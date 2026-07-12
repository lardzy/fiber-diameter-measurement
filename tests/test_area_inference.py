from __future__ import annotations

from io import BytesIO, StringIO, TextIOWrapper
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import PIL  # noqa: F401
    import torch  # noqa: F401
    import torchvision  # noqa: F401

    AREA_RUNTIME_DEPS_AVAILABLE = True
except Exception:
    AREA_RUNTIME_DEPS_AVAILABLE = False

try:
    from fdm.ui.area_inference_worker import AreaBatchInferenceWorker, AreaInferenceRequest

    QT_AREA_WORKER_AVAILABLE = True
except Exception:
    AreaBatchInferenceWorker = object  # type: ignore[assignment]
    AreaInferenceRequest = object  # type: ignore[assignment]
    QT_AREA_WORKER_AVAILABLE = False

from fdm.cancellation import CancellationSource
from fdm.services.area_inference import (
    AREA_WORKER_PROTOCOL,
    AREA_WORKER_PROTOCOL_VERSION,
    AreaInferenceCancelledError,
    AreaInferenceProtocolError,
    AreaInferenceService,
    AreaInferenceTimeoutError,
    AreaInferenceTransportError,
)
from fdm.settings import AppSettings, application_root
from fdm.workers import area_worker as area_worker_module
from fdm.workers.area_worker import _load_engine_module


class _ProtocolProcess:
    def __init__(self, *, result: dict[str, object] | None = None, request_id: str | None = None) -> None:
        self.returncode = 0
        self.result = result or {"instances": [], "engine_meta": {}}
        self.request_id = request_id
        self.request_payload: dict[str, object] | None = None
        self.terminated = False
        self.killed = False

    def communicate(self, input=None, timeout=None):
        if input is not None:
            self.request_payload = json.loads(input)
        request_id = self.request_id
        if request_id is None and self.request_payload is not None:
            request_id = str(self.request_payload["request_id"])
        response = {
            "protocol": AREA_WORKER_PROTOCOL,
            "version": AREA_WORKER_PROTOCOL_VERSION,
            "request_id": request_id,
            "ok": True,
            "result": self.result,
        }
        return json.dumps(response), ""

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class AreaInferenceTests(unittest.TestCase):
    def test_area_inference_service_uses_hidden_subprocess_flags_on_windows(self) -> None:
        service = AreaInferenceService()
        process = _ProtocolProcess()

        with (
            patch.object(sys, "platform", "win32"),
            patch.object(sys, "frozen", False, create=True),
            patch.object(service, "_worker_command", return_value=["FiberAreaWorker.exe"]),
            patch("fdm.services.area_inference.subprocess.CREATE_NO_WINDOW", 0x08000000, create=True),
            patch("fdm.services.area_inference.subprocess.Popen", return_value=process) as popen_mock,
        ):
            service.infer_image(
                image_path="/tmp/fake.png",
                model_name="棉-莱赛尔",
                model_file="b_c1_1.3.pth",
                settings=AppSettings(),
            )

        self.assertEqual(popen_mock.call_args.kwargs.get("creationflags"), 0x08000000)
        self.assertIsNotNone(process.request_payload)
        assert process.request_payload is not None
        self.assertEqual(process.request_payload["protocol"], AREA_WORKER_PROTOCOL)
        self.assertEqual(process.request_payload["image"], {"path": str(Path("/tmp/fake.png").resolve())})
        self.assertFalse(process.request_payload["options"]["include_overlay"])
        self.assertEqual(process.request_payload["runtime"]["device"], "cpu")
        self.assertFalse(process.request_payload["runtime"]["allow_untrusted_weights"])
        self.assertTrue(process.request_payload["runtime"]["require_trusted_weights"])
        self.assertTrue(process.request_payload["runtime"]["verify_trusted_weights"])
        environment = popen_mock.call_args.kwargs["env"]
        self.assertEqual(environment["PYTHONUTF8"], "1")
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")

    def test_area_worker_reconfigures_windows_code_page_streams_to_utf8(self) -> None:
        stdin_bytes = BytesIO()
        stdout_bytes = BytesIO()
        stderr_bytes = BytesIO()
        stdin_stream = TextIOWrapper(stdin_bytes, encoding="gbk")
        stdout_stream = TextIOWrapper(stdout_bytes, encoding="gbk")
        stderr_stream = TextIOWrapper(stderr_bytes, encoding="gbk")

        with (
            patch.object(sys, "stdin", stdin_stream),
            patch.object(sys, "stdout", stdout_stream),
            patch.object(sys, "stderr", stderr_stream),
        ):
            input_stream, protocol_stdout, diagnostic_stream = (
                area_worker_module._configure_protocol_streams()
            )
            self.assertEqual(input_stream.encoding.lower().replace("-", ""), "utf8")
            self.assertEqual(protocol_stdout.encoding.lower().replace("-", ""), "utf8")
            self.assertEqual(diagnostic_stream.encoding.lower().replace("-", ""), "utf8")
            protocol_stdout.write("未找到图片: F:/示例显微图像/样本.jpg")
            protocol_stdout.flush()

        self.assertEqual(
            stdout_bytes.getvalue(),
            "未找到图片: F:/示例显微图像/样本.jpg".encode("utf-8"),
        )

    def test_area_worker_round_trips_chinese_path_from_legacy_code_page_environment(self) -> None:
        request = {
            "protocol": AREA_WORKER_PROTOCOL,
            "version": AREA_WORKER_PROTOCOL_VERSION,
            "request_id": "chinese-path-request",
            "op": "infer",
            "image": {"path": "F:/Downloads/示例显微图像/样本.jpg"},
            "model": {"name": "棉", "file": "model.pth"},
            "runtime": {
                "weights_dir": "F:/missing/weights",
                "vendor_root": "F:/missing/vendor",
                "device": "cpu",
            },
            "options": {"include_overlay": False, "inference": {}},
        }
        environment = dict(os.environ)
        environment["PYTHONIOENCODING"] = "gbk"
        environment["PYTHONUTF8"] = "0"
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

        completed = subprocess.run(
            [sys.executable, str(Path(area_worker_module.__file__).resolve())],
            input=json.dumps(request, ensure_ascii=False).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=10,
            check=False,
        )

        self.assertEqual(completed.returncode, 2)
        response = json.loads(completed.stdout.decode("utf-8"))
        self.assertEqual(response["request_id"], "chinese-path-request")
        self.assertEqual(response["error"]["code"], "invalid_request")
        self.assertIn("F:/Downloads/示例显微图像/样本.jpg", response["error"]["message"])

    def test_area_worker_persists_request_phase_trace_when_diagnostics_are_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "area-worker.log"
            request = {
                "protocol": AREA_WORKER_PROTOCOL,
                "version": AREA_WORKER_PROTOCOL_VERSION,
                "request_id": "trace-request",
                "op": "infer",
                "image": {"path": str(Path(tmp) / "missing-image.png")},
                "model": {"name": "棉", "file": "model.pth"},
                "runtime": {
                    "weights_dir": str(Path(tmp) / "weights"),
                    "vendor_root": str(Path(tmp) / "vendor"),
                    "device": "cpu",
                },
                "options": {"include_overlay": False, "inference": {}},
            }
            diagnostics = StringIO()
            with patch.dict(
                os.environ,
                {
                    "FDM_AREA_WORKER_DIAGNOSTICS": "1",
                    "FDM_AREA_WORKER_LOG_PATH": str(trace_path),
                },
            ):
                response, exit_code = area_worker_module._process_request(
                    json.dumps(request, ensure_ascii=False),
                    worker_runtime=area_worker_module._AreaWorkerRuntime(),
                    diagnostic_stream=diagnostics,
                )

            records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(exit_code, 2)
        self.assertFalse(response["ok"])
        self.assertEqual(
            [record["stage"] for record in records],
            ["request_received", "request_failed"],
        )
        self.assertTrue(all(record["request_id"] == "trace-request" for record in records))

    def test_custom_model_opt_in_is_source_only_and_explicit(self) -> None:
        service = AreaInferenceService()
        process = _ProtocolProcess()
        with (
            patch.dict(os.environ, {"FDM_ALLOW_UNTRUSTED_AREA_MODELS": "1"}),
            patch.object(sys, "frozen", False, create=True),
            patch.object(service, "_worker_command", return_value=["worker"]),
            patch("fdm.services.area_inference.subprocess.Popen", return_value=process),
        ):
            service.infer_image(
                image_path="/tmp/fake.png",
                model_name="custom",
                model_file="custom.pth",
                settings=AppSettings(),
            )

        assert process.request_payload is not None
        runtime = process.request_payload["runtime"]
        self.assertTrue(runtime["allow_untrusted_weights"])
        self.assertFalse(runtime["require_trusted_weights"])
        self.assertTrue(runtime["verify_trusted_weights"])

    def test_custom_model_opt_in_is_ignored_by_frozen_release(self) -> None:
        with (
            patch.dict(os.environ, {"FDM_ALLOW_UNTRUSTED_AREA_MODELS": "1"}),
            patch.object(sys, "frozen", True, create=True),
        ):
            allowed = area_worker_module._allow_untrusted_development_weights(
                {
                    "allow_untrusted_weights": True,
                    "require_trusted_weights": False,
                    "verify_trusted_weights": False,
                }
            )

        self.assertFalse(allowed)

    def test_custom_model_opt_in_requires_worker_environment_confirmation(self) -> None:
        runtime = {
            "allow_untrusted_weights": True,
            "require_trusted_weights": False,
            "verify_trusted_weights": False,
        }
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "frozen", False, create=True),
        ):
            self.assertFalse(area_worker_module._allow_untrusted_development_weights(runtime))
        with (
            patch.dict(os.environ, {"FDM_ALLOW_UNTRUSTED_AREA_MODELS": "yes"}),
            patch.object(sys, "frozen", False, create=True),
        ):
            self.assertTrue(area_worker_module._allow_untrusted_development_weights(runtime))

    def test_custom_model_rejection_has_actionable_message(self) -> None:
        service = AreaInferenceService()
        message = service._friendly_failure_message(
            "untrusted_model_file:custom.pth",
            worker_command=["worker"],
        )

        self.assertIn("custom.pth", message)
        self.assertIn("FDM_ALLOW_UNTRUSTED_AREA_MODELS=1", message)
        self.assertIn("正式打包版本不接受", message)

    def test_area_device_setting_defaults_to_cpu_and_round_trips(self) -> None:
        settings = AppSettings()
        self.assertEqual(settings.area_infer_device, "cpu")

        restored = AppSettings.from_dict({"area_infer_device": "cuda"})
        self.assertEqual(restored.area_infer_device, "cuda:0")
        self.assertEqual(restored.to_dict()["area_infer_device"], "cuda:0")

        invalid = AppSettings.from_dict({"area_infer_device": "cuda:7"})
        self.assertEqual(invalid.area_infer_device, "cpu")

    def test_area_inference_timeout_range_is_validated_before_process_start(self) -> None:
        service = AreaInferenceService()
        with patch("fdm.services.area_inference.subprocess.Popen") as popen_mock:
            for invalid_timeout in (0, 29.9, 600.1, float("inf"), True):
                with self.subTest(timeout=invalid_timeout), self.assertRaises(ValueError):
                    service.infer_image(
                        image_path="/tmp/fake.png",
                        model_name="棉-莱赛尔",
                        model_file="b_c1_1.3.pth",
                        settings=AppSettings(),
                        timeout_s=invalid_timeout,
                    )
        popen_mock.assert_not_called()

    def test_area_inference_cancel_terminates_then_kills_hung_worker(self) -> None:
        service = AreaInferenceService()
        cancellation = CancellationSource()

        class _HangingProcess(_ProtocolProcess):
            def __init__(self) -> None:
                super().__init__()
                self.returncode = None

            def communicate(self, input=None, timeout=None):
                if input is not None and not cancellation.token.is_cancelled:
                    cancellation.cancel()
                if self.killed:
                    self.returncode = -9
                    return "", ""
                raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout or 0)

            def poll(self):
                return self.returncode

        process = _HangingProcess()
        with (
            patch.object(service, "_worker_command", return_value=["worker"]),
            patch("fdm.services.area_inference.subprocess.Popen", return_value=process),
            self.assertRaises(AreaInferenceCancelledError),
        ):
            service.infer_image(
                image_path="/tmp/fake.png",
                model_name="棉-莱赛尔",
                model_file="b_c1_1.3.pth",
                settings=AppSettings(),
                cancellation_token=cancellation.token,
            )

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_area_inference_timeout_terminates_hung_worker(self) -> None:
        service = AreaInferenceService()

        class _HangingProcess(_ProtocolProcess):
            def __init__(self) -> None:
                super().__init__()
                self.returncode = None

            def communicate(self, input=None, timeout=None):
                if self.terminated:
                    self.returncode = -15
                    return "", ""
                raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout or 0)

            def poll(self):
                return self.returncode

        process = _HangingProcess()
        with (
            patch.object(service, "_worker_command", return_value=["worker"]),
            patch("fdm.services.area_inference.subprocess.Popen", return_value=process),
            patch("fdm.services.area_inference.time.monotonic", side_effect=[0.0, 31.0]),
            self.assertRaises(AreaInferenceTimeoutError),
        ):
            service.infer_image(
                image_path="/tmp/fake.png",
                model_name="棉-莱赛尔",
                model_file="b_c1_1.3.pth",
                settings=AppSettings(),
                timeout_s=30.0,
            )

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)

    def test_area_inference_rejects_mismatched_request_id_and_non_finite_numbers(self) -> None:
        service = AreaInferenceService()
        with (
            patch.object(service, "_worker_command", return_value=["worker"]),
            patch(
                "fdm.services.area_inference.subprocess.Popen",
                return_value=_ProtocolProcess(request_id="wrong-request"),
            ),
            self.assertRaisesRegex(AreaInferenceProtocolError, "request_id"),
        ):
            service.infer_image(
                image_path="/tmp/fake.png",
                model_name="棉-莱赛尔",
                model_file="b_c1_1.3.pth",
                settings=AppSettings(),
            )

        non_finite_result = {
            "instances": [],
            "engine_meta": {"elapsed_ms": float("nan")},
        }
        with (
            patch.object(service, "_worker_command", return_value=["worker"]),
            patch(
                "fdm.services.area_inference.subprocess.Popen",
                return_value=_ProtocolProcess(result=non_finite_result),
            ),
            self.assertRaisesRegex(AreaInferenceProtocolError, "非有限"),
        ):
            service.infer_image(
                image_path="/tmp/fake.png",
                model_name="棉-莱赛尔",
                model_file="b_c1_1.3.pth",
                settings=AppSettings(),
            )

    def test_area_worker_keeps_stdout_as_single_json_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            image_path.write_bytes(b"image-bytes")
            vendor_root = root / "runtime" / "vendor" / "yolact"
            vendor_root.mkdir(parents=True)
            weights_dir = root / "weights"
            weights_dir.mkdir()
            (weights_dir / "model.pth").write_bytes(b"weights")

            class _FakeEngine:
                init_kwargs: dict[str, object] | None = None

                def __init__(self, **kwargs) -> None:
                    self.kwargs = kwargs
                    type(self).init_kwargs = kwargs

                def infer(self, **kwargs):
                    return {
                        "instances": [],
                        "engine_meta": {"effective_device": "cpu"},
                        "overlay_png_b64": "unused-overlay",
                    }

            class _FakeEngineModule:
                AreaNativeEngine = _FakeEngine

            def fake_load_engine(_vendor_root):
                print("Multiple GPUs detected!")
                return _FakeEngineModule

            request = {
                "protocol": AREA_WORKER_PROTOCOL,
                "version": AREA_WORKER_PROTOCOL_VERSION,
                "request_id": "request-1",
                "op": "infer",
                "image": {"path": str(image_path)},
                "model": {"name": "棉-莱赛尔", "file": "model.pth"},
                "runtime": {"weights_dir": str(weights_dir), "vendor_root": str(vendor_root)},
                "options": {"include_overlay": False, "inference": {}},
            }
            protocol_stdout = StringIO()
            diagnostics = StringIO()
            with (
                patch.object(sys, "stdin", StringIO(json.dumps(request))),
                patch.object(sys, "stdout", protocol_stdout),
                patch.object(sys, "stderr", diagnostics),
                patch.object(area_worker_module, "_load_engine_module", side_effect=fake_load_engine),
            ):
                exit_code = area_worker_module.main()

        self.assertEqual(exit_code, 0)
        lines = protocol_stdout.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        response = json.loads(lines[0])
        self.assertTrue(response["ok"])
        self.assertEqual(response["request_id"], "request-1")
        self.assertNotIn("overlay_png_b64", response["result"])
        self.assertIn("Multiple GPUs detected!", diagnostics.getvalue())
        self.assertNotIn("Multiple GPUs detected!", protocol_stdout.getvalue())
        assert _FakeEngine.init_kwargs is not None
        self.assertTrue(_FakeEngine.init_kwargs["require_trusted_weights"])
        self.assertTrue(_FakeEngine.init_kwargs["verify_trusted_weights"])

    def test_trusted_metadata_response_is_not_legacy_swapped_again(self) -> None:
        service = AreaInferenceService()
        result_payload = {
            "instances": [
                {
                    "class_name": "莱赛尔",
                    "score": 0.9,
                    "bbox": [0, 0, 2, 2],
                    "polygon": [[0, 0], [2, 0], [0, 2]],
                    "area_px": 2,
                }
            ],
            "engine_meta": {"class_mapping": "trusted_metadata_v1"},
        }
        with (
            patch.object(service, "_worker_command", return_value=["worker"]),
            patch(
                "fdm.services.area_inference.subprocess.Popen",
                return_value=_ProtocolProcess(result=result_payload),
            ),
        ):
            result = service.infer_image(
                image_path="/tmp/fake.png",
                model_name="棉-莱赛尔",
                model_file="b_c1_1.3.pth",
                settings=AppSettings(),
            )

        self.assertEqual(result.instances[0].class_name, "莱赛尔")

    def test_persistent_transport_failure_falls_back_to_one_shot(self) -> None:
        service = AreaInferenceService()
        process = _ProtocolProcess()

        class _BrokenSession:
            disabled = False

            def request(self, **kwargs):
                raise AreaInferenceTransportError("broken")

            def disable(self) -> None:
                self.disabled = True

        session = _BrokenSession()
        with (
            patch.object(service, "_worker_command", return_value=["worker"]),
            patch("fdm.services.area_inference.subprocess.Popen", return_value=process) as popen_mock,
        ):
            result = service.infer_image(
                image_path="/tmp/fake.png",
                model_name="棉-莱赛尔",
                model_file="b_c1_1.3.pth",
                settings=AppSettings(),
                worker_session=session,
            )

        self.assertEqual(result.instances, [])
        self.assertTrue(session.disabled)
        popen_mock.assert_called_once()

    def test_persistent_internal_error_envelope_falls_back_to_one_shot(self) -> None:
        service = AreaInferenceService()
        process = _ProtocolProcess()

        class _InternalErrorSession:
            disabled = False

            def request(self, **kwargs):
                request_id = kwargs["payload"]["request_id"]
                return (
                    json.dumps(
                        {
                            "protocol": AREA_WORKER_PROTOCOL,
                            "version": AREA_WORKER_PROTOCOL_VERSION,
                            "request_id": request_id,
                            "ok": False,
                            "error": {"code": "internal_error", "message": "corrupt cached runtime"},
                        }
                    ),
                    "",
                )

            def disable(self) -> None:
                self.disabled = True

        session = _InternalErrorSession()
        with (
            patch.object(service, "_worker_command", return_value=["worker"]),
            patch("fdm.services.area_inference.subprocess.Popen", return_value=process) as popen_mock,
        ):
            result = service.infer_image(
                image_path="/tmp/fake.png",
                model_name="棉-莱赛尔",
                model_file="b_c1_1.3.pth",
                settings=AppSettings(),
                worker_session=session,
            )

        self.assertEqual(result.instances, [])
        self.assertTrue(session.disabled)
        popen_mock.assert_called_once()

    def test_real_persistent_session_exchanges_jsonl_envelope(self) -> None:
        service = AreaInferenceService()
        session = service.create_batch_session(AppSettings())
        request_id = "persistent-invalid-image"
        payload = {
            "protocol": AREA_WORKER_PROTOCOL,
            "version": AREA_WORKER_PROTOCOL_VERSION,
            "request_id": request_id,
            "op": "infer",
            "image": {"path": "/definitely/missing/示例显微图像.png"},
            "model": {"name": "棉-莱赛尔", "file": "b_c1_1.3.pth"},
            "runtime": {
                "weights_dir": "/definitely/missing/weights",
                "vendor_root": "/definitely/missing/vendor",
                "device": "cpu",
                "require_trusted_weights": False,
                "verify_trusted_weights": True,
            },
            "options": {"include_overlay": False, "inference": {}},
        }
        try:
            stdout, _stderr = session.request(
                payload=payload,
                timeout_s=30.0,
                cancellation_token=None,
            )
        finally:
            session.close()

        response = json.loads(stdout)
        self.assertEqual(response["request_id"], request_id)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "invalid_request")
        self.assertIn("示例显微图像.png", response["error"]["message"])

    def test_persistent_worker_reuses_engine_and_recycles_at_request_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            image_path.write_bytes(b"image-bytes")
            vendor_root = root / "runtime" / "vendor" / "yolact"
            vendor_root.mkdir(parents=True)
            weights_dir = root / "weights"
            weights_dir.mkdir()
            (weights_dir / "model.pth").write_bytes(b"weights")

            class _FakeEngine:
                created = 0
                inferred = 0

                def __init__(self, **kwargs) -> None:
                    type(self).created += 1

                def infer(self, **kwargs):
                    type(self).inferred += 1
                    return {"instances": [], "engine_meta": {"effective_device": "cpu"}}

            class _FakeEngineModule:
                AreaNativeEngine = _FakeEngine

            def request(request_id: str) -> dict[str, object]:
                return {
                    "protocol": AREA_WORKER_PROTOCOL,
                    "version": AREA_WORKER_PROTOCOL_VERSION,
                    "request_id": request_id,
                    "op": "infer",
                    "image": {"path": str(image_path)},
                    "model": {"name": "custom", "file": "model.pth"},
                    "runtime": {
                        "weights_dir": str(weights_dir),
                        "vendor_root": str(vendor_root),
                        "device": "cpu",
                        "require_trusted_weights": False,
                        "verify_trusted_weights": False,
                    },
                    "options": {"include_overlay": False, "inference": {}},
                }

            input_lines = "\n".join(json.dumps(request(item)) for item in ("request-1", "request-2")) + "\n"
            protocol_stdout = StringIO()
            diagnostics = StringIO()
            with (
                patch.object(area_worker_module, "_load_engine_module", return_value=_FakeEngineModule),
                patch.object(area_worker_module, "_current_rss_bytes", return_value=0),
            ):
                exit_code = area_worker_module.serve_persistent(
                    protocol_stdout=protocol_stdout,
                    diagnostic_stream=diagnostics,
                    input_stream=StringIO(input_lines),
                    max_requests=2,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(_FakeEngine.created, 1)
        self.assertEqual(_FakeEngine.inferred, 2)
        responses = [json.loads(line) for line in protocol_stdout.getvalue().splitlines()]
        self.assertEqual([item["request_id"] for item in responses], ["request-1", "request-2"])
        self.assertIn("persistent_worker_request_limit:2", diagnostics.getvalue())

    def test_persistent_worker_recycles_at_rss_limit_and_idle_timeout(self) -> None:
        protocol_stdout = StringIO()
        diagnostics = StringIO()
        response = {
            "protocol": AREA_WORKER_PROTOCOL,
            "version": AREA_WORKER_PROTOCOL_VERSION,
            "request_id": "request-1",
            "ok": True,
            "result": {"instances": [], "engine_meta": {}},
        }
        with (
            patch.object(area_worker_module, "_process_request", return_value=(response, 0)) as process_mock,
            patch.object(
                area_worker_module,
                "_current_rss_bytes",
                return_value=area_worker_module.PERSISTENT_MAX_RSS_BYTES + 1,
            ),
        ):
            area_worker_module.serve_persistent(
                protocol_stdout=protocol_stdout,
                diagnostic_stream=diagnostics,
                input_stream=StringIO("first\nsecond\n"),
            )

        self.assertEqual(process_mock.call_count, 1)
        self.assertEqual(len(protocol_stdout.getvalue().splitlines()), 1)
        self.assertIn("persistent_worker_rss_limit", diagnostics.getvalue())

        gate = Event()

        class _DelayedEmptyInput:
            def __iter__(self):
                return self

            def __next__(self):
                gate.wait(1.0)
                raise StopIteration

        diagnostics = StringIO()
        try:
            area_worker_module.serve_persistent(
                protocol_stdout=StringIO(),
                diagnostic_stream=diagnostics,
                input_stream=_DelayedEmptyInput(),
                idle_timeout_s=0.01,
            )
        finally:
            gate.set()

        self.assertIn("persistent_worker_idle_timeout", diagnostics.getvalue())
        self.assertEqual(area_worker_module.PERSISTENT_MAX_REQUESTS, 100)
        self.assertEqual(area_worker_module.PERSISTENT_IDLE_TIMEOUT_S, 60.0)
        self.assertEqual(area_worker_module.PERSISTENT_MAX_RSS_BYTES, 1536 * 1024 * 1024)

    def test_load_engine_module_handles_dataclass_module_registration(self) -> None:
        vendor_root = application_root() / "runtime" / "area-infer" / "vendor" / "yolact"
        if not vendor_root.exists():
            self.skipTest(f"vendor root not found: {vendor_root}")

        module = _load_engine_module(vendor_root.resolve())

        self.assertTrue(hasattr(module, "AreaNativeEngine"))

    def test_area_inference_service_runs_cpu_inference_on_demo_image(self) -> None:
        if not AREA_RUNTIME_DEPS_AVAILABLE:
            self.skipTest("area inference runtime dependencies are not installed")

        image_path = application_root() / "sample_data" / "readme-demo" / "演示图片.jpg"
        weights_path = application_root() / "runtime" / "area-models" / "b_c1_1.3.pth"
        vendor_root = application_root() / "runtime" / "area-infer" / "vendor" / "yolact"
        if not image_path.exists():
            self.skipTest(f"demo image not found: {image_path}")
        if not weights_path.exists():
            self.skipTest(f"weight not found: {weights_path}")
        if not vendor_root.exists():
            self.skipTest(f"vendor root not found: {vendor_root}")

        service = AreaInferenceService()
        result = service.infer_image(
            image_path=str(image_path),
            model_name="棉-莱赛尔",
            model_file="b_c1_1.3.pth",
            settings=AppSettings(),
        )

        self.assertGreater(len(result.instances), 0)
        self.assertEqual(result.engine_meta.get("effective_device"), "cpu")
        self.assertEqual(result.engine_meta.get("requested_device"), "cpu")

    @unittest.skipUnless(QT_AREA_WORKER_AVAILABLE, "requires Qt area worker")
    def test_area_batch_inference_worker_emits_progress_and_success(self) -> None:
        emitted_progress: list[tuple[int, int, str, str, int]] = []
        emitted_success: list[tuple[str, object, str, int]] = []
        emitted_finished: list[tuple[bool, int, int, int]] = []

        worker = AreaBatchInferenceWorker(
            [
                AreaInferenceRequest(
                    document_id="doc-1",
                    image_path="/tmp/fake-image.png",
                    model_name="棉-莱赛尔",
                    model_file="b_c1_1.3.pth",
                    request_id="request-1",
                    generation=7,
                )
            ],
            settings=AppSettings(),
        )
        worker.progress.connect(
            lambda index, total, path, request_id, generation: emitted_progress.append(
                (index, total, path, request_id, generation)
            )
        )
        worker.succeeded.connect(
            lambda document_id, instances, request_id, generation: emitted_success.append(
                (document_id, instances, request_id, generation)
            )
        )
        worker.finished.connect(
            lambda cancelled, completed, failed, generation: emitted_finished.append(
                (cancelled, completed, failed, generation)
            )
        )

        class _FakeResult:
            def __init__(self) -> None:
                self.instances = ["ok"]

        from unittest.mock import patch

        with patch(
            "fdm.ui.area_inference_worker.AreaInferenceService.infer_image",
            return_value=_FakeResult(),
        ) as infer_mock:
            worker.run()

        self.assertEqual(emitted_progress, [(1, 1, "/tmp/fake-image.png", "request-1", 7)])
        self.assertEqual(emitted_success, [("doc-1", ["ok"], "request-1", 7)])
        self.assertEqual(emitted_finished, [(False, 1, 0, 7)])
        self.assertNotIn("worker_session", infer_mock.call_args.kwargs)

    @unittest.skipUnless(QT_AREA_WORKER_AVAILABLE, "requires Qt area worker")
    def test_area_batch_timeout_emits_failure_and_terminal_signal(self) -> None:
        request = AreaInferenceRequest(
            document_id="doc-timeout",
            image_path="/tmp/timeout-image.png",
            model_name="棉",
            model_file="model.pth",
            request_id="timeout-request",
            generation=3,
        )
        worker = AreaBatchInferenceWorker([request], settings=AppSettings())
        failures: list[tuple[str, str, str, str, int]] = []
        finished: list[tuple[bool, int, int, int]] = []
        worker.failed.connect(lambda *args: failures.append(args))
        worker.finished.connect(lambda *args: finished.append(args))

        with patch(
            "fdm.ui.area_inference_worker.AreaInferenceService.infer_image",
            side_effect=AreaInferenceTimeoutError("面积识别超过 180 秒"),
        ):
            worker.run()

        self.assertEqual(len(failures), 1)
        self.assertIn("超过 180 秒", failures[0][2])
        self.assertEqual(finished, [(False, 1, 1, 3)])

    @unittest.skipUnless(QT_AREA_WORKER_AVAILABLE, "requires Qt area worker")
    def test_area_batch_worker_cancel_is_thread_safe_and_stops_current_subprocess(self) -> None:
        emitted_failed: list[tuple[str, str, str, str, int]] = []
        emitted_finished: list[tuple[bool, int, int, int]] = []
        worker = AreaBatchInferenceWorker(
            [
                AreaInferenceRequest(
                    document_id="doc-1",
                    image_path="/tmp/fake-image.png",
                    model_name="棉-莱赛尔",
                    model_file="b_c1_1.3.pth",
                ),
                AreaInferenceRequest(
                    document_id="doc-2",
                    image_path="/tmp/never-started.png",
                    model_name="棉-莱赛尔",
                    model_file="b_c1_1.3.pth",
                ),
            ],
            settings=AppSettings(),
        )
        worker.failed.connect(
            lambda document_id, path, reason, request_id, generation: emitted_failed.append(
                (document_id, path, reason, request_id, generation)
            )
        )
        worker.finished.connect(
            lambda cancelled, completed, failed, generation: emitted_finished.append(
                (cancelled, completed, failed, generation)
            )
        )

        def cancel_current_request(**kwargs):
            self.assertIs(kwargs["cancellation_token"], worker.cancellation_token)
            worker.request_cancel()
            raise AreaInferenceCancelledError("cancelled")

        with patch(
            "fdm.ui.area_inference_worker.AreaInferenceService.infer_image",
            side_effect=cancel_current_request,
        ) as infer_mock:
            worker.run()

        self.assertEqual(infer_mock.call_count, 1)
        self.assertEqual(emitted_failed, [])
        self.assertEqual(emitted_finished, [(True, 0, 0, 0)])
