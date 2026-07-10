from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.engine import AreaNativeEngine, InferServiceError
from app.model_metadata import MODEL_SPECS


class _FakeCuda:
    def __init__(self, available: bool, count: int = 1, name: str = "RTX 4060") -> None:
        self._available = available
        self._count = count
        self._name = name

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count if self._available else 0

    def get_device_name(self, idx: int) -> str:
        if not self._available:
            raise RuntimeError("cuda unavailable")
        return self._name


class _FakeTorch:
    def __init__(self, available: bool) -> None:
        self.cuda = _FakeCuda(available=available)

    def device(self, token: str) -> str:
        return token


class EngineDeviceRoutingTests(unittest.TestCase):
    def _new_engine(self, infer_device: str, gpu_policy: str) -> AreaNativeEngine:
        return AreaNativeEngine(
            weights_dir="/tmp",
            vendor_root="/tmp",
            infer_device=infer_device,
            gpu_policy=gpu_policy,
        )

    def test_auto_uses_cuda_when_available(self) -> None:
        engine = self._new_engine("auto", "warn_continue")
        engine._torch = _FakeTorch(available=True)

        engine._resolve_runtime_device()

        self.assertEqual(engine._effective_device_key, "cuda")
        payload = engine._runtime_device_payload()
        self.assertEqual(payload["effective_device"], "cuda:0")
        self.assertTrue(payload["cuda_available"])
        self.assertIsNone(payload["device_warning"])

    def test_cuda_warn_continue_falls_back_to_cpu(self) -> None:
        engine = self._new_engine("cuda", "warn_continue")
        engine._torch = _FakeTorch(available=False)

        engine._resolve_runtime_device()

        self.assertEqual(engine._effective_device_key, "cpu")
        payload = engine._runtime_device_payload()
        self.assertEqual(payload["effective_device"], "cpu")
        self.assertEqual(payload["device_warning"], "cuda_requested_but_unavailable_fallback_to_cpu")

    def test_cuda_fail_raises_when_unavailable(self) -> None:
        engine = self._new_engine("cuda:0", "fail")
        engine._torch = _FakeTorch(available=False)

        with self.assertRaises(InferServiceError) as ctx:
            engine._resolve_runtime_device()

        self.assertEqual(ctx.exception.code, "infer_service_unavailable")

    def test_legacy_cuda_token_normalizes_to_cuda_zero(self) -> None:
        engine = self._new_engine("cuda", "warn_continue")

        self.assertEqual(engine._requested_device, "cuda:0")

    def test_readiness_accepts_at_least_one_verified_loadable_model(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "weights"
            vendor = root / "vendor"
            weights.mkdir()
            vendor.mkdir()
            first = str(MODEL_SPECS[0]["model_file"])
            second = str(MODEL_SPECS[1]["model_file"])
            (weights / first).write_bytes(b"first")
            (weights / second).write_bytes(b"second")
            engine = AreaNativeEngine(
                weights_dir=str(weights),
                vendor_root=str(vendor),
                require_trusted_weights=True,
                verify_trusted_weights=True,
            )

            def load_model(*, model_name: str, model_file: str):
                del model_name
                if model_file == first:
                    raise InferServiceError("infer_model_load_failed", "broken-first")
                return SimpleNamespace(model_file=model_file)

            with patch.object(engine, "_verify_weight_path", return_value=True), patch.object(
                engine,
                "_load_model",
                side_effect=load_model,
            ), patch.object(engine, "_runtime_device_payload", return_value={"effective_device": "cpu"}):
                payload = engine.readiness()

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["loadable_model"], second)


if __name__ == "__main__":
    unittest.main()
