from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_support import write_release_manifest
from fdm import app
from fdm.area_worker_protocol import AREA_WORKER_PROTOCOL, AREA_WORKER_PROTOCOL_VERSION
from fdm.release_manifest import (
    _probe_analysis_pipeline,
    _probe_image_processing_pipeline,
    _probe_pillow_raster_encoders,
    _probe_area_worker,
    _probe_tifffile_precision,
    packaged_runtime_features,
    run_release_self_check,
    runtime_capability_hint,
    verify_release_manifest,
)


def _minimal_pe_bytes() -> bytes:
    payload = bytearray(128)
    payload[:2] = b"MZ"
    payload[60:64] = (64).to_bytes(4, "little")
    payload[64:68] = b"PE\x00\x00"
    return bytes(payload)


def _create_minimal_release(root: Path, *, profile: str = "full") -> Path:
    (root / "src" / "fdm").mkdir(parents=True, exist_ok=True)
    (root / "src" / "fdm" / "version.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    config = """\
schema_version = 1
assets = []
[release]
ignored_untracked_prefixes = ["dist/", "build/"]
[profiles.core]
groups = []
required_python_modules = []
features = [
  "measurement",
  "screenshot-tool",
  "image-export",
  "image-processing",
  "image-analysis",
  "batch-processing",
]
[profiles.full]
extends = "core"
groups = []
required_python_modules = []
features = ["area-inference", "magic-segmentation"]
[groups]
"""
    (root / "runtime_assets.toml").write_text(config, encoding="utf-8")
    app_dir = root / "dist" / "windows" / "FiberDiameterMeasurement"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "FiberDiameterMeasurement.exe").write_bytes(_minimal_pe_bytes())
    (app_dir / "FiberAreaWorker.exe").write_bytes(_minimal_pe_bytes())
    if profile in {"core", "full"}:
        (app_dir / "FiberScreenshotTool.exe").write_bytes(_minimal_pe_bytes())
    (app_dir / "runtime_assets.toml").write_text(config, encoding="utf-8")
    write_release_manifest(
        app_dir,
        root,
        profile=profile,
        clean_build=True,
        build_id="self-check-build",
        source_commit="deadbeef",
        source_dirty_entries=[],
    )
    manifest_path = app_dir / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dependency_versions"]["tifffile"] = "test"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return app_dir


def _successful_tifffile_probe() -> dict[str, object]:
    return {
        "ok": True,
        "backend": "tifffile",
        "backend_version": "test",
        "cases": {
            "uint16": {"ok": True},
            "float32": {"ok": True},
        },
    }


class ReleaseSelfCheckTests(unittest.TestCase):
    def test_pillow_raster_encoder_probe_exercises_all_export_formats(self) -> None:
        report = _probe_pillow_raster_encoders()

        self.assertTrue(report["ok"], report)
        self.assertEqual(
            set(report["formats"]),
            {"png", "jpeg", "tiff", "bmp", "webp"},
        )
        self.assertTrue(
            all(result["ok"] for result in report["formats"].values())
        )
        self.assertEqual(
            set(report["production_cases"]),
            {
                "gray16_png",
                "gray16_tiff_deflate",
                "gray16_tiff_lzw",
                "gray16_tiff_none",
                "gray32_float_tiff_deflate",
                "webp_lossy",
                "webp_lossless",
            },
        )
        self.assertTrue(
            all(
                result["ok"]
                for result in report["production_cases"].values()
            )
        )
        self.assertTrue(
            report["production_cases"]["webp_lossless"]["exact_sha256"]
        )

    def test_image_processing_pipeline_probe_covers_all_scientific_types(self) -> None:
        report = _probe_image_processing_pipeline()

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["operation"], "gaussian_blur")
        self.assertEqual(
            {
                item["pixel_type"]
                for item in report["cases"].values()
            },
            {"gray8", "gray16", "gray32_float"},
        )
        for case in report["cases"].values():
            self.assertTrue(case["ok"])
            self.assertEqual(case["width"], 8)
            self.assertEqual(case["height"], 6)
            self.assertEqual(len(case["sha256"]), 64)
            self.assertGreater(case["bytes"], 0)

    def test_analysis_pipeline_probe_uses_safe_assets_registry_and_batch(self) -> None:
        report = _probe_analysis_pipeline()

        self.assertTrue(report["ok"], report)
        self.assertEqual(
            report["safe_npz"]["schema"],
            "fdm.self-check.analysis.v1",
        )
        self.assertEqual(
            set(report["safe_npz"]["members"]),
            {"mask", "values"},
        )
        self.assertIn("分析摘要", report["workbook"]["sheets"])
        self.assertIn("参数与来源", report["workbook"]["sheets"])
        self.assertTrue(report["workbook"]["unicode_path"])
        self.assertEqual(
            report["advanced_analysis"]["kind"],
            "directionality",
        )
        self.assertEqual(
            report["advanced_analysis"]["request_id"],
            "analysis-advanced-self-check",
        )
        self.assertEqual(report["advanced_analysis"]["generation"], 7)
        self.assertGreaterEqual(
            report["advanced_analysis"]["registered_tools"],
            7,
        )
        self.assertEqual(
            report["batch"]["request_id"],
            "analysis-batch-self-check",
        )
        self.assertEqual(report["batch"]["generation"], 11)
        self.assertEqual(report["batch"]["item_count"], 2)
        self.assertEqual(report["batch"]["success_count"], 2)
        self.assertIn("成功 2 张", report["batch"]["summary"])
        self.assertTrue(
            all(len(digest) == 64 for digest in report["batch"]["result_sha256"])
        )

    def test_tifffile_precision_probe_requires_bit_exact_uint16_and_float32(self) -> None:
        import numpy as np

        stored: dict[str, object] = {}
        fake_tifffile = ModuleType("tifffile")
        fake_tifffile.__version__ = "test"

        def imwrite(path, data, **_kwargs) -> None:
            stored[str(path)] = np.asarray(data).copy()
            Path(path).write_bytes(b"II-test-tiff")

        def imread(path):
            return stored[str(path)].copy()

        fake_tifffile.imwrite = imwrite
        fake_tifffile.imread = imread
        with patch.dict(sys.modules, {"tifffile": fake_tifffile}):
            report = _probe_tifffile_precision()

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["backend_version"], "test")
        self.assertTrue(report["cases"]["uint16"]["ok"])
        self.assertEqual(report["cases"]["uint16"]["dtype"], "uint16")
        self.assertTrue(report["cases"]["float32"]["ok"])
        self.assertEqual(report["cases"]["float32"]["dtype"], "float32")

    def test_packaged_area_worker_probe_uses_shared_protocol(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "FiberAreaWorker.exe").write_bytes(b"worker")
            weights_dir = root / "runtime" / "area-models"
            weights_dir.mkdir(parents=True)
            (weights_dir / "probe.pth").write_bytes(b"model")
            (root / "runtime" / "area-infer" / "vendor" / "yolact").mkdir(parents=True)

            def run_worker(command, **kwargs) -> subprocess.CompletedProcess[str]:
                self.assertEqual(command, [str(root / "FiberAreaWorker.exe")])
                request = json.loads(kwargs["input"])
                self.assertEqual(request["protocol"], AREA_WORKER_PROTOCOL)
                self.assertEqual(request["version"], AREA_WORKER_PROTOCOL_VERSION)
                self.assertIn("面积识别自检.png", request["image"]["path"])
                response = {
                    "protocol": AREA_WORKER_PROTOCOL,
                    "version": AREA_WORKER_PROTOCOL_VERSION,
                    "request_id": "self-check-infer",
                    "ok": True,
                    "result": {"instances": []},
                }
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(response) + "\n",
                    stderr="",
                )

            with patch("fdm.release_manifest.subprocess.run", side_effect=run_worker):
                result = _probe_area_worker(root)

            self.assertEqual(result["mode"], "one_shot")
            self.assertTrue(result["unicode_path"])
            self.assertEqual(result["image_size"], [1280, 960])
            self.assertEqual(result["instance_count"], 0)

    def test_development_wheel_without_runtime_assets_reports_capability_hint(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.assertEqual(
                packaged_runtime_features(root),
                frozenset({"measurement", "capture", "digital-slide"}),
            )
            hint = runtime_capability_hint(root)
        self.assertIn("开发 wheel", hint)
        self.assertIn("Windows full", hint)

    def test_self_check_passes_for_intact_release_and_ignores_installer_extras(self) -> None:
        with TemporaryDirectory() as tmpdir:
            app_dir = _create_minimal_release(Path(tmpdir))
            (app_dir / "unins000.exe").write_bytes(b"installer-added file")

            with patch(
                "fdm.release_manifest._probe_tifffile_precision",
                return_value=_successful_tifffile_probe(),
            ):
                report = run_release_self_check(app_dir)

            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["profile"], "full")
            self.assertEqual(report["version"], "1.2.3")
            self.assertEqual(report["build_id"], "self-check-build")
            self.assertTrue(report["functional_checks"]["core_measurement"])
            self.assertTrue(report["functional_checks"]["qt_local_ipc"])
            self.assertTrue(report["functional_checks"]["pe:FiberDiameterMeasurement.exe"])
            self.assertTrue(report["functional_checks"]["pe:FiberScreenshotTool.exe"])
            self.assertTrue(report["functional_checks"]["screenshot_tool"])

    def test_core_profile_runs_declared_image_feature_gates(self) -> None:
        with TemporaryDirectory() as tmpdir:
            app_dir = _create_minimal_release(Path(tmpdir), profile="core")
            with patch(
                "fdm.release_manifest._probe_tifffile_precision",
                return_value=_successful_tifffile_probe(),
            ):
                report = run_release_self_check(app_dir)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["profile"], "core")
        self.assertTrue(report["functional_checks"]["raster_encoders"]["ok"])
        self.assertTrue(
            report["functional_checks"]["image_processing_pipeline"]["ok"]
        )
        self.assertTrue(report["functional_checks"]["analysis_pipeline"]["ok"])
        self.assertNotIn("dependency:torch", report["functional_checks"])
        self.assertNotIn("dependency:torchvision", report["functional_checks"])

    def test_self_check_blocks_a_missing_pillow_export_encoder(self) -> None:
        with TemporaryDirectory() as tmpdir:
            app_dir = _create_minimal_release(Path(tmpdir))
            format_results = {
                format_name: {
                    "ok": format_name != "webp",
                    "available": format_name != "webp",
                    "message": (
                        "当前 Pillow 运行时没有 WEBP 编码器。"
                        if format_name == "webp"
                        else ""
                    ),
                }
                for format_name in ("png", "jpeg", "tiff", "bmp", "webp")
            }
            with (
                patch(
                    "fdm.release_manifest._probe_pillow_raster_encoders",
                    return_value={
                        "ok": False,
                        "backend": "Pillow",
                        "backend_version": "test",
                        "formats": format_results,
                    },
                ),
                patch(
                    "fdm.release_manifest._probe_tifffile_precision",
                    return_value=_successful_tifffile_probe(),
                ),
            ):
                report = run_release_self_check(app_dir)

        self.assertFalse(report["ok"])
        self.assertFalse(report["functional_checks"]["raster_encoder:webp"])
        self.assertTrue(
            any("WEBP encoder is unavailable" in error for error in report["errors"])
        )

    def test_self_check_blocks_tifffile_precision_failure(self) -> None:
        with TemporaryDirectory() as tmpdir:
            app_dir = _create_minimal_release(Path(tmpdir))
            with patch(
                "fdm.release_manifest._probe_tifffile_precision",
                side_effect=RuntimeError("uint16 samples changed"),
            ):
                report = run_release_self_check(app_dir)

        self.assertFalse(report["ok"])
        self.assertFalse(report["functional_checks"]["tifffile:uint16"])
        self.assertFalse(report["functional_checks"]["tifffile:float32"])
        self.assertTrue(
            any(
                "16-bit/float32 TIFF self-check failed" in error
                for error in report["errors"]
            )
        )

    def test_full_self_check_blocks_extended_raster_production_failure(self) -> None:
        raster_probe = _probe_pillow_raster_encoders()
        raster_probe["production_cases"]["gray16_tiff_lzw"] = {
            "ok": False,
            "message": "LZW frozen encoder missing",
        }
        raster_probe["ok"] = False
        with TemporaryDirectory() as tmpdir:
            app_dir = _create_minimal_release(Path(tmpdir))
            with (
                patch(
                    "fdm.release_manifest._probe_pillow_raster_encoders",
                    return_value=raster_probe,
                ),
                patch(
                    "fdm.release_manifest._probe_tifffile_precision",
                    return_value=_successful_tifffile_probe(),
                ),
                patch(
                    "fdm.release_manifest._probe_image_processing_pipeline",
                    return_value={"ok": True},
                ),
                patch(
                    "fdm.release_manifest._probe_analysis_pipeline",
                    return_value={"ok": True},
                ),
            ):
                report = run_release_self_check(app_dir)

        self.assertFalse(report["ok"])
        self.assertFalse(
            report["functional_checks"][
                "raster_production:gray16_tiff_lzw"
            ]
        )
        self.assertTrue(
            any(
                "raster production path self-check failed: gray16_tiff_lzw"
                in error
                for error in report["errors"]
            )
        )

    def test_full_self_check_blocks_processing_and_analysis_probe_errors(self) -> None:
        scenarios = (
            (
                "_probe_image_processing_pipeline",
                "image processing pipeline self-check failed",
            ),
            (
                "_probe_analysis_pipeline",
                "analysis pipeline self-check failed",
            ),
        )
        for probe_name, expected_error in scenarios:
            with self.subTest(probe=probe_name), TemporaryDirectory() as tmpdir:
                app_dir = _create_minimal_release(Path(tmpdir))
                with (
                    patch(
                        "fdm.release_manifest._probe_tifffile_precision",
                        return_value=_successful_tifffile_probe(),
                    ),
                    patch(
                        f"fdm.release_manifest.{probe_name}",
                        side_effect=RuntimeError("frozen dependency missing"),
                    ),
                ):
                    report = run_release_self_check(app_dir)

                self.assertFalse(report["ok"])
                self.assertTrue(
                    any(
                        expected_error in error
                        for error in report["errors"]
                    )
                )

    def test_self_check_rejects_hash_valid_non_pe_executables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app_dir = _create_minimal_release(root)
            (app_dir / "FiberAreaWorker.exe").write_bytes(b"not a PE executable")
            write_release_manifest(
                app_dir,
                root,
                profile="full",
                clean_build=True,
                build_id="non-pe-build",
                source_commit="deadbeef",
                source_dirty_entries=[],
            )

            report = run_release_self_check(app_dir)

            self.assertFalse(report["ok"])
            self.assertTrue(any("invalid Windows executable" in error for error in report["errors"]))

    def test_self_check_fails_after_packaged_file_is_modified(self) -> None:
        with TemporaryDirectory() as tmpdir:
            app_dir = _create_minimal_release(Path(tmpdir))
            (app_dir / "FiberDiameterMeasurement.exe").write_bytes(b"tampered")

            report = run_release_self_check(app_dir)

            self.assertFalse(report["ok"])
            self.assertTrue(any("mismatch" in error for error in report["errors"]))

    def test_release_gate_can_reject_unmanifested_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            app_dir = _create_minimal_release(Path(tmpdir))
            (app_dir / "unexpected.dll").write_bytes(b"unexpected")

            report = verify_release_manifest(app_dir, reject_extra_files=True)

            self.assertFalse(report["ok"])
            self.assertTrue(any("unmanifested packaged files" in error for error in report["errors"]))

    def test_app_self_check_json_returns_machine_readable_result_without_starting_ui(self) -> None:
        with TemporaryDirectory() as tmpdir:
            app_dir = _create_minimal_release(Path(tmpdir))
            output = StringIO()

            with (
                patch.dict(os.environ, {"FDM_SELF_CHECK_ROOT": str(app_dir)}),
                patch(
                    "fdm.release_manifest._probe_tifffile_precision",
                    return_value=_successful_tifffile_probe(),
                ),
                redirect_stdout(output),
            ):
                result = app.main(["fdm", "--self-check", "--json"])

            payload = json.loads(output.getvalue())
            self.assertEqual(result, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["build_id"], "self-check-build")


if __name__ == "__main__":
    unittest.main()
