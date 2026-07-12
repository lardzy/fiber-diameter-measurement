from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_support import write_release_manifest
from fdm import app
from fdm.area_worker_protocol import AREA_WORKER_PROTOCOL, AREA_WORKER_PROTOCOL_VERSION
from fdm.release_manifest import (
    _probe_area_worker,
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


def _create_minimal_release(root: Path) -> Path:
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
features = ["measurement"]
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
    (app_dir / "runtime_assets.toml").write_text(config, encoding="utf-8")
    write_release_manifest(
        app_dir,
        root,
        profile="full",
        clean_build=True,
        build_id="self-check-build",
        source_commit="deadbeef",
        source_dirty_entries=[],
    )
    return app_dir


class ReleaseSelfCheckTests(unittest.TestCase):
    def test_packaged_area_worker_probe_uses_shared_protocol(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "FiberAreaWorker.exe").write_bytes(b"worker")
            weights_dir = root / "runtime" / "area-models"
            weights_dir.mkdir(parents=True)
            (weights_dir / "probe.pth").write_bytes(b"model")
            (root / "runtime" / "area-infer" / "vendor" / "yolact").mkdir(parents=True)

            def run_worker(command, **kwargs) -> subprocess.CompletedProcess[str]:
                requests = [json.loads(line) for line in kwargs["input"].splitlines()]
                self.assertEqual(
                    [request["protocol"] for request in requests],
                    [AREA_WORKER_PROTOCOL, AREA_WORKER_PROTOCOL],
                )
                self.assertEqual(
                    [request["version"] for request in requests],
                    [AREA_WORKER_PROTOCOL_VERSION, AREA_WORKER_PROTOCOL_VERSION],
                )
                responses = [
                    {
                        "protocol": AREA_WORKER_PROTOCOL,
                        "version": AREA_WORKER_PROTOCOL_VERSION,
                        "request_id": "self-check-hello",
                        "ok": True,
                        "result": {"status": "ready"},
                    },
                    {
                        "protocol": AREA_WORKER_PROTOCOL,
                        "version": AREA_WORKER_PROTOCOL_VERSION,
                        "request_id": "self-check-infer",
                        "ok": True,
                        "result": {"instances": []},
                    },
                ]
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="\n".join(json.dumps(item) for item in responses) + "\n",
                    stderr="",
                )

            with patch("fdm.release_manifest.subprocess.run", side_effect=run_worker):
                result = _probe_area_worker(root)

            self.assertEqual(result["mode"], "persistent")
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

            report = run_release_self_check(app_dir)

            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["profile"], "full")
            self.assertEqual(report["version"], "1.2.3")
            self.assertEqual(report["build_id"], "self-check-build")
            self.assertTrue(report["functional_checks"]["core_measurement"])
            self.assertTrue(report["functional_checks"]["qt_local_ipc"])
            self.assertTrue(report["functional_checks"]["pe:FiberDiameterMeasurement.exe"])

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

            with patch.dict(os.environ, {"FDM_SELF_CHECK_ROOT": str(app_dir)}), redirect_stdout(output):
                result = app.main(["fdm", "--self-check", "--json"])

            payload = json.loads(output.getvalue())
            self.assertEqual(result, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["build_id"], "self-check-build")


if __name__ == "__main__":
    unittest.main()
