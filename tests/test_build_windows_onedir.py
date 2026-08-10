from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_support import RuntimeProfileCheck
from build_windows_onedir import (
    build,
    check_windows_build_dependencies,
    main as onedir_main,
    run_packaged_self_check,
)


def _prepare_build_root(root: Path) -> None:
    (root / "packaging" / "pyinstaller").mkdir(parents=True, exist_ok=True)
    (root / "packaging" / "inno-setup").mkdir(parents=True, exist_ok=True)
    (root / "src" / "fdm").mkdir(parents=True, exist_ok=True)
    (root / "packaging" / "pyinstaller" / "fdm_onedir.spec").write_text("# stub\n", encoding="utf-8")
    (root / "src" / "fdm" / "version.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    templates_root = root / "runtime" / "content-templates"
    templates_root.mkdir(parents=True)
    (templates_root / "internal-template.xlsm").write_bytes(b"template")


class BuildWindowsOnedirTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dependency_patcher = patch(
            "build_windows_onedir.check_windows_build_dependencies",
            return_value=[],
        )
        self._dependency_mock = self._dependency_patcher.start()
        self.addCleanup(self._dependency_patcher.stop)

    def test_cli_public_release_excludes_both_private_components(self) -> None:
        with (
            patch.object(sys, "argv", ["build_windows_onedir.py", "--public-release"]),
            patch("build_windows_onedir.build", return_value=0) as build_mock,
        ):
            result = onedir_main()

        self.assertEqual(result, 0)
        self.assertTrue(build_mock.call_args.kwargs["exclude_area_models"])
        self.assertTrue(build_mock.call_args.kwargs["exclude_content_templates"])

    def test_spec_uses_flat_onedir_layout_for_all_executables(self) -> None:
        spec_payload = (
            PROJECT_ROOT / "packaging" / "pyinstaller" / "fdm_onedir.spec"
        ).read_text(encoding="utf-8")

        self.assertEqual(spec_payload.count('contents_directory="."'), 3)
        self.assertIn('name="FiberScreenshotTool"', spec_payload)
        self.assertIn("FDM_EXCLUDED_COMPONENTS", spec_payload)
        self.assertIn("collect_private_content_template_datas", spec_payload)
        self.assertIn("resolve_runtime_profile", spec_payload)
        self.assertIn('("tifffile", "openpyxl", "et_xmlfile")', spec_payload)
        self.assertIn("copy_metadata(distribution_name)", spec_payload)
        self.assertIn('collection_packages.add("et_xmlfile")', spec_payload)

    def test_dependency_probe_is_derived_from_inherited_runtime_profile(self) -> None:
        def find_spec(module_name: str):
            if module_name in {"openpyxl", "torch"}:
                return None
            return object()

        with patch(
            "build_windows_onedir.importlib.util.find_spec",
            side_effect=find_spec,
        ):
            self.assertEqual(
                check_windows_build_dependencies("core", root=PROJECT_ROOT),
                ["openpyxl"],
            )
            self.assertEqual(
                check_windows_build_dependencies("full", root=PROJECT_ROOT),
                ["openpyxl", "torch"],
            )

    def test_build_blocks_before_pyinstaller_when_profile_dependency_is_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_build_root(root)
            self._dependency_mock.return_value = ["openpyxl"]

            with (
                patch.dict(sys.modules, {"PyInstaller": ModuleType("PyInstaller")}),
                patch("build_windows_onedir.check_runtime_profile") as profile_mock,
                patch("build_windows_onedir.subprocess.run") as run_mock,
            ):
                result = build(
                    clean=True,
                    console=False,
                    bootloader_debug=False,
                    profile="full",
                    root=root,
                )

            self.assertEqual(result, 1)
            profile_mock.assert_not_called()
            run_mock.assert_not_called()
            self._dependency_mock.assert_called_once_with("full", root=root)

    def test_installer_displays_the_project_license_before_installation(self) -> None:
        installer_payload = (
            PROJECT_ROOT / "packaging" / "inno-setup" / "fdm_installer.iss"
        ).read_text(encoding="utf-8")

        self.assertIn('LicenseFile={#ProjectRoot}\\LICENSE', installer_payload)
        self.assertIn('#ifnexist ProjectRoot + "\\LICENSE"', installer_payload)
        self.assertIn('ScreenshotToolExeName "FiberScreenshotTool.exe"', installer_payload)
        self.assertIn('Name: "{group}\\Fiber 截图工具"', installer_payload)
        self.assertIn("procedure RemoveOwnedScreenshotAutostart();", installer_payload)
        self.assertIn("CompareText(Trim(CurrentCommand), QuotedCommand)", installer_payload)
        self.assertIn("CompareText(Trim(CurrentCommand), UnquotedCommand)", installer_payload)
        self.assertIn("ScreenshotAutostartValueName", installer_payload)

    def test_packaged_self_check_rejects_contradictory_or_invalid_error_payloads(self) -> None:
        cases = (
            ({"ok": True, "errors": ["worker failed"]}, ["worker failed"]),
            ({"ok": True, "errors": "worker failed"}, ["packaged self-check errors field must be a list of strings"]),
            ({"ok": True, "errors": [1]}, ["packaged self-check errors field must be a list of strings"]),
            ([{"ok": True}], ["packaged self-check JSON root must be an object"]),
        )
        for payload, expected_errors in cases:
            with self.subTest(payload=payload), TemporaryDirectory() as tmpdir:
                completed = subprocess.CompletedProcess(
                    [],
                    0,
                    stdout=json.dumps(payload),
                    stderr="",
                )
                with patch("build_windows_onedir.subprocess.run", return_value=completed):
                    errors = run_packaged_self_check(Path(tmpdir))

                self.assertEqual(errors, expected_errors)

    def test_build_passes_profile_to_pyinstaller_and_generates_release_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_build_root(root)
            app_dir = root / "dist" / "windows" / "FiberDiameterMeasurement"

            def run_pyinstaller(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
                app_dir.mkdir(parents=True, exist_ok=True)
                (app_dir / "FiberDiameterMeasurement.exe").write_bytes(b"main")
                (app_dir / "FiberAreaWorker.exe").write_bytes(b"worker")
                (app_dir / "FiberScreenshotTool.exe").write_bytes(b"screenshot")
                (app_dir / "runtime_assets.toml").write_text("schema_version = 1\n", encoding="utf-8")
                return subprocess.CompletedProcess([], 0)

            manifest_path = app_dir / "release-manifest.json"
            with (
                patch.dict(sys.modules, {"PyInstaller": ModuleType("PyInstaller")}),
                patch("build_windows_onedir.check_runtime_profile", return_value=RuntimeProfileCheck("core", (), ())),
                patch("build_windows_onedir.subprocess.run", side_effect=run_pyinstaller) as run_mock,
                patch("build_windows_onedir.write_release_manifest", return_value=manifest_path) as manifest_mock,
                patch("build_windows_onedir.run_packaged_self_check", return_value=[]),
            ):
                result = build(clean=True, console=False, bootloader_debug=False, profile="core", root=root)

            self.assertEqual(result, 0)
            environment = run_mock.call_args.kwargs["env"]
            self.assertEqual(environment["FDM_BUILD_PROFILE"], "core")
            self.assertEqual(environment["FDM_STRICT_ASSET_HASHES"], "0")
            self.assertEqual(environment["FDM_EXCLUDED_COMPONENTS"], "")
            manifest_mock.assert_called_once_with(
                app_dir,
                root,
                profile="core",
                clean_build=True,
                excluded_components=(),
                included_components=("content-templates",),
            )

    def test_build_warns_and_continues_on_hash_mismatch_by_default(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_build_root(root)
            app_dir = root / "dist" / "windows" / "FiberDiameterMeasurement"
            check = RuntimeProfileCheck("full", (), (), ("runtime/model.pth (hash mismatch)",))

            def run_pyinstaller(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
                app_dir.mkdir(parents=True, exist_ok=True)
                (app_dir / "FiberDiameterMeasurement.exe").write_bytes(b"main")
                (app_dir / "FiberAreaWorker.exe").write_bytes(b"worker")
                (app_dir / "FiberScreenshotTool.exe").write_bytes(b"screenshot")
                (app_dir / "runtime_assets.toml").write_text("schema_version = 1\n", encoding="utf-8")
                return subprocess.CompletedProcess([], 0)

            with (
                patch.dict(sys.modules, {"PyInstaller": ModuleType("PyInstaller")}),
                patch("build_windows_onedir.check_runtime_profile", return_value=check),
                patch("build_windows_onedir.subprocess.run", side_effect=run_pyinstaller) as run_mock,
                patch(
                    "build_windows_onedir.write_release_manifest",
                    return_value=app_dir / "release-manifest.json",
                ),
                patch("build_windows_onedir.run_packaged_self_check", return_value=[]),
            ):
                result = build(clean=True, console=False, bootloader_debug=False, profile="full", root=root)

            self.assertEqual(result, 0)
            self.assertEqual(run_mock.call_args.kwargs["env"]["FDM_STRICT_ASSET_HASHES"], "0")

    def test_build_can_exclude_both_private_components(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_build_root(root)
            app_dir = root / "dist" / "windows" / "FiberDiameterMeasurement"

            def run_pyinstaller(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
                app_dir.mkdir(parents=True, exist_ok=True)
                (app_dir / "FiberDiameterMeasurement.exe").write_bytes(b"main")
                (app_dir / "FiberAreaWorker.exe").write_bytes(b"worker")
                (app_dir / "FiberScreenshotTool.exe").write_bytes(b"screenshot")
                (app_dir / "runtime_assets.toml").write_text("schema_version = 1\n", encoding="utf-8")
                return subprocess.CompletedProcess([], 0)

            manifest_path = app_dir / "release-manifest.json"
            with (
                patch.dict(sys.modules, {"PyInstaller": ModuleType("PyInstaller")}),
                patch(
                    "build_windows_onedir.check_runtime_profile",
                    return_value=RuntimeProfileCheck("full", (), ()),
                ) as profile_mock,
                patch("build_windows_onedir.subprocess.run", side_effect=run_pyinstaller) as run_mock,
                patch("build_windows_onedir.write_release_manifest", return_value=manifest_path) as manifest_mock,
                patch("build_windows_onedir.run_packaged_self_check", return_value=[]),
            ):
                result = build(
                    clean=True,
                    console=False,
                    bootloader_debug=False,
                    profile="full",
                    exclude_area_models=True,
                    exclude_content_templates=True,
                    root=root,
                )

            self.assertEqual(result, 0)
            exclusions = ("area-models", "content-templates")
            self.assertEqual(run_mock.call_args.kwargs["env"]["FDM_EXCLUDED_COMPONENTS"], ",".join(exclusions))
            self.assertEqual(profile_mock.call_args.kwargs["excluded_groups"], exclusions)
            manifest_mock.assert_called_once_with(
                app_dir,
                root,
                profile="full",
                clean_build=True,
                excluded_components=exclusions,
                included_components=(),
            )

    def test_default_build_requires_private_content_templates(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_build_root(root)
            template = root / "runtime" / "content-templates" / "internal-template.xlsm"
            template.unlink()

            with (
                patch.dict(sys.modules, {"PyInstaller": ModuleType("PyInstaller")}),
                patch(
                    "build_windows_onedir.check_runtime_profile",
                    return_value=RuntimeProfileCheck("full", (), ()),
                ),
                patch("build_windows_onedir.subprocess.run") as run_mock,
            ):
                result = build(
                    clean=True,
                    console=False,
                    bootloader_debug=False,
                    profile="full",
                    root=root,
                )

            self.assertEqual(result, 1)
            run_mock.assert_not_called()

    def test_build_strict_hash_mode_blocks_before_pyinstaller(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_build_root(root)
            check = RuntimeProfileCheck("full", (), (), ("runtime/model.pth (hash mismatch)",))

            with (
                patch.dict(sys.modules, {"PyInstaller": ModuleType("PyInstaller")}),
                patch("build_windows_onedir.check_runtime_profile", return_value=check),
                patch("build_windows_onedir.subprocess.run") as run_mock,
            ):
                result = build(
                    clean=True,
                    console=False,
                    bootloader_debug=False,
                    profile="full",
                    strict_asset_hashes=True,
                    root=root,
                )

            self.assertEqual(result, 1)
            run_mock.assert_not_called()

    def test_build_blocks_when_packaged_self_check_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_build_root(root)
            app_dir = root / "dist" / "windows" / "FiberDiameterMeasurement"

            def run_pyinstaller(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
                app_dir.mkdir(parents=True, exist_ok=True)
                (app_dir / "FiberDiameterMeasurement.exe").write_bytes(b"main")
                (app_dir / "FiberAreaWorker.exe").write_bytes(b"worker")
                (app_dir / "FiberScreenshotTool.exe").write_bytes(b"screenshot")
                (app_dir / "runtime_assets.toml").write_text("schema_version = 1\n", encoding="utf-8")
                return subprocess.CompletedProcess([], 0)

            with (
                patch.dict(sys.modules, {"PyInstaller": ModuleType("PyInstaller")}),
                patch("build_windows_onedir.check_runtime_profile", return_value=RuntimeProfileCheck("full", (), ())),
                patch("build_windows_onedir.subprocess.run", side_effect=run_pyinstaller),
                patch("build_windows_onedir.write_release_manifest", return_value=app_dir / "release-manifest.json"),
                patch("build_windows_onedir.run_packaged_self_check", return_value=["worker failed"]),
            ):
                result = build(clean=True, console=False, bootloader_debug=False, profile="full", root=root)

            self.assertEqual(result, 1)

    def test_build_blocks_incomplete_full_profile_before_pyinstaller(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_build_root(root)
            check = RuntimeProfileCheck("full", ("runtime/area-models/missing.pth",), ())

            with (
                patch.dict(sys.modules, {"PyInstaller": ModuleType("PyInstaller")}),
                patch("build_windows_onedir.check_runtime_profile", return_value=check),
                patch("build_windows_onedir.subprocess.run") as run_mock,
            ):
                result = build(clean=True, console=False, bootloader_debug=False, profile="full", root=root)

            self.assertEqual(result, 1)
            run_mock.assert_not_called()

    def test_build_blocks_incomplete_asset_metadata_before_pyinstaller(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_build_root(root)
            check = RuntimeProfileCheck(
                "full",
                (),
                (),
                (),
                ("required_file 'runtime/area-infer/app/engine.py' is missing complete asset metadata",),
            )

            with (
                patch.dict(sys.modules, {"PyInstaller": ModuleType("PyInstaller")}),
                patch("build_windows_onedir.check_runtime_profile", return_value=check),
                patch("build_windows_onedir.subprocess.run") as run_mock,
            ):
                result = build(clean=True, console=False, bootloader_debug=False, profile="full", root=root)

            self.assertEqual(result, 1)
            run_mock.assert_not_called()

    def test_build_returns_failure_when_pyinstaller_process_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_build_root(root)

            with (
                patch.dict(sys.modules, {"PyInstaller": ModuleType("PyInstaller")}),
                patch("build_windows_onedir.check_runtime_profile", return_value=RuntimeProfileCheck("full", (), ())),
                patch(
                    "build_windows_onedir.subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, ["PyInstaller"]),
                ),
                patch("build_windows_onedir.write_release_manifest") as manifest_mock,
                patch("build_windows_onedir.run_packaged_self_check") as self_check_mock,
            ):
                result = build(clean=True, console=False, bootloader_debug=False, profile="full", root=root)

            self.assertEqual(result, 1)
            manifest_mock.assert_not_called()
            self_check_mock.assert_not_called()

    def test_clean_build_blocks_when_stale_dist_cannot_be_removed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _prepare_build_root(root)
            stale_dist = root / "dist" / "windows"
            stale_dist.mkdir(parents=True)

            with (
                patch.dict(sys.modules, {"PyInstaller": ModuleType("PyInstaller")}),
                patch("build_windows_onedir.check_runtime_profile", return_value=RuntimeProfileCheck("full", (), ())),
                patch("build_windows_onedir.shutil.rmtree", side_effect=OSError("locked stale dist")),
                patch("build_windows_onedir.subprocess.run") as run_mock,
            ):
                result = build(clean=True, console=False, bootloader_debug=False, profile="full", root=root)

            self.assertEqual(result, 1)
            run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
