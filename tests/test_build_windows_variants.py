from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_support import PACKAGED_AREA_MODEL_FILENAMES, write_release_manifest
from build_windows_installer import main as installer_main
from build_windows_variants import build_area_variants, derive_without_area_models
from fdm.release_manifest import verify_release_manifest


def full_package(root):
    (root / "src/fdm").mkdir(parents=True, exist_ok=True)
    (root / "src/fdm/version.py").write_text('__version__ = "0.4.14"\n')
    model_names = sorted(PACKAGED_AREA_MODEL_FILENAMES)
    model_paths = ["runtime/area-models/" + name for name in model_names]
    config = '''schema_version = 1
[profiles.full]
groups = ["area-models"]
required_python_modules = ["numpy", "torch", "onnxruntime"]
features = ["measurement", "capture", "magic-segmentation", "area-inference"]
[groups.area-models]
root = "runtime/area-models"
filter = "area-models"
required_files = ''' + json.dumps(model_paths) + "\n"
    for token in model_paths:
        data = token.encode()
        config += '\n[[assets]]\n' + f'source = "{token}"\ntarget = "{token}"\ngroup = "area-models"\nfeature = "area-inference"\nrequired = true\nsize = {len(data)}\nsha256 = "{hashlib.sha256(data).hexdigest()}"\nlicense = "test"\n'
    (root / "runtime_assets.toml").write_text(config)
    app = root / "dist/windows/FiberDiameterMeasurement"
    files = {"FiberDiameterMeasurement.exe": b"shared-program", "FiberAreaWorker.exe": b"shared-worker",
             "FiberScreenshotTool.exe": b"shared-screenshot", "torch/torch.dll": b"shared-dependency",
             "runtime/segment-anything/encoder.onnx": b"shared-magic",
             "runtime/content-templates/sheet.xlt": b"user-template-pages",
             "runtime_assets.toml": config.encode()}
    files.update({path: path.encode() for path in model_paths})
    for token, data in files.items():
        path = app / token
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_release_manifest(app, root, profile="full", clean_build=True, build_id="shared123",
        source_commit="commit123", source_dirty_entries=[], included_components=("content-templates",))
    return app


def test_derived_variant_omits_only_weights_and_feature_and_keeps_source_unchanged(tmp_path):
    source = full_package(tmp_path)
    original = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    target = tmp_path / "derived"
    result = derive_without_area_models(tmp_path, source, target)
    assert result["source_build_id"] == "shared123"
    assert result["build_id"] != "shared123"
    assert {Path(p).name for p in result["omitted_files"]} == PACKAGED_AREA_MODEL_FILENAMES
    assert not (target / "runtime/area-models").exists()
    new_manifest = json.loads((target / "release-manifest.json").read_text())
    assert new_manifest["features"] == ["measurement", "capture", "magic-segmentation"]
    assert new_manifest["included_components"] == ["content-templates"]
    assert new_manifest["source_commit"] == "commit123"
    assert new_manifest["shared_program_build_id"] == "shared123"
    for name, data in original.items():
        assert (source / name).read_bytes() == data
        if name not in ("build-id.txt", "release-manifest.json") and not name.startswith("runtime/area-models/"):
            assert (target / name).read_bytes() == data
    assert verify_release_manifest(target, expected_excluded_components=["area-models"], reject_extra_files=True)["ok"]


def test_derive_copy_fallback_and_incomplete_source_rejection(tmp_path):
    source = full_package(tmp_path)
    with patch("build_windows_variants.os.link", side_effect=OSError("cross-device")):
        result = derive_without_area_models(tmp_path, source, tmp_path / "copied")
    assert result["copied_files"] > 0 and result["hardlinked_files"] == 0
    (source / "FiberDiameterMeasurement.exe").write_bytes(b"bad")
    with pytest.raises(ValueError, match="Shared package failed validation"):
        derive_without_area_models(tmp_path, source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_derived_variant_keeps_user_added_template_outside_old_manifest(tmp_path):
    source = full_package(tmp_path)
    extra = source / "runtime/content-templates/新增报告.xlt"
    extra.write_bytes(b"extra user report pages")
    derive_without_area_models(tmp_path, source, tmp_path / "derived")
    assert (tmp_path / "derived/runtime/content-templates/新增报告.xlt").read_bytes() == extra.read_bytes()


def test_two_installers_build_program_once_and_do_not_overwrite_each_other(tmp_path):
    root = tmp_path
    full_package(root)
    iss = root / "packaging/inno-setup/fdm_installer.iss"
    iss.parent.mkdir(parents=True)
    iss.write_text("; stub")
    language = iss.parent / "languages/ChineseSimplified.isl"
    language.parent.mkdir()
    language.write_text("; fixture")
    calls = []
    manifests = []
    output = root / "dist/installer"
    output.mkdir(parents=True)
    (output / "older-installer.exe").write_bytes(b"keep")

    def compile_installer(command, **kw):
        calls.append(command)
        folder = root / "dist/windows/FiberDiameterMeasurement"
        name = "fiber-diameter-measurement-setup-0.4.14"
        for arg in command:
            if arg.startswith("/DMyAppSourceDir="):
                folder = Path(arg.split("=", 1)[1])
            if arg.startswith("/DMyAppOutputBaseFilename="):
                name = arg.split("=", 1)[1]
        manifest = json.loads((folder / "release-manifest.json").read_text())
        manifests.append(manifest)
        assert verify_release_manifest(folder, reject_extra_files=True)["ok"]
        (output / (name + ".exe")).write_bytes(("installer:" + name).encode())
        return subprocess.CompletedProcess(command, 0)

    with patch("build_windows_installer.build_onedir", return_value=0) as program, \
         patch("build_windows_installer.validate_installer_release", return_value=[]), \
         patch("build_windows_installer.get_git_commit", return_value="commit123"), \
         patch("build_windows_installer.get_dirty_worktree_entries", return_value=[]), \
         patch("build_windows_installer.run_packaged_self_check", return_value=[]) as checks, \
         patch("build_windows_installer.subprocess.run", side_effect=compile_installer):
        assert build_area_variants(root=root, compiler_path="ISCC.exe") == 0
    program.assert_called_once()
    assert not program.call_args.kwargs["exclude_area_models"]
    assert not program.call_args.kwargs["exclude_content_templates"]
    assert len(calls) == 2
    checks.assert_called_once()  # full build already runs its packaged check
    assert "area-inference" in manifests[0]["features"]
    assert "area-inference" not in manifests[1]["features"]
    assert (output / "fiber-diameter-measurement-setup-0.4.14.exe").is_file()
    assert (output / "fiber-diameter-measurement-setup-0.4.14-no-area-models.exe").is_file()
    assert (output / "older-installer.exe").read_bytes() == b"keep"
    summary = json.loads((output / "area-variants-build.json").read_text())
    assert summary["ok"] and len(summary["variants"]) == 2
    assert summary["source_commit"] == "commit123"
    assert not list((root / "build/installer-variants").iterdir())


def test_double_variant_cli_and_bat_forward_options():
    with patch.object(sys, "argv", ["build_windows_installer.py", "--both-area-variants", "--no-clean"]), \
         patch("build_windows_variants.build_area_variants", return_value=0) as paired:
        assert installer_main() == 0
        assert paired.call_args.kwargs["clean"] is False
        assert paired.call_args.kwargs["rebuild_onedir"] is True
    for args in (("--public-release",), ("--exclude-area-models",), ("--sync-only",)):
        with patch.object(sys, "argv", ["build_windows_installer.py", "--both-area-variants", *args]), pytest.raises(SystemExit):
            installer_main()
    assert '--both-area-variants %*' in (PROJECT_ROOT / "scripts/build_windows_both_installers.bat").read_text()


def test_missing_compiler_does_not_start_program_build(tmp_path):
    with patch("build_windows_installer.find_inno_setup_compiler", return_value=None), \
         patch("build_windows_installer.build_onedir") as program:
        assert build_area_variants(root=tmp_path) == 2
    program.assert_not_called()
