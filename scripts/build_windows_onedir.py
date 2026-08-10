from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from build_support import (
    BUILD_COMPONENT_AREA_MODELS,
    BUILD_COMPONENT_CONTENT_TEMPLATES,
    PACKAGED_AREA_MODEL_FILENAMES,
    PACKAGED_SEGMENT_ANYTHING_DIRS,
    PACKAGED_SEGMENT_ANYTHING_FILENAMES,
    check_runtime_profile,
    normalize_build_exclusions,
    private_content_template_files,
    resolve_runtime_profile,
    summarize_runtime_hash_mismatches,
    write_release_manifest,
    write_installer_version_include,
)


_DISTRIBUTION_NAMES_BY_MODULE = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
}


def check_windows_build_dependencies(
    profile: str,
    *,
    root: Path | None = None,
) -> list[str]:
    """Return dependencies declared by the selected runtime profile that are unavailable."""

    project_root = root or Path(__file__).resolve().parents[1]
    resolved = resolve_runtime_profile(project_root, str(profile or "").strip().lower())
    missing: list[str] = []
    for module_name in resolved.required_python_modules:
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(_DISTRIBUTION_NAMES_BY_MODULE.get(module_name, module_name))
    return missing


def check_area_runtime_dependencies(*, root: Path | None = None) -> list[str]:
    """Backward-compatible full-build dependency probe."""

    return check_windows_build_dependencies("full", root=root)


def check_magic_segment_runtime_assets(root: Path) -> list[str]:
    runtime_root = root / "runtime" / "segment-anything"
    missing: list[str] = []
    expected_files = {
        "edge_sam": ("edge_sam_encoder.onnx", "edge_sam_decoder.onnx"),
        "edge_sam_3x": ("edge_sam_3x_encoder.onnx", "edge_sam_3x_decoder.onnx"),
    }
    for folder_name in sorted(PACKAGED_SEGMENT_ANYTHING_DIRS):
        for filename in expected_files.get(folder_name, ()):
            if filename not in PACKAGED_SEGMENT_ANYTHING_FILENAMES:
                continue
            if not (runtime_root / folder_name / filename).exists():
                missing.append(f"{folder_name}/{filename}")
    return missing


def check_area_model_runtime_assets(root: Path) -> list[str]:
    runtime_root = root / "runtime" / "area-models"
    missing: list[str] = []
    for filename in sorted(PACKAGED_AREA_MODEL_FILENAMES):
        if not (runtime_root / filename).exists():
            missing.append(filename)
    return missing


def run_packaged_self_check(app_dir: Path) -> list[str]:
    executable = app_dir / "FiberDiameterMeasurement.exe"
    try:
        completed = subprocess.run(
            [str(executable), "--self-check", "--json"],
            cwd=app_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"unable to execute packaged self-check: {exc}"]
    try:
        payload = json.loads(completed.stdout.strip())
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return [
            f"packaged self-check did not return JSON (rc={completed.returncode}): {exc}; "
            f"stderr={completed.stderr[-1000:]}"
        ]
    if not isinstance(payload, dict):
        return ["packaged self-check JSON root must be an object"]
    errors_payload = payload.get("errors", [])
    if not isinstance(errors_payload, list) or any(not isinstance(item, str) for item in errors_payload):
        return ["packaged self-check errors field must be a list of strings"]
    errors = [item for item in errors_payload if item]
    if errors:
        return errors
    if completed.returncode != 0 or payload.get("ok") is not True:
        return errors or [f"packaged self-check failed with exit code {completed.returncode}"]
    return []


def build(
    clean: bool,
    *,
    console: bool,
    bootloader_debug: bool,
    profile: str = "full",
    strict_asset_hashes: bool = False,
    exclude_area_models: bool = False,
    exclude_content_templates: bool = False,
    root: Path | None = None,
) -> int:
    root = root or Path(__file__).resolve().parents[1]
    spec_path = root / "packaging" / "pyinstaller" / "fdm_onedir.spec"
    dist_path = root / "dist" / "windows"
    work_path = root / "build" / "pyinstaller"
    excluded_components = normalize_build_exclusions(
        component
        for component, excluded in (
            (BUILD_COMPONENT_AREA_MODELS, exclude_area_models),
            (BUILD_COMPONENT_CONTENT_TEMPLATES, exclude_content_templates),
        )
        if excluded
    )

    if not spec_path.exists():
        print(f"Spec file not found: {spec_path}", file=sys.stderr)
        return 1

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print(
            "PyInstaller is not installed in the current environment.\n"
            "Please run: pip install pyinstaller",
            file=sys.stderr,
        )
        return 1

    try:
        missing_image_dependencies = check_windows_build_dependencies(profile, root=root)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Invalid runtime asset profile {profile!r}: {exc}", file=sys.stderr)
        return 1
    if missing_image_dependencies:
        print(
            "Windows runtime-profile build dependencies are missing: "
            + ", ".join(missing_image_dependencies),
            file=sys.stderr,
        )
        return 1

    try:
        profile_check = check_runtime_profile(
            root,
            profile,
            excluded_groups=excluded_components,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Invalid runtime asset profile {profile!r}: {exc}", file=sys.stderr)
        return 1
    if profile_check.missing_files:
        print(
            f"Runtime profile {profile!r} is incomplete. Missing files:\n  "
            + "\n  ".join(profile_check.missing_files),
            file=sys.stderr,
        )
        return 1
    if profile_check.missing_python_modules:
        print(
            f"Runtime profile {profile!r} is missing build dependencies: "
            + ", ".join(profile_check.missing_python_modules),
            file=sys.stderr,
        )
        return 1
    if profile_check.hash_mismatches:
        if strict_asset_hashes:
            print(
                f"Runtime profile {profile!r} contains files with unexpected hashes:\n  "
                + "\n  ".join(profile_check.hash_mismatches),
                file=sys.stderr,
            )
            return 1
        print(
            f"Warning: runtime profile {profile!r} contains files whose hashes differ from "
            "runtime_assets.toml. Internal-build mode will package the current files as-is; "
            "Windows line endings may account for some differences.\n  "
            + summarize_runtime_hash_mismatches(profile_check.hash_mismatches),
            file=sys.stderr,
        )
    if profile_check.metadata_errors:
        print(
            f"Runtime profile {profile!r} has incomplete or invalid asset metadata:\n  "
            + "\n  ".join(profile_check.metadata_errors),
            file=sys.stderr,
        )
        return 1
    content_template_files = private_content_template_files(root)
    if not exclude_content_templates and not content_template_files:
        print(
            "Private content templates were requested, but runtime/content-templates contains no "
            "packageable files. Add the internal templates or use --exclude-content-templates.",
            file=sys.stderr,
        )
        return 1

    if clean:
        for stale_path in (dist_path, work_path):
            try:
                if stale_path.exists():
                    shutil.rmtree(stale_path)
            except OSError as exc:
                print(f"Unable to remove stale build output {stale_path}: {exc}", file=sys.stderr)
                return 1
            if stale_path.exists():
                print(f"Stale build output still exists after cleanup: {stale_path}", file=sys.stderr)
                return 1

    dist_path.mkdir(parents=True, exist_ok=True)
    work_path.mkdir(parents=True, exist_ok=True)
    installer_version_file = write_installer_version_include(root)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(dist_path),
        "--workpath",
        str(work_path),
        str(spec_path),
    ]

    print("Running PyInstaller:")
    print(" ".join(command))
    env = os.environ.copy()
    env["FDM_PYINSTALLER_CONSOLE"] = "1" if console else "0"
    env["FDM_PYINSTALLER_BOOTLOADER_DEBUG"] = "1" if bootloader_debug else "0"
    env["FDM_BUILD_PROFILE"] = profile
    env["FDM_STRICT_ASSET_HASHES"] = "1" if strict_asset_hashes else "0"
    env["FDM_EXCLUDED_COMPONENTS"] = ",".join(excluded_components)
    try:
        subprocess.run(command, cwd=root, check=True, env=env)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"PyInstaller build failed: {exc}", file=sys.stderr)
        return 1

    app_dir = dist_path / "FiberDiameterMeasurement"
    missing_outputs = [
        path.name
        for path in (
            app_dir / "FiberDiameterMeasurement.exe",
            app_dir / "FiberAreaWorker.exe",
            app_dir / "FiberScreenshotTool.exe",
            app_dir / "runtime_assets.toml",
        )
        if not path.is_file()
    ]
    if missing_outputs:
        print("PyInstaller completed without required outputs: " + ", ".join(missing_outputs), file=sys.stderr)
        return 1
    manifest_path = write_release_manifest(
        app_dir,
        root,
        profile=profile,
        clean_build=clean,
        excluded_components=excluded_components,
        included_components=(
            ()
            if exclude_content_templates
            else (BUILD_COMPONENT_CONTENT_TEMPLATES,)
        ),
    )
    self_check_errors = run_packaged_self_check(app_dir)
    if self_check_errors:
        print(
            "Packaged runtime self-check failed:\n  " + "\n  ".join(self_check_errors),
            file=sys.stderr,
        )
        return 1
    print("\nBuild completed.")
    print(f"Output directory: {app_dir}")
    print(f"Console mode: {'on' if console else 'off'}")
    print(f"Bootloader debug: {'on' if bootloader_debug else 'off'}")
    print(f"Runtime profile: {profile}")
    print(f"Source asset hash policy: {'strict' if strict_asset_hashes else 'warn only'}")
    print(f"Excluded components: {', '.join(excluded_components) if excluded_components else 'none'}")
    if not exclude_content_templates:
        print(f"Private content templates: {len(content_template_files)} files")
    print(f"Main executable: {app_dir / 'FiberDiameterMeasurement.exe'}")
    print(f"Area worker: {app_dir / 'FiberAreaWorker.exe'}")
    print(f"Screenshot companion: {app_dir / 'FiberScreenshotTool.exe'}")
    print(f"Runtime assets: {app_dir / 'runtime'}")
    print(f"Release manifest: {manifest_path}")
    print(f"Installer version include: {installer_version_file}")
    print("Use this directory as the source folder for your Inno Setup installer.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Windows onedir package with PyInstaller.")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep the existing dist/windows and build/pyinstaller contents before building.",
    )
    parser.add_argument(
        "--profile",
        choices=("core", "full"),
        default="full",
        help="Runtime asset profile. Formal installers require a clean full-profile build.",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Build a console-enabled executable for troubleshooting startup issues.",
    )
    parser.add_argument(
        "--bootloader-debug",
        action="store_true",
        help="Enable PyInstaller bootloader debug output in the built executable.",
    )
    parser.add_argument(
        "--strict-asset-hashes",
        action="store_true",
        help="Fail when source runtime files differ from the hashes pinned in runtime_assets.toml.",
    )
    parser.add_argument(
        "--exclude-area-models",
        action="store_true",
        help="Build without runtime/area-models and disable packaged area inference.",
    )
    parser.add_argument(
        "--exclude-content-templates",
        action="store_true",
        help="Build without runtime/content-templates.",
    )
    parser.add_argument(
        "--public-release",
        action="store_true",
        help=(
            "Build a public package without private runtime/area-models "
            "or runtime/content-templates; packaged area inference is disabled."
        ),
    )
    args = parser.parse_args()
    return build(
        clean=not args.no_clean,
        console=args.console,
        bootloader_debug=args.bootloader_debug,
        profile=args.profile,
        strict_asset_hashes=args.strict_asset_hashes,
        exclude_area_models=args.exclude_area_models or args.public_release,
        exclude_content_templates=args.exclude_content_templates or args.public_release,
    )


if __name__ == "__main__":
    raise SystemExit(main())
