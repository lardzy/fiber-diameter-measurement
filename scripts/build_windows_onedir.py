from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from build_support import (
    PACKAGED_AREA_MODEL_FILENAMES,
    PACKAGED_SEGMENT_ANYTHING_DIRS,
    PACKAGED_SEGMENT_ANYTHING_FILENAMES,
    check_runtime_profile,
    write_release_manifest,
    write_installer_version_include,
)


def check_area_runtime_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name, package_name in (
        ("PIL", "Pillow"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
    ):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    return missing


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
    root: Path | None = None,
) -> int:
    root = root or Path(__file__).resolve().parents[1]
    spec_path = root / "packaging" / "pyinstaller" / "fdm_onedir.spec"
    dist_path = root / "dist" / "windows"
    work_path = root / "build" / "pyinstaller"

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
        profile_check = check_runtime_profile(root, profile)
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
        print(
            f"Runtime profile {profile!r} contains files with unexpected hashes:\n  "
            + "\n  ".join(profile_check.hash_mismatches),
            file=sys.stderr,
        )
        return 1
    if profile_check.metadata_errors:
        print(
            f"Runtime profile {profile!r} has incomplete or invalid asset metadata:\n  "
            + "\n  ".join(profile_check.metadata_errors),
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
    print(f"Main executable: {app_dir / 'FiberDiameterMeasurement.exe'}")
    print(f"Area worker: {app_dir / 'FiberAreaWorker.exe'}")
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
    args = parser.parse_args()
    return build(
        clean=not args.no_clean,
        console=args.console,
        bootloader_debug=args.bootloader_debug,
        profile=args.profile,
    )


if __name__ == "__main__":
    raise SystemExit(main())
