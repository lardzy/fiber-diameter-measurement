from __future__ import annotations

import argparse
import os
import shlex
import shutil
import string
import subprocess
import sys
from pathlib import Path

from build_support import (
    BUILD_COMPONENT_AREA_MODELS,
    BUILD_COMPONENT_CONTENT_TEMPLATES,
    normalize_build_exclusions,
    read_app_version,
    validate_installer_release,
    write_installer_version_include,
)
from build_windows_onedir import build as build_onedir


def _discard_failed_installer(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        # The release is still blocked by the caller's non-zero return code.  Keep
        # the cleanup failure visible so a locked, untrusted artifact is not
        # mistaken for a release candidate.
        print(f"Unable to remove failed installer output {path}: {exc}", file=sys.stderr)


def _render_hook_command(label: str, template: str, installer_path: Path) -> list[str]:
    try:
        fields = [
            field_name
            for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(template)
            if field_name is not None
        ]
    except ValueError as exc:
        raise ValueError(f"{label} command template is invalid: {exc}") from exc
    if "file" not in fields:
        raise ValueError(f"{label} command must contain the {{file}} placeholder")
    unsupported_fields = sorted({field_name for field_name in fields if field_name != "file"})
    if unsupported_fields:
        raise ValueError(
            f"{label} command contains unsupported placeholders: " + ", ".join(unsupported_fields)
        )
    try:
        rendered = template.format(file=str(installer_path))
    except (AttributeError, IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"{label} command template is invalid: {exc}") from exc
    command = shlex.split(rendered)
    if not command:
        raise ValueError(f"{label} command is empty")
    if not any(str(installer_path) in argument for argument in command):
        raise ValueError(f"{label} command did not resolve {{file}} to the installer path")
    return command


def _installer_is_nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _installer_variant_suffix(excluded_components: tuple[str, ...]) -> str:
    excluded = set(excluded_components)
    if excluded == {BUILD_COMPONENT_AREA_MODELS, BUILD_COMPONENT_CONTENT_TEMPLATES}:
        return "-public"
    if BUILD_COMPONENT_AREA_MODELS in excluded:
        return "-no-area-models"
    if BUILD_COMPONENT_CONTENT_TEMPLATES in excluded:
        return "-no-content-templates"
    return ""


def find_inno_setup_compiler() -> str | None:
    env_override = os.environ.get("ISCC_EXE", "").strip()
    if env_override:
        candidate = Path(env_override).expanduser()
        if candidate.exists():
            return str(candidate)

    for executable_name in ("ISCC.exe", "ISCC", "iscc.exe", "iscc"):
        resolved = shutil.which(executable_name)
        if resolved:
            return resolved

    for env_name in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env_name, "").strip()
        if not base:
            continue
        for folder_name in ("Inno Setup 6", "Inno Setup 5"):
            candidate = Path(base) / folder_name / "ISCC.exe"
            if candidate.exists():
                return str(candidate)

    return None


def build_installer(
    *,
    root: Path,
    sync_only: bool = False,
    compiler_path: str | None = None,
    sign_command: str | None = None,
    verify_signature_command: str | None = None,
    strict_asset_hashes: bool = False,
    strict_release: bool = False,
    rebuild_onedir: bool = False,
    exclude_area_models: bool = False,
    exclude_content_templates: bool = False,
) -> int:
    iss_path = root / "packaging" / "inno-setup" / "fdm_installer.iss"
    app_dir = root / "dist" / "windows" / "FiberDiameterMeasurement"
    excluded_components = normalize_build_exclusions(
        component
        for component, excluded in (
            (BUILD_COMPONENT_AREA_MODELS, exclude_area_models),
            (BUILD_COMPONENT_CONTENT_TEMPLATES, exclude_content_templates),
        )
        if excluded
    )

    if not iss_path.exists():
        print(f"Inno Setup script not found: {iss_path}", file=sys.stderr)
        return 1

    version_include = write_installer_version_include(root)
    version = read_app_version(root)
    print(f"Synchronized installer version include: {version_include} -> {version}")

    if sync_only:
        print("Sync only mode: version.auto.iss has been refreshed from src/fdm/version.py")
        return 0

    if rebuild_onedir:
        print("Building a clean full-profile onedir package before the installer...")
        onedir_result = build_onedir(
            clean=True,
            console=False,
            bootloader_debug=False,
            profile="full",
            strict_asset_hashes=strict_asset_hashes,
            exclude_area_models=exclude_area_models,
            exclude_content_templates=exclude_content_templates,
            root=root,
        )
        if onedir_result != 0:
            print("Onedir prerequisite build failed; installer compilation was not started.", file=sys.stderr)
            return 1

    if not app_dir.exists():
        print(
            "PyInstaller output not found. Run this command without --reuse-onedir, or build it first:\n"
            "  python scripts/build_windows_onedir.py",
            file=sys.stderr,
        )
        return 1

    release_warnings: list[str] = []
    release_errors = validate_installer_release(
        root,
        app_dir,
        profile="full",
        strict_asset_hashes=strict_asset_hashes,
        strict_release=strict_release,
        excluded_components=excluded_components,
        include_content_templates=not exclude_content_templates,
        warnings=release_warnings,
    )
    if release_warnings:
        print("Internal release warnings:", file=sys.stderr)
        for warning in release_warnings:
            print(f"  - {warning}", file=sys.stderr)
    if release_errors:
        heading = "Formal release checks failed:" if strict_release else "Installer package validation failed:"
        print(heading, file=sys.stderr)
        for error in release_errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    resolved_compiler = compiler_path or find_inno_setup_compiler()
    if not resolved_compiler:
        print(
            "Inno Setup compiler not found. A formal release requires ISCC output; set ISCC_EXE or --compiler.",
            file=sys.stderr,
        )
        return 2

    command = [resolved_compiler, str(iss_path)]
    output_dir = root / "dist" / "installer"
    compiled_output = output_dir / f"fiber-diameter-measurement-setup-{version}.exe"
    expected_output = output_dir / (
        f"fiber-diameter-measurement-setup-{version}"
        f"{_installer_variant_suffix(excluded_components)}.exe"
    )
    for stale_output in {compiled_output, expected_output}:
        try:
            stale_output.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Unable to remove stale installer output {stale_output}: {exc}", file=sys.stderr)
            return 1
    print("Running Inno Setup:")
    print(" ".join(command))
    try:
        subprocess.run(command, cwd=root, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Inno Setup compilation failed: {exc}", file=sys.stderr)
        return 1

    if not _installer_is_nonempty(compiled_output):
        print(
            f"ISCC returned successfully but did not produce a non-empty installer: {compiled_output}",
            file=sys.stderr,
        )
        return 1
    if compiled_output != expected_output:
        try:
            compiled_output.replace(expected_output)
        except OSError as exc:
            print(f"Unable to name installer variant {expected_output}: {exc}", file=sys.stderr)
            _discard_failed_installer(compiled_output)
            return 1

    signing_template = str(sign_command or os.environ.get("FDM_SIGN_COMMAND", "")).strip()
    verification_template = str(
        verify_signature_command or os.environ.get("FDM_VERIFY_SIGNATURE_COMMAND", "")
    ).strip()
    for label, template in (
        ("Authenticode signing", signing_template),
        ("Authenticode verification", verification_template),
    ):
        if not template:
            continue
        try:
            hook_command = _render_hook_command(label, template, expected_output)
            subprocess.run(hook_command, cwd=root, check=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"{label} failed: {exc}", file=sys.stderr)
            _discard_failed_installer(expected_output)
            return 1
        if not _installer_is_nonempty(expected_output):
            print(f"{label} did not leave a non-empty installer: {expected_output}", file=sys.stderr)
            _discard_failed_installer(expected_output)
            return 1

    print("\nInstaller build completed.")
    print(f"Output directory: {output_dir}")
    print(f"Installer: {expected_output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync installer version metadata from src/fdm/version.py and optionally build the Inno Setup installer.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only refresh packaging/inno-setup/version.auto.iss from src/fdm/version.py.",
    )
    parser.add_argument(
        "--compiler",
        default="",
        help="Optional full path to ISCC.exe. If omitted, the script will try PATH / common install locations / ISCC_EXE.",
    )
    parser.add_argument(
        "--sign-command",
        default="",
        help="Optional Authenticode command template containing {file}; also accepted via FDM_SIGN_COMMAND.",
    )
    parser.add_argument(
        "--verify-signature-command",
        default="",
        help="Optional signature verification command template containing {file}; also accepted via FDM_VERIFY_SIGNATURE_COMMAND.",
    )
    parser.add_argument(
        "--strict-asset-hashes",
        action="store_true",
        help="Fail when source runtime files differ from the hashes pinned in runtime_assets.toml.",
    )
    parser.add_argument(
        "--strict-release",
        action="store_true",
        help="Require a clean Git worktree and a clean-build manifest for a formal release.",
    )
    parser.add_argument(
        "--reuse-onedir",
        action="store_true",
        help="Reuse dist/windows/FiberDiameterMeasurement instead of rebuilding the onedir prerequisite.",
    )
    parser.add_argument(
        "--exclude-area-models",
        action="store_true",
        help="Exclude private runtime/area-models from both onedir and installer output.",
    )
    parser.add_argument(
        "--exclude-content-templates",
        action="store_true",
        help="Exclude private runtime/content-templates from both onedir and installer output.",
    )
    parser.add_argument(
        "--public-release",
        action="store_true",
        help=(
            "Build a public installer without private runtime/area-models "
            "or runtime/content-templates; the output uses the -public suffix."
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    compiler = args.compiler.strip() or None
    return build_installer(
        root=root,
        sync_only=args.sync_only,
        compiler_path=compiler,
        sign_command=args.sign_command.strip() or None,
        verify_signature_command=args.verify_signature_command.strip() or None,
        strict_asset_hashes=args.strict_asset_hashes,
        strict_release=args.strict_release,
        rebuild_onedir=not args.reuse_onedir,
        exclude_area_models=args.exclude_area_models or args.public_release,
        exclude_content_templates=args.exclude_content_templates or args.public_release,
    )


if __name__ == "__main__":
    raise SystemExit(main())
