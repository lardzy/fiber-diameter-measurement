"""Produce both internal installer variants from one verified program build."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time

from build_support import (
    BUILD_COMPONENT_AREA_MODELS, BUILD_COMPONENT_CONTENT_TEMPLATES,
    read_app_version, resolve_runtime_profile, write_release_manifest,
)
from fdm.atomic_io import atomic_write_text
from fdm.release_manifest import (
    BUILD_ID_FILENAME, RELEASE_MANIFEST_FILENAME, is_editable_template_path,
    sha256_file, verify_release_manifest,
)


def derive_without_area_models(project_root: Path, source: Path, destination: Path) -> dict:
    """Keep compiled code/dependencies and templates; omit only area weights.

    The temporary variant uses hard links where supported. Both release metadata
    files are written independently, so changing the variant identity never
    mutates the shared source. The caller must not edit linked program files.
    """
    if destination.exists():
        raise ValueError("Variant destination must be new")
    verified = verify_release_manifest(source, expected_profile="full",
        expected_version=read_app_version(project_root), reject_extra_files=True)
    if not verified["ok"]:
        raise ValueError("Shared package failed validation: " + "; ".join(verified["errors"]))
    original = json.loads((source / RELEASE_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    exclusions = tuple(original.get("excluded_components", []))
    if BUILD_COMPONENT_AREA_MODELS in exclusions or "area-inference" not in original["features"]:
        raise ValueError("Both variants require an onedir that includes area models")
    full = resolve_runtime_profile(project_root, "full", excluded_groups=exclusions)
    reduced = resolve_runtime_profile(project_root, "full", excluded_groups=(*exclusions, BUILD_COMPONENT_AREA_MODELS))
    if full.required_python_modules != reduced.required_python_modules:
        raise ValueError("Area variants now have different Python dependencies; a shared build is not valid")
    destination.mkdir(parents=True)
    omitted, linked, copied = [], 0, 0
    tokens = {str(entry["path"]) for entry in original["files"]}
    # Editable templates may legitimately be added to a reusable onedir after
    # its manifest was generated; the full installer includes them too.
    tokens.update(path.relative_to(source).as_posix() for path in source.rglob("*")
                  if path.is_file() and is_editable_template_path(path.relative_to(source).as_posix()))
    for token in sorted(tokens):
        relative = PurePosixPath(token.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid package inventory path: {token}")
        if relative.parts[:2] == ("runtime", "area-models"):
            omitted.append(token)
            continue
        if token in (BUILD_ID_FILENAME, RELEASE_MANIFEST_FILENAME):
            continue
        src, dst = source.joinpath(*relative.parts), destination.joinpath(*relative.parts)
        for item in (src, *src.parents):
            if item == source.parent:
                break
            if item.is_symlink() or getattr(item, "is_junction", lambda: False)():
                raise ValueError(f"Linked source paths are not valid release inputs: {token}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            shutil.copy2(src, dst)
            copied += 1
    if not omitted:
        raise ValueError("Shared package contains no area model files")
    path = write_release_manifest(destination, project_root, profile="full",
        clean_build=bool(original["clean_build"]), source_commit=original["source_commit"],
        source_dirty_entries=list(original["source_dirty_entries"]),
        excluded_components=(*exclusions, BUILD_COMPONENT_AREA_MODELS),
        included_components=original.get("included_components", []))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["dependency_versions"] = original["dependency_versions"]
    manifest["shared_program_build_id"] = original["build_id"]
    atomic_write_text(path, json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    # Retained executable, DLL and template bytes must be identical, not merely
    # a manifest with an area flag switched off.
    for entry in manifest["files"]:
        if entry["path"] == BUILD_ID_FILENAME:
            continue
        src, dst = source / entry["path"], destination / entry["path"]
        if src.stat().st_size != dst.stat().st_size or (
            not os.path.samefile(src, dst) and sha256_file(src) != sha256_file(dst)
        ):
            raise ValueError(f"Variant changed a shared file: {entry['path']}")
    report = verify_release_manifest(destination, expected_profile="full",
        expected_excluded_components=(*exclusions, BUILD_COMPONENT_AREA_MODELS), reject_extra_files=True)
    if not report["ok"] or "area-inference" in report["features"] or (destination / "runtime/area-models").exists():
        raise ValueError("Derived package failed validation: " + "; ".join(report["errors"]))
    return {"source_build_id": original["build_id"], "build_id": manifest["build_id"],
            "omitted_files": omitted, "hardlinked_files": linked, "copied_files": copied}


def build_area_variants(*, root: Path, compiler_path=None, sign_command=None,
                       verify_signature_command=None, strict_asset_hashes=False,
                       strict_release=False, rebuild_onedir=True, clean=True,
                       exclude_content_templates=False) -> int:
    from build_windows_installer import build_installer, find_inno_setup_compiler, _installer_variant_suffix

    compiler = compiler_path or find_inno_setup_compiler()
    if not compiler:
        print("Inno Setup compiler not found; shared program build was not started.")
        return 2
    root = root.resolve()
    source = root / "dist/windows/FiberDiameterMeasurement"
    output = root / "dist/installer"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "area-variants-build.json"
    summary = {"ok": False, "version": read_app_version(root), "variants": [], "timings": {}}
    start = time.perf_counter()
    options = dict(root=root, compiler_path=compiler, sign_command=sign_command,
        verify_signature_command=verify_signature_command, strict_asset_hashes=strict_asset_hashes,
        strict_release=strict_release, exclude_content_templates=exclude_content_templates, clean=clean)

    def record(area_excluded, timings):
        exclusions = (BUILD_COMPONENT_CONTENT_TEMPLATES,) if exclude_content_templates else ()
        if area_excluded:
            exclusions += (BUILD_COMPONENT_AREA_MODELS,)
        path = output / (f"fiber-diameter-measurement-setup-{summary['version']}"
                         + _installer_variant_suffix(exclusions) + ".exe")
        summary["variants"].append({"path": str(path), "sha256": sha256_file(path),
                                    "bytes": path.stat().st_size, "timings": timings})
        print(f"Verified installer: {path}\nSHA-256: {summary['variants'][-1]['sha256']}")

    try:
        first_times = {}
        result = build_installer(**options, rebuild_onedir=rebuild_onedir, timings=first_times)
        if result:
            summary["failed_stage"] = "with-area installer"
            summary["exit_code"] = result
            summary["timings"]["with_area_failed_seconds"] = time.perf_counter() - start
            return result
        record(False, first_times)
        original = json.loads((source / RELEASE_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        summary.update(source_commit=original["source_commit"], source_dirty=original["source_dirty"],
                       shared_program_build_id=original["build_id"])
        staging_root = root / "build/installer-variants"
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="area-pair-", dir=staging_root) as temporary:
            derived = Path(temporary) / "FiberDiameterMeasurement"
            then = time.perf_counter()
            summary["derivation"] = derive_without_area_models(root, source, derived)
            summary["timings"]["variant_derivation_seconds"] = time.perf_counter() - then
            second_times = {}
            result = build_installer(**options, app_dir=derived, exclude_area_models=True,
                                     rebuild_onedir=False, timings=second_times)
            if result:
                summary["failed_stage"] = "no-area installer"
                summary["exit_code"] = result
                return result
            record(True, second_times)
            # Keep text identities after disposing only this run's temporary
            # hardlinks. The full onedir and both installers remain available.
            for label, folder in (("with-area", source), ("no-area", derived)):
                shutil.copy2(folder / RELEASE_MANIFEST_FILENAME, output / f"{label}-release-manifest.json")
        summary["ok"] = True
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        summary["error"] = str(exc)
        print(f"Unable to finish both installer variants: {exc}")
        return 1
    finally:
        summary["timings"]["total_seconds"] = time.perf_counter() - start
        atomic_write_text(report_path, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        print(f"Build report: {report_path}")
        for phase, elapsed in summary["timings"].items():
            print(f"  {phase}: {elapsed:.2f} s")
