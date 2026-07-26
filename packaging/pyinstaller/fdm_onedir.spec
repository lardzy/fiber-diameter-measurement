# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


project_root = Path(SPECPATH).resolve().parents[1]
src_root = project_root / "src"
scripts_root = project_root / "scripts"
if str(scripts_root) not in sys.path:
    sys.path.insert(0, str(scripts_root))

from build_support import (
    BUILD_COMPONENT_CONTENT_TEMPLATES,
    collect_private_content_template_datas,
    collect_runtime_datas,
    normalize_build_exclusions,
    resolve_runtime_profile,
)

entry_script = project_root / "src" / "fdm" / "app.py"
worker_entry_script = project_root / "src" / "fdm" / "workers" / "area_worker.py"
app_icon = project_root / "packaging" / "assets" / "icons" / "app-icon.ico"
console_mode = os.environ.get("FDM_PYINSTALLER_CONSOLE", "0") == "1"
bootloader_debug = os.environ.get("FDM_PYINSTALLER_BOOTLOADER_DEBUG", "0") == "1"
build_profile = os.environ.get("FDM_BUILD_PROFILE", "full").strip().lower() or "full"
strict_asset_hashes = os.environ.get("FDM_STRICT_ASSET_HASHES", "0") == "1"
excluded_components = normalize_build_exclusions(os.environ.get("FDM_EXCLUDED_COMPONENTS", ""))
resolved_profile = resolve_runtime_profile(
    project_root,
    build_profile,
    excluded_groups=excluded_components,
)


def _collect_directory_files(root: Path, *, target_root: str) -> list[tuple[str, str]]:
    collected: list[tuple[str, str]] = []
    if not root.exists():
        return collected
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        relative_parent = file_path.parent.relative_to(root)
        target_dir = Path(target_root) / relative_parent if str(relative_parent) != "." else Path(target_root)
        collected.append((str(file_path), str(target_dir)))
    return collected

datas = [
    (str(project_root / "README.md"), "."),
    (str(project_root / "LICENSE"), "."),
    (str(project_root / "THIRD_PARTY_NOTICES.md"), "."),
    (str(project_root / "runtime_assets.toml"), "."),
]
datas += _collect_directory_files(project_root / "packaging" / "assets" / "icons", target_root="packaging/assets/icons")
datas += collect_runtime_datas(
    project_root,
    profile=build_profile,
    strict_asset_hashes=strict_asset_hashes,
    excluded_groups=excluded_components,
)
if BUILD_COMPONENT_CONTENT_TEMPLATES not in excluded_components:
    datas += collect_private_content_template_datas(project_root)
binaries = []
hiddenimports = [
    "PySide6.QtNetwork",
    "fdm.microview_helper",
]

required_packages = set(resolved_profile.required_python_modules)
collection_packages = set(required_packages)
if "openpyxl" in required_packages:
    # openpyxl imports this distribution lazily while reading/writing workbooks.
    collection_packages.add("et_xmlfile")
if "onnxruntime" in required_packages:
    hiddenimports += [
        "onnxruntime",
        "onnxruntime.capi",
        "onnxruntime.capi.onnxruntime_inference_collection",
        "onnxruntime.capi.onnxruntime_pybind11_state",
    ]

for package_name in sorted(collection_packages):
    try:
        binaries += collect_dynamic_libs(package_name)
    except Exception as exc:
        raise RuntimeError(
            f"{build_profile} profile failed to collect dynamic libraries for "
            f"{package_name}: {exc}"
        ) from exc
    try:
        hiddenimports += collect_submodules(package_name)
    except Exception as exc:
        raise RuntimeError(
            f"{build_profile} profile failed to collect submodules for "
            f"{package_name}: {exc}"
        ) from exc

for distribution_name in ("tifffile", "openpyxl", "et_xmlfile"):
    try:
        # Keep package versions and the license files stored in dist-info
        # available to the packaged self-check and the installed application.
        datas += copy_metadata(distribution_name)
    except Exception as exc:
        raise RuntimeError(
            f"{build_profile} profile failed to collect {distribution_name} metadata: {exc}"
        ) from exc

try:
    datas += collect_data_files("qtawesome", include_py_files=False)
    hiddenimports.append("qtawesome")
except Exception:
    if build_profile == "full":
        raise

try:
    hiddenimports.append("PySide6.QtMultimedia")
    hiddenimports += collect_submodules("PySide6.QtMultimedia")
except Exception:
    if build_profile == "full":
        raise

try:
    import PySide6

    pyside_root = Path(PySide6.__file__).resolve().parent
    datas += _collect_directory_files(pyside_root / "plugins" / "multimedia", target_root="PySide6/plugins/multimedia")
    datas += _collect_directory_files(pyside_root / "plugins" / "mediaservice", target_root="PySide6/plugins/mediaservice")
except Exception:
    if build_profile == "full":
        raise

main_analysis = Analysis(
    [str(entry_script)],
    pathex=[str(src_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "pytest", "IPython", "jupyter"],
    noarchive=False,
    optimize=0,
)
worker_analysis = Analysis(
    [str(worker_entry_script)],
    pathex=[str(src_root)],
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "pytest", "IPython", "jupyter"],
    noarchive=False,
    optimize=0,
)
main_pyz = PYZ(main_analysis.pure)
worker_pyz = PYZ(worker_analysis.pure)

exe = EXE(
    main_pyz,
    main_analysis.scripts,
    [],
    name="FiberDiameterMeasurement",
    exclude_binaries=True,
    debug=bootloader_debug,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=console_mode,
    icon=str(app_icon),
    disable_windowed_traceback=False,
    argv_emulation=False,
    contents_directory=".",
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
worker_exe = EXE(
    worker_pyz,
    worker_analysis.scripts,
    [],
    name="FiberAreaWorker",
    exclude_binaries=True,
    debug=bootloader_debug,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    icon=str(app_icon),
    disable_windowed_traceback=False,
    argv_emulation=False,
    contents_directory=".",
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    worker_exe,
    main_analysis.binaries,
    main_analysis.zipfiles,
    main_analysis.datas,
    worker_analysis.binaries,
    worker_analysis.zipfiles,
    worker_analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FiberDiameterMeasurement",
)
