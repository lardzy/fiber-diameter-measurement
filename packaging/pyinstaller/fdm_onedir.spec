# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


project_root = Path(SPECPATH).resolve().parents[1]
src_root = project_root / "src"
entry_script = project_root / "src" / "fdm" / "app.py"
worker_entry_script = project_root / "src" / "fdm" / "workers" / "area_worker.py"
app_icon = project_root / "packaging" / "assets" / "icons" / "app-icon.ico"
console_mode = os.environ.get("FDM_PYINSTALLER_CONSOLE", "0") == "1"
bootloader_debug = os.environ.get("FDM_PYINSTALLER_BOOTLOADER_DEBUG", "0") == "1"

datas = [
    (str(project_root / "README.md"), "."),
]
runtime_root = project_root / "runtime"
if runtime_root.exists():
    for file_path in runtime_root.rglob("*"):
        if file_path.is_file():
            if file_path.name == ".DS_Store" or file_path.name.startswith("._"):
                continue
            relative_dir = file_path.parent.relative_to(project_root)
            datas.append((str(file_path), str(relative_dir)))
binaries = collect_dynamic_libs("onnxruntime")
hiddenimports = [
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi.onnxruntime_inference_collection",
    "onnxruntime.capi.onnxruntime_pybind11_state",
]

for optional_pkg in ("cv2", "PIL", "torch", "torchvision"):
    try:
        binaries += collect_dynamic_libs(optional_pkg)
    except Exception:
        pass
    try:
        hiddenimports += collect_submodules(optional_pkg)
    except Exception:
        pass

try:
    datas += collect_data_files("qtawesome", include_py_files=False)
    hiddenimports.append("qtawesome")
except Exception:
    pass

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
