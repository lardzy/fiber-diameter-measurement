# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_dynamic_libs,
)


project_root = Path(SPECPATH).resolve().parents[1]
src_root = project_root / "src"
entry_script = project_root / "src" / "fdm" / "app.py"
console_mode = os.environ.get("FDM_PYINSTALLER_CONSOLE", "0") == "1"
bootloader_debug = os.environ.get("FDM_PYINSTALLER_BOOTLOADER_DEBUG", "0") == "1"

datas = [
    (str(project_root / "README.md"), "."),
]
binaries = collect_dynamic_libs("onnxruntime")
hiddenimports = [
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi.onnxruntime_inference_collection",
    "onnxruntime.capi.onnxruntime_pybind11_state",
]

a = Analysis(
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
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
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
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FiberDiameterMeasurement",
)
