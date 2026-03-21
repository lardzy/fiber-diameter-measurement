# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


project_root = Path(SPECPATH).resolve().parents[1]
src_root = project_root / "src"

datas = [
    (str(project_root / "README.md"), "."),
]
binaries = []
hiddenimports = []

for package_name in ("onnxruntime", "openpyxl", "pandas"):
    datas += collect_data_files(package_name, include_py_files=False)
    hiddenimports += collect_submodules(package_name)

for package_name in ("onnxruntime", "cv2"):
    binaries += collect_dynamic_libs(package_name)
    hiddenimports += collect_submodules(package_name)

a = Analysis(
    ["src/fdm/app.py"],
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
    a.binaries,
    a.datas,
    [],
    name="FiberDiameterMeasurement",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FiberDiameterMeasurement",
)
