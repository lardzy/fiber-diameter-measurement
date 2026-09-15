# Geometry dependency smoke build using the production hook. The exclusions
# apply only to this probe; it does not build the complete FDM application.
from pathlib import Path

probe_root = Path(SPECPATH)
project_root = probe_root.resolve().parents[2]
analysis = Analysis(
    [str(probe_root / "frozen_probe.py")],
    pathex=[str(project_root / "src")],
    hookspath=[str(project_root / "packaging" / "pyinstaller" / "hooks")],
    excludes=[
        "torch", "torchvision", "onnxruntime", "timm", "qtawesome",
        "pandas", "matplotlib", "pytest", "IPython", "jupyter",
    ],
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz, analysis.scripts, [], exclude_binaries=True,
    name="fdm-quick-geometry-probe", console=True,
)
coll = COLLECT(
    exe, analysis.binaries, analysis.datas,
    name="fdm-quick-geometry-probe",
)
