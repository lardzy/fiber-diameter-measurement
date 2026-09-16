# Match production hooks and ONNX native-library collection. This smoke build
# intentionally excludes unrelated application features and is not an installer.
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

probe_root = Path(SPECPATH)
project_root = probe_root.resolve().parents[2]
datas = []
for variant in ("edge_sam", "edge_sam_3x"):
    folder = Path("runtime") / "segment-anything" / variant
    for role in ("encoder", "decoder"):
        datas.append((str(project_root / folder / f"{variant}_{role}.onnx"), str(folder)))
analysis = Analysis(
    [str(probe_root / "frozen_probe.py")],
    pathex=[str(project_root / "src")],
    hookspath=[str(project_root / "packaging" / "pyinstaller" / "hooks")],
    datas=datas,
    binaries=collect_dynamic_libs("onnxruntime"),
    hiddenimports=collect_submodules("onnxruntime"),
    excludes=["torch", "torchvision", "timm", "qtawesome", "pandas", "matplotlib", "pytest", "IPython", "jupyter"],
)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name="fdm-magic-roi-probe", console=True)
coll = COLLECT(exe, analysis.binaries, analysis.datas, name="fdm-magic-roi-probe")
