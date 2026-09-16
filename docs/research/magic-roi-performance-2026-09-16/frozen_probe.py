"""Native dependency smoke probe; this is not a complete Windows release test."""
import json
from pathlib import Path
import platform
import sys

from fdm.release_manifest import _probe_fiber_quick_geometry
from fdm.services.magic_segmentation_self_check import run_magic_segmentation_self_check
from skimage.morphology import _skeletonize_various_cy
from onnxruntime.capi import onnxruntime_pybind11_state


root = Path(getattr(sys, "_MEIPASS", "/nonexistent")).resolve()
stdio = sys.stdin, sys.stdout, sys.stderr
try:
    # Also exercise the no-console case used by Windows windowed executables.
    sys.stdin = sys.stdout = sys.stderr = None
    magic = run_magic_segmentation_self_check(root)
    geometry = _probe_fiber_quick_geometry()
finally:
    sys.stdin, sys.stdout, sys.stderr = stdio
extensions = {
    "onnxruntime": Path(onnxruntime_pybind11_state.__file__).resolve(),
    "skeleton": Path(_skeletonize_various_cy.__file__).resolve(),
}
result = {
    "platform": platform.platform(), "frozen": bool(getattr(sys, "frozen", False)), "stdio_none": True,
    "extensions_in_bundle": all(path.is_relative_to(root) for path in extensions.values()),
    "extensions": {name: str(path.relative_to(root)) for name, path in extensions.items()},
    "functional_checks": {"magic_segmentation": magic, "fiber_quick_geometry": geometry},
}
result["ok"] = bool(result["frozen"] and result["extensions_in_bundle"] and magic["ok"] and geometry["ok"])
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["ok"] else 1)
