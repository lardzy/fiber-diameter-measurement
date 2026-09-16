"""Geometry-only frozen probe; this does not validate the full Windows release."""
import json
from pathlib import Path
import platform
import sys

from fdm.release_manifest import _probe_fiber_quick_geometry
from skimage.morphology import _skeletonize_various_cy


stdio = sys.stdin, sys.stdout, sys.stderr
try:
    # Windows --windowed executables have no console streams.
    sys.stdin = sys.stdout = sys.stderr = None
    result = _probe_fiber_quick_geometry()
finally:
    sys.stdin, sys.stdout, sys.stderr = stdio

extension = Path(_skeletonize_various_cy.__file__).resolve()
bundle_root = Path(getattr(sys, "_MEIPASS", "/nonexistent")).resolve()
result.update(
    platform=platform.platform(),
    frozen=bool(getattr(sys, "frozen", False)),
    extension_in_bundle=extension.is_relative_to(bundle_root),
    extension=str(extension),
    stdio_none=True,
)
result["ok"] = result["ok"] and result["frozen"] and result["extension_in_bundle"]
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["ok"] else 1)
