"""Render local mask differences; run with uv run --no-sync --with matplotlib."""
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


folder = Path(__file__).resolve().parent / "implementation-runs"
root = Path(__file__).resolve().parents[3]
photo_path = root / ".tmp/fiberseg/image/58.jpg"
if not photo_path.exists():
    print("Skipping microscopy plot: the local example image is unavailable.")
    raise SystemExit(0)
photo = cv2.cvtColor(cv2.imread(str(photo_path)), cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(2, 3, figsize=(11, 9), layout="constrained")
for row, name in enumerate(("microscopy_fiber", "microscopy_fragment")):
    old = np.load(folder / f"baseline-{name}-2.npz")["mask"]
    new = np.load(folder / f"current-{name}-2.npz")["mask"]
    yy, xx = np.where(old | new)
    x0, x1 = max(0, xx.min() - 25), min(photo.shape[1], xx.max() + 26)
    y0, y1 = max(0, yy.min() - 25), min(photo.shape[0], yy.max() + 26)
    for col, (label, mask) in enumerate((("Old", old), ("New", new), ("Differences", None))):
        ax = axes[row, col]
        ax.imshow(photo[y0:y1, x0:x1])
        ax.set_title(name + " / " + label)
        ax.axis("off")
        if mask is not None:
            ax.contour(mask[y0:y1, x0:x1], levels=[0.5], colors=["lime"], linewidths=0.65)
        else:
            rgba = np.zeros((*old.shape, 4))
            rgba[old & ~new] = [0, 1, 1, 0.9]
            rgba[new & ~old] = [1, 0, 1, 0.9]
            ax.imshow(rgba[y0:y1, x0:x1])
fig.suptitle("Final refinement: cyan = old only; magenta = new only")
fig.savefig(folder / "microscopy-replay.png", dpi=160)
