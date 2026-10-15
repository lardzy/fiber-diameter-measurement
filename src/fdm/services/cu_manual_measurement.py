"""In-memory, Qt-free CU-style assisted diameter measurement.

Coordinates are in the original frame, independent of display scale and DPI.
The native boundary snapshots only the needed ROI, including strided inputs;
callers must not concurrently modify an input through another native alias.
Calibration is deliberately separate: width_px is multiplied once by the
immutable frame's calibration by the experiment service.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from fdm import _native

ALGORITHM_REVISION = "cu-manual-portable-v1"


def measure_manual(
    image: np.ndarray,
    start: Sequence[float],
    end: Sequence[float],
    *,
    checkpoint: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Measure one crossing; reject unreliable geometry without a free-line fallback.

    uint8 grayscale and BGR images are accepted. Fractional source coordinates
    are truncated, matching the reference's display-to-image conversion. Input
    arrays are never changed. Cancellation/deadline exceptions propagate intact.
    Success has status ``ok`` and a finite positive ``width_px``; geometry
    rejection has status ``rejected``, a machine-readable reason, and width None.
    Invalid inputs raise ValueError/TypeError; binary failures never fall back.
    """
    array = np.asarray(image)
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")
    kernel = _native.require_native()
    if not hasattr(kernel, "cu_manual_measure"):
        raise RuntimeError("原生内核尚未包含人工贴边测径，请重新构建原生扩展。")
    return dict(kernel.cu_manual_measure(array, tuple(start), tuple(end), checkpoint))
