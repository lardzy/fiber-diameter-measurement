"""Real service regression for the legacy histogram's large-frame overflow."""
from __future__ import annotations

import cv2
import numpy as np

from fdm.services.cu_automatic_measurement import analyze_frame


def test_four_k_bright_histogram_does_not_wrap_before_otsu_weighting():
    # Four broad vertical regions keep graph candidates at zero, so this real
    # 4K regression costs less than one second instead of a dense C² workload.
    # The old IMUL-then-FILD rule wraps the dominant 210 bin and incorrectly
    # classifies the 194 region as foreground. The correct cutoff (127 after
    # Gaussian edge smoothing) leaves that region in the white background.
    row = np.full((1, 4096), 210, np.uint8)
    row[:, :63], row[:, 63:138], row[:, 138:201] = 20, 90, 194
    height = 2730
    blurred = cv2.GaussianBlur(row, (7, 7), 0, borderType=cv2.BORDER_REPLICATE)
    histogram = np.bincount(blurred.ravel(), minlength=256).astype(np.int64) * height
    assert int(histogram[210]) * 211 > np.iinfo(np.int32).max

    # Independent integer-weighted/double-score oracle, including Gaussian
    # edge pixels and the historical +1 gray weighting and first-maximum tie.
    weights = histogram * (np.arange(256, dtype=np.int64) + 1)
    left = np.cumsum(histogram).astype(np.float64)
    right = float(histogram.sum()) - left
    weighted_left = np.cumsum(weights).astype(np.float64)
    mean_left = np.divide(weighted_left, left, out=np.zeros(256), where=left != 0)
    mean_right = np.divide(float(weights.sum()) - weighted_left, right,
                           out=np.zeros(256), where=right != 0)
    threshold = int(np.argmax((mean_left - mean_right) ** 2 * left * right))
    assert 90 <= threshold < 194

    image = np.repeat(row, height, axis=0)
    result = analyze_frame(image, "traditional", _debug=True)
    assert result["status"] == "empty"
    assert result["diagnostics"]["circle_count"] == 0
    mask = result["diagnostics"]["mask_labeled"]
    np.testing.assert_array_equal(mask[100:-100, 10:50], 0)
    np.testing.assert_array_equal(mask[100:-100, 75:125], 0)
    np.testing.assert_array_equal(mask[100:-100, 155:185], 255)
    np.testing.assert_array_equal(mask[100:-100, 230:300], 255)
