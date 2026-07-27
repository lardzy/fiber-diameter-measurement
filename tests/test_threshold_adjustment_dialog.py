from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from fdm.raster import RasterPixelType, RasterPlane
    from fdm.ui.threshold_adjustment_dialog import (
        ThresholdAdjustmentDialog,
        ThresholdDerivationRequest,
    )

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class ThresholdAdjustmentDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _gradient() -> RasterPlane:
        values = np.arange(256, dtype=np.uint8).reshape(16, 16)
        return RasterPlane(
            width=16,
            height=16,
            pixel_type=RasterPixelType.GRAY8,
            data=values.tobytes(),
        )

    def test_auto_threshold_and_structured_derivation_request(self) -> None:
        dialog = ThresholdAdjustmentDialog(self._gradient())
        requests: list[ThresholdDerivationRequest] = []
        dialog.binaryDerivedRequested.connect(requests.append)
        try:
            self.assertFalse(dialog.isModal())
            dialog._auto_threshold()  # noqa: SLF001
            operation = dialog.operation_spec()
            self.assertEqual(operation.operation_id, "threshold")
            self.assertGreater(operation.parameters["lower"], 100.0)
            self.assertLess(operation.parameters["lower"], 160.0)
            self.assertEqual(operation.parameters["upper"], 255.0)
            dialog._request_binary_derivative()  # noqa: SLF001
            self.assertEqual(len(requests), 1)
            self.assertEqual(
                requests[0].source_sha256,
                self._gradient().sha256(),
            )
        finally:
            dialog.close()

    def test_roi_controls_statistics_and_is_not_mutated(self) -> None:
        mask = np.zeros((16, 16), dtype=np.bool_)
        mask[:, 8:] = True
        dialog = ThresholdAdjustmentDialog(
            self._gradient(),
            roi_mask=mask,
        )
        try:
            self.assertIn("当前 ROI", dialog.scopeLabel.text())
            self.assertIn("N=128", dialog.scopeLabel.text())
            self.assertFalse(dialog._roi_mask.flags.writeable)  # noqa: SLF001
            dialog._reset_threshold()  # noqa: SLF001
            self.assertEqual(dialog.lowerSpin.value(), 0.0)
            self.assertEqual(dialog.upperSpin.value(), 255.0)
            self.assertFalse(dialog.imagePreview._image.isNull())  # noqa: SLF001
            self.assertIn("灰色区域", dialog.previewLegend.text())
            dialog.previewModeCombo.setCurrentIndex(1)
            self.app.processEvents()
            self.assertIn("Over/Under", dialog.previewLegend.text())
        finally:
            dialog.close()

    def test_invalid_roi_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ROI 掩膜尺寸"):
            ThresholdAdjustmentDialog(
                self._gradient(),
                roi_mask=np.ones((2, 2), dtype=np.bool_),
            )


if __name__ == "__main__":
    unittest.main()
