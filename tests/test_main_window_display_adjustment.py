from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtWidgets import QApplication

from fdm.image_processing_models import DisplayTransform, ImageOperationSpec
from fdm.models import ImageDocument, new_id
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.raster_io import raster_plane_to_qimage
from fdm.settings import AppSettings
from fdm.ui.display_adjustment_dialog import (
    DisplayAdjustmentAction,
    DisplayAdjustmentResult,
)
from fdm.ui.image_processing_workbench import ImageProcessingWorkbench
from fdm.ui.main_window import MainWindow


class MainWindowDisplayAdjustmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.load_patch = patch(
            "fdm.ui.main_window.AppSettingsIO.load",
            return_value=AppSettings(theme_mode="dark"),
        )
        self.save_patch = patch(
            "fdm.ui.main_window.AppSettingsIO.save",
            return_value=None,
        )
        self.load_patch.start()
        self.save_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.addCleanup(self.save_patch.stop)

    @staticmethod
    def _plane(offset: int = 0) -> RasterPlane:
        values = (
            np.arange(48, dtype=np.uint16).reshape(6, 8) * 100
            + np.uint16(offset)
        ).astype("<u2")
        return RasterPlane(
            width=8,
            height=6,
            pixel_type=RasterPixelType.GRAY16,
            data=values.tobytes(),
        )

    def _window(self) -> tuple[MainWindow, ImageDocument, RasterPlane]:
        window = MainWindow()
        plane = self._plane()
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/display-source.tif",
            image_size=(plane.width, plane.height),
        )
        document.initialize_runtime_state()
        document.mark_session_saved()
        document.mark_calibration_saved()
        window._mount_document(
            document,
            raster_plane_to_qimage(plane),
            tooltip=document.path,
            raster_plane=plane,
        )
        window.show()
        self.app.processEvents()

        def cleanup() -> None:
            window._reset_workspace()
            window.close()
            self.app.processEvents()

        self.addCleanup(cleanup)
        return window, document, plane

    def test_preview_does_not_change_document_pixels_or_committed_cache(self) -> None:
        window, document, plane = self._window()
        committed_image_key = window._images[document.id].cacheKey()
        source_payload = document.to_dict()
        transform = DisplayTransform(
            channel_ranges=((200.0, 3000.0),),
            gamma=1.4,
        )

        window._preview_document_display_transform(
            document.id,
            plane.sha256(),
            transform,
        )

        self.assertEqual(document.to_dict(), source_payload)
        self.assertIs(window._rasters[document.id], plane)
        self.assertEqual(window._rasters[document.id].sha256(), plane.sha256())
        self.assertEqual(
            window._images[document.id].cacheKey(),
            committed_image_key,
        )
        self.assertNotEqual(
            window.current_canvas()._image.cacheKey(),  # noqa: SLF001
            committed_image_key,
        )

    def test_apply_is_persistent_undoable_and_refreshes_display_cache(self) -> None:
        window, document, plane = self._window()
        transform = DisplayTransform(
            channel_ranges=((100.0, 2500.0),),
            gamma=0.8,
            inverted=True,
        )
        result = DisplayAdjustmentResult(
            action=DisplayAdjustmentAction.APPLY_DISPLAY,
            transform=transform,
            source_sha256=plane.sha256(),
        )
        dialog = Mock()
        dialog.deleteLater = Mock()

        window._on_display_adjustment_result(dialog, document.id, result)

        self.assertEqual(document.display_transform, transform)
        self.assertEqual(
            window._display_cache_transforms[document.id],
            transform,
        )
        self.assertTrue(document.dirty_flags.session_dirty)
        committed_key = window._images[document.id].cacheKey()
        self.assertEqual(window.current_canvas()._image.cacheKey(), committed_key)  # noqa: SLF001

        window.undo_current_document()
        self.assertIsNone(document.display_transform)
        self.assertIsNone(window._display_cache_transforms[document.id])
        self.assertNotEqual(window._images[document.id].cacheKey(), committed_key)

        window.redo_current_document()
        self.assertEqual(document.display_transform, transform)
        self.assertEqual(
            window._display_cache_transforms[document.id],
            transform,
        )

    def test_cancel_restores_committed_canvas_without_mutating_document(self) -> None:
        window, document, plane = self._window()
        committed_key = window._images[document.id].cacheKey()
        preview = DisplayTransform(channel_ranges=((300.0, 2300.0),))
        window._preview_document_display_transform(
            document.id,
            plane.sha256(),
            preview,
        )
        self.assertNotEqual(
            window.current_canvas()._image.cacheKey(),  # noqa: SLF001
            committed_key,
        )
        result = DisplayAdjustmentResult(
            action=DisplayAdjustmentAction.CANCEL,
            transform=DisplayTransform(),
            source_sha256=plane.sha256(),
        )
        dialog = Mock()
        dialog.deleteLater = Mock()

        window._on_display_adjustment_result(dialog, document.id, result)

        self.assertIsNone(document.display_transform)
        self.assertEqual(
            window.current_canvas()._image.cacheKey(),  # noqa: SLF001
            committed_key,
        )

    def test_generate_derived_routes_explicit_bake_recipe_to_workbench(self) -> None:
        window, document, plane = self._window()
        operation = ImageOperationSpec(
            "adjust_levels",
            {
                "black_point": 100.0,
                "white_point": 2500.0,
                "gamma": 1.2,
            },
        )
        result = DisplayAdjustmentResult(
            action=DisplayAdjustmentAction.GENERATE_DERIVED,
            transform=DisplayTransform(
                channel_ranges=((100.0, 2500.0),),
                gamma=1.2,
            ),
            source_sha256=plane.sha256(),
            bake_operations=(operation,),
        )
        dialog = Mock()
        dialog.deleteLater = Mock()

        with patch.object(
            ImageProcessingWorkbench,
            "generate_derived_image",
            autospec=True,
        ) as generate:
            window._on_display_adjustment_result(
                dialog,
                document.id,
                result,
            )

        workbench = window._image_processing_workbench
        self.assertIsNotNone(workbench)
        self.assertEqual(workbench.operation_steps(), (operation,))
        generate.assert_called_once_with(workbench)
        self.assertIsNone(document.display_transform)
        self.assertIs(window._rasters[document.id], plane)
        window._close_image_processing_workbench(wait=True)

    def test_stale_display_result_is_discarded(self) -> None:
        window, document, plane = self._window()
        result = DisplayAdjustmentResult(
            action=DisplayAdjustmentAction.APPLY_DISPLAY,
            transform=DisplayTransform(channel_ranges=((1.0, 1000.0),)),
            source_sha256=plane.sha256(),
        )
        window._rasters[document.id] = self._plane(offset=1)
        dialog = Mock()
        dialog.deleteLater = Mock()
        with patch(
            "fdm.ui.main_window.QMessageBox.warning"
        ) as warning:
            window._on_display_adjustment_result(
                dialog,
                document.id,
                result,
            )

        self.assertIsNone(document.display_transform)
        self.assertFalse(document.dirty_flags.session_dirty)
        warning.assert_called_once()
        self.assertIn("晚到设置", warning.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
