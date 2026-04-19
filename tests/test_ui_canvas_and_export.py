from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QPoint, QPointF, Qt, QThread
    from PySide6.QtGui import QImage, QColor, QPalette
    from PySide6.QtWidgets import QApplication, QComboBox, QGroupBox, QListView, QMessageBox, QScrollArea, QSizePolicy, QSplitter

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import Calibration, CalibrationPreset, ImageDocument, Measurement, OverlayAnnotationKind, ProjectGroupTemplate, TextAnnotation, new_id
from fdm.settings import (
    AppSettings,
    FocusStackProfile,
    MagicSegmentModelVariant,
    MagicSegmentToolMode,
    MeasurementEndpointStyle,
    OpenImageViewMode,
    ScaleOverlayPlacementMode,
    ScaleOverlayStyle,
)
from fdm.services.area_inference import AreaInstanceResult
from fdm.services.export_service import ExportImageRenderMode
from fdm.services.prompt_segmentation import PromptSegmentationResult
from fdm.services.sidecar_io import CalibrationSidecarIO
from fdm.services.snap_service import SnapResult

if PYSIDE_AVAILABLE:
    from fdm.ui.canvas import DocumentCanvas, MagicSegmentOperationMode, ReferenceInstancePreviewCandidate
    from fdm.ui.dialogs import FiberGroupDialog, SettingsDialog, ShortcutHelpDialog
    from fdm.ui.icons import application_icon
    from fdm.ui.image_loader import ImageBatchLoaderWorker, ImageLoadRequest, qimage_to_raster
    from fdm.ui.main_window import MainWindow
    from fdm.ui.preview_analysis_dialog import PreviewAnalysisDialog
    from fdm.ui.rendering import draw_area_measurement, measurement_display_text_with_settings
    from fdm.ui.widgets import FiberGroupListItemWidget, MeasurementToolStrip, OverlayToolSplitButton
else:
    DocumentCanvas = object  # type: ignore[assignment]
    MagicSegmentOperationMode = object  # type: ignore[assignment]
    ReferenceInstancePreviewCandidate = object  # type: ignore[assignment]
    FiberGroupDialog = object  # type: ignore[assignment]
    SettingsDialog = object  # type: ignore[assignment]
    ShortcutHelpDialog = object  # type: ignore[assignment]
    application_icon = object  # type: ignore[assignment]
    ImageBatchLoaderWorker = object  # type: ignore[assignment]
    ImageLoadRequest = object  # type: ignore[assignment]
    qimage_to_raster = object  # type: ignore[assignment]
    MainWindow = object  # type: ignore[assignment]
    PreviewAnalysisDialog = object  # type: ignore[assignment]
    draw_area_measurement = object  # type: ignore[assignment]
    measurement_display_text_with_settings = object  # type: ignore[assignment]
    FiberGroupListItemWidget = object  # type: ignore[assignment]
    MeasurementToolStrip = object  # type: ignore[assignment]
    OverlayToolSplitButton = object  # type: ignore[assignment]


class RecordingPainter:
    def __init__(self) -> None:
        self.pen = None
        self.brush = None
        self.calls: list[tuple[str, object, object, int]] = []

    def setPen(self, pen) -> None:
        self.pen = pen

    def setBrush(self, brush) -> None:
        self.brush = brush

    def drawPath(self, path) -> None:
        self.calls.append(("path", self.pen, self.brush, path.elementCount()))

    def drawPolygon(self, polygon) -> None:
        self.calls.append(("polygon", self.pen, self.brush, polygon.size()))


class FakeWheelEvent:
    def __init__(self, position: QPointF, *, delta_x: int = 0, delta_y: int = 0) -> None:
        self._position = position
        self._delta = QPoint(delta_x, delta_y)

    def position(self) -> QPointF:
        return self._position

    def angleDelta(self) -> QPoint:
        return self._delta


class FakeMouseEvent:
    def __init__(self, position, *, button=None, modifiers=None) -> None:
        self._position = position
        self._button = button if button is not None else (Qt.MouseButton.LeftButton if PYSIDE_AVAILABLE else None)
        self._modifiers = modifiers if modifiers is not None else (Qt.KeyboardModifier.NoModifier if PYSIDE_AVAILABLE else None)

    def position(self) -> QPointF:
        return self._position

    def button(self):
        return self._button

    def modifiers(self):
        return self._modifiers


class FakeKeyEvent:
    def __init__(self, key, *, modifiers=None) -> None:
        self._key = key
        self._modifiers = modifiers if modifiers is not None else Qt.KeyboardModifier.NoModifier
        self.accepted = False

    def key(self):
        return self._key

    def modifiers(self):
        return self._modifiers

    def isAutoRepeat(self) -> bool:
        return False

    def accept(self) -> None:
        self.accepted = True


class FakeCloseEvent:
    def __init__(self) -> None:
        self.accepted = False
        self.ignored = False

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


class FakeIgnoredWheelEvent:
    def __init__(self) -> None:
        self.ignored = False

    def ignore(self) -> None:
        self.ignored = True


@unittest.skipUnless(PYSIDE_AVAILABLE, "requires PySide6")
class CanvasAndExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _create_canvas_document(self) -> tuple[ImageDocument, QImage, DocumentCanvas]:
        image = QImage(200, 120, QImage.Format.Format_RGB32)
        image.fill(QColor("#FFFFFF"))
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/canvas_test.png",
            image_size=(image.width(), image.height()),
        )
        document.initialize_runtime_state()
        canvas = DocumentCanvas()
        canvas.resize(320, 240)
        canvas.set_document(document, image)
        return document, image, canvas

    def _create_main_window_fixture(self) -> tuple[MainWindow, ImageDocument]:
        window = MainWindow()
        image = QImage(260, 180, QImage.Format.Format_RGB32)
        image.fill(QColor("#FFFFFF"))
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/export_visibility.png",
            image_size=(image.width(), image.height()),
        )
        document.initialize_runtime_state()
        group = document.create_group(color="#1F7A8C", label="棉")
        document.set_active_group(group.id)
        document.calibration = Calibration(
            mode="preset",
            pixels_per_unit=5.0,
            unit="um",
            source_label="demo",
        )
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=group.id,
            mode="manual",
            line_px=Line(Point(20, 30), Point(200, 140)),
            confidence=1.0,
            status="manual",
        )
        document.add_measurement(measurement)
        canvas = DocumentCanvas()
        canvas.resize(340, 240)
        canvas.set_document(document, image)

        window._images[document.id] = image
        window._canvases[document.id] = canvas
        return window, document

    def _load_document_into_window(self, window: MainWindow, document: ImageDocument, image: QImage) -> None:
        window._add_loaded_document(
            ImageLoadRequest(path=document.path, document=document),
            image,
        )

    def _group_titles_in_tab(self, dialog: SettingsDialog, index: int) -> list[str]:
        tab = dialog._tabs.widget(index)
        self.assertIsInstance(tab, QScrollArea)
        content = tab.widget()
        self.assertIsNotNone(content)
        return [group.title() for group in content.findChildren(QGroupBox) if group.title()]

    def _rect_mask(self, width: int, height: int, rect: tuple[int, int, int, int]):
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable: {exc}")
        mask = np.zeros((height, width), dtype=bool)
        x1, y1, x2, y2 = rect
        mask[y1:y2, x1:x2] = True
        return mask

    def _area_instance(self, label: str, polygon: list[Point] | None = None) -> AreaInstanceResult:
        shape = polygon or [Point(10, 10), Point(30, 10), Point(30, 30), Point(10, 30)]
        return AreaInstanceResult(
            class_name=label,
            score=0.95,
            bbox=[10, 10, 30, 30],
            polygon_px=list(shape),
            area_px=400.0,
        )

    def _count_diff_pixels(self, left: QImage, right: QImage) -> int:
        self.assertEqual(left.size(), right.size())
        diff = 0
        for y in range(left.height()):
            for x in range(left.width()):
                if left.pixel(x, y) != right.pixel(x, y):
                    diff += 1
        return diff

    def test_wheel_event_uses_horizontal_delta_fallback_for_zoom_in(self) -> None:
        _, _, canvas = self._create_canvas_document()
        before_zoom = canvas._zoom

        canvas.wheelEvent(FakeWheelEvent(QPointF(120.0, 80.0), delta_x=120, delta_y=0))

        self.assertGreater(canvas._zoom, before_zoom)

    def test_calibration_mode_does_not_grab_measurement_handles(self) -> None:
        document, _, canvas = self._create_canvas_document()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(20, 20), Point(80, 20)),
        )
        document.add_measurement(measurement)
        document.view_state.selected_measurement_id = measurement.id
        canvas.set_selected_measurement(measurement.id)
        canvas.set_tool_mode("calibration")

        endpoint_position = canvas.image_to_widget(Point(20, 20))
        canvas.mousePressEvent(FakeMouseEvent(endpoint_position))

        self.assertIsNone(canvas._dragging_handle)
        self.assertIsNotNone(canvas._drawing_line)

    def test_snap_mode_uses_two_click_commit_instead_of_release(self) -> None:
        document, _, canvas = self._create_canvas_document()
        canvas.set_tool_mode("snap")
        commits: list[tuple[str, str, object]] = []
        canvas.lineCommitted.connect(lambda document_id, mode, payload: commits.append((document_id, mode, payload)))

        start = Point(24, 18)
        end = Point(140, 76)
        start_pos = canvas.image_to_widget(start)
        end_pos = canvas.image_to_widget(end)

        canvas.mousePressEvent(FakeMouseEvent(start_pos, button=Qt.MouseButton.LeftButton))
        canvas.mouseReleaseEvent(FakeMouseEvent(start_pos, button=Qt.MouseButton.LeftButton))

        self.assertEqual(len(commits), 0)
        self.assertIsNotNone(canvas._drawing_line)
        self.assertTrue(canvas._line_commit_on_second_click)

        canvas.mouseMoveEvent(FakeMouseEvent(end_pos, button=Qt.MouseButton.LeftButton))

        self.assertEqual(canvas._drawing_line, Line(start=start, end=end))

        canvas.mousePressEvent(FakeMouseEvent(end_pos, button=Qt.MouseButton.LeftButton))
        canvas.mouseReleaseEvent(FakeMouseEvent(end_pos, button=Qt.MouseButton.LeftButton))

        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0][0], document.id)
        self.assertEqual(commits[0][1], "snap")
        self.assertEqual(commits[0][2], Line(start=start, end=end))
        self.assertIsNone(canvas._drawing_line)
        self.assertFalse(canvas._line_commit_on_second_click)

    def test_snap_mode_does_not_grab_measurement_handles(self) -> None:
        document, _, canvas = self._create_canvas_document()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(20, 20), Point(80, 20)),
        )
        document.add_measurement(measurement)
        document.view_state.selected_measurement_id = measurement.id
        canvas.set_selected_measurement(measurement.id)
        canvas.set_tool_mode("snap")

        endpoint_position = canvas.image_to_widget(Point(20, 20))
        canvas.mousePressEvent(FakeMouseEvent(endpoint_position))

        self.assertIsNone(canvas._dragging_handle)
        self.assertIsNotNone(canvas._drawing_line)
        self.assertTrue(canvas._line_commit_on_second_click)

    def test_space_drag_temporarily_pans_in_manual_mode(self) -> None:
        _, _, canvas = self._create_canvas_document()
        canvas.set_tool_mode("manual")
        start_pan = Point(canvas._pan.x, canvas._pan.y)

        canvas.keyPressEvent(FakeKeyEvent(Qt.Key.Key_Space))
        canvas.mousePressEvent(FakeMouseEvent(QPointF(120.0, 80.0), button=Qt.MouseButton.LeftButton))
        canvas.mouseMoveEvent(FakeMouseEvent(QPointF(160.0, 110.0), button=Qt.MouseButton.LeftButton))
        canvas.mouseReleaseEvent(FakeMouseEvent(QPointF(160.0, 110.0), button=Qt.MouseButton.LeftButton))

        self.assertNotEqual((canvas._pan.x, canvas._pan.y), (start_pan.x, start_pan.y))
        self.assertIsNone(canvas._drawing_line)

        canvas.keyReleaseEvent(FakeKeyEvent(Qt.Key.Key_Space))
        canvas.mousePressEvent(FakeMouseEvent(QPointF(120.0, 80.0), button=Qt.MouseButton.LeftButton))

        self.assertIsNotNone(canvas._drawing_line)

    def test_selected_endpoint_tolerance_allows_new_line_near_existing_measurement(self) -> None:
        document, _, canvas = self._create_canvas_document()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(20, 20), Point(80, 20)),
        )
        document.add_measurement(measurement)
        document.view_state.selected_measurement_id = measurement.id
        canvas.set_selected_measurement(measurement.id)
        canvas.set_tool_mode("manual")

        nearby_position = canvas.image_to_widget(Point(30, 20))
        canvas.mousePressEvent(FakeMouseEvent(nearby_position, button=Qt.MouseButton.LeftButton))

        self.assertIsNone(canvas._dragging_handle)
        self.assertIsNotNone(canvas._drawing_line)

    def test_line_hit_test_keeps_tolerance_when_point_is_outside_exact_bounds(self) -> None:
        document, _, canvas = self._create_canvas_document()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(20, 20), Point(80, 20)),
        )
        document.add_measurement(measurement)

        hit_id = canvas._hit_test_measurement(Point(50, 24))

        self.assertEqual(hit_id, measurement.id)

    def test_area_hit_test_keeps_edge_tolerance_when_point_is_outside_exact_bounds(self) -> None:
        document, _, canvas = self._create_canvas_document()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="area",
            polygon_px=[
                Point(40, 40),
                Point(90, 40),
                Point(90, 90),
                Point(40, 90),
            ],
        )
        document.add_measurement(measurement)

        hit_id = canvas._hit_test_area_measurement(Point(65, 94))

        self.assertEqual(hit_id, measurement.id)

    def test_live_preview_frame_can_be_captured_as_project_asset_document(self) -> None:
        window = MainWindow()
        preview_frame = QImage(180, 120, QImage.Format.Format_RGB32)
        preview_frame.fill(QColor("#CCE3DE"))

        window._on_live_preview_state_changed(True)
        window._capture_manager.can_capture_still = lambda: True  # type: ignore[method-assign]
        window._on_live_preview_frame_ready(preview_frame)
        window._capture_manager.last_frame = lambda: preview_frame.copy()  # type: ignore[method-assign]
        window._capture_manager.capture_still_frame = lambda: preview_frame.copy()  # type: ignore[method-assign]
        window._capture_manager.is_preview_active = lambda: False  # type: ignore[method-assign]

        self.assertIsNone(window.current_document())
        self.assertIs(window.current_canvas(), window._preview_canvas)
        self.assertTrue(window.capture_frame_action.isEnabled())

        window.capture_current_frame()

        self.assertEqual(len(window.project.documents), 1)
        captured_document = window.project.documents[0]
        self.assertEqual(captured_document.source_type, "project_asset")
        self.assertTrue(captured_document.path.startswith("captures/"))
        self.assertIn(captured_document.id, window._images)
        self.assertTrue(window._project_dirty())

    def test_native_preview_switches_to_microview_host_surface(self) -> None:
        window = MainWindow()
        window._capture_manager.preview_kind = lambda: "native_embed"  # type: ignore[method-assign]
        window._capture_manager.preview_resolution = lambda: (1760, 1328)  # type: ignore[method-assign]

        window._on_live_preview_state_changed(True)

        self.assertIsNone(window.current_canvas())
        self.assertIsNotNone(window._preview_display_stack)
        self.assertIsNotNone(window._microview_preview_host)
        self.assertIs(window._preview_display_stack.currentWidget(), window._microview_preview_scroll)
        self.assertEqual(window._microview_preview_host.size().width(), 1760)
        self.assertEqual(window._microview_preview_host.size().height(), 1328)

    def test_start_live_preview_applies_native_resolution_before_backend_start(self) -> None:
        window = MainWindow()
        observed_sizes: list[tuple[int, int]] = []
        device = type("Device", (), {"backend_key": "microview", "id": "microview:0", "name": "Microview #1"})()

        window._capture_devices = [device]
        window._refresh_capture_devices = lambda: None  # type: ignore[method-assign]
        window._capture_manager.preview_kind = lambda: "native_embed"  # type: ignore[method-assign]
        window._capture_manager.preview_resolution = lambda: (768, 576)  # type: ignore[method-assign]
        window._capture_manager.selected_device = lambda: device  # type: ignore[method-assign]
        window._capture_manager.start_preview = (  # type: ignore[method-assign]
            lambda *, preview_target=None: observed_sizes.append(preview_target.native_preview_size()) or True
        )

        window.start_live_preview()

        self.assertEqual(observed_sizes, [(768, 576)])
        self.assertIsNotNone(window._microview_preview_host)
        self.assertEqual(window._microview_preview_host.native_preview_size(), (768, 576))

    def test_native_preview_capture_uses_still_capture_path(self) -> None:
        window = MainWindow()
        preview_frame = QImage(96, 72, QImage.Format.Format_RGB32)
        preview_frame.fill(QColor("#F4D35E"))
        call_order: list[str] = []

        window._capture_manager.preview_kind = lambda: "native_embed"  # type: ignore[method-assign]
        window._capture_manager.can_capture_still = lambda: True  # type: ignore[method-assign]
        window._capture_manager.capture_still_frame = lambda: call_order.append("capture") or preview_frame.copy()  # type: ignore[method-assign]
        window._capture_manager.is_preview_active = lambda: True  # type: ignore[method-assign]
        window._capture_manager.stop_preview = lambda: call_order.append("stop")  # type: ignore[method-assign]
        window._capture_manager.selected_device = (  # type: ignore[method-assign]
            lambda: type("Device", (), {"backend_key": "microview", "id": "microview:0", "name": "Microview #1"})()
        )

        window.capture_current_frame()

        self.assertEqual(call_order, ["stop", "capture"])
        self.assertEqual(len(window.project.documents), 1)
        self.assertEqual(window.project.documents[0].source_type, "project_asset")

    def test_live_preview_stop_clears_preview_canvas_and_late_frame_is_ignored(self) -> None:
        window = MainWindow()
        preview_frame = QImage(180, 120, QImage.Format.Format_RGB32)
        preview_frame.fill(QColor("#CCE3DE"))
        late_frame = QImage(160, 90, QImage.Format.Format_RGB32)
        late_frame.fill(QColor("#F4D35E"))

        window._on_live_preview_state_changed(True)
        window._on_live_preview_frame_ready(preview_frame)
        self.assertIsNotNone(window._preview_canvas)
        self.assertEqual(window._preview_canvas.document_id, "preview_document")

        window._on_live_preview_state_changed(False)
        window._on_live_preview_frame_ready(late_frame)

        self.assertIsNone(window._preview_document)
        self.assertIsNotNone(window._preview_canvas)
        self.assertIsNone(window._preview_canvas.document_id)
        self.assertIs(window._center_stack.currentWidget(), window.tab_widget)

    def test_live_preview_stop_requests_magic_cache_clear(self) -> None:
        window = MainWindow()
        clear_calls: list[str] = []

        class FakeSignal:
            def emit(self) -> None:
                clear_calls.append("clear")

        class FakeWorker:
            def __init__(self) -> None:
                self.clearRequested = FakeSignal()

        window._prompt_seg_worker = FakeWorker()

        window._on_live_preview_state_changed(False)

        self.assertEqual(clear_calls, ["clear"])

    def test_preview_analysis_dialog_has_finish_button(self) -> None:
        dialog = PreviewAnalysisDialog("景深合成", intro_text="测试")
        try:
            triggered: list[str] = []
            dialog.finishRequested.connect(lambda: triggered.append("finish"))
            dialog._finish_button.click()
            self.assertEqual(triggered, ["finish"])
            self.assertEqual(dialog._finish_button.text(), "结束")
        finally:
            dialog.close()

    def test_preview_analysis_dialog_busy_state_disables_controls_and_shortcuts(self) -> None:
        dialog = PreviewAnalysisDialog("景深合成", intro_text="测试")
        try:
            triggered: list[str] = []
            cancelled: list[str] = []
            dialog.finishRequested.connect(lambda: triggered.append("finish"))
            dialog.cancelRequested.connect(lambda: cancelled.append("cancel"))

            dialog.set_busy(True, "正在完成景深合成，请稍候…")
            dialog.keyPressEvent(FakeKeyEvent(Qt.Key.Key_Return))
            dialog.keyPressEvent(FakeKeyEvent(Qt.Key.Key_Escape))

            self.assertTrue(dialog._busy_overlay.isVisible())
            self.assertEqual(dialog._busy_label.text(), "正在完成景深合成，请稍候…")
            self.assertGreaterEqual(dialog._busy_panel.minimumWidth(), 420)
            self.assertFalse(dialog._finish_button.isEnabled())
            self.assertFalse(dialog._cancel_button.isEnabled())
            self.assertEqual(triggered, [])
            self.assertEqual(cancelled, [])
        finally:
            dialog.close_silently()

    def test_finalize_preview_analysis_session_sets_busy_state_on_dialog(self) -> None:
        window = MainWindow()
        try:
            busy_calls: list[tuple[bool, str]] = []
            statuses: list[str] = []
            finalize_calls: list[str] = []

            class FakeSignal:
                def emit(self) -> None:
                    finalize_calls.append("finalize")

            class FakeWorker:
                def __init__(self) -> None:
                    self.finalizeRequested = FakeSignal()

            class FakeDialog:
                def set_status(self, text: str) -> None:
                    statuses.append(text)

                def set_busy(self, active: bool, text: str) -> None:
                    busy_calls.append((active, text))

            window._preview_analysis_mode = "focus_stack"
            window._preview_analysis_worker = FakeWorker()
            window._preview_analysis_dialog = FakeDialog()  # type: ignore[assignment]
            window._preview_analysis_finalizing = False

            window._finalize_preview_analysis_session()

            self.assertTrue(window._preview_analysis_finalizing)
            self.assertEqual(finalize_calls, ["finalize"])
            self.assertEqual(statuses, ["正在完成景深合成，请稍候…"])
            self.assertEqual(busy_calls, [(True, "正在完成景深合成，请稍候…")])
        finally:
            window._reset_workspace()
            window.deleteLater()
            self.app.processEvents()

    def test_map_build_button_shows_developing_message_when_unavailable(self) -> None:
        window = MainWindow()
        try:
            fake_device = type("Device", (), {"backend_key": "microview", "id": "microview:0", "name": "Microview #1"})()
            window._preview_active = True
            window._capture_manager.selected_device = lambda: fake_device  # type: ignore[method-assign]
            window._capture_manager.can_request_analysis_frame = lambda: True  # type: ignore[method-assign]

            window._update_preview_analysis_controls()

            self.assertIsNotNone(window._focus_stack_button)
            self.assertIsNotNone(window._map_build_button)
            self.assertTrue(window._focus_stack_button.isEnabled())
            self.assertTrue(window._map_build_button.isEnabled())
            self.assertIsNone(window._map_build_status_label)
            with patch.object(QMessageBox, "information", return_value=QMessageBox.StandardButton.Ok) as mock_information:
                window._map_build_button.click()
            mock_information.assert_called_once()
            self.assertIn("开发中", mock_information.call_args.args[2])
            self.assertFalse(window._map_build_button.isChecked())
        finally:
            window._reset_workspace()
            window.close()

    def test_image_resolution_label_shows_pixels_and_actual_size_when_calibrated(self) -> None:
        window = MainWindow()
        image = QImage(200, 100, QImage.Format.Format_RGB32)
        image.fill(QColor("#FFFFFF"))
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/resolution_label.png",
            image_size=(200, 100),
        )
        document.initialize_runtime_state()
        document.calibration = Calibration(
            mode="preset",
            pixels_per_unit=10.0,
            unit="um",
            source_label="10x",
        )

        window._add_loaded_document(
            ImageLoadRequest(path=document.path, document=document),
            image,
        )

        self.assertIsNotNone(window._image_resolution_label)
        label_text = window._image_resolution_label.text()
        self.assertIn("像素尺寸: 200 x 100 px", label_text)
        self.assertIn("实际尺寸: 20 x 10 um", label_text)

    def test_calibration_label_uses_red_for_uncalibrated_document(self) -> None:
        window = MainWindow()
        image = QImage(120, 80, QImage.Format.Format_RGB32)
        image.fill(QColor("#FFFFFF"))
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/uncalibrated_label.png",
            image_size=(120, 80),
        )
        document.initialize_runtime_state()

        window._add_loaded_document(
            ImageLoadRequest(path=document.path, document=document),
            image,
        )

        self.assertEqual(window.calibration_label.text(), "当前图片未标定")
        self.assertIn(window._status_color("danger"), window.calibration_label.styleSheet())
        self.assertIs(window._calibration_label_scroll.widget(), window.calibration_label)

    def test_calibration_label_uses_blue_for_calibrated_document(self) -> None:
        window = MainWindow()
        image = QImage(120, 80, QImage.Format.Format_RGB32)
        image.fill(QColor("#FFFFFF"))
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/calibrated_label.png",
            image_size=(120, 80),
        )
        document.initialize_runtime_state()
        document.calibration = Calibration(
            mode="preset",
            pixels_per_unit=5.0,
            unit="um",
            source_label="20x",
        )

        window._add_loaded_document(
            ImageLoadRequest(path=document.path, document=document),
            image,
        )

        self.assertIn("20x", window.calibration_label.text())
        self.assertIn(window._status_color("info"), window.calibration_label.styleSheet())

    def test_calibration_label_is_scrollable_and_does_not_require_expanding_width_for_long_sidecar_name(self) -> None:
        window = MainWindow()
        image = QImage(120, 80, QImage.Format.Format_RGB32)
        image.fill(QColor("#FFFFFF"))
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/calibrated_sidecar_label.png",
            image_size=(120, 80),
        )
        document.initialize_runtime_state()
        document.calibration = Calibration(
            mode="image_scale",
            pixels_per_unit=5.0,
            unit="um",
            source_label="20x",
        )
        document.sidecar_path = "/tmp/this_is_a_very_long_sidecar_filename_for_calibration_display_test_1234567890.fdm.json"

        window._add_loaded_document(
            ImageLoadRequest(path=document.path, document=document),
            image,
        )
        document.sidecar_path = "/tmp/this_is_a_very_long_sidecar_filename_for_calibration_display_test_1234567890.fdm.json"
        window._update_calibration_panel(document)

        self.assertTrue(window.calibration_label.wordWrap())
        self.assertEqual(window.calibration_label.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Ignored)
        self.assertIn("侧车:", window.calibration_label.text())
        self.assertIn("this_is_a_very_long_sidecar_filename_for_calibration_display_test_1234567890.fdm.json", window.calibration_label.toolTip())
        self.assertIsNotNone(window._calibration_label_scroll)
        self.assertEqual(window._calibration_label_scroll.verticalScrollBarPolicy(), Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.assertEqual(window._calibration_label_scroll.maximumHeight(), 118)

    def test_preset_combo_uses_eliding_friendly_size_policy_for_long_names(self) -> None:
        window = MainWindow()
        try:
            window._app_settings.calibration_presets = [
                CalibrationPreset(
                    name="激光共聚焦超长超长超长预设名称用于验证不会撑宽右侧边栏",
                    pixels_per_unit=7.5,
                    unit="um",
                )
            ]

            window._refresh_preset_combo()

            self.assertEqual(
                window.preset_combo.sizeAdjustPolicy(),
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon,
            )
            self.assertEqual(window.preset_combo.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Ignored)
            self.assertEqual(window.preset_combo.minimumContentsLength(), 10)
            self.assertIn("激光共聚焦超长超长超长预设名称", window.preset_combo.toolTip())
        finally:
            window._reset_workspace()
            window.close()

    def test_save_project_persists_project_asset_images_into_assets_directory(self) -> None:
        window = MainWindow()
        preview_frame = QImage(96, 64, QImage.Format.Format_RGB32)
        preview_frame.fill(QColor("#9AD1D4"))
        window._capture_manager.last_frame = lambda: preview_frame.copy()  # type: ignore[method-assign]
        window._capture_manager.is_preview_active = lambda: False  # type: ignore[method-assign]

        window.capture_current_frame()
        captured_document = window.project.documents[0]

        with TemporaryDirectory() as tmp_dir:
            project_path = Path(tmp_dir) / "demo.fdmproj"
            self.assertTrue(window.save_project(str(project_path)))
            asset_path = project_path.with_suffix(".assets") / captured_document.path

            self.assertTrue(asset_path.exists())
            self.assertEqual(asset_path.suffix.lower(), ".png")
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["documents"][0]["source_type"], "project_asset")
            self.assertEqual(payload["documents"][0]["path"], captured_document.path)

    def test_overlay_exports_render_visible_pixels_in_all_modes(self) -> None:
        window, document = self._create_main_window_fixture()

        try:
            with TemporaryDirectory() as tmp_dir:
                for render_mode in [
                    ExportImageRenderMode.FULL_RESOLUTION,
                    ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                    ExportImageRenderMode.CURRENT_VIEWPORT,
                ]:
                    baseline_path = Path(tmp_dir) / f"baseline_{render_mode}.png"
                    measurement_path = Path(tmp_dir) / f"measurement_{render_mode}.png"
                    scale_path = Path(tmp_dir) / f"scale_{render_mode}.png"

                    window._render_overlay_image(
                        document,
                        baseline_path,
                        include_measurements=False,
                        include_scale=False,
                        render_mode=render_mode,
                    )
                    window._render_overlay_image(
                        document,
                        measurement_path,
                        include_measurements=True,
                        include_scale=False,
                        render_mode=render_mode,
                    )
                    window._render_overlay_image(
                        document,
                        scale_path,
                        include_measurements=False,
                        include_scale=True,
                        render_mode=render_mode,
                    )

                    baseline_image = QImage(str(baseline_path))
                    measurement_image = QImage(str(measurement_path))
                    scale_image = QImage(str(scale_path))

                    self.assertFalse(measurement_image.isNull())
                    self.assertFalse(scale_image.isNull())
                    self.assertGreater(self._count_diff_pixels(baseline_image, measurement_image), 0)
                    self.assertGreater(self._count_diff_pixels(baseline_image, scale_image), 0)
        finally:
            window.close()

    def test_group_list_moves_to_left_panel_and_shows_count_badges(self) -> None:
        window = MainWindow()
        try:
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/group_list.png",
                image_size=(300, 200),
            )
            document.initialize_runtime_state()

            window._populate_group_list(document)
            self.assertEqual(window.group_list.viewMode(), QListView.ViewMode.ListMode)
            self.assertEqual(window.group_list.count(), 1)
            ungrouped_widget = window.group_list.itemWidget(window.group_list.item(0))
            self.assertIsInstance(ungrouped_widget, FiberGroupListItemWidget)
            self.assertEqual(ungrouped_widget.labelText(), "未分类")
            self.assertEqual(ungrouped_widget.countText(), "0/0")

            group = document.create_group(color="#1F7A8C", label="棉")
            document.set_active_group(group.id)
            document.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id,
                    mode="manual",
                    line_px=Line(Point(10, 10), Point(80, 10)),
                )
            )
            window._populate_group_list(document)

            self.assertEqual(window.group_list.count(), 1)
            item = window.group_list.item(0)
            widget = window.group_list.itemWidget(item)
            self.assertIsInstance(widget, FiberGroupListItemWidget)
            self.assertEqual(widget.labelText(), "1 棉")
            self.assertEqual(widget.countText(), "1/1")
            self.assertIs(window.group_list.parentWidget().parentWidget(), window._left_panel_splitter)
            margins = window.group_list.viewportMargins()
            self.assertGreaterEqual(margins.left(), 2)
            self.assertGreaterEqual(margins.top(), 2)
        finally:
            window.close()

    def test_group_list_counts_update_for_grouped_and_uncategorized_measurements(self) -> None:
        window = MainWindow()
        try:
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/group_counts.png",
                image_size=(320, 240),
            )
            document.initialize_runtime_state()
            group = document.create_group(color="#1F7A8C", label="棉")
            document.set_active_group(group.id)
            document.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id,
                    mode="manual",
                    line_px=Line(Point(10, 10), Point(90, 10)),
                )
            )
            document.set_active_group(None)
            document.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=None,
                    mode="manual",
                    line_px=Line(Point(10, 20), Point(90, 20)),
                )
            )

            window._populate_group_list(document)

            self.assertEqual(window.group_list.count(), 2)
            first_widget = window.group_list.itemWidget(window.group_list.item(0))
            second_widget = window.group_list.itemWidget(window.group_list.item(1))
            self.assertIsInstance(first_widget, FiberGroupListItemWidget)
            self.assertIsInstance(second_widget, FiberGroupListItemWidget)
            self.assertEqual(first_widget.countText(), "1/1")
            self.assertEqual(second_widget.countText(), "1/1")
            self.assertIsNone(window.group_list.item(0).data(Qt.ItemDataRole.DisplayRole))
            self.assertGreaterEqual(window._add_group_button.minimumWidth(), 100)
        finally:
            window.close()

    def test_group_list_header_and_project_counts_aggregate_across_documents(self) -> None:
        window = MainWindow()
        try:
            current = ImageDocument(
                id=new_id("image"),
                path="/tmp/group_project_counts_current.png",
                image_size=(320, 240),
            )
            current.initialize_runtime_state()
            cotton = current.create_group(color="#1F7A8C", label="棉")
            current.set_active_group(cotton.id)
            current.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=current.id,
                    fiber_group_id=cotton.id,
                    mode="manual",
                    line_px=Line(Point(10, 10), Point(90, 10)),
                )
            )

            other = ImageDocument(
                id=new_id("image"),
                path="/tmp/group_project_counts_other.png",
                image_size=(320, 240),
            )
            other.initialize_runtime_state()
            other_cotton = other.create_group(color="#E07A5F", label="棉")
            other.set_active_group(other_cotton.id)
            other.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=other.id,
                    fiber_group_id=other_cotton.id,
                    mode="manual",
                    line_px=Line(Point(10, 20), Point(90, 20)),
                )
            )
            other.set_active_group(None)
            other.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=other.id,
                    fiber_group_id=None,
                    mode="manual",
                    line_px=Line(Point(10, 30), Point(90, 30)),
                )
            )

            window.project.documents = [current, other]
            window._populate_group_list(current)

            self.assertEqual([label.text() for label in window._group_header_labels], ["颜色", "类别", "（当前/总数）"])
            cotton_widget = window.group_list.itemWidget(window.group_list.item(0))
            self.assertIsInstance(cotton_widget, FiberGroupListItemWidget)
            self.assertEqual(cotton_widget.countText(), "1/2")

            current.set_active_group(None)
            window._populate_group_list(current)
            uncategorized_widget = window.group_list.itemWidget(window.group_list.item(0))
            self.assertIsInstance(uncategorized_widget, FiberGroupListItemWidget)
            self.assertEqual(uncategorized_widget.countText(), "0/1")
        finally:
            window.project.documents = []
            window.close()

    def test_fiber_group_item_uses_stable_dark_palette_text_colors_even_when_inactive(self) -> None:
        widget = FiberGroupListItemWidget("1 棉", 2, 8, "#1F7A8C", selected=False)
        try:
            palette = widget.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor("#2B2B2B"))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor("#F0F4F8"))
            palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.ButtonText, QColor("#000000"))
            palette.setColor(QPalette.ColorRole.Text, QColor("#D9E2EC"))
            palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Text, QColor("#000000"))
            widget.setPalette(palette)

            background, border, text_color, badge_background, badge_border, badge_text = widget._resolved_colors()

            self.assertEqual(text_color.name(), "#e7ecf2")
            self.assertEqual(badge_text.name(), "#d9e2ec")
            self.assertGreater(background.alpha(), 0)
            self.assertGreater(border.alpha(), 0)
            self.assertGreater(badge_background.alpha(), 0)
            self.assertGreater(badge_border.alpha(), 0)
        finally:
            widget.close()

    def test_fiber_group_item_uses_stable_light_palette_text_colors_even_when_inactive(self) -> None:
        widget = FiberGroupListItemWidget("1 棉", 2, 8, "#1F7A8C", selected=False)
        try:
            palette = widget.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor("#F5F7FA"))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor("#182430"))
            palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.ButtonText, QColor("#000000"))
            palette.setColor(QPalette.ColorRole.Text, QColor("#223142"))
            palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Text, QColor("#000000"))
            widget.setPalette(palette)

            background, border, text_color, badge_background, badge_border, badge_text = widget._resolved_colors()

            self.assertEqual(text_color.name(), "#182430")
            self.assertEqual(badge_text.name(), "#223142")
            self.assertGreater(background.alpha(), 0)
            self.assertGreater(border.alpha(), 0)
            self.assertGreater(badge_background.alpha(), 0)
            self.assertGreater(badge_border.alpha(), 0)
        finally:
            widget.close()

    def test_measurement_table_prioritizes_group_and_result_columns(self) -> None:
        window = MainWindow()
        try:
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/measurement_table.png",
                image_size=(240, 160),
            )
            document.initialize_runtime_state()
            document.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=None,
                    mode="manual",
                    line_px=Line(Point(10, 10), Point(80, 10)),
                )
            )

            window._populate_measurement_table(document)
            headers = [window.measurement_table.horizontalHeaderItem(index).text() for index in range(window.measurement_table.columnCount())]

            self.assertEqual(headers, ["种类", "类型", "结果", "单位", "模式", "置信度", "状态", "ID"])
            self.assertIsNotNone(window.measurement_table.item(0, window.TABLE_COL_ID))
            self.assertEqual(window.measurement_table.columnWidth(window.TABLE_COL_GROUP), 150)
        finally:
            window.close()

    def test_measure_toolbar_is_separate_and_exposes_primary_modes(self) -> None:
        window = MainWindow()
        try:
            self.assertIsNotNone(window._file_toolbar)
            self.assertIsNone(window._measure_toolbar)
            self.assertIsInstance(window._measurement_tool_strip, MeasurementToolStrip)
            action_texts = window._measurement_tool_strip.primaryModeLabels()
            self.assertEqual(
                action_texts,
                ["浏览", "手动测量", "边缘吸附", "多边形面积", "自由形状面积", "标准魔棒", "比例尺标定", "叠加标注"],
            )
            visible_actions = [
                window._mode_actions[key]
                for key in [
                    "select",
                    "manual",
                    "snap",
                    "polygon_area",
                    "freehand_area",
                    MagicSegmentToolMode.STANDARD,
                    MagicSegmentToolMode.REFERENCE,
                    "calibration",
                ]
            ]
            self.assertTrue(all(not action.icon().isNull() for action in visible_actions))
            self.assertFalse(window.open_images_action.icon().isNull())
            self.assertFalse(window.save_project_action.icon().isNull())
            self.assertIsInstance(window._magic_tool_button, OverlayToolSplitButton)
            self.assertEqual(window._magic_tool_button.text(), "标准魔棒")
            self.assertEqual(window._magic_tool_button.currentToolKind(), MagicSegmentToolMode.STANDARD)
            self.assertIs(window._magic_tool_menu.parent(), window)
            self.assertIsInstance(window._overlay_tool_button, OverlayToolSplitButton)
            self.assertEqual(window._overlay_tool_button.text(), "叠加标注")
            self.assertLessEqual(window._overlay_tool_button.expandedWidthHint(), 120)
            self.assertGreaterEqual(window._overlay_tool_button.menuAreaWidth(), 28)
            self.assertEqual(window._overlay_tool_button.currentToolKind(), OverlayAnnotationKind.TEXT)
            self.assertIs(window._overlay_tool_menu.parent(), window)
        finally:
            window.close()

    def test_overlay_split_button_hit_areas_and_primary_click(self) -> None:
        button = OverlayToolSplitButton()
        button.resize(button.sizeHint())
        triggered: list[str] = []
        button.primaryTriggered.connect(lambda: triggered.append("primary"))
        try:
            primary_point = QPointF(button.primaryRect().center())
            menu_point = QPointF(button.menuRect().center())
            self.assertEqual(button._hit_part(primary_point.toPoint()), "primary")
            self.assertEqual(button._hit_part(menu_point.toPoint()), "menu")

            button.mousePressEvent(FakeMouseEvent(primary_point, button=Qt.MouseButton.LeftButton))
            button.mouseReleaseEvent(FakeMouseEvent(primary_point, button=Qt.MouseButton.LeftButton))

            self.assertEqual(triggered, ["primary"])
            self.assertEqual(button.height(), 40)
            self.assertGreaterEqual(button.menuRect().width(), 28)
        finally:
            button.close()

    def test_overlay_split_button_syncs_current_tool_and_disable_state(self) -> None:
        window = MainWindow()
        try:
            self.assertIsInstance(window._overlay_tool_button, OverlayToolSplitButton)
            window.set_tool_mode("overlay", overlay_kind=OverlayAnnotationKind.ARROW)

            self.assertEqual(window._overlay_tool_button.currentToolKind(), OverlayAnnotationKind.ARROW)
            self.assertIn("当前：箭头", window._overlay_tool_button.toolTip())
            self.assertTrue(window._overlay_subtool_actions[OverlayAnnotationKind.ARROW].isChecked())

            window._preview_active = True
            window._update_action_states()

            self.assertFalse(window._overlay_tool_button.isEnabled())
        finally:
            window.close()

    def test_measurement_tool_strip_compacts_when_width_is_small(self) -> None:
        window = MainWindow()
        try:
            strip = window._measurement_tool_strip
            self.assertIsInstance(strip, MeasurementToolStrip)
            self.assertEqual(strip.minimumWidth(), 0)
            strip.resize(1200, strip.sizeHint().height())
            strip._sync_auto_compact_mode()
            self.assertFalse(strip.isCompactMode())

            strip.resize(220, strip.sizeHint().height())
            strip._sync_auto_compact_mode()

            self.assertTrue(strip.isCompactMode())
            self.assertEqual(strip.buttonForMode("manual").toolButtonStyle(), Qt.ToolButtonStyle.ToolButtonIconOnly)
            self.assertTrue(window._overlay_tool_button.isCompactMode())
            self.assertGreaterEqual(window._overlay_tool_button.menuAreaWidth(), 28)
        finally:
            window.close()

    def test_overlay_button_matches_primary_button_height(self) -> None:
        window = MainWindow()
        try:
            strip = window._measurement_tool_strip
            self.assertIsInstance(strip, MeasurementToolStrip)
            manual_button = strip.buttonForMode("manual")
            self.assertIsNotNone(manual_button)
            self.assertEqual(window._overlay_tool_button.sizeHint().height(), manual_button.sizeHint().height())
        finally:
            window.close()

    def test_measurement_tool_strip_uses_dark_text_in_light_palette(self) -> None:
        strip = MeasurementToolStrip()
        try:
            palette = strip.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor("#F7F8FA"))
            strip.setPalette(palette)
            strip._apply_theme_styles()

            stylesheet = strip.styleSheet()
            self.assertIn("background: #F5F7FA;", stylesheet)
            self.assertIn("color: #1F2933;", stylesheet)
            self.assertIn("background: #DDF3EF;", stylesheet)
        finally:
            strip.close()

    def test_main_window_uses_application_icon(self) -> None:
        window = MainWindow()
        try:
            self.assertFalse(application_icon().isNull())
            self.assertFalse(window.windowIcon().isNull())
        finally:
            window.close()

    def test_main_window_shows_persistent_version_label_in_status_bar(self) -> None:
        window = MainWindow()
        try:
            self.assertIsNotNone(window._version_label)
            self.assertEqual(window._version_label.text(), f"v{window.project.version}")
            self.assertIs(window._version_label.parent(), window.statusBar())
            self.assertIn(window._status_color("muted"), window._version_label.styleSheet())
        finally:
            window.close()

    def test_status_bar_message_keeps_version_label_visible(self) -> None:
        window = MainWindow()
        try:
            window.show()
            self.app.processEvents()
            window.statusBar().showMessage("测试消息", 1000)
            self.app.processEvents()

            self.assertIsNotNone(window._version_label)
            self.assertTrue(window._version_label.isVisible())
            self.assertEqual(window._version_label.text(), f"v{window.project.version}")
        finally:
            window.close()

    def test_right_panel_hides_onnx_status_and_keeps_area_auto_entry(self) -> None:
        window = MainWindow()
        try:
            group_titles = [box.title() for box in window.findChildren(QGroupBox)]
            self.assertIsNone(getattr(window, "model_status_label", None))
            self.assertIsNotNone(window._area_auto_button)
            self.assertEqual(window._area_auto_button.text(), "面积自动识别...")
            self.assertIn("面积识别", group_titles)
            self.assertNotIn("模型状态", group_titles)
        finally:
            window.close()

    def test_preset_combo_reads_global_settings_presets(self) -> None:
        window = MainWindow()
        try:
            window._app_settings.calibration_presets = [
                CalibrationPreset(
                    name="40x",
                    pixels_per_unit=12.5,
                    unit="um",
                    pixel_distance=250.0,
                    actual_distance=20.0,
                    computed_pixels_per_unit=12.5,
                )
            ]

            window._refresh_preset_combo()

            self.assertEqual(window.preset_combo.count(), 1)
            self.assertIn("40x", window.preset_combo.itemText(0))
        finally:
            window.close()

    def test_default_preset_values_prioritize_current_calibration_line(self) -> None:
        window = MainWindow()
        try:
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/preset_defaults.png",
                image_size=(240, 160),
            )
            document.initialize_runtime_state()
            document.calibration = Calibration(
                mode="preset",
                pixels_per_unit=10.0,
                unit="um",
                source_label="demo preset",
            )
            document.metadata["calibration_line"] = Line(Point(5, 5), Point(105, 5)).to_dict()
            window._app_settings.calibration_presets = [
                CalibrationPreset(
                    name="demo preset",
                    pixels_per_unit=10.0,
                    unit="um",
                    pixel_distance=250.0,
                    actual_distance=25.0,
                    computed_pixels_per_unit=10.0,
                )
            ]

            pixel_distance, actual_distance, unit = window._default_preset_dialog_values(document)

            self.assertAlmostEqual(pixel_distance, 100.0)
            self.assertAlmostEqual(actual_distance, 10.0)
            self.assertEqual(unit, "um")
        finally:
            window.close()

    def test_apply_project_default_calibration_updates_all_open_documents_without_sidecars(self) -> None:
        window = MainWindow()
        try:
            with TemporaryDirectory() as tmp_dir:
                image = QImage(200, 120, QImage.Format.Format_RGB32)
                image.fill(QColor("#FFFFFF"))

                for name in ("a.png", "b.png"):
                    path = str(Path(tmp_dir) / name)
                    document = ImageDocument(
                        id=new_id("image"),
                        path=path,
                        image_size=(image.width(), image.height()),
                    )
                    document.initialize_runtime_state()
                    self._load_document_into_window(window, document, image)

                calibration = Calibration(
                    mode="preset",
                    pixels_per_unit=8.0,
                    unit="um",
                    source_label="40x",
                )

                window._apply_project_default_calibration(calibration, label="测试项目统一标尺")

                self.assertIsNotNone(window.project.project_default_calibration)
                self.assertEqual(window.project.project_default_calibration.mode, "project_default")
                self.assertEqual(len(window.project.documents), 2)
                self.assertTrue(all(document.calibration is not None for document in window.project.documents))
                self.assertTrue(all(document.calibration.mode == "project_default" for document in window.project.documents))
                self.assertFalse(any(Path(document.default_sidecar_path()).exists() for document in window.project.documents))
        finally:
            window.close()

    def test_add_loaded_document_can_prefer_project_default_over_sidecar(self) -> None:
        window = MainWindow()
        try:
            with TemporaryDirectory() as tmp_dir:
                image_path = Path(tmp_dir) / "conflict.png"
                image_path.write_bytes(b"fake")
                sidecar_source = ImageDocument(
                    id=new_id("image"),
                    path=str(image_path),
                    image_size=(200, 120),
                )
                sidecar_source.initialize_runtime_state()
                sidecar_source.calibration = Calibration(
                    mode="image_scale",
                    pixels_per_unit=5.0,
                    unit="um",
                    source_label="图片标尺",
                )
                CalibrationSidecarIO.save_document(sidecar_source)
                original_payload = json.loads(Path(sidecar_source.default_sidecar_path()).read_text(encoding="utf-8"))

                window.project.project_default_calibration = Calibration(
                    mode="project_default",
                    pixels_per_unit=9.0,
                    unit="um",
                    source_label="项目标尺",
                )
                image = QImage(200, 120, QImage.Format.Format_RGB32)
                image.fill(QColor("#FFFFFF"))

                with patch.object(window, "_prompt_project_default_conflict", return_value=True):
                    window._add_loaded_document(
                        ImageLoadRequest(path=str(image_path), document=None),
                        image,
                    )

                self.assertEqual(window.current_document().calibration.mode, "project_default")
                payload = json.loads(Path(sidecar_source.default_sidecar_path()).read_text(encoding="utf-8"))
                self.assertEqual(payload, original_payload)
        finally:
            window.close()

    def test_add_loaded_document_applies_project_group_templates(self) -> None:
        window = MainWindow()
        try:
            window.project.project_group_templates = [
                ProjectGroupTemplate(label="棉", color="#1F7A8C"),
                ProjectGroupTemplate(label="莱赛尔", color="#E07A5F"),
            ]
            image = QImage(200, 120, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/project_groups_apply.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()

            window._add_loaded_document(
                ImageLoadRequest(path=document.path, document=document),
                image,
            )

            loaded = window.current_document()
            self.assertIsNotNone(loaded)
            self.assertEqual([group.label for group in loaded.sorted_groups()], ["棉", "莱赛尔"])
            self.assertFalse(loaded.dirty_flags.session_dirty)
        finally:
            window.close()

    def test_area_auto_recognition_uses_shared_project_group_order_across_documents(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            first = ImageDocument(id=new_id("image"), path="/tmp/area_order_a.png", image_size=(220, 140))
            second = ImageDocument(id=new_id("image"), path="/tmp/area_order_b.png", image_size=(220, 140))
            first.initialize_runtime_state()
            second.initialize_runtime_state()
            first.create_group(color="#E07A5F", label="莱赛尔")
            first.create_group(color="#1F7A8C", label="棉")
            second.create_group(color="#E07A5F", label="莱赛尔")
            second.create_group(color="#1F7A8C", label="棉")
            self._load_document_into_window(window, first, image)
            self._load_document_into_window(window, second, image)

            global_labels: list[str] = []
            instances = [self._area_instance("棉"), self._area_instance("莱赛尔")]
            window._apply_area_inference_result(first, instances, global_group_labels=global_labels, model_name="棉-莱赛尔")
            window._apply_area_inference_result(second, instances, global_group_labels=global_labels, model_name="棉-莱赛尔")

            self.assertEqual(global_labels, ["棉", "莱赛尔"])
            self.assertEqual([template.label for template in window.project.project_group_templates], ["棉", "莱赛尔"])
            self.assertEqual([group.label for group in first.sorted_groups()], ["棉", "莱赛尔"])
            self.assertEqual([group.label for group in second.sorted_groups()], ["棉", "莱赛尔"])
            self.assertEqual(first.get_group_by_number(1).label, "棉")
            self.assertEqual(second.get_group_by_number(1).label, "棉")
        finally:
            window._reset_workspace()
            window.close()

    def test_area_auto_recognition_reorders_existing_groups_without_losing_manual_measurements(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(id=new_id("image"), path="/tmp/area_manual_keep.png", image_size=(220, 140))
            document.initialize_runtime_state()
            hemp = document.create_group(color="#6B7280", label="麻")
            lyocell = document.create_group(color="#E07A5F", label="莱赛尔")
            cotton = document.create_group(color="#1F7A8C", label="棉")
            manual = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=cotton.id,
                mode="manual",
                line_px=Line(Point(20, 20), Point(80, 20)),
            )
            document.add_measurement(manual)
            self._load_document_into_window(window, document, image)

            window._apply_area_inference_result(
                document,
                [self._area_instance("棉")],
                global_group_labels=[],
                model_name="棉-莱赛尔",
            )

            self.assertEqual([group.label for group in document.sorted_groups()], ["棉", "莱赛尔", "麻"])
            self.assertEqual(document.get_group(cotton.id).number, 1)
            self.assertEqual(document.get_measurement(manual.id).fiber_group_id, cotton.id)
            self.assertEqual(document.get_group(hemp.id).number, 3)
        finally:
            window._reset_workspace()
            window.close()

    def test_area_auto_recognition_merges_duplicate_same_label_groups_before_import(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(id=new_id("image"), path="/tmp/area_merge_dupes.png", image_size=(220, 140))
            document.initialize_runtime_state()
            cotton_a = document.create_group(color="#1F7A8C", label="棉")
            document.create_group(color="#6B7280", label="麻")
            cotton_b = document.create_group(color="#7DD3FC", label="棉")
            manual = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=cotton_b.id,
                mode="manual",
                line_px=Line(Point(20, 20), Point(80, 20)),
            )
            document.add_measurement(manual)
            self._load_document_into_window(window, document, image)

            window._apply_area_inference_result(
                document,
                [self._area_instance("棉")],
                global_group_labels=[],
                model_name="棉",
            )

            cotton_groups = document.groups_by_label("棉")
            self.assertEqual(len(cotton_groups), 1)
            self.assertEqual(cotton_groups[0].id, cotton_a.id)
            self.assertEqual(document.get_measurement(manual.id).fiber_group_id, cotton_a.id)
        finally:
            window._reset_workspace()
            window.close()

    def test_area_auto_recognition_restores_suppressed_project_label_when_model_hits_it(self) -> None:
        window = MainWindow()
        try:
            window.project.project_group_templates = [
                ProjectGroupTemplate(label="棉", color="#1F7A8C"),
                ProjectGroupTemplate(label="莱赛尔", color="#E07A5F"),
            ]
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/area_unsuppress.png",
                image_size=(220, 140),
                suppressed_project_group_labels=["棉"],
            )
            document.initialize_runtime_state()
            self._load_document_into_window(window, document, image)

            window._apply_area_inference_result(
                document,
                [self._area_instance("棉")],
                global_group_labels=[],
                model_name="棉-莱赛尔",
            )

            self.assertFalse(document.is_project_group_label_suppressed("棉"))
            self.assertIsNotNone(document.find_group_by_label("棉"))
            self.assertEqual([group.label for group in document.sorted_groups()], ["棉", "莱赛尔"])
        finally:
            window._reset_workspace()
            window.close()

    def test_area_auto_recognition_empty_result_clears_auto_instances_without_reordering_groups(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(id=new_id("image"), path="/tmp/area_empty.png", image_size=(220, 140))
            document.initialize_runtime_state()
            lyocell = document.create_group(color="#E07A5F", label="莱赛尔")
            cotton = document.create_group(color="#1F7A8C", label="棉")
            auto_area = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=cotton.id,
                mode="auto_instance",
                measurement_kind="area",
                polygon_px=[Point(10, 10), Point(30, 10), Point(30, 30), Point(10, 30)],
            )
            document.add_measurement(auto_area)
            self._load_document_into_window(window, document, image)

            window._apply_area_inference_result(
                document,
                [],
                global_group_labels=[],
                model_name="棉-莱赛尔",
            )

            self.assertEqual([group.label for group in document.sorted_groups()], ["莱赛尔", "棉"])
            self.assertEqual(document.get_group(lyocell.id).number, 1)
            self.assertEqual(document.get_group(cotton.id).number, 2)
            self.assertFalse(
                any(measurement.mode == "auto_instance" for measurement in document.measurements)
            )
        finally:
            window._reset_workspace()
            window.close()

    def test_magic_segment_request_uses_in_memory_image_and_cache_key(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="captures/preview.png",
                image_size=(image.width(), image.height()),
                source_type="project_asset",
            )
            document.initialize_runtime_state()
            self._load_document_into_window(window, document, image)

            class FakeRequested:
                def __init__(self) -> None:
                    self.payload = None

                def emit(self, payload) -> None:
                    self.payload = payload

            class FakeWorker:
                def __init__(self) -> None:
                    self.requested = FakeRequested()

            fake_worker = FakeWorker()
            window._prompt_seg_worker = fake_worker

            with (
                patch.object(window, "_ensure_prompt_segmentation_worker", return_value=None),
                patch("fdm.ui.main_window.resolve_interactive_segmentation_backend", return_value=(MagicSegmentModelVariant.EDGE_SAM_3X, None)),
                patch("fdm.ui.main_window.interactive_segmentation_models_ready", return_value=True),
                patch.object(QMessageBox, "warning") as warning_mock,
            ):
                window._on_canvas_magic_segment_requested(
                    document.id,
                    {
                        "request_id": 7,
                        "positive_points": [Point(20, 20)],
                        "negative_points": [Point(30, 30)],
                        "tool_mode": MagicSegmentToolMode.STANDARD,
                        "active_stage": MagicSegmentOperationMode.ADD,
                    },
                )

            self.assertIsNotNone(fake_worker.requested.payload)
            warning_mock.assert_not_called()
            self.assertIs(fake_worker.requested.payload.image, image)
            self.assertEqual(fake_worker.requested.payload.document_id, document.id)
            self.assertEqual(fake_worker.requested.payload.request_id, 7)
            self.assertTrue(fake_worker.requested.payload.cache_key.startswith(f"{document.id}:"))
            self.assertEqual(
                fake_worker.requested.payload.model_variant,
                window._app_settings.magic_segment_model_variant,
            )
            self.assertEqual(fake_worker.requested.payload.tool_mode, MagicSegmentToolMode.STANDARD)
            self.assertEqual(fake_worker.requested.payload.active_stage, MagicSegmentOperationMode.ADD)
        finally:
            window._reset_workspace()
            window.close()

    def test_reference_instance_request_uses_reference_worker_for_existing_area_seed(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="captures/reference_existing_area.png",
                image_size=(image.width(), image.height()),
                source_type="project_asset",
            )
            document.initialize_runtime_state()
            group = document.ensure_default_group()
            measurement = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=group.id,
                mode="magic_segment",
                measurement_kind="area",
                polygon_px=[Point(22, 18), Point(90, 18), Point(90, 72), Point(22, 72)],
                status="ready",
            )
            document.add_measurement(measurement)
            self._load_document_into_window(window, document, image)

            class FakeRequested:
                def __init__(self) -> None:
                    self.payload = None

                def emit(self, payload) -> None:
                    self.payload = payload

            class FakeWorker:
                def __init__(self) -> None:
                    self.requested = FakeRequested()

            fake_worker = FakeWorker()
            window._reference_instance_worker = fake_worker

            with (
                patch.object(window, "_ensure_reference_instance_worker", return_value=None),
                patch("fdm.ui.main_window.resolve_interactive_segmentation_backend", return_value=(MagicSegmentModelVariant.EDGE_SAM_3X, None)),
                patch("fdm.ui.main_window.interactive_segmentation_models_ready", return_value=True),
                patch.object(QMessageBox, "warning") as warning_mock,
            ):
                window._on_canvas_magic_segment_requested(
                    document.id,
                    {
                        "request_id": 9,
                        "tool_mode": MagicSegmentToolMode.REFERENCE,
                        "reference_measurement_id": measurement.id,
                    },
                )

            self.assertIsNotNone(fake_worker.requested.payload)
            warning_mock.assert_not_called()
            self.assertEqual(fake_worker.requested.payload.document_id, document.id)
            self.assertEqual(fake_worker.requested.payload.request_id, 9)
            self.assertEqual(fake_worker.requested.payload.model_variant, window._app_settings.magic_segment_model_variant)
            self.assertEqual(fake_worker.requested.payload.reference_box, None)
            self.assertEqual(fake_worker.requested.payload.reference_polygon_px, measurement.polygon_px)
            self.assertEqual(fake_worker.requested.payload.reference_area_rings_px, measurement.area_rings_px)
        finally:
            window._reset_workspace()
            window.close()

    def test_reference_instance_request_uses_reference_worker_for_drag_box_seed(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="captures/reference_drag_box.png",
                image_size=(image.width(), image.height()),
                source_type="project_asset",
            )
            document.initialize_runtime_state()
            self._load_document_into_window(window, document, image)

            class FakeRequested:
                def __init__(self) -> None:
                    self.payload = None

                def emit(self, payload) -> None:
                    self.payload = payload

            class FakeWorker:
                def __init__(self) -> None:
                    self.requested = FakeRequested()

            fake_worker = FakeWorker()
            window._reference_instance_worker = fake_worker

            with (
                patch.object(window, "_ensure_reference_instance_worker", return_value=None),
                patch("fdm.ui.main_window.resolve_interactive_segmentation_backend", return_value=(MagicSegmentModelVariant.EDGE_SAM_3X, None)),
                patch("fdm.ui.main_window.interactive_segmentation_models_ready", return_value=True),
                patch.object(QMessageBox, "warning") as warning_mock,
            ):
                window._on_canvas_magic_segment_requested(
                    document.id,
                    {
                        "request_id": 8,
                        "tool_mode": MagicSegmentToolMode.REFERENCE,
                        "reference_box": {
                            "start": Point(24, 28),
                            "end": Point(96, 104),
                        },
                    },
                )

            self.assertIsNotNone(fake_worker.requested.payload)
            warning_mock.assert_not_called()
            self.assertEqual(fake_worker.requested.payload.document_id, document.id)
            self.assertEqual(fake_worker.requested.payload.request_id, 8)
            self.assertEqual(fake_worker.requested.payload.reference_box, (Point(24, 28), Point(96, 104)))
            self.assertEqual(fake_worker.requested.payload.reference_polygon_px, [])
            self.assertEqual(fake_worker.requested.payload.reference_area_rings_px, [])
        finally:
            window._reset_workspace()
            window.close()

    def test_add_fiber_group_global_syncs_existing_documents(self) -> None:
        window = MainWindow()
        dialogs: list[FiberGroupDialog] = []
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            first = ImageDocument(id=new_id("image"), path="/tmp/group_global_a.png", image_size=(220, 140))
            second = ImageDocument(id=new_id("image"), path="/tmp/group_global_b.png", image_size=(220, 140))
            first.initialize_runtime_state()
            second.initialize_runtime_state()
            self._load_document_into_window(window, first, image)
            self._load_document_into_window(window, second, image)
            window._set_current_document(first.id)

            def fake_exec(dialog_self) -> int:
                dialogs.append(dialog_self)
                return dialog_self.DialogCode.Accepted

            def fake_values(dialog_self):
                return ("棉", True)

            with patch.object(FiberGroupDialog, "exec", fake_exec), patch.object(FiberGroupDialog, "values", fake_values):
                window.add_fiber_group()

            self.assertEqual([template.label for template in window.project.project_group_templates], ["棉"])
            self.assertIsNotNone(first.find_group_by_label("棉"))
            self.assertIsNotNone(second.find_group_by_label("棉"))
        finally:
            for dialog in dialogs:
                dialog.close()
            window.close()

    def test_add_fiber_group_duplicate_local_is_noop(self) -> None:
        window = MainWindow()
        dialogs: list[FiberGroupDialog] = []
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(id=new_id("image"), path="/tmp/group_local_noop.png", image_size=(220, 140))
            document.initialize_runtime_state()
            group = document.create_group(color="#1F7A8C", label="棉")
            document.set_active_group(group.id)
            self._load_document_into_window(window, document, image)

            def fake_exec(dialog_self) -> int:
                dialogs.append(dialog_self)
                return dialog_self.DialogCode.Accepted

            def fake_values(dialog_self):
                return ("棉", False)

            with patch.object(FiberGroupDialog, "exec", fake_exec), patch.object(FiberGroupDialog, "values", fake_values):
                window.add_fiber_group()

            self.assertEqual([item.label for item in document.sorted_groups()], ["棉"])
            self.assertEqual(document.active_group_id, group.id)
        finally:
            for dialog in dialogs:
                dialog.close()
            window.close()

    def test_rename_group_to_existing_merges_measurements(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(id=new_id("image"), path="/tmp/group_merge.png", image_size=(220, 140))
            document.initialize_runtime_state()
            cotton = document.create_group(color="#1F7A8C", label="棉")
            hemp = document.create_group(color="#E07A5F", label="麻")
            document.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=hemp.id,
                    mode="manual",
                    line_px=Line(Point(10, 10), Point(80, 10)),
                )
            )
            document.set_active_group(hemp.id)
            self._load_document_into_window(window, document, image)

            with patch("fdm.ui.main_window.QInputDialog.getText", return_value=("棉", True)), patch(
                "fdm.ui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ):
                window.rename_active_group()

            self.assertEqual([group.label for group in document.sorted_groups()], ["棉"])
            self.assertEqual(document.measurements[0].fiber_group_id, cotton.id)
            self.assertEqual(document.active_group_id, cotton.id)
        finally:
            window.close()

    def test_delete_project_global_group_adds_local_suppression(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(id=new_id("image"), path="/tmp/group_delete_global.png", image_size=(220, 140))
            document.initialize_runtime_state()
            group = document.create_group(color="#1F7A8C", label="棉")
            document.set_active_group(group.id)
            window.project.project_group_templates = [ProjectGroupTemplate(label="棉", color="#1F7A8C")]
            self._load_document_into_window(window, document, image)

            with patch(
                "fdm.ui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ):
                window.delete_active_group()

            self.assertIsNone(document.find_group_by_label("棉"))
            self.assertEqual(document.suppressed_project_group_labels, ["棉"])
            self.assertFalse(window._apply_project_group_templates_to_document(document))
            self.assertIsNone(document.find_group_by_label("棉"))
        finally:
            window.close()

    def test_merge_legacy_presets_renames_name_conflicts(self) -> None:
        window = MainWindow()
        try:
            window._app_settings.calibration_presets = [
                CalibrationPreset(
                    name="40x",
                    pixels_per_unit=10.0,
                    unit="um",
                    pixel_distance=100.0,
                    actual_distance=10.0,
                    computed_pixels_per_unit=10.0,
                )
            ]
            legacy_presets = [
                CalibrationPreset(
                    name="40x",
                    pixels_per_unit=12.0,
                    unit="um",
                    pixel_distance=120.0,
                    actual_distance=10.0,
                    computed_pixels_per_unit=12.0,
                ),
                CalibrationPreset(
                    name="40x",
                    pixels_per_unit=10.0,
                    unit="um",
                    pixel_distance=100.0,
                    actual_distance=10.0,
                    computed_pixels_per_unit=10.0,
                ),
            ]

            with patch.object(window, "_save_app_settings", return_value=True):
                imported_count = window._merge_legacy_calibration_presets(legacy_presets)

            self.assertEqual(imported_count, 1)
            self.assertEqual([preset.name for preset in window._app_settings.calibration_presets], ["40x", "40x (导入)"])
        finally:
            window.close()

    def test_magic_segment_controls_are_only_visible_in_magic_mode(self) -> None:
        window = MainWindow()
        try:
            self.assertIsNotNone(window._magic_controls_widget)
            self.assertIsInstance(window._measurement_tool_strip, MeasurementToolStrip)
            self.assertFalse(window._measurement_tool_strip.isMagicContextVisible())
            self.assertIsNotNone(window._magic_prompt_label)

            window._measurement_tool_strip.resize(1600, window._measurement_tool_strip.sizeHint().height())
            window.set_tool_mode(MagicSegmentToolMode.STANDARD)
            self.assertTrue(window._measurement_tool_strip.isMagicContextVisible())
            self.assertTrue(window._measurement_tool_strip.isContextInline())
            self.assertTrue(window._magic_prompt_label.isHidden())
            window.set_tool_mode(MagicSegmentToolMode.REFERENCE)
            self.assertTrue(window._measurement_tool_strip.isMagicContextVisible())
            self.assertEqual(window._magic_tool_button.currentToolKind(), MagicSegmentToolMode.REFERENCE)
            self.assertFalse(window._magic_prompt_label.isHidden())

            window.set_tool_mode("select")
            self.assertFalse(window._measurement_tool_strip.isMagicContextVisible())
        finally:
            window.close()

    def test_preview_analysis_controls_live_in_context_row(self) -> None:
        window = MainWindow()
        try:
            self.assertIsInstance(window._measurement_tool_strip, MeasurementToolStrip)
            self.assertFalse(window._measurement_tool_strip.isPreviewContextVisible())

            window._measurement_tool_strip.resize(1600, window._measurement_tool_strip.sizeHint().height())
            window._preview_active = True
            window._update_preview_analysis_controls()

            self.assertTrue(window._measurement_tool_strip.isPreviewContextVisible())
            self.assertTrue(window._measurement_tool_strip.isContextInline())

            window._preview_active = False
            window._update_preview_analysis_controls()

            self.assertFalse(window._measurement_tool_strip.isPreviewContextVisible())
        finally:
            window.close()

    def test_measurement_tool_strip_stacks_context_only_when_width_is_insufficient(self) -> None:
        window = MainWindow()
        try:
            strip = window._measurement_tool_strip
            self.assertIsInstance(strip, MeasurementToolStrip)
            window.resize(1600, 900)
            window.show()
            self.app.processEvents()
            window.set_tool_mode("magic_segment")
            self.app.processEvents()
            self.assertTrue(strip.isMagicContextVisible())
            self.assertTrue(strip.isContextInline())
            self.assertFalse(strip.isContextStacked())
            self.assertFalse(strip.isCompactMode())

            window.resize(1000, 900)
            self.app.processEvents()
            self.assertTrue(strip.isContextStacked())
            self.assertFalse(strip.isCompactMode())

            window.resize(1600, 900)
            self.app.processEvents()
            self.assertTrue(strip.isContextInline())

            window.set_tool_mode("select")
            window._preview_active = True
            window._update_magic_segment_controls()
            window._update_preview_analysis_controls()
            self.app.processEvents()
            self.assertTrue(strip.isPreviewContextVisible())
            self.assertTrue(strip.isContextInline())
            self.assertFalse(strip.isCompactMode())

            window.resize(1000, 900)
            self.app.processEvents()
            self.assertTrue(strip.isContextStacked())
            self.assertFalse(strip.isCompactMode())

            window.resize(692, 900)
            self.app.processEvents()
            self.assertTrue(strip.isPreviewContextVisible())
            self.assertTrue(strip.isContextInline() or strip.isContextStacked())
        finally:
            window.close()

    def test_stacked_context_row_has_visible_height_when_window_is_narrow(self) -> None:
        window = MainWindow()
        try:
            strip = window._measurement_tool_strip
            self.assertIsInstance(strip, MeasurementToolStrip)
            window.resize(692, 755)
            window.show()
            self.app.processEvents()

            window.set_tool_mode("magic_segment")
            self.app.processEvents()

            self.assertTrue(strip.isContextStacked())
            self.assertTrue(strip._context_host.isVisible())
            self.assertGreater(strip._context_host.height(), 0)
            self.assertGreater(strip._context_host.geometry().top(), strip._top_row.geometry().bottom())
            self.assertTrue(window._magic_controls_widget.isVisible())
        finally:
            window.close()

    def test_context_prefers_stacking_before_compacting_primary_tools(self) -> None:
        window = MainWindow()
        try:
            strip = window._measurement_tool_strip
            self.assertIsInstance(strip, MeasurementToolStrip)
            window.resize(1000, 900)
            window.show()
            self.app.processEvents()

            window.set_tool_mode("magic_segment")
            self.app.processEvents()

            self.assertTrue(strip.isContextStacked())
            self.assertFalse(strip.isCompactMode())
        finally:
            window.close()

    def test_inline_context_keeps_full_width_without_clipping_buttons(self) -> None:
        window = MainWindow()
        try:
            strip = window._measurement_tool_strip
            self.assertIsInstance(strip, MeasurementToolStrip)
            window.resize(1280, 900)
            window.show()
            self.app.processEvents()

            window.set_tool_mode("magic_segment")
            self.app.processEvents()

            self.assertTrue(strip.isContextInline())
            self.assertEqual(strip._context_host.width(), window._magic_controls_widget.sizeHint().width())
        finally:
            window.close()

    def test_shortcut_help_dialog_opens_from_help_action(self) -> None:
        window = MainWindow()
        dialogs: list[ShortcutHelpDialog] = []

        def fake_exec(dialog_self) -> int:
            dialogs.append(dialog_self)
            return dialog_self.DialogCode.Accepted

        try:
            with patch.object(ShortcutHelpDialog, "exec", fake_exec):
                window.open_shortcut_help_dialog()
            self.assertEqual(len(dialogs), 1)
            self.assertIn("R", dialogs[0]._content.toPlainText())
            self.assertIn("T", dialogs[0]._content.toPlainText())
            self.assertIn("Enter / F", dialogs[0]._content.toPlainText())
        finally:
            for dialog in dialogs:
                dialog.close()
            window.close()

    def test_right_panel_uses_vertical_splitter_for_resizable_measurement_area(self) -> None:
        window = MainWindow()
        try:
            splitters = window.findChildren(QSplitter)
            self.assertTrue(any(splitter.orientation() == Qt.Orientation.Vertical for splitter in splitters))
            right_titles = [box.title() for box in window._right_panel.findChildren(QGroupBox)]
            self.assertNotIn("纤维类别", right_titles)
        finally:
            window.close()

    def test_tab_strip_uses_scroll_buttons_without_shrinking_left_panel(self) -> None:
        window = MainWindow()
        try:
            self.assertTrue(window.tab_widget.usesScrollButtons())
            self.assertIsNotNone(window._left_panel_splitter)
            self.assertGreaterEqual(window._left_panel.minimumWidth(), 280)
            self.assertEqual(window._left_panel_splitter.orientation(), Qt.Orientation.Vertical)
            window.resize(1280, 860)
            window.show()
            self.app.processEvents()
            self.assertGreaterEqual(window._left_panel_splitter.sizes()[1], 340)
        finally:
            window.close()

    def test_polygon_area_tool_commits_when_clicking_first_point(self) -> None:
        document, _, canvas = self._create_canvas_document()
        canvas.set_tool_mode("polygon_area")
        commits: list[tuple[str, str, object]] = []
        canvas.lineCommitted.connect(lambda document_id, mode, payload: commits.append((document_id, mode, payload)))

        points = [Point(20, 20), Point(100, 20), Point(90, 90)]
        for point in points:
            canvas.mousePressEvent(FakeMouseEvent(canvas.image_to_widget(point), button=Qt.MouseButton.LeftButton))
        canvas.mousePressEvent(FakeMouseEvent(canvas.image_to_widget(points[0]), button=Qt.MouseButton.LeftButton))

        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0][1], "polygon_area")
        self.assertEqual(commits[0][2]["measurement_kind"], "area")
        self.assertEqual(len(commits[0][2]["polygon_px"]), 3)

    def test_freehand_area_tool_commits_on_release(self) -> None:
        document, _, canvas = self._create_canvas_document()
        canvas.set_tool_mode("freehand_area")
        commits: list[tuple[str, str, object]] = []
        canvas.lineCommitted.connect(lambda document_id, mode, payload: commits.append((document_id, mode, payload)))

        canvas.mousePressEvent(FakeMouseEvent(canvas.image_to_widget(Point(20, 20)), button=Qt.MouseButton.LeftButton))
        canvas._freehand_last_sample_at -= 1.0
        canvas.mouseMoveEvent(FakeMouseEvent(canvas.image_to_widget(Point(100, 20)), button=Qt.MouseButton.LeftButton))
        canvas._freehand_last_sample_at -= 1.0
        canvas.mouseMoveEvent(FakeMouseEvent(canvas.image_to_widget(Point(100, 80)), button=Qt.MouseButton.LeftButton))
        canvas._freehand_last_sample_at -= 1.0
        canvas.mouseMoveEvent(FakeMouseEvent(canvas.image_to_widget(Point(25, 85)), button=Qt.MouseButton.LeftButton))
        canvas.mouseReleaseEvent(FakeMouseEvent(canvas.image_to_widget(Point(25, 85)), button=Qt.MouseButton.LeftButton))

        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0][1], "freehand_area")
        self.assertGreaterEqual(len(commits[0][2]["polygon_px"]), 3)

    def test_magic_segment_tool_emits_request_and_commits_preview(self) -> None:
        document, image, canvas = self._create_canvas_document()
        canvas.set_tool_mode("magic_segment")
        requests: list[tuple[str, object]] = []
        commits: list[tuple[str, str, object]] = []
        canvas.magicSegmentRequested.connect(lambda document_id, payload: requests.append((document_id, payload)))
        canvas.lineCommitted.connect(lambda document_id, mode, payload: commits.append((document_id, mode, payload)))

        canvas.mousePressEvent(FakeMouseEvent(canvas.image_to_widget(Point(30, 35)), button=Qt.MouseButton.LeftButton))

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0][0], document.id)
        self.assertEqual(len(requests[0][1]["positive_points"]), 1)
        self.assertEqual(len(requests[0][1]["negative_points"]), 0)

        request_id = requests[0][1]["request_id"]
        canvas.apply_magic_segment_result(
            request_id,
            self._rect_mask(image.width(), image.height(), (20, 20, 92, 80)),
        )

        self.assertTrue(canvas.has_magic_segment_preview())
        self.assertGreaterEqual(len(canvas._magic_segment.primary_polygon), 3)
        commit_result = canvas.commit_magic_segment_preview()
        self.assertTrue(bool(commit_result["committed"]))
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0][1], "magic_segment")
        self.assertEqual(commits[0][2]["measurement_kind"], "area")
        self.assertEqual(len(commits[0][2]["area_rings_px"]), 1)

    def test_magic_segment_shortcuts_toggle_prompt_commit_and_cancel(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/magic_segment.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()

            self._load_document_into_window(window, document, image)
            canvas = window.current_canvas()
            self.assertIsNotNone(canvas)
            window.set_tool_mode("magic_segment")

            self.assertEqual(canvas.current_magic_segment_prompt_type(), "positive")
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_R))
            self.assertEqual(canvas.current_magic_segment_prompt_type(), "negative")
            self.assertEqual(canvas.current_magic_segment_operation_mode(), MagicSegmentOperationMode.ADD)
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_T))
            self.assertEqual(canvas.current_magic_segment_operation_mode(), MagicSegmentOperationMode.ADD)
            self.assertIn("请先完成第一个形状草稿", window.statusBar().currentMessage())

            canvas._magic_segment.request_id = 1
            canvas._magic_segment.pending_stage = MagicSegmentOperationMode.ADD
            canvas.apply_magic_segment_result(
                1,
                self._rect_mask(image.width(), image.height(), (24, 24, 92, 80)),
            )

            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_T))
            self.assertEqual(canvas.current_magic_segment_operation_mode(), MagicSegmentOperationMode.SUBTRACT)
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_T))
            self.assertEqual(canvas.current_magic_segment_operation_mode(), MagicSegmentOperationMode.ADD)
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_Return))

            self.assertEqual(len(document.measurements), 1)
            self.assertEqual(document.measurements[0].mode, "magic_segment")

            canvas.set_tool_mode("magic_segment")
            canvas._magic_segment.primary_positive_points = [Point(40, 40)]
            self.assertTrue(canvas.has_magic_segment_session())
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_Escape))
            self.assertFalse(canvas.has_magic_segment_session())
        finally:
            window._reset_workspace()
            window.close()

    def test_magic_segment_subtract_stage_preserves_primary_draft_until_commit(self) -> None:
        document, image, canvas = self._create_canvas_document()
        canvas.set_tool_mode("magic_segment")
        commits: list[tuple[str, str, object]] = []
        canvas.lineCommitted.connect(lambda document_id, mode, payload: commits.append((document_id, mode, payload)))

        canvas._magic_segment.request_id = 1
        canvas._magic_segment.pending_stage = MagicSegmentOperationMode.ADD
        canvas.apply_magic_segment_result(
            1,
            self._rect_mask(image.width(), image.height(), (20, 18, 150, 102)),
        )
        self.assertTrue(canvas.has_magic_segment_preview())

        self.assertEqual(canvas.cycle_magic_segment_operation_mode(), MagicSegmentOperationMode.SUBTRACT)
        canvas._magic_segment.request_id = 2
        canvas._magic_segment.pending_stage = MagicSegmentOperationMode.SUBTRACT
        canvas.apply_magic_segment_result(
            2,
            self._rect_mask(image.width(), image.height(), (56, 38, 108, 82)),
        )

        self.assertTrue(canvas.has_magic_segment_preview())
        self.assertGreaterEqual(len(canvas._magic_segment.primary_polygon), 3)
        self.assertGreaterEqual(len(canvas._magic_segment.subtract_polygon), 3)
        self.assertIsNotNone(canvas._magic_segment.primary_mask)
        self.assertIsNotNone(canvas._magic_segment.subtract_mask)

        commit_result = canvas.commit_magic_segment_preview()

        self.assertTrue(bool(commit_result["committed"]))
        self.assertGreaterEqual(int(commit_result["opened_holes"]), 1)
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0][1], "magic_segment")

    def test_magic_segment_inference_does_not_show_cleanup_messages_before_confirm(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/magic_segment_status.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()

            self._load_document_into_window(window, document, image)
            canvas = window.current_canvas()
            self.assertIsNotNone(canvas)
            window.set_tool_mode("magic_segment")
            canvas._magic_segment.request_id = 1
            canvas._magic_segment.pending_stage = MagicSegmentOperationMode.ADD
            window._on_prompt_segmentation_succeeded(
                document.id,
                1,
                PromptSegmentationResult(
                    mask=self._rect_mask(image.width(), image.height(), (20, 18, 150, 102)),
                    polygon_px=[],
                    area_rings_px=[],
                    area_px=0.0,
                    metadata={},
                ),
            )

            self.assertNotIn("自动开缝", window.statusBar().currentMessage())
            self.assertNotIn("仅保留最大连通区域", window.statusBar().currentMessage())
        finally:
            window._reset_workspace()
            window.close()

    def test_magic_segment_preview_uses_largest_polygon_instead_of_raw_mask_fragments(self) -> None:
        document, image, canvas = self._create_canvas_document()
        canvas.set_tool_mode("magic_segment")

        fragmented_mask = self._rect_mask(image.width(), image.height(), (28, 24, 136, 92))
        fragmented_mask[16:22, 164:170] = True
        canvas._magic_segment.request_id = 1
        canvas._magic_segment.pending_stage = MagicSegmentOperationMode.ADD

        canvas.apply_magic_segment_result(1, fragmented_mask)

        self.assertGreaterEqual(len(canvas._magic_segment.primary_polygon), 3)
        self.assertIsNotNone(canvas._magic_segment.primary_mask)
        self.assertFalse(bool(canvas._magic_segment.primary_mask[18, 166]))

    def test_magic_segment_commit_opens_slit_for_enclosed_subtract_shape_even_with_raw_fragment_noise(self) -> None:
        document, image, canvas = self._create_canvas_document()
        canvas.set_tool_mode("magic_segment")

        primary_mask = self._rect_mask(image.width(), image.height(), (20, 18, 150, 102))
        subtract_mask = self._rect_mask(image.width(), image.height(), (56, 38, 108, 82))
        subtract_mask[10:16, 170:176] = True

        canvas._magic_segment.request_id = 1
        canvas._magic_segment.pending_stage = MagicSegmentOperationMode.ADD
        canvas.apply_magic_segment_result(1, primary_mask)
        self.assertEqual(canvas.cycle_magic_segment_operation_mode(), MagicSegmentOperationMode.SUBTRACT)

        canvas._magic_segment.request_id = 2
        canvas._magic_segment.pending_stage = MagicSegmentOperationMode.SUBTRACT
        canvas.apply_magic_segment_result(2, subtract_mask)

        commit_result = canvas.commit_magic_segment_preview()

        self.assertGreaterEqual(int(commit_result["opened_holes"]), 1)
        self.assertFalse(bool(commit_result["result_empty"]))

    def test_magic_segment_commit_preserves_area_rings_for_hole_geometry(self) -> None:
        document, image, canvas = self._create_canvas_document()
        canvas.set_tool_mode("magic_segment")
        commits: list[tuple[str, str, object]] = []
        canvas.lineCommitted.connect(lambda document_id, mode, payload: commits.append((document_id, mode, payload)))

        primary_mask = self._rect_mask(image.width(), image.height(), (20, 18, 150, 102))
        subtract_mask = self._rect_mask(image.width(), image.height(), (56, 38, 108, 82))

        canvas._magic_segment.request_id = 1
        canvas._magic_segment.pending_stage = MagicSegmentOperationMode.ADD
        canvas.apply_magic_segment_result(1, primary_mask)
        canvas.cycle_magic_segment_operation_mode()
        canvas._magic_segment.request_id = 2
        canvas._magic_segment.pending_stage = MagicSegmentOperationMode.SUBTRACT
        canvas.apply_magic_segment_result(2, subtract_mask)

        commit_result = canvas.commit_magic_segment_preview()

        self.assertTrue(bool(commit_result["committed"]))
        self.assertEqual(len(commits), 1)
        payload = commits[0][2]
        self.assertEqual(len(payload["area_rings_px"]), 2)
        self.assertGreaterEqual(len(payload["polygon_px"]), 3)

    def test_area_hit_test_uses_area_rings_instead_of_simplified_polygon(self) -> None:
        document, _image, canvas = self._create_canvas_document()
        area_measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="magic_segment",
            measurement_kind="area",
            polygon_px=[Point(20, 20), Point(60, 20), Point(60, 120), Point(20, 120)],
            area_rings_px=[
                [
                    Point(20, 20),
                    Point(70, 20),
                    Point(70, 60),
                    Point(30, 60),
                    Point(30, 120),
                    Point(20, 120),
                ]
            ],
        )
        document.add_measurement(area_measurement)

        self.assertEqual(canvas._hit_test_area_measurement(Point(64, 40)), area_measurement.id)

    def test_text_tool_adds_annotation_and_delete_removes_selected_text(self) -> None:
        window = MainWindow()
        try:
            image = QImage(240, 160, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/text_tool.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            self._load_document_into_window(window, document, image)

            with patch("fdm.ui.main_window.QInputDialog.getMultiLineText", return_value=("说明文本", True)):
                window._on_canvas_overlay_create_requested(
                    document.id,
                    {"kind": OverlayAnnotationKind.TEXT, "anchor_px": Point(24, 32)},
                )

            self.assertEqual(len(document.overlay_annotations), 1)
            self.assertEqual(document.overlay_annotations[0].content, "说明文本")
            self.assertEqual(document.selected_overlay_id, document.overlay_annotations[0].id)

            window.delete_selected_measurement()
            self.assertEqual(len(document.overlay_annotations), 0)
        finally:
            window.close()

    def test_overlay_shape_create_and_edit_roundtrip_through_main_window(self) -> None:
        window = MainWindow()
        try:
            image = QImage(240, 160, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/overlay_shape.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            self._load_document_into_window(window, document, image)

            window._on_canvas_overlay_create_requested(
                document.id,
                {
                    "kind": OverlayAnnotationKind.RECT,
                    "start_px": Point(20, 24),
                    "end_px": Point(90, 84),
                },
            )

            overlay = document.overlay_annotations[0]
            self.assertEqual(overlay.kind, OverlayAnnotationKind.RECT)
            self.assertAlmostEqual(overlay.start_px.x, 20.0)

            edited = overlay.clone(start_px=Point(26, 30), end_px=Point(110, 92))
            window._on_canvas_overlay_edited(document.id, overlay.id, edited)

            updated = document.get_overlay_annotation(overlay.id)
            self.assertIsNotNone(updated)
            self.assertAlmostEqual(updated.start_px.x, 26.0)
            self.assertAlmostEqual(updated.end_px.x, 110.0)
        finally:
            window.close()

    def test_overlay_rect_and_circle_snap_to_square_with_shift(self) -> None:
        document, _, canvas = self._create_canvas_document()
        created: list[object] = []
        canvas.overlayCreateRequested.connect(lambda document_id, payload: created.append(payload))

        canvas.set_tool_mode("overlay", overlay_kind=OverlayAnnotationKind.RECT)
        start = canvas.image_to_widget(Point(20, 20))
        end = canvas.image_to_widget(Point(80, 50))
        canvas.mousePressEvent(FakeMouseEvent(start, button=Qt.MouseButton.LeftButton))
        canvas.mouseMoveEvent(FakeMouseEvent(end, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.ShiftModifier))
        canvas.mouseReleaseEvent(FakeMouseEvent(end, button=Qt.MouseButton.LeftButton))

        rect_payload = created[0]
        self.assertEqual(rect_payload["kind"], OverlayAnnotationKind.RECT)
        self.assertAlmostEqual(abs(rect_payload["end_px"].x - rect_payload["start_px"].x), abs(rect_payload["end_px"].y - rect_payload["start_px"].y))

        created.clear()
        canvas.set_tool_mode("overlay", overlay_kind=OverlayAnnotationKind.CIRCLE)
        circle_end = canvas.image_to_widget(Point(55, 95))
        canvas.mousePressEvent(FakeMouseEvent(start, button=Qt.MouseButton.LeftButton))
        canvas.mouseMoveEvent(FakeMouseEvent(circle_end, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.ShiftModifier))
        canvas.mouseReleaseEvent(FakeMouseEvent(circle_end, button=Qt.MouseButton.LeftButton))

        circle_payload = created[0]
        self.assertEqual(circle_payload["kind"], OverlayAnnotationKind.CIRCLE)
        self.assertAlmostEqual(abs(circle_payload["end_px"].x - circle_payload["start_px"].x), abs(circle_payload["end_px"].y - circle_payload["start_px"].y))

    def test_open_image_view_mode_actual_is_applied_to_new_canvas(self) -> None:
        window = MainWindow()
        try:
            window._app_settings.open_image_view_mode = OpenImageViewMode.ACTUAL
            image = QImage(240, 160, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/view_mode_actual.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()

            self._load_document_into_window(window, document, image)
            canvas = window.current_canvas()

            self.assertIsNotNone(canvas)
            self.assertAlmostEqual(canvas._zoom, 1.0)
            self.assertAlmostEqual(canvas._pan.x, 20.0)
            self.assertAlmostEqual(canvas._pan.y, 20.0)
        finally:
            window.close()

    def test_manual_scale_anchor_and_text_are_visible_in_exports(self) -> None:
        window, document = self._create_main_window_fixture()
        try:
            document.scale_overlay_anchor = Point(36, 48)
            document.add_text_annotation(
                TextAnnotation(
                    id=new_id("text"),
                    image_id=document.id,
                    content="样品A",
                    anchor_px=Point(40, 44),
                )
            )
            window._app_settings.scale_overlay_placement_mode = "manual"

            with TemporaryDirectory() as tmp_dir:
                baseline_path = Path(tmp_dir) / "baseline.png"
                measurement_path = Path(tmp_dir) / "measurement.png"
                scale_path = Path(tmp_dir) / "scale.png"
                window._render_overlay_image(
                    document,
                    baseline_path,
                    include_measurements=False,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                )
                window._render_overlay_image(
                    document,
                    measurement_path,
                    include_measurements=True,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                )
                window._render_overlay_image(
                    document,
                    scale_path,
                    include_measurements=False,
                    include_scale=True,
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                )

                baseline_image = QImage(str(baseline_path))
                measurement_image = QImage(str(measurement_path))
                scale_image = QImage(str(scale_path))
                self.assertGreater(self._count_diff_pixels(baseline_image, measurement_image), 0)
                self.assertGreater(self._count_diff_pixels(baseline_image, scale_image), 0)
        finally:
            window.close()

    def test_scale_overlay_style_and_length_change_export_output(self) -> None:
        window, document = self._create_main_window_fixture()
        try:
            with TemporaryDirectory() as tmp_dir:
                line_path = Path(tmp_dir) / "scale_line.png"
                bar_path = Path(tmp_dir) / "scale_bar.png"

                window._app_settings.scale_overlay_style = "line"
                window._app_settings.scale_overlay_length_value = 20.0
                window._render_overlay_image(
                    document,
                    line_path,
                    include_measurements=False,
                    include_scale=True,
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                )

                window._app_settings.scale_overlay_style = "bar"
                window._app_settings.scale_overlay_length_value = 30.0
                window._render_overlay_image(
                    document,
                    bar_path,
                    include_measurements=False,
                    include_scale=True,
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                )

                line_image = QImage(str(line_path))
                bar_image = QImage(str(bar_path))
                self.assertGreater(self._count_diff_pixels(line_image, bar_image), 0)
        finally:
            window.close()

    def test_scale_overlay_renders_for_uncalibrated_document_using_pixel_length(self) -> None:
        window = MainWindow()
        try:
            image = QImage(260, 180, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/uncalibrated_scale.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            canvas = DocumentCanvas()
            canvas.resize(340, 240)
            canvas.set_document(document, image)
            window._images[document.id] = image
            window._canvases[document.id] = canvas
            window._app_settings.scale_overlay_length_value = 48.0

            with TemporaryDirectory() as tmp_dir:
                baseline_path = Path(tmp_dir) / "baseline.png"
                scale_path = Path(tmp_dir) / "scale.png"
                window._render_overlay_image(
                    document,
                    baseline_path,
                    include_measurements=False,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                )
                window._render_overlay_image(
                    document,
                    scale_path,
                    include_measurements=False,
                    include_scale=True,
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                )

                baseline_image = QImage(str(baseline_path))
                scale_image = QImage(str(scale_path))
                self.assertGreater(self._count_diff_pixels(baseline_image, scale_image), 0)
        finally:
            window.close()

    def test_settings_dialog_collects_scale_overlay_fields(self) -> None:
        settings = AppSettings()
        dialog = SettingsDialog(settings, document=None)
        try:
            dialog._scale_overlay_style_combo.setCurrentIndex(dialog._scale_overlay_style_combo.findData("ticks"))
            dialog._scale_overlay_length_spin.setValue(27.5)
            dialog._scale_overlay_font_size.setValue(24)
            dialog._overlay_line_width.setValue(4.5)
            dialog._focus_stack_profile_combo.setCurrentIndex(dialog._focus_stack_profile_combo.findData(FocusStackProfile.SHARP))
            dialog._focus_stack_sharpen_slider.setValue(65)
            updated = dialog.app_settings()

            self.assertEqual(updated.scale_overlay_style, "ticks")
            self.assertAlmostEqual(updated.scale_overlay_length_value, 27.5)
            self.assertEqual(updated.scale_overlay_font_size, 24)
            self.assertAlmostEqual(updated.overlay_line_width, 4.5)
            self.assertEqual(updated.focus_stack_profile, FocusStackProfile.SHARP)
            self.assertEqual(updated.focus_stack_sharpen_strength, 65)
        finally:
            dialog.close()

    def test_settings_dialog_uses_new_measurement_and_scale_defaults(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            self.assertEqual(dialog._measurement_label_color.property("color_value"), "#00FF00")
            self.assertEqual(dialog._measurement_label_decimals.value(), 2)
            self.assertFalse(dialog._measurement_label_background.isChecked())
            self.assertEqual(dialog._endpoint_style_combo.currentData(), MeasurementEndpointStyle.BAR)
            self.assertEqual(dialog._open_view_mode_combo.currentData(), OpenImageViewMode.FIT)
            self.assertEqual(dialog._scale_overlay_mode_combo.currentData(), ScaleOverlayPlacementMode.BOTTOM_RIGHT)
            self.assertEqual(dialog._scale_overlay_style_combo.currentData(), ScaleOverlayStyle.TICKS)
            self.assertAlmostEqual(dialog._scale_overlay_length_spin.value(), 50.0)
            self.assertEqual(dialog._scale_overlay_color.property("color_value"), "#FF0000")
            self.assertEqual(dialog._scale_overlay_text_color.property("color_value"), "#FF0000")
        finally:
            dialog.close()

    def test_settings_dialog_scale_overlay_length_uses_current_calibration_unit_suffix(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/scale_unit.png",
            image_size=(200, 120),
        )
        document.initialize_runtime_state()
        document.calibration = Calibration(
            mode="preset",
            pixels_per_unit=5.0,
            unit="um",
            source_label="demo",
        )
        dialog = SettingsDialog(AppSettings(), document=document)
        try:
            self.assertTrue(dialog._scale_overlay_length_spin.suffix().endswith(" um"))
        finally:
            dialog.close()

    def test_settings_dialog_combobox_ignores_wheel_without_popup(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            combo = dialog._scale_overlay_style_combo
            current_index = combo.currentIndex()
            event = FakeIgnoredWheelEvent()

            combo.wheelEvent(event)

            self.assertEqual(combo.currentIndex(), current_index)
            self.assertTrue(event.ignored)
        finally:
            dialog.close()

    def test_settings_dialog_font_combobox_ignores_wheel_without_popup(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            combo = dialog._text_font
            current_family = combo.currentFont().family()
            event = FakeIgnoredWheelEvent()

            combo.wheelEvent(event)

            self.assertEqual(combo.currentFont().family(), current_family)
            self.assertTrue(event.ignored)
        finally:
            dialog.close()

    def test_settings_dialog_spinbox_ignores_wheel(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            spinbox = dialog._scale_overlay_length_spin
            current_value = spinbox.value()
            event = FakeIgnoredWheelEvent()

            spinbox.wheelEvent(event)

            self.assertEqual(spinbox.value(), current_value)
            self.assertTrue(event.ignored)
        finally:
            dialog.close()

    def test_settings_dialog_focus_stack_slider_ignores_wheel(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            slider = dialog._focus_stack_sharpen_slider
            current_value = slider.value()
            event = FakeIgnoredWheelEvent()

            slider.wheelEvent(event)

            self.assertEqual(slider.value(), current_value)
            self.assertTrue(event.ignored)
        finally:
            dialog.close()

    def test_combined_overlay_export_renders_measurements_and_scale_together(self) -> None:
        window, document = self._create_main_window_fixture()
        try:
            with TemporaryDirectory() as tmp_dir:
                baseline_path = Path(tmp_dir) / "baseline.png"
                combined_path = Path(tmp_dir) / "combined.png"
                window._render_overlay_image(
                    document,
                    baseline_path,
                    include_measurements=False,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                )
                window._render_overlay_image(
                    document,
                    combined_path,
                    include_measurements=True,
                    include_scale=True,
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                )

                baseline_image = QImage(str(baseline_path))
                combined_image = QImage(str(combined_path))
                self.assertGreater(self._count_diff_pixels(baseline_image, combined_image), 0)
        finally:
            window.close()

    def test_settings_dialog_does_not_auto_request_scale_anchor_for_unrelated_changes(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/manual_scale_settings.png",
            image_size=(240, 160),
        )
        document.initialize_runtime_state()
        settings = AppSettings(
            scale_overlay_placement_mode="manual",
        )

        dialog = SettingsDialog(settings, document=document)
        try:
            dialog._measurement_label_size.setValue(dialog._measurement_label_size.value() + 1)
            self.assertFalse(dialog.wants_scale_anchor_pick())
        finally:
            dialog.close()

    def test_settings_dialog_requests_scale_anchor_only_after_explicit_pick_button(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/manual_scale_explicit.png",
            image_size=(240, 160),
        )
        document.initialize_runtime_state()
        settings = AppSettings(
            scale_overlay_placement_mode="manual",
        )

        dialog = SettingsDialog(settings, document=document)
        try:
            dialog._trigger_scale_anchor_pick()
            self.assertTrue(dialog.wants_scale_anchor_pick())
        finally:
            dialog.close()

    def test_settings_dialog_uses_separate_area_models_tab(self) -> None:
        settings = AppSettings()
        dialog = SettingsDialog(settings, document=None)
        try:
            self.assertEqual(dialog._tabs.count(), 6)
            self.assertEqual(dialog._tabs.tabText(0), "测量标注")
            self.assertEqual(dialog._tabs.tabText(1), "比例尺叠加")
            self.assertEqual(dialog._tabs.tabText(2), "图像处理")
            self.assertEqual(dialog._tabs.tabText(3), "叠加标注")
            self.assertEqual(dialog._tabs.tabText(4), "面积识别")
            self.assertEqual(dialog._tabs.tabText(5), "当前图片")
            self.assertLessEqual(dialog.width(), 720)
            self.assertIsInstance(dialog._tabs.widget(0), QScrollArea)
            self.assertIsInstance(dialog._tabs.widget(1), QScrollArea)
            self.assertEqual(dialog._area_weights_dir_edit.text(), "runtime/area-models")
            self.assertEqual(dialog._area_vendor_root_edit.text(), "runtime/area-infer/vendor/yolact")
            self.assertEqual(dialog._area_worker_python_edit.text(), "")
            self.assertEqual(self._group_titles_in_tab(dialog, 0), ["结果文字", "测量线与端点"])
            self.assertEqual(self._group_titles_in_tab(dialog, 1), ["默认视图", "位置与长度", "样式"])
            self.assertEqual(self._group_titles_in_tab(dialog, 2), ["景深合成默认参数", "魔棒分割"])
            self.assertEqual(
                dialog._magic_segment_model_variant_combo.currentData(),
                MagicSegmentModelVariant.EDGE_SAM_3X,
            )
            self.assertFalse(dialog._magic_segment_fill_draft_holes_checkbox.isChecked())
        finally:
            dialog.close()

    def test_settings_dialog_roundtrips_magic_segment_controls(self) -> None:
        settings = AppSettings(
            magic_segment_model_variant=MagicSegmentModelVariant.EDGE_SAM,
        )
        dialog = SettingsDialog(settings, document=None)
        try:
            dialog._magic_segment_model_variant_combo.setCurrentIndex(
                dialog._magic_segment_model_variant_combo.findData(MagicSegmentModelVariant.EDGE_SAM_3X)
            )
            dialog._magic_segment_fill_draft_holes_checkbox.setChecked(True)

            collected = dialog.app_settings()

            self.assertEqual(collected.magic_segment_model_variant, MagicSegmentModelVariant.EDGE_SAM_3X)
            self.assertTrue(collected.magic_segment_fill_draft_holes_enabled)
        finally:
            dialog.close()

    def test_magic_segment_draft_hole_fill_setting_only_applies_when_enabled(self) -> None:
        document, image, canvas = self._create_canvas_document()
        canvas.set_tool_mode("magic_segment")
        hole_mask = self._rect_mask(image.width(), image.height(), (24, 24, 112, 96))
        hole_mask[46:72, 54:82] = False

        canvas._magic_segment.request_id = 1
        canvas._magic_segment.pending_stage = MagicSegmentOperationMode.ADD
        canvas.apply_magic_segment_result(1, hole_mask)
        self.assertEqual(len(canvas._magic_segment.primary_rings), 2)

        canvas.clear_magic_segment_session()
        canvas.set_settings(AppSettings(magic_segment_fill_draft_holes_enabled=True))
        canvas._magic_segment.request_id = 2
        canvas._magic_segment.pending_stage = MagicSegmentOperationMode.ADD
        canvas.apply_magic_segment_result(2, hole_mask)
        self.assertEqual(len(canvas._magic_segment.primary_rings), 1)

        canvas.clear_magic_segment_session()
        canvas.set_tool_mode(MagicSegmentToolMode.REFERENCE)
        canvas._magic_segment.request_id = 3
        canvas._magic_segment.pending_stage = MagicSegmentOperationMode.ADD
        canvas.apply_magic_segment_result(3, hole_mask)
        self.assertEqual(len(canvas._magic_segment.primary_rings), 2)

    def test_reference_instance_commit_adds_measurements_with_reference_mode_labels(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/reference_instance_commit.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            self._load_document_into_window(window, document, image)
            canvas = window.current_canvas()
            self.assertIsNotNone(canvas)

            reference_polygon = [Point(20, 18), Point(88, 18), Point(88, 72), Point(20, 72)]
            candidate_polygon = [Point(112, 18), Point(180, 18), Point(180, 72), Point(112, 72)]
            canvas._reference_instance.request_id = 1
            canvas._reference_instance.reference_polygon = list(reference_polygon)
            canvas._reference_instance.preview_candidates = [
                ReferenceInstancePreviewCandidate(
                    polygon_px=list(candidate_polygon),
                    area_rings_px=[],
                    confidence=0.91,
                )
            ]
            window.set_tool_mode(MagicSegmentToolMode.REFERENCE)

            committed = window._commit_reference_instance_preview()

            self.assertTrue(committed)
            self.assertEqual(len(document.measurements), 1)
            measurement = document.measurements[0]
            self.assertEqual(measurement.mode, "reference_instance")
            self.assertEqual(measurement.status, "reference_instance")
            self.assertIsNotNone(measurement.fiber_group_id)

            window._populate_measurement_table(document)
            self.assertEqual(window.measurement_table.item(0, window.TABLE_COL_MODE).text(), "同类扩选")
            self.assertEqual(window.measurement_table.item(0, window.TABLE_COL_STATUS).text(), "同类扩选")
        finally:
            window._reset_workspace()
            window.close()

    def test_close_event_stops_idle_prompt_and_reference_threads(self) -> None:
        window = MainWindow()
        try:
            prompt_thread = QThread(window)
            prompt_thread.start()
            reference_thread = QThread(window)
            reference_thread.start()
            window._prompt_seg_thread = prompt_thread
            window._reference_instance_thread = reference_thread
            event = FakeCloseEvent()

            with patch.object(window, "_confirm_close_documents", return_value=True):
                window.closeEvent(event)

            self.assertTrue(event.accepted)
            self.assertFalse(event.ignored)
            self.assertFalse(prompt_thread.isRunning())
            self.assertFalse(reference_thread.isRunning())
        finally:
            window.close()

    def test_settings_dialog_current_image_tab_uses_reorganized_group_titles(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/current_image_settings.png",
            image_size=(240, 160),
        )
        document.initialize_runtime_state()

        dialog = SettingsDialog(AppSettings(), document=document)
        try:
            self.assertEqual(self._group_titles_in_tab(dialog, 5), ["类别颜色", "比例尺锚点"])
        finally:
            dialog.close()

    def test_measurement_display_text_preserves_trailing_zeroes(self) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/measurement_label_text.png",
            image_size=(100, 80),
        )
        document.initialize_runtime_state()
        document.calibration = Calibration(
            mode="preset",
            pixels_per_unit=2.0,
            unit="um",
            source_label="demo",
        )

        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(0, 0), Point(20.0, 0)),
        )
        measurement.recalculate(document.calibration)
        settings = AppSettings(measurement_label_decimals=4)
        self.assertEqual(measurement_display_text_with_settings(measurement, document, settings), "10.0000 um")

        measurement.line_px = Line(Point(0, 0), Point(22.246, 0))
        measurement.recalculate(document.calibration)
        self.assertEqual(measurement_display_text_with_settings(measurement, document, settings), "11.1230 um")

    def test_main_window_progress_dialog_uses_standard_qprogressdialog_with_width(self) -> None:
        window = MainWindow()
        try:
            dialog = window._create_progress_dialog(
                title="面积自动识别",
                label_text="正在识别 (1/1)\nexample.jpg",
                maximum=1,
            )
            self.assertEqual(dialog.windowTitle(), "面积自动识别")
            self.assertEqual(dialog.labelText(), "正在识别 (1/1)\nexample.jpg")
            self.assertGreaterEqual(dialog.minimumWidth(), 420)
            self.assertEqual(dialog.maximum(), 1)
            self.assertEqual(dialog.value(), 0)
            dialog.close()
        finally:
            window.close()

    def test_main_window_persist_window_geometry_updates_app_settings(self) -> None:
        window = MainWindow()
        try:
            window.setGeometry(120, 140, 980, 720)
            window._persist_window_geometry()

            self.assertTrue(window._app_settings.main_window_geometry)
            self.assertFalse(window._app_settings.main_window_is_maximized)
        finally:
            window.close()

    def test_main_window_restores_saved_geometry_on_startup(self) -> None:
        probe = MainWindow()
        try:
            probe.setGeometry(150, 170, 920, 680)
            geometry_token = bytes(probe.saveGeometry().toBase64()).decode("ascii")
        finally:
            probe.close()

        with patch("fdm.ui.main_window.AppSettingsIO.load", return_value=AppSettings(main_window_geometry=geometry_token)):
            window = MainWindow()
        try:
            self.assertGreaterEqual(window.width(), 880)
            self.assertGreaterEqual(window.height(), 640)
        finally:
            window.close()

    def test_number_hotkey_switches_active_group_without_changing_measurement_group(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/group_hotkey.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            first_group = document.create_group(color="#1F7A8C", label="棉")
            second_group = document.create_group(color="#E07A5F", label="麻")
            document.set_active_group(first_group.id)
            measurement = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=first_group.id,
                mode="manual",
                line_px=Line(Point(20, 20), Point(120, 20)),
            )
            document.add_measurement(measurement)

            self._load_document_into_window(window, document, image)
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_2))

            self.assertEqual(document.active_group_id, second_group.id)
            self.assertEqual(measurement.fiber_group_id, first_group.id)
        finally:
            window.close()

    def test_number_hotkey_scrolls_selected_group_into_view(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/group_hotkey_scroll.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            for index in range(1, 9):
                document.create_group(color=f"#{index:02X}7A8C", label=f"类别{index}")
            document.set_active_group(document.get_group_by_number(1).id)

            self._load_document_into_window(window, document, image)
            window.resize(960, 700)
            window.show()
            self.app.processEvents()

            window.group_list.setFixedHeight(120)
            window._left_panel_splitter.setSizes([420, 220])
            self.app.processEvents()
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_8))
            self.app.processEvents()

            target_group = document.get_group_by_number(8)
            self.assertIsNotNone(target_group)
            target_item = None
            for index in range(window.group_list.count()):
                item = window.group_list.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == target_group.id:
                    target_item = item
                    break
            self.assertIsNotNone(target_item)
            rect = window.group_list.visualItemRect(target_item)
            self.assertGreater(window.group_list.verticalScrollBar().value(), 0)
            self.assertTrue(window.group_list.viewport().rect().contains(rect.center()))
        finally:
            window.close()

    def test_selected_area_handle_prefers_nearest_vertex_over_first_matching_vertex(self) -> None:
        document, _, canvas = self._create_canvas_document()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            measurement_kind="area",
            fiber_group_id=None,
            mode="polygon_area",
            polygon_px=[Point(20, 20), Point(28, 20), Point(28, 60), Point(20, 60)],
        )
        measurement.recalculate(None)
        document.add_measurement(measurement)
        document.select_measurement(measurement.id)

        hit = canvas._hit_test_selected_area_handle(Point(26.5, 20))

        self.assertEqual(hit, (measurement.id, "vertex", 1))

    def test_selected_area_handle_uses_center_only_when_no_vertex_matches(self) -> None:
        document, _, canvas = self._create_canvas_document()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            measurement_kind="area",
            fiber_group_id=None,
            mode="polygon_area",
            polygon_px=[Point(20, 20), Point(60, 20), Point(60, 60), Point(20, 60)],
        )
        measurement.recalculate(None)
        document.add_measurement(measurement)
        document.select_measurement(measurement.id)

        vertex_hit = canvas._hit_test_selected_area_handle(Point(23.4, 20.4))
        center_hit = canvas._hit_test_selected_area_handle(Point(40, 40))

        self.assertEqual(vertex_hit, (measurement.id, "vertex", 0))
        self.assertEqual(center_hit, (measurement.id, "center", None))

    def test_area_measurement_draws_fill_path_without_second_outline_geometry(self) -> None:
        document, _, _canvas = self._create_canvas_document()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            measurement_kind="area",
            fiber_group_id=None,
            mode="magic_segment",
            polygon_px=[Point(24, 20), Point(84, 24), Point(78, 92), Point(18, 86)],
            area_rings_px=[
                [Point(20, 20), Point(88, 20), Point(88, 88), Point(20, 88)],
                [Point(42, 42), Point(60, 42), Point(60, 60), Point(42, 60)],
            ],
        )
        measurement.recalculate(None)
        painter = RecordingPainter()

        draw_area_measurement(
            painter,
            document,
            measurement,
            lambda point: QPointF(point.x, point.y),
            AppSettings(),
            line_width=2.0,
            endpoint_radius=4.0,
            selected=False,
            show_fill=True,
            show_handles=False,
        )

        self.assertEqual([call[0] for call in painter.calls], ["path", "polygon", "polygon"])
        self.assertEqual(painter.calls[0][1], Qt.PenStyle.NoPen)
        self.assertEqual(painter.calls[1][1].color().name().lower(), "#0b0b0b")
        self.assertEqual(painter.calls[2][1].color().name().lower(), QColor(AppSettings().default_measurement_color).name().lower())

    def test_dragging_area_vertex_emits_polygon_payload_and_clears_magic_rings(self) -> None:
        document, _image, canvas = self._create_canvas_document()
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            measurement_kind="area",
            fiber_group_id=None,
            mode="magic_segment",
            polygon_px=[Point(24, 20), Point(84, 24), Point(78, 92), Point(18, 86)],
            area_rings_px=[
                [Point(20, 20), Point(88, 20), Point(88, 88), Point(20, 88)],
                [Point(42, 42), Point(60, 42), Point(60, 60), Point(42, 60)],
            ],
        )
        measurement.recalculate(None)
        document.add_measurement(measurement)
        document.select_measurement(measurement.id)
        payloads: list[object] = []
        canvas.measurementEdited.connect(lambda _document_id, _measurement_id, payload: payloads.append(payload))

        preview = [Point(26, 18), Point(84, 24), Point(78, 92), Point(18, 86)]
        canvas._dragging_area_handle = (measurement.id, "vertex", 0)
        canvas._drag_area_preview_points = list(preview)
        canvas.mouseReleaseEvent(FakeMouseEvent(canvas.image_to_widget(preview[0]), button=Qt.MouseButton.LeftButton))

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["measurement_kind"], "area")
        self.assertEqual(payloads[0]["mode"], "polygon_area")
        self.assertEqual(payloads[0]["area_rings_px"], [])
        self.assertEqual(payloads[0]["polygon_px"], preview)

    def test_measurement_group_combo_ignores_wheel_without_popup(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/group_combo.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            group = document.create_group(color="#1F7A8C", label="棉")
            document.set_active_group(group.id)
            document.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id,
                    mode="manual",
                    line_px=Line(Point(20, 20), Point(100, 20)),
                )
            )

            self._load_document_into_window(window, document, image)
            combo = window.measurement_table.cellWidget(0, window.TABLE_COL_GROUP)
            event = FakeIgnoredWheelEvent()
            current_index = combo.currentIndex()

            combo.wheelEvent(event)

            self.assertEqual(combo.focusPolicy(), Qt.FocusPolicy.NoFocus)
            self.assertEqual(combo.currentIndex(), current_index)
            self.assertTrue(event.ignored)
        finally:
            window.close()

    def test_a_shortcut_toggles_between_select_and_previous_tool(self) -> None:
        window = MainWindow()
        try:
            self.assertEqual(window._tool_mode, "select")
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_A))
            self.assertEqual(window._tool_mode, "select")

            window.set_tool_mode("manual")
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_A))
            self.assertEqual(window._tool_mode, "select")
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_A))
            self.assertEqual(window._tool_mode, "manual")
        finally:
            window.close()

    def test_snap_tool_is_available_and_can_be_selected(self) -> None:
        window = MainWindow()
        try:
            self.assertIn("snap", window._mode_actions)

            window.set_tool_mode("snap")

            self.assertEqual(window._tool_mode, "snap")
            self.assertTrue(window._mode_actions["snap"].isChecked())
        finally:
            window.close()

    def test_snap_line_commit_creates_snap_measurement(self) -> None:
        window = MainWindow()
        try:
            image = QImage(220, 140, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/snap_measurement.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            group = document.create_group(color="#1F7A8C", label="棉")
            document.set_active_group(group.id)
            self._load_document_into_window(window, document, image)
            original_line = Line(Point(40, 70), Point(180, 70))
            snapped_line = Line(Point(84.0, 70.0), Point(136.0, 70.0))

            with patch.object(
                window.snap_service,
                "snap_measurement",
                return_value=SnapResult(
                    status="snapped",
                    original_line=original_line,
                    snapped_line=snapped_line,
                    diameter_px=52.0,
                    confidence=0.88,
                    debug_payload={"polarity": "dark_on_light"},
                ),
            ):
                window._on_canvas_line_committed(document.id, "snap", original_line)

            self.assertEqual(len(document.measurements), 1)
            measurement = document.measurements[0]
            self.assertEqual(measurement.mode, "snap")
            self.assertEqual(measurement.status, "snapped")
            self.assertEqual(measurement.line_px, original_line)
            self.assertEqual(measurement.snapped_line_px, snapped_line)
            self.assertEqual(window._format_measurement_mode(measurement.mode), "边缘吸附")
        finally:
            window.close()

    def test_prepare_image_load_requests_skips_duplicates(self) -> None:
        window = MainWindow()
        try:
            existing_document = ImageDocument(
                id=new_id("image"),
                path="/tmp/already_open.png",
                image_size=(100, 60),
            )
            existing_document.initialize_runtime_state()
            window.project.documents.append(existing_document)

            requests, skipped_count, focus_document_id = window._prepare_image_load_requests(
                [
                    (existing_document.path, None),
                    ("/tmp/new_image.png", None),
                    ("/tmp/new_image.png", None),
                ]
            )

            self.assertEqual(len(requests), 1)
            self.assertEqual(skipped_count, 2)
            self.assertEqual(focus_document_id, existing_document.id)
        finally:
            window.close()

    def test_batch_loader_worker_reports_progress_and_failures(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "worker_image.png"
            second_image_path = Path(tmp_dir) / "worker_image_2.png"
            missing_path = Path(tmp_dir) / "missing.png"

            image = QImage(40, 20, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            image.save(str(image_path))
            image.save(str(second_image_path))

            worker = ImageBatchLoaderWorker(
                [
                    ImageLoadRequest(path=str(image_path)),
                    ImageLoadRequest(path=str(missing_path)),
                    ImageLoadRequest(path=str(second_image_path)),
                ],
                skipped_count=1,
            )
            progress_calls: list[tuple[int, int, str]] = []
            loaded_paths: list[str] = []
            failed_paths: list[str] = []
            finished_payloads: list[tuple[bool, int, int, int]] = []

            worker.progress.connect(lambda index, total, path: progress_calls.append((index, total, path)))
            worker.loaded.connect(lambda request, *_: loaded_paths.append(request.path))
            worker.failed.connect(lambda path, _reason: failed_paths.append(path))
            worker.finished.connect(lambda cancelled, loaded_count, skipped_count, failed_count: finished_payloads.append((cancelled, loaded_count, skipped_count, failed_count)))

            worker.run()

            self.assertEqual(len(progress_calls), 3)
            self.assertEqual(len(loaded_paths), 2)
            self.assertEqual(failed_paths, [str(missing_path)])
            self.assertEqual(finished_payloads, [(False, 2, 1, 1)])

    def test_qimage_to_raster_drops_scanline_padding_and_preserves_python_int_pixels(self) -> None:
        image = QImage(3, 2, QImage.Format.Format_RGB32)
        values = [
            [0, 32, 64],
            [96, 128, 160],
        ]
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                image.setPixelColor(x, y, QColor(value, value, value))

        raster = qimage_to_raster(image)

        self.assertEqual((raster.width, raster.height), (3, 2))
        self.assertEqual(raster.pixels, [0, 32, 64, 96, 128, 160])
        self.assertTrue(all(type(value) is int for value in raster.pixels))

    def test_full_resolution_metrics_scale_above_old_cap_for_large_images(self) -> None:
        window = MainWindow()
        try:
            metrics = window._overlay_metrics(12000, 8000, ExportImageRenderMode.FULL_RESOLUTION)
            self.assertLessEqual(metrics["line_width"], 2.5)
            self.assertLessEqual(metrics["font_px"], 20.0)
            self.assertGreater(metrics["line_width"], 1.0)
            self.assertGreater(metrics["endpoint_radius"], metrics["line_width"])
        finally:
            window.close()

    def test_full_resolution_export_remains_visible_after_preview_downscale(self) -> None:
        window = MainWindow()
        try:
            image = QImage(6000, 4000, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/fullres_preview.png",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            group = document.create_group(color="#E07A5F", label="棉")
            document.set_active_group(group.id)
            document.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id,
                    mode="manual",
                    line_px=Line(Point(500, 800), Point(5200, 3200)),
                )
            )
            window._images[document.id] = image

            with TemporaryDirectory() as tmp_dir:
                baseline_path = Path(tmp_dir) / "fullres_baseline.png"
                measurement_path = Path(tmp_dir) / "fullres_measurement.png"
                window._render_overlay_image(
                    document,
                    baseline_path,
                    include_measurements=False,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                )
                window._render_overlay_image(
                    document,
                    measurement_path,
                    include_measurements=True,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                )

                baseline_preview = QImage(str(baseline_path)).scaledToWidth(1400)
                measurement_preview = QImage(str(measurement_path)).scaledToWidth(1400)
                self.assertGreater(self._count_diff_pixels(baseline_preview, measurement_preview), 500)
        finally:
            window.close()

    def test_full_resolution_export_works_for_non_paintable_source_format(self) -> None:
        if not hasattr(QImage.Format, "Format_CMYK8888"):
            self.skipTest("current Qt build does not expose CMYK8888")

        window = MainWindow()
        try:
            image = QImage(800, 600, getattr(QImage.Format, "Format_CMYK8888"))
            image.fill(QColor("#FFFFFF"))
            document = ImageDocument(
                id=new_id("image"),
                path="/tmp/fullres_cmyk.jpg",
                image_size=(image.width(), image.height()),
            )
            document.initialize_runtime_state()
            group = document.create_group(color="#E07A5F", label="棉")
            document.set_active_group(group.id)
            document.add_measurement(
                Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id,
                    mode="manual",
                    line_px=Line(Point(100, 100), Point(700, 500)),
                )
            )
            window._images[document.id] = image

            with TemporaryDirectory() as tmp_dir:
                baseline_path = Path(tmp_dir) / "cmyk_baseline.png"
                measurement_path = Path(tmp_dir) / "cmyk_measurement.png"
                window._render_overlay_image(
                    document,
                    baseline_path,
                    include_measurements=False,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                )
                window._render_overlay_image(
                    document,
                    measurement_path,
                    include_measurements=True,
                    include_scale=False,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                )

                baseline_image = QImage(str(baseline_path))
                measurement_image = QImage(str(measurement_path))
                self.assertFalse(measurement_image.isNull())
                self.assertGreater(self._count_diff_pixels(baseline_image, measurement_image), 0)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
