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
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QImage, QColor
    from PySide6.QtWidgets import QApplication, QGroupBox, QListView, QMessageBox, QSplitter

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import Calibration, CalibrationPreset, ImageDocument, Measurement, ProjectGroupTemplate, TextAnnotation, new_id
from fdm.settings import AppSettings, OpenImageViewMode
from fdm.services.export_service import ExportImageRenderMode
from fdm.services.sidecar_io import CalibrationSidecarIO
from fdm.services.snap_service import SnapResult

if PYSIDE_AVAILABLE:
    from fdm.ui.canvas import DocumentCanvas
    from fdm.ui.dialogs import FiberGroupDialog, SettingsDialog, ShortcutHelpDialog
    from fdm.ui.image_loader import ImageBatchLoaderWorker, ImageLoadRequest
    from fdm.ui.main_window import MainWindow
else:
    DocumentCanvas = object  # type: ignore[assignment]
    FiberGroupDialog = object  # type: ignore[assignment]
    SettingsDialog = object  # type: ignore[assignment]
    ShortcutHelpDialog = object  # type: ignore[assignment]
    ImageBatchLoaderWorker = object  # type: ignore[assignment]
    ImageLoadRequest = object  # type: ignore[assignment]
    MainWindow = object  # type: ignore[assignment]


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

    def test_group_list_keeps_color_icon_and_hides_uncategorized_after_first_group(self) -> None:
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
            self.assertFalse(window.group_list.item(0).icon().isNull())

            group = document.create_group(color="#1F7A8C", label="棉")
            document.set_active_group(group.id)
            window._populate_group_list(document)

            self.assertEqual(window.group_list.count(), 1)
            item = window.group_list.item(0)
            self.assertIn("棉", item.text())
            self.assertFalse(item.icon().isNull())
            self.assertIn("background: #FFFDF8", window.group_list.styleSheet())
            self.assertIn("color: #182430", window.group_list.styleSheet())
        finally:
            window.close()

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
        finally:
            window.close()

    def test_measure_toolbar_is_separate_and_exposes_primary_modes(self) -> None:
        window = MainWindow()
        try:
            self.assertIsNotNone(window._file_toolbar)
            self.assertIsNotNone(window._measure_toolbar)
            action_texts = [action.text() for action in window._measure_toolbar.actions() if action.text()]
            self.assertEqual(
                action_texts,
                ["浏览", "手动测量", "多边形面积", "自由形状面积", "魔棒分割", "比例尺标定", "文字"],
            )
            visible_actions = [action for action in window._measure_toolbar.actions() if action.text()]
            self.assertTrue(all(not action.icon().isNull() for action in visible_actions))
            self.assertFalse(window.open_images_action.icon().isNull())
            self.assertFalse(window.save_project_action.icon().isNull())
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

            with patch.object(window, "_ensure_prompt_segmentation_worker", return_value=None), patch(
                "fdm.ui.main_window.PromptSegmentationService.models_ready",
                return_value=True,
            ):
                window._on_canvas_magic_segment_requested(
                    document.id,
                    {
                        "request_id": 7,
                        "positive_points": [Point(20, 20)],
                        "negative_points": [Point(30, 30)],
                    },
                )

            self.assertIsNotNone(fake_worker.requested.payload)
            self.assertIs(fake_worker.requested.payload.image, image)
            self.assertEqual(fake_worker.requested.payload.document_id, document.id)
            self.assertEqual(fake_worker.requested.payload.request_id, 7)
            self.assertTrue(fake_worker.requested.payload.cache_key.startswith(f"{document.id}:"))
        finally:
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
            self.assertIsNotNone(window._magic_controls_action)
            self.assertFalse(window._magic_controls_action.isVisible())

            window.set_tool_mode("magic_segment")
            self.assertTrue(window._magic_controls_action.isVisible())

            window.set_tool_mode("select")
            self.assertFalse(window._magic_controls_action.isVisible())
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
        finally:
            window.close()

    def test_tab_strip_uses_scroll_buttons_without_shrinking_left_panel(self) -> None:
        window = MainWindow()
        try:
            self.assertTrue(window.tab_widget.usesScrollButtons())
            self.assertGreaterEqual(window.image_list.parentWidget().minimumWidth(), 180)
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
        document, _, canvas = self._create_canvas_document()
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
            [Point(20, 20), Point(90, 18), Point(92, 76), Point(24, 80)],
        )

        self.assertTrue(canvas.has_magic_segment_preview())
        self.assertTrue(canvas.commit_magic_segment_preview())
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0][1], "magic_segment")
        self.assertEqual(commits[0][2]["measurement_kind"], "area")

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

            canvas._magic_segment.request_id = 1
            canvas.apply_magic_segment_result(
                1,
                [Point(24, 24), Point(88, 22), Point(92, 76), Point(26, 80)],
            )
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_Return))

            self.assertEqual(len(document.measurements), 1)
            self.assertEqual(document.measurements[0].mode, "magic_segment")

            canvas.set_tool_mode("magic_segment")
            canvas._magic_segment.positive_points = [Point(40, 40)]
            self.assertTrue(canvas.has_magic_segment_session())
            window.keyPressEvent(FakeKeyEvent(Qt.Key.Key_Escape))
            self.assertFalse(canvas.has_magic_segment_session())
        finally:
            window._reset_workspace()
            window.close()

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
                window._on_canvas_text_placement_requested(document.id, Point(24, 32))

            self.assertEqual(len(document.text_annotations), 1)
            self.assertEqual(document.text_annotations[0].content, "说明文本")
            self.assertEqual(document.selected_text_id, document.text_annotations[0].id)

            window.delete_selected_measurement()
            self.assertEqual(len(document.text_annotations), 0)
        finally:
            window.close()

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
            self.assertEqual(dialog._tabs.count(), 3)
            self.assertEqual(dialog._tabs.tabText(1), "面积识别")
            self.assertLessEqual(dialog.width(), 720)
            self.assertEqual(dialog._area_weights_dir_edit.text(), "runtime/area-models")
            self.assertEqual(dialog._area_vendor_root_edit.text(), "runtime/area-infer/vendor/yolact")
            self.assertEqual(dialog._area_worker_python_edit.text(), "")
        finally:
            dialog.close()

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
