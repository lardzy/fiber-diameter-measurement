from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QImage, QColor
    from PySide6.QtWidgets import QApplication, QListView

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, TextAnnotation, new_id
from fdm.settings import AppSettings, OpenImageViewMode
from fdm.services.export_service import ExportImageRenderMode

if PYSIDE_AVAILABLE:
    from fdm.ui.canvas import DocumentCanvas
    from fdm.ui.dialogs import SettingsDialog
    from fdm.ui.image_loader import ImageBatchLoaderWorker, ImageLoadRequest, qimage_to_raster
    from fdm.ui.main_window import MainWindow
else:
    DocumentCanvas = object  # type: ignore[assignment]
    SettingsDialog = object  # type: ignore[assignment]
    ImageBatchLoaderWorker = object  # type: ignore[assignment]
    ImageLoadRequest = object  # type: ignore[assignment]
    qimage_to_raster = object  # type: ignore[assignment]
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
            qimage_to_raster(image),
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

            self.assertEqual(headers, ["种类", "结果", "单位", "模式", "置信度", "状态", "ID"])
            self.assertIsNotNone(window.measurement_table.item(0, window.TABLE_COL_ID))
        finally:
            window.close()

    def test_measure_toolbar_is_separate_and_exposes_primary_modes(self) -> None:
        window = MainWindow()
        try:
            self.assertIsNotNone(window._file_toolbar)
            self.assertIsNotNone(window._measure_toolbar)
            action_texts = [action.text() for action in window._measure_toolbar.actions()]
            self.assertEqual(action_texts, ["浏览", "手动测量", "半自动吸附", "比例尺标定", "文字"])
            self.assertTrue(all(not action.icon().isNull() for action in window._measure_toolbar.actions()))
            self.assertFalse(window.open_images_action.icon().isNull())
            self.assertFalse(window.save_project_action.icon().isNull())
        finally:
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
