"""Capture current FDM widgets and produce a real, reproducible teaching project.

Run from the FDM repository: uv run --no-sync python tutorials/fdm-basics/tools/capture_fdm.py
The input is a deterministic synthetic specimen, not a microscope observation.
"""
from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_SCALE_FACTOR"] = "2"

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QAbstractButton, QFileDialog, QWidget

from fdm import runtime_logging, screenshot_settings, settings
from fdm.geometry import Point
from fdm.models import ImageDocument, new_id
from fdm.services.export_service import ExportSelection
from fdm.ui.dialogs import CalibrationInputDialog
from fdm.ui.image_loader import ImageLoadRequest
from fdm.ui.main_window import MainWindow
from fdm.ui.theme import apply_application_theme
from fdm.version import __version__

ROOT = Path(__file__).resolve().parents[1]
SCREENS = ROOT / "public" / "screens"
DEMO = ROOT / "demo"
SCREENS.mkdir(parents=True, exist_ok=True)
DEMO.mkdir(parents=True, exist_ok=True)
manifest: dict = {"software_version": __version__, "capture": "Qt offscreen, current source", "synthetic": True, "screens": {}}


def settle(ms=200):
    deadline = time.monotonic() + ms / 1000
    while time.monotonic() < deadline:
        QApplication.processEvents()
        QTest.qWait(10)


def make_specimen():
    image = QImage(1280, 820, QImage.Format.Format_RGB32)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    bg = QLinearGradient(0, 0, 1280, 820)
    bg.setColorAt(0, QColor("#deded9"))
    bg.setColorAt(1, QColor("#b8c6c2"))
    p.fillRect(image.rect(), bg)
    for center, width in [(250, 76), (600, 100), (950, 124)]:
        shade = QLinearGradient(center - width / 2, 0, center + width / 2, 0)
        shade.setColorAt(0, QColor("#4d5a59"))
        shade.setColorAt(0.1, QColor("#c4ceca"))
        shade.setColorAt(0.45, QColor("#dce2dc"))
        shade.setColorAt(0.8, QColor("#bacac4"))
        shade.setColorAt(1, QColor("#4d5a59"))
        p.fillRect(int(center - width / 2), 0, width, 684, shade)
        p.setPen(QPen(QColor("#657a73"), 2))
        for side in (-1, 1):
            p.drawLine(int(center + side * width / 2), 0, int(center + side * width / 2), 684)
        p.setPen(QPen(QColor(96, 126, 113, 52), 1))
        for y in range(15, 680, 22):
            p.drawLine(int(center - width / 2 + 5), y, int(center + width / 2 - 5), y + 10)
    p.fillRect(0, 684, 1280, 136, QColor("#e8eeea"))
    p.setFont(QFont("PingFang SC", 20))
    p.setPen(QColor("#485e58"))
    p.drawText(35, 745, "合成教学图像")
    p.setFont(QFont("PingFang SC", 13))
    p.drawText(35, 780, "仅演示操作步骤 · 非实测样品")
    p.setPen(QPen(QColor("#182a25"), 5))
    p.drawLine(800, 744, 1200, 744)
    p.drawLine(800, 734, 800, 754)
    p.drawLine(1200, 734, 1200, 754)
    p.setFont(QFont("Arial", 24))
    p.drawText(936, 794, "100 μm")
    p.end()
    sample = DEMO / "合成纤维教学图.png"
    assert image.save(str(sample))
    return image, sample


def rect(widget, parent):
    pos = widget.mapTo(parent, QPoint(0, 0))
    return [pos.x(), pos.y(), widget.width(), widget.height()]


def capture(name, window, dialog=None):
    settle(120)
    pix = window.grab()
    entry = {"size": [pix.width(), pix.height()], "logical_size": [window.width(), window.height()],
             "device_pixel_ratio": pix.devicePixelRatio(), "buttons": {}}
    for button in window.findChildren(QAbstractButton):
        label = button.text().replace("&", "").strip()
        if label and button.isVisible():
            entry["buttons"].setdefault(label, rect(button, window))
    canvas = window.current_canvas()
    if canvas is not None:
        entry["canvas"] = rect(canvas, window)
        offset = canvas.mapTo(window, QPoint(0, 0))
        origin = canvas.image_to_widget(Point(0, 0))
        unit = canvas.image_to_widget(Point(1, 0))
        entry["image_transform"] = {
            "origin": [origin.x() + offset.x(), origin.y() + offset.y()],
            "scale": unit.x() - origin.x(),
        }
        doc = window.current_document()
        entry["objects"] = []
        if doc is not None:
            for obj in doc.measurements:
                pts = list(obj.polygon_px or obj.polyline_px)
                if obj.line_px is not None:
                    line = obj.effective_line()
                    pts = [line.start, line.end]
                elif obj.point_px is not None:
                    pts = [obj.point_px]
                if pts:
                    xs, ys = [p.x for p in pts], [p.y for p in pts]
                    entry["objects"].append({"kind": obj.measurement_kind,
                        "bounds": [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)],
                        "line": [[p.x, p.y] for p in pts] if obj.line_px is not None else None})
        entry["image_points"] = {}
        for key, pt in {
            "scale_start": Point(800, 744), "scale_end": Point(1200, 744),
            "line1_start": Point(212, 250), "line1_end": Point(288, 250),
            "line2_start": Point(550, 350), "line2_end": Point(650, 350),
            "line3_start": Point(888, 205), "line3_end": Point(1012, 205),
        }.items():
            mapped = canvas.image_to_widget(pt)
            entry["image_points"][key] = [mapped.x() + offset.x(), mapped.y() + offset.y()]
    if dialog is not None:
        popup = dialog.grab()
        x = (window.width() - dialog.width()) // 2
        y = (window.height() - dialog.height()) // 2
        painter = QPainter(pix)
        painter.fillRect(pix.rect(), QColor(6, 20, 23, 70))
        painter.drawPixmap(x, y, popup)
        painter.end()
        entry["dialog"] = [x, y, dialog.width(), dialog.height()]
    host = dialog or window
    shift = entry.get("dialog", [0, 0])[:2] if dialog is not None else [0, 0]
    entry["controls"] = []
    for widget in host.findChildren(QWidget):
        if not widget.isVisible():
            continue
        label = ""
        for method in ("text", "currentText", "placeholderText"):
            getter = getattr(widget, method, None)
            if callable(getter):
                try:
                    label = str(getter())
                except TypeError:
                    pass
                if label:
                    break
        if label or widget.objectName():
            wx, wy, ww, wh = rect(widget, host)
            entry["controls"].append({"type": widget.metaObject().className(),
                "name": widget.objectName(), "text": label.replace("&", ""),
                "rect": [wx+shift[0], wy+shift[1], ww, wh]})
    if dialog is not None:
        entry["dialog_buttons"] = {
            b.text().replace("&", "").strip(): [r[0]+shift[0], r[1]+shift[1], r[2], r[3]]
            for b in dialog.findChildren(QAbstractButton) if b.isVisible()
            for r in [rect(b, dialog)] if b.text().strip()
        }
    assert pix.save(str(SCREENS / f"{name}.png"))
    manifest["screens"][name] = entry
    print("captured", name, flush=True)


def drag(window, start, end, preview_name=None):
    canvas = window.current_canvas()
    a = canvas.image_to_widget(start).toPoint()
    b = canvas.image_to_widget(end).toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    for i in range(1, 9):
        pt = QPoint(round(a.x() + (b.x() - a.x()) * i / 8), round(a.y() + (b.y() - a.y()) * i / 8))
        QTest.mouseMove(canvas, pt, 15)
    if preview_name:
        capture(preview_name, window)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
    settle(200)


def main():
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)
    app = QApplication([])
    app.setFont(QFont("PingFang SC", 10))
    preferences = settings.AppSettings(theme_mode="light", object_snap_enabled=False,
        measurement_label_font_family="PingFang SC", measurement_label_font_size=24,
        measurement_label_color="#ffffff")
    apply_application_theme(app, "light")
    specimen, specimen_path = make_specimen()
    # Reset only this generated specimen's sidecar so repeated captures start uncalibrated.
    specimen_path.with_suffix(specimen_path.suffix + ".fdm.json").unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="fdm-tutorial-profile-") as profile, ExitStack() as stack:
        profile = Path(profile)
        stack.enter_context(patch.object(settings, "settings_file_path", lambda: profile / "settings.json"))
        stack.enter_context(patch.object(screenshot_settings, "screenshot_settings_file_path", lambda: profile / "screenshot.json"))
        stack.enter_context(patch.object(runtime_logging, "runtime_log_path", lambda: profile / "startup.log"))
        stack.enter_context(patch("fdm.ui.main_window.AppSettingsIO.load", return_value=preferences))
        stack.enter_context(patch("fdm.ui.main_window.AppSettingsIO.save", return_value=None))
        win = MainWindow()
        win.resize(1600, 900)
        win.show()
        settle(500)
        capture("00-empty", win)
        chooser = QFileDialog(win, "打开图片", str(DEMO), "图片 (*.png *.jpg *.tif)")
        chooser.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        chooser.setFileMode(QFileDialog.FileMode.ExistingFile)
        chooser.selectFile(str(specimen_path))
        chooser.resize(920, 560)
        chooser.show()
        capture("01-open-dialog", win, chooser)
        chooser.hide()

        doc = ImageDocument(id=new_id("image"), path=str(specimen_path), image_size=(1280, 820))
        doc.initialize_runtime_state()
        group = doc.create_group(color="#2a9d8f", label="示例纤维")
        doc.set_active_group(group.id)
        win._add_loaded_document(ImageLoadRequest(path=doc.path, document=doc), specimen)
        assert doc.calibration is None, "Teaching sequence must start without calibration"
        settle(350)
        win.fit_current_image()
        capture("02-opened", win)
        win.set_tool_mode("calibration")
        capture("03-calibration-tool", win)

        def fill_calibration():
            dialog = next((w for w in app.topLevelWidgets() if isinstance(w, CalibrationInputDialog) and w.isVisible()), None)
            if dialog is None:
                QTimer.singleShot(50, fill_calibration)
                return
            dialog._length_spin.setValue(100)
            dialog._apply_to_project.setChecked(False)
            capture("05-calibration-input", win, dialog)
            dialog.accept()

        QTimer.singleShot(450, fill_calibration)
        drag(win, Point(800, 744), Point(1200, 744), "04-calibration-line")
        assert doc.calibration is not None, "Calibration did not commit"
        assert abs(doc.calibration.pixels_per_unit - 4) < 0.03
        capture("06-calibrated", win)
        win.set_tool_mode("manual")
        capture("07-manual-tool", win)
        drag(win, Point(212, 250), Point(288, 250), "08-measure-preview")
        win._flush_pending_measurements(for_snapshot=True)
        assert len(doc.measurements) == 1
        capture("09-first-measurement", win)
        drag(win, Point(550, 350), Point(650, 350))
        drag(win, Point(888, 205), Point(1012, 205))
        win._flush_pending_measurements(for_snapshot=True)
        assert len(doc.measurements) == 3
        win.set_tool_mode("pan")
        capture("10-measurements", win)
        win._toggle_results_panel()
        win._results_tabs.setCurrentIndex(0)
        settle(350)
        capture("11-results", win)
        win._results_tabs.setCurrentIndex(1)
        capture("12-statistics", win)
        win._toggle_results_panel()
        win.fit_current_image()
        result = win.save_project(str(DEMO / "入门测量示例.fdmproj"))
        assert result.success, str(result)
        win.statusBar().clearMessage()
        capture("13-project-saved", win)

        selection = ExportSelection(include_excel=True, include_csv=True, include_measurement_overlay=True)
        export_dialog = win._create_export_options_dialog(selection)
        export_dialog.show()
        export_dialog.resize(1040, 690)
        export_dialog._export_navigation.setCurrentRow(0)
        capture("14-export-files", win, export_dialog)
        export_dialog._export_navigation.setCurrentRow(1)
        capture("15-export-image", win, export_dialog)
        export_dialog.hide()
        exports = win.export_service.export_project(win.project, DEMO / "导出结果", selection=selection,
            documents=[doc], overlay_renderer=win._render_overlay_image)
        assert exports.success, str(exports)
        rows = win.export_service.build_measurement_rows([doc])
        manifest["measurements"] = rows
        manifest["calibration"] = doc.calibration.to_dict()
        manifest["export_files"] = [p.name for p in sorted((DEMO / "导出结果").iterdir())]
        (ROOT / "public" / "capture-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        win._mark_project_saved()
        with patch.object(win, "_confirm_close_documents", return_value=True):
            win.close()
        print(json.dumps({"version": __version__, "screens": len(manifest["screens"]), "measurements": rows,
            "exports": manifest["export_files"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
