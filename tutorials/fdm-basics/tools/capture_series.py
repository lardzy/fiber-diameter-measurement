"""Capture tutorials 02, 03, 04/05/08/06/07 and 09 from the current Qt app.

The user's profile and source images are isolated. Segmentation uses the real
local model and normal worker pipeline; no inferred mask or value is mocked.
Run with uv run --no-sync python tutorials/fdm-basics/tools/capture_series.py [02 ...]
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import capture_fdm as base
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtCore import QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QAbstractButton, QFileDialog, QInputDialog, QMessageBox
from fdm import runtime_logging, screenshot_settings, settings
from fdm.geometry import Point
from fdm.models import Calibration, ImageDocument, OverlayAnnotationKind, new_id
from fdm.services.export_service import ExportSelection
from fdm.ui.dialogs import CalibrationInputDialog, CalibrationPresetDialog, FiberGroupDialog
from fdm.ui.image_loader import ImageLoadRequest
from fdm.ui.main_window import MainWindow
from fdm.ui.scale_overlay_editor import freeze_scale_contexts
from fdm.ui.theme import apply_application_theme
from fdm.version import __version__

ROOT = base.ROOT
DEMO = ROOT / "demo-series"
SCREENS = ROOT / "public/screens/series"
MANIFEST = ROOT / "public/series-capture.json"
DEMO.mkdir(exist_ok=True)
SCREENS.mkdir(parents=True, exist_ok=True)
manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"version": __version__, "screens": {}, "checks": {}}
settle = base.settle
errors = []


def persist():
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def capture(name, win, dialog=None, points=None):
    base.SCREENS = SCREENS
    base.manifest = manifest
    base.capture(name, win, dialog)
    entry = manifest["screens"][name]
    canvas = win.current_canvas()
    if canvas and points:
        origin = canvas.mapTo(win, QPoint(0, 0))
        for key, pt in points.items():
            q = canvas.image_to_widget(Point(*pt))
            entry["image_points"][key] = [q.x() + origin.x(), q.y() + origin.y()]
    if dialog:
        dx, dy, _, _ = entry["dialog"]
        entry["dialog_buttons"] = {}
        for button in dialog.findChildren(QAbstractButton):
            label = button.text().replace("&", "").strip()
            if label and button.isVisible():
                x, y, w, h = base.rect(button, dialog)
                entry["dialog_buttons"][label] = [x + dx, y + dy, w, h]
    persist()


def wait_for(predicate, label, timeout=90):
    start = time.monotonic()
    while not predicate():
        settle(80)
        if errors:
            raise RuntimeError(errors[-1])
        if time.monotonic() - start > timeout:
            raise TimeoutError(label)
    settle(200)


def modal(kind, fn):
    started = time.monotonic()
    def poll():
        widget = next((x for x in QApplication.topLevelWidgets() if isinstance(x, kind) and x.isVisible()), None)
        if widget is None:
            if time.monotonic() - started < 20:
                QTimer.singleShot(40, poll)
            else:
                errors.append(f"Dialog did not appear: {kind}")
            return
        try:
            fn(widget)
        except Exception as exc:
            errors.append(repr(exc))
            widget.reject()
    QTimer.singleShot(100, poll)


def click(win, xy):
    canvas = win.current_canvas()
    canvas.setFocus()
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, canvas.image_to_widget(Point(*xy)).toPoint())
    settle(120)


def path(win, points, preview=None):
    for xy in points:
        click(win, xy)
    if preview:
        capture(preview, win, points={f"p{i}": p for i, p in enumerate(points)})
    assert win._commit_active_path_drawing()
    settle(200)
    win._flush_pending_measurements(for_snapshot=True)


def specimen(name):
    base.DEMO = DEMO
    image, original = base.make_specimen()
    target = DEMO / f"{name}.png"
    image.save(str(target))
    target.with_suffix(".png.fdm.json").unlink(missing_ok=True)
    return image, target


def add(win, image, filename, calibrated=False):
    doc = ImageDocument(id=new_id("image"), path=str(filename), image_size=(image.width(), image.height()))
    doc.initialize_runtime_state()
    group = doc.create_group(color="#248d79", label="示例纤维")
    doc.set_active_group(group.id)
    if calibrated:
        doc.calibration = Calibration(mode="manual", pixels_per_unit=4, unit="um", source_label="合成教学标尺 400 px = 100 μm")
    win._add_loaded_document(ImageLoadRequest(path=doc.path, document=doc), image)
    win._set_current_document(doc.id)
    settle(250)
    win.fit_current_image()
    return doc


def measure(win, pairs=((212, 250, 288, 250), (550, 350, 650, 350), (888, 205, 1012, 205))):
    win.set_tool_mode("manual")
    for x1, y1, x2, y2 in pairs:
        base.drag(win, Point(x1, y1), Point(x2, y2))
    win._flush_pending_measurements(for_snapshot=True)


def save(win, name):
    target = DEMO / f"{name}.fdmproj"
    result = win.save_project(str(target))
    assert result.success, result
    win.statusBar().clearMessage()
    return target


def results(win, tab=0):
    if not win._results_tabs.isVisible():
        win._toggle_results_panel()
    win._results_tabs.setCurrentIndex(tab)
    settle(300)


def close_results(win):
    if win._results_tabs.isVisible():
        win._toggle_results_panel()
    win.fit_current_image()


def episode02(win):
    a, ap = specimen("02-样本A")
    b, bp = specimen("02-样本B")
    chooser = QFileDialog(win, "打开图片", str(DEMO), "图片 (*.png *.jpg *.tif)")
    chooser.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    chooser.setFileMode(QFileDialog.FileMode.ExistingFiles)
    chooser.selectFile('"02-样本A.png" "02-样本B.png"')
    chooser.resize(1000, 600)
    chooser.show()
    capture("02-multi-open", win, chooser)
    chooser.hide()
    da = add(win, a, ap, True)
    measure(win, [(212, 250, 288, 250)])
    db = add(win, b, bp, True)
    measure(win, [(550, 350, 650, 350), (888, 205, 1012, 205)])
    capture("02-image-b", win)
    win._set_current_document(da.id)
    win.fit_current_image()
    capture("02-image-a", win)
    results(win)
    capture("02-records-a", win)
    win._set_current_document(db.id)
    capture("02-records-b", win)
    close_results(win)
    target = DEMO / "02-多图片项目.fdmproj"
    chooser = QFileDialog(win, "保存项目", str(DEMO), "FDM 项目 (*.fdmproj)")
    chooser.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    chooser.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    chooser.selectFile(str(target))
    chooser.resize(1000, 600)
    chooser.show()
    capture("02-save-dialog", win, chooser)
    chooser.hide()
    save(win, "02-多图片项目")
    capture("02-saved", win)
    with patch.object(win, "_confirm_close_documents", return_value=True):
        result = win._load_project_from_path(target)
    assert result.success, result
    wait_for(lambda: len(win._images) == 2, "project images reload")
    win._set_current_document(win.project.documents[0].id)
    win.fit_current_image()
    results(win)
    capture("02-reopened", win)
    counts = sorted(len(d.measurements) for d in win.project.documents)
    assert counts == [1, 2], counts
    manifest["checks"]["02"] = {"reload_measurement_counts": counts, "saved_project": str(target)}


def episode03(win):
    a, ap = specimen("03-同条件A")
    b, bp = specimen("03-同条件B")
    da = add(win, a, ap)
    db = add(win, b, bp)
    win._set_current_document(da.id)
    win.fit_current_image()
    win.set_tool_mode("calibration")
    capture("03-before", win)
    def fill(d):
        d._length_spin.setValue(100)
        d._apply_to_project.setChecked(False)
        capture("03-calibration-input", win, d)
        d.accept()
    modal(CalibrationInputDialog, fill)
    base.drag(win, Point(800, 744), Point(1200, 744))
    assert da.calibration and not db.calibration
    capture("03-calibrated", win)
    win._show_inspector_section(win._calibration_section)
    settle(250)
    capture("03-preset-panel", win)
    def preset(d):
        d._name_edit.setText("教学相机 · 1280×820 · 同条件")
        d._pixel_distance_spin.setValue(400)
        d._actual_distance_spin.setValue(100)
        d._unit_combo.setCurrentIndex(d._unit_combo.findData("um"))
        capture("03-preset-new", win, d)
        d.accept()
    modal(CalibrationPresetDialog, preset)
    win.add_calibration_preset()
    assert len(win._app_settings.calibration_presets) == 1
    win._set_current_document(db.id)
    win.fit_current_image()
    win._show_inspector_section(win._calibration_section)
    capture("03-target-unscaled", win)
    def scope_current(d):
        capture("03-scope-current", win, d)
        next(x for x in d.buttons() if x.text() == "当前图片").click()
    modal(QMessageBox, scope_current)
    win.apply_selected_preset()
    assert db.calibration
    measure(win, [(550, 350, 650, 350)])
    capture("03-preset-applied", win)
    def scope_all(d):
        capture("03-scope-project", win, d)
        next(x for x in d.buttons() if x.text() == "项目所有图片").click()
    modal(QMessageBox, scope_all)
    win.apply_selected_preset()
    assert all(x.calibration and x.calibration.pixels_per_unit == 4 for x in win.project.documents)
    capture("03-project-scaled", win)
    def units(d):
        d._name_edit.setText("教学相机 · 同条件 · 毫米单位")
        d._pixel_distance_spin.setValue(400)
        d._actual_distance_spin.setValue(0.1)
        d._unit_combo.setCurrentIndex(d._unit_combo.findData("mm"))
        capture("03-units", win, d)
        d.accept()
    modal(CalibrationPresetDialog, units)
    win.add_calibration_preset()
    modal(QMessageBox, lambda d: next(x for x in d.buttons() if x.text() == "当前图片").click())
    win.apply_selected_preset()
    results(win)
    capture("03-mm-result", win)
    assert db.calibration.unit == "mm" and abs(db.calibration.pixels_per_unit - 4000) < 0.001
    save(win, "03-标定与预设")
    manifest["checks"]["03"] = {"presets": [p.to_dict() for p in win._app_settings.calibration_presets], "units": [d.calibration.unit for d in win.project.documents]}


def area_image():
    img = QImage(1280, 820, QImage.Format.Format_RGB32)
    img.fill(QColor("#e4ede7"))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QColor("#385f54"))
    p.setBrush(QColor("#87ada0"))
    outer = [(220,160),(490,130),(620,300),(570,520),(360,570),(200,430)]
    hole = [(345,265),(430,250),(468,330),(395,382),(335,335)]
    p.drawPolygon(QPolygonF([QPointF(*x) for x in outer]))
    p.drawEllipse(QPointF(900,310), 150,130)
    p.drawEllipse(QPointF(900,610), 75,65)
    p.setBrush(QColor("#e4ede7"))
    p.drawPolygon(QPolygonF([QPointF(*x) for x in hole]))
    p.setFont(QFont("PingFang SC", 18))
    p.drawText(40, 780, "合成面积与计数练习 · 非实测截面")
    p.end()
    file = DEMO / "06-面积计数练习.png"
    img.save(str(file))
    file.with_suffix(".png.fdm.json").unlink(missing_ok=True)
    return img, file, outer, hole


def reference_image():
    img = QImage(1280, 820, QImage.Format.Format_RGB32)
    img.fill(QColor("#e1e7e2"))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    for x, y in [(140,130),(740,130),(140,340),(740,340),(140,550),(740,550)]:
        grad = QLinearGradient(x,y,x,y+65)
        for at, color in [(0,"#476b60"),(.2,"#9fbeb0"),(.5,"#d1dfd4"),(.8,"#7ca391"),(1,"#476b60")]:
            grad.setColorAt(at,QColor(color))
        p.setBrush(grad)
        p.setPen(QPen(QColor("#3c6054"),3))
        p.drawRoundedRect(x,y,340,65,15,15)
        p.setPen(QPen(QColor("#759687"),1))
        for step in range(24,325,28):
            p.drawLine(x+step,y+8,x+step+13,y+57)
    p.setPen(QColor("#35594d"))
    p.setFont(QFont("PingFang SC",19))
    p.drawText(40,770,"合成同类扩选练习 · 六个相似对象 · 非实测样品")
    p.end()
    file = DEMO / "07-同类扩选合成练习.png"
    img.save(str(file))
    file.with_suffix(".png.fdm.json").unlink(missing_ok=True)
    return img, file


def episode_measure(win):
    a, ap = specimen("04-分类与线段练习")
    da = add(win, a, ap, True)
    measure(win)
    capture("04-before", win)
    def category(d):
        d._label_edit.setText("复核样本")
        d._apply_button_color("#b16c35")
        d._apply_to_project.setChecked(True)
        capture("04-new-category", win, d)
        d.accept()
    modal(FiberGroupDialog, category)
    win.add_fiber_group()
    group = da.find_group_by_label("复核样本")
    assert group
    measure(win, [(212, 500, 288, 500)])
    capture("04-new-measurement", win)
    win._on_measurement_group_change_requested(da.measurements[1].id, group.id)
    results(win)
    capture("04-reclassified", win)
    win._bottom_records_pane.controller.set_filters(kind="length", group="复核样本")
    capture("04-filtered", win)
    win._activate_measurement_id(da.measurements[1].id)
    capture("04-located", win)
    win._bottom_records_pane.controller.set_filters()
    results(win, 1)
    capture("04-statistics", win)
    results(win, 2)
    capture("04-distribution", win)
    close_results(win)
    win.set_tool_mode("continuous_manual")
    capture("05-continuous-tool", win)
    path(win, [(420,120),(420,285),(465,395),(445,545)], "05-polyline-preview")
    capture("05-polyline-result", win)
    win.set_tool_mode("snap")
    base.drag(win, Point(550,520), Point(650,520), "05-snap-preview")
    if win.current_canvas().has_pending_path_drawing():
        assert win._commit_active_path_drawing()
    win._flush_pending_measurements(for_snapshot=True)
    capture("05-snap-result", win)
    before_undo = len(da.measurements)
    win.undo_current_document()
    settle(200)
    capture("05-undo", win)
    assert len(da.measurements) == before_undo - 1

    source = ROOT / "assets/optical-test.jpg"
    target = DEMO / "08-光学测试图-无已知标尺.jpg"
    shutil.copyfile(source, target)
    target.with_suffix(".jpg.fdm.json").unlink(missing_ok=True)
    optical = add(win, QImage(str(target)), target)
    assert optical.calibration is None
    win.set_tool_mode(settings.MagicSegmentToolMode.FIBER_QUICK)
    capture("08-quick-tool", win, points={"seed": (1230,674)})
    click(win, (1230,674))
    canvas = win.current_canvas()
    wait_for(lambda: not canvas.is_fiber_quick_busy(), "quick diameter segmentation", 180)
    assert canvas.has_fiber_quick_shape_preview(), win.statusBar().currentMessage()
    capture("08-quick-contour", win, points={"seed": (1230,674)})
    assert win._commit_fiber_quick_preview()
    wait_for(lambda: len(optical.measurements) >= 1, "quick diameter geometry", 120)
    win._flush_pending_measurements(for_snapshot=True)
    capture("08-quick-result", win, points={"seed": (1230,674)})
    results(win)
    capture("08-quick-record", win)
    close_results(win)

    ai, af, outer, hole = area_image()
    area = add(win, ai, af)
    win.set_tool_mode("polygon_area")
    path(win, outer, "06-polygon-preview")
    capture("06-polygon-result", win)
    area_id = area.measurements[0].id
    canvas = win.current_canvas()
    area.select_measurement(area_id)
    canvas.set_selected_measurement(area_id)
    assert win._cycle_area_edit_operation_mode()
    capture("06-subtract-tool", win)
    path(win, hole, "06-hole-preview")
    wait_for(lambda: not canvas.has_pending_path_drawing(), "area subtract")
    capture("06-hole-result", win)
    win.set_tool_mode("freehand_area")
    canvas.set_selected_measurement(None)
    area.select_measurement(None)
    canvas.set_area_edit_operation_mode("add")
    points = [(900+150*math.cos(i*2*math.pi/60),310+130*math.sin(i*2*math.pi/60)) for i in range(61)]
    begin = canvas.image_to_widget(Point(*points[0])).toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, begin)
    for xy in points[1:]:
        pos = canvas.image_to_widget(Point(*xy))
        settle(80)
        QApplication.sendEvent(canvas, QMouseEvent(QEvent.Type.MouseMove, pos, canvas.mapToGlobal(pos.toPoint()), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    assert len(canvas._drawing_polygon_points) >= 20, len(canvas._drawing_polygon_points)
    capture("06-freehand-preview", win)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, begin)
    settle(300)
    win._flush_pending_measurements(for_snapshot=True)
    assert len(area.measurements) == 2, len(area.measurements)
    capture("06-freehand-result", win)
    win.set_tool_mode("count")
    for pt in [(400,450),(900,310),(900,610)]:
        click(win, pt)
    win._flush_pending_measurements(for_snapshot=True)
    results(win)
    capture("06-count-result", win)
    close_results(win)
    assert len(area.measurements) == 5, len(area.measurements)

    win._set_current_document(optical.id)
    win.fit_current_image()
    win.set_tool_mode(settings.MagicSegmentToolMode.STANDARD)
    capture("07-magic-tool", win, points={"seed": (1230,674)})
    click(win, (1230,674))
    canvas = win.current_canvas()
    wait_for(lambda: not canvas.is_magic_segment_busy(), "standard magic", 120)
    capture("07-positive-preview", win, points={"seed": (1230,674)})
    win._cycle_magic_segment_prompt_type()
    click(win, (1230,780))
    wait_for(lambda: not canvas.is_magic_segment_busy(), "negative magic prompt", 120)
    capture("07-negative-preview", win, points={"seed": (1230,674), "negative": (1230,780)})
    assert win._commit_magic_segment_preview()
    wait_for(lambda: len(optical.measurements) >= 2, "magic commit")
    win._flush_pending_measurements(for_snapshot=True)
    capture("07-area-result", win)
    win.set_tool_mode(settings.MagicSegmentToolMode.REFERENCE)
    capture("07-reference-optical", win)
    click(win, (1230,674))
    wait_for(lambda: not canvas.is_reference_instance_busy(), "similar instance search", 180)
    capture("07-reference-duplicate", win)
    optical_candidates = len(canvas._reference_instance.preview_candidates)
    win._commit_reference_instance_preview()
    # Use an explicitly synthetic fixture for a multi-candidate example.
    # Both the mask and candidates still use the actual production worker path.
    ri, rf = reference_image()
    reference = add(win, ri, rf)
    win.set_tool_mode(settings.MagicSegmentToolMode.STANDARD)
    click(win,(310,162))
    canvas = win.current_canvas()
    wait_for(lambda: not canvas.is_magic_segment_busy(),"reference fixture mask",120)
    assert win._commit_magic_segment_preview()
    wait_for(lambda: len(reference.measurements) == 1,"reference fixture commit")
    win.set_tool_mode(settings.MagicSegmentToolMode.REFERENCE)
    capture("07-reference-tool",win,points={"seed":(310,162)})
    click(win,(310,162))
    wait_for(lambda: not canvas.is_reference_instance_busy(),"synthetic reference search",180)
    capture("07-reference-preview",win,points={"seed":(310,162)})
    candidate_count = len(canvas._reference_instance.preview_candidates)
    print("synthetic reference candidates", candidate_count, flush=True)
    assert candidate_count > 1, win.statusBar().currentMessage()
    assert win._commit_reference_instance_preview()
    settle(500)
    win._flush_pending_measurements(for_snapshot=True)
    capture("07-reference-result", win)
    results(win)
    capture("07-review", win)
    save(win, "04-05-08-06-07-测量操作")
    manifest["checks"]["measure"] = {"line_measurements": len(da.measurements), "area_and_count": len(area.measurements), "optical_measurements": len(optical.measurements), "optical_reference_candidates": optical_candidates, "actual_reference_candidates": candidate_count, "synthetic_reference_measurements": len(reference.measurements), "model": win._app_settings.magic_segment_model_variant, "optical_has_calibration": optical.calibration is not None, "source": str(source)}


def episode04(win):
    result = win._load_project_from_path(DEMO / "04-05-08-06-07-测量操作.fdmproj")
    assert result.success, result
    wait_for(lambda: len(win._images) == 4, "load category examples")
    doc = next(d for d in win.project.documents if "04-分类" in d.path)
    win._set_current_document(doc.id)
    # Return this capture-only copy to the category chapter, before the polyline.
    for obj in list(doc.measurements):
        if obj.measurement_kind == "polyline":
            win._apply_document_change(doc,"删除练习折线",lambda obj=obj: doc.remove_measurement(obj.id))
    win.fit_current_image()
    results(win)
    win._bottom_records_pane.search_edit.setText("复核样本")
    settle(200)
    capture("04-filtered",win)
    target = next(m for m in doc.measurements if doc.get_group(m.fiber_group_id).label == "复核样本")
    win._activate_measurement_id(target.id)
    capture("04-located",win)
    manifest["checks"]["04"]={"search_query":"复核样本","filtered_records":win._bottom_records_pane.controller.proxy.rowCount(),"total_records":len(doc.measurements)}


def episode05(win):
    result = win._load_project_from_path(DEMO / "04-05-08-06-07-测量操作.fdmproj")
    assert result.success, result
    wait_for(lambda: len(win._images) == 4, "load measurement examples")
    doc = next(d for d in win.project.documents if "04-分类" in d.path)
    win._set_current_document(doc.id)
    for obj in list(doc.measurements):
        if obj.measurement_kind == "polyline" or obj.mode == "snap":
            win._apply_document_change(doc,"重置线段练习",lambda obj=obj: doc.remove_measurement(obj.id))
    close_results(win)
    win.set_tool_mode("continuous_manual")
    capture("05-continuous-tool",win)
    path(win,[(420,120),(420,285),(465,395),(445,545)],"05-polyline-preview")
    capture("05-polyline-result",win)
    before = len(doc.measurements)
    win.set_tool_mode("snap")
    click(win,(550,520))
    canvas = win.current_canvas()
    pos = canvas.image_to_widget(Point(650,520))
    QApplication.sendEvent(canvas,QMouseEvent(QEvent.Type.MouseMove,pos,canvas.mapToGlobal(pos.toPoint()),Qt.MouseButton.NoButton,Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier))
    capture("05-snap-preview",win,points={"edge_start":(550,520),"edge_end":(650,520)})
    click(win,(650,520))
    win._flush_pending_measurements(for_snapshot=True)
    assert len(doc.measurements) == before+1
    capture("05-snap-result",win,points={"edge_start":(550,520),"edge_end":(650,520)})
    win.undo_current_document()
    capture("05-undo",win)
    assert len(doc.measurements) == before
    win.set_tool_mode("select")
    obj = doc.measurements[0]
    win._activate_measurement_id(obj.id)
    settle(250)
    start = Point(obj.line_px.end.x,obj.line_px.end.y)
    end = Point(start.x+15,start.y)
    a = canvas.image_to_widget(start)
    b = canvas.image_to_widget(end)
    QApplication.sendEvent(canvas,QMouseEvent(QEvent.Type.MouseButtonPress,a,canvas.mapToGlobal(a.toPoint()),Qt.MouseButton.LeftButton,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier))
    assert canvas._dragging_handle is not None, (a,canvas._tool_mode,doc.selected_measurement_id)
    QApplication.sendEvent(canvas,QMouseEvent(QEvent.Type.MouseMove,b,canvas.mapToGlobal(b.toPoint()),Qt.MouseButton.NoButton,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier))
    QApplication.sendEvent(canvas,QMouseEvent(QEvent.Type.MouseButtonRelease,b,canvas.mapToGlobal(b.toPoint()),Qt.MouseButton.LeftButton,Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier))
    settle(300)
    edited = doc.get_measurement(obj.id)
    assert edited.effective_line().end.x > start.x+5, (start,edited.effective_line().end)
    capture("05-endpoint-edited",win,points={"endpoint":(edited.effective_line().end.x,edited.effective_line().end.y)})
    win.undo_current_document()
    capture("05-endpoint-undo",win)
    save(win,"04-05-08-06-07-测量操作")
    manifest["checks"]["05"] = {"snap_created":True,"endpoint_edit_verified":True,"undo_verified":True,"remaining_line_measurements":len(doc.measurements)}
    manifest["checks"]["measure"]["line_measurements"] = len(doc.measurements)


def episode09(win):
    a, ap = specimen("09-标注与导出练习")
    doc = add(win, a, ap, True)
    measure(win)
    win.set_tool_mode("select")
    win.scale_preview.begin()
    capture("09-scale-panel", win)
    win.scale_preview.change(length_mode="custom", length=50, unit="um", position="top_right", font_mode="custom", font_size=28, font_family="PingFang SC", style="ticks", color="#163c33", text_color="#163c33", line_width=4)
    capture("09-scale-custom", win)
    win.scale_preview.change(style="bar", text_position="below")
    capture("09-scale-style", win)
    win.scale_preview.confirm()
    win.scale_preview.finish()
    win.fit_current_image()
    capture("09-scale-finished", win)
    win._activate_overlay_tool(OverlayAnnotationKind.TEXT)
    def text_input(d):
        d.setTextValue("示例纤维 · 待复核")
        capture("09-text-dialog", win, d)
        d.accept()
    modal(QInputDialog, text_input)
    click(win, (360,130))
    assert len(doc.overlay_annotations) == 1
    capture("09-text-result", win)
    win._activate_overlay_tool(OverlayAnnotationKind.ARROW)
    base.drag(win, Point(520,185), Point(300,260))
    assert len(doc.overlay_annotations) == 2
    capture("09-arrow-result", win)
    selection = ExportSelection(include_excel=True, include_csv=True, include_measurement_overlay=True, include_combined_overlay=True, include_scale_json=True, scale_overlay=win.scale_preview.spec)
    dialog = win._create_export_options_dialog(selection)
    dialog.resize(1040,690)
    dialog.show()
    dialog._export_navigation.setCurrentRow(0)
    capture("09-export-files", win, dialog)
    dialog._export_navigation.setCurrentRow(1)
    capture("09-export-images", win, dialog)
    dialog.hide()
    contexts = freeze_scale_contexts(win, [doc], selection, {})
    # The standard service receives the same frozen scale layout as GUI exports.
    import inspect
    kwargs = {"selection": selection, "documents": [doc], "overlay_renderer": win._render_overlay_image}
    signature = inspect.signature(win.export_service.export_project)
    if "render_contexts" in signature.parameters:
        kwargs["render_contexts"] = contexts
    exported = win.export_service.export_project(win.project, DEMO / "09-导出结果", **kwargs)
    assert exported.success, exported
    artifact = ROOT / "public/exports/09-combined.png"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        DEMO / "09-导出结果/09-标注与导出练习_measurements_scale_fullres.png",
        artifact,
    )
    save(win, "09-比例尺标注导出")
    capture("09-finished", win)
    manifest["checks"]["09"] = {"overlays": len(doc.overlay_annotations), "export": str(exported), "scale": win.scale_preview.spec.to_dict()}


def main():
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)
    app = QApplication([])
    app.setFont(QFont("PingFang SC", 10))
    apply_application_theme(app, "light")
    choices = sys.argv[1:] or ["02", "03", "measure", "04", "05", "09"]
    for episode in choices:
        prefs = settings.AppSettings(theme_mode="light", object_snap_enabled=False,
            measurement_label_font_family="PingFang SC", measurement_label_font_size=24,
            measurement_label_color="#ffffff", text_font_family="PingFang SC",
            text_font_size=28, text_color="#163c33", overlay_line_color="#ad612c", overlay_line_width=5)
        with tempfile.TemporaryDirectory(prefix="fdm-series-profile-") as profile, ExitStack() as stack:
            p = Path(profile)
            stack.enter_context(patch.object(settings, "settings_file_path", lambda: p / "settings.json"))
            stack.enter_context(patch.object(screenshot_settings, "screenshot_settings_file_path", lambda: p / "screenshot.json"))
            stack.enter_context(patch.object(runtime_logging, "runtime_log_path", lambda: p / "startup.log"))
            stack.enter_context(patch("fdm.ui.main_window.AppSettingsIO.load", return_value=prefs))
            stack.enter_context(patch("fdm.ui.main_window.AppSettingsIO.save", return_value=None))
            win = MainWindow()
            win.resize(1600,900)
            win.show()
            settle(400)
            try:
                {"02": episode02, "03": episode03, "measure": episode_measure, "04": episode04, "05": episode05, "09": episode09}[episode](win)
                if errors:
                    raise RuntimeError(errors)
                print("VERIFIED", episode, manifest["checks"][episode], flush=True)
            finally:
                persist()
                with patch.object(win, "_confirm_close_documents", return_value=True):
                    win.close()
                settle(300)


if __name__ == "__main__":
    main()
