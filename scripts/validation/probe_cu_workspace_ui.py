#!/usr/bin/env python3
"""Render and exercise the nine-category CU transition on the target Qt platform."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QAction, QImage, QPainter, QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow, QInputDialog

from fdm.live_experiment import ExperimentSession
from fdm.live_experiment.models import built_in_materials
from fdm.models import Calibration, ImageDocument, ProjectState
from fdm.ui.live_experiment_workspace import open_live_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    report = {"platform": platform.platform(), "qt_platform": app.platformName(),
              "limits": ["Synthetic static image and QWidget rendering, not a real CU camera stream."],
              "sizes": [], "checks": {}}

    class Host(QMainWindow):
        def current_document(self):
            return self.project.documents[0]

    host = Host()
    host.project = ProjectState.empty()
    host.project.documents.append(ImageDocument(id="sample", path="synthetic.png", image_size=(960, 640),
        calibration=Calibration("manual", 1, "um", "synthetic")))
    image = QImage(960, 640, QImage.Format.Format_RGB32)
    image.fill(QColor("#e7e5e2"))
    painter = QPainter(image)
    for y, width in ((100, 20), (300, 40), (500, 60)):
        painter.fillRect(70, y, 820, width, QColor("#343a3c"))
    painter.end()
    host._images = {"sample": image}
    host.save_project_action = QAction(host)
    session = ExperimentSession(save_images=False, sample_name="CU 操作布局 · 合成样品")
    session.add_operator("操作者 A")
    session.set_materials(built_in_materials()[:9])
    session.establish_history_baseline()
    host.project.live_experiments.append(session)
    view = open_live_experiment(host)
    try:
        view.activateWindow()
        app.processEvents()
        materials = session.operators[session.active_operator_id].materials["warp"]
        for index, material in enumerate(materials):
            x = 180 + index * 65
            for y, width in ((100, 20), (300, 40)):
                record = session.add_measurement(width, material.id, field_id=view.controller.field_id,
                    endpoints=((x, y), (x, y + width)))
                view.controller.current_records.append(record.id)
        view._rebuild_records()
        view._refresh_summary()
        selected = view.material.currentData()
        QTest.mouseClick(view.material_rows[materials[1].id]["add"], Qt.MouseButton.LeftButton)
        view._refresh_summary()
        report["checks"]["row_count_preserves_selected_category"] = (
            view.material.currentData() == selected and view.material_rows[materials[1].id]["extra_count"].text() == "1")
        # The probe's last measurement must enter through the real gesture and
        # async publication path so post-selection exercises its operation ID.
        view.material.setCurrentIndex(1)
        view.measure_mode.setCurrentIndex(1)
        before_count = len(session.measurements)
        scale, offset = view.canvas._transform()
        for x, y in ((100, 100), (100, 120)):
            position = QPoint(round(x * scale + offset[0]), round(y * scale + offset[1]))
            QTest.mouseClick(view.canvas, Qt.MouseButton.LeftButton, pos=position)
        deadline = time.perf_counter() + 2
        while len(session.measurements) == before_count and time.perf_counter() < deadline:
            app.processEvents()
            QTest.qWait(1)
        if len(session.measurements) != before_count + 1:
            raise AssertionError("Real two-click free measurement did not publish")
        view.classification_mode.setCurrentIndex(1)
        last_id = view.controller.current_records[-1]
        report["checks"]["last_measurement_has_real_operation_identity"] = bool(
            session.measurements[last_id].provenance.get("measurement_operation_id"))
        QTest.mouseClick(view.canvas, Qt.MouseButton.RightButton, pos=QPoint(100, 100))
        menu = view._category_menu
        report["checks"]["right_menu_is_nonmodal"] = not menu.isModal()
        next(action for action in menu.actions() if action.data() == materials[0].id).trigger()
        menu.close()
        report["checks"]["right_menu_obeys_after_selection"] = session.measurements[last_id].material_id == materials[0].id
        operator_id = session.active_operator_id
        def fill_operator_name():
            dialog = app.activeModalWidget()
            if isinstance(dialog, QInputDialog):
                dialog.setTextValue("操作者 张三")
                dialog.accept()
        QTimer.singleShot(0, fill_operator_name)
        view._rename_operator()
        report["checks"]["operator_rename_preserves_identity"] = (
            session.active_operator_id == operator_id and len(session.operators) == 1
            and session.operators[operator_id].name == "操作者 张三"
            and session.measurements[last_id].operator_id == operator_id)
        view.classification_mode.setCurrentIndex(0)
        view._refresh_summary()
        for width, height in ((1280, 720), (1330, 900)):
            view.resize(width, height)
            app.processEvents()
            visible = {}
            for material_id, row in view.material_rows.items():
                button = row["select"]
                visible[material_id] = button.visibleRegion().boundingRect().contains(button.rect())
            screenshot = args.output / f"cu-workspace-{width}x{height}.png"
            if not view.grab().save(str(screenshot)):
                raise OSError("Failed to save workspace screenshot")
            report["sizes"].append({"requested": [width, height], "actual": [view.width(), view.height()],
                "device_pixel_ratio": view.devicePixelRatioF(), "category_buttons_visible": visible,
                "keyboard_mode_visible": view.keyboard_mode.visibleRegion().boundingRect().contains(view.keyboard_mode.rect()),
                "screenshot": screenshot.name})
        report["ok"] = (all(report["checks"].values()) and
            all(all(row["category_buttons_visible"].values()) and row["keyboard_mode_visible"] for row in report["sizes"]))
        (args.output / "ui-probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 0 if report["ok"] else 1
    finally:
        view.close()
        host.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
