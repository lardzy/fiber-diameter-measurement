"""Source UI evidence, isolated settings and simulated camera only.

Run each scale in a separate process. --scale requests the actual device pixel
ratio, rather than multiplying the host's existing DPI. --size is the logical
main-window size. Images belong under build/, not Git. A platform that cannot
honor the requested ratio is reported as a failed check with its actual ratio.
This collector never starts desktop Excel or a physical camera.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    parser.add_argument("--scale", type=float, choices=(1., 1.25, 1.5, 2.), default=1.)
    parser.add_argument("--size", default="1366x768")
    parser.add_argument("--settings-size", default="900x640")
    parser.add_argument("--platform", default="windows" if sys.platform == "win32" else "offscreen")
    return parser.parse_args()


def collect(args):
    os.environ["QT_QPA_PLATFORM"] = args.platform
    # QT_SCALE_FACTOR multiplies the native ratio (e.g. Windows 200% × 1.5
    # becomes 300%). Some native backends also multiply SCREEN_SCALE_FACTORS.
    # Probe in a separate process, before this process creates QApplication,
    # and compensate without changing any OS display setting.
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"
    for name in ("QT_SCREEN_SCALE_FACTORS", "QT_AUTO_SCREEN_SCALE_FACTOR", "QT_FONT_DPI"):
        os.environ.pop(name, None)
    probe_environment = dict(os.environ, QT_SCALE_FACTOR="1")
    probe = subprocess.run([sys.executable, "-c",
        "import json; from PySide6.QtWidgets import QApplication; "
        "app=QApplication([]); screen=app.primaryScreen(); "
        "print(json.dumps({'name':screen.name(),'dpr':screen.devicePixelRatio(),"
        "'logical_dpi':screen.logicalDotsPerInch(),'physical_dpi':screen.physicalDotsPerInch()}))"],
        env=probe_environment, check=True, capture_output=True, text=True, timeout=30)
    native_screen = json.loads(probe.stdout.strip().splitlines()[-1])
    native_dpr = float(native_screen["dpr"])
    if not 0 < native_dpr < 16:
        raise RuntimeError(f"无法确定 Qt 原生屏幕比例：{native_screen!r}")
    compensation = args.scale / native_dpr
    os.environ["QT_SCALE_FACTOR"] = str(compensation)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from PySide6.QtCore import QEvent, QObject, QPoint
    from PySide6.QtGui import QCursor
    from PySide6.QtTest import QTest
    from fdm.live_experiment import default_materials
    from fdm.live_experiment.excel_live import PowerShellLiveTransport
    from fdm.live_experiment.excel_template import ExcelTemplateError
    from fdm.ui.dialogs import SettingsDialog
    from fdm.ui.trial_preferences_binding import trial_preferences_binding
    from fdm.version import __version__
    from validate_workflow_redesign import WorkflowHarness, dimensions, settle

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "version": __version__,
              "platform": platform.platform(), "qt_platform": args.platform, "scale": args.scale,
              "requested_dpr": args.scale,
              "native_dpr": native_dpr, "native_screen": native_screen,
              "scale_compensation": compensation,
              "scaling_environment": {key: os.environ.get(key) for key in (
                  "QT_SCALE_FACTOR", "QT_SCREEN_SCALE_FACTORS", "QT_SCALE_FACTOR_ROUNDING_POLICY")},
              "theme": args.theme, "source_only": True, "hardware_validated": False,
              "physical_camera_started": False, "desktop_excel_started": False, "screenshots": []}

    def no_excel(*_args, **_kwargs):
        raise ExcelTemplateError("界面验证使用模拟帧，不连接 Excel。", code="unsupported_platform")

    def shot(widget, name):
        settle(15)
        pixmap = widget.grab()
        path = output / (name + ".png")
        if not pixmap.save(str(path)):
            raise RuntimeError(f"无法保存截图：{path}")
        report["screenshots"].append({"file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "logical_size": [widget.width(), widget.height()],
            "raster_size": [pixmap.width(), pixmap.height()], "dpr": pixmap.devicePixelRatio(),
            "widget_dpr": widget.devicePixelRatioF(),
            "window_dpr": widget.windowHandle().devicePixelRatio() if widget.windowHandle() else None,
            "fully_visible": not widget.visibleRegion().isEmpty()})

    def hover_record(window, view):
        """Exercise native mouse delivery; never call the canvas handler."""
        class Moves(QObject):
            def __init__(self):
                super().__init__()
                self.count = 0
                self.spontaneous = 0
                self.last_position = None

            def eventFilter(self, watched, event):
                if event.type() == QEvent.Type.MouseMove:
                    self.count += 1
                    self.spontaneous += int(event.spontaneous())
                    self.last_position = [event.position().x(), event.position().y()]
                return False

        native_windows = sys.platform == "win32" and args.platform == "windows"
        def active_state():
            state = {"qt_active": window.isActiveWindow(), "native_foreground_matches": None}
            if native_windows:
                import ctypes
                foreground = ctypes.windll.user32.GetForegroundWindow
                foreground.restype = ctypes.c_void_p
                state["native_foreground_matches"] = int(foreground() or 0) == int(window.winId())
            return state

        window.show()
        window.raise_()
        window.activateWindow()
        if window.windowHandle() is not None:
            window.windowHandle().requestActivate()
        QTest.qWaitForWindowActive(window, 1500)
        view.canvas.repaint()
        settle()
        before = active_state()
        result = {"delivery": "native_windows" if native_windows else "qt_window",
                  "active_before": before, "passed": False}
        target = next((item for item in view.canvas._overlay_items()[0]
            if item["kind"] == "record" and item["key"] in view.canvas._label_placements
            and view.canvas.visibleRegion().contains(
                view.canvas._label_placements[item["key"]].rect.center().toPoint())), None)
        if target is None:
            result["reason"] = "没有可见的已绘制测量标签，未执行鼠标悬停。"
            return result
        position = view.canvas._label_placements[target["key"]].rect.center().toPoint()
        record_id = target["key"].split(":", 1)[1]
        number = view.session.measurements[record_id].provenance["measurement_number"]
        result.update(expected_record_id=record_id, expected_number=number,
                      target_label=target["label"], target_position=[position.x(), position.y()])
        moves = Moves()
        view.canvas.installEventFilter(moves)
        started = time.perf_counter()
        try:
            # Move elsewhere first so repeated runs do not rely on an already
            # stationary cursor causing the platform to emit a move event.
            QTest.mouseMove(window, QPoint(10, 10), delay=10)
            QTest.qWait(50)
            QTest.mouseMove(view.canvas, position, delay=10)
            deadline = time.monotonic() + 2.
            while time.monotonic() < deadline:
                QTest.qWait(25)
                if (moves.count and view.canvas._hovered_record_id == record_id
                        and f"#{number}" in view.canvas.toolTip()):
                    break
            after = active_state()
            delivered = bool(moves.count and view.canvas._hovered_record_id == record_id
                             and f"#{number}" in view.canvas.toolTip())
            foreground = after["qt_active"] and (not native_windows or after["native_foreground_matches"])
            cursor = view.canvas.mapFromGlobal(QCursor.pos())
            result.update(active_after=after, move_events=moves.count,
                spontaneous_move_events=moves.spontaneous, last_event_position=moves.last_position,
                cursor_position=[cursor.x(), cursor.y()], hovered_record_id=view.canvas._hovered_record_id,
                tooltip=view.canvas.toolTip(), elapsed_ms=(time.perf_counter() - started) * 1000,
                event_delivery_verified=delivered, passed=bool(foreground and delivered))
            if not foreground:
                result["reason"] = "未获得系统前台焦点；本次不作为原生鼠标交互通过证据。"
            elif not delivered:
                result["reason"] = "等待真实事件投递后，标签编号提示或记录命中仍不符合预期。"
        finally:
            view.canvas.removeEventFilter(moves)
        return result

    with patch.object(PowerShellLiveTransport, "request", no_excel):
        with WorkflowHarness(output / "profile", theme=args.theme, size=dimensions(args.size)) as h:
            h.enter()
            screen = h.window.screen()
            report["actual_dpr"] = h.window.devicePixelRatioF()
            report["window_dpr"] = h.window.windowHandle().devicePixelRatio()
            report["screen"] = {"name": screen.name(), "dpr": screen.devicePixelRatio(),
                "logical_dpi": screen.logicalDotsPerInch(), "physical_dpi": screen.physicalDotsPerInch(),
                "available_logical_size": [screen.availableGeometry().width(), screen.availableGeometry().height()]}
            binding = trial_preferences_binding(h.window)
            for index, mode in ((1, "content"), (2, "diameter")):
                view = h.select(index)
                if index == 1:
                    for direction in ("warp", "weft"):
                        view.session.set_materials(default_materials(), direction=direction)
                    view._refresh_materials()
                view.ruler_choice.setCurrentIndex(1)
                view.ruler_choice.activated.emit(1)
                for number, (material_id, start, end, value) in enumerate((
                    ("wool", [140, 170], [220, 190], 18.25),
                    ("cashmere", [270, 280], [345, 295], 16.50),
                    ("wool", [395, 390], [475, 410], 19.00)), 1):
                    record = view.session.add_measurement(value, material_id,
                        field_id=view.controller.field_id, endpoints=[start, end],
                        provenance={"measurement_number": number}, evidence={"state": "disabled"})
                    view.controller.current_records.append(record.id)
                view._changed()
                view.controller._timer.stop()
                view._refresh_summary()
                shot(h.window, f"{mode}-default")
                report[mode] = {"label_text": [item["label"] for item in view.canvas._overlay_items()[0]
                    if item["kind"] == "record"], "right_status": view.live_state_label.text(),
                    "evidence_mode": view.controller.effective_evidence_options["mode"],
                    "hover": hover_record(h.window, view)}
            dialog = SettingsDialog(h.window._app_settings, document=None, parent=h.window)
            try:
                dialog.show()
                settle()
                dialog.resize(*dimensions(args.settings_size))
                for section in ("display", "evidence", "analysis"):
                    dialog.select_trial_section(section)
                    shot(dialog, "settings-" + section)
                    if section == "evidence":
                        page = dialog._trial_settings_widget
                        page.tabs.widget(1).ensureWidgetVisible(page.cleanup_button)
                        settle()
                        report["cleanup_button_reachable"] = page.cleanup_button.visibleRegion().contains(page.cleanup_button.rect())
                dialog.select_trial_section("display")
                page = dialog._trial_settings_widget
                # Preview is scrollable on shorter clients. Capture its actual
                # reachable layout separately rather than cropping it away.
                scroll = page.tabs.widget(0)
                scroll.ensureWidgetVisible(page.preview)
                shot(dialog, "settings-display-preview")
                report["settings"] = {"navigation": [dialog._settings_navigation.item(i).text()
                    for i in range(dialog._settings_navigation.count())],
                    "apply_visible": dialog.button_box.visibleRegion().contains(dialog.button_box.rect()),
                    "preview_visible": page.preview.visibleRegion().contains(page.preview.rect()),
                    "cleanup_label": page.cleanup_button.text()}
            finally:
                dialog.close()
            binding.flush()
            report["preferences"] = binding.preferences.to_dict()
            report["font"] = h.font_report()
            report["slot_errors"] = list(h.slot_errors)
            report["modal_messages"] = list(h.modal_messages)
            report["checks"] = {
                "no_qt_slot_errors": not h.slot_errors,
                "no_unexpected_modal_messages": not h.modal_messages,
                "requested_window_dpr": abs(report["window_dpr"] - args.scale) < 1e-6,
                "requested_raster_dpr": all(abs(item["dpr"] - args.scale) < 1e-6
                    and abs(item["widget_dpr"] - args.scale) < 1e-6
                    and item["window_dpr"] is not None and abs(item["window_dpr"] - args.scale) < 1e-6
                    for item in report["screenshots"]),
                "all_captured_windows_visible": all(item["fully_visible"] for item in report["screenshots"]),
                "settings_footer_visible": report["settings"]["apply_visible"],
                "settings_preview_reachable": report["settings"]["preview_visible"],
                "cleanup_button_reachable": report["cleanup_button_reachable"],
                "content_native_hover": report["content"]["hover"]["passed"],
                "diameter_native_hover": report["diameter"]["hover"]["passed"],
                "image_storage_default_off": all(report[mode]["evidence_mode"] == "disabled"
                    for mode in ("content", "diameter")),
            }
            report["passed"] = all(report["checks"].values())
            report["acceptance_boundaries"] = [mode + ": " + report[mode]["hover"]["reason"]
                for mode in ("content", "diameter") if "reason" in report[mode]["hover"]]
    paths = ("scripts/validation/collect_trial_settings.py", "src/fdm/settings.py", "src/fdm/trial_preferences.py", "src/fdm/ui/trial_settings_page.py",
             "src/fdm/ui/trial_label_style.py", "src/fdm/ui/live_measurement_canvas.py",
             "src/fdm/ui/live_experiment_workspace.py", "src/fdm/ui/trial_assistance.py")
    report["sources"] = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(output),
                      "requested_dpr": args.scale, "actual_dpr": report["actual_dpr"],
                      "native_dpr": native_dpr, "scale_compensation": compensation,
                      "failed_checks": [name for name, passed in report["checks"].items() if not passed],
                      "screenshots": len(report["screenshots"]), "slot_errors": report["slot_errors"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(collect(arguments()))
