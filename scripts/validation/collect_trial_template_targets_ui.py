"""Isolated source UI evidence for template sources and quantity preferences.

Use a fresh process per theme/scale; --scale is actual Qt device-pixel ratio.
Screenshots and profiles are local validation artifacts (normally under build/).
Excel state is simulated here; this script does not validate desktop Excel or
open a physical camera. The separate live-template probe covers Excel itself.
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
from unittest.mock import patch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    parser.add_argument("--scale", type=float, choices=(1., 1.25, 1.5, 2.), default=1.)
    parser.add_argument("--size", default="1366x768", help="Main-window logical size")
    parser.add_argument("--platform", default="windows" if sys.platform == "win32" else "offscreen")
    return parser.parse_args()


def configure_scale(args):
    # Same independent native-DPR probe as collect_trial_settings.py. Requesting
    # 125% on a native 200% desktop must use .625, not multiply it to 250%.
    os.environ["QT_QPA_PLATFORM"] = args.platform
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"
    for name in ("QT_SCREEN_SCALE_FACTORS", "QT_AUTO_SCREEN_SCALE_FACTOR", "QT_FONT_DPI"):
        os.environ.pop(name, None)
    probe = subprocess.run([sys.executable, "-c",
        "import json; from PySide6.QtWidgets import QApplication; "
        "app=QApplication([]); screen=app.primaryScreen(); "
        "print(json.dumps({'name':screen.name(),'dpr':screen.devicePixelRatio(),"
        "'logical_dpi':screen.logicalDotsPerInch(),'physical_dpi':screen.physicalDotsPerInch()}))"],
        env=dict(os.environ, QT_SCALE_FACTOR="1"), check=True,
        capture_output=True, text=True, timeout=30)
    native = json.loads(probe.stdout.strip().splitlines()[-1])
    ratio = float(native["dpr"])
    if not 0 < ratio < 16:
        raise RuntimeError(f"Cannot determine native Qt DPR: {native!r}")
    compensation = args.scale / ratio
    os.environ["QT_SCALE_FACTOR"] = str(compensation)
    return native, compensation


def collect(args):
    native, compensation = configure_scale(args)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from PySide6.QtWidgets import QDialogButtonBox, QScrollArea
    from fdm.live_experiment.excel_live import PowerShellLiveTransport
    from fdm.live_experiment.excel_template import ExcelTemplateError, save_template_preferences
    from fdm.ui.dialogs import SettingsDialog
    from fdm.ui.material_catalog import MaterialCatalogDialog
    from fdm.ui.trial_preferences_binding import trial_preferences_binding
    from fdm.version import __version__
    from validate_workflow_redesign import WorkflowHarness, dimensions, settle

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "version": __version__,
        "platform": platform.platform(), "qt_platform": args.platform,
        "requested_dpr": args.scale, "native_screen": native, "scale_compensation": compensation,
        "theme": args.theme, "source_only": True, "hardware_validated": False,
        "desktop_excel_started": False, "excel_state_simulated": True,
        "physical_camera_started": False, "screenshots": [], "checks": {}}

    def check(name, value):
        report["checks"][name] = bool(value)

    def visible(widget):
        return widget.isVisible() and widget.visibleRegion().contains(widget.rect())

    def shot(widget, name):
        settle(15)
        pixmap = widget.grab()
        path = output / (name + ".png")
        if not pixmap.save(str(path)):
            raise RuntimeError(f"Cannot save screenshot: {path}")
        report["screenshots"].append({"file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "logical_size": [widget.width(), widget.height()],
            "raster_size": [pixmap.width(), pixmap.height()], "dpr": pixmap.devicePixelRatio(),
            "widget_dpr": widget.devicePixelRatioF(),
            "window_dpr": widget.windowHandle().devicePixelRatio() if widget.windowHandle() else None,
            "visible": not widget.visibleRegion().isEmpty()})

    def size_dialog(dialog, width, height):
        available = dialog.screen().availableGeometry()
        dialog.resize(min(width, available.width() - 40), min(height, available.height() - 40))
        settle(15)

    def no_excel(*_args, **_kwargs):
        raise ExcelTemplateError("界面验证不连接桌面 Excel。", code="unsupported_platform")

    with patch.object(PowerShellLiveTransport, "request", no_excel):
        with WorkflowHarness(output / "profile", theme=args.theme, size=dimensions(args.size)) as harness:
            harness.enter()
            view = harness.select(1)
            view.controller._timer.stop()
            view._refresh_timer.stop()
            host = harness.window
            binding = trial_preferences_binding(host)
            report["actual_dpr"] = host.devicePixelRatioF()
            report["font"] = harness.font_report()

            settings = SettingsDialog(host._app_settings, document=None, parent=host)
            try:
                binding.prepare_quantity_target_dialog(settings)
                settings.show()
                size_dialog(settings, 960, 640)
                settings.select_trial_section("targets")
                page = settings._trial_settings_widget
                page.total_target.setValue(1200)  # Draft only, never committed.
                page.tabs.widget(1).ensureWidgetVisible(page.total_target)
                settle(10)
                check("total_goal_reachable", visible(page.total_target))
                check("settings_apply_visible", visible(settings.button_box.button(QDialogButtonBox.StandardButton.Apply)))
                check("settings_ok_visible", visible(settings.button_box.button(QDialogButtonBox.StandardButton.Ok)))
                shot(settings, "settings-quantity-target")
                size_dialog(settings, 720, 560)
                page.tabs.widget(1).ensureWidgetVisible(page.total_target)
                settle(10)
                check("narrow_total_goal_reachable", visible(page.total_target))
                check("narrow_settings_apply_visible", visible(settings.button_box.button(QDialogButtonBox.StandardButton.Apply)))
                shot(settings, "settings-quantity-target-narrow")
            finally:
                settings.reject()
                settings.deleteLater()

            catalog = MaterialCatalogDialog(host, path=output / "profile" / "validation-catalog.json")
            try:
                catalog.show()
                size_dialog(catalog, 1000, 640)
                catalog.edit()
                catalog.editor.measured_target.setText("300")
                catalog.editor.count_target.setText("1200")
                scroll = catalog.editor.findChild(QScrollArea)
                for name in ("measured_target", "count_target"):
                    editor = getattr(catalog.editor, name)
                    scroll.ensureWidgetVisible(editor)
                    settle(10)
                    check("catalog_" + name + "_reachable", visible(editor))
                check("catalog_save_visible", visible(catalog.buttons.button(QDialogButtonBox.StandardButton.Save)))
                report["wide_catalog_narrow_layout"] = catalog.panes.narrow
                shot(catalog, "catalog-quantity-targets")
                size_dialog(catalog, 700, 560)
                catalog.panes.show_right()
                settle(10)
                check("narrow_catalog_uses_tabs", catalog.panes.narrow and catalog.panes.tabs.currentWidget() is catalog.panes.right)
                for name in ("measured_target", "count_target"):
                    editor = getattr(catalog.editor, name)
                    scroll.ensureWidgetVisible(editor)
                    settle(10)
                    check("narrow_catalog_" + name + "_reachable", visible(editor))
                check("narrow_catalog_save_visible", visible(catalog.buttons.button(QDialogButtonBox.StandardButton.Save)))
                shot(catalog, "catalog-quantity-targets-narrow")
            finally:
                catalog.reject()
                catalog.deleteLater()

            # The visual check deliberately uses distinct configured/current
            # sources. No template bytes, COM calls or workbook state are used.
            configured = output / "profile" / "templates" / "已选择的自定义模板.xlt"
            active = output / "profile" / "templates" / "当前关联模板.xlt"
            save_template_preferences({"template": str(configured)}, trial_mode="content")
            state = {"state": "template_pending", "message": "等待按所选模板建立新工作簿（界面模拟）",
                "active": True, "connected": True, "ever_connected": True,
                "pending": True, "unsynced": False, "input_pending": False,
                "template_replacement_pending": True,
                "template_path": str(active), "pending_template_path": str(configured),
                "save_state": "never_saved"}
            with patch.object(view.live_excel, "status", lambda *_: dict(state)):
                view.workflow_tabs.setCurrentIndex(2)
                view.report_panel.refresh_path()
                view.report_panel._live_changed(view.session.id)
                view.live_excel_controls._refresh(view.session.id)
                scroll = view.workflow_tabs.widget(2)
                for name in ("path_label", "live_template_label"):
                    label = getattr(view.report_panel, name)
                    scroll.ensureWidgetVisible(label)
                    settle(10)
                    check("report_" + name + "_reachable", visible(label))
                report["template_labels"] = {
                    "configured": view.report_panel.path_label.text(),
                    "current": view.report_panel.live_template_label.text()}
                check("report_distinguishes_sources", str(configured) in report["template_labels"]["configured"]
                      and str(active) in report["template_labels"]["current"]
                      and "正在更换为" in report["template_labels"]["current"])
                shot(host, "report-template-sources")
            report["slot_errors"] = list(harness.slot_errors)
            report["modal_messages"] = list(harness.modal_messages)
            check("no_qt_slot_errors", not harness.slot_errors)
            check("no_unexpected_modals", not harness.modal_messages)

    check("requested_window_dpr", abs(report["actual_dpr"] - args.scale) < 1e-6)
    check("requested_raster_dpr", all(abs(item["dpr"] - args.scale) < 1e-6
        and abs(item["widget_dpr"] - args.scale) < 1e-6
        and item["window_dpr"] is not None and abs(item["window_dpr"] - args.scale) < 1e-6
        for item in report["screenshots"]))
    check("all_captured_windows_visible", all(item["visible"] for item in report["screenshots"]))
    paths = ("scripts/validation/collect_trial_template_targets_ui.py", "scripts/validation/validate_workflow_redesign.py",
        "src/fdm/ui/trial_settings_page.py", "src/fdm/ui/trial_preferences_binding.py", "src/fdm/ui/material_components.py",
        "src/fdm/ui/material_catalog.py", "src/fdm/ui/trial_report_panel.py", "src/fdm/ui/live_experiment_workspace.py")
    report["sources"] = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}
    report["passed"] = all(report["checks"].values())
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(output),
        "requested_dpr": args.scale, "actual_dpr": report["actual_dpr"],
        "failed_checks": [name for name, passed in report["checks"].items() if not passed],
        "screenshots": len(report["screenshots"]), "slot_errors": report["slot_errors"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(collect(arguments()))
