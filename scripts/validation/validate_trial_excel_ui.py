"""Real Qt trial navigation with a deterministic Excel transport, without hardware.

The separate native-worker probe validates Microsoft Excel. Screenshots from
this script validate source UI/layout and must remain local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme", default="dark", choices=("dark", "light"))
    parser.add_argument("--scale", default=1, type=float)
    parser.add_argument("--size", default="1366x768")
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "windows" if sys.platform == "win32" else "offscreen")
    for key in ("QT_SCALE_FACTOR", "QT_SCREEN_SCALE_FACTORS", "QT_FONT_DPI"):
        os.environ.pop(key, None)
    baseline = float(subprocess.check_output([sys.executable, "-c",
        "from PySide6.QtWidgets import QApplication; a=QApplication([]); print(a.primaryScreen().devicePixelRatio())"], text=True).strip())
    os.environ["QT_SCALE_FACTOR"] = str(args.scale / baseline)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from fdm.ui import live_excel
    from fdm.live_experiment import default_materials
    from fdm.live_experiment.models import DiameterRangePolicy
    from fdm.ui.material_catalog import MaterialCatalogDialog, DiameterRangeDialog, TrialMaterialsDialog
    from validate_workflow_redesign import WorkflowHarness, dimensions, settle

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    template = output / "layout-fixture.xlt"
    template.write_bytes(b"UI fixture only; no Excel process opens this file")
    calls = []
    class Transport:
        def request(self, operation, sid, generation, revision, payload=None, **kwargs):
            calls.append({"operation": operation, "session": sid, "revision": revision, "payload": payload})
            return {"ok": True, "status": "synced", "saved": False, "name": "试验工作簿"}
        def close(self):
            return None

    def wait(predicate):
        until = time.monotonic() + 4
        while not predicate() and time.monotonic() < until:
            QTest.qWait(10)
        assert predicate(), "Qt/worker did not finish"

    report = {"theme": args.theme, "scale": args.scale, "native_excel": False, "hardware": False, "screenshots": {}}
    with patch.object(live_excel, "PowerShellLiveTransport", Transport), \
            patch.object(live_excel, "working_template", lambda **kwargs: template), \
            WorkflowHarness(output / "profile", theme=args.theme,
                            size=tuple(round(v / args.scale) for v in dimensions(args.size))) as h:
        window = h.window
        report["font"] = h.font_report()
        report["actual_dpr"] = window.devicePixelRatioF()
        report["baseline_dpr"] = baseline
        assert abs(report["actual_dpr"] - args.scale) < .015
        assert [a.text() for a in window.menuBar().actions()].index("试验") < [a.text() for a in window.menuBar().actions()].index("视图")
        assert window.trial_material_catalog_action.text() == "纤维类别库…"
        h.enter()
        page = h.select(1)
        page.session.set_materials(default_materials(), direction="warp")
        page._refresh_materials()
        page._changed()
        manager = page.live_excel
        wait(lambda: manager.status(page.session.id)["state"] == "synced")
        original = page.session.id

        def snap(name, widget=window):
            settle()
            h.assert_size()
            path = output / (name + ".png")
            assert widget.grab().save(str(path))
            report["screenshots"][name] = hashlib.sha256(path.read_bytes()).hexdigest()

        for diameter in (20, 21, 22):
            page.session.add_measurement(diameter, "wool")
            page._changed()
        wait(lambda: manager.flush(page.session))
        assert len(page.session.measurements) == 3
        assert page.live_excel_controls.show_button.isVisible()
        point = page.live_excel_controls.mapTo(page.trial_sidebar, QPoint())
        assert page.trial_sidebar.rect().contains(point)
        snap("content-live-controls")
        h.select(0)
        page = h.select(1)
        assert page.session.id == original
        assert sum(call["operation"] == "open" and call["session"] == original for call in calls) == 1
        for name, dialog in (("fiber-library", MaterialCatalogDialog(window)),
                             ("trial-materials", TrialMaterialsDialog(page.session, window))):
            dialog.show()
            snap(name, dialog)
            dialog.close()
        diameter_page = h.select(2)
        diameter_page.session.configure(diameter_range_policy=DiameterRangePolicy(10, 40))
        diameter_page._changed()
        diameter_page._refresh_summary()
        wait(lambda: manager.flush(diameter_page.session))
        assert diameter_page.diameter_range_button.isVisible()
        assert not diameter_page.pending_buttons[1].isVisible()
        snap("diameter-live-controls")
        dialog = DiameterRangeDialog(diameter_page.session, window)
        dialog.show()
        snap("diameter-range", dialog)
        dialog.close()
        assert not h.slot_errors, h.slot_errors
        assert not h.modal_messages, h.modal_messages
        report["operations"] = [{k: v for k, v in call.items() if k != "payload"} for call in calls]
        report["passed"] = True
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "screenshots": len(report["screenshots"]), "output": str(output)}))


if __name__ == "__main__":
    main()
