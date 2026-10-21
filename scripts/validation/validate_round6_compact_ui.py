#!/usr/bin/env python3
"""Round-six album entry screenshots, native button actions and geometry checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme", choices=("light", "dark", "navy", "warm"), default="light")
    parser.add_argument("--size", default="1600x900")
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--matrix", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.matrix:
        cases = []
        for theme in ("light", "dark", "navy", "warm"):
            for size, scale in (("1600x900", 1), ("1600x900", 1.25),
                                ("1600x900", 1.5), ("1366x768", 1)):
                key = f"{theme}-{size}-{scale:g}"
                result = subprocess.run([sys.executable, __file__, "--theme", theme,
                    "--size", size, "--scale", str(scale), "--output", str(args.output / key)],
                    capture_output=True, text=True)
                (args.output / f"{key}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
                cases.append({"case": key, "passed": result.returncode == 0})
        (args.output / "matrix.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
        print(json.dumps(cases))
        return 0 if all(case["passed"] for case in cases) else 1

    os.environ.setdefault("QT_QPA_PLATFORM", "windows" if sys.platform == "win32" else "offscreen")
    os.environ["QT_SCALE_FACTOR"] = str(args.scale)
    sys.path.insert(0, str(ROOT / "src"))
    from PySide6.QtCore import QPoint, QSize, Qt, QTimer
    from PySide6.QtTest import QSignalSpy, QTest
    from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionToolButton
    from fdm.live_experiment import built_in_materials
    from validate_workflow_redesign import WorkflowHarness, dimensions, settle

    physical = dimensions(args.size)
    logical = tuple(round(value / args.scale) for value in physical)
    profile = Path(tempfile.mkdtemp(prefix="profile-", dir=args.output))
    report = {"theme": args.theme, "physical_client": physical, "logical_client": logical,
              "scale": args.scale, "platform": os.environ["QT_QPA_PLATFORM"], "passed": False,
              "limitations": ["Synthetic camera and seeded category configuration.",
                              "Actual navigation and mouse button/menu actions.",
                              "Qt scale factor is not physical monitor migration."]}
    try:
        with WorkflowHarness(profile, theme=args.theme, size=logical) as harness:
            harness.enter()
            page = harness.select(1)
            page.session.set_materials(built_in_materials("normal")[:9])
            page.session.rename_operator("测试员")
            page._refresh_operators()
            page.ruler_choice.setCurrentIndex(1)
            page.ruler_choice.activated.emit(1)
            page._refresh_summary()
            settle()
            button = page.evidence_button
            assert button.size() == QSize(40, 22)
            assert not button.text() and not button.icon().isNull()
            option = QStyleOptionToolButton()
            button.initStyleOption(option)
            arrow = button.style().subControlRect(QStyle.ComplexControl.CC_ToolButton,
                option, QStyle.SubControl.SC_ToolButtonMenu, button)
            assert arrow.left() >= 26 and arrow.width() >= 10
            before = page.category_scroll.geometry()
            visible = []
            viewport = page.category_scroll.viewport()
            for row in page.material_rows.values():
                selector = row["select"]
                point = selector.mapTo(viewport, QPoint())
                if viewport.rect().contains(point) and viewport.rect().contains(point + selector.rect().bottomRight()):
                    visible.append(selector.text())
            if physical == (1600, 900) and args.scale in (1, 1.25):
                assert len(visible) >= (6 if args.scale == 1 else 4)
            for control in (page.save_button, page.export_button):
                assert control.isVisible()
                assert page.trial_sidebar.visibleRegion().contains(control.mapTo(page.trial_sidebar, control.rect().center()))
            harness.window.grab().save(str(args.output / "trial.png"))
            button.grab().save(str(args.output / "album-button.png"))
            shown = QSignalSpy(button.menu().aboutToShow)
            QTimer.singleShot(30, button.menu().close)
            QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=arrow.center())
            assert shown.count() == 1
            assert getattr(harness.window, "_evidence_gallery", None) is None
            QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=QPoint(13, 10))
            settle()
            gallery = harness.window._evidence_gallery
            assert gallery.dialog.isVisible() and not gallery.dialog.isModal()
            assert page.category_scroll.geometry() == before
            gallery.dialog.close()
            harness.assert_size()
            assert not harness.slot_errors
            report.update(passed=True, visible_categories=len(visible),
                button_size=[button.width(), button.height()],
                menu_rectangle=[arrow.x(), arrow.y(), arrow.width(), arrow.height()],
                category_rectangle=[before.x(), before.y(), before.width(), before.height()],
                outcome_width=page.measurement_outcome.width(),
                dpr=button.devicePixelRatioF(), font=harness.font_report())
    finally:
        report["source_fingerprints"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "src/fdm/ui/trial_assistance.py",
                         ROOT / "src/fdm/ui/live_experiment_view.py")}
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
