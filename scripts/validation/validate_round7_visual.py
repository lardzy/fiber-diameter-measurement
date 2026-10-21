"""Round 7 visual matrix; synthetic capture and seeded candidate geometry.

Real main navigation, canvas mouse events and the New Field button are used.
Candidate detection is intentionally seeded: this is visual/input-isolation
evidence, not algorithm, Windows-native or physical Microview acceptance.
Run every case in a fresh process because Qt reads DPI at application creation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=("light", "dark", "navy", "warm"), required=True)
    parser.add_argument("--size", required=True, help="Physical client raster, e.g. 1600x900")
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = str(args.scale)
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts/validation")]

    from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt, QTimer
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QPushButton
    from fdm.live_experiment import built_in_materials
    from fdm.ui.live_measurement_annotations import segment_intersects_rect
    from validate_workflow_redesign import WorkflowHarness, dimensions, settle

    physical = dimensions(args.size)
    logical = tuple(round(value / args.scale) for value in physical)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(theme=args.theme, requested_physical=physical, scale=args.scale,
        requested_logical=logical, platform="macOS / Qt offscreen",
        scope="Real navigation, actual canvas picks and New Field button; synthetic camera, seeded candidate geometry. No real algorithm or Windows/Microview acceptance.",
        checks={}, screenshots=[])

    def check(name, success, details=None):
        report["checks"][name] = dict(passed=bool(success), details=details)
        if not success:
            raise AssertionError(f"{name}: {details}")

    def move(canvas, position, *, held=False):
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(position),
            QPointF(canvas.mapToGlobal(position)), Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(canvas, event)

    def rect_data(rect):
        return [rect.x(), rect.y(), rect.width(), rect.height()]

    def fully_visible(widget, window):
        """Qt visibleRegion alone can include a partly clipped scroll child."""
        if not widget.isVisible():
            return False
        rect = QRect(widget.mapTo(window, QPoint()), widget.size())
        parent = widget.parentWidget()
        while parent is not None:
            boundary = QRect(parent.mapTo(window, QPoint()), parent.size())
            if not boundary.contains(rect):
                return False
            if parent is window:
                break
            parent = parent.parentWidget()
        return widget.visibleRegion().contains(widget.rect())

    try:
        with tempfile.TemporaryDirectory(prefix="fdm-round7-visual-") as profile:
            with WorkflowHarness(profile, theme=args.theme, size=logical) as h:
                h.enter()
                p = h.select(1)
                p.session.set_materials(built_in_materials("normal")[:9])
                p.session.rename_operator("测试员")
                p._refresh_operators()
                p.ruler_choice.setCurrentIndex(1)
                p.ruler_choice.activated.emit(1)
                p._refresh_summary()
                settle()
                # Ten seconds leaves enough time for screen/layout inspection.
                # This is a real preferences control, not a changed manager clock.
                p.selection_retention_seconds.setValue(10.)
                p.measure_mode.setCurrentIndex(p.measure_mode.findData("free"))
                for _ in range(8):
                    h.backend.tick()
                    QTest.qWait(50)
                deadline = time.monotonic() + 2
                while p.controller.state != "ready" and time.monotonic() < deadline:
                    QTest.qWait(5)
                c = p.controller
                check("stable_before_input", c.state == "ready", c.state)

                token = c.presented
                c._automatic = SimpleNamespace(cancel=lambda: None, poll=lambda: [], close=lambda: None)
                c._automatic_context = dict(task_id="visual-batch", batch_id="visual-batch",
                    token=token, lease=c._acquire_intent_frame(token.frame), calibration=c.calibration,
                    operator_id=p.session.active_operator_id, direction=p.session.active_direction,
                    algorithm="traditional", completed=True, diagnostics={})
                c.candidates = [dict(candidate_id=f"visual:{i}", ordinal=i + 1, state="available",
                    width_px=14, start=[243 + i * 100, 200], end=[257 + i * 100, 200]) for i in range(3)]
                c.candidatesChanged.emit(c.candidates)
                p.canvas.repaint()

                def point(x, y):
                    scale, origin = p.canvas._transform()
                    return QPoint(round(x * scale + origin[0]), round(y * scale + origin[1]))

                def snap(name, widget=None):
                    path = args.output / f"{name}.png"
                    pixmap = (widget or h.window).grab()
                    check(f"save_{name}", pixmap.save(str(path)))
                    report["screenshots"].append(dict(file=path.name, raster=[pixmap.width(), pixmap.height()],
                        device_pixel_ratio=pixmap.devicePixelRatio()))

                for x in (250, 350):
                    QTest.mouseClick(p.canvas, Qt.MouseButton.LeftButton, pos=point(x, 200))
                    settle()
                QTest.qWait(120)  # Existing deferred statistics UI update is 80 ms.
                check("canvas_picks_accepted", len(p.session.measurements) == 2)
                check("partial_accept_keeps_candidate", len(p.canvas._painted_candidates) == 1)
                snap("candidates")
                before_nav = rect_data(h.hub.navigation.geometry())
                QTest.mouseClick(next(button for button in p.findChildren(QPushButton)
                                     if button.text() == "新视场"), Qt.MouseButton.LeftButton)
                p.canvas.repaint()
                check("two_picks_retained", len(c.selection_retention.visible_marks()) == 2)
                for _ in range(6):
                    h.backend.tick()
                    QTest.qWait(50)
                check("new_field_stable", c.state == "ready", c.state)
                check("retention_survives_new_frames", len(c.selection_retention.visible_marks()) == 2)
                check("retained_not_candidate_hit", not p.canvas.candidate_hits(QPointF(point(250, 200))))
                QTest.mouseClick(p.canvas, Qt.MouseButton.LeftButton, pos=point(250, 200))
                draft = c.draft
                check("retained_click_starts_manual", draft is not None)
                for _ in range(20):
                    move(p.canvas, point(350, 200))
                p.canvas.repaint()
                check("hover_does_not_finish_manual", c.draft is draft and len(p.session.measurements) == 2)
                check("retained_has_no_hover", p.canvas._hovered_record_id is None and p.canvas.hovered_candidate_id is None)
                items, protected = p.canvas._overlay_items()
                placements = p.canvas._arrange_labels(items, protected)
                clearance = [not segment_intersects_rect(*segment,
                    placements[item["key"]].rect.adjusted(-12., -12., 12., 12.))
                    for item in items if item["kind"] == "retained" and item["key"] in placements
                    for segment in protected]
                check("retained_labels_avoid_draft", bool(clearance) and all(clearance))
                check("retained_geometry_under_draft", [item["kind"] for item in items] == ["retained", "retained"])
                report["overlay"] = dict(marks=list(c.selection_retention.visible_marks()),
                    protected_segments=protected,
                    labels={key: rect_data(value.rect) for key, value in placements.items()})
                snap("retained-draft")

                controls = {}
                for name, widget in (("save", p.save_button), ("excel", p.export_button),
                                     ("album", p.evidence_button), ("candidate_display", p.candidate_display_button)):
                    rect = QRect(widget.mapTo(h.window, QPoint()), widget.size())
                    complete = fully_visible(widget, h.window)
                    controls[name] = dict(rect=rect_data(rect), visible=complete, text=widget.text(),
                        accessible_name=widget.accessibleName(), icon_available=not widget.icon().isNull())
                    check(f"{name}_visible", complete, controls[name])
                    if name != "candidate_display":
                        check(f"{name}_icon", not widget.icon().isNull(), controls[name])
                check("album_icon_only", p.evidence_button.text() == "" and p.evidence_button.width() == 40)
                report["controls"] = controls
                report["font"] = h.font_report()
                report["actual_logical"] = [h.window.width(), h.window.height()]
                report["actual_dpr"] = h.window.devicePixelRatioF()
                report["fully_visible_categories"] = sum(
                    fully_visible(row["select"], h.window)
                    for row in p.material_rows.values())
                if args.size == "1600x900" and args.scale in (1., 1.25):
                    minimum = 6 if args.scale == 1. else 4
                    check("category_baseline", report["fully_visible_categories"] >= minimum,
                          dict(actual=report["fully_visible_categories"], minimum=minimum))
                check("navigation_fixed", rect_data(h.hub.navigation.geometry()) == before_nav)
                h.assert_navigation()
                QTest.keyClick(p.canvas, Qt.Key.Key_Escape)
                check("escape_cancels_only_draft", c.draft is None and len(c.selection_retention.visible_marks()) == 2)

                outer_scroll = p.workflow_tabs.currentWidget().verticalScrollBar()
                inner_scroll = p.category_scroll.verticalScrollBar()
                positions = (outer_scroll.value(), inner_scroll.value())
                outer_scroll.setValue(outer_scroll.maximum())
                inner_scroll.setValue(inner_scroll.maximum())
                settle()
                check("last_category_scroll_reachable", fully_visible(list(p.material_rows.values())[-1]["select"], h.window))
                check("scroll_preserves_save_excel", fully_visible(p.save_button, h.window) and fully_visible(p.export_button, h.window))
                if args.scale == 1.5:
                    snap("scroll-bottom")
                outer_scroll.setValue(positions[0])
                inner_scroll.setValue(positions[1])
                settle()

                captured_menu = []
                def capture_menu():
                    captured_menu.append(True)
                    snap("candidate-menu", p.candidate_display_menu)
                    report["menu"] = dict(size=[p.candidate_display_menu.width(), p.candidate_display_menu.height()],
                        duration=p.selection_retention_seconds.value(),
                        spin_visible=p.selection_retention_seconds.visibleRegion().contains(p.selection_retention_seconds.rect()))
                    p.candidate_display_menu.close()
                QTimer.singleShot(50, capture_menu)
                QTest.mouseClick(p.candidate_display_button, Qt.MouseButton.LeftButton)
                check("menu_opened", bool(captured_menu))
                check("duration_control_unclipped", report["menu"]["spin_visible"])
                report["window_events"] = h.observer.events
                unexpected = [event for event in h.observer.unexpected() if event["class"] != "QMenu"]
                check("no_unexpected_windows", not unexpected, unexpected)
                check("no_modal_messages", not h.modal_messages, h.modal_messages)
                check("no_slot_errors", not h.slot_errors, h.slot_errors)
                check("marks_not_project_fields", "selection_retention" not in p.session.to_dict())
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["error"] = repr(error)
        raise
    finally:
        report["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "src/fdm/ui/live_measurement_canvas.py", root / "src/fdm/ui/trial_assistance.py",
                         root / "src/fdm/live_experiment/controller.py", root / "src/fdm/live_experiment/selection_retention.py")}
        (args.output / "geometry.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "theme": args.theme, "size": args.size,
                      "scale": args.scale, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
