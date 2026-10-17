"""Real-route second-round visual regression using an isolated synthetic camera.

Run one theme/physical-client/DPI case per process. Binary evidence and profiles
default to ignored .tmp; pass --output to relocate them. This is presentation
evidence, not a Windows/Microview or measurement-algorithm acceptance test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=("dark", "light", "navy", "warm"), default="dark")
    parser.add_argument("--size", default="1366x768", help="physical client raster WIDTHxHEIGHT")
    parser.add_argument("--scale", type=float, choices=(1, 1.25, 1.5, 2), default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--platform", default="windows" if sys.platform == "win32" else "offscreen")
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--motion-cycles", type=int, default=100)
    parser.add_argument("--sidebar-width", type=int, choices=range(390, 521), default=432)
    parser.add_argument("--video", action="store_true", help="record real Qt changes with ffmpeg")
    return parser.parse_args()


def fingerprints():
    from validate_workflow_redesign import source_fingerprints
    result = source_fingerprints()
    for name in ("src/fdm/ui/acquisition_layout.py", "src/fdm/ui/quantity_targets.py",
                 "src/fdm/ui/measurement_feedback_overlay.py", "src/fdm/live_experiment/feedback.py",
                 "src/fdm/live_experiment/reminders.py", "src/fdm/live_experiment/models.py"):
        result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return result


def rect(widget, window):
    from PySide6.QtCore import QPoint
    point = widget.mapTo(window, QPoint())
    return [point.x(), point.y(), widget.width(), widget.height()]


def main(args):
    os.environ["QT_QPA_PLATFORM"] = args.platform
    os.environ["QT_SCALE_FACTOR"] = str(args.scale)
    sys.path.insert(0, str(ROOT / "src"))
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPushButton, QScrollArea
    from fdm.live_experiment.reminders import ReminderPolicy
    from validate_workflow_redesign import WorkflowHarness, dimensions, settle

    physical = dimensions(args.size)
    logical = tuple(round(value / args.scale) for value in physical)
    key = f"{args.theme}-{args.size}-{str(args.scale).replace('.', 'p')}"
    if args.sidebar_width != 432:
        key += f"-sidebar{args.sidebar_width}"
    output = (args.output or ROOT / ".tmp/acquisition-second-round-20261017" / key).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="profile-", dir=output))
    began = time.perf_counter()
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "theme": args.theme, "requested_raster": list(physical), "requested_logical": list(logical),
              "requested_scale": args.scale, "requested_sidebar_width": args.sidebar_width, "platform": args.platform,
              "runtime": {"system": platform.platform(), "python": sys.version.split()[0]},
              "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_fingerprints": fingerprints(), "screenshots": [], "geometry": [],
              "motion": {}, "transitions": {}, "feedback_history": {}, "video": None, "passed": False,
              "real_hardware": False, "real_windows": sys.platform == "win32",
              "limitations": ["Real MainWindow/navigation with isolated synthetic capture frames; no SDK or physical device.",
                  "Failure reason is injected at the current controller rejection boundary; numerical rejection is tested separately.",
                  "Accepted quantity fixtures use the domain ledger and normal controller publication signals.",
                  "Motion test exercises field retirement and status delivery; it is not detector accuracy or throughput testing.",
                  "QT_SCALE_FACTOR checks Qt layout/raster scaling, not mixed-monitor OS DPI switching."]}
    harness = None
    recorder = {"process": None, "timer": None, "frames": 0, "timeline": [], "errors": [], "timestamps": []}
    log = None
    try:
        with WorkflowHarness(profile, theme=args.theme, size=logical) as harness:
            window = harness.window
            report["font"] = harness.font_report()
            report["actual_dpr"] = window.devicePixelRatioF()
            screen = window.screen()
            report["screen"] = {"geometry": screen.geometry().getRect(),
                                "available_geometry": screen.availableGeometry().getRect(),
                                "logical_dpi": screen.logicalDotsPerInch(), "physical_dpi": screen.physicalDotsPerInch()}

            if args.video:
                ffmpeg = shutil.which("ffmpeg")
                if ffmpeg is None:
                    raise RuntimeError("--video requires ffmpeg on PATH")
                raster = window.grab().size()
                video_path = output / "process.mp4"
                log = (output / "ffmpeg.log").open("w", encoding="utf-8")
                recorder["process"] = subprocess.Popen([ffmpeg, "-y", "-loglevel", "warning",
                    "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", f"{raster.width()}x{raster.height()}",
                    "-framerate", "15", "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                timer = QTimer(window)
                timer.setInterval(67)
                recorder["timer"] = timer

                def record_frame():
                    try:
                        harness.assert_size()
                        image = window.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
                        assert image.size() == raster, "video client changed size"
                        recorder["process"].stdin.write(image.constBits())
                        recorder["frames"] += 1
                        recorder["timestamps"].append(round(time.perf_counter() - began, 4))
                    except Exception as error:
                        recorder["errors"].append(repr(error))
                        timer.stop()

                timer.timeout.connect(record_frame)
                timer.start()

            def mark(label):
                recorder["timeline"].append({"label": label, "frame": recorder["frames"],
                    "wall_seconds": round(time.perf_counter() - began, 4)})

            def geometry(label, page=None):
                harness.assert_size()
                harness.assert_navigation()
                image = page.image_scroll if page is not None else window._preview_display_stack
                sidebar = page.trial_sidebar if page is not None else window._acquisition_right_panel
                row = {"label": label, "client": [window.width(), window.height()],
                       "image": rect(image, window), "sidebar": rect(sidebar, window),
                       "navigation": rect(harness.hub.navigation, window)}
                if page is not None:
                    for name in ("workflow_tabs", "sample", "save_button", "live_state_label", "view_controls"):
                        row[name] = rect(getattr(page, name), window)
                    row["readiness_visible"] = page.readiness_label.isVisible()
                    row["state"] = page.controller.state
                assert window.rect().contains(image.mapTo(window, image.rect().topLeft()))
                assert window.rect().contains(image.mapTo(window, image.rect().bottomRight()))
                assert sidebar.isVisible()
                report["geometry"].append(row)
                return row

            def check_text(label):
                required = label.heightForWidth(label.width()) if label.wordWrap() else label.sizeHint().height()
                result = {"text": label.text(), "width": label.width(), "height": label.height(),
                          "required_height": required}
                assert required <= label.height(), f"clipped text: {result}"
                return result

            def snapshot(name, page=None, *, overlay=False):
                mark(name)
                # The real summary/status presentation timer is 80 ms. Inspect
                # its settled output instead of capturing the preceding label.
                QTest.qWait(280 if args.video else 120)
                settle()
                row = geometry(name, page)
                path = output / f"{name}.png"
                pixmap = window.grab()
                assert pixmap.save(str(path))
                item = {"name": name, "path": str(path), "raster": [pixmap.width(), pixmap.height()],
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "geometry": row}
                for index in range(harness.hub.navigation.count()):
                    assert not harness.hub.navigation.tabIcon(index).isNull()
                if page is not None:
                    assert not page.quantity_targets.settings_button.icon().isNull()
                    item["progress"] = check_text(page.quantity_targets.progress)
                    item["footer_message"] = page.statusBar().currentMessage()
                    item["feedback_visible"] = page.measurement_feedback.isVisible()
                    item["latest_value"] = page.latest_value.text()
                    item["measurement_outcome"] = page.measurement_outcome.text()
                if overlay:
                    card = page.measurement_feedback
                    assert card.isVisible() and card.parentWidget() is page.image_scroll.viewport()
                    assert card.viewport.rect().contains(card.geometry()), "feedback card exceeds viewport"
                    item["overlay"] = {"rect": rect(card, window), "code": card.feedback.code,
                        "title": check_text(card.title_label), "message": check_text(card.message_label),
                        "hint": check_text(card.hint_label), "mouse_transparent": card.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)}
                    if card.feedback.code == "no_complete_crossing":
                        value = page.latest_value
                        item["previous_value"] = {"text": value.text(), "width": value.width(),
                            "required_width": value.sizeHint().width(), "rect": rect(value, window)}
                        assert "123.456" in value.text() and "μm" in value.text()
                        assert value.sizeHint().width() <= value.width(), item["previous_value"]
                    elif card.feedback.code == "target_reached":
                        assert "已计入记录" in item["footer_message"], item["footer_message"]
                        assert "未识别到" not in item["footer_message"]
                report["screenshots"].append(item)
                return row

            def publish_fixture(page, diameter):
                record = page.session.add_measurement(diameter, "wool", field_id=page.controller.field_id,
                    frame_id=page.controller.latest_frame.frame_id, provenance={"validation_fixture": True})
                page.controller.current_records.append(record.id)
                page.controller.recordAccepted.emit(record)
                page.controller.changed.emit()
                page._refresh_summary()
                return record

            harness.enter()
            harness.select(0)
            harness.hub._sidebar_width = args.sidebar_width
            harness.hub.apply_sidebar_width()
            display_index = window._preview_display_choice.findData("fit")
            window._preview_display_choice.setCurrentIndex(display_index)
            window._preview_display_choice.activated.emit(display_index)
            assert window._preview_display_choice.currentData() == "fit"
            reference = snapshot("realtime")
            transition_rows = []
            for step in range(args.cycles):
                index = (1, 2, 0)[step % 3]
                page = harness.select(index)
                image = page.image_scroll if index else window._preview_display_stack
                current = rect(image, window)
                assert current == reference["image"], (step, index, current, reference["image"])
                if step < 3 or step >= args.cycles - 3:
                    transition_rows.append(geometry(f"cycle-{step}-{index}", page if index else None))
            assert not harness.observer.unexpected(), harness.observer.unexpected()
            harness.assert_clean()
            report["transitions"] = {"switches": args.cycles, "image_frame_identical": True,
                                     "samples": transition_rows, "backend_starts": harness.backend.starts}

            for index, mode in ((1, "content"), (2, "diameter")):
                page = harness.select(index)
                page.ruler_choice.setCurrentIndex(1)
                page.ruler_choice.activated.emit(1)
                page.session.rename_operator("视觉测试员")
                page._refresh_operators()
                page._refresh_measurement_readiness()
                page._changed()
                display_index = page.display_choice.findData("fit")
                page.display_choice.setCurrentIndex(display_index)
                page.display_choice.activated.emit(display_index)
                page._refresh_summary()
                settle()
                initial = snapshot(f"{mode}-operation", page)
                fixed = ("image", "workflow_tabs", "sample", "save_button", "sidebar")
                motion_samples = []
                mark(f"{mode}-100-field-status-changes")
                for step in range(args.motion_cycles):
                    page.controller.end_field("视场已变化，请重新测量")
                    page._refresh_measurement_readiness()
                    page.controller.statusChanged.emit("视场稳定，可以测量" if step % 2 else "正在核验视场，请稍候")
                    settle(2)
                    observed = {name: rect(getattr(page, {"image": "image_scroll", "sidebar": "trial_sidebar"}.get(name, name)), window)
                                for name in fixed}
                    assert all(observed[name] == initial[name] for name in fixed), (mode, step, observed, initial)
                    assert not page.readiness_label.isVisible()
                    if step in (0, args.motion_cycles - 1):
                        motion_samples.append({"step": step, "rects": observed})
                report["motion"][mode] = {"cycles": args.motion_cycles, "geometry_unchanged": True,
                    "presentation_only": True, "samples": motion_samples}

                # Exercise the real rejection publication path with the current
                # context; numerical edge rejection belongs to kernel tests.
                policy = ReminderPolicy(content_measured_target=2, content_total_target=3, diameter_measured_target=2)
                page._configure(reminder_policy=policy)
                publish_fixture(page, 123.456)
                before = page.session.to_dict()
                page.controller._reject("no_complete_crossing")
                snapshot(f"{mode}-invalid-measurement", page, overlay=True)
                assert page.session.to_dict() == before
                assert "本次未通过" in page.measurement_outcome.text()
                assert "no_complete_crossing" not in page.measurement_feedback.message_label.text()
                delivered = []
                page.canvas.pointPressed.connect(delivered.append)
                page.canvas.pointPressed.disconnect(page._press)
                try:
                    QTest.mouseClick(page.canvas, Qt.MouseButton.LeftButton, pos=page.canvas.rect().center())
                finally:
                    page.canvas.pointPressed.connect(page._press)
                    page.canvas.pointPressed.disconnect(delivered.append)
                assert delivered and not page.measurement_feedback.isVisible()

                publish_fixture(page, 20)
                assert page.measurement_feedback.feedback.code == "target_reached"
                assert "2 / 2" in page.quantity_targets.progress.text()
                scroll = page.workflow_tabs.currentWidget()
                if isinstance(scroll, QScrollArea):
                    scroll.ensureWidgetVisible(page.quantity_targets, 0, 8)
                snapshot(f"{mode}-quantity-target", page, overlay=True)

                # Do not pre-dismiss the target card: exercising the visible
                # undo button must invalidate obsolete feedback on its own.
                seen = set(page._reminder_tracker._seen)

                def click_history(text):
                    button = next(button for button in page.findChildren(QPushButton) if button.text() == text)
                    assert button.isVisible() and button.isEnabled()
                    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
                    settle()

                click_history("撤销")
                assert not page.measurement_feedback.isVisible()
                assert page.latest_value.text() == "123.456 μm"
                assert page.measurement_outcome.text() == "最近有效测量"
                assert "1 / 2" in page.quantity_targets.progress.text()
                snapshot(f"{mode}-target-undo", page)
                click_history("撤销")
                assert not page.session.measurements
                assert page.latest_value.text() == "— μm" and not page.measurement_outcome.text()
                assert not page.measurement_feedback.isVisible()
                assert "0 / 2" in page.quantity_targets.progress.text()
                snapshot(f"{mode}-target-undo-empty", page)
                click_history("重做")
                click_history("重做")
                assert len(page.session.measurements) == 2
                assert page.latest_value.text() == "20.000 μm"
                assert "2 / 2" in page.quantity_targets.progress.text()
                assert not page.measurement_feedback.isVisible()
                assert page._reminder_tracker._seen == seen
                snapshot(f"{mode}-target-redo", page)

                page._configure(reminder_policy=ReminderPolicy(content_measured_target=3,
                    content_total_target=4, diameter_measured_target=3))
                publish_fixture(page, 22)
                assert page.measurement_feedback.isVisible()
                assert page.measurement_feedback.feedback.code == "target_reached"
                page._configure(reminder_policy=ReminderPolicy(content_measured_target=4,
                    content_total_target=4, diameter_measured_target=4))
                assert not page.measurement_feedback.isVisible()
                assert "3 / 4" in page.quantity_targets.progress.text()
                snapshot(f"{mode}-target-changed", page)
                report["feedback_history"][mode] = {"real_undo_button": True,
                    "undo_refreshes_previous_value": True, "undo_all_clears_value": True,
                    "redo_restores_value_without_replaying_card": True,
                    "changed_target_dismisses_obsolete_card": True}
                if mode == "content":
                    def click_count(key):
                        button = page.material_rows["wool"][key]
                        if isinstance(scroll, QScrollArea):
                            scroll.ensureWidgetVisible(button, 0, 8)
                        settle()
                        assert button.isVisible() and button.isEnabled()
                        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
                        QTest.qWait(120)
                        settle()

                    click_count("add")
                    assert page.measurement_feedback.isVisible()
                    assert page.measurement_feedback.feedback.code == "target_reached"
                    assert "4 / 4" in page.quantity_targets.progress.text().splitlines()[-1]
                    snapshot("content-extra-target", page, overlay=True)
                    click_count("subtract")
                    assert not page.measurement_feedback.isVisible()
                    assert "3 / 4" in page.quantity_targets.progress.text().splitlines()[-1]
                    snapshot("content-extra-reduced", page)
                    click_count("add")
                    assert not page.measurement_feedback.isVisible()
                    click_count("subtract")
                    report["feedback_history"][mode].update(real_count_buttons=True,
                        reduced_count_dismisses_obsolete_card=True,
                        repeated_count_threshold_does_not_replay=True)
                for tab, title in ((1, "records"), (2, "report")):
                    page.workflow_tabs.setCurrentIndex(tab)
                    settle()
                    row = snapshot(f"{mode}-{title}", page)
                    assert row["image"] == initial["image"]
                page.workflow_tabs.setCurrentIndex(0)
                settle()
                row = snapshot(f"{mode}-operation-restored", page)
                assert row["image"] == initial["image"]

            harness.select(0)
            snapshot("realtime-return")
            assert not harness.slot_errors, harness.slot_errors
            assert not harness.modal_messages, harness.modal_messages
            assert not harness.observer.unexpected(), harness.observer.unexpected()
            report["show_events"] = harness.observer.events
            report["slot_errors"] = harness.slot_errors
            report["modal_messages"] = harness.modal_messages
            report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        if harness is not None:
            report["slot_errors"] = harness.slot_errors
            report["show_events"] = harness.observer.events
    finally:
        if recorder["timer"] is not None:
            from shiboken6 import isValid
            if isValid(recorder["timer"]):
                recorder["timer"].stop()
        if recorder["process"] is not None:
            process = recorder["process"]
            process.stdin.close()
            code = process.wait(timeout=30)
            log.close()
            report["video"] = {"path": str(output / "process.mp4"), "fps": 15,
                "frames": recorder["frames"], "seconds": recorder["frames"] / 15,
                "exit_code": code, "errors": recorder["errors"], "timeline": recorder["timeline"],
                "frame_wall_timestamps": recorder["timestamps"], "type": "Qt window grabs during actual navigation"}
            if code != 0 or recorder["errors"]:
                report["passed"] = False
        report["source_changed_during_run"] = fingerprints() != report["source_fingerprints"]
        report["duration_seconds"] = round(time.perf_counter() - began, 3)
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(output / "result.json"),
                      "error": report.get("error"), "source_changed_during_run": report["source_changed_during_run"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(arguments()))
