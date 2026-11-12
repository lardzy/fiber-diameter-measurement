"""Source UI/latency validation with deliberately slow, hardware-free cameras.

Uses the real Qt event loop, MainWindow, capture manager and workspace. All
preferences and screenshots stay in --output. A fake Excel transport validates
controls/layout here; verify_excel_presentation.py exercises desktop Excel.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme", choices=("light", "dark"), default="dark")
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--roundtrips", type=int, default=30)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("QT_QPA_PLATFORM", "windows" if sys.platform == "win32" else "offscreen")
    for key in ("QT_SCALE_FACTOR", "QT_SCREEN_SCALE_FACTORS", "QT_FONT_DPI"):
        os.environ.pop(key, None)
    baseline = float(subprocess.check_output([sys.executable, "-c",
        "from PySide6.QtWidgets import QApplication; a=QApplication([]); print(a.primaryScreen().devicePixelRatio())"], text=True).strip())
    os.environ["QT_SCALE_FACTOR"] = str(args.scale / baseline)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from PySide6.QtCore import QObject, QEvent, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from fdm.ui import live_excel
    from fdm.ui.live_experiment_workspace import LiveExperimentWorkspace
    from validate_workflow_redesign import WorkflowHarness, FakeCaptureBackend, synthetic_frame, source_fingerprints

    report = {"passed": False, "platform": sys.platform, "theme": args.theme,
              "scale": args.scale, "native_excel": False, "physical_camera": False,
              "source": source_fingerprints(), "checks": [], "timings": [], "screenshots": []}

    class SlowBackend(FakeCaptureBackend):
        enumerate_in_worker = True
        lifecycle_in_worker = True

        def __init__(self):
            super().__init__(kind="frame_stream")
            self.delay = .65
            self.probes = 0
            self.inflight = 0
            self.max_inflight = 0
            self.threads = []
            self._lock = threading.Lock()

        def _begin(self):
            with self._lock:
                self.inflight += 1
                self.max_inflight = max(self.max_inflight, self.inflight)
                self.threads.append(threading.get_ident())
            time.sleep(self.delay)

        def _end(self):
            with self._lock:
                self.inflight -= 1

        def list_devices(self):
            self._begin()
            try:
                self.probes += 1
                return super().list_devices()
            finally:
                self._end()

        def start_preview(self, device, **kwargs):
            self._begin()
            try:
                super().start_preview(device, **kwargs)
                kwargs["frame_callback"](synthetic_frame(sequence=self.starts))
            finally:
                self._end()

        def stop_preview(self):
            super().stop_preview()

    class ExcelTransport:
        def request(self, operation, sid, generation, revision, payload=None, **kwargs):
            return {"ok": True, "status": "synced", "saved": False, "save_state": "never_saved",
                    "input_written": operation in ("open", "sync"), "name": "界面验证工作簿",
                    "presentation_status": "shown" if operation == "show" else None}

        def close(self):
            pass

    def check(value, text):
        if not value:
            raise AssertionError(text)
        report["checks"].append(text)

    def wait(predicate, text, timeout=8):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            QTest.qWait(5)
            if predicate():
                return
        raise AssertionError("Timeout: " + text)

    class Paints(QObject):
        def __init__(self):
            super().__init__()
            self.started = None
            self.first = None

        def eventFilter(self, obj, event):
            if (self.started is not None and self.first is None and event.type() == QEvent.Type.Paint
                    and type(obj).__name__ == "AcquisitionWorkbench"):
                self.first = (time.perf_counter() - self.started) * 1000
            return False

    template = output / "ui-only.xlt"
    template.write_bytes(b"No desktop Excel opens this mock template")
    backend = SlowBackend()
    try:
        with patch.object(live_excel, "PowerShellLiveTransport", ExcelTransport), \
                patch.object(live_excel, "working_template", lambda **_: template), \
                WorkflowHarness(output / "profile", theme=args.theme, backend=backend,
                    size=(max(900, round(1366 / args.scale)), max(600, round(850 / args.scale)))) as harness:
            app, window = harness.app, harness.window
            observer = Paints()
            app.installEventFilter(observer)
            beats = []
            timer = QTimer(window)
            timer.setInterval(10)
            timer.timeout.connect(lambda: beats.append(time.perf_counter()))
            timer.start()
            try:
                observer.started = time.perf_counter()
                hub = window._open_acquisition_workbench(trial_index=0, auto_start=True)
                wait(lambda: observer.first is not None, "first framework paint")
                report["first_shell_ms"] = observer.first
                first_at = len(beats)
                wait(window._capture_manager.is_preview_active, "slow asynchronous camera open")
                wait(lambda: backend.inflight == 0, "camera worker settles")
                segment = beats[first_at:]
                gaps = [(b - a) * 1000 for a, b in zip(segment, segment[1:])]
                report["slow_camera"] = {"heartbeat_samples": len(segment), "maximum_heartbeat_gap_ms": max(gaps, default=0),
                                         "stage_timings": window._capture_manager.device_operation_timings()}
                check(len(segment) >= 20, "GUI timer continues while camera discovery and open each wait 650 ms")
                check(max(gaps, default=0) < 200, "camera waits do not block GUI for 200 ms")
                check(threading.get_ident() not in backend.threads, "device probe/open execute away from GUI thread")

                # Delay stage advancement without blocking Qt. This validates
                # the delayed spinner and navigation during real page creation.
                original = LiveExperimentWorkspace.initialize_next_stage
                gate_until = time.perf_counter() + .30

                def staged(page):
                    return False if time.perf_counter() < gate_until else original(page)

                with patch.object(LiveExperimentWorkspace, "initialize_next_stage", staged):
                    hub.select_page(1)
                    wait(lambda: hub.loading_progress.isVisible(), "delayed page spinner")
                    shot = output / "content-loading.png"
                    window.grab().save(str(shot))
                    report["screenshots"].append(shot.name)
                    wait(lambda: hub._page_load_state == "ready" and hub._current_trial() is not None, "content page ready")
                page = hub._current_trial()
                backend.tick()
                QTest.qWait(20)
                wait(lambda: page.live_excel.status(page.session.id)["connected"], "mock Excel connected")
                page.live_excel_controls._refresh(page.session.id)
                shot = output / "content-ready.png"
                window.grab().save(str(shot))
                report["screenshots"].append(shot.name)

                # Errors retain their leading text and semantic colour inside
                # the actual trial stylesheet, not just an isolated label.
                link = page.live_excel._links[page.session.id]
                link.template_replacement_failure = {"code": "template_mapping", "message": "验证模板更换失败"}
                page.live_excel_controls._refresh(page.session.id)
                check(page.live_excel_controls.status_label.property("statusTone") == "error", "template failure has error tone in real trial page")
                shot = output / "excel-error.png"
                window.grab().save(str(shot))
                report["screenshots"].append(shot.name)
                link.template_replacement_failure = None
                page.live_excel_controls._refresh(page.session.id)
                check(page.live_excel_controls.excel_button.property("emphasis") == "primary", "Excel main action uses primary emphasis")
                check(not page.live_excel_controls.excel_button.icon().isNull(), "Excel action has a vector icon")

                # Ready pages are reused on repeated workspace entry. Delayed
                # camera tasks are cancelled and never allowed to overlap.
                samples = []
                for _ in range(args.roundtrips):
                    window._activate_measure_workspace()
                    observer.first = None
                    observer.started = time.perf_counter()
                    window._open_acquisition_workbench(trial_index=1, auto_start=False)
                    wait(lambda: observer.first is not None, "return shell paint")
                    samples.append(observer.first)
                report["warm_shell"] = {"count": len(samples), "p50_ms": statistics.median(samples),
                    "p95_ms": sorted(samples)[math.ceil(.95 * len(samples)) - 1], "max_ms": max(samples)}
                report["timings"] = hub.transition_timings()
                check(hub._current_trial() is page, "reentry reuses existing trial page")

                # Begin an uncached page then leave before publication.
                hub.select_page(2)
                window._activate_measure_workspace()
                QTest.qWait(500)
                check(not window._acquisition_workspace_selected, "late page preparation does not return to acquisition")
                check(backend.max_inflight == 1, "device jobs remain serial")
                report["actual_dpr"] = window.devicePixelRatioF()
                check(abs(report["actual_dpr"] - args.scale) < .015, "requested physical display scaling is active")
                check(not harness.slot_errors, "no Qt slot exceptions")
                check(not harness.modal_messages, "no blocking error dialogs")
                report["passed"] = True
            finally:
                timer.stop()
                app.removeEventFilter(observer)
    except Exception as exc:
        import traceback
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(output), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
