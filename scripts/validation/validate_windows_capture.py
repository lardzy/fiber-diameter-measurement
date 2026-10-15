#!/usr/bin/env python3
"""Interactive-desktop WGC validation against a controlled native child HWND.

The probe creates its own animated test window and occluder, captures only that
window, and closes both on completion. It never drives or patches CU software.
Run with the Windows desktop session active; this is not a headless test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QApplication, QWidget


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("wgc", "compat"), default="wgc")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Run on an interactive Windows desktop")
    from fdm._native import _windows_capture as native
    application = QApplication([])
    application.setQuitOnLastWindowClosed(False)
    records, failures = [], []
    state = {"phase": "initial", "changed": time.perf_counter()}
    stop = threading.Event()

    class Pattern(QWidget):
        def __init__(self, parent):
            super().__init__(parent)
            self.tick = 0
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.animate)
            self.timer.start(40)

        def animate(self):
            self.tick += 1
            self.update()

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.fillRect(0, 0, self.width() // 2, self.height(), QColor(210, 40, 80))
            painter.fillRect(self.width() // 2, 0, self.width(), self.height(), QColor(30, 190, 70))
            painter.fillRect(self.tick % max(1, self.width() - 20), 0, 20, 12, QColor(20, 90, 230))

    root = QWidget()
    root.setWindowTitle("FDM controlled WGC validation")
    root.setStyleSheet("background-color: rgb(45, 45, 45)")
    root.setGeometry(80, 80, 460, 370)
    child = Pattern(root)
    child.setGeometry(37, 53, 320, 240)
    root.show()
    root_handle, child_handle = int(root.winId()), int(child.winId())
    occluder = QWidget()
    occluder.setWindowTitle("FDM controlled occluder")
    occluder.setStyleSheet("background-color: rgb(10, 10, 240)")

    def collect():
        capture = None
        session = 0
        def open_capture():
            if args.backend == "compat":
                from fdm.live_experiment.window_capture_helper import PrintWindowCaptureClient
                return PrintWindowCaptureClient(root_handle, child_handle)
            return native.WindowCapture(root_handle, child_handle)
        try:
            capture = open_capture()
            records.append({"event": "identity", "value": dict(capture.identity())})
            while not stop.is_set():
                phase = state["phase"]
                if capture is None:
                    if phase == "minimized":
                        time.sleep(.05)
                        continue
                    capture = open_capture()
                    capture.identity()
                    session += 1
                started = time.perf_counter()
                try:
                    packet = capture.poll(100)
                except Exception as exc:
                    phase = state["phase"]
                    records.append({"event": "unavailable", "phase": phase, "message": str(exc)})
                    if phase not in ("minimized", "closed"):
                        raise
                    if phase == "closed":
                        break
                    if args.backend == "compat":
                        capture.close()
                        capture = None
                    # Stay on the original session after restore, so the probe
                    # also exercises recovery without reusing a stale frame.
                    time.sleep(.05)
                    continue
                if packet is None:
                    continue
                pixels = packet["pixels"]
                height, width = pixels.shape[:2]
                # Ignore the animated top stripe. Physical DPI scaling may
                # change dimensions, so sample fractional interior positions.
                left = pixels[height // 2, width // 4, :3].tolist()
                right = pixels[height // 2, width * 3 // 4, :3].tolist()
                budget = {key: packet[key] for key in ("gpu_pool_buffers", "gpu_pool_bytes",
                    "cpu_staging_row_pitch", "cpu_staging_bytes", "next_delivery_bytes",
                    "acquisition_reserved_bytes") if key in packet}
                records.append({"event": "frame", "phase": phase,
                    "settled": started - state["changed"] > .3,
                    "sequence": packet["sequence"], "generation": packet["generation"],
                    "capture_session": session,
                    "timestamp_100ns": packet["source_timestamp_100ns"],
                    "shape": list(pixels.shape), "crop": list(packet["crop"]), "dpi": packet["dpi"],
                    "left_bgr": left, "right_bgr": right,
                    "correct_pattern": bool(np.all(pixels[height // 2:, :width // 2, :3] == [80, 40, 210])
                        and np.all(pixels[height // 2:, width // 2:, :3] == [70, 190, 30])),
                    "buffer_budget": budget,
                    "poll_ms": (time.perf_counter() - started) * 1000})
        except Exception:
            failures.append(traceback.format_exc())
        finally:
            if capture is not None:
                capture.close()

    collector = threading.Thread(target=collect, name="fdm-wgc-validation", daemon=True)

    def phase(name, operation):
        state.update(phase=name, changed=time.perf_counter())
        operation()

    def cover():
        occluder.setGeometry(root.frameGeometry())
        occluder.show()
        occluder.raise_()

    def resize():
        occluder.hide()
        root.resize(550, 440)
        child.setGeometry(61, 83, 360, 270)

    def close_window():
        root.close()
        root.destroy()

    QTimer.singleShot(300, collector.start)
    QTimer.singleShot(1800, lambda: phase("occluded", cover))
    QTimer.singleShot(3400, lambda: phase("resized", resize))
    QTimer.singleShot(5000, lambda: phase("moved", lambda: root.move(180, 130)))
    QTimer.singleShot(6400, lambda: phase("minimized", root.showMinimized))
    QTimer.singleShot(7200, lambda: phase("restored", root.showNormal))
    QTimer.singleShot(8600, lambda: phase("closed", close_window))
    QTimer.singleShot(9300, application.quit)
    application.exec()
    stop.set()
    collector.join(4)
    root.close()
    occluder.close()
    frames = [row for row in records if row["event"] == "frame"]
    phases = {}
    for name in ("initial", "occluded", "resized", "moved", "restored"):
        subset = [row for row in frames if row["phase"] == name and row["settled"]]
        phases[name] = {"settled_frames": len(subset), "correct": sum(row["correct_pattern"] for row in subset),
                        "shapes": sorted({tuple(row["shape"]) for row in subset}),
                        "generations": sorted({row["generation"] for row in subset})}
    checks = {"worker_stopped": not collector.is_alive(), "no_unexpected_exception": not failures,
              "phases_correct": all(row["settled_frames"] >= 3 and row["correct"] == row["settled_frames"] for row in phases.values()),
              "timestamps_increase": all(a["timestamp_100ns"] < b["timestamp_100ns"] for a, b in zip(frames, frames[1:]) if a["capture_session"] == b["capture_session"]),
              "sequence_increases": all(a["sequence"] < b["sequence"] for a, b in zip(frames, frames[1:]) if a["capture_session"] == b["capture_session"]),
              "minimize_unavailable": any(row["event"] == "unavailable" and row["phase"] == "minimized" for row in records),
              "close_unavailable": any(row["event"] == "unavailable" and row["phase"] == "closed" for row in records),
              "resize_changes_generation": bool(phases["initial"]["generations"] and phases["resized"]["generations"] and phases["initial"]["generations"] != phases["resized"]["generations"])}
    if args.backend == "wgc":
        checks["bgr_full_resolution_and_accounted_buffers"] = bool(frames) and all(
            row["shape"][2] == 3 and row["buffer_budget"].get("gpu_pool_buffers") == 1
            and row["buffer_budget"].get("next_delivery_bytes") == int(np.prod(row["shape"]))
            and row["buffer_budget"].get("cpu_staging_bytes") == row["buffer_budget"].get("cpu_staging_row_pitch", 0) * row["shape"][0]
            and row["buffer_budget"].get("acquisition_reserved_bytes") == sum(row["buffer_budget"].get(key, 0)
                for key in ("gpu_pool_bytes", "cpu_staging_bytes", "next_delivery_bytes"))
            for row in frames)
    report = {"schema_version": 1, "passed": all(checks.values()), "checks": checks,
              "platform": platform.platform(), "python": sys.version, "backend": args.backend, "native_build_info": native.build_info(),
              "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
              "phases": phases, "failures": failures, "records": records,
              "limits": ["Controlled HWND render capture, not a physical CU camera or Microview board acceptance.",
                         "DPI change across physical monitors is not covered by this single VM desktop probe."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks, "phases": phases, "output": str(args.output)}, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
