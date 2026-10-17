"""Measure the Windows callback-to-subscriber path with the native SDK test DLL.

This measures software transport after AFTER_PROCESS, not sensor exposure,
vendor processing, screen scan-out, or real Microview throughput.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QWidget

from fdm.services.capture import CaptureDevice, CaptureSessionManager, MicroviewIsolatedBackend
from fdm.services.microview_transport import FrameReceiver, load_bridge


def instrumented_helper(argv):
    """Probe-only hooks; the shipping helper/protocol and pixel path stay intact."""
    import ctypes
    from fdm import microview_helper
    from fdm.services import microview_transport

    ack_arrivals = {}
    command_waits = []
    original_queue = microview_helper.queue.Queue

    class CommandQueue(original_queue):
        def get(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                return super().get(*args, **kwargs)
            finally:
                command_waits.append((time.perf_counter() - started) * 1000)

        def put(self, item, *args, **kwargs):
            if isinstance(item, dict) and item.get("command") == "ack":
                ack_arrivals[int(item["sequence"])] = (
                    time.perf_counter(), float(item.get("_probe_ack_sent_at", 0)))
            return super().put(item, *args, **kwargs)

    class Tap:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.previous_poll = time.perf_counter()

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def poll_into(self, destination):
            started = time.perf_counter()
            info = self.wrapped.poll_into(destination)
            finished = time.perf_counter()
            if info is not None:
                info["_transport_probe"] = {"poll_started": started, "poll_finished": finished,
                    "helper_poll_gap_ms": (started - self.previous_poll) * 1000}
            self.previous_poll = started
            return info

    class Publisher(microview_transport.FramePublisher):
        def __init__(self, backend, emit):
            self.sent = {}
            self.last_ack = {"previous_ack_pending_ms": 0, "ack_gui_to_reader_ms": 0,
                             "ack_reader_to_helper_ms": 0}
            self.previous_emit_ms = 0
            self.pending_checks = 0
            backend._frame_tap = Tap(backend._frame_tap)
            metric = backend._dll.MV_TestMetric
            metric.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            metric.restype = ctypes.c_double

            def traced_emit(message):
                if message.get("type") != "frame":
                    emit(message)
                    return
                probe = message["_transport_probe"]
                probe["command_wait_samples"] = command_waits.copy()
                command_waits.clear()
                probe.update(self.last_ack)
                probe.update(previous_emit_ms=self.previous_emit_ms,
                             pending_checks=self.pending_checks,
                             source_work_ms=metric(backend._device_handle, 0),
                             source_schedule_late_ms=metric(backend._device_handle, 1),
                             source_completed=metric(backend._device_handle, 2),
                             source_missed_periods=metric(backend._device_handle, 3),
                             source_fixed_period=bool(metric(backend._device_handle, 4)),
                             source_reuse_processed=bool(metric(backend._device_handle, 5)))
                started = time.perf_counter()
                probe["publish_started"] = started
                emit(message)
                self.previous_emit_ms = (time.perf_counter() - started) * 1000
                self.sent[int(message["sequence"])] = started
                self.pending_checks = 0

            super().__init__(backend, traced_emit)

        def publish(self):
            if self.pending is not None:
                self.pending_checks += 1
            return super().publish()

        def acknowledge(self, sequence):
            processed = time.perf_counter()
            received, ack_sent = ack_arrivals.pop(int(sequence))
            sent = self.sent.pop(int(sequence))
            super().acknowledge(sequence)
            self.last_ack = {"previous_ack_pending_ms": (processed - sent) * 1000,
                             "ack_gui_to_reader_ms": (received - ack_sent) * 1000,
                             "ack_reader_to_helper_ms": (processed - received) * 1000}

    microview_helper.queue.Queue = CommandQueue
    microview_transport.FramePublisher = Publisher
    return microview_helper.main(argv)


def distribution(values):
    data = np.asarray(values, dtype=float)
    return {"count": len(data), "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)), "p99": float(np.percentile(data, 99)),
            "max": float(data.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--source-mode", choices=("fixed", "legacy"), default="fixed",
                        help="fixed: absolute 25 Hz ticks and reused processed test image; legacy: work + sleep")
    args = parser.parse_args()
    if sys.platform != "win32" or not args.sdk.is_file():
        parser.error("Windows and the compiled tests/native/microview_test_sdk.cpp DLL are required")
    if args.samples < 30:
        parser.error("At least 30 samples are required")
    os.environ["FDM_MICROVIEW_DLL_DIR"] = str(args.sdk.resolve().parent)
    os.environ["FDM_TEST_MICROVIEW_FIXED_PERIOD"] = "1" if args.source_mode == "fixed" else "0"
    os.environ["FDM_TEST_MICROVIEW_REUSE_PROCESSED"] = "1" if args.source_mode == "fixed" else "0"
    app = QApplication.instance() or QApplication([])
    backend = MicroviewIsolatedBackend()
    manager = CaptureSessionManager(backends=[backend], refresh_on_init=False)
    device = CaptureDevice("microview:0", "Native test DLL", "microview", 0)
    manager._devices = [device]
    manager._selected_device_id = device.id
    # The SDK host stays alive even when the trial is the presentation owner.
    target = QWidget()
    target.resize(768, 576)
    target.winId()
    presentation = QWidget()
    presentation.setWindowTitle("Microview transport probe — no sensor or rendering measurement")
    presentation.resize(1024, 720)
    presentation.show()
    errors, observations = [], []
    manager.errorOccurred.connect(errors.append)
    manager.set_native_preview_visible(False)
    subscription = None
    waiting = None
    stages = {}
    command_wait_samples = []
    original_receive = FrameReceiver.receive
    original_deliver = manager._deliver_frame_threadsafe
    original_send = backend._send_preview_command
    from fdm.services import capture
    original_helper = capture._microview_helper_command

    def helper(command, **kwargs):
        program, arguments = original_helper(command, **kwargs)
        if command == "preview":
            arguments = [str(Path(__file__).resolve()), "--instrument-helper",
                         *arguments[arguments.index("preview"):]]
        return program, arguments

    def timed_send(message):
        if message.get("command") == "ack":
            message = {**message, "_probe_ack_sent_at": time.perf_counter()}
        return original_send(message)

    capture._microview_helper_command = helper
    backend._send_preview_command = timed_send

    def timed_decode(receiver, message):
        started = time.perf_counter()
        pixels = original_receive(receiver, message)
        stages[int(message["index"])] = {
            "native_sequence": int(message["sequence"]),
            "callback_to_decode_start_ms": (started - float(message["received_at"])) * 1000,
            "decode_ms": (time.perf_counter() - started) * 1000,
        }
        probe = message["_transport_probe"]
        command_wait_samples.extend(probe["command_wait_samples"])
        stages[int(message["index"])].update({
            "callback_to_shared_ready_ms": (probe["poll_finished"] - float(message["received_at"])) * 1000,
            "native_to_shared_copy_ms": (probe["poll_finished"] - probe["poll_started"]) * 1000,
            "pipe_and_gui_wakeup_ms": (started - probe["publish_started"]) * 1000,
            **{key: value for key, value in probe.items() if key.endswith("_ms")},
            "pending_checks": probe["pending_checks"],
            "source_completed": probe["source_completed"],
            "source_missed_periods": probe["source_missed_periods"],
            "source_fixed_period": probe["source_fixed_period"],
            "source_reuse_processed": probe["source_reuse_processed"],
        })
        return pixels

    def timed_deliver(generation, frame):
        started = time.perf_counter()
        original_deliver(generation, frame)
        stages[int(frame.metadata["callback_index"])].update(
            manager_copy_ms=(time.perf_counter() - started) * 1000,
            manager_enqueued=time.perf_counter())

    FrameReceiver.receive = timed_decode
    manager._deliver_frame_threadsafe = timed_deliver

    def receive(frame):
        delivered = time.perf_counter()
        detail = stages[int(frame.metadata["callback_index"])].copy()
        detail["manager_queue_ms"] = (delivered - detail.pop("manager_enqueued")) * 1000
        observations.append({"sequence": frame.sequence,
                             "sdk_index": int(frame.metadata["callback_index"]),
                             "received": frame.received_at,
                             "delivered": delivered,
                             "callback_stage": frame.metadata["sdk_callback_stage"],
                             **detail})
        if waiting is not None and waiting[0]():
            waiting[1].quit()

    def wait_for(predicate, timeout=30):
        nonlocal waiting
        if not predicate() and not errors:
            loop = QEventLoop()
            timer = QTimer(loop)
            timer.setSingleShot(True)
            timer.timeout.connect(loop.quit)
            waiting = predicate, loop
            timer.start(int(timeout * 1000))
            loop.exec()
            timer.stop()
            waiting = None
        if errors or not predicate():
            raise RuntimeError(f"Transport probe failed: {errors or 'timeout'}")

    try:
        if not manager.start_preview(preview_target=target):
            raise RuntimeError("Native test source did not start")
        subscription = manager.acquire_session("transport-probe", receive, lambda _: None)
        wait_for(lambda: len(observations) >= 5)
        warmup = len(observations)
        wait_for(lambda: len(observations) >= warmup + args.samples)
        steady = observations[warmup:warmup + args.samples]
        before_pause = len(observations)
        pause_start = time.perf_counter()
        time.sleep(.250)  # Deliberately stop GUI dispatch, not the helper or SDK.
        pause_end = time.perf_counter()
        receiver = backend._frame_receiver
        newest_sequence_at_resume = receiver._read_latest_marker(receiver._latest_marker)
        wait_for(lambda: len(observations) >= before_pause + 10)
        recovery = observations[before_pause:before_pause + 10]
        ages = [(sample["delivered"] - sample["received"]) * 1000 for sample in steady]
        intervals = np.diff([sample["delivered"] for sample in steady]) * 1000
        elapsed = steady[-1]["delivered"] - steady[0]["delivered"]
        source_count = steady[-1]["sdk_index"] - steady[0]["sdk_index"]
        source_elapsed = steady[-1]["received"] - steady[0]["received"]
        report = {
            "scope": "Native SDK test DLL AFTER_PROCESS to manager subscriber; no real camera or painting",
            "platform": platform.platform(), "python": sys.version,
            "native_build": load_bridge().build_info(),
            "resolution": [768, 576], "test_sdk_requested_fps": 25,
            "source_mode": args.source_mode,
            "source_callback_window_seconds": source_elapsed,
            "source_callback_count_between_endpoints": source_count,
            "source_fixed_period": steady[-1]["source_fixed_period"],
            "source_reuse_processed": steady[-1]["source_reuse_processed"],
            "source_missed_periods": steady[-1]["source_missed_periods"] - steady[0]["source_missed_periods"],
            "native_host_visible": target.isVisible(), "warmup": warmup,
            "event_dispatch": "QEventLoop with visible separate presentation window",
            "helper_wait_strategy": "native frame/control auto-reset event; queue get_nowait",
            "helper_queue_get_timeout_requested_ms": 0,
            "helper_queue_get_actual_ms": distribution(command_wait_samples),
            "callback_to_subscriber_ms": distribution(ages),
            "stages_ms": {key: distribution([sample[key] for sample in steady])
                          for key in steady[0] if key.endswith("_ms")},
            "delivery_interval_ms": distribution(intervals),
            "effective_callback_fps": source_count / source_elapsed,
            "delivered_fps": (len(steady) - 1) / elapsed,
            "delivered_fraction": (len(steady) - 1) / source_count,
            "first_half_age_ms": distribution(ages[:len(ages) // 2]),
            "second_half_age_ms": distribution(ages[len(ages) // 2:]),
            "forced_gui_pause_ms": (pause_end - pause_start) * 1000,
            "newest_sequence_at_resume": newest_sequence_at_resume,
            "first_recovery_manager_sequence": recovery[0]["sequence"],
            "first_recovery_sequence": recovery[0]["native_sequence"],
            "first_recovery_is_current_or_newer": recovery[0]["native_sequence"] >= newest_sequence_at_resume,
            "recovery": [{"sdk_index": sample["sdk_index"],
                          "age_ms": (sample["delivered"] - sample["received"]) * 1000,
                          "after_resume_ms": (sample["delivered"] - pause_end) * 1000}
                         for sample in recovery],
            "all_after_process": all(sample["callback_stage"] == "after_process"
                                     for sample in observations),
            "manager_stats": manager.preview_frame_stats(), "errors": errors,
            "superseded_shared_slots": backend._frame_receiver.superseded_frames,
        }
        if not report["all_after_process"] or report["manager_stats"]["pending_wakeups"] > 1:
            raise AssertionError("Unexpected callback stage or unbounded wakeups")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if subscription is not None:
            subscription.close()
        manager.stop_preview()
        FrameReceiver.receive = original_receive
        capture._microview_helper_command = original_helper
        target.close()
        presentation.close()
        app.processEvents()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--instrument-helper":
        raise SystemExit(instrumented_helper(sys.argv[2:]))
    main()
