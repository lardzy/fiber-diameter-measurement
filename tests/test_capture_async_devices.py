"""Slow native work cannot block Qt or escape the single-device lease."""
from threading import Event, get_ident
from time import monotonic

from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.services.capture import CaptureBackend, CaptureDevice, CaptureSessionManager


def wait(predicate, timeout=2):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    raise AssertionError("camera asynchronous operation did not finish")


class ControlledBackend(CaptureBackend):
    backend_key = "controlled"
    lifecycle_in_worker = True

    def __init__(self):
        self.enumerate_gate, self.open_gate, self.stop_gate = Event(), Event(), Event()
        for event in (self.enumerate_gate, self.open_gate, self.stop_gate):
            event.set()
        self.enumerating, self.opening, self.stopping = Event(), Event(), Event()
        self.calls = []
        self.active = False

    def list_devices(self):
        self.calls.append(("enumerate", get_ident()))
        self.enumerating.set()
        assert self.enumerate_gate.wait(2)
        return [CaptureDevice("controlled:0", "Test camera", self.backend_key, 0)]

    def start_preview(self, device, *, frame_callback, error_callback, **kwargs):
        assert not self.active
        self.calls.append(("open", get_ident()))
        self.opening.set()
        assert self.open_gate.wait(2)
        self.active = True
        image = QImage(20, 12, QImage.Format.Format_RGB32)
        image.fill(0xFF123456)
        frame_callback(image)

    def stop_preview(self):
        self.calls.append(("stop", get_ident()))
        self.stopping.set()
        assert self.stop_gate.wait(2)
        self.active = False


def test_slow_probe_keeps_qt_responsive_and_completes_on_gui_thread():
    backend = ControlledBackend()
    backend.enumerate_gate.clear()
    manager = CaptureSessionManager([backend], refresh_on_init=False)
    replies, ticks = [], []
    timer = QTimer()
    timer.setInterval(1)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    manager.refresh_devices_async(lambda devices: replies.append((devices, get_ident())))
    wait(backend.enumerating.is_set)
    QTest.qWait(30)
    assert ticks and not replies and backend.calls[0][1] != get_ident()
    backend.enumerate_gate.set()
    wait(lambda: bool(replies))
    timer.stop()
    assert replies[0][1] == get_ident()
    assert manager.devices()[0].id == "controlled:0"


def test_cancelled_probe_cannot_publish_devices_or_restart_camera():
    backend = ControlledBackend()
    backend.enumerate_gate.clear()
    manager = CaptureSessionManager([backend], refresh_on_init=False)
    replies = []
    manager.refresh_devices_async(lambda devices: replies.append(devices))
    wait(backend.enumerating.is_set)
    manager.cancel_pending_operations()
    backend.enumerate_gate.set()
    wait(lambda: not manager.device_operations_pending())
    assert not replies and not manager.devices() and not backend.active


def test_no_device_probe_finishes_with_faulted_state_without_modal_error_signal():
    manager = CaptureSessionManager([], refresh_on_init=False)
    replies, errors = [], []
    manager.errorOccurred.connect(errors.append)
    manager.refresh_devices_async(replies.append)
    wait(lambda: replies == [[]])
    assert manager.current_capture_state() == "faulted"
    assert manager.current_capture_stage() == "enumerate_complete"
    assert not errors and not manager.is_starting() and not manager.is_stopping()


def test_cancelled_open_keeps_lease_until_cleanup_and_discards_first_frame():
    backend = ControlledBackend()
    manager = CaptureSessionManager([backend], refresh_on_init=False)
    manager.refresh_devices()
    backend.open_gate.clear()
    frames, opened = [], []
    manager.frameReady.connect(frames.append)
    assert manager.start_preview_async(on_done=opened.append)
    wait(backend.opening.is_set)
    assert not manager.is_preview_active()
    manager.cancel_pending_operations()
    assert manager.is_stopping()
    assert not manager.start_preview_async()
    backend.open_gate.set()
    wait(lambda: not manager.device_operations_pending())
    assert not frames and not opened and not backend.active
    assert not manager.is_starting() and not manager.is_stopping()
    assert [call[0] for call in backend.calls] == ["enumerate", "open", "stop"]
    assert manager.start_preview_async(on_done=opened.append)
    wait(lambda: opened == [True])
    assert manager.is_preview_active() and frames
    manager.request_stop_preview()
    wait(lambda: not manager.is_preview_active())


def test_cancelling_queued_open_never_sticks_in_stopping_state():
    backend = ControlledBackend()
    manager = CaptureSessionManager([backend], refresh_on_init=False)
    manager.refresh_devices()
    manager.start_preview_async()
    manager.cancel_pending_operations()  # before Qt dispatches the worker
    QApplication.processEvents()
    assert not manager.is_starting() and not manager.is_stopping()
    assert not backend.opening.is_set()


def test_slow_stop_is_off_gui_and_prevents_a_second_connection():
    backend = ControlledBackend()
    manager = CaptureSessionManager([backend], refresh_on_init=False)
    manager.refresh_devices()
    manager.start_preview_async()
    wait(manager.is_preview_active)
    backend.stop_gate.clear()
    manager.request_stop_preview()
    wait(backend.stopping.is_set)
    assert manager.is_stopping() and backend.active
    assert not manager.start_preview_async()
    assert [stage for stage, _ in backend.calls].count("open") == 1
    backend.stop_gate.set()
    wait(lambda: not manager.is_stopping())
    assert not manager.is_preview_active() and not backend.active
    assert all(thread != get_ident() for stage, thread in backend.calls if stage in ("open", "stop"))


def test_qt_owned_backend_lifecycle_stays_on_the_gui_thread():
    backend = ControlledBackend()
    backend.enumerate_in_worker = False
    backend.lifecycle_in_worker = False
    manager = CaptureSessionManager([backend], refresh_on_init=False)
    manager.refresh_devices_async(lambda devices: manager.start_preview_async())
    wait(manager.is_preview_active)
    manager.request_stop_preview()
    assert not manager.is_preview_active()
    assert all(thread == get_ident() for _, thread in backend.calls)


def test_owner_destroyed_after_result_queued_still_releases_device_once():
    import shiboken6
    from fdm.services.capture_device_jobs import CaptureDeviceJobs
    jobs = CaptureDeviceJobs()
    cleanup = Event()
    completed = []
    job = jobs.submit("open", lambda: True, lambda *_: completed.append(True),
                      orphan_cleanup=cleanup.set)
    # Dispatch the worker without pumping Qt again: its finished signal can
    # queue successfully just before the GUI receiver is deleted.
    jobs._pump()
    assert job.ready.wait(1)
    shiboken6.delete(jobs)
    assert cleanup.wait(1)
    QApplication.processEvents()
    assert not completed


def test_owner_destroyed_before_queued_release_runs_still_releases_lease():
    import shiboken6
    from fdm.services.capture_device_jobs import CaptureDeviceJobs
    jobs = CaptureDeviceJobs()
    released = Event()
    jobs.submit("release", released.set, lambda *_: None, always=True, worker=True)
    shiboken6.delete(jobs)
    assert released.wait(1)


def test_failed_async_stop_can_be_retried_without_a_second_open():
    backend = ControlledBackend()
    original_stop = backend.stop_preview
    failures = [True]
    def stop():
        if failures.pop() if failures else False:
            raise RuntimeError("driver has not released yet")
        original_stop()
    backend.stop_preview = stop
    manager = CaptureSessionManager([backend], refresh_on_init=False)
    manager.refresh_devices()
    manager.start_preview_async()
    wait(manager.is_preview_active)
    manager.request_stop_preview()
    wait(lambda: manager.current_capture_state() == "faulted")
    assert manager.is_preview_active() and not manager.is_stopping()
    manager.request_stop_preview()
    wait(lambda: not manager.is_preview_active())
    assert not backend.active
    assert [stage for stage, _ in backend.calls].count("open") == 1


def test_async_refresh_reports_active_device_loss_and_stops_owner():
    backend = ControlledBackend()
    manager = CaptureSessionManager([backend], refresh_on_init=False)
    manager.refresh_devices()
    manager.start_preview_async()
    wait(manager.is_preview_active)
    lost = []
    manager.activeDeviceLost.connect(lost.append)
    backend.list_devices = lambda: []
    manager.refresh_devices_async()
    wait(lambda: not manager.is_preview_active())
    assert lost == ["controlled:0"]


def test_failed_cancelled_open_cleanup_retains_lease_and_allows_explicit_retry():
    backend = ControlledBackend()
    backend.open_gate.clear()
    original_stop = backend.stop_preview
    attempts = []
    def stop():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("release has not completed")
        original_stop()
    backend.stop_preview = stop
    manager = CaptureSessionManager([backend], refresh_on_init=False)
    manager.refresh_devices()
    manager.start_preview_async()
    wait(backend.opening.is_set)
    manager.cancel_pending_operations()
    backend.open_gate.set()
    wait(lambda: manager.current_capture_state() == "faulted")
    assert manager.is_stopping() and backend.active
    assert not manager.start_preview_async()
    manager.request_stop_preview()
    wait(lambda: not manager.is_stopping())
    assert len(attempts) == 2 and not backend.active


def test_main_window_uses_known_listing_and_resumes_digital_slide_after_open(monkeypatch):
    from fdm.ui.main_window import MainWindow
    backend = ControlledBackend()
    backend.open_gate.clear()
    monkeypatch.setattr("fdm.services.capture.available_capture_backends", lambda: [backend])
    window = MainWindow()
    try:
        window.show()
        window._open_acquisition_workbench(trial_index=0, auto_start=False)
        window._capture_manager.refresh_devices()
        for name in ("_apply_digital_slide_motion_settings", "_reset_digital_slide_motion_zero",
                     "_refresh_digital_slide_ports", "_on_digital_slide_motor_enabled_changed"):
            monkeypatch.setattr(window, name, lambda *args, **kwargs: None)
        monkeypatch.setattr(window, "_check_digital_slide_motion_status", lambda: False)
        window._toggle_digital_slide_mode(True)
        wait(backend.opening.is_set)
        assert not window._digital_slide_mode
        assert [stage for stage, _ in backend.calls].count("enumerate") == 1
        backend.open_gate.set()
        wait(lambda: window._digital_slide_mode)
        assert window._preview_active
        assert [stage for stage, _ in backend.calls].count("open") == 1
    finally:
        backend.open_gate.set()
        window._digital_slide_mode = False
        window.stop_live_preview()
        wait(lambda: not window._capture_manager.device_operations_pending())
        monkeypatch.setattr(window, "_confirm_close_documents_for_transition", lambda *_: True)
        window.close()


def test_faulted_cancelled_open_offers_retry_stop_without_reopening_in_ui(monkeypatch):
    from fdm.ui.main_window import MainWindow
    backend = ControlledBackend()
    backend.open_gate.clear()
    original_stop = backend.stop_preview
    attempts = []
    def stop():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("test driver still owns the connection")
        original_stop()
    backend.stop_preview = stop
    monkeypatch.setattr("fdm.services.capture.available_capture_backends", lambda: [backend])
    window = MainWindow()
    try:
        window.show()
        hub = window._open_acquisition_workbench(trial_index=0, auto_start=False)
        window._capture_manager.refresh_devices()
        hub.request_start()
        wait(backend.opening.is_set)
        window.stop_live_preview()
        backend.open_gate.set()
        wait(lambda: window._capture_manager.current_capture_state() == "faulted")
        assert window._capture_manager.is_stopping()
        assert hub.capture_button.text() == "重试停止" and hub.capture_button.isEnabled()
        hub.capture_button.click()
        assert not hub.capture_button.isEnabled()
        wait(lambda: not window._capture_manager.is_stopping())
        assert len(attempts) == 2 and not backend.active
        assert [stage for stage, _ in backend.calls].count("open") == 1
        assert not window._capture_manager.is_preview_active()
    finally:
        backend.open_gate.set()
        monkeypatch.setattr(window, "_confirm_close_documents_for_transition", lambda *_: True)
        window.close()
