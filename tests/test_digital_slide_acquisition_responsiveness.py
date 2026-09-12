"""Acquisition backpressure, source-frame identity and GUI ownership regressions."""
from __future__ import annotations

from pathlib import Path
from threading import Event, Thread, get_ident
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from fdm.lifecycle import DigitalSlideAcquisitionSession
from fdm.services import capture, motion_control
from fdm.services.capture import CaptureBackend, CaptureDevice, CaptureSessionManager
from fdm.services.digital_slide_cache import DigitalSlideCacheCancelled, DigitalSlideSessionCache
from fdm.services.digital_slide_renderer import DigitalSlideDerivedCache
from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore, DigitalSlideTile
from fdm.services.motion_control import AXIS_X, AXIS_Z, DIR_POS, MotionPortInfo
from fdm.ui import main_window
from fdm.ui.main_window import DigitalSlideWriteWorker, MainWindow, TransitionIntent
from fdm.ui.responsive_io import run_responsive_io


class Camera(CaptureBackend):
    backend_key = "test-source"

    def list_devices(self):
        return [CaptureDevice("test:1", "Camera", self.backend_key, 1)]

    def start_preview(self, device, *, frame_callback, error_callback, **kwargs):
        self.send = frame_callback

    def stop_preview(self):
        pass


class Serial:
    def __init__(self, *args, **kwargs):
        self.is_open = True
        self.options = kwargs
        self.writes = []
        self.fail_write = False

    def write(self, payload):
        self.writes.append(bytes(payload))
        if self.fail_write:
            raise TimeoutError("serial write timeout")
        return len(payload)

    def close(self):
        self.is_open = False


def image(color="red", width=64, height=48):
    result = QImage(width, height, QImage.Format.Format_RGB32)
    result.fill(QColor(color))
    return result


@pytest.fixture
def manager():
    camera = Camera()
    manager = CaptureSessionManager(backends=[camera])
    assert manager.start_preview()
    yield manager, camera
    manager.stop_preview()
    manager.deleteLater()
    QApplication.processEvents()


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    camera = Camera()
    monkeypatch.setattr(capture, "available_capture_backends", lambda: [camera])
    monkeypatch.setattr(motion_control, "serial", SimpleNamespace(Serial=Serial))
    monkeypatch.setattr(main_window, "list_motion_ports", lambda: [
        MotionPortInfo("COM7", is_ftdi_motion=True), MotionPortInfo("COM9"),
    ])
    monkeypatch.setattr("fdm.ui.digital_slide_canvas.digital_slide_render_cache_directory", lambda: tmp_path / "render-cache")

    def unexpected_warning(_parent, title, message, *_args, **_kwargs):
        raise AssertionError(f"{title}: {message}")

    monkeypatch.setattr(QMessageBox, "warning", unexpected_warning)
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    window._toggle_digital_slide_mode(True)
    camera.send(image())
    QApplication.processEvents()
    monkeypatch.setattr(window, "_show_digital_slide_completion_dialog", lambda **kwargs: None)
    monkeypatch.setattr(window, "_confirm_close_documents", lambda *args, **kwargs: True)
    yield window, camera
    window._slide_acquisition_timer.stop()
    assert window.close()


def test_preview_burst_has_one_wakeup_and_latest_owned_pixels(manager):
    manager, camera = manager
    displayed = []
    manager.frameReady.connect(lambda frame: displayed.append(frame.pixelColor(0, 0).red()))
    source = image()
    for red in range(120):
        source.fill(QColor(red, 0, 0))
        camera.send(source)
    assert displayed == []
    snapshot = manager.latest_preview_frame()
    assert snapshot.sequence == 120
    assert manager.preview_frame_stats()["pending_wakeups"] == 1
    assert manager.preview_frame_stats()["coalesced"] == 119
    source.fill(QColor("white"))
    assert snapshot.image.pixelColor(0, 0).red() == 119
    snapshot.image.fill(QColor("black"))
    assert manager.last_frame().pixelColor(0, 0).red() == 119
    QApplication.processEvents()
    assert displayed == [119]
    assert manager.preview_frame_stats()["pending_wakeups"] == 0


def test_old_session_wakeup_cannot_consume_new_session_frame(manager):
    manager, camera = manager
    displayed = []
    manager.frameReady.connect(lambda frame: displayed.append(frame.pixelColor(0, 0).name()))
    old_send = camera.send
    old_send(image("red"))
    manager.stop_preview()
    manager.start_preview()
    camera.send(image("blue"))
    old_send(image("green"))
    QApplication.processEvents()
    assert displayed == ["#0000ff"]
    assert manager.preview_frame_stats()["received"] == 1


def test_frame_arriving_during_delivery_is_not_lost(manager):
    manager, camera = manager
    displayed = []

    def receive(frame):
        displayed.append(frame.pixelColor(0, 0).name())
        if len(displayed) == 1:
            camera.send(image("blue"))

    manager.frameReady.connect(receive)
    camera.send(image("red"))
    QApplication.processEvents()
    QApplication.processEvents()
    assert displayed == ["#ff0000", "#0000ff"]


def test_steady_frames_update_pixels_without_rebuilding_workspace(workspace):
    window, camera = workspace
    camera.send(image())
    QApplication.processEvents()
    strip = window._measurement_tool_strip
    with patch.object(window, "_update_action_states") as actions, patch.object(
        window._preview_canvas, "fit_to_view"
    ) as fit, patch.object(strip._top_row_layout, "insertWidget", wraps=strip._top_row_layout.insertWidget) as insert:
        for _ in range(50):
            camera.send(image("blue"))
            QApplication.processEvents()
        actions.assert_not_called()
        fit.assert_not_called()
        insert.assert_not_called()
    assert window._preview_canvas._image.pixelColor(0, 0) == QColor("blue")
    camera.send(image("green", width=96, height=72))
    QApplication.processEvents()
    assert window._preview_document.image_size == (96, 72)


@pytest.mark.parametrize("discard", [0, 3])
def test_capture_ignores_delayed_gui_frames_and_counts_source_frames(workspace, tmp_path, discard):
    window, camera = workspace
    store = DigitalSlideStore.create(tmp_path / "source.fdmslide", DigitalSlideManifest(1, 64, 48, 64, 48, [0]))
    window._slide_acquisition_store = store
    window._active_slide_acquisition = DigitalSlideAcquisitionSession(1, "test", store.path)
    window._slide_acquisition_plan = [dict(z_index=0, global_x=0, global_y=0, stage_x=10, stage_y=0, focus_z=0, row=0, col=0)]
    window._slide_acquisition_viewport_size = (64, 48)
    window._app_settings.digital_slide_discard_frames = discard
    accepted = []
    try:
        # These frames arrive before settle completes, but their GUI wakeup
        # deliberately remains queued until after the source cursor is saved.
        for _ in range(12):
            camera.send(image("red"))
        window._begin_digital_slide_frame_wait()
        window._slide_acquisition_timer.stop()
        QApplication.processEvents()
        with patch.object(window, "_enqueue_digital_slide_tile", lambda tile, frame, *args: accepted.append(frame.pixelColor(0, 0))):
            window._capture_next_digital_slide_frame()
            window._slide_acquisition_timer.stop()
            assert accepted == []
            for _ in range(discard):
                camera.send(image("yellow"))
            window._capture_next_digital_slide_frame()
            window._slide_acquisition_timer.stop()
            assert accepted == []
            camera.send(image("blue"))
            # No GUI delivery occurs here. Capture must still use the newest
            # source pixels and all source arrivals for the discard threshold.
            window._capture_next_digital_slide_frame()
            assert accepted == [QColor("blue")]
    finally:
        window._clear_digital_slide_acquisition_session()
        store.close()


def test_transition_and_device_loss_reenable_motor_on_reentry(workspace):
    window, camera = workspace
    for _ in range(4):
        assert window._slide_motion.enabled
        result = window._prepare_transition(TransitionIntent.SWITCH_DEVICE)
        assert result.completed
        assert not window._digital_slide_motor_enable.isChecked()
        window._toggle_digital_slide_mode(True)
        camera.send(image())
        QApplication.processEvents()
        assert window._digital_slide_motor_enable.isChecked()
        assert window._slide_motion.move(AXIS_X, 1, DIR_POS)
        assert window._slide_motion.move(AXIS_Z, 1, DIR_POS)
    window._on_active_capture_device_lost("test:1")
    assert not window._slide_motion.enabled
    assert not window._digital_slide_motor_enable.isChecked()
    window._toggle_digital_slide_mode(True)
    assert window._slide_motion.move(AXIS_Z, 1, DIR_POS)


def test_reentry_preserves_manual_serial_choice_and_write_timeout(workspace):
    window, _camera = workspace
    window._digital_slide_port_combo.setCurrentIndex(window._digital_slide_port_combo.findData("COM9"))
    window._toggle_digital_slide_mode(False)
    window._toggle_digital_slide_mode(True)
    assert window._slide_motion.port == "COM9"
    serial = window._slide_motion._serial
    assert serial.options["write_timeout"] == window._slide_motion.timeout
    serial.fail_write = True
    before = dict(window._slide_motion.relative_pos)
    with pytest.raises(TimeoutError):
        window._slide_motion.move(AXIS_X, 2, DIR_POS)
    assert window._slide_motion.relative_pos == before
    assert not window._digital_slide_motor_enable.isChecked()
    assert not serial.is_open
    assert len(serial.writes) == 1


def test_io_worker_keeps_gui_running_and_guards_owner_transitions(workspace):
    window, camera = workspace
    released = Event()
    beats = []
    threads = []
    transitions = []
    close_attempts = []
    timer = QTimer(window)
    timer.setInterval(5)

    def beat():
        beats.append(perf_counter())
        camera.send(image("blue"))
        if len(beats) == 3:
            transitions.append(window._prepare_transition(TransitionIntent.SWITCH_DEVICE))
            close_attempts.append(window.close())
            for dialog in window.findChildren(QProgressDialog):
                dialog.reject()  # Escape cannot detach an unfinished publisher.
            assert window._slide_io_busy
        if len(beats) >= 8:
            released.set()

    def operation(progress):
        threads.append(get_ident())
        for i in range(1000):
            progress(i, 1000)
        assert released.wait(3), "GUI stopped dispatching while worker waited"
        return 42

    timer.timeout.connect(beat)
    timer.start()
    try:
        assert window._run_slide_io("test I/O", operation) == 42
    finally:
        timer.stop()
        released.set()
    assert threads != [get_ident()]
    assert len(beats) >= 8
    assert transitions and not transitions[0].completed
    assert close_attempts == [False]
    assert not window._slide_io_busy
    assert window._capture_manager.preview_frame_stats()["pending_wakeups"] <= 1
    with pytest.raises(OSError, match="failed"):
        window._run_slide_io("test failure", lambda _progress: (_ for _ in ()).throw(OSError("failed")))
    assert not window._slide_io_busy


def test_network_read_cancellation_waits_for_worker_exit(workspace):
    window, _camera = workspace
    cancelled = Event()
    finished = Event()
    clicked = []

    def cancel_dialog():
        for dialog in window.findChildren(QProgressDialog):
            dialog.cancel()
            dialog.canceled.emit()
            clicked.append(True)

    def operation(_progress):
        assert cancelled.wait(3)
        finished.set()
        raise DigitalSlideCacheCancelled("cancelled")

    QTimer.singleShot(50, window, cancel_dialog)
    with pytest.raises(DigitalSlideCacheCancelled):
        run_responsive_io(window, title="test", label="test", operation=operation, cancellation_event=cancelled)
    assert clicked and finished.is_set()


def test_writer_accepts_one_oversized_frame_without_unbounded_queue(tmp_path):
    path = tmp_path / "large-frame.fdmslide"
    DigitalSlideStore.create(path, DigitalSlideManifest(1, 64, 48, 64, 48, [0])).close()
    worker = DigitalSlideWriteWorker(path, max_queue_bytes=100)
    source = image("red")
    tile = DigitalSlideTile(0, 0, 0, 64, 48)
    assert source.sizeInBytes() > worker._max_queue_bytes
    assert worker.enqueue(tile, source)
    assert not worker.enqueue(tile, source)
    assert worker._queued_bytes == source.sizeInBytes()
    source.fill(QColor("blue"))
    worker.start()
    worker.finish()
    worker.wait()
    assert not worker.is_running()
    store = DigitalSlideStore(path)
    try:
        assert store.tile_count() == 1
        assert store.render_viewport(x=0, y=0, width=64, height=48, z_index=0).pixelColor(0, 0) == QColor("red")
    finally:
        store.close()


def test_unviewed_slides_preserve_focus_without_rendering_then_open(workspace, tmp_path):
    window, _camera = workspace
    for index in range(3):
        path = tmp_path / f"capture-{index}.fdmslide"
        store = DigitalSlideStore.create(path, DigitalSlideManifest(1, 64, 48, 64, 48, [0, 10]))
        store.write_tile(DigitalSlideTile(1, 0, 0, 64, 48), image("blue"))
        store.close()
        window._add_digital_slide_document_from_path(path, document=None, metadata={"digital_slide": {"focus_index": 1}})
        QApplication.processEvents()
    canvases = list(window._canvases.values())
    assert len(canvases) == 3
    assert all(canvas._renderer is None and canvas._image is None for canvas in canvases)
    assert all(canvas.focus_index() == 1 for canvas in canvases)
    assert all(document.metadata["digital_slide"]["focus_index"] == 1 for document in window.project.documents)
    window.stop_live_preview()
    QTest.qWait(200)
    current = window.current_canvas()
    assert current._deferred_slide_document is None
    assert current._image is not None and current._renderer is not None
    assert current.focus_index() == 1
    assert all(canvas._renderer is None for canvas in canvases if canvas is not current)


def test_prepared_fingerprint_never_stats_original_source(tmp_path):
    path = tmp_path / "source.fdmslide"
    manifest = DigitalSlideManifest(1, 64, 48, 64, 48, [0])
    DigitalSlideStore.create(path, manifest).close()
    revision = path.stat()
    expected = DigitalSlideDerivedCache.source_fingerprint(path, manifest)
    with patch.object(Path, "stat", side_effect=AssertionError("UI stat")), patch.object(Path, "resolve", side_effect=AssertionError("UI resolve")):
        actual = DigitalSlideDerivedCache.source_fingerprint(
            path, manifest, source_identity=path,
            source_stat=(revision.st_size, revision.st_mtime_ns),
        )
    assert actual == expected


def test_network_preflight_and_publication_do_io_outside_gui(workspace, tmp_path, monkeypatch):
    window, _camera = workspace
    target = tmp_path / "network" / "slide.fdmslide"
    local = tmp_path / "local.fdmslide"
    DigitalSlideStore.create(local, DigitalSlideManifest(1, 64, 48, 64, 48, [0])).close()
    main_thread = get_ident()
    threads = []
    original_stat = Path.stat
    original_publish = window._digital_slide_local_cache.publish
    published = []

    def stat(path, *args, **kwargs):
        if path == target or path == target.parent:
            threads.append(get_ident())
        return original_stat(path, *args, **kwargs)

    def publish(*args, **kwargs):
        published.append(get_ident())
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(window._digital_slide_local_cache, "publish", publish)
    assert window._confirm_digital_slide_capture_budget(
        output_path=target, image_count=1, estimated_bytes=100, estimated_total_ms=1,
    )
    window._publish_network_digital_slide(local, target)
    source, interaction, revision = window._prepare_digital_slide_source(target, interaction_path_override=local)
    assert source == target and interaction == local
    assert revision[0] > 0
    assert threads and all(thread != main_thread for thread in threads)
    assert published and published[0] != main_thread


def test_captured_slide_preparation_never_revisits_published_source(workspace, tmp_path, monkeypatch):
    window, _camera = workspace
    target = tmp_path / "network" / "清水.fdmslide"
    local = tmp_path / "capture.fdmslide"
    DigitalSlideStore.create(local, DigitalSlideManifest(1, 64, 48, 64, 48, [0])).close()
    local_stat = local.stat()
    original_resolve = Path.resolve
    original_stat = Path.stat
    network_calls = []

    def guard(method):
        def guarded(path, *args, **kwargs):
            if path == target:
                network_calls.append(method.__name__)
                raise AssertionError("published network source must not be queried again")
            return method(path, *args, **kwargs)
        return guarded

    monkeypatch.setattr(Path, "resolve", guard(original_resolve))
    monkeypatch.setattr(Path, "stat", guard(original_stat))
    source, interaction, revision = window._prepare_digital_slide_source(
        target, interaction_path_override=local,
    )
    assert source == target
    assert interaction == original_resolve(local)
    assert revision == (local_stat.st_size, local_stat.st_mtime_ns)
    assert network_calls == []
    assert not window._slide_io_busy


@pytest.mark.parametrize("status", ["ready", "interrupted"])
def test_published_capture_completes_and_views_without_reopening_share(
    workspace, tmp_path, monkeypatch, status,
):
    window, _camera = workspace
    target = tmp_path / "network" / "清水.fdmslide"
    cache = DigitalSlideSessionCache(
        root=tmp_path / "read-cache", output_staging_root=tmp_path / "staging",
        network_path_predicate=lambda path: Path(path) == target,
    )
    window._digital_slide_local_cache = cache
    local = cache.working_output_path(target)
    store = DigitalSlideStore.create(local, DigitalSlideManifest(1, 64, 48, 64, 48, [0]))
    store.write_tile(DigitalSlideTile(0, 0, 0, 64, 48), image("blue"))
    store.close()
    window._slide_acquisition_store = store
    window._slide_acquisition_path = local
    window._slide_acquisition_publish_path = target
    window._slide_acquisition_document_path = str(target)
    completed = []
    published = Event()
    network_calls = []
    publish = cache.publish

    def publish_then_disconnect(*args, **kwargs):
        result = publish(*args, **kwargs)
        published.set()
        return result

    def guard(method):
        def guarded(path, *args, **kwargs):
            if published.is_set() and (path == target.parent or target.parent in path.parents):
                network_calls.append((method.__name__, str(path)))
                raise AssertionError("share stopped responding after successful publication")
            return method(path, *args, **kwargs)
        return guarded

    with monkeypatch.context() as guarded_share:
        guarded_share.setattr(cache, "publish", publish_then_disconnect)
        guarded_share.setattr(cache, "localize", lambda *args, **kwargs: pytest.fail("capture copied back from share"))
        guarded_share.setattr(window, "_show_digital_slide_completion_dialog", lambda **result: completed.append(result))
        for method_name in ("stat", "resolve", "open"):
            guarded_share.setattr(Path, method_name, guard(getattr(Path, method_name)))
        window._finish_digital_slide_acquisition(status=status, message="complete")
        assert published.is_set()
        assert len(completed) == 1
        assert completed[0]["path"] == target
        assert completed[0]["status"] == status
        assert not window._slide_io_busy
        assert not window._slide_acquisition_active()
        assert window._digital_slide_motor_enable.isChecked()
        assert window._preview_active
        assert len(window.project.documents) == 1
        document = window.project.documents[0]
        assert Path(document.path) == target
        assert Path(document.absolute_path) == target
        canvas = window._canvases[document.id]
        assert canvas._renderer is None and canvas._image is None
        assert window._slide_stores[document.id].path == local.resolve()

        # The completion dialog's "view" action must use this existing local
        # document as well, even if the share no longer answers metadata I/O.
        window.stop_live_preview()
        window._digital_slide_mode = False
        window._sync_digital_slide_mode_ui()
        window._focus_digital_slide_path(target)
        deadline = perf_counter() + 5
        while not canvas.pixel_work_enabled() and perf_counter() < deadline:
            QTest.qWait(10)
        assert window.current_document() is document
        assert canvas.pixel_work_enabled()
        viewport = canvas._slide_store.render_viewport(x=0, y=0, width=64, height=48, z_index=0)
        assert viewport.pixelColor(0, 0) == QColor("blue")
        assert network_calls == []
    assert DigitalSlideStore.read_manifest_read_only(target).tile_count == 1
    assert local.is_file()


def test_continuous_captures_publish_without_starting_hidden_renderers(workspace, tmp_path, monkeypatch):
    window, camera = workspace
    target_dir = tmp_path / "network"
    window._digital_slide_local_cache = DigitalSlideSessionCache(
        root=tmp_path / "read-cache", output_staging_root=tmp_path / "staging",
        network_path_predicate=lambda path: Path(path).parent == target_dir,
    )
    window._digital_slide_cols_edit.setText("3")
    window._digital_slide_rows_edit.setText("2")
    window._digital_slide_z_lower_edit.setText("0")
    window._digital_slide_z_upper_edit.setText("0")
    settings = window._app_settings
    settings.digital_slide_x_stage_step = 10
    settings.digital_slide_y_stage_step = 10
    settings.digital_slide_xy_settle_ms = 1
    settings.digital_slide_xy_post_settle_ms = 0
    settings.digital_slide_first_tile_extra_wait_ms = 0
    settings.digital_slide_discard_frames = 0
    completions = []
    monkeypatch.setattr(window, "_show_digital_slide_completion_dialog", lambda **result: completions.append(result))
    stop = Event()

    def produce():
        while not stop.is_set():
            camera.send(image())
            stop.wait(0.015)

    producer = Thread(target=produce, daemon=True)
    producer.start()
    try:
        for index in range(5):
            target = target_dir / f"scan-{index}.fdmslide"
            window._digital_slide_output_path_edit.setText(str(target))
            window._start_digital_slide_acquisition()
            deadline = perf_counter() + 8
            while len(completions) <= index and perf_counter() < deadline:
                QTest.qWait(10)
            assert len(completions) == index + 1
            assert completions[-1]["status"] == "ready"
            assert DigitalSlideStore.read_manifest_read_only(target).tile_count == 6
            assert not window._slide_acquisition_active()
            assert window._slide_acquisition_writer is None
            assert window._digital_slide_motor_enable.isChecked()
            assert window._preview_active
            assert all(canvas._renderer is None and canvas._image is None for canvas in window._canvases.values())
        assert len(window.project.documents) == 5
        assert window._slide_motion.move(AXIS_Z, 1, DIR_POS)
    finally:
        stop.set()
        producer.join(2)


@pytest.mark.parametrize("status, button_text", [("ready", "关闭"), ("interrupted", "查看切片")])
def test_finished_loader_stays_hidden_during_real_completion_dialog(
    workspace, tmp_path, monkeypatch, status, button_text,
):
    window, camera = workspace
    target = tmp_path / "network" / "清水.fdmslide"
    cache = DigitalSlideSessionCache(
        root=tmp_path / "read-cache", output_staging_root=tmp_path / "staging",
        network_path_predicate=lambda path: Path(path) == target,
    )
    window._digital_slide_local_cache = cache
    local = cache.working_output_path(target)
    store = DigitalSlideStore.create(local, DigitalSlideManifest(1, 64, 48, 64, 48, [0]))
    store.write_tile(DigitalSlideTile(0, 0, 0, 64, 48), image("blue"))
    window._slide_acquisition_store = store
    window._slide_acquisition_path = local
    window._slide_acquisition_publish_path = target
    window._slide_acquisition_document_path = str(target)
    # Previous tests replaced this modal with a no-op and missed the loader's
    # delayed forceShow while deferred deletion waited in this nested loop.
    monkeypatch.setattr(
        window, "_show_digital_slide_completion_dialog",
        MainWindow._show_digital_slide_completion_dialog.__get__(window),
    )
    observed = []
    completion_seen = False
    monitor = QTimer(window)

    def check_completion():
        nonlocal completion_seen
        camera.send(image())
        for box in window.findChildren(QMessageBox):
            if not box.isVisible() or box.text() != "capture complete":
                continue
            observed.append((
                any(dialog.isVisible() for dialog in window.findChildren(QProgressDialog)),
                QApplication.activeModalWidget() is box,
            ))
            if not completion_seen:
                completion_seen = True
                button = next(button for button in box.buttons() if button.text() == button_text)
                # Keep the actual modal open past the loader's 250 ms delay.
                QTimer.singleShot(400, box, button.click)

    monitor.timeout.connect(check_completion)
    monitor.start(10)
    loop = QEventLoop(window)
    errors = []

    def finish_capture():
        try:
            window._finish_digital_slide_acquisition(status=status, message="capture complete")
        except BaseException as exc:
            errors.append(exc)
        finally:
            loop.quit()

    QTimer.singleShot(0, window, finish_capture)
    try:
        loop.exec()
        assert not errors, errors
        assert completion_seen and len(observed) > 1
        assert all(not loader_visible and modal_active for loader_visible, modal_active in observed), observed
        assert not window._slide_acquisition_active()
        assert DigitalSlideStore.read_manifest_read_only(target).tile_count == 1
        assert len(window.project.documents) == 1
        canvas = window._canvases[window.project.documents[0].id]
        if button_text == "关闭":
            assert window._preview_active
            assert canvas._renderer is None and canvas._image is None
        else:
            assert not window._preview_active
            deadline = perf_counter() + 5
            while not canvas.pixel_work_enabled() and perf_counter() < deadline:
                QTest.qWait(10)
            assert canvas.pixel_work_enabled()
    finally:
        monitor.stop()
        loop.deleteLater()


def test_device_loss_during_output_preparation_prevents_capture(workspace, tmp_path, monkeypatch):
    window, _camera = workspace
    target = tmp_path / "lost-camera.fdmslide"
    window._digital_slide_output_path_edit.setText(str(target))
    window._digital_slide_cols_edit.setText("1")
    window._digital_slide_rows_edit.setText("1")
    window._digital_slide_z_lower_edit.setText("0")
    window._digital_slide_z_upper_edit.setText("0")
    entered = Event()
    released = Event()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))

    def prepare(*args, **kwargs):
        entered.set()
        assert released.wait(3)
        return target

    def disconnect():
        if entered.is_set():
            window._on_active_capture_device_lost("test:1")
            released.set()

    monkeypatch.setattr(window._digital_slide_local_cache, "working_output_path", prepare)
    timer = QTimer(window)
    timer.setInterval(5)
    timer.timeout.connect(disconnect)
    timer.start()
    try:
        window._start_digital_slide_acquisition()
    finally:
        timer.stop()
        released.set()
    assert warnings and "连接已变化" in warnings[-1]
    assert window._slide_acquisition_writer is None
    assert not window._slide_acquisition_active()
    assert not target.exists()


def test_publication_failure_keeps_local_capture_and_existing_target(workspace, tmp_path, monkeypatch):
    window, _camera = workspace
    target = tmp_path / "network" / "failed.fdmslide"
    target.parent.mkdir()
    target.write_bytes(b"old-file")
    cache = DigitalSlideSessionCache(
        root=tmp_path / "read-cache", output_staging_root=tmp_path / "staging",
        network_path_predicate=lambda path: Path(path) == target,
    )
    window._digital_slide_local_cache = cache
    local = cache.working_output_path(target)
    store = DigitalSlideStore.create(local, DigitalSlideManifest(1, 64, 48, 64, 48, [0]))
    store.write_tile(DigitalSlideTile(0, 0, 0, 64, 48), image("blue"))
    store.close()
    window._slide_acquisition_store = store
    window._slide_acquisition_path = local
    window._slide_acquisition_publish_path = target
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))

    def fail_publish(*args, **kwargs):
        raise OSError("network disconnected")

    monkeypatch.setattr(cache, "publish", fail_publish)
    window._finish_digital_slide_acquisition(status="ready", message="complete")
    assert warnings and "本机恢复副本" in warnings[0]
    assert target.read_bytes() == b"old-file"
    assert local.exists()
    assert DigitalSlideStore.read_manifest_read_only(local).tile_count == 1
    assert len(window.project.documents) == 1
    assert Path(window.project.documents[0].path) == local
    assert not window._slide_acquisition_active()
