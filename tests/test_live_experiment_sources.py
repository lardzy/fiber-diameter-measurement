"""Ownership/arrival and protocol regression without a real capture board."""
from __future__ import annotations

from collections import deque
import ctypes
import multiprocessing
from multiprocessing import shared_memory
import sys
import sysconfig
from types import SimpleNamespace
import time
import weakref

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from fdm.live_experiment.capture_helper import decode_microview_buffer, copy_microview_to_shared, MicroviewMemoryClient, MAX_FRAME_BYTES
from fdm.live_experiment.sources import CameraFrameSource, ExternalCuFrameSource
from fdm.services.capture import PreviewFrameSnapshot


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(predicate, app=None, timeout=2):
    deadline = time.perf_counter() + timeout
    while not predicate() and time.perf_counter() < deadline:
        if app is not None:
            app.processEvents()
        time.sleep(.005)
    assert predicate()


def test_printwindow_geometry_retries_a_move_between_win32_calls():
    from fdm.live_experiment.window_capture_helper import _WindowIdentity
    from fdm.platform.windows_window_locator import PhysicalRect
    old, moved = PhysicalRect(100, 100, 740, 580), PhysicalRect(300, 200, 940, 680)
    rectangles = iter((old, moved, moved, moved))
    identity = object.__new__(_WindowIdentity)
    identity.root, identity.child, identity.pinned = 1, 2, {}
    identity.read = lambda: {}

    def client_rect(_hwnd, pointer):
        pointer._obj.right, pointer._obj.bottom = 640, 480
        return True

    def origin(_hwnd, pointer):
        pointer._obj.x, pointer._obj.y = 300, 200
        return True

    identity.user = SimpleNamespace(IsWindowVisible=lambda _hwnd: True,
        IsIconic=lambda _hwnd: False, GetClientRect=client_rect,
        ClientToScreen=origin, GetDpiForWindow=lambda _hwnd: 192)
    identity.dwm = SimpleNamespace(DwmGetWindowAttribute=lambda *_args: 0)
    rectangle, crop, dpi = identity.geometry(SimpleNamespace(get_window_rect=lambda _hwnd: next(rectangles)))
    assert rectangle == moved and crop == (0, 0, 640, 480) and dpi == 192


class FakeManager(QObject):
    frameReady = Signal(object)
    errorOccurred = Signal(str)
    activeDeviceLost = Signal(str)

    def __init__(self):
        super().__init__()
        self.snapshot = None
        self.active = True
        self.starts = self.stops = 0
        self.device = SimpleNamespace(native_id=0, backend_key="opencv")

    def selected_device(self):
        return self.device

    def preview_kind(self):
        return "frame_stream"

    def is_preview_active(self):
        return self.active

    def latest_preview_frame(self):
        return self.snapshot

    def start_preview(self):
        self.starts += 1
        self.active = True
        return True

    def stop_preview(self):
        self.stops += 1
        self.active = False
        return True

    def frame(self, sequence, *, generation=4, width=16):
        image = QImage(width, 12, QImage.Format.Format_ARGB32)
        image.fill(0xffaabbcc)
        self.snapshot = PreviewFrameSnapshot(generation, sequence, time.perf_counter(), image)
        self.frameReady.emit(image)


def test_camera_same_pixels_new_arrivals_are_distinct_and_owned(app):
    manager = FakeManager()
    frames, errors = [], []
    source = CameraFrameSource(manager, frames.append, errors.append)
    source.start()
    wait_until(lambda: source._connected, app)
    manager.frame(1)
    manager.frame(2)
    manager.frameReady.emit(manager.snapshot.image)
    assert len(frames) == 2
    assert frames[0].sequence == 1 and frames[1].sequence == 2
    assert frames[0].frame_id != frames[1].frame_id
    assert not frames[0].pixels.flags.writeable
    before = frames[0].pixels.copy()
    manager.snapshot.image.fill(0)
    np.testing.assert_array_equal(frames[0].pixels, before)
    source.stop()
    manager.frame(3)
    assert len(frames) == 2
    assert manager.stops == 0  # an existing preview belongs to its original owner
    assert errors == []
    source.deleteLater()
    manager.deleteLater()
    app.processEvents()


def test_camera_resolution_and_manager_generation_change_frame_generation(app):
    manager = FakeManager()
    frames = []
    source = CameraFrameSource(manager, frames.append, lambda _: None)
    source.start()
    wait_until(lambda: source._connected, app)
    manager.frame(1)
    manager.frame(2, width=20)
    manager.frame(1, generation=5, width=20)
    assert len({f.generation for f in frames}) == 3
    source.stop()
    source.deleteLater()
    manager.deleteLater()
    app.processEvents()


def test_camera_start_is_deferred_and_stop_cancels_pending_start(app):
    manager = FakeManager()
    manager.active = False
    source = CameraFrameSource(manager, lambda _: None, lambda _: None)
    source.start()
    assert manager.starts == 0
    source.stop()
    app.processEvents()
    assert manager.starts == 0
    source.deleteLater()
    manager.deleteLater()
    app.processEvents()


def test_camera_watchdog_uses_arrival_time_not_pixel_difference(app):
    manager = FakeManager()
    errors = []
    source = CameraFrameSource(manager, lambda _: None, errors.append)
    source.start()
    wait_until(lambda: source._connected, app)
    manager.frame(1)
    source._last_frame_at -= 4
    source._check_arrivals()
    assert errors and "新的帧" in errors[0]
    source.stop()
    source.deleteLater()
    manager.deleteLater()
    app.processEvents()


class RetiringThread:
    alive = True

    def join(self, _timeout):
        pass

    def is_alive(self):
        return self.alive


@pytest.mark.parametrize("kind", ["camera", "external"])
def test_source_stop_keeps_live_thread_owner_until_confirmed_finished(app, kind):
    manager, errors = FakeManager(), []
    source = (CameraFrameSource(manager, lambda _: None, errors.append) if kind == "camera"
              else ExternalCuFrameSource(lambda _: None, errors.append))
    thread = source._thread = RetiringThread()
    assert source.stop() is False
    assert source._thread is thread
    source.start()
    assert source._thread is thread
    thread.alive = False
    assert source.stop() is True
    assert source._thread is None
    assert source.stop() is True
    assert errors


@pytest.mark.parametrize("raises", [False, True])
def test_camera_retains_owned_manager_preview_after_failed_stop(app, raises):
    manager, errors = FakeManager(), []
    source = CameraFrameSource(manager, lambda _: None, errors.append)
    source._owns_preview = True
    def fail():
        if raises:
            raise RuntimeError("device still stopping")
        return False
    manager.stop_preview = fail
    assert source.stop() is False
    assert source._owns_preview is True
    source.start()
    assert not source._active  # cannot restart an owner whose stop failed
    manager.stop_preview = lambda: True
    assert source.stop() is True
    assert not source._owns_preview
    assert source.stop() is True


class RetiringClient:
    closed = False
    can_close = False

    def start(self):
        raise RuntimeError("controlled startup failure")

    def close(self):
        if not self.can_close:
            raise RuntimeError("worker exit not confirmed")
        self.closed = True


def test_camera_preserves_helper_after_monitor_thread_cleanup_failure(app):
    source = CameraFrameSource(FakeManager(), lambda _: None, lambda _: None)
    client = RetiringClient()
    source._client_factory = lambda _: client
    source._memory_loop(0, source._epoch)
    assert source._retiring_client is client
    assert source.stop() is False
    source.start()
    assert not source._active
    client.can_close = True
    assert source.stop() is True
    assert client.closed and source._retiring_client is None


def test_external_preserves_helper_after_monitor_thread_cleanup_failure():
    source, capture = setup_external([RuntimeError("controlled polling failure")], [], [])
    client = RetiringClient()
    capture.close = client.close
    source._run(source._epoch)
    assert source._retiring_capture is capture
    assert source.stop() is False
    source.start()
    assert source._thread is None
    client.can_close = True
    assert source.stop() is True
    assert client.closed and source._retiring_capture is None


@pytest.mark.parametrize("kind", ["camera", "external"])
def test_unretained_delivery_is_released_before_next_poll(app, kind):
    references, observations = [], []
    class Client:
        polls = 0
        def start(self):
            pass
        def identity(self):
            return {}
        def poll(self, _timeout):
            self.polls += 1
            if self.polls > 1:
                observations.append(references[0]() is None)
                raise RuntimeError("controlled end of stream")
            return {**packet(1), "received_at": time.perf_counter()}
        def close(self):
            pass
    client = Client()
    callback = lambda frame: references.append(weakref.ref(frame.pixels))
    if kind == "camera":
        source = CameraFrameSource(FakeManager(), callback, lambda _: None)
        source._client_factory = lambda _: client
        source._memory_loop(0, source._epoch)
    else:
        source, _ = setup_external([], [], [])
        source.frame_callback = callback
        source._capture_factory = lambda *_: client
        source._run(source._epoch)
    assert observations == [True]
    assert source.stop() is True


class FakeCapture:
    def __init__(self, packets):
        self.packets = deque(packets)
        self.closed = False

    def identity(self):
        return {"pid": 9, "process_created": 100, "root_hwnd": 1, "child_hwnd": 2}

    def poll(self, _timeout):
        if self.packets:
            value = self.packets.popleft()
            if isinstance(value, Exception):
                raise value
            return value
        time.sleep(.01)
        return None

    def close(self):
        self.closed = True


def packet(sequence, generation=0):
    return {"pixels": np.full((24, 32, 4), 55, np.uint8), "sequence": sequence,
            "generation": generation, "dpi": 96, "crop": (10, 20, 32, 24),
            "source_timestamp_100ns": sequence * 10000}


def setup_external(packets, frames, errors):
    capture = FakeCapture(packets)
    source = ExternalCuFrameSource(frames.append, errors.append)
    source._capture_factory = lambda *_: capture
    source._locator_factory = lambda **_: SimpleNamespace(locate=lambda: SimpleNamespace(record=SimpleNamespace(root_hwnd=1), hwnd=2))
    return source, capture


def test_external_source_uses_native_arrival_not_pixel_hash():
    frames, errors = [], []
    source, capture = setup_external([packet(1), packet(2), packet(3, 1)], frames, errors)
    source.start()
    wait_until(lambda: len(frames) == 3)
    source.stop()
    assert capture.closed
    assert frames[0].generation == frames[1].generation
    assert frames[2].generation != frames[1].generation
    assert frames[0].metadata["raw_camera"] is False
    assert frames[0].metadata["process_created"] == 100
    assert frames[0].metadata["rendered_preview"] is True
    assert not frames[0].pixels.flags.writeable
    assert errors == []


def test_external_window_unavailable_closes_capture_without_fake_fallback():
    frames, errors = [], []
    source, capture = setup_external([packet(1), RuntimeError("capture_window_unavailable")], frames, errors)
    source.start()
    wait_until(lambda: bool(errors))
    source.stop()
    assert len(frames) == 1
    assert errors == ["capture_window_unavailable"]
    assert capture.closed


def test_external_nonadvancing_sequence_rejected():
    frames, errors = [], []
    source, capture = setup_external([packet(1), packet(1)], frames, errors)
    source.start()
    wait_until(lambda: bool(errors))
    source.stop()
    assert len(frames) == 1
    assert "序号" in errors[0]
    assert capture.closed


@pytest.mark.parametrize("channels,format_value", [(1, 0), (3, 2), (4, 3)])
def test_microview_decode_stride_and_owned_memory(channels, format_value):
    width, height, padding = 5, 4, 2
    data = np.arange(height * (width + padding) * channels, dtype=np.uint8)
    buffer = ctypes.create_string_buffer(data.tobytes())
    info = SimpleNamespace(Width=width, Heigth=height, Length=len(data), nColor=format_value, SkipPixel=padding)
    pixels = decode_microview_buffer(buffer, info)
    expected = data.reshape(height, (width + padding) * channels)[:, :width * channels].copy()
    if channels != 1:
        expected = expected.reshape(height, width, channels)
    if channels == 4:
        expected[..., 3] = 255
    np.testing.assert_array_equal(pixels, expected)
    ctypes.memset(ctypes.addressof(buffer), 0, len(buffer))
    np.testing.assert_array_equal(pixels, expected)


@pytest.mark.parametrize("format_value,word,expected", [(5, 0xf800, [0, 0, 255]), (5, 0x07e0, [0, 255, 0]),
    (1, 0x7c00, [0, 0, 255]), (1, 0x03e0, [0, 255, 0])])
def test_microview_supported_packed_rgb_formats(format_value, word, expected):
    data = np.array([word] * 8, dtype="<u2").tobytes()
    info = SimpleNamespace(Width=4, Heigth=2, Length=len(data), nColor=format_value, SkipPixel=0)
    result = decode_microview_buffer(data, info)
    assert result[0, 0].tolist() == expected


def test_microview_decode_rejects_invalid_metadata():
    for values in ({"Length": 100}, {"Width": 100}, {"nColor": 9}):
        properties = dict(Width=4, Heigth=2, Length=8, nColor=0, SkipPixel=0)
        properties.update(values)
        with pytest.raises(ValueError):
            decode_microview_buffer(b"12345678", SimpleNamespace(**properties))


def test_microview_client_rejects_invalid_memory_budget():
    with pytest.raises(ValueError):
        MicroviewMemoryClient(0, capacity=0)


@pytest.mark.parametrize("pixel_format,bpp", [(0, 1), (1, 2), (2, 3), (3, 4), (5, 2), (6, 2)])
def test_streaming_conversion_writes_directly_to_shared_slot(pixel_format, bpp):
    width, height, padding = 13, 7, 3
    source = np.random.default_rng(25).integers(0, 256, (height, (width + padding) * bpp), np.uint8).tobytes()
    info = SimpleNamespace(Width=width, Heigth=height, Length=len(source), nColor=pixel_format, SkipPixel=padding)
    expected = decode_microview_buffer(source, info)
    slot = bytearray(expected.nbytes)
    shape = copy_microview_to_shared(source, info, slot)
    actual = np.ndarray(shape, np.uint8, buffer=slot)
    np.testing.assert_array_equal(actual, expected)


def test_microview_defaults_support_6k_without_eager_128mb_allocation():
    client = MicroviewMemoryClient(0)
    assert client.capacity == MAX_FRAME_BYTES and client.fps == 30
    assert client._memory is None


def test_dynamic_shared_slot_is_detached_before_resize_ack(monkeypatch):
    from fdm.live_experiment import capture_helper
    events = []
    owner = SimpleNamespace(name="controlled-shared-slot", size=16,
                            close=lambda: events.append("closed"))
    monkeypatch.setattr(capture_helper.shared_memory, "SharedMemory", lambda **_: owner)
    messages, acknowledgements = deque(), []
    client = MicroviewMemoryClient(0, capacity=16)
    def acknowledge(message):
        events.append("acknowledged")
        acknowledgements.append(message)
    client._connection = SimpleNamespace(poll=lambda _: bool(messages), recv=messages.popleft, send=acknowledge)
    client._process = SimpleNamespace(is_alive=lambda: True)
    messages.append({"type": "buffer", "name": owner.name, "capacity": 16})
    assert client.poll() is None
    assert client._memory.size == 16
    messages.append({"type": "release_buffer"})
    assert client.poll() is None
    assert client._memory is None
    assert acknowledgements == [{"command": "buffer_released"}]
    assert events == ["closed", "acknowledged"]


def test_hung_capture_process_is_bounded_and_terminated():
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=time.sleep, args=(60,), daemon=True)
    process.start()
    client = MicroviewMemoryClient(0)
    client._process, client._connection = process, parent
    client._deadline = time.perf_counter() - 1
    started = time.perf_counter()
    try:
        with pytest.raises(TimeoutError):
            client.poll(.01)
    finally:
        client.close()
        child.close()
    assert time.perf_counter() - started < 3.5
    assert client._process is None


def test_microview_client_keeps_resources_until_writer_confirmed_dead():
    class Process:
        pid = 10
        dead = False
        closed = False
        def join(self, *_): pass
        def is_alive(self): return not self.dead
        def terminate(self): pass
        def kill(self): pass
        def close(self): self.closed = True
    process = Process()
    connection = SimpleNamespace(send=lambda _: None, close=lambda: None)
    memory = SimpleNamespace(close=lambda: None, unlink=lambda: None)
    client = MicroviewMemoryClient(0, capacity=16)
    client._process, client._connection, client._memory = process, connection, memory
    with pytest.raises(RuntimeError, match="退出"):
        client.close()
    assert client._process is process and client._memory is memory
    process.dead = True
    client.close()
    assert process.closed and client._process is client._memory is None


def test_microview_client_copies_then_acknowledges_slot_and_rejects_replay():
    messages, acknowledgements = deque(), []
    connection = SimpleNamespace(poll=lambda _: bool(messages), recv=messages.popleft, send=acknowledgements.append)
    client = MicroviewMemoryClient(0, capacity=16)
    client._connection = connection
    client._process = SimpleNamespace(is_alive=lambda: True)
    client._memory = SimpleNamespace(buf=bytearray(range(16)), size=16)
    message = {"type": "frame", "sequence": 1, "shape": (4, 4), "nbytes": 16, "received_at": time.perf_counter()}
    messages.append(message)
    first = client.poll()
    client._memory.buf[:] = bytes(16)
    assert first["pixels"][3, 3] == 15
    assert acknowledgements == [{"command": "ack", "sequence": 1}]
    messages.append(message)
    with pytest.raises(RuntimeError, match="序号"):
        client.poll()


@pytest.mark.parametrize("failure,expected_fallback", [("windows_graphics_capture_not_supported", True),
    ("capture_window_unavailable", False), ("D3D device lost", False), ("bad_abi", False)])
def test_compatibility_is_only_for_startup_capability_failure(monkeypatch, failure, expected_fallback):
    from fdm import _native
    from fdm.live_experiment import sources, window_capture_helper
    calls = []
    monkeypatch.setattr(sources.sys, "platform", "win32")
    monkeypatch.setattr(sources.sys, "getwindowsversion", lambda: SimpleNamespace(build=26100), raising=False)
    monkeypatch.setattr(_native, "_source_root", lambda: None)
    monkeypatch.setattr(window_capture_helper, "PrintWindowCaptureClient", lambda *args: calls.append(args) or "compat")
    def create(*_):
        raise RuntimeError(failure)
    info = {"api_version": 99 if failure == "bad_abi" else 1,
            "python_abi": sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag}
    module = SimpleNamespace(__file__="_windows_capture" + sources.EXTENSION_SUFFIXES[0],
                             build_info=lambda: info, WindowCapture=create)
    monkeypatch.setattr(sources.importlib, "import_module", lambda _: module)
    if expected_fallback:
        assert sources._create_external_capture(1, 2) == "compat"
        assert calls == [(1, 2)]
    else:
        with pytest.raises(RuntimeError):
            sources._create_external_capture(1, 2)
        assert calls == []
