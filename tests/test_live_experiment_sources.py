"""One device owner, immutable arrival frames and the raw callback protocol."""
import ctypes
from multiprocessing import shared_memory
from types import SimpleNamespace
import time

import numpy as np
import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from fdm.live_experiment.capture_helper import decode_microview_buffer, copy_microview_to_shared
from fdm.live_experiment.frames import FrameHandle
from fdm.live_experiment.sources import CameraFrameSource
from fdm.services.capture import CaptureSessionManager
from fdm.services.capture_subscription import CaptureState, CaptureSubscription
from fdm.services.microview_transport import FrameReceiver


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class Manager:
    def __init__(self):
        self.subscriptions = {}
        self.opens = self.closes = 0
    def acquire_session(self, owner, on_frame, on_state):
        sub = CaptureSubscription(self, owner, on_frame, on_state)
        self.subscriptions[owner] = sub
        self.opens += 1
        return sub
    def _release_subscription(self, sub):
        self.subscriptions.pop(sub.owner_id)
        self.closes += 1
    def frame(self, seq, generation=1, width=16):
        frame = FrameHandle.snapshot(np.full((12, width, 3), 70, np.uint8),
            source_id="microview:0", sequence=seq, generation=generation)
        for sub in self.subscriptions.values():
            sub.on_frame(frame)
        return frame


def test_same_pixels_are_new_owned_arrivals_and_close_is_idempotent(app):
    manager, received, errors = Manager(), [], []
    source = CameraFrameSource(manager, received.append, errors.append)
    source.start(); source.start()
    first = manager.frame(1)
    manager.frame(2)
    assert manager.opens == 1 and len(received) == 2
    assert received[0] is first and first.frame_id != received[1].frame_id
    assert not first.pixels.flags.writeable
    source.stop(); source.stop()
    manager.frame(3)
    assert len(received) == 2 and manager.closes == 1 and not errors


def test_disconnect_and_recovery_rearm_arrival_watchdog(app):
    manager, errors = Manager(), []
    source = CameraFrameSource(manager, lambda _: None, errors.append)
    source.start()
    manager.frame(1)
    source._last_frame_at -= 4
    source._check_arrivals()
    assert errors and not source._watchdog.isActive()
    manager.frame(2)
    assert source._watchdog.isActive()
    source._on_state(CaptureState("stopping", 1))
    assert "停止" in errors[-1]
    source.stop()
    source._on_state(CaptureState("faulted", 1, message="late"))
    assert errors[-1] != "late"


def test_subscription_release_does_not_stop_an_existing_camera(app, monkeypatch):
    manager = CaptureSessionManager(refresh_on_init=False)
    monkeypatch.setattr(manager, "is_preview_active", lambda: True)
    stops = []
    monkeypatch.setattr(manager, "request_stop_preview", lambda: stops.append(True))
    a = manager.acquire_session("preview", lambda _: None, lambda _: None)
    b = manager.acquire_session("trial", lambda _: None, lambda _: None)
    a.close(); b.close(); b.close()
    assert not stops and not manager._subscriptions


def test_pending_subscription_start_is_cancelled_on_close(app, monkeypatch):
    manager = CaptureSessionManager(refresh_on_init=False)
    starts = []
    monkeypatch.setattr(manager, "start_preview", lambda: starts.append(True))
    manager.acquire_session("trial", lambda _: None, lambda _: None).close()
    app.processEvents()
    assert not starts


@pytest.mark.parametrize("channels,format_value", [(1, 0), (3, 24), (4, 32)])
def test_microview_skip_pixel_is_row_prefix_and_output_is_owned(channels, format_value):
    width, height, prefix = 7, 4, 2
    data = np.arange(height * width * channels, dtype=np.uint8)
    buffer = ctypes.create_string_buffer(data.tobytes())
    info = SimpleNamespace(Width=width, Heigth=height, Length=len(data), nColor=format_value, SkipPixel=prefix)
    pixels = decode_microview_buffer(buffer, info)
    expected = data.reshape(height, width, channels)[:, prefix:, :].copy()
    if channels == 1:
        expected = expected[..., 0]
    if channels == 4:
        expected[..., 3] = 255
    np.testing.assert_array_equal(pixels, expected)
    ctypes.memset(ctypes.addressof(buffer), 0, len(buffer))
    np.testing.assert_array_equal(pixels, expected)


@pytest.mark.parametrize("format_value,word,expected", [(5, 0xf800, [0,0,255]), (5, 0x07e0, [0,255,0]),
    (1, 0x7c00, [0,0,255]), (1, 0x03e0, [0,255,0])])
def test_packed_rgb_formats(format_value, word, expected):
    data = np.array([word] * 8, dtype="<u2").tobytes()
    info = SimpleNamespace(Width=4, Heigth=2, Length=len(data), nColor=format_value, SkipPixel=1)
    result = decode_microview_buffer(data, info)
    assert result.shape == (2,3,3) and result[0,0].tolist() == expected


@pytest.mark.parametrize("values", [{"Length":100}, {"Width":100}, {"nColor":9}, {"SkipPixel":4}, {"SkipPixel":-1}])
def test_invalid_metadata(values):
    properties = dict(Width=4, Heigth=2, Length=8, nColor=0, SkipPixel=0)
    properties.update(values)
    with pytest.raises(ValueError):
        decode_microview_buffer(b"12345678", SimpleNamespace(**properties))


@pytest.mark.parametrize("pixel_format,bpp", [(0,1), (1,2), (2,3), (3,4), (5,2), (6,2)])
def test_conversion_into_slot_matches_owned_conversion(pixel_format, bpp):
    source = np.random.default_rng(25).integers(0,256,(7,13*bpp),np.uint8).tobytes()
    info = SimpleNamespace(Width=13, Heigth=7, Length=len(source), nColor=pixel_format, SkipPixel=3)
    expected = decode_microview_buffer(source, info)
    slot = bytearray(expected.nbytes)
    shape = copy_microview_to_shared(source, info, slot)
    np.testing.assert_array_equal(np.ndarray(shape,np.uint8,buffer=slot),expected)


def test_receiver_owns_pixels_before_reuse_and_rejects_replay():
    memory = shared_memory.SharedMemory(create=True, size=12)
    receiver = FrameReceiver()
    try:
        memory.buf[:12] = bytes(range(12))
        receiver.attach({"name":memory.name,"capacity":12})
        packet = dict(sequence=1,length=12,width=3,height=2,color=8,skip=1)
        frame = receiver.receive(packet)
        assert frame.tolist() == [[1,2],[4,5]]
        memory.buf[:12] = b'\0' * 12
        assert frame.tolist() == [[1,2],[4,5]]
        with pytest.raises(ValueError): receiver.receive(packet)
        with pytest.raises(ValueError): receiver.receive({**packet,"sequence":2,"length":13})
    finally:
        receiver.close(); memory.close(); memory.unlink()


def test_receiver_drops_only_proven_superseded_slot_once_before_decode(monkeypatch):
    """Parser test; atomic marker writes are separately tested with the real DLL."""
    import struct
    from fdm.services import microview_transport
    reads = []
    def read_marker(buffer):
        reads.append(True)
        return struct.unpack_from("<Q", buffer)[0]
    monkeypatch.setattr(microview_transport, "load_bridge", lambda: SimpleNamespace(read_latest_marker=read_marker))
    memory = shared_memory.SharedMemory(create=True, size=24)
    receiver = FrameReceiver()
    try:
        memory.buf[:12] = bytes(range(12))
        struct.pack_into("<Q", memory.buf, 16, 1)
        receiver.attach({"name": memory.name, "capacity": 12, "latest_marker_offset": 16})
        packet = dict(sequence=1, length=12, width=3, height=2, color=8, skip=1, received_at=-1000)
        # A current frame from a slow/stopped source has no arbitrary age limit.
        owned = receiver.receive(packet)
        assert owned.tolist() == [[1, 2], [4, 5]] and len(reads) == 1
        struct.pack_into("<Q", memory.buf, 16, 3)
        assert receiver.receive({**packet, "sequence": 2}) is None
        assert receiver.superseded_frames == 1 and len(reads) == 2
        with pytest.raises(ValueError):
            receiver.receive({**packet, "sequence": 2})
        memory.buf[:12] = b'\x07' * 12
        assert receiver.receive({**packet, "sequence": 3}).tolist() == [[7, 7], [7, 7]]
        assert len(reads) == 3  # Do not re-check after copying and starve 4K consumers.
        assert owned.tolist() == [[1, 2], [4, 5]]
    finally:
        receiver.close(); memory.close(); memory.unlink()


@pytest.mark.parametrize("offset", [8, 13, "end"])
def test_receiver_rejects_marker_overlap_misalignment_or_truncation(offset):
    memory = shared_memory.SharedMemory(create=True, size=24)
    receiver = FrameReceiver()
    try:
        if offset == "end": offset = memory.size
        with pytest.raises(ValueError, match="协议"):
            receiver.attach({"name": memory.name, "capacity": 12, "latest_marker_offset": offset})
        assert receiver.memory is None
    finally:
        receiver.close(); memory.close(); memory.unlink()
