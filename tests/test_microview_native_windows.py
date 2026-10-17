"""Run against a compiled SDK test DLL, never a Python callback substitute."""
import ctypes
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows SDK ABI')


@pytest.fixture
def sdk():
    path=os.environ.get('FDM_TEST_MICROVIEW_DLL')
    if not path or not Path(path).is_file(): pytest.skip('Compile tests/native/microview_test_sdk.cpp first')
    dll=ctypes.WinDLL(path)
    dll.MV_OpenDevice.argtypes=[ctypes.c_uint32,ctypes.c_int32]; dll.MV_OpenDevice.restype=ctypes.c_void_p
    dll.MV_CloseDevice.argtypes=[ctypes.c_void_p]; dll.MV_CloseDevice.restype=None
    dll.MV_TestEmit.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32]
    dll.MV_TestEmit.restype=ctypes.c_int32
    dll.MV_TestGetCounter.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    dll.MV_TestGetCounter.restype=ctypes.c_uint32
    dll.MV_TestIgnoreAfterReturn.argtypes=[ctypes.c_void_p,ctypes.c_int32]
    dll.MV_TestIgnoreAfterReturn.restype=None
    dll.MV_TestConfigure.argtypes=[ctypes.c_void_p,*([ctypes.c_uint32]*4),ctypes.c_int32]
    dll.MV_OperateDevice.argtypes=[ctypes.c_void_p,ctypes.c_int]; dll.MV_OperateDevice.restype=ctypes.c_int
    handle=dll.MV_OpenDevice(0,True)
    yield dll,handle
    dll.MV_CloseDevice(handle)


def test_real_callback_preserves_64bit_addresses_prefix_and_formats(sdk):
    from fdm._native import _microview
    from fdm.services.capture import MicroviewCaptureBackend
    from fdm.live_experiment.capture_helper import decode_microview_buffer
    dll,handle=sdk
    tap=_microview.FrameTap(dll._handle,handle,512,4)
    assert handle>2**32 and tap.diagnostics()['context_address']>2**32
    tap.start()
    try:
        for color,bpp in [(8,1),(24,3),(32,4)]:
            source=ctypes.create_string_buffer(bytes(range(8*4*bpp)))
            assert ctypes.addressof(source)>2**32
            info=MicroviewCaptureBackend._MVImageInfo()
            info.Width,info.Heigth,info.Length,info.nColor,info.SkipPixel=8,4,8*4*bpp,color,2
            assert dll.MV_TestEmit(handle,source,ctypes.byref(info),9)
            destination=bytearray(512); packet=tap.poll_into(destination)
            assert packet['index']==9 and packet['skip']==2 and packet['width']==8
            assert packet['callback_stage']=='after_process'
            result=decode_microview_buffer(destination,info)
            assert result.shape[:2]==(4,6)
            assert int(result.reshape(4,-1)[0,0])==((2*bpp)^0x40)
            assert source.raw[:info.Length]==bytes(range(8*4*bpp))
            assert tap.poll_into(destination) is None
        assert dll.MV_TestGetCounter(handle,0)==0  # No raw callback was registered.
        assert dll.MV_TestGetCounter(handle,1)==3
        assert dll.MV_TestGetCounter(handle,2)==3
        assert dll.MV_TestGetCounter(handle,3)==3
        info.Heigth=2
        dll.MV_TestEmit(handle,source,ctypes.byref(info),10)
        assert tap.poll_into(destination) is None and tap.diagnostics()['fields_rejected']==1
    finally:
        tap.stop(); tap.wait_idle()
    dll.MV_TestEmit(handle,source,ctypes.byref(info),11)
    assert tap.poll_into(destination) is None


def test_vendor_variant_ignores_after_return_but_processed_frames_continue(sdk):
    """The native HWND must also be hidden; FALSE alone is not that guarantee."""
    from fdm._native import _microview
    from fdm.services.capture import MicroviewCaptureBackend
    dll,handle=sdk
    dll.MV_TestIgnoreAfterReturn(handle,True)
    tap=_microview.FrameTap(dll._handle,handle,64,2,False)
    info=MicroviewCaptureBackend._MVImageInfo(16,32,2,2,0)
    source=ctypes.create_string_buffer(bytes(range(16)))
    tap.start()
    try:
        assert dll.MV_TestEmit(handle,source,ctypes.byref(info),0)
        destination=bytearray(64)
        assert tap.poll_into(destination) is not None
        assert destination[:16]==bytes(value^0x40 for value in range(16))
        assert dll.MV_TestGetCounter(handle,3)==1
        assert tap.diagnostics()['native_preview_visible'] is False
    finally:
        tap.stop(); tap.wait_idle()
    dll.MV_TestEmit(handle,source,ctypes.byref(info),11)
    assert tap.poll_into(destination) is None


def test_processed_tap_switches_display_owner_without_stopping_frames(sdk):
    from fdm._native import _microview
    from fdm.services.capture import MicroviewCaptureBackend
    dll,handle=sdk
    # Start a trial with SDK drawing already disabled; it must not draw even
    # its first frame. The source and processed pixels remain distinct.
    tap=_microview.FrameTap(dll._handle,handle,64,2,False)
    info=MicroviewCaptureBackend._MVImageInfo(16,32,2,2,0)
    source=ctypes.create_string_buffer(bytes(range(16)))
    destination=bytearray(64)
    tap.start()
    try:
        assert not dll.MV_TestEmit(handle,source,ctypes.byref(info),0)
        packet=tap.poll_into(destination)
        assert packet and destination[:16]==bytes(value^0x40 for value in range(16))
        assert dll.MV_TestGetCounter(handle,3)==0
        tap.set_native_preview_visible(True)
        assert dll.MV_TestEmit(handle,source,ctypes.byref(info),1)
        assert dll.MV_TestGetCounter(handle,3)==1
        assert dll.MV_TestGetCounter(handle,4)==destination[0]
        assert tap.poll_into(destination)['sequence']>packet['sequence']
        tap.set_native_preview_visible(False)
        for index in range(3):
            assert not dll.MV_TestEmit(handle,source,ctypes.byref(info),index+2)
            assert tap.poll_into(destination) is not None
        assert dll.MV_TestGetCounter(handle,1)==5
        assert dll.MV_TestGetCounter(handle,2)==5
        assert dll.MV_TestGetCounter(handle,3)==1
        assert tap.diagnostics()['native_preview_visible'] is False
    finally:
        tap.stop(); tap.wait_idle()


def test_real_callback_resize_and_stop_during_delivery(sdk):
    from fdm._native import _microview
    dll,handle=sdk
    dll.MV_TestConfigure(handle,768,576,3,100,False)
    tap=_microview.FrameTap(dll._handle,handle,128,576); tap.start()
    dll.MV_OperateDevice(handle,1)
    try:
        deadline=time.perf_counter()+2
        while not tap.diagnostics()['required_capacity'] and time.perf_counter()<deadline: time.sleep(.005)
        required=tap.diagnostics()['required_capacity']
        assert required==768*576*4
        # A full measurement tap does not freeze the SDK's native preview.
        assert dll.MV_TestGetCounter(handle,3)>0
        tap.reserve(required)
        dest=bytearray(required)
        deadline=time.perf_counter()+2
        packet=None
        while packet is None and time.perf_counter()<deadline:
            packet=tap.poll_into(dest); time.sleep(.005)
        assert packet and packet['skip']==3
        tap.stop(); tap.wait_idle()
        before=tap.diagnostics()['callbacks']
        time.sleep(.05)
        assert tap.diagnostics()['callbacks']==before
    finally:
        tap.stop(); dll.MV_OperateDevice(handle,0); tap.wait_idle()


def test_native_latest_marker_retains_buffer_export_and_detaches_safely(sdk):
    from fdm._native import _microview
    from fdm.services.capture import MicroviewCaptureBackend
    dll, handle = sdk
    tap = _microview.FrameTap(dll._handle, handle, 64, 2, False)
    marker, replacement = bytearray(8), bytearray(8)
    view = memoryview(marker)
    info = MicroviewCaptureBackend._MVImageInfo(6, 8, 2, 3, 0)
    source = ctypes.create_string_buffer(bytes(range(6)))
    tap.bind_latest_marker(view)
    tap.start()
    try:
        dll.MV_TestEmit(handle, source, ctypes.byref(info), 0)
        assert _microview.read_latest_marker(marker) == 1
        with pytest.raises(BufferError): marker.extend(b'\0')
        with pytest.raises(BufferError): view.release()
        tap.bind_latest_marker(replacement)
        view.release(); marker.extend(b'\0')  # Old export was actually retired.
        assert _microview.read_latest_marker(replacement) == 1
        dll.MV_TestEmit(handle, source, ctypes.byref(info), 1)
        assert int.from_bytes(marker[:8], 'little') == 1
        assert _microview.read_latest_marker(replacement) == 2
        tap.stop(); tap.wait_idle()
        dll.MV_TestEmit(handle, source, ctypes.byref(info), 2)
        assert _microview.read_latest_marker(replacement) == 2
    finally:
        tap.stop(); tap.wait_idle(); tap.bind_latest_marker(None)
    replacement.extend(b'\0')


def test_native_event_retains_coalesced_frames_control_capacity_and_stop(sdk):
    from fdm._native import _microview
    from fdm.services.capture import MicroviewCaptureBackend
    dll, handle = sdk
    tap = _microview.FrameTap(dll._handle, handle, 64, 2, False)
    tap.start()
    source = ctypes.create_string_buffer(bytes(range(80)))
    info = MicroviewCaptureBackend._MVImageInfo(6, 8, 2, 3, 0)
    try:
        assert not tap.wait(0)
        for index in range(3): dll.MV_TestEmit(handle, source, ctypes.byref(info), index)
        assert tap.wait(0) and not tap.wait(0)  # A pre-wait burst remains signalled once.
        assert tap.poll_into(bytearray(64))['index'] == 2
        info.Length, info.Width = 80, 40
        dll.MV_TestEmit(handle, source, ctypes.byref(info), 3)
        assert tap.wait(0) and tap.diagnostics()['required_capacity'] == 80
        assert not tap.wait(0)
        signal = threading.Thread(target=lambda: (time.sleep(.02), tap.wake()))
        signal.start()
        assert tap.wait(1000)  # Releases the GIL so the stdin/control thread can wake it.
        signal.join()
        assert not tap.wait(0)
        tap.stop()
        assert tap.wait(0)  # Stop also interrupts a no-frame wait.
        assert not tap.wait(0)
    finally:
        tap.stop(); tap.wait_idle()


def test_native_latest_marker_skips_pending_slot_and_survives_capacity_change(sdk):
    from fdm._native import _microview
    from fdm.services.capture import MicroviewCaptureBackend
    from fdm.services.microview_transport import FramePublisher, FrameReceiver
    dll, handle = sdk
    tap = _microview.FrameTap(dll._handle, handle, 64, 2, False)
    backend = SimpleNamespace(_frame_tap=tap, _frame_tap_capacity=64)
    messages = []
    publisher = FramePublisher(backend, messages.append)
    receiver = FrameReceiver()
    receiver.attach(messages[-1])
    tap.start()
    def emit(index, value, width=3):
        info = MicroviewCaptureBackend._MVImageInfo(width * 2, 8, 2, width, 0)
        source = ctypes.create_string_buffer(bytes([value]) * info.Length)
        dll.MV_TestEmit(handle, source, ctypes.byref(info), index)
    try:
        emit(0, 1); assert publisher.publish()
        first = messages[-1]
        owned = receiver.receive(first)
        assert owned.tolist() == [[65] * 3] * 2
        publisher.acknowledge(first['sequence'])
        emit(1, 2); assert publisher.publish()
        pending = messages[-1]
        emit(2, 3)  # GUI and helper have not consumed/ACKed the pending slot.
        assert not publisher.publish()
        assert receiver.receive(pending) is None and receiver.superseded_frames == 1
        publisher.acknowledge(pending['sequence'])
        assert publisher.publish()
        latest = messages[-1]
        # No age threshold: current source pixels remain valid indefinitely.
        assert receiver.receive({**latest, 'received_at': -1000}).tolist() == [[67] * 3] * 2
        publisher.acknowledge(latest['sequence'])
        assert owned.tolist() == [[65] * 3] * 2
        old_name = publisher.memory.name
        emit(3, 4, width=40)
        assert tap.diagnostics()['required_capacity'] == 80
        assert not publisher.publish()  # Reserve and rebind without a successful larger callback yet.
        assert messages[-1]['type'] == 'buffer' and publisher.memory.name != old_name
        receiver.attach(messages[-1])
        emit(4, 5, width=40); assert publisher.publish()
        packet = messages[-1]
        assert receiver.receive(packet).tolist() == [[69] * 40] * 2
        publisher.acknowledge(packet['sequence'])
        tap.stop(); tap.wait_idle()
        before = _microview.read_latest_marker(receiver._latest_marker)
        emit(5, 6, width=40)
        assert _microview.read_latest_marker(receiver._latest_marker) == before
    finally:
        tap.stop(); tap.wait_idle(); receiver.close(); publisher.close()


def test_real_backend_reuses_standard_initialization_and_stops_20_times(sdk,monkeypatch):
    from fdm.services.capture import MicroviewCaptureBackend,CaptureDevice
    dll,_=sdk
    monkeypatch.setenv('FDM_MICROVIEW_DLL_DIR',str(Path(os.environ['FDM_TEST_MICROVIEW_DLL']).parent))
    target=SimpleNamespace(native_window_id=lambda:1,width=lambda:768,height=lambda:576)
    # Match the target contract without a Qt image/window in the native callback.
    target.winId=lambda:1
    for _ in range(20):
        backend=MicroviewCaptureBackend(frame_tap=True)
        backend.start_preview(CaptureDevice('microview:0','Test SDK','microview',0),
            preview_target=target,frame_callback=lambda _:None,error_callback=lambda _:None)
        deadline=time.perf_counter()+2
        dest=bytearray(backend._frame_tap_capacity); packet=None
        while packet is None and time.perf_counter()<deadline:
            packet=backend._frame_tap.poll_into(dest); time.sleep(.005)
        assert packet and packet['width']==768 and packet['height']==576
        backend.stop_preview()
        assert backend._frame_tap is None and backend._device_handle is None


def test_hidden_live_native_hwnd_keeps_processed_frames_and_never_targets_trial(sdk,monkeypatch,desktop_application):
    from PySide6.QtWidgets import QWidget
    from PySide6.QtTest import QTest
    from fdm.services.capture import MicroviewCaptureBackend,CaptureDevice
    dll,_=sdk
    monkeypatch.setenv('FDM_MICROVIEW_DLL_DIR',str(Path(os.environ['FDM_TEST_MICROVIEW_DLL']).parent))
    native=QWidget(); native.resize(768,576)
    trial=QWidget(); trial.resize(768,576); trial.show()
    native_id=int(native.winId())
    trial_id=int(trial.winId())
    user32=ctypes.WinDLL('user32',use_last_error=True)
    user32.IsWindow.argtypes=[ctypes.c_void_p]; user32.IsWindow.restype=ctypes.c_int32
    user32.IsWindowVisible.argtypes=[ctypes.c_void_p]; user32.IsWindowVisible.restype=ctypes.c_int32
    assert user32.IsWindow(native_id) and not user32.IsWindowVisible(native_id)
    backend=MicroviewCaptureBackend(frame_tap=True)
    backend.set_native_preview_visible(False)
    try:
        backend.start_preview(CaptureDevice('microview:0','Test SDK','microview',0),
            preview_target=native,frame_callback=lambda _:None,error_callback=lambda _:None)
        handle=backend._device_handle
        dll.MV_TestIgnoreAfterReturn(handle,True)
        def until(predicate):
            deadline=time.perf_counter()+3
            while not predicate() and time.perf_counter()<deadline: QTest.qWait(5)
            assert predicate()
        until(lambda:dll.MV_TestGetCounter(handle,2)>=3)
        assert dll.MV_TestGetCounter(handle,3)>0  # DLL ignored FALSE, as SAA7130 does.
        assert dll.MV_TestGetCounter(handle,5)==0  # Dedicated native HWND is hidden.
        assert dll.MV_TestGetCounter(handle,6)==native_id and native_id!=trial_id
        assert dll.MV_TestGetCounter(handle,7)==1  # Processing was not switched off.
        dest=bytearray(backend._frame_tap_capacity)
        assert backend._frame_tap.poll_into(dest)['callback_stage']=='after_process'
        backend.set_native_preview_visible(True)
        native.show()
        until(lambda:dll.MV_TestGetCounter(handle,5)>0)
        assert int(native.winId())==native_id and user32.IsWindow(native_id)
        native.hide(); backend.set_native_preview_visible(False)
        visible=dll.MV_TestGetCounter(handle,5)
        after=dll.MV_TestGetCounter(handle,2)
        until(lambda:dll.MV_TestGetCounter(handle,2)>=after+3)
        assert dll.MV_TestGetCounter(handle,5)==visible
        assert dll.MV_TestGetCounter(handle,6)==native_id
    finally:
        backend.stop_preview(); native.close(); trial.close()


def test_isolated_helper_shares_one_session_and_recovers_crash(sdk,monkeypatch,desktop_application):
    from PySide6.QtWidgets import QWidget
    from PySide6.QtTest import QTest
    from fdm.services.capture import CaptureSessionManager,MicroviewIsolatedBackend,CaptureDevice
    from fdm.live_experiment.sources import CameraFrameSource
    path=Path(os.environ['FDM_TEST_MICROVIEW_DLL'])
    monkeypatch.setenv('FDM_MICROVIEW_DLL_DIR',str(path.parent))
    backend=MicroviewIsolatedBackend()
    manager=CaptureSessionManager(backends=[backend],refresh_on_init=False)
    device=CaptureDevice('microview:0','Test SDK','microview',0)
    manager._devices=[device]; manager._selected_device_id=device.id
    target=QWidget(); target.resize(768,576); target.show()
    errors=[]; manager.errorOccurred.connect(errors.append)
    frames=[]; source=CameraFrameSource(manager,frames.append,errors.append)
    def until(predicate,seconds=5):
        deadline=time.perf_counter()+seconds
        while not predicate() and time.perf_counter()<deadline: QTest.qWait(5)
        assert predicate(), errors
    try:
        assert manager.start_preview(preview_target=target)
        assert manager.preview_kind()=='native_embed'
        source.start(); until(lambda:len(frames)>=5)
        process=backend._preview_process
        owned=frames[0]; saved=owned.pixels.copy()
        source.stop(); source.start(); until(lambda:len(frames)>=8)
        assert backend._preview_process is process
        import numpy as np
        np.testing.assert_array_equal(owned.pixels,saved)
        assert not owned.pixels.flags.writeable
        manager.set_native_preview_visible(False)
        target.hide()
        until(lambda:frames[-1].metadata['diagnostics']['native_preview_visible'] is False)
        assert frames[-1].metadata['sdk_callback_stage']=='after_process'
        count=len(frames)
        until(lambda:len(frames)>=count+3)
        manager.set_native_preview_visible(True)
        target.show()
        until(lambda:frames[-1].metadata['diagnostics']['native_preview_visible'] is True)
        process.kill(); until(lambda:not manager.is_preview_active())
        assert errors and manager._session_state in {'faulted','stopped'}
        count=len(frames)
        assert manager.start_preview(preview_target=target)
        until(lambda:len(frames)>=count+3)
        assert frames[-1].generation!=owned.generation
        for cycle in range(20):
            process=backend._preview_process
            source.stop(); source.start()
            count=len(frames)
            until(lambda:len(frames)>count)
            assert backend._preview_process is process
            started=time.perf_counter(); manager.request_stop_preview()
            assert time.perf_counter()-started<.05
            until(lambda:not manager.is_preview_active())
            if cycle<19:
                assert manager.start_preview(preview_target=target)
    finally:
        source.stop(); manager.stop_preview(); target.close()


def test_isolated_helper_stops_with_no_frames_after_control_burst(sdk, monkeypatch, desktop_application):
    from PySide6.QtWidgets import QWidget
    from PySide6.QtTest import QTest
    from fdm.services.capture import CaptureSessionManager, MicroviewIsolatedBackend, CaptureDevice
    monkeypatch.setenv('FDM_MICROVIEW_DLL_DIR', str(Path(os.environ['FDM_TEST_MICROVIEW_DLL']).parent))
    monkeypatch.setenv('FDM_TEST_MICROVIEW_NO_FRAMES', '1')
    backend = MicroviewIsolatedBackend()
    manager = CaptureSessionManager(backends=[backend], refresh_on_init=False)
    device = CaptureDevice('microview:0', 'No-frame test SDK', 'microview', 0)
    manager._devices = [device]; manager._selected_device_id = device.id
    target = QWidget(); target.resize(768, 576); target.winId()
    errors = []; manager.errorOccurred.connect(errors.append)
    try:
        assert manager.start_preview(preview_target=target)
        deadline = time.perf_counter() + 4
        while backend._frame_receiver is None and time.perf_counter() < deadline:
            QTest.qWait(5)
        assert backend._frame_receiver is not None and not errors
        for index in range(40):
            manager.set_native_preview_visible(index % 2 == 0)
            backend.update_preview_target(target)
        process = backend._preview_process
        manager.request_stop_preview()
        deadline = time.perf_counter() + 3
        while manager.is_preview_active() and time.perf_counter() < deadline:
            QTest.qWait(5)
        assert not manager.is_preview_active() and not errors
        assert process.exitCode() == 0
    finally:
        manager.stop_preview(); target.close()
