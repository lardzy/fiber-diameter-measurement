"""Frame adapters with explicit ownership, arrival identity and stop boundaries."""
from __future__ import annotations

import importlib
import json
import sys
import sysconfig
from pathlib import Path
from importlib.machinery import EXTENSION_SUFFIXES
import threading
import time
import uuid

from PySide6.QtCore import QObject, QProcess, QTimer

from .capture_helper import MicroviewMemoryClient
from .frames import FrameHandle, frame_from_qimage


def _create_external_capture(root_hwnd: int, child_hwnd: int):
    """Select compatibility only for a genuinely unavailable startup capability."""
    if sys.platform != "win32":
        raise RuntimeError("外部 CU 实时预览采集需要 Windows。")
    from .window_capture_helper import PrintWindowCaptureClient
    if sys.getwindowsversion().build < 18362:
        return PrintWindowCaptureClient(root_hwnd, child_hwnd)
    from fdm import _native
    name = "fdm._native._windows_capture"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        stale = [p for directory in _native.__path__ for p in Path(directory).glob("_windows_capture.*")
                 if p.suffix.lower() in {".pyd", ".so"}]
        if stale or getattr(sys, "frozen", False):
            raise RuntimeError("Windows 捕获扩展缺失或 ABI 不匹配，请重新构建。") from exc
        return PrintWindowCaptureClient(root_hwnd, child_hwnd)
    info = module.build_info()
    abi = sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag
    if (info.get("api_version") != 1 or info.get("python_abi") != abi or
            not any(str(module.__file__).endswith(suffix) for suffix in EXTENSION_SUFFIXES)):
        raise RuntimeError("Windows 捕获扩展接口或 ABI 不匹配，请重新构建。")
    root = _native._source_root()
    if root is not None and info.get("source_sha256") != _native.native_source_sha256(root):
        raise RuntimeError("Windows 捕获扩展与源码不一致，请重新构建。")
    try:
        return module.WindowCapture(root_hwnd, child_hwnd)
    except RuntimeError as exc:
        if str(exc) != "windows_graphics_capture_not_supported":
            raise
        return PrintWindowCaptureClient(root_hwnd, child_hwnd)


class CameraFrameSource(QObject):
    """Reuse a manager stream, or take exclusive ownership of Microview memory IO.

    start/stop are GUI-thread lifecycle methods. Source callbacks may arrive on
    the GUI or the dedicated helper-monitor thread; consumers marshal UI work.
    Existing frame-stream previews stay open when this subscriber stops.
    """
    def __init__(self, capture_manager, frame_callback, error_callback):
        super().__init__()
        self.manager = capture_manager
        self.frame_callback = frame_callback
        self.error_callback = error_callback
        self._active = False
        self._epoch = 0
        self._cursor = None
        self._shape = None
        self._generation = 0
        self._source_id = "camera:" + uuid.uuid4().hex
        self._stop_event = threading.Event()
        self._thread = None
        self._retiring_client = None
        self._client_factory = MicroviewMemoryClient
        self._owns_preview = False
        self._connected = False
        self._handoff_process = None
        self._handoff_slot = None
        self._last_frame_at = 0.0
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(500)
        self._watchdog.timeout.connect(self._check_arrivals)

    def start(self) -> None:
        if self._active:
            return
        if ((self._thread is not None and self._thread.is_alive()) or
                self._retiring_client is not None or self._owns_preview):
            self.error_callback("上一次相机来源仍在退出，请稍候。")
            return
        self._active = True
        self._epoch += 1
        epoch = self._epoch
        self._stop_event.clear()
        self._cursor = self._shape = None
        self._generation += 1
        self._last_frame_at = time.perf_counter()
        QTimer.singleShot(0, self, lambda: self._start_on_gui(epoch))

    def _start_on_gui(self, epoch: int) -> None:
        if not self._active or epoch != self._epoch:
            return
        try:
            device = self.manager.selected_device()
            if device is None:
                raise RuntimeError("请先在采集设置中选择相机。")
            if self.manager.preview_kind() == "native_embed" or device.backend_key == "microview":
                self._take_microview_ownership(device, epoch)
                return
            self.manager.frameReady.connect(self._on_frame)
            self.manager.errorOccurred.connect(self._on_error)
            self.manager.activeDeviceLost.connect(self._on_error)
            self._connected = True
            if not self.manager.is_preview_active():
                self._owns_preview = True
                if not self.manager.start_preview():
                    raise RuntimeError("相机预览未能启动。")
            self._watchdog.start()
            self._on_frame()
        except Exception as exc:
            self._on_error(str(exc))

    def _take_microview_ownership(self, device, epoch: int) -> None:
        active = getattr(self.manager, "_active_capture", None)
        process = getattr(getattr(active, "backend", None), "_preview_process", None)
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            # Ask the existing isolated preview to leave and await its finished
            # signal. No GUI-thread waitForFinished, and never open the board
            # while another helper still owns its handle.
            self._handoff_process = process
            self._handoff_slot = lambda *_: self._finish_handoff(device, epoch)
            process.finished.connect(self._handoff_slot)
            process.write(json.dumps({"command": "stop"}, allow_nan=False).encode() + b"\n")
            process.closeWriteChannel()
            QTimer.singleShot(2000, self, lambda: self._handoff_timeout(process, epoch, False))
        else:
            self._finish_handoff(device, epoch)

    def _handoff_timeout(self, process, epoch: int, killed: bool) -> None:
        if epoch != self._epoch or not self._active or self._handoff_process is not process:
            return
        if process.state() == QProcess.ProcessState.NotRunning:
            return
        if not killed:
            process.terminate()
            QTimer.singleShot(1000, self, lambda: self._handoff_timeout(process, epoch, True))
        else:
            process.kill()
            QTimer.singleShot(1000, self, lambda: self._handoff_failed(process, epoch))

    def _handoff_failed(self, process, epoch: int) -> None:
        if epoch == self._epoch and self._active and self._handoff_process is process:
            if process.state() != QProcess.ProcessState.NotRunning:
                self._on_error("旧 Microview 预览未确认退出，未打开第二个设备句柄。")

    def _finish_handoff(self, device, epoch: int) -> None:
        if epoch != self._epoch or not self._active:
            return
        self._disconnect_handoff()
        try:
            if self.manager.is_preview_active() and not self.manager.stop_preview():
                raise RuntimeError("旧 Microview 预览未确认退出，无法接管设备。")
            self._thread = threading.Thread(target=self._memory_loop, args=(int(device.native_id), epoch),
                                            name="fdm-live-camera", daemon=True)
            self._thread.start()
        except Exception as exc:
            self._on_error(str(exc))

    def _disconnect_handoff(self) -> None:
        if self._handoff_process is not None and self._handoff_slot is not None:
            try:
                self._handoff_process.finished.disconnect(self._handoff_slot)
            except (RuntimeError, TypeError):
                pass
        self._handoff_process = self._handoff_slot = None

    def _memory_loop(self, index: int, epoch: int) -> None:
        client = self._client_factory(index)
        frame = packet = None
        last_shape = None
        generation = (epoch << 32) + self._generation
        try:
            client.start()
            while not self._stop_event.is_set() and epoch == self._epoch:
                packet = client.poll(.1)
                if packet is None:
                    continue
                shape = tuple(packet["pixels"].shape)
                if last_shape is not None and shape != last_shape:
                    generation += 1
                last_shape = shape
                frame = FrameHandle.snapshot(packet["pixels"], source_id=self._source_id,
                    generation=generation, sequence=packet["sequence"], received_at=packet["received_at"],
                    transfer_ownership=True, metadata={"backend": "microview_memory", "raw_camera": True,
                        **{key: packet[key] for key in ("acquisition_reserved_bytes", "shared_slot_bytes",
                           "sdk_raw_bytes", "decoded_frame_bytes", "requested_fps_limit") if key in packet}})
                if not self._stop_event.is_set() and epoch == self._epoch:
                    self.frame_callback(frame)
                # The callback owns accepted frames. Rejected frames must die
                # before poll allocates the next delivery, not one loop later.
                del frame, packet
        except Exception as exc:
            if not self._stop_event.is_set() and epoch == self._epoch:
                self.error_callback(str(exc))
        finally:
            frame = packet = None
            try:
                client.close()
            except Exception as exc:
                # close() retains a live process and its shared-memory handles
                # on failure. Keep the client reachable after this thread exits
                # so stop() can retry without forgetting the SDK owner.
                self._retiring_client = client
                if epoch == self._epoch:
                    self.error_callback(str(exc))

    def _on_frame(self, *_ignored) -> None:
        if not self._active:
            return
        snapshot = self.manager.latest_preview_frame()
        if snapshot is None:
            return
        cursor = snapshot.generation, snapshot.sequence
        if cursor == self._cursor:
            return
        shape = snapshot.image.width(), snapshot.image.height()
        if self._cursor is not None and (cursor[0] != self._cursor[0] or shape != self._shape):
            self._generation += 1
        self._cursor, self._shape = cursor, shape
        self._last_frame_at = time.perf_counter()
        try:
            frame = frame_from_qimage(snapshot.image, source_id=self._source_id,
                generation=self._generation, sequence=snapshot.sequence, received_at=snapshot.received_at,
                metadata={"backend": "capture_manager", "raw_camera": True,
                          "capture_generation": snapshot.generation})
            self.frame_callback(frame)
        except Exception as exc:
            self._on_error(str(exc))

    def _check_arrivals(self) -> None:
        if self._active and time.perf_counter() - self._last_frame_at > 3:
            self._watchdog.stop()
            self.error_callback("相机没有新的帧到达；已暂停实时测量，请检查设备。")

    def _on_error(self, message: str) -> None:
        if self._active:
            self.error_callback(message)

    def stop(self) -> bool:
        """Cancel delivery; True only once every source-owned resource is free.

        A False result keeps the owner reachable and forbids replacing it.
        Calling stop again retries cleanup. A pre-existing manager preview is
        owned by its original subscriber and is deliberately left running.
        """
        self._active = False
        self._epoch += 1
        self._stop_event.set()
        self._watchdog.stop()
        self._disconnect_handoff()
        if self._connected:
            for signal, slot in ((self.manager.frameReady, self._on_frame),
                                 (self.manager.errorOccurred, self._on_error),
                                 (self.manager.activeDeviceLost, self._on_error)):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            self._connected = False
        if self._thread is not None:
            self._thread.join(3)
            if self._thread.is_alive():
                self.error_callback("采集线程仍在结束 SDK 调用，暂时不能切换设备。")
                return False
            self._thread = None
        if self._retiring_client is not None:
            try:
                self._retiring_client.close()
            except Exception as exc:
                self.error_callback(str(exc))
                return False
            self._retiring_client = None
        if self._owns_preview:
            try:
                stopped = self.manager.stop_preview()
            except Exception as exc:
                self.error_callback(str(exc))
                return False
            if not stopped:
                self.error_callback("相机预览仍未确认退出，暂时不能切换设备。")
                return False
            self._owns_preview = False
        return True


class ExternalCuFrameSource:
    """Capture a pinned CU window's rendered child through WGC, never the SDK.

    Unavailable/ambiguous windows fail explicitly. A startup-only PrintWindow
    compatibility process is allowed; desktop screen-scraping is never used.
    """
    def __init__(self, frame_callback, error_callback, selector=None):
        self.frame_callback = frame_callback
        self.error_callback = error_callback
        self.selector = selector
        self._stop = threading.Event()
        self._thread = None
        self._retiring_capture = None
        self._epoch = 0
        self._source_id = "external-cu:" + uuid.uuid4().hex
        self._locator_factory = None
        self._capture_factory = None

    def start(self) -> None:
        if ((self._thread is not None and self._thread.is_alive()) or
                self._retiring_capture is not None):
            return
        self._epoch += 1
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(self._epoch,), name="fdm-cu-wgc", daemon=True)
        self._thread.start()

    def _run(self, epoch: int) -> None:
        capture = None
        frame = packet = None
        try:
            if self._capture_factory is None:
                capture_factory = _create_external_capture
            else:
                capture_factory = self._capture_factory
            from fdm.services.cu5_preview_locator import Cu5PreviewLocator
            locator = (self._locator_factory or Cu5PreviewLocator)(selector=self.selector)
            match = locator.locate()
            capture = capture_factory(match.record.root_hwnd, match.hwnd)
            identity = dict(capture.identity())
            last_arrival = time.perf_counter()
            last_sequence = None
            while not self._stop.is_set() and epoch == self._epoch:
                packet = capture.poll(100)
                if packet is None:
                    if time.perf_counter() - last_arrival > 3:
                        raise TimeoutError("CU 预览没有新的捕获帧到达；不能确认实时性，已暂停测量。")
                    continue
                sequence = int(packet["sequence"])
                if last_sequence is not None and sequence <= last_sequence:
                    raise RuntimeError("CU 捕获帧序号未前进，已暂停测量。")
                last_sequence = sequence
                last_arrival = time.perf_counter()
                frame = FrameHandle.snapshot(packet["pixels"], source_id=self._source_id,
                    generation=(epoch << 32) + int(packet["generation"]), sequence=sequence,
                    received_at=last_arrival, transfer_ownership=True,
                    metadata={**identity, "backend": packet.get("backend", "windows_graphics_capture"), "raw_camera": False,
                              "rendered_preview": True, "dpi": packet["dpi"], "crop": packet["crop"],
                              "acquisition_reserved_bytes": packet.get("acquisition_reserved_bytes", packet["pixels"].nbytes * 2),
                              **{key: packet[key] for key in ("gpu_pool_buffers", "gpu_pool_bytes",
                                 "cpu_staging_row_pitch", "cpu_staging_bytes", "next_delivery_bytes") if key in packet},
                              "source_timestamp_100ns": packet["source_timestamp_100ns"]})
                if not self._stop.is_set() and epoch == self._epoch:
                    self.frame_callback(frame)
                del frame, packet
        except Exception as exc:
            if not self._stop.is_set() and epoch == self._epoch:
                self.error_callback(str(exc))
        finally:
            frame = packet = None
            if capture is not None:
                try:
                    capture.close()
                except Exception as exc:
                    # WGC closes on its owner thread and cannot throw. A
                    # compatibility helper may still own a live process; keep
                    # that client for a subsequent, explicit stop retry.
                    self._retiring_capture = capture
                    if not self._stop.is_set() and epoch == self._epoch:
                        self.error_callback(str(exc))

    def stop(self) -> bool:
        """Return True only after the capture thread/helper has released ownership."""
        self._stop.set()
        self._epoch += 1
        if self._thread is not None:
            self._thread.join(2)
            if self._thread.is_alive():
                self.error_callback("Windows 捕获仍在释放资源，请稍候再启动来源。")
                return False
            self._thread = None
        if self._retiring_capture is not None:
            try:
                self._retiring_capture.close()
            except Exception as exc:
                self.error_callback(str(exc))
                return False
            self._retiring_capture = None
        return True
