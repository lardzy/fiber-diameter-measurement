"""Trial sources subscribe to the existing camera session; they never open SDKs."""
from __future__ import annotations

import time
import uuid
from PySide6.QtCore import QObject, QTimer


class CameraFrameSource(QObject):
    def __init__(self, capture_manager, frame_callback, error_callback):
        super().__init__()
        self.manager = capture_manager
        self.frame_callback, self.error_callback = frame_callback, error_callback
        self._subscription = None
        self._active = False
        self._connected = False
        self._last_frame_at = 0.0
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(500)
        self._watchdog.timeout.connect(self._check_arrivals)

    def start(self):
        if self._active:
            return
        self._active = True
        self._last_frame_at = time.perf_counter()
        self._subscription = self.manager.acquire_session(
            "trial:" + uuid.uuid4().hex, self._on_frame, self._on_state)
        self._connected = True
        self._watchdog.start()

    def _on_frame(self, frame):
        if self._active:
            self._last_frame_at = time.perf_counter()
            if not self._watchdog.isActive():
                self._watchdog.start()
            self.frame_callback(frame)

    def _on_state(self, state):
        if self._active and state.state in {"faulted", "stopped", "stopping"}:
            self.error_callback(state.message or {"stopped": "采集已停止", "stopping": "正在停止采集",
                                                "faulted": "采集发生错误"}[state.state])

    def _check_arrivals(self):
        if self._active and time.perf_counter() - self._last_frame_at > 3:
            self._watchdog.stop()
            self.error_callback("相机没有新的完整帧到达，测量已暂停。")

    def stop(self):
        self._active = self._connected = False
        self._watchdog.stop()
        if self._subscription is not None:
            self._subscription.close()
            self._subscription = None
        return True
