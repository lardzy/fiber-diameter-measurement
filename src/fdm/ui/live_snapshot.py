"""Grab the next complete frame from the active preview without releasing its owner."""
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from fdm.ui.workspace import WorkspacePage


@dataclass(frozen=True)
class _Request:
    project: object
    device: str
    generation: int
    after_sequence: int


class LiveSnapshotControls(QWidget):
    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.manager = host._capture_manager
        self.request = None
        self._committing = False
        self.latest_document_id = None
        self.count = 0
        self.timeout = QTimer(self)
        self.timeout.setSingleShot(True)
        self.timeout.setInterval(2500)
        self.timeout.timeout.connect(lambda: self.cancel("抓拍超时，未收到新画面。"))

        self.grab_button = QToolButton(self)
        self.grab_button.setDefaultAction(host.capture_frame_action)
        self.grab_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.grab_button.setProperty("emphasis", "primary")
        self.notice = QLabel(self)
        self.notice.setObjectName("liveSnapshotNotice")
        self.view = QToolButton(self)
        self.view.setText("查看刚拍图像")
        self.view.setAccessibleName("查看刚拍图像")
        self.view.clicked.connect(self.view_latest)
        self.view.hide()
        row = QHBoxLayout()
        row.addWidget(self.notice)
        row.addWidget(self.view)
        row.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.grab_button)
        layout.addLayout(row)
        self.manager.frameReady.connect(self._frame_ready)
        if hasattr(self.manager, "captureStateChanged"):
            self.manager.captureStateChanged.connect(self._capture_changed)

    def available(self):
        layout = self.host._adaptive_layout
        return bool(not self._committing and layout and layout.page is WorkspacePage.CAPTURE
                    and not self.host.project.is_read_only_compatible
                    and self.host._preview_analysis_mode == "none" and not self.host._digital_slide_mode
                    and self.manager.is_preview_active() and not self.manager.is_stopping()
                    and self.manager.can_capture_still())

    def _message(self, text, tone):
        self.notice.setProperty("statusTone", tone)
        self.notice.setText(text)
        self.notice.style().unpolish(self.notice)
        self.notice.style().polish(self.notice)
        self.notice.update()

    def start(self):
        if self.request is not None or not self.available():
            return
        cursor = self.manager.preview_frame_cursor()
        device = self.manager.selected_device()
        if cursor is None or device is None:
            self._message("尚未收到设备画面。", "warning")
            return
        self.request = _Request(self.host.project, device.id, *cursor)
        self.host.capture_frame_action.setEnabled(False)
        self._message("正在抓拍…", "info")
        self.view.hide()
        self.timeout.start()

    def cancel(self, reason):
        if self.request is not None:
            self.request = None
            self.timeout.stop()
            self._message(reason, "warning")
            self._restore_view_button()
            self.host.capture_frame_action.setEnabled(self.available())

    def _restore_view_button(self):
        self.view.setVisible(self.latest_document_id is not None
                             and self.host.project.get_document(self.latest_document_id) is not None)

    @staticmethod
    def _failure_message(exc):
        detail = "内存不足" if isinstance(exc, MemoryError) else str(exc) or type(exc).__name__
        return f"抓拍失败：{detail}"

    def _capture_changed(self, state):
        if state.state in {"stopping", "stopped", "faulted"}:
            self.cancel("抓拍已取消，采集会话已结束。")

    def page_changed(self, page):
        if page is not WorkspacePage.CAPTURE:
            self.cancel("已离开预览与抓拍，抓拍取消。")

    def _frame_ready(self, _image):
        pending = self.request
        if pending is None:
            return
        device = self.manager.selected_device()
        if (not self.available() or pending.project is not self.host.project or
                device is None or device.id != pending.device):
            self.cancel("抓拍已取消，采集上下文已改变。")
            return
        try:
            frame = self.manager.latest_preview_frame()
        except Exception as exc:
            self.cancel(self._failure_message(exc))
            return
        if frame is None:
            return
        if frame.generation != pending.generation:
            self.cancel("抓拍已取消，设备会话已改变。")
            return
        if frame.sequence <= pending.after_sequence or frame.image.isNull():
            return
        # Consume before constructing widgets. Re-entrant frame delivery, or a
        # failed allocation followed by a later frame, cannot retry this click.
        self.request = None
        self._committing = True
        self.timeout.stop()
        try:
            # Use the full owned source frame, independent of zoom, alignment
            # and any later camera buffers. Leave the visible workspace alone.
            owned_image = frame.image.copy()
            if owned_image.isNull():
                raise MemoryError("无法复制预览帧")
            doc = self.host._add_project_asset_image(owned_image, metadata={"capture": {
                "device_id": pending.device, "generation": frame.generation, "sequence": frame.sequence,
            }}, status_message="已抓拍到项目", activate=False)
        except Exception as exc:  # Boundary for image conversion, Qt and asset registration.
            self._message(self._failure_message(exc), "warning")
        else:
            self.latest_document_id = doc.id
            self.count += 1
            self._message(f"已抓拍第 {self.count} 张", "success")
        finally:
            self._committing = False
            self._restore_view_button()
            self.host.capture_frame_action.setEnabled(self.available())

    def view_latest(self):
        identity = self.latest_document_id
        if self.host.project.get_document(identity) is None:
            self.reset()
            return
        if self.host._activate_measure_workspace():
            self.host._set_current_document(identity)

    def reset(self):
        self.cancel("抓拍已取消")
        self.count = 0
        self.latest_document_id = None
        self.notice.clear()
        self.view.hide()
