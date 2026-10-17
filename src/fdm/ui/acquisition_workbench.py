"""Stable acquisition navigation; pages subscribe to one managed device session."""
from __future__ import annotations

from dataclasses import dataclass
import json

from PySide6.QtCore import QTimer, QSize, QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabBar, QPushButton, QLabel

from fdm.ui.icons import themed_icon
from fdm.ui.workspace_stack import WorkspaceStack


@dataclass(frozen=True)
class TrialDefinition:
    id: str
    title: str


TRIALS = (TrialDefinition("content", "含量试验"), TrialDefinition("diameter", "直径试验"))


class AcquisitionWorkbench(QWidget):
    def __init__(self, host, capture_page):
        super().__init__(host)
        self.host = host
        self._selected = 0
        self._switching = False
        self._entered = False
        self._want_capture = False
        self._restart_pending = False
        self._error_message = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        row = QHBoxLayout()
        row.setContentsMargins(8, 4, 8, 4)
        self.navigation = QTabBar(self)
        self.navigation.setObjectName("acquisitionNavigation")
        self.navigation.setAccessibleName("采集与试验功能")
        self.navigation.setExpanding(False)
        self.navigation.setIconSize(QSize(20, 20))
        self.navigation.addTab("实时采集")
        for trial in TRIALS:
            self.navigation.addTab(trial.title)
        row.addWidget(self.navigation)
        row.addStretch()
        self.device_status = QLabel("采集未启动", self)
        self.device_status.setMaximumWidth(280)
        row.addWidget(self.device_status)
        self.device_button = QPushButton("设备设置", self)
        self.device_button.clicked.connect(host.show_capture_device_menu)
        row.addWidget(self.device_button)
        self.capture_button = QPushButton("开始采集", self)
        self.capture_button.clicked.connect(self.toggle_capture)
        row.addWidget(self.capture_button)
        layout.addLayout(row)
        self.feedback = QLabel(self)
        self.feedback.setObjectName("acquisitionFeedback")
        self.feedback.setWordWrap(True)
        self.feedback.setContentsMargins(12, 4, 12, 6)
        self.feedback.hide()
        layout.addWidget(self.feedback)
        self.pages = WorkspaceStack(self)
        self.capture_page = capture_page
        self.pages.addWidget(capture_page)
        layout.addWidget(self.pages, 1)
        self.navigation.currentChanged.connect(self.select_page)
        manager = host._capture_manager
        manager.previewStateChanged.connect(self._preview_state_changed)
        manager.captureStateChanged.connect(self._capture_state_changed)
        manager.devicesChanged.connect(self.refresh_capture_state)
        self.refresh_capture_state()
        self._refresh_icons()

    def _refresh_icons(self):
        color = self.palette().windowText().color().name()
        for index, icon in enumerate(("acquisition", "content_trial", "diameter_trial")):
            self.navigation.setTabIcon(index, themed_icon(icon, color=color))
        self.device_button.setIcon(themed_icon("capture_device", color=color))
        self.capture_button.setIcon(themed_icon(
            "stop_capture" if self.host._capture_manager.is_preview_active() else "start_capture", color=color))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange) and hasattr(self, "capture_button"):
            self._refresh_icons()

    @staticmethod
    def _preferences_path():
        from fdm.settings import settings_file_path
        return settings_file_path().parent / "acquisition-workbench.json"

    def restore_page(self):
        try:
            data = json.loads(self._preferences_path().read_text(encoding="utf-8"))
            index = int(data.get("page", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            index = 0
        self.select_page(index if 0 <= index <= len(TRIALS) else 0)

    def _save_page(self):
        from fdm.atomic_io import atomic_write_text
        try:
            path = self._preferences_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, json.dumps({"version": 1, "page": self._selected}), encoding="utf-8")
        except OSError:
            pass

    def enter(self, *, auto_start=True):
        first_entry = not self._entered
        self._entered = True
        if first_entry and auto_start:
            self.request_start()

    def leave(self):
        self.retire_entry()
        self.suspend_trial("已离开采集与试验")
        binding = getattr(self.host, "_preview_display_binding", None)
        if binding is not None:
            binding.flush()
        self.host.stop_live_preview()

    def retire_entry(self):
        """Invalidate delayed starts when an accepted transition retires this entry."""
        self._entered = False
        self._want_capture = False
        self._restart_pending = False

    def suspend_trial(self, reason):
        trial = self.pages.currentWidget()
        if trial is self.capture_page:
            return True
        trial.controller.cancel_draft()
        trial.controller.end_field(reason)
        trial.settings_dialog.hide()
        return trial._stop_source()

    def clear_trial_pages(self):
        """A project replacement must not leave navigation on a disposed page."""
        self._switching = True
        try:
            for index in range(self.pages.count() - 1, -1, -1):
                page = self.pages.widget(index)
                if page is not self.capture_page:
                    self.pages.removeWidget(page)
            self._selected = 0
            self.navigation.setCurrentIndex(0)
            self.pages.setCurrentWidget(self.capture_page)
        finally:
            self._switching = False

    def select_page(self, index):
        if self._switching or not 0 <= index <= len(TRIALS):
            return
        current = self.pages.currentWidget()
        if index == self._selected and (current is self.capture_page if index == 0 else
                current is getattr(self.host, "_live_experiment_workspace", None)):
            return
        self._switching = True
        try:
            binding = getattr(self.host, "_preview_display_binding", None)
            if current is self.capture_page and binding is not None:
                binding.flush()
            if not self.suspend_trial("已切换试验页面，累计记录保留"):
                self.navigation.setCurrentIndex(self._selected)
                return
            if index == 0:
                self.host._preview_ruler_key = None
                self.pages.setCurrentWidget(self.capture_page)
                frame = self.host._capture_manager.latest_preview_frame()
                if frame is not None:
                    self.host._on_live_preview_frame_ready(frame.image)
            else:
                from fdm.ui.live_experiment_workspace import open_live_experiment
                trial = open_live_experiment(self.host, embedded=True, trial_mode=TRIALS[index - 1].id)
                if trial is None:
                    self.navigation.setCurrentIndex(self._selected)
                    return
                if self.pages.indexOf(trial) < 0:
                    self.pages.addWidget(trial)
                # Retire live-preview toolbars/status before exposing the
                # taller trial page. Qt otherwise briefly combines both
                # minimum heights and permanently grows a 670px client area.
                # This also covers programmatic/restored navigation, where
                # QTabBar has not selected the requested tab yet.
                self.navigation.setCurrentIndex(index)
                self.host._sync_workspace_mode()
                self.pages.setCurrentWidget(trial)
                if self.host._capture_manager.is_preview_active() and not self.host._capture_manager.is_stopping():
                    trial._start_camera()
                else:
                    trial.controller.unavailable("等待采集画面；可在上方选择设备并重试。")
            self._selected = index
            self.navigation.setCurrentIndex(index)
            self.host._capture_manager.set_native_preview_visible(False)
            self.host._sync_workspace_mode()
            self._save_page()
        finally:
            self._switching = False

    def request_start(self):
        self._want_capture = True
        self._error_message = ""
        manager = self.host._capture_manager
        if manager.is_stopping():
            self._restart_pending = True
            self.feedback.setText("正在释放上一采集会话，完成后自动连接…")
            self.feedback.show()
            return
        self._restart_pending = False
        if not manager.is_preview_active():
            self.host.start_live_preview(automatic=True)
        self._resume_trial()
        self.refresh_capture_state()

    def _resume_trial(self):
        manager = self.host._capture_manager
        trial = self.pages.currentWidget()
        if (self._entered and trial is not self.capture_page and manager.is_preview_active()
                and not manager.is_stopping() and trial.source is None):
            trial._start_camera()

    def toggle_capture(self):
        manager = self.host._capture_manager
        if manager.is_preview_active() or manager.is_stopping():
            self._want_capture = False
            self._restart_pending = False
            self.suspend_trial("已停止采集，累计记录保留")
            self.host.stop_live_preview()
        else:
            self.request_start()
        self.refresh_capture_state()

    def _preview_state_changed(self, active):
        if active:
            self._resume_trial()
        elif self._entered and self._want_capture and self._restart_pending:
            QTimer.singleShot(0, self, self._restart_if_requested)
        self.refresh_capture_state()

    def _restart_if_requested(self):
        if self._entered and self._want_capture and self._restart_pending:
            self.request_start()

    def _capture_state_changed(self, state):
        if state.state == "faulted":
            self._want_capture = False
            self._restart_pending = False
            self.show_error(state.message or "设备连接失败，请检查设备后重试。")
        self.refresh_capture_state()

    def show_error(self, message):
        self._error_message = str(message)
        self.feedback.setText(self._error_message)
        self.feedback.show()
        self.refresh_capture_state()

    def refresh_capture_state(self, *_):
        manager = self.host._capture_manager
        device = manager.selected_device()
        active = manager.is_preview_active()
        stopping = manager.is_stopping()
        state = manager.current_capture_state()
        label = {"starting": "连接中", "streaming": "采集中", "stopping": "停止中", "faulted": "连接异常"}.get(state, "未启动")
        self.capture_button.setText("停止中…" if stopping else "停止采集" if active else "重试采集" if self._error_message else "开始采集")
        self.capture_button.setEnabled(not stopping)
        self.device_status.setText((device.name if device else "未选择设备") + " · " + label)
        self.device_status.setToolTip(self.device_status.text())
        if active and not stopping and not self._error_message:
            self.feedback.hide()
        elif not active and not self._error_message and not self._restart_pending:
            self.feedback.setText("选择可用的采集设备后点击开始；进入试验后选择标尺即可测量。" if device else "尚未发现采集设备。连接设备后，打开“设备设置”刷新列表，再重试采集。")
            self.feedback.show()
        self._refresh_icons()
