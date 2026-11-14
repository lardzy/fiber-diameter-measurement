"""Stable acquisition navigation; pages subscribe to one managed device session."""
from __future__ import annotations

from dataclasses import dataclass
import json
import time
import weakref

from PySide6.QtCore import QTimer, QSize, QEvent, Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabBar, QPushButton, QLabel, QProgressBar, QToolButton, QMenu

from fdm.ui.icons import themed_icon
from fdm.ui.workspace_stack import WorkspaceStack


@dataclass(frozen=True)
class TrialDefinition:
    id: str
    title: str


TRIALS = (TrialDefinition("content", "含量试验"), TrialDefinition("diameter", "直径试验"))


class AcquisitionWorkbench(QWidget):
    pageLoadStateChanged = Signal(str)

    def __init__(self, host, capture_page):
        super().__init__(host)
        self.host = host
        self._selected = 0
        self._switching = False
        self._entered = False
        self._want_capture = False
        self._restart_pending = False
        self._error_message = ""
        self._entry_generation = 0
        self._page_generation = 0
        self._page_load = None
        self._page_load_state = "ready"
        self._auto_start_after_paint = False
        self._navigation_timings = []
        self._timing = None
        self._splitters = []
        try:
            preferences = json.loads(self._preferences_path().read_text(encoding="utf-8"))
            if "sidebar_width" not in preferences:
                legacy = json.loads(self._preferences_path().with_name("trial-workbench.json").read_text(encoding="utf-8"))
                preferences["sidebar_width"] = legacy.get("content", {}).get("sidebar_width", 432)
            self._sidebar_width = max(390, min(520, int(preferences.get("sidebar_width", 432))))
        except (OSError, ValueError, TypeError, AttributeError):
            self._sidebar_width = 432
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
        self.navigation.addTab("预览与抓拍")
        for trial in TRIALS:
            self.navigation.addTab(trial.title)
        row.addWidget(self.navigation)
        row.addStretch()
        self.trial_panel_button = QToolButton(self)
        self.trial_panel_button.setText("试验面板")
        self.trial_panel_button.setAccessibleName("显示或隐藏含量试验面板")
        self.trial_panel_button.setCheckable(True)
        self.trial_panel_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.trial_panel_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.trial_panel_button.clicked.connect(self._toggle_trial_panel)
        self.trial_panel_button.hide()
        row.addWidget(self.trial_panel_button)
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
        from fdm.ui.trial_records import AcquisitionDockHost, RecordsPresentationController
        self.body_host = AcquisitionDockHost(self)
        self.body_host.setCentralWidget(self.pages)
        self.records = RecordsPresentationController(self.body_host,
            is_current=lambda workspace: (self.pages.currentWidget() is workspace
                and not workspace._closed and self.host._acquisition_workspace_selected
                and workspace.project_identity == id(self.host.project)),
            action_callback=lambda workspace, action, key: workspace._guard(lambda: workspace._record_action(action, key)))
        from fdm.ui.trial_sidebar import TrialSidebarPresentationController
        self.trial_sidebar = TrialSidebarPresentationController(self.body_host,
            is_current=lambda workspace: (self.pages.currentWidget() is workspace
                and not workspace._closed and self.host._acquisition_workspace_selected
                and workspace.project_identity == id(self.host.project)))
        self.trial_sidebar.visibilityChanged.connect(self._trial_panel_visibility)
        menu = QMenu(self.trial_panel_button)
        for label, position in (("停靠左侧", "left"), ("停靠右侧", "right"),
                                ("停靠底部", "bottom"), ("浮动窗口", "floating")):
            menu.addAction(label, lambda checked=False, value=position: self.trial_sidebar.set_position(value))
        menu.addSeparator()
        menu.addAction("恢复默认位置", self.trial_sidebar.reset_layout)
        self.trial_panel_button.setMenu(menu)
        self.capture_page = capture_page
        self.pages.addWidget(capture_page)
        self.loading_page = QWidget(self.pages)
        self.loading_page.setObjectName("acquisitionPageLoading")
        loading_layout = QVBoxLayout(self.loading_page)
        loading_layout.addStretch(1)
        self.loading_label = QLabel("正在准备采集界面…", self.loading_page)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setWordWrap(True)
        self.loading_label.setTextFormat(Qt.TextFormat.PlainText)
        loading_layout.addWidget(self.loading_label)
        self.loading_progress = QProgressBar(self.loading_page)
        self.loading_progress.setRange(0, 0)
        self.loading_progress.setTextVisible(False)
        self.loading_progress.setMaximumWidth(280)
        self.loading_progress.hide()
        loading_layout.addWidget(self.loading_progress, 0, Qt.AlignmentFlag.AlignCenter)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.loading_retry = QPushButton("重试", self.loading_page)
        self.loading_retry.clicked.connect(lambda: self.select_page(self._selected))
        self.loading_retry.hide()
        actions.addWidget(self.loading_retry)
        self.loading_back = QPushButton("返回测量", self.loading_page)
        self.loading_back.clicked.connect(host._activate_measure_workspace)
        actions.addWidget(self.loading_back)
        actions.addStretch(1)
        loading_layout.addLayout(actions)
        loading_layout.addStretch(1)
        self.pages.addWidget(self.loading_page)
        self._loading_timer = QTimer(self)
        self._loading_timer.setSingleShot(True)
        self._loading_timer.setInterval(150)
        self._loading_timer.timeout.connect(self._show_loading_animation)
        if hasattr(host, "_preview_body_splitter"):
            self.register_splitter(host._preview_body_splitter)
        layout.addWidget(self.body_host, 1)
        self.navigation.currentChanged.connect(self.select_page)
        manager = host._capture_manager
        manager.previewStateChanged.connect(self._preview_state_changed)
        manager.captureStateChanged.connect(self._capture_state_changed)
        manager.devicesChanged.connect(self.refresh_capture_state)
        self.refresh_capture_state()
        self._refresh_icons()

    def _toggle_trial_panel(self, checked):
        if checked:
            self.trial_sidebar.open()
        else:
            self.trial_sidebar.close()

    def _trial_panel_visibility(self, visible):
        self.trial_panel_button.blockSignals(True)
        self.trial_panel_button.setChecked(visible)
        self.trial_panel_button.blockSignals(False)

    def _present_trial_sidebar(self, page):
        content = page is not None and page.session.workspace_options.get("mode") == "content"
        self.trial_panel_button.setVisible(content)
        if content:
            self.trial_sidebar.bind(page)
        else:
            self.trial_sidebar.suspend()

    @property
    def entry_generation(self):
        return self._entry_generation

    def is_entry_current(self, generation):
        return (generation == self._entry_generation and self._entered
                and self.host._acquisition_workspace_selected)

    def _begin_timing(self, index, requested_at=None):
        self._timing = {"page": index, "requested_at": time.perf_counter() if requested_at is None else requested_at,
                        "events": [{"phase": "requested", "elapsed_ms": 0.0}]}
        self._navigation_timings.append(self._timing)
        del self._navigation_timings[:-100]

    def record_transition_phase(self, name, *, once=False):
        if self._timing is not None:
            if once and any(event["phase"] == name for event in self._timing["events"]):
                return
            self._timing["events"].append({"phase": name,
                "elapsed_ms": (time.perf_counter() - self._timing["requested_at"]) * 1000})

    def transition_timings(self):
        from copy import deepcopy
        return deepcopy(self._navigation_timings)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._timing is not None and not any(e["phase"] == "framework_painted" for e in self._timing["events"]):
            self.record_transition_phase("framework_painted")
        load = self._page_load
        if load is not None and load["awaiting_paint"]:
            load["awaiting_paint"] = False
            generation = load["generation"]
            QTimer.singleShot(0, self, lambda token=generation: self._prepare_page_stage(token))
        if self._auto_start_after_paint and self._entered:
            self._auto_start_after_paint = False
            generation = self._entry_generation
            QTimer.singleShot(0, self, lambda token=generation: self._start_after_shell(token))

    def _start_after_shell(self, generation):
        if self.is_entry_current(generation) and self._want_capture:
            self.record_transition_phase("device_start_requested")
            self.request_start()

    def _current_trial(self):
        current = self.pages.currentWidget()
        return None if current in (self.capture_page, self.loading_page) else current

    def _cancel_page_load(self):
        self._page_generation += 1
        self._loading_timer.stop()
        load, self._page_load = self._page_load, None
        if load is not None and load.get("page") is not None:
            page = load["page"]
            cache = getattr(self.host, "_live_experiment_workspaces", {})
            for mode, cached in tuple(cache.items()):
                if cached is page:
                    del cache[mode]
            if getattr(self.host, "_live_experiment_workspace", None) is page:
                self.host._live_experiment_workspace = None
            if self.pages.indexOf(page) >= 0:
                self.pages.removeWidget(page)
            page.discard_initialization()
        self.loading_progress.hide()

    def _set_load_state(self, state):
        self._page_load_state = state
        self.pageLoadStateChanged.emit(state)

    def _show_loading_animation(self):
        if self._page_load is not None and self.pages.currentWidget() is self.loading_page:
            self.loading_progress.show()

    def _cancel_external_presentation(self):
        manager = getattr(self.host, "_trial_live_excel", None)
        cancel = getattr(manager, "cancel_presentation", None)
        if cancel is not None:
            cancel()

    def _prepare_page_stage(self, generation):
        load = self._page_load
        if (load is None or load["generation"] != generation or load["project"] != id(self.host.project)
                or not self.host._acquisition_workspace_selected or not self._entered):
            if load is not None and load["generation"] == generation:
                self._cancel_page_load()
            return
        try:
            if load["page"] is None:
                self.loading_label.setText("正在准备" + TRIALS[load["index"] - 1].title + "…")
                from fdm.ui.live_experiment_workspace import open_live_experiment
                load["page"] = open_live_experiment(self.host, embedded=True,
                    trial_mode=TRIALS[load["index"] - 1].id, deferred=True)
                if load["page"] is None:
                    raise RuntimeError("当前项目无法进入试验，请先另存为可编辑项目。")
                complete = load["page"]._initialized
                self.record_transition_phase("trial_model_ready")
            else:
                page = load["page"]
                descriptions = ("准备测量工具…", "恢复显示设置…", "准备报告入口…", "恢复试验记录…")
                self.loading_label.setText(descriptions[min(page._initialization_step, len(descriptions) - 1)])
                complete = page.initialize_next_stage()
                self.record_transition_phase("trial_stage_" + str(page._initialization_step))
            if not complete:
                # A positive timer gives queued navigation/paint events a turn;
                # widget construction remains on its owning GUI thread.
                QTimer.singleShot(1, self, lambda: self._prepare_page_stage(generation))
                return
            page = load["page"]
            mode = TRIALS[load["index"] - 1].id
            self.host._live_experiment_workspaces[mode] = page
            self.host._live_experiment_workspace = page
            self._loading_timer.stop()
            self.loading_progress.hide()
            self._publish_page(page)
            self._page_load = None
        except Exception as exc:
            self._cancel_page_load()
            self.pages.setCurrentWidget(self.loading_page)
            self.loading_label.setText("试验界面未能准备完成：\n" + str(exc))
            self.loading_retry.show()
            self._set_load_state("error")
            self.record_transition_phase("page_failed")

    def _publish_page(self, page):
        if self.pages.indexOf(page) < 0:
            self.pages.addWidget(page)
        self.pages.setCurrentWidget(page)
        self.records.bind(page)
        self._present_trial_sidebar(page)
        self.apply_sidebar_width()
        QTimer.singleShot(0, self, self.apply_sidebar_width)
        self._set_load_state("ready")
        self.record_transition_phase("page_interactive")
        self.host._update_history_action_states()
        if self.host._capture_manager.is_preview_active() and not self.host._capture_manager.is_stopping():
            page._start_camera()
        else:
            page.controller.unavailable("等待采集画面；可在上方选择设备并重试。")
        if self._entered:
            QTimer.singleShot(0, page, page.ensure_live_excel)

    def _refresh_icons(self):
        color = self.palette().windowText().color().name()
        for index, icon in enumerate(("acquisition", "content_trial", "diameter_trial")):
            self.navigation.setTabIcon(index, themed_icon(icon, color=color))
        self.trial_panel_button.setIcon(themed_icon("content_trial", color=color))
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

    def restore_page(self, *, requested_at=None):
        try:
            data = json.loads(self._preferences_path().read_text(encoding="utf-8"))
            index = int(data.get("page", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            index = 0
        self.select_page(index if 0 <= index <= len(TRIALS) else 0, requested_at=requested_at)

    def _save_page(self):
        from fdm.atomic_io import atomic_write_text
        try:
            path = self._preferences_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, json.dumps({"version": 2, "page": self._selected,
                                               "sidebar_width": self._sidebar_width}, allow_nan=False), encoding="utf-8")
        except OSError:
            pass

    def register_splitter(self, splitter):
        self._splitters.append(weakref.ref(splitter))
        splitter.splitterMoved.connect(lambda *_: self._sidebar_moved(splitter))
        self.apply_sidebar_width()

    def _sidebar_moved(self, splitter):
        if splitter.count() < 2:
            return
        self._sidebar_width = max(390, min(520, splitter.sizes()[1]))
        self.apply_sidebar_width()
        self._save_page()

    def apply_sidebar_width(self):
        from shiboken6 import isValid
        live = []
        for ref in self._splitters:
            splitter = ref()
            if splitter is None or not isValid(splitter):
                continue
            if splitter.count() < 2:
                continue
            live.append(ref)
            splitter.setSizes([max(320, splitter.width() - splitter.handleWidth() - self._sidebar_width),
                               self._sidebar_width])
        self._splitters = live

    def enter(self, *, auto_start=True, requested_at=None):
        first_entry = not self._entered
        self._entered = True
        if first_entry:
            self._entry_generation += 1
            if self._timing is None or self._page_load is None:
                self._begin_timing(self._selected, requested_at)
            if self._page_load is None:
                self.record_transition_phase("page_interactive")
        trial = self._current_trial()
        if trial is not None:
            self.records.bind(trial)
            self._present_trial_sidebar(trial)
            QTimer.singleShot(0, trial, trial.ensure_live_excel)
        if first_entry and auto_start:
            self._want_capture = True
            self._auto_start_after_paint = True
        if self.pages.currentWidget() is self.loading_page and self._page_load is None:
            self._set_load_state("error")
            self.select_page(self._selected)
        self.update()

    def leave(self):
        self.retire_entry()
        self.suspend_trial("已离开采集与试验")
        binding = getattr(self.host, "_preview_display_binding", None)
        if binding is not None:
            binding.flush()
        self.host.stop_live_preview()

    def confirm_leave(self):
        trial = self._current_trial()
        return trial is None or trial._confirm_trial_navigation()

    def prepare_leave(self):
        """Accept trial navigation before changing the main-window layout."""
        trial = self._current_trial()
        if trial is None:
            return True
        if not self.confirm_leave():
            return False
        try:
            trial.commit_pending_edits()
        except (ValueError, RuntimeError, OSError) as exc:
            trial.statusBar().showMessage(str(exc), 8000)
            return False
        return self.suspend_trial("已离开采集与试验")

    def retire_entry(self):
        """Invalidate delayed starts when an accepted transition retires this entry."""
        self._entered = False
        self._entry_generation += 1
        self._auto_start_after_paint = False
        self._cancel_page_load()
        self._cancel_external_presentation()
        cancel_start = getattr(self.host, "_cancel_pending_capture_start", None)
        if cancel_start is not None:
            cancel_start(reason="acquisition workspace left")
        self._want_capture = False
        self._restart_pending = False
        self.records.suspend()
        self.trial_sidebar.suspend()

    def suspend_trial(self, reason):
        trial = self._current_trial()
        if trial is None:
            return True
        trial.controller.cancel_draft()
        trial.controller.end_field(reason)
        return trial._stop_source()

    def clear_trial_pages(self):
        """A project replacement must not leave navigation on a disposed page."""
        self._switching = True
        self.retire_entry()
        self.records.suspend()
        self.trial_sidebar.detach()
        self.trial_panel_button.hide()
        try:
            for index in range(self.pages.count() - 1, -1, -1):
                page = self.pages.widget(index)
                if page not in (self.capture_page, self.loading_page):
                    self.pages.removeWidget(page)
            self._selected = 0
            self.navigation.setCurrentIndex(0)
            self.pages.setCurrentWidget(self.capture_page)
            self._set_load_state("ready")
        finally:
            self._switching = False

    def select_page(self, index, *, requested_at=None):
        if self._switching or not 0 <= index <= len(TRIALS):
            return False
        current = self._current_trial()
        if index == self._selected and self._page_load_state != "error" and (
                self._page_load is not None or (index == 0 and self.pages.currentWidget() is self.capture_page)
                or (current is not None and current.session.workspace_options.get("mode") == TRIALS[index - 1].id)):
            return True
        self._switching = True
        try:
            if index and self.host._digital_slide_mode and self.host._slide_acquisition_active():
                self.navigation.setCurrentIndex(self._selected)
                self.host.statusBar().showMessage("请先停止当前数字化切片采集，再切换试验页面。", 5000)
                return False
            if current is not None and not current._confirm_trial_navigation():
                self.navigation.setCurrentIndex(self._selected)
                return False
            if current is not None:
                try:
                    current.commit_pending_edits()
                except (ValueError, RuntimeError, OSError) as exc:
                    current.statusBar().showMessage(str(exc), 8000)
                    self.navigation.setCurrentIndex(self._selected)
                    return False
            binding = getattr(self.host, "_preview_display_binding", None)
            if self.pages.currentWidget() is self.capture_page and binding is not None:
                binding.flush()
            if not self.suspend_trial("已切换试验页面，累计记录保留"):
                self.navigation.setCurrentIndex(self._selected)
                return False
            self._cancel_page_load()
            self._cancel_external_presentation()
            self._entry_generation += 1
            cancel_start = getattr(self.host, "_cancel_pending_capture_start", None)
            if cancel_start is not None:
                cancel_start(reason="acquisition page changed")
            self._auto_start_after_paint = self._entered and self._want_capture
            self._selected = index
            self.navigation.setCurrentIndex(index)
            self._begin_timing(index, requested_at)
            self.records.suspend()
            self.trial_sidebar.suspend()
            self.trial_panel_button.hide()
            if self.host._digital_slide_mode and index:
                self.host._toggle_digital_slide_mode(False)
            # Target-page policy applies before preparing or exposing its body.
            self.host._sync_workspace_mode()
            if index == 0:
                self.host._preview_ruler_key = None
                self.pages.setCurrentWidget(self.capture_page)
                self._set_load_state("ready")
                self.record_transition_phase("page_interactive")
                frame = self.host._capture_manager.latest_preview_frame()
                if frame is not None:
                    self.host._on_live_preview_frame_ready(frame.image)
            else:
                mode = TRIALS[index - 1].id
                page = getattr(self.host, "_live_experiment_workspaces", {}).get(mode)
                if page is not None and not page._closed and page.project_identity == id(self.host.project):
                    self.host._live_experiment_workspace = page
                    self._publish_page(page)
                else:
                    self.loading_label.setText("正在准备" + TRIALS[index - 1].title + "…")
                    self.loading_retry.hide()
                    self.pages.setCurrentWidget(self.loading_page)
                    self._page_load = {"generation": self._page_generation, "project": id(self.host.project),
                                       "index": index, "page": None, "awaiting_paint": True}
                    self._set_load_state("loading")
                    self._loading_timer.start()
            self.host._capture_manager.set_native_preview_visible(False)
            self.host._update_history_action_states()
            self.apply_sidebar_width()
            self._save_page()
            self.update()
            return True
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
        trial = self._current_trial()
        if (self._entered and trial is not None and manager.is_preview_active()
                and not manager.is_stopping() and trial.source is None):
            trial._start_camera()

    def toggle_capture(self):
        manager = self.host._capture_manager
        if manager.is_stopping() and manager.current_capture_state() == "faulted":
            self._want_capture = False
            self._restart_pending = False
            manager.request_stop_preview()
            self.refresh_capture_state()
            return
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
        elif state.state == "streaming":
            self._error_message = ""
            self.feedback.hide()
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
        if state == "starting":
            label = {"enumerate": "查找设备中", "open_device": "连接中", "first_frame": "等待第一帧"}.get(
                manager.current_capture_stage(), label)
        retry_stop = stopping and state == "faulted"
        self.capture_button.setText("重试停止" if retry_stop else "停止中…" if stopping else "停止采集" if active else "重试采集" if self._error_message else "开始采集")
        self.capture_button.setEnabled(not stopping or retry_stop)
        emphasis = "stop" if active or stopping else "primary"
        if self.capture_button.property("emphasis") != emphasis:
            self.capture_button.setProperty("emphasis", emphasis)
            self.capture_button.style().unpolish(self.capture_button)
            self.capture_button.style().polish(self.capture_button)
        tone = {"streaming": "success", "faulted": "error", "stopping": "warning"}.get(state, "info")
        if self.device_status.property("statusTone") != tone:
            self.device_status.setProperty("statusTone", tone)
            self.device_status.style().unpolish(self.device_status)
            self.device_status.style().polish(self.device_status)
        self.device_status.setText(("● " if state == "streaming" else "") + (device.name if device else "未选择设备") + " · " + label)
        self.device_status.setToolTip(self.device_status.text())
        if active and not stopping and not self._error_message:
            self.feedback.hide()
        elif not active and not self._error_message and not self._restart_pending:
            self.feedback.setText("选择可用的采集设备后点击开始；进入试验后选择标尺即可测量。" if device else "尚未发现采集设备。连接设备后，打开“设备设置”刷新列表，再重试采集。")
            self.feedback.show()
        self._refresh_icons()
