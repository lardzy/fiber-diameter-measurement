from __future__ import annotations

from enum import StrEnum
import json

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

from fdm.settings import WorkspaceLayoutSettings


class WorkspaceMode(StrEnum):
    """Top-level task workspaces exposed by the main window."""

    MEASURE = "measure"
    ACQUIRE = "acquire"
    DIGITAL_SLIDE_ACQUIRE = "digital_slide_acquire"


class WorkspacePage(StrEnum):
    """Accepted navigation identity, independent of camera running state."""

    MEASURE = "measure"
    CAPTURE = "capture"
    CONTENT_TRIAL = "content_trial"
    DIAMETER_TRIAL = "diameter_trial"
    DIGITAL_SLIDE_ACQUIRE = "digital_slide_acquire"


WORKSPACE_PANEL_SUPPORT = {
    WorkspacePage.MEASURE: frozenset({"project", "inspector", "results", "professional"}),
    WorkspacePage.CAPTURE: frozenset(),
    WorkspacePage.CONTENT_TRIAL: frozenset(),
    WorkspacePage.DIAMETER_TRIAL: frozenset(),
    WorkspacePage.DIGITAL_SLIDE_ACQUIRE: frozenset({"project", "inspector"}),
}


def workspace_mode_for_page(page: WorkspacePage | str) -> WorkspaceMode:
    page = WorkspacePage(page)
    if page is WorkspacePage.MEASURE:
        return WorkspaceMode.MEASURE
    if page is WorkspacePage.DIGITAL_SLIDE_ACQUIRE:
        return WorkspaceMode.DIGITAL_SLIDE_ACQUIRE
    return WorkspaceMode.ACQUIRE


class WorkspaceDockWidget(QDockWidget):
    """A restrained tool window used by the task-oriented workspace.

    The application intentionally does not expose free-floating expert docks.
    Users can resize, show and hide panels while the application keeps a
    predictable layout on the company workstations it targets.
    """

    extentChanged = Signal(QSize)

    def __init__(self, title: str, object_name: str, parent: QMainWindow) -> None:
        super().__init__(title, parent)
        self.setObjectName(object_name)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.setTitleBarWidget(None)

    def setPanelWidget(self, widget: QWidget) -> None:
        self.setWidget(widget)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.isVisible():
            self.extentChanged.emit(event.size())


class AdaptiveLayoutController(QObject):
    """Apply responsive panel rules without owning business state."""

    layoutChanged = Signal(bool)
    pageChanged = Signal(object)

    COMPACT_WIDTH = 1180
    MEDIUM_WIDTH = 1440
    MINIMUM_CENTRAL_WIDTH = 560
    MINIMUM_CENTRAL_HEIGHT = 120

    def __init__(
        self,
        window: QMainWindow,
        *,
        project_dock: QDockWidget,
        inspector_dock: QDockWidget,
        results_dock: QDockWidget,
        layout_settings: WorkspaceLayoutSettings | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent or window)
        self._window = window
        self._project_dock = project_dock
        self._inspector_dock = inspector_dock
        self._results_dock = results_dock
        self._layout_settings = layout_settings or WorkspaceLayoutSettings()
        self._workspace = WorkspaceMode.MEASURE
        self._page = WorkspacePage.MEASURE
        self._compact = False
        self._compact_side_panel = "inspector"
        self._applying = False
        self._layout_apply_pending = False
        self._suppress_extent_capture = False
        self._presentation_suspend_depth = 0
        self._pending_results_height: int | None = None
        self._programmatic_side_widths: dict[QDockWidget, int] = {}
        self._pending_side_extents: dict[QDockWidget, QSize] = {}
        self._last_extent_window_size = QSize(window.size())
        self._wide_visibility = {
            "project": True,
            "inspector": True,
            "results": False,
        }
        self._panels = {
            "project": project_dock, "inspector": inspector_dock, "results": results_dock,
        }
        self._panel_defaults = dict(self._wide_visibility)
        # Becomes True once dock visibility is authoritative: either restored
        # from a saved window state or applied once while the window is shown.
        # Before that, isHidden() on the docks reports the never-shown
        # transitional state and must not be sampled into _wide_visibility.
        self._visibility_ready = False
        self._embedded_controls = False
        self._profiles = self._read_profiles()
        if self._workspace.value in self._profiles:
            self._restore_profile()
        self._profile_save_timer = QTimer(self)
        self._profile_save_timer.setSingleShot(True)
        self._profile_save_timer.setInterval(200)
        self._profile_save_timer.timeout.connect(self.persist_profiles)
        self.layoutPreferencesChanged.connect(self._profile_save_timer.start)
        for dock in (self._project_dock, self._inspector_dock, self._results_dock):
            self._watch_panel(dock)
            if isinstance(dock, WorkspaceDockWidget):
                dock.extentChanged.connect(
                    lambda size, target=dock: self._on_dock_extent_changed(target, size)
                )

    layoutPreferencesChanged = Signal()

    @property
    def workspace(self) -> WorkspaceMode:
        return self._workspace

    @property
    def page(self) -> WorkspacePage:
        return self._page

    def register_panel(self, key: str, dock: QDockWidget, *, default_visible: bool = False) -> None:
        """Include a later-created dock in persistence and capability checks."""
        if key in self._panels:
            if self._panels[key] is dock:
                return
            raise ValueError(f"Panel already registered: {key}")
        if key not in WORKSPACE_PANEL_SUPPORT[WorkspacePage.MEASURE]:
            raise ValueError(f"Unknown workspace panel: {key}")
        self._panels[key] = dock
        self._panel_defaults[key] = bool(default_visible)
        profile = self._profiles.get(self._workspace.value, {})
        visibility = profile.get("visibility", {}) if isinstance(profile, dict) else {}
        self._wide_visibility[key] = bool(visibility.get(key, default_visible)) if isinstance(visibility, dict) else bool(default_visible)
        self._watch_panel(dock)
        self.enforce_panel_policy()

    def _watch_panel(self, dock: QDockWidget) -> None:
        dock.installEventFilter(self)
        dock.visibilityChanged.connect(self._panel_visibility_changed)

    def panel_allowed(self, key: str) -> bool:
        return not self._embedded_controls and key in WORKSPACE_PANEL_SUPPORT[self._page]

    def _panel_visibility_changed(self, _visible: bool) -> None:
        # Also covers QAction.trigger(), restoreState(), and direct dock.show().
        self.enforce_panel_policy()
        professional = self._panels.get("professional")
        if (self.sender() is professional and professional is not None
                and self._compact and self._visibility_ready and not self._applying
                and not self.is_presentation_suspended and self.panel_allowed("professional")):
            # Unlike compact's mutually exclusive legacy sidebars, the user
            # explicitly opens/closes this tab in either responsive layout.
            self._wide_visibility["professional"] = not professional.isHidden()
            self.layoutPreferencesChanged.emit()
        self.note_visibility_change()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Show and watched in self._panels.values():
            key = next(key for key, dock in self._panels.items() if dock is watched)
            if not self.panel_allowed(key):
                watched.hide()
                return True
        return super().eventFilter(watched, event)

    def enforce_panel_policy(self) -> None:
        """Suppress unsupported docks without changing saved user preferences."""
        was_applying = self._applying
        self._applying = True
        try:
            for key, dock in self._panels.items():
                allowed = self.panel_allowed(key)
                action = dock.toggleViewAction()
                action.setEnabled(allowed)
                action.setVisible(allowed)
                if not allowed and not dock.isHidden():
                    dock.hide()
        finally:
            self._applying = was_applying

    def set_panel_visible(self, key: str, visible: bool) -> bool:
        if not self.panel_allowed(key):
            self.enforce_panel_policy()
            return False
        dock = self._panels[key]
        self._wide_visibility[key] = bool(visible)
        dock.setVisible(bool(visible))
        self.layoutPreferencesChanged.emit()
        self.layoutChanged.emit(self._compact)
        return True

    @property
    def is_compact(self) -> bool:
        return self._compact

    def apply_pending_layout(self) -> None:
        """Apply a layout change deferred while the main window was hidden."""

        if not self._layout_apply_pending or not self._window.isVisible():
            return
        self.apply_for_width(self._window.width(), force=True)

    @property
    def is_presentation_suspended(self) -> bool:
        """Whether temporary presentation chrome changes are being ignored."""

        return self._presentation_suspend_depth > 0

    def begin_presentation_mode(self) -> None:
        """Freeze responsive layout and preference capture for temporary chrome.

        Full-screen measurement temporarily hides docks and changes the window
        extent.  Neither operation represents a user layout preference.  The
        depth counter makes the API safe for nested presentation surfaces and
        keeps already queued resize callbacks from releasing the capture guard.
        """

        self._presentation_suspend_depth += 1
        if self._presentation_suspend_depth != 1:
            return
        self._suppress_extent_capture = True
        self._pending_results_height = None
        self._pending_side_extents.clear()

    def end_presentation_mode(self, *, reapply_layout: bool = False) -> None:
        """Release a presentation freeze without capturing its temporary state.

        ``reapply_layout`` is opt-in because a full-screen controller normally
        restores an exact ``QMainWindow`` snapshot before releasing the guard.
        Reapplying responsive rules in that path could overwrite the restored
        visibility.  Other callers may request a fresh responsive pass.
        """

        if self._presentation_suspend_depth <= 0:
            return
        self._presentation_suspend_depth -= 1
        if self._presentation_suspend_depth:
            return
        self._suppress_extent_capture = False
        self._last_extent_window_size = QSize(self._window.size())
        if reapply_layout:
            self.apply_for_width(self._window.width(), force=True)
            self.restore_preferred_extents()
        self.enforce_panel_policy()

    def set_workspace(self, workspace: WorkspaceMode | str) -> None:
        target_workspace = WorkspaceMode(workspace)
        self.set_page({WorkspaceMode.MEASURE: WorkspacePage.MEASURE,
                       WorkspaceMode.ACQUIRE: WorkspacePage.CAPTURE,
                       WorkspaceMode.DIGITAL_SLIDE_ACQUIRE: WorkspacePage.DIGITAL_SLIDE_ACQUIRE}[target_workspace])

    def set_page(self, page: WorkspacePage | str) -> None:
        page = WorkspacePage(page)
        target_workspace = workspace_mode_for_page(page)
        page_changed = page is not self._page
        workspace_changed = target_workspace is not self._workspace
        if workspace_changed:
            self._remember_profile()
        self._workspace = target_workspace
        self._page = page
        if workspace_changed:
            self._restore_profile()
            self._visibility_ready = False
            self._profile_save_timer.start()
        self.enforce_panel_policy()
        if page_changed:
            self.pageChanged.emit(page)
        if self.is_presentation_suspended:
            return
        compact = int(self._window.width()) < self.COMPACT_WIDTH
        # Frame-stream previews resync the workspace for every frame; preserve
        # the user's dock visibility unless the mode or breakpoint changed.
        if not page_changed and compact == self._compact:
            return
        self.apply_for_width(self._window.width(), force=True)

    @staticmethod
    def _profiles_path():
        from fdm.settings import settings_file_path
        return settings_file_path().parent / "workspace-layouts.json"

    def _read_profiles(self):
        try:
            payload = json.loads(self._profiles_path().read_text(encoding="utf-8"))
            profiles = payload.get("workspaces", {})
            return profiles if isinstance(profiles, dict) else {}
        except (OSError, ValueError, AttributeError):
            return {}

    def _remember_profile(self):
        self._profiles[self._workspace.value] = {
            "visibility": dict(self._wide_visibility),
            "extents": self._layout_settings.to_dict(),
            "compact_side": self._compact_side_panel,
        }

    def _restore_profile(self):
        default = dict(self._panel_defaults)
        default["project"] = self._workspace is not WorkspaceMode.ACQUIRE
        profile = self._profiles.get(self._workspace.value, {})
        if not isinstance(profile, dict):
            profile = {}
        visibility = profile.get("visibility", {})
        if not isinstance(visibility, dict):
            visibility = {}
        self._wide_visibility = {key: bool(visibility.get(key, value)) for key, value in default.items()}
        default_side = "project" if self._workspace is WorkspaceMode.DIGITAL_SLIDE_ACQUIRE else "inspector"
        side = profile.get("compact_side", default_side)
        self._compact_side_panel = side if side in {"project", "inspector"} else default_side
        extents = WorkspaceLayoutSettings.from_dict(profile.get("extents", {}))
        for key in ("project_width", "inspector_width", "results_height"):
            setattr(self._layout_settings, key, getattr(extents, key))

    def persist_profiles(self):
        from fdm.atomic_io import atomic_write_text
        self._remember_profile()
        path = self._profiles_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, json.dumps({"version": 1, "workspaces": self._profiles},
                ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        except OSError:
            # Preferences must never make a measurement or application close fail.
            pass

    def set_embedded_controls(self, enabled: bool):
        """Trial controls temporarily replace docks without changing preferences."""
        if self._embedded_controls == bool(enabled):
            return
        self._embedded_controls = bool(enabled)
        self.enforce_panel_policy()
        self.apply_for_width(self._window.width(), force=True)

    def begin_window_resize(self) -> None:
        self._suppress_extent_capture = True

    def end_window_resize(self, width: int) -> None:
        if self.is_presentation_suspended:
            return
        self.apply_for_width(width)
        # Apply the saved dock widths in the same resize turn so a compact →
        # wide transition never exposes an oversized sidebar for one frame.
        self.restore_preferred_extents()
        QTimer.singleShot(0, self, self._finish_window_resize)

    def _finish_window_resize(self) -> None:
        if self.is_presentation_suspended:
            return
        self.restore_preferred_extents()
        QTimer.singleShot(0, self, self._release_extent_capture)

    def _release_extent_capture(self) -> None:
        if not self.is_presentation_suspended:
            self._suppress_extent_capture = False

    def apply_for_width(self, width: int, *, force: bool = False) -> None:
        if self.is_presentation_suspended:
            return
        compact = int(width) < self.COMPACT_WIDTH
        if not force and compact == self._compact:
            return
        self._applying = True
        try:
            if compact and not self._compact and self._window.isVisible() and not self._embedded_controls and self._visibility_ready:
                self._capture_allowed_visibility()
            self._compact = compact
            if not self._window.isVisible():
                # 首启时窗口尚未映射，传入宽度可能来自错误的屏幕或在映射时被
                # 窗口管理器再次调整。此时只记录断点状态，dock 可见性交给
                # show 之后用最终宽度的强制应用决定。运行期隐藏窗口时发生的
                # 断点或工作区变化也走同一路径，避免逻辑状态先行、dock 状态
                # 留在旧布局。
                self._layout_apply_pending = True
                return
            if self._embedded_controls:
                for dock in self._panels.values():
                    dock.hide()
            elif compact:
                self._results_dock.hide()
                if self._compact_side_panel == "project":
                    self._inspector_dock.hide()
                    self._project_dock.show()
                else:
                    self._project_dock.hide()
                    self._inspector_dock.show()
            else:
                for key, dock in self._panels.items():
                    dock.setVisible(self.panel_allowed(key) and self._wide_visibility[key])
            if compact and "professional" in self._panels:
                self._panels["professional"].setVisible(
                    self.panel_allowed("professional") and self._wide_visibility["professional"])
            self.enforce_panel_policy()
        finally:
            self._applying = False
        self._layout_apply_pending = False
        # Dock visibility has been applied explicitly while the window is
        # shown, so later visibilityChanged signals reflect real state.
        self._visibility_ready = True
        self._last_extent_window_size = QSize(self._window.size())
        QTimer.singleShot(0, self, self.restore_preferred_extents)
        self.layoutChanged.emit(compact)

    def toggle_project(self) -> None:
        if not self.panel_allowed("project"):
            self.enforce_panel_policy()
            return
        if self._compact:
            show = not self._project_dock.isVisible()
            self._compact_side_panel = "project" if show else "inspector"
            self._inspector_dock.hide()
            self._project_dock.setVisible(show)
            if not show:
                self._inspector_dock.show()
            QTimer.singleShot(0, self, self.restore_preferred_extents)
            self.layoutPreferencesChanged.emit()
            return
        self._project_dock.setVisible(not self._project_dock.isVisible())
        self._wide_visibility["project"] = self._project_dock.isVisible()
        if self._project_dock.isVisible():
            QTimer.singleShot(0, self, self.restore_preferred_extents)

    def toggle_inspector(self) -> None:
        if not self.panel_allowed("inspector"):
            self.enforce_panel_policy()
            return
        if self._compact:
            show = not self._inspector_dock.isVisible()
            self._compact_side_panel = "inspector" if show else "project"
            self._project_dock.hide()
            self._inspector_dock.setVisible(show)
            if not show:
                self._project_dock.show()
            QTimer.singleShot(0, self, self.restore_preferred_extents)
            self.layoutPreferencesChanged.emit()
            return
        self._inspector_dock.setVisible(not self._inspector_dock.isVisible())
        self._wide_visibility["inspector"] = self._inspector_dock.isVisible()
        if self._inspector_dock.isVisible():
            QTimer.singleShot(0, self, self.restore_preferred_extents)

    def toggle_results(self) -> None:
        if not self.panel_allowed("results"):
            self.enforce_panel_policy()
            return
        show = not self._results_dock.isVisible()
        self._applying = True
        try:
            if not self._compact:
                self._wide_visibility["results"] = show
            self._results_dock.setVisible(show)
        finally:
            self._applying = False
        if show:
            QTimer.singleShot(0, self, self.restore_preferred_extents)
        self.layoutChanged.emit(self._compact)

    def reset_defaults(self) -> None:
        defaults = WorkspaceLayoutSettings()
        self._layout_settings.project_width = defaults.project_width
        self._layout_settings.inspector_width = defaults.inspector_width
        self._layout_settings.results_height = defaults.results_height
        self._layout_settings.inspector_records_height = defaults.inspector_records_height
        self._layout_settings.statistics_expanded = defaults.statistics_expanded
        self._layout_settings.calibration_expanded = defaults.calibration_expanded
        self._layout_settings.records_expanded = defaults.records_expanded
        self._layout_settings.area_recognition_expanded = defaults.area_recognition_expanded
        self._layout_settings.object_properties_expanded = defaults.object_properties_expanded
        self._wide_visibility = dict(self._panel_defaults)
        self._compact_side_panel = "project" if self._workspace is WorkspaceMode.DIGITAL_SLIDE_ACQUIRE else "inspector"
        self.apply_for_width(self._window.width(), force=True)
        self.restore_preferred_extents()
        self.layoutPreferencesChanged.emit()

    @property
    def layout_settings(self) -> WorkspaceLayoutSettings:
        return self._layout_settings

    def set_layout_settings(self, settings: WorkspaceLayoutSettings) -> None:
        self._layout_settings = settings

    def capture_restored_visibility(self) -> None:
        """Adopt QMainWindow.restoreState() visibility before responsive rules run."""

        if self.is_presentation_suspended:
            return
        self._visibility_ready = True
        if self._workspace.value in self._profiles:
            self._restore_profile()
            self.enforce_panel_policy()
            return
        self._capture_allowed_visibility()
        self.enforce_panel_policy()

    def restore_preferred_extents(self) -> None:
        if (
            self.is_presentation_suspended
            or self._applying
            or not self._window.isVisible()
        ):
            return
        self._applying = True
        try:
            if self._compact:
                visible_side_dock: QDockWidget | None = None
                preferred_width = 0
                if self._project_dock.isVisible():
                    visible_side_dock = self._project_dock
                    preferred_width = int(self._layout_settings.project_width)
                elif self._inspector_dock.isVisible():
                    visible_side_dock = self._inspector_dock
                    preferred_width = int(self._layout_settings.inspector_width)
                if visible_side_dock is not None:
                    minimum_width = max(
                        120,
                        int(visible_side_dock.minimumSizeHint().width()),
                    )
                    document_area = getattr(self._window, "tab_widget", None)
                    combined_width = (
                        int(document_area.width()) + int(visible_side_dock.width())
                        if isinstance(document_area, QWidget) and document_area.isVisible()
                        else int(self._window.width())
                    )
                    combined_width = min(combined_width, self._window.width())
                    maximum_width = max(
                        minimum_width,
                        combined_width - self.MINIMUM_CENTRAL_WIDTH,
                    )
                    compact_width = max(
                        minimum_width,
                        min(preferred_width, maximum_width),
                    )
                    self._programmatic_side_widths[visible_side_dock] = compact_width
                    self._window.resizeDocks(
                        [visible_side_dock],
                        [compact_width],
                        Qt.Orientation.Horizontal,
                    )
            else:
                horizontal_docks: list[QDockWidget] = []
                horizontal_sizes: list[int] = []
                if self._project_dock.isVisible():
                    horizontal_docks.append(self._project_dock)
                    horizontal_sizes.append(int(self._layout_settings.project_width))
                if self._inspector_dock.isVisible():
                    horizontal_docks.append(self._inspector_dock)
                    horizontal_sizes.append(int(self._layout_settings.inspector_width))
                if horizontal_docks:
                    maximum_total = max(240, self._window.width() - 560)
                    if sum(horizontal_sizes) > maximum_total:
                        scale = maximum_total / max(1, sum(horizontal_sizes))
                        horizontal_sizes = [max(120, int(round(size * scale))) for size in horizontal_sizes]
                    self._programmatic_side_widths.update(
                        zip(horizontal_docks, horizontal_sizes, strict=True)
                    )
                    self._window.resizeDocks(
                        horizontal_docks,
                        horizontal_sizes,
                        Qt.Orientation.Horizontal,
                    )
            if self._results_dock.isVisible():
                central = self._window.centralWidget()
                document_area = getattr(self._window, "tab_widget", None)
                if not isinstance(document_area, QWidget) or not document_area.isVisible():
                    document_area = getattr(self._window, "_center_stack", central)
                central_height = (
                    max(0, document_area.height())
                    if document_area is not None
                    else 0
                )
                # The dock and central widget already share the usable vertical
                # workspace.  Their current combined height is therefore a more
                # reliable budget than subtracting a fixed allowance from the
                # whole window (which includes toolbars and the status bar).
                maximum_height = max(
                    120,
                    central_height
                    + max(0, self._results_dock.height())
                    - self.MINIMUM_CENTRAL_HEIGHT,
                )
                preferred_height = min(int(self._layout_settings.results_height), maximum_height)
                self._pending_results_height = preferred_height
                self._window.resizeDocks(
                    [self._results_dock],
                    [preferred_height],
                    Qt.Orientation.Vertical,
                )
        finally:
            self._applying = False

    def _on_dock_extent_changed(self, dock: QDockWidget, size: QSize) -> None:
        if (
            self._applying
            or self._suppress_extent_capture
            or not dock.isVisible()
        ):
            return
        if dock in {self._project_dock, self._inspector_dock}:
            expected = self._programmatic_side_widths.get(dock)
            if expected is not None and abs(int(size.width()) - expected) <= 2:
                self._programmatic_side_widths.pop(dock, None)
                return
            self._pending_side_extents[dock] = QSize(size)
            if self._window.size() == self._last_extent_window_size:
                self._capture_pending_side_extent(dock)
            else:
                QTimer.singleShot(
                    0,
                    self,
                    lambda target=dock: self._capture_pending_side_extent(target),
                )
            return
        if dock is self._results_dock and self._pending_results_height is not None:
            expected = self._pending_results_height
            self._pending_results_height = None
            if abs(int(size.height()) - expected) <= 12:
                return
        changed = False
        if dock is self._results_dock and not self._compact:
            value = max(120, int(size.height()))
            changed = value != self._layout_settings.results_height
            self._layout_settings.results_height = value
        if changed:
            self.layoutPreferencesChanged.emit()

    def _capture_pending_side_extent(self, dock: QDockWidget) -> None:
        size = self._pending_side_extents.pop(dock, None)
        if (
            size is None
            or self._applying
            or self._suppress_extent_capture
            or self._compact
            or not dock.isVisible()
        ):
            return
        value = max(120, int(dock.width()))
        expected = self._programmatic_side_widths.get(dock)
        if expected is not None:
            if abs(value - expected) <= 2:
                self._programmatic_side_widths.pop(dock, None)
                return
            self._programmatic_side_widths.pop(dock, None)
        if dock is self._project_dock:
            changed = value != self._layout_settings.project_width
            self._layout_settings.project_width = value
        elif dock is self._inspector_dock:
            changed = value != self._layout_settings.inspector_width
            self._layout_settings.inspector_width = value
        else:
            return
        if changed:
            self.layoutPreferencesChanged.emit()

    def note_visibility_change(self) -> None:
        if (
            self.is_presentation_suspended
            or self._applying
            or self._embedded_controls
            or self._compact
            or not self._visibility_ready
        ):
            return
        self._capture_allowed_visibility()
        self.layoutPreferencesChanged.emit()

    def _capture_allowed_visibility(self) -> None:
        for key, dock in self._panels.items():
            if self.panel_allowed(key):
                self._wide_visibility[key] = not dock.isHidden()
