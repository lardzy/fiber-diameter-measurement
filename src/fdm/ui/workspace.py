from __future__ import annotations

from enum import StrEnum

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

from fdm.settings import WorkspaceLayoutSettings


class WorkspaceMode(StrEnum):
    """Top-level task workspaces exposed by the main window."""

    MEASURE = "measure"
    ACQUIRE = "acquire"
    DIGITAL_SLIDE_ACQUIRE = "digital_slide_acquire"


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
        self._compact = False
        self._applying = False
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
        # Becomes True once dock visibility is authoritative: either restored
        # from a saved window state or applied once while the window is shown.
        # Before that, isHidden() on the docks reports the never-shown
        # transitional state and must not be sampled into _wide_visibility.
        self._visibility_ready = False
        for dock in (self._project_dock, self._inspector_dock, self._results_dock):
            if isinstance(dock, WorkspaceDockWidget):
                dock.extentChanged.connect(
                    lambda size, target=dock: self._on_dock_extent_changed(target, size)
                )

    layoutPreferencesChanged = Signal()

    @property
    def workspace(self) -> WorkspaceMode:
        return self._workspace

    @property
    def is_compact(self) -> bool:
        return self._compact

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

    def set_workspace(self, workspace: WorkspaceMode | str) -> None:
        target_workspace = WorkspaceMode(workspace)
        workspace_changed = target_workspace is not self._workspace
        self._workspace = target_workspace
        if self.is_presentation_suspended:
            return
        compact = int(self._window.width()) < self.COMPACT_WIDTH
        # Frame-stream previews resync the workspace for every frame; preserve
        # the user's dock visibility unless the mode or breakpoint changed.
        if not workspace_changed and compact == self._compact:
            return
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
        QTimer.singleShot(0, self._finish_window_resize)

    def _finish_window_resize(self) -> None:
        if self.is_presentation_suspended:
            return
        self.restore_preferred_extents()
        QTimer.singleShot(0, self._release_extent_capture)

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
            if compact and not self._compact and self._window.isVisible():
                self._wide_visibility = {
                    "project": not self._project_dock.isHidden(),
                    "inspector": not self._inspector_dock.isHidden(),
                    "results": not self._results_dock.isHidden(),
                }
            self._compact = compact
            if not self._window.isVisible():
                # 首启时窗口尚未映射，传入宽度可能来自错误的屏幕或在映射时被
                # 窗口管理器再次调整。此时只记录断点状态，dock 可见性交给
                # show 之后用最终宽度的强制应用决定，避免同一设备首启时
                # 「项目与类别」时有时无。
                return
            if compact:
                self._results_dock.hide()
                if self._workspace == WorkspaceMode.DIGITAL_SLIDE_ACQUIRE:
                    self._inspector_dock.hide()
                    self._project_dock.show()
                else:
                    self._project_dock.hide()
                    self._inspector_dock.show()
            else:
                if self._workspace is WorkspaceMode.ACQUIRE:
                    self._project_dock.hide()
                    self._inspector_dock.show()
                    self._results_dock.hide()
                elif self._workspace is WorkspaceMode.DIGITAL_SLIDE_ACQUIRE:
                    self._project_dock.show()
                    self._inspector_dock.show()
                    self._results_dock.hide()
                else:
                    results_visible = self._wide_visibility["results"]
                    self._project_dock.setVisible(self._wide_visibility["project"])
                    self._inspector_dock.setVisible(self._wide_visibility["inspector"])
                    self._results_dock.setVisible(results_visible)
        finally:
            self._applying = False
        # Dock visibility has been applied explicitly while the window is
        # shown, so later visibilityChanged signals reflect real state.
        self._visibility_ready = True
        self._last_extent_window_size = QSize(self._window.size())
        QTimer.singleShot(0, self.restore_preferred_extents)
        self.layoutChanged.emit(compact)

    def toggle_project(self) -> None:
        if self._compact:
            show = not self._project_dock.isVisible()
            self._inspector_dock.hide()
            self._project_dock.setVisible(show)
            if not show:
                self._inspector_dock.show()
            QTimer.singleShot(0, self.restore_preferred_extents)
            return
        self._project_dock.setVisible(not self._project_dock.isVisible())
        self._wide_visibility["project"] = self._project_dock.isVisible()
        if self._project_dock.isVisible():
            QTimer.singleShot(0, self.restore_preferred_extents)

    def toggle_inspector(self) -> None:
        if self._compact:
            show = not self._inspector_dock.isVisible()
            self._project_dock.hide()
            self._inspector_dock.setVisible(show)
            if not show:
                self._project_dock.show()
            QTimer.singleShot(0, self.restore_preferred_extents)
            return
        self._inspector_dock.setVisible(not self._inspector_dock.isVisible())
        self._wide_visibility["inspector"] = self._inspector_dock.isVisible()
        if self._inspector_dock.isVisible():
            QTimer.singleShot(0, self.restore_preferred_extents)

    def toggle_results(self) -> None:
        show = not self._results_dock.isVisible()
        self._applying = True
        try:
            if not self._compact:
                self._wide_visibility["results"] = show
            self._results_dock.setVisible(show)
        finally:
            self._applying = False
        if show:
            QTimer.singleShot(0, self.restore_preferred_extents)
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
        self._wide_visibility = {"project": True, "inspector": True, "results": False}
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
        self._wide_visibility = {
            "project": not self._project_dock.isHidden(),
            "inspector": not self._inspector_dock.isHidden(),
            "results": not self._results_dock.isHidden(),
        }

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
                        if isinstance(document_area, QWidget)
                        else int(self._window.width())
                    )
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
                if not isinstance(document_area, QWidget):
                    document_area = central
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
            or self._compact
            or not self._visibility_ready
        ):
            return
        self._wide_visibility = {
            "project": not self._project_dock.isHidden(),
            "inspector": not self._inspector_dock.isHidden(),
            "results": not self._results_dock.isHidden(),
        }
