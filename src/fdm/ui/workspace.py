from __future__ import annotations

from enum import StrEnum

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget


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

    def __init__(self, title: str, object_name: str, parent: QMainWindow) -> None:
        super().__init__(title, parent)
        self.setObjectName(object_name)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.setTitleBarWidget(None)

    def setPanelWidget(self, widget: QWidget) -> None:
        self.setWidget(widget)


class AdaptiveLayoutController(QObject):
    """Apply responsive panel rules without owning business state."""

    layoutChanged = Signal(bool)

    COMPACT_WIDTH = 1180
    MEDIUM_WIDTH = 1440

    def __init__(
        self,
        window: QMainWindow,
        *,
        project_dock: QDockWidget,
        inspector_dock: QDockWidget,
        results_dock: QDockWidget,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent or window)
        self._window = window
        self._project_dock = project_dock
        self._inspector_dock = inspector_dock
        self._results_dock = results_dock
        self._workspace = WorkspaceMode.MEASURE
        self._compact = False
        self._applying = False
        self._wide_visibility = {
            "project": True,
            "inspector": True,
            "results": False,
        }

    @property
    def workspace(self) -> WorkspaceMode:
        return self._workspace

    @property
    def is_compact(self) -> bool:
        return self._compact

    def set_workspace(self, workspace: WorkspaceMode | str) -> None:
        self._workspace = WorkspaceMode(workspace)
        self.apply_for_width(self._window.width(), force=True)

    def apply_for_width(self, width: int, *, force: bool = False) -> None:
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
                    project_visible = self._wide_visibility["project"] and not (
                        results_visible and int(width) < self.MEDIUM_WIDTH
                    )
                    self._project_dock.setVisible(project_visible)
                    self._inspector_dock.setVisible(self._wide_visibility["inspector"])
                    self._results_dock.setVisible(results_visible)
        finally:
            self._applying = False
        self.layoutChanged.emit(compact)

    def toggle_project(self) -> None:
        if self._compact:
            show = not self._project_dock.isVisible()
            self._inspector_dock.hide()
            self._project_dock.setVisible(show)
            if not show:
                self._inspector_dock.show()
            return
        self._project_dock.setVisible(not self._project_dock.isVisible())
        self._wide_visibility["project"] = self._project_dock.isVisible()

    def toggle_inspector(self) -> None:
        if self._compact:
            show = not self._inspector_dock.isVisible()
            self._project_dock.hide()
            self._inspector_dock.setVisible(show)
            if not show:
                self._project_dock.show()
            return
        self._inspector_dock.setVisible(not self._inspector_dock.isVisible())
        self._wide_visibility["inspector"] = self._inspector_dock.isVisible()

    def toggle_results(self) -> None:
        show = not self._results_dock.isVisible()
        self._applying = True
        try:
            if not self._compact:
                self._wide_visibility["results"] = show
                if show and self._window.width() < self.MEDIUM_WIDTH:
                    self._project_dock.hide()
            self._results_dock.setVisible(show)
            if not self._compact and not show and self._window.width() < self.MEDIUM_WIDTH:
                self._project_dock.setVisible(self._wide_visibility["project"])
        finally:
            self._applying = False
        self.layoutChanged.emit(self._compact)

    def reset_defaults(self) -> None:
        self._wide_visibility = {"project": True, "inspector": True, "results": False}
        self.apply_for_width(self._window.width(), force=True)
        if not self._compact:
            self._window.resizeDocks(
                [self._project_dock, self._inspector_dock],
                [260, 340],
                Qt.Orientation.Horizontal,
            )

    def note_visibility_change(self) -> None:
        if self._applying or self._compact:
            return
        project_visible = not self._project_dock.isHidden()
        if (
            self._workspace is WorkspaceMode.MEASURE
            and not self._results_dock.isHidden()
            and self._window.width() < self.MEDIUM_WIDTH
        ):
            project_visible = self._wide_visibility["project"]
        self._wide_visibility = {
            "project": project_visible,
            "inspector": not self._inspector_dock.isHidden(),
            "results": not self._results_dock.isHidden(),
        }
