from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PySide6.QtCore import QByteArray, QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMenuBar,
    QStatusBar,
    QToolBar,
    QWidget,
)

from fdm.ui.workspace import AdaptiveLayoutController


@dataclass(frozen=True, slots=True)
class FullscreenComponentSnapshot:
    """Visibility of one piece of application chrome before full screen."""

    widget: QWidget
    visible: bool


@dataclass(frozen=True, slots=True)
class FullscreenUiSnapshot:
    """Complete restorable UI state captured before full-screen measurement."""

    geometry: bytes
    main_window_state: bytes
    window_state: Qt.WindowState
    menu_bars: tuple[FullscreenComponentSnapshot, ...]
    status_bars: tuple[FullscreenComponentSnapshot, ...]
    toolbars: tuple[FullscreenComponentSnapshot, ...]
    docks: tuple[FullscreenComponentSnapshot, ...]
    extra_widgets: tuple[FullscreenComponentSnapshot, ...]

    def components(self) -> tuple[FullscreenComponentSnapshot, ...]:
        return (
            self.menu_bars
            + self.status_bars
            + self.toolbars
            + self.docks
            + self.extra_widgets
        )


@dataclass(frozen=True, slots=True)
class FullscreenPersistenceState:
    """Normal-window state that should be persisted while full screen is active."""

    geometry: bytes
    main_window_state: bytes
    window_state: Qt.WindowState


class FullscreenMeasurementController(QObject):
    """Own temporary full-screen chrome without changing workspace preferences.

    The controller deliberately knows nothing about measurement cancellation or
    Escape-key precedence.  MainWindow remains responsible for that business
    interaction and calls :meth:`enter`, :meth:`exit`, or :meth:`toggle`.
    """

    activeChanged = Signal(bool)
    entered = Signal()
    exited = Signal()

    def __init__(
        self,
        window: QMainWindow,
        *,
        adaptive_layout: AdaptiveLayoutController | None = None,
        extra_chrome: Iterable[QWidget] = (),
        preserved_widgets: Iterable[QWidget] = (),
        state_version: int = 0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent or window)
        self._window = window
        self._adaptive_layout = adaptive_layout
        self._extra_chrome = tuple(extra_chrome)
        self._preserved_widgets = tuple(preserved_widgets)
        self._state_version = int(state_version)
        self._snapshot: FullscreenUiSnapshot | None = None
        self._transitioning = False
        self._external_sync_pending = False
        self._window.installEventFilter(self)

    @property
    def is_active(self) -> bool:
        return self._snapshot is not None

    @property
    def entry_snapshot(self) -> FullscreenUiSnapshot | None:
        """The immutable pre-full-screen snapshot, if full screen is active."""

        return self._snapshot

    def persistence_state_for_close(self) -> FullscreenPersistenceState:
        """Return normal-window geometry/state even when closing in full screen."""

        snapshot = self._snapshot
        if snapshot is not None:
            return FullscreenPersistenceState(
                geometry=snapshot.geometry,
                main_window_state=snapshot.main_window_state,
                window_state=snapshot.window_state,
            )
        return FullscreenPersistenceState(
            geometry=bytes(self._window.saveGeometry()),
            main_window_state=bytes(
                self._window.saveState(self._state_version)
            ),
            window_state=self._window.windowState(),
        )

    def enter(self) -> bool:
        """Enter owned full-screen measurement mode.

        Returns ``False`` when the controller is already active, another owner
        already put the window in full screen, or a transition is in progress.
        """

        if self.is_active or self._transitioning or self._window.isFullScreen():
            return False
        self._transitioning = True
        snapshot = self._capture_snapshot()
        self._snapshot = snapshot
        if self._adaptive_layout is not None:
            self._adaptive_layout.begin_presentation_mode()
        try:
            self._hide_presentation_chrome(snapshot)
            self._window.showFullScreen()
        except Exception:
            self._restore_component_visibility(snapshot)
            if self._adaptive_layout is not None:
                self._adaptive_layout.end_presentation_mode(reapply_layout=False)
            self._snapshot = None
            raise
        finally:
            self._transitioning = False
        self.activeChanged.emit(True)
        self.entered.emit()
        return True

    def exit(self) -> bool:
        """Leave full screen and restore the exact pre-entry UI state."""

        if not self.is_active or self._transitioning:
            return False
        self._restore_from_snapshot(window_already_left_fullscreen=False)
        return True

    def toggle(self) -> bool:
        """Toggle the controller-owned full-screen mode."""

        return self.exit() if self.is_active else self.enter()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        window = getattr(self, "_window", None)
        if (
            window is not None
            and watched is window
            and event.type() == QEvent.Type.WindowStateChange
            and self.is_active
            and not self._transitioning
            and not self._external_sync_pending
        ):
            # Window-state delivery differs between Qt platform plugins.  Read
            # the settled state on the next event-loop turn and treat an
            # external departure from full screen like an explicit exit.
            self._external_sync_pending = True
            QTimer.singleShot(0, self._synchronize_external_window_state)
        return super().eventFilter(watched, event)

    def _synchronize_external_window_state(self) -> None:
        self._external_sync_pending = False
        if (
            self.is_active
            and not self._transitioning
            and not self._window.isFullScreen()
        ):
            self._restore_from_snapshot(window_already_left_fullscreen=True)

    def _capture_snapshot(self) -> FullscreenUiSnapshot:
        menu_bars = tuple(
            self._component_snapshot(widget)
            for widget in self._window.findChildren(
                QMenuBar,
                options=Qt.FindChildOption.FindDirectChildrenOnly,
            )
        )
        status_bars = tuple(
            self._component_snapshot(widget)
            for widget in self._window.findChildren(
                QStatusBar,
                options=Qt.FindChildOption.FindDirectChildrenOnly,
            )
        )
        toolbars = tuple(
            self._component_snapshot(widget)
            for widget in self._window.findChildren(
                QToolBar,
                options=Qt.FindChildOption.FindDirectChildrenOnly,
            )
        )
        docks = tuple(
            self._component_snapshot(widget)
            for widget in self._window.findChildren(
                QDockWidget,
                options=Qt.FindChildOption.FindDirectChildrenOnly,
            )
        )
        captured_ids = {
            id(component.widget)
            for group in (menu_bars, status_bars, toolbars, docks)
            for component in group
        }
        extra_widgets = tuple(
            self._component_snapshot(widget)
            for widget in self._extra_chrome
            if id(widget) not in captured_ids
        )
        return FullscreenUiSnapshot(
            geometry=bytes(self._window.saveGeometry()),
            main_window_state=bytes(
                self._window.saveState(self._state_version)
            ),
            window_state=self._window.windowState(),
            menu_bars=menu_bars,
            status_bars=status_bars,
            toolbars=toolbars,
            docks=docks,
            extra_widgets=extra_widgets,
        )

    @staticmethod
    def _component_snapshot(widget: QWidget) -> FullscreenComponentSnapshot:
        # isHidden() represents the widget's explicit visibility preference;
        # isVisible() would also become false when a parent is temporarily
        # hidden or before the top-level window is shown.
        return FullscreenComponentSnapshot(widget=widget, visible=not widget.isHidden())

    def _hide_presentation_chrome(self, snapshot: FullscreenUiSnapshot) -> None:
        for component in snapshot.components():
            widget = component.widget
            if self._is_preserved(widget):
                continue
            try:
                widget.hide()
            except RuntimeError:
                # A caller-owned optional widget may have been deleted between
                # construction and entry; it must not block full-screen mode.
                continue

    def _is_preserved(self, chrome: QWidget) -> bool:
        for preserved in self._preserved_widgets:
            try:
                if chrome is preserved or chrome.isAncestorOf(preserved):
                    return True
            except RuntimeError:
                continue
        return False

    def _restore_from_snapshot(self, *, window_already_left_fullscreen: bool) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return
        self._transitioning = True
        try:
            if not window_already_left_fullscreen and self._window.isFullScreen():
                self._window.showNormal()

            # Restore the normal geometry before reapplying maximized state.
            self._window.restoreGeometry(QByteArray(snapshot.geometry))
            self._window.restoreState(
                QByteArray(snapshot.main_window_state),
                self._state_version,
            )

            if snapshot.window_state & Qt.WindowState.WindowMaximized:
                self._window.showMaximized()
            elif snapshot.window_state & Qt.WindowState.WindowMinimized:
                self._window.showMinimized()
            else:
                self._window.showNormal()
                # Some platforms replace the normal geometry while leaving
                # full-screen state, so apply it once more after showNormal().
                self._window.restoreGeometry(QByteArray(snapshot.geometry))

            # QMainWindow.restoreState() handles dock placement but explicit
            # visibility restoration also covers menu/status/custom chrome and
            # makes the result independent of whether a component has a name.
            self._restore_component_visibility(snapshot)
        finally:
            if self._adaptive_layout is not None:
                self._adaptive_layout.end_presentation_mode(reapply_layout=False)
            self._snapshot = None
            self._transitioning = False
        self.activeChanged.emit(False)
        self.exited.emit()

    @staticmethod
    def _restore_component_visibility(snapshot: FullscreenUiSnapshot) -> None:
        for component in snapshot.components():
            try:
                component.widget.setVisible(component.visible)
            except RuntimeError:
                continue
