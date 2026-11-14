"""Application-owned presentation for the content trial's complete sidebar."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication, QBoxLayout, QDockWidget, QFrame, QScrollArea, QWidget
from shiboken6 import isValid

from fdm.atomic_io import atomic_write_text


class _SidebarDock(QDockWidget):
    closedByUser = Signal()

    def closeEvent(self, event):
        self.closedByUser.emit()
        super().closeEvent(event)


class _SidebarScroll(QScrollArea):
    def minimumSizeHint(self):
        return QSize(280, 100)

    def sizeHint(self):
        return QSize(432, 620)


class TrialSidebarPresentationController(QObject):
    """Move the existing controls, without copying or replacing trial state.

    User visibility is independent of temporary page suspension. The wrapper
    absorbs a long sidebar's minimum height, so bottom docking cannot enlarge
    the main window. Detached controls are parked with their owning workspace.
    """
    visibilityChanged = Signal(bool)

    def __init__(self, dock_host, *, is_current=None, preferences_path=None, parent=None):
        super().__init__(parent or dock_host)
        self.host = dock_host
        self.workspace = None
        self._active = False
        self._closed = False
        self._programmatic = False
        self._applied_position = None
        self._last_host_size = self.host.size()
        self._target_extent = None
        self._is_current = is_current or (lambda workspace: not getattr(workspace, "_closed", False))
        if preferences_path is None:
            from fdm.settings import settings_file_path
            preferences_path = settings_file_path().parent / "content-trial-sidebar.json"
        self.preferences_path = Path(preferences_path)
        self.preferences = self._read_preferences()
        self.dock = _SidebarDock("含量试验", dock_host)
        self.dock.setObjectName("contentTrialSidebarDock")
        self.dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea)
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.scroll = _SidebarScroll(self.dock)
        self.scroll.setObjectName("acquisitionTrial")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.dock.setWidget(self.scroll)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self.flush_preferences)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._fit_current)
        self._programmatic = True
        self.host.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.hide()
        self._programmatic = False
        self.dock.closedByUser.connect(self._user_closed)
        self.dock.topLevelChanged.connect(self._position_changed)
        self.dock.dockLocationChanged.connect(self._position_changed)
        self.dock.installEventFilter(self)
        self.host.installEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.screenRemoved.connect(self._screen_removed)

    @staticmethod
    def _defaults():
        return {"version": 1, "visible": True, "position": "right", "width": 432,
                "bottom_height": 310, "floating_geometry": None}

    def _read_preferences(self):
        result = self._defaults()
        try:
            values = json.loads(self.preferences_path.read_text(encoding="utf-8"))
            if not isinstance(values, dict) or values.get("version") != 1:
                return result
            if values.get("position") in ("left", "right", "bottom", "floating"):
                result["position"] = values["position"]
            result["visible"] = values.get("visible", True) is True
            for key, minimum, maximum in (("width", 280, 1000), ("bottom_height", 160, 700)):
                result[key] = max(minimum, min(maximum, int(values.get(key, result[key]))))
            geometry = values.get("floating_geometry")
            if isinstance(geometry, list) and len(geometry) == 4:
                result["floating_geometry"] = [int(value) for value in geometry]
        except (OSError, ValueError, TypeError, OverflowError):
            return self._defaults()
        return result

    @property
    def is_open(self):
        return self._active and not self.dock.isHidden()

    def _current(self):
        return (not self._closed and self._active and self.workspace is not None
                and self._is_current(self.workspace))

    def contains_focus(self, widget):
        panel = self.scroll.widget()
        return bool(widget is not None and panel is not None and
                    (widget is panel or panel.isAncestorOf(widget)))

    def owns_window(self, window):
        return self._current() and window is self.dock and self.dock.isFloating()

    def _prepare_panel(self, workspace):
        panel = workspace.trial_sidebar
        panel.setMinimumWidth(0)
        panel.setMaximumWidth(16777215)
        # Reuse the original widgets; metadata and workflow become side by side
        # only in wide bottom docks. Save/Excel/export remain a common footer.
        if not hasattr(panel, "_dock_content_box"):
            layout = panel.layout()
            metadata = getattr(workspace, "metadata_scroll", None)
            workflow = getattr(workspace, "workflow_tabs", None)
            if metadata is not None and workflow is not None:
                layout.removeWidget(metadata)
                layout.removeWidget(workflow)
                content = QWidget(panel)
                box = QBoxLayout(QBoxLayout.Direction.TopToBottom, content)
                box.setContentsMargins(0, 0, 0, 0)
                box.setSpacing(4)
                box.addWidget(metadata)
                box.addWidget(workflow, 1)
                layout.insertWidget(0, content, 1)
                panel._dock_content_box = box
        return panel

    def bind(self, workspace):
        if self._closed:
            return
        if workspace.session.workspace_options.get("mode") != "content":
            self.suspend()
            return
        if self.workspace is not workspace:
            self.detach()
            panel = self._prepare_panel(workspace)
            self.workspace = workspace
            self.scroll.setWidget(panel)
            panel.show()
            workspace.installEventFilter(self)
        self._active = True
        self.scroll.setEnabled(True)
        self.refresh_theme()
        self._refresh_title()
        self._apply_position()
        self.visibilityChanged.emit(self.is_open)

    def refresh_theme(self):
        if self.workspace is not None and isValid(self.workspace):
            # Reparenting into the dock removes the workspace stylesheet ancestor.
            self.scroll.setStyleSheet(self.workspace.styleSheet())

    def _refresh_title(self):
        if self.workspace is not None:
            session = self.workspace.session
            name = session.sample_name or session.sample_number or session.name or "新试验"
            self.dock.setWindowTitle(f"含量试验 · {name}")

    def suspend(self):
        if self._closed:
            return
        self._active = False
        self.scroll.setEnabled(False)
        self._programmatic = True
        self.dock.hide()
        self._programmatic = False
        self.visibilityChanged.emit(False)

    def detach(self, workspace=None):
        if workspace is not None and workspace is not self.workspace:
            return
        self.suspend()
        old, self.workspace = self.workspace, None
        panel = self.scroll.takeWidget()
        if old is not None and isValid(old):
            old.removeEventFilter(self)
            if panel is not None:
                panel.setParent(old)
                panel.hide()

    def open(self):
        if not self._current():
            return False
        self.preferences["visible"] = True
        self._apply_position()
        self._save_timer.start()
        self.visibilityChanged.emit(True)
        return True

    def close(self):
        if not self._closed:
            self.dock.close()

    def _user_closed(self):
        if not self._programmatic and not self._closed:
            self.preferences["visible"] = False
            self._save_timer.start()
            self.visibilityChanged.emit(False)

    def set_position(self, position):
        if position not in ("left", "right", "bottom", "floating"):
            raise ValueError("未知试验面板位置")
        if not self._closed:
            self.preferences["position"] = position
            self._apply_position()
            self._save_timer.start()

    def _set_compact(self):
        panel = self.scroll.widget()
        box = getattr(panel, "_dock_content_box", None)
        if box is not None:
            compact = self.preferences["position"] == "bottom" and self.host.width() >= 900
            box.setDirection(QBoxLayout.Direction.LeftToRight if compact else QBoxLayout.Direction.TopToBottom)
            box.setStretch(0, 1 if compact else 0)
            box.setStretch(1, 2 if compact else 1)

    def _floating_rect(self):
        geometry = self.preferences["floating_geometry"]
        rect = QRect(*geometry) if geometry else QRect(self.host.mapToGlobal(QPoint(24, 24)), QSize(460, 680))
        screens = QApplication.screens()
        screen = next((item for item in screens if item.availableGeometry().intersects(rect)),
                      self.host.screen() or QApplication.primaryScreen())
        if screen is not None:
            available = screen.availableGeometry().adjusted(12, 36, -12, -12)
            rect.setWidth(min(max(320, rect.width()), available.width()))
            rect.setHeight(min(max(220, rect.height()), available.height()))
            rect.moveLeft(max(available.left(), min(rect.left(), available.right() - rect.width() + 1)))
            rect.moveTop(max(available.top(), min(rect.top(), available.bottom() - rect.height() + 1)))
        return rect

    def _apply_position(self):
        position = self.preferences["position"]
        self._programmatic = True
        try:
            if position != self._applied_position:
                self.dock.hide()
            self._set_compact()
            if position == "floating":
                self.dock.setFloating(True)
                self.dock.setGeometry(self._floating_rect())
            else:
                self.dock.setFloating(False)
                area = {"left": Qt.DockWidgetArea.LeftDockWidgetArea,
                        "right": Qt.DockWidgetArea.RightDockWidgetArea,
                        "bottom": Qt.DockWidgetArea.BottomDockWidgetArea}[position]
                self.host.addDockWidget(area, self.dock)
                if position == "bottom":
                    dimension = min(self.preferences["bottom_height"], max(160, self.host.height() // 2))
                else:
                    dimension = min(self.preferences["width"], max(280, self.host.width() // 2))
                self._target_extent = dimension
                self.host.resizeDocks([self.dock], [dimension],
                    Qt.Orientation.Vertical if position == "bottom" else Qt.Orientation.Horizontal)
            self.dock.setVisible(self._active and self.preferences["visible"])
            self._applied_position = position
            self._last_host_size = self.host.size()
        finally:
            self._programmatic = False

    def _position_changed(self, *_):
        if self._programmatic or not self._active or self._closed:
            return
        area = self.host.dockWidgetArea(self.dock)
        self.preferences["position"] = ("floating" if self.dock.isFloating() else
            "bottom" if area == Qt.DockWidgetArea.BottomDockWidgetArea else
            "left" if area == Qt.DockWidgetArea.LeftDockWidgetArea else "right")
        self._set_compact()
        self._capture_extent()

    def _capture_extent(self):
        if self._programmatic or not self._active or self._closed or not self.dock.isVisible():
            return
        if self.dock.isFloating():
            rect = self.dock.geometry()
            self.preferences["floating_geometry"] = [rect.x(), rect.y(), rect.width(), rect.height()]
        elif self.host.size() != self._last_host_size:
            self._resize_timer.start(0)
            return
        elif self.preferences["position"] == "bottom":
            # Do not replace the preferred height when a small host clamps it.
            if self.dock.height() != self._target_extent or self.dock.height() >= self.preferences["bottom_height"]:
                self.preferences["bottom_height"] = max(160, min(700, self.dock.height()))
        else:
            if self.dock.width() != self._target_extent or self.dock.width() >= self.preferences["width"]:
                self.preferences["width"] = max(280, min(1000, self.dock.width()))
        self._save_timer.start()

    def _fit_current(self):
        if self._active and not self._closed:
            self._apply_position()

    def _screen_removed(self, *_):
        if not self._closed and self.preferences["position"] == "floating":
            self._resize_timer.start(0)

    def eventFilter(self, watched, event):
        if watched is self.dock and event.type() in (QEvent.Type.Resize, QEvent.Type.Move):
            self._capture_extent()
        if watched is self.host and event.type() == QEvent.Type.Resize and self._active:
            self._resize_timer.start(0)
        if watched is self.workspace and event.type() == QEvent.Type.StyleChange:
            self.refresh_theme()
        return super().eventFilter(watched, event)

    def reset_layout(self):
        self.preferences = self._defaults()
        self._apply_position()
        self._save_timer.start()
        self.visibilityChanged.emit(self.is_open)

    def flush_preferences(self):
        self._save_timer.stop()
        try:
            self.preferences_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.preferences_path, json.dumps(self.preferences, ensure_ascii=False,
                              indent=2, allow_nan=False), encoding="utf-8")
        except OSError as exc:
            if self.workspace is not None and not getattr(self.workspace, "_closed", False):
                self.workspace.statusBar().showMessage(f"试验面板布局已应用，但未能保存偏好：{exc}", 8000)

    def shutdown(self):
        if self._closed:
            return
        if self._save_timer.isActive():
            self.flush_preferences()
        self.detach()
        self._closed = True
        self._resize_timer.stop()
        self.dock.removeEventFilter(self)
        self.host.removeEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.screenRemoved.disconnect(self._screen_removed)
