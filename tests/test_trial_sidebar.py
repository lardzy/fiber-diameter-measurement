"""The content controls follow accepted navigation without altering trial data."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QRect, Qt
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from fdm.ui.trial_records import AcquisitionDockHost
from fdm.ui.trial_sidebar import TrialSidebarPresentationController


def settle():
    for _ in range(8):
        QApplication.processEvents()


def workspace():
    value = QMainWindow()
    value.session = SimpleNamespace(workspace_options={"mode": "content"}, sample_name="样品", sample_number="", name="")
    value._closed = False
    value.trial_sidebar = QFrame(value)
    value.trial_sidebar.setMinimumWidth(390)
    value.trial_sidebar.setMaximumWidth(520)
    layout = QVBoxLayout(value.trial_sidebar)
    value.metadata_scroll = QLabel("操作员／标尺", value.trial_sidebar)
    layout.addWidget(value.metadata_scroll)
    value.workflow_tabs = QTabWidget(value.trial_sidebar)
    for title in ("操作", "审核", "报告"):
        value.workflow_tabs.addTab(QLabel(title), title)
    layout.addWidget(value.workflow_tabs)
    layout.addWidget(QLabel("保存 / Excel / 导出"))
    return value


@pytest.fixture
def presenter(tmp_path):
    host = AcquisitionDockHost()
    host.setCentralWidget(QWidget())
    host.resize(1200, 700)
    host.show()
    value = TrialSidebarPresentationController(host, preferences_path=tmp_path / "layout.json")
    page = workspace()
    value.bind(page)
    settle()
    yield value, page, host
    value.shutdown()
    page.close()
    host.close()
    page.deleteLater()
    host.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    settle()


@pytest.mark.parametrize("position", ["left", "right", "bottom", "floating"])
def test_content_sidebar_position_and_suspend_restore(presenter, position):
    control, page, host = presenter
    control.set_position(position)
    settle()
    assert control.is_open
    assert control.contains_focus(page.workflow_tabs)
    assert control.scroll.widget() is page.trial_sidebar
    assert page.trial_sidebar.maximumWidth() > 520
    assert host.size().width() == 1200 and host.size().height() == 700
    assert control.dock.isFloating() is (position == "floating")
    if position != "floating":
        assert host.dockWidgetArea(control.dock) == {"left": Qt.LeftDockWidgetArea,
            "right": Qt.RightDockWidgetArea, "bottom": Qt.BottomDockWidgetArea}[position]
    control.flush_preferences()
    before = control.preferences_path.read_bytes()
    control.suspend()
    settle()
    assert not control.is_open and control.dock.isHidden()
    assert control.preferences["visible"]
    assert control.preferences_path.read_bytes() == before
    control.bind(page)
    settle()
    assert control.is_open and control.preferences["position"] == position


def test_explicit_close_persists_and_reset_recovers(presenter):
    control, page, _ = presenter
    control.close()
    settle()
    assert not control.preferences["visible"]
    control.suspend()
    control.bind(page)
    assert not control.is_open
    control.flush_preferences()
    assert control._read_preferences()["visible"] is False
    control.reset_layout()
    settle()
    assert control.is_open and control.preferences["position"] == "right"


def test_rebind_parks_old_controls_and_diameter_stays_undocked(presenter):
    control, old, _ = presenter
    new = workspace()
    control.bind(new)
    assert old.trial_sidebar.parentWidget() is old and old.trial_sidebar.isHidden()
    assert control.scroll.widget() is new.trial_sidebar
    diameter = workspace()
    diameter.session.workspace_options["mode"] = "diameter"
    original_parent = diameter.trial_sidebar.parentWidget()
    control.bind(diameter)
    assert diameter.trial_sidebar.parentWidget() is original_parent
    assert control.workspace is new and not control.is_open
    control.detach(old)
    assert control.workspace is new
    control.detach(new)
    assert control.workspace is None and control.scroll.widget() is None
    new.close()
    diameter.close()


def test_offscreen_floating_geometry_is_clamped_and_restored(presenter):
    control, page, host = presenter
    control.preferences["floating_geometry"] = [100000, -100000, 3000, 3000]
    control.set_position("floating")
    settle()
    available = host.screen().availableGeometry()
    assert available.contains(control.dock.geometry())
    control.flush_preferences()
    control.detach()
    restored = TrialSidebarPresentationController(host, preferences_path=control.preferences_path)
    restored.bind(page)
    settle()
    assert restored.dock.isFloating()
    assert available.contains(restored.dock.geometry())
    restored.shutdown()


def test_narrow_bottom_has_internal_scroll_without_enlarging_host(presenter):
    control, page, host = presenter
    page.trial_sidebar.setMinimumHeight(850)
    host.resize(640, 480)
    control.set_position("bottom")
    settle()
    assert host.size().width() == 640 and host.size().height() == 480
    assert control.scroll.verticalScrollBar().maximum() > 0
    assert control.dock.height() <= host.height() // 2 + 30
