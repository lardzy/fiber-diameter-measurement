"""Trial navigation changes presentation, while only business edits create data."""
import json

import pytest
from PySide6.QtCore import QObject, QEvent, QPoint, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QStackedWidget

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.models import ProjectState
from fdm.ui.live_experiment_workspace import open_live_experiment
from test_live_experiment_content_workspace import workspace
from test_acquisition_workflow_redesign import harness


@pytest.fixture
def trial_host():
    class Host(QMainWindow):
        def current_document(self):
            return None
    host = Host()
    host.project = ProjectState.empty()
    host.save_project_action = QAction(host)
    host.pages = QStackedWidget(host)
    host.setCentralWidget(host.pages)
    host.resize(1366, 650)
    host.show()
    yield host
    for view in tuple(getattr(host, "_live_experiment_workspaces", {}).values()):
        view.close()
    host.close()
    QApplication.processEvents()


def show_trial(host, mode="content"):
    view = open_live_experiment(host, embedded=True, trial_mode=mode)
    host.pages.addWidget(view)
    host.pages.setCurrentWidget(view)
    QApplication.processEvents()
    return view


def test_page_visits_are_cached_drafts_without_unsaved_business_data(trial_host):
    start = trial_host.project.extension_state_id
    views = {}
    for mode in ("content", "diameter") * 10:
        view = show_trial(trial_host, mode)
        if mode in views:
            assert views[mode] is view
        views[mode] = view
        for index in range(3):
            view.workflow_tabs.setCurrentIndex(index)
        assert not view.save_button.isEnabled()
    assert len(trial_host._live_experiment_workspaces) == 2
    assert not trial_host.project.live_experiments
    assert trial_host.project.extension_state_id == start


@pytest.mark.parametrize("edit", ["name", "number", "count", "measurement", "weights"])
def test_first_business_edit_registers_only_the_edited_draft(trial_host, edit):
    view = show_trial(trial_host)
    other = show_trial(trial_host, "diameter")
    trial_host.pages.setCurrentWidget(view)
    if edit == "name":
        view.sample.setText("样品甲")
        view.sample.editingFinished.emit()
    elif edit == "number":
        view.sample_number.setText("F-001")
        view.sample_number.editingFinished.emit()
    elif edit == "count":
        view.session.set_materials(default_materials())
        view._count_material("wool", 1)
    elif edit == "measurement":
        view.session.set_materials(default_materials())
        view.session.add_measurement(20, "wool")
        view._changed()
    else:
        view.warp_weight.setValue(70)
        view.weft_weight.setValue(30)
        view.warp_weight.editingFinished.emit()
    assert trial_host.project.live_experiments == [view.session]
    assert other.session not in trial_host.project.live_experiments
    assert trial_host.project.extension_state_id > 0
    previous = trial_host.project.extension_state_id
    view._changed()
    assert trial_host.project.extension_state_id == previous
    assert view.save_button.isEnabled()


def test_existing_empty_experiment_is_preserved_without_recreating_it(trial_host):
    session = ExperimentSession()
    session.add_operator("甲")
    session.set_materials(default_materials())
    trial_host.project.live_experiments.append(session)
    before = trial_host.project.extension_state_id
    view = show_trial(trial_host)
    assert view.session is session
    assert trial_host.project.live_experiments == [session]
    assert trial_host.project.extension_state_id == before


def test_explicit_save_flushes_focused_metadata_before_project_snapshot(trial_host):
    view = show_trial(trial_host)
    captured = []
    trial_host.save_project_action.triggered.connect(lambda: captured.append(view.session.sample_name))
    view.sample.setText("尚未离开输入框")
    view.warp_weight.lineEdit().setText("70")
    view.weft_weight.lineEdit().setText("30")
    assert not trial_host.project.live_experiments
    view._save_project()
    assert captured == ["尚未离开输入框"]
    assert trial_host.project.live_experiments == [view.session]
    operator = view.session.operators[view.session.active_operator_id]
    assert (operator.warp_weight, operator.weft_weight) == (70, 30)


@pytest.mark.parametrize("route", ["sync", "background", "global_action"])
@pytest.mark.parametrize("field", ["sample", "weights"])
def test_global_save_commits_focused_trial_editor_before_snapshot(harness, tmp_path, route, field):
    from fdm.project_io import ProjectIO
    from test_live_experiment_save_rebase import wait_saved

    harness.enter()
    view = harness.select(1)
    target = tmp_path / f"focused-{route}-{field}.fdmproj"
    harness.window._project_path = target
    editor = view.sample if field == "sample" else view.warp_weight
    if field == "sample":
        view.sample_details_toggle.click()
        QApplication.processEvents()
    editor.setFocus()
    QApplication.processEvents()
    if field == "sample":
        editor.setText("焦点中的样品")
    else:
        editor.lineEdit().setText("0.123456")
    assert editor.hasFocus()
    assert not harness.window.project.live_experiments
    if route == "sync":
        result = harness.window.save_project(str(target))
        assert result.success, result.message
    elif route == "background":
        assert harness.window.request_save_project(str(target))
        wait_saved(harness.app, harness.window)
    else:
        harness.window.save_project_action.trigger()
        wait_saved(harness.app, harness.window)
    restored = ProjectIO.load(target).live_experiments[0]
    if field == "sample":
        assert restored.sample_name == "焦点中的样品"
    else:
        assert restored.operators[restored.active_operator_id].warp_weight == pytest.approx(.123456)
    assert not harness.window._project_dirty()
    assert not harness.modal_messages and not harness.slot_errors


def test_expanded_metadata_stays_visible_and_review_report_tabs_do_not_shrink_image(trial_host):
    view = show_trial(trial_host)
    view.sample_details_toggle.click()
    QApplication.processEvents()
    viewport = view.image_scroll.geometry()
    for index in range(3):
        view.workflow_tabs.setCurrentIndex(index)
        QApplication.processEvents()
        assert view.sample.isVisible() and view.sample_number.isVisible()
        assert view.operator.isVisible() and view.ruler_choice.isVisible()
        assert view.image_scroll.geometry() == viewport
        image_rect = view.image_scroll.rect().translated(view.image_scroll.mapTo(view, QPoint()))
        side_rect = view.trial_sidebar.rect().translated(view.trial_sidebar.mapTo(view, QPoint()))
        assert not image_rect.intersects(side_rect)
    assert not view.settings_dialog.isAncestorOf(view.sample)


def test_category_rebuild_never_shows_a_parentless_label(trial_host):
    class ShowObserver(QObject):
        def __init__(self):
            super().__init__()
            self.windows = []

        def eventFilter(self, watched, event):
            if event.type() == QEvent.Type.Show and isinstance(watched, QLabel) and watched.isWindow():
                self.windows.append(watched.text())
            return False

    observer = ShowObserver()
    app = QApplication.instance()
    app.installEventFilter(observer)
    try:
        for mode in ("content", "diameter") * 3:
            view = show_trial(trial_host, mode)
            view._refresh_materials()
            QApplication.processEvents()
    finally:
        app.removeEventFilter(observer)
    assert observer.windows == []


def test_weights_are_directly_editable_and_undo_restores_them(trial_host):
    view = show_trial(trial_host)
    assert view.warp_weight.isVisible() and view.weft_weight.isVisible()
    view.warp_weight.setValue(.123456)
    QApplication.processEvents()
    assert view.warp_weight.text() == "0.123456"
    assert view.warp_weight.fontMetrics().horizontalAdvance(view.warp_weight.text()) <= view.warp_weight.lineEdit().width() - 4
    view.warp_weight.setValue(70)
    view.weft_weight.setValue(30)
    view.warp_weight.editingFinished.emit()
    operator = view.session.operators[view.session.active_operator_id]
    assert (operator.warp_weight, operator.weft_weight) == (70, 30)
    view._history_action("undo")
    assert (view.warp_weight.value(), view.weft_weight.value()) == (0, 0)
    diameter = show_trial(trial_host, "diameter")
    assert not diameter.weights_button.isVisible()
    assert not diameter.material_panel.isVisible()
    assert diameter.diameter_summary.isVisible()


def test_trial_layout_preferences_survive_reopen_without_dirtying_project(trial_host):
    view = show_trial(trial_host)
    view.workflow_tabs.setCurrentIndex(2)
    view.body_splitter.setSizes([850, 460])
    view._persist_view_preferences()
    saved = json.loads(view._view_preferences_path.read_text())["content"]
    assert saved["tab"] == 2
    view.close()
    trial_host.pages.removeWidget(view)
    reopened = show_trial(trial_host)
    assert reopened.workflow_tabs.currentIndex() == 2
    assert abs(reopened.trial_sidebar.width() - saved["sidebar_width"]) <= 2
    assert not trial_host.project.live_experiments
    assert trial_host.project.extension_state_id == 0


def test_next_step_links_route_to_the_relevant_control(trial_host, monkeypatch):
    view = show_trial(trial_host)
    calls = []
    monkeypatch.setattr(trial_host, "show_capture_device_menu", lambda: calls.append("device"), raising=False)
    monkeypatch.setattr(view.ruler_choice, "showPopup", lambda: calls.append("ruler"))
    monkeypatch.setattr(view, "_rename_operator", lambda: calls.append("operator"))
    monkeypatch.setattr(view, "_materials", lambda: calls.append("materials"))
    for target in ("device", "ruler", "operator", "materials"):
        view.readiness_label.linkActivated.emit(target)
    assert calls == ["device", "ruler", "operator", "materials"]
    assert not trial_host.project.live_experiments


def test_pending_measurement_opens_review_in_sidebar_without_covering_image(workspace):
    from PySide6.QtTest import QTest
    workspace.workflow_tabs.setCurrentIndex(2)
    QApplication.processEvents()
    image_geometry = workspace.image_scroll.geometry()
    workspace.controller.start((30, 30))
    workspace.controller.finish((30, 70), "cashmere", method="free")
    for _ in range(100):
        QTest.qWait(2)
        workspace.controller.poll()
        if workspace.controller.pending_classification:
            break
    assert workspace.controller.pending_classification
    workspace._refresh_summary()
    QApplication.processEvents()
    assert workspace.workflow_tabs.currentIndex() == 0
    assert workspace.pending_panel.isVisible()
    assert workspace.image_scroll.geometry() == image_geometry
    scroll = workspace.workflow_tabs.widget(0)
    assert scroll.viewport().rect().contains(workspace.pending_buttons[0].mapTo(scroll.viewport(), QPoint(1, 1)))
