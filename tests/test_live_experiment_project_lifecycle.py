"""An experiment is project data even after the last static document closes."""
from types import SimpleNamespace
from unittest.mock import Mock

from fdm.models import ProjectState
from fdm.live_experiment import ExperimentSession
from fdm.ui.main_window import MainWindow


def test_last_document_does_not_drop_experiment_only_project():
    project = ProjectState.empty()
    project.live_experiments.append(ExperimentSession())
    host = SimpleNamespace(project=project)
    assert MainWindow._end_project_session_if_empty(host) is False
    assert host.project is project
    assert host.project.live_experiments


def test_project_transition_stops_live_producer_before_save_confirmation():
    events = []
    host = SimpleNamespace(project=ProjectState.empty(),
        _prepare_live_experiment_close=lambda: events.append('stop') or True,
        _close_live_experiment=lambda **_: events.append('dispose') or True,
        _resume_live_experiment_after_cancel=lambda: events.append('resume'),
        _confirm_close_documents=lambda docs: events.append('confirm') or True)
    assert MainWindow._confirm_close_documents_for_transition(host, [], None)
    assert events == ['stop', 'confirm', 'dispose']


def test_pending_evidence_blocks_transition_before_confirmation():
    confirm = Mock()
    host = SimpleNamespace(project=ProjectState.empty(),
        _prepare_live_experiment_close=lambda: False, _confirm_close_documents=confirm)
    assert not MainWindow._confirm_close_documents_for_transition(host, [], None)
    confirm.assert_not_called()


def test_reset_workspace_does_not_drop_pending_live_evidence():
    import pytest
    project = ProjectState.empty()
    coordinator = SimpleNamespace(busy=False, reset=Mock())
    host = SimpleNamespace(project=project, project_save_coordinator=coordinator,
                           _close_live_experiment=lambda **_: False)
    with pytest.raises(RuntimeError, match='实时试验'):
        MainWindow._reset_workspace(host)
    assert host.project is project
    coordinator.reset.assert_not_called()


def test_experiment_only_project_saves_through_existing_window_entry(tmp_path, desktop_application):
    from fdm.project_io import ProjectIO
    from fdm.live_experiment import default_materials
    window = MainWindow()
    experiment = ExperimentSession(save_images=False)
    experiment.add_operator('操作员')
    experiment.set_materials(default_materials())
    experiment.add_measurement(14, 'wool')
    window.project.live_experiments.append(experiment)
    window.project.mark_extension_changed()
    path = tmp_path / 'live-only.fdmproj'
    try:
        result = window.save_project(str(path))
        assert result.success, result.message
        loaded = ProjectIO.load(path)
        assert not loaded.documents
        assert len(loaded.live_experiments[0].measurements) == 1
    finally:
        window._reset_workspace()
        window.close()
