"""User presentation choices must never become scientific project edits."""
import json

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from fdm.ui.main_window import MainWindow


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr("fdm.services.capture.available_capture_backends", lambda: [])
    result = MainWindow()
    result.resize(1600, 900)
    result.show()
    QApplication.processEvents()
    yield result
    monkeypatch.setattr(result, "_confirm_close_documents_for_transition", lambda *_: True)
    result.close()


def test_empty_trial_visits_and_display_choices_do_not_offer_project_save(window, monkeypatch):
    window.acquisition_workspace_action.trigger()
    hub = window._acquisition_workbench
    for index in (1, 2, 0, 1):
        hub.select_page(index)
        QApplication.processEvents()
    window._active_acquisition_display_binding().set_mode("percent", 175)
    window.measure_workspace_action.trigger()
    QApplication.processEvents()
    assert not window.project.live_experiments
    assert not window._project_dirty()
    def unexpected_dialog(*_):
        raise AssertionError("An untouched trial must not ask to save")
    monkeypatch.setattr(QMessageBox, "exec", unexpected_dialog)
    assert window._confirm_close_documents(window.project.documents)


def test_save_summary_names_only_changed_trial_and_rebases_after_save(window):
    window._open_acquisition_workbench(trial_index=1)
    content = window._live_experiment_workspace
    content._configure(sample_name="羊毛样品 A")
    assert window.project.live_experiments == [content.session]
    summary = "\n".join(window._unsaved_project_summary())
    assert "含量试验“羊毛样品 A”" in summary
    assert "项目默认标尺" not in summary
    window._mark_project_saved()
    assert window._unsaved_project_summary() == []
    window._acquisition_workbench.select_page(2)
    diameter = window._live_experiment_workspace
    diameter._configure(sample_name="直径样品 B")
    summary = "\n".join(window._unsaved_project_summary())
    assert "直径试验“直径样品 B”" in summary
    assert "羊毛样品 A" not in summary
    assert window._project_dirty()


def test_measurement_panels_keep_user_visibility_across_hidden_trial_and_restart(window):
    window._project_dock.show()
    window._inspector_dock.show()
    window._results_dock.show()
    QApplication.processEvents()
    layout = window._adaptive_layout
    original = dict(layout._wide_visibility)
    assert all(original.values())
    window._open_acquisition_workbench(trial_index=1)
    QApplication.processEvents()
    assert all(not dock.isVisible() for dock in (window._project_dock, window._inspector_dock, window._results_dock))
    layout.persist_profiles()
    saved = json.loads(layout._profiles_path().read_text())
    assert saved["workspaces"]["measure"]["visibility"] == original
    window.measure_workspace_action.trigger()
    QApplication.processEvents()
    assert all(dock.isVisible() for dock in (window._project_dock, window._inspector_dock, window._results_dock))
    # The stored preference, rather than trial's temporary hidden docks, is
    # restored even if Qt's global window snapshot came from a trial page.
    window._open_acquisition_workbench(trial_index=2)
    window._persist_window_geometry()
    other = MainWindow()
    try:
        other.resize(1600, 900)
        other.show()
        QApplication.processEvents()
        assert other._adaptive_layout._wide_visibility == original
        assert all(dock.isVisible() for dock in (other._project_dock, other._inspector_dock, other._results_dock))
    finally:
        other.close()
