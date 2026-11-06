"""Real navigation, stopped cameras and restored Qt state share one policy."""
import pytest
from PySide6.QtWidgets import QApplication

from fdm.ui.workspace import WorkspacePage
from test_metrology_workspace import mounted_window, professional_slide


@pytest.fixture(autouse=True)
def no_camera(monkeypatch):
    monkeypatch.setattr("fdm.services.capture.available_capture_backends", lambda: [])


def settle():
    for _ in range(4):
        QApplication.processEvents()


def test_welcome_layout_disables_document_tools(desktop_application):
    from fdm.ui.main_window import MainWindow
    window = MainWindow()
    try:
        window.show()
        settle()
        assert window._center_stack.currentWidget() is window._welcome_panel
        assert window._adaptive_layout.page is WorkspacePage.MEASURE
        assert window._adaptive_layout.panel_allowed("professional")
        window.professional_workspace.show()
        assert not window.professional_workspace.panel.isEnabled()
        for mode in ("manual", "snap", "construction", "professional", "polygon_area"):
            assert not window._mode_actions[mode].isEnabled()
            window.set_tool_mode(mode)
            assert window._tool_mode == "select"
        assert not window._manual_tool_button.isEnabled()
    finally:
        window.close()


@pytest.mark.parametrize("entry", ["file", "folder", "drop", "associated"])
def test_open_document_waits_for_accepted_trial_navigation(entry, mounted_window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from fdm.application_launch import build_application_open_request
    from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore

    window, original, _, _ = mounted_window()
    hub = window._open_acquisition_workbench(trial_index=1, auto_start=False)
    folder = tmp_path / "图片"
    folder.mkdir()
    slide = folder / "样品.fdmslide"
    DigitalSlideStore.create(slide, DigitalSlideManifest(1, 64, 48, 32, 24, [0])).close()
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *_: ([str(slide)], ""))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_: str(folder))
    def association():
        window.enqueue_application_open_request(build_application_open_request([slide], source="test"))
        window.associated_file_open_controller._dispatch_next()
    navigate = {"file": window.open_images, "folder": window.open_folder,
                "drop": lambda: window._open_dropped_paths([slide]), "associated": association}[entry]
    trial = hub.pages.currentWidget()
    for accept in (False, True):
        monkeypatch.setattr(trial, "_confirm_trial_navigation", lambda: accept)
        navigate()
        settle()
        if not accept:
            assert window.project.documents == [original]
            assert window._adaptive_layout.page is WorkspacePage.CONTENT_TRIAL
            assert window._center_stack.currentWidget() is hub and hub.pages.currentWidget() is trial
        else:
            assert len(window.project.documents) == 2
            assert window.current_document().is_digital_slide()
            assert window._adaptive_layout.page is WorkspacePage.MEASURE
            assert window._center_stack.currentWidget() is window.tab_widget


def test_cancelled_chooser_or_busy_slide_keeps_page_and_skips_loading(mounted_window, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    window, original, _, _ = mounted_window()
    hub = window._open_acquisition_workbench(trial_index=0, auto_start=False)
    window._digital_slide_mode = True
    window._sync_digital_slide_mode_ui()
    with monkeypatch.context() as guarded:
        guarded.setattr(window, "stop_live_preview", lambda: pytest.fail("Chooser stopped the camera"))
        guarded.setattr(window, "_open_image_requests", lambda *a, **kw: pytest.fail("Unexpected file load"))
        guarded.setattr(QFileDialog, "getOpenFileNames", lambda *_: ([], ""))
        guarded.setattr(QFileDialog, "getExistingDirectory", lambda *_: "")
        window.open_images()
        window.open_folder()
        guarded.setattr(QFileDialog, "getOpenFileNames", lambda *_: (["requested.png"], ""))
        guarded.setattr(QMessageBox, "information", lambda *_: None)
        guarded.setattr(window, "_slide_acquisition_active", lambda: True)
        window.open_images()
    assert window._adaptive_layout.page is WorkspacePage.DIGITAL_SLIDE_ACQUIRE
    assert window._center_stack.currentWidget() is hub
    assert window.project.documents == [original]
    window._activate_measure_workspace()


def test_project_or_standalone_request_respects_retained_trial(mounted_window, tmp_path, monkeypatch):
    from fdm.models import ProjectState
    from fdm.project_io import ProjectIO
    from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore
    from fdm.ui.associated_file_controller import AssociatedSlideDisposition

    window, original, _, _ = mounted_window()
    hub = window._open_acquisition_workbench(trial_index=2, auto_start=False)
    trial = hub.pages.currentWidget()
    monkeypatch.setattr(trial, "_confirm_trial_navigation", lambda: False)
    slide = tmp_path / "standalone.fdmslide"
    DigitalSlideStore.create(slide, DigitalSlideManifest(1, 64, 48, 32, 24, [0])).close()
    project = tmp_path / "next.fdmproj"
    ProjectIO.save(ProjectState.empty(), project)
    with monkeypatch.context() as retained:
        retained.setattr(trial, "prepare_close", lambda: pytest.fail("Retained trial was retired"))
        result = window._load_project_from_path(project)
        assert result.cancelled
        window.associated_file_open_controller._open_digital_slides([slide], AssociatedSlideDisposition.STANDALONE_WORKSPACE)
    assert window.project.documents == [original]
    assert hub.pages.currentWidget() is trial and window._adaptive_layout.page is WorkspacePage.DIAMETER_TRIAL


@pytest.mark.parametrize("replacement", ["project", "standalone"])
@pytest.mark.parametrize("confirm_close", [True, False], ids=["accept", "stay_after_save_prompt"])
def test_replacement_asks_once_and_preserves_page_if_cancelled(replacement, confirm_close, mounted_window, tmp_path, monkeypatch):
    from fdm.models import ProjectState
    from fdm.project_io import ProjectIO
    from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore
    from fdm.ui.associated_file_controller import AssociatedSlideDisposition

    window, original, _, _ = mounted_window()
    hub = window._open_acquisition_workbench(trial_index=2, auto_start=False)
    trial = hub.pages.currentWidget()
    confirmations = []
    monkeypatch.setattr(trial, "_confirm_trial_navigation",
                        lambda: confirmations.append(True) or True)
    if replacement == "project":
        path = tmp_path / "next.fdmproj"
        ProjectIO.save(ProjectState.empty(), path)
        with monkeypatch.context() as saving:
            saving.setattr(window, "_confirm_close_documents", lambda *_: confirm_close)
            assert window._load_project_from_path(path).success is confirm_close
        if confirm_close:
            assert window._project_path == path
    else:
        path = tmp_path / "standalone.fdmslide"
        DigitalSlideStore.create(path, DigitalSlideManifest(1, 64, 48, 32, 24, [0])).close()
        with monkeypatch.context() as saving:
            saving.setattr(window, "_confirm_close_documents", lambda *_: confirm_close)
            window.associated_file_open_controller._open_digital_slides([path], AssociatedSlideDisposition.STANDALONE_WORKSPACE)
        if confirm_close:
            assert window.current_document().is_digital_slide()
    assert confirmations == [True]
    if confirm_close:
        assert window._adaptive_layout.page is WorkspacePage.MEASURE
    else:
        assert window.project.documents == [original]
        assert window._adaptive_layout.page is WorkspacePage.DIAMETER_TRIAL
        assert hub.pages.currentWidget() is trial


def test_real_navigation_blocks_dock_search_actions_and_background_tools(mounted_window):
    window, document, professional, plane = mounted_window()
    window.resize(1500, 850)
    settle()
    layout = window._adaptive_layout
    for key in ("project", "inspector", "results", "professional"):
        layout.set_panel_visible(key, True)
    layout.set_panel_visible("project", False)
    settle()
    original = dict(layout._wide_visibility)
    saved = window.saveState(2)
    before = document.to_dict()
    before.pop("view_state")  # Existing fit-to-window follows dock extents.
    dirty = window._document_has_unsaved_project_changes(document)
    mode = window._tool_mode
    hub = window._open_acquisition_workbench(trial_index=0, auto_start=False)
    docks = (window._project_dock, window._inspector_dock, window._results_dock, professional.dock)
    for index, page in ((0, WorkspacePage.CAPTURE), (1, WorkspacePage.CONTENT_TRIAL),
                        (2, WorkspacePage.DIAMETER_TRIAL), (0, WorkspacePage.CAPTURE)):
        hub.select_page(index)
        settle()
        assert not window._preview_active
        assert layout.page is page
        for show in (window._toggle_project_panel, window._toggle_inspector_panel,
                     window._toggle_results_panel, professional.show):
            show()
        for dock in docks:
            dock.show()
            dock.toggleViewAction().trigger()
        window.restoreState(saved, 2)
        window._update_action_states()
        window.set_tool_mode("professional")
        window.set_tool_mode("manual")
        settle()
        assert all(dock.isHidden() for dock in docks)
        assert window._tool_mode == mode
        assert not window._mode_actions["professional"].isEnabled()
        assert not window._mode_actions["manual"].isEnabled()
        assert not window.export_overlay_action.isEnabled()
        actions = [entry[0] for entry in window._command_search_entries()]
        assert window._mode_actions["professional"] not in actions
        assert window.toggle_project_panel_action not in actions
        menu = window.createPopupMenu()
        assert all(dock.toggleViewAction() not in menu.actions() for dock in docks)
        menu.deleteLater()
    window._activate_measure_workspace()
    settle()
    assert layout.page is WorkspacePage.MEASURE
    assert layout._wide_visibility == original
    assert professional.dock in window.tabifiedDockWidgets(window._inspector_dock)
    assert window._project_dock.isHidden()
    assert not professional.dock.isHidden()
    after = document.to_dict()
    after.pop("view_state")
    assert after == before
    assert window._document_has_unsaved_project_changes(document) == dirty
    assert plane.data == window._rasters[document.id].data


def test_cancelled_trial_or_measure_navigation_keeps_policy_and_records(mounted_window, monkeypatch):
    window, document, professional, _ = mounted_window()
    hub = window._open_acquisition_workbench(trial_index=1, auto_start=False)
    settle()
    trial = hub.pages.currentWidget()
    monkeypatch.setattr(trial, "_confirm_trial_navigation", lambda: False)
    starts = []
    monkeypatch.setattr(window, "start_live_preview", lambda: starts.append(True))
    before = window.saveState(2)
    for change in (lambda: hub.navigation.setCurrentIndex(2), window._activate_measure_workspace,
                   lambda: window._toggle_digital_slide_mode(True)):
        change()
        settle()
        assert hub.pages.currentWidget() is trial
        assert hub.navigation.currentIndex() == 1
        assert window._acquisition_workspace_selected
        assert window._adaptive_layout.page is WorkspacePage.CONTENT_TRIAL
        assert professional.dock.isHidden()
        assert window.saveState(2) == before
        assert not starts
        assert not window._digital_slide_mode
    monkeypatch.setattr(trial, "_confirm_trial_navigation", lambda: True)
    window._activate_measure_workspace()
    assert window._adaptive_layout.page is WorkspacePage.MEASURE


def test_slide_acquisition_policy_does_not_depend_on_preview_state(mounted_window):
    window, _, professional, _ = mounted_window()
    window._digital_slide_mode = True
    window._acquisition_workspace_selected = True
    try:
        window._sync_workspace_mode()
        window._sync_digital_slide_mode_ui()
        assert not window._preview_active
        assert window._adaptive_layout.page is WorkspacePage.DIGITAL_SLIDE_ACQUIRE
        assert window.toggle_project_panel_action.isVisible()
        assert window.toggle_project_panel_action.text() == "采集范围与焦层"
        assert window.toggle_inspector_panel_action.isVisible()
        assert not window.toggle_results_panel_action.isVisible()
        assert professional.dock.isHidden()
        assert not window._mode_actions["professional"].isEnabled()
        assert not window._digital_slide_left_panel.isHidden()
        assert window._left_standard_splitter.isHidden()
    finally:
        window._digital_slide_mode = False
        window._acquisition_workspace_selected = False
        window._sync_workspace_mode()


def test_background_slide_view_commands_and_popup_are_unavailable(professional_slide):
    window, document, _, _, _ = professional_slide
    canvas = window.current_canvas()
    window._open_slide_brightness_popup()
    assert window._slide_brightness_popup is not None
    window._open_acquisition_workbench(trial_index=1, auto_start=False)
    settle()
    assert window._slide_brightness_popup is None
    for action in (window.digital_slide_brightness_action, window.digital_slide_native_fit_action,
                   window.digital_slide_smooth_navigation_action, window.digital_slide_shift_navigation_action):
        assert not action.isEnabled()
    window._open_slide_brightness_popup()
    assert window._slide_brightness_popup is None
    assert canvas.display_adjustments == (0, 0)
