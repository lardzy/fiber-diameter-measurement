"""UI contracts for explicit candidate picking and application-only preferences."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.ui.live_experiment_workspace import CandidatesModel, _new_trial_session
from test_live_experiment_content_workspace import workspace


def candidates(view):
    controller = view.controller
    token = controller.presented
    controller._automatic = SimpleNamespace(cancel=lambda: None, poll=lambda: [], close=lambda: None)
    controller._automatic_context = {"task_id": "ui-batch", "batch_id": "ui-batch", "token": token,
        "lease": controller._acquire_intent_frame(token.frame), "calibration": controller.calibration,
        "operator_id": view.session.active_operator_id, "direction": view.session.active_direction,
        "algorithm": "traditional", "completed": True, "diagnostics": {}}
    controller.candidates = [dict(candidate_id=f"ui-batch:{i}", ordinal=i + 1, state="available",
        width_px=20, start=[30 + i * 60, 30], end=[30 + i * 60, 50]) for i in range(3)]
    controller.candidatesChanged.emit(controller.candidates)
    view.canvas.repaint()
    return [c["candidate_id"] for c in controller.candidates]


def point(view, x, y):
    scale, origin = view.canvas._transform()
    return QPoint(round(x * scale + origin[0]), round(y * scale + origin[1]))


def choose_mode(view, mode):
    control = view.measure_mode if mode in {"assisted", "free"} else view.input_mode
    control.setCurrentIndex(control.findData(mode))
    QApplication.processEvents()


def test_new_trials_default_crop_but_old_payload_remains_unchanged():
    from fdm.live_experiment import ExperimentSession
    for mode in ("content", "diameter"):
        session = _new_trial_session(mode)
        assert session.save_images
        assert session.effective_evidence_options["mode"] == "crop_jpeg"
        assert not session.can_undo
        assert not session.measurements
    for save in (False, True):
        session = ExperimentSession(save_images=save)
        payload = session.to_dict()
        restored = ExperimentSession.from_dict(deepcopy(payload))
        assert restored.to_dict() == payload
        assert restored.effective_evidence_options["mode"] == ("full_png" if save else "disabled")


def test_preferences_do_not_edit_trial_and_are_persisted(workspace):
    revision = workspace.session.revision
    workspace.auto_analysis.setChecked(True)
    workspace.candidate_opacity.setValue(37)
    workspace.show_candidates_action.setChecked(False)
    choose_mode(workspace, "candidate")
    workspace.algorithm.setCurrentIndex(1)
    workspace._persist_view_preferences()
    saved = json.loads(workspace._view_preferences_path.read_text())["assistance"]
    assert saved == {"version": 3, "automatic": True, "algorithm": "multiscale", "visible": False,
                     "opacity": 37, "input_mode": "candidate", "manual_algorithm": "assisted", "retention_seconds": 1.0}
    assert workspace.session.revision == revision
    workspace._apply_assistance_preferences({})
    assert not workspace.auto_analysis.isChecked()
    workspace._apply_assistance_preferences(saved)
    assert workspace.auto_analysis.isChecked()
    assert workspace.canvas.candidate_input_enabled
    assert not workspace.canvas.candidate_visible
    assert workspace.canvas.candidate_opacity == .37
    assert workspace.session.revision == revision


def test_automatic_defaults_off_and_does_not_run_for_static_picture(workspace):
    assert not workspace.auto_analysis.isChecked()
    workspace.auto_analysis.setChecked(True)
    workspace._refresh_assistance_state()
    assert not workspace.controller.automatic_active
    assert workspace.controller._automatic_context is None


def test_offline_diagnostics_include_metric_definitions_without_dirtying_trial(workspace, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    path = tmp_path / "offline-performance.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(path), "JSON (*.json)"))
    revision = workspace.session.revision
    workspace._export_assistance_diagnostics()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == "0.4.10"
    assert payload["performance_schema"]["version"] == 1
    assert "frame_age_ms" in payload["performance_schema"]["metrics"]
    assert payload["performance"]["analysis_ms"]["p95"] is None
    assert payload["frame_sampling"]["presented"]["window_fps"] is None
    assert workspace.session.revision == revision


def test_candidate_arrival_keeps_operation_tab_and_geometry(workspace):
    workspace.workflow_tabs.setCurrentIndex(0)
    QApplication.processEvents()
    geometry = workspace.category_scroll.geometry()
    viewport = workspace.image_scroll.geometry()
    candidates(workspace)
    QApplication.processEvents()
    assert workspace.workflow_tabs.currentIndex() == 0
    assert workspace.category_scroll.geometry() == geometry
    assert workspace.image_scroll.geometry() == viewport
    assert not workspace.candidate_feedback.isVisible()


def test_real_canvas_click_accepts_one_keeps_remaining_and_current_category(workspace):
    ids = candidates(workspace)
    choose_mode(workspace, "candidate")
    workspace.material.setCurrentIndex(workspace.material.findData("cashmere"))
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 90, 40))
    QApplication.processEvents()
    assert len(workspace.session.measurements) == 1
    record = next(iter(workspace.session.measurements.values()))
    assert record.material_id == "cashmere"
    assert [item["candidate_id"] for item in workspace.controller.candidates] == ids
    assert [item["state"] for item in workspace.controller.candidates] == ["available", "accepted", "available"]
    assert workspace.workflow_tabs.currentIndex() == 0
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 90, 40))
    assert len(workspace.session.measurements) == 1


def test_manual_tool_retains_its_click_and_point_tool_blank_is_inert(workspace):
    candidates(workspace)
    choose_mode(workspace, "manual")
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 30, 40))
    assert workspace.controller.draft is not None
    assert not workspace.session.measurements
    choose_mode(workspace, "candidate")
    assert workspace.controller.draft is None
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 230, 170))
    assert workspace.controller.draft is None
    assert not workspace.session.measurements


def test_accepted_candidate_category_tracks_reclassification_and_history(workspace):
    candidates(workspace)
    choose_mode(workspace, "candidate")
    QTest.mouseClick(workspace.material_rows["wool"]["select"], Qt.MouseButton.LeftButton)
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 30, 40))
    item = next(iter(workspace.session.measurements.values()))
    model = workspace.candidate_model
    assert model.data(model.index(0, 3)) == "绵羊毛"
    choose_mode(workspace, "mixed")
    workspace.classification_mode.setCurrentIndex(1)
    QTest.mouseClick(workspace.material_rows["cashmere"]["select"], Qt.MouseButton.LeftButton)
    assert workspace.session.measurements[item.id].material_id == "wool"
    # Deliberate history edits still update the linked candidate; merely
    # choosing the next category must not reclassify a candidate acceptance.
    workspace.session.edit_measurement(item.id, material_id="cashmere")
    workspace._rebuild_records()
    assert workspace.session.measurements[item.id].material_id == "cashmere"
    assert model.data(model.index(0, 3)) == "山羊绒"
    assert model.data(model.index(0, 3), Qt.ItemDataRole.UserRole) == "cashmere"
    workspace._history_action("undo")
    assert workspace.session.measurements[item.id].material_id == "wool"
    assert model.data(model.index(0, 3)) == "绵羊毛"
    workspace._history_action("redo")
    assert workspace.session.measurements[item.id].material_id == "cashmere"
    assert model.data(model.index(0, 3)) == "山羊绒"
    workspace.session.edit_measurement(item.id, diameter_um=23)
    workspace._rebuild_records()
    assert "原候选" in model.data(model.index(0, 2))
    assert "23.0000" in model.data(model.index(0, 2), Qt.ItemDataRole.ToolTipRole)


def test_hidden_candidates_cannot_be_clicked(workspace):
    candidates(workspace)
    choose_mode(workspace, "candidate")
    workspace.show_candidates_action.setChecked(False)
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 30, 40))
    assert not workspace.session.measurements
    assert workspace.controller.draft is None


def test_excluded_candidate_is_not_clickable_and_can_be_reenabled(workspace):
    candidates(workspace)
    choose_mode(workspace, "candidate")
    model = workspace.candidate_model
    index = model.index(0, 0)
    assert model.setData(index, Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    assert workspace.controller.candidates[0]["state"] == "excluded"
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 30, 40))
    assert not workspace.session.measurements
    assert model.setData(model.index(0, 0), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
    assert workspace.controller.candidates[0]["state"] == "available"
    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point(workspace, 30, 40))
    assert len(workspace.session.measurements) == 1


def test_hover_table_link_never_changes_tab_or_view(workspace):
    ids = candidates(workspace)
    view = workspace.canvas._transform()
    workspace._candidate_hovered(ids[2])
    assert workspace.candidate_table.currentIndex().row() == 2
    assert workspace.workflow_tabs.currentIndex() == 0
    workspace.candidate_table.setCurrentIndex(workspace.candidate_model.index(1, 1))
    assert workspace.canvas.selected_candidate_id == ids[1]
    assert workspace.canvas._transform() == view


def test_model_retains_assignment_exclusion_and_range_confirmation_by_id(workspace):
    candidates(workspace)
    model = workspace.candidate_model
    model.setData(model.index(1, 3), "cashmere")
    model.setData(model.index(2, 0), Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    workspace.controller.candidates[0]["state"] = "accepted"
    workspace.controller.candidatesChanged.emit(workspace.controller.candidates)
    assert model.assignments == {1: "cashmere"}
    assert model.excluded == {2}
    assert model.data(model.index(0, 0)) == "已计入"
    assert model.data(model.index(2, 1)) == "C3"
    assert not model.setData(model.index(0, 3), "cashmere")


def test_point_mode_does_not_apply_manual_post_classification(workspace):
    record = workspace.session.add_measurement(20, "wool", field_id=workspace.controller.field_id)
    workspace.controller.current_records.append(record.id)
    workspace.classification_mode.setCurrentIndex(1)
    choose_mode(workspace, "candidate")
    workspace.material.setCurrentIndex(workspace.material.findData("cashmere"))
    workspace._after_select()
    assert workspace.session.measurements[record.id].material_id == "wool"
    choose_mode(workspace, "mixed")
    assert workspace.classification_mode.currentIndex() == 1
    assert workspace.classification_mode.isEnabled()


def test_storage_mode_is_business_configuration_and_survives_sync(workspace):
    before = workspace.session.revision
    workspace.evidence_mode_actions["crop_jpeg"].trigger()
    assert workspace.session.revision > before
    workspace._sync_configuration_controls()
    workspace._refresh_assistance_state()
    assert workspace.session.effective_evidence_options["mode"] == "crop_jpeg"
    assert workspace.evidence_storage.currentData() == "crop_jpeg"
    assert "局部" in workspace.evidence_button.toolTip()
    assert workspace.evidence_button.text() == ""
    assert not workspace.evidence_button.icon().isNull()


def test_current_value_opens_last_evidence_after_field_moves(workspace, monkeypatch):
    record = workspace.session.add_measurement(20, "wool", field_id=workspace.controller.field_id)
    workspace.evidence_review.accepted(record)
    workspace.controller.end_field("moving")
    shown = []
    monkeypatch.setattr(workspace.evidence_review, "show_record", lambda value, **kw: shown.append((value.id, kw)))
    QTest.mouseClick(workspace.latest_value, Qt.MouseButton.LeftButton)
    assert shown == [(record.id, {"latest": True})]
    assert "20.000" in workspace.latest_value.text()
    assert workspace.measurement_outcome.text() == "上一条"


def test_fully_consumed_candidates_do_not_prompt_navigation(workspace, monkeypatch):
    ids = candidates(workspace)
    for key in ids:
        workspace.controller.submit_candidate(key, "wool")
    assert not workspace.controller.has_unaccepted_candidates
    assert workspace._confirm_trial_navigation()


def test_pending_click_does_not_insert_a_sidebar_message_row(workspace):
    from concurrent.futures import Future
    from test_live_experiment_content_workspace import present
    workspace.controller._motion_future = (workspace.controller.field_epoch, Future())
    frame = present(workspace.controller, 2, live=True)
    from fdm.live_experiment.controller import CalibrationSnapshot
    workspace.controller.set_calibration(CalibrationSnapshot(1, 1))
    workspace.canvas.set_frame(frame)
    workspace.canvas.repaint()
    candidates(workspace)
    workspace.controller.state = "ready"
    workspace.controller.submit_candidate("ui-batch:0", "wool")
    workspace._refresh_summary()
    assert workspace.controller.candidate_pending_count == 1
    assert not workspace.pending_panel.isVisible()
    with pytest.raises(ValueError):
        workspace._history_action("undo")
    workspace._cancel_trial_gesture()
    assert workspace.controller.candidate_pending_count == 0


@pytest.mark.parametrize('legacy,algorithm', [('assisted','assisted'), ('free','free'), ('candidate','assisted')])
def test_legacy_input_preferences_migrate_only_application_choice(workspace, legacy, algorithm):
    payload = workspace.session.to_dict()
    workspace._apply_assistance_preferences({'version': 1, 'input_mode': legacy})
    assert workspace.input_mode.currentData() == 'mixed'
    assert workspace.measure_mode.currentData() == algorithm
    assert workspace.canvas.input_mode == 'mixed'
    assert workspace.gesture.isEnabled()
    assert workspace.session.to_dict() == payload
    for mode in ('manual', 'candidate', 'mixed'):
        preferences = dict(workspace._assistance_preferences(), input_mode=mode)
        workspace._apply_assistance_preferences(preferences)
        assert workspace.canvas.input_mode == mode
        assert workspace.gesture.isEnabled() == (mode != 'candidate')
        assert workspace.manual_gesture_button.isEnabled() == (mode != 'candidate')
        assert workspace.session.to_dict() == payload


def test_visible_manual_gesture_menu_updates_existing_saved_choice_and_undo(workspace):
    from fdm.live_experiment import ExperimentSession
    choose_mode(workspace, "mixed")
    button = workspace.manual_gesture_button
    assert button.isVisible() and button.isEnabled()
    assert button.text() == "两点"
    assert workspace.gesture.itemText(0) == "两点测量"
    assert workspace.gesture.itemText(1) == "按住拖线"
    assert workspace.manual_gesture_actions["click"].isChecked()
    preferences = workspace._assistance_preferences()
    category_geometry = workspace.category_scroll.geometry()
    viewport_geometry = workspace.image_scroll.geometry()
    controls_height = workspace.view_controls.height()
    menu = button.menu()
    menu.popup(button.mapToGlobal(button.rect().bottomLeft()))
    QApplication.processEvents()
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton,
                     pos=menu.actionGeometry(workspace.manual_gesture_actions["drag"]).center())
    QApplication.processEvents()
    assert workspace.gesture.currentIndex() == 1
    assert workspace.canvas.manual_gesture == "drag"
    assert button.text() == "拖线"
    assert "按住拖线" in button.accessibleName()
    assert "单击不建立手工首点" in button.toolTip()
    assert workspace.manual_gesture_actions["drag"].isChecked()
    assert ExperimentSession.from_dict(workspace.session.to_dict()).workspace_options["gesture"] == "drag"
    assert workspace._assistance_preferences() == preferences
    assert workspace.category_scroll.geometry() == category_geometry
    assert workspace.image_scroll.geometry() == viewport_geometry
    assert workspace.view_controls.height() == controls_height
    workspace._history_action("undo")
    assert workspace.gesture.currentIndex() == 0
    assert workspace.canvas.manual_gesture == "click"
    assert button.text() == "两点"
    assert workspace.manual_gesture_actions["click"].isChecked()
    workspace._history_action("redo")
    assert workspace.gesture.currentIndex() == 1
    assert workspace.canvas.manual_gesture == "drag"
    assert button.text() == "拖线"


def test_advanced_manual_gesture_and_interaction_mode_keep_quick_control_in_sync(workspace):
    choose_mode(workspace, "manual")
    workspace.gesture.setCurrentIndex(1)
    assert workspace.manual_gesture_button.text() == "拖线"
    choose_mode(workspace, "candidate")
    assert not workspace.gesture.isEnabled()
    assert not workspace.manual_gesture_button.isEnabled()
    assert "仅候选点选" in workspace.manual_gesture_button.toolTip()
    choose_mode(workspace, "mixed")
    assert workspace.gesture.isEnabled() and workspace.manual_gesture_button.isEnabled()
    assert workspace.gesture.currentIndex() == 1
    assert workspace.canvas.manual_gesture == "drag"
    assert workspace.manual_gesture_button.text() == "拖线"
    assert "短点击候选" in workspace.manual_gesture_button.toolTip()


def test_advanced_evidence_storage_and_menu_share_undoable_policy(workspace):
    original = workspace.session.effective_evidence_options['mode']
    record = workspace.session.add_measurement(20, 'wool', field_id=workspace.controller.field_id,
        evidence={'state': 'saved', 'path': 'existing.png', 'mode': 'full_png'})
    original_record = deepcopy(record.evidence)
    workspace.settings_dialog.show()
    for mode in ('crop_jpeg', 'full_png', 'disabled'):
        workspace.evidence_storage.setCurrentIndex(workspace.evidence_storage.findData(mode))
        workspace._refresh_assistance_state()
        assert workspace.session.effective_evidence_options['mode'] == mode
        assert workspace.evidence_mode_actions[mode].isChecked()
        assert workspace.session.measurements[record.id].evidence == original_record
    workspace.evidence_mode_actions['crop_jpeg'].trigger()
    assert workspace.evidence_storage.currentData() == 'crop_jpeg'
    workspace._history_action('undo')
    assert workspace.evidence_storage.currentData() == 'disabled'
    workspace._history_action('redo')
    assert workspace.evidence_storage.currentData() == 'crop_jpeg'
    assert original == 'disabled'


def test_album_button_opens_single_nonmodal_window_without_sidebar_reflow(workspace):
    revision = workspace.session.revision
    geometry = workspace.image_scroll.geometry()
    QTest.mouseClick(workspace.evidence_button, Qt.MouseButton.LeftButton, pos=QPoint(5, 8))
    gallery = workspace.host._evidence_gallery
    QApplication.processEvents()
    assert gallery.workspace is workspace and gallery.dialog.isVisible()
    assert not gallery.dialog.isModal()
    assert workspace.image_scroll.geometry() == geometry
    assert workspace.session.revision == revision
    QTest.mouseClick(workspace.evidence_button, Qt.MouseButton.LeftButton, pos=QPoint(5, 8))
    assert workspace.host._evidence_gallery is gallery
    workspace.hide()
    assert not gallery.dialog.isVisible()
    workspace.show()
    QApplication.processEvents()
    assert gallery.dialog.isVisible()
    gallery.dialog.close()
    assert not gallery.dialog.isVisible()
    assert workspace.session.revision == revision


@pytest.mark.parametrize('origin', ['automatic', 'manual'])
def test_empty_analysis_keeps_last_measurement_outcome_and_does_not_show_card(workspace, origin):
    from test_measurement_feedback import automatic_context
    record = workspace.session.add_measurement(20, 'wool', field_id=workspace.controller.field_id)
    workspace._record_accepted(record)
    before = workspace.latest_value.text(), workspace.measurement_outcome.text()
    revision = workspace.session.revision
    if origin == 'automatic':
        # A running automatic task requires the user's switch to remain on.
        workspace.auto_analysis.setChecked(True)
    automatic_context(workspace.controller, {'task_id': 'analysis-7',
        'result': {'status': 'empty', 'candidates': [], 'diagnostics': {}}})
    workspace.controller._automatic_context['origin'] = origin
    workspace.controller.poll()
    workspace._refresh_assistance_state()
    assert not workspace.measurement_feedback.isVisible()
    assert (workspace.latest_value.text(), workspace.measurement_outcome.text()) == before
    assert workspace.candidate_display_button.text() == '候选 0'
    assert '本视场未发现候选' in workspace.statusBar().currentMessage()
    assert workspace.session.revision == revision


def test_replacing_trial_rebinds_open_album_and_resets_filters(workspace):
    from test_live_experiment_content_workspace import make_session
    # The shared fixture tears down its original object after the replacement.
    workspace.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    workspace._show_evidence_gallery()
    gallery = workspace.host._evidence_gallery
    gallery.dialog.direction.setCurrentIndex(1)
    target = make_session()
    workspace._replace_experiment(target)
    replacement = workspace.host._live_experiment_workspace
    try:
        QApplication.processEvents()
        assert replacement is not workspace
        assert workspace.host._evidence_gallery is gallery
        assert gallery.workspace is replacement
        assert gallery.dialog.isVisible()
        assert gallery.dialog.direction.currentData() is None
        assert gallery.model.session is target
    finally:
        replacement.close()
