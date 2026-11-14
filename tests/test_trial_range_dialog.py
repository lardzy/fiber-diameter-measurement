"""Real Qt choices retain a verified frame while the live view changes."""
from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from fdm.live_experiment.selection_retention import FieldEndCause
from test_live_experiment_content_workspace import workspace, present


def pending(view, material="cashmere", diameter=40):
    controller = view.controller
    controller.start((30, 30))
    controller.finish((30, 30 + diameter), material, method="free")
    for _ in range(200):
        QTest.qWait(2)
        controller.poll()
        if controller.has_range_decisions:
            break
    assert controller.has_range_decisions
    view._refresh_summary()
    QApplication.processEvents()
    return view._range_dialog


def test_modal_survives_escape_close_motion_timeout_disconnect_and_commits_source(workspace):
    controller = workspace.controller
    original_field = controller.field_id
    original_frame = controller.presented.frame.frame_id
    dialog = pending(workspace)
    original_preview = dialog.preview.pixmap().toImage()
    assert dialog.windowModality() == Qt.WindowModality.ApplicationModal
    assert not dialog.remember.isChecked()
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    dialog.close()
    assert dialog.isVisible() and controller.has_range_decisions
    controller.end_field(cause=FieldEndCause.MOTION)
    present(controller, 55)
    controller.unavailable("测试断流")
    QTest.qWait(2100)
    workspace._refresh_summary()
    assert workspace._range_dialog is dialog and dialog.isVisible()
    assert dialog.preview.pixmap().toImage() == original_preview
    assert not workspace._confirm_trial_navigation()
    assert not workspace.prepare_close()
    dialog.keep.click()
    assert not controller.has_range_decisions
    assert workspace._range_dialog is None
    record = next(iter(workspace.session.measurements.values()))
    assert record.field_id == original_field and record.frame_id == original_frame
    assert record.material_id == "cashmere" and record.diameter_um == 40
    assert not controller.current_records and workspace.current_model.ids == []
    assert workspace.history_model.ids == [record.id]


def test_remembered_reclassification_leaves_selected_category_and_normal_results_alone(workspace):
    controller = workspace.controller
    workspace._select_material("cashmere")
    selected = workspace.material.currentData()
    dialog = pending(workspace)
    dialog.target.setCurrentIndex(dialog.target.findData("wool"))
    dialog.remember.setChecked(True)
    dialog.reclassify.click()
    assert len(workspace.session.measurements) == 1
    assert workspace.material.currentData() == selected
    controller.start((30, 30))
    controller.finish((30, 70), "cashmere", method="free")
    for _ in range(200):
        QTest.qWait(2)
        controller.poll()
        if len(workspace.session.measurements) == 2:
            break
    assert not controller.has_range_decisions
    assert list(workspace.session.measurements.values())[-1].material_id == "wool"
    assert "改为" in workspace.statusBar().currentMessage()
    assert workspace.material.currentData() == selected
    controller.start((30, 30))
    controller.finish((30, 50), "cashmere", method="free")
    for _ in range(200):
        QTest.qWait(2)
        controller.poll()
        if len(workspace.session.measurements) == 3:
            break
    assert list(workspace.session.measurements.values())[-1].material_id == "cashmere"


def test_reclassify_still_outside_requires_explicit_second_choice_and_discard_writes_nothing(workspace):
    controller = workspace.controller
    materials = workspace.session.operators[workspace.session.active_operator_id].materials["warp"]
    workspace.session.set_materials([replace(m, upper_limit=25) if m.id == "wool" else m for m in materials])
    dialog = pending(workspace)
    dialog.target.setCurrentIndex(dialog.target.findData("wool"))
    dialog.remember.setChecked(True)
    dialog.reclassify.click()
    assert controller.has_range_decisions and not workspace.session.measurements
    assert "仍然越限" in dialog.details.text()
    assert not controller.range_rules.list_rules(workspace.session.id)
    assert not dialog.remember.isChecked()
    revision = workspace.session.revision
    dialog.discard.click()
    assert not controller.has_range_decisions and not workspace.session.measurements
    assert workspace.session.revision == revision


def test_missing_suggestion_config_editor_is_owned_by_modal(workspace, monkeypatch):
    dialog = pending(workspace)
    calls = []
    monkeypatch.setattr(workspace, "_materials", lambda **kwargs: calls.append(kwargs))
    dialog.configureRequested.emit("missing")
    assert calls == [{"suggested_id": "missing", "parent": dialog}]
    assert dialog.isVisible()
    dialog.discard.click()


def test_external_category_removal_keeps_original_preview_and_discard_available(workspace):
    dialog = pending(workspace)
    materials = workspace.session.operators[workspace.session.active_operator_id].materials["warp"]
    # Normal configuration protects in-use categories; exercise an external
    # revision as well so a diagnostic failure cannot trap the modal dialog.
    workspace.session.set_materials([item for item in materials if item.id != "cashmere"])
    workspace._refresh_summary()
    assert dialog.isVisible() and not dialog.keep.isEnabled()
    assert "类别已不可用" in dialog.details.text()
    dialog.discard.click()
    assert not workspace.controller.has_range_decisions
