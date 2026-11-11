"""Category defaults keep measured and count-only goals independent."""
from dataclasses import replace

import pytest

from fdm.live_experiment.material_catalog import MaterialCatalog
from fdm.live_experiment.models import ExperimentSession, MaterialSnapshot
from fdm.live_experiment.reminders import ReminderPolicy, ReminderTracker, reminder_progress
from fdm.ui.material_catalog import MaterialCatalogDialog, TrialMaterialsDialog
from fdm.ui.quantity_targets import QuantityTargetDialog, QuantityTargets


def trial(*, measured=2, counted=3, mode="content", policy=None):
    session = ExperimentSession(workspace_options={"mode": mode}, reminder_policy=policy or ReminderPolicy())
    session.add_operator("甲", id="a")
    session.set_materials([MaterialSnapshot("wool", "羊毛", 1.31,
                                          measured_target=measured, count_target=counted)])
    session.establish_history_baseline()
    return session


def test_measured_and_count_only_goals_are_independent_and_total_includes_both():
    session = trial()
    tracker = ReminderTracker(session)
    session.add_measurement(20, "wool")
    assert tracker.update(session) is None
    session.adjust_count("wool", 2)
    assert tracker.update(session) is None
    session.add_measurement(21, "wool")
    notice = tracker.update(session)
    assert [(item.kind, item.count, item.target) for item in notice.crossings] == [("measured", 2, 2)]
    assert session.reminder_state()["total"] == 4
    session.adjust_count("wool", 1)
    notice = tracker.update(session)
    assert [(item.kind, item.count, item.target) for item in notice.crossings] == [("counted", 3, 3)]
    assert "补计／计数" in notice.text
    state = session.reminder_state()
    assert state["measured"] == {"wool": 2} and state["counted"] == {"wool": 3}
    assert state["counted_reached"] == ["wool"] and state["total"] == 5
    session.adjust_count("wool", 10)
    session.add_measurement(22, "wool")
    assert tracker.update(session) is None  # Allowed to continue, not a hard cap.


def test_count_batch_crossing_undo_redo_and_reopen_do_not_repeat_notice():
    session = trial(counted=5)
    tracker = ReminderTracker(session)
    session.adjust_count("wool", 8)
    assert tracker.update(session).crossings[0].target == 5
    session.undo()
    assert tracker.update(session) is None
    session.redo()
    assert tracker.update(session) is None
    restored = ExperimentSession.from_dict(session.to_dict())
    assert restored.material("wool").count_target == 5
    assert ReminderTracker(restored).update(restored) is None


def test_category_goals_are_scoped_by_operator_and_direction():
    session = trial(measured=10, counted=2)
    original = session.material("wool")
    session.set_materials([original], direction="weft")
    session.add_operator("乙", id="b")
    session.set_materials([original], operator_id="b", direction="warp")
    tracker = ReminderTracker(session)
    for owner, direction in (("a", "warp"), ("a", "weft"), ("b", "warp")):
        session.adjust_count("wool", 2, operator_id=owner, direction=direction)
        notice = tracker.update(session)
        assert [(x.kind, x.operator_id, x.direction) for x in notice.crossings] == [("counted", owner, direction)]


def test_trial_override_wins_and_blank_category_goals_preserve_existing_defaults():
    session = trial(measured=12, counted=None, policy=ReminderPolicy(material_targets={"wool": 8}))
    assert session.reminder_state()["measured_targets"] == {"wool": 8}
    assert session.reminder_state()["counted_targets"] == {}
    session.configure(reminder_policy=ReminderPolicy(content_measured_target=30))
    assert session.reminder_state()["measured_targets"] == {"wool": 12}
    session.set_materials([replace(session.material("wool"), measured_target=None)])
    assert session.reminder_state()["measured_targets"] == {"wool": 30}


def test_explicit_category_goals_apply_to_legacy_content_trials_at_equality():
    session = trial(measured=2, counted=3)
    session.configure(reminder_policy=None, measured_reminder=7)
    tracker = ReminderTracker(session)
    session.add_measurement(20, "wool")
    assert tracker.update(session) is None
    session.add_measurement(20, "wool")
    assert [row.kind for row in tracker.update(session).crossings] == ["measured"]
    session.adjust_count("wool", 3)
    notice = tracker.update(session)
    assert [(row.kind, row.count, row.target, row.legacy) for row in notice.crossings] == [("counted", 3, 3, False)]
    assert session.reminder_state()["total"] == 5
    saved = session.to_dict()
    assert "reminder_policy" not in saved
    restored = ExperimentSession.from_dict(saved)
    assert restored.reminder_state()["measured_targets"] == {"wool": 2}
    assert restored.reminder_state()["counted_targets"] == {"wool": 3}
    assert ReminderTracker(restored).update(restored) is None
    restored.set_materials([replace(restored.material("wool"), measured_target=None, count_target=None)])
    assert restored.reminder_state()["measured_targets"] == {"wool": 7}
    assert restored.reminder_state()["counted_targets"] == {}


def test_diameter_trial_uses_trial_goal_not_content_category_counts():
    session = trial(mode="diameter", measured=1, counted=1)
    tracker = ReminderTracker(session)
    session.add_measurement(20, "wool")
    session.adjust_count("wool", 1)
    assert tracker.update(session) is None
    rows = reminder_progress(session)
    assert len(rows) == 1 and rows[0].kind == "diameter_measured" and rows[0].target == 300


def test_changing_category_goal_seeds_existing_records_and_only_future_crossing_notifies():
    session = trial(measured=10, counted=None)
    session.adjust_count("wool", 3)
    tracker = ReminderTracker(session)
    session.set_materials([replace(session.material("wool"), count_target=2)])
    assert tracker.update(session) is None  # Enabling a goal does not replay old counts.
    session.set_materials([replace(session.material("wool"), count_target=5)])
    assert tracker.update(session) is None
    session.adjust_count("wool", 2)
    assert tracker.update(session).crossings[0].target == 5
    session.set_materials([replace(session.material("wool"), name="羊毛新名")])
    assert tracker.update(session) is None


@pytest.mark.parametrize("key", ["measured_target", "count_target"])
@pytest.mark.parametrize("value", [0, -1, 2.5, True, "10"])
def test_category_targets_reject_invalid_values(key, value):
    with pytest.raises(ValueError, match="提醒根数"):
        MaterialSnapshot("wool", "羊毛", **{key: value})


def test_library_targets_roundtrip_without_mutating_trial_snapshots(tmp_path):
    path = tmp_path / "materials.json"
    catalog = MaterialCatalog()
    catalog.upsert(replace(catalog.get("wool"), measured_target=17, count_target=60))
    catalog.save(path)
    loaded = MaterialCatalog.load(path)
    session = trial()
    session.set_materials([loaded.get("wool")])
    loaded.upsert(replace(loaded.get("wool"), count_target=90))
    assert session.material("wool").count_target == 60
    loaded.delete("wool")
    assert session.material("wool").measured_target == 17
    assert ExperimentSession.from_dict(session.to_dict()).material("wool").count_target == 60


def test_category_goals_survive_project_file_and_library_file_independently(tmp_path):
    from fdm.models import ProjectState
    from fdm.project_io import ProjectIO
    session = trial(measured=13, counted=47)
    session._content_total_target = 99
    session.add_measurement(20, "wool", evidence={"state": "disabled"})
    project = ProjectState.empty()
    project.live_experiments.append(session)
    path = tmp_path / "类别提醒.fdmproj"
    ProjectIO.save(project, path)
    restored = ProjectIO.load(path).live_experiments[0]
    assert (restored.material("wool").measured_target, restored.material("wool").count_target) == (13, 47)
    assert not hasattr(restored, "_content_total_target")
    library = MaterialCatalog([restored.material("wool")])
    library_path = tmp_path / "类别库.json"
    library.save(library_path)
    exported = MaterialCatalog.load(library_path)
    assert exported.get("wool") == session.material("wool")


def test_library_inline_editor_saves_both_targets_and_invalid_edit_stays_draft(tmp_path):
    path = tmp_path / "materials.json"
    dialog = MaterialCatalogDialog(path=path)
    try:
        identity = dialog.editor.original.id
        dialog.editor.measured_target.setText("23")
        dialog.editor.count_target.setText("77")
        dialog.accept()
        material = MaterialCatalog.load(path).get(identity)
        assert (material.measured_target, material.count_target) == (23, 77)
    finally:
        dialog.deleteLater()
    invalid = MaterialCatalogDialog(path=path)
    try:
        invalid.editor.count_target.setText("2.5")
        invalid.accept()
        assert invalid.result() == 0 and "正整数" in invalid.editor.error.text()
        assert MaterialCatalog.load(path).get(identity).count_target == 77
    finally:
        invalid.deleteLater()


def test_trial_editor_and_quantity_dialog_show_inherited_category_goals():
    session = trial(measured=17, counted=41)
    dialog = TrialMaterialsDialog(session, catalog=MaterialCatalog())
    quantities = QuantityTargetDialog(session)
    widget = QuantityTargets(session, session.configure)
    try:
        dialog.edit()
        assert dialog.editor.measured_target.text() == "17"
        assert dialog.editor.count_target.text() == "41"
        assert quantities.material_editors["wool"].text() == "使用类别参数（17 根）"
        assert "羊毛实测 0 / 17" in widget.progress.toolTip()
        assert "羊毛补计／计数 0 / 41" in widget.progress.toolTip()
        before = session.to_dict()
        dialog.editor.count_target.clear()
        assert dialog._back()
        assert dialog.materials()[0].count_target is None
        dialog.reject()
        assert session.to_dict() == before
    finally:
        dialog.deleteLater()
        quantities.deleteLater()
        widget.deleteLater()


def test_shared_total_override_is_runtime_only_and_reseeds_reminders(tmp_path):
    session = trial(measured=10, counted=None)
    session.adjust_count("wool", 3)
    before = session.to_dict()
    tracker = ReminderTracker(session)
    session._content_total_target = 2
    assert session.reminder_state()["total_target"] == 2
    assert tracker.update(session) is None
    assert session.to_dict() == before
    session._content_total_target = 5
    assert tracker.update(session) is None
    session.adjust_count("wool", 3)
    notice = tracker.update(session)
    assert [(row.kind, row.target) for row in notice.crossings] == [("total", 5)]
    dialog = QuantityTargetDialog(session, defaults_file=tmp_path / "defaults.json")
    try:
        assert dialog.total.value() == 5 and not dialog.total.isEnabled()
        assert not dialog.legacy_total.isEnabled()
        assert dialog.changes()["reminder_policy"].content_total_target == 1500
        dialog.save_default.click()
        from fdm.live_experiment.trial_defaults import new_trial_defaults
        assert new_trial_defaults("content", path=tmp_path / "defaults.json")["reminder_policy"].content_total_target == 5
    finally:
        dialog.deleteLater()
