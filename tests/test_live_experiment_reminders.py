"""Quantity goals use accepted data and transient, non-replaying deliveries."""
from copy import deepcopy
from dataclasses import replace

import pytest

from fdm.live_experiment.models import ExperimentSession, default_materials
from fdm.live_experiment.reminders import (
    ReminderPolicy, ReminderTracker, all_reminder_progress, reminder_progress,
)


def session(mode="content", policy=None):
    value = ExperimentSession(workspace_options={"mode": mode},
                              reminder_policy=policy or ReminderPolicy())
    value.add_operator("甲", id="a")
    for direction in ("warp", "weft"):
        value.set_materials(default_materials(), direction=direction)
    value.establish_history_baseline()
    return value


def add(s, count, material="wool", **kwargs):
    return s.add_measurements_batch([
        {"diameter_um": 15 + i / 100, "material_id": material, **kwargs}
        for i in range(count)])


def test_new_defaults_and_versioned_owned_roundtrip():
    s = session()
    assert s.reminder_policy == ReminderPolicy()
    assert s.reminder_state()["measured_targets"] == {"wool": 300, "cashmere": 300}
    assert s.reminder_state()["total_target"] == 1500
    source = {"wool": 12}
    policy = ReminderPolicy(material_targets=source)
    source["wool"] = 99
    assert policy.material_target("wool") == 12
    s.configure(reminder_policy=policy)
    payload = s.to_dict()
    assert payload["reminder_policy"]["version"] == 1
    restored = ExperimentSession.from_dict(payload)
    assert restored.to_dict() == payload
    payload["reminder_policy"]["material_targets"]["wool"] = 77
    assert restored.reminder_policy.material_target("wool") == 12
    assert not restored.can_undo


@pytest.mark.parametrize("target", [1, 2, 300])
def test_n_minus_one_n_and_after_n_notify_once(target):
    s = session(policy=ReminderPolicy(content_measured_target=target, content_total_target=1000))
    tracker = ReminderTracker(s)
    add(s, target - 1)
    assert tracker.update(s) is None
    s.add_measurement(18, "wool")
    notice = tracker.update(s)
    assert len(notice.crossings) == 1
    assert (notice.crossings[0].count, notice.crossings[0].target) == (target, target)
    assert tracker.update(s) is None
    s.add_measurement(19, "wool")
    assert tracker.update(s) is None


def test_batch_crosses_multiple_targets_in_one_notice_and_one_undo():
    s = session(policy=ReminderPolicy(content_measured_target=2, content_total_target=3))
    tracker = ReminderTracker(s)
    revision = s.revision
    records = s.add_measurements_batch([
        {"diameter_um": 15, "material_id": material}
        for material in ("wool", "wool", "wool", "cashmere", "cashmere")])
    assert s.revision == revision + 1
    notice = tracker.update(s)
    assert {(item.kind, item.material_id, item.count) for item in notice.crossings} == {
        ("measured", "wool", 3), ("measured", "cashmere", 2), ("total", None, 5)}
    assert "当前 5 根" in notice.text
    s.undo()
    assert not s.measurements and tracker.update(s) is None
    s.redo()
    assert set(s.measurements) == {item.id for item in records}
    assert tracker.update(s) is None


def test_extra_counts_reach_total_but_not_measured_target_or_diameter_goal():
    s = session(policy=ReminderPolicy(content_measured_target=2, content_total_target=3))
    tracker = ReminderTracker(s)
    s.adjust_count("wool", 100)
    notice = tracker.update(s)
    assert [(item.kind, item.count) for item in notice.crossings] == [("total", 100)]
    assert s.reminder_state()["measured"]["wool"] == 0
    assert not s.measurements
    s.adjust_count("wool", -97)
    assert tracker.update(s) is None  # Decreasing back to N is not a crossing.
    diameter = session("diameter", ReminderPolicy(diameter_measured_target=2))
    tracker = ReminderTracker(diameter)
    diameter.adjust_count("wool", 1000)
    assert tracker.update(diameter) is None
    assert diameter.reminder_state()["total"] == 0


def test_diameter_combines_directions_and_classes_only_for_current_operator():
    s = session("diameter", ReminderPolicy(diameter_measured_target=3))
    s.add_operator("乙", id="b")
    tracker = ReminderTracker(s)
    add(s, 2, operator_id="a", direction="warp")
    add(s, 1, "cashmere", operator_id="b", direction="weft")
    assert tracker.update(s) is None
    s.add_measurement(15, "cashmere", operator_id="a", direction="weft")
    notice = tracker.update(s)
    assert [(item.operator_id, item.direction, item.kind, item.count)
            for item in notice.crossings] == [("a", None, "diameter_measured", 3)]
    assert len(all_reminder_progress(s)) == 2
    for direction in ("warp", "weft"):
        state = s.reminder_state(operator_id="a", direction=direction)
        assert state["total"] == 3 and state["total_target"] == 3
        assert state["measured"] == {} and state["total_scope"] == "operator_measured"


def test_three_operators_two_directions_have_independent_goals():
    s = session(policy=ReminderPolicy(content_measured_target=2, content_total_target=2))
    s.add_operator("乙", id="b")
    s.add_operator("丙", id="c")
    tracker = ReminderTracker(s)
    for owner in ("a", "b", "c"):
        for direction in ("warp", "weft"):
            add(s, 1, operator_id=owner, direction=direction)
    assert tracker.update(s) is None
    for owner in ("a", "b", "c"):
        for direction in ("warp", "weft"):
            add(s, 1, operator_id=owner, direction=direction)
            notice = tracker.update(s)
            assert len(notice.crossings) == 2
            assert all(item.operator_id == owner and item.direction == direction for item in notice.crossings)
    assert tracker.update(s) is None


def test_material_overrides_follow_id_after_reorder_rename_and_reclassification():
    policy = ReminderPolicy(content_measured_target=100, material_targets={"cashmere": 2})
    s = session(policy=policy)
    first = s.add_measurement(15, "wool")
    s.add_measurement(18, "cashmere")
    tracker = ReminderTracker(s)
    s.edit_measurement(first.id, material_id="cashmere")
    notice = tracker.update(s)
    assert [(item.material_id, item.target, item.count) for item in notice.crossings] == [("cashmere", 2, 2)]
    assert s.reminder_state()["total"] == 2
    s.undo()
    assert tracker.update(s) is None
    s.redo()
    assert tracker.update(s) is None
    wool, cashmere = s.operators["a"].materials["warp"]
    s.set_materials([replace(cashmere, name="改名羊绒"), wool])
    s.rename_operator("改名甲")
    assert tracker.update(s) is None
    assert s.reminder_state()["measured_targets"] == {"cashmere": 2, "wool": 100}
    assert reminder_progress(s)[0].material_name == "改名羊绒"


def test_delete_undo_redo_and_new_measurement_do_not_replay_goal():
    s = session(policy=ReminderPolicy(content_measured_target=2))
    tracker = ReminderTracker(s)
    records = add(s, 2)
    assert tracker.update(s) is not None
    s.delete_measurement(records[0].id)
    assert tracker.update(s) is None
    s.undo()
    assert tracker.update(s) is None
    s.redo()
    assert tracker.update(s) is None
    s.add_measurement(17, "wool")
    assert tracker.update(s) is None


def test_undo_of_old_deletion_cannot_deliver_first_notification():
    s = session(policy=ReminderPolicy(content_measured_target=2))
    records = add(s, 2)
    s.delete_measurement(records[0].id)
    tracker = ReminderTracker(s)
    s.undo()
    assert tracker.update(s) is None
    s.redo()
    assert tracker.update(s) is None
    s.add_measurement(17, "wool")
    assert tracker.update(s) is None


def test_open_and_reopen_seed_without_project_changes_or_notification_history():
    s = session(policy=ReminderPolicy(content_measured_target=2))
    add(s, 4)
    payload = deepcopy(s.to_dict())
    tracker = ReminderTracker(s)
    assert tracker.update(s) is None
    assert s.to_dict() == payload
    restored = ExperimentSession.from_dict(payload)
    other = ReminderTracker(restored)
    assert other.update(restored) is None
    assert restored.to_dict() == payload
    assert not any("seen" in key or "notification" in key for key in payload)


def test_changed_threshold_seeds_then_future_crossing_notifies_and_config_undo_works():
    s = session()
    add(s, 3)
    tracker = ReminderTracker(s)
    revision = s.revision
    s.configure(reminder_policy=ReminderPolicy(content_measured_target=2))
    assert s.revision == revision + 1
    assert tracker.update(s) is None  # Lowering a goal does not replay old data.
    s.undo()
    assert s.reminder_policy == ReminderPolicy()
    assert tracker.update(s) is None
    s.redo()
    assert s.reminder_policy.content_measured_target == 2
    assert tracker.update(s) is None
    s.configure(reminder_policy=ReminderPolicy(content_measured_target=5))
    assert tracker.update(s) is None
    add(s, 3)
    assert tracker.update(s).crossings[0].target == 5
    revision = s.revision
    s.configure(reminder_policy=s.reminder_policy.to_dict())
    assert s.revision == revision  # No-op edits do not dirty the project.


@pytest.mark.parametrize("mode", ["content", "diameter"])
def test_legacy_payload_keeps_original_thresholds_scope_revision_and_missing_policy(mode):
    s = session(mode)
    s.configure(measured_reminder=7, total_reminder=13)
    s.add_measurement(18, "wool", direction="warp")
    s.add_measurement(19, "cashmere", direction="weft")
    s.adjust_count("wool", 12)
    payload = s.to_dict()
    payload.pop("reminder_policy")
    payload["revision"] = 271
    restored = ExperimentSession.from_dict(payload)
    assert restored.reminder_policy is None
    assert restored.to_dict() == payload
    assert restored.revision == 271 and not restored.can_undo
    warp = restored.reminder_state(direction="warp")
    weft = restored.reminder_state(direction="weft")
    assert warp["legacy"] and warp["total"] == 13 and warp["total_target"] == 13
    assert warp["measured_targets"] == {"wool": 7, "cashmere": 7}
    assert weft["total"] == 1 and not weft["total_reached"]
    tracker = ReminderTracker(restored)
    assert tracker.update(restored) is None
    assert restored.to_dict() == payload
    restored.configure(measured_reminder=9)
    assert restored.reminder_policy is None and restored.measured_reminder == 9
    restored.undo()
    assert restored.measured_reminder == 7


def test_explicit_legacy_migration_is_reversible_and_seeds_new_diameter_scope():
    s = session("diameter")
    add(s, 3)
    s.configure(reminder_policy=None)
    tracker = ReminderTracker(s)
    revision = s.revision
    s.configure(reminder_policy=ReminderPolicy(diameter_measured_target=2))
    assert s.revision == revision + 1
    assert tracker.update(s) is None
    assert s.reminder_state()["total_scope"] == "operator_measured"
    s.undo()
    assert s.reminder_policy is None and "reminder_policy" not in s.to_dict()
    assert tracker.update(s) is None


def test_invalid_measurements_and_rejected_batch_never_contribute():
    s = session(policy=ReminderPolicy(content_measured_target=1, content_total_target=1))
    tracker = ReminderTracker(s)
    original = s.to_dict()
    with pytest.raises(ValueError):
        s.add_measurement(0, "wool")
    with pytest.raises(ValueError):
        s.add_measurements_batch([{"diameter_um": 15, "material_id": "wool"},
                                  {"diameter_um": -1, "material_id": "wool"}])
    assert s.to_dict() == original and tracker.update(s) is None


@pytest.mark.parametrize("policy", [
    {"version": 2}, {"version": True}, {"version": 1.0}, {},
    {"version": 1, "content_total_target": 0},
    {"version": 1, "diameter_measured_target": 2.5},
    {"version": 1, "content_measured_target": True},
    {"version": 1, "material_targets": {"wool": -2}},
    {"version": 1, "material_targets": {"": 5}},
    {"version": 1, "material_targets": [("wool", 1), ("wool", 2)]},
    {"version": 1, "unknown": 5}, "invalid",
])
def test_invalid_policy_is_atomic_and_not_silently_interpreted(policy):
    s = session()
    before = s.to_dict()
    with pytest.raises(ValueError):
        s.configure(reminder_policy=policy)
    assert s.to_dict() == before
    before["reminder_policy"] = policy
    with pytest.raises(ValueError):
        ExperimentSession.from_dict(before)


def test_empty_session_and_new_operator_are_safe_and_notification_scope_follows_id():
    s = ExperimentSession(reminder_policy=ReminderPolicy(content_measured_target=1))
    tracker = ReminderTracker(s)
    assert tracker.update(s) is None
    s.add_operator("甲", id="new")
    s.set_materials(default_materials())
    s.add_measurement(15, "wool")
    notice = tracker.update(s)
    assert notice.crossings[0].operator_id == "new"


def test_unrelated_evidence_and_field_updates_do_not_replay_history_crossings():
    s = session(policy=ReminderPolicy(content_measured_target=2))
    records = add(s, 2)
    s.delete_measurement(records[0].id)
    tracker = ReminderTracker(s)
    s.undo()
    s.set_evidence(records[0].id, {"state": "disabled"})
    s.update_field("field", status="ended")
    assert tracker.update(s) is None
