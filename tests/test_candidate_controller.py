"""Field-owned candidate acceptance plus standalone alignment utility regression.

The independent alignment utility is kept covered for offline consumers, but
its former live-preview deadline and recovery policies were removed in round 7.
"""
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.candidate_alignment import (CandidateAlignmentMonitor, CandidateReferenceCache,
                                                     compare_candidate_frames)
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameBudget, FrameHandle, PresentedFrameToken


class LocalAlignmentWorker:
    """Deterministic test double; real spawn/IPC is covered in worker tests."""
    def __init__(self):
        self.prepared = CandidateReferenceCache()
    def compare(self, reference, current):
        from fdm.live_experiment.candidate_alignment import compare_candidate_frames
        return compare_candidate_frames(reference, current, prepared=self.prepared)
    def invalidate(self):
        self.prepared = CandidateReferenceCache()
    def request_stop(self):
        pass
    def close(self):
        self.invalidate()


def texture():
    rng = np.random.default_rng(192)
    return cv2.GaussianBlur(rng.integers(0, 256, (240, 320), dtype=np.uint8), (5, 5), 0)


def frame(sequence=1, pixels=None, live=False):
    return FrameHandle.snapshot(texture() if pixels is None else pixels, source_id="camera",
                                sequence=sequence, live=live)


def present(controller, value):
    controller.receive_frame(value)
    controller.mark_presented(PresentedFrameToken(value, controller.field_epoch, 0, 1., (0., 0.)))


@pytest.fixture
def controller(tmp_path):
    session = ExperimentSession(save_images=False)
    session.add_operator("A")
    session.set_materials(default_materials())
    result = LiveExperimentController(session, evidence_directory=tmp_path)
    result._timer.stop()
    result.set_calibration(CalibrationSnapshot(1, 1))
    present(result, frame())
    result._automatic = SimpleNamespace(cancel=lambda: None, close=lambda: None, poll=lambda: [])
    yield result
    result.close()
    assert result.budget.used_bytes == 0


def batch(controller, count=3, guarded=False):
    token = controller.presented
    controller._automatic_context = {"task_id": "analysis", "token": token,
        "calibration": controller.calibration, "algorithm": "traditional",
        "operator_id": controller.session.active_operator_id, "direction": "warp",
        "lease": controller.budget.acquire(token.frame), "completed": True,
        "guarded": guarded, "started_at": time.perf_counter()}
    controller.candidates = [{"width_px": 20, "start": [10 + i * 10, 30], "end": [10 + i * 10, 50]}
                             for i in range(count)]
    controller._ensure_candidate_ids()
    return [c["candidate_id"] for c in controller.candidates]


def test_fixed_reference_rejects_slow_cumulative_drift_without_modifying_source():
    source = texture()
    original = source.copy()
    assert compare_candidate_frames(source, source).valid
    for pixels in (np.roll(source, 2, 1), np.roll(source, 15, 1), cv2.GaussianBlur(source, (21, 21), 5)):
        assert not compare_candidate_frames(source, pixels).valid
    assert np.array_equal(source, original)
    assert not compare_candidate_frames(np.full_like(source, 128), np.full_like(source, 128)).valid


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_cached_reference_matches_fresh_checks_and_is_not_windowed_repeatedly(channels):
    source = texture()
    if channels != 1:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR if channels == 3 else cv2.COLOR_GRAY2BGRA)
    prepared = CandidateReferenceCache()
    variants = [source, np.roll(source, 1, 1), np.roll(source, 2, 1),
                cv2.GaussianBlur(source, (21, 21), 5), source]
    original = source.copy()
    for current in variants * 3:
        cached = compare_candidate_frames(source, current, prepared=prepared)
        fresh = compare_candidate_frames(source, current)
        assert cached.valid == fresh.valid
        assert cached.reason == fresh.reason
        assert cached.shift == pytest.approx(fresh.shift, abs=1e-10)
        assert cached.response == pytest.approx(fresh.response, abs=1e-10)
        assert cached.correlation == fresh.correlation
        assert cached.gradient_ratio == fresh.gradient_ratio
    assert not prepared.normalized.flags.writeable
    assert np.array_equal(source, original)


def test_alignment_preparation_remains_budgeted_between_checks_and_is_released():
    budget = FrameBudget()
    monitor = CandidateAlignmentMonitor(budget, lambda *_: None, worker=LocalAlignmentWorker())
    source = frame()
    try:
        monitor.reset(source)
        monitor.poll(source)
        monitor.future[0].result(timeout=3)
        monitor.poll(source)
        assert monitor.worker.prepared.normalized is not None
        assert budget.used_bytes >= source.nbytes + source.size[0] * source.size[1] * 64
        monitor.reset()
        monitor.pool.submit(lambda: None).result(timeout=3)
        assert monitor.preparation is None
        assert budget.used_bytes == 0
    finally:
        monitor.close()


def test_partial_acceptance_keeps_ids_geometry_and_other_candidates(controller):
    ids = batch(controller)
    controller.submit_candidate(ids[1], "wool")
    assert [c["candidate_id"] for c in controller.candidates] == ids
    assert [c["state"] for c in controller.candidates] == ["available", "accepted", "available"]
    assert [c["ordinal"] for c in controller.candidates] == [1, 2, 3]
    record = next(iter(controller.session.measurements.values()))
    assert record.provenance["candidate_id"] == ids[1]
    assert record.frame_id == controller.presented.frame.frame_id
    with pytest.raises(ValueError):
        controller.submit_candidate(ids[1], "wool")
    assert len(controller.session.measurements) == 1
    controller.submit_candidate(ids[2], "cashmere")
    assert [r.material_id for r in controller.session.measurements.values()] == ["wool", "cashmere"]


def test_undo_redo_and_delete_sync_candidate_record_mapping(controller):
    ids = batch(controller)
    controller.submit_candidate(ids[0], "wool")
    record_id = controller.candidates[0]["record_id"]
    assert controller.session.undo()
    controller.sync_candidate_records()
    assert controller.candidates[0]["state"] == "available"
    assert controller.session.redo()
    controller.sync_candidate_records()
    assert controller.candidates[0]["state"] == "accepted"
    controller.session.delete_measurement(record_id)
    controller.sync_candidate_records()
    assert controller.candidates[0]["state"] == "available"


def test_excluding_undone_candidate_survives_unrelated_history_sync(controller):
    ids = batch(controller)
    controller.submit_candidate(ids[0], "wool")
    assert controller.session.undo()
    controller.sync_candidate_records()
    controller.set_candidate_excluded(ids[0], True)
    controller.sync_candidate_records()
    assert controller.candidates[0]["state"] == "excluded"
    with pytest.raises(ValueError):
        controller.submit_candidate(ids[0], "wool")
    assert controller.session.redo()
    controller.sync_candidate_records()
    assert controller.candidates[0]["state"] == "accepted"


def test_exclusion_only_changes_available_runtime_state(controller):
    ids = batch(controller)
    revision = controller.session.revision
    controller.set_candidate_excluded(ids[0], True)
    assert controller.candidates[0]["state"] == "excluded"
    with pytest.raises(ValueError):
        controller.submit_candidate(ids[0], "wool")
    controller.set_candidate_excluded(ids[0], False)
    assert controller.session.revision == revision
    controller.submit_candidate(ids[0], "wool")
    with pytest.raises(ValueError):
        controller.set_candidate_excluded(ids[0], True)
    controller.candidates[1]["state"] = "confirming"
    with pytest.raises(ValueError):
        controller.set_candidate_excluded(ids[1], True)
    with pytest.raises(ValueError):
        controller.set_candidate_excluded("not-current-batch", True)


def test_queue_snapshots_category_is_bounded_and_escape_does_not_remove_records(controller):
    ids = batch(controller, 20)
    controller.submit_candidate(ids[0], "wool")
    # Keep global frame verification outstanding while many clicks arrive.
    controller._motion_future = (controller.field_epoch, Future())
    present(controller, frame(2, live=True))
    controller.state = "ready"
    controller._motion_sequence = 0
    for candidate_id in ids[1:17]:
        controller.submit_candidate(candidate_id, "cashmere")
    assert controller.candidate_pending_count == 16
    with pytest.raises(ValueError, match="16"):
        controller.submit_candidate(ids[17], "wool")
    with pytest.raises(ValueError, match="核验"):
        controller.sync_candidate_records()
    assert len(controller.session.measurements) == 1
    controller.cancel_candidate_submissions()
    assert controller.candidate_pending_count == 0
    assert controller.candidates[0]["state"] == "accepted"
    assert all(c["state"] == "available" for c in controller.candidates[1:])


def test_queue_uses_click_category_even_if_selected_class_changes(controller):
    ids = batch(controller)
    controller._motion_future = (controller.field_epoch, Future())
    present(controller, frame(2, live=True))
    controller.state = "ready"
    controller._motion_sequence = 0
    controller.submit_candidate(ids[0], "wool")
    controller.submit_candidate(ids[1], "cashmere")
    controller._motion_sequence = 2
    controller._poll_confirmation()
    controller._advance_candidate_queue()
    assert [r.material_id for r in controller.session.measurements.values()] == ["wool", "cashmere"]


def test_queued_geometric_duplicate_is_rechecked_after_first_record_commits(controller):
    ids = batch(controller)
    controller.candidates[1].update(start=[10, 30], end=[10, 50])
    controller._motion_future = (controller.field_epoch, Future())
    present(controller, frame(2, live=True))
    controller.state = "ready"
    controller._motion_sequence = 0
    controller.submit_candidate(ids[0], "wool")
    controller.submit_candidate(ids[1], "wool")
    controller._motion_sequence = 2
    controller._poll_confirmation()
    controller._advance_candidate_queue()
    assert len(controller.session.measurements) == 1
    assert controller.candidates[1]["state"] == "available"


def test_out_of_range_click_is_not_accepted_or_marked_pending(controller):
    material = controller.session.material("wool")
    controller.session.set_materials([replace(material, upper_limit=15)])
    ids = batch(controller)
    with pytest.raises(ValueError, match="越限"):
        controller.submit_candidate(ids[0], "wool")
    assert not controller.session.measurements and controller.candidate_pending_count == 0
    assert controller.candidates[0]["state"] == "available"


def test_automatic_defaults_off_and_runs_only_once_after_stable_cycle(controller, monkeypatch):
    assert not controller.automatic_enabled
    controller._motion_future = (controller.field_epoch, Future())
    present(controller, frame(2, live=True))
    controller.state = "ready"
    calls = []
    monkeypatch.setattr(controller, "analyze", lambda algorithm, **kw: calls.append((algorithm, kw)))
    controller.configure_automatic(active=True, prepared=True)
    controller._schedule_automatic()
    assert not calls
    controller.configure_automatic(enabled=True, algorithm="multiscale")
    controller._update_candidate_state()
    controller._schedule_automatic()
    assert not calls
    controller._automatic_stable_since -= .31
    controller._schedule_automatic()
    controller._schedule_automatic()
    assert calls == [("multiscale", {"_origin": "automatic"})]
    # Pixel changes alone cannot rearm the business field's cycle.
    present(controller, frame(3, np.roll(texture(), 4, 1), live=True))
    controller._update_candidate_state()
    controller._schedule_automatic()
    assert len(calls) == 1
    controller.end_field("explicit next field")
    present(controller, frame(4, live=True))
    controller.state = "ready"
    controller._update_candidate_state()
    controller._automatic_stable_since -= .31
    controller._schedule_automatic()
    assert len(calls) == 2


def test_automatic_does_not_run_for_static_source_even_if_ui_is_active(controller, monkeypatch):
    calls = []
    monkeypatch.setattr(controller, "analyze", lambda *args, **kw: calls.append(args))
    controller.configure_automatic(enabled=True, active=True, prepared=True)
    controller._update_candidate_state()
    controller._automatic_stable_since -= 1
    controller._schedule_automatic()
    assert not calls


def test_empty_batch_is_a_completed_cycle_not_repeated_analysis(controller):
    controller.configure_automatic(enabled=True, active=True, prepared=True)
    batch(controller, count=0, guarded=True)
    controller._automatic_cycle_done = True
    controller._automatic_stable_since = time.perf_counter() - 1
    controller._schedule_automatic()
    assert controller._automatic_context["task_id"] == "analysis"
    assert not controller.busy


def test_disabling_auto_cancels_running_auto_only_and_keeps_completed_candidates(controller):
    batch(controller)
    controller._automatic_context["origin"] = "automatic"
    controller.configure_automatic(enabled=False)
    assert controller.candidates
    controller._automatic_context["completed"] = False
    controller.configure_automatic(enabled=False)
    assert not controller.candidates and controller._automatic_context is None


def test_refresh_waits_for_clicked_candidates_then_refreshes(controller, monkeypatch):
    ids = batch(controller)
    controller._motion_future = (controller.field_epoch, Future())
    present(controller, frame(2, live=True))
    controller.state = "ready"
    controller._motion_sequence = 0
    controller.submit_candidate(ids[0], "wool")
    controller.analyze("multiscale")
    assert controller._refresh_pending == "multiscale"
    calls = []
    monkeypatch.setattr(controller, "analyze", lambda algorithm: calls.append(algorithm))
    controller._schedule_automatic()
    assert not calls
    controller._motion_sequence = 2
    controller._poll_confirmation()
    controller._schedule_automatic()
    assert calls == ["multiscale"] and len(controller.session.measurements) == 1


def test_monitor_discards_obsolete_work_after_reference_reset_and_releases_budget():
    budget = FrameBudget()
    monitor = CandidateAlignmentMonitor(budget, lambda *_: None, worker=LocalAlignmentWorker())
    first, second = frame(), frame(2, np.roll(texture(), 4, 1))
    try:
        monitor.reset(first)
        monitor.poll(first)
        old = monitor.future[0]
        monitor.reset(second)
        old.result(timeout=3)
        assert monitor.poll(second) is None
        assert monitor.result is None
        monitor.future[0].result(timeout=3)
        result = monitor.poll(second)
        assert result[1] == 2 and result[3].valid
    finally:
        monitor.close()
    assert budget.used_bytes == 0


def test_offline_frame_metrics_follow_actual_admission_and_first_presentation(controller):
    first = frame(2, live=True)
    present(controller, first)
    controller.mark_presented(controller.presented)  # Overlay-only repaint.
    present(controller, frame(3, live=True))
    values = controller.performance_diagnostics()
    assert values["ingest_fps"]["n"] == 1
    assert values["presented_fps"]["n"] == 1
    assert values["frame_age_ms"]["n"] == 2
    controller.unavailable("user stopped")
    assert controller.performance_diagnostics()["frame_age_ms"] == values["frame_age_ms"]
    assert controller.performance_frame_context()["stopped"]
    assert controller.performance_diagnostics_schema()["version"] == 1


def test_old_analysis_response_cannot_restore_replaced_batch(controller):
    responses, submissions = [], []
    def submit(pixels, **kwargs):
        submissions.append(kwargs)
        return str(len(submissions))
    def poll():
        items, responses[:] = list(responses), []
        return items
    controller._automatic = SimpleNamespace(submit=submit, poll=poll, cancel=lambda: None, close=lambda: None)
    controller.analyze()
    first_batch = controller._automatic_context["batch_id"]
    controller.analyze("multiscale")
    second_batch = controller._automatic_context["batch_id"]
    assert first_batch != second_batch
    result = {"status": "ok", "candidates": [{"width_px": 20, "start": [10, 30], "end": [10, 50]}]}
    responses.append({"task_id": "1", "result": result})
    controller._poll_automatic()
    assert controller.candidates == []
    responses.append({"task_id": "2", "result": result})
    controller._poll_automatic()
    assert len(controller.candidates) == 1
    assert controller.candidates[0]["candidate_id"].startswith(second_batch)
    assert controller.candidate_metrics["analysis_ms"]


def test_monitor_never_queues_intermediate_frames(monkeypatch):
    from threading import Event
    from fdm.live_experiment import candidate_alignment as module
    entered, release = Event(), Event()
    calls = []
    compare = module.compare_candidate_frames
    def delayed(a, b, **kwargs):
        calls.append(b)
        entered.set()
        assert release.wait(3)
        return compare(a, b, **kwargs)
    monkeypatch.setattr(module, "compare_candidate_frames", delayed)
    budget = FrameBudget()
    monitor = CandidateAlignmentMonitor(budget, lambda *_: None, worker=LocalAlignmentWorker())
    frames = [frame(i) for i in range(1, 20)]
    try:
        monitor.reset(frames[0])
        monitor.poll(frames[0])
        assert entered.wait(3)
        for item in frames[1:]:
            monitor.poll(item)
        assert len(calls) == 1
        first_future = monitor.future[0]
        release.set()
        first_future.result(timeout=3)
        monitor.poll(frames[-1])
        monitor.future[0].result(timeout=3)
        assert len(calls) == 2
        assert calls[-1] is frames[-1].pixels
    finally:
        release.set()
        monitor.close()
    assert budget.used_bytes == 0


def test_completed_alignment_owns_pixels_until_consumed_or_reset():
    monitor = CandidateAlignmentMonitor(FrameBudget(), lambda *_: None, worker=LocalAlignmentWorker())
    source, newer = frame(), frame(2)
    try:
        monitor.reset(source)
        monitor.poll(newer)
        monitor.future[0].result(timeout=3)
        monitor.poll(newer)
        assert monitor.completed_frame is newer
        assert monitor._completed_lease in monitor.budget._leases
        assert monitor.result_for(newer)[3].valid
        monitor.reset()
        monitor.pool.submit(lambda: None).result(timeout=3)
        assert monitor.completed_frame is None and monitor.budget.used_bytes == 0
    finally:
        monitor.close()


def test_retained_press_uses_original_frame_after_preview_advances(controller):
    token = controller.presented
    lease = controller.retain_pointer_frame(token)
    present(controller, frame(2))
    assert controller.pointer_token_is_current(token)
    controller.start((30, 30), token=token)
    controller.release_pointer_frame(lease)
    controller.release_pointer_frame(lease)  # Cancellation and release may race.
    controller.finish((30, 50), "wool", method="free")
    controller._jobs[0][1].result(timeout=3)
    controller.poll()
    assert next(iter(controller.session.measurements.values())).frame_id == token.frame.frame_id
    assert not controller._pointer_frames


@pytest.mark.parametrize("change", ["field", "source", "calibration", "operator", "direction", "display"])
def test_press_context_cannot_cross_measurement_or_display_changes(controller, change):
    token = controller.presented
    lease = controller.retain_pointer_frame(token)
    if change == "field":
        controller.end_field()
    elif change == "source":
        controller.receive_frame(FrameHandle.snapshot(texture(), source_id="other", live=False))
    elif change == "calibration":
        controller.set_calibration(CalibrationSnapshot(2, 2))
    elif change == "operator":
        controller.session.add_operator("B")
    elif change == "direction":
        controller.session.active_direction = "weft"
    else:
        controller.mark_presented(replace(token, display_revision=token.display_revision + 1))
    assert not controller.pointer_token_is_current(token)
    with pytest.raises(ValueError):
        controller.start((30, 30), token=token)
    controller.release_pointer_frame(lease)
    assert controller.draft is None and not controller.session.measurements


def test_late_manual_result_cannot_reclaim_postselection_after_candidate_action(controller, monkeypatch):
    future = Future()
    monkeypatch.setattr(controller._manual_pool, "submit", lambda *_: future)
    controller.start((30, 30))
    controller.finish((30, 50), "wool", method="free")
    intent = controller._jobs[0][0]
    controller.candidate_interaction_started()
    assert controller.reclassify_latest("cashmere") is None
    assert intent.material_id == "wool"
    future.set_result({"status": "ok", "width_px": 20, "diameter_um": 20,
                       "start": (30, 30), "end": (30, 50)})
    controller.poll()
    record = next(iter(controller.session.measurements.values()))
    assert controller.reclassify_latest("cashmere") is None
    assert record.material_id == "wool"
    assert controller._classification_target["state"] == "candidate"


def test_candidate_review_cancels_only_unsubmitted_draft_not_queued_manual(controller, monkeypatch):
    future = Future()
    monkeypatch.setattr(controller._manual_pool, "submit", lambda *_: future)
    controller.start((30, 30))
    controller.finish((30, 50), "wool", method="free")
    intent = controller._jobs[0][0]
    controller.start((60, 30))
    assert controller.draft is not None and controller._draft_lease is not None
    controller.candidate_interaction_started()
    assert controller.draft is None and controller._draft_lease is None
    assert len(controller._jobs) == 1 and not intent.cancel.is_set()
    future.set_result({"status": "ok", "width_px": 20, "diameter_um": 20,
                       "start": (30, 30), "end": (30, 50)})
    controller.poll()
    assert len(controller.session.measurements) == 1
    assert controller._classification_target["state"] == "candidate"


def test_offline_visibility_diagnostics_are_actual_bounded_paint_observations(controller, monkeypatch):
    from fdm.live_experiment import controller as module
    batch(controller, guarded=True)
    current = controller.presented.frame
    clock = [100.]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    controller.record_candidate_visibility(current, False)
    clock[0] = 100.1
    controller.record_candidate_visibility(current, True)
    clock[0] = 100.5
    controller.record_candidate_visibility(current, False)
    clock[0] = 100.8
    result = controller.performance_frame_context()["candidate_preview"]["painted_visibility"]
    assert result["transition_count"] == 2
    assert result["current_hidden_ms"] == pytest.approx(300)
    assert result["max_visible_ms"] == pytest.approx(400)
    assert result["max_hidden_ms"] == pytest.approx(300)
    assert result["current_visible_ms"] is None
    for index in range(300):
        controller.record_candidate_visibility(current, bool(index % 2))
    assert len(controller.performance_frame_context()["candidate_preview"]["painted_visibility"]["events"]) == 128
    controller.record_candidate_visibility(current, False, display_enabled=False)
    clock[0] = 200.
    result = controller.performance_frame_context()["candidate_preview"]["painted_visibility"]
    assert result["current_hidden_ms"] is None  # User hiding is not a verification outage.
    assert result["max_hidden_ms"] == pytest.approx(300)
    controller.record_candidate_visibility(current, True)
    clock[0] = 200.4
    controller._suspend_candidate_visibility()
    clock[0] = 201.4
    result = controller.performance_frame_context()["candidate_preview"]["painted_visibility"]
    assert result["visible"] is None and not result["observation_active"]
    assert result["current_visible_ms"] is None
    assert result["max_visible_ms"] == pytest.approx(400)
    controller.record_candidate_visibility(current, True)
    clock[0] = 201.5
    assert controller.performance_frame_context()["candidate_preview"]["painted_visibility"]["current_visible_ms"] == pytest.approx(100)


def test_manual_finish_rejects_operator_change_even_when_field_not_explicitly_ended(controller):
    controller.start((30, 30))
    controller.session.add_operator("B")
    with pytest.raises(ValueError, match="视场已变化"):
        controller.finish((30, 50), "wool", method="free")
    assert not controller._jobs and not controller.session.measurements


def test_configuration_notification_does_not_recurse_when_observer_refreshes_readiness(controller):
    notifications = []
    def observe():
        notifications.append(True)
        controller.configure_automatic(active=True, prepared=True)
    controller.candidateStateChanged.connect(observe)
    controller.configure_automatic(active=True, prepared=True)
    assert len(notifications) == 1


def test_paint_metric_is_once_per_batch_and_diagnostics_are_bounded_scalars(controller):
    batch(controller, guarded=True)
    controller.record_candidate_painted(controller.presented.frame)
    controller.record_candidate_painted(controller.presented.frame)
    metric = controller.performance_diagnostics()["analysis_to_paint_ms"]
    assert metric["n"] == 1 and metric["p50"] >= 0 and metric["p50"] == metric["p95"]


def test_hide_show_preserves_completed_cycle_and_batch(controller):
    controller.configure_automatic(enabled=True, active=True, prepared=True)
    ids = batch(controller, guarded=True)
    context = controller._automatic_context
    controller._automatic_cycle_done = True
    controller.configure_automatic(active=False)
    assert controller._automatic_cycle_done
    assert controller._automatic_context is context
    assert [c["candidate_id"] for c in controller.candidates] == ids
    assert not controller.candidate_overlay_ready
    controller.configure_automatic(active=True)
    assert controller._automatic_cycle_done
    assert controller._automatic_context is context


def test_pending_cancel_does_not_rearm_automatic_cycle(controller):
    controller._automatic_cycle_done = True
    controller.cancel_pending_measurements()
    assert controller._automatic_cycle_done


def test_acceptance_emits_one_notification_per_state_change(controller):
    ids = batch(controller)
    states = []
    controller.candidatesChanged.connect(lambda values: states.append(tuple(c["state"] for c in values)))
    controller.submit_candidate(ids[0], "wool")
    assert states == [("confirming", "available", "available"), ("accepted", "available", "available")]


def test_live_controller_never_constructs_independent_alignment_worker(tmp_path, monkeypatch):
    from fdm.live_experiment.candidate_alignment import CandidateAlignmentMonitor
    def forbidden(*args, **kwargs):
        raise AssertionError("Live controller must not create independent alignment")
    monkeypatch.setattr(CandidateAlignmentMonitor, "__init__", forbidden)
    session = ExperimentSession(save_images=False)
    session.add_operator("A")
    value = LiveExperimentController(session, evidence_directory=tmp_path)
    try:
        assert not hasattr(value, "_alignment")
        present(value, frame())
        value.configure_automatic(enabled=True, active=True, prepared=True)
        value._update_candidate_state()
    finally:
        value.close()
    assert value.budget.used_bytes == 0


def test_same_field_candidate_survives_every_new_preview_without_pixel_comparison(controller):
    ids = batch(controller, guarded=True)
    context = controller._automatic_context
    controller._automatic_cycle_done = True
    controller.candidates[1]["material_id"] = "cashmere"
    for sequence, pixels in enumerate((texture(), np.roll(texture(), 4, 1),
            cv2.GaussianBlur(texture(), (21, 21), 5), texture()), 2):
        # This fixture controls the business field, so the test proves that
        # candidate validity no longer has a second independent pixel gate.
        present(controller, frame(sequence, pixels))
        controller._update_candidate_state()
        assert controller.candidate_overlay_ready
        assert controller._automatic_context is context
        assert [candidate["candidate_id"] for candidate in controller.candidates] == ids
        assert controller.candidates[1]["material_id"] == "cashmere"
        assert controller._automatic_cycle_done
    controller.submit_candidate(ids[0], "wool")
    assert len(controller.session.measurements) == 1
    assert controller.candidate_overlay_ready


def test_same_context_frame_is_clickable_before_next_presentation_token(controller):
    batch(controller, guarded=True)
    upcoming = frame(2)
    assert controller.candidates_aligned_for(upcoming)
    assert not controller.candidates_aligned_for(replace(upcoming, source_id="different"))
    assert not controller.candidates_aligned_for(replace(upcoming, generation=1))
    assert not controller.candidates_aligned_for(replace(upcoming, pixels=np.zeros((10, 10), np.uint8)))


def test_global_field_change_cancels_pending_candidate_and_starts_new_cycle(controller):
    from fdm.live_experiment.selection_retention import FieldEndCause
    ids = batch(controller)
    controller._automatic_cycle_done = True
    controller._motion_future = (controller.field_epoch, Future())
    present(controller, frame(2, live=True))
    controller.state = "ready"
    controller._motion_sequence = 0
    controller.submit_candidate(ids[0], "wool")
    assert controller.candidate_pending_count == 1
    before = controller.field_epoch
    controller.end_field("motion", cause=FieldEndCause.MOTION)
    assert controller.field_epoch == before + 1
    assert not controller.candidates and controller._automatic_context is None
    assert not controller.candidate_pending_count and not controller.session.measurements
    assert not controller._automatic_cycle_done


def test_pending_candidate_waits_only_for_global_motion_without_hiding_remainder(controller):
    ids = batch(controller)
    controller._motion_future = (controller.field_epoch, Future())
    present(controller, frame(2, live=True))
    controller.state = "ready"
    controller._motion_sequence = 0
    controller.submit_candidate(ids[0], "wool")
    assert controller.candidate_pending_count == 1
    assert controller.candidate_overlay_ready
    assert controller.candidates[0]["state"] == "confirming"
    assert all(item["state"] == "available" for item in controller.candidates[1:])
    controller._motion_sequence = 2
    controller._poll_confirmation()
    assert len(controller.session.measurements) == 1
    assert controller.candidate_overlay_ready


def test_preview_publishes_immediately_with_field_candidate_context(controller):
    batch(controller, guarded=True)
    delivered = []
    controller.frameReady.connect(delivered.append)
    for sequence in range(2, 8):
        current = frame(sequence)
        controller.receive_frame(current)
        controller.present_preview_frame(current)
        assert delivered[-1] is current
        assert controller.candidates_aligned_for(current)
        assert not controller.should_defer_canvas_business_update()


def test_analysis_start_failure_releases_its_frame_without_changing_batch(controller):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate analyzer launch failure")
    controller._automatic = SimpleNamespace(submit=fail, cancel=lambda: None, close=lambda: None, poll=lambda: [])
    before = controller.budget.used_bytes
    with pytest.raises(RuntimeError, match="deliberate"):
        controller.analyze()
    assert controller._automatic_context is None
    assert controller.budget.used_bytes == before
