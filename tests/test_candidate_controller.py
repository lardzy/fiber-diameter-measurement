"""Candidate acceptance, fixed-reference safety and once-per-cycle scheduling."""
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
    result._alignment.worker = LocalAlignmentWorker()
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
    if guarded:
        controller._alignment.reset(token.frame)
        wait_alignment(controller)
    return [c["candidate_id"] for c in controller.candidates]


def wait_alignment(controller):
    end = time.perf_counter() + 3
    while time.perf_counter() < end:
        controller._poll_candidate_alignment()
        result = controller._alignment.result
        if result and result[1] == controller.latest_frame.sequence:
            return result
        time.sleep(.002)
    pytest.fail("alignment did not complete")


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


def test_alignment_loss_hides_candidates_preserves_accepted_field_and_count(controller):
    ids = batch(controller, guarded=True)
    assert controller.candidate_overlay_ready
    controller.submit_candidate(ids[0], "wool")
    field_id, records = controller.field_id, list(controller.current_records)
    present(controller, frame(2, np.roll(texture(), 3, 1)))
    assert not controller.candidate_overlay_ready
    deadline = time.perf_counter() + 3
    while controller._automatic_context is not None and time.perf_counter() < deadline:
        controller._poll_candidate_alignment()
        time.sleep(.002)
    assert not controller.candidates and not controller.candidate_overlay_ready
    assert controller.field_id == field_id and controller.current_records == records
    assert len(controller.session.measurements) == 1


def test_new_display_frame_blocks_click_until_independently_aligned(controller):
    ids = batch(controller, guarded=True)
    present(controller, frame(2))
    assert not controller.candidate_overlay_ready
    with pytest.raises(ValueError, match="对齐"):
        controller.submit_candidate(ids[0], "wool")
    wait_alignment(controller)
    assert controller.candidate_overlay_ready
    controller.submit_candidate(ids[0], "wool")
    assert len(controller.session.measurements) == 1


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
    wait_alignment(controller)
    controller._schedule_automatic()
    assert not calls
    controller._automatic_stable_since -= .31
    controller._schedule_automatic()
    controller._schedule_automatic()
    assert calls == [("multiscale", {"_origin": "automatic"})]
    # Fixed-source drift rearms a candidate cycle without touching business data.
    present(controller, frame(3, np.roll(texture(), 4, 1), live=True))
    wait_alignment(controller)
    controller._automatic_stable_since -= .31
    controller._schedule_automatic()
    assert len(calls) == 2


def test_automatic_does_not_run_for_static_source_even_if_ui_is_active(controller, monkeypatch):
    calls = []
    monkeypatch.setattr(controller, "analyze", lambda *args, **kw: calls.append(args))
    controller.configure_automatic(enabled=True, active=True, prepared=True)
    wait_alignment(controller)
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


def test_paint_gate_covers_next_canvas_frame_before_presented_token_changes(controller):
    batch(controller, guarded=True)
    assert controller.candidate_overlay_ready
    next_frame = frame(2)
    # Canvas still has the old presented token while drawing its next frame.
    assert not controller.candidates_aligned_for(next_frame)
    assert not controller.candidates_aligned_for(replace(next_frame, source_id="different"))


def test_newer_alignment_does_not_certify_skipped_intermediate_frame(controller):
    batch(controller, guarded=True)
    first = controller.presented.frame
    skipped = frame(2, np.roll(texture(), 8, 1))
    present(controller, frame(3))
    wait_alignment(controller)
    assert controller.candidates_aligned_for(first)
    assert controller.candidate_overlay_ready
    assert not controller.candidates_aligned_for(skipped)
    # The same sequence from a reopened source is a different frame identity.
    assert not controller.candidates_aligned_for(replace(controller.presented.frame, generation=1))
    for sequence in range(4, 13):
        present(controller, frame(sequence))
        wait_alignment(controller)
    assert len(controller._alignment._results) == 8
    assert not controller.candidates_aligned_for(first)


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


def test_alignment_admission_failure_does_not_start_analysis_or_leak_frames(controller):
    source_bytes = controller.budget.used_bytes
    controller.budget.limit_bytes = source_bytes - 1
    from fdm.live_experiment.frames import FrameBudgetExceeded
    with pytest.raises(FrameBudgetExceeded):
        controller.analyze()
    assert controller._automatic_context is None
    assert controller.budget.used_bytes == source_bytes


def test_low_texture_does_not_busy_loop_same_static_frame(controller):
    present(controller, frame(2, np.full((240, 320), 128, np.uint8)))
    controller.configure_automatic(enabled=True, active=True, prepared=True)
    controller._poll_candidate_alignment()
    controller._alignment.future[0].result(timeout=3)
    controller._poll_candidate_alignment()
    generation = controller._alignment.generation
    for _ in range(10):
        controller._poll_candidate_alignment()
    assert controller._alignment.future is None
    assert controller._alignment.generation == generation
    assert controller._automatic_stable_since is None


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


def test_preview_waits_for_alignment_without_hiding_verified_overlay(controller):
    batch(controller, guarded=True)
    delivered = []
    def paint(value):
        delivered.append(value)
        controller.mark_presented(PresentedFrameToken(value, controller.field_epoch, 0, 1., (0., 0.)))
    controller.frameReady.connect(paint)
    next_frame = frame(2)
    controller.receive_frame(next_frame)
    controller.present_preview_frame(next_frame)
    assert not delivered
    assert controller.candidate_overlay_ready  # Existing displayed frame remains verified.
    assert controller._candidate_preview_lease is not None
    wait_alignment(controller)
    assert delivered == [next_frame]
    assert controller.candidate_overlay_ready
    assert controller._candidate_preview_lease is None


def test_slow_alignment_falls_back_to_live_pixels_without_clickable_overlay(controller, monkeypatch):
    ids = batch(controller, guarded=True)
    delivered = []
    def paint(value):
        delivered.append(value)
        controller.mark_presented(PresentedFrameToken(value, controller.field_epoch, 0, 1., (0., 0.)))
    controller.frameReady.connect(paint)
    real_poll = controller._alignment.poll
    monkeypatch.setattr(controller._alignment, "poll", lambda frame: None)
    next_frame = frame(2)
    controller.receive_frame(next_frame)
    controller.present_preview_frame(next_frame)
    assert not delivered
    controller._candidate_preview_deadline = time.perf_counter() - .01
    controller._flush_candidate_preview()
    assert delivered == [next_frame]
    assert not controller.candidate_overlay_ready
    assert controller._candidate_preview_lease is None
    assert controller._candidate_preview_fallbacks == 1
    # One late check must not make every following frame wait again. Capture
    # keeps advancing while an unavailable worker cannot certify any new frame.
    for sequence in range(3, 14):
        latest = frame(sequence)
        controller.receive_frame(latest)
        controller.present_preview_frame(latest)
        assert delivered[-1] is latest
        assert controller._candidate_preview_pending is None
        assert not controller.candidates_aligned_for(latest)
        with pytest.raises(ValueError, match="正在核验候选"):
            controller.submit_candidate(ids[0], "wool")
    assert len(controller.session.measurements) == 0
    # One timely completion must not flash the overlay back on. Recovery needs
    # a sustained run, including on static images without another capture tick.
    monkeypatch.setattr(controller._alignment, "poll", real_poll)
    wait_alignment(controller)
    assert not controller.candidate_overlay_ready
    limit = time.perf_counter() + 3
    while controller._candidate_preview_backoff_until and time.perf_counter() < limit:
        controller._poll_candidate_alignment()
        time.sleep(.01)
    assert controller.candidate_overlay_ready
    controller.submit_candidate(ids[0], "wool")
    assert len(controller.session.measurements) == 1


def test_preview_backoff_recovers_only_after_time_and_consecutive_timely_checks(controller, monkeypatch):
    from fdm.live_experiment.candidate_alignment import AlignmentResult
    from fdm.live_experiment import controller as module
    batch(controller, guarded=True)
    current = controller.latest_frame
    identity = (current.source_id, current.generation, current.size)
    clock = [100.]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    controller._back_off_candidate_preview(clock[0])
    timely = (identity, current.sequence, current.frame_id, AlignmentResult(True, "aligned", elapsed_ms=1.))
    controller._alignment.completed_submitted_at = clock[0] - .001
    controller._alignment.completed_ready_at = clock[0]
    for _ in range(8):
        controller._note_candidate_preview_alignment(timely, current)
    assert controller._candidate_preview_backoff_until
    # Eight checks do not satisfy the 300 ms recovery interval. A slow check
    # resets the healthy run, but cannot make individual frames visible.
    clock[0] = 100.2
    controller._alignment.completed_submitted_at = clock[0] - .08
    controller._alignment.completed_ready_at = clock[0]
    controller._note_candidate_preview_alignment(timely, current)
    assert controller._candidate_preview_timely_checks == 0
    assert not controller.candidate_overlay_ready
    for i in range(8):
        clock[0] = 100.3 + i * .04
        controller._alignment.completed_submitted_at = clock[0] - .001
        controller._alignment.completed_ready_at = clock[0]
        controller._note_candidate_preview_alignment(timely, current)
    assert controller._candidate_preview_backoff_until
    clock[0] = 100.63
    controller._alignment.completed_submitted_at = clock[0] - .001
    controller._alignment.completed_ready_at = clock[0]
    controller._note_candidate_preview_alignment(timely, current)
    assert controller._candidate_preview_backoff_until == 0.
    assert controller.performance_frame_context()["candidate_preview"]["fallback_count"] == 1


def test_valid_completion_for_previous_input_publishes_its_own_exact_frame(controller):
    batch(controller, guarded=True)
    delivered = []
    controller.frameReady.connect(delivered.append)
    checked, newest = frame(2), frame(3)
    controller.receive_frame(checked)
    controller._alignment.poll(checked)
    controller._alignment.future[0].result(timeout=3)
    # Qt receives a newer capture before it consumes the valid result. This
    # must not be counted as failure, nor certify the unchecked newer image.
    controller.receive_frame(newest)
    controller._poll_candidate_alignment()
    assert delivered[-1] is checked
    assert controller._candidate_preview_backoff_until == 0
    assert controller.candidates_aligned_for(checked)
    assert not controller.candidates_aligned_for(newest)
    assert controller._automatic_context is not None and len(controller.candidates) == 3


def test_arrival_triggered_recovery_never_overwrites_checked_pixels_with_unchecked_input(controller, monkeypatch):
    from fdm.live_experiment.candidate_alignment import AlignmentResult
    batch(controller, guarded=True)
    checked, newest = frame(2), frame(3)
    identity = (checked.source_id, checked.generation, checked.size)
    controller._alignment._results.append((identity, checked.sequence, checked.frame_id,
                                          AlignmentResult(True, "aligned", elapsed_ms=1.)))
    controller.receive_frame(newest)
    controller._back_off_candidate_preview(time.perf_counter())
    delivered = []
    controller.frameReady.connect(delivered.append)
    def recover():
        if controller._candidate_preview_backoff_until:
            controller._candidate_preview_backoff_until = 0.
            controller._publish_verified_candidate_frame(checked)
    monkeypatch.setattr(controller, "_poll_candidate_alignment", recover)
    controller.present_preview_frame(newest)
    assert delivered == [checked]
    assert controller._candidate_preview_pending is newest
    assert controller.candidates_aligned_for(checked)
    assert not controller.candidates_aligned_for(newest)


def test_paused_gate_cannot_flash_on_individual_verified_frames(controller):
    ids = batch(controller, guarded=True)
    controller._back_off_candidate_preview(time.perf_counter())
    assert controller._candidate_frame_aligned(controller.presented.frame)
    for _ in range(20):
        assert not controller.candidates_aligned_for(controller.presented.frame)
        assert not controller.candidate_overlay_ready
        with pytest.raises(ValueError, match="核验候选"):
            controller.submit_candidate(ids[0], "wool")
    assert controller.session.measurements == {}


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


def test_result_readiness_gui_delay_and_publishable_age_are_separate(controller, monkeypatch):
    from fdm.live_experiment.candidate_alignment import AlignmentResult
    from fdm.live_experiment import controller as module
    batch(controller, guarded=True)
    checked = replace(controller.latest_frame, live=True, received_at=100.)
    identity = (checked.source_id, checked.generation, checked.size)
    completed = (identity, checked.sequence, checked.frame_id, AlignmentResult(True, "aligned", elapsed_ms=19.))
    controller._alignment.completed_frame = checked
    controller._alignment.completed_submitted_at = 100.
    controller._alignment.completed_ready_at = 100.020
    controller._preview_interval = 1 / 30
    monkeypatch.setattr(module.time, "perf_counter", lambda: 100.031)
    controller._note_candidate_preview_alignment(completed, checked)
    assert controller._candidate_preview_backoff_until == 0
    assert controller._candidate_preview_wait_budget() == pytest.approx(.85 / 30)
    metrics = controller.performance_diagnostics()
    assert metrics["candidate_verification_ready_ms"]["p50"] >= 0
    assert controller._performance.samples["candidate_verification_ready_ms"][-1] == pytest.approx(20.)
    assert controller._performance.samples["candidate_verification_gui_wait_ms"][-1] == pytest.approx(11.)
    # A finished check never grants unbounded reuse of its source pixels.
    monkeypatch.setattr(module.time, "perf_counter", lambda: 100.034)
    controller._note_candidate_preview_alignment(completed, checked)
    assert controller._candidate_preview_backoff_until
    assert controller._candidate_presentation_events[-1]["reason"] == "presentation_stale"


def test_deadline_first_consumes_already_finished_exact_frame(controller, monkeypatch):
    batch(controller, guarded=True)
    delivered = []
    controller.frameReady.connect(delivered.append)
    current = frame(2)
    controller.receive_frame(current)
    real_poll = controller._poll_candidate_alignment
    monkeypatch.setattr(controller, "_poll_candidate_alignment", lambda: None)
    controller.present_preview_frame(current)
    controller._alignment.poll(current)
    controller._alignment.future[0].result(3)
    controller._candidate_preview_deadline = time.perf_counter() - .01
    monkeypatch.setattr(controller, "_poll_candidate_alignment", real_poll)
    controller._flush_candidate_preview()
    assert delivered == [current]
    assert controller._candidate_preview_pending is None
    assert controller._candidate_preview_backoff_until == 0
    assert controller._candidate_frame_aligned(current)


def test_business_redraw_is_deferred_only_for_guarded_live_pending_publication(controller):
    batch(controller, guarded=True)
    assert not controller.should_defer_canvas_business_update()
    controller._candidate_preview_pending = replace(frame(2), live=True)
    controller._candidate_preview_deadline = time.perf_counter() + .02
    assert controller.should_defer_canvas_business_update()
    controller.draft = ((1, 1),)
    assert not controller.should_defer_canvas_business_update()
    controller.draft = None
    controller._candidate_preview_backoff_until = time.perf_counter()
    assert not controller.should_defer_canvas_business_update()
    controller._candidate_preview_backoff_until = 0
    controller._candidate_preview_pending = frame(2)
    assert not controller.should_defer_canvas_business_update()
