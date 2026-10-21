"""Optical task boundaries: snapshot, raw evidence, field lifetime and acceptance."""
from copy import deepcopy
from types import SimpleNamespace
import time

import numpy as np
import pytest

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameHandle, PresentedFrameToken
from fdm.services.cu_automatic_worker import CuAutomaticWorker


def image():
    pixels = np.full((240, 320, 3), 220, np.uint8)
    pixels[104:136, 35:285] = 190
    return pixels


@pytest.fixture
def optical_controller(tmp_path):
    session = ExperimentSession(save_images=False)
    session.add_operator("Optical operator")
    session.set_materials(default_materials())
    controller = LiveExperimentController(session, evidence_directory=tmp_path)
    controller._timer.stop()
    controller.set_calibration(CalibrationSnapshot(.5, .5))
    frame = FrameHandle.snapshot(image(), source_id="optical-test", live=False)
    controller.receive_frame(frame)
    controller.mark_presented(PresentedFrameToken(frame, controller.field_epoch, 0, 1., (0., 0.)))
    controller._automatic = SimpleNamespace(cancel=lambda: None, close=lambda: None, poll=lambda: [])
    yield controller
    controller.close()
    assert controller.budget.used_bytes == 0


def install_batch(controller):
    frame = controller.presented.frame
    controller._automatic_context = {
        "task_id": "optical-task", "token": controller.presented,
        "calibration": controller.calibration, "algorithm": "optical_adaptive",
        "operator_id": controller.session.active_operator_id, "direction": "warp",
        "lease": controller.budget.acquire(frame), "completed": True,
        "batch_id": "optical-batch", "started_at": time.perf_counter(),
        "analysis_config": {"algorithm": "optical_adaptive", "imaging_mode": "mixed"},
        "diagnostics": {"revision": 1, "illumination": {"background": 220}},
    }
    controller.candidates = [
        {"width_px": 32., "start": [80., 103.5], "end": [80., 135.5],
         "hypothesis_id": "h1", "conflicts_with": ["h2"], "warning_codes": ["low_contrast"],
         "quality_features": {"valid_samples": 5}, "structure": "solid"},
        {"width_px": 40., "start": [140., 100.], "end": [140., 140.],
         "hypothesis_id": "h2", "conflicts_with": ["h1"]},
        {"width_px": 32., "start": [200., 103.5], "end": [200., 135.5],
         "hypothesis_id": "h3", "conflicts_with": []},
    ]
    controller._ensure_candidate_ids()


def test_preferences_do_not_invalidate_existing_candidates_draft_or_dirty_session(optical_controller):
    c = optical_controller
    install_batch(c)
    c.start((80., 80.))
    draft, batch, before = c.draft, c._automatic_context, c.session.to_dict()
    c.configure_optical_analysis({"imaging_mode": "double_edge"})
    c.configure_automatic(algorithm="optical_adaptive")
    assert c.draft is draft and c._automatic_context is batch
    assert c.candidate_overlay_ready
    assert c.session.to_dict() == before
    assert c.optical_analysis_config["imaging_mode"] == "double_edge"
    assert batch["analysis_config"]["imaging_mode"] == "mixed"


def test_task_config_and_frame_are_frozen_until_explicit_refresh(optical_controller):
    c = optical_controller
    submitted = []
    c._automatic.submit = lambda pixels, **kwargs: submitted.append((pixels, kwargs)) or "task"
    c.configure_optical_analysis({"imaging_mode": "solid"})
    c.analyze("optical_adaptive")
    c.configure_optical_analysis({"imaging_mode": "double_edge"})
    assert submitted[0][0] is c.presented.frame.pixels
    assert submitted[0][1]["config"]["imaging_mode"] == "solid"
    assert c._automatic_context["analysis_config"]["imaging_mode"] == "solid"
    assert c._automatic_cycle_done


def test_quality_warning_does_not_add_confirmation_and_records_own_provenance(optical_controller):
    c = optical_controller
    install_batch(c)
    ids = [row["candidate_id"] for row in c.candidates]
    c.submit_candidate(ids[0], "wool")
    record = next(iter(c.session.measurements.values()))
    assert record.diameter_um == 16.
    assert record.measurement_method == "automatic-optical_adaptive"
    assert record.provenance["algorithm_revision"] == "optical-adaptive-v1"
    assert record.provenance["analysis_config"]["imaging_mode"] == "mixed"
    assert record.provenance["candidate_diagnostics"]["warning_codes"] == ["low_contrast"]
    assert record.frame_id == c.presented.frame.frame_id
    assert [row["candidate_id"] for row in c.candidates] == ids
    assert [row["state"] for row in c.candidates] == ["accepted", "available", "available"]
    c._automatic_context["analysis_config"]["imaging_mode"] = "solid"
    assert record.provenance["analysis_config"]["imaging_mode"] == "mixed"
    loaded = ExperimentSession.from_dict(c.session.to_dict())
    assert loaded.measurements[record.id].provenance == record.provenance


def test_conflicting_interpretation_uses_existing_duplicate_confirmation(optical_controller):
    c = optical_controller
    install_batch(c)
    assert c.possible_duplicate_candidates([0, 1, 2]) == [0, 1]
    assert c.possible_duplicate_candidates([0]) == []
    c.submit_candidate(c.candidates[0]["candidate_id"], "wool")
    assert c.possible_duplicate_candidates([1, 2]) == [1]
    with pytest.raises(ValueError, match="重复"):
        c.submit_candidate(c.candidates[1]["candidate_id"], "wool")
    assert len(c.session.measurements) == 1
    # Existing review confirmation is still the explicit acceptance path.
    c.accept_candidates([(1, "wool")])
    assert len(c.session.measurements) == 2
    c.submit_candidate(c.candidates[2]["candidate_id"], "wool")
    assert len(c.session.measurements) == 3


def test_invalid_reference_context_cannot_leak_analysis_lease(optical_controller, monkeypatch):
    import fdm.services.optical_flat_field as flat
    c = optical_controller
    before = c.budget.used_bytes
    monkeypatch.setattr(flat, "reference_context", lambda *_: (_ for _ in ()).throw(ValueError("invalid metadata")))
    with pytest.raises(ValueError, match="invalid metadata"):
        c.analyze("optical_adaptive")
    assert c.budget.used_bytes == before


def test_out_of_range_check_remains(optical_controller):
    c = optical_controller
    install_batch(c)
    c.candidates[0]["width_px"] = 2000.
    with pytest.raises(ValueError, match="越限"):
        c.submit_candidate(c.candidates[0]["candidate_id"], "cashmere")
    assert not c.session.measurements


def test_optical_frame_is_original_evidence_source(optical_controller, monkeypatch):
    c = optical_controller
    install_batch(c)
    seen = []
    monkeypatch.setattr(c, "_reserve_evidence", lambda intent, result:
                        seen.append(intent.token.frame) or {"state": "disabled"})
    c.submit_candidate(c.candidates[0]["candidate_id"], "wool")
    assert seen == [c._automatic_context["token"].frame]
    np.testing.assert_array_equal(seen[0].pixels, image())


def test_real_spawn_optical_configuration_is_owned_and_worker_reused():
    pixels = image()
    before = pixels.copy()
    with CuAutomaticWorker() as worker:
        config = {"imaging_mode": "solid", "background_mode": "local", "flat_enabled": False}
        task = worker.submit(pixels, "optical_adaptive", config=config)
        config["imaging_mode"] = "not-a-mode"
        deadline = time.monotonic() + 20
        messages = []
        while not messages and time.monotonic() < deadline:
            messages = worker.poll()
            time.sleep(.01)
        assert messages, "optical child did not respond"
        result = messages[0]
        assert result["task_id"] == task and result["error"] is None, result
        assert result["result"]["status"] == "ok", result
        assert len(result["result"]["candidates"]) >= 1
        assert any(abs(row["width_px"] - 32) <= 1 for row in result["result"]["candidates"])
        pid = worker._process.pid
        worker.submit(pixels, "traditional")
        messages = []
        deadline = time.monotonic() + 10
        while not messages and time.monotonic() < deadline:
            messages = worker.poll()
            time.sleep(.01)
        assert messages and messages[0]["error"] is None
        assert worker._process.pid == pid
        assert worker._segment is None
    np.testing.assert_array_equal(pixels, before)
