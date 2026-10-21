from copy import deepcopy

import pytest

from fdm.services import cu_live_self_check as probe


CONFIGURATION_CASES = ("content_deferred_rebuild", "content_width_undo", "content_width_redo")
EVIDENCE_CASES = ("evidence_png_gray", "evidence_png_bgr", "evidence_png_bgra",
                  "evidence_leases_released", "project_asset_rebase")
ALIGNMENT_CASES = ("candidate_alignment_worker", "candidate_alignment_same",
                   "candidate_alignment_noisy", "candidate_alignment_drift",
                   "candidate_alignment_reference_reset", "candidate_alignment_closed",
                   "candidate_alignment_child_process")


def complete_report():
    names = ("manual_native", "traditional_native", "multiscale_native", "optical_adaptive", "optical_empty", "optical_worker", "content_formula",
             "field_motion", "field_preparation", "project_roundtrip", "microview_bridge_import", "microview_bridge_build", "microview_bridge_transport",
             *CONFIGURATION_CASES, *EVIDENCE_CASES, *ALIGNMENT_CASES)
    return {"ok": True, "revision": 6, "cases": {name: True for name in names}}


def test_field_preparation_probe_runs_native_moments_and_ordered_decisions():
    assert probe._field_preparation_probe()


def test_field_preparation_probe_rejects_wrong_moments_even_if_both_paths_share_them(monkeypatch):
    import fdm.live_experiment.motion as motion
    monkeypatch.setattr(motion, "dark_centroid", lambda gray: (0, 0))
    assert not probe._field_preparation_probe()


def test_configuration_probe_runs_real_deferred_custom_width_and_history():
    assert probe._content_configuration_probe() == {name: True for name in CONFIGURATION_CASES}


def test_evidence_probe_runs_real_writer_save_and_history_rebase():
    assert probe._evidence_persistence_probe() == {name: True for name in EVIDENCE_CASES}


@pytest.mark.parametrize("name", ("field_preparation", "optical_adaptive", "optical_empty", "optical_worker", *CONFIGURATION_CASES, *EVIDENCE_CASES,
                                 *ALIGNMENT_CASES))
@pytest.mark.parametrize("failure", ["missing", "false"])
def test_release_gate_requires_each_actual_configuration_and_evidence_case(name, failure):
    report = complete_report()
    if failure == "missing":
        del report["cases"][name]
    else:
        report["cases"][name] = False
    assert not probe.cu_live_probe_passed(report)


@pytest.mark.parametrize("revision", [1, 2, 3, 4, 5])
def test_previous_probe_revision_cannot_validate_alignment_worker_release(revision):
    report = complete_report()
    report["revision"] = revision
    assert not probe.cu_live_probe_passed(report)


def test_current_report_is_accepted_without_weakening_microview_gate(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "win32")
    report = complete_report()
    assert probe.cu_live_probe_passed(report)
    missing_capture = deepcopy(report)
    del missing_capture["cases"]["microview_bridge_build"]
    assert not probe.cu_live_probe_passed(missing_capture)


def test_alignment_probe_uses_real_child_process_and_retains_runtime():
    import os

    checks, details = probe._candidate_alignment_worker_probe()
    assert checks == {name: True for name in ALIGNMENT_CASES}, details
    assert details["worker_runtime"]["pid"] > 0
    assert details["worker_runtime"]["pid"] != os.getpid()
    assert details["worker_runtime"]["opencv_threads"] >= 1


@pytest.mark.parametrize("field", ("ok", "same", "noisy", "drift", "reference_reset",
                                  "closed", "child_process"))
@pytest.mark.parametrize("failure", ("missing", "false", "truthy"))
def test_alignment_probe_requires_each_strict_boolean(monkeypatch, field, failure):
    from fdm.live_experiment import candidate_alignment_worker as worker

    details = {name: True for name in ("ok", "same", "noisy", "drift", "reference_reset",
                                      "closed", "child_process")}
    if failure == "missing":
        del details[field]
    else:
        details[field] = False if failure == "false" else 1
    monkeypatch.setattr(worker, "run_alignment_worker_self_check", lambda: details)
    checks, actual = probe._candidate_alignment_worker_probe()
    assert actual is details
    assert sum(not result for result in checks.values()) == 1
    report = complete_report()
    report["cases"].update(checks)
    assert not probe.cu_live_probe_passed(report)


@pytest.mark.parametrize("failure", (RuntimeError("child failed to start"), None, []))
def test_alignment_probe_preserves_failure_diagnostics(monkeypatch, failure):
    from fdm.live_experiment import candidate_alignment_worker as worker

    def fail():
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(worker, "run_alignment_worker_self_check", fail)
    checks, details = probe._candidate_alignment_worker_probe()
    assert checks == {name: False for name in ALIGNMENT_CASES}
    assert details["ok"] is False
    assert "error" in details
    if isinstance(failure, Exception):
        assert "child failed to start" in details["error"]


def test_live_report_keeps_child_diagnostics_and_fails_on_alignment_failure(monkeypatch):
    from fdm.live_experiment import candidate_alignment_worker as worker

    details = {"ok": False, "error": "failed to read shared frame",
               "worker_runtime": {"pid": 12345, "opencv_threads": 2}}
    monkeypatch.setattr(worker, "run_alignment_worker_self_check", lambda: details)
    report = probe.run_cu_live_self_check(validate_capture=False)
    assert report["revision"] == 6
    assert report["ok"] is False
    assert report["candidate_alignment_worker"] is details
    assert all(report["cases"][name] is False for name in ALIGNMENT_CASES)
    assert all(result for name, result in report["cases"].items() if name not in ALIGNMENT_CASES)


def test_probe_detects_missing_deferred_transaction(monkeypatch):
    from fdm.live_experiment import ExperimentSession

    monkeypatch.setattr(ExperimentSession, "configure_deferred", lambda self, **changes: None)
    assert probe._content_configuration_probe() == {name: False for name in CONFIGURATION_CASES}


def test_evidence_probe_rejects_changed_decoded_pixels(monkeypatch):
    import cv2
    original = cv2.imdecode

    def corrupt(*args, **kwargs):
        decoded = original(*args, **kwargs)
        decoded.flat[0] ^= 1
        return decoded

    monkeypatch.setattr(cv2, "imdecode", corrupt)
    checks = probe._evidence_persistence_probe()
    assert all(not checks[name] for name in EVIDENCE_CASES if name != "evidence_leases_released")
    assert checks["evidence_leases_released"]


def test_evidence_probe_requires_actual_stream_encoder_identity(monkeypatch):
    from fdm.live_experiment.evidence import EvidenceStore
    original = EvidenceStore.close

    def obsolete_encoder(self):
        results = original(self)
        for entry in self._entries.values():
            entry["encoder"] = "obsolete-encoder"
        return results

    monkeypatch.setattr(EvidenceStore, "close", obsolete_encoder)
    checks = probe._evidence_persistence_probe()
    assert all(not checks[name] for name in EVIDENCE_CASES if name != "evidence_leases_released")
    assert checks["evidence_leases_released"]


def test_evidence_probe_detects_missing_live_asset_rebase(monkeypatch):
    import fdm.live_experiment.project_assets as assets
    monkeypatch.setattr(assets, "rebase_experiment_assets", lambda *args: None)
    checks = probe._evidence_persistence_probe()
    assert not checks["project_asset_rebase"]
    assert all(checks[name] for name in EVIDENCE_CASES if name != "project_asset_rebase")
