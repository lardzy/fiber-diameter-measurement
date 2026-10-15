"""Persist failed evidence without losing measurements or claiming saved pixels."""
import hashlib
import json
from pathlib import Path

import pytest

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.project_assets import hydrate_experiment_assets, stage_experiment_assets
from fdm.models import ProjectState
from fdm.project_io import ProjectIO, project_assets_root


def make_project(evidence):
    session = ExperimentSession()
    session.add_operator("A")
    session.set_materials(default_materials())
    record = session.add_measurement(20, "wool", evidence=evidence)
    project = ProjectState.empty()
    project.live_experiments.append(session)
    return project, session, record


@pytest.mark.parametrize("failed_path", ["/tmp/vanished.png", r"C:\\Temp\\missing.png", "../outside.png", "", None])
def test_failed_evidence_saves_record_without_temporary_path(tmp_path, failed_path):
    evidence = {"state": "failed", "frame_id": "frame1", "error": "disk full", "sha256": None}
    if failed_path is not None:
        evidence["path"] = failed_path
    project, session, record = make_project(evidence)
    path = tmp_path / "experiment.fdm.json"
    ProjectIO.save(project, path)
    payload = json.loads(path.read_text())
    saved = payload["live_experiments"][0]["measurements"][0]
    assert saved["diameter_um"] == 20
    assert saved["evidence"] == {"state": "failed", "frame_id": "frame1", "error": "disk full", "sha256": None}
    assert session.measurements[record.id].evidence == evidence  # staging is a snapshot
    loaded = ProjectIO.load(path)
    restored = loaded.live_experiments[0].measurements[record.id]
    assert restored.evidence["state"] == "failed"
    ProjectIO.save(loaded, path)


def test_missing_durable_asset_retains_relative_failure_reference(tmp_path):
    path = tmp_path / "experiment.fdm.json"
    asset = project_assets_root(path) / "experiments/missing.png"
    project, _, record = make_project({"state": "failed", "path": str(asset), "error": "missing", "sha256": "a" * 64})
    ProjectIO.save(project, path)
    stored = ProjectIO.load(path).live_experiments[0].measurements[record.id].evidence
    assert stored["state"] == "failed"
    assert stored["path"] == "experiments/missing.png"


def test_hydrated_missing_saved_image_becomes_savable_failure(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"opaque image evidence")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    project, _, record = make_project({"state": "saved", "path": str(source), "sha256": digest})
    path = tmp_path / "experiment.fdm.json"
    ProjectIO.save(project, path)
    (project_assets_root(path) / f"experiments/{digest}.png").unlink()
    loaded = ProjectIO.load(path)
    assert loaded.live_experiments[0].measurements[record.id].evidence["state"] == "failed"
    ProjectIO.save(loaded, path)
    assert ProjectIO.load(path).live_experiments[0].measurements[record.id].evidence["state"] == "failed"


def test_pending_evidence_still_blocks_incomplete_publication(tmp_path):
    project, _, _ = make_project({"state": "pending", "path": str(tmp_path / "unfinished.png")})
    path = tmp_path / "experiment.fdm.json"
    path.write_text("previous valid project")
    with pytest.raises(ValueError, match="尚未保存成功"):
        ProjectIO.save(project, path)
    assert path.read_text() == "previous valid project"


def test_hydration_hashes_shared_asset_once_and_checks_every_expected_digest(tmp_path, monkeypatch):
    import fdm.live_experiment.project_assets as assets
    source = tmp_path / "experiments" / "shared.png"
    source.parent.mkdir()
    pixels = b"same-frame-evidence" * 100
    source.write_bytes(pixels)
    digest = hashlib.sha256(pixels).hexdigest()
    sessions = []
    for expected in (digest, digest, "0" * 64):
        _, session, _ = make_project({"state": "saved", "path": "experiments/shared.png", "sha256": expected})
        sessions.append(session)
    original = assets._file_sha256
    reads = []
    def count_hash(path):
        reads.append(path)
        return original(path)
    monkeypatch.setattr(assets, "_file_sha256", count_hash)
    hydrate_experiment_assets(sessions, tmp_path)
    assert reads == [source.resolve()]
    records = [next(iter(session.measurements.values())) for session in sessions]
    assert [record.evidence["state"] for record in records] == ["saved", "saved", "failed"]
    assert "摘要不匹配" in records[-1].evidence["error"]
    # The cache belongs only to one load, never to a later changed file.
    source.write_bytes(b"changed")
    _, next_session, _ = make_project({"state": "saved", "path": "experiments/shared.png", "sha256": digest})
    hydrate_experiment_assets([next_session], tmp_path)
    assert len(reads) == 2
    assert next(iter(next_session.measurements.values())).evidence["state"] == "failed"


def test_evidence_hashing_streams_instead_of_allocating_complete_image(tmp_path, monkeypatch):
    from fdm.live_experiment.project_assets import _file_sha256
    source = tmp_path / "large.png"
    payload = b"a" * (2 * 1024 * 1024 + 13)
    source.write_bytes(payload)
    def reject_read_bytes(path):
        raise AssertionError("whole-image allocation is forbidden")
    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    assert _file_sha256(source) == hashlib.sha256(payload).hexdigest()
