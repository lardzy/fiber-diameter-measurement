"""Original-frame crop, byte ownership and legacy persistence contracts."""
import hashlib
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin
import pytest

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.evidence import EvidenceStore
from fdm.live_experiment.evidence_options import evidence_crop, measurement_evidence_points
from fdm.live_experiment.frames import FrameBudget, FrameBudgetExceeded, FrameHandle
from fdm.live_experiment.jpeg_evidence import jpeg_workspace_bytes
from fdm.live_experiment.persistence import load_session, save_session
from fdm.live_experiment.project_assets import (hydrate_experiment_assets,
    rebase_experiment_assets, stage_experiment_assets)
from fdm.models import ProjectState
from fdm.project_io import ProjectIO, project_assets_root


CROP = {"version": 1, "mode": "crop_jpeg"}


def frame(pixels=None, sequence=0):
    if pixels is None:
        pixels = np.zeros((576, 768, 3), np.uint8)
        pixels[:, :384] = (13, 80, 220)
        pixels[:, 384:] = (220, 80, 13)
    return FrameHandle.snapshot(pixels, source_id="crop", sequence=sequence, live=False)


@pytest.mark.parametrize("points", [((0, 0), (5, 4)), ((767, 575), (768, 576)),
    ((4, 570), (8, 575)), ((500, 100), (510, 120)), ((100, 100), (700, 500))])
def test_crop_keeps_endpoints_and_minimum_context_at_all_boundaries(points):
    x, y, width, height = evidence_crop((768, 576), points)
    assert width >= 256 and height >= 256
    assert 0 <= x <= 768 - width and 0 <= y <= 576 - height
    for px, py in points:
        assert x <= px <= x + width and y <= py <= y + height
        assert px - x >= min(64, px) and x + width - px >= min(64, 768 - px)
        assert py - y >= min(64, py) and y + height - py >= min(64, 576 - py)


def test_tiny_image_and_automatic_all_sample_geometry_are_preserved():
    assert evidence_crop((23, 18), [(10, 10)]) == (0, 0, 23, 18)
    result = {"start": [380, 250], "end": [400, 250],
              "input_start": [375, 250], "input_end": [405, 250],
              "samples": [{"start": [50, 30], "end": [60, 30]},
                          {"start": [700, 540], "end": [710, 540]}],
              "path": [[55, 30], [705, 540]]}
    points = measurement_evidence_points(result)
    assert len(points) == 10
    assert evidence_crop((768, 576), points) == (0, 0, 768, 576)


@pytest.mark.parametrize("points", [[], [[float("nan"), 0]], [[-1, 3]], [[769, 0]], [[True, 0]]])
def test_invalid_crop_is_not_silently_clamped(points):
    with pytest.raises(ValueError):
        evidence_crop((768, 576), points)


def test_same_frame_distinct_regions_are_independent_jpeg_444_and_deduped(tmp_path):
    value = frame()
    store = EvidenceStore(tmp_path)
    left = store.reserve(value, options=CROP, points=[(100, 100), (120, 100)])
    right = store.reserve(value, options=CROP, points=[(600, 400), (620, 400)])
    assert store.reserve(value, options=CROP, points=[(100, 100), (120, 100)]) is left
    assert left["evidence_id"] != right["evidence_id"]
    results = store.close()
    assert len(results) == 2 and all(result["state"] == "saved" for result in results)
    for result, color in zip(results, [(13, 80, 220), (220, 80, 13)]):
        assert result["codec"] == "jpeg" and result["source_size"] == [768, 576]
        assert result["quality"] == 95 and result["subsampling"] == "4:4:4"
        assert result["sha256"] == hashlib.sha256(Path(result["path"]).read_bytes()).hexdigest()
        with Image.open(result["path"]) as image:
            assert image.format == "JPEG" and image.size == (256, 256)
            assert JpegImagePlugin.get_sampling(image) == 0
        decoded = cv2.imread(result["path"])
        np.testing.assert_allclose(decoded[120, 120], color, atol=2)
    assert store.budget.used_bytes == 0


@pytest.mark.parametrize("alpha", [254, 255])
def test_real_alpha_uses_local_lossless_png_but_opaque_bgra_is_jpeg(tmp_path, alpha):
    pixels = np.full((576, 768, 4), (13, 80, 220, 255), np.uint8)
    pixels[120, 120, 3] = alpha
    value = frame(pixels)
    store = EvidenceStore(tmp_path)
    store.reserve(value, options=CROP, points=[(100, 100), (120, 100)])
    result, = store.close()
    assert result["codec"] == ("png" if alpha < 255 else "jpeg")
    if alpha < 255:
        x, y, w, h = result["crop"]
        np.testing.assert_array_equal(cv2.imread(result["path"], cv2.IMREAD_UNCHANGED), pixels[y:y+h, x:x+w])


def test_crop_budget_reservation_rejects_before_ledger_write_and_rolls_back(tmp_path):
    value = frame()
    budget = FrameBudget(value.nbytes + jpeg_workspace_bytes(256, 256) - 1)
    store = EvidenceStore(tmp_path, budget=budget)
    try:
        with pytest.raises(FrameBudgetExceeded):
            store.reserve(value, options=CROP, points=[(100, 100)])
        assert budget.used_bytes == 0 and not store.pending
        assert not list(tmp_path.iterdir())
    finally:
        store.close()


def test_jpeg_encoder_failure_is_durable_and_never_leaks_raw_frame_or_pending_file(tmp_path, monkeypatch):
    import fdm.live_experiment.evidence as module
    store = EvidenceStore(tmp_path)
    original = module.write_jpeg_evidence
    def fail(pixels, target, **kwargs):
        target.write(b"partial jpeg")
        raise OSError("JPEG encoder fault")
    monkeypatch.setattr(module, "write_jpeg_evidence", fail)
    try:
        store.reserve(frame(), options=CROP, points=[(100, 100)])
        failure, = store.flush()
        assert failure["state"] == "failed" and "JPEG encoder fault" in failure["error"]
        assert not list(tmp_path.iterdir()) and store.budget.used_bytes == 0
        monkeypatch.setattr(module, "write_jpeg_evidence", original)
        store.reserve(frame(sequence=1), options=CROP, points=[(100, 100)])
        done, = store.close()
        assert done["state"] == "saved"
    finally:
        store.close()


def test_jpeg_releases_source_before_fsync_and_keeps_codec_ticket(tmp_path, monkeypatch):
    import fdm.live_experiment.evidence as module
    store = EvidenceStore(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = module.os.fsync
    def gate(fd):
        entered.set()
        assert release.wait(3)
        return original(fd)
    monkeypatch.setattr(module.os, "fsync", gate)
    try:
        store.reserve(frame(), options=CROP, points=[(100, 100)])
        assert entered.wait(3)
        assert not store.budget._frames
        assert store.budget.used_bytes == jpeg_workspace_bytes(256, 256)
    finally:
        release.set()
        store.close()
    assert store.budget.used_bytes == 0


def test_legacy_policies_do_not_migrate_and_explicit_policy_is_reversible(tmp_path):
    for enabled in (True, False):
        old = ExperimentSession(save_images=enabled)
        before = old.to_dict()
        loaded = ExperimentSession.from_dict(before)
        assert loaded.to_dict() == before and "evidence_options" not in before
        assert loaded.effective_evidence_options["mode"] == ("full_png" if enabled else "disabled")
    session = ExperimentSession(evidence_options=CROP)
    assert session.save_images and session.revision == 0
    session.configure(evidence_options={"mode": "disabled"})
    assert not session.save_images
    session.undo()
    assert session.save_images and session.effective_evidence_options["mode"] == "crop_jpeg"
    path = tmp_path / "session.json"
    save_session(session, path)
    assert load_session(path).to_dict() == session.to_dict()
    session.configure(save_images=False)
    assert session.effective_evidence_options["mode"] == "disabled"
    session.undo()
    assert session.effective_evidence_options["mode"] == "crop_jpeg"


def project_with_crop(tmp_path):
    store = EvidenceStore(tmp_path / "temporary")
    pending = dict(store.reserve(frame(), options=CROP, points=[(100, 100), (120, 100)]))
    session = ExperimentSession(evidence_options=CROP)
    session.add_operator("A")
    session.set_materials(default_materials())
    record = session.add_measurement(20, "wool", endpoints=((100, 100), (120, 100)), evidence=pending)
    assert session.undo()
    done, = store.close()
    assert session.set_evidence(record.id, done)
    assert session.redo()
    project = ProjectState.empty()
    project.live_experiments.append(session)
    return project, session, record.id


def test_jpeg_save_as_rebase_undo_and_reload_preserve_metadata_and_coordinates(tmp_path):
    project, session, record_id = project_with_crop(tmp_path)
    first, second = tmp_path / "first.fdmproj", tmp_path / "next" / "second.fdmproj"
    ProjectIO.save(project, first)
    loaded = ProjectIO.load(first)
    ProjectIO.save(loaded, second)
    saved_payload = loaded.to_dict()
    restored = loaded.live_experiments[0]
    revision = restored.revision
    rebase_experiment_assets([restored], saved_payload, project_assets_root(second))
    assert restored.revision == revision
    evidence = restored.measurements[record_id].evidence
    assert Path(evidence["path"]).suffix == ".jpg"
    assert Path(evidence["path"]).is_relative_to(project_assets_root(second).resolve())
    (project_assets_root(first)).rename(tmp_path / "moved-assets")
    assert Path(evidence["path"]).is_file()
    loaded_again = ProjectIO.load(second).live_experiments[0]
    assert loaded_again.measurements[record_id].endpoints == session.measurements[record_id].endpoints
    assert loaded_again.measurements[record_id].evidence["crop"] == evidence["crop"]
    assert loaded_again.effective_evidence_options["mode"] == "crop_jpeg"


@pytest.mark.parametrize("mutation", ["codec", "crop", "extension"])
def test_declared_codec_and_crop_must_match_asset(tmp_path, mutation):
    project, session, record_id = project_with_crop(tmp_path)
    payload = project.to_dict()
    evidence = payload["live_experiments"][0]["measurements"][0]["evidence"]
    if mutation == "codec":
        evidence["codec"] = "png"
    elif mutation == "crop":
        evidence["crop"][2] += 1
    else:
        original = Path(evidence["path"])
        renamed = original.with_suffix(".png")
        original.rename(renamed)
        evidence["path"] = str(renamed)
    with pytest.raises(ValueError, match="证据"):
        stage_experiment_assets(payload, tmp_path / "assets", [])


def test_shared_asset_checks_each_records_crop_not_just_first_hash(tmp_path):
    project, session, record_id = project_with_crop(tmp_path)
    evidence = dict(session.measurements[record_id].evidence)
    evidence["crop"] = [*evidence["crop"][:2], 257, 256]
    session.add_measurement(21, "wool", evidence=evidence)
    with pytest.raises(ValueError, match="裁剪尺寸不一致"):
        stage_experiment_assets(project.to_dict(), tmp_path / "assets", [])


def test_corrupted_codec_metadata_hydrates_as_savable_failure(tmp_path):
    project, session, record_id = project_with_crop(tmp_path)
    created = []
    root = tmp_path / "assets"
    payload = stage_experiment_assets(project.to_dict(), root, created)
    evidence = payload["live_experiments"][0]["measurements"][0]["evidence"]
    evidence["crop"][2] += 1
    restored = ExperimentSession.from_dict(payload["live_experiments"][0])
    hydrate_experiment_assets([restored], root)
    assert restored.measurements[record_id].evidence["state"] == "failed"
    saved = stage_experiment_assets({"live_experiments": [restored.to_dict()]}, root, [])
    assert saved["live_experiments"][0]["measurements"][0]["evidence"]["state"] == "failed"
