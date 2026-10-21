import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fdm.live_experiment.frames import FrameHandle
from fdm.services.optical_config import OpticalAnalysisConfig
from fdm.services.optical_flat_field import (
    apply_flat_field, capture_blank_reference, estimate_flat_field_scratch_bytes,
    flat_field_export_files, illumination_diagnostics, load_flat_field_reference,
    prepare_detection_image, prepare_flat_field_reference, reference_context,
    save_flat_field_reference,
)


def _frames(pixels, *, metadata=None):
    return [FrameHandle.snapshot(pixels, source_id="camera:1", generation=2, sequence=index,
                                 metadata=metadata) for index in range(5)]


@pytest.fixture
def reference(tmp_path):
    pixels = np.tile(np.arange(120, 200, dtype=np.uint8), (64, 1))
    frames = _frames(pixels, metadata={"orientation": {"flip_x": False}, "imaging_settings": {"exposure_us": 100}})
    context = reference_context(frames[0], SimpleNamespace(x_um=.63, y_um=.63, source="10x", revision=1))
    descriptor = capture_blank_reference(frames, context, tmp_path)
    return pixels, context, descriptor


def test_context_survives_reconnect_but_binds_readable_image_conditions():
    pixels = np.full((20, 30, 3), 180, np.uint8)
    frame = _frames(pixels, metadata={"device_id": "mv1", "orientation": {"rotation": 0}, "exposure": 15})[0]
    other = FrameHandle.snapshot(pixels, source_id=frame.source_id, generation=17, sequence=55, metadata=frame.metadata)
    before = reference_context(frame, SimpleNamespace(x_um=.63, y_um=.63, revision=1, source="10x"))
    after = reference_context(other, SimpleNamespace(x_um=.63, y_um=.63, revision=9, source="10x"))
    assert before == after
    assert before["width"] == 30
    assert before["pixel_format"] == "BGR"
    assert before["imaging_settings"] == {"exposure": 15}
    assert "generation" not in before


def test_five_frame_median_rejects_transient_outliers_without_changing_sources():
    sources = [np.full((20, 30), value, np.uint8) for value in (150, 151, 149, 255, 2)]
    originals = [image.copy() for image in sources]
    result = prepare_flat_field_reference(sources, reference_context(sources[0]))
    assert result["status"] == "ready"
    assert np.all(result["image"] == 150)
    assert not result["image"].flags.writeable
    for before, after in zip(originals, sources):
        np.testing.assert_array_equal(before, after)


def test_capture_requires_five_unique_frames_and_stable_source_context():
    pixels = np.full((20, 30), 150, np.uint8)
    frames = _frames(pixels, metadata={"imaging_settings": {"gain": 3}})
    context = reference_context(frames[0])
    assert prepare_flat_field_reference(frames[:4], context)["status"] == "invalid"
    assert prepare_flat_field_reference([frames[0]] * 5, context)["status"] == "invalid"
    changed = FrameHandle.snapshot(pixels, source_id="camera:1", generation=2, sequence=7,
                                  metadata={"imaging_settings": {"gain": 4}})
    assert prepare_flat_field_reference(frames[:4] + [changed], context)["status"] == "invalid"
    replaced = FrameHandle.snapshot(pixels, source_id="camera:1", generation=3, sequence=7)
    assert prepare_flat_field_reference(frames[:4] + [replaced], context)["status"] == "invalid"


def test_reference_persisted_with_hash_and_atomic_manifest(reference):
    pixels, context, descriptor = reference
    result = load_flat_field_reference(descriptor, context)
    assert result["status"] == "ready"
    np.testing.assert_array_equal(result["image"], pixels)
    manifest, image = flat_field_export_files(descriptor)
    assert hashlib.sha256(image.read_bytes()).hexdigest() == descriptor["sha256"]
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert metadata["frame_count"] == 5
    assert metadata["image_file"] == image.name
    assert metadata["context"] == context
    assert not list(manifest.parent.glob("*.tmp"))


@pytest.mark.parametrize("key,value", [
    ("source_id", "different-camera"), ("width", 81), ("height", 65),
    ("pixel_format", "BGR"), ("orientation", {"flip_x": True}),
    ("calibration", {"x_um": .7, "y_um": .7, "source": "10x"}),
    ("imaging_settings", {"exposure_us": 101}),
])
def test_known_change_disables_reference_without_changing_frame(reference, key, value):
    pixels, context, descriptor = reference
    changed = {**context, key: value}
    result = apply_flat_field(pixels, descriptor, changed)
    assert result["status"] == "not_applied"
    assert result["reason"] == "context_mismatch"
    assert not result["applied"]
    assert not np.shares_memory(result["image"], pixels)
    np.testing.assert_array_equal(result["image"], pixels)


def test_flat_field_flattens_only_detection_copy(reference):
    pixels, context, descriptor = reference
    original = pixels.copy()
    result = apply_flat_field(pixels, descriptor, context)
    assert result["applied"]
    assert np.ptp(result["image"]) == 0
    assert int(result["image"][0, 0]) == round(float(pixels.mean()))
    np.testing.assert_array_equal(pixels, original)


def test_bgra_alpha_and_channel_white_balance_are_preserved(tmp_path):
    reference = np.full((16, 20, 4), (100, 150, 200, 0), np.uint8)
    reference[:, 10:, :3] //= 2
    context = reference_context(reference)
    descriptor = capture_blank_reference([reference] * 5, context, tmp_path)
    source = reference.copy()
    source[..., 3] = np.arange(20)
    result = apply_flat_field(source, descriptor, context)
    assert result["applied"]
    for channel in range(3):
        assert np.ptp(result["image"][..., channel]) == 0
    np.testing.assert_array_equal(result["image"][..., 3], source[..., 3])


@pytest.mark.parametrize("fault", ["missing_image", "corrupt_png", "hash", "truncated_json", "traversal"])
def test_corrupt_reference_is_recoverable_and_explicit(reference, fault):
    pixels, context, descriptor = reference
    manifest = Path(descriptor["path"])
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    path = manifest.parent / metadata["image_file"]
    if fault == "missing_image":
        path.unlink()
    elif fault == "corrupt_png":
        path.write_bytes(b"not a PNG")
    elif fault == "hash":
        descriptor["sha256"] = "f" * 64
    elif fault == "truncated_json":
        manifest.write_text('{"version":', encoding="utf-8")
    elif fault == "traversal":
        metadata["image_file"] = "../outside.png"
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
    result = apply_flat_field(pixels, descriptor, context)
    assert result["status"] == "not_applied"
    assert result["reason"] in ("invalid", "missing")
    np.testing.assert_array_equal(result["image"], pixels)


def test_manifest_failure_removes_only_unpublished_new_png(tmp_path, monkeypatch):
    pixels = np.full((16, 20), 160, np.uint8)
    prepared = prepare_flat_field_reference([pixels] * 5, reference_context(pixels))
    old = capture_blank_reference([pixels] * 5, reference_context(pixels), tmp_path)
    existing = set(tmp_path.iterdir())

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("fdm.services.optical_flat_field.atomic_write_json", fail)
    with pytest.raises(OSError, match="disk full"):
        save_flat_field_reference(prepared, tmp_path)
    assert set(tmp_path.iterdir()) == existing
    assert load_flat_field_reference(old)["status"] == "ready"


def test_oversized_png_header_is_rejected_before_decoder_allocation(reference, monkeypatch):
    _, context, descriptor = reference
    manifest = Path(descriptor["path"])
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    path = manifest.parent / metadata["image_file"]
    content = bytearray(path.read_bytes())
    content[16:24] = struct.pack(">II", 100000, 100000)
    path.write_bytes(content)
    descriptor["sha256"] = hashlib.sha256(content).hexdigest()
    descriptor["context"].update(width=100000, height=100000)
    metadata.update(sha256=descriptor["sha256"], shape=[100000, 100000], context=descriptor["context"])
    manifest.write_text(json.dumps(metadata), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized reference must not reach the decoder")

    monkeypatch.setattr(cv2, "imdecode", forbidden)
    assert load_flat_field_reference(descriptor)["status"] == "resource_limit"


def test_memory_guards_use_wide_integer_and_cancel_before_work(reference):
    pixels, context, descriptor = reference
    assert estimate_flat_field_scratch_bytes((2**35, 2**25, 4)) > 2**63
    result = prepare_flat_field_reference([pixels] * 5, context, memory_budget=10)
    assert result["status"] == "resource_limit"
    assert load_flat_field_reference(descriptor, memory_budget=10)["status"] == "resource_limit"
    assert apply_flat_field(pixels, descriptor, context, memory_budget=10)["status"] == "resource_limit"
    assert prepare_flat_field_reference([pixels] * 5, context, checkpoint=lambda: True)["status"] == "cancelled"
    assert apply_flat_field(pixels, descriptor, context, checkpoint=lambda: True)["status"] == "cancelled"


def test_disabled_correction_never_reads_reference_and_only_copies_source(monkeypatch):
    pixels = np.full((16, 20), 160, np.uint8)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled correction must not load a reference")

    monkeypatch.setattr("fdm.services.optical_flat_field.load_flat_field_reference", forbidden)
    result = apply_flat_field(pixels, None, {}, enabled=False, memory_budget=pixels.nbytes)
    assert result["status"] == "disabled"
    np.testing.assert_array_equal(result["image"], pixels)


def test_saturation_low_contrast_and_unevenness_are_advisory():
    saturated = np.full((64, 80), 255, np.uint8)
    diagnostics = illumination_diagnostics(saturated)
    assert diagnostics["saturated_high_fraction"] == 1
    assert diagnostics["robust_contrast"] == 0
    assert diagnostics["advisory_only"] is True
    assert diagnostics["suggestions"]
    json.dumps(diagnostics, allow_nan=False)
    result = prepare_detection_image(saturated, OpticalAnalysisConfig(algorithm="optical_adaptive").to_dict())
    assert result["image"] is not None
    assert result["status"] == "disabled"
    assert result["diagnostics"]["illumination"]["advisory_only"]
    assert "image" not in result["diagnostics"]


def test_zero_reference_rejected_but_transparent_alpha_allowed():
    black = np.zeros((16, 20), np.uint8)
    assert prepare_flat_field_reference([black] * 5, reference_context(black))["status"] == "invalid"
    rgba = np.full((16, 20, 4), (120, 160, 180, 0), np.uint8)
    assert prepare_flat_field_reference([rgba] * 5, reference_context(rgba))["status"] == "ready"
