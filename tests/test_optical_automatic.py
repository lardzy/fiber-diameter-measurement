from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from fdm.services.cu_automatic_measurement import CuAutomaticCancelled
from fdm.services.optical_automatic import analyze_optical_frame


def optical_strip(width=32., angle=15., kind="solid_dark", *, gradient=False, shape=(256, 384), seed=18):
    """Analytic pixel coverage; truth is distance between the two outer edges."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    theta = math.radians(angle)
    x, y = xx - (shape[1] - 1) / 2, yy - (shape[0] - 1) / 2
    along = x * math.cos(theta) + y * math.sin(theta)
    across = -x * math.sin(theta) + y * math.cos(theta)
    body = np.clip(width / 2 - np.abs(across) + .5, 0, 1)
    body *= np.clip(100 - np.abs(along) + .5, 0, 1)
    background = 220 + (xx / shape[1] * 30 - 15 if gradient else 0)
    background = np.broadcast_to(background, shape).astype(float)
    if kind == "solid_dark":
        image = background - 170 * body
    elif kind == "solid_pale":
        image = background - 18 * body
    else:
        inner = np.clip(width / 2 - 2.5 - np.abs(across) + .5, 0, 1)
        inner *= np.clip(100 - np.abs(along) + .5, 0, 1)
        image = background - 170 * body + 152 * inner
    image = cv2.GaussianBlur(image.astype(np.float32), (3, 3), .65)
    image += np.random.default_rng(seed).normal(0, .35, shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("kind", ["solid_dark", "solid_pale", "double_edge"])
@pytest.mark.parametrize("width", [8, 12, 20, 32, 48, 80])
@pytest.mark.parametrize("angle", [0, 25, 55, 90, 135])
def test_source_resolution_outer_diameter(kind, width, angle):
    image = optical_strip(width, angle, kind, gradient=True)
    before = image.copy()
    result = analyze_optical_frame(image)
    assert result["status"] == "ok", result
    assert len(result["candidates"]) == 1, [(c["width_px"], c["structure"]) for c in result["candidates"]]
    candidate = result["candidates"][0]
    assert abs(candidate["width_px"] - width) <= 1.0
    assert candidate["samples"]
    assert candidate["quality_features"]["valid_sections"] >= 3
    np.testing.assert_array_equal(image, before)


def test_two_neighboring_real_solid_fibers_are_not_joined():
    image = np.full((256, 384), 220, np.uint8)
    image[70:90, 40:340] = 40
    image[110:130, 40:340] = 40
    result = analyze_optical_frame(image)
    individual = [c for c in result["candidates"] if abs(c["width_px"] - 20) < 1]
    assert len(individual) == 2
    # Background-like space between two bands cannot establish whether they
    # are distinct fibers or two dark edges. Keep both actual alternatives.
    wide = [c for c in result["candidates"] if c not in individual]
    assert len(wide) == 1 and "ambiguous_pair" in wide[0]["warning_codes"]
    assert set(wide[0]["conflicts_with"]) == {c["hypothesis_id"] for c in individual}


def test_color_gray_bgra_and_strided_inputs_preserve_geometry():
    gray = optical_strip()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    widths = []
    for image in (gray, bgr, bgra, bgr[:, ::-1]):
        image.setflags(write=False)
        widths.append(analyze_optical_frame(image)["candidates"][0]["width_px"])
    assert np.ptp(widths) < .05


def test_empty_budget_and_cancellation_have_no_partial_candidates():
    image = optical_strip()
    result = analyze_optical_frame(image, memory_budget=1024)
    assert result["status"] == "resource_limit" and not result["candidates"]
    assert analyze_optical_frame(np.full_like(image, 220))["status"] == "empty"
    with pytest.raises(CuAutomaticCancelled):
        analyze_optical_frame(image, checkpoint=lambda: True)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls >= 6
    with pytest.raises(CuAutomaticCancelled):
        analyze_optical_frame(image, checkpoint=cancel)


def test_explicit_imaging_modes_and_invalid_settings():
    image = optical_strip(kind="double_edge")
    assert analyze_optical_frame(image, config={"imaging_mode": "double_edge"})["status"] == "ok"
    assert analyze_optical_frame(image, config={"imaging_mode": "solid"})["status"] == "ok"
    with pytest.raises(ValueError):
        analyze_optical_frame(image, config={"imaging_mode": "mystery"})


def test_crossing_uses_local_arms_not_the_intersection():
    image = np.full((384, 384), 220, np.uint8)
    image[176:208, 35:350] = 40
    image[35:350, 176:208] = 40
    result = analyze_optical_frame(image)
    assert result["status"] == "ok"
    assert len(result["candidates"]) >= 2
    for c in result["candidates"]:
        assert abs(c["width_px"] - 32) < 1
        center = (np.asarray(c["start"]) + c["end"]) / 2
        assert np.linalg.norm(center - 192) > 40
        assert "near_crossing" in c["warning_codes"]


def test_curved_fiber_is_measured_in_a_local_section():
    yy, xx = np.mgrid[:384, :384]
    dx, dy = xx - 95., yy - 192.
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    mask = (np.abs(r - 140) <= 10) & (np.abs(theta) < 1.15)
    image = np.full((384, 384), 220, np.uint8)
    image[mask] = 40
    result = analyze_optical_frame(image)
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert abs(candidate["width_px"] - 20) < 1
    assert np.ptp(np.asarray(candidate["path"]), axis=0).max() <= 40


@pytest.mark.parametrize("core", [218, 219, 220])
def test_ambiguous_clear_core_preserves_rim_alternatives_and_links(core):
    image = np.full((256, 384), 220, np.uint8)
    image[110:142, 40:340] = core
    image[110:113, 40:340] = 40
    image[139:142, 40:340] = 40
    result = analyze_optical_frame(image)
    assert result["status"] == "ok"
    wide = [c for c in result["candidates"] if c["width_px"] > 20]
    assert len(wide) == 1
    assert "ambiguous_pair" in wide[0]["warning_codes"]
    assert len(wide[0]["conflicts_with"]) == 2
    by_id = {c["hypothesis_id"]: c for c in result["candidates"]}
    for other in wide[0]["conflicts_with"]:
        assert wide[0]["hypothesis_id"] in by_id[other]["conflicts_with"]
        assert by_id[other]["diagnostics"]["warning_codes"] == by_id[other]["warning_codes"]


def test_unavailable_flat_field_is_explicit_and_does_not_change_source():
    image = optical_strip()
    before = image.copy()
    result = analyze_optical_frame(image, config={"flat_enabled": True})
    assert result["status"] == "ok"
    assert result["diagnostics"]["flat_field"]["applied"] is False
    assert result["diagnostics"]["flat_field"]["reason"] == "missing"
    np.testing.assert_array_equal(before, image)


def test_refinement_value_and_representative_use_actual_retained_sections():
    from fdm.services.optical_automatic import PARAMETERS, _Hypothesis, _refine
    image = optical_strip(width=32, angle=30).astype(np.float32)
    # Deliberately wrong proposal direction: projection must not silently
    # report a short perpendicular width alongside the longer actual line.
    hypothesis = _Hypothesis(np.array([191.5, 127.5]), np.array([1., 0.]),
                             32., 160., "solid", "test-oblique")
    result = _refine(hypothesis, image, .5, PARAMETERS, None)
    kept = [s for s in result["samples"] if s["retained"]]
    expected = float(np.median([s["raw_width"] for s in kept]))
    assert result["width_px"] == pytest.approx(expected)
    actual = np.linalg.norm(np.asarray(result["end"]) - result["start"])
    closest = min(abs(s["raw_width"] - expected) for s in kept)
    assert abs(actual - expected) == pytest.approx(closest, abs=.01)
    assert any(s["start"] == result["start"] and s["end"] == result["end"] for s in kept)


@pytest.mark.parametrize("width,angle,phase,seed", [
    (6, 9, .17, 9241), (8, 43, -.22, 7135), (10, 81, .39, 8127),
    (16, 123, -.31, 5117), (8, 69, .43, 9113), (8, 133, -.19, 2317)])
def test_complete_body_owns_remote_end_cap_rims(width, angle, phase, seed):
    factor, height, width_image = 4, 256, 384
    yy, xx = np.mgrid[:height * factor, :width_image * factor].astype(float)
    xx, yy = (xx + .5) / factor - .5, (yy + .5) / factor - .5
    theta = math.radians(angle)
    dx, dy = xx - width_image / 2 - phase, yy - height / 2 + phase
    along = dx * math.cos(theta) + dy * math.sin(theta)
    across = -dx * math.sin(theta) + dy * math.cos(theta)
    body = (np.abs(across) < width / 2) & (np.abs(along) < 90)
    image = 210 + 20 * xx / width_image + 5 * yy / height
    image[body] -= 125
    image[body & (np.abs(across) < width / 2 - min(3, width * .2))] += 113
    image = cv2.resize(image.astype(np.float32), (width_image, height), interpolation=cv2.INTER_AREA)
    image = cv2.GaussianBlur(image, (5, 5), .8)
    image += np.random.default_rng(seed).normal(0, .55, image.shape)
    image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    result = analyze_optical_frame(image)
    assert len(result["candidates"]) == 1, [(c["hypothesis_id"], c["width_px"]) for c in result["candidates"]]
    assert abs(result["candidates"][0]["width_px"] - width) < 1


def test_complete_pale_bodies_do_not_pair_their_rims_across_background():
    image = np.full((576, 768), 220, np.uint8)
    for top in (130, 280, 430):
        image[top:top+32, 70:698] = 202
        image[top:top+3, 70:698] = 40
        image[top+29:top+32, 70:698] = 40
    result = analyze_optical_frame(image)
    assert len(result["candidates"]) == 3
    assert all(abs(c["width_px"] - 32) < 1 for c in result["candidates"])
    assert all(not c["conflicts_with"] for c in result["candidates"])


def test_pair_index_limits_sparse_queries_by_position_and_axis():
    from fdm.services.optical_automatic import PARAMETERS, _Hypothesis, _RimIndex
    rims = [_Hypothesis(np.array([float(x), float(y)]), np.array([1., 0.]),
                        3., 60., "solid", f"rim-{i}")
            for i, (x, y) in enumerate((x, y) for x in range(0, 1440, 120) for y in range(0, 1440, 120))]
    index = _RimIndex(rims, 40., PARAMETERS, None)
    visited = sum(len(list(index.query(rim, 40., None))) for rim in rims)
    assert visited == len(rims)
    # Opposite tangent signs share an undirected axis; orthogonal axes do not.
    reverse = _Hypothesis(rims[0].center, np.array([-1., 0.]), 3., 60., "solid", "reverse")
    assert list(index.query(reverse, 40., None)) == [0]
    upright = _Hypothesis(rims[0].center, np.array([0., 1.]), 3., 60., "solid", "upright")
    assert not list(index.query(upright, 40., None))
