from __future__ import annotations

import json
import math

import numpy as np
import pytest

from fdm.services.metrology_fitting import (
    CaliperOptions, fit_circle_points, fit_line_points,
    run_circle_caliper, run_line_caliper,
)


def _edge(y: np.ndarray, location: float, blur: float = 0.8) -> np.ndarray:
    # Analytic, continuously located edge independent of the sampling code.
    return (0.5 + 0.5 * np.tanh((y - location) / blur)).astype(np.float32)


def test_exact_line_physical_space_and_global_coordinates() -> None:
    origin = np.array([100_000_000.0, -200_000_000.0])
    x = np.linspace(-30, 30, 31)
    points = np.column_stack((x, 2 * x + 4)) + origin
    result = fit_line_points(points, scale=(2, 0.5))
    assert result.success, result.reason
    direction = np.asarray(result.geometry["direction"])
    assert abs(np.dot(direction, np.array([2, 1]) / math.sqrt(5))) == pytest.approx(1)
    assert result.rms < 1e-7
    assert len(result.accepted_indices) == 31
    assert len(result.geometry_px["endpoints"]) == 2
    json.dumps(result.to_dict(), allow_nan=False)


def test_line_robust_rejects_distant_outliers_without_changing_input() -> None:
    rng = np.random.default_rng(43)
    x = np.linspace(0, 100, 80)
    clean = np.column_stack((x, x * 0.3 + 7 + rng.normal(0, 0.06, len(x))))
    points = np.vstack((clean, [[15, 40], [65, -30], [25, 60], [0, -55]]))
    before = points.copy()
    result = fit_line_points(points)
    assert result.success
    assert set(range(80, 84)) <= set(result.rejected_indices)
    assert result.rms < 0.1
    direction = result.geometry["direction"]
    assert direction[1] / direction[0] == pytest.approx(0.3, abs=0.003)
    np.testing.assert_array_equal(points, before)


@pytest.mark.parametrize("points", [[], [(1, 1)], [(1, 1), (1, 1)], [(0, 0), (math.nan, 1)]])
def test_line_invalid_inputs_are_not_zero_success(points) -> None:
    result = fit_line_points(points)
    assert not result.success
    assert result.status == "invalid_input"
    assert result.reason
    assert not result.geometry


def test_circle_exact_anisotropic_image_is_physical_circle() -> None:
    angles = np.linspace(0, 2 * math.pi, 100, endpoint=False)
    physical = np.column_stack((40 + 12 * np.cos(angles), 90 + 12 * np.sin(angles)))
    pixels = physical / (2.0, 0.5)
    result = fit_circle_points(pixels, scale=(2, 0.5))
    assert result.success, result.reason
    assert result.geometry["center"] == pytest.approx((40, 90), abs=1e-10)
    assert result.geometry["radius"] == pytest.approx(12, abs=1e-10)
    assert result.geometry_px["center"] == pytest.approx((20, 180))
    assert result.geometry_px["radii"] == pytest.approx((6, 24))
    assert result.rms < 1e-10
    assert result.coverage_deg == pytest.approx(356.4)


def test_circle_robust_with_noise_and_outliers() -> None:
    rng = np.random.default_rng(77)
    angles = np.linspace(0, 2 * math.pi, 80, endpoint=False)
    radii = 30 + rng.normal(0, 0.08, 80)
    points = np.column_stack((100 + radii * np.cos(angles), 80 + radii * np.sin(angles)))
    points = np.vstack((points, [[20, 20], [150, 180], [100, 80], [170, 90]]))
    result = fit_circle_points(points)
    assert result.success, result.reason
    assert result.geometry["center"] == pytest.approx((100, 80), abs=0.1)
    assert result.geometry["radius"] == pytest.approx(30, abs=0.1)
    assert set(range(80, 84)) <= set(result.rejected_indices)
    assert result.rms < 0.1


def test_three_point_circle_and_collinear_rejection() -> None:
    result = fit_circle_points([(1, 0), (0, 1), (-1, 0)])
    assert result.success
    assert result.geometry["radius"] == pytest.approx(1)
    invalid = fit_circle_points([(0, 0), (1, 1), (2, 2)])
    assert not invalid.success
    assert "共线" in invalid.reason


@pytest.mark.parametrize("method", ["gradient", "threshold"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_subpixel_horizontal_edge_and_integer_bit_depth(method, dtype) -> None:
    y, _x = np.mgrid[:128, :180]
    image = _edge(y, 62.35)
    if dtype != np.float32:
        image = np.rint(image * np.iinfo(dtype).max).astype(dtype)
    original = image.copy()
    result = run_line_caliper(image, (12, 62), (165, 62), options=CaliperOptions(method=method, polarity="rising"))
    assert result.success, result.reason
    assert result.fit.geometry_px["point"][1] == pytest.approx(62.35, abs=0.15)
    assert result.valid_fraction == 1
    assert all(profile.candidates and profile.search_polygon_px for profile in result.profiles)
    np.testing.assert_array_equal(image, original)
    json.dumps(result.to_dict(), allow_nan=False)


def test_diagonal_caliper_and_anisotropic_physical_normal() -> None:
    y, x = np.mgrid[:180, :220]
    image = _edge(y - 0.5 * x, 35.2)
    result = run_line_caliper(image, (35, 52.5), (175, 122.5), scale=(2, 0.5))
    assert result.success, result.reason
    axis = np.asarray(result.fit.geometry["direction"])
    assert axis[1] / axis[0] == pytest.approx(0.125, abs=0.002)
    assert result.fit.rms < 0.1
    for p in result.profiles:
        polygon = np.asarray(p.search_polygon_px)
        search_axis_physical = (polygon[1] - polygon[0]) * (2, 0.5)
        assert abs(float(np.dot(search_axis_physical, axis))) < 0.1


def test_polarity_and_reverse_search_are_explicit() -> None:
    y, _x = np.mgrid[:100, :120]
    image = _edge(y, 50)
    bad = run_line_caliper(image, (10, 50), (110, 50), options=CaliperOptions(polarity="falling"))
    assert not bad.success
    assert all(p.candidates for p in bad.profiles)  # rejected candidates are still reviewable
    reversed_result = run_line_caliper(image, (10, 50), (110, 50), options=CaliperOptions(polarity="falling", reverse_search=True))
    assert reversed_result.success
    assert reversed_result.fit.geometry_px["point"][1] == pytest.approx(50, abs=0.1)


def test_multi_edge_selects_nearest_expected_contour() -> None:
    y, _x = np.mgrid[:140, :180]
    image = np.clip(_edge(y, 48) - _edge(y, 60) + _edge(y, 74), 0, 1)
    result = run_line_caliper(image, (15, 61), (165, 61), options=CaliperOptions(search_half_width_px=20, polarity="falling"))
    assert result.success
    assert result.fit.geometry_px["point"][1] == pytest.approx(60, abs=0.15)
    assert len(result.profiles[10].candidates) >= 3


def test_threshold_percentage_is_explicit_and_differs_from_gradient_peak() -> None:
    y, _x = np.mgrid[:100, :140]
    image = np.clip((y - 40.0) / 20.0, 0, 1).astype(np.float32)
    values = []
    for percent in [25, 50, 75]:
        result = run_line_caliper(image, (15, 50), (125, 50), options=CaliperOptions(method="threshold", search_half_width_px=18, threshold_percent=percent))
        assert result.success
        values.append(result.fit.geometry_px["point"][1])
    assert values == pytest.approx([45, 50, 55], abs=0.1)


def test_double_edge_sections_preserve_individual_values_and_physical_units() -> None:
    y, x = np.mgrid[:150, :200]
    first = np.full_like(x, 58.3, dtype=float)
    second = 80.7 + x * 0.02
    image = np.clip(_edge(y - first, 0) - _edge(y - second, 0), 0, 1)
    result = run_line_caliper(image, (20, 70), (180, 70), double_edge=True, scale=(0.5, 2), options=CaliperOptions(search_half_width_px=25, method="threshold"))
    assert result.success, result.reason
    assert len(result.sections) == 32
    for section in result.sections:
        expected = (80.7 + section.first_px[0] * 0.02 - 58.3) * 2
        assert section.width == pytest.approx(expected, abs=0.3)
    assert result.statistics["mean"] == pytest.approx(48.8, abs=0.2)
    assert result.statistics["stddev"] > 0
    assert result.statistics["maximum"] > result.statistics["minimum"]


def test_double_edge_refuses_same_polarity_or_pair_not_straddling_baseline() -> None:
    y, _x = np.mgrid[:130, :160]
    image = 0.5 * _edge(y, 55) + 0.5 * _edge(y, 75)
    result = run_line_caliper(image, (15, 65), (145, 65), double_edge=True, options=CaliperOptions(search_half_width_px=25))
    assert not result.success
    assert not result.sections


@pytest.mark.parametrize("method", ["gradient", "threshold"])
def test_circle_caliper_fits_radial_edges(method) -> None:
    y, x = np.mgrid[:200, :220]
    image = _edge(np.hypot(x - 108.3, y - 94.7), 45.4)
    result = run_circle_caliper(image, (108, 95), 45, options=CaliperOptions(method=method, polarity="rising", sample_count=64))
    assert result.success, result.reason
    assert result.fit.geometry["center"] == pytest.approx((108.3, 94.7), abs=0.2)
    assert result.fit.geometry["radius"] == pytest.approx(45.4, abs=0.2)
    # Coverage excludes the largest gap between *accepted* edge points. Raster
    # interpolation can make a few axial points robust outliers; that missing
    # support must not be reported as a complete 360-degree ring.
    accepted = np.asarray(result.fit.accepted_points_px)
    center = np.asarray(result.fit.geometry_px["center"])
    angles = np.sort(np.degrees(np.arctan2(*(accepted - center).T[::-1])) % 360)
    expected = 360 - np.diff(np.r_[angles, angles[0] + 360]).max()
    assert result.fit.coverage_deg == pytest.approx(expected)
    assert result.fit.coverage_deg > 340


def test_circle_caliper_anisotropic_pixels_fit_in_physical_plane() -> None:
    y, x = np.mgrid[:220, :260]
    image = _edge(np.hypot((x - 130) * 1.5, (y - 110) * 0.75), 54)
    result = run_circle_caliper(image, (130, 110), 36, radius_y_px=72, scale=(1.5, 0.75), options=CaliperOptions(sample_count=64))
    assert result.success, result.reason
    assert result.fit.geometry["radius"] == pytest.approx(54, abs=0.2)
    assert result.fit.geometry_px["radii"] == pytest.approx((36, 72), abs=0.3)


def test_partial_circle_records_insufficient_angular_support() -> None:
    y, x = np.mgrid[:160, :180]
    image = _edge(np.hypot(x - 90, y - 80), 40)
    result = run_circle_caliper(image, (90, 80), 40, sweep_angle_deg=45)
    assert result.status == "insufficient_coverage"
    assert result.fit is not None
    assert result.fit.coverage_deg < 50


def test_noisy_ring_with_a_missing_sector_keeps_missing_evidence() -> None:
    y, x = np.mgrid[:220, :220]
    radius = np.hypot(x - 110, y - 110)
    angle = np.arctan2(y - 110, x - 110)
    image = _edge(radius, 52.3)
    image[np.abs(angle) < 0.45] = 0.5
    image = np.clip(image + np.random.default_rng(28).normal(0, 0.004, image.shape), 0, 1).astype(np.float32)
    result = run_circle_caliper(image, (110, 110), 52, options=CaliperOptions(sample_count=96))
    assert result.success, result.reason
    assert result.fit.geometry["radius"] == pytest.approx(52.3, abs=0.2)
    missing = [p for p in result.profiles if p.status == "missing_edge"]
    assert len(missing) >= 8
    assert result.fit.coverage_deg < 320
    assert result.valid_fraction < 0.95
    # Raising the evidence threshold must fail explicitly, with no fallback.
    stricter = run_circle_caliper(image, (110, 110), 52, options=CaliperOptions(sample_count=96, min_valid_fraction=0.98))
    assert stricter.status == "low_valid_fraction"
    assert stricter.fit is None
    assert stricter.statistics["valid_fraction"] < 0.98


def test_non_circular_edge_rejects_a_plausible_but_poor_circle_fit() -> None:
    y, x = np.mgrid[:220, :220]
    angle = np.arctan2(y - 110, x - 110)
    image = _edge(np.hypot(x - 110, y - 110), 50 + 4 * np.sin(3 * angle))
    result = run_circle_caliper(image, (110, 110), 50, options=CaliperOptions(sample_count=64, max_residual_px=0.5))
    assert result.status == "poor_fit"
    assert not result.success
    assert result.fit is not None  # retained for review, never reported valid
    assert result.fit.max_residual > 3
    assert result.reason


def test_crop_origin_preserves_global_coordinates() -> None:
    y, _x = np.mgrid[:100, :140]
    image = _edge(y, 45.3)
    result = run_line_caliper(image, (1015, 2045), (1125, 2045), origin_px=(1000, 2000))
    assert result.success
    assert result.fit.geometry_px["point"][1] == pytest.approx(2045.3, abs=0.15)
    assert result.profiles[0].center_px == (1015, 2045)


def test_full_profile_bounds_are_not_silently_clipped() -> None:
    y, _x = np.mgrid[:100, :140]
    image = _edge(y, 3)
    result = run_line_caliper(image, (15, 3), (125, 3))
    assert not result.success
    assert all(p.status == "out_of_bounds" for p in result.profiles)
    assert result.fit is None


def test_weak_texture_and_constant_images_do_not_get_valid_geometry() -> None:
    image = np.full((100, 140), 127, dtype=np.uint8)
    result = run_line_caliper(image, (15, 50), (125, 50))
    assert not result.success
    assert all(not p.candidates for p in result.profiles)
    assert result.fit is None
    noisy = image.astype(np.float32) / 255 + np.random.default_rng(0).normal(0, .0002, image.shape)
    result = run_line_caliper(noisy.astype(np.float32), (15, 50), (125, 50))
    assert not result.success


def test_rgb_bgr_agree_and_original_alpha_is_not_measured() -> None:
    y, _x = np.mgrid[:100, :140]
    red = np.rint(_edge(y, 50) * 255).astype(np.uint8)
    rgb = np.stack((red, np.zeros_like(red), np.zeros_like(red)), axis=2)
    first = run_line_caliper(rgb, (15, 50), (125, 50), channel_order="RGB")
    second = run_line_caliper(rgb[..., ::-1], (15, 50), (125, 50), channel_order="BGR")
    assert first.success and second.success
    assert first.fit.geometry == second.fit.geometry
    rgba = np.dstack((rgb, np.zeros_like(red)))
    third = run_line_caliper(rgba, (15, 50), (125, 50), channel_order="RGBA")
    assert not third.success
    assert all(p.status == "invalid_pixels" for p in third.profiles)


@pytest.mark.parametrize("scale", [(0, 1), (-1, 1), (1, math.nan)])
def test_invalid_calibration_is_explicit(scale) -> None:
    assert not fit_line_points([(0, 0), (1, 1)], scale=scale).success
    assert not fit_circle_points([(0, 1), (1, 0), (-1, 0)], scale=scale).success


def test_cancellation_preserves_partial_profiles() -> None:
    y, _x = np.mgrid[:100, :140]
    calls = 0
    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls > 8
    result = run_line_caliper(_edge(y, 50), (15, 50), (125, 50), scale=(0.5, 2), cancel=cancelled)
    assert result.status == "cancelled"
    assert 0 < len(result.profiles) < 32
    assert result.scale == (0.5, 2)
    assert fit_line_points([(0, 0), (1, 1)], cancel=lambda: True).status == "cancelled"
    assert fit_circle_points([(0, 1), (1, 0), (-1, 0)], cancel=lambda: True).status == "cancelled"


@pytest.mark.parametrize("changes", [
    {"method": "unknown"}, {"polarity": "guess"}, {"sample_step_px": 0},
    {"sample_count": 3.5}, {"sample_count": 5000}, {"threshold_percent": 100},
    {"search_half_width_px": math.nan}, {"min_valid_fraction": 2},
    {"reverse_search": "false"}, {"robust": 0},
])
def test_options_validate_before_worker_starts(changes) -> None:
    with pytest.raises(ValueError):
        CaliperOptions(**changes)


def test_config_round_trip_and_float_input_contract() -> None:
    options = CaliperOptions(method="threshold", threshold_percent=35)
    assert CaliperOptions.from_dict(options.to_dict()) == options
    with pytest.raises(ValueError):
        CaliperOptions.from_dict({"mystery": 1})
    bad = run_line_caliper(np.full((100, 140), 255, dtype=np.float32), (15, 50), (125, 50))
    assert bad.status == "invalid_input"
    assert "[0,1]" in bad.reason
