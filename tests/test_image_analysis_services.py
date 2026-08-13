from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from fdm.services.image_analysis import (
    FftPowerSpectrumRequest,
    FindMaximaRequest,
    HistogramRequest,
    IntensityAnalysisRequest,
    IntensityProfileRequest,
    ParticleAnalysisRequest,
    ShapeAnalysisRequest,
    analyze_intensity,
    analyze_particles,
    analyze_shape,
    calculate_fft_power_spectrum,
    calculate_histogram,
    find_local_maxima,
    sample_intensity_profile,
)
from fdm.services.image_processing import fft_power_spectrum


class ImageAnalysisServiceTests(unittest.TestCase):
    def test_fft_analysis_whole_image_matches_legacy_v1_kernel(self) -> None:
        image = np.arange(48, dtype=np.uint16).reshape(6, 8)

        result = calculate_fft_power_spectrum(
            FftPowerSpectrumRequest(
                image=image,
                logarithmic=True,
                centered=True,
                window="tukey",
                tukey_alpha=0.4,
                request_id="fft-1",
                generation=3,
            )
        )

        expected = fft_power_spectrum(
            image,
            logarithmic=True,
            centered=True,
            window="tukey",
            tukey_alpha=0.4,
        )
        np.testing.assert_array_equal(result.power, expected)
        self.assertEqual(result.source_size, (8, 6))
        self.assertEqual(result.analysis_bounds, (0, 0, 8, 6))
        self.assertEqual(result.mask_policy, "full_image")
        self.assertFalse(result.roi_applied)
        self.assertFalse(result.power.flags.writeable)
        self.assertEqual(result.request_id, "fft-1")
        self.assertEqual(result.generation, 3)

    def test_fft_analysis_freezes_exact_roi_policy_and_rejects_nonfinite(self) -> None:
        image = np.arange(36, dtype=np.float32).reshape(6, 6)
        mask = np.zeros((6, 6), dtype=bool)
        mask[1:5, 2:5] = True
        mask[2, 3] = False

        result = calculate_fft_power_spectrum(
            FftPowerSpectrumRequest(image=image, roi_mask=mask)
        )

        expected_input = np.where(mask[1:5, 2:5], image[1:5, 2:5], 0.0)
        np.testing.assert_array_equal(
            result.power,
            fft_power_spectrum(expected_input),
        )
        self.assertEqual(result.analysis_bounds, (2, 1, 3, 4))
        self.assertEqual(
            result.mask_policy,
            "tight_bounds_zero_outside_exact_mask",
        )
        self.assertTrue(result.roi_applied)

        image[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            calculate_fft_power_spectrum(FftPowerSpectrumRequest(image=image))

    def test_shape_uses_exact_area_and_preserves_hole_metrics(self) -> None:
        outer = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
        hole = ((3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0))

        result = analyze_shape(
            ShapeAnalysisRequest(
                rings=(outer, hole),
                exact_area_px=77.0,
                pixel_size_x=0.5,
                pixel_size_y=2.0,
                unit="µm",
            )
        )

        self.assertTrue(result.area_from_exact_mask)
        self.assertEqual(result.area_px, 77.0)
        self.assertAlmostEqual(result.vector_area_px, 84.0)
        self.assertEqual(result.area, 77.0)
        self.assertEqual(result.hole_count, 1)
        self.assertAlmostEqual(result.hole_area_px, 16.0)
        self.assertAlmostEqual(result.outer_perimeter_px, 40.0)
        self.assertAlmostEqual(result.hole_perimeter_px, 16.0)
        self.assertEqual(result.unit, "µm")

    def test_shape_uses_odd_even_topology_for_multiple_components(self) -> None:
        first_outer = (
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        )
        first_hole = (
            (3.0, 3.0),
            (5.0, 3.0),
            (5.0, 5.0),
            (3.0, 5.0),
        )
        second_outer = (
            (20.0, 0.0),
            (24.0, 0.0),
            (24.0, 4.0),
            (20.0, 4.0),
        )

        vector = analyze_shape(
            ShapeAnalysisRequest(
                # Deliberately put a hole first: ring order is not topology.
                rings=(first_hole, second_outer, first_outer),
            )
        )
        exact = analyze_shape(
            ShapeAnalysisRequest(
                rings=(first_hole, second_outer, first_outer),
                exact_area_px=999.0,
            )
        )

        self.assertEqual(vector.vector_area_px, 112.0)
        self.assertEqual(vector.component_count, 2)
        self.assertEqual(vector.hole_count, 1)
        self.assertEqual(vector.euler_number, 1)
        self.assertAlmostEqual(vector.extent, 112.0 / 240.0)
        self.assertEqual(
            sorted(item.area_px for item in vector.component_table),
            [16.0, 96.0],
        )
        self.assertEqual(
            sorted(item.hole_count for item in vector.component_table),
            [0, 1],
        )
        self.assertIsNotNone(vector.solidity)
        self.assertLessEqual(vector.solidity, 1.0)

        # A mask-derived exact area is authoritative only for reported area.
        self.assertEqual(exact.area_px, 999.0)
        self.assertEqual(exact.area, 999.0)
        self.assertEqual(exact.vector_area_px, vector.vector_area_px)
        self.assertEqual(exact.equivalent_circle_diameter, vector.equivalent_circle_diameter)
        self.assertEqual(exact.circularity, vector.circularity)
        self.assertEqual(exact.roundness, vector.roundness)
        self.assertEqual(exact.solidity, vector.solidity)
        self.assertEqual(exact.extent, vector.extent)
        self.assertEqual(exact.component_table, vector.component_table)

    def test_intensity_odd_even_mask_excludes_hole(self) -> None:
        image = np.ones((11, 11), dtype=np.uint8)
        image[4:7, 4:7] = 100
        outer = ((1.0, 1.0), (9.0, 1.0), (9.0, 9.0), (1.0, 9.0))
        hole = ((3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0))

        result = analyze_intensity(
            IntensityAnalysisRequest(
                image=image,
                rings=(outer, hole),
                channel="luminance",
            )
        )

        self.assertEqual(result.non_finite_count, 0)
        self.assertGreater(result.valid_pixel_count, 0)
        self.assertEqual(result.mean, 1.0)
        self.assertEqual(result.minimum, 1.0)
        self.assertEqual(result.maximum, 1.0)

    def test_intensity_counts_non_finite_values_without_serializing_nan(self) -> None:
        image = np.asarray([[1.0, np.nan], [3.0, np.inf]], dtype=np.float32)

        result = analyze_intensity(IntensityAnalysisRequest(image=image))

        self.assertEqual(result.included_pixel_count, 4)
        self.assertEqual(result.valid_pixel_count, 2)
        self.assertEqual(result.non_finite_count, 2)
        self.assertEqual(result.mean, 2.0)
        self.assertTrue(all(math.isfinite(value) for _level, value in result.percentiles))

    def test_intensity_reports_mode_shape_moments_threshold_fraction_and_rgb(self) -> None:
        image = np.asarray(
            [
                [[0, 10, 20], [0, 20, 40]],
                [[10, 30, 60], [20, 40, 80]],
            ],
            dtype=np.uint8,
        )

        result = analyze_intensity(
            IntensityAnalysisRequest(
                image=image,
                channel="rgb",
                threshold_low=10,
                threshold_high=30,
            )
        )

        self.assertEqual([item.channel for item in result.channel_statistics], ["red", "green", "blue"])
        red = result.channel_statistics[0]
        self.assertEqual(red.mode, 0.0)
        self.assertEqual(red.threshold_area_fraction, 0.5)
        self.assertIsNotNone(red.skewness)
        self.assertIsNotNone(red.excess_kurtosis)

    def test_histogram_count_matches_finite_roi_pixels(self) -> None:
        image = np.arange(16, dtype=np.uint8).reshape(4, 4)
        roi = np.zeros((4, 4), dtype=bool)
        roi[:, :2] = True

        result = calculate_histogram(
            HistogramRequest(
                image=image,
                roi_mask=roi,
                bins=4,
                value_range=(0.0, 16.0),
            )
        )

        self.assertEqual(result.included_pixel_count, 8)
        self.assertEqual(sum(result.counts), 8)
        self.assertEqual(result.non_finite_count, 0)

        logarithmic = calculate_histogram(
            HistogramRequest(
                image=image,
                bins=4,
                value_range=(0.0, 16.0),
                log_counts=True,
            )
        )
        np.testing.assert_allclose(
            logarithmic.display_counts,
            np.log1p(logarithmic.counts),
        )

    def test_line_profile_matches_analytic_horizontal_gradient(self) -> None:
        image = np.tile(np.arange(11, dtype=np.float32), (5, 1))

        result = sample_intensity_profile(
            IntensityProfileRequest(
                image=image,
                points=((0.0, 2.0), (10.0, 2.0)),
                sample_spacing=1.0,
                line_width=3.0,
                pixel_size_x=0.5,
                pixel_size_y=1.0,
            )
        )

        self.assertEqual(result.valid_sample_count, 11)
        np.testing.assert_allclose(result.values, np.arange(11), atol=1e-6)
        self.assertAlmostEqual(result.distances[-1], 5.0)

    def test_long_polyline_profile_uses_linear_physical_prefix_lookup(self) -> None:
        image = np.tile(np.arange(400, dtype=np.float32), (3, 1))
        points = tuple((float(index), 1.0) for index in range(400))

        result = sample_intensity_profile(
            IntensityProfileRequest(
                image=image,
                points=points,
                sample_spacing=1.0,
                pixel_size_x=0.25,
                pixel_size_y=1.0,
            )
        )

        self.assertEqual(len(result.distances), 400)
        np.testing.assert_allclose(
            result.distances,
            np.arange(400, dtype=np.float64) * 0.25,
        )

    def test_rectangle_profile_supports_row_and_column_averages(self) -> None:
        image = np.arange(20, dtype=np.float32).reshape(4, 5)

        rows = sample_intensity_profile(
            IntensityProfileRequest(
                image=image,
                points=((1.0, 1.0), (3.0, 3.0)),
                aggregation="rectangle_rows",
                sample_spacing=2.0,
            )
        )
        columns = sample_intensity_profile(
            IntensityProfileRequest(
                image=image,
                points=((1.0, 1.0), (3.0, 3.0)),
                aggregation="rectangle_columns",
            )
        )

        self.assertEqual(rows.values, (7.0, 17.0))
        self.assertEqual(columns.values, (11.0, 12.0, 13.0))
        self.assertEqual(rows.aggregation, "rectangle_rows")

    def test_particle_analysis_preserves_hole_or_includes_it_explicitly(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[3:15, 3:15] = True
        mask[7:11, 7:11] = False

        excluding_hole = analyze_particles(
            ParticleAnalysisRequest(mask=mask, include_holes=False)
        )
        including_hole = analyze_particles(
            ParticleAnalysisRequest(mask=mask, include_holes=True)
        )

        self.assertEqual(excluding_hole.accepted_count, 1)
        self.assertEqual(excluding_hole.particles[0].exact_area_px, 128)
        self.assertEqual(excluding_hole.particles[0].hole_count, 1)
        self.assertEqual(including_hole.particles[0].exact_area_px, 144)
        self.assertEqual(including_hole.particles[0].hole_count, 0)

    def test_particle_filters_size_circularity_and_edges(self) -> None:
        mask = np.zeros((30, 30), dtype=bool)
        mask[0:4, 0:4] = True
        mask[10:20, 10:20] = True
        mask[25:27, 25:27] = True

        result = analyze_particles(
            ParticleAnalysisRequest(
                mask=mask,
                min_area_px=10,
                exclude_edge=True,
            )
        )

        self.assertEqual(result.total_component_count, 3)
        self.assertEqual(result.accepted_count, 1)
        self.assertEqual(result.rejected_edge_count, 1)
        self.assertEqual(result.rejected_by_area_count, 1)
        self.assertEqual(result.particles[0].exact_area_px, 100)
        self.assertAlmostEqual(result.area_fraction, 100 / 900)
        self.assertEqual(result.label_image.dtype, np.int32)
        self.assertFalse(result.label_image.flags.writeable)
        self.assertGreater(np.count_nonzero(result.contour_image), 0)

    def test_particle_watershed_separates_touching_objects_without_losing_area(self) -> None:
        source = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(source, (25, 32), 12, 1, -1)
        cv2.circle(source, (39, 32), 12, 1, -1)
        mask = source.astype(bool)

        connected = analyze_particles(ParticleAnalysisRequest(mask=mask))
        separated = analyze_particles(
            ParticleAnalysisRequest(
                mask=mask,
                watershed=True,
                watershed_min_distance=4,
            )
        )

        self.assertEqual(connected.accepted_count, 1)
        self.assertEqual(separated.accepted_count, 2)
        self.assertEqual(
            separated.accepted_foreground_pixel_count,
            separated.foreground_pixel_count,
        )

    def test_find_maxima_collapses_plateau_and_applies_distance(self) -> None:
        image = np.zeros((15, 15), dtype=np.float32)
        image[3:5, 3:5] = 10.0
        image[10, 10] = 8.0
        image[10, 12] = 7.0

        result = find_local_maxima(
            FindMaximaRequest(
                image=image,
                minimum_value=1.0,
                prominence=1.0,
                neighborhood_radius=1,
                min_distance=3.0,
            )
        )

        self.assertEqual(len(result.maxima), 2)
        self.assertEqual((result.maxima[0].x, result.maxima[0].y), (3.0, 3.0))
        self.assertEqual(result.maxima[0].value, 10.0)
        self.assertEqual(result.suppressed_count, 1)

    def test_topographic_prominence_v2_is_distinct_from_local_v1(self) -> None:
        image = np.zeros((7, 7), dtype=np.float32)
        image[3, 1:6] = (10.0, 7.0, 7.0, 7.0, 9.0)

        local = find_local_maxima(
            FindMaximaRequest(
                image=image,
                prominence=3.0,
                algorithm_version="1",
            )
        )
        topographic = find_local_maxima(
            FindMaximaRequest(
                image=image,
                prominence=3.0,
                algorithm_version="2",
            )
        )

        self.assertEqual(local.algorithm_version, "1")
        self.assertEqual(topographic.algorithm_version, "2")
        self.assertIn(9.0, [item.value for item in local.maxima])
        self.assertNotIn(9.0, [item.value for item in topographic.maxima])

    def test_analysis_requests_are_detached_from_caller_arrays(self) -> None:
        image = np.zeros((4, 4), dtype=np.uint8)
        request = HistogramRequest(image=image, bins=2)
        image[:, :] = 255

        result = calculate_histogram(request)

        self.assertEqual(result.edges[0], -0.5)
        self.assertEqual(sum(result.counts), 16)
        with self.assertRaises(ValueError):
            request.image[0, 0] = 1


if __name__ == "__main__":
    unittest.main()
