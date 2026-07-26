from __future__ import annotations

import math
import unittest

import numpy as np

from fdm.services.image_analysis import (
    FindMaximaRequest,
    HistogramRequest,
    IntensityAnalysisRequest,
    IntensityProfileRequest,
    ParticleAnalysisRequest,
    ShapeAnalysisRequest,
    analyze_intensity,
    analyze_particles,
    analyze_shape,
    calculate_histogram,
    find_local_maxima,
    sample_intensity_profile,
)


class ImageAnalysisServiceTests(unittest.TestCase):
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
