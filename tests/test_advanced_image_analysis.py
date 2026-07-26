from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from fdm.cancellation import CancellationError, CancellationTokenSource
from fdm.services.advanced_image_analysis import (
    AdvancedAnalysisError,
    AdvancedAnalysisErrorCode,
    AdvancedAnalysisKind,
    AdvancedAnalysisLimits,
    DirectionalityRequest,
    GlcmHaralickRequest,
    IntensitySurfaceRequest,
    LocalThicknessRequest,
    SkeletonNetworkRequest,
    SpatialDistributionRequest,
    TubenessRequest,
    analyze_fiber_directionality,
    analyze_skeleton_network,
    analyze_spatial_distribution,
    build_intensity_surface,
    calculate_glcm_haralick,
    calculate_local_thickness,
    calculate_multiscale_tubeness,
    estimate_advanced_analysis_resources,
)


def _axial_distance(first: float, second: float) -> float:
    difference = abs((first - second) % 180.0)
    return min(difference, 180.0 - difference)


class DirectionalityAnalysisTests(unittest.TestCase):
    def test_horizontal_fiber_reports_zero_degree_axis(self) -> None:
        image = np.zeros((96, 128), dtype=np.uint8)
        image[43:53, 10:118] = 255

        result = analyze_fiber_directionality(
            DirectionalityRequest(
                image=image,
                bins=90,
                gradient_sigma=1.0,
                histogram_smoothing_bins=1.0,
                request_id="direction",
                generation=4,
            )
        )

        self.assertGreater(result.total_weight, 0)
        self.assertGreater(result.valid_gradient_pixels, 0)
        self.assertLess(_axial_distance(result.peaks[0].angle_degrees, 0.0), 3.0)
        self.assertEqual(result.request_id, "direction")
        self.assertEqual(result.generation, 4)
        self.assertAlmostEqual(sum(result.normalized_weights), 1.0, places=12)

    def test_two_fiber_families_are_returned_as_multiple_peaks(self) -> None:
        image = np.zeros((160, 160), dtype=np.uint8)
        image[25:33, 10:145] = 255
        image[70:150, 105:113] = 255

        result = analyze_fiber_directionality(
            DirectionalityRequest(
                image=image,
                bins=90,
                peak_min_fraction=0.15,
                max_peaks=6,
            )
        )

        angles = tuple(peak.angle_degrees for peak in result.peaks)
        self.assertTrue(any(_axial_distance(angle, 0.0) < 4.0 for angle in angles))
        self.assertTrue(any(_axial_distance(angle, 90.0) < 4.0 for angle in angles))
        self.assertGreaterEqual(len(result.peaks), 2)

    def test_non_finite_directionality_input_has_structured_chinese_error(self) -> None:
        image = np.zeros((4, 4), dtype=np.float32)
        image[1, 1] = np.nan

        with self.assertRaises(AdvancedAnalysisError) as raised:
            DirectionalityRequest(image=image)

        self.assertEqual(
            raised.exception.code,
            AdvancedAnalysisErrorCode.NON_FINITE_INPUT,
        )
        self.assertIn("NaN", raised.exception.message)


class SkeletonNetworkAnalysisTests(unittest.TestCase):
    def test_t_network_counts_nodes_lengths_and_geodesic_distance(self) -> None:
        mask = np.zeros((15, 15), dtype=bool)
        mask[7, 2:13] = True
        mask[3:8, 7] = True

        result = analyze_skeleton_network(
            SkeletonNetworkRequest(
                mask=mask,
                already_skeletonized=True,
                pixel_size_x=1.0,
                pixel_size_y=1.0,
            )
        )

        self.assertEqual(result.endpoint_count, 3)
        self.assertEqual(result.branchpoint_count, 1)
        self.assertEqual(result.loop_count, 0)
        self.assertEqual(result.connected_component_count, 1)
        self.assertEqual(len(result.branches), 3)
        self.assertAlmostEqual(result.total_length, 14.0)
        self.assertAlmostEqual(result.maximum_geodesic_distance, 10.0)
        self.assertEqual(
            sorted(round(branch.length, 6) for branch in result.branches),
            [4.0, 5.0, 5.0],
        )
        self.assertFalse(result.skeleton.flags.writeable)

    def test_closed_ring_reports_loop_without_false_endpoints(self) -> None:
        mask = np.zeros((11, 11), dtype=np.uint8)
        cv2.rectangle(mask, (2, 2), (8, 8), 1, thickness=1)

        result = analyze_skeleton_network(
            SkeletonNetworkRequest(mask=mask, already_skeletonized=True)
        )

        self.assertEqual(result.endpoint_count, 0)
        self.assertEqual(result.branchpoint_count, 0)
        self.assertEqual(result.loop_count, 1)
        self.assertEqual(len(result.branches), 1)
        self.assertTrue(result.branches[0].closed)
        self.assertAlmostEqual(result.total_length, 24.0)
        self.assertAlmostEqual(result.maximum_geodesic_distance, 12.0)

    def test_thick_bar_is_skeletonized_before_network_analysis(self) -> None:
        mask = np.zeros((25, 40), dtype=bool)
        mask[9:16, 5:35] = True

        result = analyze_skeleton_network(
            SkeletonNetworkRequest(mask=mask, already_skeletonized=False)
        )

        self.assertEqual(result.connected_component_count, 1)
        self.assertEqual(result.endpoint_count, 2)
        self.assertLess(np.count_nonzero(result.skeleton), np.count_nonzero(mask))


class LocalThicknessAnalysisTests(unittest.TestCase):
    def test_maximal_circle_is_propagated_instead_of_returning_two_times_edt(self) -> None:
        mask = np.zeros((11, 21), dtype=bool)
        mask[3:8, 2:19] = True

        result = calculate_local_thickness(LocalThicknessRequest(mask=mask))
        padded = np.pad(mask.astype(np.uint8), 1)
        edt = cv2.distanceTransform(
            padded,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )[1:-1, 1:-1]

        # 条带边缘点自身的 2×EDT 为 2，但它被中轴最大圆覆盖，局部厚度为 6。
        self.assertAlmostEqual(float(edt[3, 10] * 2.0), 2.0)
        self.assertAlmostEqual(float(result.thickness_px[3, 10]), 6.0)
        self.assertTrue(
            np.allclose(
                result.thickness_px[mask],
                result.maximum_thickness_px,
            )
        )
        self.assertGreater(len(result.maximal_circles), 0)
        self.assertFalse(result.thickness_px.flags.writeable)

    def test_empty_mask_returns_empty_finite_map(self) -> None:
        result = calculate_local_thickness(
            LocalThicknessRequest(mask=np.zeros((6, 8), dtype=bool))
        )

        self.assertEqual(result.foreground_pixel_count, 0)
        self.assertEqual(result.maximal_circles, ())
        self.assertEqual(result.maximum_thickness_px, 0.0)
        self.assertIsNone(result.mean_thickness_px)
        self.assertTrue(np.all(result.thickness_px == 0))

    def test_dynamic_local_thickness_work_limit_is_structured(self) -> None:
        mask = np.ones((15, 15), dtype=bool)
        limits = AdvancedAnalysisLimits(
            max_working_bytes=1 << 30,
            max_work_units=1_000_000,
            max_local_thickness_work_units=10,
            max_output_values=1_000_000,
            max_skeleton_pixels=1_000_000,
            max_local_thickness_centers=1_000_000,
        )

        with self.assertRaises(AdvancedAnalysisError) as raised:
            calculate_local_thickness(
                LocalThicknessRequest(mask=mask),
                limits=limits,
            )

        self.assertEqual(
            raised.exception.code,
            AdvancedAnalysisErrorCode.RESOURCE_LIMIT,
        )
        self.assertIn("局部厚度", raised.exception.message)


class TubenessAnalysisTests(unittest.TestCase):
    def test_bright_ridge_response_is_stronger_than_background(self) -> None:
        image = np.zeros((101, 101), dtype=np.float32)
        image[:, 47:54] = 1.0

        result = calculate_multiscale_tubeness(
            TubenessRequest(
                image=image,
                scales=(1.0, 2.0, 3.0),
                bright_ridges=True,
                request_id="tube",
                generation=7,
            )
        )

        self.assertGreater(float(result.response[50, 50]), 0.5)
        self.assertLess(float(result.response[10, 10]), 1e-6)
        self.assertIn(float(result.best_scale[50, 50]), result.scales)
        self.assertEqual(result.request_id, "tube")
        self.assertEqual(result.generation, 7)
        self.assertFalse(result.response.flags.writeable)
        self.assertFalse(result.best_scale.flags.writeable)

    def test_dark_ridge_polarity_is_explicit(self) -> None:
        image = np.ones((81, 81), dtype=np.float32)
        image[:, 38:43] = 0.0

        dark = calculate_multiscale_tubeness(
            TubenessRequest(image=image, scales=(1.0, 2.0), bright_ridges=False)
        )
        bright = calculate_multiscale_tubeness(
            TubenessRequest(image=image, scales=(1.0, 2.0), bright_ridges=True)
        )

        self.assertGreater(float(dark.response[40, 40]), 0.5)
        self.assertLess(float(bright.response[40, 40]), 1e-6)


class GlcmHaralickAnalysisTests(unittest.TestCase):
    def test_checkerboard_horizontal_glcm_has_known_features(self) -> None:
        checkerboard = (np.indices((8, 8)).sum(axis=0) % 2).astype(np.uint8)

        result = calculate_glcm_haralick(
            GlcmHaralickRequest(
                image=checkerboard,
                levels=2,
                distances=(1,),
                directions_degrees=(0.0,),
                symmetric=True,
            )
        )
        feature = result.features[0]

        np.testing.assert_allclose(
            feature.matrix,
            np.asarray([[0.0, 0.5], [0.5, 0.0]]),
            atol=1e-12,
        )
        self.assertAlmostEqual(feature.contrast, 1.0)
        self.assertAlmostEqual(feature.dissimilarity, 1.0)
        self.assertAlmostEqual(feature.homogeneity, 0.5)
        self.assertAlmostEqual(feature.angular_second_moment, 0.5)
        self.assertAlmostEqual(feature.energy, math.sqrt(0.5))
        self.assertAlmostEqual(feature.correlation or 0.0, -1.0)
        self.assertAlmostEqual(feature.entropy, math.log(2.0))
        self.assertFalse(feature.matrix.flags.writeable)

    def test_glcm_records_quantization_distance_direction_roi_and_nonfinite(self) -> None:
        image = np.arange(36, dtype=np.float32).reshape(6, 6)
        image[0, 0] = np.nan
        roi = np.zeros((6, 6), dtype=bool)
        roi[:4, :] = True

        result = calculate_glcm_haralick(
            GlcmHaralickRequest(
                image=image,
                roi_mask=roi,
                levels=4,
                distances=(1, 2),
                directions_degrees=(0.0, 90.0),
                value_range=(0.0, 36.0),
            )
        )

        self.assertEqual(len(result.features), 4)
        self.assertEqual(result.non_finite_pixel_count, 1)
        self.assertEqual(result.valid_pixel_count, 23)
        self.assertEqual(result.quantization_range, (0.0, 36.0))
        self.assertEqual(
            {(item.distance_px, item.direction_degrees) for item in result.features},
            {(1, 0.0), (1, 90.0), (2, 0.0), (2, 90.0)},
        )

    def test_duplicate_rounded_offsets_are_rejected(self) -> None:
        with self.assertRaises(AdvancedAnalysisError) as raised:
            GlcmHaralickRequest(
                image=np.zeros((8, 8), dtype=np.uint8),
                directions_degrees=(0.0, 180.0),
            )

        self.assertEqual(
            raised.exception.code,
            AdvancedAnalysisErrorCode.INVALID_INPUT,
        )
        self.assertIn("重复偏移", raised.exception.message)


class SpatialAndSurfaceAnalysisTests(unittest.TestCase):
    def test_square_points_have_exact_nearest_neighbor_and_density(self) -> None:
        result = analyze_spatial_distribution(
            SpatialDistributionRequest(
                points=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
                pixel_size_x=2.0,
                pixel_size_y=3.0,
                study_area=24.0,
                unit="µm",
            )
        )

        self.assertEqual(result.nearest_neighbor_distances, (2.0, 2.0, 2.0, 2.0))
        self.assertAlmostEqual(result.mean_nearest_neighbor_distance, 2.0)
        self.assertAlmostEqual(result.spatial_density, 4.0 / 24.0)
        self.assertEqual(result.area_source, "用户指定")
        self.assertEqual(result.unit, "µm")

    def test_collinear_points_require_explicit_study_area(self) -> None:
        with self.assertRaises(AdvancedAnalysisError) as raised:
            analyze_spatial_distribution(
                SpatialDistributionRequest(
                    points=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
                )
            )

        self.assertEqual(
            raised.exception.code,
            AdvancedAnalysisErrorCode.INVALID_INPUT,
        )
        self.assertIn("研究区域面积", raised.exception.message)

    def test_surface_uses_explicit_sampling_and_serializable_missing_values(self) -> None:
        image = np.arange(30, dtype=np.float32).reshape(5, 6)
        image[2, 2] = np.nan
        roi = np.ones((5, 6), dtype=bool)
        roi[0, 0] = False

        result = build_intensity_surface(
            IntensitySurfaceRequest(
                image=image,
                roi_mask=roi,
                sample_step_x=2,
                sample_step_y=2,
                pixel_size_x=0.5,
                pixel_size_y=2.0,
                unit="µm",
            )
        )

        self.assertEqual(result.x_coordinates, (0.0, 1.0, 2.0))
        self.assertEqual(result.y_coordinates, (0.0, 4.0, 8.0))
        self.assertIsNone(result.z_values[0][0])
        self.assertIsNone(result.z_values[1][1])
        self.assertEqual(result.masked_sample_count, 1)
        self.assertEqual(result.non_finite_sample_count, 1)
        self.assertEqual(result.finite_sample_count, 7)
        self.assertTrue(
            all(
                value is None or math.isfinite(value)
                for row in result.z_values
                for value in row
            )
        )


class ResourceCancellationAndImmutabilityTests(unittest.TestCase):
    def test_all_request_kinds_have_resource_estimates(self) -> None:
        requests = (
            DirectionalityRequest(np.zeros((8, 8), dtype=np.uint8)),
            SkeletonNetworkRequest(np.zeros((8, 8), dtype=bool)),
            LocalThicknessRequest(np.zeros((8, 8), dtype=bool)),
            TubenessRequest(np.zeros((8, 8), dtype=np.uint8)),
            GlcmHaralickRequest(np.zeros((8, 8), dtype=np.uint8)),
            SpatialDistributionRequest(
                ((0.0, 0.0), (1.0, 1.0)),
                study_area=1.0,
            ),
            IntensitySurfaceRequest(np.zeros((8, 8), dtype=np.uint8)),
        )

        estimates = tuple(estimate_advanced_analysis_resources(item) for item in requests)

        self.assertEqual(
            {estimate.operation for estimate in estimates},
            set(AdvancedAnalysisKind),
        )
        self.assertTrue(all(estimate.allowed for estimate in estimates))
        self.assertTrue(all(estimate.estimated_peak_bytes > 0 for estimate in estimates))

    def test_resource_limit_rejects_before_execution(self) -> None:
        request = IntensitySurfaceRequest(np.zeros((100, 100), dtype=np.uint8))
        limits = AdvancedAnalysisLimits(
            max_working_bytes=1024,
            max_work_units=1_000_000,
            max_local_thickness_work_units=1_000_000,
            max_output_values=1_000_000,
            max_skeleton_pixels=1_000_000,
            max_local_thickness_centers=1_000_000,
        )

        estimate = estimate_advanced_analysis_resources(request, limits=limits)
        self.assertFalse(estimate.allowed)
        with self.assertRaises(AdvancedAnalysisError) as raised:
            build_intensity_surface(request, limits=limits)
        self.assertEqual(
            raised.exception.code,
            AdvancedAnalysisErrorCode.RESOURCE_LIMIT,
        )

    def test_pre_cancelled_request_does_not_run(self) -> None:
        source = CancellationTokenSource()
        source.cancel()

        with self.assertRaises(CancellationError):
            calculate_multiscale_tubeness(
                TubenessRequest(
                    image=np.zeros((64, 64), dtype=np.uint8),
                    scales=(1.0, 2.0),
                ),
                cancellation_token=source.token,
            )

    def test_requests_are_detached_from_caller_arrays(self) -> None:
        image = np.zeros((16, 16), dtype=np.uint8)
        image[7:9, :] = 255
        request = DirectionalityRequest(image=image, bins=18)
        image[:, :] = 0

        result = analyze_fiber_directionality(request)

        self.assertGreater(result.total_weight, 0)
        self.assertFalse(request.image.flags.writeable)

    def test_non_finite_point_parameter_is_rejected_structurally(self) -> None:
        with self.assertRaises(AdvancedAnalysisError) as raised:
            SpatialDistributionRequest(
                points=((0.0, 0.0), (math.inf, 1.0)),
                study_area=1.0,
            )

        self.assertEqual(
            raised.exception.code,
            AdvancedAnalysisErrorCode.NON_FINITE_INPUT,
        )


if __name__ == "__main__":
    unittest.main()
