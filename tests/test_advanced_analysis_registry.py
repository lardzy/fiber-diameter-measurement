from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from fdm.analysis_artifacts import AnalysisToolSpec
from fdm.cancellation import CancellationError, CancellationTokenSource
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.advanced_image_analysis import AdvancedAnalysisKind
from fdm.services.advanced_analysis_registry import (
    AdvancedAnalysisInvocation,
    AdvancedAnalysisRegistration,
    AdvancedAnalysisRegistry,
)
from fdm.services.raster_io import numpy_to_raster_plane


class AdvancedAnalysisRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = AdvancedAnalysisRegistry()

    def test_all_high_value_analyses_are_registered_with_chinese_names(self) -> None:
        registrations = self.registry.registrations()

        self.assertEqual(
            {item.kind for item in registrations},
            set(AdvancedAnalysisKind),
        )
        self.assertTrue(all(item.chinese_name for item in registrations))
        versions = {item.kind: item.algorithm_version for item in registrations}
        self.assertEqual(versions[AdvancedAnalysisKind.DIRECTIONALITY], "2")
        self.assertEqual(versions[AdvancedAnalysisKind.SKELETON_NETWORK], "2")
        self.assertEqual(versions[AdvancedAnalysisKind.SPATIAL_DISTRIBUTION], "2")
        self.assertTrue(
            all(
                version == "1"
                for kind, version in versions.items()
                if kind
                not in {
                    AdvancedAnalysisKind.DIRECTIONALITY,
                    AdvancedAnalysisKind.SKELETON_NETWORK,
                    AdvancedAnalysisKind.SPATIAL_DISTRIBUTION,
                }
            )
        )
        self.assertEqual(
            {item.tool_spec.tool_id for item in registrations},
            {f"fdm.{kind.value}" for kind in AdvancedAnalysisKind},
        )
        self.assertEqual(
            len({id(item.tool_spec) for item in registrations}),
            len(AdvancedAnalysisKind),
        )
        self.assertTrue(all(item.tool_spec.version == item.algorithm_version for item in registrations))

    def test_registration_uses_one_serializable_tool_contract(self) -> None:
        registration = self.registry.registration(
            AdvancedAnalysisKind.SPATIAL_DISTRIBUTION
        )

        restored = AnalysisToolSpec.from_dict(registration.tool_spec.to_dict())

        self.assertEqual(restored, registration.tool_spec)
        self.assertEqual(
            restored.convertible_kinds,
            ("point_set", "measurement_group"),
        )

    def test_directionality_preserves_request_contract_and_reports_arrays(self) -> None:
        image = np.zeros((48, 64), dtype=np.uint8)
        image[20:28, 5:59] = 255
        invocation = AdvancedAnalysisInvocation(
            AdvancedAnalysisKind.DIRECTIONALITY,
            request_id="advanced-1",
            generation=9,
            plane=numpy_to_raster_plane(image),
            parameters={"bins": 36, "channel": "luminance"},
        )

        execution = self.registry.execute(invocation)

        self.assertEqual(execution.request_id, "advanced-1")
        self.assertEqual(execution.generation, 9)
        self.assertEqual(execution.kind, AdvancedAnalysisKind.DIRECTIONALITY)
        self.assertGreater(execution.result.total_weight, 0.0)
        self.assertIn("valid_gradient_pixels", execution.scalar_report_map)

    def test_rgb_channel_selection_is_explicit(self) -> None:
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb[:, 8:12, 0] = 255
        plane = numpy_to_raster_plane(rgb)

        red = self.registry.execute(
            AdvancedAnalysisInvocation(
                AdvancedAnalysisKind.INTENSITY_SURFACE,
                request_id="red",
                generation=1,
                plane=plane,
                parameters={"channel": "red", "sample_step_x": 5, "sample_step_y": 5},
            )
        )
        blue = self.registry.execute(
            AdvancedAnalysisInvocation(
                AdvancedAnalysisKind.INTENSITY_SURFACE,
                request_id="blue",
                generation=1,
                plane=plane,
                parameters={"channel": "blue", "sample_step_x": 5, "sample_step_y": 5},
            )
        )

        self.assertGreater(red.result.z_maximum, 0.0)
        self.assertEqual(blue.result.z_maximum, 0.0)

    def test_luminance_matches_basic_analysis_rec709_definition(self) -> None:
        rgb = np.asarray([[[0, 255, 0]]], dtype=np.uint8)
        execution = self.registry.execute(
            AdvancedAnalysisInvocation(
                AdvancedAnalysisKind.INTENSITY_SURFACE,
                request_id="luminance",
                generation=1,
                plane=numpy_to_raster_plane(rgb),
                parameters={
                    "channel": "luminance",
                    "sample_step_x": 1,
                    "sample_step_y": 1,
                },
            )
        )

        self.assertAlmostEqual(
            execution.result.z_maximum,
            255.0 * 0.7152,
            places=4,
        )

    def test_binary_algorithms_refuse_implicit_thresholding(self) -> None:
        plane = RasterPlane(
            width=4,
            height=4,
            pixel_type=RasterPixelType.GRAY8,
            data=bytes(16),
        )

        with self.assertRaisesRegex(ValueError, "显式二值掩膜"):
            self.registry.execute(
                AdvancedAnalysisInvocation(
                    AdvancedAnalysisKind.LOCAL_THICKNESS,
                    request_id="missing-mask",
                    generation=0,
                    plane=plane,
                )
            )

    def test_each_builtin_executes_from_the_generic_boundary(self) -> None:
        scalar = np.zeros((20, 20), dtype=np.uint8)
        scalar[7:13, 3:17] = 255
        plane = numpy_to_raster_plane(scalar)
        binary = scalar > 0
        invocations = (
            AdvancedAnalysisInvocation(
                AdvancedAnalysisKind.DIRECTIONALITY,
                request_id="direction",
                generation=2,
                plane=plane,
                parameters={"bins": 18},
            ),
            AdvancedAnalysisInvocation(
                AdvancedAnalysisKind.SKELETON_NETWORK,
                request_id="skeleton",
                generation=2,
                plane=plane,
                binary_mask=binary,
            ),
            AdvancedAnalysisInvocation(
                AdvancedAnalysisKind.LOCAL_THICKNESS,
                request_id="thickness",
                generation=2,
                plane=plane,
                binary_mask=binary,
            ),
            AdvancedAnalysisInvocation(
                AdvancedAnalysisKind.TUBENESS,
                request_id="tube",
                generation=2,
                plane=plane,
                parameters={"scales": [1.0]},
            ),
            AdvancedAnalysisInvocation(
                AdvancedAnalysisKind.GLCM_HARALICK,
                request_id="texture",
                generation=2,
                plane=plane,
                parameters={
                    "levels": 4,
                    "distances": [1],
                    "directions_degrees": [0.0],
                },
            ),
            AdvancedAnalysisInvocation(
                AdvancedAnalysisKind.SPATIAL_DISTRIBUTION,
                request_id="space",
                generation=2,
                points=((0.0, 0.0), (3.0, 4.0), (6.0, 0.0)),
                parameters={
                    "study_bounds": [0.0, 0.0, 6.0, 6.0],
                    "ripley_radii": [1.0, 3.0],
                },
            ),
            AdvancedAnalysisInvocation(
                AdvancedAnalysisKind.INTENSITY_SURFACE,
                request_id="surface",
                generation=2,
                plane=plane,
                parameters={"sample_step_x": 4, "sample_step_y": 4},
            ),
        )

        executions = tuple(self.registry.execute(item) for item in invocations)

        self.assertEqual(
            {item.kind for item in executions},
            set(AdvancedAnalysisKind),
        )
        self.assertTrue(all(item.generation == 2 for item in executions))

    def test_input_masks_are_copied_and_read_only(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        invocation = AdvancedAnalysisInvocation(
            AdvancedAnalysisKind.LOCAL_THICKNESS,
            request_id="immutable",
            generation=0,
            binary_mask=mask,
        )
        mask[:] = False

        self.assertTrue(invocation.binary_mask[1, 1])
        with self.assertRaises(ValueError):
            invocation.binary_mask[1, 1] = False

    def test_cancellation_is_checked_before_executor(self) -> None:
        source = CancellationTokenSource()
        source.cancel()
        invocation = AdvancedAnalysisInvocation(
            AdvancedAnalysisKind.SPATIAL_DISTRIBUTION,
            request_id="cancelled",
            generation=1,
            points=((0.0, 0.0), (1.0, 1.0)),
        )

        with self.assertRaises(CancellationError):
            self.registry.execute(
                invocation,
                cancellation_token=source.token,
            )

    def test_registry_rejects_an_executor_that_breaks_generation_contract(self) -> None:
        self.registry.register(
            AdvancedAnalysisRegistration(
                kind=AdvancedAnalysisKind.SPATIAL_DISTRIBUTION,
                chinese_name="测试替身",
                algorithm_version="test",
                input_description="测试",
                executor=lambda invocation, token, limits: SimpleNamespace(
                    request_id=invocation.request_id,
                    generation=invocation.generation + 1,
                ),
            ),
            replace=True,
        )

        with self.assertRaisesRegex(RuntimeError, "request_id/generation"):
            self.registry.execute(
                AdvancedAnalysisInvocation(
                    AdvancedAnalysisKind.SPATIAL_DISTRIBUTION,
                    request_id="mismatch",
                    generation=3,
                    points=((0.0, 0.0), (1.0, 1.0)),
                )
            )


if __name__ == "__main__":
    unittest.main()
