from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtWidgets import QApplication

from fdm.analysis_artifacts import (
    AnalysisAssetReference,
    AnalysisDependencySignature,
    AnalysisObjectKind,
    AnalysisObjectReference,
    AnalysisRegionSnapshot,
    AnalysisSourceDescriptor,
)
from fdm.cancellation import CancellationTokenSource
from fdm.raster import RasterPixelType, RasterPlane
from fdm.ui.image_analysis_controller import (
    MAX_ANALYSIS_WORKING_BYTES,
    AnalysisCalibrationSnapshot,
    AnalysisTaskPhase,
    AnalysisTool,
    ImageAnalysisTaskController,
    ImageAnalysisTaskRequest,
    ImageAnalysisTaskResult,
    MaximaConversionPayload,
    ParticleConversionPayload,
    estimate_analysis_resources,
    execute_analysis_task,
    rebuild_analysis_conversion_payload,
    rebuild_particle_conversion_payload,
)
from fdm.services.analysis_asset_io import write_safe_analysis_npz
from fdm.services.analysis_profiles import ANALYSIS_OUTPUT_FIELDS_PARAMETER


def _gray_plane(width: int = 16, height: int = 16) -> RasterPlane:
    values = np.arange(width * height, dtype=np.uint8)
    return RasterPlane(
        width=width,
        height=height,
        pixel_type=RasterPixelType.GRAY8,
        data=values.tobytes(),
    )


def _rgb_plane(width: int = 4, height: int = 4) -> RasterPlane:
    values = np.zeros((height, width, 3), dtype=np.uint8)
    values[..., 0] = np.arange(width * height, dtype=np.uint8).reshape(height, width)
    values[..., 1] = 20
    values[..., 2] = 40
    return RasterPlane(
        width=width,
        height=height,
        pixel_type=RasterPixelType.RGB8,
        data=values.tobytes(),
    )


def _execute(request: ImageAnalysisTaskRequest) -> ImageAnalysisTaskResult:
    return execute_analysis_task(
        request,
        CancellationTokenSource().token,
        lambda _phase: None,
    )


def _result_for(request: ImageAnalysisTaskRequest) -> ImageAnalysisTaskResult:
    return ImageAnalysisTaskResult(
        tool=request.tool,
        request_id=request.request_id,
        generation=request.generation,
        document_id=request.document_id,
        source_pixel_revision=request.source_pixel_revision,
        source_reference=request.source_reference,
        calibration_signature=request.calibration.signature,
        parameters=request.parameters,
        scalars={"n": 1},
    )


class AnalysisRequestAndPackagingTests(unittest.TestCase):
    def test_saved_output_field_snapshot_is_not_forwarded_to_kernel_parameters(
        self,
    ) -> None:
        request = ImageAnalysisTaskRequest(
            tool=AnalysisTool.INTENSITY,
            request_id="saved-output-fields",
            generation=1,
            document_id="doc",
            source_pixel_revision=0,
            plane=_gray_plane(),
            parameters={
                "channel": "luminance",
                ANALYSIS_OUTPUT_FIELDS_PARAMETER: [
                    "central_tendency",
                    "percentiles",
                ],
            },
        )

        self.assertEqual(
            request.output_fields,
            ("central_tendency", "percentiles"),
        )
        self.assertEqual(request.parameters, {"channel": "luminance"})
        with self.assertRaisesRegex(ValueError, "不一致"):
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.INTENSITY,
                request_id="conflicting-output-fields",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                parameters={
                    ANALYSIS_OUTPUT_FIELDS_PARAMETER: ["central_tendency"],
                },
                output_fields=("range",),
            )

    def test_shape_and_intensity_outputs_are_projected_after_full_calculation(
        self,
    ) -> None:
        shape = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.SHAPE,
                request_id="shape-selected-fields",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                raw_rings=(
                    ((1, 1), (13, 1), (13, 13), (1, 13)),
                    ((4, 4), (8, 4), (8, 8), (4, 8)),
                ),
                exact_area_px=101.0,
                output_fields=("net_area",),
            )
        )
        intensity = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.INTENSITY,
                request_id="intensity-selected-fields",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                parameters={"channel": "luminance"},
                output_fields=("central_tendency",),
            )
        )

        self.assertEqual(
            set(shape.scalars),
            {
                "area_px",
                "vector_area_px",
                "area",
                "unit",
                "area_from_exact_mask",
            },
        )
        self.assertEqual(shape.scalars["area_px"], 101.0)
        self.assertEqual(shape.tables, ())
        self.assertEqual(
            set(intensity.scalars),
            {
                "included_pixel_count",
                "valid_pixel_count",
                "non_finite_count",
                "channel",
                "mean",
                "median",
                "mode",
            },
        )
        self.assertEqual(intensity.tables, ())
        self.assertEqual(
            intensity.parameters[ANALYSIS_OUTPUT_FIELDS_PARAMETER],
            ["central_tendency"],
        )
        artifact = intensity.to_analysis_artifact(
            artifact_id="selected-intensity"
        )
        replay = ImageAnalysisTaskRequest(
            tool=AnalysisTool.INTENSITY,
            request_id="selected-intensity-replay",
            generation=2,
            document_id="doc",
            source_pixel_revision=0,
            plane=_gray_plane(),
            parameters=artifact.parameters,
        )
        self.assertEqual(replay.output_fields, ("central_tendency",))
        self.assertNotIn(
            ANALYSIS_OUTPUT_FIELDS_PARAMETER,
            replay.parameters,
        )

    def test_result_warnings_are_preserved_in_analysis_artifact(self) -> None:
        result = ImageAnalysisTaskResult(
            tool=AnalysisTool.INTENSITY,
            request_id="warnings",
            generation=1,
            document_id="doc",
            source_pixel_revision=0,
            source_reference=None,
            calibration_signature=None,
            parameters={},
            scalars={"valid_pixel_count": 0},
            warnings=("没有有限像素，部分统计量为空。",),
        )

        artifact = result.to_analysis_artifact(artifact_id="warnings-artifact")

        self.assertEqual(artifact.warnings, result.warnings)

    def test_glcm_selection_projects_columns_rows_and_optional_asset(
        self,
    ) -> None:
        common = {
            "tool": AnalysisTool.GLCM,
            "generation": 1,
            "document_id": "doc",
            "source_pixel_revision": 0,
            "plane": _gray_plane(),
            "parameters": {
                "levels": 8,
                "distances": [1],
                "directions_degrees": [0, 90],
            },
        }
        contrast = _execute(
            ImageAnalysisTaskRequest(
                request_id="glcm-contrast",
                output_fields=("contrast",),
                **common,
            )
        )
        matrices = _execute(
            ImageAnalysisTaskRequest(
                request_id="glcm-matrices",
                output_fields=("glcm_matrices",),
                **common,
            )
        )

        self.assertEqual(
            contrast.tables[0].columns,
            ("距离(px)", "方向(°)", "像素对数", "Contrast"),
        )
        self.assertEqual(
            tuple(row[0] for row in contrast.tables[1].rows),
            ("Contrast",),
        )
        self.assertEqual(contrast.asset_payloads, ())
        self.assertEqual(matrices.tables, ())
        self.assertEqual(
            tuple(asset.schema for asset in matrices.asset_payloads),
            ("fdm.glcm-matrices.v1",),
        )

    def test_spatial_v2_parameters_reach_kernel_and_package_ripley_outputs(
        self,
    ) -> None:
        result = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.SPATIAL_DISTRIBUTION,
                request_id="spatial-v2",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(20, 20),
                parameters={
                    "points": [[2.0, 2.0], [8.0, 5.0], [14.0, 13.0]],
                    "study_bounds": [0.0, 0.0, 20.0, 20.0],
                    "ripley_radii": [2.0, 5.0, 10.0],
                    "algorithm_version": 2,
                },
            )
        )

        self.assertEqual(result.scalars["algorithm_version"], 2)
        self.assertIn("平移边界校正", result.scalars["boundary_correction"])
        self.assertIn("Ripley K/L", {table.name for table in result.tables})
        self.assertEqual(
            {curve.name for curve in result.curves},
            {"Ripley K(r)", "Ripley L(r)"},
        )

    def test_fft_is_packaged_as_analysis_asset_not_conversion_or_derived_image(
        self,
    ) -> None:
        roi = np.zeros((16, 16), dtype=bool)
        roi[2:14, 3:12] = True
        result = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.FFT_POWER_SPECTRUM,
                request_id="fft-analysis-1",
                generation=4,
                document_id="doc",
                source_pixel_revision=7,
                plane=_gray_plane(),
                roi_mask=roi,
                parameters={
                    "channel": "luminance",
                    "logarithmic": True,
                    "centered": True,
                    "window": "tukey",
                    "tukey_alpha": 0.25,
                },
            )
        )

        self.assertIsNone(result.conversion_payload)
        self.assertFalse(result.curves)
        self.assertEqual(len(result.asset_payloads), 1)
        payload = result.asset_payloads[0]
        self.assertEqual(payload.metadata["schema"], "fdm.fft-power-spectrum.v1")
        self.assertEqual(payload.metadata["allow_pickle"], False)
        self.assertEqual(
            payload.metadata["mask_policy"],
            "tight_bounds_zero_outside_exact_mask",
        )
        power = payload.array_mapping()["power"]
        self.assertEqual(power.dtype, np.float32)
        self.assertFalse(power.flags.writeable)
        reference = AnalysisAssetReference(
            kind=payload.kind,
            path="analysis/result/fft-power-spectrum.npz",
            sha256="f" * 64,
            media_type="application/x-npz",
            metadata=payload.metadata,
        )
        artifact = result.to_analysis_artifact(
            artifact_id="analysis-fft",
            asset_references=(reference,),
        )
        self.assertEqual(artifact.tool_id, "fdm.fft_power_spectrum")
        self.assertEqual(artifact.tool_version, "1")
        self.assertEqual(artifact.assets, (reference,))

    def test_request_defensively_freezes_roi_raw_rings_and_parameters(self) -> None:
        roi = np.zeros((16, 16), dtype=bool)
        roi[2:10, 3:12] = True
        rings = [[(1.0, 1.0), (12.0, 1.0), (12.0, 12.0), (1.0, 12.0)]]
        parameters = {"bins": 16, "value_range": [0, 255]}

        request = ImageAnalysisTaskRequest(
            tool=AnalysisTool.HISTOGRAM,
            request_id="histogram-1",
            generation=3,
            document_id="doc-1",
            source_pixel_revision=7,
            plane=_gray_plane(),
            roi_mask=roi,
            raw_rings=rings,
            parameters=parameters,
        )
        roi[:] = False
        rings[0][0] = (99.0, 99.0)
        parameters["bins"] = 1
        returned = request.parameters
        returned["bins"] = 2

        self.assertEqual(int(np.count_nonzero(request.roi_mask)), 72)
        self.assertFalse(request.roi_mask.flags.writeable)
        self.assertEqual(request.raw_rings[0][0], (1.0, 1.0))
        self.assertEqual(request.parameters["bins"], 16)

    def test_provenance_is_strongly_typed_and_transferred_to_artifact(self) -> None:
        region_snapshot = AnalysisRegionSnapshot(
            mask_sha256="a" * 64,
            pixel_center_rule="pixel-center-inclusion",
            components=1,
            holes=0,
            rings=(((1.0, 1.0), (5.0, 1.0), (5.0, 5.0)),),
            source="roi-mask",
        )
        source_descriptor = AnalysisSourceDescriptor(
            kind="raster",
            pixel_sha256="b" * 64,
        )
        dependency_signature = AnalysisDependencySignature(
            calibration={"pixel_size_x": 1.0, "pixel_size_y": 1.0},
            roi_transitive_refs={"roi-1": 3},
        )
        request = ImageAnalysisTaskRequest(
            tool=AnalysisTool.INTENSITY,
            request_id="provenance-1",
            generation=1,
            document_id="doc",
            source_pixel_revision=4,
            plane=_gray_plane(),
            region_snapshot=region_snapshot,
            source_descriptor=source_descriptor,
            dependency_signature=dependency_signature,
        )

        result = _execute(request)
        artifact = result.to_analysis_artifact(artifact_id="provenance-result")

        self.assertIs(result.region_snapshot, region_snapshot)
        self.assertIs(result.source_descriptor, source_descriptor)
        self.assertIs(result.dependency_signature, dependency_signature)
        self.assertEqual(artifact.region_snapshot, region_snapshot)
        self.assertEqual(artifact.source_descriptor, source_descriptor)
        self.assertEqual(artifact.dependency_signature, dependency_signature)

        common = {
            "tool": AnalysisTool.INTENSITY,
            "request_id": "invalid-provenance",
            "generation": 1,
            "document_id": "doc",
            "source_pixel_revision": 0,
            "plane": _gray_plane(),
        }
        with self.assertRaisesRegex(TypeError, "region_snapshot"):
            ImageAnalysisTaskRequest(
                **common,
                region_snapshot={},  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(TypeError, "source_descriptor"):
            ImageAnalysisTaskRequest(
                **common,
                source_descriptor={},  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(TypeError, "dependency_signature"):
            ImageAnalysisTaskRequest(
                **common,
                dependency_signature={},  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(TypeError, "region_snapshot"):
            ImageAnalysisTaskResult(
                tool=AnalysisTool.INTENSITY,
                request_id="invalid-result-provenance",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                source_reference=None,
                calibration_signature=None,
                parameters={},
                scalars={},
                region_snapshot={},  # type: ignore[arg-type]
            )

    def test_shape_keeps_raw_hole_geometry_and_exact_area_priority(self) -> None:
        rings = (
            ((1, 1), (13, 1), (13, 13), (1, 13)),
            ((4, 4), (8, 4), (8, 8), (4, 8)),
        )
        reference = AnalysisObjectReference(
            AnalysisObjectKind.MEASUREMENT,
            "measurement-1",
            9,
        )
        request = ImageAnalysisTaskRequest(
            tool=AnalysisTool.SHAPE,
            request_id="shape-1",
            generation=2,
            document_id="doc-1",
            source_pixel_revision=5,
            plane=_gray_plane(),
            raw_rings=rings,
            exact_area_px=101.0,
            source_reference=reference,
            calibration=AnalysisCalibrationSnapshot(
                pixel_size_x=0.5,
                pixel_size_y=0.5,
                unit="µm",
                signature="sha256:" + "a" * 64,
            ),
        )

        result = _execute(request)

        self.assertEqual(result.scalars["area_px"], 101.0)
        self.assertEqual(result.scalars["vector_area_px"], 128.0)
        self.assertEqual(result.scalars["area"], 25.25)
        self.assertEqual(result.scalars["hole_count"], 1)
        self.assertTrue(result.scalars["area_from_exact_mask"])
        self.assertTrue(
            any("未混用精确掩膜面积" in item for item in result.warnings)
        )
        artifact = result.to_analysis_artifact(artifact_id="analysis-shape")
        self.assertEqual(artifact.source_reference, reference)
        self.assertEqual(artifact.scalars["area_px"], 101.0)
        self.assertEqual(artifact.calibration_signature, "sha256:" + "a" * 64)

    def test_local_thickness_uses_physical_units_only_for_isotropic_pixels(
        self,
    ) -> None:
        result = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.LOCAL_THICKNESS,
                request_id="thickness-isotropic",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                calibration=AnalysisCalibrationSnapshot(
                    pixel_size_x=0.5,
                    pixel_size_y=0.5,
                    unit="µm",
                ),
                parameters={"threshold": 128},
            )
        )

        self.assertEqual(result.scalars["unit"], "µm")
        self.assertEqual(result.scalars["physical_unit_available"], True)
        self.assertEqual(result.curves[0].x_unit, "µm")
        self.assertEqual(result.warnings, ())

    def test_histogram_and_profile_are_packaged_as_curves(self) -> None:
        histogram = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.HISTOGRAM,
                request_id="histogram-1",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                parameters={"bins": 16},
            )
        )
        profile = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.PROFILE,
                request_id="profile-1",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                parameters={"points": [[0, 0], [15, 15]]},
            )
        )

        self.assertEqual(histogram.curves[0].name, "直方图")
        self.assertEqual(len(histogram.curves[0].x), 16)
        self.assertEqual(profile.curves[0].name, "强度剖面")
        self.assertEqual(profile.curves[0].x_unit, "px")

    def test_particle_and_maxima_keep_explicit_conversion_payloads(self) -> None:
        particle_result = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.PARTICLES,
                request_id="particle-1",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                parameters={"threshold": 128},
            )
        )
        maxima_result = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.MAXIMA,
                request_id="maxima-1",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                viewport_origin=(37, -9),
            )
        )

        self.assertIsInstance(
            particle_result.conversion_payload,
            ParticleConversionPayload,
        )
        candidate = particle_result.conversion_payload.candidates[0]
        self.assertGreater(candidate.exact_area_px, 0)
        self.assertGreaterEqual(len(candidate.rings[0]), 3)
        self.assertIsInstance(
            maxima_result.conversion_payload,
            MaximaConversionPayload,
        )
        self.assertGreaterEqual(len(maxima_result.conversion_payload.points), 1)

    def test_v2_results_package_new_statistics_assets_and_versions(self) -> None:
        intensity = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.INTENSITY,
                request_id="rgb-v2",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_rgb_plane(),
                parameters={
                    "channel": "rgb",
                    "threshold_low": 5,
                    "threshold_high": 30,
                },
            )
        )
        histogram = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.HISTOGRAM,
                request_id="hist-v2",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                parameters={"bins": 8, "log_counts": True},
            )
        )
        particles = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.PARTICLES,
                request_id="particles-v2",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                parameters={"threshold": 128},
            )
        )
        maxima = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.MAXIMA,
                request_id="maxima-v2",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                parameters={"algorithm_version": "2"},
            )
        )
        thickness = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.LOCAL_THICKNESS,
                request_id="thickness-v2",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                calibration=AnalysisCalibrationSnapshot(
                    pixel_size_x=0.5,
                    pixel_size_y=2.0,
                    unit="µm",
                ),
                parameters={"threshold": 128},
            )
        )
        glcm = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.GLCM,
                request_id="glcm-v2",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                parameters={
                    "levels": 8,
                    "distances": [1],
                    "directions_degrees": [0, 90],
                },
            )
        )

        self.assertIn("通道统计", [table.name for table in intensity.tables])
        self.assertIn("mode", intensity.scalars)
        self.assertEqual(histogram.scalars["log_counts"], True)
        self.assertEqual(histogram.curves[0].y_unit, "log(1+频数)")
        self.assertIn("直方图明细", [table.name for table in histogram.tables])
        self.assertEqual(
            {payload.schema for payload in particles.asset_payloads},
            {"fdm.particle-labels.v2", "fdm.particle-contours.v2"},
        )
        self.assertIn("粒子面积汇总", [table.name for table in particles.tables])
        self.assertEqual(maxima.scalars["algorithm_version"], "2")
        self.assertEqual(
            maxima.to_analysis_artifact().tool_version,
            "2",
        )
        self.assertEqual(thickness.scalars["unit"], "px")
        self.assertEqual(
            thickness.scalars["physical_unit_available"],
            False,
        )
        self.assertIn("局部厚度分位数", [table.name for table in thickness.tables])
        self.assertEqual(thickness.curves[0].x_unit, "px")
        self.assertTrue(
            any("横向与纵向像素尺寸不同" in item for item in thickness.warnings)
        )
        self.assertIn("Haralick 聚合", [table.name for table in glcm.tables])

    def test_saved_particle_v2_asset_rebuilds_exact_conversion_payload(self) -> None:
        result = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.PARTICLES,
                request_id="particles-saved",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                viewport_origin=(120, -30),
                parameters={"threshold": 128},
            )
        )
        self.assertIsInstance(result.conversion_payload, ParticleConversionPayload)

        with TemporaryDirectory() as directory:
            mappings: dict[str, Path] = {}
            references: list[AnalysisAssetReference] = []
            for index, payload in enumerate(result.asset_payloads):
                relative = f"analysis/saved/asset-{index}.npz"
                source = Path(directory) / f"asset-{index}.npz"
                info = write_safe_analysis_npz(
                    source,
                    schema=payload.schema,
                    arrays=payload.array_mapping(),
                )
                mappings[relative] = source
                references.append(
                    AnalysisAssetReference(
                        kind=payload.kind,
                        path=relative,
                        sha256=info.sha256,
                        media_type="application/x-npz",
                        metadata=payload.metadata,
                    )
                )
            artifact = result.to_analysis_artifact(
                artifact_id="saved-particles",
                asset_references=references,
            )

            rebuilt = rebuild_particle_conversion_payload(
                artifact,
                asset_source_paths=mappings,
            )
            unified = rebuild_analysis_conversion_payload(
                artifact,
                asset_source_paths=mappings,
            )

            self.assertEqual(rebuilt, result.conversion_payload)
            self.assertEqual(unified, rebuilt)
            self.assertEqual(rebuilt.viewport_origin, (120, -30))
            self.assertEqual(
                rebuilt.candidates[0].rings,
                result.conversion_payload.candidates[0].rings,
            )

            label_source = mappings[references[0].path]
            label_source.write_bytes(label_source.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                rebuild_particle_conversion_payload(
                    artifact,
                    asset_source_paths=mappings,
                )

    def test_inline_maxima_artifact_rebuilds_through_unified_api(self) -> None:
        result = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.MAXIMA,
                request_id="maxima-saved",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                viewport_origin=(37, -9),
            )
        )
        artifact = result.to_analysis_artifact(artifact_id="saved-maxima")

        rebuilt = rebuild_analysis_conversion_payload(artifact)

        self.assertEqual(rebuilt, result.conversion_payload)
        self.assertEqual(rebuilt.viewport_origin, (37, -9))

    def test_saved_maxima_asset_rebuilds_through_unified_api(self) -> None:
        result = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.MAXIMA,
                request_id="maxima-asset",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
            )
        )
        self.assertIsInstance(result.conversion_payload, MaximaConversionPayload)
        values = np.asarray(
            [
                (index, x, y, value, 0.0)
                for index, (x, y, value) in enumerate(
                    result.conversion_payload.points,
                    start=1,
                )
            ],
            dtype=np.float64,
        ).reshape((-1, 5))

        with TemporaryDirectory() as directory:
            source = Path(directory) / "maxima.npz"
            info = write_safe_analysis_npz(
                source,
                schema="fdm.maxima-table.v1",
                arrays={"values": values},
            )
            reference = AnalysisAssetReference(
                kind="table",
                path="analysis/maxima/maxima.npz",
                sha256=info.sha256,
                media_type="application/x-npz",
                metadata={
                    "schema": info.schema,
                    "allow_pickle": False,
                    "columns": [
                        "index",
                        "x_px",
                        "y_px",
                        "value",
                        "local_prominence",
                    ],
                    "members": {
                        name: {"dtype": dtype, "shape": list(shape)}
                        for name, dtype, shape in info.members
                    },
                },
            )
            payload = result.to_analysis_artifact().to_dict()
            payload["tables"] = []
            payload["assets"] = [reference.to_dict()]
            artifact = type(result.to_analysis_artifact()).from_dict(payload)

            rebuilt = rebuild_analysis_conversion_payload(
                artifact,
                asset_source_paths={reference.path: source},
            )

        self.assertEqual(rebuilt, result.conversion_payload)

    def test_saved_maxima_v2_asset_rebuilds_viewport_origin(self) -> None:
        result = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.MAXIMA,
                request_id="maxima-asset-v2",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                viewport_origin=(81, 144),
            )
        )
        self.assertIsInstance(result.conversion_payload, MaximaConversionPayload)
        values = np.asarray(
            [
                (index, x, y, value, 0.0)
                for index, (x, y, value) in enumerate(
                    result.conversion_payload.points,
                    start=1,
                )
            ],
            dtype=np.float64,
        ).reshape((-1, 5))

        with TemporaryDirectory() as directory:
            source = Path(directory) / "maxima-v2.npz"
            info = write_safe_analysis_npz(
                source,
                schema="fdm.maxima-table.v2",
                arrays={
                    "values": values,
                    "viewport_origin": np.asarray(
                        result.conversion_payload.viewport_origin,
                        dtype=np.int64,
                    ),
                },
            )
            reference = AnalysisAssetReference(
                kind="table",
                path="analysis/maxima/maxima-v2.npz",
                sha256=info.sha256,
                media_type="application/x-npz",
                metadata={
                    "schema": info.schema,
                    "allow_pickle": False,
                    "columns": [
                        "index",
                        "x_px",
                        "y_px",
                        "value",
                        "local_prominence",
                    ],
                    "coordinate_space": "viewport_pixel",
                    "conversion_schema": "fdm.maxima-conversion.v2",
                    "members": {
                        name: {"dtype": dtype, "shape": list(shape)}
                        for name, dtype, shape in info.members
                    },
                },
            )
            payload = result.to_analysis_artifact().to_dict()
            payload["tables"] = []
            payload["assets"] = [reference.to_dict()]
            artifact = type(result.to_analysis_artifact()).from_dict(payload)

            rebuilt = rebuild_analysis_conversion_payload(
                artifact,
                asset_source_paths={reference.path: source},
            )

        self.assertEqual(rebuilt, result.conversion_payload)
        self.assertEqual(rebuilt.viewport_origin, (81, 144))

    def test_large_maxima_packaging_uses_v2_conversion_asset(self) -> None:
        with patch(
            "fdm.ui.image_analysis_controller._INLINE_DETAIL_ROWS",
            0,
        ):
            result = _execute(
                ImageAnalysisTaskRequest(
                    tool=AnalysisTool.MAXIMA,
                    request_id="maxima-large-v2",
                    generation=1,
                    document_id="doc",
                    source_pixel_revision=0,
                    plane=_gray_plane(),
                    viewport_origin=(23, 45),
                )
            )

        self.assertFalse(result.tables)
        self.assertEqual(len(result.asset_payloads), 1)
        payload = result.asset_payloads[0]
        self.assertEqual(payload.schema, "fdm.maxima-table.v2")
        self.assertEqual(
            payload.metadata["conversion_schema"],
            "fdm.maxima-conversion.v2",
        )
        np.testing.assert_array_equal(
            payload.array_mapping()["viewport_origin"],
            np.asarray([23, 45], dtype=np.int64),
        )

    def test_large_array_payload_requires_matching_atomic_asset_references(self) -> None:
        result = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.LOCAL_THICKNESS,
                request_id="thickness-1",
                generation=1,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                parameters={"threshold": 128},
            )
        )
        payload = result.asset_payloads[0]

        self.assertFalse(payload.arrays[0][1].flags.writeable)
        self.assertEqual(payload.metadata["allow_pickle"], False)
        self.assertEqual(payload.metadata["schema"], "fdm.local-thickness.v2")
        self.assertIn("thickness", payload.array_mapping())
        self.assertTrue(result.curves)
        with self.assertRaisesRegex(ValueError, "必须先由项目保存层"):
            result.to_analysis_artifact()

        reference = AnalysisAssetReference(
            kind=payload.kind,
            path="analysis/result/local-thickness.npz",
            sha256="b" * 64,
            media_type="application/x-npz",
            metadata=payload.metadata,
        )
        artifact = result.to_analysis_artifact(
            artifact_id="analysis-thickness",
            asset_references=(reference,),
        )
        self.assertEqual(artifact.assets, (reference,))

    def test_all_analysis_tools_dispatch_on_small_deterministic_inputs(self) -> None:
        cases: tuple[tuple[AnalysisTool, dict[str, object]], ...] = (
            (
                AnalysisTool.SHAPE,
                {"raw_rings": (((1, 1), (10, 1), (10, 10), (1, 10)),)},
            ),
            (AnalysisTool.INTENSITY, {}),
            (AnalysisTool.HISTOGRAM, {"parameters": {"bins": 8}}),
            (AnalysisTool.FFT_POWER_SPECTRUM, {}),
            (
                AnalysisTool.PROFILE,
                {"parameters": {"points": [[0, 0], [15, 15]]}},
            ),
            (
                AnalysisTool.PARTICLES,
                {"parameters": {"threshold": 128}},
            ),
            (AnalysisTool.MAXIMA, {}),
            (AnalysisTool.DIRECTIONALITY, {"parameters": {"bins": 18}}),
            (
                AnalysisTool.SKELETON,
                {"parameters": {"threshold": 128}},
            ),
            (
                AnalysisTool.LOCAL_THICKNESS,
                {"parameters": {"threshold": 128}},
            ),
            (
                AnalysisTool.TUBENESS,
                {"parameters": {"scales": [1.0]}},
            ),
            (
                AnalysisTool.GLCM,
                {"parameters": {"levels": 8, "directions_degrees": [0.0]}},
            ),
            (
                AnalysisTool.SPATIAL_DISTRIBUTION,
                {
                    "parameters": {
                        "points": [[0, 0], [10, 10]],
                        "study_area": 256,
                    }
                },
            ),
            (
                AnalysisTool.SURFACE,
                {"parameters": {"sample_step_x": 4, "sample_step_y": 4}},
            ),
        )

        for tool, kwargs in cases:
            with self.subTest(tool=tool.value):
                result = _execute(
                    ImageAnalysisTaskRequest(
                        tool=tool,
                        request_id=f"{tool.value}-1",
                        generation=1,
                        document_id="doc",
                        source_pixel_revision=0,
                        plane=_gray_plane(),
                        **kwargs,
                    )
                )
                self.assertEqual(result.tool, tool)
                self.assertEqual(result.request_id, f"{tool.value}-1")

    def test_skeleton_v2_preserves_tubeness_chain_audit_parameters(self) -> None:
        parameters = {
            "threshold": 128,
            "algorithm_version": 2,
            "chain_parent_artifact_id": "analysis_mask",
            "chain_source_tubeness_artifact_id": "analysis_tubeness",
            "chain_threshold": 0.25,
            "chain_mask_sha256": "a" * 64,
            "chain_response_asset_sha256": "b" * 64,
        }
        result = _execute(
            ImageAnalysisTaskRequest(
                tool=AnalysisTool.SKELETON,
                request_id="skeleton-chain",
                generation=2,
                document_id="doc",
                source_pixel_revision=3,
                plane=_gray_plane(),
                parameters=parameters,
            )
        )

        self.assertEqual(dict(result.parameters), parameters)
        artifact = result.to_analysis_artifact(
            asset_references=tuple(
                AnalysisAssetReference(
                    kind=payload.kind,
                    path=f"analysis/skeleton/{index}.npz",
                    sha256=str(index + 1) * 64,
                    media_type="application/x-npz",
                    metadata=payload.metadata,
                )
                for index, payload in enumerate(result.asset_payloads)
            )
        )
        self.assertEqual(artifact.tool_version, "2")
        self.assertEqual(
            artifact.parameters["chain_source_tubeness_artifact_id"],
            "analysis_tubeness",
        )

    def test_spatial_quadratic_workset_over_one_gib_is_blocked_in_chinese(self) -> None:
        points = [[float(index), float(index % 17)] for index in range(6_000)]
        request = ImageAnalysisTaskRequest(
            tool=AnalysisTool.SPATIAL_DISTRIBUTION,
            request_id="large-spatial",
            generation=1,
            document_id="doc",
            source_pixel_revision=0,
            plane=_gray_plane(1, 1),
            parameters={"points": points, "study_area": 1_000_000.0},
        )

        estimate = estimate_analysis_resources(request)

        self.assertGreater(estimate.estimated_peak_bytes, MAX_ANALYSIS_WORKING_BYTES)
        self.assertFalse(estimate.allowed)
        self.assertIn("超过", estimate.reason)
        self.assertIn("安全上限", estimate.reason)
        with self.assertRaisesRegex(MemoryError, "缩小 ROI"):
            _execute(request)


class AnalysisTaskControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _wait_until(self, predicate, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        self.fail("等待异步图像分析条件超时")

    def test_controller_emits_real_ordered_phases_and_result(self) -> None:
        controller = ImageAnalysisTaskController()
        phases = []
        results = []
        controller.phaseChanged.connect(phases.append)
        controller.analysisReady.connect(results.append)
        try:
            request = controller.start(
                tool=AnalysisTool.HISTOGRAM,
                document_id="doc",
                source_pixel_revision=4,
                plane=_gray_plane(),
                viewport_origin=(23, -5),
                parameters={"bins": 8},
            )
            self._wait_until(lambda: bool(results))

            self.assertEqual(results[0].request_id, request.request_id)
            self.assertEqual(request.viewport_origin, (23, -5))
            self.assertEqual(
                [update.phase for update in phases],
                [
                    AnalysisTaskPhase.PREPARING,
                    AnalysisTaskPhase.ANALYZING,
                    AnalysisTaskPhase.PACKAGING,
                ],
            )
            self.assertFalse(controller.is_busy())
        finally:
            controller.close()
            controller.wait_for_done()

    def test_late_generation_is_discarded_and_latest_request_runs(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def executor(request, _token, _phase):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                if request.generation == 1:
                    first_started.set()
                    release_first.wait(2.0)
                return _result_for(request)
            finally:
                with lock:
                    active -= 1

        controller = ImageAnalysisTaskController(executor=executor)
        ready = []
        controller.analysisReady.connect(ready.append)
        try:
            first = controller.start(
                tool=AnalysisTool.INTENSITY,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
            )
            self.assertTrue(first_started.wait(1.0))
            second = controller.start(
                tool=AnalysisTool.HISTOGRAM,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
                parameters={"bins": 8},
            )
            release_first.set()
            self._wait_until(lambda: len(ready) == 1)

            self.assertEqual(ready[0].request_id, second.request_id)
            self.assertNotEqual(ready[0].request_id, first.request_id)
            self.assertEqual(maximum_active, 1)
        finally:
            release_first.set()
            controller.close()
            controller.wait_for_done()

    def test_cancelled_task_never_emits_success(self) -> None:
        started = threading.Event()

        def executor(request, token, _phase):
            started.set()
            while not token.is_cancelled:
                time.sleep(0.002)
            token.raise_if_cancelled()
            return _result_for(request)  # pragma: no cover

        controller = ImageAnalysisTaskController(executor=executor)
        ready = []
        cancelled = []
        controller.analysisReady.connect(ready.append)
        controller.taskCancelled.connect(cancelled.append)
        try:
            request = controller.start(
                tool=AnalysisTool.INTENSITY,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(),
            )
            self.assertTrue(started.wait(1.0))
            controller.cancel()
            self._wait_until(lambda: not controller.is_busy())
            self.assertEqual(ready, [])
            self.assertEqual(cancelled, [request.request_id])
        finally:
            controller.close()
            controller.wait_for_done()

    def test_resource_failure_is_structured_and_does_not_emit_success(self) -> None:
        controller = ImageAnalysisTaskController()
        failures = []
        ready = []
        controller.taskFailed.connect(
            lambda request_id, message: failures.append((request_id, message))
        )
        controller.analysisReady.connect(ready.append)
        points = [[float(index), float(index % 19)] for index in range(6_000)]
        try:
            request = controller.start(
                tool=AnalysisTool.SPATIAL_DISTRIBUTION,
                document_id="doc",
                source_pixel_revision=0,
                plane=_gray_plane(1, 1),
                parameters={"points": points, "study_area": 1_000_000.0},
            )
            self._wait_until(lambda: bool(failures))

            self.assertEqual(failures[0][0], request.request_id)
            self.assertIn("安全上限", failures[0][1])
            self.assertEqual(ready, [])
        finally:
            controller.close()
            controller.wait_for_done()


if __name__ == "__main__":
    unittest.main()
