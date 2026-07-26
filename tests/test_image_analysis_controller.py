from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtWidgets import QApplication

from fdm.analysis_artifacts import (
    AnalysisAssetReference,
    AnalysisObjectKind,
    AnalysisObjectReference,
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
)


def _gray_plane(width: int = 16, height: int = 16) -> RasterPlane:
    values = np.arange(width * height, dtype=np.uint8)
    return RasterPlane(
        width=width,
        height=height,
        pixel_type=RasterPixelType.GRAY8,
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
        artifact = result.to_analysis_artifact(artifact_id="analysis-shape")
        self.assertEqual(artifact.source_reference, reference)
        self.assertEqual(artifact.scalars["area_px"], 101.0)
        self.assertEqual(artifact.calibration_signature, "sha256:" + "a" * 64)

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
        self.assertEqual(payload.metadata["schema"], "fdm.local-thickness.v1")
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
                parameters={"bins": 8},
            )
            self._wait_until(lambda: bool(results))

            self.assertEqual(results[0].request_id, request.request_id)
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
