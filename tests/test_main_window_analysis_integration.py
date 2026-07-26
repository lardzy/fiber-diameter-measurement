from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

from fdm.analysis_artifacts import AnalysisAssetKind
from fdm.geometry import Point
from fdm.models import ImageDocument, Measurement, new_id
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.raster_io import raster_plane_to_qimage
from fdm.settings import AppSettings
from fdm.ui.image_analysis_controller import (
    AnalysisAssetPayload,
    AnalysisTool,
    ImageAnalysisTaskResult,
    ParticleConversionPayload,
    ParticleMeasurementCandidate,
)
from fdm.ui.main_window import (
    ImageAnalysisRunContext,
    ImageAnalysisSourceContext,
    MainWindow,
)
from fdm.ui.analysis_results_center import AnalysisActionRequest


class MainWindowAnalysisIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.load_patch = patch(
            "fdm.ui.main_window.AppSettingsIO.load",
            return_value=AppSettings(theme_mode="dark"),
        )
        self.save_patch = patch(
            "fdm.ui.main_window.AppSettingsIO.save",
            return_value=None,
        )
        self.load_patch.start()
        self.save_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.addCleanup(self.save_patch.stop)

    def _process_events(self, turns: int = 4) -> None:
        for _ in range(turns):
            self.app.processEvents()

    def _window(self) -> tuple[MainWindow, Path]:
        session = TemporaryDirectory()
        self.addCleanup(session.cleanup)
        window = MainWindow()
        window._session_analysis_root = Path(session.name)
        window.resize(1100, 700)
        window.show()
        self._process_events()

        def cleanup() -> None:
            window._reset_workspace()
            window.close()
            self._process_events()

        self.addCleanup(cleanup)
        return window, Path(session.name)

    @staticmethod
    def _plane(value: int = 30) -> RasterPlane:
        return RasterPlane(
            width=16,
            height=12,
            pixel_type=RasterPixelType.GRAY8,
            data=bytes([value]) * (16 * 12),
        )

    def _mount_document(
        self,
        window: MainWindow,
        *,
        plane: RasterPlane | None = None,
    ) -> ImageDocument:
        raster = plane or self._plane()
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/analysis-source.png",
            image_size=(raster.width, raster.height),
        )
        document.initialize_runtime_state()
        document.mark_session_saved()
        document.mark_calibration_saved()
        window._mount_document(
            document,
            raster_plane_to_qimage(raster),
            tooltip=document.path,
            raster_plane=raster,
        )
        self._process_events()
        return document

    def test_analysis_menu_contains_six_core_tools(self) -> None:
        window, _root = self._window()
        self.assertEqual(
            tuple(window._analysis_actions),
            (
                AnalysisTool.SHAPE,
                AnalysisTool.INTENSITY,
                AnalysisTool.HISTOGRAM,
                AnalysisTool.PROFILE,
                AnalysisTool.PARTICLES,
                AnalysisTool.MAXIMA,
            ),
        )
        self.assertEqual(
            window.analysis_results_center_action.text(),
            "分析结果中心…",
        )

    def test_shape_scope_keeps_raw_hole_rings_and_exact_area(self) -> None:
        window, _root = self._window()
        plane = self._plane()
        document = self._mount_document(window, plane=plane)
        area = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="magic_segment",
            measurement_kind="area",
            polygon_px=[
                Point(1, 1),
                Point(12, 1),
                Point(12, 10),
                Point(1, 10),
            ],
            area_rings_px=[
                [
                    Point(1, 1),
                    Point(12, 1),
                    Point(12, 10),
                    Point(1, 10),
                ],
                [
                    Point(4, 4),
                    Point(8, 4),
                    Point(8, 7),
                    Point(4, 7),
                ],
            ],
            exact_area_px=87.0,
        )
        document.add_measurement(area)
        document.select_measurement(area.id)
        source = ImageAnalysisSourceContext(
            document_id=document.id,
            plane_sha256=plane.sha256(),
            source_signature=(document.id, "raster", plane.sha256()),
        )

        _mask, rings, exact, reference, summary, _points = (
            window._analysis_scope_snapshot(
                tool=AnalysisTool.SHAPE,
                document=document,
                plane=plane,
                source=source,
            )
        )

        self.assertEqual(len(rings), 2)
        self.assertEqual(exact, 87.0)
        self.assertEqual(reference.object_id, area.id)
        self.assertEqual(summary, "当前选中的面积对象")

    def test_digital_viewport_geometry_is_shifted_to_local_pixels(self) -> None:
        window, _root = self._window()
        plane = self._plane()
        document = self._mount_document(window, plane=plane)
        area = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=None,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=[
                Point(101, 202),
                Point(108, 202),
                Point(108, 208),
                Point(101, 208),
            ],
            area_rings_px=[
                [
                    Point(101, 202),
                    Point(108, 202),
                    Point(108, 208),
                    Point(101, 208),
                ]
            ],
        )
        document.add_measurement(area)
        document.select_measurement(area.id)
        source = ImageAnalysisSourceContext(
            document_id=document.id,
            plane_sha256=plane.sha256(),
            source_signature=(
                document.id,
                "digital_slide_viewport",
                0,
                100,
                200,
                plane.width,
                plane.height,
                plane.sha256(),
            ),
            viewport_origin=(100, 200),
        )

        _mask, rings, _exact, _reference, _summary, _points = (
            window._analysis_scope_snapshot(
                tool=AnalysisTool.SHAPE,
                document=document,
                plane=plane,
                source=source,
            )
        )

        self.assertEqual(rings[0][0], (1.0, 2.0))
        self.assertEqual(rings[0][2], (8.0, 8.0))

    def test_ready_result_writes_safe_npz_before_project_commit(self) -> None:
        window, session_root = self._window()
        plane = self._plane()
        document = self._mount_document(window, plane=plane)
        source = ImageAnalysisSourceContext(
            document_id=document.id,
            plane_sha256=plane.sha256(),
            source_signature=(document.id, "raster", plane.sha256()),
        )
        request_id = "analysis-request"
        window._analysis_run_contexts[request_id] = ImageAnalysisRunContext(
            request_id=request_id,
            generation=7,
            tool=AnalysisTool.HISTOGRAM,
            source=source,
        )
        result = ImageAnalysisTaskResult(
            tool=AnalysisTool.HISTOGRAM,
            request_id=request_id,
            generation=7,
            document_id=document.id,
            source_pixel_revision=0,
            source_reference=None,
            calibration_signature=None,
            parameters={"bins": 4},
            scalars={"included_pixel_count": 192},
            asset_payloads=(
                AnalysisAssetPayload(
                    kind=AnalysisAssetKind.CURVE,
                    schema="fdm.test-histogram.v1",
                    suggested_stem="histogram",
                    arrays={
                        "counts": np.asarray([1, 2, 3, 4], dtype=np.int64)
                    },
                ),
            ),
        )

        with patch.object(
            window,
            "_open_analysis_results_center",
            return_value=None,
        ):
            window._on_image_analysis_ready(result)

        self.assertEqual(len(window.project.analysis_artifacts), 1)
        artifact = window.project.analysis_artifacts[0]
        self.assertEqual(len(artifact.assets), 1)
        asset_path = session_root / artifact.assets[0].path
        self.assertTrue(asset_path.is_file())
        self.assertEqual(
            window._session_analysis_assets[artifact.assets[0].path],
            asset_path,
        )

    def test_particle_conversion_applies_viewport_origin_once(self) -> None:
        window, _root = self._window()
        plane = self._plane()
        document = self._mount_document(window, plane=plane)
        result = ImageAnalysisTaskResult(
            tool=AnalysisTool.PARTICLES,
            request_id="particles",
            generation=1,
            document_id=document.id,
            source_pixel_revision=0,
            source_reference=None,
            calibration_signature=None,
            parameters={"threshold": 10},
            scalars={"accepted_count": 1},
            conversion_payload=ParticleConversionPayload(
                candidates=(
                    ParticleMeasurementCandidate(
                        index=1,
                        exact_area_px=12,
                        centroid_px=(3.0, 4.0),
                        rings=(
                            (
                                (1.0, 2.0),
                                (5.0, 2.0),
                                (5.0, 5.0),
                                (1.0, 5.0),
                            ),
                        ),
                    ),
                )
            ),
        )
        artifact = result.to_analysis_artifact(
            artifact_id="analysis-particles"
        )
        window.project.analysis_artifacts.append(artifact)
        window._analysis_conversion_payloads[artifact.id] = (
            result.conversion_payload
        )
        window._analysis_conversion_offsets[artifact.id] = (100, 200)

        with patch(
            "fdm.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window._on_analysis_convert_requested(
                AnalysisActionRequest((artifact.id,))
            )

        self.assertEqual(len(document.measurements), 1)
        measurement = document.measurements[0]
        self.assertEqual(measurement.exact_area_px, 12.0)
        self.assertEqual(
            (measurement.polygon_px[0].x, measurement.polygon_px[0].y),
            (101.0, 202.0),
        )


if __name__ == "__main__":
    unittest.main()
