from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox

from fdm.analysis_artifacts import (
    AnalysisArtifact,
    AnalysisAssetKind,
    AnalysisAssetReference,
    AnalysisDependencySignature,
    AnalysisRegionSnapshot,
    AnalysisSourceDescriptor,
    AnalysisTable,
)
from fdm.geometry import Point
from fdm.models import (
    ImageDocument,
    Measurement,
    new_id,
    project_assets_root,
)
from fdm.raster import RasterPixelType, RasterPlane
from fdm.project_roi import (
    ProjectRoi,
    RectangleRoiGeometry,
    RoiBooleanExpression,
    RoiBooleanOperator,
)
from fdm.services.raster_io import raster_plane_to_qimage
from fdm.services.analysis_asset_io import write_safe_analysis_npz
from fdm.services.analysis_profiles import ANALYSIS_OUTPUT_FIELDS_PARAMETER
from fdm.services.image_processing import ImageOperation
from fdm.settings import AppSettings
from fdm.ui.image_analysis_controller import (
    AnalysisAssetPayload,
    AnalysisTool,
    ImageAnalysisTaskResult,
    MaximaConversionPayload,
    ParticleConversionPayload,
    ParticleMeasurementCandidate,
)
from fdm.ui.main_window import (
    ImageAnalysisRunContext,
    ImageAnalysisSourceContext,
    MainWindow,
)
from fdm.ui.analysis_results_center import (
    AnalysisActionRequest,
    AnalysisExportRequest,
)
from fdm.ui.advanced_analysis_dialog import (
    SPATIAL_POINT_SCOPE_KEY,
    SPATIAL_STUDY_AREA_MODE_KEY,
)


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

    def _install_particle_artifact(
        self,
        window: MainWindow,
        document: ImageDocument,
        *,
        artifact_id: str,
        asset_root: Path,
        viewport_origin: tuple[int, int],
        local_x: float,
        map_session_asset: bool = True,
    ):
        ring = (
            (local_x, 2.0),
            (local_x + 4.0, 2.0),
            (local_x + 4.0, 5.0),
            (local_x, 5.0),
        )
        asset_payload = AnalysisAssetPayload(
            kind=AnalysisAssetKind.LABEL_IMAGE,
            schema="fdm.particle-labels.v2",
            suggested_stem="particle-labels",
            arrays={
                "labels": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
                "coordinates": np.asarray(ring, dtype=np.float64),
                "ring_offsets": np.asarray([0, 4], dtype=np.int64),
                "particle_ring_offsets": np.asarray([0, 1], dtype=np.int64),
                "particle_index": np.asarray([1], dtype=np.int64),
                "exact_area_px": np.asarray([12], dtype=np.int64),
                "centroid_px": np.asarray(
                    [[local_x + 2.0, 3.5]],
                    dtype=np.float64,
                ),
                "viewport_origin": np.asarray(
                    viewport_origin,
                    dtype=np.int64,
                ),
            },
            metadata={
                "background_label": 0,
                "coordinate_space": "viewport_pixel",
                "conversion_schema": "fdm.particle-conversion.v2",
            },
        )
        relative = f"analysis/{artifact_id}/particle-labels.npz"
        source = asset_root / relative
        info = write_safe_analysis_npz(
            source,
            schema=asset_payload.schema,
            arrays=asset_payload.array_mapping(),
            metadata=asset_payload.metadata,
        )
        reference = AnalysisAssetReference(
            kind=asset_payload.kind,
            path=relative,
            sha256=info.sha256,
            media_type="application/x-npz",
            metadata=asset_payload.metadata,
        )
        conversion = ParticleConversionPayload(
            candidates=(
                ParticleMeasurementCandidate(
                    index=1,
                    exact_area_px=12,
                    centroid_px=(local_x + 2.0, 3.5),
                    rings=(ring,),
                ),
            ),
            viewport_origin=viewport_origin,
        )
        result = ImageAnalysisTaskResult(
            tool=AnalysisTool.PARTICLES,
            request_id=f"request-{artifact_id}",
            generation=1,
            document_id=document.id,
            source_pixel_revision=0,
            source_reference=None,
            calibration_signature=None,
            parameters={"threshold": 10},
            scalars={"accepted_count": 1},
            asset_payloads=(asset_payload,),
            conversion_payload=conversion,
        )
        artifact = result.to_analysis_artifact(
            artifact_id=artifact_id,
            asset_references=(reference,),
        )
        window.project.analysis_artifacts.append(artifact)
        if map_session_asset:
            window._session_analysis_assets[relative] = source
        return artifact, source

    def test_tubeness_chain_commits_audited_mask_without_measurements(
        self,
    ) -> None:
        window, asset_root = self._window()
        document = self._mount_document(window)
        source_result = window._create_analysis_source_context()
        self.assertIsNotNone(source_result)
        _source_document, plane, source = source_result
        calibration = window._analysis_calibration_snapshot(document)
        region = window._analysis_region_snapshot(
            tool=AnalysisTool.TUBENESS,
            document=document,
            plane=plane,
            source=source,
            roi_mask=None,
            raw_rings=(),
            source_reference=None,
        )
        descriptor = window._analysis_source_descriptor(
            document,
            plane,
            source,
        )
        dependency = window._analysis_dependency_signature(
            tool=AnalysisTool.TUBENESS,
            document=document,
            source=source,
            calibration=calibration,
            source_reference=None,
            region_snapshot=region,
            parameters={"scales": [1.0]},
        )
        response = np.zeros((plane.height, plane.width), dtype=np.float32)
        response[2:6, 3:8] = 0.75
        source_path = asset_root / "analysis/tubeness/source.npz"
        info = write_safe_analysis_npz(
            source_path,
            schema="fdm.tubeness.v1",
            arrays={
                "response": response,
                "best_scale": np.where(response > 0, 2.0, 0.0).astype(
                    np.float32
                ),
                "scales": np.asarray((1.0, 2.0), dtype=np.float64),
            },
        )
        reference = AnalysisAssetReference(
            kind=AnalysisAssetKind.OTHER,
            path="analysis/tubeness/source.npz",
            sha256=info.sha256,
            media_type="application/x-npz",
            metadata={
                "schema": info.schema,
                "allow_pickle": False,
                "members": {
                    name: {"dtype": dtype, "shape": list(shape)}
                    for name, dtype, shape in info.members
                },
            },
        )
        source_artifact = AnalysisArtifact(
            id="analysis_tubeness_source",
            source_document_id=document.id,
            source_pixel_revision=0,
            region_snapshot=region,
            source_descriptor=descriptor,
            dependency_signature=dependency,
            tool_id="fdm.tubeness",
            tool_version="1",
            parameters={"scales": [1.0]},
            calibration_signature=calibration.signature,
            scalars={"maximum_response": 0.75},
            assets=(reference,),
        )
        window.project.analysis_artifacts.append(source_artifact)
        window._session_analysis_assets[reference.path] = source_path
        before_measurements = tuple(document.measurements)

        with (
            patch(
                "fdm.ui.main_window.QInputDialog.getDouble",
                return_value=(0.5, True),
            ),
            patch(
                "fdm.ui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ),
        ):
            window._on_tubeness_chain_requested(
                AnalysisActionRequest((source_artifact.id,))
            )

        self.assertEqual(tuple(document.measurements), before_measurements)
        self.assertEqual(len(window.project.analysis_artifacts), 2)
        mask_artifact = window.project.analysis_artifacts[-1]
        self.assertEqual(
            mask_artifact.tool_id,
            "fdm.tubeness_threshold_mask",
        )
        self.assertEqual(mask_artifact.source_descriptor, descriptor)
        self.assertEqual(mask_artifact.region_snapshot, region)
        self.assertEqual(mask_artifact.dependency_signature, dependency)
        self.assertEqual(
            mask_artifact.parameters["source_response_asset_sha256"],
            reference.sha256,
        )
        self.assertEqual(mask_artifact.assets[1], reference)
        self.assertEqual(
            mask_artifact.calibration_signature,
            calibration.signature,
        )
        mask_path = window._session_analysis_assets[
            mask_artifact.assets[0].path
        ]
        with np.load(mask_path, allow_pickle=False) as archive:
            mask = np.asarray(archive["mask"])
        self.assertEqual(int(mask.sum()), 20)
        self.assertTrue(window._project_dirty())

    def test_analysis_menu_contains_fft_while_process_menu_hides_legacy_fft(
        self,
    ) -> None:
        window, _root = self._window()
        self.assertEqual(tuple(window._analysis_actions), tuple(AnalysisTool))
        self.assertEqual(
            window.analysis_results_center_action.text(),
            "分析结果中心…",
        )
        analysis_menu = next(
            menu
            for menu in window.menuBar().findChildren(QMenu)
            if menu.title() == "分析"
        )
        advanced_menu = next(
            menu
            for menu in analysis_menu.findChildren(QMenu)
            if menu.title() == "纤维高级分析"
        )
        self.assertEqual(
            tuple(action.text() for action in advanced_menu.actions()),
            tuple(
                f"{tool.chinese_name}…"
                for tool in (
                    AnalysisTool.DIRECTIONALITY,
                    AnalysisTool.SKELETON,
                    AnalysisTool.LOCAL_THICKNESS,
                    AnalysisTool.TUBENESS,
                    AnalysisTool.GLCM,
                    AnalysisTool.SPATIAL_DISTRIBUTION,
                    AnalysisTool.SURFACE,
                )
            ),
        )
        self.assertIn(
            window._analysis_actions[AnalysisTool.FFT_POWER_SPECTRUM],
            analysis_menu.actions(),
        )
        process_menu = next(
            menu
            for menu in window.menuBar().findChildren(QMenu)
            if menu.title() == "处理"
        )
        process_actions = {
            action
            for submenu in process_menu.findChildren(QMenu)
            for action in submenu.actions()
        }
        self.assertNotIn(
            window._image_operation_actions[
                ImageOperation.FFT_POWER_SPECTRUM.value
            ],
            process_actions,
        )

    def test_spatial_distribution_uses_raw_count_points_group_roi_and_origin(
        self,
    ) -> None:
        window, _root = self._window()
        plane = self._plane()
        document = self._mount_document(window, plane=plane)
        active_group = document.create_group(color="#2A9D8F", label="目标类别")
        other_group = document.create_group(color="#E76F51", label="其他类别")
        document.set_active_group(active_group.id)
        for index, (x, y, group_id) in enumerate(
            (
                (101.2, 202.2, active_group.id),
                (103.5, 203.5, active_group.id),
                (110.0, 208.0, active_group.id),
                (102.0, 202.0, other_group.id),
            )
        ):
            document.add_measurement(
                Measurement(
                    id=f"count_{index}",
                    image_id=document.id,
                    fiber_group_id=group_id,
                    mode="count",
                    measurement_kind="count",
                    point_px=Point(x, y),
                )
            )
        roi = ProjectRoi(
            id="roi_spatial",
            document_id=document.id,
            name="当前视窗 ROI",
            geometry=RectangleRoiGeometry(100.0, 200.0, 6.0, 6.0),
        )
        window.project.project_rois.append(roi)
        window._selected_project_roi_ids = (roi.id,)
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
        parameters: dict[str, object] = {
            SPATIAL_POINT_SCOPE_KEY: "active_group",
            SPATIAL_STUDY_AREA_MODE_KEY: "scope",
        }

        mask, _rings, _exact, reference, summary, point_dependencies = (
            window._spatial_distribution_scope_snapshot(
                document=document,
                plane=plane,
                source=source,
                parameters=parameters,
            )
        )

        self.assertIsNotNone(mask)
        self.assertEqual(reference.object_id, roi.id)
        points = parameters["points"]
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0][0], 1.2)
        self.assertAlmostEqual(points[0][1], 2.2)
        self.assertEqual(points[1], (3.5, 3.5))
        self.assertEqual(parameters["point_scope"], "active_group")
        self.assertEqual(parameters["point_group_id"], active_group.id)
        self.assertEqual(parameters["study_area"], 36.0)
        self.assertIn("目标类别", summary)
        self.assertEqual(
            tuple(item[0] for item in point_dependencies),
            ("count_0", "count_1"),
        )

        with patch.object(
            window,
            "_create_analysis_source_context",
            return_value=(document, plane, source),
        ), patch.object(
            window.image_analysis_task_controller,
            "start",
            return_value=SimpleNamespace(
                request_id="spatial-provenance",
                generation=1,
            ),
        ) as start:
            window._start_image_analysis(
                AnalysisTool.SPATIAL_DISTRIBUTION,
                parameters={
                    SPATIAL_POINT_SCOPE_KEY: "active_group",
                    SPATIAL_STUDY_AREA_MODE_KEY: "scope",
                },
                prompt_for_parameters=False,
            )

        start_kwargs = start.call_args.kwargs
        self.assertEqual(start_kwargs["viewport_origin"], (100, 200))
        descriptor = start_kwargs["source_descriptor"]
        self.assertIsInstance(descriptor, AnalysisSourceDescriptor)
        self.assertEqual(descriptor.kind, "digital_slide_viewport")
        self.assertEqual(descriptor.store_id, document.id)
        self.assertEqual(descriptor.focus, 0)
        self.assertEqual(descriptor.origin, (100, 200))
        self.assertEqual(
            descriptor.viewport_size,
            (plane.width, plane.height),
        )
        dependency = start_kwargs["dependency_signature"]
        self.assertIsInstance(dependency, AnalysisDependencySignature)
        dependencies = dependency.dependencies
        self.assertEqual(dependencies["point_set"]["count"], 2)
        self.assertEqual(
            set(dependencies["measurement_revisions"]),
            {"count_0", "count_1"},
        )
        self.assertEqual(dependencies["group"]["id"], active_group.id)
        self.assertEqual(dependencies["group"]["scope"], "active_group")
        self.assertEqual(dependencies["study_region"]["mode"], "scope")
        self.assertEqual(
            dependencies["study_region"]["viewport_origin"],
            [100, 200],
        )
        self.assertEqual(
            set(dependencies["roi_transitive_refs"]),
            {roi.id},
        )
        self.assertIsInstance(
            start_kwargs["region_snapshot"],
            AnalysisRegionSnapshot,
        )

    def test_spatial_distribution_blocks_fewer_than_two_points(self) -> None:
        window, _root = self._window()
        plane = self._plane()
        document = self._mount_document(window, plane=plane)
        document.add_measurement(
            Measurement(
                id="only_count",
                image_id=document.id,
                fiber_group_id=None,
                mode="count",
                measurement_kind="count",
                point_px=Point(2.0, 3.0),
            )
        )
        source = ImageAnalysisSourceContext(
            document_id=document.id,
            plane_sha256=plane.sha256(),
            source_signature=(document.id, "raster", plane.sha256()),
        )

        with self.assertRaisesRegex(ValueError, "至少需要 2 个"):
            window._spatial_distribution_scope_snapshot(
                document=document,
                plane=plane,
                source=source,
                parameters={
                    SPATIAL_POINT_SCOPE_KEY: "all",
                    SPATIAL_STUDY_AREA_MODE_KEY: "scope",
                },
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

        with patch.object(
            window,
            "_create_analysis_source_context",
            return_value=(document, plane, source),
        ), patch.object(
            window.image_analysis_task_controller,
            "start",
            return_value=SimpleNamespace(
                request_id="shape-provenance",
                generation=1,
            ),
        ) as start:
            window._start_image_analysis(
                AnalysisTool.SHAPE,
                parameters={},
                prompt_for_parameters=False,
            )

        start_kwargs = start.call_args.kwargs
        self.assertEqual(start_kwargs["raw_rings"], rings)
        self.assertEqual(start_kwargs["exact_area_px"], 87.0)
        self.assertEqual(start_kwargs["viewport_origin"], (0, 0))
        descriptor = start_kwargs["source_descriptor"]
        self.assertIsInstance(descriptor, AnalysisSourceDescriptor)
        self.assertEqual(descriptor.kind, "raster")
        self.assertEqual(descriptor.pixel_sha256, plane.sha256())
        region = start_kwargs["region_snapshot"]
        self.assertIsInstance(region, AnalysisRegionSnapshot)
        self.assertEqual(region.rings, rings)
        self.assertEqual(
            region.pixel_center_rule,
            "integer-coordinate-is-pixel-center",
        )
        self.assertEqual(region.components, 1)
        self.assertEqual(region.holes, 1)
        dependency = start_kwargs["dependency_signature"]
        self.assertIsInstance(dependency, AnalysisDependencySignature)
        self.assertEqual(
            dependency.dependencies["measurement_revisions"],
            {area.id: area.geometry_revision},
        )

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

    def test_rectangle_profile_uses_selected_roi_in_viewport_coordinates(
        self,
    ) -> None:
        window, _root = self._window()
        plane = self._plane()
        document = self._mount_document(window, plane=plane)
        roi = ProjectRoi(
            id="roi_profile",
            document_id=document.id,
            name="剖面矩形",
            geometry=RectangleRoiGeometry(103.0, 204.0, 8.0, 5.0),
        )
        window.project.project_rois.append(roi)
        window._selected_project_roi_ids = (roi.id,)
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

        _mask, _rings, _exact, reference, summary, points = (
            window._analysis_scope_snapshot(
                tool=AnalysisTool.PROFILE,
                document=document,
                plane=plane,
                source=source,
                profile_aggregation="rectangle_rows",
            )
        )

        self.assertEqual(points, ((3.0, 4.0), (11.0, 9.0)))
        self.assertEqual(reference.object_id, roi.id)
        self.assertEqual(summary, "矩形 ROI：剖面矩形")

    def test_composite_roi_provenance_tracks_transitive_revisions(self) -> None:
        window, _root = self._window()
        plane = self._plane()
        document = self._mount_document(window, plane=plane)
        left = ProjectRoi(
            id="roi_left",
            document_id=document.id,
            name="左侧",
            geometry=RectangleRoiGeometry(1.0, 1.0, 3.0, 3.0),
            revision=2,
        )
        right = ProjectRoi(
            id="roi_right",
            document_id=document.id,
            name="右侧",
            geometry=RectangleRoiGeometry(10.0, 1.0, 3.0, 3.0),
            revision=3,
        )
        composite = ProjectRoi(
            id="roi_union",
            document_id=document.id,
            name="组合区域",
            geometry=RoiBooleanExpression(
                RoiBooleanOperator.UNION,
                (left.id, right.id),
            ),
            revision=4,
        )
        window.project.project_rois.extend((left, right, composite))
        window._selected_project_roi_ids = (composite.id,)
        source = ImageAnalysisSourceContext(
            document_id=document.id,
            plane_sha256=plane.sha256(),
            source_signature=(document.id, "raster", plane.sha256()),
        )

        with patch.object(
            window,
            "_create_analysis_source_context",
            return_value=(document, plane, source),
        ), patch.object(
            window.image_analysis_task_controller,
            "start",
            return_value=SimpleNamespace(
                request_id="roi-provenance",
                generation=1,
            ),
        ) as start:
            window._start_image_analysis(
                AnalysisTool.INTENSITY,
                parameters={},
                prompt_for_parameters=False,
            )

        start_kwargs = start.call_args.kwargs
        region = start_kwargs["region_snapshot"]
        self.assertIsInstance(region, AnalysisRegionSnapshot)
        self.assertEqual(region.components, 2)
        self.assertEqual(region.holes, 0)
        self.assertEqual(region.rings, ())
        self.assertEqual(
            region.pixel_center_rule,
            "pixel-center-at-(column+0.5,row+0.5)",
        )
        dependency = start_kwargs["dependency_signature"].dependencies
        roi_dependencies = dependency["roi_transitive_refs"]
        self.assertEqual(
            set(roi_dependencies),
            {left.id, right.id, composite.id},
        )
        self.assertEqual(roi_dependencies[left.id]["revision"], 2)
        self.assertEqual(roi_dependencies[right.id]["revision"], 3)
        self.assertEqual(roi_dependencies[composite.id]["revision"], 4)
        self.assertEqual(
            roi_dependencies[composite.id]["operand_ids"],
            [left.id, right.id],
        )

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

    def test_ready_result_is_discarded_when_frozen_dependency_changed(
        self,
    ) -> None:
        window, session_root = self._window()
        plane = self._plane()
        document = self._mount_document(window, plane=plane)
        source = ImageAnalysisSourceContext(
            document_id=document.id,
            plane_sha256=plane.sha256(),
            source_signature=(document.id, "raster", plane.sha256()),
        )
        request_id = "analysis-stale-dependency"
        window._analysis_run_contexts[request_id] = ImageAnalysisRunContext(
            request_id=request_id,
            generation=9,
            tool=AnalysisTool.HISTOGRAM,
            source=source,
        )
        result = ImageAnalysisTaskResult(
            tool=AnalysisTool.HISTOGRAM,
            request_id=request_id,
            generation=9,
            document_id=document.id,
            source_pixel_revision=0,
            source_reference=None,
            source_descriptor=AnalysisSourceDescriptor(
                kind="raster",
                pixel_sha256=plane.sha256(),
            ),
            dependency_signature=AnalysisDependencySignature(
                calibration=None,
            ),
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
            "_analysis_frozen_inputs_are_current",
            return_value=False,
        ) as validate:
            window._on_image_analysis_ready(result)

        validate.assert_called_once()
        self.assertEqual(window.project.analysis_artifacts, [])
        self.assertEqual(list(session_root.rglob("*.npz")), [])
        self.assertIn("依赖已变化", window.statusBar().currentMessage())

    def test_project_load_marks_changed_source_sha_stale_and_dirty(
        self,
    ) -> None:
        window, _session_root = self._window()
        plane = self._plane()
        document = self._mount_document(window, plane=plane)
        artifact = ImageAnalysisTaskResult(
            tool=AnalysisTool.HISTOGRAM,
            request_id="loaded-analysis",
            generation=1,
            document_id=document.id,
            source_pixel_revision=0,
            source_reference=None,
            source_descriptor=AnalysisSourceDescriptor(
                kind="raster",
                pixel_sha256="f" * 64,
            ),
            calibration_signature=None,
            parameters={"bins": 4},
            scalars={"included_pixel_count": 192},
        ).to_analysis_artifact(artifact_id="loaded-artifact")
        window.project.analysis_artifacts.append(artifact)
        window._pending_project_load_snapshot = True

        window._mark_project_saved()

        refreshed = window.project.analysis_artifacts[0]
        self.assertFalse(refreshed.is_current)
        self.assertEqual(
            refreshed.stale_reason,
            "来源图片内容或冻结视窗已变化",
        )
        self.assertTrue(window._project_dirty())

    def test_analysis_deletion_is_atomic_dirty_and_preserves_shared_asset(
        self,
    ) -> None:
        window, session_root = self._window()
        document = self._mount_document(window)
        relative = "analysis/shared/curve.npz"
        source = session_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"shared-analysis-asset")
        reference = AnalysisAssetReference(
            kind=AnalysisAssetKind.CURVE,
            path=relative,
            sha256="a" * 64,
            media_type="application/x-npz",
            metadata={"schema": "fdm.test-curve.v1"},
        )
        artifacts = tuple(
            AnalysisArtifact(
                id=artifact_id,
                source_document_id=document.id,
                source_pixel_revision=0,
                tool_id="fdm.histogram",
                tool_version="1",
                scalars={"included_pixel_count": 1},
                assets=(reference,),
            )
            for artifact_id in ("shared-a", "shared-b")
        )
        window.project.analysis_artifacts.extend(artifacts)
        window._session_analysis_assets[relative] = source
        window.project.mark_extension_changed()
        window._mark_project_saved()
        self.assertFalse(window._project_dirty())

        self.assertEqual(
            window._delete_analysis_artifacts(("shared-a", "missing")),
            1,
        )
        self.assertEqual(
            [item.id for item in window.project.analysis_artifacts],
            ["shared-b"],
        )
        self.assertTrue(source.is_file())
        self.assertEqual(
            window._session_analysis_assets[relative],
            source,
        )
        self.assertTrue(window._project_dirty())

        self.assertEqual(
            window._delete_analysis_artifacts(("shared-b",)),
            1,
        )
        self.assertEqual(window.project.analysis_artifacts, [])
        self.assertFalse(source.exists())
        self.assertNotIn(relative, window._session_analysis_assets)

    def test_analysis_delete_confirmation_declares_no_undo(self) -> None:
        window, _root = self._window()
        document = self._mount_document(window)
        artifact = AnalysisArtifact(
            id="delete-warning",
            source_document_id=document.id,
            source_pixel_revision=0,
            tool_id="fdm.histogram",
            tool_version="1",
        )
        window.project.analysis_artifacts.append(artifact)

        with patch(
            "fdm.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ) as question:
            window._on_analysis_delete_requested(
                AnalysisActionRequest((artifact.id,))
            )

        self.assertIn(
            "不能通过“撤销”恢复",
            question.call_args.args[2],
        )
        self.assertEqual(window.project.analysis_artifacts, [artifact])

    def test_recalculate_preserves_artifact_output_field_selection(self) -> None:
        window, _root = self._window()
        document = self._mount_document(window)
        artifact = AnalysisArtifact(
            id="recalculate-selected-fields",
            source_document_id=document.id,
            source_pixel_revision=0,
            tool_id="fdm.intensity",
            tool_version="2",
            parameters={
                "channel": "luminance",
                ANALYSIS_OUTPUT_FIELDS_PARAMETER: [
                    "central_tendency",
                    "percentiles",
                ],
            },
        )
        window.project.analysis_artifacts.append(artifact)

        with patch.object(window, "_start_image_analysis") as start:
            window._on_analysis_recalculate_requested(
                AnalysisActionRequest((artifact.id,))
            )

        start.assert_called_once_with(
            AnalysisTool.INTENSITY,
            parameters=artifact.parameters,
            prompt_for_parameters=False,
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

    def test_saved_inline_maxima_applies_persisted_origin_once(self) -> None:
        window, _root = self._window()
        document = self._mount_document(window)
        result = ImageAnalysisTaskResult(
            tool=AnalysisTool.MAXIMA,
            request_id="maxima-inline",
            generation=1,
            document_id=document.id,
            source_pixel_revision=0,
            source_reference=None,
            calibration_signature=None,
            parameters={},
            scalars={
                "accepted_count": 1,
                "conversion_schema": "fdm.maxima-conversion.v2",
                "conversion_viewport_origin_x": 100,
                "conversion_viewport_origin_y": 200,
            },
            tables=(
                AnalysisTable(
                    name="极值点",
                    columns=(
                        "序号",
                        "X(px)",
                        "Y(px)",
                        "强度",
                    ),
                    rows=((1, 3.0, 4.0, 255.0),),
                ),
            ),
            conversion_payload=MaximaConversionPayload(
                points=((3.0, 4.0, 255.0),),
                viewport_origin=(100, 200),
            ),
        )
        artifact = result.to_analysis_artifact(
            artifact_id="saved-inline-maxima"
        )
        window.project.analysis_artifacts.append(artifact)
        window._analysis_conversion_offsets[artifact.id] = (100, 200)
        window._analysis_conversion_payloads.clear()

        with patch(
            "fdm.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window._on_analysis_convert_requested(
                AnalysisActionRequest((artifact.id,))
            )

        self.assertEqual(len(document.measurements), 1)
        point = document.measurements[0].point_px
        self.assertIsNotNone(point)
        self.assertEqual((point.x, point.y), (103.0, 204.0))

    def test_saved_particle_multi_selection_is_one_undoable_change(self) -> None:
        window, session_root = self._window()
        document = self._mount_document(window)
        first, _first_source = self._install_particle_artifact(
            window,
            document,
            artifact_id="saved-particle-a",
            asset_root=session_root,
            viewport_origin=(100, 200),
            local_x=1.0,
        )
        second, _second_source = self._install_particle_artifact(
            window,
            document,
            artifact_id="saved-particle-b",
            asset_root=session_root,
            viewport_origin=(100, 200),
            local_x=10.0,
        )
        window._analysis_conversion_offsets[first.id] = (100, 200)
        window._analysis_conversion_offsets[second.id] = (100, 200)
        window._analysis_conversion_payloads.clear()
        history = document.history
        self.assertIsNotNone(history)
        before_commands = history.command_count

        with patch(
            "fdm.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window._on_analysis_convert_requested(
                AnalysisActionRequest((first.id, second.id))
            )

        self.assertEqual(len(document.measurements), 2)
        self.assertEqual(
            [
                (item.polygon_px[0].x, item.polygon_px[0].y)
                for item in document.measurements
            ],
            [(101.0, 202.0), (110.0, 202.0)],
        )
        self.assertEqual(history.command_count, before_commands + 1)

        window.undo_current_document()

        self.assertEqual(document.measurements, [])
        self.assertEqual(document.fiber_groups, [])

    def test_multi_conversion_validation_failure_creates_nothing(self) -> None:
        window, session_root = self._window()
        document = self._mount_document(window)
        first, _first_source = self._install_particle_artifact(
            window,
            document,
            artifact_id="valid-particle",
            asset_root=session_root,
            viewport_origin=(0, 0),
            local_x=1.0,
        )
        second, second_source = self._install_particle_artifact(
            window,
            document,
            artifact_id="tampered-particle",
            asset_root=session_root,
            viewport_origin=(0, 0),
            local_x=10.0,
        )
        second_source.write_bytes(second_source.read_bytes() + b"tampered")
        history = document.history
        self.assertIsNotNone(history)
        before_commands = history.command_count

        with patch(
            "fdm.ui.main_window.QMessageBox.question",
        ) as question, patch(
            "fdm.ui.main_window.QMessageBox.warning",
        ):
            window._on_analysis_convert_requested(
                AnalysisActionRequest((first.id, second.id))
            )

        question.assert_not_called()
        self.assertEqual(document.measurements, [])
        self.assertEqual(document.fiber_groups, [])
        self.assertEqual(history.command_count, before_commands)

    def test_portable_analysis_export_resolves_project_assets(self) -> None:
        window, root = self._window()
        document = self._mount_document(window)
        project_path = root / "portable-source.fdmproj"
        window._project_path = project_path
        artifact, _source = self._install_particle_artifact(
            window,
            document,
            artifact_id="portable-particle",
            asset_root=project_assets_root(project_path),
            viewport_origin=(0, 0),
            local_x=1.0,
            map_session_asset=False,
        )
        target = root / "portable-analysis.zip"

        with patch(
            "fdm.ui.main_window.QFileDialog.getSaveFileName",
            return_value=(
                str(target),
                "便携分析包 ZIP (*.zip)",
            ),
        ):
            window._on_analysis_export_requested(
                AnalysisExportRequest((artifact.id,), None)
            )

        self.assertTrue(target.is_file())
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
        self.assertIn("manifest.json", names)
        self.assertIn("analysis-results.xlsx", names)
        self.assertTrue(any(name.startswith("assets/") for name in names))


if __name__ == "__main__":
    unittest.main()
