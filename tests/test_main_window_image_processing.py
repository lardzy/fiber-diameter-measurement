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

from fdm.geometry import Line, Point
from fdm.image_processing_models import (
    ImageOperationSpec,
    ImageProcessingRecipe,
    ProcessingRoiSnapshot,
    RasterSemantic,
)
from fdm.models import (
    Calibration,
    ImageDocument,
    Measurement,
    OverlayAnnotation,
    OverlayAnnotationKind,
    new_id,
)
from fdm.project_roi import (
    ProjectRoi,
    RectangleRoiGeometry,
    RoiBooleanExpression,
    RoiBooleanOperator,
)
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.image_processing import ImageOperation, image_operation_registry
from fdm.services.raster_io import numpy_to_raster_plane, raster_plane_to_numpy, raster_plane_to_qimage
from fdm.settings import AppSettings
from fdm.ui.image_processing_workbench import (
    WorkbenchTaskKind,
    WorkbenchTaskResult,
)
from fdm.ui.main_window import MainWindow
from fdm.ui.raster_derivation_dialogs import (
    Gray8RasterDocumentDescriptor,
    RasterChannelMergeRequest,
    RasterCopyDerivationRequest,
    RasterCopyScope,
)


class MainWindowImageProcessingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.settings = AppSettings(theme_mode="dark")
        self.load_patch = patch(
            "fdm.ui.main_window.AppSettingsIO.load",
            return_value=self.settings,
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
        session_directory = TemporaryDirectory()
        self.addCleanup(session_directory.cleanup)
        window = MainWindow()
        window._session_processed_root = Path(session_directory.name)
        window.resize(1280, 720)
        window.show()
        self._process_events()

        def cleanup() -> None:
            window._reset_workspace()
            window.close()
            self._process_events()

        self.addCleanup(cleanup)
        return window, Path(session_directory.name)

    @staticmethod
    def _plane(
        width: int = 8,
        height: int = 6,
        *,
        pixel_type: RasterPixelType = RasterPixelType.GRAY8,
        value: int = 0x33,
    ) -> RasterPlane:
        return RasterPlane(
            width=width,
            height=height,
            pixel_type=pixel_type,
            data=bytes([value & 0xFF])
            * (width * height * pixel_type.bytes_per_pixel),
        )

    def _mount_document(
        self,
        window: MainWindow,
        *,
        name: str,
        plane: RasterPlane | None = None,
        calibration: Calibration | None = None,
        document_kind: str = "image",
    ) -> ImageDocument:
        authoritative_plane = plane or self._plane()
        document = ImageDocument(
            id=new_id("image"),
            path=f"/tmp/{name}.png",
            image_size=(
                authoritative_plane.width,
                authoritative_plane.height,
            ),
            document_kind=document_kind,
            calibration=calibration,
        )
        document.initialize_runtime_state()
        document.mark_session_saved()
        document.mark_calibration_saved()
        window._mount_document(
            document,
            raster_plane_to_qimage(authoritative_plane),
            tooltip=document.path,
            raster_plane=authoritative_plane,
        )
        self._process_events()
        return document

    def test_rgb_color_balance_uses_pixel_processing_workbench(self) -> None:
        window, _session_root = self._window()
        with (
            patch.object(window, "_open_image_processing_workbench") as workbench,
            patch.object(window, "_open_display_adjustment_dialog") as display,
        ):
            window._open_registered_image_operation(ImageOperation.COLOR_BALANCE)

        workbench.assert_called_once_with(ImageOperation.COLOR_BALANCE)
        display.assert_not_called()

    @staticmethod
    def _result(
        document_id: str,
        raster: RasterPlane,
        operation: ImageOperationSpec,
        *,
        generation: int = 1,
    ) -> WorkbenchTaskResult:
        return WorkbenchTaskResult(
            kind=WorkbenchTaskKind.FINAL,
            request_id=f"request-{generation}",
            generation=generation,
            source_document_id=document_id,
            raster=raster,
            recipe=ImageProcessingRecipe.from_operations((operation,)),
        )

    def _rich_source_document(
        self,
        window: MainWindow,
    ) -> tuple[ImageDocument, RasterPlane, ProjectRoi]:
        plane = self._plane(value=0x24)
        calibration = Calibration(
            mode="preset",
            pixels_per_unit=4.0,
            unit="um",
            source_label="激光共聚焦 40x",
        )
        document = ImageDocument(
            id=new_id("image"),
            path="/tmp/source-rich.png",
            image_size=(plane.width, plane.height),
            calibration=calibration,
        )
        document.initialize_runtime_state()
        first_group = document.create_group(
            color="#EF476F",
            label="纤维 A",
        )
        document.create_group(
            color="#118AB2",
            label="纤维 B",
        )
        document.add_measurement(
            Measurement(
                id=new_id("measurement"),
                image_id=document.id,
                fiber_group_id=first_group.id,
                mode="manual",
                line_px=Line(Point(1, 1), Point(6, 1)),
            )
        )
        document.add_overlay_annotation(
            OverlayAnnotation(
                id=new_id("overlay"),
                image_id=document.id,
                kind=OverlayAnnotationKind.RECT,
                start_px=Point(1, 2),
                end_px=Point(5, 5),
            )
        )
        document.mark_session_saved()
        document.mark_calibration_saved()
        window._mount_document(
            document,
            raster_plane_to_qimage(plane),
            tooltip=document.path,
            raster_plane=plane,
        )
        roi = ProjectRoi(
            id=new_id("roi"),
            document_id=document.id,
            name="源图片 ROI",
            geometry=RectangleRoiGeometry(1, 1, 4, 3),
        )
        window.project.project_rois.append(roi)
        window.project.mark_extension_changed()
        self._process_events()
        return document, plane, roi

    def test_derived_image_preserves_source_and_only_inherits_safe_metadata(
        self,
    ) -> None:
        window, _session_root = self._window()
        source, source_plane, source_roi = self._rich_source_document(window)
        source_payload_before = source.to_dict()
        source_sha_before = window._rasters[source.id].sha256()
        source_rois_before = tuple(window.project.project_rois)
        source_measurement = source.measurements[0]
        source_overlay = source.overlay_annotations[0]
        source_group_ids = {group.id for group in source.fiber_groups}

        window._selected_project_roi_ids = (source_roi.id,)
        window._refresh_project_roi_ui()
        window._roi_manager.select_rois((source_roi.id,))
        context_result = window._create_processing_source_context()
        self.assertIsNotNone(context_result)
        context, _roi_mask, _roi_summary = context_result
        window._image_processing_source_context = context
        result_plane = self._plane(value=0x81)
        operation = ImageOperationSpec(
            ImageOperation.FLIP_HORIZONTAL.value,
            {},
        )

        window._on_derived_image_ready(
            self._result(source.id, result_plane, operation)
        )
        self._process_events()

        self.assertEqual(source.to_dict(), source_payload_before)
        self.assertIs(window._rasters[source.id], source_plane)
        self.assertEqual(window._rasters[source.id].sha256(), source_sha_before)
        self.assertIs(window.project.get_document(source.id), source)
        self.assertIs(source.measurements[0], source_measurement)
        self.assertIs(source.overlay_annotations[0], source_overlay)
        self.assertEqual(tuple(window.project.project_rois), source_rois_before)
        self.assertIs(window.project.get_project_roi(source_roi.id), source_roi)

        self.assertEqual(len(window.project.documents), 2)
        derived = window.current_document()
        self.assertIsNotNone(derived)
        self.assertIsNot(derived, source)
        self.assertEqual(derived.source_type, "project_asset")
        self.assertTrue(derived.path.startswith("processed/"))
        self.assertEqual(
            window._rasters[derived.id].sha256(),
            result_plane.sha256(),
        )

        self.assertIsNotNone(derived.calibration)
        self.assertIsNot(derived.calibration, source.calibration)
        self.assertEqual(
            derived.calibration.to_dict(),
            source.calibration.to_dict(),
        )
        self.assertEqual(
            [
                (group.number, group.label, group.color)
                for group in derived.sorted_groups()
            ],
            [
                (group.number, group.label, group.color)
                for group in source.sorted_groups()
            ],
        )
        self.assertTrue(
            source_group_ids.isdisjoint(
                {group.id for group in derived.fiber_groups}
            )
        )
        self.assertTrue(
            all(
                group.image_id == derived.id
                and not group.measurement_ids
                for group in derived.fiber_groups
            )
        )
        self.assertEqual(derived.measurements, [])
        self.assertEqual(derived.overlay_annotations, [])
        self.assertIsNone(derived.scale_overlay_anchor)
        self.assertFalse(
            any(
                roi.document_id == derived.id
                for roi in window.project.project_rois
            )
        )
        self.assertIsNotNone(derived.derivation)
        self.assertEqual(derived.derivation.source_document_id, source.id)
        self.assertEqual(derived.derivation.source_sha256, source_plane.sha256())
        self.assertEqual(
            derived.derivation.recipe.operations,
            (operation,),
        )
        self.assertIsNotNone(derived.derivation.roi_snapshot)
        self.assertEqual(
            derived.derivation.roi_snapshot.source_id,
            source_roi.id,
        )
        self.assertTrue(window._session_processed_assets[derived.id].is_file())

    def test_every_registered_workbench_operation_has_a_main_window_action(
        self,
    ) -> None:
        window, _session_root = self._window()

        expected = set(image_operation_registry()) - {
            ImageOperation.COPY.value
        }
        self.assertEqual(set(window._image_operation_actions), expected)
        for operation in (
            ImageOperation.ADAPTIVE_THRESHOLD,
            ImageOperation.ROLLING_BALL_BACKGROUND_SUBTRACT,
            ImageOperation.WATERSHED_V2,
            ImageOperation.LOG_V2,
            ImageOperation.MORPHOLOGICAL_RECONSTRUCTION,
            ImageOperation.FLAT_FIELD_CORRECTION,
            ImageOperation.FFT_POWER_SPECTRUM,
        ):
            action = window._image_operation_actions[operation.value]
            self.assertTrue(action.text().strip())

    def test_changed_source_discards_late_result_without_writing_asset(
        self,
    ) -> None:
        window, session_root = self._window()
        source = self._mount_document(
            window,
            name="stale-source",
            plane=self._plane(value=0x11),
        )
        context_result = window._create_processing_source_context()
        self.assertIsNotNone(context_result)
        context, _roi_mask, _roi_summary = context_result
        window._image_processing_source_context = context
        source_payload_before = source.to_dict()

        window._rasters[source.id] = self._plane(value=0x22)
        operation = ImageOperationSpec(
            ImageOperation.FLIP_VERTICAL.value,
            {},
        )
        with (
            patch(
                "fdm.ui.main_window.QMessageBox.warning",
                return_value=QMessageBox.StandardButton.Ok,
            ) as warning,
            patch(
                "fdm.ui.main_window.write_native_raster_asset",
            ) as asset_writer,
        ):
            window._on_derived_image_ready(
                self._result(
                    source.id,
                    self._plane(value=0x44),
                    operation,
                    generation=7,
                )
            )

        self.assertEqual(len(window.project.documents), 1)
        self.assertEqual(source.to_dict(), source_payload_before)
        self.assertEqual(window._session_processed_assets, {})
        self.assertEqual(list(session_root.rglob("*")), [])
        asset_writer.assert_not_called()
        warning.assert_called_once()
        self.assertIn("已丢弃晚到结果", warning.call_args.args[2])

    def test_roi_copy_commits_cropped_authoritative_pixels_only(self) -> None:
        window, _session_root = self._window()
        source, source_plane, roi = self._rich_source_document(window)
        source.raster_semantic = RasterSemantic.BINARY_MASK
        window._selected_project_roi_ids = (roi.id,)
        window._refresh_project_roi_ui()
        window._roi_manager.select_rois((roi.id,))
        context_result = window._create_processing_source_context()
        self.assertIsNotNone(context_result)
        context, roi_mask, _summary = context_result
        frozen_roi = window._frozen_raster_roi(roi_mask)
        request = RasterCopyDerivationRequest(
            source=source_plane,
            source_sha256=source_plane.sha256(),
            scope=RasterCopyScope.ROI_BOUNDS,
            bounds=frozen_roi.bounds,
        )

        window._commit_raster_copy_request(request, context)

        self.assertEqual(len(window.project.documents), 2)
        derived = window.current_document()
        self.assertIsNotNone(derived)
        self.assertEqual(derived.image_size, (4, 3))
        self.assertEqual(source.image_size, (8, 6))
        self.assertEqual(derived.measurements, [])
        self.assertIs(
            derived.raster_semantic,
            RasterSemantic.BINARY_MASK,
        )
        self.assertIs(
            derived.derivation.result_semantic,
            RasterSemantic.BINARY_MASK,
        )
        self.assertEqual(
            derived.derivation.recipe.operations[-1].operation_id,
            ImageOperation.COPY.value,
        )
        self.assertIsInstance(
            derived.derivation.roi_snapshot,
            ProcessingRoiSnapshot,
        )
        self.assertEqual(
            derived.derivation.roi_snapshot.source_id,
            roi.id,
        )

    def test_processing_snapshot_freezes_composite_roi_dependencies(
        self,
    ) -> None:
        window, _session_root = self._window()
        source, source_plane, first = self._rich_source_document(window)
        second = ProjectRoi(
            id=new_id("roi"),
            document_id=source.id,
            name="第二块",
            geometry=RectangleRoiGeometry(5, 1, 2, 2),
            revision=3,
        )
        composite = ProjectRoi(
            id=new_id("roi"),
            document_id=source.id,
            name="组合区域",
            geometry=RoiBooleanExpression(
                RoiBooleanOperator.UNION,
                (first.id, second.id),
            ),
            revision=4,
        )
        window.project.project_rois.extend((second, composite))
        window._selected_project_roi_ids = (composite.id,)
        window._refresh_project_roi_ui()
        window._roi_manager.select_rois((composite.id,))

        context_result = window._create_processing_source_context()

        self.assertIsNotNone(context_result)
        context, roi_mask, summary = context_result
        self.assertIsNotNone(roi_mask)
        self.assertEqual(summary, "ROI：组合区域")
        snapshot = context.roi_snapshot
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.source_kind, "project_roi")
        self.assertEqual(snapshot.source_id, composite.id)
        self.assertEqual(snapshot.revision, 4)
        self.assertEqual(snapshot.bounds, (1, 1, 6, 3))
        self.assertEqual(
            snapshot.dependency_revisions,
            tuple(sorted(((first.id, 0), (second.id, 3)))),
        )
        self.assertEqual(len(snapshot.mask_sha256), 64)
        self.assertTrue(window._processing_source_is_current(context))

        replacement = second.replace_geometry(
            RectangleRoiGeometry(4, 1, 3, 2)
        )
        window.project.project_rois[
            window.project.project_rois.index(second)
        ] = replacement
        self.assertFalse(window._processing_source_is_current(context))

        # Full-image copy does not consume the selected ROI and therefore
        # remains valid even when that ROI changes while the dialog is open.
        self.assertTrue(
            window._processing_source_is_current(
                context,
                require_roi_current=False,
            )
        )

    def test_processing_snapshot_freezes_selected_area_geometry(self) -> None:
        window, _session_root = self._window()
        source = self._mount_document(
            window,
            name="area-source",
            plane=self._plane(width=8, height=6),
        )
        area = Measurement(
            id=new_id("measurement"),
            image_id=source.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="area",
            polygon_px=[
                Point(1, 1),
                Point(6, 1),
                Point(6, 5),
                Point(1, 5),
            ],
        )
        source.add_measurement(area)
        source.view_state.selected_measurement_id = area.id

        context_result = window._create_processing_source_context()

        self.assertIsNotNone(context_result)
        context, roi_mask, _summary = context_result
        self.assertIsNotNone(roi_mask)
        snapshot = context.roi_snapshot
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.source_kind, "measurement_area")
        self.assertEqual(snapshot.source_id, area.id)
        self.assertEqual(snapshot.revision, area.geometry_revision)
        self.assertEqual(snapshot.bounds, (1, 1, 5, 4))
        self.assertEqual(snapshot.dependency_revisions, ())

        area.replace_area_geometry(
            polygon_px=[
                Point(2, 1),
                Point(6, 1),
                Point(6, 5),
                Point(2, 5),
            ],
        )
        self.assertFalse(window._processing_source_is_current(context))

    def test_split_and_merge_channels_create_new_documents(self) -> None:
        window, _session_root = self._window()
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        rgb[..., 0] = 10
        rgb[..., 1] = 20
        rgb[..., 2] = 30
        source = self._mount_document(
            window,
            name="rgb-source",
            plane=numpy_to_raster_plane(rgb),
        )
        ignored_roi = ProjectRoi(
            id=new_id("roi"),
            document_id=source.id,
            name="不应进入通道派生来源",
            geometry=RectangleRoiGeometry(0, 0, 2, 2),
        )
        window.project.project_rois.append(ignored_roi)
        window._selected_project_roi_ids = (ignored_roi.id,)
        window._refresh_project_roi_ui()
        window._roi_manager.select_rois((ignored_roi.id,))
        with patch(
            "fdm.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window._split_current_rgb_channels()

        self.assertEqual(len(window.project.documents), 4)
        split_documents = [
            document
            for document in window.project.documents
            if document.id != source.id
        ]
        self.assertEqual(
            sorted(
                int(raster_plane_to_numpy(window._rasters[item.id])[0, 0])
                for item in split_documents
            ),
            [10, 20, 30],
        )
        descriptors = tuple(
            Gray8RasterDocumentDescriptor(
                document_id=document.id,
                display_name=window._document_display_name(document),
                width=window._rasters[document.id].width,
                height=window._rasters[document.id].height,
                pixel_sha256=window._rasters[document.id].sha256(),
                calibration_signature="uncalibrated",
            )
            for document in split_documents
        )

        window._commit_rgb_channel_merge(
            RasterChannelMergeRequest(
                red=next(
                    item
                    for item in descriptors
                    if int(
                        raster_plane_to_numpy(
                            window._rasters[item.document_id]
                        )[0, 0]
                    )
                    == 10
                ),
                green=next(
                    item
                    for item in descriptors
                    if int(
                        raster_plane_to_numpy(
                            window._rasters[item.document_id]
                        )[0, 0]
                    )
                    == 20
                ),
                blue=next(
                    item
                    for item in descriptors
                    if int(
                        raster_plane_to_numpy(
                            window._rasters[item.document_id]
                        )[0, 0]
                    )
                    == 30
                ),
            )
        )

        self.assertEqual(len(window.project.documents), 5)
        merged = window.current_document()
        self.assertIsNotNone(merged)
        np.testing.assert_array_equal(
            raster_plane_to_numpy(window._rasters[merged.id]),
            rgb,
        )
        self.assertTrue(
            all(
                document.derivation is not None
                and document.derivation.roi_snapshot is None
                for document in window.project.documents
                if document.id != source.id
            )
        )

    def test_non_uniform_resize_requires_explicit_calibration_clear(
        self,
    ) -> None:
        window, session_root = self._window()
        calibration = Calibration(
            mode="preset",
            pixels_per_unit=8.0,
            unit="um",
            source_label="20x",
        )
        source = self._mount_document(
            window,
            name="non-uniform",
            plane=self._plane(width=8, height=6, value=0x20),
            calibration=calibration,
        )
        context_result = window._create_processing_source_context()
        self.assertIsNotNone(context_result)
        context, _roi_mask, _roi_summary = context_result
        window._image_processing_source_context = context
        operation = ImageOperationSpec(
            ImageOperation.RESIZE.value,
            {
                "width": 4,
                "height": 6,
                "interpolation": "nearest",
            },
        )
        result = self._result(
            source.id,
            self._plane(width=4, height=6, value=0x70),
            operation,
        )
        source_payload_before = source.to_dict()

        with patch(
            "fdm.ui.main_window.write_native_raster_asset",
        ) as asset_writer:
            self._run_calibration_choice(
                window,
                selected_label="取消",
                callback=lambda: window._on_derived_image_ready(result),
            )

        self.assertEqual(len(window.project.documents), 1)
        self.assertEqual(source.to_dict(), source_payload_before)
        self.assertEqual(window._session_processed_assets, {})
        self.assertEqual(list(session_root.rglob("*")), [])
        asset_writer.assert_not_called()

        default_label = self._run_calibration_choice(
            window,
            selected_label="清除标定并继续",
            callback=lambda: window._on_derived_image_ready(result),
        )
        self._process_events()

        self.assertEqual(default_label, "取消")
        self.assertEqual(len(window.project.documents), 2)
        derived = window.current_document()
        self.assertIsNotNone(derived)
        self.assertIsNone(derived.calibration)
        self.assertEqual(source.to_dict(), source_payload_before)
        self.assertEqual(source.calibration.to_dict(), calibration.to_dict())
        self.assertTrue(window._session_processed_assets[derived.id].is_file())

    @staticmethod
    def _run_calibration_choice(
        window: MainWindow,
        *,
        selected_label: str,
        callback,
    ) -> str | None:
        buttons: dict[str, object] = {}
        default_label: list[str | None] = []
        original_add_button = QMessageBox.addButton

        def tracking_add_button(
            box: QMessageBox,
            text: str,
            role: QMessageBox.ButtonRole,
        ):
            button = original_add_button(box, text, role)
            buttons[text] = button
            return button

        def fake_exec(box: QMessageBox) -> int:
            default_button = box.defaultButton()
            default_label.append(
                default_button.text()
                if default_button is not None
                else None
            )
            return 0

        def fake_clicked_button(_box: QMessageBox):
            return buttons[selected_label]

        with (
            patch(
                "fdm.ui.main_window.QMessageBox.addButton",
                new=tracking_add_button,
            ),
            patch(
                "fdm.ui.main_window.QMessageBox.exec",
                new=fake_exec,
            ),
            patch(
                "fdm.ui.main_window.QMessageBox.clickedButton",
                new=fake_clicked_button,
            ),
        ):
            callback()
        return default_label[0] if default_label else None

    def test_image_calculator_only_exposes_compatible_second_images(
        self,
    ) -> None:
        window, _session_root = self._window()
        source_calibration = Calibration(
            mode="preset",
            pixels_per_unit=5.0,
            unit="um",
            source_label="source",
        )
        source_plane = self._plane(value=0x10)
        source = self._mount_document(
            window,
            name="calculator-source",
            plane=source_plane,
            calibration=source_calibration,
        )
        eligible = self._mount_document(
            window,
            name="calculator-eligible",
            plane=self._plane(value=0x20),
            calibration=Calibration(
                mode="derived",
                pixels_per_unit=5.0,
                unit="um",
                source_label="compatible",
            ),
        )
        self._mount_document(
            window,
            name="calculator-wrong-size",
            plane=self._plane(width=9, height=6, value=0x30),
            calibration=source_calibration.clone(),
        )
        self._mount_document(
            window,
            name="calculator-wrong-type",
            plane=self._plane(
                pixel_type=RasterPixelType.RGB8,
                value=0x40,
            ),
            calibration=source_calibration.clone(),
        )
        self._mount_document(
            window,
            name="calculator-wrong-calibration",
            plane=self._plane(value=0x50),
            calibration=Calibration(
                mode="preset",
                pixels_per_unit=6.0,
                unit="um",
                source_label="different",
            ),
        )
        self._mount_document(
            window,
            name="calculator-slide",
            plane=self._plane(value=0x60),
            calibration=source_calibration.clone(),
            document_kind="digital_slide",
        )

        candidates, names = window._processing_secondary_images(
            source,
            source_plane,
        )

        self.assertEqual(tuple(candidates), (eligible.id,))
        self.assertIs(candidates[eligible.id], window._rasters[eligible.id])
        self.assertEqual(tuple(names), (eligible.id,))
        self.assertIn("calculator-eligible", names[eligible.id])

        window._set_current_document(source.id)
        self._process_events()
        window._open_image_processing_workbench(
            ImageOperation.IMAGE_CALCULATOR
        )
        workbench = window._image_processing_workbench
        self.assertIsNotNone(workbench)
        self.assertEqual(tuple(workbench._secondary_images), (eligible.id,))
        self.assertEqual(len(workbench.operation_steps()), 1)
        calculator_step = workbench.operation_steps()[0]
        self.assertEqual(
            calculator_step.operation_id,
            ImageOperation.IMAGE_CALCULATOR.value,
        )
        self.assertEqual(
            calculator_step.parameters["secondary_document_id"],
            eligible.id,
        )


if __name__ == "__main__":
    unittest.main()
