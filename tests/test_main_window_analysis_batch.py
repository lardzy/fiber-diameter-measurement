from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox

from fdm.lifecycle import TransitionIntent
from fdm.models import ImageDocument, new_id
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.analysis_batch import (
    AnalysisBatchItemResult,
    AnalysisBatchResult,
    AnalysisSourceKind,
    execute_analysis_batch,
)
from fdm.services.digital_slide_store import (
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTile,
)
from fdm.services.raster_io import raster_plane_to_qimage
from fdm.settings import AppSettings
from fdm.ui.main_window import (
    AnalysisBatchRunContext,
    MainWindow,
)


class MainWindowAnalysisBatchTests(unittest.TestCase):
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

    def _events(self, count: int = 12) -> None:
        for _ in range(count):
            self.app.processEvents()

    def _window(self) -> tuple[MainWindow, Path]:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        window = MainWindow()
        window._session_analysis_root = Path(directory.name)
        window.resize(1100, 700)
        window.show()
        self._events()

        def cleanup() -> None:
            try:
                window._reset_workspace()
            except RuntimeError:
                pass
            window.close()
            self._events()

        self.addCleanup(cleanup)
        return window, Path(directory.name)

    @staticmethod
    def _plane(seed: int, *, width: int = 16, height: int = 12) -> RasterPlane:
        payload = bytes(
            (seed + row * 11 + column * 7) % 256
            for row in range(height)
            for column in range(width)
        )
        return RasterPlane(
            width=width,
            height=height,
            pixel_type=RasterPixelType.GRAY8,
            data=payload,
        )

    def _mount(
        self,
        window: MainWindow,
        name: str,
        plane: RasterPlane,
    ) -> ImageDocument:
        document = ImageDocument(
            id=new_id("image"),
            path=f"/tmp/{name}.png",
            image_size=(plane.width, plane.height),
        )
        document.initialize_runtime_state()
        document.mark_session_saved()
        document.mark_calibration_saved()
        window._mount_document(
            document,
            raster_plane_to_qimage(plane),
            tooltip=document.path,
            raster_plane=plane,
        )
        self._events()
        return document

    def _add_unmounted_slide(
        self,
        window: MainWindow,
        name: str,
    ) -> ImageDocument:
        document = ImageDocument(
            id=new_id("slide"),
            path=f"/tmp/{name}.fdmslide",
            image_size=(512, 512),
            document_kind="digital_slide",
        )
        document.initialize_runtime_state()
        window.project.documents.append(document)
        window._document_order.append(document.id)
        return document

    def _mount_slide(
        self,
        window: MainWindow,
        root: Path,
        name: str,
        *,
        color: str = "#335577",
    ) -> ImageDocument:
        path = root / f"{name}.fdmslide"
        store = DigitalSlideStore.create(
            path,
            DigitalSlideManifest(
                version=1,
                width=16,
                height=12,
                viewport_width=16,
                viewport_height=12,
                focus_levels=[0],
            ),
        )
        image = QImage(16, 12, QImage.Format.Format_RGB32)
        image.fill(QColor(color))
        store.write_tile(
            DigitalSlideTile(
                z_index=0,
                x=0,
                y=0,
                width=16,
                height=12,
            ),
            image,
        )
        store.close()
        window._add_digital_slide_document_from_path(
            path,
            document=None,
        )
        self._events()
        document = window.current_document()
        assert document is not None
        self.assertTrue(document.is_digital_slide())
        return document

    def test_analysis_menu_opens_matrix_and_disables_digital_slides(self) -> None:
        window, _root = self._window()
        first = self._mount(window, "first", self._plane(1))
        second = self._mount(window, "second", self._plane(2))
        slide = self._add_unmounted_slide(window, "slide")

        analysis_menu = next(
            menu
            for menu in window.menuBar().findChildren(QMenu)
            if menu.title() == "分析"
        )
        self.assertIn(window.analysis_batch_action, analysis_menu.actions())

        window._open_analysis_batch_dialog()
        dialog = window._analysis_batch_dialog
        self.assertIsNotNone(dialog)
        assert dialog is not None
        self.assertEqual(
            dialog.selected_item_ids(),
            (first.id, second.id),
        )
        slide_row = dialog._row_by_item_id[slide.id]
        slide_item = dialog.items_table.item(slide_row, 0)
        self.assertFalse(bool(slide_item.flags() & Qt.ItemFlag.ItemIsEnabled))
        self.assertIn(
            "不会隐式冻结",
            dialog.items_table.item(slide_row, 2).text(),
        )

    def test_digital_slide_only_entry_opens_explicit_freeze_prompt(self) -> None:
        window, _root = self._window()
        self._add_unmounted_slide(window, "slide-only")
        window._update_action_states()

        self.assertTrue(window.analysis_batch_action.isEnabled())
        window._open_analysis_batch_dialog()

        dialog = window._analysis_batch_dialog
        self.assertIsNotNone(dialog)
        assert dialog is not None
        self.assertTrue(dialog.freeze_viewport_button.isEnabled())
        self.assertEqual(dialog.selected_item_ids(), ())
        self.assertIn("显式冻结", dialog.summary_label.text())

    def test_user_can_freeze_only_current_slide_viewport_for_batch(self) -> None:
        window, root = self._window()
        slide = self._mount_slide(window, root, "slide")
        window._open_analysis_batch_dialog()
        dialog = window._analysis_batch_dialog
        assert dialog is not None

        store = window._slide_stores[slide.id]
        with patch(
            "fdm.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as question, patch.object(
            store,
            "render_viewport",
            wraps=store.render_viewport,
        ) as render_viewport:
            window._freeze_current_analysis_batch_viewport(dialog)

        question.assert_called_once()
        render_viewport.assert_called_once_with(
            x=0,
            y=0,
            width=16,
            height=12,
            z_index=0,
        )
        selected = dialog.selected_item_ids()
        self.assertEqual(len(selected), 1)
        frozen = window._analysis_batch_frozen_viewports[selected[0]]
        self.assertEqual(frozen.document_id, slide.id)
        self.assertEqual(
            (
                frozen.viewport.level,
                frozen.viewport.x,
                frozen.viewport.y,
                frozen.viewport.width,
                frozen.viewport.height,
            ),
            (0, 0, 0, 16, 12),
        )
        self.assertEqual(frozen.pixel_sha256, frozen.plane.sha256())

        invocations = window._analysis_batch_invocations(
            window._analysis_batch_recipes[-1],
            selected,
            request_id="frozen-slide",
            generation=3,
        )
        self.assertEqual(len(invocations), 1)
        self.assertEqual(
            invocations[0].source_kind,
            AnalysisSourceKind.DIGITAL_SLIDE,
        )
        self.assertEqual(invocations[0].viewport, frozen.viewport)

    def test_cancelled_slide_freeze_does_not_read_or_add_viewport(self) -> None:
        window, root = self._window()
        slide = self._mount_slide(window, root, "slide-cancel")
        window._open_analysis_batch_dialog()
        dialog = window._analysis_batch_dialog
        assert dialog is not None
        store = window._slide_stores[slide.id]

        with patch(
            "fdm.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Cancel,
        ), patch.object(
            store,
            "render_viewport",
            wraps=store.render_viewport,
        ) as render_viewport:
            window._freeze_current_analysis_batch_viewport(dialog)

        render_viewport.assert_not_called()
        self.assertEqual(window._analysis_batch_frozen_viewports, {})
        self.assertEqual(dialog.selected_item_ids(), ())

    def test_frozen_slide_viewport_batch_commits_persisted_source_identity(
        self,
    ) -> None:
        window, root = self._window()
        slide = self._mount_slide(window, root, "slide-commit")
        window._open_analysis_batch_dialog()
        dialog = window._analysis_batch_dialog
        assert dialog is not None
        with patch(
            "fdm.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window._freeze_current_analysis_batch_viewport(dialog)

        with patch.object(
            window,
            "_open_analysis_results_center",
            return_value=None,
        ):
            window._start_analysis_batch(
                dialog,
                "intensity-surface-v1",
            )
            self.assertTrue(window.analysis_batch_controller.wait_for_done())
            self._events(30)

        self.assertEqual(len(window.project.analysis_artifacts), 1)
        artifact = window.project.analysis_artifacts[0]
        self.assertEqual(artifact.source_document_id, slide.id)
        descriptor = artifact.source_descriptor
        self.assertIsNotNone(descriptor)
        assert descriptor is not None
        self.assertEqual(descriptor.kind, "digital_slide_viewport")
        self.assertEqual(descriptor.focus, 0)
        self.assertEqual(descriptor.origin, (0, 0))
        self.assertEqual(descriptor.viewport_size, (16, 12))
        self.assertEqual(
            descriptor.pixel_sha256,
            next(iter(window._analysis_batch_frozen_viewports.values()))
            .pixel_sha256,
        )

    def test_frozen_slide_batch_records_descriptor_and_detects_changed_pixels(
        self,
    ) -> None:
        window, root = self._window()
        slide = self._mount_slide(window, root, "slide-stale")
        window._open_analysis_batch_dialog()
        dialog = window._analysis_batch_dialog
        assert dialog is not None
        with patch(
            "fdm.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window._freeze_current_analysis_batch_viewport(dialog)

        with patch.object(
            window.analysis_batch_controller,
            "start",
            return_value=True,
        ):
            window._start_analysis_batch(dialog, "intensity-surface-v1")
        context = window._analysis_batch_run_context
        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(len(context.items), 1)
        item = context.items[0]
        descriptor = item.task_request.source_descriptor
        self.assertIsNotNone(descriptor)
        assert descriptor is not None
        self.assertEqual(descriptor.kind, "digital_slide_viewport")
        self.assertEqual(descriptor.store_id, slide.id)
        self.assertEqual(descriptor.focus, 0)
        self.assertEqual(descriptor.origin, (0, 0))
        self.assertEqual(descriptor.viewport_size, (16, 12))
        self.assertEqual(descriptor.pixel_sha256, item.source_sha256)
        self.assertTrue(window._analysis_batch_source_is_current(item))

        changed = QImage(16, 12, QImage.Format.Format_RGB32)
        changed.fill(QColor("#CC2244"))
        window._slide_stores[slide.id].write_tile(
            DigitalSlideTile(
                z_index=0,
                x=0,
                y=0,
                width=16,
                height=12,
            ),
            changed,
        )
        self.assertFalse(window._analysis_batch_source_is_current(item))
        window._analysis_batch_run_context = None
        dialog.set_busy(False)

    def test_surface_batch_commits_all_artifacts_once_without_mutating_sources(
        self,
    ) -> None:
        window, _root = self._window()
        first = self._mount(window, "first", self._plane(10))
        second = self._mount(window, "second", self._plane(20))
        source_payloads = {}
        for document in (first, second):
            payload = document.to_dict()
            # A delayed offscreen Qt resize can legitimately settle fit-to-view
            # while the modeless dialog is opening.  View state is not an
            # authoritative analysis source; keep the regression focused on
            # pixels, measurements, calibration and persistent provenance.
            payload.pop("view_state", None)
            source_payloads[document.id] = payload
        window._open_analysis_batch_dialog()
        dialog = window._analysis_batch_dialog
        assert dialog is not None
        before_next_state = window.project._next_extension_state_id

        with patch.object(
            window,
            "_open_analysis_results_center",
            return_value=None,
        ):
            window._start_analysis_batch(
                dialog,
                "intensity-surface-v1",
            )
            self.assertTrue(window.analysis_batch_controller.wait_for_done())
            self._events(30)

        self.assertEqual(len(window.project.analysis_artifacts), 2)
        self.assertEqual(
            window.project._next_extension_state_id,
            before_next_state + 1,
        )
        first_payload = first.to_dict()
        second_payload = second.to_dict()
        first_payload.pop("view_state", None)
        second_payload.pop("view_state", None)
        self.assertEqual(first_payload, source_payloads[first.id])
        self.assertEqual(second_payload, source_payloads[second.id])
        self.assertTrue(
            all(
                artifact.tool_id == "fdm.surface"
                and artifact.source_descriptor is not None
                and artifact.region_snapshot is not None
                and artifact.dependency_signature is not None
                for artifact in window.project.analysis_artifacts
            )
        )
        self.assertIn("已一次提交 2 项", dialog.summary_label.text())

    def test_histogram_batch_commits_one_result_per_selected_image(self) -> None:
        window, _root = self._window()
        first = self._mount(window, "histogram-first", self._plane(3))
        second = self._mount(window, "histogram-second", self._plane(9))
        window._open_analysis_batch_dialog()
        dialog = window._analysis_batch_dialog
        assert dialog is not None

        self.assertEqual(window.analysis_batch_action.text(), "批量分析…")
        self.assertEqual(dialog.windowTitle(), "批量分析")
        with patch.object(
            window,
            "_open_analysis_results_center",
            return_value=None,
        ):
            window._start_analysis_batch(dialog, "histogram-v2")
            self.assertTrue(window.analysis_batch_controller.wait_for_done())
            self._events(30)

        self.assertEqual(len(window.project.analysis_artifacts), 2)
        self.assertEqual(
            {
                artifact.source_document_id
                for artifact in window.project.analysis_artifacts
            },
            {first.id, second.id},
        )
        self.assertTrue(
            all(
                artifact.tool_id == "fdm.histogram"
                and artifact.tool_version == "2"
                and artifact.curves
                and artifact.tables
                for artifact in window.project.analysis_artifacts
            )
        )
        self.assertIn("已一次提交 2 项", dialog.summary_label.text())

    def test_multi_tool_recipe_commits_all_steps_for_one_source_atomically(
        self,
    ) -> None:
        window, _root = self._window()
        document = self._mount(
            window,
            "multi-tool",
            self._plane(27, width=32, height=32),
        )
        window._open_analysis_batch_dialog()
        dialog = window._analysis_batch_dialog
        assert dialog is not None
        before_next_state = window.project._next_extension_state_id

        with patch.object(
            window,
            "_open_analysis_results_center",
            return_value=None,
        ):
            window._start_analysis_batch(
                dialog,
                "directionality-and-glcm-v2",
            )
            context = window._analysis_batch_run_context
            self.assertIsNotNone(context)
            assert context is not None
            self.assertEqual(len(context.items), 1)
            self.assertEqual(
                tuple(
                    request.tool.value
                    for request in context.items[0].task_requests
                ),
                ("directionality", "glcm"),
            )
            self.assertEqual(
                tuple(
                    request.request_id.rsplit(":", 1)[-1]
                    for request in context.items[0].task_requests
                ),
                ("step-1", "step-2"),
            )
            self.assertTrue(window.analysis_batch_controller.wait_for_done())
            self._events(30)

        self.assertEqual(len(window.project.analysis_artifacts), 2)
        self.assertEqual(
            {
                (artifact.tool_id, artifact.tool_version)
                for artifact in window.project.analysis_artifacts
            },
            {
                ("fdm.directionality", "2"),
                ("fdm.glcm", "1"),
            },
        )
        self.assertTrue(
            all(
                artifact.source_document_id == document.id
                for artifact in window.project.analysis_artifacts
            )
        )
        self.assertEqual(
            window.project._next_extension_state_id,
            before_next_state + 1,
        )
        row = dialog._row_by_item_id[document.id]
        self.assertEqual(
            dialog.items_table.item(row, 2).text(),
            "已提交（2 项）",
        )

    def test_multi_tool_validation_failure_discards_source_assets_and_artifacts(
        self,
    ) -> None:
        window, root = self._window()
        document = self._mount(
            window,
            "multi-invalid",
            self._plane(31, width=32, height=32),
        )
        window._open_analysis_batch_dialog()
        dialog = window._analysis_batch_dialog
        assert dialog is not None
        captured = []
        with patch.object(
            window.analysis_batch_controller,
            "start",
            side_effect=lambda request: captured.append(request) or True,
        ):
            window._start_analysis_batch(
                dialog,
                "directionality-and-glcm-v2",
            )
        self.assertEqual(len(captured), 1)
        completed = execute_analysis_batch(captured[0])
        self.assertEqual(completed.success_count, 1)
        executions = completed.item_results[0].executions
        self.assertEqual(len(executions), 2)
        mismatched = replace(executions[1], request_id="wrong-step-request")
        invalid_item = AnalysisBatchItemResult(
            item_id=document.id,
            display_name=completed.item_results[0].display_name,
            success=True,
            execution=executions[0],
            executions=(executions[0], mismatched),
        )
        before_next_state = window.project._next_extension_state_id

        window._on_analysis_batch_ready(
            replace(completed, item_results=(invalid_item,))
        )

        self.assertEqual(window.project.analysis_artifacts, [])
        self.assertEqual(
            window.project._next_extension_state_id,
            before_next_state,
        )
        self.assertEqual(tuple(root.rglob("*.npz")), ())
        row = dialog._row_by_item_id[document.id]
        self.assertEqual(
            dialog.items_table.item(row, 2).text(),
            "提交失败",
        )

    def test_source_sha_change_discards_only_changed_batch_artifact(self) -> None:
        window, _root = self._window()
        changed = self._mount(window, "changed", self._plane(30))
        current = self._mount(window, "current", self._plane(40))
        window._open_analysis_batch_dialog()
        dialog = window._analysis_batch_dialog
        assert dialog is not None

        with patch.object(
            window,
            "_open_analysis_results_center",
            return_value=None,
        ):
            window._start_analysis_batch(
                dialog,
                "intensity-surface-v1",
            )
            self.assertTrue(window.analysis_batch_controller.wait_for_done())
            window._rasters[changed.id] = self._plane(99)
            self._events(30)

        self.assertEqual(len(window.project.analysis_artifacts), 1)
        self.assertEqual(
            window.project.analysis_artifacts[0].source_document_id,
            current.id,
        )
        changed_row = dialog._row_by_item_id[changed.id]
        self.assertEqual(
            dialog.items_table.item(changed_row, 2).text(),
            "来源已变化",
        )

    def test_late_or_cancelled_batch_callbacks_do_not_commit(self) -> None:
        window, _root = self._window()
        context = AnalysisBatchRunContext(
            request_id="current-request",
            generation=4,
            recipe_id="intensity-surface-v1",
            items=(),
        )
        window._analysis_batch_run_context = context
        late = AnalysisBatchResult(
            request_id="late-request",
            generation=3,
            recipe_id="intensity-surface-v1",
            item_results=(
                AnalysisBatchItemResult(
                    item_id="missing",
                    display_name="missing",
                    success=True,
                ),
            ),
        )

        window._on_analysis_batch_ready(late)
        self.assertIs(window._analysis_batch_run_context, context)
        self.assertEqual(window.project.analysis_artifacts, [])

        window._on_analysis_batch_cancelled(
            context.request_id,
            context.generation,
        )
        self.assertIsNone(window._analysis_batch_run_context)
        self.assertEqual(window.project.analysis_artifacts, [])

    def test_transition_is_blocked_when_batch_worker_cannot_stop(self) -> None:
        window, _root = self._window()

        with patch.object(
            window,
            "_stop_analysis_batch_tasks",
            return_value=False,
        ):
            transition = window._prepare_transition(
                TransitionIntent.OPEN_PROJECT
            )

        self.assertFalse(transition.completed)
        self.assertTrue(transition.timed_out)
        self.assertIn("批量分析", transition.reason)


if __name__ == "__main__":
    unittest.main()
