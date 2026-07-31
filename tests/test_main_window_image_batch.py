from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication, QMessageBox

from fdm.image_processing_models import ImageOperationSpec, ImageProcessingRecipe
from fdm.lifecycle import TransitionIntent
from fdm.models import ImageDocument, ProjectState, new_id
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.image_batch import (
    BatchRecipeRequest,
    BatchRasterInput,
    execute_batch_recipe,
)
from fdm.services.image_processing import ImageOperation
from fdm.services.raster_io import raster_plane_to_qimage, read_raster_file
from fdm.settings import AppSettings
from fdm.ui.image_batch_dialog import (
    BatchDocumentOption,
    ImageBatchProcessingDialog,
)
from fdm.ui.main_window import (
    DerivedImageCommitResult,
    ImageBatchRunContext,
    MainWindow,
)


class MainWindowImageBatchIntegrationTests(unittest.TestCase):
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

    def _events(self, count: int = 5) -> None:
        for _ in range(count):
            self.app.processEvents()

    def _window(self) -> tuple[MainWindow, Path]:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        window = MainWindow()
        window._session_processed_root = Path(directory.name) / "session"
        window._session_processed_root.mkdir()
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
    def _plane(value: int, *, width: int = 8, height: int = 6) -> RasterPlane:
        return RasterPlane(
            width=width,
            height=height,
            pixel_type=RasterPixelType.GRAY8,
            data=bytes([value]) * (width * height),
        )

    def _mount(
        self,
        window: MainWindow,
        name: str,
        plane: RasterPlane,
        *,
        digital_slide: bool = False,
    ) -> ImageDocument:
        document = ImageDocument(
            id=new_id("image"),
            path=f"/tmp/{name}.png",
            image_size=(plane.width, plane.height),
            document_kind="digital_slide" if digital_slide else "image",
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

    @staticmethod
    def _recipe() -> ImageProcessingRecipe:
        return ImageProcessingRecipe.from_operations(
            (ImageOperationSpec(ImageOperation.INVERT.value),)
        )

    def _result_and_context(
        self,
        window: MainWindow,
        documents: tuple[ImageDocument, ...],
        *,
        request_id: str = "batch-request",
        generation: int = 4,
    ):
        recipe = self._recipe()
        inputs = tuple(
            BatchRasterInput(
                document.id,
                Path(document.path).stem,
                window._rasters[document.id],
                source_path=document.path,
            )
            for document in documents
        )
        result = execute_batch_recipe(
            BatchRecipeRequest(
                request_id=request_id,
                generation=generation,
                recipe=recipe,
                inputs=inputs,
                available_disk_bytes=12 << 30,
            )
        )
        context = ImageBatchRunContext(
            request_id=request_id,
            generation=generation,
            recipe=recipe,
            source_sha256=tuple(
                (item.document_id, item.raster.sha256())
                for item in inputs
            ),
        )
        return result, context

    def _attach_dialog(
        self,
        window: MainWindow,
        documents: tuple[ImageDocument, ...],
        context: ImageBatchRunContext,
    ) -> ImageBatchProcessingDialog:
        dialog = ImageBatchProcessingDialog(
            self._recipe(),
            tuple(
                BatchDocumentOption(
                    document.id,
                    Path(document.path).stem,
                    "GRAY8 · 8×6",
                )
                for document in documents
            ),
            parent=window,
        )
        window._image_batch_dialog = dialog
        dialog.begin_request(context.request_id, context.generation)
        return dialog

    def test_batch_commits_each_candidate_without_mutating_sources_and_saves_assets(
        self,
    ) -> None:
        window, root = self._window()
        first = self._mount(window, "batch-first", self._plane(0x12))
        second = self._mount(window, "batch-second", self._plane(0x34))
        source_payloads = {
            document.id: document.to_dict()
            for document in (first, second)
        }
        result, context = self._result_and_context(
            window,
            (first, second),
        )
        dialog = self._attach_dialog(window, (first, second), context)
        window._image_batch_run_context = context

        window._on_image_batch_ready(result)
        self._events()

        self.assertEqual(
            first.to_dict(),
            source_payloads[first.id],
        )
        self.assertEqual(
            second.to_dict(),
            source_payloads[second.id],
        )
        derived = [
            document
            for document in window.project.documents
            if document.source_type == "project_asset"
        ]
        self.assertEqual(len(derived), 2)
        self.assertTrue(all(not item.measurements for item in derived))
        self.assertTrue(all(not item.overlay_annotations for item in derived))
        self.assertTrue(
            all(
                item.derivation is not None
                and item.derivation.source_document_id in {first.id, second.id}
                for item in derived
            )
        )
        self.assertEqual(dialog._documents_table.item(0, 3).text(), "已加入项目")
        self.assertEqual(dialog._documents_table.item(1, 3).text(), "已加入项目")

        project_path = root / "batch-project.fdmproj"
        self.assertTrue(window.save_project(str(project_path)))
        payload = json.loads(project_path.read_text(encoding="utf-8"))
        restored = ProjectState.from_dict(payload)
        restored_derived = [
            document
            for document in restored.documents
            if document.source_type == "project_asset"
        ]
        self.assertEqual(len(restored_derived), 2)
        for document in restored_derived:
            self.assertIsNotNone(document.derivation)
            asset = project_path.with_suffix(".assets") / document.path
            self.assertTrue(asset.is_file())
            loaded = read_raster_file(asset).require_success()
            self.assertEqual(
                loaded.plane.sha256(),
                document.derivation.result_sha256,
            )

    def test_one_commit_failure_does_not_block_other_candidates(self) -> None:
        window, _root = self._window()
        first = self._mount(window, "failure-first", self._plane(0x11))
        second = self._mount(window, "failure-second", self._plane(0x22))
        result, context = self._result_and_context(
            window,
            (first, second),
        )
        dialog = self._attach_dialog(window, (first, second), context)
        window._image_batch_run_context = context
        original = window._commit_derived_image

        def commit_one(**kwargs):
            if kwargs["source_document"].id == first.id:
                return DerivedImageCommitResult(None, "注入的单项写入失败")
            return original(**kwargs)

        with patch.object(
            window,
            "_commit_derived_image",
            side_effect=commit_one,
        ):
            window._on_image_batch_ready(result)

        self.assertEqual(
            len(
                [
                    item
                    for item in window.project.documents
                    if item.source_type == "project_asset"
                ]
            ),
            1,
        )
        self.assertEqual(dialog._documents_table.item(0, 3).text(), "提交失败")
        self.assertEqual(dialog._documents_table.item(1, 3).text(), "已加入项目")
        self.assertIn("提交失败 1 张", dialog._summary_label.text())

    def test_source_sha_change_discards_pending_candidate(self) -> None:
        window, _root = self._window()
        source = self._mount(window, "sha-source", self._plane(0x10))
        result, context = self._result_and_context(window, (source,))
        dialog = self._attach_dialog(window, (source,), context)
        window._image_batch_run_context = context
        window._rasters[source.id] = self._plane(0x99)

        with patch.object(
            window,
            "_commit_derived_image",
        ) as commit:
            window._on_image_batch_ready(result)

        commit.assert_not_called()
        self.assertEqual(len(window.project.documents), 1)
        self.assertEqual(
            dialog._documents_table.item(0, 3).text(),
            "来源已变化",
        )

    def test_stale_or_cancelled_result_never_commits(self) -> None:
        window, _root = self._window()
        source = self._mount(window, "cancel-source", self._plane(0x10))
        result, context = self._result_and_context(window, (source,))
        cancelled = type(result)(
            request_id=result.request_id,
            generation=result.generation,
            items=result.items,
            preflight=result.preflight,
            cancelled=True,
        )
        self._attach_dialog(window, (source,), context)
        window._image_batch_run_context = context

        with patch.object(
            window,
            "_commit_derived_image",
        ) as commit:
            window._on_image_batch_ready(cancelled)

        commit.assert_not_called()
        self.assertEqual(len(window.project.documents), 1)

    def test_calculator_recipe_is_rejected_before_dialog_or_worker(self) -> None:
        window, _root = self._window()
        self._mount(window, "calculator-source", self._plane(0x10))
        recipe = ImageProcessingRecipe.from_operations(
            (
                ImageOperationSpec(
                    ImageOperation.IMAGE_CALCULATOR.value,
                    {"secondary_document_id": "other"},
                ),
            )
        )
        with patch(
            "fdm.ui.main_window.QMessageBox.warning",
            return_value=QMessageBox.StandardButton.Ok,
        ) as warning:
            window._open_image_batch_dialog(recipe)
        self.assertIsNone(window._image_batch_dialog)
        warning.assert_called_once()
        self.assertIn("不支持图像计算器", warning.call_args.args[2])

    def test_document_close_and_transition_block_when_batch_cannot_stop(self) -> None:
        window, _root = self._window()
        source = self._mount(window, "active-source", self._plane(0x10))
        window._image_batch_run_context = ImageBatchRunContext(
            request_id="active",
            generation=1,
            recipe=self._recipe(),
            source_sha256=((source.id, window._rasters[source.id].sha256()),),
        )
        with patch.object(
            window,
            "_stop_image_batch_tasks",
            return_value=False,
        ):
            window._remove_document(source.id)
        self.assertIs(window.project.get_document(source.id), source)

        with patch.object(
            window,
            "_stop_image_batch_tasks",
            return_value=False,
        ):
            transition = window._prepare_transition(
                TransitionIntent.OPEN_PROJECT
            )
        self.assertFalse(transition.completed)
        self.assertTrue(transition.timed_out)
        self.assertIn("批处理", transition.reason)


if __name__ == "__main__":
    unittest.main()
