from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QSignalSpy
    from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QSpinBox

    from fdm.image_processing_models import ImageOperationSpec, ImageProcessingRecipe
    from fdm.services.image_batch import (
        BatchExecutionResult,
        BatchItemResourceEstimate,
        BatchItemResult,
        BatchItemStatus,
        BatchProgressPhase,
        BatchProgressUpdate,
        BatchResourceEstimate,
    )
    from fdm.ui.image_batch_dialog import (
        BatchDialogRequest,
        BatchDocumentOption,
        ImageBatchProcessingDialog,
    )
    from fdm.ui.image_processing_workbench import ImageProcessingWorkbench
    from fdm.raster import RasterPixelType, RasterPlane

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


def _recipe() -> "ImageProcessingRecipe":
    return ImageProcessingRecipe.from_operations(
        (
            ImageOperationSpec(
                "gaussian_blur",
                {
                    "sigma": 1.2,
                    "border_mode": "reflect",
                },
            ),
            ImageOperationSpec("invert"),
        )
    )


def _options() -> tuple["BatchDocumentOption", ...]:
    return (
        BatchDocumentOption(
            "doc-a",
            "显微图片 A",
            "GRAY8 · 1280×820",
        ),
        BatchDocumentOption(
            "doc-b",
            "显微图片 B",
            "RGB8 · 1920×1080",
            selected=False,
        ),
        BatchDocumentOption(
            "slide-c",
            "数字化切片 C",
            "数字化切片",
            is_digital_slide=True,
        ),
    )


def _preflight(
    document_ids: tuple[str, ...],
    *,
    disk_allowed: bool = True,
) -> "BatchResourceEstimate":
    return BatchResourceEstimate(
        items=tuple(
            BatchItemResourceEstimate(
                document_id=document_id,
                source_bytes=2 << 20,
                estimated_output_bytes=3 << 20,
                estimated_peak_bytes=48 << 20,
                allowed=True,
            )
            for document_id in document_ids
        ),
        estimated_total_output_bytes=3 << 20,
        available_disk_bytes=12 << 30,
        reserve_disk_bytes=2 << 30,
        disk_allowed=disk_allowed,
        reason="" if disk_allowed else "磁盘空间不足。",
    )


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class ImageBatchProcessingDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_is_non_modal_small_screen_safe_and_disables_slides(self) -> None:
        dialog = ImageBatchProcessingDialog(_recipe(), _options())
        try:
            self.assertFalse(dialog.isModal())
            self.assertEqual(dialog.windowTitle(), "批量应用图像处理配方")
            self.assertLessEqual(dialog.minimumWidth(), 760)
            self.assertLessEqual(dialog.minimumHeight(), 500)
            self.assertEqual(dialog.selected_document_ids(), ("doc-a",))
            self.assertIn("1. 高斯滤波", dialog._recipe_steps.item(0).text())  # noqa: SLF001
            self.assertIn("2. 反相", dialog._recipe_steps.item(1).text())  # noqa: SLF001
            slide_check = dialog._documents_table.item(2, 0)  # noqa: SLF001
            self.assertFalse(bool(slide_check.flags() & Qt.ItemFlag.ItemIsEnabled))
            self.assertIn(
                "冻结当前焦层",
                dialog._documents_table.item(2, 3).toolTip(),  # noqa: SLF001
            )
            self.assertFalse(dialog._start_button.isEnabled())  # noqa: SLF001
            self.assertLessEqual(
                dialog._documents_table.columnWidth(2),  # noqa: SLF001
                240,
            )
            self.assertFalse(dialog.findChildren(QSpinBox))
            self.assertFalse(dialog.findChildren(QDoubleSpinBox))
            self.assertFalse(dialog.findChildren(QComboBox))
        finally:
            dialog.close()

    def test_preflight_is_bound_to_current_selection_before_start(self) -> None:
        dialog = ImageBatchProcessingDialog(_recipe(), _options())
        preflight_spy = QSignalSpy(dialog.preflightRequested)
        start_spy = QSignalSpy(dialog.batchStartRequested)
        try:
            dialog._preflight_button.click()  # noqa: SLF001
            self.assertEqual(preflight_spy.count(), 1)
            self.assertEqual(start_spy.count(), 0)
            request = preflight_spy.at(0)[0]
            self.assertIsInstance(request, BatchDialogRequest)
            self.assertEqual(request.document_ids, ("doc-a",))

            dialog.apply_preflight(_preflight(("doc-a",)))
            self.assertTrue(dialog._start_button.isEnabled())  # noqa: SLF001
            dialog._start_button.click()  # noqa: SLF001
            self.assertEqual(start_spy.count(), 1)
            self.assertEqual(start_spy.at(0)[0].document_ids, ("doc-a",))

            dialog._documents_table.item(1, 0).setCheckState(  # noqa: SLF001
                Qt.CheckState.Checked
            )
            self.app.processEvents()
            self.assertFalse(dialog._start_button.isEnabled())  # noqa: SLF001
            dialog.apply_preflight(
                _preflight(("doc-a",)),
                document_ids=("doc-a",),
            )
            self.assertFalse(dialog._start_button.isEnabled())  # noqa: SLF001
        finally:
            dialog.close()

    def test_real_progress_and_result_are_request_scoped_and_show_pending_commit(self) -> None:
        dialog = ImageBatchProcessingDialog(_recipe(), _options())
        try:
            dialog.apply_preflight(_preflight(("doc-a",)))
            dialog.begin_request("request-current", 7)
            stale = BatchProgressUpdate(
                request_id="request-old",
                generation=6,
                phase=BatchProgressPhase.PROCESSING,
                item_index=1,
                item_total=1,
                document_id="doc-a",
                display_name="显微图片 A",
                completed_operations=1,
                total_operations=2,
                message="旧请求",
            )
            self.assertFalse(dialog.apply_progress(stale))
            self.assertEqual(
                dialog._documents_table.item(0, 3).text(),  # noqa: SLF001
                "等待处理",
            )
            current = BatchProgressUpdate(
                request_id="request-current",
                generation=7,
                phase=BatchProgressPhase.PROCESSING,
                item_index=1,
                item_total=1,
                document_id="doc-a",
                display_name="显微图片 A",
                completed_operations=1,
                total_operations=2,
                message="正在处理显微图片 A。",
            )
            self.assertTrue(dialog.apply_progress(current))
            self.assertEqual(
                dialog._documents_table.item(0, 3).text(),  # noqa: SLF001
                "处理中 1/2",
            )
            self.assertGreater(dialog._progress_bar.value(), 5)  # noqa: SLF001

            result = BatchExecutionResult(
                request_id="request-current",
                generation=7,
                items=(
                    BatchItemResult(
                        "doc-a",
                        "显微图片 A",
                        BatchItemStatus.SUCCESS,
                        "已生成待提交的派生图片候选。",
                        completed_operations=2,
                    ),
                ),
                preflight=_preflight(("doc-a",)),
            )
            self.assertTrue(dialog.apply_result(result))
            self.assertEqual(
                dialog._documents_table.item(0, 3).text(),  # noqa: SLF001
                "待加入项目",
            )
            self.assertIn("成功 1 张", dialog._summary_label.text())  # noqa: SLF001
            self.assertEqual(dialog._progress_bar.value(), 100)  # noqa: SLF001
            self.assertFalse(dialog._busy)  # noqa: SLF001
        finally:
            dialog.close()

    def test_busy_cancel_is_a_host_request_and_does_not_close_dialog(self) -> None:
        dialog = ImageBatchProcessingDialog(_recipe(), _options())
        cancel_spy = QSignalSpy(dialog.cancelRequested)
        try:
            dialog.apply_preflight(_preflight(("doc-a",)))
            dialog.begin_request("request-current", 2)
            dialog._cancel_button.click()  # noqa: SLF001
            self.assertEqual(cancel_spy.count(), 1)
            self.assertTrue(dialog._busy)  # noqa: SLF001
            self.assertIn("不会提交", dialog._status_label.text())  # noqa: SLF001
        finally:
            dialog.set_busy(False)
            dialog.close()

    def test_controller_terminal_signals_are_request_scoped(self) -> None:
        dialog = ImageBatchProcessingDialog(_recipe(), _options())
        try:
            dialog.apply_preflight(_preflight(("doc-a",)))
            dialog.begin_request("request-current", 3)
            self.assertFalse(
                dialog.apply_task_failure("request-old", "旧请求失败")
            )
            self.assertTrue(dialog._busy)  # noqa: SLF001
            self.assertTrue(
                dialog.apply_task_failure("request-current", "编码器异常")
            )
            self.assertFalse(dialog._busy)  # noqa: SLF001
            self.assertIn("编码器异常", dialog._summary_label.text())  # noqa: SLF001
            self.assertEqual(
                dialog._documents_table.item(0, 3).text(),  # noqa: SLF001
                "未完成",
            )
        finally:
            dialog.close()


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class WorkbenchRecipeSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_footer_emits_recipe_requests_without_writing_project(self) -> None:
        source = RasterPlane(
            width=8,
            height=6,
            pixel_type=RasterPixelType.GRAY8,
            data=bytes(range(48)),
        )
        dialog = ImageProcessingWorkbench(
            source,
            source_document_id="doc-a",
        )
        save_spy = QSignalSpy(dialog.recipeSaveRequested)
        load_spy = QSignalSpy(dialog.recipeLoadRequested)
        batch_spy = QSignalSpy(dialog.batchApplyRequested)
        try:
            self.assertFalse(dialog._save_recipe_button.isEnabled())  # noqa: SLF001
            self.assertTrue(dialog._load_recipe_button.isEnabled())  # noqa: SLF001
            dialog.apply_loaded_recipe(_recipe())
            self.assertEqual(
                tuple(item.operation_id for item in dialog.operation_steps()),
                ("gaussian_blur", "invert"),
            )
            self.assertEqual(dialog.operation_steps()[0].parameters["sigma"], 1.2)
            dialog._save_recipe_button.click()  # noqa: SLF001
            dialog._load_recipe_button.click()  # noqa: SLF001
            dialog._batch_apply_button.click()  # noqa: SLF001
            self.assertEqual(save_spy.count(), 1)
            self.assertEqual(load_spy.count(), 1)
            self.assertEqual(batch_spy.count(), 1)
            for emitted in (save_spy.at(0)[0], batch_spy.at(0)[0]):
                self.assertIsInstance(emitted, ImageProcessingRecipe)
                self.assertEqual(
                    tuple(item.operation_id for item in emitted.operations),
                    ("gaussian_blur", "invert"),
                )
                self.assertEqual(emitted.operations[0].parameters["sigma"], 1.2)
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
