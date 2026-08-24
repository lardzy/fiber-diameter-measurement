"""Standalone batch-analysis configuration and progress dialog."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from fdm.services.analysis_batch import (
    AnalysisBatchProgress,
    AnalysisBatchResult,
    AnalysisInvocation,
    AnalysisRecipe,
)
from fdm.ui.widgets import NoWheelComboBox


@dataclass(frozen=True, slots=True)
class AnalysisBatchCommitUpdate:
    item_id: str
    status: str
    detail: str = ""


class AnalysisBatchDialog(QDialog):
    runRequested = Signal(str)
    cancelRequested = Signal()
    recipeChanged = Signal(str)
    freezeViewportRequested = Signal()
    exportRequested = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("批量分析")
        self.setObjectName("analysisBatchDialog")
        self.resize(720, 480)
        self._recipes: dict[str, AnalysisRecipe] = {}
        self._row_by_item_id: dict[str, int] = {}
        self._available_item_ids: set[str] = set()
        self._last_result: AnalysisBatchResult | None = None

        self.recipe_combo = NoWheelComboBox(self)
        self.recipe_combo.setObjectName("analysisBatchRecipeCombo")
        self.recipe_combo.currentIndexChanged.connect(
            self._emit_recipe_changed
        )
        form = QFormLayout()
        form.addRow("分析配方：", self.recipe_combo)

        self.items_table = QTableWidget(0, 3, self)
        self.items_table.setObjectName("analysisBatchItemsTable")
        self.items_table.setHorizontalHeaderLabels(("项目", "输入范围", "状态"))
        self.items_table.itemChanged.connect(self._update_run_button)

        self.progress = QProgressBar(self)
        self.progress.setObjectName("analysisBatchProgress")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.summary_label = QLabel("尚未运行。", self)
        self.summary_label.setObjectName("analysisBatchSummary")

        self.buttons = QDialogButtonBox(self)
        self.run_button = self.buttons.addButton(
            "开始分析",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self.run_button.setObjectName("analysisBatchRunButton")
        self.cancel_button = self.buttons.addButton(
            "取消任务",
            QDialogButtonBox.ButtonRole.RejectRole,
        )
        self.cancel_button.setObjectName("analysisBatchCancelButton")
        self.cancel_button.setEnabled(False)
        self.export_button = self.buttons.addButton(
            "导出…",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.export_button.setObjectName("analysisBatchExportButton")
        self.export_button.setToolTip("导出总览、逐图片、逐 ROI 和失败明细工作簿")
        self.export_button.setEnabled(False)
        self.freeze_viewport_button = self.buttons.addButton(
            "冻结当前切片视窗",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.freeze_viewport_button.setObjectName(
            "analysisBatchFreezeViewportButton"
        )
        self.freeze_viewport_button.setToolTip(
            "仅冻结当前已打开数字化切片的当前焦层和当前原始像素视窗；"
            "不会扫描整张切片"
        )
        self.run_button.clicked.connect(self._emit_run)
        self.cancel_button.clicked.connect(self.cancelRequested)
        self.export_button.clicked.connect(self._emit_export)
        self.freeze_viewport_button.clicked.connect(
            self.freezeViewportRequested
        )

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.items_table, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.buttons)

    def set_recipes(self, recipes: tuple[AnalysisRecipe, ...]) -> None:
        self._recipes = {recipe.recipe_id: recipe for recipe in recipes}
        self.recipe_combo.clear()
        for recipe in recipes:
            self.recipe_combo.addItem(recipe.name, recipe.recipe_id)
        self.run_button.setEnabled(bool(recipes) and self.items_table.rowCount() > 0)

    def set_invocations(
        self,
        invocations: tuple[AnalysisInvocation, ...],
        *,
        unavailable_items: tuple[tuple[str, str, str], ...] = (),
    ) -> None:
        self.items_table.setRowCount(len(invocations) + len(unavailable_items))
        self._row_by_item_id.clear()
        self._available_item_ids = {
            invocation.item_id
            for invocation in invocations
        }
        for row, invocation in enumerate(invocations):
            self._row_by_item_id[invocation.item_id] = row
            scope = (
                "冻结 ROI"
                if invocation.analysis.roi_mask is not None
                else "完整图像"
                if invocation.viewport is None
                else (
                    f"viewport ({invocation.viewport.x}, {invocation.viewport.y}, "
                    f"{invocation.viewport.width}×{invocation.viewport.height}, "
                    f"焦层 {invocation.viewport.level})"
                )
            )
            self.items_table.setItem(
                row,
                0,
                self._checkable_item(invocation.display_name),
            )
            self.items_table.setItem(row, 1, QTableWidgetItem(scope))
            self.items_table.setItem(row, 2, QTableWidgetItem("等待"))
        for offset, (item_id, display_name, reason) in enumerate(
            unavailable_items,
            start=len(invocations),
        ):
            self._row_by_item_id[item_id] = offset
            item = self._checkable_item(display_name)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.items_table.setItem(offset, 0, item)
            self.items_table.setItem(offset, 1, QTableWidgetItem("不可用"))
            status = QTableWidgetItem(reason)
            status.setToolTip(reason)
            self.items_table.setItem(offset, 2, status)
        self.progress.setRange(0, max(1, len(invocations)))
        self.progress.setValue(0)
        if unavailable_items:
            self.summary_label.setText(
                f"{len(unavailable_items)} 个来源当前不可直接使用；"
                "数字化切片需逐个显式冻结当前视窗。"
            )
        else:
            self.summary_label.setText("请选择图片或已冻结切片视窗并开始分析。")
        self.run_button.setEnabled(
            bool(self._recipes) and bool(self.selected_item_ids())
        )

    def add_invocation(self, invocation: AnalysisInvocation) -> None:
        """Append one explicitly frozen viewport without rebuilding the matrix."""

        if not isinstance(invocation, AnalysisInvocation):
            raise TypeError("invocation 必须是 AnalysisInvocation")
        existing_row = self._row_by_item_id.get(invocation.item_id)
        if existing_row is None:
            row = self.items_table.rowCount()
            self.items_table.insertRow(row)
            self._row_by_item_id[invocation.item_id] = row
        else:
            row = existing_row
        self._available_item_ids.add(invocation.item_id)
        self.items_table.setItem(
            row,
            0,
            self._checkable_item(invocation.display_name),
        )
        viewport = invocation.viewport
        if viewport is None:
            scope = (
                "冻结 ROI"
                if invocation.analysis.roi_mask is not None
                else "完整图像"
            )
        else:
            scope = (
                f"冻结视窗 ({viewport.x}, {viewport.y}, "
                f"{viewport.width}×{viewport.height}, "
                f"焦层 {viewport.level})"
            )
        self.items_table.setItem(row, 1, QTableWidgetItem(scope))
        self.items_table.setItem(row, 2, QTableWidgetItem("已冻结"))
        self.progress.setRange(0, max(1, len(self._available_item_ids)))
        self.summary_label.setText(
            f"已显式冻结 {sum(1 for item_id in self._available_item_ids if '::viewport::' in item_id)} "
            "个数字化切片视窗；可切换切片后继续添加。"
        )
        self._update_run_button()

    def selected_item_ids(self) -> tuple[str, ...]:
        selected: list[str] = []
        for item_id, row in self._row_by_item_id.items():
            if item_id not in self._available_item_ids:
                continue
            item = self.items_table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.append(item_id)
        return tuple(selected)

    def set_busy(self, busy: bool) -> None:
        if busy:
            self._last_result = None
        self.recipe_combo.setEnabled(not busy)
        self.run_button.setEnabled(
            not busy
            and bool(self._recipes)
            and bool(self.selected_item_ids())
        )
        self.cancel_button.setEnabled(busy)
        self.freeze_viewport_button.setEnabled(not busy)
        self.export_button.setEnabled(
            not busy and self._last_result is not None
        )

    def update_progress(self, update: AnalysisBatchProgress) -> None:
        self.progress.setRange(0, max(1, update.total))
        self.progress.setValue(update.completed)
        row = self._row_by_item_id.get(update.item_id)
        if row is not None:
            self.items_table.setItem(row, 2, QTableWidgetItem("已完成"))
        self.summary_label.setText(
            f"已完成 {update.completed}/{update.total} 项，结果将在整批结束后提交。"
        )

    def show_result(self, result: AnalysisBatchResult) -> None:
        self._last_result = result
        by_id = {item.item_id: item for item in result.item_results}
        for item_id, row in self._row_by_item_id.items():
            item = by_id.get(item_id)
            if item is None:
                status = "已取消" if result.cancelled else "未执行"
            elif item.success:
                status = "成功"
            else:
                status = f"失败：{item.error_message or item.error_type or '未知错误'}"
            self.items_table.setItem(row, 2, QTableWidgetItem(status))
        self.summary_label.setText(
            f"完成：成功 {result.success_count} 项，失败 {result.failure_count} 项"
            + ("，任务已取消。" if result.cancelled else "。")
        )
        self.set_busy(False)

    def apply_commit_updates(
        self,
        updates: tuple[AnalysisBatchCommitUpdate, ...],
        *,
        summary: str,
    ) -> None:
        for update in updates:
            row = self._row_by_item_id.get(update.item_id)
            if row is None:
                continue
            status = QTableWidgetItem(update.status)
            if update.detail:
                status.setToolTip(update.detail)
            self.items_table.setItem(row, 2, status)
        self.summary_label.setText(summary)
        self.set_busy(False)

    def show_task_failure(self, message: str) -> None:
        self.summary_label.setText(f"批量分析失败：{message}")
        self.set_busy(False)

    def show_cancelled(self) -> None:
        self.summary_label.setText("批量分析已取消，未提交任何结果。")
        self.set_busy(False)

    def show_stale_discard(self) -> None:
        self.summary_label.setText("批量分析结果已过期，未提交任何结果。")
        self.set_busy(False)

    def _emit_run(self) -> None:
        recipe_id = self.recipe_combo.currentData()
        if recipe_id and self.selected_item_ids():
            self.runRequested.emit(str(recipe_id))

    def _emit_recipe_changed(self, _index: int) -> None:
        recipe_id = self.recipe_combo.currentData()
        if recipe_id:
            self.recipeChanged.emit(str(recipe_id))

    def _emit_export(self) -> None:
        if self._last_result is not None:
            self.exportRequested.emit(self._last_result)

    def _update_run_button(self, _item: QTableWidgetItem | None = None) -> None:
        self.run_button.setEnabled(
            not self.cancel_button.isEnabled()
            and bool(self._recipes)
            and bool(self.selected_item_ids())
        )

    def _checkable_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        return item


__all__ = ["AnalysisBatchCommitUpdate", "AnalysisBatchDialog"]
