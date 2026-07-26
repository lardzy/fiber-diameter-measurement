"""Presentation-only dialog for applying an image-processing recipe in batch.

The dialog deliberately does not own ``ImageBatchTaskController`` and never
writes project state.  A host supplies mounted-document choices, performs the
resource preflight, owns request generations, and commits successful
``DerivedRasterCandidate`` objects only after the dialog has displayed them as
pending.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fdm.image_processing_models import ImageProcessingRecipe
from fdm.services.image_batch import (
    BatchExecutionResult,
    BatchItemStatus,
    BatchProgressPhase,
    BatchProgressUpdate,
    BatchResourceEstimate,
)
from fdm.ui.image_processing_workbench import image_operation_display_name


@dataclass(frozen=True, slots=True)
class BatchDocumentOption:
    """A mounted document row offered by the batch dialog."""

    document_id: str
    display_name: str
    source_summary: str
    selected: bool = True
    enabled: bool = True
    unavailable_reason: str = ""
    is_digital_slide: bool = False

    def __post_init__(self) -> None:
        document_id = str(self.document_id or "").strip()
        display_name = str(self.display_name or "").strip()
        if not document_id:
            raise ValueError("批处理候选文档 ID 不能为空")
        if not display_name:
            raise ValueError("批处理候选图片名称不能为空")
        enabled = bool(self.enabled) and not bool(self.is_digital_slide)
        unavailable_reason = str(self.unavailable_reason or "").strip()
        if not enabled and not unavailable_reason:
            unavailable_reason = (
                "数字化切片不能直接整张处理；"
                "请先冻结当前焦层的原始像素视窗并生成普通图片。"
                if self.is_digital_slide
                else "当前图片不可用于批处理。"
            )
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(
            self,
            "source_summary",
            str(self.source_summary or "").strip() or "普通图片",
        )
        object.__setattr__(self, "selected", bool(self.selected) and enabled)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "unavailable_reason", unavailable_reason)


@dataclass(frozen=True, slots=True)
class BatchDialogRequest:
    """Immutable UI request handed to the host for preflight or execution."""

    recipe: ImageProcessingRecipe
    document_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.recipe, ImageProcessingRecipe):
            raise TypeError("recipe 必须是 ImageProcessingRecipe")
        document_ids = tuple(str(item).strip() for item in self.document_ids)
        if not document_ids or any(not item for item in document_ids):
            raise ValueError("批处理至少需要选择一张普通图片")
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("批处理文档 ID 不能重复")
        object.__setattr__(self, "document_ids", document_ids)


@dataclass(frozen=True, slots=True)
class BatchCommitUpdate:
    """Host-confirmed disposition of one pending derived-image candidate."""

    document_id: str
    status: str
    message: str = ""

    def __post_init__(self) -> None:
        document_id = str(self.document_id or "").strip()
        status = str(self.status or "").strip()
        if not document_id or not status:
            raise ValueError("批处理提交状态必须包含文档 ID 和状态")
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "message", str(self.message or "").strip())


def _format_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or suffix == "TiB":
            return (
                f"{size:.0f} {suffix}"
                if suffix == "B"
                else f"{size:.1f} {suffix}"
            )
        size /= 1024.0
    return f"{size:.1f} TiB"  # pragma: no cover - loop is exhaustive


class ImageBatchProcessingDialog(QDialog):
    """Non-modal batch review surface with no project mutation authority."""

    preflightRequested = Signal(object)
    batchStartRequested = Signal(object)
    cancelRequested = Signal()
    selectionChanged = Signal(object)

    _COLUMN_SELECT = 0
    _COLUMN_NAME = 1
    _COLUMN_SOURCE = 2
    _COLUMN_STATUS = 3

    def __init__(
        self,
        recipe: ImageProcessingRecipe,
        documents: tuple[BatchDocumentOption, ...],
        *,
        recipe_name: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(recipe, ImageProcessingRecipe):
            raise TypeError("recipe 必须是 ImageProcessingRecipe")
        options = tuple(documents)
        if not options:
            raise ValueError("批处理对话框至少需要一个候选文档")
        if len({item.document_id for item in options}) != len(options):
            raise ValueError("批处理候选文档 ID 不能重复")

        self._recipe = recipe
        self._documents = options
        self._recipe_name = str(recipe_name or "").strip() or "当前工作台配方"
        self._preflight: BatchResourceEstimate | None = None
        self._preflight_signature: tuple[str, ...] = ()
        self._active_request_id = ""
        self._active_generation = -1
        self._busy = False
        self._updating_rows = False

        self.setWindowTitle("批量应用图像处理配方")
        self.setModal(False)
        self.setMinimumSize(760, 500)
        self.resize(980, 680)
        self._build_ui()
        self._populate_recipe()
        self._populate_documents()
        self._invalidate_preflight(
            "请选择普通图片，然后执行资源预检。"
        )

    @property
    def recipe(self) -> ImageProcessingRecipe:
        return self._recipe

    def selected_document_ids(self) -> tuple[str, ...]:
        selected: list[str] = []
        for row, option in enumerate(self._documents):
            item = self._documents_table.item(row, self._COLUMN_SELECT)
            if (
                option.enabled
                and item is not None
                and item.checkState() == Qt.CheckState.Checked
            ):
                selected.append(option.document_id)
        return tuple(selected)

    def current_request(self) -> BatchDialogRequest | None:
        document_ids = self.selected_document_ids()
        if not document_ids:
            return None
        return BatchDialogRequest(
            recipe=self._recipe,
            document_ids=document_ids,
        )

    def request_preflight(self) -> None:
        request = self.current_request()
        if request is None:
            self._status_label.setText("请至少选择一张普通图片。")
            return
        self._status_label.setText("正在检查内存和磁盘空间…")
        self._preflight_button.setEnabled(False)
        self.preflightRequested.emit(request)

    def apply_preflight(
        self,
        estimate: BatchResourceEstimate,
        *,
        document_ids: tuple[str, ...] | None = None,
    ) -> None:
        if not isinstance(estimate, BatchResourceEstimate):
            raise TypeError("estimate 必须是 BatchResourceEstimate")
        signature = (
            self.selected_document_ids()
            if document_ids is None
            else tuple(str(item) for item in document_ids)
        )
        if signature != self.selected_document_ids():
            return
        self._preflight = estimate
        self._preflight_signature = signature
        item_estimates = {item.document_id: item for item in estimate.items}
        missing_estimates = tuple(
            document_id
            for document_id in signature
            if document_id not in item_estimates
        )
        maximum_peak = max(
            (
                item.estimated_peak_bytes
                for item in estimate.items
                if item.document_id in signature
            ),
            default=0,
        )
        self._memory_value.setText(_format_bytes(maximum_peak))
        self._output_value.setText(
            _format_bytes(estimate.estimated_total_output_bytes)
        )
        self._disk_value.setText(
            f"可用 {_format_bytes(estimate.available_disk_bytes)}；"
            f"需保留 {_format_bytes(estimate.reserve_disk_bytes)}"
        )
        for row, option in enumerate(self._documents):
            if not option.enabled:
                continue
            status_item = self._documents_table.item(row, self._COLUMN_STATUS)
            item_estimate = item_estimates.get(option.document_id)
            if status_item is None or option.document_id not in signature:
                continue
            if item_estimate is None:
                status_item.setText("未获得预检结果")
                status_item.setToolTip("宿主没有返回此图片的资源估算。")
            elif item_estimate.allowed:
                status_item.setText("预检通过")
                status_item.setToolTip(
                    "预计峰值 "
                    f"{_format_bytes(item_estimate.estimated_peak_bytes)}；"
                    "尚未开始处理。"
                )
            else:
                status_item.setText("资源阻断")
                status_item.setToolTip(item_estimate.reason)
        if missing_estimates:
            self._status_label.setText(
                "资源预检结果不完整，请重新预检后再开始批处理。"
            )
        elif estimate.allowed:
            blocked = sum(
                not item.allowed
                for item in estimate.items
                if item.document_id in signature
            )
            self._status_label.setText(
                "资源预检通过。"
                + (
                    f"其中 {blocked} 张将被单独阻断，其余图片仍可继续。"
                    if blocked
                    else "可以开始批处理。"
                )
            )
        else:
            self._status_label.setText(
                estimate.reason or "资源预检未通过，无法开始批处理。"
            )
        self._update_actions()

    def apply_preflight_error(self, message: str) -> None:
        """Return the dialog to a retryable state after host preflight failure."""

        self._invalidate_preflight(
            "资源预检失败："
            + (str(message or "").strip() or "未提供具体原因。")
        )

    def begin_request(self, request_id: str, generation: int) -> None:
        request_id = str(request_id or "").strip()
        if not request_id:
            raise ValueError("批处理 request_id 不能为空")
        self._active_request_id = request_id
        self._active_generation = int(generation)
        self.set_busy(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._phase_label.setText("预扫描")
        self._status_label.setText("已提交批处理请求，等待真实进度…")
        for row, option in enumerate(self._documents):
            if (
                option.document_id in self._preflight_signature
                and option.enabled
            ):
                self._documents_table.item(
                    row, self._COLUMN_STATUS
                ).setText("等待处理")

    def apply_progress(self, update: BatchProgressUpdate) -> bool:
        if not isinstance(update, BatchProgressUpdate):
            raise TypeError("update 必须是 BatchProgressUpdate")
        if not self._active_request_id or (
            update.request_id != self._active_request_id
            or update.generation != self._active_generation
        ):
            return False
        phase_names = {
            BatchProgressPhase.PREFLIGHT: "预扫描",
            BatchProgressPhase.PROCESSING: "处理",
            BatchProgressPhase.PACKAGING: "整理结果",
        }
        self._phase_label.setText(phase_names[update.phase])
        self._status_label.setText(update.message)
        if update.phase is BatchProgressPhase.PREFLIGHT:
            progress = 2
        elif update.phase is BatchProgressPhase.PACKAGING:
            progress = 96
        else:
            operations = max(1, update.total_operations)
            total_units = max(1, update.item_total * operations)
            completed_units = (
                max(0, update.item_index - 1) * operations
                + update.completed_operations
            )
            progress = 5 + int(round(88 * completed_units / total_units))
        self._progress_bar.setValue(max(0, min(99, progress)))
        if update.document_id:
            row = self._row_for_document(update.document_id)
            if row >= 0:
                status = (
                    f"处理中 {update.completed_operations}/"
                    f"{update.total_operations}"
                    if update.total_operations
                    else "处理中"
                )
                item = self._documents_table.item(row, self._COLUMN_STATUS)
                item.setText(status)
                item.setToolTip(update.message)
                self._documents_table.scrollToItem(
                    item,
                    QAbstractItemView.ScrollHint.EnsureVisible,
                )
        return True

    def apply_result(self, result: BatchExecutionResult) -> bool:
        if not isinstance(result, BatchExecutionResult):
            raise TypeError("result 必须是 BatchExecutionResult")
        if not self._active_request_id or (
            result.request_id != self._active_request_id
            or result.generation != self._active_generation
        ):
            return False
        status_text = {
            BatchItemStatus.SUCCESS: "待加入项目",
            BatchItemStatus.FAILED: "失败",
            BatchItemStatus.RESOURCE_BLOCKED: "资源阻断",
            BatchItemStatus.CANCELLED: "已取消",
            BatchItemStatus.STALE: "已丢弃",
        }
        for item_result in result.items:
            row = self._row_for_document(item_result.document_id)
            if row < 0:
                continue
            item = self._documents_table.item(row, self._COLUMN_STATUS)
            item.setText(status_text[item_result.status])
            item.setToolTip(item_result.message)
        self._progress_bar.setValue(100)
        self._phase_label.setText(
            "等待项目提交"
            if result.commit_candidates
            else "已结束"
        )
        summary = result.summary_text
        if result.commit_candidates:
            summary += (
                f" {len(result.commit_candidates)} 张派生图片当前仅为待提交候选，"
                "尚未写入项目。"
            )
        self._summary_label.setText(summary)
        self._status_label.setText(summary)
        self._active_request_id = ""
        self._active_generation = -1
        self.set_busy(False)
        return True

    def apply_task_cancelled(self, request_id: str) -> bool:
        if str(request_id) != self._active_request_id:
            return False
        self._finish_without_result(
            phase="已取消",
            row_status="已取消",
            message="批处理已取消，所有派生候选均未提交。",
        )
        return True

    def apply_task_failure(self, request_id: str, message: str) -> bool:
        if str(request_id) != self._active_request_id:
            return False
        self._finish_without_result(
            phase="失败",
            row_status="未完成",
            message=(
                "批处理失败："
                + (str(message or "").strip() or "未提供具体原因。")
            ),
        )
        return True

    def apply_stale_discard(self, request_id: str, generation: int) -> bool:
        if (
            str(request_id) != self._active_request_id
            or int(generation) != self._active_generation
        ):
            return False
        self._finish_without_result(
            phase="已丢弃",
            row_status="已丢弃",
            message="批处理请求已过期，晚到结果已丢弃且未提交。",
        )
        return True

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self._documents_table.setEnabled(not self._busy)
        self._select_all_button.setEnabled(not self._busy)
        self._clear_selection_button.setEnabled(not self._busy)
        self._cancel_button.setText("取消任务" if self._busy else "关闭")
        self._update_actions()

    def set_commit_summary(self, text: str, *, completed: bool = True) -> None:
        """Show the host's post-commit summary without mutating project state."""

        message = str(text or "").strip()
        if message:
            self._summary_label.setText(message)
            self._status_label.setText(message)
        if completed:
            self._phase_label.setText("提交完成")

    def apply_commit_updates(
        self,
        updates: tuple[BatchCommitUpdate, ...],
        *,
        summary: str,
    ) -> None:
        """Apply host-confirmed commit outcomes to the per-image status table."""

        for update in tuple(updates):
            if not isinstance(update, BatchCommitUpdate):
                raise TypeError("updates 必须全部是 BatchCommitUpdate")
            row = self._row_for_document(update.document_id)
            if row < 0:
                continue
            item = self._documents_table.item(row, self._COLUMN_STATUS)
            item.setText(update.status)
            item.setToolTip(update.message)
        self.set_commit_summary(summary, completed=True)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._busy:
            self.cancelRequested.emit()
            self._status_label.setText("正在请求取消；确认任务退出后方可关闭。")
            event.ignore()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        intro = QLabel(
            "将同一配方应用到选中的普通图片。每张图片独立执行；"
            "成功结果会先停留在“待加入项目”状态，由工作区统一确认提交。",
            self,
        )
        intro.setWordWrap(True)
        intro.setObjectName("imageBatchIntroduction")

        top_row = QHBoxLayout()
        top_row.addWidget(self._build_recipe_group(), 3)
        top_row.addWidget(self._build_resource_group(), 2)

        documents_group = QGroupBox("处理图片", self)
        documents_layout = QVBoxLayout(documents_group)
        selection_row = QHBoxLayout()
        self._selection_summary = QLabel("", documents_group)
        self._select_all_button = QPushButton("全选普通图片", documents_group)
        self._clear_selection_button = QPushButton("清除选择", documents_group)
        self._select_all_button.clicked.connect(
            lambda: self._set_all_enabled_checked(True)
        )
        self._clear_selection_button.clicked.connect(
            lambda: self._set_all_enabled_checked(False)
        )
        selection_row.addWidget(self._selection_summary, 1)
        selection_row.addWidget(self._select_all_button)
        selection_row.addWidget(self._clear_selection_button)
        documents_layout.addLayout(selection_row)

        self._documents_table = QTableWidget(0, 4, documents_group)
        self._documents_table.setObjectName("imageBatchDocuments")
        self._documents_table.setHorizontalHeaderLabels(
            ("选择", "图片", "像素来源", "阶段 / 状态")
        )
        self._documents_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._documents_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._documents_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._documents_table.setAlternatingRowColors(True)
        self._documents_table.verticalHeader().setVisible(False)
        header = self._documents_table.horizontalHeader()
        header.setSectionResizeMode(self._COLUMN_SELECT, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(self._COLUMN_SELECT, 58)
        header.setSectionResizeMode(self._COLUMN_NAME, QHeaderView.ResizeMode.Stretch)
        # User-owned names and source descriptions may be arbitrarily long.
        # Keep this column bounded so content can never widen the dialog.
        header.setSectionResizeMode(
            self._COLUMN_SOURCE, QHeaderView.ResizeMode.Interactive
        )
        header.resizeSection(self._COLUMN_SOURCE, 210)
        header.setSectionResizeMode(self._COLUMN_STATUS, QHeaderView.ResizeMode.Stretch)
        self._documents_table.itemChanged.connect(self._on_item_changed)
        documents_layout.addWidget(self._documents_table, 1)

        progress_group = QGroupBox("执行进度与提交前汇总", self)
        progress_layout = QVBoxLayout(progress_group)
        progress_row = QHBoxLayout()
        self._phase_label = QLabel("尚未开始", progress_group)
        self._phase_label.setObjectName("imageBatchPhase")
        self._progress_bar = QProgressBar(progress_group)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        progress_row.addWidget(self._phase_label)
        progress_row.addWidget(self._progress_bar, 1)
        progress_layout.addLayout(progress_row)
        self._summary_label = QLabel(
            "尚无执行结果。逐图片状态会显示在上方表格中。",
            progress_group,
        )
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        progress_layout.addWidget(self._summary_label)

        footer = QHBoxLayout()
        self._status_label = QLabel("", self)
        self._status_label.setWordWrap(True)
        self._status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self._preflight_button = QPushButton("重新预检", self)
        self._start_button = QPushButton("开始批处理", self)
        self._cancel_button = QPushButton("关闭", self)
        self._preflight_button.clicked.connect(self.request_preflight)
        self._start_button.clicked.connect(self._emit_start_request)
        self._cancel_button.clicked.connect(self._cancel_or_close)
        footer.addWidget(self._status_label, 1)
        footer.addWidget(self._preflight_button)
        footer.addWidget(self._start_button)
        footer.addWidget(self._cancel_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        root.addWidget(intro)
        root.addLayout(top_row)
        root.addWidget(documents_group, 1)
        root.addWidget(progress_group)
        root.addLayout(footer)

    def _build_recipe_group(self) -> QGroupBox:
        group = QGroupBox("处理配方", self)
        layout = QVBoxLayout(group)
        self._recipe_title = QLabel(
            f"<b>{self._recipe_name}</b> · {len(self._recipe.operations)} 个步骤",
            group,
        )
        self._recipe_title.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._recipe_title)
        self._recipe_steps = QListWidget(group)
        self._recipe_steps.setMaximumHeight(132)
        self._recipe_steps.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        layout.addWidget(self._recipe_steps)
        return group

    def _build_resource_group(self) -> QGroupBox:
        group = QGroupBox("资源预检", self)
        layout = QVBoxLayout(group)
        for label_text, attribute in (
            ("单图最高预计工作集", "_memory_value"),
            ("预计派生像素总量", "_output_value"),
            ("磁盘安全余量", "_disk_value"),
        ):
            row = QHBoxLayout()
            label = QLabel(label_text, group)
            value = QLabel("等待预检", group)
            value.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            value.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            setattr(self, attribute, value)
            row.addWidget(label)
            row.addWidget(value, 1)
            layout.addLayout(row)
        note = QLabel(
            "预计工作集超过 1 GiB，或不能预留至少 2 GiB 磁盘空间时，"
            "对应任务会明确阻断，不会静默降分辨率。",
            group,
        )
        note.setWordWrap(True)
        note.setObjectName("imageBatchResourceNote")
        layout.addWidget(note)
        layout.addStretch(1)
        return group

    def _populate_recipe(self) -> None:
        self._recipe_steps.clear()
        for index, operation in enumerate(self._recipe.operations, start=1):
            label = image_operation_display_name(operation.operation_id)
            self._recipe_steps.addItem(f"{index}. {label}")

    def _populate_documents(self) -> None:
        self._updating_rows = True
        try:
            self._documents_table.setRowCount(len(self._documents))
            for row, option in enumerate(self._documents):
                check_item = QTableWidgetItem("")
                check_item.setData(Qt.ItemDataRole.UserRole, option.document_id)
                if option.enabled:
                    check_item.setFlags(
                        Qt.ItemFlag.ItemIsEnabled
                        | Qt.ItemFlag.ItemIsSelectable
                        | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    check_item.setCheckState(
                        Qt.CheckState.Checked
                        if option.selected
                        else Qt.CheckState.Unchecked
                    )
                else:
                    check_item.setFlags(Qt.ItemFlag.NoItemFlags)
                    check_item.setCheckState(Qt.CheckState.Unchecked)
                    check_item.setToolTip(option.unavailable_reason)
                name_item = QTableWidgetItem(option.display_name)
                source_item = QTableWidgetItem(option.source_summary)
                name_item.setToolTip(option.display_name)
                source_item.setToolTip(option.source_summary)
                status_item = QTableWidgetItem(
                    "等待预检" if option.enabled else "不可批处理"
                )
                for item in (name_item, source_item, status_item):
                    if not option.enabled:
                        item.setFlags(Qt.ItemFlag.NoItemFlags)
                        item.setToolTip(option.unavailable_reason)
                self._documents_table.setItem(row, self._COLUMN_SELECT, check_item)
                self._documents_table.setItem(row, self._COLUMN_NAME, name_item)
                self._documents_table.setItem(row, self._COLUMN_SOURCE, source_item)
                self._documents_table.setItem(row, self._COLUMN_STATUS, status_item)
            self._documents_table.resizeRowsToContents()
        finally:
            self._updating_rows = False
        self._update_selection_summary()

    def _row_for_document(self, document_id: str) -> int:
        for row, option in enumerate(self._documents):
            if option.document_id == document_id:
                return row
        return -1

    def _set_all_enabled_checked(self, checked: bool) -> None:
        self._updating_rows = True
        try:
            for row, option in enumerate(self._documents):
                if option.enabled:
                    self._documents_table.item(
                        row, self._COLUMN_SELECT
                    ).setCheckState(
                        Qt.CheckState.Checked
                        if checked
                        else Qt.CheckState.Unchecked
                    )
        finally:
            self._updating_rows = False
        self._on_selection_changed()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if (
            self._updating_rows
            or item.column() != self._COLUMN_SELECT
            or self._busy
        ):
            return
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        self._update_selection_summary()
        self._invalidate_preflight("图片选择已变化，请重新执行资源预检。")
        request = self.current_request()
        self.selectionChanged.emit(
            () if request is None else request.document_ids
        )

    def _update_selection_summary(self) -> None:
        selected = len(self.selected_document_ids())
        available = sum(option.enabled for option in self._documents)
        unavailable = len(self._documents) - available
        suffix = (
            f"；{unavailable} 张数字化切片或不可用文档已禁用"
            if unavailable
            else ""
        )
        self._selection_summary.setText(
            f"已选择 {selected}/{available} 张普通图片{suffix}"
        )

    def _invalidate_preflight(self, message: str) -> None:
        self._preflight = None
        self._preflight_signature = ()
        self._memory_value.setText("等待预检")
        self._output_value.setText("等待预检")
        self._disk_value.setText("等待预检")
        self._status_label.setText(message)
        for row, option in enumerate(self._documents):
            if option.enabled:
                self._documents_table.item(
                    row, self._COLUMN_STATUS
                ).setText("等待预检")
        self._update_actions()

    def _emit_start_request(self) -> None:
        request = self.current_request()
        if request is None:
            self._status_label.setText("请至少选择一张普通图片。")
            return
        if (
            self._preflight is None
            or self._preflight_signature != request.document_ids
        ):
            self._status_label.setText("开始前必须完成当前选择的资源预检。")
            self.request_preflight()
            return
        if not self._preflight.allowed:
            self._status_label.setText(
                self._preflight.reason or "资源预检未通过，不能开始批处理。"
            )
            return
        self.batchStartRequested.emit(request)

    def _cancel_or_close(self) -> None:
        if self._busy:
            self.cancelRequested.emit()
            self._status_label.setText("正在请求取消；取消后不会提交派生图片。")
            return
        self.reject()

    def _finish_without_result(
        self,
        *,
        phase: str,
        row_status: str,
        message: str,
    ) -> None:
        for row, option in enumerate(self._documents):
            if (
                option.enabled
                and option.document_id in self._preflight_signature
            ):
                self._documents_table.item(
                    row, self._COLUMN_STATUS
                ).setText(row_status)
        self._phase_label.setText(phase)
        self._summary_label.setText(message)
        self._status_label.setText(message)
        self._active_request_id = ""
        self._active_generation = -1
        self.set_busy(False)

    def _update_actions(self) -> None:
        signature = self.selected_document_ids()
        estimate_ids = (
            set()
            if self._preflight is None
            else {item.document_id for item in self._preflight.items}
        )
        preflight_current = (
            self._preflight is not None
            and self._preflight_signature == signature
            and set(signature).issubset(estimate_ids)
        )
        self._preflight_button.setEnabled(bool(signature) and not self._busy)
        self._start_button.setEnabled(
            bool(signature)
            and not self._busy
            and preflight_current
            and bool(self._preflight and self._preflight.allowed)
        )


__all__ = [
    "BatchCommitUpdate",
    "BatchDialogRequest",
    "BatchDocumentOption",
    "ImageBatchProcessingDialog",
]
