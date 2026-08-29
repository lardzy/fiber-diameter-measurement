from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fdm.models import UNCATEGORIZED_LABEL
from fdm.services.dataset_export import DatasetExportFormat


@dataclass(frozen=True, slots=True)
class DatasetDocumentOption:
    document_id: str
    label: str
    is_current: bool = False
    is_available: bool = True


@dataclass(frozen=True, slots=True)
class DatasetExportDialogOptions:
    output_directory: Path
    document_ids: tuple[str, ...]
    formats: tuple[DatasetExportFormat, ...]
    category_mapping: dict[str, str | None]
    annotation_complete: bool
    split_train_validation: bool
    yolo_complex_policy: str
    convert_high_bit_to_uint8: bool


class DatasetExportDialog(QDialog):
    """Low-frequency export wizard for confirmed training annotations."""

    def __init__(
        self,
        documents: list[DatasetDocumentOption],
        category_names: list[str],
        *,
        initial_directory: str | Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出训练数据")
        self.resize(820, 680)
        self._documents = list(documents)
        self._category_names = list(dict.fromkeys(category_names))

        intro = QLabel(
            "从已确认的面积对象生成训练数据。线段、计数点和辅助几何不会被猜测为纤维掩码。"
        )
        intro.setWordWrap(True)

        source_group = QGroupBox("1. 样本范围")
        source_layout = QVBoxLayout(source_group)
        self._scope_combo = QComboBox(source_group)
        self._scope_combo.addItem("当前图片", "current")
        self._scope_combo.addItem("勾选图片", "checked")
        self._scope_combo.addItem("整个项目", "project")
        self._scope_combo.currentIndexChanged.connect(self._apply_scope)
        source_layout.addWidget(self._scope_combo)
        self._document_table = QTableWidget(len(documents), 2, source_group)
        self._document_table.setHorizontalHeaderLabels(("导出", "图片 / 数字切片"))
        self._document_table.verticalHeader().setVisible(False)
        self._document_table.horizontalHeader().setStretchLastSection(True)
        self._document_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        for row, option in enumerate(documents):
            include = QTableWidgetItem()
            include.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            include.setCheckState(
                Qt.CheckState.Checked if option.is_current and option.is_available else Qt.CheckState.Unchecked
            )
            include.setData(Qt.ItemDataRole.UserRole, option.document_id)
            if not option.is_available:
                include.setFlags(Qt.ItemFlag.NoItemFlags)
            name = QTableWidgetItem(option.label)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if not option.is_available:
                name.setToolTip("当前来源无法读取，将不会导出")
                name.setForeground(self.palette().color(self.foregroundRole()).darker(160))
            self._document_table.setItem(row, 0, include)
            self._document_table.setItem(row, 1, name)
        self._document_table.setColumnWidth(0, 68)
        source_layout.addWidget(self._document_table)

        format_group = QGroupBox("2. 输出格式")
        format_layout = QVBoxLayout(format_group)
        self._format_checks: dict[DatasetExportFormat, QCheckBox] = {}
        for format_value in DatasetExportFormat:
            checkbox = QCheckBox(format_value.label, format_group)
            checkbox.setChecked(format_value is DatasetExportFormat.COCO_INSTANCE)
            self._format_checks[format_value] = checkbox
            format_layout.addWidget(checkbox)
        format_hint = QLabel(
            "COCO RLE 可无损保留孔洞、多连通域和重叠。YOLO 分割无法无损表达孔洞，默认跳过复杂对象。"
        )
        format_hint.setWordWrap(True)
        # QLabel's wrapped size hint can underestimate the third line after
        # responsive two-column layout, clipping the last few characters on
        # short/high-DPI screens.
        format_hint.setMinimumHeight(format_hint.fontMetrics().lineSpacing() * 3)
        format_layout.addWidget(format_hint)
        yolo_row = QWidget(format_group)
        yolo_layout = QHBoxLayout(yolo_row)
        yolo_layout.setContentsMargins(0, 0, 0, 0)
        yolo_layout.addWidget(QLabel("YOLO 复杂轮廓：", yolo_row))
        self._yolo_policy_combo = QComboBox(yolo_row)
        self._yolo_policy_combo.addItem("跳过并写入报告（推荐）", "skip")
        self._yolo_policy_combo.addItem("仅保留最大外轮廓（有损）", "lossy_largest_outer")
        yolo_layout.addWidget(self._yolo_policy_combo, 1)
        format_layout.addWidget(yolo_row)

        category_group = QGroupBox("3. 类别映射")
        category_layout = QVBoxLayout(category_group)
        category_hint = QLabel(
            "类别名称忽略大小写合并。‘未分类’不会自动当作背景，必须明确映射或排除。"
        )
        category_hint.setWordWrap(True)
        category_layout.addWidget(category_hint)
        self._category_table = QTableWidget(len(self._category_names), 3, category_group)
        self._category_table.setHorizontalHeaderLabels(("导出", "现有纤维类别", "训练类别名称"))
        self._category_table.verticalHeader().setVisible(False)
        self._category_table.horizontalHeader().setStretchLastSection(True)
        for row, name in enumerate(self._category_names):
            include = QTableWidgetItem()
            include.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            include.setCheckState(
                Qt.CheckState.Unchecked
                if name.strip().casefold() == UNCATEGORIZED_LABEL.casefold()
                else Qt.CheckState.Checked
            )
            source = QTableWidgetItem(name)
            source.setFlags(Qt.ItemFlag.ItemIsEnabled)
            target = QTableWidgetItem("" if name == UNCATEGORIZED_LABEL else name)
            self._category_table.setItem(row, 0, include)
            self._category_table.setItem(row, 1, source)
            self._category_table.setItem(row, 2, target)
        self._category_table.setColumnWidth(0, 68)
        self._category_table.setColumnWidth(1, 210)
        category_layout.addWidget(self._category_table)

        options_group = QGroupBox("4. 质量确认与输出")
        options_form = QFormLayout(options_group)
        self._annotation_complete = QCheckBox("已确认所选样本中的目标标注完整")
        self._annotation_complete.setToolTip(
            "若只标注了部分纤维，未标对象会在许多训练流程中被视为背景"
        )
        self._split_checkbox = QCheckBox("按独立源图 / 原始切片分组，划分 80% 训练集与 20% 验证集")
        self._split_checkbox.setChecked(True)
        self._convert_high_bit = QCheckBox("将 16 位 / 浮点源图转换为 8 位（默认保留 TIFF 原始位深）")
        output_row = QWidget(options_group)
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        initial = Path(initial_directory).expanduser() if initial_directory else Path.home()
        self._output_edit = QLineEdit(str(initial / "fdm-training-dataset"), output_row)
        browse_button = QPushButton("浏览…", output_row)
        browse_button.clicked.connect(self._browse_output)
        output_layout.addWidget(self._output_edit, 1)
        output_layout.addWidget(browse_button)
        options_form.addRow("", self._annotation_complete)
        options_form.addRow("", self._split_checkbox)
        options_form.addRow("", self._convert_high_bit)
        options_form.addRow("新建目录", output_row)

        warning = QLabel(
            "导出前会检查未分类、空标注、越界/重叠/截断对象、未知焦层及格式转换损失。"
            "可解释风险需要再次确认，无法解析的个别对象只会跳过并写入 export_report.json。"
        )
        warning.setWordWrap(True)
        warning.setObjectName("datasetExportWarning")

        left = QVBoxLayout()
        left.addWidget(source_group, 1)
        left.addWidget(format_group)
        right = QVBoxLayout()
        right.addWidget(category_group, 1)
        right.addWidget(options_group)
        columns = QHBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("检查并导出")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(columns, 1)
        layout.addWidget(warning)
        layout.addWidget(buttons)
        self._apply_scope()

    def _apply_scope(self) -> None:
        mode = str(self._scope_combo.currentData() or "current")
        for row, option in enumerate(self._documents):
            item = self._document_table.item(row, 0)
            if item is None:
                continue
            if mode == "current":
                item.setCheckState(
                    Qt.CheckState.Checked
                    if option.is_current and option.is_available
                    else Qt.CheckState.Unchecked
                )
            elif mode == "project":
                item.setCheckState(
                    Qt.CheckState.Checked if option.is_available else Qt.CheckState.Unchecked
                )
            item.setFlags(
                (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                if option.is_available and mode == "checked"
                else (Qt.ItemFlag.ItemIsEnabled if option.is_available else Qt.ItemFlag.NoItemFlags)
            )

    def _browse_output(self) -> None:
        initial = Path(self._output_edit.text().strip() or Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择训练数据父目录",
            str(initial.parent if initial.suffix or not initial.exists() else initial),
        )
        if not selected:
            return
        self._output_edit.setText(str(Path(selected) / "fdm-training-dataset"))

    def _selected_document_ids(self) -> tuple[str, ...]:
        values: list[str] = []
        for row in range(self._document_table.rowCount()):
            item = self._document_table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                values.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return tuple(values)

    def _category_mapping(self) -> dict[str, str | None]:
        mapping: dict[str, str | None] = {}
        for row in range(self._category_table.rowCount()):
            include = self._category_table.item(row, 0)
            source = self._category_table.item(row, 1)
            target = self._category_table.item(row, 2)
            if source is None:
                continue
            mapping[source.text()] = (
                target.text().strip()
                if include is not None and include.checkState() == Qt.CheckState.Checked and target is not None
                else None
            )
        return mapping

    def _validate_and_accept(self) -> None:
        output_text = self._output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(self, "导出训练数据", "请选择输出目录。")
            return
        output = Path(output_text).expanduser()
        if output.exists():
            QMessageBox.warning(self, "导出训练数据", "目标目录已存在，请使用一个新目录，避免覆盖数据集。")
            return
        if not self._selected_document_ids():
            QMessageBox.warning(self, "导出训练数据", "至少选择一张图片或一个数字切片。")
            return
        if not any(checkbox.isChecked() for checkbox in self._format_checks.values()):
            QMessageBox.warning(self, "导出训练数据", "至少选择一种训练数据格式。")
            return
        for row in range(self._category_table.rowCount()):
            include = self._category_table.item(row, 0)
            source = self._category_table.item(row, 1)
            target = self._category_table.item(row, 2)
            if (
                include is not None
                and include.checkState() == Qt.CheckState.Checked
                and (target is None or not target.text().strip())
            ):
                source_name = source.text() if source is not None else f"第 {row + 1} 行"
                QMessageBox.warning(
                    self,
                    "导出训练数据",
                    f"类别“{source_name}”已勾选，但训练类别名称为空。请填写名称或取消勾选。",
                )
                return
        mapping = self._category_mapping()
        if self._category_table.rowCount() and not any(target for target in mapping.values()):
            QMessageBox.warning(self, "导出训练数据", "至少保留并命名一个训练类别。")
            return
        self.accept()

    def options(self) -> DatasetExportDialogOptions:
        return DatasetExportDialogOptions(
            output_directory=Path(self._output_edit.text().strip()).expanduser(),
            document_ids=self._selected_document_ids(),
            formats=tuple(
                format_value
                for format_value, checkbox in self._format_checks.items()
                if checkbox.isChecked()
            ),
            category_mapping=self._category_mapping(),
            annotation_complete=self._annotation_complete.isChecked(),
            split_train_validation=self._split_checkbox.isChecked(),
            yolo_complex_policy=str(self._yolo_policy_combo.currentData() or "skip"),
            convert_high_bit_to_uint8=self._convert_high_bit.isChecked(),
        )
