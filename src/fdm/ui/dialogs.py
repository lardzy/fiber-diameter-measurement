from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
)

from fdm.services.export_service import ExportScope, ExportSelection


class CalibrationInputDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("图内标尺标定")
        self._length_spin = QDoubleSpinBox()
        self._length_spin.setDecimals(6)
        self._length_spin.setRange(0.000001, 1_000_000.0)
        self._length_spin.setValue(100.0)
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["um", "mm"])

        form = QFormLayout()
        form.addRow("真实长度", self._length_spin)
        form.addRow("单位", self._unit_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> tuple[float, str]:
        return self._length_spin.value(), self._unit_combo.currentText()


class CalibrationPresetDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("新增标定预设")
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("例如 40x 显微镜")
        self._pixels_spin = QDoubleSpinBox()
        self._pixels_spin.setDecimals(6)
        self._pixels_spin.setRange(0.000001, 1_000_000.0)
        self._pixels_spin.setValue(10.0)
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["um", "mm"])

        form = QFormLayout()
        form.addRow("预设名称", self._name_edit)
        form.addRow("像素/单位", self._pixels_spin)
        form.addRow("单位", self._unit_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, float, str]:
        return self._name_edit.text().strip(), self._pixels_spin.value(), self._unit_combo.currentText()


class ExportOptionsDialog(QDialog):
    def __init__(
        self,
        selection: ExportSelection,
        *,
        allow_all_scope: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出选项")

        self._measurement_overlay = QCheckBox("测量叠加图 PNG")
        self._measurement_overlay.setChecked(selection.include_measurement_overlay)
        self._scale_overlay = QCheckBox("比例尺图 PNG")
        self._scale_overlay.setChecked(selection.include_scale_overlay)
        self._scale_json = QCheckBox("比例尺 JSON")
        self._scale_json.setChecked(selection.include_scale_json)
        self._excel = QCheckBox("Excel 文档")
        self._excel.setChecked(selection.include_excel)
        self._csv = QCheckBox("CSV 文档")
        self._csv.setChecked(selection.include_csv)

        export_group = QGroupBox("导出内容")
        export_layout = QVBoxLayout(export_group)
        export_layout.addWidget(self._measurement_overlay)
        export_layout.addWidget(self._scale_overlay)
        export_layout.addWidget(self._scale_json)
        export_layout.addWidget(self._excel)
        export_layout.addWidget(self._csv)

        scope_group = QGroupBox("导出范围")
        scope_layout = QVBoxLayout(scope_group)
        self._scope_current = QRadioButton("当前图片")
        self._scope_all = QRadioButton("全部已打开图片")
        self._scope_current.setChecked(selection.scope != ExportScope.ALL_OPEN)
        self._scope_all.setChecked(selection.scope == ExportScope.ALL_OPEN)
        self._scope_all.setEnabled(allow_all_scope)
        scope_layout.addWidget(self._scope_current)
        scope_layout.addWidget(self._scope_all)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(export_group)
        layout.addWidget(scope_group)
        layout.addWidget(buttons)

    def selection(self) -> ExportSelection:
        return ExportSelection(
            include_measurement_overlay=self._measurement_overlay.isChecked(),
            include_scale_overlay=self._scale_overlay.isChecked(),
            include_scale_json=self._scale_json.isChecked(),
            include_excel=self._excel.isChecked(),
            include_csv=self._csv.isChecked(),
            scope=ExportScope.ALL_OPEN if self._scope_all.isChecked() and self._scope_all.isEnabled() else ExportScope.CURRENT,
        )
