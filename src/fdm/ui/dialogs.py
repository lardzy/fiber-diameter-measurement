from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
)

from fdm.services.export_service import ExportImageRenderMode, ExportScope, ExportSelection


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
        self._pixel_distance_spin = QDoubleSpinBox()
        self._pixel_distance_spin.setDecimals(6)
        self._pixel_distance_spin.setRange(0.000001, 1_000_000.0)
        self._pixel_distance_spin.setValue(100.0)
        self._actual_distance_spin = QDoubleSpinBox()
        self._actual_distance_spin.setDecimals(6)
        self._actual_distance_spin.setRange(0.000001, 1_000_000.0)
        self._actual_distance_spin.setValue(10.0)
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["um", "mm"])
        self._computed_label = QLabel()
        self._computed_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._pixel_distance_spin.valueChanged.connect(self._refresh_computed_value)
        self._actual_distance_spin.valueChanged.connect(self._refresh_computed_value)
        self._unit_combo.currentIndexChanged.connect(self._refresh_computed_value)

        form = QFormLayout()
        form.addRow("预设名称", self._name_edit)
        form.addRow("像素距离", self._pixel_distance_spin)
        form.addRow("实际距离", self._actual_distance_spin)
        form.addRow("单位", self._unit_combo)
        form.addRow("自动计算", self._computed_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._refresh_computed_value()

    def _refresh_computed_value(self) -> None:
        pixels_per_unit = self._pixel_distance_spin.value() / self._actual_distance_spin.value()
        self._computed_label.setText(f"{pixels_per_unit:.6f} px/{self._unit_combo.currentText()}")

    def values(self) -> tuple[str, float, float, float, str]:
        pixels_per_unit = self._pixel_distance_spin.value() / self._actual_distance_spin.value()
        return (
            self._name_edit.text().strip(),
            self._pixel_distance_spin.value(),
            self._actual_distance_spin.value(),
            pixels_per_unit,
            self._unit_combo.currentText(),
        )


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

        render_group = QGroupBox("图片导出模式")
        render_layout = QFormLayout(render_group)
        self._render_mode_combo = QComboBox()
        self._render_mode_combo.addItem("整图按屏显比例导出", ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE)
        self._render_mode_combo.addItem("完整分辨率", ExportImageRenderMode.FULL_RESOLUTION)
        self._render_mode_combo.addItem("当前视窗截图", ExportImageRenderMode.CURRENT_VIEWPORT)
        render_index = self._render_mode_combo.findData(selection.render_mode)
        self._render_mode_combo.setCurrentIndex(max(0, render_index))
        self._render_mode_hint = QLabel("图片类导出会使用这里的渲染模式；表格和 JSON 不受影响。")
        self._render_mode_hint.setWordWrap(True)
        render_layout.addRow("渲染方式", self._render_mode_combo)
        render_layout.addRow("", self._render_mode_hint)

        self._measurement_overlay.toggled.connect(self._update_render_mode_state)
        self._scale_overlay.toggled.connect(self._update_render_mode_state)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(export_group)
        layout.addWidget(scope_group)
        layout.addWidget(render_group)
        layout.addWidget(buttons)
        self._update_render_mode_state()

    def _update_render_mode_state(self) -> None:
        enabled = self._measurement_overlay.isChecked() or self._scale_overlay.isChecked()
        self._render_mode_combo.setEnabled(enabled)
        self._render_mode_hint.setEnabled(enabled)

    def selection(self) -> ExportSelection:
        return ExportSelection(
            include_measurement_overlay=self._measurement_overlay.isChecked(),
            include_scale_overlay=self._scale_overlay.isChecked(),
            include_scale_json=self._scale_json.isChecked(),
            include_excel=self._excel.isChecked(),
            include_csv=self._csv.isChecked(),
            scope=ExportScope.ALL_OPEN if self._scope_all.isChecked() and self._scope_all.isEnabled() else ExportScope.CURRENT,
            render_mode=self._render_mode_combo.currentData(),
        )
