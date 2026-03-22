from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QColorDialog,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFontComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fdm.models import ImageDocument
from fdm.settings import (
    AppSettings,
    MeasurementEndpointStyle,
    OpenImageViewMode,
    ScaleOverlayPlacementMode,
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


class SettingsDialog(QDialog):
    def __init__(
        self,
        settings: AppSettings,
        *,
        document: ImageDocument | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.resize(620, 520)
        self._initial_settings = replace(settings)
        self._document = document
        self._group_color_buttons: dict[str | None, QPushButton] = {}
        self._request_scale_anchor_pick = False

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_app_settings_tab(settings), "应用设置")
        self._tabs.addTab(self._build_current_image_tab(document), "当前图片样式")

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs)
        layout.addWidget(self._button_box)

    @property
    def button_box(self) -> QDialogButtonBox:
        return self._button_box

    def app_settings(self) -> AppSettings:
        return AppSettings(
            show_measurement_labels=self._show_measurement_labels.isChecked(),
            measurement_label_font_family=self._measurement_label_font.currentFont().family(),
            measurement_label_font_size=self._measurement_label_size.value(),
            measurement_label_color=self._measurement_label_color.property("color_value") or self._initial_settings.measurement_label_color,
            measurement_endpoint_style=self._endpoint_style_combo.currentData(),
            default_measurement_color=self._default_measurement_color.property("color_value") or self._initial_settings.default_measurement_color,
            open_image_view_mode=self._open_view_mode_combo.currentData(),
            scale_overlay_placement_mode=self._scale_overlay_mode_combo.currentData(),
            text_font_family=self._text_font.currentFont().family(),
            text_font_size=self._text_size.value(),
            text_color=self._text_color.property("color_value") or self._initial_settings.text_color,
        )

    def group_colors(self) -> dict[str, str]:
        if self._document is None:
            return {}
        colors: dict[str, str] = {}
        for group in self._document.sorted_groups():
            button = self._group_color_buttons.get(group.id)
            if button is not None:
                colors[group.id] = str(button.property("color_value") or group.color)
        return colors

    def wants_scale_anchor_pick(self) -> bool:
        if self._document is None:
            return False
        if self._request_scale_anchor_pick:
            return True
        settings = self.app_settings()
        if settings.scale_overlay_placement_mode != ScaleOverlayPlacementMode.MANUAL:
            return False
        initial_mode = self._initial_settings.scale_overlay_placement_mode
        if initial_mode != ScaleOverlayPlacementMode.MANUAL:
            return True
        return self._document.scale_overlay_anchor is None

    def _build_app_settings_tab(self, settings: AppSettings) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        measurement_group = QGroupBox("测量线样式")
        measurement_form = QFormLayout(measurement_group)
        self._show_measurement_labels = QCheckBox("在测量线旁显示结果文字")
        self._show_measurement_labels.setChecked(settings.show_measurement_labels)
        self._measurement_label_font = QFontComboBox()
        self._measurement_label_font.setCurrentFont(QFont(settings.measurement_label_font_family))
        self._measurement_label_size = QSpinBox()
        self._measurement_label_size.setRange(8, 96)
        self._measurement_label_size.setValue(settings.measurement_label_font_size)
        self._measurement_label_color = self._create_color_button(settings.measurement_label_color)
        self._endpoint_style_combo = QComboBox()
        self._endpoint_style_combo.addItem("圆点", MeasurementEndpointStyle.CIRCLE)
        self._endpoint_style_combo.addItem("内侧箭头", MeasurementEndpointStyle.ARROW_INSIDE)
        self._endpoint_style_combo.addItem("外侧箭头", MeasurementEndpointStyle.ARROW_OUTSIDE)
        self._endpoint_style_combo.addItem("竖线", MeasurementEndpointStyle.BAR)
        self._endpoint_style_combo.addItem("无端点", MeasurementEndpointStyle.NONE)
        self._endpoint_style_combo.setCurrentIndex(max(0, self._endpoint_style_combo.findData(settings.measurement_endpoint_style)))
        self._default_measurement_color = self._create_color_button(settings.default_measurement_color)
        measurement_form.addRow("", self._show_measurement_labels)
        measurement_form.addRow("结果文字字体", self._measurement_label_font)
        measurement_form.addRow("结果文字字号", self._measurement_label_size)
        measurement_form.addRow("结果文字颜色", self._measurement_label_color)
        measurement_form.addRow("端点样式", self._endpoint_style_combo)
        measurement_form.addRow("未分类测量线颜色", self._default_measurement_color)

        behavior_group = QGroupBox("打开与导出")
        behavior_form = QFormLayout(behavior_group)
        self._open_view_mode_combo = QComboBox()
        self._open_view_mode_combo.addItem("缺省", OpenImageViewMode.DEFAULT)
        self._open_view_mode_combo.addItem("适合窗口", OpenImageViewMode.FIT)
        self._open_view_mode_combo.addItem("原始像素", OpenImageViewMode.ACTUAL)
        self._open_view_mode_combo.setCurrentIndex(max(0, self._open_view_mode_combo.findData(settings.open_image_view_mode)))
        self._scale_overlay_mode_combo = QComboBox()
        self._scale_overlay_mode_combo.addItem("左上", ScaleOverlayPlacementMode.TOP_LEFT)
        self._scale_overlay_mode_combo.addItem("右上", ScaleOverlayPlacementMode.TOP_RIGHT)
        self._scale_overlay_mode_combo.addItem("左下", ScaleOverlayPlacementMode.BOTTOM_LEFT)
        self._scale_overlay_mode_combo.addItem("右下", ScaleOverlayPlacementMode.BOTTOM_RIGHT)
        self._scale_overlay_mode_combo.addItem("手动选定", ScaleOverlayPlacementMode.MANUAL)
        self._scale_overlay_mode_combo.setCurrentIndex(max(0, self._scale_overlay_mode_combo.findData(settings.scale_overlay_placement_mode)))
        behavior_form.addRow("打开图片默认视图", self._open_view_mode_combo)
        behavior_form.addRow("比例尺叠加位置", self._scale_overlay_mode_combo)

        text_group = QGroupBox("文字工具")
        text_form = QFormLayout(text_group)
        self._text_font = QFontComboBox()
        self._text_font.setCurrentFont(QFont(settings.text_font_family))
        self._text_size = QSpinBox()
        self._text_size.setRange(8, 144)
        self._text_size.setValue(settings.text_font_size)
        self._text_color = self._create_color_button(settings.text_color)
        text_form.addRow("文字字体", self._text_font)
        text_form.addRow("文字字号", self._text_size)
        text_form.addRow("文字颜色", self._text_color)

        layout.addWidget(measurement_group)
        layout.addWidget(behavior_group)
        layout.addWidget(text_group)
        layout.addStretch(1)
        return page

    def _build_current_image_tab(self, document: ImageDocument | None) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        if document is None:
            layout.addWidget(QLabel("当前没有打开的图片。"))
            layout.addStretch(1)
            return page

        group_box = QGroupBox("当前图片类别颜色")
        group_layout = QFormLayout(group_box)
        if not document.fiber_groups:
            group_layout.addRow(QLabel("当前图片还没有已定义类别。"))
        for group in document.sorted_groups():
            button = self._create_color_button(group.color)
            self._group_color_buttons[group.id] = button
            group_layout.addRow(group.display_name(), button)

        scale_box = QGroupBox("当前图片手动比例尺位置")
        scale_layout = QVBoxLayout(scale_box)
        anchor = document.scale_overlay_anchor
        status_text = "当前未设置手动位置。"
        if anchor is not None:
            status_text = f"当前锚点: ({anchor.x:.1f}, {anchor.y:.1f})"
        scale_layout.addWidget(QLabel(status_text))
        hint = QLabel("当比例尺位置为“手动选定”时，可点击下方按钮并在画布中重新指定位置。")
        hint.setWordWrap(True)
        scale_layout.addWidget(hint)
        pick_button = QPushButton("重新选择位置")
        pick_button.clicked.connect(self._trigger_scale_anchor_pick)
        scale_layout.addWidget(pick_button)
        scale_layout.addStretch(1)

        layout.addWidget(group_box)
        layout.addWidget(scale_box)
        layout.addStretch(1)
        return page

    def _create_color_button(self, color_value: str) -> QPushButton:
        button = QPushButton(color_value)
        button.setProperty("color_value", color_value)
        button.clicked.connect(lambda checked=False, target=button: self._choose_color(target))
        self._apply_button_color(button, color_value)
        return button

    def _apply_button_color(self, button: QPushButton, color_value: str) -> None:
        color = QColor(color_value)
        text_color = "#111111" if color.lightnessF() > 0.7 else "#FFFFFF"
        button.setText(color_value)
        button.setStyleSheet(
            f"QPushButton {{ background: {color_value}; color: {text_color}; min-height: 28px; border-radius: 6px; }}"
        )
        button.setProperty("color_value", color_value)

    def _choose_color(self, button: QPushButton) -> None:
        initial = QColor(str(button.property("color_value") or "#FFFFFF"))
        color = QColorDialog.getColor(initial, self, "选择颜色")
        if not color.isValid():
            return
        self._apply_button_color(button, color.name())

    def _trigger_scale_anchor_pick(self) -> None:
        manual_index = self._scale_overlay_mode_combo.findData(ScaleOverlayPlacementMode.MANUAL)
        if manual_index >= 0:
            self._scale_overlay_mode_combo.setCurrentIndex(manual_index)
        self._request_scale_anchor_pick = True
        self.accept()
