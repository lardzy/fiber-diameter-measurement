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
    QFileDialog,
    QFontComboBox,
    QFormLayout,
    QGroupBox,
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fdm.models import ImageDocument
from fdm.settings import (
    AreaModelMapping,
    AppSettings,
    FocusStackProfile,
    MagicSegmentModelVariant,
    MeasurementEndpointStyle,
    OpenImageViewMode,
    ScaleOverlayStyle,
    ScaleOverlayPlacementMode,
    application_root,
    bundle_resource_root,
    resolve_app_relative_path,
    resolve_resource_relative_path,
    to_app_relative_path,
    to_resource_relative_path,
)
from fdm.services.export_service import ExportImageRenderMode, ExportScope, ExportSelection


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:
        if self.view().isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


class NoWheelFontComboBox(QFontComboBox):
    def wheelEvent(self, event) -> None:
        if self.view().isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelSlider(QSlider):
    def wheelEvent(self, event) -> None:
        event.ignore()


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
        self._apply_to_project = QCheckBox("应用到当前项目（当前及后续打开图片）")

        form = QFormLayout()
        form.addRow("真实长度", self._length_spin)
        form.addRow("单位", self._unit_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._apply_to_project)
        layout.addWidget(buttons)

    def values(self) -> tuple[float, str, bool]:
        return (
            self._length_spin.value(),
            self._unit_combo.currentText(),
            self._apply_to_project.isChecked(),
        )


class CalibrationPresetDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        title: str = "新增标定预设",
        initial_name: str = "",
        initial_pixel_distance: float = 100.0,
        initial_actual_distance: float = 10.0,
        initial_unit: str = "um",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("例如 40x 显微镜")
        self._name_edit.setText(initial_name)
        self._pixel_distance_spin = QDoubleSpinBox()
        self._pixel_distance_spin.setDecimals(6)
        self._pixel_distance_spin.setRange(0.000001, 1_000_000.0)
        self._pixel_distance_spin.setValue(initial_pixel_distance)
        self._actual_distance_spin = QDoubleSpinBox()
        self._actual_distance_spin.setDecimals(6)
        self._actual_distance_spin.setRange(0.000001, 1_000_000.0)
        self._actual_distance_spin.setValue(initial_actual_distance)
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["um", "mm"])
        initial_index = self._unit_combo.findText(initial_unit)
        if initial_index >= 0:
            self._unit_combo.setCurrentIndex(initial_index)
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


class FiberGroupDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        title: str = "新增类别",
        initial_label: str = "",
        apply_to_project_default: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText("类别名称")
        self._label_edit.setText(initial_label)
        self._apply_to_project = QCheckBox("应用到当前项目全局")
        self._apply_to_project.setChecked(apply_to_project_default)

        form = QFormLayout()
        form.addRow("类别名称", self._label_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._apply_to_project)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, bool]:
        return self._label_edit.text().strip(), self._apply_to_project.isChecked()


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
        self._combined_overlay = QCheckBox("测量 + 比例尺叠加图 PNG")
        self._combined_overlay.setChecked(selection.include_combined_overlay)
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
        export_layout.addWidget(self._combined_overlay)
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
        self._combined_overlay.toggled.connect(self._update_render_mode_state)

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
        enabled = (
            self._measurement_overlay.isChecked()
            or self._scale_overlay.isChecked()
            or self._combined_overlay.isChecked()
        )
        self._render_mode_combo.setEnabled(enabled)
        self._render_mode_hint.setEnabled(enabled)

    def selection(self) -> ExportSelection:
        return ExportSelection(
            include_measurement_overlay=self._measurement_overlay.isChecked(),
            include_scale_overlay=self._scale_overlay.isChecked(),
            include_combined_overlay=self._combined_overlay.isChecked(),
            include_scale_json=self._scale_json.isChecked(),
            include_excel=self._excel.isChecked(),
            include_csv=self._csv.isChecked(),
            scope=ExportScope.ALL_OPEN if self._scope_all.isChecked() and self._scope_all.isEnabled() else ExportScope.CURRENT,
            render_mode=self._render_mode_combo.currentData(),
        )


class ShortcutHelpDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("快捷键说明")
        self.resize(560, 460)

        self._content = QPlainTextEdit()
        self._content.setReadOnly(True)
        self._content.setPlainText(
            "\n".join(
                [
                    "基础操作",
                    "Ctrl+O  打开图片",
                    "Ctrl+S  保存项目",
                    "Ctrl+W  关闭当前图片",
                    "Ctrl+Shift+W  关闭所有图片",
                    "Ctrl+Z  撤回",
                    "Ctrl+Shift+Z  重做",
                    "Delete / Backspace  删除选中对象",
                    "",
                    "视图与工具",
                    "Space  临时抓手 / 平移画布",
                    "A  在当前工具与浏览工具之间切换",
                    "V  切换面积填充显示",
                    "1-9  切换当前激活纤维类别",
                    "",
                    "面积与魔棒",
                    "R  在正采样点 / 负采样点之间切换",
                    "T  在添加模式 / 剔除模式之间切换",
                    "Enter / F  完成当前魔棒遮罩",
                    "Esc  放弃当前测量线、多边形、自由形状或魔棒草稿",
                    "",
                    "说明",
                    "正采样点用于告诉模型“这里属于目标区域”。",
                    "负采样点用于告诉模型“这里不属于目标区域”。",
                ]
            )
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self._content)
        layout.addWidget(buttons)


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
        self.resize(700, 560)
        self._initial_settings = replace(settings)
        self._document = document
        self._group_color_buttons: dict[str | None, QPushButton] = {}
        self._request_scale_anchor_pick = False

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_measurement_tab(settings), "测量标注")
        self._tabs.addTab(self._build_scale_overlay_tab(settings), "比例尺叠加")
        self._tabs.addTab(self._build_image_processing_tab(settings), "图像处理")
        self._tabs.addTab(self._build_overlay_tab(settings), "叠加标注")
        self._tabs.addTab(self._build_area_models_tab(settings), "面积识别")
        self._tabs.addTab(self._build_current_image_tab(document), "当前图片")

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
            measurement_label_decimals=self._measurement_label_decimals.value(),
            measurement_label_parallel_to_line=self._measurement_label_parallel.isChecked(),
            measurement_label_background_enabled=self._measurement_label_background.isChecked(),
            measurement_endpoint_style=self._endpoint_style_combo.currentData(),
            default_measurement_color=self._default_measurement_color.property("color_value") or self._initial_settings.default_measurement_color,
            open_image_view_mode=self._open_view_mode_combo.currentData(),
            scale_overlay_placement_mode=self._scale_overlay_mode_combo.currentData(),
            scale_overlay_style=self._scale_overlay_style_combo.currentData(),
            scale_overlay_length_value=self._scale_overlay_length_spin.value(),
            scale_overlay_color=self._scale_overlay_color.property("color_value") or self._initial_settings.scale_overlay_color,
            scale_overlay_text_color=self._scale_overlay_text_color.property("color_value") or self._initial_settings.scale_overlay_text_color,
            scale_overlay_font_family=self._scale_overlay_font.currentFont().family(),
            scale_overlay_font_size=self._scale_overlay_font_size.value(),
            text_font_family=self._text_font.currentFont().family(),
            text_font_size=self._text_size.value(),
            text_color=self._text_color.property("color_value") or self._initial_settings.text_color,
            overlay_line_color=self._overlay_line_color.property("color_value") or self._initial_settings.overlay_line_color,
            overlay_line_width=self._overlay_line_width.value(),
            focus_stack_profile=self._focus_stack_profile_combo.currentData(),
            focus_stack_sharpen_strength=self._focus_stack_sharpen_slider.value(),
            magic_segment_auto_small_object_enabled=self._magic_segment_auto_small_object.isChecked(),
            magic_segment_model_variant=self._magic_segment_model_variant_combo.currentData(),
            area_model_mappings=self.area_model_mappings(),
            area_weights_dir=self._area_weights_dir_edit.text().strip(),
            area_vendor_root=self._area_vendor_root_edit.text().strip(),
            area_worker_python=self._area_worker_python_edit.text().strip(),
            calibration_presets=list(self._initial_settings.calibration_presets),
            selected_capture_device_id=self._initial_settings.selected_capture_device_id,
            main_window_geometry=self._initial_settings.main_window_geometry,
            main_window_is_maximized=self._initial_settings.main_window_is_maximized,
        )

    def area_model_mappings(self) -> list[AreaModelMapping]:
        mappings: list[AreaModelMapping] = []
        for row in range(self._area_mapping_table.rowCount()):
            model_item = self._area_mapping_table.item(row, 0)
            file_item = self._area_mapping_table.item(row, 1)
            model_name = (model_item.text().strip() if model_item is not None else "")
            model_file = (file_item.text().strip() if file_item is not None else "")
            if not model_name and not model_file:
                continue
            mappings.append(AreaModelMapping(model_name=model_name, model_file=model_file))
        return mappings

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
        return self._document is not None and self._request_scale_anchor_pick

    def _wrap_settings_page(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _update_focus_stack_sharpen_label(self, value: int) -> None:
        self._focus_stack_sharpen_value_label.setText(f"{value}%")

    def _scale_overlay_length_unit(self) -> str:
        calibration = self._document.calibration if self._document is not None else None
        return calibration.unit if calibration is not None else "px"

    def _build_measurement_tab(self, settings: AppSettings) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        label_group = QGroupBox("结果文字")
        label_form = QFormLayout(label_group)
        self._show_measurement_labels = QCheckBox("在测量线旁显示结果文字")
        self._show_measurement_labels.setChecked(settings.show_measurement_labels)
        self._measurement_label_font = NoWheelFontComboBox()
        self._measurement_label_font.setCurrentFont(QFont(settings.measurement_label_font_family))
        self._measurement_label_size = NoWheelSpinBox()
        self._measurement_label_size.setRange(8, 96)
        self._measurement_label_size.setValue(settings.measurement_label_font_size)
        self._measurement_label_color = self._create_color_button(settings.measurement_label_color)
        self._measurement_label_decimals = NoWheelSpinBox()
        self._measurement_label_decimals.setRange(0, 8)
        self._measurement_label_decimals.setValue(settings.measurement_label_decimals)
        self._measurement_label_parallel = QCheckBox("结果文字与测量线平行")
        self._measurement_label_parallel.setChecked(settings.measurement_label_parallel_to_line)
        self._measurement_label_background = QCheckBox("显示结果文字浅黑底")
        self._measurement_label_background.setChecked(settings.measurement_label_background_enabled)
        self._endpoint_style_combo = NoWheelComboBox()
        self._endpoint_style_combo.addItem("圆点", MeasurementEndpointStyle.CIRCLE)
        self._endpoint_style_combo.addItem("内侧箭头", MeasurementEndpointStyle.ARROW_INSIDE)
        self._endpoint_style_combo.addItem("外侧箭头", MeasurementEndpointStyle.ARROW_OUTSIDE)
        self._endpoint_style_combo.addItem("竖线", MeasurementEndpointStyle.BAR)
        self._endpoint_style_combo.addItem("无端点", MeasurementEndpointStyle.NONE)
        self._endpoint_style_combo.setCurrentIndex(max(0, self._endpoint_style_combo.findData(settings.measurement_endpoint_style)))
        self._default_measurement_color = self._create_color_button(settings.default_measurement_color)
        label_form.addRow("", self._show_measurement_labels)
        label_form.addRow("结果文字字体", self._measurement_label_font)
        label_form.addRow("结果文字字号", self._measurement_label_size)
        label_form.addRow("结果文字颜色", self._measurement_label_color)
        label_form.addRow("结果文字小数位", self._measurement_label_decimals)
        label_form.addRow("", self._measurement_label_parallel)
        label_form.addRow("", self._measurement_label_background)

        measurement_group = QGroupBox("测量线与端点")
        measurement_form = QFormLayout(measurement_group)
        measurement_form.addRow("端点样式", self._endpoint_style_combo)
        measurement_form.addRow("未分类测量线颜色", self._default_measurement_color)

        layout.addWidget(label_group)
        layout.addWidget(measurement_group)
        layout.addStretch(1)
        return self._wrap_settings_page(page)

    def _build_image_processing_tab(self, settings: AppSettings) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        focus_stack_group = QGroupBox("景深合成默认参数")
        focus_stack_form = QFormLayout(focus_stack_group)
        self._focus_stack_profile_combo = NoWheelComboBox()
        self._focus_stack_profile_combo.addItem("锐利优先", FocusStackProfile.SHARP)
        self._focus_stack_profile_combo.addItem("平衡", FocusStackProfile.BALANCED)
        self._focus_stack_profile_combo.addItem("柔和", FocusStackProfile.SOFT)
        self._focus_stack_profile_combo.setCurrentIndex(
            max(0, self._focus_stack_profile_combo.findData(settings.focus_stack_profile))
        )
        sharpen_row = QWidget()
        sharpen_layout = QHBoxLayout(sharpen_row)
        sharpen_layout.setContentsMargins(0, 0, 0, 0)
        self._focus_stack_sharpen_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self._focus_stack_sharpen_slider.setRange(0, 100)
        self._focus_stack_sharpen_slider.setSingleStep(5)
        self._focus_stack_sharpen_slider.setPageStep(10)
        self._focus_stack_sharpen_slider.setTickInterval(5)
        self._focus_stack_sharpen_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._focus_stack_sharpen_slider.setValue(settings.focus_stack_sharpen_strength)
        self._focus_stack_sharpen_value_label = QLabel()
        self._focus_stack_sharpen_value_label.setMinimumWidth(44)
        self._focus_stack_sharpen_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._focus_stack_sharpen_slider.valueChanged.connect(self._update_focus_stack_sharpen_label)
        self._update_focus_stack_sharpen_label(self._focus_stack_sharpen_slider.value())
        sharpen_layout.addWidget(self._focus_stack_sharpen_slider, 1)
        sharpen_layout.addWidget(self._focus_stack_sharpen_value_label)
        focus_stack_hint = QLabel("作为景深合成预览与最终导入的默认参数使用。")
        focus_stack_hint.setWordWrap(True)
        focus_stack_form.addRow("默认合成风格", self._focus_stack_profile_combo)
        focus_stack_form.addRow("默认锐化强度", sharpen_row)
        focus_stack_form.addRow("", focus_stack_hint)

        magic_segment_group = QGroupBox("魔棒分割")
        magic_segment_form = QFormLayout(magic_segment_group)
        self._magic_segment_auto_small_object = QCheckBox("自动启用小物体增强")
        self._magic_segment_auto_small_object.setChecked(settings.magic_segment_auto_small_object_enabled)
        self._magic_segment_model_variant_combo = NoWheelComboBox()
        self._magic_segment_model_variant_combo.addItem("标准 (EdgeSAM)", MagicSegmentModelVariant.EDGE_SAM)
        self._magic_segment_model_variant_combo.addItem("高精度 (EdgeSAM-3x)", MagicSegmentModelVariant.EDGE_SAM_3X)
        self._magic_segment_model_variant_combo.setCurrentIndex(
            max(0, self._magic_segment_model_variant_combo.findData(settings.magic_segment_model_variant))
        )
        magic_hint = QLabel("小物体增强会在目标相对整图较小时自动进行二次局部裁剪推理；高精度模型优先使用 EdgeSAM-3x，缺失时运行时会自动回退。")
        magic_hint.setWordWrap(True)
        magic_segment_form.addRow("", self._magic_segment_auto_small_object)
        magic_segment_form.addRow("分割模型", self._magic_segment_model_variant_combo)
        magic_segment_form.addRow("", magic_hint)

        layout.addWidget(focus_stack_group)
        layout.addWidget(magic_segment_group)
        layout.addStretch(1)
        return self._wrap_settings_page(page)

    def _build_scale_overlay_tab(self, settings: AppSettings) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        display_group = QGroupBox("默认视图")
        display_form = QFormLayout(display_group)
        self._open_view_mode_combo = NoWheelComboBox()
        self._open_view_mode_combo.addItem("缺省", OpenImageViewMode.DEFAULT)
        self._open_view_mode_combo.addItem("适合窗口", OpenImageViewMode.FIT)
        self._open_view_mode_combo.addItem("原始像素", OpenImageViewMode.ACTUAL)
        self._open_view_mode_combo.setCurrentIndex(max(0, self._open_view_mode_combo.findData(settings.open_image_view_mode)))
        display_form.addRow("打开图片默认视图", self._open_view_mode_combo)

        placement_group = QGroupBox("位置与长度")
        placement_form = QFormLayout(placement_group)
        self._scale_overlay_mode_combo = NoWheelComboBox()
        self._scale_overlay_mode_combo.addItem("左上", ScaleOverlayPlacementMode.TOP_LEFT)
        self._scale_overlay_mode_combo.addItem("右上", ScaleOverlayPlacementMode.TOP_RIGHT)
        self._scale_overlay_mode_combo.addItem("左下", ScaleOverlayPlacementMode.BOTTOM_LEFT)
        self._scale_overlay_mode_combo.addItem("右下", ScaleOverlayPlacementMode.BOTTOM_RIGHT)
        self._scale_overlay_mode_combo.addItem("手动选定", ScaleOverlayPlacementMode.MANUAL)
        self._scale_overlay_mode_combo.setCurrentIndex(max(0, self._scale_overlay_mode_combo.findData(settings.scale_overlay_placement_mode)))
        self._scale_overlay_length_spin = NoWheelDoubleSpinBox()
        self._scale_overlay_length_spin.setDecimals(4)
        self._scale_overlay_length_spin.setRange(0.01, 1_000_000.0)
        self._scale_overlay_length_spin.setValue(settings.scale_overlay_length_value)
        self._scale_overlay_length_spin.setSuffix(f" {self._scale_overlay_length_unit()}")
        placement_form.addRow("比例尺叠加位置", self._scale_overlay_mode_combo)
        placement_form.addRow("目标长度", self._scale_overlay_length_spin)

        style_group = QGroupBox("样式")
        style_form = QFormLayout(style_group)
        self._scale_overlay_style_combo = NoWheelComboBox()
        self._scale_overlay_style_combo.addItem("纯线", ScaleOverlayStyle.LINE)
        self._scale_overlay_style_combo.addItem("端点刻度", ScaleOverlayStyle.TICKS)
        self._scale_overlay_style_combo.addItem("粗条", ScaleOverlayStyle.BAR)
        self._scale_overlay_style_combo.setCurrentIndex(max(0, self._scale_overlay_style_combo.findData(settings.scale_overlay_style)))
        self._scale_overlay_color = self._create_color_button(settings.scale_overlay_color)
        self._scale_overlay_font = NoWheelFontComboBox()
        self._scale_overlay_font.setCurrentFont(QFont(settings.scale_overlay_font_family))
        self._scale_overlay_font_size = NoWheelSpinBox()
        self._scale_overlay_font_size.setRange(8, 96)
        self._scale_overlay_font_size.setValue(settings.scale_overlay_font_size)
        self._scale_overlay_text_color = self._create_color_button(settings.scale_overlay_text_color)
        style_form.addRow("比例尺样式", self._scale_overlay_style_combo)
        style_form.addRow("线条颜色", self._scale_overlay_color)
        style_form.addRow("文字字体", self._scale_overlay_font)
        style_form.addRow("文字字号", self._scale_overlay_font_size)
        style_form.addRow("文字颜色", self._scale_overlay_text_color)
        display_hint = QLabel("目标长度按当前图片标定单位输入；未标定时按 px 输入。文字会自动补对比描边。")
        display_hint.setWordWrap(True)
        layout.addWidget(display_group)
        layout.addWidget(placement_group)
        layout.addWidget(style_group)
        layout.addWidget(display_hint)

        layout.addStretch(1)
        return self._wrap_settings_page(page)

    def _build_overlay_tab(self, settings: AppSettings) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        text_group = QGroupBox("文字默认样式")
        text_form = QFormLayout(text_group)
        self._text_font = NoWheelFontComboBox()
        self._text_font.setCurrentFont(QFont(settings.text_font_family))
        self._text_size = NoWheelSpinBox()
        self._text_size.setRange(8, 144)
        self._text_size.setValue(settings.text_font_size)
        self._text_color = self._create_color_button(settings.text_color)
        text_form.addRow("文字字体", self._text_font)
        text_form.addRow("文字字号", self._text_size)
        text_form.addRow("文字颜色", self._text_color)

        shape_group = QGroupBox("图形默认样式")
        shape_form = QFormLayout(shape_group)
        self._overlay_line_color = self._create_color_button(settings.overlay_line_color)
        self._overlay_line_width = NoWheelDoubleSpinBox()
        self._overlay_line_width.setDecimals(1)
        self._overlay_line_width.setRange(0.5, 24.0)
        self._overlay_line_width.setSingleStep(0.5)
        self._overlay_line_width.setValue(settings.overlay_line_width)
        shape_form.addRow("线条颜色", self._overlay_line_color)
        shape_form.addRow("线条宽度", self._overlay_line_width)
        shape_hint = QLabel("适用于矩形、圆形、直线和箭头，首版均为无填充描边。")
        shape_hint.setWordWrap(True)
        shape_form.addRow("", shape_hint)

        layout.addWidget(text_group)
        layout.addWidget(shape_group)
        layout.addStretch(1)
        return self._wrap_settings_page(page)

    def _build_area_models_tab(self, settings: AppSettings) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        area_group = QGroupBox("面积自动识别模型")
        area_layout = QVBoxLayout(area_group)
        area_hint = QLabel("模型名称会用于解析识别标签，权重文件名用于定位本地权重文件。默认映射已参考面积识别项目写入。")
        area_hint.setWordWrap(True)
        area_layout.addWidget(area_hint)
        self._area_mapping_table = QTableWidget(0, 2)
        self._area_mapping_table.setHorizontalHeaderLabels(["模型名称", "权重文件名"])
        self._area_mapping_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._area_mapping_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._area_mapping_table.verticalHeader().setVisible(False)
        self._area_mapping_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._area_mapping_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for mapping in settings.area_model_mappings:
            self._append_area_mapping_row(mapping)
        if self._area_mapping_table.rowCount() == 0:
            self._append_area_mapping_row(AreaModelMapping(model_name="", model_file=""))
        area_layout.addWidget(self._area_mapping_table)
        area_mapping_buttons = QHBoxLayout()
        add_mapping_button = QPushButton("新增映射")
        add_mapping_button.clicked.connect(lambda: self._append_area_mapping_row(AreaModelMapping(model_name="", model_file="")))
        remove_mapping_button = QPushButton("删除选中映射")
        remove_mapping_button.clicked.connect(self._remove_selected_area_mapping_row)
        area_mapping_buttons.addWidget(add_mapping_button)
        area_mapping_buttons.addWidget(remove_mapping_button)
        area_mapping_buttons.addStretch(1)
        area_layout.addLayout(area_mapping_buttons)
        self._area_weights_dir_edit = QLineEdit(settings.area_weights_dir)
        self._area_vendor_root_edit = QLineEdit(settings.area_vendor_root)
        self._area_worker_python_edit = QLineEdit(settings.area_worker_python)
        self._area_worker_python_edit.setPlaceholderText("留空表示自动：打包后优先使用 FiberAreaWorker.exe")
        area_form = QFormLayout()
        area_form.addRow("权重目录", self._with_browse_button(self._area_weights_dir_edit, directory=True, resource_relative=True))
        area_form.addRow("YOLACT vendor 目录", self._with_browse_button(self._area_vendor_root_edit, directory=True, resource_relative=True))
        area_form.addRow("Worker 可执行文件 / Python", self._with_browse_button(self._area_worker_python_edit, directory=False, resource_relative=False))
        area_layout.addLayout(area_form)
        path_hint = QLabel("权重和 vendor 支持相对运行时资源目录填写；Worker 支持相对程序目录填写。保持 Worker 为空时，会自动选择打包后的 FiberAreaWorker 或当前 Python。")
        path_hint.setWordWrap(True)
        area_layout.addWidget(path_hint)

        layout.addWidget(area_group)
        layout.addStretch(1)
        return self._wrap_settings_page(page)

    def _build_current_image_tab(self, document: ImageDocument | None) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        if document is None:
            layout.addWidget(QLabel("当前没有打开的图片。"))
            layout.addStretch(1)
            return self._wrap_settings_page(page)

        group_box = QGroupBox("类别颜色")
        group_layout = QFormLayout(group_box)
        if not document.fiber_groups:
            group_layout.addRow(QLabel("当前图片还没有已定义类别。"))
        for group in document.sorted_groups():
            button = self._create_color_button(group.color)
            self._group_color_buttons[group.id] = button
            group_layout.addRow(group.display_name(), button)

        scale_box = QGroupBox("比例尺锚点")
        scale_layout = QVBoxLayout(scale_box)
        anchor = document.scale_overlay_anchor
        status_text = "当前未设置手动位置。"
        if anchor is not None:
            status_text = f"当前锚点: ({anchor.x:.1f}, {anchor.y:.1f})"
        scale_layout.addWidget(QLabel(status_text))
        hint = QLabel("手动比例尺位置只会在你显式点击“重新选择位置”后进入画布选点；单独修改其它设置不会触发选点。")
        hint.setWordWrap(True)
        scale_layout.addWidget(hint)
        pick_button = QPushButton("重新选择位置")
        pick_button.clicked.connect(self._trigger_scale_anchor_pick)
        scale_layout.addWidget(pick_button)
        scale_layout.addStretch(1)

        layout.addWidget(group_box)
        layout.addWidget(scale_box)
        layout.addStretch(1)
        return self._wrap_settings_page(page)

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

    def _append_area_mapping_row(self, mapping: AreaModelMapping) -> None:
        row = self._area_mapping_table.rowCount()
        self._area_mapping_table.insertRow(row)
        self._area_mapping_table.setItem(row, 0, QTableWidgetItem(mapping.model_name))
        self._area_mapping_table.setItem(row, 1, QTableWidgetItem(mapping.model_file))

    def _remove_selected_area_mapping_row(self) -> None:
        selected_rows = self._area_mapping_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        self._area_mapping_table.removeRow(selected_rows[0].row())
        if self._area_mapping_table.rowCount() == 0:
            self._append_area_mapping_row(AreaModelMapping(model_name="", model_file=""))

    def _browse_path(self, line_edit: QLineEdit, *, directory: bool, resource_relative: bool) -> None:
        current_text = line_edit.text().strip()
        if resource_relative:
            base_root = bundle_resource_root()
            start_path = resolve_resource_relative_path(current_text) if current_text else base_root
        else:
            base_root = application_root()
            start_path = resolve_app_relative_path(current_text) if current_text else base_root
        start_dir = str(start_path if start_path.exists() else base_root)
        if directory:
            path = QFileDialog.getExistingDirectory(self, "选择目录", start_dir)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", start_dir)
        if path:
            if resource_relative:
                line_edit.setText(to_resource_relative_path(path))
            else:
                line_edit.setText(to_app_relative_path(path))

    def _with_browse_button(self, line_edit: QLineEdit, *, directory: bool, resource_relative: bool) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton("浏览...")
        button.clicked.connect(
            lambda checked=False, target=line_edit, is_dir=directory, use_resource_root=resource_relative:
            self._browse_path(target, directory=is_dir, resource_relative=use_resource_root)
        )
        layout.addWidget(button)
        return row


class AreaAutoRecognitionDialog(QDialog):
    def __init__(
        self,
        model_mappings: list[AreaModelMapping],
        *,
        allow_all_scope: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("面积自动识别")
        self.resize(420, 220)
        self._model_combo = QComboBox()
        for mapping in model_mappings:
            self._model_combo.addItem(mapping.model_name, mapping.model_file)
        self._scope_all = QCheckBox("处理全部已打开图片")
        self._scope_all.setEnabled(allow_all_scope)
        self._weight_hint = QLabel("权重文件: -")
        self._weight_hint.setWordWrap(True)
        self._model_combo.currentIndexChanged.connect(self._refresh_weight_hint)

        form = QFormLayout()
        form.addRow("模型", self._model_combo)
        form.addRow("权重文件", self._weight_hint)
        form.addRow("", self._scope_all)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._refresh_weight_hint()

    def _refresh_weight_hint(self) -> None:
        model_file = self._model_combo.currentData() or ""
        self._weight_hint.setText(str(model_file or "-"))

    def values(self) -> tuple[str, str, bool]:
        return (
            self._model_combo.currentText().strip(),
            str(self._model_combo.currentData() or "").strip(),
            self._scope_all.isChecked() and self._scope_all.isEnabled(),
        )
