from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fdm.models import (
    ImageDocument,
    Measurement,
    ObjectAppearanceOverride,
    OverlayAnnotation,
    OverlayAnnotationKind,
    OverlayTextAnchorAlignment,
    OverlayTextLayoutSpec,
    OverlayTextSizeSpace,
    UNCATEGORIZED_LABEL,
)
from fdm.construction_document import make_construction_resolver
from fdm.construction_geometry import (
    ArraySide,
    CircleCenterDiameterDefinition,
    CircleCenterRadiusDefinition,
    CommonTangentDefinition,
    ConcentricCircleDefinition,
    LineDefinition,
    LineExtent,
    OffsetCircleDefinition,
    OffsetParallelDefinition,
    ParallelArrayDefinition,
    PointCircleTangentDefinition,
    TangentTangentRadiusCircleDefinition,
    iter_live_refs,
)
from fdm.settings import AppSettings
from fdm.ui.measurement_results_model import format_measurement_mode, format_measurement_status
from fdm.ui.construction_widgets import construction_kind_label
from fdm.ui.widgets import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelFontComboBox,
    NoWheelSpinBox,
)


class CurrentObjectInspector(QWidget):
    """Selection-driven object metadata and per-object appearance editor."""

    appearanceChangeRequested = Signal(str, str, object)
    measurementGroupChangeRequested = Signal(str, object)
    overlayContentChangeRequested = Signal(str, str)
    overlayTextLayoutChangeRequested = Signal(str, object)
    overlayTextLayoutConversionRequested = Signal(str)
    overlayTextActualSizePreviewRequested = Signal(str)
    constructionDetachRequested = Signal(str)
    constructionLocateSourcesRequested = Signal(str)
    constructionDefinitionChangeRequested = Signal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._document: ImageDocument | None = None
        self._settings = AppSettings()
        self._object_type = ""
        self._object_id = ""
        self._appearance: ObjectAppearanceOverride | None = None
        self._text_layout_spec: OverlayTextLayoutSpec | None = None
        self._construction_definition: object | None = None
        self._construction_distance_field = ""
        self._construction_pixels_per_display_unit = 1.0
        self._view_zoom = 1.0
        self._control_values: dict[str, object] = {}
        self._updating = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        self._stack = QStackedWidget(self)
        root.addWidget(self._stack)

        self._empty_label = QLabel("请选择一个测量、辅助几何或叠加对象。", self._stack)
        self._empty_label.setWordWrap(True)
        self._empty_label.setMinimumWidth(0)
        self._empty_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(self._empty_label)

        self._editor_page = QWidget(self._stack)
        page_layout = QVBoxLayout(self._editor_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)
        self._summary_label = QLabel(self._editor_page)
        self._summary_label.setWordWrap(True)
        self._summary_label.setMinimumWidth(0)
        self._summary_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self._summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        page_layout.addWidget(self._summary_label)

        self._content_group = QGroupBox("文字内容", self._editor_page)
        content_layout = QVBoxLayout(self._content_group)
        self._content_edit = QPlainTextEdit(self._content_group)
        self._content_edit.setMaximumHeight(88)
        self._content_apply_button = QPushButton("应用文字", self._content_group)
        self._content_apply_button.clicked.connect(self._request_content_change)
        content_layout.addWidget(self._content_edit)
        content_layout.addWidget(self._content_apply_button)
        page_layout.addWidget(self._content_group)

        self._text_layout_group = QGroupBox("文字布局与导出", self._editor_page)
        text_layout_root = QVBoxLayout(self._text_layout_group)
        text_layout_root.setContentsMargins(8, 8, 8, 8)
        text_layout_root.setSpacing(8)
        text_layout_form = QFormLayout()
        text_layout_form.setContentsMargins(0, 0, 0, 0)
        self._text_size_space_label = QLabel(self._text_layout_group)
        self._text_size_space_label.setWordWrap(True)
        self._text_size_space_label.setMinimumWidth(0)
        self._text_size_space_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        text_layout_form.addRow("尺寸基准", self._text_size_space_label)

        self._text_image_font_size_label = QLabel("原图字号", self._text_layout_group)
        self._text_image_font_size_spin = NoWheelDoubleSpinBox(self._text_layout_group)
        self._text_image_font_size_spin.setRange(1.0, 8192.0)
        self._text_image_font_size_spin.setDecimals(1)
        self._text_image_font_size_spin.setSingleStep(1.0)
        self._text_image_font_size_spin.setSuffix(" px")
        self._text_image_font_size_spin.setMinimumWidth(0)
        self._text_image_font_size_spin.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self._text_image_font_size_spin.editingFinished.connect(
            self._request_text_image_font_size_change
        )
        text_layout_form.addRow(
            self._text_image_font_size_label,
            self._text_image_font_size_spin,
        )

        anchor_host = QWidget(self._text_layout_group)
        anchor_layout = QGridLayout(anchor_host)
        anchor_layout.setContentsMargins(0, 0, 0, 0)
        anchor_layout.setHorizontalSpacing(4)
        anchor_layout.setVerticalSpacing(4)
        self._text_anchor_group = QButtonGroup(anchor_host)
        self._text_anchor_group.setExclusive(True)
        self._text_anchor_buttons: dict[str, QPushButton] = {}
        anchor_definitions = (
            (OverlayTextAnchorAlignment.TOP_LEFT, "↖", "左上", 0, 0),
            (OverlayTextAnchorAlignment.TOP_CENTER, "↑", "上中", 0, 1),
            (OverlayTextAnchorAlignment.TOP_RIGHT, "↗", "右上", 0, 2),
            (OverlayTextAnchorAlignment.CENTER_LEFT, "←", "左中", 1, 0),
            (OverlayTextAnchorAlignment.CENTER, "•", "中心", 1, 1),
            (OverlayTextAnchorAlignment.CENTER_RIGHT, "→", "右中", 1, 2),
            (OverlayTextAnchorAlignment.BOTTOM_LEFT, "↙", "左下", 2, 0),
            (OverlayTextAnchorAlignment.BOTTOM_CENTER, "↓", "下中", 2, 1),
            (OverlayTextAnchorAlignment.BOTTOM_RIGHT, "↘", "右下", 2, 2),
        )
        for alignment, symbol, description, row, column in anchor_definitions:
            button = QPushButton(symbol, anchor_host)
            button.setCheckable(True)
            button.setToolTip(f"{description}锚点")
            button.setAccessibleName(f"文字{description}锚点")
            button.setMinimumSize(28, 26)
            button.setMaximumWidth(42)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            button.clicked.connect(
                lambda _checked=False, value=alignment: self._request_text_anchor_change(value)
            )
            self._text_anchor_group.addButton(button)
            self._text_anchor_buttons[alignment] = button
            anchor_layout.addWidget(button, row, column)
        anchor_host.setMaximumWidth(138)
        text_layout_form.addRow("锚点", anchor_host)
        text_layout_root.addLayout(text_layout_form)

        self._text_layout_summary_label = QLabel(self._text_layout_group)
        self._text_layout_summary_label.setWordWrap(True)
        self._text_layout_summary_label.setMinimumWidth(0)
        self._text_layout_summary_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        text_layout_root.addWidget(self._text_layout_summary_label)
        self._text_actual_size_button = QPushButton(
            "1:1 查看完整分辨率大小",
            self._text_layout_group,
        )
        self._text_actual_size_button.setToolTip(
            "将画布切换到原始像素 1:1，并把当前文字移到视图中央；不会修改对象。"
        )
        self._text_actual_size_button.clicked.connect(
            self._request_text_actual_size_preview
        )
        text_layout_root.addWidget(self._text_actual_size_button)

        self._legacy_text_layout_label = QLabel(self._text_layout_group)
        self._legacy_text_layout_label.setWordWrap(True)
        self._legacy_text_layout_label.setMinimumWidth(0)
        self._legacy_text_layout_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        text_layout_root.addWidget(self._legacy_text_layout_label)
        self._text_layout_conversion_button = QPushButton(
            "按当前画布外观转换",
            self._text_layout_group,
        )
        self._text_layout_conversion_button.setToolTip(
            "把当前屏幕中的文字大小换算成原图像素字号，并改用随图像缩放的布局。"
        )
        self._text_layout_conversion_button.clicked.connect(
            self._request_text_layout_conversion
        )
        text_layout_root.addWidget(self._text_layout_conversion_button)
        page_layout.addWidget(self._text_layout_group)

        self._metadata_group = QGroupBox("对象属性", self._editor_page)
        metadata_form = QFormLayout(self._metadata_group)
        self._group_combo = NoWheelComboBox(self._metadata_group)
        self._group_combo.setMinimumWidth(0)
        self._group_combo.setMinimumContentsLength(10)
        self._group_combo.setSizeAdjustPolicy(
            NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._group_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self._group_combo.currentTextChanged.connect(self._group_combo.setToolTip)
        self._group_combo.activated.connect(self._request_group_change)
        metadata_form.addRow("所属类别", self._group_combo)
        page_layout.addWidget(self._metadata_group)

        self._construction_group = QGroupBox("构造关系", self._editor_page)
        construction_layout = QVBoxLayout(self._construction_group)
        self._construction_definition_label = QLabel(self._construction_group)
        self._construction_definition_label.setWordWrap(True)
        self._construction_definition_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        construction_layout.addWidget(self._construction_definition_label)
        self._construction_parameters_widget = QWidget(self._construction_group)
        self._construction_parameters_form = QFormLayout(
            self._construction_parameters_widget
        )
        self._construction_parameters_form.setContentsMargins(0, 2, 0, 2)
        self._construction_distance_label = QLabel(
            "距离",
            self._construction_parameters_widget,
        )
        self._construction_distance_spin = NoWheelDoubleSpinBox(
            self._construction_parameters_widget
        )
        self._construction_distance_spin.setDecimals(6)
        self._construction_distance_spin.editingFinished.connect(
            self._request_construction_distance_change
        )
        self._construction_parameters_form.addRow(
            self._construction_distance_label,
            self._construction_distance_spin,
        )
        self._construction_count_spin = NoWheelSpinBox(
            self._construction_parameters_widget
        )
        self._construction_count_spin.setRange(1, 10_000)
        self._construction_count_spin.editingFinished.connect(
            self._request_construction_count_change
        )
        self._construction_parameters_form.addRow(
            "每侧数量",
            self._construction_count_spin,
        )
        self._construction_side_combo = NoWheelComboBox(
            self._construction_parameters_widget
        )
        self._construction_side_combo.addItem("正向单侧", ArraySide.POSITIVE.value)
        self._construction_side_combo.addItem("负向单侧", ArraySide.NEGATIVE.value)
        self._construction_side_combo.addItem("双侧", ArraySide.BOTH.value)
        self._construction_side_combo.activated.connect(
            self._request_construction_side_change
        )
        self._construction_parameters_form.addRow(
            "阵列方向",
            self._construction_side_combo,
        )
        self._construction_extent_combo = NoWheelComboBox(
            self._construction_parameters_widget
        )
        self._construction_extent_combo.addItem("有限线段", LineExtent.SEGMENT.value)
        self._construction_extent_combo.addItem("射线", LineExtent.RAY.value)
        self._construction_extent_combo.addItem("两端无限", LineExtent.INFINITE.value)
        self._construction_extent_combo.activated.connect(
            self._request_construction_extent_change
        )
        self._construction_parameters_form.addRow(
            "线范围",
            self._construction_extent_combo,
        )
        construction_layout.addWidget(self._construction_parameters_widget)
        construction_buttons = QHBoxLayout()
        self._construction_locate_sources_button = QPushButton(
            "定位源对象",
            self._construction_group,
        )
        self._construction_locate_sources_button.clicked.connect(
            lambda: self.constructionLocateSourcesRequested.emit(self._object_id)
        )
        construction_buttons.addWidget(self._construction_locate_sources_button)
        self._construction_detach_button = QPushButton(
            "解除关联",
            self._construction_group,
        )
        self._construction_detach_button.clicked.connect(
            lambda: self.constructionDetachRequested.emit(self._object_id)
        )
        construction_buttons.addWidget(self._construction_detach_button)
        construction_layout.addLayout(construction_buttons)
        page_layout.addWidget(self._construction_group)

        self._appearance_group = QGroupBox("对象样式覆盖", self._editor_page)
        appearance_form = QFormLayout(self._appearance_group)
        self._appearance_form = appearance_form
        self._stroke_color_button = self._make_color_button("stroke_color")
        self._stroke_width_spin = NoWheelDoubleSpinBox(self._appearance_group)
        self._stroke_width_spin.setRange(0.5, 24.0)
        self._stroke_width_spin.setDecimals(1)
        self._stroke_width_spin.setSingleStep(0.5)
        self._stroke_width_spin.editingFinished.connect(
            lambda: self._request_appearance_change("stroke_width", self._stroke_width_spin.value())
        )
        self._marker_scale_spin = NoWheelDoubleSpinBox(self._appearance_group)
        self._marker_scale_spin.setRange(0.25, 4.0)
        self._marker_scale_spin.setDecimals(2)
        self._marker_scale_spin.setSingleStep(0.25)
        self._marker_scale_spin.editingFinished.connect(
            lambda: self._request_appearance_change("marker_scale", self._marker_scale_spin.value())
        )
        self._text_color_button = self._make_color_button("text_color")
        self._font_combo = NoWheelFontComboBox(self._appearance_group)
        self._font_combo.setMinimumWidth(0)
        self._font_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self._font_combo.activated.connect(
            lambda _index: self._request_font_change(self._font_combo.currentFont())
        )
        self._font_size_spin = NoWheelSpinBox(self._appearance_group)
        self._font_size_spin.setRange(8, 144)
        self._font_size_spin.editingFinished.connect(
            lambda: self._request_appearance_change("font_size", self._font_size_spin.value())
        )
        appearance_form.addRow("线条/点颜色", self._stroke_color_button)
        appearance_form.addRow("线条宽度", self._stroke_width_spin)
        appearance_form.addRow("计数点尺寸", self._marker_scale_spin)
        appearance_form.addRow("文字颜色", self._text_color_button)
        appearance_form.addRow("字体", self._font_combo)
        appearance_form.addRow("字号", self._font_size_spin)
        page_layout.addWidget(self._appearance_group)

        self._inheritance_label = QLabel("当前对象继承类别和首选项样式。", self._editor_page)
        self._inheritance_label.setWordWrap(True)
        self._inheritance_label.setMinimumWidth(0)
        self._inheritance_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        page_layout.addWidget(self._inheritance_label)
        self._reset_button = QPushButton("恢复继承样式", self._editor_page)
        self._reset_button.clicked.connect(self._request_reset)
        page_layout.addWidget(self._reset_button)
        page_layout.addStretch(1)
        self._stack.addWidget(self._editor_page)

    def set_context(
        self,
        document: ImageDocument | None,
        *,
        settings: AppSettings,
        measurement_ids: Sequence[str] = (),
        overlay_id: str | None = None,
        construction_id: str | None = None,
        view_zoom: float = 1.0,
    ) -> None:
        self._document = document
        self._settings = settings
        try:
            normalized_zoom = float(view_zoom)
        except (TypeError, ValueError):
            normalized_zoom = 1.0
        self._view_zoom = (
            normalized_zoom
            if math.isfinite(normalized_zoom) and normalized_zoom > 0.0
            else 1.0
        )
        if document is None:
            self._show_empty("当前没有打开的图片。")
            return
        unique_measurements = tuple(dict.fromkeys(item for item in measurement_ids if item))
        if len(unique_measurements) > 1:
            self._show_empty(f"已选择 {len(unique_measurements)} 个测量对象；批量样式编辑暂未开放。")
            return
        if overlay_id:
            overlay = document.get_overlay_annotation(overlay_id)
            if overlay is not None:
                self._load_overlay(overlay)
                return
        if construction_id:
            getter = getattr(document, "get_construction_entity", None)
            entity = getter(construction_id) if callable(getter) else None
            if entity is not None:
                self._load_construction(entity)
                return
        measurement_id = unique_measurements[0] if unique_measurements else document.view_state.selected_measurement_id
        measurement = document.get_measurement(measurement_id) if measurement_id else None
        if measurement is not None:
            self._load_measurement(measurement)
            return
        self._show_empty("请选择一个测量、辅助几何或叠加对象。")

    def _show_empty(self, text: str) -> None:
        self._object_type = ""
        self._object_id = ""
        self._appearance = None
        self._text_layout_spec = None
        self._construction_definition = None
        self._empty_label.setText(text)
        self._stack.setCurrentWidget(self._empty_label)

    def _load_measurement(self, measurement: Measurement) -> None:
        document = self._document
        if document is None:
            return
        self._updating = True
        try:
            self._object_type = "measurement"
            self._object_id = measurement.id
            self._appearance = measurement.appearance
            group = document.get_group(measurement.fiber_group_id)
            category = group.display_name() if group is not None else UNCATEGORIZED_LABEL
            unit = measurement.display_unit(document.calibration)
            if measurement.measurement_kind == "area":
                pixel_text = f"{(measurement.area_px or 0.0):.4g} px²"
            elif measurement.measurement_kind == "count":
                pixel_text = "1 个"
            else:
                pixel_text = f"{(measurement.diameter_px or 0.0):.4g} px"
            self._summary_label.setText(
                f"{self._kind_label(measurement.measurement_kind)} · {measurement.display_value():.4g} {unit}\n"
                f"像素值：{pixel_text}\n类别：{category}\n模式：{format_measurement_mode(measurement.mode)}\n"
                f"状态：{format_measurement_status(measurement.status)}\n"
                f"创建时间：{self._friendly_time(measurement.created_at)}"
            )
            self._content_group.hide()
            self._text_layout_group.hide()
            self._construction_group.hide()
            self._metadata_group.show()
            self._appearance_group.show()
            self._inheritance_label.show()
            self._reset_button.show()
            self._populate_group_combo(measurement.fiber_group_id)
            group_color = group.color if group is not None else self._settings.default_measurement_color
            is_count = measurement.measurement_kind == "count"
            label_style = (
                self._settings.area_measurement_label_style
                if measurement.measurement_kind == "area"
                else self._settings.length_measurement_label_style
            )
            self._set_appearance_controls(
                stroke_color=(measurement.appearance.stroke_color if measurement.appearance else None) or group_color,
                stroke_width=(measurement.appearance.stroke_width if measurement.appearance else None) or 2.0,
                marker_scale=(measurement.appearance.marker_scale if measurement.appearance else None) or 1.0,
                text_color=(measurement.appearance.text_color if measurement.appearance else None)
                or (self._settings.count_number_color if is_count else label_style.color),
                font_family=(measurement.appearance.font_family if measurement.appearance else None)
                or (
                    self._settings.count_number_font_family
                    if is_count
                    else label_style.font_family
                ),
                font_size=(measurement.appearance.font_size if measurement.appearance else None)
                or (
                    self._settings.count_number_font_size
                    if is_count
                    else label_style.font_size
                ),
                show_stroke=True,
                show_stroke_width=not is_count,
                show_marker=is_count,
                show_text=True,
            )
            self._finish_load()
        finally:
            self._updating = False

    def _load_construction(self, entity: object) -> None:
        self._updating = True
        try:
            self._object_type = "construction"
            self._object_id = str(getattr(entity, "id", ""))
            self._appearance = None
            self._text_layout_spec = None
            self._construction_definition = definition = getattr(
                entity,
                "definition",
                None,
            )
            name = str(getattr(entity, "name", "") or construction_kind_label(entity))
            state_bits = [
                "显示" if bool(getattr(entity, "visible", True)) else "隐藏",
                "锁定" if bool(getattr(entity, "locked", False)) else "可编辑",
                (
                    "可捕捉"
                    if bool(
                        getattr(
                            entity,
                            "snap_enabled",
                            getattr(entity, "snappable", True),
                        )
                    )
                    else "不捕捉"
                ),
            ]
            resolved = (
                make_construction_resolver(self._document).resolve(self._object_id)
                if self._document is not None
                else None
            )
            resolution_valid = bool(getattr(resolved, "valid", False))
            resolution_error = getattr(resolved, "error", None)
            resolution_reason = str(
                getattr(resolution_error, "message", "") or ""
            )
            resolution_text = "有效" if resolution_valid else "不可解"
            if resolution_reason:
                resolution_text = f"{resolution_text}\n原因：{resolution_reason}"
            self._summary_label.setText(
                f"{name} · {construction_kind_label(entity)}\n"
                f"解析状态：{resolution_text}\n"
                f"对象状态：{' · '.join(state_bits)}\nID：{self._object_id}"
            )
            sources = [
                reference.object_id
                for reference in iter_live_refs(definition)
            ]
            source_text = "、".join(dict.fromkeys(sources)) if sources else "无实时源对象"
            self._construction_definition_label.setText(
                f"定义：{type(definition).__name__}\n源对象：{source_text}"
            )
            locked = bool(getattr(entity, "locked", False))
            self._construction_locate_sources_button.setEnabled(bool(sources))
            self._construction_detach_button.setEnabled(bool(sources) and not locked)
            self._set_construction_parameter_controls(definition)
            self._construction_parameters_widget.setEnabled(not locked)
            self._content_group.hide()
            self._text_layout_group.hide()
            self._metadata_group.hide()
            self._construction_group.hide()
            self._appearance_group.show()
            self._inheritance_label.show()
            self._reset_button.show()
            self._appearance_group.hide()
            self._inheritance_label.hide()
            self._reset_button.hide()
            self._construction_group.show()
            self._stack.setCurrentWidget(self._editor_page)
        finally:
            self._updating = False

    def _set_construction_parameter_controls(self, definition: object) -> None:
        distance_field = ""
        distance_label = "距离"
        signed_distance = False
        if isinstance(definition, (CircleCenterRadiusDefinition, ConcentricCircleDefinition)):
            distance_field = "radius"
            distance_label = "半径"
        elif isinstance(definition, CircleCenterDiameterDefinition):
            distance_field = "diameter"
            distance_label = "直径"
        elif isinstance(definition, ParallelArrayDefinition):
            distance_field = "spacing"
            distance_label = "阵列间距"
        elif isinstance(definition, OffsetParallelDefinition):
            distance_field = "offset"
            distance_label = "偏移距离"
            signed_distance = True
        elif isinstance(definition, OffsetCircleDefinition):
            distance_field = "offset"
            distance_label = "半径偏移"
            signed_distance = True
        elif isinstance(definition, TangentTangentRadiusCircleDefinition):
            distance_field = "radius"
            distance_label = "固定半径"

        calibration = self._document.calibration if self._document is not None else None
        factor = (
            float(calibration.pixels_per_unit)
            if calibration is not None
            else 1.0
        )
        if not math.isfinite(factor) or factor <= 0.0:
            factor = 1.0
        self._construction_pixels_per_display_unit = factor
        self._construction_distance_field = distance_field
        unit = str(calibration.unit) if calibration is not None else "px"

        show_distance = bool(distance_field)
        self._construction_parameters_form.setRowVisible(
            self._construction_distance_spin,
            show_distance,
        )
        if show_distance:
            value_px = float(getattr(definition, distance_field))
            maximum = 1_000_000_000.0 / factor
            minimum = -maximum if signed_distance else max(1e-9, 1e-6 / factor)
            self._construction_distance_label.setText(distance_label)
            self._construction_distance_spin.blockSignals(True)
            self._construction_distance_spin.setRange(minimum, maximum)
            self._construction_distance_spin.setSingleStep(max(1e-6, 1.0 / factor))
            self._construction_distance_spin.setSuffix(f" {unit}")
            self._construction_distance_spin.setValue(value_px / factor)
            self._construction_distance_spin.blockSignals(False)

        show_array = isinstance(definition, ParallelArrayDefinition)
        self._construction_parameters_form.setRowVisible(
            self._construction_count_spin,
            show_array,
        )
        self._construction_parameters_form.setRowVisible(
            self._construction_side_combo,
            show_array,
        )
        if show_array:
            self._construction_count_spin.blockSignals(True)
            self._construction_count_spin.setValue(definition.count)
            self._construction_count_spin.blockSignals(False)
            self._construction_side_combo.blockSignals(True)
            index = self._construction_side_combo.findData(definition.side.value)
            self._construction_side_combo.setCurrentIndex(max(0, index))
            self._construction_side_combo.blockSignals(False)

        extent = getattr(definition, "extent", None)
        # 水平/垂直工具是定位用的无限构造线。它们只暴露一个平移柄，
        # 因而不能把范围改成射线或线段，否则会得到无法调长的 1 px 对象。
        show_extent = isinstance(extent, LineExtent) and self._construction_extent_is_editable(
            definition
        )
        self._construction_parameters_form.setRowVisible(
            self._construction_extent_combo,
            show_extent,
        )
        if show_extent:
            self._construction_extent_combo.blockSignals(True)
            index = self._construction_extent_combo.findData(extent.value)
            self._construction_extent_combo.setCurrentIndex(max(0, index))
            self._construction_extent_combo.blockSignals(False)
        self._construction_parameters_widget.setVisible(
            show_distance or show_array or show_extent
        )

    def _emit_construction_definition_change(self, **changes: object) -> None:
        if (
            self._updating
            or self._object_type != "construction"
            or not self._object_id
            or self._construction_definition is None
        ):
            return
        try:
            next_definition = replace(self._construction_definition, **changes)
        except (TypeError, ValueError):
            self._set_construction_parameter_controls(self._construction_definition)
            return
        if next_definition == self._construction_definition:
            return
        self._construction_definition = next_definition
        self.constructionDefinitionChangeRequested.emit(
            self._object_id,
            next_definition,
        )

    def _request_construction_distance_change(self) -> None:
        field_name = self._construction_distance_field
        if not field_name:
            return
        self._emit_construction_definition_change(
            **{
                field_name: (
                    self._construction_distance_spin.value()
                    * self._construction_pixels_per_display_unit
                )
            }
        )

    def _request_construction_count_change(self) -> None:
        self._emit_construction_definition_change(
            count=self._construction_count_spin.value()
        )

    def _request_construction_side_change(self, _index: int = -1) -> None:
        value = self._construction_side_combo.currentData()
        if value is not None:
            self._emit_construction_definition_change(side=ArraySide(str(value)))

    def _request_construction_extent_change(self, _index: int = -1) -> None:
        if not self._construction_extent_is_editable(self._construction_definition):
            self._set_construction_parameter_controls(self._construction_definition)
            return
        value = self._construction_extent_combo.currentData()
        if value is not None:
            self._emit_construction_definition_change(extent=LineExtent(str(value)))

    @staticmethod
    def _construction_extent_is_editable(definition: object) -> bool:
        return (
            isinstance(definition, LineDefinition)
            and definition.axis_constraint is None
        ) or isinstance(
            definition,
            (PointCircleTangentDefinition, CommonTangentDefinition),
        )

    def _load_overlay(self, overlay: OverlayAnnotation) -> None:
        self._updating = True
        try:
            self._object_type = "overlay"
            self._object_id = overlay.id
            self._appearance = overlay.appearance
            kind = overlay.normalized_kind()
            self._summary_label.setText(
                f"叠加对象：{self._overlay_kind_label(kind)}\n创建时间：{self._friendly_time(overlay.created_at)}"
            )
            self._metadata_group.hide()
            is_text = kind == OverlayAnnotationKind.TEXT
            self._content_group.setVisible(is_text)
            self._content_edit.setPlainText(overlay.content if is_text else "")
            self._text_layout_group.setVisible(is_text)
            resolved_font_size = (
                (overlay.appearance.font_size if overlay.appearance else None)
                or self._settings.text_font_size
            )
            if is_text:
                self._set_text_layout_controls(
                    overlay,
                    legacy_font_size=float(resolved_font_size),
                )
            else:
                self._text_layout_spec = None
            self._set_appearance_controls(
                stroke_color=(overlay.appearance.stroke_color if overlay.appearance else None)
                or self._settings.overlay_line_color,
                stroke_width=(overlay.appearance.stroke_width if overlay.appearance else None)
                or self._settings.overlay_line_width,
                marker_scale=1.0,
                text_color=(overlay.appearance.text_color if overlay.appearance else None)
                or self._settings.text_color,
                font_family=(overlay.appearance.font_family if overlay.appearance else None)
                or self._settings.text_font_family,
                font_size=resolved_font_size,
                show_stroke=not is_text,
                show_stroke_width=not is_text,
                show_marker=False,
                show_text=is_text,
                show_font_size=is_text and overlay.text_layout is None,
            )
            self._finish_load()
        finally:
            self._updating = False

    def _finish_load(self) -> None:
        has_override = self._appearance is not None and not self._appearance.is_empty()
        self._inheritance_label.setText(
            "当前对象含独立样式；未覆盖的项目继续继承类别或首选项。"
            if has_override
            else "当前对象继承类别和首选项样式。"
        )
        self._reset_button.setEnabled(has_override)
        self._stack.setCurrentWidget(self._editor_page)

    def _set_text_layout_controls(
        self,
        overlay: OverlayAnnotation,
        *,
        legacy_font_size: float,
    ) -> None:
        spec = overlay.text_layout
        self._text_layout_spec = spec
        is_implicit_legacy = spec is None
        size_space = (
            OverlayTextSizeSpace.LEGACY_OUTPUT_PX
            if spec is None
            else spec.size_space
        )
        is_image_space = size_space == OverlayTextSizeSpace.IMAGE_PX
        if spec is None:
            anchor_alignment = OverlayTextAnchorAlignment.TOP_LEFT
            fixed_output_size = max(1.0, float(legacy_font_size))
            converted_image_size = min(
                8192.0,
                max(1.0, fixed_output_size / self._view_zoom),
            )
            current_canvas_size = fixed_output_size
            full_resolution_size = fixed_output_size
            editor_size = converted_image_size
        else:
            anchor_alignment = spec.anchor_alignment
            explicit_size = max(1.0, float(spec.image_font_size_px))
            if is_image_space:
                current_canvas_size = explicit_size * self._view_zoom
                full_resolution_size = explicit_size
                editor_size = explicit_size
            else:
                current_canvas_size = explicit_size
                full_resolution_size = explicit_size
                editor_size = explicit_size

        if is_image_space:
            self._text_size_space_label.setText("随图像缩放（推荐）")
        elif is_implicit_legacy:
            self._text_size_space_label.setText("旧版固定输出像素")
        else:
            self._text_size_space_label.setText("固定输出像素")
        if is_image_space:
            self._text_image_font_size_label.setText("原图字号")
        elif is_implicit_legacy:
            self._text_image_font_size_label.setText("转换后原图字号")
        else:
            self._text_image_font_size_label.setText("固定输出字号")
        self._text_image_font_size_spin.blockSignals(True)
        self._text_image_font_size_spin.setValue(editor_size)
        self._text_image_font_size_spin.blockSignals(False)
        self._text_image_font_size_spin.setEnabled(not is_implicit_legacy)
        self._text_image_font_size_spin.setToolTip(
            "完整分辨率输出时的字体像素高度。"
            if is_image_space
            else (
                "按当前画布外观转换后将采用的原图像素字号。"
                if is_implicit_legacy
                else "当前固定用于屏幕和各种导出模式的字体像素高度。"
            )
        )
        for alignment, button in self._text_anchor_buttons.items():
            button.blockSignals(True)
            button.setChecked(alignment == anchor_alignment)
            button.setEnabled(not is_implicit_legacy)
            button.blockSignals(False)

        self._text_layout_summary_label.setText(
            f"当前画布约 {self._format_pixel_size(current_canvas_size)} px"
            f" · 完整分辨率导出 {self._format_pixel_size(full_resolution_size)} px"
        )
        is_legacy = not is_image_space
        self._legacy_text_layout_label.setVisible(is_legacy)
        self._text_layout_conversion_button.setVisible(is_legacy)
        if is_implicit_legacy:
            self._legacy_text_layout_label.setText(
                "此文字沿用旧项目的固定输出像素和左上锚点。"
                "原始分辨率导出时可能显得过小；转换前不会改动旧项目外观。"
            )
        elif is_legacy:
            self._legacy_text_layout_label.setText(
                "此文字使用固定输出像素。可按当前画布外观转换为随图像缩放，"
                "使画布与完整分辨率导出的相对大小一致。"
            )

    @staticmethod
    def _format_pixel_size(value: float) -> str:
        finite = float(value)
        if abs(finite - round(finite)) <= 0.05:
            return str(int(round(finite)))
        if finite < 10.0:
            return f"{finite:.1f}"
        return f"{finite:.3g}"

    def _populate_group_combo(self, selected_group_id: str | None) -> None:
        document = self._document
        self._group_combo.blockSignals(True)
        self._group_combo.clear()
        self._group_combo.addItem(UNCATEGORIZED_LABEL, None)
        if document is not None:
            for group in document.sorted_groups():
                self._group_combo.addItem(group.display_name(), group.id)
        index = self._group_combo.findData(selected_group_id)
        self._group_combo.setCurrentIndex(max(0, index))
        self._group_combo.blockSignals(False)
        self._group_combo.setToolTip(self._group_combo.currentText())

    def _set_appearance_controls(
        self,
        *,
        stroke_color: str,
        stroke_width: float,
        marker_scale: float,
        text_color: str,
        font_family: str,
        font_size: int,
        show_stroke: bool,
        show_stroke_width: bool,
        show_marker: bool,
        show_text: bool,
        show_font_size: bool | None = None,
    ) -> None:
        show_font_size = show_text if show_font_size is None else bool(show_font_size)
        self._set_color_button(self._stroke_color_button, stroke_color)
        self._set_color_button(self._text_color_button, text_color)
        self._stroke_color_button.setVisible(show_stroke)
        self._stroke_width_spin.setVisible(show_stroke_width)
        self._marker_scale_spin.setVisible(show_marker)
        self._text_color_button.setVisible(show_text)
        self._font_combo.setVisible(show_text)
        self._font_size_spin.setVisible(show_font_size)
        self._appearance_form.setRowVisible(self._stroke_color_button, show_stroke)
        self._appearance_form.setRowVisible(self._stroke_width_spin, show_stroke_width)
        self._appearance_form.setRowVisible(self._marker_scale_spin, show_marker)
        self._appearance_form.setRowVisible(self._text_color_button, show_text)
        self._appearance_form.setRowVisible(self._font_combo, show_text)
        self._appearance_form.setRowVisible(self._font_size_spin, show_font_size)
        self._stroke_width_spin.setValue(float(stroke_width))
        self._marker_scale_spin.setValue(float(marker_scale))
        self._font_combo.setCurrentFont(QFont(font_family))
        self._font_size_spin.setValue(int(font_size))
        self._control_values = {
            "stroke_color": str(self._stroke_color_button.property("color_value") or ""),
            "stroke_width": self._stroke_width_spin.value(),
            "marker_scale": self._marker_scale_spin.value(),
            "text_color": str(self._text_color_button.property("color_value") or ""),
            "font_family": self._font_combo.currentFont().family(),
            "font_size": self._font_size_spin.value(),
        }

    def _make_color_button(self, field_name: str) -> QPushButton:
        button = QPushButton("选择颜色", self)
        button.clicked.connect(lambda _checked=False, target=button, field=field_name: self._choose_color(target, field))
        return button

    @staticmethod
    def _set_color_button(button: QPushButton, color: str) -> None:
        resolved = QColor(color)
        value = resolved.name().upper() if resolved.isValid() else "#FFFFFF"
        button.setProperty("color_value", value)
        button.setText(value)
        foreground = "#111111" if resolved.lightnessF() > 0.62 else "#FFFFFF"
        button.setStyleSheet(f"background: {value}; color: {foreground};")

    def _choose_color(self, button: QPushButton, field_name: str) -> None:
        initial = QColor(str(button.property("color_value") or "#FFFFFF"))
        color = QColorDialog.getColor(initial, self, "选择对象颜色")
        if not color.isValid():
            return
        value = color.name().upper()
        self._set_color_button(button, value)
        self._request_appearance_change(field_name, value)

    def _request_font_change(self, font: QFont) -> None:
        if not self._updating:
            self._request_appearance_change("font_family", font.family())

    def _request_appearance_change(self, field_name: str, value: object) -> None:
        if self._updating or not self._object_type or not self._object_id:
            return
        previous = self._control_values.get(field_name)
        if isinstance(previous, (int, float)) and isinstance(value, (int, float)):
            if abs(float(previous) - float(value)) <= 1e-9:
                return
        elif previous == value:
            return
        self._control_values[field_name] = value
        base = self._appearance or ObjectAppearanceOverride()
        updated = base.clone(**{field_name: value})
        self.appearanceChangeRequested.emit(self._object_type, self._object_id, updated)

    def _request_reset(self) -> None:
        if self._object_type and self._object_id:
            self.appearanceChangeRequested.emit(self._object_type, self._object_id, None)

    def _request_group_change(self, _index: int) -> None:
        if not self._updating and self._object_type == "measurement" and self._object_id:
            self.measurementGroupChangeRequested.emit(self._object_id, self._group_combo.currentData())

    def _request_content_change(self) -> None:
        if self._object_type == "overlay" and self._object_id:
            self.overlayContentChangeRequested.emit(self._object_id, self._content_edit.toPlainText())

    def _request_text_image_font_size_change(self) -> None:
        spec = self._text_layout_spec
        if (
            self._updating
            or self._object_type != "overlay"
            or not self._object_id
            or spec is None
        ):
            return
        value = float(self._text_image_font_size_spin.value())
        if abs(value - float(spec.image_font_size_px)) <= 1e-9:
            return
        updated = replace(spec, image_font_size_px=value)
        self._text_layout_spec = updated
        self.overlayTextLayoutChangeRequested.emit(self._object_id, updated)

    def _request_text_anchor_change(self, alignment: str) -> None:
        spec = self._text_layout_spec
        if (
            self._updating
            or self._object_type != "overlay"
            or not self._object_id
            or spec is None
        ):
            return
        normalized = OverlayTextAnchorAlignment.normalize(alignment)
        if normalized == spec.anchor_alignment:
            return
        updated = replace(spec, anchor_alignment=normalized)
        self._text_layout_spec = updated
        self.overlayTextLayoutChangeRequested.emit(self._object_id, updated)

    def _request_text_layout_conversion(self) -> None:
        if self._object_type != "overlay" or not self._object_id:
            return
        overlay = (
            self._document.get_overlay_annotation(self._object_id)
            if self._document is not None
            else None
        )
        if overlay is None or not overlay.is_text():
            return
        if (
            overlay.text_layout is not None
            and overlay.text_layout.size_space == OverlayTextSizeSpace.IMAGE_PX
        ):
            return
        self.overlayTextLayoutConversionRequested.emit(self._object_id)

    def _request_text_actual_size_preview(self) -> None:
        if self._object_type != "overlay" or not self._object_id:
            return
        overlay = (
            self._document.get_overlay_annotation(self._object_id)
            if self._document is not None
            else None
        )
        if overlay is not None and overlay.is_text():
            self.overlayTextActualSizePreviewRequested.emit(self._object_id)

    @staticmethod
    def _kind_label(kind: str) -> str:
        return {"line": "线段", "polyline": "折线", "area": "面积", "count": "计数点"}.get(kind, kind)

    @staticmethod
    def _overlay_kind_label(kind: str) -> str:
        return {
            OverlayAnnotationKind.TEXT: "文字",
            OverlayAnnotationKind.RECT: "矩形",
            OverlayAnnotationKind.CIRCLE: "圆形",
            OverlayAnnotationKind.LINE: "直线",
            OverlayAnnotationKind.ARROW: "箭头",
        }.get(kind, kind)

    @staticmethod
    def _friendly_time(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "—"
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone()
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return raw.replace("T", " ").removesuffix("Z")
