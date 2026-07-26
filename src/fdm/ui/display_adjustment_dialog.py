from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fdm.image_processing_models import DisplayTransform, ImageOperationSpec
from fdm.raster import RasterPixelType, RasterPlane
from fdm.ui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox


class DisplayAdjustmentAction(str, Enum):
    APPLY_DISPLAY = "apply_display"
    GENERATE_DERIVED = "generate_derived"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class DisplayBakePlan:
    supported: bool
    operations: tuple[ImageOperationSpec, ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class DisplayAdjustmentResult:
    action: DisplayAdjustmentAction
    transform: DisplayTransform
    source_sha256: str
    bake_operations: tuple[ImageOperationSpec, ...] = ()
    message: str = ""


class DisplayAdjustmentDialog(QDialog):
    """Edit presentation settings without mutating authoritative pixels."""

    previewTransformChanged = Signal(object)
    displaySettingsApplied = Signal(object)
    derivedImageRequested = Signal(object)
    adjustmentCancelled = Signal(object)
    resultReady = Signal(object)

    _LUT_ITEMS = (
        ("灰度（默认）", None),
        ("红色", "red"),
        ("绿色", "green"),
        ("蓝色", "blue"),
        ("Fire", "fire"),
        ("Ice", "ice"),
        ("Spectrum", "spectrum"),
    )

    def __init__(
        self,
        source: RasterPlane,
        initial_transform: DisplayTransform | None = None,
        *,
        source_name: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(source, RasterPlane):
            raise TypeError("source 必须是 RasterPlane")
        initial = initial_transform or DisplayTransform()
        if not isinstance(initial, DisplayTransform):
            raise TypeError("initial_transform 必须是 DisplayTransform")
        initial.ranges_for_pixel_type(source.pixel_type)

        self._source = source
        self._initial_transform = initial
        self._source_name = str(source_name or "").strip()
        self._changed = False
        self._updating = True
        self._completed = False
        self._last_valid_transform = initial

        self.setWindowTitle("亮度、对比度与显示范围")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumSize(520, 420)
        self._apply_available_screen_size()

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)

        scroll = QScrollArea(self)
        scroll.setObjectName("displayAdjustmentScroll")
        scroll.setProperty("redirectEditorWheel", True)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll, 1)

        content = QWidget(scroll)
        content.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(2, 2, 8, 2)
        content_layout.setSpacing(12)
        scroll.setWidget(content)

        title = QLabel("非破坏性显示调整", content)
        title.setObjectName("displayAdjustmentTitle")
        title_font = title.font()
        title_font.setPointSizeF(title_font.pointSizeF() + 2.0)
        title_font.setBold(True)
        title.setFont(title_font)
        content_layout.addWidget(title)

        source_text = (
            f"{self._source_name} · " if self._source_name else ""
        ) + (
            f"{source.width} × {source.height} · "
            f"{self._pixel_type_label(source.pixel_type)}"
        )
        self.sourceLabel = QLabel(source_text, content)
        self.sourceLabel.setObjectName("displayAdjustmentSource")
        self.sourceLabel.setWordWrap(True)
        content_layout.addWidget(self.sourceLabel)

        explanation = QLabel(
            "这些参数只改变画布显示，不会改写原始像素、测量值或标定。"
            "实时预览显示在当前画布中。",
            content,
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("displayAdjustmentExplanation")
        content_layout.addWidget(explanation)

        range_group = QGroupBox("显示范围", content)
        range_layout = QVBoxLayout(range_group)
        range_layout.setSpacing(8)
        self.rangeModeCombo = NoWheelComboBox(range_group)
        self.rangeModeCombo.addItem("黑场 / 白场", "range")
        if source.pixel_type.is_grayscale:
            self.rangeModeCombo.addItem("窗宽 / 窗位", "window")
        mode_form = QFormLayout()
        mode_form.addRow("调整方式", self.rangeModeCombo)
        range_layout.addLayout(mode_form)

        self.rangeEditor = QWidget(range_group)
        range_grid = QGridLayout(self.rangeEditor)
        range_grid.setContentsMargins(0, 0, 0, 0)
        range_grid.setHorizontalSpacing(10)
        range_grid.setVerticalSpacing(7)
        range_grid.addWidget(QLabel("通道", self.rangeEditor), 0, 0)
        range_grid.addWidget(QLabel("黑场 / 下限", self.rangeEditor), 0, 1)
        range_grid.addWidget(QLabel("白场 / 上限", self.rangeEditor), 0, 2)
        self.channelLowSpins: list[NoWheelDoubleSpinBox] = []
        self.channelHighSpins: list[NoWheelDoubleSpinBox] = []
        channel_labels = (
            ("灰度",)
            if source.pixel_type.is_grayscale
            else ("红色 R", "绿色 G", "蓝色 B")
        )
        native_low, native_high = self._native_display_range()
        spin_limit = self._spin_limit(native_low, native_high, initial)
        for row, label_text in enumerate(channel_labels, start=1):
            label = QLabel(label_text, self.rangeEditor)
            low_spin = self._new_value_spin(
                native_low,
                minimum=-spin_limit,
                maximum=spin_limit,
            )
            high_spin = self._new_value_spin(
                native_high,
                minimum=-spin_limit,
                maximum=spin_limit,
            )
            range_grid.addWidget(label, row, 0)
            range_grid.addWidget(low_spin, row, 1)
            range_grid.addWidget(high_spin, row, 2)
            self.channelLowSpins.append(low_spin)
            self.channelHighSpins.append(high_spin)
        range_layout.addWidget(self.rangeEditor)

        self.windowEditor = QWidget(range_group)
        window_form = QFormLayout(self.windowEditor)
        window_form.setContentsMargins(0, 0, 0, 0)
        self.windowCenterSpin = self._new_value_spin(
            (native_low + native_high) / 2.0,
            minimum=-spin_limit,
            maximum=spin_limit,
        )
        self.windowWidthSpin = self._new_value_spin(
            native_high - native_low,
            minimum=max(1e-12, abs(native_high - native_low) * 1e-12),
            maximum=spin_limit * 2.0,
        )
        window_form.addRow("窗位（中心）", self.windowCenterSpin)
        window_form.addRow("窗宽", self.windowWidthSpin)
        range_layout.addWidget(self.windowEditor)
        content_layout.addWidget(range_group)

        appearance_group = QGroupBox("显示映射", content)
        appearance_form = QFormLayout(appearance_group)
        self.gammaSpin = NoWheelDoubleSpinBox(appearance_group)
        self.gammaSpin.setDecimals(3)
        self.gammaSpin.setRange(0.01, 20.0)
        self.gammaSpin.setSingleStep(0.05)
        self.gammaSpin.setValue(initial.gamma)
        appearance_form.addRow("Gamma", self.gammaSpin)

        self.lutCombo = NoWheelComboBox(appearance_group)
        for label, value in self._LUT_ITEMS:
            self.lutCombo.addItem(label, value)
        appearance_form.addRow("LUT", self.lutCombo)
        self.invertCheck = QCheckBox("反相显示", appearance_group)
        appearance_form.addRow("", self.invertCheck)
        if not source.pixel_type.is_grayscale:
            self.lutCombo.setEnabled(False)
            self.lutCombo.setToolTip("彩色图片不使用灰度 LUT。")
        content_layout.addWidget(appearance_group)

        self.validationLabel = QLabel(content)
        self.validationLabel.setObjectName("displayAdjustmentValidation")
        self.validationLabel.setWordWrap(True)
        self.validationLabel.setVisible(False)
        content_layout.addWidget(self.validationLabel)

        self.bakeHintLabel = QLabel(content)
        self.bakeHintLabel.setObjectName("displayBakeHint")
        self.bakeHintLabel.setWordWrap(True)
        content_layout.addWidget(self.bakeHintLabel)
        content_layout.addStretch(1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addStretch(1)
        self.applyDisplayButton = QPushButton("应用显示设置", self)
        self.generateDerivedButton = QPushButton("应用并生成派生图片", self)
        self.cancelButton = QPushButton("取消", self)
        self.applyDisplayButton.setDefault(True)
        footer.addWidget(self.applyDisplayButton)
        footer.addWidget(self.generateDerivedButton)
        footer.addWidget(self.cancelButton)
        root.addLayout(footer)

        self._load_initial_controls()
        self._connect_controls()
        self._updating = False
        self._refresh_state(emit_preview=False)

        self.applyDisplayButton.clicked.connect(self._apply_display)
        self.generateDerivedButton.clicked.connect(self._request_derived)
        self.cancelButton.clicked.connect(self.reject)

    @property
    def source(self) -> RasterPlane:
        return self._source

    @property
    def initial_transform(self) -> DisplayTransform:
        return self._initial_transform

    def current_transform(self) -> DisplayTransform:
        if not self._changed:
            return self._initial_transform
        return self._transform_from_controls()

    def bake_plan(self) -> DisplayBakePlan:
        try:
            transform = self.current_transform()
        except (TypeError, ValueError) as exc:
            return DisplayBakePlan(False, message=str(exc))
        return self._build_bake_plan(transform)

    def _apply_available_screen_size(self) -> None:
        screen = (
            self.parentWidget().screen()
            if self.parentWidget() is not None
            else QGuiApplication.primaryScreen()
        )
        available = screen.availableGeometry() if screen is not None else None
        if available is None:
            self.resize(720, 600)
            return
        self.resize(
            max(self.minimumWidth(), min(760, available.width() - 40)),
            max(self.minimumHeight(), min(640, available.height() - 40)),
        )

    @staticmethod
    def _pixel_type_label(pixel_type: RasterPixelType) -> str:
        return {
            RasterPixelType.GRAY8: "8 位灰度",
            RasterPixelType.GRAY16: "16 位灰度",
            RasterPixelType.GRAY32_FLOAT: "32 位浮点灰度",
            RasterPixelType.RGB8: "RGB 8 位",
            RasterPixelType.RGBA8: "RGBA 8 位",
        }[pixel_type]

    def _native_display_range(self) -> tuple[float, float]:
        pixel_type = self._source.pixel_type
        if pixel_type is RasterPixelType.GRAY16:
            return 0.0, 65_535.0
        if pixel_type is not RasterPixelType.GRAY32_FLOAT:
            return 0.0, 255.0
        values = np.frombuffer(self._source.data, dtype="<f4")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        low = float(np.min(finite))
        high = float(np.max(finite))
        if high > low:
            return low, high
        padding = max(0.5, abs(low) * 1e-6)
        return low - padding, high + padding

    @staticmethod
    def _spin_limit(
        native_low: float,
        native_high: float,
        initial: DisplayTransform,
    ) -> float:
        candidates = [abs(native_low), abs(native_high), 1.0]
        candidates.extend(abs(value) for item in initial.effective_channel_ranges for value in item)
        if initial.window_center is not None:
            candidates.append(abs(initial.window_center))
        if initial.window_width is not None:
            candidates.append(abs(initial.window_width))
        return min(1e100, max(1e12, max(candidates) * 10.0))

    @staticmethod
    def _new_value_spin(
        value: float,
        *,
        minimum: float,
        maximum: float,
    ) -> NoWheelDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(float(minimum), float(maximum))
        span = max(abs(maximum - minimum), 1.0)
        spin.setSingleStep(max(1e-6, min(1000.0, span / 10000.0)))
        spin.setValue(float(value))
        spin.setKeyboardTracking(False)
        return spin

    def _load_initial_controls(self) -> None:
        initial = self._initial_transform
        native_low, native_high = self._native_display_range()
        if initial.window_center is not None and initial.window_width is not None:
            index = self.rangeModeCombo.findData("window")
            if index >= 0:
                self.rangeModeCombo.setCurrentIndex(index)
            self.windowCenterSpin.setValue(initial.window_center)
            self.windowWidthSpin.setValue(initial.window_width)
        else:
            ranges = initial.ranges_for_pixel_type(self._source.pixel_type)
            if not ranges:
                range_values = [(native_low, native_high)] * len(self.channelLowSpins)
            elif len(ranges) == 1 and len(self.channelLowSpins) == 3:
                range_values = [ranges[0]] * 3
            else:
                range_values = list(ranges)
            for low_spin, high_spin, (low, high) in zip(
                self.channelLowSpins,
                self.channelHighSpins,
                range_values,
                strict=True,
            ):
                low_spin.setValue(low)
                high_spin.setValue(high)
        lut_index = self.lutCombo.findData(initial.lut_id)
        self.lutCombo.setCurrentIndex(max(0, lut_index))
        self.invertCheck.setChecked(initial.inverted)
        self._update_mode_visibility()

    def _connect_controls(self) -> None:
        self.rangeModeCombo.currentIndexChanged.connect(self._on_control_changed)
        for spin in (*self.channelLowSpins, *self.channelHighSpins):
            spin.valueChanged.connect(self._on_control_changed)
        self.windowCenterSpin.valueChanged.connect(self._on_control_changed)
        self.windowWidthSpin.valueChanged.connect(self._on_control_changed)
        self.gammaSpin.valueChanged.connect(self._on_control_changed)
        self.lutCombo.currentIndexChanged.connect(self._on_control_changed)
        self.invertCheck.toggled.connect(self._on_control_changed)

    def _on_control_changed(self, *_args: object) -> None:
        if self._updating:
            return
        self._changed = True
        self._update_mode_visibility()
        self._refresh_state(emit_preview=True)

    def _update_mode_visibility(self) -> None:
        window_mode = self.rangeModeCombo.currentData() == "window"
        self.rangeEditor.setVisible(not window_mode)
        self.windowEditor.setVisible(window_mode)

    def _transform_from_controls(self) -> DisplayTransform:
        gamma = float(self.gammaSpin.value())
        lut_id = (
            self.lutCombo.currentData()
            if self._source.pixel_type.is_grayscale
            else None
        )
        inverted = bool(self.invertCheck.isChecked())
        if self.rangeModeCombo.currentData() == "window":
            return DisplayTransform(
                gamma=gamma,
                lut_id=lut_id,
                window_center=float(self.windowCenterSpin.value()),
                window_width=float(self.windowWidthSpin.value()),
                inverted=inverted,
            )

        ranges: list[tuple[float, float]] = []
        labels = (
            ("灰度",)
            if self._source.pixel_type.is_grayscale
            else ("红色", "绿色", "蓝色")
        )
        for label, low_spin, high_spin in zip(
            labels,
            self.channelLowSpins,
            self.channelHighSpins,
            strict=True,
        ):
            low = float(low_spin.value())
            high = float(high_spin.value())
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(f"{label}通道显示范围必须是有限数值。")
            if high <= low:
                raise ValueError(f"{label}通道显示上限必须大于下限。")
            ranges.append((low, high))
        return DisplayTransform(
            channel_ranges=tuple(ranges),
            gamma=gamma,
            lut_id=lut_id,
            inverted=inverted,
        )

    def _refresh_state(self, *, emit_preview: bool) -> None:
        try:
            transform = self.current_transform()
            transform.ranges_for_pixel_type(self._source.pixel_type)
        except (TypeError, ValueError) as exc:
            self.validationLabel.setText(f"参数错误：{exc}")
            self.validationLabel.setVisible(True)
            self.applyDisplayButton.setEnabled(False)
            self.generateDerivedButton.setEnabled(False)
            self.bakeHintLabel.setText("请先修正参数。")
            return

        self.validationLabel.clear()
        self.validationLabel.setVisible(False)
        self.applyDisplayButton.setEnabled(True)
        self._last_valid_transform = transform
        plan = self._build_bake_plan(transform)
        self.generateDerivedButton.setEnabled(plan.supported)
        self.generateDerivedButton.setToolTip(
            "" if plan.supported else plan.message
        )
        self.bakeHintLabel.setText(plan.message)
        if emit_preview:
            self.previewTransformChanged.emit(transform)

    def _build_bake_plan(
        self,
        transform: DisplayTransform,
    ) -> DisplayBakePlan:
        if transform.lut_id not in {None, "grayscale"}:
            return DisplayBakePlan(
                False,
                message=(
                    "当前彩色 LUT 只用于显示；现有处理操作无法逐像素精确复现该 LUT，"
                    "因此不能生成派生图片。可先选择“灰度（默认）”。"
                ),
            )

        ranges = transform.ranges_for_pixel_type(self._source.pixel_type)
        if (
            not self._source.pixel_type.is_grayscale
            and len(ranges) == 3
            and not (ranges[0] == ranges[1] == ranges[2])
        ):
            return DisplayBakePlan(
                False,
                message=(
                    "RGB 独立通道显示范围当前只能非破坏性预览；"
                    "现有操作链没有逐通道色阶步骤，不能无提示地近似烘焙。"
                ),
            )

        operations: list[ImageOperationSpec] = []
        pixel_type = self._source.pixel_type
        native_low, native_high = self._native_display_range()
        selected_range = ranges[0] if ranges else (native_low, native_high)
        needs_levels = bool(ranges) or not math.isclose(transform.gamma, 1.0)
        if pixel_type is RasterPixelType.GRAY32_FLOAT and transform.inverted:
            # The automatic float display uses the finite data range, whereas
            # the math operation's native range is [0, 1].  Normalising first
            # preserves the exact declared presentation semantics.
            needs_levels = True
        if needs_levels:
            operations.append(
                ImageOperationSpec(
                    "adjust_levels",
                    {
                        "black_point": selected_range[0],
                        "white_point": selected_range[1],
                        "gamma": transform.gamma,
                    },
                )
            )
        if transform.inverted:
            if needs_levels:
                invert_low, invert_high = (
                    (0.0, 1.0)
                    if pixel_type is RasterPixelType.GRAY32_FLOAT
                    else (
                        0.0,
                        float(pixel_type.sample_maximum or 1),
                    )
                )
            else:
                invert_low, invert_high = native_low, native_high
            operations.append(
                ImageOperationSpec(
                    "invert",
                    {
                        "minimum": invert_low,
                        "maximum": invert_high,
                    },
                )
            )
        if not operations:
            return DisplayBakePlan(
                False,
                message="当前显示设置与默认显示一致，无需生成派生图片。",
            )
        return DisplayBakePlan(
            True,
            tuple(operations),
            (
                "将以显式色阶、Gamma 和反相步骤生成新图片；"
                "源图片及其测量对象保持不变。"
            ),
        )

    def _apply_display(self) -> None:
        try:
            transform = self.current_transform()
        except (TypeError, ValueError):
            self._refresh_state(emit_preview=False)
            return
        result = DisplayAdjustmentResult(
            action=DisplayAdjustmentAction.APPLY_DISPLAY,
            transform=transform,
            source_sha256=self._source.sha256(),
            message="显示设置已应用；原始像素未改变。",
        )
        self._complete(result, self.displaySettingsApplied)

    def _request_derived(self) -> None:
        try:
            transform = self.current_transform()
        except (TypeError, ValueError):
            self._refresh_state(emit_preview=False)
            return
        plan = self._build_bake_plan(transform)
        if not plan.supported:
            self._refresh_state(emit_preview=False)
            return
        result = DisplayAdjustmentResult(
            action=DisplayAdjustmentAction.GENERATE_DERIVED,
            transform=transform,
            source_sha256=self._source.sha256(),
            bake_operations=plan.operations,
            message=plan.message,
        )
        self._complete(result, self.derivedImageRequested)

    def _complete(self, result: DisplayAdjustmentResult, signal: Signal) -> None:
        if self._completed:
            return
        self._completed = True
        signal.emit(result)
        self.resultReady.emit(result)
        self.done(QDialog.DialogCode.Accepted)

    def reject(self) -> None:
        if not self._completed:
            self.previewTransformChanged.emit(self._initial_transform)
            result = DisplayAdjustmentResult(
                action=DisplayAdjustmentAction.CANCEL,
                transform=self._initial_transform,
                source_sha256=self._source.sha256(),
                message="已取消并恢复打开窗口时的显示设置。",
            )
            self._completed = True
            self.adjustmentCancelled.emit(result)
            self.resultReady.emit(result)
        super().reject()


__all__ = [
    "DisplayAdjustmentAction",
    "DisplayAdjustmentDialog",
    "DisplayAdjustmentResult",
    "DisplayBakePlan",
]
