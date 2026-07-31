from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fdm.image_processing_models import ImageOperationSpec
from fdm.raster import RasterPixelType, RasterPlane
from fdm.ui.display_adjustment_dialog import IntensityHistogramWidget
from fdm.ui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox


@dataclass(frozen=True, slots=True)
class ThresholdDerivationRequest:
    operation: ImageOperationSpec
    source_sha256: str


class ThresholdImagePreview(QWidget):
    """Aspect-preserving lightweight preview for B&W and Over/Under modes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QImage()
        self.setMinimumHeight(150)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(500, 190)

    def set_image(self, image: QImage) -> None:
        self._image = image.copy()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        super().paintEvent(event)
        painter = QPainter(self)
        frame = self.rect().adjusted(1, 1, -2, -2)
        painter.fillRect(frame, self.palette().base())
        painter.setPen(self.palette().mid().color())
        painter.drawRect(frame)
        if self._image.isNull():
            painter.setPen(self.palette().placeholderText().color())
            painter.drawText(frame, Qt.AlignmentFlag.AlignCenter, "无可预览像素")
            return
        scaled = self._image.size()
        scaled.scale(frame.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = frame
        target.setSize(scaled)
        target.moveCenter(frame.center())
        painter.drawImage(target, self._image)


class ThresholdAdjustmentDialog(QDialog):
    """Non-destructive threshold editor.

    The dialog only computes statistics and a preview description.  Pixel
    mutation happens later in the processing workbench after the user requests
    a managed binary derivative.
    """

    binaryDerivedRequested = Signal(object)

    def __init__(
        self,
        source: RasterPlane,
        *,
        source_name: str = "",
        roi_mask: NDArray[np.bool_] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._source_name = str(source_name or "").strip()
        self._roi_mask = self._normalize_mask(roi_mask)
        self._completed = False

        self.setWindowTitle("阈值")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumSize(540, 430)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)
        source_label = QLabel(
            f"<b>源图片：</b>{self._source_name or '未命名图片'}　"
            f"{source.width} × {source.height}　{source.pixel_type.value}",
            self,
        )
        source_label.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(source_label)

        form = QFormLayout()
        self.channelCombo = NoWheelComboBox(self)
        self.channelCombo.addItem("加权亮度", "luminance")
        if not source.pixel_type.is_grayscale:
            self.channelCombo.addItem("红色通道", "red")
            self.channelCombo.addItem("绿色通道", "green")
            self.channelCombo.addItem("蓝色通道", "blue")
        self.previewModeCombo = NoWheelComboBox(self)
        self.previewModeCombo.addItem("B&W（二值预览）", "black_white")
        self.previewModeCombo.addItem("Over/Under（区间内外）", "over_under")
        form.addRow("分析通道", self.channelCombo)
        form.addRow("预览方式", self.previewModeCombo)
        root.addLayout(form)

        self.histogramWidget = IntensityHistogramWidget(self)
        root.addWidget(self.histogramWidget)
        self.imagePreview = ThresholdImagePreview(self)
        root.addWidget(self.imagePreview, 1)
        self.previewLegend = QLabel(self)
        self.previewLegend.setWordWrap(True)
        root.addWidget(self.previewLegend)
        self.scopeLabel = QLabel(self)
        self.scopeLabel.setWordWrap(True)
        root.addWidget(self.scopeLabel)

        threshold_form = QFormLayout()
        self.lowerSpin = NoWheelDoubleSpinBox(self)
        self.upperSpin = NoWheelDoubleSpinBox(self)
        for spin in (self.lowerSpin, self.upperSpin):
            spin.setDecimals(6)
            spin.setRange(-1e100, 1e100)
            spin.setKeyboardTracking(False)
        threshold_form.addRow("阈值下限", self.lowerSpin)
        threshold_form.addRow("阈值上限", self.upperSpin)
        self.invertCheck = QCheckBox("反相（二值前景取区间外）", self)
        threshold_form.addRow("", self.invertCheck)
        root.addLayout(threshold_form)

        actions = QHBoxLayout()
        self.autoButton = QPushButton("Auto（Otsu）", self)
        self.resetButton = QPushButton("Reset", self)
        self.setButton = QPushButton("Set", self)
        actions.addWidget(self.autoButton)
        actions.addWidget(self.resetButton)
        actions.addWidget(self.setButton)
        actions.addStretch(1)
        root.addLayout(actions)

        note = QLabel(
            "此窗口只调整阈值预览，不修改源像素。只有点击“生成二值派生图片”"
            "后，才会按原始分辨率执行并生成受管新图片。",
            self,
        )
        note.setWordWrap(True)
        root.addWidget(note)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self.generateButton = QPushButton("生成二值派生图片", self)
        self.cancelButton = QPushButton("取消", self)
        footer.addWidget(self.generateButton)
        footer.addWidget(self.cancelButton)
        root.addLayout(footer)

        self.channelCombo.currentIndexChanged.connect(self._refresh_statistics)
        self.previewModeCombo.currentIndexChanged.connect(
            self._refresh_preview_description
        )
        self.lowerSpin.valueChanged.connect(self._threshold_changed)
        self.upperSpin.valueChanged.connect(self._threshold_changed)
        self.invertCheck.toggled.connect(self._refresh_preview_description)
        self.autoButton.clicked.connect(self._auto_threshold)
        self.resetButton.clicked.connect(self._reset_threshold)
        self.setButton.clicked.connect(self.lowerSpin.setFocus)
        self.generateButton.clicked.connect(self._request_binary_derivative)
        self.cancelButton.clicked.connect(self.reject)
        self._refresh_statistics()

    def _normalize_mask(
        self,
        mask: NDArray[np.bool_] | None,
    ) -> NDArray[np.bool_] | None:
        if mask is None:
            return None
        normalized = np.ascontiguousarray(mask, dtype=np.bool_)
        expected = (self._source.height, self._source.width)
        if normalized.shape != expected:
            raise ValueError(
                f"ROI 掩膜尺寸 {normalized.shape!r} 与图片尺寸 {expected!r} 不一致。"
            )
        normalized.setflags(write=False)
        return normalized

    def _source_array(self) -> NDArray[np.generic]:
        pixel_type = self._source.pixel_type
        if pixel_type is RasterPixelType.GRAY8:
            dtype, channels = np.dtype(np.uint8), 1
        elif pixel_type is RasterPixelType.GRAY16:
            dtype, channels = np.dtype("<u2"), 1
        elif pixel_type is RasterPixelType.GRAY32_FLOAT:
            dtype, channels = np.dtype("<f4"), 1
        else:
            dtype, channels = np.dtype(np.uint8), pixel_type.channel_count
        shape = (
            (self._source.height, self._source.width)
            if channels == 1
            else (self._source.height, self._source.width, channels)
        )
        return np.frombuffer(self._source.data, dtype=dtype).reshape(shape)

    def _channel_values(self) -> NDArray[np.float64]:
        scalar = self._channel_plane()
        values = (
            scalar[self._roi_mask]
            if self._roi_mask is not None
            else scalar.reshape(-1)
        )
        finite = np.asarray(values, dtype=np.float64)
        return finite[np.isfinite(finite)]

    def _channel_plane(self) -> NDArray[np.generic]:
        array = self._source_array()
        channel = str(self.channelCombo.currentData() or "luminance")
        if array.ndim == 2:
            scalar = array
        elif channel == "red":
            scalar = array[..., 0]
        elif channel == "green":
            scalar = array[..., 1]
        elif channel == "blue":
            scalar = array[..., 2]
        else:
            scalar = (
                array[..., 0].astype(np.float64) * 0.299
                + array[..., 1].astype(np.float64) * 0.587
                + array[..., 2].astype(np.float64) * 0.114
            )
        return np.asarray(scalar)

    def _native_range(self) -> tuple[float, float]:
        values = self._channel_values()
        if values.size == 0:
            return 0.0, 1.0
        if self._source.pixel_type is RasterPixelType.GRAY16:
            return 0.0, 65_535.0
        if self._source.pixel_type is not RasterPixelType.GRAY32_FLOAT:
            return 0.0, 255.0
        low = float(np.min(values))
        high = float(np.max(values))
        if high <= low:
            padding = max(0.5, abs(low) * 1e-6)
            return low - padding, high + padding
        return low, high

    def _refresh_statistics(self, *_args: object) -> None:
        values = self._channel_values()
        low, high = self._native_range()
        histogram, _edges = np.histogram(
            values,
            bins=256,
            range=(low, high),
        )
        self.histogramWidget.set_histograms(
            (np.asarray(histogram, dtype=np.float64),),
            value_range=(low, high),
            colors=(),
        )
        self.lowerSpin.setValue(low)
        self.upperSpin.setValue(high)
        scope = "当前 ROI" if self._roi_mask is not None else "整张图片"
        self.scopeLabel.setText(
            f"统计范围：{scope} · 有效 N={values.size:,}；"
            "Auto 使用当前范围像素计算 Otsu 阈值。"
        )
        self._threshold_changed()

    def _threshold_changed(self, *_args: object) -> None:
        low = float(self.lowerSpin.value())
        high = float(self.upperSpin.value())
        valid = math.isfinite(low) and math.isfinite(high) and high >= low
        self.generateButton.setEnabled(valid)
        self.histogramWidget.set_markers(((low, high),))
        self._refresh_preview_description()

    def _refresh_preview_description(self, *_args: object) -> None:
        mode = str(self.previewModeCombo.currentData())
        invert = self.invertCheck.isChecked()
        self.previewModeCombo.setToolTip(
            (
                "区间内显示为白色、区间外显示为黑色。"
                if mode == "black_white"
                else "区间内与低于/高于阈值的像素使用不同语义色。"
            )
            + (" 当前已反相。" if invert else "")
        )
        self._refresh_image_preview(mode=mode, invert=invert)

    def _refresh_image_preview(self, *, mode: str, invert: bool) -> None:
        scalar = self._channel_plane()
        row_step = max(1, int(math.ceil(scalar.shape[0] / 256)))
        column_step = max(1, int(math.ceil(scalar.shape[1] / 512)))
        sample = np.asarray(
            scalar[::row_step, ::column_step],
            dtype=np.float64,
        )
        finite = np.isfinite(sample)
        lower = float(self.lowerSpin.value())
        upper = float(self.upperSpin.value())
        inside = finite & (sample >= lower) & (sample <= upper)
        if invert:
            inside = finite & ~inside
        rgb = np.zeros((*sample.shape, 3), dtype=np.uint8)
        if mode == "black_white":
            rgb[inside] = (245, 245, 245)
            rgb[finite & ~inside] = (18, 18, 18)
            legend = "B&W：白色为当前二值前景，黑色为背景"
        else:
            below = finite & (sample < lower)
            above = finite & (sample > upper)
            rgb[below] = (44, 123, 182)
            rgb[above] = (214, 67, 62)
            rgb[finite & ~(below | above)] = (42, 157, 143)
            if invert:
                rgb[inside] = (242, 181, 55)
            legend = (
                "Over/Under：蓝色低于下限，绿色位于区间内，"
                "红色高于上限"
            )
        rgb[~finite] = (204, 76, 180)
        if self._roi_mask is not None:
            roi_sample = self._roi_mask[::row_step, ::column_step]
            rgb[~roi_sample] = (72, 76, 82)
            legend += "；灰色区域不属于当前 ROI"
        if np.any(~finite):
            legend += "；洋红色为 NaN/Inf"
        contiguous = np.ascontiguousarray(rgb)
        image = QImage(
            contiguous.data,
            contiguous.shape[1],
            contiguous.shape[0],
            contiguous.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self.imagePreview.set_image(image)
        self.previewLegend.setText(legend)

    def _auto_threshold(self) -> None:
        values = self._channel_values()
        if values.size == 0:
            return
        low, high = self._native_range()
        histogram, edges = np.histogram(values, bins=256, range=(low, high))
        counts = histogram.astype(np.float64)
        total = float(np.sum(counts))
        if total <= 0.0:
            return
        centers = (edges[:-1] + edges[1:]) / 2.0
        weight_background = np.cumsum(counts)
        weight_foreground = total - weight_background
        mean_background = np.cumsum(counts * centers) / np.maximum(
            weight_background,
            1.0,
        )
        reverse_sum = np.cumsum((counts * centers)[::-1])[::-1]
        mean_foreground = reverse_sum / np.maximum(weight_foreground, 1.0)
        between = (
            weight_background
            * weight_foreground
            * np.square(mean_background - mean_foreground)
        )
        threshold = float(centers[int(np.argmax(between[:-1]))])
        self.lowerSpin.setValue(threshold)
        self.upperSpin.setValue(high)

    def _reset_threshold(self) -> None:
        low, high = self._native_range()
        self.lowerSpin.setValue(low)
        self.upperSpin.setValue(high)
        self.invertCheck.setChecked(False)

    def operation_spec(self) -> ImageOperationSpec:
        low = float(self.lowerSpin.value())
        high = float(self.upperSpin.value())
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise ValueError("阈值上限必须大于或等于下限。")
        return ImageOperationSpec(
            "threshold",
            {
                "lower": low,
                "upper": high,
                "invert": bool(self.invertCheck.isChecked()),
                "channel": str(self.channelCombo.currentData() or "luminance"),
            },
        )

    def _request_binary_derivative(self) -> None:
        try:
            operation = self.operation_spec()
        except ValueError:
            return
        if self._completed:
            return
        self._completed = True
        self.binaryDerivedRequested.emit(
            ThresholdDerivationRequest(
                operation=operation,
                source_sha256=self._source.sha256(),
            )
        )
        self.accept()


__all__ = [
    "ThresholdAdjustmentDialog",
    "ThresholdDerivationRequest",
]
