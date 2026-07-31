from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.raster_derivation import (
    FillValue,
    FrozenRasterRoi,
    RasterBounds,
    RasterDerivationError,
)
from fdm.ui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class RasterCopyScope(str, Enum):
    FULL_IMAGE = "full_image"
    ROI_BOUNDS = "roi_bounds"
    ROI_MASK = "roi_mask"


class RasterMaskOutsideMode(str, Enum):
    TRANSPARENT = "transparent"
    FILL_VALUE = "fill_value"


@dataclass(frozen=True, slots=True)
class RasterCopyDerivationRequest:
    source: RasterPlane
    source_sha256: str
    scope: RasterCopyScope
    bounds: RasterBounds | None = None
    roi: FrozenRasterRoi | None = None
    transparent_outside: bool = False
    fill_value: FillValue | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, RasterPlane) or self.source.is_empty:
            raise RasterDerivationError("复制来源必须是非空 RasterPlane")
        source_sha256 = _normalize_sha256(self.source_sha256, "复制来源")
        if source_sha256 != self.source.sha256():
            raise RasterDerivationError("复制来源 SHA256 与冻结像素不一致")
        try:
            scope = (
                self.scope
                if isinstance(self.scope, RasterCopyScope)
                else RasterCopyScope(str(self.scope))
            )
        except ValueError as exc:
            raise RasterDerivationError("不支持的图片复制范围") from exc
        if scope is RasterCopyScope.FULL_IMAGE:
            if (
                self.bounds is not None
                or self.roi is not None
                or self.transparent_outside
                or self.fill_value is not None
            ):
                raise RasterDerivationError("整图复制不能携带 ROI 或遮罩参数")
        elif scope is RasterCopyScope.ROI_BOUNDS:
            if not isinstance(self.bounds, RasterBounds):
                raise RasterDerivationError("ROI 包围框复制缺少有效 bounds")
            if self.roi is not None:
                raise RasterDerivationError("ROI 包围框复制不应携带掩膜")
            if self.transparent_outside or self.fill_value is not None:
                raise RasterDerivationError(
                    "ROI 包围框复制不能设置遮罩外像素策略"
                )
            _validate_bounds(self.bounds, self.source)
        else:
            if not isinstance(self.roi, FrozenRasterRoi):
                raise RasterDerivationError("ROI 遮罩复制缺少冻结掩膜")
            if self.bounds != self.roi.bounds:
                raise RasterDerivationError(
                    "ROI 遮罩请求的 bounds 必须与冻结掩膜一致"
                )
            _validate_bounds(self.roi.bounds, self.source)
            if self.transparent_outside == (self.fill_value is not None):
                raise RasterDerivationError(
                    "ROI 遮罩必须且只能选择透明或显式填充值"
                )
            if (
                self.transparent_outside
                and self.source.pixel_type
                in {
                    RasterPixelType.GRAY16,
                    RasterPixelType.GRAY32_FLOAT,
                }
            ):
                raise RasterDerivationError(
                    "科学灰度图片不能隐式转换为透明 8 位图片"
                )
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "scope", scope)


@dataclass(frozen=True, slots=True)
class Gray8RasterDocumentDescriptor:
    document_id: str
    display_name: str
    width: int
    height: int
    pixel_sha256: str
    calibration_signature: str
    pixel_type: RasterPixelType = RasterPixelType.GRAY8

    def __post_init__(self) -> None:
        document_id = str(self.document_id or "").strip()
        display_name = str(self.display_name or "").strip()
        if not document_id:
            raise RasterDerivationError("通道来源文档 ID 不能为空")
        if not display_name:
            display_name = document_id
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
        ):
            raise RasterDerivationError("通道来源图片尺寸必须是正整数")
        pixel_type = RasterPixelType.parse(self.pixel_type)
        if pixel_type is not RasterPixelType.GRAY8:
            raise RasterDerivationError("RGB 通道合并只接受 GRAY8 来源")
        calibration_signature = str(self.calibration_signature or "").strip()
        if not calibration_signature:
            raise RasterDerivationError(
                "通道来源必须提供标定签名；未标定图片请使用统一的 uncalibrated"
            )
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(
            self,
            "pixel_sha256",
            _normalize_sha256(self.pixel_sha256, display_name),
        )
        object.__setattr__(
            self,
            "calibration_signature",
            calibration_signature,
        )
        object.__setattr__(self, "pixel_type", pixel_type)

    @property
    def compatibility_key(self) -> tuple[int, int, str]:
        return (self.width, self.height, self.calibration_signature)


@dataclass(frozen=True, slots=True)
class RasterChannelMergeRequest:
    red: Gray8RasterDocumentDescriptor
    green: Gray8RasterDocumentDescriptor
    blue: Gray8RasterDocumentDescriptor

    def __post_init__(self) -> None:
        sources = (self.red, self.green, self.blue)
        if not all(
            isinstance(source, Gray8RasterDocumentDescriptor)
            for source in sources
        ):
            raise TypeError("RGB 通道来源必须是 Gray8RasterDocumentDescriptor")
        document_ids = tuple(source.document_id for source in sources)
        if len(set(document_ids)) != 3:
            raise RasterDerivationError("红、绿、蓝通道必须选择三个不同来源")
        compatibility_key = sources[0].compatibility_key
        if any(source.compatibility_key != compatibility_key for source in sources[1:]):
            raise RasterDerivationError(
                "三个通道的尺寸和标定必须完全兼容"
            )

    @property
    def document_ids(self) -> tuple[str, str, str]:
        return (
            self.red.document_id,
            self.green.document_id,
            self.blue.document_id,
        )


class RasterCopyDialog(QDialog):
    copyRequested = Signal(object)

    def __init__(
        self,
        source: RasterPlane,
        *,
        source_name: str = "",
        roi: FrozenRasterRoi | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(source, RasterPlane) or source.is_empty:
            raise RasterDerivationError("复制来源必须是非空 RasterPlane")
        if roi is not None:
            if not isinstance(roi, FrozenRasterRoi):
                raise TypeError("roi 必须是 FrozenRasterRoi")
            _validate_bounds(roi.bounds, source)
        self._source = source
        self._source_sha256 = source.sha256()
        self._roi = roi

        self.setWindowTitle("复制当前图片或 ROI")
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)

        source_label = QLabel(
            f"<b>来源：</b>{str(source_name or '').strip() or '当前图片'}　"
            f"{source.width} × {source.height}　{source.pixel_type.value}",
            self,
        )
        source_label.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(source_label)

        form = QFormLayout()
        self.scopeCombo = NoWheelComboBox(self)
        self.scopeCombo.setObjectName("rasterCopyScopeCombo")
        self.scopeCombo.addItem(
            "整张图片",
            RasterCopyScope.FULL_IMAGE.value,
        )
        if roi is not None:
            self.scopeCombo.addItem(
                "当前 ROI 包围框",
                RasterCopyScope.ROI_BOUNDS.value,
            )
            self.scopeCombo.addItem(
                "当前 ROI 遮罩",
                RasterCopyScope.ROI_MASK.value,
            )
            self.scopeCombo.setCurrentIndex(
                self.scopeCombo.findData(RasterCopyScope.ROI_BOUNDS.value)
            )
        form.addRow("复制范围", self.scopeCombo)

        self.outsideModeCombo = NoWheelComboBox(self)
        self.outsideModeCombo.setObjectName("rasterCopyOutsideModeCombo")
        if source.pixel_type in {
            RasterPixelType.GRAY8,
            RasterPixelType.RGB8,
            RasterPixelType.RGBA8,
        }:
            self.outsideModeCombo.addItem(
                "透明（输出 RGBA8）",
                RasterMaskOutsideMode.TRANSPARENT.value,
            )
        self.outsideModeCombo.addItem(
            "使用显式填充值（保持像素类型）",
            RasterMaskOutsideMode.FILL_VALUE.value,
        )
        if source.pixel_type in {
            RasterPixelType.GRAY16,
            RasterPixelType.GRAY32_FLOAT,
        }:
            self.outsideModeCombo.setCurrentIndex(
                self.outsideModeCombo.findData(
                    RasterMaskOutsideMode.FILL_VALUE.value
                )
            )
        form.addRow("遮罩外区域", self.outsideModeCombo)
        root.addLayout(form)

        self.fillGroup = QGroupBox("遮罩外填充值", self)
        fill_form = QFormLayout(self.fillGroup)
        self.fillSpins: list[NoWheelDoubleSpinBox] = []
        channel_names = _fill_channel_names(source.pixel_type)
        minimum, maximum, decimals = _fill_editor_range(source.pixel_type)
        for index, channel_name in enumerate(channel_names):
            spin = NoWheelDoubleSpinBox(self.fillGroup)
            spin.setObjectName(f"rasterCopyFillSpin{index}")
            spin.setRange(minimum, maximum)
            spin.setDecimals(decimals)
            spin.setKeyboardTracking(False)
            spin.setValue(0.0)
            fill_form.addRow(channel_name, spin)
            self.fillSpins.append(spin)
        root.addWidget(self.fillGroup)

        self.hintLabel = QLabel(self)
        self.hintLabel.setWordWrap(True)
        root.addWidget(self.hintLabel)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.createButton = self.buttons.button(
            QDialogButtonBox.StandardButton.Ok
        )
        self.createButton.setText("生成派生图片")
        self.buttons.button(
            QDialogButtonBox.StandardButton.Cancel
        ).setText("取消")
        self.buttons.accepted.connect(self._emit_request)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self.scopeCombo.currentIndexChanged.connect(self._refresh_controls)
        self.outsideModeCombo.currentIndexChanged.connect(
            self._refresh_controls
        )
        self._refresh_controls()

    def request(self) -> RasterCopyDerivationRequest:
        scope = RasterCopyScope(str(self.scopeCombo.currentData()))
        if scope is RasterCopyScope.FULL_IMAGE:
            return RasterCopyDerivationRequest(
                source=self._source,
                source_sha256=self._source_sha256,
                scope=scope,
            )
        if self._roi is None:
            raise RasterDerivationError("当前没有可复制的 ROI")
        if scope is RasterCopyScope.ROI_BOUNDS:
            return RasterCopyDerivationRequest(
                source=self._source,
                source_sha256=self._source_sha256,
                scope=scope,
                bounds=self._roi.bounds,
            )
        outside_mode = RasterMaskOutsideMode(
            str(self.outsideModeCombo.currentData())
        )
        transparent = outside_mode is RasterMaskOutsideMode.TRANSPARENT
        return RasterCopyDerivationRequest(
            source=self._source,
            source_sha256=self._source_sha256,
            scope=RasterCopyScope.ROI_MASK,
            bounds=self._roi.bounds,
            roi=self._roi,
            transparent_outside=transparent,
            fill_value=None if transparent else self._fill_value(),
        )

    def _fill_value(self) -> FillValue:
        values: list[int | float] = []
        integer_type = self._source.pixel_type is not RasterPixelType.GRAY32_FLOAT
        for spin in self.fillSpins:
            value = spin.value()
            values.append(int(round(value)) if integer_type else float(value))
        return values[0] if len(values) == 1 else tuple(values)

    def _refresh_controls(self, *_args: object) -> None:
        scope = RasterCopyScope(str(self.scopeCombo.currentData()))
        is_mask = scope is RasterCopyScope.ROI_MASK
        self.outsideModeCombo.setVisible(is_mask)
        label = self._form_label_for(self.outsideModeCombo)
        if label is not None:
            label.setVisible(is_mask)
        fill_visible = (
            is_mask
            and str(self.outsideModeCombo.currentData())
            == RasterMaskOutsideMode.FILL_VALUE.value
        )
        self.fillGroup.setVisible(fill_visible)
        if scope is RasterCopyScope.FULL_IMAGE:
            text = "复制完整权威像素；源图片不会被修改。"
        elif scope is RasterCopyScope.ROI_BOUNDS:
            text = (
                "按冻结 ROI 的包围框裁剪；包围框内、ROI 外的像素仍会保留。"
            )
        elif fill_visible:
            text = (
                "输出 ROI 包围框尺寸，遮罩外写入显式填充值，"
                "保持原像素类型。"
            )
        else:
            text = (
                "输出 ROI 包围框尺寸，遮罩外透明，并生成 RGBA8 派生图片。"
            )
        self.hintLabel.setText(text)

    def _form_label_for(self, field: QWidget) -> QWidget | None:
        layout = self.layout()
        for index in range(layout.count()):
            item = layout.itemAt(index)
            child_layout = item.layout()
            if not isinstance(child_layout, QFormLayout):
                continue
            label = child_layout.labelForField(field)
            if label is not None:
                return label
        return None

    def _emit_request(self) -> None:
        try:
            request = self.request()
        except (RasterDerivationError, ValueError) as exc:
            self.hintLabel.setText(f"无法生成请求：{exc}")
            return
        self.copyRequested.emit(request)
        self.accept()


class RasterChannelMergeDialog(QDialog):
    mergeRequested = Signal(object)

    def __init__(
        self,
        documents: tuple[Gray8RasterDocumentDescriptor, ...],
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._documents = tuple(documents)
        by_id = {document.document_id: document for document in self._documents}
        if len(by_id) != len(self._documents):
            raise RasterDerivationError("通道来源文档 ID 不能重复")
        self._by_id = by_id

        self.setWindowTitle("合并 RGB 通道")
        self.setMinimumWidth(560)
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)

        introduction = QLabel(
            "分别选择红、绿、蓝三张 8 位灰度图片。三个来源必须不同，"
            "且尺寸与标定完全一致；源图片不会被修改。",
            self,
        )
        introduction.setWordWrap(True)
        root.addWidget(introduction)

        form = QFormLayout()
        self.redCombo = self._make_source_combo("rasterMergeRedCombo")
        self.greenCombo = self._make_source_combo("rasterMergeGreenCombo")
        self.blueCombo = self._make_source_combo("rasterMergeBlueCombo")
        form.addRow("红色通道", self.redCombo)
        form.addRow("绿色通道", self.greenCombo)
        form.addRow("蓝色通道", self.blueCombo)
        root.addLayout(form)

        self.validationLabel = QLabel(self)
        self.validationLabel.setWordWrap(True)
        root.addWidget(self.validationLabel)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.createButton = self.buttons.button(
            QDialogButtonBox.StandardButton.Ok
        )
        self.createButton.setText("合并为 RGB 图片")
        self.buttons.button(
            QDialogButtonBox.StandardButton.Cancel
        ).setText("取消")
        self.buttons.accepted.connect(self._emit_request)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        for combo in (self.redCombo, self.greenCombo, self.blueCombo):
            combo.currentIndexChanged.connect(self._refresh_validation)
        self._select_first_compatible_triplet()
        self._refresh_validation()

    def request(self) -> RasterChannelMergeRequest:
        sources: list[Gray8RasterDocumentDescriptor] = []
        for combo in (self.redCombo, self.greenCombo, self.blueCombo):
            document_id = str(combo.currentData() or "")
            try:
                sources.append(self._by_id[document_id])
            except KeyError as exc:
                raise RasterDerivationError("请选择有效的通道来源") from exc
        return RasterChannelMergeRequest(
            red=sources[0],
            green=sources[1],
            blue=sources[2],
        )

    def _make_source_combo(self, object_name: str) -> NoWheelComboBox:
        combo = NoWheelComboBox(self)
        combo.setObjectName(object_name)
        combo.setSizeAdjustPolicy(
            NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(24)
        for document in self._documents:
            combo.addItem(
                f"{document.display_name} · {document.width}×{document.height}",
                document.document_id,
            )
            item_index = combo.count() - 1
            combo.setItemData(
                item_index,
                (
                    f"像素 SHA256：{document.pixel_sha256}\n"
                    f"标定签名：{document.calibration_signature}"
                ),
                Qt.ItemDataRole.ToolTipRole,
            )
        return combo

    def _select_first_compatible_triplet(self) -> None:
        groups: dict[
            tuple[int, int, str],
            list[Gray8RasterDocumentDescriptor],
        ] = {}
        for document in self._documents:
            groups.setdefault(document.compatibility_key, []).append(document)
        compatible = next(
            (group[:3] for group in groups.values() if len(group) >= 3),
            None,
        )
        if compatible is None:
            if len(self._documents) >= 3:
                compatible = list(self._documents[:3])
            else:
                return
        for combo, document in zip(
            (self.redCombo, self.greenCombo, self.blueCombo),
            compatible,
            strict=True,
        ):
            combo.setCurrentIndex(combo.findData(document.document_id))

    def _refresh_validation(self, *_args: object) -> None:
        if len(self._documents) < 3:
            self.validationLabel.setText(
                "无法合并：至少需要三张可用的 GRAY8 图片。"
            )
            self.createButton.setEnabled(False)
            return
        try:
            request = self.request()
        except (RasterDerivationError, ValueError) as exc:
            self.validationLabel.setText(f"无法合并：{exc}")
            self.createButton.setEnabled(False)
            return
        self.validationLabel.setText(
            f"输出：{request.red.width}×{request.red.height} RGB8；"
            "标定将由调用方按共同签名继承。"
        )
        self.createButton.setEnabled(True)

    def _emit_request(self) -> None:
        try:
            request = self.request()
        except (RasterDerivationError, ValueError) as exc:
            self.validationLabel.setText(f"无法合并：{exc}")
            return
        self.mergeRequested.emit(request)
        self.accept()


def _fill_channel_names(
    pixel_type: RasterPixelType,
) -> tuple[str, ...]:
    if pixel_type is RasterPixelType.RGB8:
        return ("红色 R", "绿色 G", "蓝色 B")
    if pixel_type is RasterPixelType.RGBA8:
        return ("红色 R", "绿色 G", "蓝色 B", "透明度 Alpha")
    return ("填充值",)


def _fill_editor_range(
    pixel_type: RasterPixelType,
) -> tuple[float, float, int]:
    if pixel_type is RasterPixelType.GRAY32_FLOAT:
        return (-3.4028234e38, 3.4028234e38, 6)
    if pixel_type is RasterPixelType.GRAY16:
        return (0.0, 65_535.0, 0)
    return (0.0, 255.0, 0)


def _normalize_sha256(value: object, label: str) -> str:
    token = str(value or "").strip()
    if not _SHA256_PATTERN.fullmatch(token):
        raise RasterDerivationError(f"{label} SHA256 必须是 64 位十六进制")
    return token.lower()


def _validate_bounds(bounds: RasterBounds, plane: RasterPlane) -> None:
    if bounds.right > plane.width or bounds.bottom > plane.height:
        raise RasterDerivationError("ROI 边界超出复制来源图片")


__all__ = [
    "Gray8RasterDocumentDescriptor",
    "RasterChannelMergeDialog",
    "RasterChannelMergeRequest",
    "RasterCopyDerivationRequest",
    "RasterCopyDialog",
    "RasterCopyScope",
    "RasterMaskOutsideMode",
]
