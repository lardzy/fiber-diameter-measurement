from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fdm.services.image_information import ImageInformationSnapshot


class ImageInformationDialog(QDialog):
    """Read-only, selectable scientific image metadata."""

    def __init__(
        self,
        snapshot: ImageInformationSnapshot,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(snapshot, ImageInformationSnapshot):
            raise TypeError("snapshot 必须是 ImageInformationSnapshot")
        self._snapshot = snapshot
        self.setWindowTitle("图像信息与属性")
        self.setModal(False)
        self.setMinimumSize(560, 460)
        self.resize(680, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)
        title = QLabel(snapshot.display_name, self)
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSizeF(title_font.pointSizeF() + 2.0)
        title.setFont(title_font)
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(title)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(2, 2, 8, 2)
        content_layout.setSpacing(10)
        content_layout.addWidget(self._pixel_group(content))
        content_layout.addWidget(self._source_group(content))
        content_layout.addWidget(self._calibration_group(content))
        content_layout.addWidget(self._derivation_group(content))
        content_layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        footer = QHBoxLayout()
        copy_button = QPushButton("复制 JSON", self)
        copy_button.clicked.connect(self._copy_json)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        footer.addWidget(copy_button)
        footer.addStretch(1)
        footer.addWidget(buttons)
        root.addLayout(footer)

    @staticmethod
    def _selectable_label(text: object, parent: QWidget) -> QLabel:
        label = QLabel("—" if text in (None, "") else str(text), parent)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        return label

    def _add_row(
        self,
        form: QFormLayout,
        label: str,
        value: object,
        parent: QWidget,
    ) -> None:
        form.addRow(label, self._selectable_label(value, parent))

    def _pixel_group(self, parent: QWidget) -> QGroupBox:
        snapshot = self._snapshot
        group = QGroupBox("权威像素", parent)
        form = QFormLayout(group)
        self._add_row(form, "尺寸", f"{snapshot.width} × {snapshot.height} px", group)
        self._add_row(form, "像素类型", snapshot.pixel_type, group)
        self._add_row(form, "通道数", snapshot.channel_count, group)
        self._add_row(form, "Alpha", "有" if snapshot.has_alpha else "无", group)
        self._add_row(
            form,
            "未压缩大小",
            f"{snapshot.byte_count / float(1 << 20):.2f} MiB",
            group,
        )
        self._add_row(form, "像素 SHA256", snapshot.pixel_sha256, group)
        return group

    def _source_group(self, parent: QWidget) -> QGroupBox:
        snapshot = self._snapshot
        group = QGroupBox("来源与文件元数据", parent)
        form = QFormLayout(group)
        self._add_row(form, "项目来源类型", snapshot.source_type, group)
        self._add_row(form, "路径", snapshot.source_path, group)
        self._add_row(form, "原格式", snapshot.source_format, group)
        self._add_row(form, "原模式", snapshot.source_mode, group)
        dpi = (
            "—"
            if snapshot.dpi_x is None or snapshot.dpi_y is None
            else f"{snapshot.dpi_x:g} × {snapshot.dpi_y:g} DPI"
        )
        self._add_row(form, "分辨率", dpi, group)
        self._add_row(
            form,
            "ICC",
            (
                "无"
                if snapshot.icc_profile_bytes <= 0
                else (
                    f"{snapshot.icc_profile_bytes} bytes · "
                    f"{snapshot.icc_profile_sha256}"
                )
            ),
            group,
        )
        return group

    def _calibration_group(self, parent: QWidget) -> QGroupBox:
        snapshot = self._snapshot
        group = QGroupBox("标定", parent)
        form = QFormLayout(group)
        self._add_row(form, "模式", snapshot.calibration_mode or "未标定", group)
        scale = (
            "—"
            if snapshot.pixels_per_unit is None
            else (
                f"{snapshot.pixels_per_unit:g} px/"
                f"{snapshot.calibration_unit or 'unit'}"
            )
        )
        self._add_row(form, "换算关系", scale, group)
        return group

    def _derivation_group(self, parent: QWidget) -> QGroupBox:
        snapshot = self._snapshot
        group = QGroupBox("派生与可复现性", parent)
        form = QFormLayout(group)
        self._add_row(
            form,
            "源文档 ID",
            snapshot.derivation_source_document_id or "原始/非派生图片",
            group,
        )
        self._add_row(form, "处理步骤数", snapshot.derivation_step_count, group)
        self._add_row(
            form,
            "派生结果 SHA256",
            snapshot.derivation_result_sha256,
            group,
        )
        return group

    def _copy_json(self) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._snapshot.to_json())


__all__ = ["ImageInformationDialog"]
