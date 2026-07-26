from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from fdm.services.raster_export import (
    RasterEncodingOptions,
    RasterExportFormat,
    TiffCompression,
)
from fdm.ui.widgets import NoWheelComboBox, NoWheelSpinBox


class CurrentImageExportMode:
    RAW_PIXELS = "raw_pixels"
    CURRENT_DISPLAY = "current_display"


class CurrentImageExportDialog(QDialog):
    """Collect a path, pixel intent and deterministic encoder options."""

    def __init__(
        self,
        default_path: str | Path,
        *,
        initial_options: RasterEncodingOptions | None = None,
        digital_slide_viewport: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出当前图像")
        self.setMinimumWidth(560)
        options = initial_options or RasterEncodingOptions()
        self._background = tuple(options.jpeg_background)

        path_row = QWidget(self)
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(8)
        self.path_edit = QLineEdit(str(default_path), path_row)
        self.path_edit.setPlaceholderText("请选择导出文件")
        browse = QPushButton("浏览…", path_row)
        browse.clicked.connect(self._browse)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(browse)

        form = QFormLayout()
        form.addRow("输出文件", path_row)

        self.raw_radio = QRadioButton(
            "当前焦层原始视窗像素"
            if digital_slide_viewport
            else "原始像素（推荐用于后续测量）"
        )
        self.display_radio = QRadioButton("当前显示效果")
        self.raw_radio.setChecked(True)
        mode_row = QWidget(self)
        mode_layout = QVBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)
        mode_layout.addWidget(self.raw_radio)
        mode_layout.addWidget(self.display_radio)
        form.addRow("像素来源", mode_row)

        self.format_combo = NoWheelComboBox(self)
        for label, value in (
            ("PNG（无损）", RasterExportFormat.PNG),
            ("JPEG（有损）", RasterExportFormat.JPEG),
            ("TIFF（无损）", RasterExportFormat.TIFF),
            ("BMP", RasterExportFormat.BMP),
            ("WebP", RasterExportFormat.WEBP),
        ):
            self.format_combo.addItem(label, value)
        self.format_combo.setCurrentIndex(
            max(0, self.format_combo.findData(options.format))
        )
        form.addRow("格式", self.format_combo)

        self.quality_spin = NoWheelSpinBox(self)
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setSuffix("%")
        self.quality_spin.setValue(int(options.resolved_quality or 95))
        form.addRow("质量", self.quality_spin)

        self.jpeg_progressive = QCheckBox("渐进式 JPEG", self)
        self.jpeg_progressive.setChecked(options.jpeg_progressive)
        form.addRow("", self.jpeg_progressive)

        self.png_compression_spin = NoWheelSpinBox(self)
        self.png_compression_spin.setRange(0, 9)
        self.png_compression_spin.setValue(options.png_compression)
        form.addRow("PNG 压缩级别", self.png_compression_spin)

        self.tiff_compression_combo = NoWheelComboBox(self)
        self.tiff_compression_combo.addItem("Deflate", TiffCompression.DEFLATE)
        self.tiff_compression_combo.addItem("LZW", TiffCompression.LZW)
        self.tiff_compression_combo.addItem("无压缩", TiffCompression.NONE)
        self.tiff_compression_combo.setCurrentIndex(
            max(
                0,
                self.tiff_compression_combo.findData(
                    options.tiff_compression
                ),
            )
        )
        form.addRow("TIFF 压缩", self.tiff_compression_combo)

        self.webp_lossless = QCheckBox("使用无损 WebP", self)
        self.webp_lossless.setChecked(options.webp_lossless)
        form.addRow("", self.webp_lossless)

        self.webp_method_spin = NoWheelSpinBox(self)
        self.webp_method_spin.setRange(0, 6)
        self.webp_method_spin.setValue(options.webp_method)
        form.addRow("WebP 编码强度", self.webp_method_spin)

        self.background_button = QPushButton(self)
        self.background_button.clicked.connect(self._choose_background)
        self._refresh_background_button()
        form.addRow("透明区域背景", self.background_button)

        self.hint_label = QLabel(self)
        self.hint_label.setWordWrap(True)
        form.addRow("", self.hint_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("导出")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self.format_combo.currentIndexChanged.connect(
            self._on_format_changed
        )
        self.webp_lossless.toggled.connect(self._update_format_controls)
        self._update_format_controls()

    def export_path(self) -> Path:
        path = Path(self.path_edit.text().strip()).expanduser()
        suffix = self.encoding_options().canonical_suffix
        if path.suffix.lower() not in self.encoding_options().format.accepted_suffixes:
            path = path.with_suffix(suffix)
        return path

    def export_mode(self) -> str:
        return (
            CurrentImageExportMode.RAW_PIXELS
            if self.raw_radio.isChecked()
            else CurrentImageExportMode.CURRENT_DISPLAY
        )

    def encoding_options(self) -> RasterEncodingOptions:
        export_format = self.format_combo.currentData()
        return RasterEncodingOptions(
            format=export_format,
            quality=(
                self.quality_spin.value()
                if export_format
                in {RasterExportFormat.JPEG, RasterExportFormat.WEBP}
                else None
            ),
            jpeg_progressive=self.jpeg_progressive.isChecked(),
            png_compression=self.png_compression_spin.value(),
            tiff_compression=self.tiff_compression_combo.currentData(),
            webp_lossless=self.webp_lossless.isChecked(),
            webp_method=self.webp_method_spin.value(),
            jpeg_background=self._background,
        )

    def _accept_if_valid(self) -> None:
        if not self.path_edit.text().strip():
            self.path_edit.setFocus()
            return
        self.path_edit.setText(str(self.export_path()))
        self.accept()

    def _browse(self) -> None:
        options = self.encoding_options()
        selected, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出当前图像",
            str(self.export_path()),
            _dialog_filter(options.format),
        )
        if selected:
            self.path_edit.setText(str(Path(selected)))
            self._normalize_path_suffix()

    def _on_format_changed(self) -> None:
        self._normalize_path_suffix()
        self._update_format_controls()

    def _normalize_path_suffix(self) -> None:
        token = self.path_edit.text().strip()
        if not token:
            return
        path = Path(token)
        options = self.encoding_options()
        if path.suffix.lower() not in options.format.accepted_suffixes:
            self.path_edit.setText(str(path.with_suffix(options.canonical_suffix)))

    def _update_format_controls(self) -> None:
        export_format = self.format_combo.currentData()
        is_png = export_format == RasterExportFormat.PNG
        is_jpeg = export_format == RasterExportFormat.JPEG
        is_tiff = export_format == RasterExportFormat.TIFF
        is_bmp = export_format == RasterExportFormat.BMP
        is_webp = export_format == RasterExportFormat.WEBP
        self.quality_spin.setEnabled(
            is_jpeg or (is_webp and not self.webp_lossless.isChecked())
        )
        self.jpeg_progressive.setEnabled(is_jpeg)
        self.png_compression_spin.setEnabled(is_png)
        self.tiff_compression_combo.setEnabled(is_tiff)
        self.webp_lossless.setEnabled(is_webp)
        self.webp_method_spin.setEnabled(is_webp)
        self.background_button.setEnabled(is_jpeg or is_bmp)
        if is_jpeg:
            self.hint_label.setText(
                "JPEG 是有损格式，不建议将此文件再次用于定量测量或面积识别。"
            )
        elif is_png:
            self.hint_label.setText(
                "PNG 压缩级别只影响编码速度和文件大小，不影响像素质量。"
            )
        elif is_tiff:
            self.hint_label.setText(
                "TIFF 适合保存高位深像素；不兼容的位深不会被静默转换。"
            )
        elif is_webp and self.webp_lossless.isChecked():
            self.hint_label.setText("当前使用无损 WebP。")
        elif is_webp:
            self.hint_label.setText("当前使用有损 WebP 质量设置。")
        else:
            self.hint_label.setText("BMP 不提供压缩或质量选项。")

    def _choose_background(self) -> None:
        selected = QColorDialog.getColor(
            QColor(*self._background),
            self,
            "选择透明区域背景色",
        )
        if not selected.isValid():
            return
        self._background = (
            selected.red(),
            selected.green(),
            selected.blue(),
        )
        self._refresh_background_button()

    def _refresh_background_button(self) -> None:
        red, green, blue = self._background
        color = QColor(red, green, blue)
        foreground = "#111111" if color.lightness() > 160 else "#FFFFFF"
        self.background_button.setText(f"RGB({red}, {green}, {blue})")
        self.background_button.setStyleSheet(
            "QPushButton {"
            f"background-color: rgb({red}, {green}, {blue});"
            f"color: {foreground};"
            "}"
        )


def _dialog_filter(export_format: RasterExportFormat) -> str:
    return {
        RasterExportFormat.PNG: "PNG 图片 (*.png)",
        RasterExportFormat.JPEG: "JPEG 图片 (*.jpg *.jpeg)",
        RasterExportFormat.TIFF: "TIFF 图片 (*.tif *.tiff)",
        RasterExportFormat.BMP: "BMP 图片 (*.bmp)",
        RasterExportFormat.WEBP: "WebP 图片 (*.webp)",
    }[export_format]
