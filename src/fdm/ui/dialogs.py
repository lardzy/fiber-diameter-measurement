from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Thread

from PySide6.QtCore import QEvent, QLineF, QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontInfo, QFontMetrics, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QApplication,
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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QHeaderView,
    QFrame,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fdm.models import (
    ImageDocument,
    OverlayTextAnchorAlignment,
    OverlayTextSizeSpace,
)
from fdm.settings import (
    AppThemeMode,
    AreaInferDevice,
    AreaModelMapping,
    AppSettings,
    DEFAULT_MEASUREMENT_LABEL_COLOR,
    FocusStackProfile,
    MagicSegmentModelVariant,
    MeasurementEndpointStyle,
    MeasurementLabelStyleSettings,
    OpenImageViewMode,
    RawRecordDataSource,
    RawRecordExportDirection,
    RawRecordExportRule,
    RawRecordMeasurementFilter,
    RawRecordTemplate,
    ScaleOverlayStyle,
    ScaleOverlayPlacementMode,
    SUPPORTED_RAW_RECORD_TEMPLATE_SUFFIXES,
    application_root,
    bundle_resource_root,
    resolve_app_relative_path,
    resolve_resource_relative_path,
    to_app_relative_path,
    to_resource_relative_path,
)
from fdm.services.export_service import ExportImageRenderMode, ExportScope, ExportSelection
from fdm.services.digital_slide_store import (
    DIGITAL_SLIDE_SUFFIX,
    DIGITAL_SLIDE_TILE_CODEC_JPEG,
    DIGITAL_SLIDE_TILE_CODEC_PNG,
    compress_slide_file,
    normalize_jpeg_quality,
    normalize_tile_codec,
)
from fdm.services.raw_record_export import RAW_RECORD_FIELD_NAMES


RAW_RECORD_DATA_SOURCE_ITEMS = [
    ("直径结果", RawRecordDataSource.DIAMETER_RESULT),
    ("面积结果", RawRecordDataSource.AREA_RESULT),
    ("测量明细字段", RawRecordDataSource.MEASUREMENT_FIELD),
    ("去重字段范围", RawRecordDataSource.UNIQUE_FIELD_RANGE),
]

RAW_RECORD_FILTER_ITEMS = [
    ("全部", RawRecordMeasurementFilter.ALL),
    ("直径/线段", RawRecordMeasurementFilter.LINE),
    ("面积", RawRecordMeasurementFilter.AREA),
    ("折线", RawRecordMeasurementFilter.POLYLINE),
    ("计数点", RawRecordMeasurementFilter.COUNT),
]

RAW_RECORD_DIRECTION_ITEMS = [
    ("纵向", RawRecordExportDirection.VERTICAL),
    ("横向", RawRecordExportDirection.HORIZONTAL),
]

OVERLAY_TEXT_ANCHOR_ITEMS = [
    ("左上", OverlayTextAnchorAlignment.TOP_LEFT),
    ("上中", OverlayTextAnchorAlignment.TOP_CENTER),
    ("右上", OverlayTextAnchorAlignment.TOP_RIGHT),
    ("左中", OverlayTextAnchorAlignment.CENTER_LEFT),
    ("中心", OverlayTextAnchorAlignment.CENTER),
    ("右中", OverlayTextAnchorAlignment.CENTER_RIGHT),
    ("左下", OverlayTextAnchorAlignment.BOTTOM_LEFT),
    ("下中", OverlayTextAnchorAlignment.BOTTOM_CENTER),
    ("右下", OverlayTextAnchorAlignment.BOTTOM_RIGHT),
]


class _MeasurementStylePreview(QWidget):
    """Small live preview for clean-profile measurement appearance settings."""

    def __init__(self, parent: QWidget | None = None, *, metric: str = "length") -> None:
        super().__init__(parent)
        self._metric = "area" if metric == "area" else "length"
        self._show_label = True
        self._font = QFont()
        self._label_color = QColor(DEFAULT_MEASUREMENT_LABEL_COLOR)
        self._line_color = QColor("#2A9D8F")
        self._background_enabled = True
        self._decimals = 2
        self._endpoint_style = MeasurementEndpointStyle.BAR
        self.setMinimumHeight(86)

    def set_preview_style(
        self,
        *,
        show_label: bool,
        font: QFont,
        label_color: str,
        line_color: str,
        background_enabled: bool,
        decimals: int,
        endpoint_style: str,
    ) -> None:
        self._show_label = bool(show_label)
        self._font = QFont(font)
        self._label_color = QColor(label_color)
        self._line_color = QColor(line_color)
        self._background_enabled = bool(background_enabled)
        self._decimals = max(0, min(8, int(decimals)))
        self._endpoint_style = str(endpoint_style)
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.fillRect(rect, self.palette().color(QPalette.ColorRole.AlternateBase))
        painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Mid), 1))
        painter.drawRoundedRect(rect, 6, 6)

        left = rect.left() + 28
        right = rect.right() - 28
        y = rect.center().y() + 10
        painter.setPen(QPen(self._line_color, 2.5))
        if self._metric == "area":
            path = QPainterPath()
            path.moveTo(QPointF(left + 30, y + 11))
            path.lineTo(QPointF(left + 52, y - 18))
            path.lineTo(QPointF(right - 36, y - 12))
            path.lineTo(QPointF(right - 18, y + 14))
            path.lineTo(QPointF(left + 30, y + 11))
            painter.setBrush(QColor(self._line_color.red(), self._line_color.green(), self._line_color.blue(), 48))
            painter.drawPath(path)
        else:
            painter.drawLine(QLineF(left, y, right, y))
            if self._endpoint_style == MeasurementEndpointStyle.CIRCLE:
                painter.setBrush(self._line_color)
                painter.drawEllipse(QRectF(left - 3, y - 3, 6, 6))
                painter.drawEllipse(QRectF(right - 3, y - 3, 6, 6))
            elif self._endpoint_style == MeasurementEndpointStyle.BAR:
                painter.drawLine(QLineF(left, y - 7, left, y + 7))
                painter.drawLine(QLineF(right, y - 7, right, y + 7))
            elif self._endpoint_style in {
                MeasurementEndpointStyle.ARROW_INSIDE,
                MeasurementEndpointStyle.ARROW_OUTSIDE,
            }:
                direction = 1 if self._endpoint_style == MeasurementEndpointStyle.ARROW_INSIDE else -1
                painter.drawLine(QLineF(left, y, left + 8 * direction, y - 5))
                painter.drawLine(QLineF(left, y, left + 8 * direction, y + 5))
                painter.drawLine(QLineF(right, y, right - 8 * direction, y - 5))
                painter.drawLine(QLineF(right, y, right - 8 * direction, y + 5))

        if not self._show_label:
            return
        painter.setFont(self._font)
        suffix = " μm²" if self._metric == "area" else " μm"
        text = f"{12.3456:.{self._decimals}f}{suffix}"
        metrics = QFontMetrics(self._font)
        text_rect = QRectF(metrics.boundingRect(text)).adjusted(-6, -3, 6, 3)
        text_rect.moveCenter(QRectF(rect.left(), rect.top(), rect.width(), rect.height() * 0.55).center())
        if self._background_enabled:
            painter.fillRect(text_rect, QColor(15, 23, 42, 190))
        painter.setPen(self._label_color)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)


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


class DigitalSlideCompressionWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, source: Path, target: Path, *, codec: str, quality: int | None) -> None:
        super().__init__()
        self._source = source
        self._target = target
        self._codec = normalize_tile_codec(codec)
        self._quality = normalize_jpeg_quality(quality) if self._codec == DIGITAL_SLIDE_TILE_CODEC_JPEG else None
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = Thread(target=self._run, name=f"fdm-slide-compress-{self._source.name}", daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            result = compress_slide_file(
                self._source,
                self._target,
                codec=self._codec,
                quality=self._quality,
                progress_callback=lambda completed, total: self.progress.emit(int(completed), int(total)),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(str(result))


def _digital_slide_quality_label_text(value: int) -> str:
    quality = normalize_jpeg_quality(value)
    if quality <= 80:
        level = "中等留档"
    elif quality <= 90:
        level = "高质量"
    else:
        level = "更高质量/更大文件"
    return f"{quality} ({level})"


class DigitalSlideCompressionDialog(QDialog):
    """Standalone maintenance task for creating a compressed slide copy."""

    compression_finished = Signal(str)

    def __init__(
        self,
        settings: AppSettings,
        *,
        source_path: str | Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("压缩数字化切片副本")
        self.resize(680, 360)
        self.setMinimumSize(560, 320)
        self._worker: DigitalSlideCompressionWorker | None = None
        self._running = False
        self._completed_path: Path | None = None

        heading = QLabel("压缩数字化切片副本", self)
        heading.setObjectName("digitalSlideCompressionTitle")
        hint = QLabel(
            "源文件保持不变，压缩结果始终另存为新的 .fdmslide 文件。"
            "JPEG 可以减小体积，但可能引入压缩伪影；精确测量建议保留 PNG 无损原件。",
            self,
        )
        hint.setWordWrap(True)

        paths_group = QGroupBox("文件", self)
        paths_layout = QVBoxLayout(paths_group)
        source_row = QHBoxLayout()
        self._source_edit = QLineEdit(paths_group)
        self._source_edit.setPlaceholderText("源 .fdmslide 文件")
        self._source_button = QPushButton("选择源文件", paths_group)
        self._source_button.clicked.connect(self._choose_source)
        source_row.addWidget(self._source_edit, 1)
        source_row.addWidget(self._source_button)
        paths_layout.addLayout(source_row)

        target_row = QHBoxLayout()
        self._target_edit = QLineEdit(paths_group)
        self._target_edit.setPlaceholderText("压缩副本保存位置")
        self._target_button = QPushButton("另存为", paths_group)
        self._target_button.clicked.connect(self._choose_target)
        target_row.addWidget(self._target_edit, 1)
        target_row.addWidget(self._target_button)
        paths_layout.addLayout(target_row)

        options_group = QGroupBox("压缩选项", self)
        options_form = QFormLayout(options_group)
        self._codec_combo = NoWheelComboBox(options_group)
        self._codec_combo.addItem("JPEG 压缩", DIGITAL_SLIDE_TILE_CODEC_JPEG)
        self._codec_combo.addItem("PNG 无损副本", DIGITAL_SLIDE_TILE_CODEC_PNG)
        self._quality_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self._quality_slider.setRange(70, 95)
        self._quality_slider.setValue(normalize_jpeg_quality(settings.digital_slide_capture_jpeg_quality))
        self._quality_label = QLabel(options_group)
        self._quality_label.setMinimumWidth(150)
        self._quality_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        quality_row = QWidget(options_group)
        quality_layout = QHBoxLayout(quality_row)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.addWidget(self._quality_slider, 1)
        quality_layout.addWidget(self._quality_label)
        options_form.addRow("输出格式", self._codec_combo)
        options_form.addRow("JPEG 质量", quality_row)
        self._quality_slider.valueChanged.connect(self._update_quality_label)
        self._codec_combo.currentIndexChanged.connect(self._sync_quality_visibility)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("等待开始")

        self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        close_button = self._button_box.button(QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.setText("关闭")
        self._start_button = self._button_box.addButton("开始压缩", QDialogButtonBox.ButtonRole.ActionRole)
        self._start_button.clicked.connect(self.start_compression)
        self._button_box.rejected.connect(self.reject)

        self._task_controls = [
            self._source_edit,
            self._source_button,
            self._target_edit,
            self._target_button,
            self._codec_combo,
            self._quality_slider,
            self._start_button,
        ]
        self._sync_quality_visibility()
        if source_path is not None and str(source_path).strip():
            self.set_source_path(source_path)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)
        layout.addWidget(heading)
        layout.addWidget(hint)
        layout.addWidget(paths_group)
        layout.addWidget(options_group)
        layout.addWidget(self._progress)
        layout.addStretch(1)
        layout.addWidget(self._button_box)
        self.setStyleSheet(
            "QLabel#digitalSlideCompressionTitle { font-size: 18px; font-weight: 700; }"
        )

    def set_source_path(self, path: str | Path) -> None:
        source = Path(path).expanduser()
        self._source_edit.setText(str(source))
        self._target_edit.setText(str(self._default_target_path(source)))

    def source_path(self) -> Path | None:
        token = self._source_edit.text().strip()
        return Path(token).expanduser() if token else None

    def target_path(self) -> Path | None:
        token = self._target_edit.text().strip()
        if not token:
            return None
        target = Path(token).expanduser()
        return target if target.suffix.lower() == DIGITAL_SLIDE_SUFFIX else target.with_suffix(DIGITAL_SLIDE_SUFFIX)

    def completed_path(self) -> Path | None:
        return self._completed_path

    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def _default_target_path(source: Path) -> Path:
        return source.with_name(f"{source.stem}_compressed{DIGITAL_SLIDE_SUFFIX}")

    def _choose_source(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择数字化切片文件",
            "",
            f"数字化切片 (*{DIGITAL_SLIDE_SUFFIX});;所有文件 (*)",
        )
        if path:
            self.set_source_path(path)

    def _choose_target(self) -> None:
        source = self.source_path()
        default_path = str(self._default_target_path(source)) if source is not None else ""
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存压缩数字化切片",
            default_path,
            f"数字化切片 (*{DIGITAL_SLIDE_SUFFIX});;所有文件 (*)",
        )
        if not path:
            return
        target = Path(path).expanduser()
        if target.suffix.lower() != DIGITAL_SLIDE_SUFFIX:
            target = target.with_suffix(DIGITAL_SLIDE_SUFFIX)
        self._target_edit.setText(str(target))

    def _update_quality_label(self, value: int) -> None:
        self._quality_label.setText(_digital_slide_quality_label_text(value))

    def _sync_quality_visibility(self) -> None:
        is_jpeg = normalize_tile_codec(self._codec_combo.currentData()) == DIGITAL_SLIDE_TILE_CODEC_JPEG
        self._quality_slider.setEnabled(is_jpeg and not self._running)
        self._quality_label.setEnabled(is_jpeg)
        self._update_quality_label(self._quality_slider.value())

    def _set_task_controls_enabled(self, enabled: bool) -> None:
        for control in self._task_controls:
            control.setEnabled(enabled)
        self._sync_quality_visibility()

    def start_compression(self) -> bool:
        if self._running:
            return False
        source = self.source_path()
        if source is None:
            QMessageBox.information(self, "切片压缩", "请先选择源 .fdmslide 文件。")
            return False
        if not source.exists() or source.suffix.lower() != DIGITAL_SLIDE_SUFFIX:
            QMessageBox.warning(self, "切片压缩", "源文件不存在或不是 .fdmslide 文件。")
            return False
        target = self.target_path() or self._default_target_path(source)
        if source.resolve() == target.resolve():
            QMessageBox.warning(self, "切片压缩", "压缩目标不能与源文件相同，请选择另存副本。")
            return False
        if target.exists():
            response = QMessageBox.question(
                self,
                "覆盖压缩文件",
                f"目标文件已存在，是否覆盖？\n{target}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return False
        self._target_edit.setText(str(target))
        codec = normalize_tile_codec(self._codec_combo.currentData())
        quality = self._quality_slider.value() if codec == DIGITAL_SLIDE_TILE_CODEC_JPEG else None
        self._running = True
        self._completed_path = None
        self._set_task_controls_enabled(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("准备压缩...")
        worker = DigitalSlideCompressionWorker(source, target, codec=codec, quality=quality)
        self._worker = worker
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        try:
            worker.start()
        except Exception as exc:
            self._progress.setFormat("压缩失败")
            self._finish_task_ui()
            QMessageBox.warning(self, "切片压缩", f"无法启动压缩任务：\n{exc}")
            return False
        return True

    def _on_progress(self, completed: int, total: int) -> None:
        total = max(1, int(total))
        completed = max(0, min(int(completed), total))
        self._progress.setRange(0, total)
        self._progress.setValue(completed)
        self._progress.setFormat(f"{completed}/{total} 张")

    def _finish_task_ui(self) -> None:
        self._running = False
        self._worker = None
        self._set_task_controls_enabled(True)

    def _on_finished(self, path: str) -> None:
        self._completed_path = Path(path)
        self._progress.setFormat("压缩完成")
        self._finish_task_ui()
        self.compression_finished.emit(path)
        QMessageBox.information(self, "切片压缩", f"压缩完成：\n{path}")

    def _on_failed(self, message: str) -> None:
        self._progress.setFormat("压缩失败")
        self._finish_task_ui()
        QMessageBox.warning(self, "切片压缩", f"压缩失败：\n{message}")

    def accept(self) -> None:
        if self._running:
            QMessageBox.information(self, "切片压缩", "切片压缩正在进行，请等待完成后再关闭窗口。")
            return
        super().accept()

    def reject(self) -> None:
        if self._running:
            QMessageBox.information(self, "切片压缩", "切片压缩正在进行，请等待完成后再关闭窗口。")
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self._running:
            QMessageBox.information(self, "切片压缩", "切片压缩正在进行，请等待完成后再关闭窗口。")
            event.ignore()
            return
        super().closeEvent(event)


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
        self._apply_to_project.setChecked(True)

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
        initial_color: str = "#1F7A8C",
        apply_to_project_default: bool = True,
        show_apply_to_project: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._show_apply_to_project = show_apply_to_project
        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText("类别名称")
        self._label_edit.setText(initial_label)
        self._apply_to_project = QCheckBox("应用到当前项目全局")
        self._apply_to_project.setChecked(apply_to_project_default)
        self._color_button = QPushButton()
        self._color_button.clicked.connect(self._choose_color)
        self._apply_button_color(initial_color)

        form = QFormLayout()
        form.addRow("类别名称", self._label_edit)
        form.addRow("类别颜色", self._color_button)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        if self._show_apply_to_project:
            layout.addWidget(self._apply_to_project)
        layout.addWidget(buttons)

    def _apply_button_color(self, color_value: str) -> None:
        color = QColor(color_value)
        normalized = color.name() if color.isValid() else color_value
        text_color = "#111111" if color.isValid() and color.lightnessF() > 0.7 else "#FFFFFF"
        self._color_button.setText(normalized)
        self._color_button.setStyleSheet(
            f"QPushButton {{ background: {normalized}; color: {text_color}; min-height: 28px; border-radius: 6px; }}"
        )
        self._color_button.setProperty("color_value", normalized)

    def _choose_color(self) -> None:
        initial = QColor(str(self._color_button.property("color_value") or "#1F7A8C"))
        color = QColorDialog.getColor(initial, self, "选择颜色")
        if not color.isValid():
            return
        self._apply_button_color(color.name())

    def values(self) -> tuple[str, str, bool]:
        return (
            self._label_edit.text().strip(),
            str(self._color_button.property("color_value") or "#1F7A8C"),
            self._apply_to_project.isChecked() if self._show_apply_to_project else False,
        )


class ExportOptionsDialog(QDialog):
    def __init__(
        self,
        selection: ExportSelection,
        *,
        allow_all_scope: bool,
        legacy_overlay_text_count_current: int = 0,
        legacy_overlay_text_count_all: int = 0,
        raw_record_templates: list[RawRecordTemplate] | None = None,
        last_raw_record_template_path: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出选项")
        self._legacy_overlay_text_count_current = max(
            0,
            int(legacy_overlay_text_count_current),
        )
        self._legacy_overlay_text_count_all = max(
            0,
            int(legacy_overlay_text_count_all),
        )

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
        self._scope_current.setChecked(not allow_all_scope)
        self._scope_all.setChecked(allow_all_scope)
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
        self._legacy_overlay_text_warning = QLabel()
        self._legacy_overlay_text_warning.setObjectName(
            "exportLegacyOverlayTextWarning"
        )
        self._legacy_overlay_text_warning.setWordWrap(True)
        render_layout.addRow("渲染方式", self._render_mode_combo)
        render_layout.addRow("", self._render_mode_hint)
        render_layout.addRow("", self._legacy_overlay_text_warning)

        raw_record_group = QGroupBox("原始记录模板")
        raw_record_layout = QFormLayout(raw_record_group)
        self._raw_record_template_combo = QComboBox()
        self._raw_record_template_combo.addItem("不使用模板", "")
        for template in raw_record_templates or []:
            display_name = template.name or Path(template.path).stem or template.path
            self._raw_record_template_combo.addItem(display_name, template.path)
        selected_template_path = selection.raw_record_template_path or last_raw_record_template_path
        template_index = self._raw_record_template_combo.findData(selected_template_path)
        self._raw_record_template_combo.setCurrentIndex(max(0, template_index))
        self._raw_record_template_hint = QLabel("选择后会复制模板并写入测量数据；模板文件本身不会被修改。")
        self._raw_record_template_hint.setWordWrap(True)
        raw_record_layout.addRow("模板", self._raw_record_template_combo)
        raw_record_layout.addRow("", self._raw_record_template_hint)

        self._measurement_overlay.toggled.connect(self._update_render_mode_state)
        self._scale_overlay.toggled.connect(self._update_render_mode_state)
        self._combined_overlay.toggled.connect(self._update_render_mode_state)
        self._scope_current.toggled.connect(
            self._update_legacy_overlay_text_warning
        )
        self._scope_all.toggled.connect(
            self._update_legacy_overlay_text_warning
        )
        self._render_mode_combo.currentIndexChanged.connect(
            self._update_legacy_overlay_text_warning
        )
        self._excel.toggled.connect(self._update_raw_record_template_state)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(export_group)
        layout.addWidget(scope_group)
        layout.addWidget(render_group)
        layout.addWidget(raw_record_group)
        layout.addWidget(buttons)
        self._update_render_mode_state()
        self._update_raw_record_template_state()

    def _update_render_mode_state(self) -> None:
        enabled = (
            self._measurement_overlay.isChecked()
            or self._scale_overlay.isChecked()
            or self._combined_overlay.isChecked()
        )
        self._render_mode_combo.setEnabled(enabled)
        self._render_mode_hint.setEnabled(enabled)
        self._update_legacy_overlay_text_warning()

    def _update_legacy_overlay_text_warning(self) -> None:
        image_overlay_selected = (
            self._measurement_overlay.isChecked()
            or self._scale_overlay.isChecked()
            or self._combined_overlay.isChecked()
        )
        full_resolution = (
            self._render_mode_combo.currentData()
            == ExportImageRenderMode.FULL_RESOLUTION
        )
        legacy_count = (
            self._legacy_overlay_text_count_all
            if self._scope_all.isChecked() and self._scope_all.isEnabled()
            else self._legacy_overlay_text_count_current
        )
        visible = image_overlay_selected and full_resolution and legacy_count > 0
        if visible:
            self._legacy_overlay_text_warning.setText(
                "⚠ 当前导出范围含 "
                f"{legacy_count} 个旧版固定像素文字。完整分辨率导出时，"
                "文字可能显得过小；建议先在“当前对象属性”中转换为随图像缩放文字。"
            )
        else:
            self._legacy_overlay_text_warning.clear()
        self._legacy_overlay_text_warning.setVisible(visible)

    def _update_raw_record_template_state(self) -> None:
        enabled = self._excel.isChecked()
        self._raw_record_template_combo.setEnabled(enabled)
        self._raw_record_template_hint.setEnabled(enabled)

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
            raw_record_template_path=str(self._raw_record_template_combo.currentData() or "") if self._excel.isChecked() else "",
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
                    "数字化切片浏览",
                    "M  切换步进移动 / 平滑移动",
                    "方向键  移动当前视场",
                    "Shift+方向键  按整视场移动",
                    "鼠标滚轮  切换焦层",
                    "Ctrl+鼠标滚轮  缩放当前视场",
                    "",
                    "数字化切片采集地图",
                    "地图坐标为 X 向右、Y 向下；绿色框表示当前视场，浅色框表示接下来采集的范围。",
                    "方向按钮用于把当前视场的对应边缘指定为采集范围边界。",
                    "如果设备实际左右或上下移动与界面相反，请在设置 > 数字化切片 > 运动控制中启用对应方向反转。",
                    "",
                    "面积与魔棒",
                    "R  在正采样点 / 负采样点之间切换",
                    "Y  切换 ROI 限制区域",
                    "T  标准魔棒或选中面积时，在添加模式 / 剔除模式之间切换",
                    "1 / 2 / 3  在智能 / 多边形 / 自由圈选剔除之间切换",
                    "S  确认当前剔除形状并继续添加下一块（仅剔除模式）",
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


class _SettingsTabsCompatibility:
    """Read-only adapter for legacy tests and integrations using ``_tabs``.

    Settings pages are no longer presented as horizontal tabs.  Keeping this
    deliberately small adapter avoids coupling callers to the new navigation
    widgets while callers migrate to the category navigation.
    """

    def __init__(self, labels: list[str], pages: list[QWidget]) -> None:
        self._labels = list(labels)
        self._pages = list(pages)

    def count(self) -> int:
        return len(self._pages)

    def tabText(self, index: int) -> str:
        return self._labels[index]

    def widget(self, index: int) -> QWidget:
        return self._pages[index]


class SettingsDialog(QDialog):
    _NAVIGATION_DEFINITIONS = (
        ("常规", "主题与默认视图", "主题 深色 浅色 系统 打开 图片 默认 视图"),
        ("测量与显示", "测量结果、计数和线条外观", "测量 结果 文字 标签 计数 编号 端点 颜色"),
        ("标注与比例尺", "比例尺和图形标注默认样式", "比例尺 叠加 文字 图形 矩形 圆形 箭头 线条"),
        ("图像与智能分析", "景深合成、魔棒和快速测径", "图像 景深 合成 锐化 魔棒 EdgeSAM ROI 快速测径"),
        ("面积识别", "面积模型、权重和推理设备", "面积 模型 权重 Python CPU CUDA 推理"),
        ("采集与数字切片", "预览、运动控制和切片参数", "采集 预览 数字化切片 电机 运动 焦层"),
        ("导出与模板", "原始记录模板和导出规则", "导出 原始记录 模板 规则 Excel 工作表"),
    )

    def __init__(
        self,
        settings: AppSettings,
        *,
        document: ImageDocument | None,
        digital_slide_locked: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("首选项")
        # Keep the pre-show size compatible with callers that inspect a newly
        # constructed dialog.  The preferred clamped size is applied when the
        # real window is shown, once its target screen is known.
        self.resize(700, 560)
        self._preferred_size_applied = False
        self._initial_settings = replace(settings)
        self._document = document
        self._request_scale_anchor_pick = False
        self._raw_record_templates_data = [template.normalized_copy() for template in settings.raw_record_templates]
        self._raw_record_current_template_index = -1

        general_page = self._build_general_tab(settings)
        measurement_page = self._build_measurement_tab(settings)
        annotation_scale_page = self._build_scale_overlay_tab(settings)
        image_processing_page = self._build_image_processing_tab(settings)
        area_models_page = self._build_area_models_tab(settings)
        digital_slide_page = self._build_digital_slide_tab(settings, locked=digital_slide_locked)
        raw_record_page = self._build_raw_record_templates_tab(settings)

        self._settings_pages = QStackedWidget(self)
        self._settings_pages.setObjectName("settingsPages")
        navigation_pages = [
            general_page,
            measurement_page,
            annotation_scale_page,
            image_processing_page,
            area_models_page,
            digital_slide_page,
            raw_record_page,
        ]
        for page in navigation_pages:
            self._settings_pages.addWidget(page)

        # ``_tabs`` is private legacy surface retained only as a read-only
        # compatibility layer.  The two proxy pages preserve the previous
        # group-title structure without re-parenting live controls.
        legacy_scale_page = self._build_legacy_group_page(("默认视图", "位置与长度", "样式"))
        legacy_overlay_page = self._build_legacy_group_page(("文字默认样式", "图形默认样式"))
        legacy_scale_page.hide()
        legacy_overlay_page.hide()
        self._tabs = _SettingsTabsCompatibility(
            ["测量标注", "比例尺叠加", "图像处理", "叠加标注", "数字化切片", "面积识别", "原始记录模板"],
            [
                measurement_page,
                legacy_scale_page,
                image_processing_page,
                legacy_overlay_page,
                digital_slide_page,
                area_models_page,
                raw_record_page,
            ],
        )

        self._settings_navigation = QListWidget(self)
        self._settings_navigation.setObjectName("settingsNavigation")
        self._settings_navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._settings_navigation.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._settings_navigation.setUniformItemSizes(True)
        self._settings_navigation.setSpacing(2)
        self._settings_navigation_items: list[QListWidgetItem] = []
        self._settings_search_texts: list[str] = []
        for index, ((label, description, keywords), page) in enumerate(
            zip(self._NAVIGATION_DEFINITIONS, navigation_pages, strict=True)
        ):
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setToolTip(description)
            self._settings_navigation.addItem(item)
            self._settings_navigation_items.append(item)
            self._settings_search_texts.append(
                self._settings_page_search_text(label, description, keywords, page)
            )

        self._settings_search = QLineEdit(self)
        self._settings_search.setObjectName("settingsSearch")
        self._settings_search.setPlaceholderText("搜索设置")
        self._settings_search.setClearButtonEnabled(True)
        self._settings_search.textChanged.connect(self._filter_settings_navigation)
        self._settings_navigation.currentItemChanged.connect(self._activate_settings_navigation_item)

        sidebar = QFrame(self)
        sidebar.setObjectName("settingsSidebar")
        sidebar.setMinimumWidth(190)
        sidebar.setMaximumWidth(240)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(8)
        sidebar_title = QLabel("首选项", sidebar)
        sidebar_title.setObjectName("settingsSidebarTitle")
        sidebar_layout.addWidget(sidebar_title)
        sidebar_layout.addWidget(self._settings_search)
        sidebar_layout.addWidget(self._settings_navigation, 1)
        self._settings_search_empty = QLabel("没有匹配的设置", sidebar)
        self._settings_search_empty.setObjectName("settingsSearchEmpty")
        self._settings_search_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._settings_search_empty.setWordWrap(True)
        self._settings_search_empty.hide()
        sidebar_layout.addWidget(self._settings_search_empty)

        self._settings_page_title = QLabel(self)
        self._settings_page_title.setObjectName("settingsPageTitle")
        self._settings_page_description = QLabel(self)
        self._settings_page_description.setObjectName("settingsPageDescription")
        self._settings_page_description.setWordWrap(True)
        page_layout = QVBoxLayout()
        page_layout.setContentsMargins(16, 12, 12, 8)
        page_layout.setSpacing(4)
        page_layout.addWidget(self._settings_page_title)
        page_layout.addWidget(self._settings_page_description)
        page_layout.addSpacing(6)
        page_layout.addWidget(self._settings_pages, 1)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(sidebar)
        content_layout.addLayout(page_layout, 1)

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        self._restore_page_defaults_button = self._button_box.addButton(
            "恢复本页默认值",
            QDialogButtonBox.ButtonRole.ResetRole,
        )
        self._restore_page_defaults_button.setToolTip("恢复当前分类的专业默认样式与参数；点击应用或确定后才会保存")
        self._restore_page_defaults_button.clicked.connect(self._restore_current_page_defaults)
        for standard_button, text in (
            (QDialogButtonBox.StandardButton.Ok, "确定"),
            (QDialogButtonBox.StandardButton.Cancel, "取消"),
            (QDialogButtonBox.StandardButton.Apply, "应用"),
        ):
            button = self._button_box.button(standard_button)
            if button is not None:
                button.setText(text)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 12, 12)
        layout.setSpacing(8)
        layout.addLayout(content_layout, 1)
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(12, 0, 0, 0)
        footer_layout.setSpacing(0)
        footer_layout.addWidget(self._button_box)
        layout.addLayout(footer_layout)
        self.setStyleSheet(
            """
            QFrame#settingsSidebar {
                border: none;
                border-right: 1px solid palette(mid);
            }
            QLabel#settingsSidebarTitle, QLabel#settingsPageTitle {
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#settingsPageDescription, QLabel#settingsSearchEmpty {
                color: palette(placeholder-text);
            }
            QListWidget#settingsNavigation {
                border: none;
                background: transparent;
                outline: none;
            }
            QListWidget#settingsNavigation::item {
                padding: 6px 9px;
                border-radius: 6px;
            }
            QListWidget#settingsNavigation::item:hover {
                background: rgba(42, 157, 143, 45);
            }
            QListWidget#settingsNavigation::item:selected {
                background: #2A9D8F;
                color: white;
            }
            """
        )
        self._update_settings_navigation_item_sizes()
        self._settings_navigation.setCurrentRow(0)

    @property
    def button_box(self) -> QDialogButtonBox:
        return self._button_box

    def _update_settings_navigation_item_sizes(self) -> None:
        """Keep navigation rows readable for the active system UI font."""

        navigation = getattr(self, "_settings_navigation", None)
        items = getattr(self, "_settings_navigation_items", ())
        if navigation is None:
            return
        metrics = QFontMetrics(navigation.font())
        row_height = max(40, metrics.height() + 16)
        row_size = QSize(0, row_height)
        for item in items:
            item.setSizeHint(row_size)
        navigation.scheduleDelayedItemsLayout()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in {
            QEvent.Type.ApplicationFontChange,
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
        }:
            self._update_settings_navigation_item_sizes()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._preferred_size_applied:
            return
        self._preferred_size_applied = True
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(900, 640)
            return
        available = screen.availableGeometry()
        margin = 32
        target_width = max(1, min(900, available.width() - margin))
        target_height = max(1, min(640, available.height() - margin))
        self.resize(target_width, target_height)

    def _settings_page_search_text(
        self,
        label: str,
        description: str,
        keywords: str,
        page: QWidget,
    ) -> str:
        parts = [label, description, keywords]
        parts.extend(group.title() for group in page.findChildren(QGroupBox))
        parts.extend(child.text() for child in page.findChildren(QLabel))
        parts.extend(child.text() for child in page.findChildren(QCheckBox))
        parts.extend(child.text() for child in page.findChildren(QPushButton))
        parts.extend(child.placeholderText() for child in page.findChildren(QLineEdit))
        for combo in page.findChildren(QComboBox):
            parts.extend(combo.itemText(index) for index in range(combo.count()))
        return " ".join(part for part in parts if part).casefold()

    def _filter_settings_navigation(self, query: str) -> None:
        tokens = [token.casefold() for token in query.split() if token.strip()]
        first_visible: QListWidgetItem | None = None
        current_visible = False
        current_item = self._settings_navigation.currentItem()
        for item, searchable_text in zip(
            self._settings_navigation_items,
            self._settings_search_texts,
            strict=True,
        ):
            visible = all(token in searchable_text for token in tokens)
            item.setHidden(not visible)
            if visible and first_visible is None:
                first_visible = item
            if item is current_item and visible:
                current_visible = True
        has_results = first_visible is not None
        self._settings_search_empty.setVisible(not has_results)
        if has_results and not current_visible:
            self._settings_navigation.setCurrentItem(first_visible)
        elif not has_results:
            self._settings_navigation.clearSelection()

    def _activate_settings_navigation_item(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None or current.isHidden():
            return
        index = int(current.data(Qt.ItemDataRole.UserRole))
        if not (0 <= index < self._settings_pages.count()):
            return
        self._settings_pages.setCurrentIndex(index)
        label, description, _keywords = self._NAVIGATION_DEFINITIONS[index]
        self._settings_page_title.setText(label)
        self._settings_page_description.setText(description)

    def _build_legacy_group_page(self, titles: tuple[str, ...]) -> QScrollArea:
        content = QWidget(self)
        layout = QVBoxLayout(content)
        for title in titles:
            layout.addWidget(QGroupBox(title, content))
        layout.addStretch(1)
        return self._wrap_settings_page(content)

    def _restore_current_page_defaults(self) -> None:
        """Restore only the visible preference category to clean-profile defaults."""

        page_index = self._settings_pages.currentIndex()
        page = self._settings_pages.currentWidget()
        if page is None:
            return
        defaults_dialog = SettingsDialog(
            AppSettings(),
            document=self._document,
            digital_slide_locked=False,
            parent=self,
        )
        try:
            default_page = defaults_dialog._settings_pages.widget(page_index)
            for name, target in vars(self).items():
                source = vars(defaults_dialog).get(name)
                if not isinstance(target, QWidget) or not isinstance(source, QWidget):
                    continue
                if not page.isAncestorOf(target) or not default_page.isAncestorOf(source):
                    continue
                self._copy_default_control_value(target, source)

            if page_index == 4:
                self._copy_table_contents(
                    self._area_mapping_table,
                    defaults_dialog._area_mapping_table,
                )
            elif page_index == 6:
                self._raw_record_templates_data = []
                self._raw_record_current_template_index = -1
                self._raw_record_template_table.setRowCount(0)
                self._raw_record_rule_table.setRowCount(0)
            self._settings_page_description.setText(
                self._NAVIGATION_DEFINITIONS[page_index][1]
                + "（已恢复本页默认值，尚未应用）"
            )
        finally:
            defaults_dialog.deleteLater()

    def _copy_default_control_value(self, target: QWidget, source: QWidget) -> None:
        if isinstance(target, QFontComboBox) and isinstance(source, QFontComboBox):
            target.setCurrentFont(source.currentFont())
            target.setProperty("requested_font_family", source.property("requested_font_family"))
            target.setProperty("font_user_changed", source.property("font_user_changed"))
            return
        if isinstance(target, QComboBox) and isinstance(source, QComboBox):
            index = target.findData(source.currentData())
            if index < 0:
                index = target.findText(source.currentText())
            if index >= 0:
                target.setCurrentIndex(index)
            return
        if isinstance(target, (QCheckBox, QRadioButton)) and isinstance(source, (QCheckBox, QRadioButton)):
            target.setChecked(source.isChecked())
            return
        if isinstance(target, (QSpinBox, QDoubleSpinBox)) and isinstance(source, (QSpinBox, QDoubleSpinBox)):
            target.setValue(source.value())
            return
        if isinstance(target, QSlider) and isinstance(source, QSlider):
            target.setValue(source.value())
            return
        if isinstance(target, QLineEdit) and isinstance(source, QLineEdit):
            target.setText(source.text())
            return
        if isinstance(target, QPushButton) and isinstance(source, QPushButton):
            color_value = source.property("color_value")
            if color_value:
                self._apply_button_color(target, str(color_value))

    @staticmethod
    def _copy_table_contents(target: QTableWidget, source: QTableWidget) -> None:
        target.setRowCount(source.rowCount())
        target.setColumnCount(source.columnCount())
        for row in range(source.rowCount()):
            for column in range(source.columnCount()):
                item = source.item(row, column)
                if item is not None:
                    target.setItem(row, column, item.clone())

    def app_settings(self) -> AppSettings:
        length_label_style = MeasurementLabelStyleSettings(
            enabled=self._show_length_measurement_labels.isChecked(),
            font_family=self._font_combo_family_value(self._length_measurement_label_font),
            font_size=self._length_measurement_label_size.value(),
            color=str(
                self._length_measurement_label_color.property("color_value")
                or self._initial_settings.length_measurement_label_style.color
            ),
            decimals=self._length_measurement_label_decimals.value(),
            background_enabled=self._length_measurement_label_background.isChecked(),
            parallel_to_line=self._length_measurement_label_parallel.isChecked(),
        )
        area_label_style = MeasurementLabelStyleSettings(
            enabled=self._show_area_measurement_labels.isChecked(),
            font_family=self._font_combo_family_value(self._area_measurement_label_font),
            font_size=self._area_measurement_label_size.value(),
            color=str(
                self._area_measurement_label_color.property("color_value")
                or self._initial_settings.area_measurement_label_style.color
            ),
            decimals=self._area_measurement_label_decimals.value(),
            background_enabled=self._area_measurement_label_background.isChecked(),
            parallel_to_line=False,
        )
        return AppSettings(
            theme_mode=self._theme_mode_combo.currentData(),
            length_measurement_label_style=length_label_style,
            area_measurement_label_style=area_label_style,
            show_count_numbers=self._show_count_numbers.isChecked(),
            count_number_font_family=self._font_combo_family_value(self._count_number_font),
            count_number_font_size=self._count_number_size.value(),
            count_number_color=self._count_number_color.property("color_value") or self._initial_settings.count_number_color,
            measurement_endpoint_style=self._endpoint_style_combo.currentData(),
            default_measurement_color=self._default_measurement_color.property("color_value") or self._initial_settings.default_measurement_color,
            open_image_view_mode=self._open_view_mode_combo.currentData(),
            scale_overlay_placement_mode=self._scale_overlay_mode_combo.currentData(),
            scale_overlay_style=self._scale_overlay_style_combo.currentData(),
            scale_overlay_length_value=self._scale_overlay_length_spin.value(),
            scale_overlay_color=self._scale_overlay_color.property("color_value") or self._initial_settings.scale_overlay_color,
            scale_overlay_text_color=self._scale_overlay_text_color.property("color_value") or self._initial_settings.scale_overlay_text_color,
            scale_overlay_font_family=self._font_combo_family_value(self._scale_overlay_font),
            scale_overlay_font_size=self._scale_overlay_font_size.value(),
            text_font_family=self._font_combo_family_value(self._text_font),
            text_font_size=self._text_size.value(),
            text_color=self._text_color.property("color_value") or self._initial_settings.text_color,
            text_size_space=self._text_size_space_combo.currentData(),
            text_anchor_alignment=self._text_anchor_combo.currentData(),
            overlay_line_color=self._overlay_line_color.property("color_value") or self._initial_settings.overlay_line_color,
            overlay_line_width=self._overlay_line_width.value(),
            focus_stack_profile=self._focus_stack_profile_combo.currentData(),
            focus_stack_sharpen_strength=self._focus_stack_sharpen_slider.value(),
            magic_segment_model_variant=self._magic_segment_model_variant_combo.currentData(),
            magic_segment_fill_draft_holes_enabled=self._magic_segment_fill_draft_holes_checkbox.isChecked(),
            magic_segment_standard_roi_enabled=self._magic_segment_standard_add_roi_checkbox.isChecked(),
            magic_segment_standard_add_roi_enabled=self._magic_segment_standard_add_roi_checkbox.isChecked(),
            magic_segment_standard_subtract_roi_enabled=self._magic_segment_standard_subtract_roi_checkbox.isChecked(),
            magic_segment_standard_subtract_input_mode=self._initial_settings.magic_segment_standard_subtract_input_mode,
            magic_segment_restrict_subtract_roi_to_primary_bounds=self._magic_segment_restrict_subtract_roi_checkbox.isChecked(),
            magic_segment_small_object_subtract_enhancement_enabled=self._magic_segment_small_object_enhancement_checkbox.isChecked(),
            magic_segment_small_object_roi_area_threshold_px=self._magic_segment_small_object_threshold_spin.value(),
            fiber_quick_roi_enabled=self._fiber_quick_roi_checkbox.isChecked(),
            fiber_quick_edge_trim_enabled=self._fiber_quick_edge_trim_checkbox.isChecked(),
            fiber_quick_line_extension_px=self._fiber_quick_line_extension_spin.value(),
            recent_export_dir=self._initial_settings.recent_export_dir,
            recent_project_dir=self._initial_settings.recent_project_dir,
            area_model_mappings=self.area_model_mappings(),
            area_weights_dir=self._area_weights_dir_edit.text().strip(),
            area_vendor_root=self._area_vendor_root_edit.text().strip(),
            area_worker_python=self._area_worker_python_edit.text().strip(),
            area_infer_device=str(self._area_infer_device_combo.currentData() or AreaInferDevice.CPU),
            calibration_presets=list(self._initial_settings.calibration_presets),
            selected_capture_device_id=self._initial_settings.selected_capture_device_id,
            raw_record_templates=self.raw_record_templates(),
            last_raw_record_template_path=self._initial_settings.last_raw_record_template_path,
            main_window_geometry=self._initial_settings.main_window_geometry,
            main_window_state=self._initial_settings.main_window_state,
            measurement_results_header_state=self._initial_settings.measurement_results_header_state,
            main_window_is_maximized=self._initial_settings.main_window_is_maximized,
            digital_slide_last_output_path=self._initial_settings.digital_slide_last_output_path,
            digital_slide_preview_max_width=int(self._digital_slide_preview_width_combo.currentData() or 0),
            digital_slide_capture_max_width=int(self._digital_slide_capture_width_combo.currentData() or 0),
            digital_slide_capture_tile_codec=normalize_tile_codec(self._digital_slide_capture_codec_combo.currentData()),
            digital_slide_capture_jpeg_quality=self._digital_slide_capture_quality_slider.value(),
            digital_slide_xy_soft_limit=self._digital_slide_xy_soft_limit_spin.value(),
            digital_slide_z_soft_limit=self._digital_slide_z_soft_limit_spin.value(),
            digital_slide_xy_jog_step=self._digital_slide_xy_jog_step_spin.value(),
            digital_slide_z_jog_step=self._digital_slide_z_jog_step_spin.value(),
            digital_slide_z_capture_lower=self._initial_settings.digital_slide_z_capture_lower,
            digital_slide_z_capture_upper=self._initial_settings.digital_slide_z_capture_upper,
            digital_slide_z_capture_step=self._initial_settings.digital_slide_z_capture_step,
            digital_slide_jog_rate=self._digital_slide_jog_rate_spin.value(),
            digital_slide_motor_output_enabled=self._digital_slide_motor_output_checkbox.isChecked(),
            digital_slide_x_stage_step=self._digital_slide_x_stage_step_spin.value(),
            digital_slide_y_stage_step=self._digital_slide_y_stage_step_spin.value(),
            digital_slide_reverse_x_axis=self._digital_slide_reverse_x_axis_checkbox.isChecked(),
            digital_slide_reverse_y_axis=self._digital_slide_reverse_y_axis_checkbox.isChecked(),
            digital_slide_overlap_percent=self._digital_slide_overlap_spin.value(),
            digital_slide_pixel_stride_mode=self._digital_slide_pixel_stride_mode_combo.currentData(),
            digital_slide_x_pixel_stride=self._digital_slide_x_pixel_stride_spin.value(),
            digital_slide_y_pixel_stride=self._digital_slide_y_pixel_stride_spin.value(),
            digital_slide_blend_width=self._digital_slide_blend_width_spin.value(),
            digital_slide_xy_settle_ms=self._digital_slide_xy_settle_spin.value(),
            digital_slide_xy_post_settle_ms=self._digital_slide_xy_post_settle_spin.value(),
            digital_slide_z_settle_ms=self._digital_slide_z_settle_spin.value(),
            digital_slide_z_post_settle_ms=self._digital_slide_z_post_settle_spin.value(),
            digital_slide_first_tile_extra_wait_ms=self._digital_slide_first_tile_extra_wait_spin.value(),
            digital_slide_discard_frames=self._digital_slide_discard_frames_spin.value(),
            digital_slide_focus_wheel_step=self._digital_slide_focus_wheel_slider.value(),
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

    def raw_record_templates(self) -> list[RawRecordTemplate]:
        self._store_raw_record_rules_from_table(self._raw_record_current_template_index)
        templates: list[RawRecordTemplate] = []
        seen_paths: set[str] = set()
        for row in range(self._raw_record_template_table.rowCount()):
            name_item = self._raw_record_template_table.item(row, 0)
            path_item = self._raw_record_template_table.item(row, 1)
            name = name_item.text().strip() if name_item is not None else ""
            path_token = path_item.text().strip() if path_item is not None else ""
            if not path_token or Path(path_token).suffix.lower() not in SUPPORTED_RAW_RECORD_TEMPLATE_SUFFIXES:
                continue
            normalized_path = to_resource_relative_path(path_token)
            key = normalized_path.casefold()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            rules = (
                list(self._raw_record_templates_data[row].rules)
                if row < len(self._raw_record_templates_data)
                else [RawRecordExportRule()]
            )
            templates.append(
                RawRecordTemplate(
                    name=name or Path(normalized_path).stem,
                    path=normalized_path,
                    rules=rules,
                ).normalized_copy()
            )
        return templates

    def wants_scale_anchor_pick(self) -> bool:
        return self._document is not None and self._request_scale_anchor_pick

    def _wrap_settings_page(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _build_general_tab(self, settings: AppSettings) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        display_group = QGroupBox("界面与默认视图")
        display_form = QFormLayout(display_group)
        self._theme_mode_combo = NoWheelComboBox()
        self._theme_mode_combo.addItem("跟随系统", AppThemeMode.SYSTEM)
        self._theme_mode_combo.addItem("深色", AppThemeMode.DARK)
        self._theme_mode_combo.addItem("浅色", AppThemeMode.LIGHT)
        self._theme_mode_combo.setCurrentIndex(
            max(0, self._theme_mode_combo.findData(settings.theme_mode))
        )
        self._open_view_mode_combo = NoWheelComboBox()
        self._open_view_mode_combo.addItem("缺省", OpenImageViewMode.DEFAULT)
        self._open_view_mode_combo.addItem("适合窗口", OpenImageViewMode.FIT)
        self._open_view_mode_combo.addItem("原始像素", OpenImageViewMode.ACTUAL)
        self._open_view_mode_combo.setCurrentIndex(
            max(0, self._open_view_mode_combo.findData(settings.open_image_view_mode))
        )
        display_form.addRow("界面主题", self._theme_mode_combo)
        display_form.addRow("打开图片默认视图", self._open_view_mode_combo)

        hint = QLabel(
            "这些选项是长期偏好；当前图片标定由工作台的标定区管理，类别颜色由左侧项目导航管理；"
            "比例尺锚点仅在“标注与比例尺”页显式触发选点。"
        )
        hint.setWordWrap(True)
        layout.addWidget(display_group)
        layout.addWidget(hint)
        layout.addStretch(1)
        return self._wrap_settings_page(page)

    def _update_focus_stack_sharpen_label(self, value: int) -> None:
        self._focus_stack_sharpen_value_label.setText(f"{value}%")

    def _update_digital_slide_focus_wheel_label(self, value: int) -> None:
        self._digital_slide_focus_wheel_value_label.setText(f"{value} 层/格")

    def _scale_overlay_length_unit(self) -> str:
        calibration = self._document.calibration if self._document is not None else None
        return calibration.unit if calibration is not None else "px"

    @staticmethod
    def _configure_font_combo(combo: QFontComboBox, requested_family: str) -> None:
        requested = str(requested_family or "").strip()
        available = {
            family.casefold(): family
            for family in QFontDatabase.families()
        }
        resolved_system = QFontInfo(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        ).family()
        fallback_candidates = (
            resolved_system,
            "Segoe UI",
            "PingFang SC",
            "Noto Sans CJK SC",
            "Noto Sans",
            "DejaVu Sans",
            "Arial",
            "Helvetica Neue",
        )
        fallback = next(
            (
                available[candidate.casefold()]
                for candidate in fallback_candidates
                if candidate.casefold() in available and not candidate.startswith(".")
            ),
            next(iter(available.values()), "Sans Serif"),
        )
        resolved = available.get(requested.casefold(), fallback) if requested else fallback
        combo.setCurrentFont(QFont(resolved))
        combo.setProperty("requested_font_family", requested or resolved)
        combo.setProperty("font_user_changed", False)
        combo.currentFontChanged.connect(
            lambda _font, widget=combo: widget.setProperty("font_user_changed", True)
        )

    @staticmethod
    def _font_combo_family_value(combo: QFontComboBox) -> str:
        requested = str(combo.property("requested_font_family") or "").strip()
        if requested and not bool(combo.property("font_user_changed")):
            return requested
        return combo.currentFont().family()

    def _update_length_measurement_style_preview(self, *_args) -> None:
        preview = getattr(self, "_length_measurement_style_preview", None)
        if preview is None:
            return
        font = QFont(self._length_measurement_label_font.currentFont())
        font.setPointSize(self._length_measurement_label_size.value())
        preview.set_preview_style(
            show_label=self._show_length_measurement_labels.isChecked(),
            font=font,
            label_color=str(
                self._length_measurement_label_color.property("color_value")
                or DEFAULT_MEASUREMENT_LABEL_COLOR
            ),
            line_color=str(self._default_measurement_color.property("color_value") or "#2A9D8F"),
            background_enabled=self._length_measurement_label_background.isChecked(),
            decimals=self._length_measurement_label_decimals.value(),
            endpoint_style=str(self._endpoint_style_combo.currentData() or MeasurementEndpointStyle.BAR),
        )

    def _update_area_measurement_style_preview(self, *_args) -> None:
        preview = getattr(self, "_area_measurement_style_preview", None)
        if preview is None:
            return
        font = QFont(self._area_measurement_label_font.currentFont())
        font.setPointSize(self._area_measurement_label_size.value())
        preview.set_preview_style(
            show_label=self._show_area_measurement_labels.isChecked(),
            font=font,
            label_color=str(
                self._area_measurement_label_color.property("color_value")
                or DEFAULT_MEASUREMENT_LABEL_COLOR
            ),
            line_color=str(self._default_measurement_color.property("color_value") or "#2A9D8F"),
            background_enabled=self._area_measurement_label_background.isChecked(),
            decimals=self._area_measurement_label_decimals.value(),
            endpoint_style=str(self._endpoint_style_combo.currentData() or MeasurementEndpointStyle.BAR),
        )

    def _update_measurement_style_preview(self, *_args) -> None:
        """Refresh both metric previews; retained for compatibility callers."""
        self._update_length_measurement_style_preview()
        self._update_area_measurement_style_preview()

    def _build_measurement_tab(self, settings: AppSettings) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        length_style = settings.length_measurement_label_style
        area_style = settings.area_measurement_label_style

        length_group = QGroupBox("直径/长度结果")
        length_form = QFormLayout(length_group)
        self._show_length_measurement_labels = QCheckBox("在线段和折线旁显示结果文字")
        self._show_length_measurement_labels.setChecked(length_style.enabled)
        self._length_measurement_label_font = NoWheelFontComboBox()
        self._configure_font_combo(
            self._length_measurement_label_font,
            length_style.font_family,
        )
        self._length_measurement_label_size = NoWheelSpinBox()
        self._length_measurement_label_size.setRange(8, 96)
        self._length_measurement_label_size.setValue(length_style.font_size)
        self._length_measurement_label_color = self._create_color_button(length_style.color)
        self._length_measurement_label_decimals = NoWheelSpinBox()
        self._length_measurement_label_decimals.setRange(0, 8)
        self._length_measurement_label_decimals.setValue(length_style.decimals)
        self._length_measurement_label_parallel = QCheckBox("结果文字与测量线平行")
        self._length_measurement_label_parallel.setChecked(length_style.parallel_to_line)
        self._length_measurement_label_background = QCheckBox("显示结果文字浅黑底")
        self._length_measurement_label_background.setChecked(length_style.background_enabled)
        self._length_measurement_style_preview = _MeasurementStylePreview(
            length_group,
            metric="length",
        )
        length_form.addRow("", self._show_length_measurement_labels)
        length_form.addRow("结果文字字体", self._length_measurement_label_font)
        length_form.addRow("结果文字字号", self._length_measurement_label_size)
        length_form.addRow("结果文字颜色", self._length_measurement_label_color)
        length_form.addRow("结果文字小数位", self._length_measurement_label_decimals)
        length_form.addRow("", self._length_measurement_label_parallel)
        length_form.addRow("", self._length_measurement_label_background)
        length_form.addRow("预览", self._length_measurement_style_preview)

        area_group = QGroupBox("面积结果")
        area_form = QFormLayout(area_group)
        self._show_area_measurement_labels = QCheckBox("在面积对象旁显示结果文字")
        self._show_area_measurement_labels.setChecked(area_style.enabled)
        self._area_measurement_label_font = NoWheelFontComboBox()
        self._configure_font_combo(
            self._area_measurement_label_font,
            area_style.font_family,
        )
        self._area_measurement_label_size = NoWheelSpinBox()
        self._area_measurement_label_size.setRange(8, 96)
        self._area_measurement_label_size.setValue(area_style.font_size)
        self._area_measurement_label_color = self._create_color_button(area_style.color)
        self._area_measurement_label_decimals = NoWheelSpinBox()
        self._area_measurement_label_decimals.setRange(0, 8)
        self._area_measurement_label_decimals.setValue(area_style.decimals)
        self._area_measurement_label_background = QCheckBox("显示结果文字浅黑底")
        self._area_measurement_label_background.setChecked(area_style.background_enabled)
        self._area_measurement_style_preview = _MeasurementStylePreview(
            area_group,
            metric="area",
        )
        area_form.addRow("", self._show_area_measurement_labels)
        area_form.addRow("结果文字字体", self._area_measurement_label_font)
        area_form.addRow("结果文字字号", self._area_measurement_label_size)
        area_form.addRow("结果文字颜色", self._area_measurement_label_color)
        area_form.addRow("结果文字小数位", self._area_measurement_label_decimals)
        area_form.addRow("", self._area_measurement_label_background)
        area_form.addRow("预览", self._area_measurement_style_preview)

        # Private aliases preserve existing integrations while directing them
        # to the length-specific controls.
        self._show_measurement_labels = self._show_length_measurement_labels
        self._measurement_label_font = self._length_measurement_label_font
        self._measurement_label_size = self._length_measurement_label_size
        self._measurement_label_color = self._length_measurement_label_color
        self._measurement_label_decimals = self._length_measurement_label_decimals
        self._measurement_label_parallel = self._length_measurement_label_parallel
        self._measurement_label_background = self._length_measurement_label_background
        self._measurement_style_preview = self._length_measurement_style_preview

        self._show_count_numbers = QCheckBox("显示计数点编号")
        self._show_count_numbers.setChecked(settings.show_count_numbers)
        self._count_number_font = NoWheelFontComboBox()
        self._configure_font_combo(self._count_number_font, settings.count_number_font_family)
        self._count_number_size = NoWheelSpinBox()
        self._count_number_size.setRange(8, 96)
        self._count_number_size.setValue(settings.count_number_font_size)
        self._count_number_color = self._create_color_button(settings.count_number_color)
        self._endpoint_style_combo = NoWheelComboBox()
        self._endpoint_style_combo.addItem("圆点", MeasurementEndpointStyle.CIRCLE)
        self._endpoint_style_combo.addItem("内侧箭头", MeasurementEndpointStyle.ARROW_INSIDE)
        self._endpoint_style_combo.addItem("外侧箭头", MeasurementEndpointStyle.ARROW_OUTSIDE)
        self._endpoint_style_combo.addItem("竖线", MeasurementEndpointStyle.BAR)
        self._endpoint_style_combo.addItem("无端点", MeasurementEndpointStyle.NONE)
        self._endpoint_style_combo.setCurrentIndex(max(0, self._endpoint_style_combo.findData(settings.measurement_endpoint_style)))
        self._default_measurement_color = self._create_color_button(settings.default_measurement_color)
        count_group = QGroupBox("计数点编号")
        count_form = QFormLayout(count_group)
        count_form.addRow("", self._show_count_numbers)
        count_form.addRow("编号字体", self._count_number_font)
        count_form.addRow("编号字号", self._count_number_size)
        count_form.addRow("编号颜色", self._count_number_color)

        measurement_group = QGroupBox("测量线与端点")
        measurement_form = QFormLayout(measurement_group)
        measurement_form.addRow("端点样式", self._endpoint_style_combo)
        measurement_form.addRow("未分类测量线颜色", self._default_measurement_color)

        self._show_length_measurement_labels.toggled.connect(self._update_length_measurement_style_preview)
        self._length_measurement_label_font.currentFontChanged.connect(self._update_length_measurement_style_preview)
        self._length_measurement_label_size.valueChanged.connect(self._update_length_measurement_style_preview)
        self._length_measurement_label_color.clicked.connect(self._update_length_measurement_style_preview)
        self._length_measurement_label_decimals.valueChanged.connect(self._update_length_measurement_style_preview)
        self._length_measurement_label_background.toggled.connect(self._update_length_measurement_style_preview)
        self._show_area_measurement_labels.toggled.connect(self._update_area_measurement_style_preview)
        self._area_measurement_label_font.currentFontChanged.connect(self._update_area_measurement_style_preview)
        self._area_measurement_label_size.valueChanged.connect(self._update_area_measurement_style_preview)
        self._area_measurement_label_color.clicked.connect(self._update_area_measurement_style_preview)
        self._area_measurement_label_decimals.valueChanged.connect(self._update_area_measurement_style_preview)
        self._area_measurement_label_background.toggled.connect(self._update_area_measurement_style_preview)
        self._endpoint_style_combo.currentIndexChanged.connect(self._update_measurement_style_preview)
        self._default_measurement_color.clicked.connect(self._update_measurement_style_preview)
        self._update_measurement_style_preview()

        layout.addWidget(length_group)
        layout.addWidget(area_group)
        layout.addWidget(count_group)
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
        self._magic_segment_model_variant_combo = NoWheelComboBox()
        self._magic_segment_model_variant_combo.addItem("标准 (EdgeSAM)", MagicSegmentModelVariant.EDGE_SAM)
        self._magic_segment_model_variant_combo.addItem("高精度 (EdgeSAM-3x)", MagicSegmentModelVariant.EDGE_SAM_3X)
        self._magic_segment_model_variant_combo.setCurrentIndex(
            max(0, self._magic_segment_model_variant_combo.findData(settings.magic_segment_model_variant))
        )
        self._magic_segment_fill_draft_holes_checkbox = QCheckBox("草稿阶段自动填充内部孔洞")
        self._magic_segment_fill_draft_holes_checkbox.setChecked(settings.magic_segment_fill_draft_holes_enabled)
        self._magic_segment_standard_add_roi_checkbox = QCheckBox("标准魔棒添加模式默认启用 ROI")
        self._magic_segment_standard_add_roi_checkbox.setChecked(settings.magic_segment_standard_add_roi_enabled)
        self._magic_segment_standard_subtract_roi_checkbox = QCheckBox("标准魔棒剔除模式默认启用 ROI")
        self._magic_segment_standard_subtract_roi_checkbox.setChecked(settings.magic_segment_standard_subtract_roi_enabled)
        self._magic_segment_restrict_subtract_roi_checkbox = QCheckBox("剔除模式 ROI 限制在主体范围内")
        self._magic_segment_restrict_subtract_roi_checkbox.setChecked(settings.magic_segment_restrict_subtract_roi_to_primary_bounds)
        self._magic_segment_small_object_enhancement_checkbox = QCheckBox("剔除小目标增强")
        self._magic_segment_small_object_enhancement_checkbox.setChecked(settings.magic_segment_small_object_subtract_enhancement_enabled)
        self._magic_segment_small_object_threshold_spin = NoWheelSpinBox()
        self._magic_segment_small_object_threshold_spin.setRange(4096, 4_000_000)
        self._magic_segment_small_object_threshold_spin.setSingleStep(10000)
        self._magic_segment_small_object_threshold_spin.setValue(settings.magic_segment_small_object_roi_area_threshold_px)
        self._magic_segment_small_object_threshold_spin.setSuffix(" px^2")
        self._fiber_quick_roi_checkbox = QCheckBox("快速测径默认启用 ROI")
        self._fiber_quick_roi_checkbox.setChecked(settings.fiber_quick_roi_enabled)
        self._fiber_quick_edge_trim_checkbox = QCheckBox("快速测径启用边缘剔除")
        self._fiber_quick_edge_trim_checkbox.setChecked(settings.fiber_quick_edge_trim_enabled)
        self._fiber_quick_line_extension_spin = NoWheelDoubleSpinBox()
        self._fiber_quick_line_extension_spin.setDecimals(1)
        self._fiber_quick_line_extension_spin.setRange(-20.0, 20.0)
        self._fiber_quick_line_extension_spin.setSingleStep(0.5)
        self._fiber_quick_line_extension_spin.setValue(settings.fiber_quick_line_extension_px)
        self._fiber_quick_line_extension_spin.setSuffix(" px")
        magic_hint = QLabel("标准魔棒与同类扩选都会复用这里的 EdgeSAM / EdgeSAM-3x 设置；若缺失高精度模型文件，运行时会自动回退到标准模型。")
        magic_hint.setWordWrap(True)
        fill_holes_hint = QLabel("开启后，标准魔棒的主体与剔除形状草稿都会先填充内部孔洞；同类扩选不受此开关影响。")
        fill_holes_hint.setWordWrap(True)
        roi_hint = QLabel("ROI 开关会同时出现在标准魔棒与快速测径右侧工具区，快捷键为 Y。快速测径在 ROI 失败时仍会自动回退到整图分割。")
        roi_hint.setWordWrap(True)
        small_object_hint = QLabel("剔除模式启用 ROI 且限制在主体内时，小 ROI 会进入局部上采样增强工作区，便于处理低分辨率下的细小剔除目标。")
        small_object_hint.setWordWrap(True)
        quick_hint = QLabel("快速测径确认后会在后台异步生成线段；边缘剔除只影响快速测径，不影响标准魔棒与同类扩选。")
        quick_hint.setWordWrap(True)
        magic_segment_form.addRow("标准模型", self._magic_segment_model_variant_combo)
        magic_segment_form.addRow("", self._magic_segment_fill_draft_holes_checkbox)
        magic_segment_form.addRow("", self._magic_segment_standard_add_roi_checkbox)
        magic_segment_form.addRow("", self._magic_segment_standard_subtract_roi_checkbox)
        magic_segment_form.addRow("", self._magic_segment_restrict_subtract_roi_checkbox)
        magic_segment_form.addRow("", self._magic_segment_small_object_enhancement_checkbox)
        magic_segment_form.addRow("小目标 ROI 阈值", self._magic_segment_small_object_threshold_spin)
        magic_segment_form.addRow("", self._fiber_quick_roi_checkbox)
        magic_segment_form.addRow("", self._fiber_quick_edge_trim_checkbox)
        magic_segment_form.addRow("快速测径扩展像素", self._fiber_quick_line_extension_spin)
        magic_segment_form.addRow("", fill_holes_hint)
        magic_segment_form.addRow("", roi_hint)
        magic_segment_form.addRow("", small_object_hint)
        magic_segment_form.addRow("", quick_hint)
        magic_segment_form.addRow("", magic_hint)

        layout.addWidget(focus_stack_group)
        layout.addWidget(magic_segment_group)
        layout.addStretch(1)
        return self._wrap_settings_page(page)

    def _build_scale_overlay_tab(self, settings: AppSettings) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        placement_group = QGroupBox("比例尺位置与长度")
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

        style_group = QGroupBox("比例尺样式")
        style_form = QFormLayout(style_group)
        self._scale_overlay_style_combo = NoWheelComboBox()
        self._scale_overlay_style_combo.addItem("纯线", ScaleOverlayStyle.LINE)
        self._scale_overlay_style_combo.addItem("端点刻度", ScaleOverlayStyle.TICKS)
        self._scale_overlay_style_combo.addItem("粗条", ScaleOverlayStyle.BAR)
        self._scale_overlay_style_combo.setCurrentIndex(max(0, self._scale_overlay_style_combo.findData(settings.scale_overlay_style)))
        self._scale_overlay_color = self._create_color_button(settings.scale_overlay_color)
        self._scale_overlay_font = NoWheelFontComboBox()
        self._configure_font_combo(self._scale_overlay_font, settings.scale_overlay_font_family)
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

        text_group = QGroupBox("文字标注默认样式")
        text_form = QFormLayout(text_group)
        self._text_font = NoWheelFontComboBox()
        self._configure_font_combo(self._text_font, settings.text_font_family)
        self._text_size = NoWheelSpinBox()
        self._text_size.setRange(8, 144)
        self._text_size.setValue(settings.text_font_size)
        self._text_color = self._create_color_button(settings.text_color)
        self._text_size_space_combo = NoWheelComboBox()
        self._text_size_space_combo.addItem(
            "随图像缩放（推荐）",
            OverlayTextSizeSpace.IMAGE_PX,
        )
        self._text_size_space_combo.addItem(
            "固定输出像素",
            OverlayTextSizeSpace.LEGACY_OUTPUT_PX,
        )
        self._text_size_space_combo.setCurrentIndex(
            max(
                0,
                self._text_size_space_combo.findData(settings.text_size_space),
            )
        )
        self._text_anchor_combo = NoWheelComboBox()
        for label, value in OVERLAY_TEXT_ANCHOR_ITEMS:
            self._text_anchor_combo.addItem(label, value)
        self._text_anchor_combo.setCurrentIndex(
            max(
                0,
                self._text_anchor_combo.findData(
                    settings.text_anchor_alignment
                ),
            )
        )
        text_form.addRow("文字字体", self._text_font)
        text_form.addRow("新建时屏显字号", self._text_size)
        text_form.addRow("文字颜色", self._text_color)
        text_form.addRow("尺寸基准", self._text_size_space_combo)
        text_form.addRow("默认锚点", self._text_anchor_combo)
        text_hint = QLabel(
            "“随图像缩放”会在创建时按当前缩放率换算并冻结原图字号，"
            "使画布、完整分辨率和其它图片导出模式保持同一相对大小。"
        )
        text_hint.setWordWrap(True)
        text_form.addRow("", text_hint)

        shape_group = QGroupBox("图形标注默认样式")
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

        layout.addWidget(placement_group)
        layout.addWidget(style_group)
        layout.addWidget(display_hint)
        layout.addWidget(self._build_current_scale_anchor_group())
        layout.addWidget(text_group)
        layout.addWidget(shape_group)

        layout.addStretch(1)
        return self._wrap_settings_page(page)

    def _build_current_scale_anchor_group(self) -> QGroupBox:
        group = QGroupBox("当前图片比例尺位置")
        group_layout = QVBoxLayout(group)
        document = self._document
        if document is None:
            status = QLabel("当前没有打开的图片，无法设置手动比例尺位置。", group)
        elif document.scale_overlay_anchor is None:
            status = QLabel("当前图片尚未设置手动位置。", group)
        else:
            anchor = document.scale_overlay_anchor
            status = QLabel(f"当前锚点：({anchor.x:.1f}, {anchor.y:.1f})", group)
        self._scale_anchor_status_label = status
        status.setWordWrap(True)
        group_layout.addWidget(status)
        hint = QLabel(
            "只有点击下方按钮才会关闭首选项并进入画布选点；修改其它比例尺设置不会触发选点。",
            group,
        )
        hint.setWordWrap(True)
        group_layout.addWidget(hint)
        self._scale_anchor_pick_button = QPushButton("在画布重新选择位置", group)
        self._scale_anchor_pick_button.setEnabled(document is not None)
        self._scale_anchor_pick_button.clicked.connect(self._trigger_scale_anchor_pick)
        group_layout.addWidget(self._scale_anchor_pick_button)
        return group

    def _build_digital_slide_tab(self, settings: AppSettings, *, locked: bool = False) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        if locked:
            locked_hint = QLabel("数字化切片正在采集中，本页参数已锁定；本次采集会继续使用开始时的参数快照。")
            locked_hint.setWordWrap(True)
            locked_hint.setStyleSheet("font-weight: 700; color: #B45309;")
            layout.addWidget(locked_hint)

        capture_group = QGroupBox("采集与预览")
        capture_form = QFormLayout(capture_group)
        self._digital_slide_preview_width_combo = NoWheelComboBox()
        self._add_digital_slide_width_options(
            self._digital_slide_preview_width_combo,
            current=settings.digital_slide_preview_max_width,
            options=(960, 1280, 1600, 2400),
        )
        self._digital_slide_capture_width_combo = NoWheelComboBox()
        self._add_digital_slide_width_options(
            self._digital_slide_capture_width_combo,
            current=settings.digital_slide_capture_max_width,
            options=(1600, 2400, 3200),
        )
        self._digital_slide_capture_codec_combo = NoWheelComboBox()
        self._digital_slide_capture_codec_combo.addItem("PNG 无损", DIGITAL_SLIDE_TILE_CODEC_PNG)
        self._digital_slide_capture_codec_combo.addItem("JPEG 压缩", DIGITAL_SLIDE_TILE_CODEC_JPEG)
        codec_index = self._digital_slide_capture_codec_combo.findData(normalize_tile_codec(settings.digital_slide_capture_tile_codec))
        self._digital_slide_capture_codec_combo.setCurrentIndex(codec_index if codec_index >= 0 else 0)
        quality_row = QWidget()
        quality_layout = QHBoxLayout(quality_row)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        self._digital_slide_capture_quality_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self._digital_slide_capture_quality_slider.setRange(70, 95)
        self._digital_slide_capture_quality_slider.setValue(normalize_jpeg_quality(settings.digital_slide_capture_jpeg_quality))
        self._digital_slide_capture_quality_label = QLabel()
        self._digital_slide_capture_quality_label.setMinimumWidth(150)
        self._digital_slide_capture_quality_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._digital_slide_capture_quality_slider.valueChanged.connect(self._update_digital_slide_capture_quality_label)
        self._digital_slide_capture_codec_combo.currentIndexChanged.connect(self._sync_digital_slide_capture_quality_visibility)
        quality_layout.addWidget(self._digital_slide_capture_quality_slider, 1)
        quality_layout.addWidget(self._digital_slide_capture_quality_label)
        self._digital_slide_overlap_spin = NoWheelSpinBox()
        self._digital_slide_overlap_spin.setRange(0, 90)
        self._digital_slide_overlap_spin.setSuffix(" %")
        self._digital_slide_overlap_spin.setValue(settings.digital_slide_overlap_percent)
        self._digital_slide_blend_width_spin = NoWheelSpinBox()
        self._digital_slide_blend_width_spin.setRange(0, 10000)
        self._digital_slide_blend_width_spin.setSuffix(" px")
        self._digital_slide_blend_width_spin.setValue(settings.digital_slide_blend_width)
        capture_form.addRow("预览最大宽度", self._digital_slide_preview_width_combo)
        capture_form.addRow("采集最大宽度", self._digital_slide_capture_width_combo)
        capture_form.addRow("默认存储格式", self._digital_slide_capture_codec_combo)
        capture_form.addRow("JPEG 质量", quality_row)
        capture_form.addRow("视场重叠", self._digital_slide_overlap_spin)
        capture_form.addRow("重叠融合宽度", self._digital_slide_blend_width_spin)
        self._sync_digital_slide_capture_quality_visibility()

        motion_group = QGroupBox("运动控制")
        motion_form = QFormLayout(motion_group)
        self._digital_slide_xy_soft_limit_spin = NoWheelSpinBox()
        self._digital_slide_xy_soft_limit_spin.setRange(0, 10_000_000)
        self._digital_slide_xy_soft_limit_spin.setSingleStep(10_000)
        self._digital_slide_xy_soft_limit_spin.setSuffix(" steps")
        self._digital_slide_xy_soft_limit_spin.setValue(settings.digital_slide_xy_soft_limit)
        self._digital_slide_z_soft_limit_spin = NoWheelSpinBox()
        self._digital_slide_z_soft_limit_spin.setRange(0, 10_000_000)
        self._digital_slide_z_soft_limit_spin.setSingleStep(5000)
        self._digital_slide_z_soft_limit_spin.setSuffix(" steps")
        self._digital_slide_z_soft_limit_spin.setValue(settings.digital_slide_z_soft_limit)
        self._digital_slide_xy_jog_step_spin = NoWheelSpinBox()
        self._digital_slide_xy_jog_step_spin.setRange(1, 1_000_000)
        self._digital_slide_xy_jog_step_spin.setSingleStep(100)
        self._digital_slide_xy_jog_step_spin.setSuffix(" steps")
        self._digital_slide_xy_jog_step_spin.setValue(settings.digital_slide_xy_jog_step)
        self._digital_slide_z_jog_step_spin = NoWheelSpinBox()
        self._digital_slide_z_jog_step_spin.setRange(1, 1_000_000)
        self._digital_slide_z_jog_step_spin.setSingleStep(100)
        self._digital_slide_z_jog_step_spin.setSuffix(" steps")
        self._digital_slide_z_jog_step_spin.setValue(settings.digital_slide_z_jog_step)
        self._digital_slide_jog_rate_spin = NoWheelSpinBox()
        self._digital_slide_jog_rate_spin.setRange(1, 50)
        self._digital_slide_jog_rate_spin.setSuffix(" 次/秒")
        self._digital_slide_jog_rate_spin.setValue(settings.digital_slide_jog_rate)
        self._digital_slide_motor_output_checkbox = QCheckBox("进入数字化切片界面后自动启用电机输出")
        self._digital_slide_motor_output_checkbox.setChecked(settings.digital_slide_motor_output_enabled)
        self._digital_slide_reverse_x_axis_checkbox = QCheckBox("左右方向反转")
        self._digital_slide_reverse_x_axis_checkbox.setChecked(settings.digital_slide_reverse_x_axis)
        self._digital_slide_reverse_y_axis_checkbox = QCheckBox("上下方向反转")
        self._digital_slide_reverse_y_axis_checkbox.setChecked(settings.digital_slide_reverse_y_axis)
        motion_form.addRow("XY 软限位", self._digital_slide_xy_soft_limit_spin)
        motion_form.addRow("Z 软限位", self._digital_slide_z_soft_limit_spin)
        motion_form.addRow("XY 步距", self._digital_slide_xy_jog_step_spin)
        motion_form.addRow("对焦步距", self._digital_slide_z_jog_step_spin)
        motion_form.addRow("长按速度", self._digital_slide_jog_rate_spin)
        motion_form.addRow("坐标方向", self._digital_slide_reverse_x_axis_checkbox)
        motion_form.addRow("", self._digital_slide_reverse_y_axis_checkbox)
        motion_form.addRow("", self._digital_slide_motor_output_checkbox)

        advanced_group = QGroupBox("高级采集")
        advanced_form = QFormLayout(advanced_group)
        self._digital_slide_x_stage_step_spin = NoWheelSpinBox()
        self._digital_slide_x_stage_step_spin.setRange(-10_000_000, 10_000_000)
        self._digital_slide_x_stage_step_spin.setSingleStep(100)
        self._digital_slide_x_stage_step_spin.setSuffix(" steps")
        self._digital_slide_x_stage_step_spin.setValue(settings.digital_slide_x_stage_step)
        self._digital_slide_y_stage_step_spin = NoWheelSpinBox()
        self._digital_slide_y_stage_step_spin.setRange(-10_000_000, 10_000_000)
        self._digital_slide_y_stage_step_spin.setSingleStep(100)
        self._digital_slide_y_stage_step_spin.setSuffix(" steps")
        self._digital_slide_y_stage_step_spin.setValue(settings.digital_slide_y_stage_step)
        self._digital_slide_pixel_stride_mode_combo = NoWheelComboBox()
        self._digital_slide_pixel_stride_mode_combo.addItem("按视场重叠自动", "auto_overlap")
        self._digital_slide_pixel_stride_mode_combo.addItem("手动像素步距", "manual_pixels")
        self._digital_slide_pixel_stride_mode_combo.setCurrentIndex(
            max(0, self._digital_slide_pixel_stride_mode_combo.findData(settings.digital_slide_pixel_stride_mode))
        )
        self._digital_slide_x_pixel_stride_spin = NoWheelSpinBox()
        self._digital_slide_x_pixel_stride_spin.setRange(1, 100_000)
        self._digital_slide_x_pixel_stride_spin.setSuffix(" px")
        self._digital_slide_x_pixel_stride_spin.setValue(settings.digital_slide_x_pixel_stride)
        self._digital_slide_y_pixel_stride_spin = NoWheelSpinBox()
        self._digital_slide_y_pixel_stride_spin.setRange(1, 100_000)
        self._digital_slide_y_pixel_stride_spin.setSuffix(" px")
        self._digital_slide_y_pixel_stride_spin.setValue(settings.digital_slide_y_pixel_stride)
        self._digital_slide_xy_settle_spin = NoWheelSpinBox()
        self._digital_slide_xy_settle_spin.setRange(0, 10_000)
        self._digital_slide_xy_settle_spin.setSuffix(" ms")
        self._digital_slide_xy_settle_spin.setValue(settings.digital_slide_xy_settle_ms)
        self._digital_slide_xy_post_settle_spin = NoWheelSpinBox()
        self._digital_slide_xy_post_settle_spin.setRange(0, 5000)
        self._digital_slide_xy_post_settle_spin.setSuffix(" ms")
        self._digital_slide_xy_post_settle_spin.setValue(settings.digital_slide_xy_post_settle_ms)
        self._digital_slide_z_settle_spin = NoWheelSpinBox()
        self._digital_slide_z_settle_spin.setRange(0, 10_000)
        self._digital_slide_z_settle_spin.setSuffix(" ms")
        self._digital_slide_z_settle_spin.setValue(settings.digital_slide_z_settle_ms)
        self._digital_slide_z_post_settle_spin = NoWheelSpinBox()
        self._digital_slide_z_post_settle_spin.setRange(0, 5000)
        self._digital_slide_z_post_settle_spin.setSuffix(" ms")
        self._digital_slide_z_post_settle_spin.setValue(settings.digital_slide_z_post_settle_ms)
        self._digital_slide_first_tile_extra_wait_spin = NoWheelSpinBox()
        self._digital_slide_first_tile_extra_wait_spin.setRange(0, 60_000)
        self._digital_slide_first_tile_extra_wait_spin.setSingleStep(500)
        self._digital_slide_first_tile_extra_wait_spin.setSuffix(" ms")
        self._digital_slide_first_tile_extra_wait_spin.setValue(settings.digital_slide_first_tile_extra_wait_ms)
        self._digital_slide_discard_frames_spin = NoWheelSpinBox()
        self._digital_slide_discard_frames_spin.setRange(0, 20)
        self._digital_slide_discard_frames_spin.setSuffix(" 帧")
        self._digital_slide_discard_frames_spin.setValue(settings.digital_slide_discard_frames)
        advanced_form.addRow("X 自动采集步距", self._digital_slide_x_stage_step_spin)
        advanced_form.addRow("Y 自动采集步距", self._digital_slide_y_stage_step_spin)
        advanced_form.addRow("像素步距模式", self._digital_slide_pixel_stride_mode_combo)
        advanced_form.addRow("X 像素步距", self._digital_slide_x_pixel_stride_spin)
        advanced_form.addRow("Y 像素步距", self._digital_slide_y_pixel_stride_spin)
        advanced_form.addRow("XY 停稳等待", self._digital_slide_xy_settle_spin)
        advanced_form.addRow("XY 停稳后等待", self._digital_slide_xy_post_settle_spin)
        advanced_form.addRow("Z 停稳等待", self._digital_slide_z_settle_spin)
        advanced_form.addRow("Z 停稳后等待", self._digital_slide_z_post_settle_spin)
        advanced_form.addRow("首张额外等待", self._digital_slide_first_tile_extra_wait_spin)
        advanced_form.addRow("丢弃帧数", self._digital_slide_discard_frames_spin)

        browsing_group = QGroupBox("浏览与快捷键")
        browsing_form = QFormLayout(browsing_group)
        wheel_row = QWidget()
        wheel_layout = QHBoxLayout(wheel_row)
        wheel_layout.setContentsMargins(0, 0, 0, 0)
        self._digital_slide_focus_wheel_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self._digital_slide_focus_wheel_slider.setRange(1, 10)
        self._digital_slide_focus_wheel_slider.setValue(settings.digital_slide_focus_wheel_step)
        self._digital_slide_focus_wheel_value_label = QLabel()
        self._digital_slide_focus_wheel_value_label.setMinimumWidth(70)
        self._digital_slide_focus_wheel_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._digital_slide_focus_wheel_slider.valueChanged.connect(self._update_digital_slide_focus_wheel_label)
        self._update_digital_slide_focus_wheel_label(self._digital_slide_focus_wheel_slider.value())
        wheel_layout.addWidget(self._digital_slide_focus_wheel_slider, 1)
        wheel_layout.addWidget(self._digital_slide_focus_wheel_value_label)
        shortcuts = QLabel("M 切换步进/平滑移动；方向键移动视场；Shift+方向键按整视场移动；Ctrl+滚轮缩放；普通滚轮切换焦层。")
        shortcuts.setWordWrap(True)
        browsing_form.addRow("焦层滚轮速度", wheel_row)
        browsing_form.addRow("快捷键", shortcuts)

        layout.addWidget(capture_group)
        layout.addWidget(motion_group)
        layout.addWidget(advanced_group)
        layout.addWidget(browsing_group)
        layout.addStretch(1)
        for group in (capture_group, motion_group, advanced_group, browsing_group):
            group.setEnabled(not locked)
        return self._wrap_settings_page(page)

    def _add_digital_slide_width_options(self, combo: QComboBox, *, current: int, options: tuple[int, ...]) -> None:
        for width in options:
            combo.addItem(f"{width} px", int(width))
        combo.addItem("原始尺寸", 0)
        index = combo.findData(int(current))
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _digital_slide_quality_label_text(self, value: int) -> str:
        return _digital_slide_quality_label_text(value)

    def _update_digital_slide_capture_quality_label(self, value: int) -> None:
        self._digital_slide_capture_quality_label.setText(self._digital_slide_quality_label_text(value))

    def _sync_digital_slide_capture_quality_visibility(self) -> None:
        is_jpeg = normalize_tile_codec(self._digital_slide_capture_codec_combo.currentData()) == DIGITAL_SLIDE_TILE_CODEC_JPEG
        self._digital_slide_capture_quality_slider.setEnabled(is_jpeg)
        self._digital_slide_capture_quality_label.setEnabled(is_jpeg)
        self._update_digital_slide_capture_quality_label(self._digital_slide_capture_quality_slider.value())

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
        self._area_infer_device_combo = NoWheelComboBox()
        self._area_infer_device_combo.addItem("CPU（默认，兼容性最佳）", AreaInferDevice.CPU)
        self._area_infer_device_combo.addItem("自动选择 CPU / CUDA", AreaInferDevice.AUTO)
        self._area_infer_device_combo.addItem("CUDA 0", AreaInferDevice.CUDA_0)
        device_index = self._area_infer_device_combo.findData(settings.area_infer_device)
        self._area_infer_device_combo.setCurrentIndex(max(0, device_index))
        area_form = QFormLayout()
        area_form.addRow("权重目录", self._with_browse_button(self._area_weights_dir_edit, directory=True, resource_relative=True))
        area_form.addRow("YOLACT vendor 目录", self._with_browse_button(self._area_vendor_root_edit, directory=True, resource_relative=True))
        area_form.addRow("Worker 可执行文件 / Python", self._with_browse_button(self._area_worker_python_edit, directory=False, resource_relative=False))
        area_form.addRow("推理设备", self._area_infer_device_combo)
        area_layout.addLayout(area_form)
        path_hint = QLabel("权重和 vendor 支持相对运行时资源目录填写；Worker 支持相对程序目录填写。保持 Worker 为空时，会自动选择打包后的 FiberAreaWorker 或当前 Python。")
        path_hint.setWordWrap(True)
        area_layout.addWidget(path_hint)

        layout.addWidget(area_group)
        layout.addStretch(1)
        return self._wrap_settings_page(page)

    def _build_raw_record_templates_tab(self, settings: AppSettings) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        template_group = QGroupBox("原始记录模板")
        template_layout = QVBoxLayout(template_group)
        template_hint = QLabel("模板文件建议放在 runtime/content-templates 下，并使用 .xlsx/.xlsm/.xltx/.xltm 格式。旧版 .xls/.xlt 请先用 Excel 另存为新格式。")
        template_hint.setWordWrap(True)
        template_layout.addWidget(template_hint)

        self._raw_record_template_table = QTableWidget(0, 2)
        self._raw_record_template_table.setHorizontalHeaderLabels(["名称", "模板文件"])
        self._raw_record_template_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._raw_record_template_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._raw_record_template_table.verticalHeader().setVisible(False)
        self._raw_record_template_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._raw_record_template_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for template in self._raw_record_templates_data:
            self._insert_raw_record_template_row(template)
        self._raw_record_template_table.currentCellChanged.connect(self._on_raw_record_template_selection_changed)
        template_layout.addWidget(self._raw_record_template_table)

        template_buttons = QHBoxLayout()
        add_template_button = QPushButton("新增模板")
        add_template_button.clicked.connect(self._add_raw_record_template)
        remove_template_button = QPushButton("删除选中模板")
        remove_template_button.clicked.connect(self._remove_selected_raw_record_template)
        browse_template_button = QPushButton("更换模板文件")
        browse_template_button.clicked.connect(self._browse_selected_raw_record_template)
        template_buttons.addWidget(add_template_button)
        template_buttons.addWidget(remove_template_button)
        template_buttons.addWidget(browse_template_button)
        template_buttons.addStretch(1)
        template_layout.addLayout(template_buttons)

        rules_group = QGroupBox("导出规则")
        rules_layout = QVBoxLayout(rules_group)
        self._raw_record_rule_table = QTableWidget(0, 7)
        self._raw_record_rule_table.setHorizontalHeaderLabels(["数据", "字段", "筛选", "工作表", "起始单元格", "结束单元格", "方向"])
        self._raw_record_rule_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._raw_record_rule_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._raw_record_rule_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._raw_record_rule_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._raw_record_rule_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._raw_record_rule_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self._raw_record_rule_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self._raw_record_rule_table.verticalHeader().setVisible(False)
        self._raw_record_rule_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._raw_record_rule_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        rules_layout.addWidget(self._raw_record_rule_table)

        rule_buttons = QHBoxLayout()
        self._raw_record_add_rule_button = QPushButton("新增规则")
        self._raw_record_add_rule_button.clicked.connect(self._add_raw_record_rule)
        self._raw_record_remove_rule_button = QPushButton("删除选中规则")
        self._raw_record_remove_rule_button.clicked.connect(self._remove_selected_raw_record_rule)
        rule_buttons.addWidget(self._raw_record_add_rule_button)
        rule_buttons.addWidget(self._raw_record_remove_rule_button)
        rule_buttons.addStretch(1)
        rules_layout.addLayout(rule_buttons)

        layout.addWidget(template_group)
        layout.addWidget(rules_group)
        layout.addStretch(1)
        if self._raw_record_template_table.rowCount() > 0:
            self._raw_record_template_table.setCurrentCell(0, 0)
            self._load_raw_record_rules_into_table(0)
        else:
            self._load_raw_record_rules_into_table(-1)
        return self._wrap_settings_page(page)

    def _insert_raw_record_template_row(self, template: RawRecordTemplate) -> None:
        row = self._raw_record_template_table.rowCount()
        self._raw_record_template_table.insertRow(row)
        self._raw_record_template_table.setItem(row, 0, QTableWidgetItem(template.name))
        self._raw_record_template_table.setItem(row, 1, QTableWidgetItem(template.path))

    def _on_raw_record_template_selection_changed(
        self,
        current_row: int,
        _current_column: int,
        previous_row: int,
        _previous_column: int,
    ) -> None:
        if previous_row >= 0:
            self._store_raw_record_rules_from_table(previous_row)
        self._raw_record_current_template_index = current_row
        self._load_raw_record_rules_into_table(current_row)

    def _load_raw_record_rules_into_table(self, template_index: int) -> None:
        self._raw_record_rule_table.setRowCount(0)
        enabled = 0 <= template_index < len(self._raw_record_templates_data)
        self._raw_record_rule_table.setEnabled(enabled)
        self._raw_record_add_rule_button.setEnabled(enabled)
        self._raw_record_remove_rule_button.setEnabled(enabled)
        if not enabled:
            return
        for rule in self._raw_record_templates_data[template_index].rules:
            self._append_raw_record_rule_row(rule)

    def _store_raw_record_rules_from_table(self, template_index: int) -> None:
        if not (0 <= template_index < len(self._raw_record_templates_data)):
            return
        rules: list[RawRecordExportRule] = []
        for row in range(self._raw_record_rule_table.rowCount()):
            source_combo = self._raw_record_rule_table.cellWidget(row, 0)
            field_combo = self._raw_record_rule_table.cellWidget(row, 1)
            filter_combo = self._raw_record_rule_table.cellWidget(row, 2)
            direction_combo = self._raw_record_rule_table.cellWidget(row, 6)
            sheet_item = self._raw_record_rule_table.item(row, 3)
            cell_item = self._raw_record_rule_table.item(row, 4)
            end_cell_item = self._raw_record_rule_table.item(row, 5)
            data_source = source_combo.currentData() if isinstance(source_combo, QComboBox) else RawRecordDataSource.DIAMETER_RESULT
            field_name = field_combo.currentText().strip() if isinstance(field_combo, QComboBox) else "结果"
            measurement_filter = filter_combo.currentData() if isinstance(filter_combo, QComboBox) else RawRecordMeasurementFilter.ALL
            direction = direction_combo.currentData() if isinstance(direction_combo, QComboBox) else RawRecordExportDirection.VERTICAL
            rules.append(
                RawRecordExportRule(
                    data_source=str(data_source),
                    field_name=field_name or "结果",
                    measurement_filter=str(measurement_filter),
                    sheet_name=(sheet_item.text().strip() if sheet_item is not None else "Sheet1") or "Sheet1",
                    start_cell=(cell_item.text().strip() if cell_item is not None else "B2") or "B2",
                    end_cell=(end_cell_item.text().strip() if end_cell_item is not None else ""),
                    direction=str(direction),
                ).normalized_copy()
            )
        current = self._raw_record_templates_data[template_index]
        self._raw_record_templates_data[template_index] = RawRecordTemplate(
            name=current.name,
            path=current.path,
            rules=rules,
        ).normalized_copy()

    def _append_raw_record_rule_row(self, rule: RawRecordExportRule) -> None:
        normalized = rule.normalized_copy()
        row = self._raw_record_rule_table.rowCount()
        self._raw_record_rule_table.insertRow(row)
        source_combo = self._raw_record_combo(RAW_RECORD_DATA_SOURCE_ITEMS, normalized.data_source)
        self._raw_record_rule_table.setCellWidget(
            row,
            0,
            source_combo,
        )
        field_combo = NoWheelComboBox()
        field_combo.setEditable(True)
        for field_name in RAW_RECORD_FIELD_NAMES:
            field_combo.addItem(field_name, field_name)
        field_index = field_combo.findText(normalized.field_name)
        if field_index >= 0:
            field_combo.setCurrentIndex(field_index)
        else:
            field_combo.setEditText(normalized.field_name)
        self._raw_record_rule_table.setCellWidget(row, 1, field_combo)
        filter_combo = self._raw_record_combo(RAW_RECORD_FILTER_ITEMS, normalized.measurement_filter)
        self._raw_record_rule_table.setCellWidget(
            row,
            2,
            filter_combo,
        )
        self._raw_record_rule_table.setItem(row, 3, QTableWidgetItem(normalized.sheet_name))
        self._raw_record_rule_table.setItem(row, 4, QTableWidgetItem(normalized.start_cell))
        self._raw_record_rule_table.setItem(row, 5, QTableWidgetItem(normalized.end_cell))
        self._raw_record_rule_table.setCellWidget(
            row,
            6,
            self._raw_record_combo(RAW_RECORD_DIRECTION_ITEMS, normalized.direction),
        )
        source_combo.currentIndexChanged.connect(
            lambda _index, source=source_combo: self._sync_raw_record_rule_row_state(
                self._raw_record_rule_row_for_widget(source)
            )
        )
        self._sync_raw_record_rule_row_state(row)

    def _raw_record_combo(self, items: list[tuple[str, str]], current_value: str) -> NoWheelComboBox:
        combo = NoWheelComboBox()
        for label, value in items:
            combo.addItem(label, value)
        index = combo.findData(current_value)
        combo.setCurrentIndex(max(0, index))
        return combo

    def _sync_raw_record_rule_row_state(self, row: int) -> None:
        if row < 0:
            return
        source_combo = self._raw_record_rule_table.cellWidget(row, 0)
        field_combo = self._raw_record_rule_table.cellWidget(row, 1)
        filter_combo = self._raw_record_rule_table.cellWidget(row, 2)
        end_cell_item = self._raw_record_rule_table.item(row, 5)
        if not isinstance(source_combo, QComboBox) or not isinstance(field_combo, QComboBox) or not isinstance(filter_combo, QComboBox):
            return
        data_source = str(source_combo.currentData() or "")
        self._reset_raw_record_filter_combo_labels(filter_combo)
        if end_cell_item is not None:
            end_cell_item.setFlags(self._raw_record_table_item_flags(enabled=True))
            if data_source == RawRecordDataSource.UNIQUE_FIELD_RANGE:
                end_cell_item.setToolTip("去重字段范围需要填写结束单元格，例如 BG11。")
            else:
                end_cell_item.setToolTip("普通规则填写结束单元格后，会按纤维类别分列或分行导出。")
        if data_source == RawRecordDataSource.DIAMETER_RESULT:
            self._set_combo_current_data(field_combo, "结果", text_fallback="结果")
            self._set_combo_item_text_for_data(filter_combo, RawRecordMeasurementFilter.LINE, "自动: 直径/线段")
            self._set_combo_current_data(filter_combo, RawRecordMeasurementFilter.LINE)
            field_combo.setEnabled(False)
            filter_combo.setEnabled(False)
            field_combo.setToolTip("直径结果固定导出“结果”字段。")
            filter_combo.setToolTip("直径结果会自动只导出直径/线段测量。")
            return
        if data_source == RawRecordDataSource.AREA_RESULT:
            self._set_combo_current_data(field_combo, "结果", text_fallback="结果")
            self._set_combo_item_text_for_data(filter_combo, RawRecordMeasurementFilter.AREA, "自动: 面积")
            self._set_combo_current_data(filter_combo, RawRecordMeasurementFilter.AREA)
            field_combo.setEnabled(False)
            filter_combo.setEnabled(False)
            field_combo.setToolTip("面积结果固定导出“结果”字段。")
            filter_combo.setToolTip("面积结果会自动只导出面积测量。")
            return
        field_combo.setEnabled(True)
        filter_combo.setEnabled(True)
        field_combo.setToolTip("")
        filter_combo.setToolTip("")

    def _raw_record_table_item_flags(self, *, enabled: bool) -> Qt.ItemFlag:
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if enabled:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def _set_combo_current_data(self, combo: QComboBox, value: str, *, text_fallback: str | None = None) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
            return
        text = text_fallback or value
        text_index = combo.findText(text)
        if text_index >= 0:
            combo.setCurrentIndex(text_index)
        elif combo.isEditable():
            combo.setEditText(text)

    def _reset_raw_record_filter_combo_labels(self, combo: QComboBox) -> None:
        for label, value in RAW_RECORD_FILTER_ITEMS:
            self._set_combo_item_text_for_data(combo, value, label)

    def _set_combo_item_text_for_data(self, combo: QComboBox, value: str, label: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setItemText(index, label)

    def _raw_record_rule_row_for_widget(self, widget: QWidget) -> int:
        for row in range(self._raw_record_rule_table.rowCount()):
            if self._raw_record_rule_table.cellWidget(row, 0) is widget:
                return row
        return -1

    def _add_raw_record_template(self) -> None:
        path = self._choose_raw_record_template_path()
        if not path:
            return
        self._store_raw_record_rules_from_table(self._raw_record_current_template_index)
        template = RawRecordTemplate(
            name=Path(path).stem,
            path=to_resource_relative_path(path),
            rules=[RawRecordExportRule()],
        ).normalized_copy()
        self._raw_record_templates_data.append(template)
        self._insert_raw_record_template_row(template)
        self._raw_record_template_table.setCurrentCell(self._raw_record_template_table.rowCount() - 1, 0)

    def _remove_selected_raw_record_template(self) -> None:
        row = self._selected_raw_record_template_row()
        if row < 0:
            return
        self._raw_record_template_table.removeRow(row)
        if row < len(self._raw_record_templates_data):
            self._raw_record_templates_data.pop(row)
        if self._raw_record_template_table.rowCount() == 0:
            self._raw_record_current_template_index = -1
            self._load_raw_record_rules_into_table(-1)
            return
        self._raw_record_template_table.setCurrentCell(min(row, self._raw_record_template_table.rowCount() - 1), 0)

    def _browse_selected_raw_record_template(self) -> None:
        row = self._selected_raw_record_template_row()
        if row < 0:
            return
        path = self._choose_raw_record_template_path()
        if not path:
            return
        path_token = to_resource_relative_path(path)
        self._raw_record_template_table.setItem(row, 1, QTableWidgetItem(path_token))
        name_item = self._raw_record_template_table.item(row, 0)
        if name_item is None or not name_item.text().strip():
            self._raw_record_template_table.setItem(row, 0, QTableWidgetItem(Path(path).stem))
        if row < len(self._raw_record_templates_data):
            current = self._raw_record_templates_data[row]
            self._raw_record_templates_data[row] = RawRecordTemplate(
                name=current.name or Path(path).stem,
                path=path_token,
                rules=list(current.rules),
            ).normalized_copy()

    def _choose_raw_record_template_path(self) -> str:
        start_dir = bundle_resource_root() / "runtime" / "content-templates"
        if not start_dir.exists():
            start_dir = bundle_resource_root()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择原始记录模板",
            str(start_dir),
            "Excel 模板 (*.xlsx *.xlsm *.xltx *.xltm);;旧版 Excel (*.xls *.xlt);;所有文件 (*)",
        )
        if not path:
            return ""
        suffix = Path(path).suffix.lower()
        if suffix not in SUPPORTED_RAW_RECORD_TEMPLATE_SUFFIXES:
            QMessageBox.warning(
                self,
                "原始记录模板",
                "当前只支持 .xlsx/.xlsm/.xltx/.xltm 模板。\n请先用 Excel 将 .xls/.xlt 另存为新格式后再配置。",
            )
            return ""
        return path

    def _selected_raw_record_template_row(self) -> int:
        selected_rows = self._raw_record_template_table.selectionModel().selectedRows()
        if selected_rows:
            return selected_rows[0].row()
        return self._raw_record_template_table.currentRow()

    def _add_raw_record_rule(self) -> None:
        if not (0 <= self._raw_record_current_template_index < len(self._raw_record_templates_data)):
            return
        self._append_raw_record_rule_row(RawRecordExportRule())

    def _remove_selected_raw_record_rule(self) -> None:
        selected_rows = self._raw_record_rule_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        self._raw_record_rule_table.removeRow(selected_rows[0].row())

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
