from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QDate, QDateTime, QPointF, QRectF, Qt, QTime, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QDateTimeEdit, QDialog, QDialogButtonBox, QFileDialog,
    QFontComboBox, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from fdm.ui.watermark_rendering import draw_watermark, import_logo, logo_image
from fdm.ui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox
from fdm.watermark import ANCHORS, WatermarkSpec


DATETIME_DISPLAY_FORMAT = "yyyy-MM-dd HH:mm:ss"


class WatermarkDateTimeEdit(QDateTimeEdit):
    def wheelEvent(self, event):
        # Scrolling the settings form must not silently alter the timestamp.
        event.ignore()


class WatermarkPreview(QWidget):
    def __init__(self, document, image: QImage, parent=None):
        super().__init__(parent)
        self.document = document
        self.image = image
        self.setMinimumSize(260, 260)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#252b34"))
        scale = min((self.width() - 24) / self.image.width(), (self.height() - 24) / self.image.height())
        left = (self.width() - self.image.width() * scale) / 2
        top = (self.height() - self.image.height() * scale) / 2
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRectF(left, top, self.image.width() * scale, self.image.height() * scale), self.image)
        draw_watermark(painter, self.document, lambda point: QPointF(left + point.x * scale, top + point.y * scale))
        painter.end()


class WatermarkDialog(QDialog):
    """Edit an isolated draft; only the accepted result changes live documents."""

    def __init__(self, document, image: QImage, parent=None, *, default_spec=None, default_assets=None):
        super().__init__(parent)
        self.setWindowTitle("图片水印")
        self.setMinimumSize(660, 460)
        self.resize(980, 660)
        self._removed = False
        self._draft = replace(document, watermark_assets=dict(document.watermark_assets))
        spec = document.watermark
        if spec is None:
            spec = default_spec or WatermarkSpec(
                enabled=True, text="GTTC", bold=True, include_datetime=True,
                layout="tile", opacity=0.75, rotation=45.0, gap_x=0.25, gap_y=0.25,
            )
            self._draft.watermark_assets.update(default_assets or {})
            # Remember the datetime option, not another image's timestamp.
            # An existing document's timestamp stays fixed across edits.
            spec = replace(
                spec,
                datetime_text=QDateTime.currentDateTime().toString(DATETIME_DISPLAY_FORMAT)
                if spec.include_datetime else "",
            )
        self._logo_sha256 = spec.logo_sha256
        self._color = spec.color
        self._font_family = spec.font_family
        self._datetime_text = spec.datetime_text
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.enabled_check = QCheckBox("启用水印")
        self.enabled_check.setChecked(spec.enabled)
        form.addRow(self.enabled_check)
        self.kind_combo = NoWheelComboBox()
        self.kind_combo.addItem("文字", "text")
        self.kind_combo.addItem("图片 Logo", "logo")
        self.kind_combo.setCurrentIndex(self.kind_combo.findData(spec.kind))
        form.addRow("水印内容", self.kind_combo)
        self.text_edit = QPlainTextEdit(spec.text)
        self.text_edit.setPlaceholderText("输入机构名称、版权说明等；支持多行")
        self.text_edit.setMaximumHeight(86)
        form.addRow("文字", self.text_edit)
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(spec.font_family) if spec.font_family else self.font())
        # System font aliases (notably macOS' private UI family) are absent
        # from QFontComboBox's list. Show the real selection instead of its
        # first alphabetical entry while retaining the default in the spec.
        self.font_combo.setEditText(spec.font_family or "系统默认")
        form.addRow("字体", self.font_combo)
        font_style = QWidget()
        font_style_layout = QHBoxLayout(font_style)
        font_style_layout.setContentsMargins(0, 0, 0, 0)
        self.bold_check = QCheckBox("粗体")
        self.italic_check = QCheckBox("斜体")
        self.bold_check.setChecked(spec.bold)
        self.italic_check.setChecked(spec.italic)
        font_style_layout.addWidget(self.bold_check)
        font_style_layout.addWidget(self.italic_check)
        form.addRow("字形", font_style)
        self.color_button = QPushButton(self._color)
        self.color_button.clicked.connect(self._choose_color)
        form.addRow("文字颜色", self.color_button)
        self.logo_button = QPushButton("更换 Logo…" if self._logo_sha256 else "选择 Logo…")
        self.logo_button.clicked.connect(self._choose_logo)
        form.addRow("Logo 图片", self.logo_button)
        self.datetime_check = QCheckBox("附加日期和时间")
        self.datetime_check.setChecked(spec.include_datetime)
        self.datetime_check.setToolTip("显示在水印内容下方，随水印一起旋转、平铺和调整不透明度。")
        form.addRow(self.datetime_check)
        datetime_row = QWidget()
        datetime_layout = QHBoxLayout(datetime_row)
        datetime_layout.setContentsMargins(0, 0, 0, 0)
        self.datetime_edit = WatermarkDateTimeEdit()
        self.datetime_edit.setDisplayFormat(DATETIME_DISPLAY_FORMAT)
        self.datetime_edit.setDateTimeRange(
            QDateTime(QDate(1, 1, 1), QTime(0, 0)),
            QDateTime(QDate(9999, 12, 31), QTime(23, 59, 59)),
        )
        self.datetime_edit.setCalendarPopup(True)
        self.datetime_edit.setDateTime(
            QDateTime.fromString(spec.datetime_text, DATETIME_DISPLAY_FORMAT)
            if spec.datetime_text else QDateTime.currentDateTime()
        )
        self.datetime_edit.setToolTip("默认记录设置时的本机日期时间，可手动修改；保存后保持固定。")
        self.datetime_now_button = QPushButton("当前时间")
        self.datetime_now_button.clicked.connect(self._use_current_datetime)
        datetime_layout.addWidget(self.datetime_edit, 1)
        datetime_layout.addWidget(self.datetime_now_button)
        form.addRow("日期时间", datetime_row)
        self.layout_combo = NoWheelComboBox()
        self.layout_combo.addItem("单个", "single")
        self.layout_combo.addItem("重复平铺", "tile")
        self.layout_combo.setCurrentIndex(self.layout_combo.findData(spec.layout))
        form.addRow("布局", self.layout_combo)
        self.anchor_combo = NoWheelComboBox()
        for key, label in zip(ANCHORS, ("左上", "上中", "右上", "左中", "居中", "右中", "左下", "下中", "右下")):
            self.anchor_combo.addItem(label, key)
        self.anchor_combo.setCurrentIndex(self.anchor_combo.findData(spec.anchor))
        form.addRow("位置", self.anchor_combo)

        def spin(label, low, high, value, suffix="%"):
            control = NoWheelDoubleSpinBox()
            control.setRange(low, high)
            control.setDecimals(1)
            control.setSuffix(suffix)
            control.setValue(value)
            form.addRow(label, control)
            return control

        self.opacity_spin = spin("不透明度", 0, 100, spec.opacity * 100)
        self.width_spin = spin("宽度 / 图像短边", 1, 200, spec.width_ratio * 100)
        self.rotation_spin = spin("旋转", -180, 180, spec.rotation, "°")
        self.x_spin = spin("水平边距 / 偏移", -100, 100, spec.offset_x * 100)
        self.y_spin = spin("垂直边距 / 偏移", -100, 100, spec.offset_y * 100)
        self.x_spin.setToolTip("占原图宽度；靠边时正数向内，居中和平铺时正数向右。")
        self.y_spin.setToolTip("占原图高度；靠边时正数向内，居中和平铺时正数向下。")
        self.gap_x_spin = spin("水平间距 / 水印宽度", 0, 1000, spec.gap_x * 100)
        self.gap_y_spin = spin("垂直间距 / 水印高度", 0, 1000, spec.gap_y * 100)
        self.all_check = QCheckBox("应用到全部已打开的普通图片")
        form.addRow(self.all_check)
        controls = QWidget()
        controls.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(330)
        self.preview = WatermarkPreview(self._draft, image)
        self.hint = QLabel("水印随项目保存；数字切片不会添加水印。")
        self.hint.setWordWrap(True)
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.preview, 1)
        preview_layout.addWidget(self.hint)
        row = QHBoxLayout()
        row.addWidget(scroll, 0)
        row.addLayout(preview_layout, 1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("应用")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.remove_button = self.buttons.addButton("移除水印", QDialogButtonBox.ButtonRole.DestructiveRole)
        self.remove_button.clicked.connect(self._remove)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(row, 1)
        layout.addWidget(self.buttons)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(40)
        self._preview_timer.timeout.connect(self._refresh_preview)
        for control in (self.opacity_spin, self.width_spin, self.rotation_spin, self.x_spin, self.y_spin, self.gap_x_spin, self.gap_y_spin):
            control.valueChanged.connect(self._schedule_preview)
        for control in (self.enabled_check, self.bold_check, self.italic_check):
            control.toggled.connect(self._schedule_preview)
        for control in (self.kind_combo, self.layout_combo, self.anchor_combo):
            control.currentIndexChanged.connect(self._schedule_preview)
        self.font_combo.currentFontChanged.connect(self._font_changed)
        self.datetime_check.toggled.connect(self._datetime_toggled)
        self.datetime_edit.dateTimeChanged.connect(self._datetime_changed)
        self.text_edit.textChanged.connect(self._schedule_preview)
        self._refresh_preview()
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(self.width(), available.width() - 60), min(self.height(), available.height() - 60))

    def _schedule_preview(self, *_):
        self._preview_timer.start()

    def _font_changed(self, font):
        self._font_family = "" if font.family() == "系统默认" else font.family()
        self._schedule_preview()

    def _datetime_toggled(self, enabled):
        if enabled and not self._datetime_text:
            self._use_current_datetime()
        self._schedule_preview()

    def _datetime_changed(self, value):
        self._datetime_text = value.toString(DATETIME_DISPLAY_FORMAT)
        self._schedule_preview()

    def _use_current_datetime(self):
        self.datetime_edit.setDateTime(QDateTime.currentDateTime())
        self._datetime_changed(self.datetime_edit.dateTime())

    def done(self, result):
        self._preview_timer.stop()
        super().done(result)

    def watermark(self) -> WatermarkSpec | None:
        if self._removed:
            return None
        return WatermarkSpec(
            enabled=self.enabled_check.isChecked(), kind=self.kind_combo.currentData(),
            text=self.text_edit.toPlainText(), font_family=self._font_family,
            bold=self.bold_check.isChecked(), italic=self.italic_check.isChecked(), color=self._color,
            logo_sha256=self._logo_sha256 if self.kind_combo.currentData() == "logo" else "",
            layout=self.layout_combo.currentData(),
            anchor=self.anchor_combo.currentData(), opacity=self.opacity_spin.value() / 100,
            width_ratio=self.width_spin.value() / 100, rotation=self.rotation_spin.value(),
            offset_x=self.x_spin.value() / 100, offset_y=self.y_spin.value() / 100,
            gap_x=self.gap_x_spin.value() / 100, gap_y=self.gap_y_spin.value() / 100,
            include_datetime=self.datetime_check.isChecked(), datetime_text=self._datetime_text,
        )

    def assets(self) -> dict[str, bytes]:
        digest = self._logo_sha256
        return {digest: self._draft.watermark_assets[digest]} if digest in self._draft.watermark_assets else {}

    def apply_to_all(self) -> bool:
        return self.all_check.isChecked()

    def _refresh_preview(self):
        spec = self.watermark()
        self._draft.watermark = spec
        text = spec.kind == "text"
        self.text_edit.setEnabled(text)
        for control in (self.font_combo, self.bold_check, self.italic_check, self.color_button):
            control.setEnabled(text or spec.include_datetime)
        self.datetime_edit.setEnabled(spec.include_datetime)
        self.datetime_now_button.setEnabled(spec.include_datetime)
        self.logo_button.setEnabled(not text)
        self.anchor_combo.setEnabled(spec.layout == "single")
        self.gap_x_spin.setEnabled(spec.layout == "tile")
        self.gap_y_spin.setEnabled(spec.layout == "tile")
        try:
            spec.validate_content()
            if spec.enabled and not text:
                logo_image(self._draft, spec)
            self.hint.setText("点击“应用”后记忆本次设置，供下次新建水印使用。水印随项目保存，数字切片不参与。")
        except ValueError as exc:
            self.hint.setText(str(exc))
        self.preview.update()

    def _choose_color(self):
        color = QColorDialog.getColor(QColor(self._color), self, "水印文字颜色")
        if color.isValid():
            self._color = color.name()
            self.color_button.setText(self._color)
            self._refresh_preview()

    def _choose_logo(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择水印 Logo", "", "水印图片 (*.png *.jpg *.jpeg *.webp)")
        if not path:
            return
        try:
            digest, data = import_logo(path)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "无法导入 Logo", str(exc))
            return
        self._draft.watermark_assets[digest] = data
        self._logo_sha256 = digest
        self.logo_button.setText(f"更换：{Path(path).name}")
        self._refresh_preview()

    def _accept(self):
        if self.datetime_check.isChecked():
            self.datetime_edit.interpretText()
        spec = self.watermark()
        try:
            spec.validate_content()
            if spec.enabled and spec.kind == "logo":
                logo_image(self._draft, spec)
        except ValueError as exc:
            self.hint.setText(str(exc))
            return
        self.accept()

    def _remove(self):
        self._removed = True
        self.accept()
