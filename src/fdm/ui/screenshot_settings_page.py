from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fdm.screenshot_settings import (
    AfterCaptureTask,
    CollisionPolicy,
    HotkeyBinding,
    ImageFormat,
    ScreenshotSettings,
)
from fdm.services.screenshot_capture import CaptureMode
from fdm.services.cu5_preview_locator import Cu5PreviewSelector
from fdm.ui.widgets import NoWheelComboBox, NoWheelSpinBox


_HOTKEY_ROWS = (
    (CaptureMode.REGION, "自由区域"),
    (CaptureMode.WINDOW, "窗口 / 子窗口"),
    (CaptureMode.FULL_SCREEN, "全部屏幕"),
    (CaptureMode.LAST_REGION, "上次区域"),
    (CaptureMode.CU5, "CU 系列实时预览"),
)


class ScreenshotSettingsPage(QWidget):
    """Preferences page for the independent screenshot companion process."""

    cu5DiagnosticRequested = Signal()
    cu5CandidateSelectionRequested = Signal(object)

    def __init__(
        self,
        settings: ScreenshotSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._initial_settings = (settings or ScreenshotSettings()).normalized()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        resident_group = QGroupBox("常驻与启动", self)
        resident_layout = QVBoxLayout(resident_group)
        self.resident_checkbox = QCheckBox("启动独立常驻截图工具", resident_group)
        self.resident_checkbox.setChecked(self._initial_settings.enabled)
        self.resident_checkbox.setToolTip(
            "截图工具在独立进程中运行；关闭测量工作台不会中断它。"
        )
        self.autostart_checkbox = QCheckBox("登录 Windows 后自动启动", resident_group)
        self.autostart_checkbox.setChecked(self._initial_settings.autostart)
        self.agent_status_label = QLabel("运行状态：尚未检测", resident_group)
        self.agent_status_label.setWordWrap(True)
        resident_layout.addWidget(self.resident_checkbox)
        resident_layout.addWidget(self.autostart_checkbox)
        resident_layout.addWidget(self.agent_status_label)
        layout.addWidget(resident_group)

        output_group = QGroupBox("保存与捕获后操作", self)
        output_form = QFormLayout(output_group)
        output_row = QWidget(output_group)
        output_row_layout = QHBoxLayout(output_row)
        output_row_layout.setContentsMargins(0, 0, 0, 0)
        self.output_directory_edit = QLineEdit(
            self._initial_settings.output_directory,
            output_row,
        )
        self.output_directory_edit.setPlaceholderText("截图保存目录")
        browse_button = QPushButton("浏览...", output_row)
        browse_button.clicked.connect(self._browse_output_directory)
        output_row_layout.addWidget(self.output_directory_edit, 1)
        output_row_layout.addWidget(browse_button)
        output_form.addRow("输出目录", output_row)

        self.filename_template_edit = QLineEdit(
            self._initial_settings.filename_template,
            output_group,
        )
        self.filename_template_edit.setToolTip(
            "可用变量：{date}、{time}、{datetime}、{mode}、{counter}"
        )
        output_form.addRow("文件名模板", self.filename_template_edit)

        self.image_format_combo = NoWheelComboBox(output_group)
        for label, image_format in (
            ("PNG（无损）", ImageFormat.PNG),
            ("JPEG", ImageFormat.JPEG),
            ("WebP", ImageFormat.WEBP),
        ):
            self.image_format_combo.addItem(label, image_format.value)
        self.image_format_combo.setCurrentIndex(
            max(0, self.image_format_combo.findData(self._initial_settings.image_format.value))
        )
        output_form.addRow("图片格式", self.image_format_combo)

        self.quality_spin = NoWheelSpinBox(output_group)
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setSuffix("%")
        self.quality_spin.setValue(self._quality_for(self._initial_settings.image_format))
        self._quality_values = {
            ImageFormat.PNG: self._quality_for(ImageFormat.PNG),
            ImageFormat.JPEG: self._quality_for(ImageFormat.JPEG),
            ImageFormat.WEBP: self._quality_for(ImageFormat.WEBP),
        }
        self._quality_format = self._initial_settings.image_format
        output_form.addRow("有损质量", self.quality_spin)

        self.collision_combo = NoWheelComboBox(output_group)
        self.collision_combo.addItem("自动追加序号", CollisionPolicy.INCREMENT.value)
        self.collision_combo.addItem("覆盖同名文件", CollisionPolicy.OVERWRITE.value)
        self.collision_combo.addItem("同名时报告失败", CollisionPolicy.FAIL.value)
        self.collision_combo.setCurrentIndex(
            max(0, self.collision_combo.findData(self._initial_settings.collision_policy.value))
        )
        output_form.addRow("重名处理", self.collision_combo)

        task_row = QWidget(output_group)
        task_layout = QHBoxLayout(task_row)
        task_layout.setContentsMargins(0, 0, 0, 0)
        self.save_checkbox = QCheckBox("保存文件", task_row)
        self.copy_checkbox = QCheckBox("复制到剪贴板", task_row)
        tasks = set(self._initial_settings.after_capture_tasks)
        self.save_checkbox.setChecked(AfterCaptureTask.SAVE in tasks)
        self.copy_checkbox.setChecked(AfterCaptureTask.COPY_CLIPBOARD in tasks)
        task_layout.addWidget(self.save_checkbox)
        task_layout.addWidget(self.copy_checkbox)
        task_layout.addStretch(1)
        output_form.addRow("完成后", task_row)

        behavior_row = QWidget(output_group)
        behavior_layout = QHBoxLayout(behavior_row)
        behavior_layout.setContentsMargins(0, 0, 0, 0)
        self.editor_checkbox = QCheckBox("打开标注编辑器", behavior_row)
        self.editor_checkbox.setChecked(self._initial_settings.show_editor)
        self.notification_checkbox = QCheckBox("显示完成通知", behavior_row)
        self.notification_checkbox.setChecked(self._initial_settings.notification)
        self.cursor_checkbox = QCheckBox("包含鼠标指针", behavior_row)
        self.cursor_checkbox.setChecked(self._initial_settings.include_cursor)
        behavior_layout.addWidget(self.editor_checkbox)
        behavior_layout.addWidget(self.notification_checkbox)
        behavior_layout.addWidget(self.cursor_checkbox)
        behavior_layout.addStretch(1)
        output_form.addRow("行为", behavior_row)

        self.delay_spin = NoWheelSpinBox(output_group)
        self.delay_spin.setRange(0, 60_000)
        self.delay_spin.setSingleStep(500)
        self.delay_spin.setSuffix(" ms")
        self.delay_spin.setValue(self._initial_settings.delay_ms)
        output_form.addRow("截图延时", self.delay_spin)
        layout.addWidget(output_group)

        hotkey_group = QGroupBox("全局快捷键", self)
        hotkey_form = QFormLayout(hotkey_group)
        self.hotkey_edits: dict[CaptureMode, QKeySequenceEdit] = {}
        for mode, label in _HOTKEY_ROWS:
            edit = QKeySequenceEdit(hotkey_group)
            binding = self._initial_settings.hotkeys.get(mode, HotkeyBinding())
            edit.setKeySequence(QKeySequence(_portable_sequence(binding.sequence)))
            edit.setClearButtonEnabled(True)
            edit.setMaximumSequenceLength(1)
            self.hotkey_edits[mode] = edit
            hotkey_form.addRow(label, edit)
        hotkey_hint = QLabel(
            "快捷键由常驻进程在 Windows 全局注册；若被其它软件占用，会保留原绑定并明确提示。",
            hotkey_group,
        )
        hotkey_hint.setWordWrap(True)
        hotkey_form.addRow(hotkey_hint)
        layout.addWidget(hotkey_group)

        cu5_group = QGroupBox("CU 系列 / Microview 实时预览", self)
        cu5_layout = QVBoxLayout(cu5_group)
        cu5_description = QLabel(
            "专用模式会按 CU 系列软件的原生子窗口层级自动识别视频画面区域，"
            "不会重新打开或抢占 Microview 设备，也不会截取完整软件窗口。",
            cu5_group,
        )
        cu5_description.setWordWrap(True)
        self.cu5_diagnostic_button = QPushButton("检测 CU 系列实时预览区域", cu5_group)
        self.cu5_diagnostic_button.clicked.connect(self.cu5DiagnosticRequested)
        self.cu5_status_label = QLabel(
            "尚未在本机检测；需要在 Windows 设备上先打开 CU 系列实时预览。",
            cu5_group,
        )
        self.cu5_status_label.setWordWrap(True)
        candidate_row = QWidget(cu5_group)
        candidate_layout = QHBoxLayout(candidate_row)
        candidate_layout.setContentsMargins(0, 0, 0, 0)
        candidate_label = QLabel("预览对象", candidate_row)
        self.cu5_candidate_combo = NoWheelComboBox(candidate_row)
        self.cu5_candidate_combo.setSizeAdjustPolicy(
            NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.cu5_candidate_combo.setMinimumContentsLength(24)
        self.cu5_candidate_combo.addItem("请先检测实时预览区域", None)
        self.cu5_candidate_combo.setEnabled(False)
        self.cu5_candidate_apply_button = QPushButton("使用所选对象", candidate_row)
        self.cu5_candidate_apply_button.setEnabled(False)
        self.cu5_candidate_apply_button.clicked.connect(
            self._request_cu5_candidate_selection
        )
        candidate_layout.addWidget(candidate_label)
        candidate_layout.addWidget(self.cu5_candidate_combo, 1)
        candidate_layout.addWidget(self.cu5_candidate_apply_button)
        self.cu5_candidate_hint = QLabel(
            "检测后会默认选中推荐对象；只有点击“使用所选对象”才会更改后续抓取目标。",
            cu5_group,
        )
        self.cu5_candidate_hint.setWordWrap(True)
        cu5_layout.addWidget(cu5_description)
        cu5_layout.addWidget(self.cu5_diagnostic_button)
        cu5_layout.addWidget(self.cu5_status_label)
        cu5_layout.addWidget(candidate_row)
        cu5_layout.addWidget(self.cu5_candidate_hint)
        layout.addWidget(cu5_group)
        layout.addStretch(1)

        self.image_format_combo.currentIndexChanged.connect(self._sync_quality_state)
        self.autostart_checkbox.toggled.connect(self._sync_resident_dependency)
        self.resident_checkbox.toggled.connect(self._sync_autostart_dependency)
        self._sync_quality_state()

    def settings(self) -> ScreenshotSettings:
        image_format = ImageFormat.parse(
            self.image_format_combo.currentData(),
            default=ImageFormat.PNG,
        )
        self._quality_values[image_format] = self.quality_spin.value()
        tasks: list[AfterCaptureTask] = []
        if self.save_checkbox.isChecked():
            tasks.append(AfterCaptureTask.SAVE)
        if self.copy_checkbox.isChecked():
            tasks.append(AfterCaptureTask.COPY_CLIPBOARD)
        # The domain model guarantees a useful pipeline even if both boxes are
        # cleared; keep that normalization visible here as well.
        if not tasks:
            tasks.append(AfterCaptureTask.SAVE)
        hotkeys = dict(self._initial_settings.hotkeys)
        for mode, edit in self.hotkey_edits.items():
            sequence = edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            hotkeys[mode] = HotkeyBinding(sequence=sequence, enabled=bool(sequence))
        updated = replace(
            self._initial_settings,
            enabled=self.resident_checkbox.isChecked(),
            autostart=self.autostart_checkbox.isChecked(),
            output_directory=self.output_directory_edit.text().strip(),
            filename_template=self.filename_template_edit.text().strip(),
            image_format=image_format,
            collision_policy=CollisionPolicy.parse(
                self.collision_combo.currentData(),
                default=CollisionPolicy.INCREMENT,
            ),
            after_capture_tasks=tuple(tasks),
            delay_ms=self.delay_spin.value(),
            include_cursor=self.cursor_checkbox.isChecked(),
            show_editor=self.editor_checkbox.isChecked(),
            notification=self.notification_checkbox.isChecked(),
            hotkeys=hotkeys,
            # Retain the compatibility field without exposing a switch that
            # currently has no runtime behavior.  Diagnostics remain an
            # explicit button action.
            cu5_diagnostics_enabled=self._initial_settings.cu5_diagnostics_enabled,
        )
        updated.png_compression = max(
            0,
            min(9, round((100 - self._quality_values[ImageFormat.PNG]) * 9 / 99)),
        )
        updated.jpeg_quality = self._quality_values[ImageFormat.JPEG]
        updated.webp_quality = self._quality_values[ImageFormat.WEBP]
        return updated.normalized()

    def set_agent_status(self, running: bool, detail: str = "") -> None:
        state = "正在运行" if running else "未运行"
        suffix = f"；{detail}" if detail else ""
        self.agent_status_label.setText(f"运行状态：{state}{suffix}")

    def restore_defaults(self) -> None:
        defaults = ScreenshotSettings().normalized()
        self._initial_settings = defaults
        self.resident_checkbox.setChecked(defaults.enabled)
        self.autostart_checkbox.setChecked(defaults.autostart)
        self.output_directory_edit.setText(defaults.output_directory)
        self.filename_template_edit.setText(defaults.filename_template)
        self.image_format_combo.setCurrentIndex(
            max(0, self.image_format_combo.findData(defaults.image_format.value))
        )
        self._quality_values = {
            ImageFormat.PNG: self._quality_for(ImageFormat.PNG),
            ImageFormat.JPEG: self._quality_for(ImageFormat.JPEG),
            ImageFormat.WEBP: self._quality_for(ImageFormat.WEBP),
        }
        self._quality_format = defaults.image_format
        self.quality_spin.setValue(self._quality_values[defaults.image_format])
        self.collision_combo.setCurrentIndex(
            max(0, self.collision_combo.findData(defaults.collision_policy.value))
        )
        tasks = set(defaults.after_capture_tasks)
        self.save_checkbox.setChecked(AfterCaptureTask.SAVE in tasks)
        self.copy_checkbox.setChecked(AfterCaptureTask.COPY_CLIPBOARD in tasks)
        self.delay_spin.setValue(defaults.delay_ms)
        self.cursor_checkbox.setChecked(defaults.include_cursor)
        self.editor_checkbox.setChecked(defaults.show_editor)
        self.notification_checkbox.setChecked(defaults.notification)
        for mode, edit in self.hotkey_edits.items():
            binding = defaults.hotkeys.get(mode, HotkeyBinding())
            edit.setKeySequence(QKeySequence(_portable_sequence(binding.sequence)))
        self._sync_quality_state()

    def set_cu5_diagnostic_status(self, message: str, *, success: bool) -> None:
        self.cu5_status_label.setText(str(message).strip())
        self.cu5_status_label.setProperty("diagnosticSuccess", bool(success))
        self.cu5_status_label.style().unpolish(self.cu5_status_label)
        self.cu5_status_label.style().polish(self.cu5_status_label)

    def set_cu5_candidates(
        self,
        candidates: object,
        *,
        selected_selector: object = None,
    ) -> None:
        """Show adjustable native preview objects without changing settings."""

        selected = Cu5PreviewSelector.from_value(selected_selector).to_dict()
        entries = candidates if isinstance(candidates, (tuple, list)) else ()
        self.cu5_candidate_combo.blockSignals(True)
        try:
            self.cu5_candidate_combo.clear()
            selected_index = -1
            for raw in entries:
                if not isinstance(raw, dict):
                    continue
                selector = Cu5PreviewSelector.from_value(raw.get("selector"))
                rect = raw.get("rect")
                if not selector.active or not isinstance(rect, dict):
                    continue
                item_selector = selector.to_dict()
                title = str(raw.get("title", "") or "").strip()
                class_name = str(raw.get("class_name", "") or "").strip()
                process_name = str(raw.get("process_name", "") or "").strip()
                identity = title or class_name or process_name or "原生窗口对象"
                if title and class_name and title.casefold() != class_name.casefold():
                    identity = f"{title} · {class_name}"
                width = _safe_int(rect.get("width"))
                height = _safe_int(rect.get("height"))
                x = _safe_int(rect.get("x"))
                y = _safe_int(rect.get("y"))
                score = _safe_float(raw.get("score"))
                if (
                    width is None
                    or height is None
                    or width <= 0
                    or height <= 0
                    or x is None
                    or y is None
                    or score is None
                ):
                    continue
                recommended = bool(selected and item_selector == selected)
                prefix = "推荐 · " if recommended else ""
                self.cu5_candidate_combo.addItem(
                    f"{prefix}{identity} · {width}×{height} · ({x}, {y}) · {score:.1f}",
                    item_selector,
                )
                if recommended:
                    selected_index = self.cu5_candidate_combo.count() - 1
            if self.cu5_candidate_combo.count() == 0:
                self.cu5_candidate_combo.addItem("未发现可调整的预览对象", None)
            elif selected_index < 0:
                selected_index = 0
            self.cu5_candidate_combo.setCurrentIndex(max(0, selected_index))
        finally:
            self.cu5_candidate_combo.blockSignals(False)
        available = isinstance(self.cu5_candidate_combo.currentData(), dict)
        self.cu5_candidate_combo.setEnabled(available)
        self.cu5_candidate_apply_button.setEnabled(available)

    def _request_cu5_candidate_selection(self) -> None:
        selector = self.cu5_candidate_combo.currentData()
        if isinstance(selector, dict) and selector:
            self.cu5CandidateSelectionRequested.emit(dict(selector))

    def _quality_for(self, image_format: ImageFormat) -> int:
        if image_format is ImageFormat.PNG:
            return max(1, min(100, round(100 - self._initial_settings.png_compression * 99 / 9)))
        if image_format is ImageFormat.JPEG:
            return self._initial_settings.jpeg_quality
        return self._initial_settings.webp_quality

    def _sync_quality_state(self) -> None:
        image_format = ImageFormat.parse(
            self.image_format_combo.currentData(),
            default=ImageFormat.PNG,
        )
        previous = getattr(self, "_quality_format", image_format)
        self._quality_values[previous] = self.quality_spin.value()
        self._quality_format = image_format
        self.quality_spin.blockSignals(True)
        self.quality_spin.setValue(self._quality_values[image_format])
        self.quality_spin.blockSignals(False)
        self.quality_spin.setEnabled(image_format is not ImageFormat.PNG)
        self.quality_spin.setToolTip(
            "PNG 使用无损压缩" if image_format is ImageFormat.PNG else "数值越高，画质越高、文件越大"
        )

    def _sync_resident_dependency(self, enabled: bool) -> None:
        if enabled and not self.resident_checkbox.isChecked():
            self.resident_checkbox.setChecked(True)

    def _sync_autostart_dependency(self, enabled: bool) -> None:
        if not enabled and self.autostart_checkbox.isChecked():
            self.autostart_checkbox.setChecked(False)

    def _browse_output_directory(self) -> None:
        current = Path(self.output_directory_edit.text().strip()).expanduser()
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择截图输出目录",
            str(current if current.exists() else Path.home()),
        )
        if directory:
            self.output_directory_edit.setText(directory)


def _portable_sequence(value: str) -> str:
    # Qt calls the Print Screen key "Print" in PortableText.  Accept the more
    # familiar persisted spelling as a compatibility alias.
    return str(value or "").replace("PrintScreen", "Print")


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


__all__ = ["ScreenshotSettingsPage"]
