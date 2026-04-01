from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from fdm.models import ImageDocument, new_id
from fdm.services.preview_analysis import FocusStackRenderConfig
from fdm.settings import FocusStackProfile
from fdm.ui.canvas import DocumentCanvas


class PreviewAnalysisDialog(QDialog):
    finishRequested = Signal()
    cancelRequested = Signal()
    renderConfigChanged = Signal(object)

    def __init__(
        self,
        title: str,
        *,
        intro_text: str,
        show_focus_stack_controls: bool = False,
        initial_render_config: FocusStackRenderConfig | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle(title)
        self.resize(1100, 760)
        self._ignore_close_signal = False
        self._document: ImageDocument | None = None
        self._render_config_controls: QWidget | None = None
        self._profile_combo: QComboBox | None = None
        self._sharpen_slider: QSlider | None = None
        self._sharpen_value_label: QLabel | None = None
        self._render_config_timer = QTimer(self)
        self._render_config_timer.setSingleShot(True)
        self._render_config_timer.setInterval(150)
        self._render_config_timer.timeout.connect(self._emit_render_config_changed)
        self._initial_render_config = (initial_render_config or FocusStackRenderConfig()).normalized_copy()

        layout = QVBoxLayout(self)
        self._summary_label = QLabel(intro_text)
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._status_label = QLabel("等待采样…")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        if show_focus_stack_controls:
            self._render_config_controls = self._build_focus_stack_controls()
            layout.addWidget(self._render_config_controls)

        self.canvas = DocumentCanvas(self)
        self.canvas.set_read_only(True)
        self.canvas.set_tool_mode("select")
        self.canvas.set_fit_alignment("top_left")
        layout.addWidget(self.canvas, 1)

    def _build_focus_stack_controls(self) -> QWidget:
        container = QWidget(self)
        form = QFormLayout(container)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._profile_combo = QComboBox(container)
        self._profile_combo.addItem("锐利优先", FocusStackProfile.SHARP)
        self._profile_combo.addItem("平衡", FocusStackProfile.BALANCED)
        self._profile_combo.addItem("柔和", FocusStackProfile.SOFT)
        profile_index = self._profile_combo.findData(self._initial_render_config.profile)
        self._profile_combo.setCurrentIndex(max(0, profile_index))
        self._profile_combo.currentIndexChanged.connect(self._schedule_render_config_changed)

        self._sharpen_slider = QSlider(Qt.Orientation.Horizontal, container)
        self._sharpen_slider.setRange(0, 100)
        self._sharpen_slider.setSingleStep(5)
        self._sharpen_slider.setPageStep(10)
        self._sharpen_slider.setValue(self._initial_render_config.sharpen_strength)
        self._sharpen_slider.valueChanged.connect(self._on_sharpen_slider_changed)
        self._sharpen_value_label = QLabel(container)
        self._sharpen_value_label.setMinimumWidth(44)
        self._on_sharpen_slider_changed(self._sharpen_slider.value())

        sharpen_row = QWidget(container)
        sharpen_layout = QHBoxLayout(sharpen_row)
        sharpen_layout.setContentsMargins(0, 0, 0, 0)
        sharpen_layout.addWidget(self._sharpen_slider, 1)
        sharpen_layout.addWidget(self._sharpen_value_label)
        hint = QLabel("预览与最终结果共用同一组景深合成参数。")
        hint.setWordWrap(True)

        form.addRow("合成风格", self._profile_combo)
        form.addRow("锐化强度", sharpen_row)
        form.addRow("", hint)
        return container

    def _on_sharpen_slider_changed(self, value: int) -> None:
        if self._sharpen_value_label is not None:
            self._sharpen_value_label.setText(f"{int(value)}")
        self._schedule_render_config_changed()

    def _schedule_render_config_changed(self) -> None:
        if self._render_config_controls is None:
            return
        self._render_config_timer.start()

    def _emit_render_config_changed(self) -> None:
        if self._render_config_controls is None:
            return
        self.renderConfigChanged.emit(self.render_config())

    def set_summary(self, text: str) -> None:
        self._summary_label.setText(text)

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_result_image(self, image: QImage) -> None:
        if image.isNull():
            return
        if self._document is None or self._document.image_size != (image.width(), image.height()):
            self._document = ImageDocument(
                id=new_id("preview"),
                path="preview_analysis.png",
                image_size=(image.width(), image.height()),
            )
            self._document.initialize_runtime_state()
            self.canvas.set_document(self._document, image)
            self.canvas.fit_to_view()
        else:
            self.canvas.set_image(image)
        self.canvas.focus_canvas()

    def close_silently(self) -> None:
        self._ignore_close_signal = True
        self.close()
        self._ignore_close_signal = False

    def render_config(self) -> FocusStackRenderConfig:
        if self._profile_combo is None or self._sharpen_slider is None:
            return self._initial_render_config.normalized_copy()
        return FocusStackRenderConfig(
            profile=str(self._profile_combo.currentData() or FocusStackProfile.BALANCED),
            sharpen_strength=int(self._sharpen_slider.value()),
        ).normalized_copy()

    def post_sharpen_enabled(self) -> bool:
        return bool(self.render_config().sharpen_strength > 0)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F):
            self.finishRequested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.cancelRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if not self._ignore_close_signal:
            self.cancelRequested.emit()
        super().closeEvent(event)
