from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QCheckBox, QDialog, QLabel, QVBoxLayout

from fdm.models import ImageDocument, new_id
from fdm.ui.canvas import DocumentCanvas


class PreviewAnalysisDialog(QDialog):
    finishRequested = Signal()
    cancelRequested = Signal()

    def __init__(self, title: str, *, intro_text: str, show_post_sharpen_option: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle(title)
        self.resize(1100, 760)
        self._ignore_close_signal = False
        self._document: ImageDocument | None = None
        self._post_sharpen_checkbox: QCheckBox | None = None

        layout = QVBoxLayout(self)
        self._summary_label = QLabel(intro_text)
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._status_label = QLabel("等待采样…")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        if show_post_sharpen_option:
            self._post_sharpen_checkbox = QCheckBox("合成后锐化")
            self._post_sharpen_checkbox.setChecked(False)
            self._post_sharpen_checkbox.setToolTip("默认关闭。仅对景深合成结束后的最终导出结果生效，不影响实时预览。")
            layout.addWidget(self._post_sharpen_checkbox)

        self.canvas = DocumentCanvas(self)
        self.canvas.set_read_only(True)
        self.canvas.set_tool_mode("select")
        self.canvas.set_fit_alignment("top_left")
        layout.addWidget(self.canvas, 1)

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

    def post_sharpen_enabled(self) -> bool:
        return bool(self._post_sharpen_checkbox and self._post_sharpen_checkbox.isChecked())

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
