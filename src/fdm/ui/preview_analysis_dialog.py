from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from fdm.models import ImageDocument, new_id
from fdm.ui.canvas import DocumentCanvas


class PreviewAnalysisDialog(QDialog):
    finishRequested = Signal()
    cancelRequested = Signal()

    def __init__(
        self,
        title: str,
        *,
        intro_text: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle(title)
        self.resize(1100, 760)
        self._ignore_close_signal = False
        self._busy = False
        self._document: ImageDocument | None = None

        layout = QVBoxLayout(self)
        self._summary_label = QLabel(intro_text)
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._status_label = QLabel("等待采样…")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self.canvas = DocumentCanvas(self)
        self.canvas.set_read_only(True)
        self.canvas.set_tool_mode("select")
        self.canvas.set_fit_alignment("top_left")
        layout.addWidget(self.canvas, 1)
        buttons_row = QHBoxLayout()
        buttons_row.addStretch(1)
        self._finish_button = QPushButton("结束")
        self._finish_button.clicked.connect(self.finishRequested.emit)
        buttons_row.addWidget(self._finish_button)
        self._cancel_button = QPushButton("取消")
        self._cancel_button.clicked.connect(self.cancelRequested.emit)
        buttons_row.addWidget(self._cancel_button)
        layout.addLayout(buttons_row)

        self._busy_overlay = QWidget(self)
        self._busy_overlay.setVisible(False)
        self._busy_overlay.setStyleSheet("background: rgba(12, 16, 24, 150);")
        overlay_layout = QVBoxLayout(self._busy_overlay)
        overlay_layout.setContentsMargins(24, 24, 24, 24)
        overlay_layout.addStretch(1)
        self._busy_panel = QWidget(self._busy_overlay)
        self._busy_panel.setObjectName("preview_analysis_busy_panel")
        self._busy_panel.setStyleSheet(
            "QWidget#preview_analysis_busy_panel {"
            " background: rgba(16, 24, 32, 228);"
            " border: 1px solid rgba(255, 255, 255, 32);"
            " border-radius: 12px;"
            "}"
        )
        busy_panel_layout = QVBoxLayout(self._busy_panel)
        busy_panel_layout.setContentsMargins(28, 22, 28, 22)
        busy_panel_layout.setSpacing(12)
        self._busy_label = QLabel("正在完成景深合成，请稍候…")
        self._busy_label.setStyleSheet("color: #F7F4EA; font-weight: 600;")
        self._busy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._busy_label.setWordWrap(True)
        self._busy_progress = QProgressBar(self._busy_panel)
        self._busy_progress.setRange(0, 0)
        self._busy_progress.setTextVisible(False)
        self._busy_progress.setMinimumHeight(14)
        busy_panel_layout.addWidget(self._busy_label)
        busy_panel_layout.addWidget(self._busy_progress)
        overlay_layout.addWidget(self._busy_panel, 0, Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addStretch(1)
        self._busy_overlay.raise_()
        self._sync_busy_overlay_geometry()

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

    def set_busy(self, active: bool, text: str = "正在处理，请稍候…") -> None:
        self._busy = active
        self._busy_label.setText(text)
        self._finish_button.setEnabled(not active)
        self._cancel_button.setEnabled(not active)
        self.canvas.setEnabled(not active)
        self._busy_overlay.setVisible(active)
        if active:
            self._busy_overlay.raise_()
        self._busy_overlay.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_busy_overlay_geometry()

    def _sync_busy_overlay_geometry(self) -> None:
        self._busy_overlay.setGeometry(self.rect())
        target_width = min(max(420, int(self.width() * 0.54)), 680)
        self._busy_panel.setFixedWidth(target_width)

    def keyPressEvent(self, event) -> None:
        if self._busy:
            event.accept()
            return
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
        if self._busy and not self._ignore_close_signal:
            event.ignore()
            return
        if not self._ignore_close_signal:
            self.cancelRequested.emit()
        super().closeEvent(event)
