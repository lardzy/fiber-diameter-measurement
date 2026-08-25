from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from weakref import ref

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid as is_qobject_valid

from fdm.services.digital_slide_cache import DigitalSlideCacheCancelled
from fdm.services.digital_slide_calibration import (
    CALIBRATION_AXIS_X,
    CALIBRATION_AXIS_Y,
    DigitalSlideCalibrationEstimate,
    DigitalSlideCalibrationPair,
    DigitalSlideCalibrationSession,
)
from fdm.services.digital_slide_store import DIGITAL_SLIDE_SUFFIX
from fdm.settings import AppSettings
from fdm.ui.dialogs import NoWheelComboBox, NoWheelSlider, NoWheelSpinBox


class CalibrationPairPreview(QWidget):
    offsetChanged = Signal(int, int)
    viewChanged = Signal(float, str)
    escapeRequested = Signal()

    _MIN_VIEW_ZOOM = 0.01
    _MAX_VIEW_ZOOM = 32.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Keep the comparison usable on a 720 px-high screen.  The preview
        # expands with the dialog, while the compact minimum leaves enough
        # room for the complete estimate and the guarded motor-step option.
        self.setMinimumSize(480, 90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._reference = QImage()
        self._candidate = QImage()
        self._nominal_dx = 0
        self._nominal_dy = 0
        self._offset_x = 0
        self._offset_y = 0
        self._mode = "alpha"
        self._overlay_opacity = 0.52
        self._split_fraction = 0.5
        self._view_mode = "fit"
        self._view_zoom = 1.0
        self._view_center = QPointF()
        self._offset_drag_origin: QPointF | None = None
        self._offset_drag_value = (0, 0)
        self._view_drag_origin: QPointF | None = None
        self._view_drag_center = QPointF()
        self._pan_mode = False
        self._space_pan = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName("视场对比预览")
        self.setAccessibleDescription(
            "左键拖动校准偏移，滚轮缩放，空格加左键或中键平移查看。"
        )
        self.setToolTip(
            "左键拖动待校准视场；滚轮围绕指针缩放；"
            "空格+左键或中键平移查看。"
        )
        self._update_cursor()

    def set_pair(
        self,
        reference: QImage,
        candidate: QImage,
        *,
        nominal_dx: int,
        nominal_dy: int,
    ) -> None:
        self._reference = QImage(reference)
        self._candidate = QImage(candidate)
        self._nominal_dx = int(nominal_dx)
        self._nominal_dy = int(nominal_dy)
        self.set_offset(0, 0, emit=False)
        self.fit_to_view()

    def set_mode(self, mode: str) -> None:
        self._mode = mode if mode in {"alpha", "split", "difference"} else "alpha"
        self.update()

    def mode(self) -> str:
        return self._mode

    def set_overlay_opacity(self, value: float) -> None:
        self._overlay_opacity = max(0.05, min(0.95, float(value)))
        self.update()

    def overlay_opacity(self) -> float:
        return self._overlay_opacity

    def set_split_fraction(self, value: float) -> None:
        self._split_fraction = max(0.05, min(0.95, float(value)))
        self.update()

    def split_fraction(self) -> float:
        return self._split_fraction

    def set_pan_mode(self, enabled: bool) -> None:
        self._pan_mode = bool(enabled)
        self._update_cursor()

    def pan_mode(self) -> bool:
        return self._pan_mode

    def has_pair(self) -> bool:
        return not self._reference.isNull() and not self._candidate.isNull()

    def view_mode(self) -> str:
        return self._view_mode

    def view_zoom(self) -> float:
        if self._view_mode == "fit":
            return self._fit_zoom()
        return self._bounded_zoom(self._view_zoom)

    @classmethod
    def _bounded_zoom(cls, value: float) -> float:
        return max(cls._MIN_VIEW_ZOOM, min(cls._MAX_VIEW_ZOOM, float(value)))

    def _content_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(16.0, 16.0, -16.0, -34.0)

    def _fit_zoom(self) -> float:
        bounds = self._virtual_bounds()
        content = self._content_rect()
        if bounds.isEmpty() or content.isEmpty():
            return 1.0
        return self._bounded_zoom(
            min(content.width() / bounds.width(), content.height() / bounds.height())
        )

    def fit_to_view(self) -> None:
        bounds = self._virtual_bounds()
        if not bounds.isEmpty():
            self._view_center = bounds.center()
        self._view_mode = "fit"
        self.update()
        self._emit_view_changed()

    def actual_size(self) -> None:
        bounds = self._virtual_bounds()
        if bounds.isEmpty():
            return
        self._view_center = bounds.center()
        self._view_zoom = 1.0
        self._view_mode = "actual"
        self.update()
        self._emit_view_changed()

    def center_view(self) -> None:
        bounds = self._virtual_bounds()
        if bounds.isEmpty():
            return
        self._view_center = bounds.center()
        self.update()

    def set_view_zoom(
        self,
        zoom: float,
        *,
        anchor: QPointF | None = None,
        mode: str = "custom",
    ) -> None:
        bounds = self._virtual_bounds()
        content = self._content_rect()
        if bounds.isEmpty() or content.isEmpty():
            return
        old_scale, old_origin, _bounds = self._display_transform()
        anchor_position = QPointF(anchor) if anchor is not None else content.center()
        image_anchor = QPointF(
            (anchor_position.x() - old_origin.x()) / max(old_scale, 1e-9),
            (anchor_position.y() - old_origin.y()) / max(old_scale, 1e-9),
        )
        self._view_zoom = self._bounded_zoom(zoom)
        self._view_mode = mode if mode in {"actual", "custom"} else "custom"
        self._view_center = QPointF(
            image_anchor.x()
            - ((anchor_position.x() - content.center().x()) / self._view_zoom),
            image_anchor.y()
            - ((anchor_position.y() - content.center().y()) / self._view_zoom),
        )
        self._clamp_view_center()
        self.update()
        self._emit_view_changed()

    def zoom_in(self) -> None:
        self.set_view_zoom(self.view_zoom() * 1.25)

    def zoom_out(self) -> None:
        self.set_view_zoom(self.view_zoom() / 1.25)

    def reset_view(self) -> None:
        self._overlay_opacity = 0.52
        self._split_fraction = 0.5
        self.set_pan_mode(False)
        self.fit_to_view()

    def _emit_view_changed(self) -> None:
        self.viewChanged.emit(self.view_zoom(), self._view_mode)

    def _clamp_view_center(self) -> None:
        bounds = self._virtual_bounds()
        content = self._content_rect()
        if bounds.isEmpty() or content.isEmpty() or self._view_mode == "fit":
            return
        scale = max(self.view_zoom(), 1e-9)
        keep_visible = 28.0
        horizontal_slack = max(0.0, (content.width() / 2.0) - keep_visible) / scale
        vertical_slack = max(0.0, (content.height() / 2.0) - keep_visible) / scale
        self._view_center.setX(
            max(
                bounds.left() - horizontal_slack,
                min(bounds.right() + horizontal_slack, self._view_center.x()),
            )
        )
        self._view_center.setY(
            max(
                bounds.top() - vertical_slack,
                min(bounds.bottom() + vertical_slack, self._view_center.y()),
            )
        )

    def set_offset(self, x: int, y: int, *, emit: bool = False) -> None:
        changed = (int(x), int(y)) != (self._offset_x, self._offset_y)
        self._offset_x = int(x)
        self._offset_y = int(y)
        self.update()
        if emit and changed:
            self.offsetChanged.emit(self._offset_x, self._offset_y)

    def _virtual_bounds(self) -> QRectF:
        if self._reference.isNull() or self._candidate.isNull():
            return QRectF()
        candidate_x = self._nominal_dx + self._offset_x
        candidate_y = self._nominal_dy + self._offset_y
        return QRectF(0, 0, self._reference.width(), self._reference.height()).united(
            QRectF(candidate_x, candidate_y, self._candidate.width(), self._candidate.height())
        )

    def _display_transform(self) -> tuple[float, QPointF, QRectF]:
        bounds = self._virtual_bounds()
        content = self._content_rect()
        if bounds.isEmpty() or content.isEmpty():
            return 1.0, QPointF(), bounds
        scale = self.view_zoom()
        center = bounds.center() if self._view_mode == "fit" else self._view_center
        origin = QPointF(
            content.center().x() - (center.x() * scale),
            content.center().y() - (center.y() * scale),
        )
        return scale, origin, bounds

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0B1220"))
        if self._reference.isNull() or self._candidate.isNull():
            painter.setPen(QColor("#94A3B8"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "请选择有效的相邻视场")
            return
        scale, origin, bounds = self._display_transform()
        # Downsampling benefits from smoothing, while magnified pixels should
        # remain crisp enough for seam and texture inspection.
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            scale < 1.0,
        )
        content = self._content_rect()
        painter.save()
        painter.setClipRect(content)
        painter.translate(origin)
        painter.scale(scale, scale)
        candidate_x = self._nominal_dx + self._offset_x
        candidate_y = self._nominal_dy + self._offset_y
        painter.drawImage(QPointF(0, 0), self._reference)
        if self._mode == "alpha":
            painter.setOpacity(self._overlay_opacity)
            painter.drawImage(QPointF(candidate_x, candidate_y), self._candidate)
        elif self._mode == "difference":
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Difference)
            painter.drawImage(QPointF(candidate_x, candidate_y), self._candidate)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.restore()
        else:
            painter.restore()
            displayed_bounds = QRectF(
                origin.x() + (bounds.left() * scale),
                origin.y() + (bounds.top() * scale),
                bounds.width() * scale,
                bounds.height() * scale,
            ).intersected(content)
            split_x = displayed_bounds.left() + (
                displayed_bounds.width() * self._split_fraction
            )
            painter.save()
            painter.setClipRect(
                QRectF(split_x, content.top(), content.right() - split_x, content.height())
            )
            painter.translate(origin)
            painter.scale(scale, scale)
            painter.drawImage(QPointF(candidate_x, candidate_y), self._candidate)
            painter.restore()
            painter.setPen(QPen(QColor("#F8FAFC"), 1))
            painter.drawLine(
                QPointF(split_x, displayed_bounds.top()),
                QPointF(split_x, displayed_bounds.bottom()),
            )
        if self._mode == "alpha":
            painter.restore()
        painter.setPen(QColor("#94A3B8"))
        if self._mode == "split":
            action = "左参考/右待校准"
        elif self._pan_mode or self._space_pan:
            action = "平移查看"
        else:
            action = "左键拖动校准"
        status = (
            f"{action} · 滚轮缩放 {self.view_zoom() * 100:.0f}% · "
            f"ΔX={self._offset_x:+d}px  ΔY={self._offset_y:+d}px"
        )
        status = painter.fontMetrics().elidedText(
            status,
            Qt.TextElideMode.ElideRight,
            max(1, self.width() - 24),
        )
        painter.drawText(
            12,
            self.height() - 10,
            status,
        )

    def _uses_pan_gesture(self, button: Qt.MouseButton) -> bool:
        return button == Qt.MouseButton.MiddleButton or (
            button == Qt.MouseButton.LeftButton and (self._pan_mode or self._space_pan)
        )

    def _update_cursor(self) -> None:
        if self._view_drag_origin is not None:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._pan_mode or self._space_pan:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._uses_pan_gesture(event.button()) and self.has_pair():
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._view_drag_origin = QPointF(event.position())
            self._view_drag_center = QPointF(self._view_center)
            self._update_cursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and not self._candidate.isNull():
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._offset_drag_origin = QPointF(event.position())
            self._offset_drag_value = (self._offset_x, self._offset_y)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._view_drag_origin is not None:
            scale = max(self.view_zoom(), 1e-9)
            delta = event.position() - self._view_drag_origin
            if self._view_mode == "fit":
                self._view_mode = "custom"
                self._view_zoom = scale
            self._view_center = QPointF(
                self._view_drag_center.x() - (delta.x() / scale),
                self._view_drag_center.y() - (delta.y() / scale),
            )
            self._clamp_view_center()
            self.update()
            self._emit_view_changed()
            event.accept()
            return
        if self._offset_drag_origin is None:
            super().mouseMoveEvent(event)
            return
        scale, _origin, _bounds = self._display_transform()
        delta = event.position() - self._offset_drag_origin
        self.set_offset(
            self._offset_drag_value[0] + int(round(delta.x() / max(0.01, scale))),
            self._offset_drag_value[1] + int(round(delta.y() / max(0.01, scale))),
            emit=True,
        )
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._view_drag_origin is not None and event.button() in {
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        }:
            self._view_drag_origin = None
            self._update_cursor()
            event.accept()
            return
        if self._offset_drag_origin is not None and event.button() == Qt.MouseButton.LeftButton:
            self._offset_drag_origin = None
            self._update_cursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta == 0 or not self.has_pair():
            super().wheelEvent(event)
            return
        steps = max(-8.0, min(8.0, float(delta) / 120.0))
        self.set_view_zoom(
            self.view_zoom() * (1.2**steps),
            anchor=event.position(),
        )
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = True
            self._update_cursor()
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Plus, Qt.Key.Key_Equal}:
            self.zoom_in()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
            event.accept()
            return
        if event.key() == Qt.Key.Key_0:
            self.fit_to_view()
            event.accept()
            return
        if event.key() == Qt.Key.Key_1:
            self.actual_size()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Home:
            self.center_view()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.escapeRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = False
            self._update_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        self._space_pan = False
        self._view_drag_origin = None
        self._offset_drag_origin = None
        self._update_cursor()
        super().focusOutEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._view_mode == "fit":
            self._emit_view_changed()


class DigitalSlideCalibrationDialog(QDialog):
    _estimateFinished = Signal(int, object, str)

    def __init__(
        self,
        settings: AppSettings,
        *,
        source_path: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("数字切片校准辅助")
        self.resize(1120, 760)
        self.setMinimumSize(860, 640)
        self._settings = settings.normalized_copy()
        self._session = DigitalSlideCalibrationSession()
        self._current_pair: DigitalSlideCalibrationPair | None = None
        self._current_estimate: DigitalSlideCalibrationEstimate | None = None
        self._applied_values: dict[str, object] = {}
        self._estimate_generation = 0
        self._estimate_cancel = Event()
        self._estimate_thread: Thread | None = None
        self._session_cleanup_scheduled = False
        self._preview_focus_mode = False
        self._estimateFinished.connect(self._on_estimate_finished)

        self._heading = QLabel("利用已有切片校准视场拼接", self)
        self._heading.setObjectName("digitalSlideCalibrationTitle")
        self._intro = QLabel(
            "源切片只读。选择同焦层相邻视场后，可拖动或自动估算实际像素步距；"
            "电机步距仅在明确勾选并确认后回填。",
            self,
        )
        self._intro.setWordWrap(True)

        self._source_group = QGroupBox("校准源", self)
        source_layout = QHBoxLayout(self._source_group)
        self._source_label = QLabel("尚未选择 .fdmslide", self._source_group)
        self._source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._source_button = QPushButton("选择切片…", self._source_group)
        self._source_button.clicked.connect(self._choose_source)
        source_layout.addWidget(self._source_label, 1)
        source_layout.addWidget(self._source_button)

        self._selection_group = QGroupBox("视场选择", self)
        selection_form = QFormLayout(self._selection_group)
        self._focus_combo = NoWheelComboBox(self._selection_group)
        self._axis_combo = NoWheelComboBox(self._selection_group)
        self._axis_combo.addItem("横向相邻视场", CALIBRATION_AXIS_X)
        self._axis_combo.addItem("纵向相邻视场", CALIBRATION_AXIS_Y)
        self._pair_combo = NoWheelComboBox(self._selection_group)
        self._mode_combo = NoWheelComboBox(self._selection_group)
        self._mode_combo.addItem("半透明叠加", "alpha")
        self._mode_combo.addItem("分割线比较", "split")
        self._mode_combo.addItem("差异混合", "difference")
        selection_form.addRow("焦层", self._focus_combo)
        selection_form.addRow("方向", self._axis_combo)
        selection_form.addRow("相邻视场", self._pair_combo)
        selection_form.addRow("比较方式", self._mode_combo)

        self._preview = CalibrationPairPreview(self)
        self._preview.offsetChanged.connect(self._on_preview_offset_changed)
        self._preview.viewChanged.connect(self._on_preview_view_changed)
        self._preview.escapeRequested.connect(self._on_preview_escape_requested)

        self._adjustment_group = QGroupBox("平移与计算", self)
        adjustment_layout = QVBoxLayout(self._adjustment_group)
        offset_form = QFormLayout()
        self._offset_x_spin = NoWheelSpinBox(self._adjustment_group)
        self._offset_y_spin = NoWheelSpinBox(self._adjustment_group)
        for spin in (self._offset_x_spin, self._offset_y_spin):
            spin.setRange(-100_000, 100_000)
            spin.setSuffix(" px")
        self._offset_x_spin.valueChanged.connect(self._on_offset_spin_changed)
        self._offset_y_spin.valueChanged.connect(self._on_offset_spin_changed)
        offset_form.addRow("X 微调", self._offset_x_spin)
        offset_form.addRow("Y 微调", self._offset_y_spin)
        adjustment_layout.addLayout(offset_form)
        nudge_row = QHBoxLayout()
        for label, dx, dy in (
            ("X−10", -10, 0), ("X−1", -1, 0), ("X+1", 1, 0), ("X+10", 10, 0),
            ("Y−10", 0, -10), ("Y−1", 0, -1), ("Y+1", 0, 1), ("Y+10", 0, 10),
        ):
            button = QPushButton(label, self._adjustment_group)
            button.clicked.connect(lambda _checked=False, x=dx, y=dy: self._nudge(x, y))
            nudge_row.addWidget(button)
        adjustment_layout.addLayout(nudge_row)
        action_row = QHBoxLayout()
        self._auto_button = QPushButton("自动估算（最多 10 对）", self._adjustment_group)
        self._auto_button.clicked.connect(self._start_auto_estimate)
        reset_button = QPushButton("重置微调", self._adjustment_group)
        reset_button.clicked.connect(lambda: self._set_offsets(0, 0))
        action_row.addWidget(self._auto_button)
        action_row.addWidget(reset_button)
        action_row.addStretch(1)
        adjustment_layout.addLayout(action_row)
        # Estimates range from a compact manual summary to several warning
        # lines.  A bounded read-only text area keeps long diagnostics
        # accessible without allowing the layout to overlap the guarded motor
        # option on short/high-DPI screens.
        self._result_label = QPlainTextEdit(self._adjustment_group)
        self._result_label.setPlainText("等待选择相邻视场")
        self._result_label.setReadOnly(True)
        self._result_label.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._result_label.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._result_label.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._result_label.setFrameShape(QFrame.Shape.NoFrame)
        self._result_label.setMinimumHeight(64)
        self._result_label.setMaximumHeight(96)
        self._result_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self._result_label.setStyleSheet(
            "QPlainTextEdit { background: transparent; border: none; padding: 0px; }"
        )
        adjustment_layout.addWidget(self._result_label)
        self._apply_stage_checkbox = QCheckBox(
            "同时应用电机自动采集步距（需要确认）",
            self._adjustment_group,
        )
        self._apply_stage_checkbox.setChecked(False)
        self._apply_stage_checkbox.setEnabled(False)
        adjustment_layout.addWidget(self._apply_stage_checkbox)

        self._selection_container = QWidget(self)
        selection_row = QHBoxLayout(self._selection_container)
        selection_row.setContentsMargins(0, 0, 0, 0)
        selection_row.addWidget(self._selection_group, 1)
        selection_row.addWidget(self._adjustment_group, 2)

        self._preview_controls = self._build_preview_controls()

        self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        cancel_button = self._button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button is not None:
            cancel_button.setText("关闭")
        self._apply_button = self._button_box.addButton("应用到当前配置", QDialogButtonBox.ButtonRole.AcceptRole)
        self._apply_button.setEnabled(False)
        self._apply_button.clicked.connect(self._apply_result)
        self._button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)
        layout.addWidget(self._heading)
        layout.addWidget(self._intro)
        layout.addWidget(self._source_group)
        layout.addWidget(self._selection_container)
        layout.addWidget(self._preview_controls)
        layout.addWidget(self._preview, 1)
        layout.addWidget(self._button_box)
        self.setStyleSheet("QLabel#digitalSlideCalibrationTitle { font-size: 18px; font-weight: 700; }")

        self._focus_combo.currentIndexChanged.connect(self._refresh_pairs)
        self._axis_combo.currentIndexChanged.connect(self._refresh_pairs)
        self._pair_combo.currentIndexChanged.connect(self._load_selected_pair)
        self._mode_combo.currentIndexChanged.connect(self._on_preview_mode_changed)
        self._on_preview_mode_changed()
        self._update_preview_controls_enabled()
        self._on_preview_view_changed(self._preview.view_zoom(), self._preview.view_mode())
        self._enforce_minimum_layout_size()
        if source_path is not None and str(source_path).strip():
            self._open_source(Path(source_path))

    def applied_profile_values(self) -> dict[str, object]:
        return dict(self._applied_values)

    def _enforce_minimum_layout_size(self) -> None:
        layout = self.layout()
        if layout is not None:
            layout.activate()
        hint = self.minimumSizeHint()
        # The application theme can increase form-row and checkbox heights.
        # Respect the polished layout hint instead of allowing Qt to compress
        # the estimate and guarded stage option into the same pixels.
        self.setMinimumSize(max(860, hint.width()), max(640, hint.height()))

    def _build_preview_controls(self) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("calibrationPreviewControls")
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(frame)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        title = QLabel("细节查看", frame)
        self._zoom_out_button = QPushButton("−", frame)
        self._zoom_out_button.setFixedWidth(34)
        self._zoom_out_button.setToolTip("缩小预览（−）")
        self._zoom_out_button.clicked.connect(self._preview.zoom_out)

        self._zoom_spin = NoWheelSpinBox(frame)
        self._zoom_spin.setRange(1, 3200)
        self._zoom_spin.setSuffix(" %")
        self._zoom_spin.setKeyboardTracking(False)
        self._zoom_spin.setFixedWidth(88)
        self._zoom_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._zoom_spin.setAccessibleName("预览缩放比例")
        self._zoom_spin.setToolTip(
            "预览显示比例；100% 表示一个切片像素对应一个界面逻辑像素。"
        )
        self._zoom_spin.valueChanged.connect(self._on_preview_zoom_value_changed)

        self._zoom_in_button = QPushButton("+", frame)
        self._zoom_in_button.setFixedWidth(34)
        self._zoom_in_button.setToolTip("放大预览（+）")
        self._zoom_in_button.clicked.connect(self._preview.zoom_in)

        self._fit_button = QPushButton("适合窗口", frame)
        self._fit_button.setCheckable(True)
        self._fit_button.setToolTip("完整显示两个视场（0）")
        self._fit_button.toggled.connect(self._on_preview_fit_toggled)

        self._actual_button = QPushButton("1:1", frame)
        self._actual_button.setCheckable(True)
        self._actual_button.setToolTip("按切片存储像素原始大小显示（1）")
        self._actual_button.toggled.connect(self._on_preview_actual_toggled)

        self._center_button = QPushButton("居中", frame)
        self._center_button.setToolTip("保持当前缩放比例并将视场重新居中（Home）")
        self._center_button.clicked.connect(self._preview.center_view)

        self._pan_button = QPushButton("平移查看", frame)
        self._pan_button.setCheckable(True)
        self._pan_button.setToolTip(
            "启用后左键拖动画布；也可随时按住空格并左键拖动，或使用中键。"
        )
        self._pan_button.toggled.connect(self._preview.set_pan_mode)

        self._reset_view_button = QPushButton("复位视图", frame)
        self._reset_view_button.setToolTip(
            "恢复适应窗口、默认透明度和居中的分割线；不会改变校准偏移。"
        )
        self._reset_view_button.clicked.connect(self._reset_preview_view)

        self._focus_preview_button = QPushButton("展开预览", frame)
        self._focus_preview_button.setCheckable(True)
        self._focus_preview_button.setToolTip(
            "隐藏上方参数，使用窗口主体查看图像细节；按 Esc 返回。"
        )
        self._focus_preview_button.toggled.connect(self._set_preview_focus_mode)

        grid.addWidget(title, 0, 0)
        grid.addWidget(self._zoom_out_button, 0, 1)
        grid.addWidget(self._zoom_spin, 0, 2)
        grid.addWidget(self._zoom_in_button, 0, 3)
        grid.addWidget(self._fit_button, 0, 4)
        grid.addWidget(self._actual_button, 0, 5)
        grid.addWidget(self._center_button, 0, 6)
        grid.addWidget(self._pan_button, 0, 7)
        grid.addWidget(self._reset_view_button, 0, 8)
        grid.setColumnStretch(9, 1)
        grid.addWidget(self._focus_preview_button, 0, 10)

        self._opacity_label = QLabel("叠加透明度", frame)
        self._opacity_slider = NoWheelSlider(Qt.Orientation.Horizontal, frame)
        self._opacity_slider.setRange(5, 95)
        self._opacity_slider.setValue(52)
        self._opacity_slider.setMinimumWidth(120)
        self._opacity_slider.setAccessibleName("叠加透明度")
        self._opacity_slider.setToolTip("调整半透明叠加模式中待校准视场的透明度。")
        self._opacity_value_label = QLabel("52%", frame)
        self._opacity_value_label.setMinimumWidth(42)
        self._opacity_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._opacity_label.setBuddy(self._opacity_slider)
        self._opacity_slider.valueChanged.connect(self._on_preview_opacity_changed)

        self._split_label = QLabel("分割线位置", frame)
        self._split_slider = NoWheelSlider(Qt.Orientation.Horizontal, frame)
        self._split_slider.setRange(5, 95)
        self._split_slider.setValue(50)
        self._split_slider.setMinimumWidth(120)
        self._split_slider.setAccessibleName("分割线位置")
        self._split_slider.setToolTip("调整分割线比较模式中参考视场与待校准视场的分界。")
        self._split_value_label = QLabel("50%", frame)
        self._split_value_label.setMinimumWidth(42)
        self._split_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._split_label.setBuddy(self._split_slider)
        self._split_slider.valueChanged.connect(self._on_preview_split_changed)

        grid.addWidget(self._opacity_label, 1, 0)
        grid.addWidget(self._opacity_slider, 1, 1, 1, 4)
        grid.addWidget(self._opacity_value_label, 1, 5)
        grid.addWidget(self._split_label, 1, 6)
        grid.addWidget(self._split_slider, 1, 7, 1, 3)
        grid.addWidget(self._split_value_label, 1, 10)
        return frame

    def _on_preview_zoom_value_changed(self, percentage: int) -> None:
        self._preview.set_view_zoom(max(1, int(percentage)) / 100.0)

    def _on_preview_fit_toggled(self, enabled: bool) -> None:
        if enabled:
            self._preview.fit_to_view()
        elif self._preview.view_mode() == "fit":
            self._fit_button.blockSignals(True)
            self._fit_button.setChecked(True)
            self._fit_button.blockSignals(False)

    def _on_preview_actual_toggled(self, enabled: bool) -> None:
        if enabled:
            self._preview.actual_size()
        elif self._preview.view_mode() == "actual":
            self._actual_button.blockSignals(True)
            self._actual_button.setChecked(True)
            self._actual_button.blockSignals(False)

    def _on_preview_view_changed(self, zoom: float, mode: str) -> None:
        percentage = max(
            self._zoom_spin.minimum(),
            min(self._zoom_spin.maximum(), int(round(float(zoom) * 100.0))),
        )
        self._zoom_spin.blockSignals(True)
        self._zoom_spin.setValue(percentage)
        self._zoom_spin.blockSignals(False)
        self._fit_button.blockSignals(True)
        self._actual_button.blockSignals(True)
        self._fit_button.setChecked(mode == "fit")
        self._actual_button.setChecked(mode == "actual")
        self._fit_button.blockSignals(False)
        self._actual_button.blockSignals(False)

    def _on_preview_opacity_changed(self, percentage: int) -> None:
        self._opacity_value_label.setText(f"{int(percentage)}%")
        self._preview.set_overlay_opacity(int(percentage) / 100.0)

    def _on_preview_split_changed(self, percentage: int) -> None:
        self._split_value_label.setText(f"{int(percentage)}%")
        self._preview.set_split_fraction(int(percentage) / 100.0)

    def _on_preview_mode_changed(self, _index: int = -1) -> None:
        self._preview.set_mode(str(self._mode_combo.currentData() or "alpha"))
        self._update_preview_controls_enabled()

    def _update_preview_controls_enabled(self) -> None:
        available = self._preview.has_pair()
        for widget in (
            self._zoom_out_button,
            self._zoom_spin,
            self._zoom_in_button,
            self._fit_button,
            self._actual_button,
            self._center_button,
            self._pan_button,
            self._reset_view_button,
            self._focus_preview_button,
        ):
            widget.setEnabled(available)
        alpha_enabled = available and self._preview.mode() == "alpha"
        split_enabled = available and self._preview.mode() == "split"
        for widget in (
            self._opacity_label,
            self._opacity_slider,
            self._opacity_value_label,
        ):
            widget.setEnabled(alpha_enabled)
        for widget in (
            self._split_label,
            self._split_slider,
            self._split_value_label,
        ):
            widget.setEnabled(split_enabled)

    def _reset_preview_view(self) -> None:
        self._opacity_slider.setValue(52)
        self._split_slider.setValue(50)
        self._pan_button.setChecked(False)
        self._preview.reset_view()

    def _set_preview_focus_mode(self, enabled: bool) -> None:
        enabled = bool(enabled and self._preview.has_pair())
        if self._focus_preview_button.isChecked() != enabled:
            self._focus_preview_button.blockSignals(True)
            self._focus_preview_button.setChecked(enabled)
            self._focus_preview_button.blockSignals(False)
        self._preview_focus_mode = enabled
        for widget in (
            self._heading,
            self._intro,
            self._source_group,
            self._selection_container,
        ):
            widget.setVisible(not enabled)
        self._focus_preview_button.setText("返回校准" if enabled else "展开预览")
        if enabled:
            self._preview.setFocus(Qt.FocusReason.ShortcutFocusReason)
        QTimer.singleShot(0, self._refresh_preview_after_layout_change)

    def _refresh_preview_after_layout_change(self) -> None:
        if self._preview.view_mode() == "fit":
            self._preview.fit_to_view()
        else:
            self._preview.update()

    def _on_preview_escape_requested(self) -> None:
        if self._preview_focus_mode:
            self._focus_preview_button.setChecked(False)
        else:
            self.reject()

    def _choose_source(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择用于校准的数字切片",
            str(self._session.source_path.parent) if self._session.source_path else "",
            f"数字化切片 (*{DIGITAL_SLIDE_SUFFIX})",
        )
        if path:
            self._open_source(Path(path))

    def _open_source(self, path: Path) -> None:
        self._invalidate_auto_estimate()
        if path.suffix.lower() != DIGITAL_SLIDE_SUFFIX:
            self._show_error("请选择 .fdmslide 文件。")
            return
        progress = QProgressDialog("正在准备数字切片…", "取消", 0, 1000, self)
        progress.setWindowTitle("校准辅助")
        progress.setMinimumDuration(300)
        cancelled = Event()
        progress.canceled.connect(cancelled.set)

        def update(copied: int, total: int) -> None:
            progress.setMaximum(1000)
            progress.setValue(
                1000 if total <= 0 else max(0, min(1000, int(copied * 1000 / total)))
            )
            QApplication.processEvents()

        try:
            manifest = self._session.open(
                path,
                progress_callback=update,
                cancellation_requested=cancelled.is_set,
            )
        except DigitalSlideCacheCancelled:
            progress.close()
            return
        except Exception as exc:  # noqa: BLE001 - normalize file/cache errors
            progress.close()
            self._show_error(str(exc))
            return
        progress.close()
        self._source_label.setText(str(path))
        self._focus_combo.blockSignals(True)
        self._focus_combo.clear()
        for index, focus_z in enumerate(manifest.focus_levels):
            self._focus_combo.addItem(f"第 {index + 1} 层（Z={focus_z}）", index)
        middle = max(0, len(manifest.focus_levels) // 2)
        self._focus_combo.setCurrentIndex(middle if self._focus_combo.count() else -1)
        self._focus_combo.blockSignals(False)
        self._refresh_pairs()

    def _refresh_pairs(self, _index: int = -1) -> None:
        self._invalidate_auto_estimate()
        if self._focus_combo.currentIndex() < 0:
            return
        focus_index = int(self._focus_combo.currentData())
        axis = str(self._axis_combo.currentData())
        pairs = self._session.adjacent_pairs(focus_index, axis)
        self._pair_combo.blockSignals(True)
        self._pair_combo.clear()
        for index, pair in enumerate(pairs):
            label = (
                f"{index + 1}: ({pair.reference.x}, {pair.reference.y}) → "
                f"({pair.candidate.x}, {pair.candidate.y})"
            )
            self._pair_combo.addItem(label, pair)
        self._pair_combo.setCurrentIndex(0 if pairs else -1)
        self._pair_combo.blockSignals(False)
        self._load_selected_pair()

    def _load_selected_pair(self, _index: int = -1) -> None:
        self._invalidate_auto_estimate()
        # The offsets are calibration parameters shared while the user samples
        # different focus levels, axes, and adjacent field pairs.  Loading a
        # different pair must replace only the compared images; resetting these
        # values here made iterative calibration lose the user's work.
        offset_x = self._offset_x_spin.value()
        offset_y = self._offset_y_spin.value()
        pair = self._pair_combo.currentData()
        self._current_pair = pair if isinstance(pair, DigitalSlideCalibrationPair) else None
        self._current_estimate = None
        self._apply_stage_checkbox.setChecked(False)
        self._apply_stage_checkbox.setEnabled(False)
        if self._current_pair is None:
            self._preview.set_pair(QImage(), QImage(), nominal_dx=0, nominal_dy=0)
            self._result_label.setPlainText("当前焦层在所选方向没有相邻视场。")
            self._apply_button.setEnabled(False)
            if self._preview_focus_mode:
                self._focus_preview_button.setChecked(False)
            self._update_preview_controls_enabled()
            return
        reference, candidate = self._session.read_pair(self._current_pair)
        self._preview.set_pair(
            reference,
            candidate,
            nominal_dx=self._current_pair.nominal_dx,
            nominal_dy=self._current_pair.nominal_dy,
        )
        self._set_offsets(offset_x, offset_y)
        self._update_preview_controls_enabled()

    def _set_offsets(self, x: int, y: int) -> None:
        self._offset_x_spin.blockSignals(True)
        self._offset_y_spin.blockSignals(True)
        self._offset_x_spin.setValue(int(x))
        self._offset_y_spin.setValue(int(y))
        self._offset_x_spin.blockSignals(False)
        self._offset_y_spin.blockSignals(False)
        self._preview.set_offset(int(x), int(y))
        self._update_manual_result()

    def _on_preview_offset_changed(self, x: int, y: int) -> None:
        self._offset_x_spin.blockSignals(True)
        self._offset_y_spin.blockSignals(True)
        self._offset_x_spin.setValue(int(x))
        self._offset_y_spin.setValue(int(y))
        self._offset_x_spin.blockSignals(False)
        self._offset_y_spin.blockSignals(False)
        self._update_manual_result()

    def _on_offset_spin_changed(self, _value: int) -> None:
        self._preview.set_offset(self._offset_x_spin.value(), self._offset_y_spin.value())
        self._update_manual_result()

    def _nudge(self, dx: int, dy: int) -> None:
        self._set_offsets(
            self._offset_x_spin.value() + int(dx),
            self._offset_y_spin.value() + int(dy),
        )

    def _target_frame_size(self, pair: DigitalSlideCalibrationPair) -> tuple[int, int]:
        width = int(pair.reference.width)
        height = int(pair.reference.height)
        if width <= 0 or height <= 0:
            return 0, 0
        maximum = int(self._settings.digital_slide_capture_max_width)
        if maximum <= 0 or width <= maximum:
            return width, height
        return maximum, max(1, int(round(height * maximum / max(1, width))))

    def _manual_estimate(self) -> DigitalSlideCalibrationEstimate | None:
        pair = self._current_pair
        if pair is None:
            return None
        axis = str(self._axis_combo.currentData())
        dx = pair.nominal_dx + self._offset_x_spin.value()
        dy = pair.nominal_dy + self._offset_y_spin.value()
        source = (pair.reference.width, pair.reference.height)
        target = self._target_frame_size(pair)
        if min(*source, *target) <= 0:
            return None
        primary_scale = target[0] / max(1, source[0]) if axis == CALIBRATION_AXIS_X else target[1] / max(1, source[1])
        cross_scale = target[1] / max(1, source[1]) if axis == CALIBRATION_AXIS_X else target[0] / max(1, source[0])
        primary = abs(dx if axis == CALIBRATION_AXIS_X else dy) * primary_scale
        cross = (dy if axis == CALIBRATION_AXIS_X else dx) * cross_scale
        stage_delta = pair.stage_dx if axis == CALIBRATION_AXIS_X else pair.stage_dy
        pixels_per_step = (primary / abs(stage_delta)) if stage_delta else None
        current_step = (
            self._settings.digital_slide_x_stage_step
            if axis == CALIBRATION_AXIS_X
            else self._settings.digital_slide_y_stage_step
        )
        suggested: int | None = None
        if pixels_per_step and pixels_per_step > 0:
            frame_axis = target[0] if axis == CALIBRATION_AXIS_X else target[1]
            target_stride = frame_axis * (1.0 - self._settings.digital_slide_overlap_percent / 100.0)
            magnitude = max(1, int(round(target_stride / pixels_per_step)))
            suggested = -magnitude if current_step < 0 else magnitude
        warnings = () if stage_delta else ("该视场对缺少有效电机位移",)
        return DigitalSlideCalibrationEstimate(
            axis=axis,
            primary_stride_px=float(primary),
            cross_axis_drift_px=float(cross),
            pixels_per_step=pixels_per_step,
            suggested_stage_step=suggested,
            confidence=0.0,
            sample_count=1,
            accepted_count=1,
            source_frame_size=source,
            target_frame_size=target,
            warnings=warnings,
        )

    def _update_manual_result(self) -> None:
        self._current_estimate = self._manual_estimate()
        if self._current_pair is not None and self._current_estimate is None:
            self._result_label.setPlainText(
                "无法确认源视场或目标采集尺寸，不能直接应用校准结果。"
            )
            self._apply_button.setEnabled(False)
            return
        self._show_estimate(self._current_estimate, automatic=False)

    def _show_estimate(
        self,
        estimate: DigitalSlideCalibrationEstimate | None,
        *,
        automatic: bool,
    ) -> None:
        if estimate is None:
            self._result_label.setPlainText("等待选择相邻视场")
            self._apply_button.setEnabled(False)
            return
        axis_name = "X" if estimate.axis == CALIBRATION_AXIS_X else "Y"
        source = f"自动 {estimate.accepted_count}/{estimate.sample_count} 对" if automatic else "手动当前视场对"
        lines = [
            f"{source} | {axis_name} 像素步距 {estimate.primary_stride_px:.2f} px",
            f"交叉轴漂移 {estimate.cross_axis_drift_px:+.2f} px | "
            f"尺寸 {estimate.source_frame_size[0]}×{estimate.source_frame_size[1]} → "
            f"{estimate.target_frame_size[0]}×{estimate.target_frame_size[1]}",
        ]
        if estimate.source_frame_size != estimate.target_frame_size:
            lines.append(
                "尺寸换算比例 "
                f"X {estimate.target_frame_size[0] / max(1, estimate.source_frame_size[0]):.4f}，"
                f"Y {estimate.target_frame_size[1] / max(1, estimate.source_frame_size[1]):.4f}"
            )
        conversion_parts: list[str] = []
        if estimate.pixels_per_step is not None:
            conversion_parts.append(f"换算 {estimate.pixels_per_step:.5f} px/step")
        if estimate.suggested_stage_step is not None:
            conversion_parts.append(f"电机步距建议 {estimate.suggested_stage_step} steps")
        if conversion_parts:
            lines.append(" | ".join(conversion_parts))
        if automatic:
            lines.append(f"置信度 {estimate.confidence:.2f}")
            if estimate.confidence < 0.12:
                lines.append("注意：自动结果置信度不足，请改用手动像素校准。")
        if estimate.warnings:
            lines.append("注意：" + "；".join(estimate.warnings))
        self._result_label.setPlainText("\n".join(lines))
        pixel_result_safe = bool(
            estimate.can_apply_pixel_stride
            and (not automatic or estimate.confidence >= 0.12)
        )
        self._apply_button.setEnabled(pixel_result_safe)
        self._apply_stage_checkbox.setEnabled(estimate.can_apply_stage_step)
        if not estimate.can_apply_stage_step:
            self._apply_stage_checkbox.setChecked(False)

    def _start_auto_estimate(self) -> None:
        if self._focus_combo.currentIndex() < 0 or self._estimate_thread is not None:
            return
        focus_index = int(self._focus_combo.currentData())
        axis = str(self._axis_combo.currentData())
        pair = self._current_pair
        if pair is None:
            return
        target_size = self._target_frame_size(pair)
        current_step = (
            self._settings.digital_slide_x_stage_step
            if axis == CALIBRATION_AXIS_X
            else self._settings.digital_slide_y_stage_step
        )
        working_path = self._session.working_path
        if working_path is None:
            return
        self._estimate_generation += 1
        generation = self._estimate_generation
        self._estimate_cancel = Event()
        cancel_event = self._estimate_cancel
        self._auto_button.setEnabled(False)
        self._auto_button.setText("正在估算…")
        self._source_button.setEnabled(False)
        dialog_ref = ref(self)

        def run() -> None:
            session = DigitalSlideCalibrationSession()
            try:
                session.open(working_path, cancellation_requested=cancel_event.is_set)
                estimate = session.estimate(
                    focus_index=focus_index,
                    axis=axis,
                    target_frame_size=target_size,
                    target_overlap_percent=self._settings.digital_slide_overlap_percent,
                    current_stage_step=current_step,
                    maximum_pairs=10,
                    cancellation_requested=cancel_event.is_set,
                )
                error = ""
            except Exception as exc:  # noqa: BLE001 - worker publishes normalized error
                estimate = None
                error = str(exc)
            finally:
                session.close()
            dialog = dialog_ref()
            if dialog is not None and is_qobject_valid(dialog):
                dialog._estimateFinished.emit(generation, estimate, error)

        self._estimate_thread = Thread(target=run, name="fdm-slide-calibration", daemon=True)
        self._estimate_thread.start()

    def _on_estimate_finished(self, generation: int, estimate: object, error: str) -> None:
        self._estimate_thread = None
        self._auto_button.setEnabled(True)
        self._auto_button.setText("自动估算（最多 10 对）")
        self._source_button.setEnabled(True)
        if generation != self._estimate_generation:
            return
        if error:
            self._show_error(error)
            self._update_manual_result()
            return
        if not isinstance(estimate, DigitalSlideCalibrationEstimate):
            self._show_error("自动估算未返回有效结果。")
            return
        self._current_estimate = estimate
        pair = self._current_pair
        if pair is not None:
            scale_x = estimate.target_frame_size[0] / max(1, estimate.source_frame_size[0])
            scale_y = estimate.target_frame_size[1] / max(1, estimate.source_frame_size[1])
            if estimate.axis == CALIBRATION_AXIS_X:
                actual_dx = estimate.primary_stride_px / max(0.0001, scale_x)
                actual_dy = estimate.cross_axis_drift_px / max(0.0001, scale_y)
            else:
                actual_dx = estimate.cross_axis_drift_px / max(0.0001, scale_x)
                actual_dy = estimate.primary_stride_px / max(0.0001, scale_y)
            self._set_offsets(
                int(round(actual_dx - pair.nominal_dx)),
                int(round(actual_dy - pair.nominal_dy)),
            )
            self._current_estimate = estimate
        self._show_estimate(estimate, automatic=True)

    def _invalidate_auto_estimate(self) -> None:
        thread = self._estimate_thread
        if thread is None or not thread.is_alive():
            return
        self._estimate_generation += 1
        self._estimate_cancel.set()

    def _apply_result(self) -> None:
        estimate = self._current_estimate
        if estimate is None or not estimate.can_apply_pixel_stride:
            return
        values: dict[str, object] = {"digital_slide_pixel_stride_mode": "manual_pixels"}
        if estimate.axis == CALIBRATION_AXIS_X:
            values["digital_slide_x_pixel_stride"] = max(1, int(round(estimate.primary_stride_px)))
            stage_key = "digital_slide_x_stage_step"
        else:
            values["digital_slide_y_pixel_stride"] = max(1, int(round(estimate.primary_stride_px)))
            stage_key = "digital_slide_y_stage_step"
        if self._apply_stage_checkbox.isChecked() and estimate.suggested_stage_step is not None:
            from PySide6.QtWidgets import QMessageBox

            response = QMessageBox.question(
                self,
                "应用电机步距",
                f"将 {'X' if estimate.axis == CALIBRATION_AXIS_X else 'Y'} 自动采集步距改为 {estimate.suggested_stage_step} steps？\n"
                "该参数会改变后续真机运动距离。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
            values[stage_key] = int(estimate.suggested_stage_step)
        self._applied_values = values
        self.accept()

    def _show_error(self, message: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(self, "校准辅助", str(message))

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape and self._preview_focus_mode:
            self._focus_preview_button.setChecked(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def reject(self) -> None:
        self._shutdown_calibration_session()
        super().reject()

    def accept(self) -> None:
        self._shutdown_calibration_session()
        super().accept()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._shutdown_calibration_session()
        super().closeEvent(event)

    def _shutdown_calibration_session(self) -> None:
        if self._session_cleanup_scheduled:
            return
        self._session_cleanup_scheduled = True
        self._estimate_generation += 1
        self._estimate_cancel.set()
        estimate_thread = self._estimate_thread
        self._session.close_store()
        if estimate_thread is None or not estimate_thread.is_alive():
            self._session.close()
            return
        session = self._session

        def cleanup_after_estimate() -> None:
            estimate_thread.join()
            session.close()

        Thread(
            target=cleanup_after_estimate,
            name="fdm-slide-calibration-cleanup",
            daemon=True,
        ).start()


__all__ = ["CalibrationPairPreview", "DigitalSlideCalibrationDialog"]
