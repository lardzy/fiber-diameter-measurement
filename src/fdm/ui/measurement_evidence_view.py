"""Nonmodal review of stored measurement evidence, independent of live frames."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from fdm.atomic_io import atomic_write_text


class EvidenceImageView(QGraphicsView):
    """Image pixels stay untouched; scene items provide optional annotations."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMinimumSize(220, 160)
        self.overlay_items = []
        self.image_size = None
        self.fit_mode = True
        self._annotations = True

    def clear_image(self):
        self.scene().clear()
        self.overlay_items = []
        self.image_size = None

    def set_image(self, image, record, *, reset_view=True):
        transform = self.transform()
        center = self.mapToScene(self.viewport().rect().center())
        self.clear_image()
        self.scene().addPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(0, 0, image.width(), image.height())
        self.image_size = image.size()
        crop = (record.evidence or {}).get("crop", (0, 0, image.width(), image.height()))
        origin = crop[:2] if isinstance(crop, (list, tuple)) and len(crop) == 4 else (0, 0)
        if record.endpoints and len(record.endpoints) == 2:
            start, end = [QPointF(point[0] - origin[0], point[1] - origin[1]) for point in record.endpoints]
            halo = QPen(QColor(255, 255, 255, 215), 4.)
            halo.setCosmetic(True)
            pen = QPen(QColor("#c92828"), 1.8)
            pen.setCosmetic(True)
            self.overlay_items.append(self.scene().addLine(start.x(), start.y(), end.x(), end.y(), halo))
            self.overlay_items.append(self.scene().addLine(start.x(), start.y(), end.x(), end.y(), pen))
            for point in (start, end):
                item = self.scene().addEllipse(-3., -3., 6., 6., pen)
                item.setPos(point)
                item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations)
                self.overlay_items.append(item)
            # The number is shown in the fixed information line above the
            # viewport. Do not introduce another label over the measured fiber.
        self.set_annotations(self._annotations)
        if reset_view:
            self.fit_image()
        else:
            self.setTransform(transform)
            self.centerOn(center)

    def set_annotations(self, visible):
        self._annotations = bool(visible)
        for item in self.overlay_items:
            item.setVisible(self._annotations)

    def fit_image(self):
        self.fit_mode = True
        if self.image_size is not None:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def actual_pixels(self):
        self.fit_mode = False
        self.resetTransform()
        scale = 1. / self.devicePixelRatioF()
        self.scale(scale, scale)

    def zoom(self, factor):
        if self.image_size is None:
            return
        current = self.transform().m11()
        target = max(.05, min(32., current * factor))
        self.fit_mode = False
        self.scale(target / current, target / current)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self.zoom(1.2 if delta > 0 else 1. / 1.2)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit_image()


class _EvidenceDialog(QDialog):
    closedByUser = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("测量证据")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.resize(600, 470)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        options = QHBoxLayout()
        self.pin = QCheckBox("固定为小窗")
        self.follow = QCheckBox("跟随最新")
        self.annotations = QCheckBox("显示测量线")
        self.collapse = QPushButton("收起")
        for widget in (self.pin, self.follow, self.annotations):
            options.addWidget(widget)
        options.addStretch()
        options.addWidget(self.collapse)
        layout.addLayout(options)
        self.details = QLabel("暂无已接受的测量")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)
        self.image = EvidenceImageView(self)
        layout.addWidget(self.image, 1)
        controls = QHBoxLayout()
        for title, action in (("适应窗口", self.image.fit_image), ("实际像素", self.image.actual_pixels),
                              ("放大", lambda: self.image.zoom(1.25)), ("缩小", lambda: self.image.zoom(.8))):
            button = QPushButton(title)
            button.clicked.connect(action)
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.annotations.toggled.connect(self.image.set_annotations)
        self.collapse.clicked.connect(self.close)

    def reject(self):
        # Escape follows QDialog.reject rather than QWidget.closeEvent.
        # Treat both as an explicit user collapse, never temporary suspension.
        self.closedByUser.emit()
        super().reject()


class EvidenceReviewController(QObject):
    """Keep preference writes and file decoding outside measurement commits."""
    def __init__(self, workspace, *, preferences_path=None):
        super().__init__(workspace)
        self.workspace = workspace
        from fdm.settings import settings_file_path
        self.preferences_path = Path(preferences_path) if preferences_path else settings_file_path().parent / "evidence-review.json"
        # Both trial pages use the same application preference file. Share the
        # live values so a delayed save or inactive page shutdown cannot write
        # an older snapshot over the user's most recent choice on another page.
        app = QApplication.instance()
        stores = getattr(app, "_fdm_evidence_review_preferences", {})
        app._fdm_evidence_review_preferences = stores
        key = str(self.preferences_path.resolve())
        if key not in stores:
            values = self._read_preferences()
            stores[key] = {"values": values, "saved": deepcopy(values)}
        self._preference_store = stores[key]
        self.preferences = self._preference_store["values"]
        self.dialog = _EvidenceDialog(workspace.window())
        self.latest_record_id = None
        self.record_id = None
        self._closed = False
        self._suspended = False
        self._programmatic = True
        self._shown_record = None
        self._loaded_key = None
        self._loading_key = None
        self._decode_future = None
        self._decoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-evidence-review")
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self.flush_preferences)
        self.dialog.pin.setChecked(self.preferences["pinned"])
        self.dialog.follow.setChecked(self.preferences["follow"])
        self.dialog.annotations.setChecked(self.preferences["annotations"])
        self._apply_window_type()
        self._restore_geometry()
        self.dialog.pin.toggled.connect(self._pin_changed)
        self.dialog.follow.toggled.connect(self._follow_changed)
        self.dialog.annotations.toggled.connect(self._annotations_changed)
        self.dialog.closedByUser.connect(self._user_closed)
        self.dialog.installEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.screenRemoved.connect(self._screen_removed)
        self._programmatic = False

    @staticmethod
    def _defaults():
        return dict(version=1, pinned=False, open=False, follow=True, annotations=True,
                    geometry=None, pinned_geometry=None)

    def _read_preferences(self):
        result = self._defaults()
        try:
            values = json.loads(self.preferences_path.read_text(encoding="utf-8"))
            if not isinstance(values, dict) or values.get("version") != 1:
                return result
            for key in ("pinned", "open", "follow", "annotations"):
                if isinstance(values.get(key), bool):
                    result[key] = values[key]
            for key in ("geometry", "pinned_geometry"):
                rect = values.get(key)
                if isinstance(rect, list) and len(rect) == 4:
                    result[key] = [max(-100000, min(100000, int(value))) for value in rect]
        except (OSError, TypeError, ValueError, OverflowError):
            return self._defaults()
        return result

    def _apply_window_type(self):
        flag = Qt.WindowType.Tool if self.preferences["pinned"] else Qt.WindowType.Dialog
        self.dialog.setWindowFlags(flag | Qt.WindowType.WindowCloseButtonHint |
                                   Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowSystemMenuHint)
        self.dialog.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

    def _sync_preference_controls(self):
        self._programmatic = True
        try:
            for control, key in ((self.dialog.pin, "pinned"), (self.dialog.follow, "follow"),
                                 (self.dialog.annotations, "annotations")):
                blocked = control.blockSignals(True)
                control.setChecked(self.preferences[key])
                control.blockSignals(blocked)
            self.dialog.image.set_annotations(self.preferences["annotations"])
            flag = Qt.WindowType.Tool if self.preferences["pinned"] else Qt.WindowType.Dialog
            if self.dialog.windowType() != flag:
                self._apply_window_type()
                self._restore_geometry()
        finally:
            self._programmatic = False

    def _restore_geometry(self):
        pinned = self.preferences["pinned"]
        stored = self.preferences["pinned_geometry" if pinned else "geometry"]
        rect = QRect(*stored) if stored else QRect(*self.workspace.mapToGlobal(QPoint(30, 30)).toTuple(),
                                                  400 if pinned else 600, 320 if pinned else 470)
        screens = QApplication.screens()
        screen = next((screen for screen in screens if screen.availableGeometry().intersects(rect)),
                      self.workspace.screen() or QApplication.primaryScreen())
        if screen is not None:
            available = screen.availableGeometry().adjusted(10, 34, -10, -10)
            rect.setWidth(min(max(360, rect.width()), available.width()))
            rect.setHeight(min(max(270, rect.height()), available.height()))
            rect.moveLeft(max(available.left(), min(rect.left(), available.right() - rect.width() + 1)))
            rect.moveTop(max(available.top(), min(rect.top(), available.bottom() - rect.height() + 1)))
        self.dialog.setGeometry(rect)

    def _screen_removed(self, *_):
        if not self._closed:
            self._programmatic = True
            try:
                self._restore_geometry()
            finally:
                self._programmatic = False

    def _pin_changed(self, enabled):
        if self.dialog.isVisible():
            rect = self.dialog.geometry()
            self.preferences["pinned_geometry" if self.preferences["pinned"] else "geometry"] = [
                rect.x(), rect.y(), rect.width(), rect.height()]
        self.preferences["pinned"] = bool(enabled)
        self._programmatic = True
        try:
            self._apply_window_type()
            self._restore_geometry()
            if not self._suspended and self.preferences["open"]:
                self.dialog.show()
        finally:
            self._programmatic = False
        self._save_timer.start()

    def _follow_changed(self, enabled):
        self.preferences["follow"] = bool(enabled)
        if enabled and self.latest_record_id is not None:
            self.record_id = self.latest_record_id
        self._save_timer.start()
        self.refresh()

    def _annotations_changed(self, enabled):
        self.preferences["annotations"] = bool(enabled)
        self._save_timer.start()

    def _user_closed(self):
        if not self._programmatic and not self._closed:
            self.preferences["open"] = False
            self._save_timer.start()

    def show_record(self, record, *, latest=False):
        if self._closed or record is None:
            return
        self._sync_preference_controls()
        self.record_id = record.id
        if latest:
            self.latest_record_id = record.id
        # Explicit history review locks the image; newly accepted measurements
        # must not silently replace what the user is checking.
        self.dialog.follow.setChecked(bool(latest))
        self.preferences["open"] = True
        if not self._suspended:
            self.dialog.show()
            if not self.preferences["pinned"]:
                self.dialog.raise_()
        self._save_timer.start()
        self.refresh()

    def accepted(self, record, *, refresh=True):
        if self._closed or record is None:
            return
        self.latest_record_id = record.id
        if self.preferences["follow"]:
            self.record_id = record.id
        # Merely accepting a record never opens an unsolicited review window.
        if refresh:
            self.refresh()

    @staticmethod
    def _details(record):
        original = record.provenance.get("original_diameter_um")
        number = record.provenance.get("measurement_number") or record.id[:6]
        value = f"#{number} · {record.diameter_um:.4f} μm"
        if isinstance(original, (int, float)) and math.isfinite(original):
            if not math.isclose(original, record.diameter_um, abs_tol=1e-9):
                value += f"（记录已修订；原测值 {original:.4f} μm）"
            else:
                value += " · 原测值"
        else:
            value += " · 记录值（旧记录未留存原测值）"
        evidence = record.evidence or {}
        if evidence.get("state") not in {"disabled", None}:
            if evidence.get("mode") == "crop_jpeg":
                description = (f"局部 JPEG · 质量 {evidence.get('quality', 95)} · 4:4:4"
                               if evidence.get("codec") == "jpeg" else "局部无损 PNG")
            else:
                description = "完整无损 PNG"
            value += f"\n{description}；叠加线为原始测量几何"
        return value

    @staticmethod
    def _decode(path, expected_size):
        image = QImage(path)
        if image.isNull():
            return None, "无法读取测量证据文件，请检查文件是否存在或损坏。"
        if expected_size is not None and (image.width(), image.height()) != expected_size:
            return None, "证据尺寸与裁剪元数据不一致，无法可靠叠加测量线。"
        return image, None

    def refresh(self):
        if self._closed:
            return
        session = self.workspace.session
        # Establish a useful latest entry when opening an existing project.
        if self.latest_record_id not in session.measurements:
            previous = self.latest_record_id
            self.latest_record_id = next(reversed(session.measurements), None)
            if self.preferences["follow"] and self.record_id == previous:
                self.record_id = self.latest_record_id
        if self.record_id is None and self.preferences["follow"]:
            self.record_id = self.latest_record_id
        if self._suspended or not self.dialog.isVisible():
            return
        record = session.measurements.get(self.record_id)
        if record is None:
            self.dialog.image.clear_image()
            self.dialog.details.setText("该测量已撤销或删除" if self.record_id else "暂无已接受的测量")
            self.dialog.status.setText("")
            self._shown_record = self._loaded_key = None
            return
        self.dialog.details.setText(self._details(record))
        evidence = record.evidence or {}
        state = evidence.get("state", "disabled")
        if state != "saved":
            self.dialog.image.clear_image()
            self._shown_record = self._loaded_key = None
            self.dialog.status.setText({"pending": "测量证据正在写入，完成后自动显示。",
                                       "disabled": "此测量未保存图像证据。",
                                       "failed": "测量证据写入失败：" + str(evidence.get("error") or "未知原因")}.get(state, "测量证据暂不可用。"))
            return
        crop = evidence.get("crop")
        expected_size = tuple(crop[2:]) if isinstance(crop, (list, tuple)) and len(crop) == 4 else None
        path = str(evidence.get("path", ""))
        key = (record.id, path, evidence.get("sha256"), expected_size)
        if self._decode_future is not None and self._decode_future.done():
            try:
                image, error = self._decode_future.result()
            except Exception as exc:
                image, error = None, f"测量证据读取失败：{exc}"
            completed_key = self._loading_key
            self._decode_future = self._loading_key = None
            if key == completed_key:
                self._loaded_key = key
                if image is None:
                    self.dialog.image.clear_image()
                    self.dialog.status.setText(error)
                else:
                    self.dialog.image.set_image(image, record, reset_view=self._shown_record != record.id)
                    self._shown_record = record.id
                    self.dialog.status.setText("拖动平移，滚轮缩放；可关闭测量线检查纯图。")
        if key == self._loaded_key:
            return
        if self._decode_future is None:
            self.dialog.image.clear_image()
            self.dialog.status.setText("正在读取测量证据…")
            self._loading_key = key
            self._decode_future = self._decoder.submit(self._decode, path, expected_size)

    def suspend(self):
        if self._closed:
            return
        self._suspended = True
        self.dialog.hide()

    def resume(self):
        if self._closed:
            return
        self._suspended = False
        self._sync_preference_controls()
        if self.preferences["open"]:
            self._programmatic = True
            try:
                self._restore_geometry()
                self.dialog.show()
            finally:
                self._programmatic = False
        self.refresh()

    def eventFilter(self, watched, event):
        if (watched is self.dialog and event.type() in (QEvent.Type.Move, QEvent.Type.Resize)
                and not self._programmatic and not self._suspended and not self._closed and self.dialog.isVisible()):
            rect = self.dialog.geometry()
            self.preferences["pinned_geometry" if self.preferences["pinned"] else "geometry"] = [
                rect.x(), rect.y(), rect.width(), rect.height()]
            self._save_timer.start()
        return super().eventFilter(watched, event)

    def flush_preferences(self):
        self._save_timer.stop()
        if self.preferences == self._preference_store["saved"]:
            return
        try:
            atomic_write_text(self.preferences_path, json.dumps(self.preferences, ensure_ascii=False, allow_nan=False, indent=2))
            self._preference_store["saved"] = deepcopy(self.preferences)
        except OSError as exc:
            self.dialog.status.setText(f"复核窗口偏好未能保存：{exc}")

    def close(self):
        if self._closed:
            return
        self.flush_preferences()
        self._closed = True
        self._decoder.shutdown(wait=False, cancel_futures=True)
        self.dialog.removeEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.screenRemoved.disconnect(self._screen_removed)
        self.dialog.hide()
        self.dialog.deleteLater()
