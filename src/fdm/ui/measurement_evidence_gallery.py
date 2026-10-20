"""A bounded, ledger-backed album of accepted measurement evidence.

Only thumbnails requested by the visible list are decoded. The album neither
owns acquisition frames nor discovers images by scanning a directory.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

from PySide6.QtCore import (
    QEvent, QMargins, QModelIndex, QObject, QPoint, QPointF, QRect, QRectF, QSize,
    QSortFilterProxyModel, Qt, QTimer, Signal,
)
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QListView,
    QPushButton, QSlider, QStyle, QStyledItemDelegate, QVBoxLayout,
)

from fdm.atomic_io import atomic_write_text
from fdm.ui.trial_records import KEY_ROLE, TrialTimelineModel, _record_time


RECORD_ROLE = int(Qt.ItemDataRole.UserRole) + 20
_MIB = 1024 * 1024


def _read_windows_window_bounds(hwnd):
    """Read outer/client rectangles in the calling Qt thread's existing DPI context."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    for name, structure in (("GetWindowRect", wintypes.RECT), ("GetClientRect", wintypes.RECT),
                            ("ClientToScreen", wintypes.POINT)):
        function = getattr(user32, name)
        function.argtypes = [wintypes.HWND, ctypes.POINTER(structure)]
        function.restype = wintypes.BOOL
    outer, client, origin = wintypes.RECT(), wintypes.RECT(), wintypes.POINT()
    if not (user32.GetWindowRect(hwnd, ctypes.byref(outer))
            and user32.GetClientRect(hwnd, ctypes.byref(client))
            and user32.ClientToScreen(hwnd, ctypes.byref(origin))):
        raise ctypes.WinError(ctypes.get_last_error())
    return ((outer.left, outer.top, outer.right, outer.bottom), (origin.x, origin.y),
            (client.right - client.left, client.bottom - client.top))


def _windows_frame_margins(hwnd, dpr, *, read_bounds=None):
    """Include Windows' invisible resize border, which Qt may omit from its frame."""
    try:
        if not math.isfinite(dpr) or dpr <= 0:
            return QMargins()
        outer, origin, size = (read_bounds or _read_windows_window_bounds)(hwnd)
        left, top, right, bottom = outer
        x, y = origin
        width, height = size
        if right <= left or bottom <= top or width <= 0 or height <= 0:
            return QMargins()
        return QMargins(*(math.ceil(max(0, value) / dpr) for value in
                          (x - left, y - top, right - x - width, bottom - y - height)))
    except (OSError, TypeError, ValueError, OverflowError):
        # A not-yet-mapped or closing HWND can fail. Qt margins still apply;
        # the ordinary post-show check retries without changing DPI awareness.
        return QMargins()


def _read_windows_managed_state(hwnd):
    """Read OS state before Qt has necessarily processed WM_SIZE/state changes."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    for name in ("IsWindow", "IsZoomed", "IsIconic"):
        function = getattr(user32, name)
        function.argtypes = [wintypes.HWND]
        function.restype = wintypes.BOOL
    if not user32.IsWindow(hwnd):
        raise OSError("Gallery native window is not available")
    return bool(user32.IsZoomed(hwnd)), bool(user32.IsIconic(hwnd))


def _fit_client_geometry(rect, available, margins):
    """Fit the native outer frame while keeping persisted coordinates client based."""
    area = available.adjusted(8, 8, -8, -8)
    client_area = area.marginsRemoved(margins)
    width = min(max(520, rect.width()), max(1, client_area.width()))
    height = min(max(360, rect.height()), max(1, client_area.height()))
    result = QRect(rect)
    result.setSize(QSize(width, height))
    result.moveLeft(max(client_area.left(), min(rect.left(), client_area.right() - width + 1)))
    result.moveTop(max(client_area.top(), min(rect.top(), client_area.bottom() - height + 1)))
    return result


class EvidenceGalleryFilter(QSortFilterProxyModel):
    """A record remains a separate card even when several share one image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.operator_id = self.direction = self.material_id = self.evidence_state = None
        self.has_read_error = lambda record: False
        self._order_session = None
        self._order_revision = -1
        self._accepted_rank = {}
        self.setDynamicSortFilter(True)

    def filterAcceptsRow(self, row, parent):
        source = self.sourceModel()
        key = source.keys[row]
        if key[0] != "measurement":
            return False
        record = source.record(key)
        state = (record.evidence or {}).get("state") if record is not None else None
        return record is not None and state in {"saved", "pending", "failed"} and (
            (self.operator_id is None or record.operator_id == self.operator_id)
            and (self.direction is None or record.direction == self.direction)
            and (self.material_id is None or record.material_id == self.material_id)
            and (self.evidence_state is None or state == self.evidence_state
                 or self.evidence_state == "read_error" and self.has_read_error(record)))

    def lessThan(self, left, right):
        source = self.sourceModel()
        def order(index):
            key = source.keys[index.row()]
            # Windows can assign identical timestamps to consecutive accepts.
            # The ledger's insertion order is persisted, unlike UUID ordering;
            # per-field measurement numbers also restart and cannot order them.
            return (*source._sort_keys[index.row()][:2], self._accepted_rank.get(key[1], -1), *key)
        return order(left) < order(right)

    def refresh_order(self, session):
        """Build once per binding/business revision, never in a comparator."""
        revision = session.revision if session is not None else -1
        if session is self._order_session and revision == self._order_revision:
            return
        self._accepted_rank = ({record_id: rank for rank, record_id in enumerate(session.measurements)}
                               if session is not None else {})
        self._order_session, self._order_revision = session, revision

    def record(self, index):
        if not index.isValid():
            return None
        source_index = self.mapToSource(index)
        source = self.sourceModel()
        return source.record(source.keys[source_index.row()])

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        record = self.record(index)
        if record is None:
            return None
        if role == RECORD_ROLE:
            return record
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return "\n".join(_card_lines(self.sourceModel().session, record))
        return super().data(index, role)

    def apply(self, *, operator_id=None, direction=None, material_id=None, evidence_state=None):
        values = (operator_id, direction, material_id, evidence_state)
        if values == (self.operator_id, self.direction, self.material_id, self.evidence_state):
            return
        modern = hasattr(self, "endFilterChange")
        if modern:
            self.beginFilterChange()
        self.operator_id, self.direction, self.material_id, self.evidence_state = values
        if modern:
            self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
        else:
            self.invalidateFilter()

    def refresh_read_errors(self):
        if self.evidence_state != "read_error":
            return
        if hasattr(self, "endFilterChange"):
            self.beginFilterChange()
            self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
        else:
            self.invalidateFilter()


def _card_lines(session, record):
    operator = session.operators.get(record.operator_id)
    materials = operator.materials.get(record.direction, ()) if operator else ()
    material = next((item.name for item in materials if item.id == record.material_id), record.material_id)
    evidence = record.evidence or {}
    kind = ("局部 JPEG" if evidence.get("codec") == "jpeg" else "局部 PNG") if evidence.get("mode") == "crop_jpeg" else "完整 PNG"
    if evidence.get("state") == "pending":
        kind += " · 写入中"
    elif evidence.get("state") == "failed":
        kind += " · 写入失败"
    number = record.provenance.get("measurement_number") or record.id[:8]
    timestamp = _record_time(record.created_at)
    try:
        time_text = timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S") if timestamp else "时间未知"
    except (OSError, ValueError, OverflowError):
        time_text = "时间未知"
    return (f"#{number} · {record.diameter_um:.4f} μm", f"{material} · {kind}",
            f"{operator.name if operator else record.operator_id} · {'经向' if record.direction == 'warp' else '纬向'}", time_text)


@dataclass(frozen=True)
class _ThumbnailPlan:
    path: str
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    reserved_bytes: int
    pixel_bytes: int = 4


def _probe_thumbnail(path, expected_size, target_size):
    """Read only the header before the controller reserves decoding memory."""
    reader = QImageReader(path)
    size = reader.size()
    if not size.isValid() or bytes(reader.format()).lower() not in (b"png", b"jpeg", b"jpg"):
        return None, "图片缺失或损坏；双击查看详情"
    source = (size.width(), size.height())
    if expected_size is not None and source != expected_size:
        return None, "图片尺寸与证据元数据不一致"
    target = size.scaled(QSize(*target_size), Qt.AspectRatioMode.KeepAspectRatio)
    # PNG may decode at full size before scaling. Include a full RGBA image,
    # both derived thumbnails and decoder row/scratch storage in the budget.
    pixel_bytes = max(4, (QImage(1, 1, reader.imageFormat()).depth() + 7) // 8)
    required = (source[0] * source[1] + target.width() * target.height() * 2) * pixel_bytes + 2 * _MIB
    return _ThumbnailPlan(path, source, (target.width(), target.height()), required, pixel_bytes), None


def _decode_thumbnail(plan):
    reader = QImageReader(plan.path)
    if (reader.size().width(), reader.size().height()) != plan.source_size:
        return None, "证据文件在读取期间发生变化，请重新打开"
    if max(4, (QImage(1, 1, reader.imageFormat()).depth() + 7) // 8) > plan.pixel_bytes:
        return None, "证据文件在读取期间发生变化，请重新打开"
    reader.setScaledSize(QSize(*plan.target_size))
    image = reader.read()
    if image.isNull():
        return None, "图片缺失或损坏；双击查看详情"
    # Some plugins do not implement ScaledSize. Never retain a full image in
    # the cache; the full-size intermediate was included in the reservation.
    if image.size() != QSize(*plan.target_size):
        image = image.scaled(QSize(*plan.target_size), Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
    return image, None


class _GalleryDelegate(QStyledItemDelegate):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller

    def sizeHint(self, option, index):
        return self.controller.card_size()

    def paint(self, painter, option, index):
        record = self.controller.proxy.record(index)
        if record is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = option.palette
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        rect = QRectF(option.rect).adjusted(4, 4, -4, -4)
        painter.setPen(QPen(palette.highlight().color() if selected else palette.mid().color(), 1))
        painter.setBrush(palette.highlight() if selected else palette.base())
        painter.drawRoundedRect(rect, 5, 5)
        foreground = palette.highlightedText().color() if selected else palette.text().color()
        image_rect = QRectF(rect.x() + 7, rect.y() + 7, rect.width() - 14,
                            self.controller.thumbnail_height())
        painter.fillRect(image_rect, palette.alternateBase())
        image, error = self.controller.thumbnail(record)
        if image is not None:
            scale = min(image_rect.width() / image.width(), image_rect.height() / image.height())
            displayed = QRectF(0, 0, image.width() * scale, image.height() * scale)
            displayed.moveCenter(image_rect.center())
            painter.drawImage(displayed, image)
            if self.controller.dialog.annotations.isChecked():
                self._paint_measurement(painter, displayed, record, image)
        else:
            state = (record.evidence or {}).get("state")
            text = error or {"pending": "证据正在写入", "failed": "证据写入失败"}.get(state, "正在读取缩略图…")
            painter.setPen(palette.text().color())
            painter.drawText(image_rect.adjusted(8, 8, -8, -8),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)
        painter.setPen(foreground)
        metrics = option.fontMetrics
        y = image_rect.bottom() + 5
        for line in _card_lines(self.controller.model.session, record):
            text = metrics.elidedText(line, Qt.TextElideMode.ElideRight, int(rect.width() - 14))
            painter.drawText(QRectF(rect.x() + 7, y, rect.width() - 14, metrics.height() + 2),
                             Qt.AlignmentFlag.AlignVCenter, text)
            y += metrics.height() + 2
        painter.restore()

    @staticmethod
    def _paint_measurement(painter, rect, record, image):
        endpoints = record.endpoints
        if not endpoints or len(endpoints) != 2:
            return
        evidence = record.evidence or {}
        crop = evidence.get("crop")
        if not isinstance(crop, (tuple, list)) or len(crop) != 4:
            source_size = evidence.get("source_size")
            # Legacy full PNG dimensions are known only after probing. The
            # cached thumbnail keeps its original source size separately.
            source_size = source_size or image.text("fdm_source_size").split(",")
            try:
                crop = (0, 0, int(source_size[0]), int(source_size[1]))
            except (TypeError, ValueError, IndexError):
                return
        try:
            x, y, width, height = map(float, crop)
            if width <= 0 or height <= 0:
                return
            points = [QPointF(rect.x() + (float(point[0]) - x) * rect.width() / width,
                             rect.y() + (float(point[1]) - y) * rect.height() / height)
                      for point in endpoints]
            if any(not math.isfinite(point.x()) or not math.isfinite(point.y()) for point in points):
                return
        except (TypeError, ValueError, IndexError):
            return
        painter.save()
        painter.setClipRect(rect)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 3))
        painter.drawLine(*points)
        painter.setPen(QPen(QColor("#c92828"), 1.4))
        painter.drawLine(*points)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for point in points:
            painter.drawEllipse(point, 2.5, 2.5)
        painter.restore()


class _GalleryDialog(QDialog):
    closedByUser = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.WindowCloseButtonHint)
        self.resize(1040, 720)
        layout = QVBoxLayout(self)
        filters = QHBoxLayout()
        self.operator = QComboBox(self)
        self.direction = QComboBox(self)
        self.material = QComboBox(self)
        self.state = QComboBox(self)
        for name, widget in (("操作者", self.operator), ("方向", self.direction),
                             ("纤维类别", self.material), ("证据状态", self.state)):
            widget.setAccessibleName(name)
            widget.setMinimumWidth(90)
            filters.addWidget(widget, 1)
        layout.addLayout(filters)
        controls = QHBoxLayout()
        self.annotations = QCheckBox("显示测量线", self)
        self.latest = QPushButton("回到最新", self)
        self.latest.setAccessibleName("回到最新图像证据")
        self.size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.size_slider.setRange(144, 320)
        self.size_slider.setSingleStep(16)
        self.size_slider.setMaximumWidth(140)
        self.size_slider.setAccessibleName("缩略图大小")
        controls.addWidget(self.annotations)
        controls.addWidget(self.latest)
        controls.addStretch(1)
        controls.addWidget(QLabel("缩略图", self))
        controls.addWidget(self.size_slider)
        layout.addLayout(controls)
        self.view = QListView(self)
        self.view.setObjectName("measurementEvidenceGallery")
        self.view.setAccessibleName("测量证据相册")
        self.view.setViewMode(QListView.ViewMode.IconMode)
        self.view.setFlow(QListView.Flow.LeftToRight)
        self.view.setWrapping(True)
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
        self.view.setMovement(QListView.Movement.Static)
        self.view.setUniformItemSizes(True)
        self.view.setLayoutMode(QListView.LayoutMode.Batched)
        self.view.setBatchSize(100)
        self.view.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
        self.view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        layout.addWidget(self.view, 1)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def closeEvent(self, event):
        self.closedByUser.emit()
        super().closeEvent(event)

    def reject(self):
        self.closedByUser.emit()
        super().reject()


class EvidenceGalleryController(QObject):
    """One rebindable application-owned album, with a 64 MiB image budget."""

    MEMORY_LIMIT = 64 * _MIB
    HEADER_RESERVATION = 64 * 1024
    MAX_QUEUED = 128

    def __init__(self, host, *, preferences_path=None, parent=None):
        super().__init__(parent or host)
        self.host = host
        self.workspace = None
        if preferences_path is None:
            from fdm.settings import settings_file_path
            preferences_path = settings_file_path().parent / "evidence-gallery.json"
        self.preferences_path = Path(preferences_path)
        self.preferences = self._read_preferences()
        self.dialog = _GalleryDialog(host.window())
        self.model = TrialTimelineModel(parent=self)
        self.proxy = EvidenceGalleryFilter(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.sort(0, Qt.SortOrder.DescendingOrder)
        self.dialog.view.setModel(self.proxy)
        self.dialog.view.setItemDelegate(_GalleryDelegate(self, self.dialog.view))
        self.dialog.annotations.setChecked(self.preferences["annotations"])
        self.dialog.size_slider.setValue(self.preferences["thumbnail_width"])
        self._closed = self._suspended = self._programmatic = False
        self._normal_restore_rect = None
        self._generation = 0
        self._cache = OrderedDict()
        self._known_read_errors = {}
        self._read_error_records = {}
        self.proxy.has_read_error = lambda record: self._known_read_errors.get(record.id) == self._key(record)
        self._cache_bytes = self._reserved_bytes = 0
        self._queue = OrderedDict()
        self._wanted = set()
        self._future = self._job = None
        self._decoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-evidence-album")
        self._metadata = None
        self._new_items = 0
        self.decode_count = 0
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self.flush_preferences)
        self._request_timer = QTimer(self)
        self._request_timer.setSingleShot(True)
        self._request_timer.timeout.connect(self._request_visible)
        self._geometry_timer = QTimer(self)
        self._geometry_timer.setSingleShot(True)
        self._geometry_timer.timeout.connect(self._clamp_visible_geometry)
        self.dialog.closedByUser.connect(self._user_closed)
        self.dialog.view.doubleClicked.connect(self._review)
        self.dialog.view.verticalScrollBar().valueChanged.connect(self._scrolled)
        for combo in (self.dialog.operator, self.dialog.direction, self.dialog.material, self.dialog.state):
            combo.currentIndexChanged.connect(self._filters_changed)
        self.dialog.size_slider.valueChanged.connect(self._size_changed)
        self.dialog.annotations.toggled.connect(self._annotations_changed)
        self.dialog.latest.clicked.connect(self._go_latest)
        self.dialog.view.viewport().installEventFilter(self)
        self.dialog.view.installEventFilter(self)
        self.dialog.installEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.screenRemoved.connect(self._screen_removed)
        self._size_changed(self.preferences["thumbnail_width"])
        self._restore_geometry()

    @property
    def is_open(self):
        return not self._closed and bool(self.preferences["open"])

    @property
    def memory_bytes(self):
        return self._cache_bytes + self._reserved_bytes

    def _read_preferences(self):
        result = {"version": 1, "open": False, "geometry": None, "maximized": False,
                  "thumbnail_width": 224, "annotations": True}
        try:
            saved = json.loads(self.preferences_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or saved.get("version") != 1:
                return result
            for key in ("open", "maximized", "annotations"):
                if isinstance(saved.get(key), bool):
                    result[key] = saved[key]
            if type(saved.get("thumbnail_width")) is int:
                result["thumbnail_width"] = max(144, min(320, saved["thumbnail_width"]))
            geometry = saved.get("geometry")
            if isinstance(geometry, list) and len(geometry) == 4 and all(type(v) is int for v in geometry):
                result["geometry"] = geometry
        except (OSError, ValueError, TypeError):
            pass
        return result

    def flush_preferences(self):
        self._save_timer.stop()
        try:
            self.preferences_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.preferences_path,
                              json.dumps(self.preferences, ensure_ascii=False, indent=2, allow_nan=False),
                              encoding="utf-8")
        except OSError:
            if not self._closed:
                self.dialog.status.setText("相册显示已应用，本次窗口偏好未能保存。")

    def bind(self, workspace):
        if self._closed or (workspace is self.workspace and self.model.session is getattr(workspace, "session", None)):
            return
        self._invalidate_requests()
        self._cache.clear()
        self._known_read_errors.clear()
        self._read_error_records.clear()
        self._cache_bytes = 0
        self.workspace = workspace
        session = workspace.session if workspace is not None else None
        self.proxy.refresh_order(session)
        self.model.set_session(session)
        self._metadata = None
        self._refresh_filters(reset=True)
        self._new_items = 0
        self._go_latest()
        self._update_status()

    def open(self, workspace=None):
        if self._closed:
            return
        if workspace is not None:
            self.bind(workspace)
        if self.workspace is None:
            return
        self.preferences["open"] = True
        self._suspended = False
        # Reopening is also an explicit retry after a missing file was restored.
        for key, entry in list(self._cache.items()):
            if entry[0] is None:
                self._cache_bytes -= self._cache.pop(key)[2]
        self._restore_geometry(create_native=True)
        self.dialog.showMaximized() if self.preferences["maximized"] else self.dialog.show()
        self._clamp_visible_geometry()
        self.dialog.raise_()
        self._save_timer.start()
        self.refresh()
        self._request_timer.start(0)

    def suspend(self):
        if self._closed:
            return
        self._suspended = True
        self._invalidate_requests()
        self.dialog.hide()

    def resume(self, workspace=None):
        if self._closed:
            return
        if workspace is not None:
            self.bind(workspace)
        self._suspended = False
        if self.is_open and self.workspace is not None:
            self._restore_geometry(create_native=True)
            self.dialog.showMaximized() if self.preferences["maximized"] else self.dialog.show()
            self._clamp_visible_geometry()
            self.refresh()
            self._request_timer.start(0)

    def refresh(self, keys=None, *, rebuild=False):
        if self._closed or self.workspace is None:
            return
        session = self.workspace.session
        if self.model.session is not session:
            self.bind(self.workspace)
        if session.revision != self.model.revision or keys is not None or rebuild:
            following = self.dialog.view.verticalScrollBar().value() <= 2
            anchor = self._anchor() if not following else None
            self.proxy.refresh_order(session)
            added = self.model.sync(keys=keys, rebuild=rebuild)
            self._refresh_filters()
            if following:
                self._go_latest()
            else:
                self._new_items += sum(1 for key in added if key[0] == "measurement"
                    and self.proxy.mapFromSource(self.model.index(self.model.rows_by_key[key], 0)).isValid())
                self._restore_anchor(anchor)
            # sync() already emits row/structure changes. Do not widen Qt's
            # precise invalidation to every visible evidence card.
            self._update_status()
        if self.dialog.isVisible() and not self._suspended:
            # Also re-evaluate visible evidence keys after Save As, whose path
            # rebase deliberately does not create a business revision.
            self._prune_changed_read_errors()
            if not self._request_timer.isActive():
                self._request_timer.start(0)

    def _prune_changed_read_errors(self):
        changed = False
        for record_id, key in list(self._known_read_errors.items()):
            record = self.model.session.measurements.get(record_id)
            if record is self._read_error_records.get(record_id):
                continue
            if record is None or self._key(record) != key:
                self._known_read_errors.pop(record_id, None)
                self._read_error_records.pop(record_id, None)
                changed = True
            else:
                self._read_error_records[record_id] = record
        if changed:
            self.proxy.refresh_read_errors()
            self._update_status()

    def thumbnail_height(self):
        return round(self.preferences["thumbnail_width"] * .75)

    def card_size(self):
        return QSize(self.preferences["thumbnail_width"] + 22,
                     self.thumbnail_height() + (self.dialog.fontMetrics().height() + 2) * 4 + 30)

    def _key(self, record):
        evidence = record.evidence or {}
        crop = evidence.get("crop")
        size = tuple(crop[2:]) if isinstance(crop, (tuple, list)) and len(crop) == 4 else None
        dpr = self.dialog.devicePixelRatioF()
        target = (round(self.preferences["thumbnail_width"] * dpr), round(self.thumbnail_height() * dpr))
        return (self.model.session.id, record.id, evidence.get("evidence_id"), evidence.get("sha256"),
                str(evidence.get("path", "")), size, target)

    def thumbnail(self, record):
        entry = self._cache.get(self._key(record))
        if entry is not None:
            self._cache.move_to_end(self._key(record))
            return entry[0], entry[1]
        return None, None

    def _visible_rows(self, *, preload=True):
        count = self.proxy.rowCount()
        if not count:
            return range(0)
        view = self.dialog.view
        grid = self.card_size()
        columns = max(1, view.viewport().width() // grid.width())
        scroll = view.verticalScrollBar().value()
        height = view.viewport().height()
        margin = height if preload else 0
        first = max(0, (scroll - margin) // grid.height()) * columns
        last = ((scroll + height + margin) // grid.height() + 2) * columns
        return range(min(first, count - 1), min(count, last, first + self.MAX_QUEUED))

    def _request_visible(self):
        if self._closed or self._suspended or not self.dialog.isVisible() or self.workspace is None:
            return
        queue = OrderedDict()
        wanted = set()
        # Visible cards take priority over the adjacent screen's prefetch.
        rows = dict.fromkeys([*self._visible_rows(preload=False), *self._visible_rows()])
        for row in rows:
            record = self.proxy.record(self.proxy.index(row, 0))
            if record is None or (record.evidence or {}).get("state") != "saved":
                continue
            key = self._key(record)
            wanted.add(key)
            if key not in self._cache and (self._job is None or self._job[1] != key):
                queue[key] = record
            if len(queue) >= self.MAX_QUEUED:
                break
        self._wanted, self._queue = wanted, queue
        self._launch_next()

    def _reserve(self, amount):
        if amount > self.MEMORY_LIMIT:
            return False
        while self._cache and self._cache_bytes + amount > self.MEMORY_LIMIT:
            _, entry = self._cache.popitem(last=False)
            self._cache_bytes -= entry[2]
        self._reserved_bytes = amount
        return True

    def _launch_next(self):
        if self._future is not None or self._closed or self._suspended or not self.dialog.isVisible():
            return
        while self._queue:
            key, record = self._queue.popitem(last=False)
            if key in self._cache or key not in self._wanted:
                continue
            self._reserve(self.HEADER_RESERVATION)
            self._job = (self._generation, key, "header", None)
            self._future = self._decoder.submit(_probe_thumbnail, key[4], key[5], key[6])
            break

    def _tick(self):
        if self._closed:
            return
        if self._future is not None and self._future.done():
            job, future = self._job, self._future
            self._job = self._future = None
            try:
                value, error = future.result()
            except Exception:
                value, error = None, "无法读取证据缩略图，双击查看详情"
            self._reserved_bytes = 0
            generation, key, stage, plan = job
            current = generation == self._generation and key in self._wanted and not self._suspended
            if current and stage == "header" and value is not None:
                if self._reserve(value.reserved_bytes):
                    self._job = (generation, key, "decode", value)
                    self._future = self._decoder.submit(_decode_thumbnail, value)
                    self.decode_count += 1
                else:
                    self._cache_result(key, None, "图像较大，双击打开单图复核", read_error=False)
            elif current:
                if value is not None and stage == "decode":
                    value.setText("fdm_source_size", f"{plan.source_size[0]},{plan.source_size[1]}")
                self._cache_result(key, value, error)
            # A successful header only starts decoding; its loading card has
            # not changed. Completed pixels/errors repaint in _cache_result.
        self._launch_next()

    def _cache_result(self, key, image, error, *, read_error=True):
        size = int(image.sizeInBytes()) if image is not None else 512
        previous = self._cache.pop(key, None)
        if previous:
            self._cache_bytes -= previous[2]
        while self._cache and (self._cache_bytes + self._reserved_bytes + size > self.MEMORY_LIMIT or len(self._cache) >= 512):
            _, entry = self._cache.popitem(last=False)
            self._cache_bytes -= entry[2]
        if self._cache_bytes + self._reserved_bytes + size <= self.MEMORY_LIMIT:
            self._cache[key] = (image, error, size)
            self._cache_bytes += size
        if image is None and error and read_error:
            self._known_read_errors[key[1]] = key
            self._read_error_records[key[1]] = self.model.session.measurements.get(key[1])
        else:
            self._known_read_errors.pop(key[1], None)
            self._read_error_records.pop(key[1], None)
        self.proxy.refresh_read_errors()
        self._update_status()
        self._update_record_card(key[1])

    def _update_record_card(self, record_id):
        if self._closed or self._suspended or not self.dialog.isVisible():
            return
        row = self.model.rows_by_key.get(("measurement", record_id))
        if row is None:
            return
        index = self.proxy.mapFromSource(self.model.index(row, 0))
        if not index.isValid():
            return
        view = self.dialog.view
        rect = view.visualRect(index).intersected(view.viewport().rect())
        if not rect.isEmpty():
            view.viewport().update(rect)

    def _invalidate_requests(self):
        self._generation += 1
        self._request_timer.stop()
        self._queue.clear()
        self._wanted.clear()
        # An active file decode cannot be forcibly interrupted. Its reservation
        # remains accounted for until completion, and its result is discarded.
        if self._future is not None and self._future.cancel():
            self._future = self._job = None
            self._reserved_bytes = 0

    def _refresh_filters(self, *, reset=False):
        session = self.model.session
        metadata = tuple((item.id, item.name, tuple((direction, material.id, material.name)
                         for direction, materials in item.materials.items() for material in materials))
                         for item in session.operators.values()) if session is not None else ()
        if not reset and metadata == self._metadata:
            return
        self._metadata = metadata
        operators = [("全部操作者", None)]
        operators += [(item.name, item.id) for item in session.operators.values()] if session is not None else []
        materials = {}
        if session is not None:
            for operator in session.operators.values():
                for values in operator.materials.values():
                    materials.update((item.id, item.name) for item in values)
        choices = ((self.dialog.operator, operators),
                   (self.dialog.direction, [("全部方向", None), ("经向", "warp"), ("纬向", "weft")]),
                   (self.dialog.material, [("全部类别", None)] + [(name, key) for key, name in materials.items()]),
                   (self.dialog.state, [("全部证据", None), ("已保存", "saved"), ("写入中", "pending"),
                                        ("写入失败", "failed"), ("读取异常（已检查）", "read_error")]))
        for combo, entries in choices:
            selected = None if reset else combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for label, value in entries:
                combo.addItem(label, value)
            combo.setCurrentIndex(max(0, combo.findData(selected)))
            combo.blockSignals(False)
        self._filters_changed(reset_scroll=reset)

    def _filters_changed(self, *_, reset_scroll=True):
        self._invalidate_requests()
        self.proxy.apply(operator_id=self.dialog.operator.currentData(), direction=self.dialog.direction.currentData(),
                         material_id=self.dialog.material.currentData(), evidence_state=self.dialog.state.currentData())
        if reset_scroll:
            self._go_latest()
        self._update_status()
        self._request_timer.start(0)

    def _size_changed(self, value):
        self.preferences["thumbnail_width"] = max(144, min(320, int(value)))
        if hasattr(self, "_request_timer"):
            self._invalidate_requests()
        self.dialog.view.setGridSize(self.card_size())
        self.dialog.view.doItemsLayout()
        self._save_timer.start()
        self._request_timer.start(0)

    def _annotations_changed(self, visible):
        self.preferences["annotations"] = bool(visible)
        self.dialog.view.viewport().update()
        self._save_timer.start()

    def _scrolled(self, value):
        if value <= 2:
            self._new_items = 0
            self._update_status()
        self._request_timer.start(0)

    def _go_latest(self):
        self._new_items = 0
        self.dialog.view.scrollToTop()
        self._update_status()
        self._request_timer.start(0)

    def _anchor(self):
        rows = self._visible_rows(preload=False)
        if not rows:
            return None
        index = self.proxy.index(rows.start, 0)
        return self.proxy.data(index, KEY_ROLE), self.dialog.view.visualRect(index).top()

    def _restore_anchor(self, anchor):
        if not anchor:
            return
        key, offset = anchor
        generation = self._generation
        def restore():
            if self._closed or generation != self._generation:
                return
            row = self.model.rows_by_key.get(tuple(key))
            if row is None:
                return
            index = self.proxy.mapFromSource(self.model.index(row, 0))
            if index.isValid():
                self.dialog.view.scrollTo(index, QListView.ScrollHint.PositionAtTop)
                bar = self.dialog.view.verticalScrollBar()
                bar.setValue(bar.value() - offset)
        QTimer.singleShot(0, restore)

    def _update_status(self):
        if self._closed:
            return
        session = self.model.session
        name = session.sample_name or session.sample_number or session.name or "新试验" if session is not None else "未选择试验"
        self.dialog.setWindowTitle(f"证据相册 · {name}")
        count = self.proxy.rowCount()
        self.dialog.latest.setText(f"新增 {self._new_items} 项 · 回到最新" if self._new_items else "回到最新")
        if count:
            self.dialog.status.setText(f"{count} 项测量证据 · 最新在前 · 双击查看原测量线与图像；补计不产生图像证据。")
        else:
            self.dialog.status.setText("当前筛选没有图像证据。高级选项可设置局部、全图或不保存；设置只影响之后接受的测量，未保存的图片无法补回。")
        if self.proxy.evidence_state == "read_error":
            self.dialog.status.setText(self.dialog.status.text() + " 读取异常仅包含已经按需检查的图片，未扫描全部证据。")

    def _review(self, index):
        record = self.proxy.record(index)
        if self._closed or self.workspace is None or record is None:
            return
        self.workspace.evidence_review.show_record(record, latest=False)

    def _user_closed(self):
        if self._closed or self._programmatic:
            return
        self.preferences["open"] = False
        self._invalidate_requests()
        self._save_timer.start()

    def _frame_margins(self):
        handle = self.dialog.windowHandle()
        if handle is None:
            return QMargins()
        # Native decorations are available after handle creation on Windows.
        # Other window systems may only finalize them when showing the window.
        margins = handle.frameMargins()
        native = (_windows_frame_margins(int(self.dialog.winId()), self.dialog.devicePixelRatioF())
                  if sys.platform == "win32" else QMargins())
        return QMargins(max(0, margins.left(), native.left()), max(0, margins.top(), native.top()),
                        max(0, margins.right(), native.right()), max(0, margins.bottom(), native.bottom()))

    def _apply_geometry(self, rect):
        previous = self._programmatic
        self._programmatic = True
        try:
            if rect != self.dialog.geometry():
                self.dialog.setGeometry(rect)
        finally:
            self._programmatic = previous
        actual = self.dialog.geometry()
        saved = [actual.x(), actual.y(), actual.width(), actual.height()]
        if self.preferences["geometry"] != saved:
            self.preferences["geometry"] = saved
            self._save_timer.start()

    def _native_managed_state(self):
        if sys.platform == "win32" and self.dialog.windowHandle() is not None:
            try:
                return _read_windows_managed_state(int(self.dialog.winId()))
            except OSError:
                pass  # Qt state and normalGeometry remain the fallback.
        return False, False

    def _window_is_managed(self):
        return (self.dialog.isMaximized() or self.dialog.isFullScreen() or self.dialog.isMinimized()
                or any(self._native_managed_state()))

    def _restore_geometry(self, *, create_native=False):
        if self._window_is_managed():
            return
        app = QApplication.instance()
        raw = self.preferences["geometry"]
        rect = QRect(*raw) if raw is not None else self.dialog.geometry()
        screen = app.screenAt(rect.center()) or self.host.screen() or app.primaryScreen()
        if screen is None:
            return
        if create_native:
            self.dialog.ensurePolished()
            self.dialog.winId()  # Create decorations without displaying an off-screen title bar.
        self._apply_geometry(_fit_client_geometry(rect, screen.availableGeometry(), self._frame_margins()))

    def _clamp_visible_geometry(self):
        if (self._closed or self._suspended or not self.dialog.isVisible()
                or self._window_is_managed()):
            return
        rect = (QRect(self._normal_restore_rect) if self._normal_restore_rect is not None
                else self.dialog.geometry())
        screen = (QApplication.screenAt(rect.center()) or self.dialog.screen()
                  or self.host.screen() or QApplication.primaryScreen())
        if screen is not None:
            self._apply_geometry(_fit_client_geometry(rect, screen.availableGeometry(), self._frame_margins()))
            self._normal_restore_rect = None

    def _screen_removed(self, *_):
        if not self._closed:
            self._restore_geometry()
            self._geometry_timer.start(0)

    def eventFilter(self, watched, event):
        if self._closed:
            return False
        kind = event.type()
        if watched is self.dialog.view and kind == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._review(self.dialog.view.currentIndex())
            return True
        if watched is self.dialog.view.viewport() and kind in (QEvent.Type.Resize, QEvent.Type.Show):
            self._request_timer.start(0)
        elif watched is self.dialog:
            qt_managed = self.dialog.isMaximized() or self.dialog.isFullScreen() or self.dialog.isMinimized()
            managed = qt_managed
            native_maximized = native_minimized = False
            if (kind in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.WindowStateChange)
                    and not self._programmatic and self.dialog.isVisible()):
                # Actual WM/UIA maximization delivers Resize/Move while Qt
                # still says normal. Never persist those intermediate sizes.
                native_maximized, native_minimized = self._native_managed_state()
                managed = managed or native_maximized or native_minimized
            if kind == QEvent.Type.WindowStateChange:
                was_managed = bool(event.oldState() & (Qt.WindowState.WindowMaximized |
                                                       Qt.WindowState.WindowFullScreen |
                                                       Qt.WindowState.WindowMinimized))
                if was_managed and not managed and self.preferences["geometry"] is not None:
                    # Windows can announce the normal state before updating
                    # its maximized client rectangle. Preserve the actual
                    # restore target through those intermediate move events.
                    self._normal_restore_rect = QRect(*self.preferences["geometry"])
                elif managed:
                    self._normal_restore_rect = None
                    if qt_managed and not self._programmatic:
                        normal = self.dialog.normalGeometry()
                        if normal.isValid():
                            # At the Qt state transition this is authoritative,
                            # including when the earlier native query failed.
                            self.preferences["geometry"] = [normal.x(), normal.y(),
                                                            normal.width(), normal.height()]
            if kind == QEvent.Type.Show or (kind == QEvent.Type.WindowStateChange and not managed):
                # One check after show/normal restore, not after user moves.
                self._geometry_timer.start(0)
            if kind in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.WindowStateChange) and not self._programmatic and self.dialog.isVisible():
                if not self.dialog.isFullScreen() and not (self.dialog.isMinimized() or native_minimized):
                    self.preferences["maximized"] = self.dialog.isMaximized() or native_maximized
                if not managed and self._normal_restore_rect is None:
                    rect = self.dialog.geometry()
                    self.preferences["geometry"] = [rect.x(), rect.y(), rect.width(), rect.height()]
                self._save_timer.start()
            elif kind in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
                self.dialog.view.setGridSize(self.card_size())
                self._request_timer.start(0)
        return False

    def close(self):
        if self._closed:
            return
        self._programmatic = True
        self._invalidate_requests()
        self._closed = True
        self._timer.stop()
        self._geometry_timer.stop()
        self._save_timer.stop()
        self.flush_preferences()
        self.dialog.close()
        self._decoder.shutdown(wait=False, cancel_futures=True)
        # The executor alone owns any final bounded decode while it finishes;
        # the closed controller must not retain that future's image result.
        self._future = self._job = None
        self._reserved_bytes = 0
        self._cache.clear()
        self._cache_bytes = 0
        self._known_read_errors.clear()
        self._read_error_records.clear()
        self.workspace = None
        self.proxy.refresh_order(None)
        self.model.set_session(None)
        self.dialog.deleteLater()
        self.deleteLater()
