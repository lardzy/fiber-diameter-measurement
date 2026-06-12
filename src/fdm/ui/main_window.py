from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread, current_thread
from time import perf_counter
import math
import tempfile

from PySide6.QtCore import QByteArray, QEvent, QEventLoop, QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QCloseEvent, QFont, QGuiApplication, QIcon, QImage, QImageReader, QPainter, QPalette, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from fdm import __version__
from fdm.area_display import ensure_measurement_display_geometry, invalidate_measurement_display_geometry
from fdm.geometry import Line, Point, line_length
from fdm.models import (
    Calibration,
    CalibrationPreset,
    FiberGroup,
    ImageDocument,
    Measurement,
    OverlayAnnotation,
    OverlayAnnotationKind,
    ProjectGroupTemplate,
    ProjectState,
    UNCATEGORIZED_LABEL,
    normalize_group_label,
    new_id,
    project_capture_root,
    project_slide_root,
)
from fdm.services.digital_slide_store import (
    DIGITAL_SLIDE_SUFFIX,
    DOCUMENT_KIND_DIGITAL_SLIDE,
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTile,
    is_digital_slide_path,
)
from fdm.services.motion_control import (
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    DIR_NEG,
    DIR_POS,
    MotionController,
    axis_name,
    list_motion_ports,
    preferred_motion_port,
)
from fdm.settings import (
    AppSettings,
    AppSettingsIO,
    FocusStackProfile,
    MagicSegmentToolMode,
    OpenImageViewMode,
    RawRecordTemplate,
    ScaleOverlayPlacementMode,
    is_fiber_quick_tool_mode,
    is_magic_toolbar_tool_mode,
    is_magic_segment_tool_mode,
    is_reference_propagation_tool_mode,
    settings_file_path,
)
from fdm.services.area_inference import AreaInferenceService, parse_area_model_labels
from fdm.services.export_service import ExportImageRenderMode, ExportScope, ExportSelection, ExportService
from fdm.services.fiber_quick_geometry import DEFAULT_FIBER_QUICK_GEOMETRY_TIMEOUT_MS
from fdm.services.group_manager import GroupManager
from fdm.services.preview_analysis import (
    FocusStackFinalResult,
    FocusStackRenderConfig,
    FocusStackReport,
    MAP_BUILD_ANALYSIS_INTERVAL_MS,
    MAP_BUILD_STABLE_REQUIRED_FRAMES,
    MapBuildFinalResult,
    MapBuildReport,
)
from fdm.services.prompt_segmentation import (
    PromptSegmentationResult,
    create_interactive_segmentation_service,
    initial_interactive_segmentation_crop_box,
    interactive_segmentation_model_label,
    interactive_segmentation_model_paths,
    interactive_segmentation_models_ready,
    interactive_segmentation_runtime_root,
    resolve_interactive_segmentation_backend,
)
from fdm.services.reference_instance_propagation import (
    ReferenceInstancePropagationResult,
    area_geometry_iou,
)
from fdm.services.sidecar_io import CalibrationSidecarIO
from fdm.services.snap_service import SnapResult, SnapService
from fdm.ui.canvas import (
    AreaEditOperationMode,
    DocumentCanvas,
    MagicSegmentOperationMode,
    MagicSegmentSubtractInputMode,
    magic_prompt_visual,
)
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.dialogs import (
    AreaAutoRecognitionDialog,
    CalibrationInputDialog,
    CalibrationPresetDialog,
    ExportOptionsDialog,
    FiberGroupDialog,
    SettingsDialog,
    ShortcutHelpDialog,
)
from fdm.ui.area_inference_worker import AreaInferenceRequest
from fdm.ui.background_task_controller import AreaInferenceBatchState, BackgroundTaskController, BatchLoadState
from fdm.ui.export_controller import ExportController
from fdm.ui.icons import application_icon, themed_icon
from fdm.ui.image_loader import ImageLoadRequest
from fdm.ui.microview_preview_host import MicroviewPreviewHost
from fdm.ui.preview_analysis_task_controller import PreviewAnalysisTaskController
from fdm.ui.preview_analysis_dialog import PreviewAnalysisDialog
from fdm.ui.project_session_controller import ProjectSessionController
from fdm.ui.prompt_segmentation_worker import PromptSegmentationRequest
from fdm.ui.reference_instance_worker import ReferenceInstancePropagationRequest
from fdm.ui.fiber_quick_geometry_worker import FiberQuickGeometryRequest
from fdm.ui.rendering import draw_measurements, draw_overlay_annotations, draw_scale_overlay, overlay_metrics
from fdm.ui.theme import apply_application_theme, refresh_widget_theme
from fdm.ui.thread_task_manager import (
    TASK_AREA_INFERENCE,
    TASK_FIBER_QUICK_COMMIT_GEOMETRY,
    TASK_FIBER_QUICK_GEOMETRY,
    TASK_IMAGE_LOAD,
    TASK_PROMPT_SEGMENTATION,
    TASK_REFERENCE_INSTANCE,
    ThreadTaskManager,
)
from fdm.ui.widgets import (
    FiberGroupListItemWidget,
    FlowLayout,
    MeasurementGroupComboBox,
    MeasurementToolStrip,
    OverlayToolSplitButton,
)

try:
    from fdm.services.capture import CaptureDevice, CaptureSessionManager

    _CAPTURE_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    _CAPTURE_IMPORT_ERROR = exc

    @dataclass(slots=True)
    class CaptureDevice:
        id: str
        name: str
        backend_key: str
        native_id: object
        available: bool = True
        detail: str = ""

    class _SignalProxy:
        def __init__(self) -> None:
            self._callbacks: list[object] = []

        def connect(self, callback) -> None:
            self._callbacks.append(callback)

        def emit(self, *args) -> None:
            for callback in list(self._callbacks):
                callback(*args)

    class CaptureSessionManager:
        def __init__(self, *args, selected_device_id: str = "", refresh_on_init: bool = True, **kwargs) -> None:
            self._selected_device_id = selected_device_id
            self.devicesChanged = _SignalProxy()
            self.previewStateChanged = _SignalProxy()
            self.frameReady = _SignalProxy()
            self.analysisFrameReady = _SignalProxy()
            self.analysisFrameFailed = _SignalProxy()
            self.errorOccurred = _SignalProxy()

        def devices(self) -> list[CaptureDevice]:
            return []

        def selected_device_id(self) -> str:
            return self._selected_device_id

        def selected_device(self) -> CaptureDevice | None:
            return None

        def is_preview_active(self) -> bool:
            return False

        def last_frame(self) -> QImage | None:
            return None

        def preview_kind(self) -> str:
            return "frame_stream"

        def can_capture_still(self) -> bool:
            return False

        def capture_still_frame(self) -> QImage | None:
            return None

        def capture_fresh_frame(self, *, timeout_ms: int = 2000) -> QImage | None:
            return None

        def preview_resolution(self) -> tuple[int, int] | None:
            return None

        def can_optimize_signal(self) -> bool:
            return False

        def can_request_analysis_frame(self) -> bool:
            return False

        def optimize_signal(self) -> str:
            raise RuntimeError("当前采集设备不支持信号优化。")

        def active_warning(self) -> str:
            return ""

        def capture_diagnostics(self) -> str:
            return ""

        def device_refresh_warnings(self) -> list[str]:
            return [f"采集模块未安装: {_CAPTURE_IMPORT_ERROR}"] if _CAPTURE_IMPORT_ERROR is not None else []

        def refresh_devices(self) -> list[CaptureDevice]:
            self.devicesChanged.emit([])
            return []

        def set_selected_device(self, device_id: str) -> bool:
            return False

        def start_preview(self, *, preview_target: object | None = None) -> bool:
            detail = str(_CAPTURE_IMPORT_ERROR).strip() if _CAPTURE_IMPORT_ERROR is not None else "未知错误"
            self.errorOccurred.emit(f"当前版本缺少采集模块，实时预览不可用。\n{detail}")
            return False

        def stop_preview(self) -> None:
            return

        def update_preview_target(self, preview_target: object | None) -> None:
            return

        def request_analysis_frame(self, request_id: int) -> bool:
            return False

try:
    from fdm.services.cu_scale_io import format_cu_scale_record_summary, parse_cu_scale_file

    _CU_SCALE_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    _CU_SCALE_IMPORT_ERROR = exc
    _cu_scale_import_error_message = str(exc)

    def parse_cu_scale_file(path: str | Path):
        raise RuntimeError(f"当前版本缺少 CU 标尺导入模块，无法导入标尺。\n{_cu_scale_import_error_message}")

    def format_cu_scale_record_summary(record) -> str:
        raise RuntimeError(f"当前版本缺少 CU 标尺导入模块，无法导入标尺。\n{_cu_scale_import_error_message}")


@dataclass(slots=True)
class PresetImportPlanEntry:
    preset: CalibrationPreset
    action: str
    final_name: str


class DigitalSlideWriteWorker(QObject):
    tileWritten = Signal(int, float)
    failed = Signal(str)
    drained = Signal()

    def __init__(self, path: str | Path, *, max_queue_size: int = 3) -> None:
        super().__init__()
        self._path = Path(path)
        self._queue: Queue[tuple[DigitalSlideTile, QImage]] = Queue(maxsize=max(1, int(max_queue_size)))
        self._lock = Lock()
        self._thread: Thread | None = None
        self._finish_requested = False
        self._cancel_requested = False
        self._written_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = Thread(target=self._run, name=f"fdm-digital-slide-write-{self._path.name}", daemon=True)
        self._thread.start()

    def enqueue(self, tile: DigitalSlideTile, image: QImage) -> bool:
        if image.isNull():
            return False
        with self._lock:
            if self._finish_requested or self._cancel_requested:
                return False
        try:
            self._queue.put_nowait((tile, image.copy()))
        except Full:
            return False
        return True

    def finish(self) -> None:
        with self._lock:
            self._finish_requested = True

    def cancel(self) -> None:
        with self._lock:
            self._cancel_requested = True
            self._finish_requested = True
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout_ms: int = 2000) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not current_thread():
            thread.join(max(0.0, timeout_ms / 1000.0))

    def _run(self) -> None:
        store = DigitalSlideStore(self._path)
        try:
            store.open()
            while True:
                with self._lock:
                    cancel_requested = self._cancel_requested
                    finish_requested = self._finish_requested
                if cancel_requested:
                    break
                try:
                    tile, image = self._queue.get(timeout=0.05)
                except Empty:
                    if finish_requested:
                        break
                    continue
                started_at = perf_counter()
                store.write_tile(tile, image)
                write_ms = (perf_counter() - started_at) * 1000.0
                self._written_count += 1
                self.tileWritten.emit(self._written_count, write_ms)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            try:
                store.close()
            except Exception:
                pass
            self.drained.emit()


class DigitalSlideZRangeRail(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(112, 220)
        self._soft_limit = 200_000
        self._current_z = 0
        self._lower_z: int | None = None
        self._upper_z: int | None = None

    def set_state(self, *, soft_limit: int, current_z: int, lower_z: int | None, upper_z: int | None) -> None:
        self._soft_limit = max(1, abs(int(soft_limit)))
        self._current_z = int(current_z)
        self._lower_z = int(lower_z) if lower_z is not None else None
        self._upper_z = int(upper_z) if upper_z is not None else None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        if not painter.isActive():
            return
        rect = self.rect().adjusted(8, 10, -8, -10)
        rail_x = rect.center().x()
        top = rect.top() + 16
        bottom = rect.bottom() - 30
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#64748B"), 2))
        painter.drawLine(rail_x, top, rail_x, bottom)
        painter.setPen(QPen(QColor("#94A3B8"), 1))
        painter.drawText(rect.left(), top + 4, f"+{self._soft_limit}")
        painter.drawText(rect.left(), bottom + 4, f"-{self._soft_limit}")

        lower = self._lower_z
        upper = self._upper_z
        if lower is not None and upper is not None:
            y1 = self._value_to_y(upper, top, bottom)
            y2 = self._value_to_y(lower, top, bottom)
            painter.fillRect(QRectF(rail_x - 10, min(y1, y2), 20, abs(y2 - y1)), QColor(96, 165, 250, 70))
        for value, label, color in (
            (upper, "上限", QColor("#60A5FA")),
            (lower, "下限", QColor("#34D399")),
        ):
            if value is None:
                continue
            y = self._value_to_y(value, top, bottom)
            painter.setPen(QPen(color, 2))
            painter.drawLine(rail_x - 22, y, rail_x + 22, y)
            painter.drawText(rail_x + 26, y + 4, label)
        current_y = self._value_to_y(self._current_z, top, bottom)
        painter.setBrush(QColor("#F97316"))
        painter.setPen(QPen(QColor("#FED7AA"), 2))
        painter.drawEllipse(QPoint(rail_x, current_y), 6, 6)
        painter.setPen(QPen(QColor("#F97316"), 1))
        painter.drawText(
            QRectF(rect.left(), rect.bottom() - 20, rect.width(), 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"当前 Z={self._current_z}",
        )

    def _value_to_y(self, value: int, top: int, bottom: int) -> int:
        limit = max(1, self._soft_limit)
        clamped = max(-limit, min(limit, int(value)))
        ratio = (clamped + limit) / (2.0 * limit)
        return int(round(bottom - (ratio * (bottom - top))))


class DigitalSlideRangeMap(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(132, 132)
        self._current_xy = (0, 0)
        self._bounds: dict[str, int] = {}
        self._stage_step = (5000, 5000)

    def set_state(
        self,
        *,
        current_xy: tuple[int, int],
        bounds: dict[str, int],
        stage_step: tuple[int, int] = (5000, 5000),
    ) -> None:
        self._current_xy = (int(current_xy[0]), int(current_xy[1]))
        self._bounds = dict(bounds)
        self._stage_step = (int(stage_step[0]), int(stage_step[1]))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.fillRect(rect, QColor("#0F172A") if self.palette().window().color().lightness() < 128 else QColor("#F8FAFC"))
        painter.setPen(QPen(QColor("#64748B"), 1))
        painter.drawRect(rect)

        current_x, current_y = self._current_xy
        step_x, step_y = self._stage_step
        fallback_span = 1
        current_corners = [
            (current_x, current_y),
            (current_x + (step_x if step_x != 0 else fallback_span), current_y),
            (current_x, current_y + (step_y if step_y != 0 else fallback_span)),
            (
                current_x + (step_x if step_x != 0 else fallback_span),
                current_y + (step_y if step_y != 0 else fallback_span),
            ),
        ]
        points: list[tuple[int, int]] = [self._current_xy, *current_corners]
        if {"left", "right", "top", "bottom"}.issubset(self._bounds):
            left = self._bounds["left"]
            right = self._bounds["right"]
            top = self._bounds["top"]
            bottom = self._bounds["bottom"]
            points.extend([(left, top), (right, bottom)])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if min_x == max_x:
            min_x -= 1
            max_x += 1
        if min_y == max_y:
            min_y -= 1
            max_y += 1

        def map_point(x: int, y: int) -> QPointF:
            px = rect.left() + ((x - min_x) / max(1, max_x - min_x)) * rect.width()
            py = rect.top() + ((y - min_y) / max(1, max_y - min_y)) * rect.height()
            return QPointF(px, py)

        if {"left", "right", "top", "bottom"}.issubset(self._bounds):
            p1 = map_point(self._bounds["left"], self._bounds["top"])
            p2 = map_point(self._bounds["right"], self._bounds["bottom"])
            selected = QRectF(p1, p2).normalized()
            painter.fillRect(selected, QColor(96, 165, 250, 55))
            painter.setPen(QPen(QColor("#60A5FA"), 2))
            painter.drawRect(selected)
        p1 = map_point(current_corners[0][0], current_corners[0][1])
        p2 = map_point(current_corners[-1][0], current_corners[-1][1])
        view_rect = QRectF(p1, p2).normalized()
        if view_rect.width() < 8 or view_rect.height() < 8:
            center = map_point(*self._current_xy)
            view_rect = QRectF(center.x() - 6, center.y() - 6, 12, 12)
        painter.setPen(QPen(QColor("#22C55E"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(view_rect)
        painter.setPen(QPen(QColor("#94A3B8"), 1))
        painter.drawText(rect.left() + 6, rect.top() + 18, "绿框: 当前视场")
        painter.drawText(rect.left() + 6, rect.bottom() - 8, "浅色: 采集范围")


class SmallObjectEnhancementPreviewWindow(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(300, 300)
        self._image: QImage | None = None
        self._workspace_box: tuple[int, int, int, int] | None = None
        self._scale = 1.0
        self._enhanced_size: tuple[int, int] | None = None
        self._positive_points: list[Point] = []
        self._negative_points: list[Point] = []
        self._polygon_px: list[Point] = []
        self._reject_reason = ""

    def set_preview(self, metadata: dict[str, object], polygon_px: list[Point]) -> bool:
        image = metadata.get("small_object_preview_image")
        box = metadata.get("small_object_workspace_box")
        if not isinstance(image, QImage) or image.isNull() or not isinstance(box, (tuple, list)) or len(box) != 4:
            self.hide()
            return False
        try:
            workspace_box = tuple(int(round(float(value))) for value in box)
            scale = float(metadata.get("small_object_scale", 1.0) or 1.0)
        except (TypeError, ValueError):
            self.hide()
            return False
        if workspace_box[2] <= workspace_box[0] or workspace_box[3] <= workspace_box[1]:
            self.hide()
            return False
        self._image = image.copy()
        self._workspace_box = workspace_box
        self._scale = max(1.0, scale)
        enhanced_size = metadata.get("small_object_enhanced_size")
        if isinstance(enhanced_size, (tuple, list)) and len(enhanced_size) == 2:
            try:
                self._enhanced_size = (int(enhanced_size[0]), int(enhanced_size[1]))
            except (TypeError, ValueError):
                self._enhanced_size = None
        else:
            self._enhanced_size = None
        positive_points = metadata.get("positive_points_px")
        negative_points = metadata.get("negative_points_px")
        self._positive_points = [point for point in positive_points if isinstance(point, Point)] if isinstance(positive_points, list) else []
        self._negative_points = [point for point in negative_points if isinstance(point, Point)] if isinstance(negative_points, list) else []
        self._polygon_px = [point for point in polygon_px if isinstance(point, Point)]
        self._reject_reason = str(metadata.get("small_object_reject_reason", "") or "").strip()
        self.update()
        return True

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(8, 13, 21, 238))
        if self._image is None or self._workspace_box is None:
            painter.end()
            return
        padding = 10
        footer_height = 28
        image_rect = QRectF(
            padding,
            padding,
            max(1, self.width() - padding * 2),
            max(1, self.height() - padding * 2 - footer_height),
        )
        scaled = self._image.scaled(
            image_rect.size().toSize(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        target = QRectF(
            image_rect.x() + (image_rect.width() - scaled.width()) / 2.0,
            image_rect.y() + (image_rect.height() - scaled.height()) / 2.0,
            scaled.width(),
            scaled.height(),
        )
        painter.drawImage(target, scaled)
        sx = target.width() / max(1, self._image.width())
        sy = target.height() / max(1, self._image.height())

        def map_point(point: Point) -> QPointF:
            x0, y0, _x1, _y1 = self._workspace_box or (0, 0, 1, 1)
            return QPointF(
                target.x() + (point.x - x0) * self._scale * sx,
                target.y() + (point.y - y0) * self._scale * sy,
            )

        if len(self._polygon_px) >= 3:
            polygon = QPolygonF([map_point(point) for point in self._polygon_px])
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#050505"), 4.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawPolygon(polygon)
            painter.setPen(QPen(QColor("#F87171"), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawPolygon(polygon)

        for positive, points in ((True, self._positive_points), (False, self._negative_points)):
            color = QColor("#34D399" if positive else "#FB7185")
            for point in points:
                if not DocumentCanvas.point_in_box(point, self._workspace_box):
                    continue
                widget_point = map_point(point)
                painter.setPen(QPen(QColor("#050505"), 2.0))
                painter.setBrush(color)
                painter.drawEllipse(widget_point, 5.0, 5.0)
                painter.setPen(QPen(QColor("#FFFFFF"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(QPointF(widget_point.x() - 3.0, widget_point.y()), QPointF(widget_point.x() + 3.0, widget_point.y()))
                if positive:
                    painter.drawLine(QPointF(widget_point.x(), widget_point.y() - 3.0), QPointF(widget_point.x(), widget_point.y() + 3.0))

        painter.setPen(QColor("#E5E7EB"))
        font = QFont(self.font())
        font.setPointSize(10)
        painter.setFont(font)
        size_text = ""
        if self._enhanced_size is not None:
            size_text = f"  {self._enhanced_size[0]}x{self._enhanced_size[1]}"
        state_text = "  请补点" if self._reject_reason else ""
        painter.drawText(
            QRectF(padding, self.height() - footer_height, self.width() - padding * 2, footer_height - 4),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"小洞 x{self._scale:.1f}{size_text}{state_text}",
        )
        painter.end()


class MainWindow(QMainWindow):
    IMAGE_FILTER = "图像与数字化切片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.fdmslide);;图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;数字化切片 (*.fdmslide)"
    PROJECT_FILTER = "Fiber 项目 (*.fdmproj)"
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    DIGITAL_SLIDE_SUFFIXES = {DIGITAL_SLIDE_SUFFIX}
    SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | DIGITAL_SLIDE_SUFFIXES
    MAP_BUILD_AVAILABLE = True
    PREVIEW_ANALYSIS_INTERVAL_MS = 300
    TABLE_COL_GROUP = 0
    TABLE_COL_KIND = 1
    TABLE_COL_RESULT = 2
    TABLE_COL_UNIT = 3
    TABLE_COL_MODE = 4
    TABLE_COL_CONFIDENCE = 5
    TABLE_COL_STATUS = 6
    TABLE_COL_ID = 7

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("显微测量工作台")
        self.setWindowIcon(application_icon())
        self.setAcceptDrops(True)

        self.project = ProjectState.empty()
        self._project_path: Path | None = None
        self._app_settings = AppSettingsIO.load()
        app = QApplication.instance()
        if app is not None:
            self._app_settings.theme_mode = apply_application_theme(app, self._app_settings.theme_mode)
        try:
            AppSettingsIO.save(self._app_settings)
        except OSError:
            pass
        self._document_order: list[str] = []
        self._images: dict[str, QImage] = {}
        self._canvases: dict[str, DocumentCanvas] = {}
        self._slide_stores: dict[str, DigitalSlideStore] = {}
        self._tool_mode = "select"
        self._last_non_select_tool: str | None = None
        self._manual_tool_mode = "manual"
        self._area_tool_mode = "polygon_area"
        self._overlay_tool_kind = OverlayAnnotationKind.TEXT
        self._group_list_rebuilding = False
        self._table_rebuilding = False
        self._file_toolbar: QToolBar | None = None
        self._measure_toolbar: QToolBar | None = None
        self._measurement_tool_strip: MeasurementToolStrip | None = None
        self._magic_tool_mode = MagicSegmentToolMode.STANDARD
        self._magic_standard_add_roi_enabled = bool(self._app_settings.magic_segment_standard_add_roi_enabled)
        self._magic_standard_subtract_roi_enabled = bool(self._app_settings.magic_segment_standard_subtract_roi_enabled)
        self._magic_standard_subtract_input_mode = MagicSegmentSubtractInputMode.normalize(
            getattr(self._app_settings, "magic_segment_standard_subtract_input_mode", MagicSegmentSubtractInputMode.SMART)
        )
        self._fiber_quick_roi_enabled = bool(self._app_settings.fiber_quick_roi_enabled)
        self._magic_tool_button: OverlayToolSplitButton | None = None
        self._magic_tool_menu: QMenu | None = None
        self._magic_subtool_actions: dict[str, QAction] = {}
        self._manual_tool_button: OverlayToolSplitButton | None = None
        self._manual_tool_menu: QMenu | None = None
        self._manual_subtool_actions: dict[str, QAction] = {}
        self._area_tool_button: OverlayToolSplitButton | None = None
        self._area_tool_menu: QMenu | None = None
        self._area_subtool_actions: dict[str, QAction] = {}
        self._overlay_tool_button: OverlayToolSplitButton | None = None
        self._overlay_tool_menu: QMenu | None = None
        self._overlay_subtool_actions: dict[str, QAction] = {}
        self._left_panel: QWidget | None = None
        self._main_splitter: QSplitter | None = None
        self._left_panel_splitter: QSplitter | None = None
        self._right_panel: QWidget | None = None
        self._left_standard_splitter: QSplitter | None = None
        self._digital_slide_left_panel: QWidget | None = None
        self._right_standard_panel: QWidget | None = None
        self._digital_slide_right_panel: QWidget | None = None
        self._digital_slide_mode = False
        self.digital_slide_action: QAction | None = None
        self.digital_slide_smooth_navigation_action: QAction | None = None
        self._digital_slide_status_label: QLabel | None = None
        self._digital_slide_camera_label: QLabel | None = None
        self._digital_slide_progress_label: QLabel | None = None
        self._digital_slide_progress_bar: QProgressBar | None = None
        self._digital_slide_elapsed_label: QLabel | None = None
        self._digital_slide_remaining_label: QLabel | None = None
        self._digital_slide_eta_label: QLabel | None = None
        self._digital_slide_timing_label: QLabel | None = None
        self._digital_slide_diagnostics_toggle: QToolButton | None = None
        self._digital_slide_diagnostics_details: QWidget | None = None
        self._digital_slide_diagnostics_summary_label: QLabel | None = None
        self._digital_slide_connection_toggle: QToolButton | None = None
        self._digital_slide_connection_summary_label: QLabel | None = None
        self._digital_slide_connection_details: QWidget | None = None
        self._digital_slide_stage_toggle: QToolButton | None = None
        self._digital_slide_stage_summary_label: QLabel | None = None
        self._digital_slide_stage_details: QWidget | None = None
        self._digital_slide_motor_card_label: QLabel | None = None
        self._digital_slide_port_card_label: QLabel | None = None
        self._digital_slide_camera_card_label: QLabel | None = None
        self._digital_slide_position_label: QLabel | None = None
        self._digital_slide_port_combo: QComboBox | None = None
        self._digital_slide_motor_enable: QCheckBox | None = None
        self._digital_slide_start_button: QPushButton | None = None
        self._digital_slide_stop_button: QPushButton | None = None
        self._digital_slide_output_path_edit: QLineEdit | None = None
        self._digital_slide_z_lower_edit: QLineEdit | None = None
        self._digital_slide_z_upper_edit: QLineEdit | None = None
        self._digital_slide_z_rail: DigitalSlideZRangeRail | None = None
        self._digital_slide_cols_edit: QLineEdit | None = None
        self._digital_slide_rows_edit: QLineEdit | None = None
        self._digital_slide_range_map: DigitalSlideRangeMap | None = None
        self._digital_slide_region_bounds: dict[str, int] = {}
        self._digital_slide_region_anchor_points: dict[str, tuple[int, int]] = {}
        self._digital_slide_rows_cols_manual = False
        self._digital_slide_z_step_spin: QSpinBox | None = None
        self._digital_slide_xy_jog_step_spin: QSpinBox | None = None
        self._digital_slide_focus_jog_step_spin: QSpinBox | None = None
        self._digital_slide_motion_settings_label: QLabel | None = None
        self._digital_slide_locked_controls: list[QWidget] = []
        self._digital_slide_motion_controls: list[QWidget] = []
        self._digital_slide_direction_buttons: dict[str, QPushButton] = {}
        self._group_header_labels: list[QLabel] = []
        self._prompt_request_tool_modes: dict[tuple[str, int], str] = {}
        self._fiber_quick_geometry_request_ids: set[tuple[str, int]] = set()
        self._fiber_quick_background_job_serial = 0
        self._fiber_quick_background_jobs: dict[tuple[str, int], dict[str, object]] = {}
        self._interactive_segmentation_services: dict[str, object] = {}
        self._show_area_fill = True
        self._area_auto_button: QPushButton | None = None
        self._magic_controls_widget: QWidget | None = None
        self._magic_prompt_label: QLabel | None = None
        self._magic_toggle_button: QToolButton | None = None
        self._magic_roi_button: QToolButton | None = None
        self._magic_small_object_button: QToolButton | None = None
        self._magic_options_button: QToolButton | None = None
        self._magic_options_menu: QMenu | None = None
        self._magic_roi_option_checkbox: QCheckBox | None = None
        self._magic_small_object_option_checkbox: QCheckBox | None = None
        self._magic_small_object_option_hint: QLabel | None = None
        self._magic_subtract_mode_button: QToolButton | None = None
        self._magic_subtract_mode_menu: QMenu | None = None
        self._magic_subtract_mode_actions: dict[str, QAction] = {}
        self._magic_operation_button: QToolButton | None = None
        self._magic_confirm_subtract_button: QToolButton | None = None
        self._magic_complete_button: QToolButton | None = None
        self._magic_cancel_button: QToolButton | None = None
        self._small_object_preview_window: SmallObjectEnhancementPreviewWindow | None = None
        self._count_controls_widget: QWidget | None = None
        self._preview_analysis_widget: QWidget | None = None
        self._path_controls_widget: QWidget | None = None
        self._area_operation_button: QToolButton | None = None
        self._path_complete_button: QToolButton | None = None
        self._path_cancel_button: QToolButton | None = None
        self._focus_stack_button: QToolButton | None = None
        self._map_build_button: QToolButton | None = None
        self._map_build_status_label: QLabel | None = None
        self._count_numbers_button: QToolButton | None = None
        self._add_preset_button: QPushButton | None = None
        self._edit_preset_button: QPushButton | None = None
        self._delete_preset_button: QPushButton | None = None
        self._import_cu_preset_button: QPushButton | None = None
        self._apply_preset_button: QPushButton | None = None
        self._add_group_button: QPushButton | None = None
        self._rename_group_button: QPushButton | None = None
        self.delete_group_button: QPushButton | None = None
        self._delete_group_measurements_button: QPushButton | None = None
        self._delete_all_measurements_button: QPushButton | None = None
        self._center_stack: QStackedWidget | None = None
        self._preview_page: QWidget | None = None
        self._preview_display_stack: QStackedWidget | None = None
        self._preview_canvas: DocumentCanvas | None = None
        self._microview_preview_host: MicroviewPreviewHost | None = None
        self._microview_preview_scroll: QScrollArea | None = None
        self._preview_status_label: QLabel | None = None
        self._image_resolution_label: QLabel | None = None
        self._calibration_label_scroll: QScrollArea | None = None
        self._calibration_status_card: QFrame | None = None
        self._calibration_status_title_label: QLabel | None = None
        self._calibration_status_summary_label: QLabel | None = None
        self._calibration_details_button: QPushButton | None = None
        self._calibration_details_label: QLabel | None = None
        self._calibration_start_button: QPushButton | None = None
        self._calibration_card_action_row: QWidget | None = None
        self._calibration_preset_action_row: QWidget | None = None
        self._version_label: QLabel | None = None
        self._preview_active = False
        self._preview_document: ImageDocument | None = None
        self._latest_preview_frame: QImage | None = None
        self._preview_frame_serial = 0
        self._last_digital_slide_focus_wheel_at = 0.0
        self._slide_motion = MotionController(parent=self)
        self._slide_jog_timer = QTimer(self)
        self._slide_jog_timer.timeout.connect(self._perform_digital_slide_jog_repeat)
        self._slide_jog_single_shot_timer = QTimer(self)
        self._slide_jog_single_shot_timer.setSingleShot(True)
        self._slide_jog_single_shot_timer.timeout.connect(self._activate_digital_slide_long_jog)
        self._slide_jog_request: dict[str, object] | None = None
        self._slide_acquisition_timer = QTimer(self)
        self._slide_acquisition_timer.setSingleShot(True)
        self._slide_acquisition_timer.timeout.connect(self._on_slide_acquisition_timer_timeout)
        self._slide_acquisition_plan: list[dict[str, int]] = []
        self._slide_acquisition_index = 0
        self._slide_acquisition_store: DigitalSlideStore | None = None
        self._slide_acquisition_path: Path | None = None
        self._slide_acquisition_document_path: str = ""
        self._slide_acquisition_metadata: dict[str, object] = {}
        self._slide_acquisition_settings: AppSettings | None = None
        self._slide_acquisition_writer: DigitalSlideWriteWorker | None = None
        self._slide_acquisition_finishing: tuple[str, str] | None = None
        self._slide_acquisition_discard_message: str | None = None
        self._slide_acquisition_pending_write: tuple[
            DigitalSlideTile,
            QImage,
            dict[str, float],
            dict[str, int],
        ] | None = None
        self._slide_acquisition_viewport_size: tuple[int, int] | None = None
        self._slide_acquisition_timer_phase = "idle"
        self._slide_acquisition_frame_marker = 0
        self._slide_acquisition_wait_started_at = 0.0
        self._slide_acquisition_settle_started_at = 0.0
        self._slide_acquisition_post_settle_started_at = 0.0
        self._slide_acquisition_move_ms = 0.0
        self._slide_acquisition_settle_ms = 0.0
        self._slide_acquisition_post_settle_ms = 0.0
        self._slide_acquisition_xy_moved = False
        self._slide_acquisition_z_moved = False
        self._slide_acquisition_xy_settle_wait_ms = 0
        self._slide_acquisition_z_settle_wait_ms = 0
        self._slide_acquisition_xy_post_wait_ms = 0
        self._slide_acquisition_z_post_wait_ms = 0
        self._slide_acquisition_settle_wait_ms = 0
        self._slide_acquisition_post_wait_ms = 0
        self._slide_acquisition_required_discard_frames = 0
        self._slide_acquisition_last_write_ms = 0.0
        self._slide_acquisition_started_at = 0.0
        self._slide_acquisition_initial_estimated_total_ms = 0.0
        self._slide_acquisition_last_timing_summary = ""
        self._capture_devices: list[CaptureDevice] = []
        self._microview_optimize_hints_shown: set[str] = set()
        self._project_clean_snapshot: dict[str, object] | None = None
        self._pending_project_load_snapshot = False
        self._capture_manager = CaptureSessionManager(
            selected_device_id=self._app_settings.selected_capture_device_id,
            refresh_on_init=False,
        )
        self._color_palette = [
            "#1F7A8C",
            "#E07A5F",
            "#81B29A",
            "#3D405B",
            "#F2CC8F",
            "#6D597A",
            "#227C9D",
            "#FF7C43",
            "#2A9D8F",
        ]

        self.export_service = ExportService()
        self.area_inference_service = AreaInferenceService()
        self.snap_service = SnapService()
        self.thread_task_manager = ThreadTaskManager(parent=self)
        self.background_task_controller = BackgroundTaskController(self, self.thread_task_manager, parent=self)
        self.preview_analysis_task_controller = PreviewAnalysisTaskController(
            self,
            self.thread_task_manager,
            parent=self,
        )
        self.project_session_controller = ProjectSessionController(self)
        self.export_controller = ExportController(self)

        self._build_ui()
        self._refresh_theme_sensitive_icons()
        self._capture_manager.devicesChanged.connect(self._on_capture_devices_changed)
        self._capture_manager.previewStateChanged.connect(self._on_live_preview_state_changed)
        self._capture_manager.frameReady.connect(self._on_live_preview_frame_ready)
        self._capture_manager.analysisFrameReady.connect(self._on_preview_analysis_frame_ready)
        self._capture_manager.analysisFrameFailed.connect(self._on_preview_analysis_frame_failed)
        self._capture_manager.errorOccurred.connect(self._on_capture_error)
        self._slide_motion.statusChanged.connect(self._on_digital_slide_motion_status)
        self._slide_motion.positionChanged.connect(self._on_digital_slide_position_changed)
        self._capture_devices = self._capture_manager.devices()
        self._refresh_preset_combo()
        self._update_capture_device_ui()
        self._restore_initial_window_geometry()
        self._update_ui_for_current_document()
        self._mark_project_saved()

    @property
    def _load_thread(self):
        return self.background_task_controller.thread(TASK_IMAGE_LOAD)

    @_load_thread.setter
    def _load_thread(self, value) -> None:
        if value is None:
            self.thread_task_manager.stop(TASK_IMAGE_LOAD, cancel=False)
        else:
            self.background_task_controller.register_external_thread(TASK_IMAGE_LOAD, value)

    @property
    def _load_worker(self):
        return self.background_task_controller.worker(TASK_IMAGE_LOAD)

    @_load_worker.setter
    def _load_worker(self, value) -> None:
        self.background_task_controller.set_worker_override(TASK_IMAGE_LOAD, value)

    @property
    def _load_state(self) -> BatchLoadState | None:
        return self.background_task_controller.load_state

    @property
    def _area_infer_thread(self):
        return self.background_task_controller.thread(TASK_AREA_INFERENCE)

    @_area_infer_thread.setter
    def _area_infer_thread(self, value) -> None:
        if value is None:
            self.thread_task_manager.stop(TASK_AREA_INFERENCE, cancel=False)
        else:
            self.background_task_controller.register_external_thread(TASK_AREA_INFERENCE, value)

    @property
    def _area_infer_worker(self):
        return self.background_task_controller.worker(TASK_AREA_INFERENCE)

    @_area_infer_worker.setter
    def _area_infer_worker(self, value) -> None:
        self.background_task_controller.set_worker_override(TASK_AREA_INFERENCE, value)

    @property
    def _area_infer_state(self) -> AreaInferenceBatchState | None:
        return self.background_task_controller.area_infer_state

    @property
    def _prompt_seg_thread(self):
        return self.background_task_controller.thread(TASK_PROMPT_SEGMENTATION)

    @_prompt_seg_thread.setter
    def _prompt_seg_thread(self, value) -> None:
        if value is None:
            self.thread_task_manager.stop(TASK_PROMPT_SEGMENTATION, cancel=False)
        else:
            self.background_task_controller.register_external_thread(TASK_PROMPT_SEGMENTATION, value)

    @property
    def _prompt_seg_worker(self):
        return self.background_task_controller.worker(TASK_PROMPT_SEGMENTATION)

    @_prompt_seg_worker.setter
    def _prompt_seg_worker(self, value) -> None:
        self.background_task_controller.set_worker_override(TASK_PROMPT_SEGMENTATION, value)

    @property
    def _fiber_quick_geometry_thread(self):
        return self.background_task_controller.thread(TASK_FIBER_QUICK_GEOMETRY)

    @_fiber_quick_geometry_thread.setter
    def _fiber_quick_geometry_thread(self, value) -> None:
        if value is None:
            self.thread_task_manager.stop(TASK_FIBER_QUICK_GEOMETRY, cancel=False)
        else:
            self.background_task_controller.register_external_thread(TASK_FIBER_QUICK_GEOMETRY, value)

    @property
    def _fiber_quick_geometry_worker(self):
        return self.background_task_controller.worker(TASK_FIBER_QUICK_GEOMETRY)

    @_fiber_quick_geometry_worker.setter
    def _fiber_quick_geometry_worker(self, value) -> None:
        self.background_task_controller.set_worker_override(TASK_FIBER_QUICK_GEOMETRY, value)

    @property
    def _fiber_quick_commit_geometry_thread(self):
        return self.background_task_controller.thread(TASK_FIBER_QUICK_COMMIT_GEOMETRY)

    @_fiber_quick_commit_geometry_thread.setter
    def _fiber_quick_commit_geometry_thread(self, value) -> None:
        if value is None:
            self.thread_task_manager.stop(TASK_FIBER_QUICK_COMMIT_GEOMETRY, cancel=False)
        else:
            self.background_task_controller.register_external_thread(TASK_FIBER_QUICK_COMMIT_GEOMETRY, value)

    @property
    def _fiber_quick_commit_geometry_worker(self):
        return self.background_task_controller.worker(TASK_FIBER_QUICK_COMMIT_GEOMETRY)

    @_fiber_quick_commit_geometry_worker.setter
    def _fiber_quick_commit_geometry_worker(self, value) -> None:
        self.background_task_controller.set_worker_override(TASK_FIBER_QUICK_COMMIT_GEOMETRY, value)

    @property
    def _reference_instance_thread(self):
        return self.background_task_controller.thread(TASK_REFERENCE_INSTANCE)

    @_reference_instance_thread.setter
    def _reference_instance_thread(self, value) -> None:
        if value is None:
            self.thread_task_manager.stop(TASK_REFERENCE_INSTANCE, cancel=False)
        else:
            self.background_task_controller.register_external_thread(TASK_REFERENCE_INSTANCE, value)

    @property
    def _reference_instance_worker(self):
        return self.background_task_controller.worker(TASK_REFERENCE_INSTANCE)

    @_reference_instance_worker.setter
    def _reference_instance_worker(self, value) -> None:
        self.background_task_controller.set_worker_override(TASK_REFERENCE_INSTANCE, value)

    @property
    def _preview_analysis_mode(self) -> str:
        return self.preview_analysis_task_controller.mode

    @_preview_analysis_mode.setter
    def _preview_analysis_mode(self, value: str) -> None:
        self.preview_analysis_task_controller.mode = value

    @property
    def _preview_analysis_dialog(self):
        return self.preview_analysis_task_controller.dialog

    @_preview_analysis_dialog.setter
    def _preview_analysis_dialog(self, value) -> None:
        self.preview_analysis_task_controller.dialog = value

    @property
    def _preview_analysis_worker(self):
        return self.preview_analysis_task_controller.worker

    @_preview_analysis_worker.setter
    def _preview_analysis_worker(self, value) -> None:
        self.preview_analysis_task_controller.worker = value

    @property
    def _preview_analysis_finalizing(self) -> bool:
        return self.preview_analysis_task_controller.finalizing

    @_preview_analysis_finalizing.setter
    def _preview_analysis_finalizing(self, value: bool) -> None:
        self.preview_analysis_task_controller.finalizing = bool(value)

    def _build_ui(self) -> None:
        self.setStatusBar(QStatusBar())
        self._version_label = QLabel(f"v{__version__}")
        self._version_label.setToolTip(f"{self.windowTitle()} {__version__}")
        self._version_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.statusBar().addPermanentWidget(self._version_label, 0)
        self._update_statusbar_aux_labels()
        self._create_actions()
        self._build_menus()
        self._build_toolbar()

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if self._measurement_tool_strip is not None:
            layout.addWidget(self._measurement_tool_strip)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter = splitter
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([380, 770, 360])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(container)

    def _create_actions(self) -> None:
        self.open_images_action = QAction("打开图片", self)
        self.open_images_action.setIcon(themed_icon("open_images", color="#D7E3FC"))
        self.open_images_action.setShortcut("Ctrl+O")
        self.open_images_action.triggered.connect(self.open_images)

        self.open_folder_action = QAction("打开文件夹", self)
        self.open_folder_action.setIcon(themed_icon("open_folder", color="#D7E3FC"))
        self.open_folder_action.triggered.connect(self.open_folder)

        self.open_project_action = QAction("打开项目", self)
        self.open_project_action.setIcon(themed_icon("open_project", color="#D7E3FC"))
        self.open_project_action.triggered.connect(self.load_project)

        self.save_project_action = QAction("保存项目", self)
        self.save_project_action.setIcon(themed_icon("save_project", color="#D7E3FC"))
        self.save_project_action.setShortcut("Ctrl+S")
        self.save_project_action.triggered.connect(lambda: self.save_project())

        self.close_current_action = QAction("关闭当前图片", self)
        self.close_current_action.setIcon(themed_icon("close_current", color="#F2B5A7"))
        self.close_current_action.setShortcut("Ctrl+W")
        self.close_current_action.triggered.connect(self.close_current_document)

        self.close_all_action = QAction("关闭所有图片", self)
        self.close_all_action.setIcon(themed_icon("close_all", color="#F2B5A7"))
        self.close_all_action.setShortcut("Ctrl+Shift+W")
        self.close_all_action.triggered.connect(self.close_all_documents)

        self.switch_capture_device_action = QAction("切换采集设备", self)
        self.switch_capture_device_action.setIcon(themed_icon("capture_device", color="#D7E3FC"))
        self.switch_capture_device_action.triggered.connect(self.show_capture_device_menu)

        self.live_preview_action = QAction("实时预览", self)
        self.live_preview_action.setCheckable(True)
        self.live_preview_action.setIcon(themed_icon("live_preview", color="#7BD389"))
        self.live_preview_action.triggered.connect(self.toggle_live_preview)

        self.digital_slide_action = QAction("数字化切片", self)
        self.digital_slide_action.setCheckable(True)
        self.digital_slide_action.setIcon(themed_icon("capture_frame", color="#7BD389"))
        self.digital_slide_action.setToolTip("进入数字化切片采集工作台")
        self.digital_slide_action.triggered.connect(self._toggle_digital_slide_mode)

        self.capture_frame_action = QAction("采集一张", self)
        self.capture_frame_action.setIcon(themed_icon("capture_frame", color="#F4D35E"))
        self.capture_frame_action.triggered.connect(self.capture_current_frame)

        self.optimize_capture_signal_action = QAction("优化采集参数", self)
        self.optimize_capture_signal_action.setIcon(themed_icon("capture_device", color="#7BD389"))
        self.optimize_capture_signal_action.triggered.connect(self.optimize_capture_signal)

        self.undo_action = QAction("撤回", self)
        self.undo_action.setIcon(themed_icon("undo", color="#E7ECEF"))
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo_current_document)

        self.redo_action = QAction("重做", self)
        self.redo_action.setIcon(themed_icon("redo", color="#E7ECEF"))
        self.redo_action.setShortcut("Ctrl+Shift+Z")
        self.redo_action.triggered.connect(self.redo_current_document)

        self.delete_measurement_action = QAction("删除选中对象", self)
        self.delete_measurement_action.setIcon(themed_icon("delete", color="#F28482"))
        self.delete_measurement_action.setShortcut("Delete")
        self.delete_measurement_action.triggered.connect(self.delete_selected_measurement)

        self.add_group_action = QAction("新增类别", self)
        self.add_group_action.setIcon(themed_icon("add", color="#7BD389"))
        self.add_group_action.triggered.connect(self.add_fiber_group)

        self.rename_group_action = QAction("编辑当前类别", self)
        self.rename_group_action.setIcon(themed_icon("rename", color="#D7E3FC"))
        self.rename_group_action.triggered.connect(self.rename_active_group)

        self.delete_group_action = QAction("删除当前类别", self)
        self.delete_group_action.setIcon(themed_icon("delete", color="#F28482"))
        self.delete_group_action.triggered.connect(self.delete_active_group)

        self.fit_action = QAction("适应窗口", self)
        self.fit_action.setIcon(themed_icon("fit", color="#E7ECEF"))
        self.fit_action.triggered.connect(self.fit_current_image)

        self.actual_size_action = QAction("原始像素", self)
        self.actual_size_action.setIcon(themed_icon("actual_size", color="#E7ECEF"))
        self.actual_size_action.triggered.connect(self.actual_size_current_image)

        self.digital_slide_smooth_navigation_action = QAction("平滑移动", self)
        self.digital_slide_smooth_navigation_action.setCheckable(True)
        self.digital_slide_smooth_navigation_action.setShortcut("M")
        self.digital_slide_smooth_navigation_action.setIcon(themed_icon("select", color="#D7E3FC"))
        self.digital_slide_smooth_navigation_action.setToolTip("切换数字化切片方向键步进/平滑移动")
        self.digital_slide_smooth_navigation_action.triggered.connect(self._set_current_digital_slide_smooth_navigation)
        self.digital_slide_smooth_navigation_action.setEnabled(False)

        self.settings_action = QAction("设置", self)
        self.settings_action.setMenuRole(QAction.MenuRole.NoRole)
        self.settings_action.setIcon(themed_icon("rename", color="#D7E3FC"))
        self.settings_action.triggered.connect(self.open_settings_dialog)

        self.shortcuts_help_action = QAction("快捷键说明", self)
        self.shortcuts_help_action.triggered.connect(self.open_shortcut_help_dialog)

        self.export_actions: list[QAction] = []
        self.export_actions.append(
            self._make_export_action(
                "导出测量叠加图",
                ExportSelection(
                    include_measurement_overlay=True,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                ),
            )
        )
        self.export_actions.append(
            self._make_export_action(
                "导出比例尺图",
                ExportSelection(
                    include_scale_overlay=True,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                ),
            )
        )
        self.export_actions.append(
            self._make_export_action(
                "导出测量 + 比例尺叠加图",
                ExportSelection(
                    include_combined_overlay=True,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                ),
            )
        )
        self.export_actions.append(self._make_export_action("导出比例尺 JSON", ExportSelection(include_scale_json=True)))
        self.export_actions.append(self._make_export_action("导出 Excel", ExportSelection(include_excel=True)))
        self.export_actions.append(self._make_export_action("导出 CSV", ExportSelection(include_csv=True)))
        self.export_actions.append(
            self._make_export_action(
                "导出叠加图 + Excel",
                ExportSelection(
                    include_measurement_overlay=True,
                    include_excel=True,
                    render_mode=ExportImageRenderMode.FULL_RESOLUTION,
                ),
            )
        )

        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        self._mode_actions: dict[str, QAction] = {}
        for mode, label in [
            ("select", "浏览"),
            ("manual", "手动线段"),
            ("continuous_manual", "连续测量"),
            ("count", "计数"),
            ("snap", "边缘吸附"),
            ("polygon_area", "多边形面积"),
            ("freehand_area", "自由形状面积"),
            (MagicSegmentToolMode.STANDARD, "标准魔棒"),
            (MagicSegmentToolMode.REFERENCE, "同类扩选"),
            (MagicSegmentToolMode.FIBER_QUICK, "快速测径"),
            ("calibration", "标定"),
            ("overlay", "叠加标注"),
        ]:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, value=mode: self.set_tool_mode(value))
            self._mode_actions[mode] = action
            mode_group.addAction(action)
        self._mode_actions["select"].setChecked(True)
        self._mode_actions["select"].setIcon(themed_icon("select", color="#D4D8DD"))
        self._mode_actions["manual"].setIcon(themed_icon("manual", color="#F4D35E"))
        self._mode_actions["continuous_manual"].setIcon(themed_icon("continuous_manual", color="#F4D35E"))
        self._mode_actions["count"].setIcon(themed_icon("count", color="#F08B95"))
        self._mode_actions["snap"].setIcon(themed_icon("snap", color="#7BD389"))
        self._mode_actions["polygon_area"].setIcon(themed_icon("polygon_area", color="#7BD389"))
        self._mode_actions["freehand_area"].setIcon(themed_icon("freehand_area", color="#9C89B8"))
        self._mode_actions[MagicSegmentToolMode.STANDARD].setIcon(self._magic_tool_icon(MagicSegmentToolMode.STANDARD))
        self._mode_actions[MagicSegmentToolMode.REFERENCE].setIcon(self._magic_tool_icon(MagicSegmentToolMode.REFERENCE))
        self._mode_actions[MagicSegmentToolMode.FIBER_QUICK].setIcon(self._magic_tool_icon(MagicSegmentToolMode.FIBER_QUICK))
        self._mode_actions["calibration"].setIcon(themed_icon("calibration", color="#FF7F50"))
        self._mode_actions["overlay"].setIcon(self._overlay_tool_icon())

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        file_menu.addAction(self.open_images_action)
        file_menu.addAction(self.open_folder_action)
        file_menu.addSeparator()
        file_menu.addAction(self.open_project_action)
        file_menu.addAction(self.save_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.close_current_action)
        file_menu.addAction(self.close_all_action)
        export_menu = file_menu.addMenu("导出")
        for action in self.export_actions:
            export_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.settings_action)

        edit_menu = self.menuBar().addMenu("编辑")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.delete_measurement_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.add_group_action)
        edit_menu.addAction(self.rename_group_action)
        edit_menu.addAction(self.delete_group_action)

        tool_menu = self.menuBar().addMenu("工具")
        for action in self._mode_actions.values():
            tool_menu.addAction(action)

        view_menu = self.menuBar().addMenu("视图")
        view_menu.addAction(self.fit_action)
        view_menu.addAction(self.actual_size_action)
        view_menu.addAction(self.digital_slide_smooth_navigation_action)

        help_menu = self.menuBar().addMenu("帮助")
        help_menu.addAction(self.shortcuts_help_action)

    def _build_toolbar(self) -> None:
        file_toolbar = QToolBar("文件工具栏")
        file_toolbar.setMovable(False)
        file_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        file_toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(file_toolbar)
        self._file_toolbar = file_toolbar
        file_toolbar.addAction(self.open_images_action)
        file_toolbar.addAction(self.open_folder_action)
        file_toolbar.addAction(self.open_project_action)
        file_toolbar.addAction(self.save_project_action)
        file_toolbar.addSeparator()

        export_button = QToolButton(self)
        export_button.setText("导出")
        export_button.setIcon(themed_icon("export", color="#D7E3FC"))
        export_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QMenu(export_button)
        for action in self.export_actions:
            export_menu.addAction(action)
        export_button.setMenu(export_menu)
        file_toolbar.addWidget(export_button)
        file_toolbar.addSeparator()
        file_toolbar.addAction(self.fit_action)
        file_toolbar.addAction(self.actual_size_action)
        file_toolbar.addAction(self.digital_slide_smooth_navigation_action)
        file_toolbar.addSeparator()
        file_toolbar.addAction(self.close_current_action)
        file_toolbar.addAction(self.close_all_action)
        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        file_toolbar.addWidget(spacer)
        file_toolbar.addSeparator()
        file_toolbar.addAction(self.switch_capture_device_action)
        file_toolbar.addAction(self.live_preview_action)
        file_toolbar.addAction(self.digital_slide_action)
        file_toolbar.addAction(self.capture_frame_action)
        file_toolbar.addAction(self.optimize_capture_signal_action)

        self._measure_toolbar = None
        self._measurement_tool_strip = self._build_measurement_tool_strip()
        self._update_count_numbers_button()

    def _build_measurement_tool_strip(self) -> MeasurementToolStrip:
        strip = MeasurementToolStrip(self)
        strip.addModeAction("select", self._mode_actions["select"])
        self._manual_tool_button = self._build_manual_tool_button()
        strip.addSplitModeButton("manual", self._manual_tool_button, aliases=["continuous_manual"])
        strip.addModeAction("count", self._mode_actions["count"])
        self._count_controls_widget = self._build_count_controls()
        strip.setCountContextWidget(self._count_controls_widget)
        strip.addModeAction("snap", self._mode_actions["snap"])
        self._area_tool_button = self._build_area_tool_button()
        strip.addSplitModeButton("polygon_area", self._area_tool_button, aliases=["freehand_area"])
        self._magic_tool_button = self._build_magic_tool_button()
        strip.setMagicToolButton(self._magic_tool_button)
        strip.addModeAction("calibration", self._mode_actions["calibration"])
        self._overlay_tool_button = self._build_overlay_tool_button()
        strip.setOverlayButton(self._overlay_tool_button)
        self._magic_controls_widget = self._build_magic_segment_controls()
        strip.setMagicContextWidget(self._magic_controls_widget)
        self._preview_analysis_widget = self._build_preview_analysis_controls()
        strip.setPreviewContextWidget(self._preview_analysis_widget)
        self._path_controls_widget = self._build_path_drawing_controls()
        strip.setPathContextWidget(self._path_controls_widget)
        strip.setActiveMode(self._tool_mode)
        return strip

    def _build_count_controls(self) -> QWidget:
        container = QWidget(self)
        layout = FlowLayout(container, h_spacing=6, v_spacing=6)
        container.setLayout(layout)

        self._count_numbers_button = QToolButton(container)
        self._count_numbers_button.setProperty("contextTool", True)
        self._count_numbers_button.setCheckable(True)
        self._count_numbers_button.setText("编号关")
        self._count_numbers_button.setToolTip("显示或隐藏当前图片的计数编号")
        self._count_numbers_button.toggled.connect(self._toggle_count_numbers)
        layout.addWidget(self._count_numbers_button)

        return container

    def _build_magic_segment_controls(self) -> QWidget:
        container = QWidget(self)
        layout = FlowLayout(container, h_spacing=6, v_spacing=6)
        container.setLayout(layout)
        self._magic_prompt_label = QLabel(container)
        self._magic_prompt_label.setProperty("contextLabel", True)
        layout.addWidget(self._magic_prompt_label)

        self._magic_operation_button = QToolButton(container)
        self._magic_operation_button.setProperty("contextTool", True)
        self._magic_operation_button.setText("添加(T)")
        self._magic_operation_button.clicked.connect(self._cycle_magic_segment_operation_mode)
        layout.addWidget(self._magic_operation_button)

        self._magic_subtract_mode_button = QToolButton(container)
        self._magic_subtract_mode_button.setProperty("contextTool", True)
        self._magic_subtract_mode_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._magic_subtract_mode_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._magic_subtract_mode_menu = self._build_magic_subtract_mode_menu(self._magic_subtract_mode_button)
        self._magic_subtract_mode_button.setMenu(self._magic_subtract_mode_menu)
        layout.addWidget(self._magic_subtract_mode_button)

        self._magic_toggle_button = QToolButton(container)
        self._magic_toggle_button.setProperty("contextTool", True)
        self._magic_toggle_button.setText("正负(R)")
        self._magic_toggle_button.clicked.connect(self._cycle_active_magic_prompt_type)
        layout.addWidget(self._magic_toggle_button)

        self._magic_options_button = QToolButton(container)
        self._magic_options_button.setProperty("contextTool", True)
        self._magic_options_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._magic_options_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._magic_options_button.setMenu(self._build_magic_options_menu(self._magic_options_button))
        layout.addWidget(self._magic_options_button)

        self._magic_confirm_subtract_button = QToolButton(container)
        self._magic_confirm_subtract_button.setProperty("contextTool", True)
        self._magic_confirm_subtract_button.setText("加洞(S)")
        self._magic_confirm_subtract_button.setToolTip("确认当前剔除形状，并继续添加下一块剔除区域")
        self._magic_confirm_subtract_button.clicked.connect(self._confirm_current_magic_subtract_shape)
        layout.addWidget(self._magic_confirm_subtract_button)

        self._magic_complete_button = QToolButton(container)
        self._magic_complete_button.setProperty("contextTool", True)
        self._magic_complete_button.setText("完成")
        self._magic_complete_button.setToolTip("提交当前魔棒结果（Enter / F）")
        self._magic_complete_button.clicked.connect(self._commit_active_magic_preview)
        layout.addWidget(self._magic_complete_button)

        self._magic_cancel_button = QToolButton(container)
        self._magic_cancel_button.setProperty("contextTool", True)
        self._magic_cancel_button.setText("取消")
        self._magic_cancel_button.setToolTip("取消当前魔棒会话或草稿（Esc）")
        self._magic_cancel_button.clicked.connect(self._cancel_active_magic_session)
        layout.addWidget(self._magic_cancel_button)

        return container

    def _build_magic_subtract_mode_menu(self, parent: QWidget) -> QMenu:
        menu = QMenu(parent)
        action_group = QActionGroup(menu)
        action_group.setExclusive(True)
        definitions = [
            (MagicSegmentSubtractInputMode.SMART, "智能剔除"),
            (MagicSegmentSubtractInputMode.POLYGON, "多边形剔除"),
            (MagicSegmentSubtractInputMode.FREEHAND, "自由圈选剔除"),
        ]
        for mode, label in definitions:
            action = QAction(label, menu)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, value=mode: self._set_magic_subtract_input_mode(value))
            menu.addAction(action)
            action_group.addAction(action)
            self._magic_subtract_mode_actions[mode] = action
        return menu

    def _build_magic_options_menu(self, parent: QWidget) -> QMenu:
        menu = QMenu(parent)
        panel = QWidget(menu)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        title = QLabel("魔棒选项", panel)
        title.setStyleSheet("font-weight: 700;")
        layout.addWidget(title)

        self._magic_roi_option_checkbox = QCheckBox("ROI 局部分割", panel)
        self._magic_roi_option_checkbox.setToolTip("只在当前 ROI 区域内进行魔棒识别或剔除，快捷键 Y")
        self._magic_roi_option_checkbox.toggled.connect(self._set_active_magic_roi_checked)
        layout.addWidget(self._magic_roi_option_checkbox)

        self._magic_small_object_option_checkbox = QCheckBox("小洞增强", panel)
        self._magic_small_object_option_checkbox.setToolTip("标准魔棒剔除小目标时使用局部上采样增强")
        self._magic_small_object_option_checkbox.toggled.connect(self._toggle_magic_small_object_enhancement)
        layout.addWidget(self._magic_small_object_option_checkbox)

        self._magic_small_object_option_hint = QLabel("仅在标准魔棒的智能剔除、ROI 开启时可用", panel)
        self._magic_small_object_option_hint.setWordWrap(True)
        self._magic_small_object_option_hint.setStyleSheet(f"color: {self._status_color('muted')};")
        layout.addWidget(self._magic_small_object_option_hint)

        widget_action = QWidgetAction(menu)
        widget_action.setDefaultWidget(panel)
        menu.addAction(widget_action)
        self._magic_options_menu = menu
        return menu

    def _build_preview_analysis_controls(self) -> QWidget:
        container = QWidget(self)
        layout = FlowLayout(container, h_spacing=6, v_spacing=6)
        container.setLayout(layout)

        header_button = QToolButton(container)
        header_button.setProperty("contextTool", True)
        header_button.setText("预览分析")
        header_button.setCursor(Qt.CursorShape.ArrowCursor)
        header_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        layout.addWidget(header_button)

        self._focus_stack_button = QToolButton(container)
        self._focus_stack_button.setProperty("contextTool", True)
        self._focus_stack_button.setText("景深合成")
        self._focus_stack_button.setCheckable(True)
        self._focus_stack_button.clicked.connect(lambda checked=False: self._toggle_preview_analysis_mode("focus_stack", checked))
        layout.addWidget(self._focus_stack_button)

        self._map_build_button = QToolButton(container)
        self._map_build_button.setProperty("contextTool", True)
        self._map_build_button.setText("地图构建")
        self._map_build_button.setCheckable(True)
        self._map_build_button.clicked.connect(lambda checked=False: self._toggle_preview_analysis_mode("map_build", checked))
        layout.addWidget(self._map_build_button)
        self._map_build_status_label = None

        return container

    def _build_path_drawing_controls(self) -> QWidget:
        container = QWidget(self)
        layout = FlowLayout(container, h_spacing=6, v_spacing=6)
        container.setLayout(layout)

        self._area_operation_button = QToolButton(container)
        self._area_operation_button.setProperty("contextTool", True)
        self._area_operation_button.setText("添加(T)")
        self._area_operation_button.setToolTip("切换当前面积工具的添加/剔除状态（T）")
        self._area_operation_button.clicked.connect(self._cycle_area_edit_operation_mode)
        layout.addWidget(self._area_operation_button)

        self._path_complete_button = QToolButton(container)
        self._path_complete_button.setProperty("contextTool", True)
        self._path_complete_button.setText("完成")
        self._path_complete_button.setToolTip("完成当前绘制（Enter / F）")
        self._path_complete_button.clicked.connect(self._commit_active_path_drawing)
        layout.addWidget(self._path_complete_button)

        self._path_cancel_button = QToolButton(container)
        self._path_cancel_button.setProperty("contextTool", True)
        self._path_cancel_button.setText("取消")
        self._path_cancel_button.setToolTip("取消当前绘制（Esc）")
        self._path_cancel_button.clicked.connect(self._cancel_active_path_drawing)
        layout.addWidget(self._path_cancel_button)

        return container

    def _manual_tool_definitions(self) -> list[tuple[str, str]]:
        return [
            ("manual", "手动线段"),
            ("continuous_manual", "连续测量"),
        ]

    def _area_tool_definitions(self) -> list[tuple[str, str]]:
        return [
            ("polygon_area", "多边形面积"),
            ("freehand_area", "自由形状面积"),
        ]

    def _overlay_tool_definitions(self) -> list[tuple[str, str, str]]:
        return [
            (OverlayAnnotationKind.TEXT, "文字", "rename"),
            (OverlayAnnotationKind.RECT, "矩形", "overlay_rect"),
            (OverlayAnnotationKind.CIRCLE, "圆形", "overlay_circle"),
            (OverlayAnnotationKind.LINE, "直线", "overlay_line"),
            (OverlayAnnotationKind.ARROW, "箭头", "overlay_arrow"),
        ]

    def _magic_tool_definitions(self) -> list[tuple[str, str]]:
        return [
            (MagicSegmentToolMode.STANDARD, "标准魔棒"),
            (MagicSegmentToolMode.REFERENCE, "同类扩选"),
            (MagicSegmentToolMode.FIBER_QUICK, "快速测径"),
        ]

    def _manual_tool_label(self, tool_mode: str) -> str:
        for mode, label in self._manual_tool_definitions():
            if mode == tool_mode:
                return label
        return "手动线段"

    def _manual_tool_icon(self, tool_mode: str, *, active: bool = False) -> QIcon:
        color = "#F7C948" if active else "#D9A72A"
        if tool_mode == "continuous_manual":
            return themed_icon("continuous_manual", color=color)
        return themed_icon("manual", color=color)

    def _activate_manual_tool(self, tool_mode: str) -> None:
        if tool_mode not in {item[0] for item in self._manual_tool_definitions()}:
            tool_mode = "manual"
        self._manual_tool_mode = tool_mode
        self.set_tool_mode(tool_mode)

    def _area_tool_label(self, tool_mode: str) -> str:
        for mode, label in self._area_tool_definitions():
            if mode == tool_mode:
                return label
        return "多边形面积"

    def _area_tool_icon(self, tool_mode: str, *, active: bool = False) -> QIcon:
        if tool_mode == "freehand_area":
            color = "#C2A1E6" if active else "#9C89B8"
            return themed_icon("freehand_area", color=color)
        color = "#7BD389" if active else "#5AAE69"
        return themed_icon("polygon_area", color=color)

    def _activate_area_tool(self, tool_mode: str) -> None:
        if tool_mode not in {item[0] for item in self._area_tool_definitions()}:
            tool_mode = "polygon_area"
        self._area_tool_mode = tool_mode
        self.set_tool_mode(tool_mode)

    def _magic_tool_label(self, tool_mode: str) -> str:
        for mode, label in self._magic_tool_definitions():
            if mode == tool_mode:
                return label
        return "标准魔棒"

    def _magic_tool_icon(self, tool_mode: str, *, active: bool = False) -> QIcon:
        if tool_mode == MagicSegmentToolMode.REFERENCE:
            color = "#7FD6E0" if active else "#5CB9C9"
        elif tool_mode == MagicSegmentToolMode.FIBER_QUICK:
            color = "#F7C948" if active else "#D9A72A"
        else:
            color = "#F08B95" if active else "#D96C75"
        return themed_icon("magic_segment", color=color)

    def _activate_magic_tool(self, tool_mode: str) -> None:
        if not is_magic_toolbar_tool_mode(tool_mode):
            tool_mode = MagicSegmentToolMode.STANDARD
        self._magic_tool_mode = tool_mode
        self.set_tool_mode(tool_mode)

    def _build_split_menu_stylesheet(self, object_name: str, checked_rgba: str) -> str:
        if self._is_dark_palette():
            background = "#23282E"
            border = "rgba(255, 255, 255, 20)"
            text = "#F3F4F6"
            selected = "#2D343C"
        else:
            background = "#FFFFFF"
            border = "rgba(17, 24, 39, 16)"
            text = "#1F2933"
            selected = "#EAF2F4"
        return f"""
            QMenu#{object_name} {{
                background: {background};
                border: 1px solid {border};
                border-radius: 12px;
                padding: 8px;
            }}
            QMenu#{object_name}::item {{
                min-height: 38px;
                margin: 2px 0;
                padding: 0 16px 0 12px;
                border-radius: 8px;
                color: {text};
                font-weight: 600;
            }}
            QMenu#{object_name}::item:selected {{
                background: {selected};
            }}
            QMenu#{object_name}::item:checked {{
                background: {checked_rgba};
            }}
            QMenu#{object_name}::icon {{
                padding-left: 2px;
            }}
            QMenu#{object_name}::indicator {{
                width: 0px;
                height: 0px;
            }}
        """

    def _build_manual_tool_button(self) -> OverlayToolSplitButton:
        button = OverlayToolSplitButton(self)
        button.setText("手动测量")
        button.primaryTriggered.connect(lambda: self._activate_manual_tool(self._manual_tool_mode))

        menu = QMenu(self)
        menu.setObjectName("manualToolMenu")
        menu.setStyleSheet(self._build_split_menu_stylesheet("manualToolMenu", "rgba(217, 167, 42, 41)"))
        for tool_mode, label in self._manual_tool_definitions():
            action = QAction(label, menu)
            action.setCheckable(True)
            action.setIcon(self._manual_tool_icon(tool_mode))
            action.triggered.connect(lambda checked=False, manual_mode=tool_mode: self._activate_manual_tool(manual_mode))
            menu.addAction(action)
            self._manual_subtool_actions[tool_mode] = action
        button.setMenu(menu)
        self._manual_tool_menu = menu
        self._sync_manual_tool_button()
        return button

    def _sync_manual_tool_button(self) -> None:
        active_mode = self._tool_mode if self._tool_mode in {mode for mode, _ in self._manual_tool_definitions()} else self._manual_tool_mode
        icon = self._manual_tool_icon(active_mode, active=self._tool_mode in {mode for mode, _ in self._manual_tool_definitions()})
        tooltip = f"手动测量（当前：{self._manual_tool_label(active_mode)}）"
        if self._manual_tool_button is not None:
            self._manual_tool_button.blockSignals(True)
            self._manual_tool_button.setText(self._manual_tool_label(active_mode))
            self._manual_tool_button.setChecked(self._tool_mode in {mode for mode, _ in self._manual_tool_definitions()})
            self._manual_tool_button.setCurrentTool(active_mode, icon)
            self._manual_tool_button.setToolTip(tooltip)
            self._manual_tool_button.blockSignals(False)
        for tool_mode, action in self._manual_subtool_actions.items():
            action.setChecked(tool_mode == active_mode)
            action.setIcon(self._manual_tool_icon(tool_mode))

    def _build_area_tool_button(self) -> OverlayToolSplitButton:
        button = OverlayToolSplitButton(self)
        button.setText("面积测量")
        button.primaryTriggered.connect(lambda: self._activate_area_tool(self._area_tool_mode))

        menu = QMenu(self)
        menu.setObjectName("areaToolMenu")
        menu.setStyleSheet(self._build_split_menu_stylesheet("areaToolMenu", "rgba(90, 174, 105, 41)"))
        for tool_mode, label in self._area_tool_definitions():
            action = QAction(label, menu)
            action.setCheckable(True)
            action.setIcon(self._area_tool_icon(tool_mode))
            action.triggered.connect(lambda checked=False, area_mode=tool_mode: self._activate_area_tool(area_mode))
            menu.addAction(action)
            self._area_subtool_actions[tool_mode] = action
        button.setMenu(menu)
        self._area_tool_menu = menu
        self._sync_area_tool_button()
        return button

    def _sync_area_tool_button(self) -> None:
        active_mode = self._tool_mode if self._tool_mode in {mode for mode, _ in self._area_tool_definitions()} else self._area_tool_mode
        icon = self._area_tool_icon(active_mode, active=self._tool_mode in {mode for mode, _ in self._area_tool_definitions()})
        tooltip = f"面积测量（当前：{self._area_tool_label(active_mode)}）"
        if self._area_tool_button is not None:
            self._area_tool_button.blockSignals(True)
            self._area_tool_button.setText(self._area_tool_label(active_mode))
            self._area_tool_button.setChecked(self._tool_mode in {mode for mode, _ in self._area_tool_definitions()})
            self._area_tool_button.setCurrentTool(active_mode, icon)
            self._area_tool_button.setToolTip(tooltip)
            self._area_tool_button.blockSignals(False)
        for tool_mode, action in self._area_subtool_actions.items():
            action.setChecked(tool_mode == active_mode)
            action.setIcon(self._area_tool_icon(tool_mode))

    def _build_magic_tool_button(self) -> OverlayToolSplitButton:
        button = OverlayToolSplitButton(self)
        button.setText(self._magic_tool_label(self._magic_tool_mode))
        button.primaryTriggered.connect(lambda: self._activate_magic_tool(self._magic_tool_mode))

        menu = QMenu(self)
        menu.setObjectName("magicToolMenu")
        menu.setStyleSheet(self._build_split_menu_stylesheet("magicToolMenu", "rgba(217, 108, 117, 41)"))
        for tool_mode, label in self._magic_tool_definitions():
            action = QAction(label, menu)
            action.setCheckable(True)
            action.setIcon(self._magic_tool_icon(tool_mode))
            action.triggered.connect(lambda checked=False, magic_mode=tool_mode: self._activate_magic_tool(magic_mode))
            menu.addAction(action)
            self._magic_subtool_actions[tool_mode] = action
        button.setMenu(menu)
        self._magic_tool_menu = menu
        self._sync_magic_tool_button()
        return button

    def _sync_magic_tool_button(self) -> None:
        active_mode = self._tool_mode if is_magic_toolbar_tool_mode(self._tool_mode) else self._magic_tool_mode
        label = self._magic_tool_label(active_mode)
        tooltip = f"分割工具（当前：{label}）"
        icon = self._magic_tool_icon(active_mode, active=is_magic_toolbar_tool_mode(self._tool_mode))
        if self._magic_tool_button is not None:
            self._magic_tool_button.blockSignals(True)
            self._magic_tool_button.setText(label)
            self._magic_tool_button.setChecked(is_magic_toolbar_tool_mode(self._tool_mode))
            self._magic_tool_button.setCurrentTool(active_mode, icon)
            self._magic_tool_button.setToolTip(tooltip)
            self._magic_tool_button.blockSignals(False)
        if self._measurement_tool_strip is not None:
            self._measurement_tool_strip.setActiveMode(self._tool_mode)
            self._measurement_tool_strip.setMagicTool(
                active_mode,
                is_magic_toolbar_tool_mode(self._tool_mode),
                icon=icon,
                tooltip=tooltip,
            )
        self._mode_actions[MagicSegmentToolMode.STANDARD].setIcon(self._magic_tool_icon(MagicSegmentToolMode.STANDARD))
        self._mode_actions[MagicSegmentToolMode.REFERENCE].setIcon(self._magic_tool_icon(MagicSegmentToolMode.REFERENCE))
        self._mode_actions[MagicSegmentToolMode.FIBER_QUICK].setIcon(self._magic_tool_icon(MagicSegmentToolMode.FIBER_QUICK))
        for tool_mode, action in self._magic_subtool_actions.items():
            action.setChecked(tool_mode == active_mode)

    def _overlay_tool_icon_name(self, kind: str) -> str:
        for overlay_kind, _label, icon_name in self._overlay_tool_definitions():
            if overlay_kind == kind:
                return icon_name
        return "rename"

    def _overlay_tool_label(self, kind: str) -> str:
        for overlay_kind, label, _icon_name in self._overlay_tool_definitions():
            if overlay_kind == kind:
                return label
        return "文字"

    def _overlay_tool_icon(self, *, active: bool = False) -> QIcon:
        color = "#C9B3E5" if active else "#B79AD8"
        return themed_icon(self._overlay_tool_icon_name(self._overlay_tool_kind), color=color)

    def _activate_overlay_tool(self, kind: str) -> None:
        if kind not in {item[0] for item in self._overlay_tool_definitions()}:
            kind = OverlayAnnotationKind.TEXT
        self._overlay_tool_kind = kind
        self.set_tool_mode("overlay", overlay_kind=kind)

    def _build_overlay_tool_button(self) -> OverlayToolSplitButton:
        button = OverlayToolSplitButton(self)
        button.setText("叠加标注")
        button.primaryTriggered.connect(lambda: self._activate_overlay_tool(self._overlay_tool_kind))

        menu = QMenu(self)
        menu.setObjectName("overlayToolMenu")
        menu.setStyleSheet(self._build_split_menu_stylesheet("overlayToolMenu", "rgba(183, 154, 216, 41)"))
        for kind, label, icon_name in self._overlay_tool_definitions():
            action = QAction(label, menu)
            action.setCheckable(True)
            action.setIcon(themed_icon(icon_name, color="#B79AD8"))
            action.triggered.connect(lambda checked=False, overlay_kind=kind: self._activate_overlay_tool(overlay_kind))
            menu.addAction(action)
            self._overlay_subtool_actions[kind] = action
        button.setMenu(menu)
        self._overlay_tool_menu = menu
        self._sync_overlay_tool_button()
        return button

    def _sync_overlay_tool_button(self) -> None:
        tooltip = f"叠加标注（当前：{self._overlay_tool_label(self._overlay_tool_kind)}）"
        icon = self._overlay_tool_icon(active=self._tool_mode == "overlay")
        if self._overlay_tool_button is not None:
            self._overlay_tool_button.blockSignals(True)
            self._overlay_tool_button.setText(self._overlay_tool_label(self._overlay_tool_kind))
            self._overlay_tool_button.setChecked(self._tool_mode == "overlay")
            self._overlay_tool_button.setCurrentTool(self._overlay_tool_kind, icon)
            self._overlay_tool_button.setToolTip(tooltip)
            self._overlay_tool_button.blockSignals(False)
        if self._measurement_tool_strip is not None:
            self._measurement_tool_strip.setActiveMode(self._tool_mode)
            self._measurement_tool_strip.setOverlayTool(
                self._overlay_tool_kind,
                self._tool_mode == "overlay",
                icon=icon,
                tooltip=tooltip,
            )
        overlay_action = self._mode_actions.get("overlay")
        if overlay_action is not None:
            overlay_action.setIcon(self._overlay_tool_icon())
        for kind, action in self._overlay_subtool_actions.items():
            action.setChecked(kind == self._overlay_tool_kind)

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        self._left_panel = container
        container.setMinimumWidth(380)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        image_box = QGroupBox("已打开图片")
        image_layout = QVBoxLayout(image_box)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_list_changed)
        image_layout.addWidget(self.image_list)

        group_box = QGroupBox("纤维类别")
        group_layout = QVBoxLayout(group_box)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(14, 0, FiberGroupListItemWidget.RIGHT_MARGIN, 0)
        header_row.setSpacing(0)
        color_header = QLabel("颜色")
        color_header.setFixedWidth(36)
        name_header = QLabel("类别")
        count_header = QLabel("（当前/总数）")
        count_header.setFixedWidth(FiberGroupListItemWidget.COUNT_COLUMN_WIDTH)
        count_header.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._group_header_labels = [color_header, name_header, count_header]
        for label in self._group_header_labels:
            label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header_row.addWidget(color_header)
        header_row.addSpacing(14)
        header_row.addWidget(name_header, 1)
        header_row.addWidget(count_header)
        group_layout.addLayout(header_row)
        self.group_list = QListWidget()
        self.group_list.setViewMode(QListView.ViewMode.ListMode)
        self.group_list.setFlow(QListView.Flow.TopToBottom)
        self.group_list.setWrapping(False)
        self.group_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.group_list.setMovement(QListView.Movement.Static)
        self.group_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.group_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.group_list.setSpacing(6)
        self.group_list.setFrameShape(QFrame.Shape.NoFrame)
        self.group_list.setViewportMargins(2, 2, 2, 2)
        self.group_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.group_list.setStyleSheet(
            """
            QListWidget {
                background: transparent;
                border: none;
            }
            QListWidget::item {
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QListWidget::item:selected {
                background: transparent;
                border: none;
                outline: 0;
            }
            """
        )
        self.group_list.itemSelectionChanged.connect(self._on_group_selection_changed)
        group_layout.addWidget(self.group_list, 1)
        group_button_row = FlowLayout(h_spacing=8, v_spacing=8)
        self._add_group_button = QPushButton("新增类别")
        self._add_group_button.setIcon(themed_icon("add", color="#7BD389"))
        self._add_group_button.clicked.connect(self.add_fiber_group)
        self._add_group_button.setMinimumWidth(104)
        self._rename_group_button = QPushButton("编辑")
        self._rename_group_button.setIcon(themed_icon("rename", color="#D7E3FC"))
        self._rename_group_button.clicked.connect(self.rename_active_group)
        self._rename_group_button.setMinimumWidth(92)
        self.delete_group_button = QPushButton("删除")
        self.delete_group_button.setIcon(themed_icon("delete", color="#F28482"))
        self.delete_group_button.clicked.connect(self.delete_active_group)
        self.delete_group_button.setMinimumWidth(80)
        group_button_row.addWidget(self._add_group_button)
        group_button_row.addWidget(self._rename_group_button)
        group_button_row.addWidget(self.delete_group_button)
        group_layout.addLayout(group_button_row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self._left_panel_splitter = splitter
        self._left_standard_splitter = splitter
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(image_box)
        splitter.addWidget(group_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([280, 420])
        layout.addWidget(splitter, 1)
        self._digital_slide_left_panel = self._build_digital_slide_left_panel(container)
        self._digital_slide_left_panel.hide()
        layout.addWidget(self._digital_slide_left_panel, 1)
        self._update_group_list_header_styles()
        return container

    def _build_digital_slide_left_panel(self, parent: QWidget) -> QWidget:
        scroll = QScrollArea(parent)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumWidth(360)

        panel = QWidget(scroll)
        panel.setMinimumWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        summary_box = QGroupBox("数字化切片任务", panel)
        summary_layout = QVBoxLayout(summary_box)
        self._digital_slide_progress_label = QLabel("等待开始采集", summary_box)
        self._digital_slide_progress_label.setWordWrap(True)
        summary_layout.addWidget(self._digital_slide_progress_label)
        self._digital_slide_progress_bar = QProgressBar(summary_box)
        self._digital_slide_progress_bar.setRange(0, 1)
        self._digital_slide_progress_bar.setValue(0)
        self._digital_slide_progress_bar.setTextVisible(True)
        self._digital_slide_progress_bar.setFormat("等待开始")
        summary_layout.addWidget(self._digital_slide_progress_bar)
        eta_grid = QGridLayout()
        eta_grid.setHorizontalSpacing(10)
        eta_grid.setVerticalSpacing(4)
        self._digital_slide_elapsed_label = QLabel("已用时: -", summary_box)
        self._digital_slide_remaining_label = QLabel("预计剩余: -", summary_box)
        self._digital_slide_eta_label = QLabel("预计完成: -", summary_box)
        for label in (self._digital_slide_elapsed_label, self._digital_slide_remaining_label, self._digital_slide_eta_label):
            label.setStyleSheet(f"color: {self._status_color('muted')};")
        eta_grid.addWidget(self._digital_slide_elapsed_label, 0, 0)
        eta_grid.addWidget(self._digital_slide_remaining_label, 0, 1)
        eta_grid.addWidget(self._digital_slide_eta_label, 1, 0, 1, 2)
        summary_layout.addLayout(eta_grid)
        hint = QLabel(
            "采集时会边采集边写入 .fdmslide；完成或停止保留后可直接打开生成的切片文件。",
            summary_box,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {self._status_color('muted')};")
        summary_layout.addWidget(hint)
        layout.addWidget(summary_box)
        layout.addWidget(self._build_digital_slide_capture_box(panel))
        layout.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def _build_digital_slide_capture_box(self, parent: QWidget) -> QWidget:
        self._digital_slide_locked_controls = []
        self._digital_slide_direction_buttons = {}
        container = QWidget(parent)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        z_box = QGroupBox("Z 采集范围", container)
        z_layout = QHBoxLayout(z_box)
        z_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        z_form_panel = QWidget(z_box)
        z_form_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        z_form = QFormLayout(z_form_panel)
        self._digital_slide_z_lower_edit = QLineEdit(z_form_panel)
        self._digital_slide_z_lower_edit.setPlaceholderText("未设置")
        self._digital_slide_z_lower_edit.setMaximumWidth(128)
        self._set_optional_int_edit(self._digital_slide_z_lower_edit, self._app_settings.digital_slide_z_capture_lower)
        self._digital_slide_z_upper_edit = QLineEdit(z_form_panel)
        self._digital_slide_z_upper_edit.setPlaceholderText("未设置")
        self._digital_slide_z_upper_edit.setMaximumWidth(128)
        self._set_optional_int_edit(self._digital_slide_z_upper_edit, self._app_settings.digital_slide_z_capture_upper)
        self._digital_slide_z_step_spin = self._make_digital_slide_spinbox(1, 1_000_000, self._app_settings.digital_slide_z_capture_step, suffix=" steps")
        self._digital_slide_z_step_spin.setMaximumWidth(150)
        self._digital_slide_z_lower_edit.textChanged.connect(self._sync_digital_slide_task_state)
        self._digital_slide_z_upper_edit.textChanged.connect(self._sync_digital_slide_task_state)
        self._digital_slide_z_lower_edit.editingFinished.connect(self._remember_digital_slide_z_capture_settings)
        self._digital_slide_z_upper_edit.editingFinished.connect(self._remember_digital_slide_z_capture_settings)
        self._digital_slide_z_step_spin.valueChanged.connect(lambda _value: self._on_digital_slide_z_capture_step_changed())
        set_upper_button = QPushButton("上限", z_form_panel)
        set_upper_button.setMinimumHeight(32)
        set_upper_button.clicked.connect(lambda checked=False: self._set_digital_slide_z_bound("upper"))
        set_lower_button = QPushButton("下限", z_form_panel)
        set_lower_button.setMinimumHeight(32)
        set_lower_button.clicked.connect(lambda checked=False: self._set_digital_slide_z_bound("lower"))
        self._digital_slide_locked_controls.extend(
            [
                self._digital_slide_z_lower_edit,
                self._digital_slide_z_upper_edit,
                self._digital_slide_z_step_spin,
                set_upper_button,
                set_lower_button,
            ]
        )
        bound_buttons = QHBoxLayout()
        bound_buttons.addWidget(set_upper_button)
        bound_buttons.addWidget(set_lower_button)
        z_form.addRow("Z 上限", self._digital_slide_z_upper_edit)
        z_form.addRow("Z 下限", self._digital_slide_z_lower_edit)
        z_form.addRow("Z 步距", self._digital_slide_z_step_spin)
        z_form.addRow(bound_buttons)
        self._digital_slide_z_rail = DigitalSlideZRangeRail(z_box)
        z_layout.addWidget(z_form_panel, 0, Qt.AlignmentFlag.AlignVCenter)
        z_layout.addWidget(self._digital_slide_z_rail)
        layout.addWidget(z_box)

        range_box = QGroupBox("采集范围", container)
        range_layout = QVBoxLayout(range_box)
        count_row = QHBoxLayout()
        self._digital_slide_cols_edit = QLineEdit(range_box)
        self._digital_slide_cols_edit.setPlaceholderText("列数")
        self._digital_slide_cols_edit.setMaximumWidth(82)
        self._digital_slide_rows_edit = QLineEdit(range_box)
        self._digital_slide_rows_edit.setPlaceholderText("行数")
        self._digital_slide_rows_edit.setMaximumWidth(82)
        self._digital_slide_cols_edit.textEdited.connect(self._on_digital_slide_rows_cols_edited)
        self._digital_slide_rows_edit.textEdited.connect(self._on_digital_slide_rows_cols_edited)
        self._digital_slide_cols_edit.textChanged.connect(self._sync_digital_slide_task_state)
        self._digital_slide_rows_edit.textChanged.connect(self._sync_digital_slide_task_state)
        count_row.addWidget(QLabel("列数", range_box))
        count_row.addWidget(self._digital_slide_cols_edit)
        count_row.addWidget(QLabel("行数", range_box))
        count_row.addWidget(self._digital_slide_rows_edit)
        count_row.addStretch(1)
        range_layout.addLayout(count_row)
        self._digital_slide_locked_controls.extend([self._digital_slide_cols_edit, self._digital_slide_rows_edit])
        self._digital_slide_range_map = DigitalSlideRangeMap(range_box)
        direction_grid = QGridLayout()
        direction_grid.setHorizontalSpacing(6)
        direction_grid.setVerticalSpacing(6)
        for label, key, icon_name, row, col in (
            ("左上", "top_left", "direction_top_left", 0, 0),
            ("上方", "top", "direction_up", 0, 1),
            ("右上", "top_right", "direction_top_right", 0, 2),
            ("左侧", "left", "direction_left", 1, 0),
            ("清除", "clear", "direction_clear", 1, 1),
            ("右侧", "right", "direction_right", 1, 2),
            ("左下", "bottom_left", "direction_bottom_left", 2, 0),
            ("下方", "bottom", "direction_down", 2, 1),
            ("右下", "bottom_right", "direction_bottom_right", 2, 2),
        ):
            button = QPushButton(range_box)
            button.setToolTip(label)
            button.setAccessibleName(label)
            button.setIcon(themed_icon(icon_name, color="#D7E3FC"))
            button.setIconSize(QSize(18, 18))
            button.setFixedSize(38, 34)
            if key == "clear":
                button.clicked.connect(self._clear_digital_slide_region)
            else:
                button.clicked.connect(lambda checked=False, marker=key: self._mark_digital_slide_region(marker))
            self._digital_slide_direction_buttons[key] = button
            self._digital_slide_locked_controls.append(button)
            direction_grid.addWidget(button, row, col)
        map_direction_row = QHBoxLayout()
        map_direction_row.setSpacing(10)
        map_direction_row.addWidget(self._digital_slide_range_map)
        map_direction_row.addStretch(1)
        map_direction_row.addLayout(direction_grid)
        range_layout.addLayout(map_direction_row)
        layout.addWidget(range_box)

        self._sync_digital_slide_task_state()
        return container

    def _build_digital_slide_output_box(self, parent: QWidget) -> QGroupBox:
        output_box = QGroupBox("输出路径", parent)
        output_layout = QVBoxLayout(output_box)
        path_row = QHBoxLayout()
        self._digital_slide_output_path_edit = QLineEdit(self._app_settings.digital_slide_last_output_path, output_box)
        self._digital_slide_output_path_edit.setPlaceholderText("请选择 .fdmslide 输出文件")
        self._digital_slide_output_path_edit.textChanged.connect(self._on_digital_slide_output_path_changed)
        browse_button = QPushButton("设置路径", output_box)
        browse_button.clicked.connect(self._choose_digital_slide_output_path)
        self._digital_slide_locked_controls.extend([self._digital_slide_output_path_edit, browse_button])
        path_row.addWidget(self._digital_slide_output_path_edit, 1)
        path_row.addWidget(browse_button)
        output_layout.addLayout(path_row)

        button_row = QHBoxLayout()
        self._digital_slide_start_button = QPushButton("开始", output_box)
        self._digital_slide_start_button.setObjectName("digitalSlideStartButton")
        self._digital_slide_start_button.setIcon(themed_icon("digital_slide_start", color="#ECFDF5"))
        self._digital_slide_start_button.setMinimumHeight(46)
        self._digital_slide_start_button.clicked.connect(self._start_digital_slide_acquisition)
        self._digital_slide_stop_button = QPushButton("停止", output_box)
        self._digital_slide_stop_button.setObjectName("digitalSlideStopButton")
        self._digital_slide_stop_button.setIcon(themed_icon("digital_slide_stop", color="#FEF2F2"))
        self._digital_slide_stop_button.setMinimumHeight(46)
        self._digital_slide_stop_button.clicked.connect(self._stop_digital_slide_acquisition)
        self._digital_slide_stop_button.setEnabled(False)
        button_row.addWidget(self._digital_slide_start_button, 1)
        button_row.addWidget(self._digital_slide_stop_button, 1)
        output_layout.addLayout(button_row)
        return output_box

    def _build_center_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        self._center_stack = QStackedWidget()
        self.tab_widget = QTabWidget()
        self.tab_widget.setUsesScrollButtons(True)
        self.tab_widget.tabBar().setExpanding(False)
        self.tab_widget.tabBar().setElideMode(Qt.TextElideMode.ElideRight)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self._center_stack.addWidget(self.tab_widget)

        self._preview_page = QWidget()
        preview_layout = QVBoxLayout(self._preview_page)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_status_label = QLabel("请选择采集设备并开始实时预览")
        preview_layout.addWidget(self._preview_status_label)
        self._preview_display_stack = QStackedWidget()
        self._preview_canvas = DocumentCanvas()
        self._preview_canvas.set_read_only(True)
        self._preview_canvas.set_fit_alignment("top_left")
        self._preview_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview_canvas.installEventFilter(self)
        self._preview_display_stack.addWidget(self._preview_canvas)
        self._microview_preview_host = MicroviewPreviewHost()
        self._microview_preview_host.installEventFilter(self)
        self._microview_preview_host.metricsChanged.connect(self._on_preview_host_metrics_changed)
        self._microview_preview_scroll = QScrollArea()
        self._microview_preview_scroll.setWidget(self._microview_preview_host)
        self._microview_preview_scroll.setWidgetResizable(False)
        self._microview_preview_scroll.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._microview_preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._microview_preview_scroll.viewport().installEventFilter(self)
        self._preview_display_stack.addWidget(self._microview_preview_scroll)
        preview_layout.addWidget(self._preview_display_stack, 1)
        self._image_resolution_label = QLabel("像素尺寸: -")
        self._image_resolution_label.setWordWrap(True)
        self._image_resolution_label.setStyleSheet("padding: 6px 2px 0 2px;")
        self._center_stack.addWidget(self._preview_page)
        layout.addWidget(self._center_stack)
        layout.addWidget(self._image_resolution_label)
        return container

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        self._right_panel = container
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)

        model_box = QGroupBox("面积识别")
        model_layout = QVBoxLayout(model_box)
        self._area_auto_button = QPushButton("面积自动识别...")
        self._area_auto_button.setIcon(themed_icon("area_auto", color="#7BD389"))
        self._area_auto_button.clicked.connect(self.run_area_auto_recognition)
        model_layout.addWidget(self._area_auto_button)
        top_layout.addWidget(model_box)

        calibration_box = QGroupBox("标定")
        calibration_layout = QVBoxLayout(calibration_box)
        self._calibration_status_card = QFrame(calibration_box)
        self._calibration_status_card.setObjectName("calibrationStatusCard")
        card_layout = QVBoxLayout(self._calibration_status_card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(6)

        self._calibration_status_title_label = QLabel("未标定", self._calibration_status_card)
        self._calibration_status_title_label.setWordWrap(True)
        self._calibration_status_title_label.setStyleSheet("font-weight: 800;")
        card_layout.addWidget(self._calibration_status_title_label)

        self._calibration_status_summary_label = QLabel("测量仅显示 px，无法输出真实尺寸", self._calibration_status_card)
        self._calibration_status_summary_label.setWordWrap(True)
        self._calibration_status_summary_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        card_layout.addWidget(self._calibration_status_summary_label)
        self.calibration_label = self._calibration_status_summary_label

        self._calibration_card_action_row = QWidget(self._calibration_status_card)
        card_action_layout = QHBoxLayout(self._calibration_card_action_row)
        card_action_layout.setContentsMargins(0, 0, 0, 0)
        card_action_layout.setSpacing(8)

        self._calibration_start_button = QPushButton("开始标定", self._calibration_card_action_row)
        self._calibration_start_button.setIcon(themed_icon("calibration", color="#FF7F50"))
        self._calibration_start_button.clicked.connect(lambda checked=False: self.set_tool_mode("calibration"))
        card_action_layout.addWidget(self._calibration_start_button, 1)

        self._calibration_details_button = QPushButton("查看详情", self._calibration_card_action_row)
        self._calibration_details_button.setCheckable(True)
        self._calibration_details_button.toggled.connect(self._toggle_calibration_details)
        card_action_layout.addWidget(self._calibration_details_button, 1)
        card_layout.addWidget(self._calibration_card_action_row)

        self._calibration_details_label = QLabel("", self._calibration_status_card)
        self._calibration_details_label.setWordWrap(True)
        self._calibration_details_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._calibration_label_scroll = QScrollArea()
        self._calibration_label_scroll.setWidget(self._calibration_details_label)
        self._calibration_label_scroll.setWidgetResizable(True)
        self._calibration_label_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._calibration_label_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._calibration_label_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._calibration_label_scroll.setMinimumHeight(72)
        self._calibration_label_scroll.setMaximumHeight(118)
        self._calibration_label_scroll.hide()
        card_layout.addWidget(self._calibration_label_scroll)

        calibration_layout.addWidget(self._calibration_status_card)
        self.preset_combo = QComboBox()
        self.preset_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.preset_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_combo.setMinimumContentsLength(10)
        self.preset_combo.currentTextChanged.connect(self._update_preset_combo_tooltip)
        calibration_layout.addWidget(self.preset_combo)
        preset_row = QHBoxLayout()
        self._add_preset_button = QPushButton("新增预设")
        self._add_preset_button.setIcon(themed_icon("preset_add", color="#7BD389"))
        self._add_preset_button.clicked.connect(self.add_calibration_preset)
        self._edit_preset_button = QPushButton("编辑预设")
        self._edit_preset_button.setIcon(themed_icon("rename", color="#D7E3FC"))
        self._edit_preset_button.clicked.connect(self.edit_selected_preset)
        self._delete_preset_button = QPushButton("删除预设")
        self._delete_preset_button.setIcon(themed_icon("delete", color="#F28482"))
        self._delete_preset_button.clicked.connect(self.delete_selected_preset)
        preset_row.addWidget(self._add_preset_button)
        preset_row.addWidget(self._edit_preset_button)
        preset_row.addWidget(self._delete_preset_button)
        calibration_layout.addLayout(preset_row)
        self._calibration_preset_action_row = QWidget(calibration_box)
        preset_action_layout = QHBoxLayout(self._calibration_preset_action_row)
        preset_action_layout.setContentsMargins(0, 0, 0, 0)
        preset_action_layout.setSpacing(8)

        self._apply_preset_button = QPushButton("应用预设", self._calibration_preset_action_row)
        self._apply_preset_button.setIcon(themed_icon("preset_apply", color="#D7E3FC"))
        self._apply_preset_button.clicked.connect(self.apply_selected_preset)
        preset_action_layout.addWidget(self._apply_preset_button, 1)

        self._import_cu_preset_button = QPushButton("导入CU标尺", self._calibration_preset_action_row)
        self._import_cu_preset_button.setIcon(themed_icon("preset_import", color="#D7E3FC"))
        self._import_cu_preset_button.clicked.connect(self.import_cu_calibration_presets)
        preset_action_layout.addWidget(self._import_cu_preset_button, 1)
        calibration_layout.addWidget(self._calibration_preset_action_row)
        top_layout.addWidget(calibration_box)

        measurement_box = QGroupBox("测量记录")
        measurement_layout = QVBoxLayout(measurement_box)
        self.measurement_table = QTableWidget(0, 8)
        self.measurement_table.setHorizontalHeaderLabels(["种类", "类型", "结果", "单位", "模式", "置信度", "状态", "ID"])
        header = self.measurement_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.measurement_table.setColumnWidth(self.TABLE_COL_GROUP, 150)
        self.measurement_table.setColumnWidth(self.TABLE_COL_KIND, 80)
        self.measurement_table.setColumnWidth(self.TABLE_COL_RESULT, 120)
        self.measurement_table.setColumnWidth(self.TABLE_COL_UNIT, 70)
        self.measurement_table.setColumnWidth(self.TABLE_COL_MODE, 120)
        self.measurement_table.setColumnWidth(self.TABLE_COL_CONFIDENCE, 80)
        self.measurement_table.setColumnWidth(self.TABLE_COL_STATUS, 110)
        self.measurement_table.setColumnWidth(self.TABLE_COL_ID, 110)
        self.measurement_table.verticalHeader().setVisible(False)
        self.measurement_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.measurement_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.measurement_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.measurement_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.measurement_table.itemSelectionChanged.connect(self._on_measurement_selection_changed)
        measurement_layout.addWidget(self.measurement_table)

        measurement_action_row = QWidget(measurement_box)
        measurement_action_layout = QHBoxLayout(measurement_action_row)
        measurement_action_layout.setContentsMargins(0, 0, 0, 0)
        measurement_action_layout.setSpacing(8)

        self.delete_measurement_button = QPushButton("删除选中")
        self.delete_measurement_button.setIcon(themed_icon("delete", color="#F28482"))
        self.delete_measurement_button.clicked.connect(self.delete_selected_measurement)
        self.delete_measurement_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        measurement_action_layout.addWidget(self.delete_measurement_button)

        self._delete_group_measurements_button = QPushButton("删除类别")
        self._delete_group_measurements_button.setIcon(themed_icon("delete", color="#F28482"))
        self._delete_group_measurements_button.clicked.connect(self.delete_measurements_by_category)
        self._delete_group_measurements_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        measurement_action_layout.addWidget(self._delete_group_measurements_button)

        self._delete_all_measurements_button = QPushButton("删除全部")
        self._delete_all_measurements_button.setIcon(themed_icon("delete", color="#F28482"))
        self._delete_all_measurements_button.clicked.connect(self.delete_all_measurements)
        self._delete_all_measurements_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        measurement_action_layout.addWidget(self._delete_all_measurements_button)

        measurement_layout.addWidget(measurement_action_row)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(top_container)
        right_splitter.addWidget(measurement_box)
        right_splitter.setStretchFactor(0, 0)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([310, 470])
        self._right_standard_panel = right_splitter
        layout.addWidget(right_splitter)
        self._digital_slide_right_panel = self._build_digital_slide_right_panel(container)
        self._digital_slide_right_panel.hide()
        layout.addWidget(self._digital_slide_right_panel)

        return container

    def _build_digital_slide_right_panel(self, parent: QWidget) -> QWidget:
        scroll = QScrollArea(parent)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumWidth(340)

        panel = QWidget(scroll)
        panel.setMinimumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        connection_section, self._digital_slide_connection_details, self._digital_slide_connection_toggle, self._digital_slide_connection_summary_label = (
            self._make_digital_slide_collapsible_section(panel, "连接情况", "尚未检查", expanded=False)
        )
        connection_content = self._digital_slide_connection_details
        connection_layout = QVBoxLayout(connection_content)
        connection_layout.setContentsMargins(0, 0, 0, 0)
        self._digital_slide_motor_enable = QCheckBox("启用电机输出", connection_content)
        self._digital_slide_motor_enable.toggled.connect(self._set_digital_slide_motor_enabled)
        connection_layout.addWidget(self._digital_slide_motor_enable)
        self._digital_slide_port_combo = QComboBox(connection_content)
        self._digital_slide_port_combo.setEditable(True)
        self._digital_slide_port_combo.currentIndexChanged.connect(self._on_digital_slide_port_changed)
        if self._digital_slide_port_combo.lineEdit() is not None:
            self._digital_slide_port_combo.lineEdit().editingFinished.connect(self._on_digital_slide_port_changed)
        connection_layout.addWidget(self._digital_slide_port_combo)
        connection_buttons = QHBoxLayout()
        refresh_button = QPushButton("刷新", connection_content)
        refresh_button.clicked.connect(lambda checked=False: self._refresh_digital_slide_ports(prefer_auto=False))
        auto_button = QPushButton("Auto FTDI", connection_content)
        auto_button.clicked.connect(lambda checked=False: self._refresh_digital_slide_ports(prefer_auto=True))
        check_button = QPushButton("检查", connection_content)
        check_button.clicked.connect(self._check_digital_slide_motion_status)
        self._digital_slide_locked_controls.extend(
            [self._digital_slide_motor_enable, self._digital_slide_port_combo, refresh_button, auto_button, check_button]
        )
        connection_buttons.addWidget(refresh_button)
        connection_buttons.addWidget(auto_button)
        connection_buttons.addWidget(check_button)
        connection_layout.addLayout(connection_buttons)
        card_grid = QGridLayout()
        motor_card, self._digital_slide_motor_card_label = self._make_digital_slide_status_card(connection_content, "控制", "尚未检查")
        port_card, self._digital_slide_port_card_label = self._make_digital_slide_status_card(connection_content, "端口", "未选择")
        camera_card, self._digital_slide_camera_card_label = self._make_digital_slide_status_card(connection_content, "相机", "未检查")
        self._digital_slide_status_label = self._digital_slide_motor_card_label
        self._digital_slide_camera_label = self._digital_slide_camera_card_label
        card_grid.addWidget(motor_card, 0, 0)
        card_grid.addWidget(port_card, 0, 1)
        card_grid.addWidget(camera_card, 1, 0, 1, 2)
        connection_layout.addLayout(card_grid)

        diagnostics_box = QFrame(connection_content)
        diagnostics_box.setObjectName("digitalSlideDiagnosticsBox")
        diagnostics_box.setStyleSheet(
            "QFrame#digitalSlideDiagnosticsBox {"
            "border: 1px solid rgba(148, 163, 184, 0.28);"
            "border-radius: 8px;"
            "}"
        )
        diagnostics_layout = QVBoxLayout(diagnostics_box)
        diagnostics_layout.setContentsMargins(10, 8, 10, 8)
        diagnostics_layout.setSpacing(6)
        diagnostics_header = QHBoxLayout()
        self._digital_slide_diagnostics_toggle = QToolButton(diagnostics_box)
        self._digital_slide_diagnostics_toggle.setText("采集诊断")
        self._digital_slide_diagnostics_toggle.setCheckable(True)
        self._digital_slide_diagnostics_toggle.setChecked(False)
        self._digital_slide_diagnostics_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._digital_slide_diagnostics_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._digital_slide_diagnostics_toggle.toggled.connect(self._toggle_digital_slide_diagnostics)
        self._digital_slide_diagnostics_summary_label = QLabel("上一张: -", diagnostics_box)
        self._digital_slide_diagnostics_summary_label.setStyleSheet(f"color: {self._status_color('muted')};")
        diagnostics_header.addWidget(self._digital_slide_diagnostics_toggle)
        diagnostics_header.addStretch(1)
        diagnostics_header.addWidget(self._digital_slide_diagnostics_summary_label)
        diagnostics_layout.addLayout(diagnostics_header)
        self._digital_slide_diagnostics_details = QWidget(diagnostics_box)
        details_layout = QVBoxLayout(self._digital_slide_diagnostics_details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        self._digital_slide_timing_label = QLabel("耗时: -", self._digital_slide_diagnostics_details)
        self._digital_slide_timing_label.setWordWrap(True)
        self._digital_slide_timing_label.setStyleSheet(f"color: {self._status_color('muted')};")
        details_layout.addWidget(self._digital_slide_timing_label)
        self._digital_slide_diagnostics_details.hide()
        diagnostics_layout.addWidget(self._digital_slide_diagnostics_details)
        connection_layout.addWidget(diagnostics_box)
        layout.addWidget(connection_section)

        stage_section, self._digital_slide_stage_details, self._digital_slide_stage_toggle, self._digital_slide_stage_summary_label = (
            self._make_digital_slide_collapsible_section(panel, "样品台 / 对焦", "X=0  Y=0  Z=0", expanded=False)
        )
        stage_content = self._digital_slide_stage_details
        stage_layout = QVBoxLayout(stage_content)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        step_form = QFormLayout()
        self._digital_slide_xy_jog_step_spin = self._make_digital_slide_spinbox(1, 1_000_000, self._app_settings.digital_slide_xy_jog_step, suffix=" steps")
        self._digital_slide_focus_jog_step_spin = self._make_digital_slide_spinbox(1, 1_000_000, self._app_settings.digital_slide_z_jog_step, suffix=" steps")
        self._digital_slide_xy_jog_step_spin.valueChanged.connect(lambda _value: self._on_digital_slide_manual_step_changed())
        self._digital_slide_focus_jog_step_spin.valueChanged.connect(lambda _value: self._on_digital_slide_manual_step_changed())
        self._digital_slide_locked_controls.extend([self._digital_slide_xy_jog_step_spin, self._digital_slide_focus_jog_step_spin])
        step_form.addRow("XY 步距", self._digital_slide_xy_jog_step_spin)
        step_form.addRow("对焦步距", self._digital_slide_focus_jog_step_spin)
        stage_layout.addLayout(step_form)
        self._digital_slide_motion_settings_label = QLabel(stage_content)
        self._digital_slide_motion_settings_label.setWordWrap(True)
        self._digital_slide_motion_settings_label.setStyleSheet(f"color: {self._status_color('muted')};")
        stage_layout.addWidget(self._digital_slide_motion_settings_label)
        stage_grid = QGridLayout()
        up = self._make_jog_button("↑", AXIS_Y, DIR_POS)
        down = self._make_jog_button("↓", AXIS_Y, DIR_NEG)
        left = self._make_jog_button("←", AXIS_X, DIR_NEG)
        right = self._make_jog_button("→", AXIS_X, DIR_POS)
        focus_up = self._make_jog_button("焦点 +", AXIS_Z, DIR_POS)
        focus_down = self._make_jog_button("焦点 -", AXIS_Z, DIR_NEG)
        self._digital_slide_motion_controls.extend([up, down, left, right, focus_up, focus_down])
        stage_grid.addWidget(up, 0, 1)
        stage_grid.addWidget(left, 1, 0)
        stage_grid.addWidget(right, 1, 2)
        stage_grid.addWidget(down, 2, 1)
        stage_grid.addWidget(focus_up, 0, 3)
        stage_grid.addWidget(focus_down, 2, 3)
        stage_layout.addLayout(stage_grid)
        self._digital_slide_position_label = QLabel("", stage_content)
        self._digital_slide_position_label.setWordWrap(True)
        stage_layout.addWidget(self._digital_slide_position_label)
        zero_buttons = QHBoxLayout()
        xy_zero_button = QPushButton("设置 XY 样品台原点", stage_content)
        xy_zero_button.clicked.connect(lambda checked=False: self._reset_digital_slide_motion_zero(axes=(AXIS_X, AXIS_Y)))
        z_zero_button = QPushButton("设置 Z 轴高度原点", stage_content)
        z_zero_button.clicked.connect(lambda checked=False: self._reset_digital_slide_motion_zero(axes=(AXIS_Z,)))
        self._digital_slide_locked_controls.extend([xy_zero_button, z_zero_button])
        zero_buttons.addWidget(xy_zero_button)
        zero_buttons.addWidget(z_zero_button)
        stage_layout.addLayout(zero_buttons)
        layout.addWidget(stage_section)

        layout.addStretch(1)
        layout.addWidget(self._build_digital_slide_output_box(panel))
        scroll.setWidget(panel)
        self._refresh_digital_slide_ports(prefer_auto=True)
        self._apply_digital_slide_motion_settings()
        self._sync_digital_slide_position_label()
        self._sync_digital_slide_camera_label()
        self._sync_digital_slide_task_state()
        return scroll

    def _make_digital_slide_collapsible_section(
        self,
        parent: QWidget,
        title: str,
        summary: str,
        *,
        expanded: bool,
    ) -> tuple[QFrame, QWidget, QToolButton, QLabel]:
        frame = QFrame(parent)
        frame.setObjectName("digitalSlideCollapsibleSection")
        frame.setStyleSheet(
            "QFrame#digitalSlideCollapsibleSection {"
            "border: 1px solid rgba(148, 163, 184, 0.32);"
            "border-radius: 8px;"
            "}"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        header = QHBoxLayout()
        toggle = QToolButton(frame)
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setChecked(bool(expanded))
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        summary_label = QLabel(summary, frame)
        summary_label.setStyleSheet(f"color: {self._status_color('muted')};")
        summary_label.setWordWrap(True)
        header.addWidget(toggle)
        header.addStretch(1)
        header.addWidget(summary_label)
        layout.addLayout(header)
        content = QWidget(frame)
        content.setVisible(bool(expanded))
        layout.addWidget(content)

        def on_toggled(checked: bool) -> None:
            toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
            content.setVisible(checked)

        toggle.toggled.connect(on_toggled)
        return frame, content, toggle, summary_label

    def _make_digital_slide_status_card(self, parent: QWidget, title: str, value: str) -> tuple[QFrame, QLabel]:
        card = QFrame(parent)
        card.setObjectName("digitalSlideStatusCard")
        card.setStyleSheet(
            "QFrame#digitalSlideStatusCard {"
            "background: rgba(148, 163, 184, 0.12);"
            "border: 1px solid rgba(148, 163, 184, 0.35);"
            "border-radius: 8px;"
            "}"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        title_label = QLabel(title, card)
        title_label.setStyleSheet(f"color: {self._status_color('muted')}; font-weight: 700;")
        value_label = QLabel(value, card)
        value_label.setWordWrap(True)
        value_label.setStyleSheet("font-weight: 800;")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return card, value_label

    def _toggle_digital_slide_diagnostics(self, checked: bool) -> None:
        if self._digital_slide_diagnostics_toggle is not None:
            self._digital_slide_diagnostics_toggle.setArrowType(
                Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
            )
        if self._digital_slide_diagnostics_details is not None:
            self._digital_slide_diagnostics_details.setVisible(checked)

    def _make_digital_slide_spinbox(self, minimum: int, maximum: int, value: int, *, suffix: str = "") -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(int(minimum), int(maximum))
        spinbox.setValue(int(value))
        spinbox.setSingleStep(max(1, min(1000, abs(int(value)) or 1)))
        if suffix:
            spinbox.setSuffix(suffix)
        return spinbox

    def _make_jog_button(self, label: str, axis: str, direction: str) -> QPushButton:
        button = QPushButton(label)
        button.setMinimumHeight(38)
        button.pressed.connect(lambda axis_name=axis, dir_name=direction: self._begin_digital_slide_jog(axis_name, dir_name))
        button.released.connect(self._end_digital_slide_jog)
        return button

    def _make_export_action(self, label: str, selection: ExportSelection) -> QAction:
        action = QAction(label, self)
        action.triggered.connect(lambda checked=False, preset=selection: self.export_results(preset))
        return action

    def set_tool_mode(self, mode: str, *, overlay_kind: str | None = None) -> None:
        previous_mode = self._tool_mode
        if mode not in self._mode_actions:
            mode = "select"
        if overlay_kind in {item[0] for item in self._overlay_tool_definitions()}:
            self._overlay_tool_kind = overlay_kind
        current_canvas = self.current_canvas()
        current_document_id = current_canvas.document_id if current_canvas is not None else None
        if current_document_id is not None and mode != previous_mode:
            if self._prompt_seg_worker is not None and (
                is_magic_segment_tool_mode(previous_mode) or is_fiber_quick_tool_mode(previous_mode)
            ):
                self._prompt_seg_worker.cancel_document(current_document_id)
            if self._fiber_quick_geometry_worker is not None and is_fiber_quick_tool_mode(previous_mode):
                self._fiber_quick_geometry_worker.cancel_document(current_document_id)
        if is_magic_toolbar_tool_mode(mode):
            self._magic_tool_mode = mode
        if mode in {item[0] for item in self._manual_tool_definitions()}:
            self._manual_tool_mode = mode
        if mode in {item[0] for item in self._area_tool_definitions()}:
            self._area_tool_mode = mode
        if mode != "select":
            self._last_non_select_tool = mode
        self._tool_mode = mode
        for canvas in self._canvases.values():
            canvas.set_tool_mode(mode, overlay_kind=self._overlay_tool_kind)
            if is_magic_segment_tool_mode(mode):
                self._sync_canvas_magic_subtract_input_mode(canvas)
        if mode in self._mode_actions:
            self._mode_actions[mode].setChecked(True)
            self.statusBar().showMessage(f"当前工具: {self._mode_actions[mode].text()}", 3000)
        if self._measurement_tool_strip is not None:
            self._measurement_tool_strip.setActiveMode(mode)
        self._sync_manual_tool_button()
        self._sync_area_tool_button()
        self._sync_magic_tool_button()
        self._sync_overlay_tool_button()
        self._update_magic_segment_controls()
        self._update_count_numbers_button()
        self._update_path_drawing_controls()

    def current_document(self) -> ImageDocument | None:
        if self._preview_active:
            return None
        index = self.tab_widget.currentIndex()
        if index < 0 or index >= len(self._document_order):
            return None
        return self.project.get_document(self._document_order[index])

    def current_canvas(self) -> DocumentCanvas | None:
        if self._preview_active:
            return self._preview_canvas if self._capture_manager.preview_kind() == "frame_stream" else None
        document = self.current_document()
        if document is None:
            return None
        return self._canvases.get(document.id)

    def _preview_kind(self) -> str:
        return self._capture_manager.preview_kind()

    def _is_native_preview(self) -> bool:
        return self._preview_kind() == "native_embed"

    def _current_preview_target(self) -> object | None:
        if self._is_native_preview():
            return self._microview_preview_host
        return None

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel and self._should_intercept_digital_slide_preview_wheel(watched):
            return self._handle_digital_slide_focus_wheel(event)
        return super().eventFilter(watched, event)

    def _should_intercept_digital_slide_preview_wheel(self, watched: object) -> bool:
        if not (self._digital_slide_mode and self._preview_active) or self._slide_acquisition_active():
            return False
        targets: set[object] = set()
        if self._preview_canvas is not None:
            targets.add(self._preview_canvas)
        if self._microview_preview_host is not None:
            targets.add(self._microview_preview_host)
        if self._microview_preview_scroll is not None:
            targets.add(self._microview_preview_scroll)
            targets.add(self._microview_preview_scroll.viewport())
        return watched in targets

    def _handle_digital_slide_focus_wheel(self, event) -> bool:
        delta = event.angleDelta()
        delta_y = delta.y()
        delta_x = delta.x()
        effective_delta = delta_y if delta_y != 0 else delta_x
        if effective_delta == 0:
            return False
        now = perf_counter()
        if now - self._last_digital_slide_focus_wheel_at >= 0.08:
            self._last_digital_slide_focus_wheel_at = now
            self._perform_digital_slide_jog_step(AXIS_Z, DIR_POS if effective_delta > 0 else DIR_NEG)
        accept = getattr(event, "accept", None)
        if callable(accept):
            accept()
        return True

    def _apply_preview_surface(self, preview_kind: str) -> None:
        if (
            self._preview_display_stack is None
            or self._preview_canvas is None
            or self._microview_preview_scroll is None
        ):
            return
        target_widget = self._microview_preview_scroll if preview_kind == "native_embed" else self._preview_canvas
        self._preview_display_stack.setCurrentWidget(target_widget)

    def _refresh_preview_surface(self) -> None:
        self._apply_preview_surface(self._preview_kind())

    def _on_preview_host_metrics_changed(self) -> None:
        if not self._preview_active or not self._is_native_preview() or self._microview_preview_host is None:
            return
        self._capture_manager.update_preview_target(self._microview_preview_host)

    def _show_active_capture_warning(self) -> None:
        warning = self._capture_manager.active_warning().strip()
        if warning:
            self.statusBar().showMessage(warning, 7000)

    def _format_dimension_value(self, value: float) -> str:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"

    def _is_dark_palette(self) -> bool:
        app = QApplication.instance()
        palette = app.palette() if app is not None else self.palette()
        return palette.color(QPalette.ColorRole.Window).lightnessF() < 0.5

    def _status_color(self, kind: str) -> str:
        if kind == "danger":
            return "#FF7B72" if self._is_dark_palette() else "#C62828"
        if kind == "info":
            return "#79C0FF" if self._is_dark_palette() else "#1565C0"
        if kind == "muted":
            return "#C8D3DD" if self._is_dark_palette() else "#4E5969"
        return self.palette().color(QPalette.ColorRole.WindowText).name()

    def _tool_icon_color(self, kind: str) -> str:
        if kind == "select":
            return "#D4D8DD" if self._is_dark_palette() else "#4E5969"
        if kind == "count":
            return "#F08B95" if self._is_dark_palette() else "#C65B75"
        if kind == "snap":
            return "#7BD389" if self._is_dark_palette() else "#2F8F6B"
        if kind == "calibration":
            return "#FF7F50" if self._is_dark_palette() else "#C7662B"
        return "#D7E3FC" if self._is_dark_palette() else "#51606F"

    def _refresh_theme_sensitive_icons(self) -> None:
        if not {
            "select",
            "count",
            "snap",
            "manual",
            "continuous_manual",
            "polygon_area",
            "freehand_area",
            "calibration",
            "overlay",
            MagicSegmentToolMode.STANDARD,
            MagicSegmentToolMode.REFERENCE,
            MagicSegmentToolMode.FIBER_QUICK,
        }.issubset(self._mode_actions):
            return
        self._mode_actions["select"].setIcon(themed_icon("select", color=self._tool_icon_color("select")))
        self._mode_actions["count"].setIcon(themed_icon("count", color=self._tool_icon_color("count")))
        self._mode_actions["snap"].setIcon(themed_icon("snap", color=self._tool_icon_color("snap")))
        self._mode_actions["manual"].setIcon(self._manual_tool_icon("manual"))
        self._mode_actions["continuous_manual"].setIcon(self._manual_tool_icon("continuous_manual"))
        self._mode_actions["polygon_area"].setIcon(self._area_tool_icon("polygon_area"))
        self._mode_actions["freehand_area"].setIcon(self._area_tool_icon("freehand_area"))
        self._mode_actions[MagicSegmentToolMode.STANDARD].setIcon(self._magic_tool_icon(MagicSegmentToolMode.STANDARD))
        self._mode_actions[MagicSegmentToolMode.REFERENCE].setIcon(self._magic_tool_icon(MagicSegmentToolMode.REFERENCE))
        self._mode_actions[MagicSegmentToolMode.FIBER_QUICK].setIcon(self._magic_tool_icon(MagicSegmentToolMode.FIBER_QUICK))
        self._mode_actions["calibration"].setIcon(themed_icon("calibration", color=self._tool_icon_color("calibration")))
        self._mode_actions["overlay"].setIcon(self._overlay_tool_icon())
        if self._manual_tool_button is not None:
            self._sync_manual_tool_button()
        if self._area_tool_button is not None:
            self._sync_area_tool_button()
        if self._magic_tool_button is not None:
            self._sync_magic_tool_button()
        if self._overlay_tool_button is not None:
            self._sync_overlay_tool_button()

    def _update_statusbar_aux_labels(self) -> None:
        if self._version_label is None:
            return
        self._version_label.setStyleSheet(f"color: {self._status_color('muted')}; padding: 0 4px;")

    def _update_group_list_header_styles(self) -> None:
        if not self._group_header_labels:
            return
        muted = self._status_color("muted")
        for label in self._group_header_labels:
            label.setStyleSheet(f"color: {muted}; padding: 0 0 2px 0;")

    def _toggle_calibration_details(self, checked: bool) -> None:
        if self._calibration_label_scroll is not None:
            self._calibration_label_scroll.setVisible(bool(checked))
        if self._calibration_details_button is not None:
            self._calibration_details_button.setText("收起详情" if checked else "查看详情")

    def _clear_calibration_action_prominence(self) -> None:
        for button in (self._calibration_start_button, self._apply_preset_button):
            if button is not None:
                button.setStyleSheet("")

    def _set_calibration_status_card(
        self,
        *,
        title: str,
        summary: str,
        status: str,
        details: str = "",
        show_start_button: bool = False,
    ) -> None:
        if self._calibration_status_title_label is None or self._calibration_status_summary_label is None:
            return
        palette = {
            "uncalibrated": ("#FEF2F2", "#FCA5A5", "#7F1D1D", "#991B1B"),
            "calibrated": ("#ECFDF5", "#6EE7B7", "#064E3B", "#047857"),
            "preview": ("#F1F5F9", "#CBD5E1", "#334155", "#475569"),
            "warning": ("#FFFBEB", "#FCD34D", "#78350F", "#92400E"),
        }.get(status, ("#F8FAFC", "#CBD5E1", "#1F2937", "#475569"))
        background, border, title_color, text_color = palette
        if self._is_dark_palette():
            palette = {
                "uncalibrated": ("#3F1518", "#F87171", "#FEE2E2", "#FECACA"),
                "calibrated": ("#063C33", "#34D399", "#ECFDF5", "#A7F3D0"),
                "preview": ("#1F2937", "#64748B", "#E5E7EB", "#CBD5E1"),
                "warning": ("#3B2A08", "#FBBF24", "#FEF3C7", "#FDE68A"),
            }.get(status, ("#1F2937", "#64748B", "#E5E7EB", "#CBD5E1"))
            background, border, title_color, text_color = palette
        if self._calibration_status_card is not None:
            self._calibration_status_card.setStyleSheet(
                "QFrame#calibrationStatusCard {"
                f"background: {background};"
                f"border: 1px solid {border};"
                "border-radius: 8px;"
                "}"
            )
        self._calibration_status_title_label.setText(title)
        self._calibration_status_title_label.setStyleSheet(f"font-weight: 800; color: {title_color};")
        self._calibration_status_summary_label.setText(summary)
        self._calibration_status_summary_label.setToolTip("\n".join(part for part in [title, summary, details] if part))
        self._calibration_status_summary_label.setStyleSheet(f"color: {text_color};")
        self.calibration_label = self._calibration_status_summary_label
        if self._calibration_details_label is not None:
            self._calibration_details_label.setText(details)
            self._calibration_details_label.setToolTip(details)
            self._calibration_details_label.setStyleSheet(f"color: {text_color};")
        if self._calibration_details_button is not None:
            has_details = bool(details.strip())
            self._calibration_details_button.setVisible(has_details)
            self._calibration_details_button.setEnabled(has_details)
            if not has_details:
                self._calibration_details_button.setChecked(False)
            self._calibration_details_button.setText("收起详情" if self._calibration_details_button.isChecked() else "查看详情")
        if self._calibration_start_button is not None:
            self._calibration_start_button.setVisible(show_start_button)
        self._clear_calibration_action_prominence()

    def _set_calibration_label(self, text: str, *, status: str) -> None:
        self._set_calibration_status_card(
            title=text.splitlines()[0] if text else "",
            summary=text,
            status=status,
            details=text,
            show_start_button=status == "uncalibrated",
        )

    def _update_preset_combo_tooltip(self, text: str) -> None:
        self.preset_combo.setToolTip(text)

    def _update_image_resolution_label(self, document: ImageDocument | None = None) -> None:
        if self._image_resolution_label is None:
            return
        self._image_resolution_label.setStyleSheet(f"color: {self._status_color('muted')}; padding: 6px 2px 0 2px;")
        if self._preview_active:
            resolution = self._capture_manager.preview_resolution()
            if resolution is None:
                self._image_resolution_label.setText("实时预览分辨率: -")
            else:
                self._image_resolution_label.setText(f"实时预览分辨率: {resolution[0]} x {resolution[1]} px")
            return
        target_document = document or self.current_document()
        if target_document is None:
            self._image_resolution_label.setText("像素尺寸: -")
            return
        if target_document.is_digital_slide():
            width_px, height_px = target_document.image_size
            canvas = self._canvases.get(target_document.id)
            view_text = ""
            if isinstance(canvas, DigitalSlideCanvas):
                origin = canvas.viewport_origin()
                view_text = (
                    f"    |    视场: X={int(origin.x)}, Y={int(origin.y)}, 焦层={canvas.focus_index() + 1}"
                    f"    |    移动: {canvas.navigation_mode_label()}"
                )
            self._image_resolution_label.setText(f"数字化切片: {width_px} x {height_px} px{view_text}")
            return
        width_px, height_px = target_document.image_size
        parts = [f"像素尺寸: {width_px} x {height_px} px"]
        calibration = target_document.calibration
        if calibration is not None and calibration.pixels_per_unit > 0:
            width_unit = self._format_dimension_value(calibration.px_to_unit(width_px))
            height_unit = self._format_dimension_value(calibration.px_to_unit(height_px))
            parts.append(f"实际尺寸: {width_unit} x {height_unit} {calibration.unit}")
        self._image_resolution_label.setText("    |    ".join(parts))

    def _apply_native_preview_resolution(self) -> None:
        if self._microview_preview_host is None:
            return
        resolution = self._capture_manager.preview_resolution()
        if resolution is None:
            return
        width, height = resolution
        self._microview_preview_host.set_preview_resolution(width, height)
        if self._preview_status_label is not None:
            selected = self._selected_capture_device()
            label = selected.name if selected is not None else "采集设备"
            self._preview_status_label.setText(f"正在预览: {label}  ({width} x {height}, 原始分辨率)")
        self._update_image_resolution_label()

    def _maybe_hint_signal_optimization(self) -> None:
        selected = self._selected_capture_device()
        if selected is None or selected.id in self._microview_optimize_hints_shown:
            return
        if not self._capture_manager.can_optimize_signal():
            return
        self._microview_optimize_hints_shown.add(selected.id)
        self.statusBar().showMessage("如果预览出现横条撕裂，可尝试点击“优化采集参数”。", 7000)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in {QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange}:
            self._update_statusbar_aux_labels()
            self._update_image_resolution_label()
            self._update_group_list_header_styles()
            self._refresh_theme_sensitive_icons()
            if getattr(self, "tab_widget", None) is not None and hasattr(self, "calibration_label"):
                self._update_calibration_panel(self.current_document())
            self._apply_tool_menu_stylesheets()

    def _apply_tool_menu_stylesheets(self) -> None:
        menu_specs = (
            (self._manual_tool_menu, "manualToolMenu", "rgba(217, 167, 42, 41)"),
            (self._area_tool_menu, "areaToolMenu", "rgba(90, 174, 105, 41)"),
            (self._magic_tool_menu, "magicToolMenu", "rgba(217, 108, 117, 41)"),
            (self._overlay_tool_menu, "overlayToolMenu", "rgba(183, 154, 216, 41)"),
        )
        for menu, object_name, checked_rgba in menu_specs:
            if menu is not None:
                menu.setStyleSheet(self._build_split_menu_stylesheet(object_name, checked_rgba))

    def _apply_theme_mode(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        self._app_settings.theme_mode = apply_application_theme(app, self._app_settings.theme_mode)
        refresh_widget_theme(self)

    def _resolved_document_path(self, document: ImageDocument, *, project_path: str | Path | None = None) -> Path:
        return document.resolved_path(project_path or self._project_path)

    def _document_display_name(self, document: ImageDocument) -> str:
        token = str(document.path or "").strip()
        if token:
            return Path(token).name or token
        return document.id

    def _document_tooltip(self, document: ImageDocument, *, project_path: str | Path | None = None) -> str:
        if document.is_digital_slide():
            resolved = self._resolved_document_path(document, project_path=project_path)
            return f"数字化切片\n{resolved}"
        if document.is_project_asset():
            resolved = self._resolved_document_path(document, project_path=project_path)
            if project_path is None and self._project_path is None:
                return f"项目内采集图片\n相对路径: {document.path}"
            return f"项目资源\n{resolved}"
        return str(self._resolved_document_path(document, project_path=project_path))

    def _first_filesystem_image_directory(self) -> Path | None:
        for document in self.project.documents:
            if document.source_type != "filesystem":
                continue
            token = str(document.path or "").strip()
            if token:
                direct_path = Path(token).expanduser()
                if direct_path.is_absolute():
                    return direct_path.parent
            try:
                resolved = self._resolved_document_path(document)
            except Exception:
                continue
            if str(resolved).strip():
                return resolved.parent
        return None

    def _preferred_dialog_directory(self, *, recent_dir: str = "") -> Path:
        candidates = [
            Path(recent_dir).expanduser() if str(recent_dir).strip() else None,
            self._first_filesystem_image_directory(),
            self._project_path.parent if self._project_path is not None else None,
            Path.home(),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                if candidate.exists():
                    return candidate
            except OSError:
                continue
        return Path.home()

    def _remember_recent_directory(self, *, setting_name: str, directory: Path, context: str) -> None:
        normalized = str(directory.expanduser().resolve())
        if getattr(self._app_settings, setting_name, "") == normalized:
            return
        setattr(self._app_settings, setting_name, normalized)
        self._save_app_settings(context=context)

    def _normalize_dialog_save_path(self, selected_path: str, default_filename: str) -> Path:
        path = Path(selected_path)
        if not path.suffix:
            default_suffix = Path(default_filename).suffix
            if default_suffix:
                path = path.with_suffix(default_suffix)
        return path

    def _single_export_dialog_filter(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        return {
            ".png": "PNG 图片 (*.png)",
            ".json": "JSON 文件 (*.json)",
            ".xlsx": "Excel 工作簿 (*.xlsx)",
            ".xlsm": "启用宏的 Excel 工作簿 (*.xlsm)",
            ".csv": "CSV 文件 (*.csv)",
        }.get(suffix, "所有文件 (*)")

    def _create_export_options_dialog(self, preset: ExportSelection) -> ExportOptionsDialog:
        return ExportOptionsDialog(
            preset,
            allow_all_scope=len(self.project.documents) > 1,
            raw_record_templates=self._app_settings.raw_record_templates,
            last_raw_record_template_path=self._app_settings.last_raw_record_template_path,
            parent=self,
        )

    def _select_export_save_path(self, default_path: Path, file_filter: str) -> str:
        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择导出文件",
            str(default_path),
            file_filter,
        )
        return selected_path

    def _select_export_directory(self, default_dir: Path) -> str:
        return QFileDialog.getExistingDirectory(self, "选择导出目录", str(default_dir))

    def _show_export_information(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    def _show_export_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _document_has_unsaved_project_changes(self, document: ImageDocument) -> bool:
        return document.dirty_flags.session_dirty or (not document.uses_sidecar() and document.dirty_flags.calibration_dirty)

    def _selected_capture_device(self) -> CaptureDevice | None:
        return self._capture_manager.selected_device()

    def _sync_live_preview_action(self) -> None:
        self.live_preview_action.blockSignals(True)
        self.live_preview_action.setChecked(self._preview_active)
        self.live_preview_action.setText("终止预览" if self._preview_active else "实时预览")
        self.live_preview_action.blockSignals(False)

    def _update_capture_device_ui(self) -> None:
        if _CAPTURE_IMPORT_ERROR is not None:
            self.switch_capture_device_action.setToolTip(f"实时预览模块不可用: {_CAPTURE_IMPORT_ERROR}")
            self.live_preview_action.setToolTip("实时预览模块不可用")
            self.optimize_capture_signal_action.setToolTip("实时预览模块不可用")
            self._sync_live_preview_action()
            return
        selected = self._selected_capture_device()
        if selected is None:
            self.switch_capture_device_action.setToolTip("切换或刷新采集设备")
            self.live_preview_action.setToolTip("开始或停止实时预览")
            self.optimize_capture_signal_action.setToolTip("当前设备不支持采集参数优化")
        else:
            self.switch_capture_device_action.setToolTip(f"当前设备: {selected.name}")
            self.live_preview_action.setToolTip(f"使用 {selected.name} 进行实时预览")
            if self._capture_manager.can_optimize_signal():
                self.optimize_capture_signal_action.setToolTip("优化当前 Microview 设备的信号/场频参数")
            else:
                self.optimize_capture_signal_action.setToolTip("当前设备不支持采集参数优化")
        self._sync_live_preview_action()

    def _capture_refresh_message(self) -> str:
        lines = ["当前未检测到可用的采集设备。"]
        warnings = self._capture_manager.device_refresh_warnings()
        if warnings:
            lines.append("")
            lines.append("采集模块诊断:")
            lines.extend(warnings[:4])
        return "\n".join(lines)

    def _on_capture_devices_changed(self, devices: object) -> None:
        self._capture_devices = list(devices) if isinstance(devices, list) else []
        if self._capture_devices and not self._app_settings.selected_capture_device_id:
            selected = self._selected_capture_device()
            if selected is not None:
                self._app_settings.selected_capture_device_id = selected.id
        self._update_capture_device_ui()
        self._sync_digital_slide_camera_label()
        self._update_action_states()

    def _refresh_capture_devices(self) -> None:
        self._capture_devices = self._capture_manager.refresh_devices()
        selected = self._selected_capture_device()
        if selected is not None and not self._app_settings.selected_capture_device_id:
            self._app_settings.selected_capture_device_id = selected.id
        self._update_capture_device_ui()
        self._sync_digital_slide_camera_label()
        self._update_action_states()

    def _set_selected_capture_device(self, device_id: str) -> None:
        restart_preview = self._capture_manager.is_preview_active()
        if restart_preview:
            self.stop_live_preview()
        if not self._capture_manager.set_selected_device(device_id):
            QMessageBox.warning(self, "切换采集设备", "无法切换到所选设备。")
            return
        self._app_settings.selected_capture_device_id = device_id
        self._save_app_settings(context="切换采集设备")
        selected = self._selected_capture_device()
        if selected is not None:
            self.statusBar().showMessage(f"当前采集设备: {selected.name}", 4000)
        self._show_active_capture_warning()
        self._update_capture_device_ui()
        self._sync_digital_slide_camera_label()
        self._update_action_states()
        if restart_preview:
            self.start_live_preview()

    def show_capture_device_menu(self) -> None:
        self._refresh_capture_devices()
        if not self._capture_devices:
            QMessageBox.information(self, "切换采集设备", self._capture_refresh_message())
            return
        menu = QMenu(self)
        for device in self._capture_devices:
            action = menu.addAction(device.name)
            action.setCheckable(True)
            action.setChecked(device.id == self._capture_manager.selected_device_id())
            action.triggered.connect(
                lambda checked=False, device_id=device.id: self._set_selected_capture_device(device_id)
            )
        menu.exec(self.cursor().pos())

    def toggle_live_preview(self, checked: bool) -> None:
        if checked:
            self.start_live_preview()
            return
        self.stop_live_preview()

    def start_live_preview(self) -> None:
        self._refresh_capture_devices()
        if not self._capture_devices:
            QMessageBox.information(self, "实时预览", self._capture_refresh_message())
            self._sync_live_preview_action()
            return
        preview_kind = self._preview_kind()
        if self._center_stack is not None and self._preview_page is not None:
            self._center_stack.setCurrentWidget(self._preview_page)
        self._apply_preview_surface(preview_kind)
        if preview_kind == "native_embed" and self._microview_preview_host is not None:
            self._apply_native_preview_resolution()
            self._microview_preview_host.ensure_native_handle()
            QApplication.processEvents()
        preview_target = self._current_preview_target()
        if not self._capture_manager.start_preview(preview_target=preview_target):
            if self._center_stack is not None:
                self._center_stack.setCurrentWidget(self.tab_widget)
            self._sync_live_preview_action()
            return
        selected = self._selected_capture_device()
        if self._preview_status_label is not None:
            if preview_kind == "native_embed":
                self._preview_status_label.setText(
                    f"正在预览: {selected.name if selected is not None else '采集设备'}  (Microview 原生预览)"
                )
            else:
                self._preview_status_label.setText(f"正在预览: {selected.name if selected is not None else '采集设备'}")
        self.statusBar().showMessage("实时预览已启动", 3000)

    def stop_live_preview(self) -> None:
        if self._slide_acquisition_active():
            QMessageBox.information(self, "数字化切片", "请先停止当前数字化切片采集。")
            return
        if self._preview_analysis_mode != "none":
            self._cancel_preview_analysis_session()
        if not self._capture_manager.is_preview_active():
            self._preview_active = False
            self._clear_preview_surface_state()
            self._clear_prompt_segmentation_cache()
            self._update_ui_for_current_document()
            self._sync_live_preview_action()
            return
        self._capture_manager.stop_preview()
        self.statusBar().showMessage("实时预览已停止", 3000)

    def _on_live_preview_state_changed(self, active: bool) -> None:
        if not active and self._preview_analysis_mode != "none":
            self._cancel_preview_analysis_session()
        self._preview_active = active
        if self._center_stack is not None and self._preview_page is not None:
            self._center_stack.setCurrentWidget(self._preview_page if active else self.tab_widget)
        if active:
            self._refresh_preview_surface()
            if self._is_native_preview() and self._microview_preview_host is not None:
                self._apply_native_preview_resolution()
                QApplication.processEvents()
                self._capture_manager.update_preview_target(self._microview_preview_host)
            self._show_active_capture_warning()
            self._maybe_hint_signal_optimization()
        if not active:
            self._digital_slide_mode = False
            self._sync_digital_slide_mode_ui()
            self._clear_preview_surface_state()
            self._clear_prompt_segmentation_cache()
        self._sync_live_preview_action()
        self._update_ui_for_current_document()
        if not active:
            QTimer.singleShot(0, self._fit_current_digital_slide_after_preview_stop)

    def _fit_current_digital_slide_after_preview_stop(self) -> None:
        if self._preview_active:
            return
        canvas = self._current_digital_slide_canvas()
        if canvas is not None:
            canvas.fit_to_view()

    def _on_live_preview_frame_ready(self, image: object) -> None:
        if not self._preview_active or self._is_native_preview():
            return
        if not isinstance(image, QImage) or image.isNull() or self._preview_canvas is None:
            return
        self._preview_frame_serial += 1
        self._latest_preview_frame = image.copy()
        display_image = self._scaled_digital_slide_preview_image(image)
        if (
            self._preview_document is None
            or self._preview_document.image_size != (display_image.width(), display_image.height())
        ):
            self._preview_document = ImageDocument(
                id="preview_document",
                path="preview_frame.png",
                image_size=(display_image.width(), display_image.height()),
                source_type="project_asset",
            )
            self._preview_canvas.set_document(self._preview_document, display_image)
            self._preview_canvas.fit_to_view()
        else:
            self._preview_canvas.set_image(display_image)
        if self._preview_status_label is not None:
            selected = self._selected_capture_device()
            label = selected.name if selected is not None else "采集设备"
            if self._digital_slide_mode and (display_image.width(), display_image.height()) != (image.width(), image.height()):
                self._preview_status_label.setText(
                    f"数字化切片模式: {label}  预览 {display_image.width()} x {display_image.height()}，采集 {image.width()} x {image.height()}"
                )
            else:
                self._preview_status_label.setText(f"正在预览: {label}  ({image.width()} x {image.height()})")
        if self._image_resolution_label is not None:
            if self._digital_slide_mode and (display_image.width(), display_image.height()) != (image.width(), image.height()):
                self._image_resolution_label.setText(
                    f"实时预览: {display_image.width()} x {display_image.height()} px    |    原始帧: {image.width()} x {image.height()} px"
                )
            else:
                self._image_resolution_label.setText(f"实时预览分辨率: {image.width()} x {image.height()} px")
        self._sync_digital_slide_camera_label()
        self._update_action_states()

    def _clear_preview_surface_state(self) -> None:
        self._preview_document = None
        self._latest_preview_frame = None
        self._preview_frame_serial = 0
        self._apply_preview_surface("frame_stream")
        if self._preview_canvas is not None:
            self._preview_canvas.clear_document()
        if self._preview_status_label is not None:
            self._preview_status_label.setText("请选择采集设备并开始实时预览")

    def _on_capture_error(self, message: str) -> None:
        self._sync_live_preview_action()
        self._update_action_states()
        self.statusBar().showMessage(message, 5000)
        QMessageBox.warning(self, "实时预览", message)

    def _next_project_capture_relative_path(self) -> str:
        existing = {
            document.path
            for document in self.project.documents
            if document.is_project_asset()
        }
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = 1
        while True:
            candidate = f"captures/capture_{stamp}_{counter:02d}.png"
            if candidate not in existing:
                return candidate
            counter += 1

    def _persist_project_assets(self, target_path: Path) -> bool:
        return self.project_session_controller.persist_project_assets(target_path)

    def _project_asset_image_for_save(self, document: ImageDocument) -> QImage | None:
        return self._images.get(document.id)

    def _show_project_information(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    def _show_project_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _select_project_save_path(self, default_path: Path) -> str:
        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存项目",
            str(default_path),
            self.PROJECT_FILTER,
        )
        return selected_path

    def _select_project_open_path(self) -> str:
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", self.PROJECT_FILTER)
        return path

    def _show_status_message(self, message: str, timeout_ms: int = 0) -> None:
        self.statusBar().showMessage(message, timeout_ms)

    def is_image_loading(self) -> bool:
        return self.background_task_controller.is_image_loading()

    def _show_area_inference_warning(self, message: str) -> None:
        QMessageBox.warning(self, "面积自动识别", message)

    def _request_capture_analysis_frame(self, request_id: int) -> bool:
        return self._capture_manager.request_analysis_frame(request_id)

    def capture_current_frame(self) -> None:
        was_preview_active = self._capture_manager.is_preview_active()
        selected_device = self._selected_capture_device()
        stop_before_capture = bool(
            was_preview_active
            and selected_device is not None
            and selected_device.backend_key == "microview"
            and self._capture_manager.can_capture_still()
        )
        if stop_before_capture:
            self.stop_live_preview()
        frame: QImage | None
        try:
            frame = self._capture_manager.capture_still_frame() if self._capture_manager.can_capture_still() else self._capture_manager.last_frame()
        except Exception as exc:
            QMessageBox.warning(self, "采集一张", str(exc))
            return
        if frame is None or frame.isNull():
            diagnostics = self._capture_manager.capture_diagnostics().strip()
            if diagnostics:
                QMessageBox.warning(self, "采集一张", f"当前未抓拍到有效图像。\n\n抓拍诊断:\n{diagnostics}")
            else:
                QMessageBox.information(self, "采集一张", "当前还没有可用的预览画面。")
            return
        if was_preview_active and not stop_before_capture:
            self.stop_live_preview()
        self._add_project_asset_image(frame, status_message="已采集当前画面到项目内存")

    def optimize_capture_signal(self) -> None:
        selected = self._selected_capture_device()
        if selected is None or not self._capture_manager.can_optimize_signal():
            QMessageBox.information(self, "优化采集参数", "当前设备不支持自动优化采集参数。")
            return
        restart_preview = self._capture_manager.is_preview_active()
        if restart_preview:
            self.stop_live_preview()
        try:
            message = self._capture_manager.optimize_signal()
        except Exception as exc:
            QMessageBox.warning(self, "优化采集参数", str(exc))
            if restart_preview:
                self.start_live_preview()
            return
        QMessageBox.information(self, "优化采集参数", message)
        if restart_preview:
            self.start_live_preview()

    def _toggle_digital_slide_mode(self, checked: bool) -> None:
        if checked and not self._preview_active:
            self.start_live_preview()
            if not self._preview_active:
                self._sync_digital_slide_mode_ui()
                return
        if not checked and self._slide_acquisition_active():
            if self.digital_slide_action is not None:
                self.digital_slide_action.blockSignals(True)
                self.digital_slide_action.setChecked(True)
                self.digital_slide_action.blockSignals(False)
            QMessageBox.information(self, "数字化切片", "请先停止当前数字化切片采集。")
            return
        if checked and self._preview_analysis_mode != "none":
            self._cancel_preview_analysis_session()
        self._digital_slide_mode = bool(checked)
        self._sync_digital_slide_mode_ui()
        if checked:
            self._apply_digital_slide_motion_settings()
            self._reset_digital_slide_motion_zero(axes=(AXIS_X, AXIS_Y, AXIS_Z))
            self._refresh_digital_slide_ports(prefer_auto=True)
            self._check_digital_slide_motion_status()
            if self._app_settings.digital_slide_motor_output_enabled and not self._slide_motion.enabled:
                ok, _message = self._slide_motion.check_available()
                if ok and self._digital_slide_motor_enable is not None:
                    self._digital_slide_motor_enable.setChecked(True)
            self.statusBar().showMessage("已进入数字化切片模式", 3000)
        else:
            self._end_digital_slide_jog()
            self.statusBar().showMessage("已退出数字化切片模式", 3000)
        self._update_action_states()

    def _sync_digital_slide_mode_ui(self) -> None:
        active = bool(self._digital_slide_mode and self._preview_active)
        if self.digital_slide_action is not None:
            self.digital_slide_action.blockSignals(True)
            self.digital_slide_action.setChecked(active)
            self.digital_slide_action.setText("数字化切片")
            self.digital_slide_action.blockSignals(False)
        if self._left_standard_splitter is not None:
            self._left_standard_splitter.setVisible(not active)
        if self._digital_slide_left_panel is not None:
            self._digital_slide_left_panel.setVisible(active)
        if self._right_standard_panel is not None:
            self._right_standard_panel.setVisible(not active)
        if self._digital_slide_right_panel is not None:
            self._digital_slide_right_panel.setVisible(active)
        if active:
            self._ensure_digital_slide_left_width()
        if active and self._preview_status_label is not None:
            self._preview_status_label.setText("数字化切片模式：设置范围后点击开始采集")
        if active and self._preview_canvas is not None and self._preview_document is not None:
            self._preview_canvas.fit_to_view()

    def _ensure_digital_slide_left_width(self) -> None:
        splitter = self._main_splitter
        if splitter is None:
            return
        sizes = splitter.sizes()
        if len(sizes) < 3:
            return
        target_left = 380
        if sizes[0] >= target_left:
            return
        available_from_center = max(0, sizes[1] - 360)
        delta = min(target_left - sizes[0], available_from_center)
        if delta <= 0:
            return
        splitter.setSizes([sizes[0] + delta, sizes[1] - delta, sizes[2]])

    def _digital_slide_selected_port(self) -> str:
        if self._digital_slide_port_combo is None:
            return ""
        data = self._digital_slide_port_combo.currentData()
        text = self._digital_slide_port_combo.currentText()
        if data and str(text).startswith(str(data)):
            return str(data)
        return str(text or "").strip().split()[0] if str(text or "").strip() else ""

    def _refresh_digital_slide_ports(self, *, prefer_auto: bool) -> None:
        if self._digital_slide_port_combo is None:
            return
        ports = list_motion_ports()
        current = self._slide_motion.port or self._digital_slide_selected_port()
        preferred = current
        auto = preferred_motion_port(ports)
        if prefer_auto or not preferred:
            preferred = auto.device if auto is not None else preferred
        if not preferred:
            preferred = "COM3"
        self._digital_slide_port_combo.blockSignals(True)
        self._digital_slide_port_combo.clear()
        for item in ports:
            self._digital_slide_port_combo.addItem(item.display_label(), item.device)
        if not ports:
            self._digital_slide_port_combo.addItem(preferred, preferred)
        for index in range(self._digital_slide_port_combo.count()):
            if self._digital_slide_port_combo.itemData(index) == preferred:
                self._digital_slide_port_combo.setCurrentIndex(index)
                break
        else:
            self._digital_slide_port_combo.setEditText(preferred)
        self._digital_slide_port_combo.blockSignals(False)
        self._set_digital_slide_motion_port(preferred)
        ftdi_count = len([item for item in ports if item.is_ftdi_motion])
        self._set_digital_slide_status(f"检测到 {len(ports)} 个串口，FTDI 候选 {ftdi_count} 个")

    def _set_digital_slide_motion_port(self, port: str) -> None:
        port = str(port or "").strip().split()[0] if str(port or "").strip() else ""
        if not port or port == self._slide_motion.port:
            self._sync_digital_slide_port_card()
            return
        was_enabled = self._slide_motion.enabled
        if was_enabled and self._digital_slide_motor_enable is not None:
            self._digital_slide_motor_enable.setChecked(False)
        self._slide_motion.close()
        self._slide_motion.port = port
        self._set_digital_slide_status(f"控制串口: {port}")

    def _on_digital_slide_port_changed(self) -> None:
        self._set_digital_slide_motion_port(self._digital_slide_selected_port())

    def _check_digital_slide_motion_status(self) -> None:
        self._set_digital_slide_motion_port(self._digital_slide_selected_port())
        ok, message = self._slide_motion.check_available()
        self._set_digital_slide_status(message)
        if not ok and self._digital_slide_motor_enable is not None:
            self._digital_slide_motor_enable.setChecked(False)

    def _set_digital_slide_motor_enabled(self, enabled: bool) -> None:
        try:
            self._slide_motion.set_enabled(enabled)
        except Exception as exc:
            if self._digital_slide_motor_enable is not None:
                self._digital_slide_motor_enable.blockSignals(True)
                self._digital_slide_motor_enable.setChecked(False)
                self._digital_slide_motor_enable.blockSignals(False)
            self._slide_motion.enabled = False
            QMessageBox.warning(self, "数字化切片", f"无法启用电机输出：\n{exc}")

    def _on_digital_slide_motion_status(self, message: str) -> None:
        self._set_digital_slide_status(message)

    def _on_digital_slide_position_changed(self, position: object) -> None:
        self._sync_digital_slide_position_label()
        self._sync_digital_slide_visuals()

    def _set_digital_slide_status(self, message: str) -> None:
        if self._digital_slide_status_label is not None:
            self._digital_slide_status_label.setText(message)
        if self._digital_slide_connection_summary_label is not None:
            self._digital_slide_connection_summary_label.setText(str(message or "尚未检查").splitlines()[0])
        self._sync_digital_slide_port_card()
        self.statusBar().showMessage(message, 3000)

    def _set_digital_slide_timing(self, message: str) -> None:
        self._slide_acquisition_last_timing_summary = message
        if self._digital_slide_timing_label is not None:
            self._digital_slide_timing_label.setText(message)
        if self._digital_slide_diagnostics_summary_label is not None:
            self._digital_slide_diagnostics_summary_label.setText(self._compact_digital_slide_timing(message))

    def _compact_digital_slide_timing(self, message: str) -> str:
        body = str(message or "").replace("耗时:", "").strip()
        if not body or body == "-":
            return "上一张: -"
        parts = [part.strip() for part in body.split("|") if part.strip()]
        if not parts:
            return body[:48]
        write = next((part for part in parts if part.startswith("写入")), "")
        scale = next((part for part in parts if part.startswith("缩放")), "")
        frame_wait = next((part for part in parts if part.startswith("等帧")), "")
        first = parts[0]
        compact_parts = [part for part in (first, frame_wait, scale, write) if part]
        if compact_parts:
            return "上一张: " + " / ".join(compact_parts[:3])
        return body[:48]

    def _sync_digital_slide_port_card(self) -> None:
        if self._digital_slide_port_card_label is None:
            return
        port = self._slide_motion.port or self._digital_slide_selected_port()
        self._digital_slide_port_card_label.setText(port or "未选择")

    def _digital_slide_effective_settings(self) -> AppSettings:
        return self._slide_acquisition_settings or self._app_settings

    def _digital_slide_pixel_stride_mode(self, settings: AppSettings | None = None) -> str:
        active_settings = settings or self._digital_slide_effective_settings()
        return "manual_pixels" if active_settings.digital_slide_pixel_stride_mode == "manual_pixels" else "auto_overlap"

    def _sync_digital_slide_pixel_stride_controls(self) -> None:
        return None

    def _digital_slide_capture_max_width(self, settings: AppSettings | None = None) -> int | None:
        active_settings = settings or self._digital_slide_effective_settings()
        data = int(getattr(active_settings, "digital_slide_capture_max_width", 1600) or 0)
        if data <= 0:
            return None
        return max(1, data)

    def _digital_slide_preview_max_width(self, settings: AppSettings | None = None) -> int | None:
        active_settings = settings or self._app_settings
        data = int(getattr(active_settings, "digital_slide_preview_max_width", 1280) or 0)
        if data <= 0:
            return None
        return max(1, data)

    def _digital_slide_scaled_size(
        self,
        source_width: int,
        source_height: int,
        settings: AppSettings | None = None,
    ) -> tuple[int, int, float]:
        source_width = max(1, int(source_width))
        source_height = max(1, int(source_height))
        max_width = self._digital_slide_capture_max_width(settings)
        if max_width is None or source_width <= max_width:
            return source_width, source_height, 1.0
        scale = max_width / source_width
        return max(1, int(max_width)), max(1, int(source_height * scale)), scale

    def _scale_digital_slide_frame(self, frame: QImage) -> tuple[QImage, float]:
        target_width, target_height, scale = self._digital_slide_scaled_size(
            frame.width(),
            frame.height(),
            self._digital_slide_effective_settings(),
        )
        if target_width == frame.width() and target_height == frame.height():
            return frame.copy(), scale
        return (
            frame.scaled(
                target_width,
                target_height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ),
            scale,
        )

    def _scaled_digital_slide_preview_image(self, frame: QImage) -> QImage:
        if frame.isNull() or not self._digital_slide_mode:
            return frame
        max_width = self._digital_slide_preview_max_width()
        if max_width is None or frame.width() <= max_width:
            return frame
        target_height = max(1, int(frame.height() * (max_width / max(1, frame.width()))))
        return frame.scaled(
            max_width,
            target_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _parse_optional_int_edit(self, edit: QLineEdit | None) -> int | None:
        if edit is None:
            return None
        token = edit.text().strip()
        if not token:
            return None
        try:
            return int(token)
        except ValueError:
            return None

    def _set_optional_int_edit(self, edit: QLineEdit | None, value: int | None) -> None:
        if edit is None:
            return
        edit.setText("" if value is None else str(int(value)))

    def _remember_digital_slide_z_capture_settings(self) -> None:
        if self._slide_acquisition_active():
            return
        self._app_settings.digital_slide_z_capture_lower = self._parse_optional_int_edit(self._digital_slide_z_lower_edit)
        self._app_settings.digital_slide_z_capture_upper = self._parse_optional_int_edit(self._digital_slide_z_upper_edit)
        if self._digital_slide_z_step_spin is not None:
            self._app_settings.digital_slide_z_capture_step = int(self._digital_slide_z_step_spin.value())
        self._save_app_settings(context="数字化切片Z采集范围")

    def _on_digital_slide_z_capture_step_changed(self) -> None:
        self._sync_digital_slide_task_state()
        self._remember_digital_slide_z_capture_settings()

    def _digital_slide_output_path(self) -> Path | None:
        token = self._digital_slide_output_path_edit.text().strip() if self._digital_slide_output_path_edit is not None else ""
        if not token:
            return None
        path = Path(token).expanduser()
        if path.suffix.lower() != DIGITAL_SLIDE_SUFFIX:
            path = path.with_suffix(DIGITAL_SLIDE_SUFFIX)
        return path

    def _choose_digital_slide_output_path(self) -> Path | None:
        initial = self._digital_slide_output_path_edit.text().strip() if self._digital_slide_output_path_edit is not None else ""
        if not initial:
            initial = self._app_settings.digital_slide_last_output_path
        if not initial:
            initial_dir = self._app_settings.recent_export_dir or str(Path.home())
            initial = str(Path(initial_dir) / f"slide_{datetime.now().strftime('%Y%m%d_%H%M%S')}{DIGITAL_SLIDE_SUFFIX}")
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "选择数字化切片输出文件",
            initial,
            "数字化切片 (*.fdmslide)",
        )
        if not path:
            return None
        output_path = Path(path).expanduser()
        if output_path.suffix.lower() != DIGITAL_SLIDE_SUFFIX:
            output_path = output_path.with_suffix(DIGITAL_SLIDE_SUFFIX)
        if self._digital_slide_output_path_edit is not None:
            self._digital_slide_output_path_edit.setText(str(output_path))
        self._remember_digital_slide_output_path(output_path)
        return output_path

    def _remember_digital_slide_output_path(self, path: Path) -> None:
        self._app_settings.digital_slide_last_output_path = str(path)
        if path.parent:
            self._app_settings.recent_export_dir = str(path.parent)
        self._save_app_settings(context="数字化切片输出路径")

    def _clear_digital_slide_output_path(self) -> None:
        if self._digital_slide_output_path_edit is not None:
            self._digital_slide_output_path_edit.blockSignals(True)
            self._digital_slide_output_path_edit.clear()
            self._digital_slide_output_path_edit.blockSignals(False)
        self._app_settings.digital_slide_last_output_path = ""
        self._save_app_settings(context="数字化切片输出路径")
        self._sync_digital_slide_task_state()

    def _on_digital_slide_output_path_changed(self) -> None:
        self._sync_digital_slide_task_state()

    def _set_digital_slide_z_bound(self, bound: str) -> None:
        current_z = int(self._slide_motion.relative_pos.get(AXIS_Z, 0))
        if bound == "upper":
            self._set_optional_int_edit(self._digital_slide_z_upper_edit, current_z)
        else:
            self._set_optional_int_edit(self._digital_slide_z_lower_edit, current_z)
        self._remember_digital_slide_z_capture_settings()
        self._sync_digital_slide_task_state()

    def _on_digital_slide_rows_cols_edited(self) -> None:
        self._digital_slide_rows_cols_manual = True
        self._sync_digital_slide_task_state()

    def _mark_digital_slide_region(self, marker: str) -> None:
        pos = self._slide_motion.relative_pos
        x = int(pos.get(AXIS_X, 0))
        y = int(pos.get(AXIS_Y, 0))
        self._digital_slide_region_anchor_points[marker] = (x, y)
        if "left" in marker:
            self._digital_slide_region_bounds["left"] = x
        if "right" in marker:
            self._digital_slide_region_bounds["right"] = x
        if "top" in marker:
            self._digital_slide_region_bounds["top"] = y
        if "bottom" in marker:
            self._digital_slide_region_bounds["bottom"] = y
        self._update_digital_slide_region_counts_from_bounds()
        self._sync_digital_slide_task_state()

    def _clear_digital_slide_region(self) -> None:
        self._digital_slide_region_anchor_points.clear()
        self._digital_slide_region_bounds.clear()
        self._digital_slide_rows_cols_manual = False
        if self._digital_slide_cols_edit is not None:
            self._digital_slide_cols_edit.clear()
        if self._digital_slide_rows_edit is not None:
            self._digital_slide_rows_edit.clear()
        self._sync_digital_slide_task_state()

    def _update_digital_slide_region_counts_from_bounds(self) -> None:
        if not {"left", "right", "top", "bottom"}.issubset(self._digital_slide_region_bounds):
            return
        x_step = abs(int(self._app_settings.digital_slide_x_stage_step))
        y_step = abs(int(self._app_settings.digital_slide_y_stage_step))
        if x_step <= 0 or y_step <= 0:
            return
        cols = max(1, int(math.ceil(abs(self._digital_slide_region_bounds["right"] - self._digital_slide_region_bounds["left"]) / x_step)) + 1)
        rows = max(1, int(math.ceil(abs(self._digital_slide_region_bounds["bottom"] - self._digital_slide_region_bounds["top"]) / y_step)) + 1)
        if not self._digital_slide_rows_cols_manual:
            if self._digital_slide_cols_edit is not None:
                self._digital_slide_cols_edit.setText(str(cols))
            if self._digital_slide_rows_edit is not None:
                self._digital_slide_rows_edit.setText(str(rows))

    def _digital_slide_has_region(self) -> bool:
        return {"left", "right", "top", "bottom"}.issubset(self._digital_slide_region_bounds)

    def _sync_digital_slide_task_state(self) -> None:
        self._sync_digital_slide_visuals()
        if self._digital_slide_start_button is None:
            return
        has_path = self._digital_slide_output_path() is not None
        active = self._slide_acquisition_active()
        for widget in self._digital_slide_locked_controls:
            widget.setEnabled(not active)
        for widget in self._digital_slide_motion_controls:
            widget.setEnabled(not active)
        self._digital_slide_start_button.setEnabled(not active)
        self._digital_slide_start_button.setToolTip("" if has_path else "点击后选择 .fdmslide 输出路径")
        if self._digital_slide_stop_button is not None:
            self._digital_slide_stop_button.setEnabled(active)
        self._apply_digital_slide_action_button_styles(has_path=has_path)

    def _apply_digital_slide_action_button_styles(self, *, has_path: bool) -> None:
        if self._digital_slide_start_button is not None:
            if has_path:
                self._digital_slide_start_button.setIcon(themed_icon("digital_slide_start", color="#ECFDF5"))
                self._digital_slide_start_button.setStyleSheet(
                    "QPushButton#digitalSlideStartButton {"
                    "background: #047857;"
                    "color: #ECFDF5;"
                    "border: 1px solid #059669;"
                    "border-radius: 8px;"
                    "font-weight: 800;"
                    "padding: 8px 14px;"
                    "}"
                    "QPushButton#digitalSlideStartButton:hover { background: #059669; }"
                    "QPushButton#digitalSlideStartButton:disabled {"
                    "background: #94A3B8;"
                    "border-color: #CBD5E1;"
                    "color: #F8FAFC;"
                    "}"
                )
            else:
                self._digital_slide_start_button.setIcon(themed_icon("digital_slide_start", color="#F59E0B"))
                self._digital_slide_start_button.setStyleSheet(
                    "QPushButton#digitalSlideStartButton {"
                    "background: #1F2937;"
                    "color: #FDE68A;"
                    "border: 1px solid #F59E0B;"
                    "border-radius: 8px;"
                    "font-weight: 800;"
                    "padding: 8px 14px;"
                    "}"
                    "QPushButton#digitalSlideStartButton:hover { background: #374151; }"
                    "QPushButton#digitalSlideStartButton:disabled {"
                    "background: #E5E7EB;"
                    "border-color: #CBD5E1;"
                    "color: #94A3B8;"
                    "}"
                )
        if self._digital_slide_stop_button is not None:
            self._digital_slide_stop_button.setStyleSheet(
                "QPushButton#digitalSlideStopButton {"
                "background: #B91C1C;"
                "color: #FEF2F2;"
                "border: 1px solid #DC2626;"
                "border-radius: 8px;"
                "font-weight: 800;"
                "padding: 8px 14px;"
                "}"
                "QPushButton#digitalSlideStopButton:hover { background: #DC2626; }"
                "QPushButton#digitalSlideStopButton:disabled {"
                "background: #E5E7EB;"
                "border-color: #CBD5E1;"
                "color: #94A3B8;"
                "}"
            )

    def _sync_digital_slide_visuals(self) -> None:
        lower = self._parse_optional_int_edit(self._digital_slide_z_lower_edit)
        upper = self._parse_optional_int_edit(self._digital_slide_z_upper_edit)
        current_z = int(self._slide_motion.relative_pos.get(AXIS_Z, 0))
        if self._digital_slide_z_rail is not None:
            self._digital_slide_z_rail.set_state(
                soft_limit=self._app_settings.digital_slide_z_soft_limit,
                current_z=current_z,
                lower_z=lower,
                upper_z=upper,
            )
        if self._digital_slide_range_map is not None:
            pos = self._slide_motion.relative_pos
            self._digital_slide_range_map.set_state(
                current_xy=(int(pos.get(AXIS_X, 0)), int(pos.get(AXIS_Y, 0))),
                bounds=self._digital_slide_region_bounds,
                stage_step=(
                    int(self._app_settings.digital_slide_x_stage_step),
                    int(self._app_settings.digital_slide_y_stage_step),
                ),
            )
        self._sync_digital_slide_motion_settings_label()

    def _sync_digital_slide_motion_settings_label(self) -> None:
        if self._digital_slide_motion_settings_label is None:
            return
        self._digital_slide_motion_settings_label.setText(
            "设置: "
            f"XY步距 {self._app_settings.digital_slide_xy_jog_step} steps，"
            f"对焦步距 {self._app_settings.digital_slide_z_jog_step} steps，"
            f"长按 {self._app_settings.digital_slide_jog_rate} 次/秒\n"
            f"软限位: XY +/-{self._app_settings.digital_slide_xy_soft_limit}，"
            f"Z +/-{self._app_settings.digital_slide_z_soft_limit}"
        )
        if self._digital_slide_stage_summary_label is not None:
            pos = self._slide_motion.relative_pos
            self._digital_slide_stage_summary_label.setText(
                f"X={pos.get(AXIS_X, 0)}  Y={pos.get(AXIS_Y, 0)}  Z={pos.get(AXIS_Z, 0)}"
            )

    def _apply_digital_slide_motion_settings(self) -> None:
        self._slide_motion.set_soft_limit(AXIS_X, self._app_settings.digital_slide_xy_soft_limit)
        self._slide_motion.set_soft_limit(AXIS_Y, self._app_settings.digital_slide_xy_soft_limit)
        self._slide_motion.set_soft_limit(AXIS_Z, self._app_settings.digital_slide_z_soft_limit)
        self._sync_digital_slide_motion_settings_label()
        self._sync_digital_slide_visuals()

    def _digital_slide_scaled_calibration(
        self,
        calibration: Calibration,
        manifest: DigitalSlideManifest,
    ) -> Calibration:
        metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
        try:
            capture_scale = float(metadata.get("capture_scale", 1.0) or 1.0)
        except (TypeError, ValueError):
            capture_scale = 1.0
        return Calibration(
            mode=calibration.mode,
            pixels_per_unit=max(0.000001, calibration.pixels_per_unit * capture_scale),
            unit=calibration.unit,
            source_label=calibration.source_label,
        )

    def _sync_digital_slide_camera_label(self) -> None:
        if self._digital_slide_camera_label is None:
            return
        selected = self._selected_capture_device()
        if selected is None:
            self._digital_slide_camera_label.setText("相机: 未选择")
            if self._digital_slide_connection_summary_label is not None:
                port = self._slide_motion.port or self._digital_slide_selected_port() or "未选择"
                self._digital_slide_connection_summary_label.setText(f"端口 {port} / 相机未选择")
            return
        backend_name = {
            "opencv": "OpenCV / Do3think",
            "microview": "Microview",
            "qt_multimedia": "Qt Multimedia",
        }.get(selected.backend_key, selected.backend_key)
        lines = [f"相机: {selected.name}", f"后端: {backend_name}"]
        resolution = self._capture_manager.preview_resolution()
        if resolution is not None:
            lines.append(f"分辨率: {resolution[0]} x {resolution[1]} px")
        else:
            last_frame = self._capture_manager.last_frame()
            if last_frame is not None and not last_frame.isNull():
                lines.append(f"最后帧: {last_frame.width()} x {last_frame.height()} px")
        detail = str(getattr(selected, "detail", "")).strip()
        if detail:
            lines.append(detail)
        if selected.backend_key != "opencv":
            lines.append("目标设备建议选择 OpenCV Do3think 相机。")
        self._digital_slide_camera_label.setText("\n".join(lines))
        if self._digital_slide_connection_summary_label is not None:
            port = self._slide_motion.port or self._digital_slide_selected_port() or "未选择"
            self._digital_slide_connection_summary_label.setText(f"端口 {port} / {selected.name}")

    def _sync_digital_slide_position_label(self) -> None:
        if self._digital_slide_position_label is None:
            return
        pos = self._slide_motion.relative_pos
        self._digital_slide_position_label.setText(
            f"命令相对位置: X={pos.get(AXIS_X, 0)}  Y={pos.get(AXIS_Y, 0)}  Z={pos.get(AXIS_Z, 0)}"
        )

    def _reset_digital_slide_motion_zero(self, *, axes: tuple[str, ...] | None = None) -> None:
        if axes is None:
            axes = (AXIS_X, AXIS_Y, AXIS_Z)
        self._clear_digital_slide_region()
        self._slide_motion.reset_relative_zero(axes=axes)
        self._sync_digital_slide_position_label()
        self._sync_digital_slide_visuals()
        if set(axes) == {AXIS_X, AXIS_Y, AXIS_Z}:
            message = "已将当前命令位置设为零点"
        elif set(axes) == {AXIS_X, AXIS_Y}:
            message = "已将当前 XY 位置设为样品台原点"
        else:
            message = "已将当前 Z 位置设为高度原点"
        self._set_digital_slide_status(message)

    def _digital_slide_manual_step(self, axis: str) -> int:
        if axis == AXIS_Z:
            return int(self._app_settings.digital_slide_z_jog_step)
        return int(self._app_settings.digital_slide_xy_jog_step)

    def _on_digital_slide_manual_step_changed(self) -> None:
        if self._slide_acquisition_active():
            return
        if self._digital_slide_xy_jog_step_spin is not None:
            self._app_settings.digital_slide_xy_jog_step = int(self._digital_slide_xy_jog_step_spin.value())
        if self._digital_slide_focus_jog_step_spin is not None:
            self._app_settings.digital_slide_z_jog_step = int(self._digital_slide_focus_jog_step_spin.value())
        self._sync_digital_slide_motion_settings_label()
        self._save_app_settings(context="数字化切片运动步距")

    def _begin_digital_slide_jog(self, axis: str, direction: str) -> None:
        if not self._digital_slide_mode or self._slide_acquisition_active():
            return
        self._end_digital_slide_jog()
        self._slide_jog_request = {
            "axis": axis,
            "direction": direction,
            "long_active": False,
        }
        self._slide_jog_single_shot_timer.start(300)

    def _activate_digital_slide_long_jog(self) -> None:
        if self._slide_jog_request is None:
            return
        self._slide_jog_request["long_active"] = True
        self._perform_digital_slide_jog_repeat()
        rate = int(self._app_settings.digital_slide_jog_rate)
        self._slide_jog_timer.start(max(20, int(round(1000 / max(1, rate)))))

    def _perform_digital_slide_jog_repeat(self) -> None:
        request = self._slide_jog_request
        if request is None:
            return
        self._perform_digital_slide_jog_step(str(request["axis"]), str(request["direction"]))

    def _end_digital_slide_jog(self) -> None:
        request = self._slide_jog_request
        if request is None:
            self._slide_jog_single_shot_timer.stop()
            self._slide_jog_timer.stop()
            return
        was_long = bool(request.get("long_active"))
        self._slide_jog_single_shot_timer.stop()
        self._slide_jog_timer.stop()
        self._slide_jog_request = None
        if not was_long:
            self._perform_digital_slide_jog_step(str(request["axis"]), str(request["direction"]))

    def _perform_digital_slide_jog_step(self, axis: str, direction: str) -> None:
        try:
            self._slide_motion.move(axis, self._digital_slide_manual_step(axis), direction, label=f"{axis_name(axis)} jog")
        except Exception as exc:
            QMessageBox.warning(self, "数字化切片", f"移动失败：\n{exc}")
            self._end_digital_slide_jog()

    def _slide_acquisition_active(self) -> bool:
        writer = self._slide_acquisition_writer
        return bool(
            self._slide_acquisition_store is not None
            or self._slide_acquisition_timer.isActive()
            or self._slide_acquisition_pending_write is not None
            or self._slide_acquisition_finishing is not None
            or self._slide_acquisition_discard_message is not None
            or (writer is not None and writer.is_running())
        )

    def _start_digital_slide_acquisition(self) -> None:
        if not self._digital_slide_mode or not self._preview_active:
            QMessageBox.information(self, "数字化切片", "请先进入数字化切片模式并启动实时预览。")
            return
        if not self._slide_motion.enabled:
            QMessageBox.information(self, "数字化切片", "请先启用电机输出。")
            return
        output_path = self._digital_slide_output_path()
        if output_path is None:
            output_path = self._choose_digital_slide_output_path()
            if output_path is None:
                return
        if output_path.exists():
            response = QMessageBox.question(
                self,
                "覆盖数字化切片",
                f"输出文件已存在，是否覆盖？\n{output_path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
        acquisition_settings = self._app_settings.normalized_copy()
        frame = self._latest_preview_frame
        if frame is None or frame.isNull():
            frame = self._capture_manager.last_frame()
        if frame is None or frame.isNull():
            QMessageBox.information(self, "数字化切片", "等待第一帧预览后再开始采集。")
            return
        source_width, source_height = frame.width(), frame.height()
        view_width, view_height, capture_scale = self._digital_slide_scaled_size(
            source_width,
            source_height,
            acquisition_settings,
        )
        z_lower = self._parse_optional_int_edit(self._digital_slide_z_lower_edit)
        z_upper = self._parse_optional_int_edit(self._digital_slide_z_upper_edit)
        if z_lower is None or z_upper is None:
            QMessageBox.warning(self, "数字化切片", "请先设置 Z 上限和 Z 下限。")
            return
        z_step = self._digital_slide_z_step_spin.value() if self._digital_slide_z_step_spin is not None else self._app_settings.digital_slide_z_jog_step
        if z_upper < z_lower:
            QMessageBox.warning(self, "数字化切片", "Z 上限不能小于 Z 下限。")
            return
        z_limit = int(acquisition_settings.digital_slide_z_soft_limit)
        if z_limit > 0 and (abs(z_lower) > z_limit or abs(z_upper) > z_limit):
            QMessageBox.warning(self, "数字化切片", f"Z 上下限不能超过软限位 +/-{z_limit} steps。")
            return
        current_z = int(self._slide_motion.relative_pos.get(AXIS_Z, 0))
        if current_z < z_lower or current_z > z_upper:
            response = QMessageBox.question(
                self,
                "当前 Z 不在采集范围内",
                (
                    f"当前 Z={current_z} steps，不在采集范围 {z_lower} ~ {z_upper} steps 内。\n"
                    "开始后设备会按采集流程移动到 Z 下限并继续采集，是否继续？"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
        self._remember_digital_slide_z_capture_settings()
        focus_levels = list(range(z_lower, z_upper + 1, max(1, z_step)))
        if not focus_levels or focus_levels[-1] != z_upper:
            focus_levels.append(z_upper)
        cols = self._parse_optional_int_edit(self._digital_slide_cols_edit)
        rows = self._parse_optional_int_edit(self._digital_slide_rows_edit)
        if cols is None or rows is None or cols <= 0 or rows <= 0:
            QMessageBox.warning(self, "数字化切片", "请先设置有效的列数和行数。")
            return
        overlap = int(acquisition_settings.digital_slide_overlap_percent) / 100.0
        auto_pixel_stride_x = max(1, int(round(view_width * (1.0 - overlap))))
        auto_pixel_stride_y = max(1, int(round(view_height * (1.0 - overlap))))
        pixel_stride_mode = self._digital_slide_pixel_stride_mode(acquisition_settings)
        if pixel_stride_mode == "manual_pixels":
            pixel_stride_x = int(acquisition_settings.digital_slide_x_pixel_stride)
            pixel_stride_y = int(acquisition_settings.digital_slide_y_pixel_stride)
        else:
            pixel_stride_x = auto_pixel_stride_x
            pixel_stride_y = auto_pixel_stride_y
        x_stage_step = int(acquisition_settings.digital_slide_x_stage_step)
        y_stage_step = int(acquisition_settings.digital_slide_y_stage_step)
        if cols > 1 and x_stage_step == 0:
            QMessageBox.warning(self, "数字化切片", "列数大于 1 时，X 行内步距不能为 0。")
            return
        if rows > 1 and y_stage_step == 0:
            QMessageBox.warning(self, "数字化切片", "行数大于 1 时，Y 换行步距不能为 0。")
            return
        stage_region = dict(self._digital_slide_region_bounds) if self._digital_slide_has_region() else None
        if stage_region is not None:
            xy_limit = int(acquisition_settings.digital_slide_xy_soft_limit)
            for label, value in (("X 左", stage_region["left"]), ("X 右", stage_region["right"])):
                if xy_limit > 0 and abs(int(value)) > xy_limit:
                    QMessageBox.warning(self, "数字化切片", f"{label} 边界超过 XY 软限位 +/-{xy_limit} steps。")
                    return
            for label, value in (("Y 上", stage_region["top"]), ("Y 下", stage_region["bottom"])):
                if xy_limit > 0 and abs(int(value)) > xy_limit:
                    QMessageBox.warning(self, "数字化切片", f"{label} 边界超过 XY 软限位 +/-{xy_limit} steps。")
                    return
        xy_settle_ms = int(acquisition_settings.digital_slide_xy_settle_ms)
        xy_post_settle_ms = int(acquisition_settings.digital_slide_xy_post_settle_ms)
        z_settle_ms = int(acquisition_settings.digital_slide_z_settle_ms)
        z_post_settle_ms = int(acquisition_settings.digital_slide_z_post_settle_ms)
        discard_frames = int(acquisition_settings.digital_slide_discard_frames)
        blend_width = int(acquisition_settings.digital_slide_blend_width)
        image_width = view_width + ((cols - 1) * pixel_stride_x)
        image_height = view_height + ((rows - 1) * pixel_stride_y)
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._remember_digital_slide_output_path(output_path)
        capture_max_width = self._digital_slide_capture_max_width(acquisition_settings)
        settings_snapshot = {
            "preview_max_width": self._digital_slide_preview_max_width(acquisition_settings) if self._digital_slide_preview_max_width(acquisition_settings) is not None else "original",
            "capture_max_width": capture_max_width if capture_max_width is not None else "original",
            "xy_soft_limit": acquisition_settings.digital_slide_xy_soft_limit,
            "z_soft_limit": acquisition_settings.digital_slide_z_soft_limit,
            "xy_jog_step": acquisition_settings.digital_slide_xy_jog_step,
            "z_jog_step": acquisition_settings.digital_slide_z_jog_step,
        }
        manifest = DigitalSlideManifest(
            version=1,
            width=image_width,
            height=image_height,
            viewport_width=view_width,
            viewport_height=view_height,
            focus_levels=focus_levels,
            status="capturing",
            metadata={
                "columns": cols,
                "rows": rows,
                "overlap": overlap,
                "stage_step": [x_stage_step, y_stage_step],
                "pixel_stride_mode": pixel_stride_mode,
                "pixel_stride": [pixel_stride_x, pixel_stride_y],
                "calibrated_pixel_stride": [pixel_stride_x, pixel_stride_y],
                "auto_pixel_stride": [auto_pixel_stride_x, auto_pixel_stride_y],
                "blend_width": blend_width,
                "source_frame_size": [source_width, source_height],
                "stored_frame_size": [view_width, view_height],
                "capture_max_width": capture_max_width if capture_max_width is not None else "original",
                "capture_scale": capture_scale,
                "xy_settle_ms": xy_settle_ms,
                "xy_post_settle_ms": xy_post_settle_ms,
                "z_settle_ms": z_settle_ms,
                "z_post_settle_ms": z_post_settle_ms,
                "discard_frames": discard_frames,
                "output_path": str(output_path),
                "preview_max_width": settings_snapshot["preview_max_width"],
                "settings_snapshot": settings_snapshot,
                "region_bounds": dict(stage_region or {}),
                "region_anchors": dict(self._digital_slide_region_anchor_points),
                "row_column_source": "manual" if self._digital_slide_rows_cols_manual else ("region" if stage_region else "manual"),
            },
        )
        try:
            store = DigitalSlideStore.create(output_path, manifest)
            store.close()
        except Exception as exc:
            QMessageBox.warning(self, "数字化切片", f"无法创建切片文件：\n{exc}")
            return
        writer = DigitalSlideWriteWorker(output_path, max_queue_size=3)
        writer.tileWritten.connect(self._on_digital_slide_tile_written, Qt.ConnectionType.QueuedConnection)
        writer.failed.connect(self._on_digital_slide_writer_failed, Qt.ConnectionType.QueuedConnection)
        writer.drained.connect(self._on_digital_slide_writer_drained, Qt.ConnectionType.QueuedConnection)
        writer.start()
        self._slide_acquisition_store = store
        self._slide_acquisition_settings = acquisition_settings
        self._slide_acquisition_writer = writer
        self._slide_acquisition_path = output_path
        self._slide_acquisition_document_path = str(output_path)
        initial_focus_index = max(0, len(focus_levels) // 2)
        self._slide_acquisition_metadata = {
            "digital_slide": {
                "working_path": str(output_path),
                "viewport_origin": [0, 0],
                "focus_index": initial_focus_index,
                "capture_scale": capture_scale,
                "source_frame_size": [source_width, source_height],
            }
        }
        self._slide_acquisition_plan = self._build_digital_slide_capture_plan(
            cols=cols,
            rows=rows,
            focus_levels=focus_levels,
            pixel_stride_x=pixel_stride_x,
            pixel_stride_y=pixel_stride_y,
            stage_region=stage_region,
            settings=acquisition_settings,
        )
        self._slide_acquisition_index = 0
        self._slide_acquisition_finishing = None
        self._slide_acquisition_discard_message = None
        self._slide_acquisition_pending_write = None
        self._slide_acquisition_viewport_size = (view_width, view_height)
        self._slide_acquisition_frame_marker = self._preview_frame_serial
        self._slide_acquisition_wait_started_at = 0.0
        self._slide_acquisition_settle_started_at = 0.0
        self._slide_acquisition_post_settle_started_at = 0.0
        self._slide_acquisition_timer_phase = "idle"
        self._slide_acquisition_move_ms = 0.0
        self._slide_acquisition_settle_ms = 0.0
        self._slide_acquisition_post_settle_ms = 0.0
        self._slide_acquisition_xy_moved = False
        self._slide_acquisition_z_moved = False
        self._slide_acquisition_xy_settle_wait_ms = xy_settle_ms
        self._slide_acquisition_z_settle_wait_ms = z_settle_ms
        self._slide_acquisition_xy_post_wait_ms = xy_post_settle_ms
        self._slide_acquisition_z_post_wait_ms = z_post_settle_ms
        self._slide_acquisition_settle_wait_ms = 0
        self._slide_acquisition_post_wait_ms = 0
        self._slide_acquisition_required_discard_frames = 0
        self._slide_acquisition_last_write_ms = 0.0
        self._slide_acquisition_started_at = perf_counter()
        self._slide_acquisition_initial_estimated_total_ms = self._estimate_digital_slide_total_ms(
            plan=self._slide_acquisition_plan,
            settings=acquisition_settings,
        )
        self._slide_acquisition_last_timing_summary = ""
        if self._digital_slide_start_button is not None:
            self._digital_slide_start_button.setEnabled(False)
        if self._digital_slide_stop_button is not None:
            self._digital_slide_stop_button.setEnabled(True)
        self._set_digital_slide_progress(f"开始采集，共 {len(self._slide_acquisition_plan)} 张")
        self._set_digital_slide_progress_value(0, len(self._slide_acquisition_plan))
        self._update_digital_slide_eta()
        self._set_digital_slide_timing("耗时: 等待第一步移动")
        self._sync_digital_slide_task_state()
        QTimer.singleShot(0, self._schedule_next_digital_slide_move)

    def _build_digital_slide_capture_plan(
        self,
        *,
        cols: int,
        rows: int,
        focus_levels: list[int],
        pixel_stride_x: int,
        pixel_stride_y: int,
        stage_region: dict[str, int] | None = None,
        settings: AppSettings | None = None,
    ) -> list[dict[str, int]]:
        active_settings = settings or self._digital_slide_effective_settings()
        x_stage_step = int(active_settings.digital_slide_x_stage_step)
        y_stage_step = int(active_settings.digital_slide_y_stage_step)

        def target_x_for_col(col: int) -> int:
            if stage_region is not None and {"left", "right"}.issubset(stage_region):
                if cols <= 1:
                    return int(stage_region["left"])
                return int(round(stage_region["left"] + ((stage_region["right"] - stage_region["left"]) * (col / max(1, cols - 1)))))
            return int(col * x_stage_step)

        def target_y_for_row(row: int) -> int:
            if stage_region is not None and {"top", "bottom"}.issubset(stage_region):
                if rows <= 1:
                    return int(stage_region["top"])
                return int(round(stage_region["top"] + ((stage_region["bottom"] - stage_region["top"]) * (row / max(1, rows - 1)))))
            return int(row * y_stage_step)

        plan: list[dict[str, int]] = []
        for row in range(rows):
            col_order = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
            for col in col_order:
                for z_index, focus_z in enumerate(focus_levels):
                    plan.append(
                        {
                            "col": col,
                            "row": row,
                            "z_index": z_index,
                            "global_x": col * pixel_stride_x,
                            "global_y": row * pixel_stride_y,
                            "stage_x": target_x_for_col(col),
                            "stage_y": target_y_for_row(row),
                            "focus_z": int(focus_z),
                        }
                    )
        return plan

    def _slide_acquisition_wait_summary(self) -> str:
        xy_settle = self._slide_acquisition_xy_settle_wait_ms if self._slide_acquisition_xy_moved else 0
        z_settle = self._slide_acquisition_z_settle_wait_ms if self._slide_acquisition_z_moved else 0
        xy_post = self._slide_acquisition_xy_post_wait_ms if self._slide_acquisition_xy_moved else 0
        z_post = self._slide_acquisition_z_post_wait_ms if self._slide_acquisition_z_moved else 0
        return f"XY停稳 {xy_settle} ms | Z停稳 {z_settle} ms | XY后等 {xy_post} ms | Z后等 {z_post} ms"

    def _schedule_next_digital_slide_move(self, *, delay_ms: int | None = None) -> None:
        if self._slide_acquisition_store is None:
            return
        if self._slide_acquisition_index >= len(self._slide_acquisition_plan):
            self._request_digital_slide_acquisition_finish(status="ready", message="数字化切片采集完成")
            return
        item = self._slide_acquisition_plan[self._slide_acquisition_index]
        try:
            move_started_at = perf_counter()
            current_x = int(self._slide_motion.relative_pos.get(AXIS_X, 0))
            current_y = int(self._slide_motion.relative_pos.get(AXIS_Y, 0))
            current_z = int(self._slide_motion.relative_pos.get(AXIS_Z, 0))
            target_x = int(item["stage_x"])
            target_y = int(item["stage_y"])
            target_z = int(item["focus_z"])
            self._slide_acquisition_xy_moved = current_x != target_x or current_y != target_y
            self._slide_acquisition_z_moved = current_z != target_z
            if current_x != target_x and not self._slide_motion.move_to(AXIS_X, target_x, label="自动采集 X"):
                self._fail_digital_slide_acquisition(f"移动失败：X 未能移动到 {target_x} steps")
                return
            if current_y != target_y and not self._slide_motion.move_to(AXIS_Y, target_y, label="自动采集 Y"):
                self._fail_digital_slide_acquisition(f"移动失败：Y 未能移动到 {target_y} steps")
                return
            if current_z != target_z and not self._slide_motion.move_to(AXIS_Z, target_z, label="自动采集 Z"):
                self._fail_digital_slide_acquisition(f"移动失败：Z 未能移动到 {target_z} steps")
                return
            self._slide_acquisition_move_ms = (perf_counter() - move_started_at) * 1000.0
        except Exception as exc:
            self._fail_digital_slide_acquisition(f"移动失败：{exc}")
            return
        acquisition_settings = self._digital_slide_effective_settings()
        self._slide_acquisition_xy_settle_wait_ms = int(acquisition_settings.digital_slide_xy_settle_ms)
        self._slide_acquisition_xy_post_wait_ms = int(acquisition_settings.digital_slide_xy_post_settle_ms)
        self._slide_acquisition_z_settle_wait_ms = int(acquisition_settings.digital_slide_z_settle_ms)
        self._slide_acquisition_z_post_wait_ms = int(acquisition_settings.digital_slide_z_post_settle_ms)
        xy_settle_wait_ms = self._slide_acquisition_xy_settle_wait_ms if self._slide_acquisition_xy_moved else 0
        z_settle_wait_ms = self._slide_acquisition_z_settle_wait_ms if self._slide_acquisition_z_moved else 0
        xy_post_wait_ms = self._slide_acquisition_xy_post_wait_ms if self._slide_acquisition_xy_moved else 0
        z_post_wait_ms = self._slide_acquisition_z_post_wait_ms if self._slide_acquisition_z_moved else 0
        self._slide_acquisition_settle_wait_ms = max(xy_settle_wait_ms, z_settle_wait_ms)
        if delay_ms is not None:
            self._slide_acquisition_settle_wait_ms = max(0, int(delay_ms))
        self._slide_acquisition_post_wait_ms = max(xy_post_wait_ms, z_post_wait_ms)
        self._slide_acquisition_settle_started_at = perf_counter()
        self._slide_acquisition_post_settle_started_at = 0.0
        self._slide_acquisition_wait_started_at = 0.0
        self._slide_acquisition_settle_ms = 0.0
        self._slide_acquisition_post_settle_ms = 0.0
        self._slide_acquisition_timer_phase = "settle"
        self._set_digital_slide_timing(
            f"耗时: 移动 {self._slide_acquisition_move_ms:.0f} ms | {self._slide_acquisition_wait_summary()}"
        )
        self._slide_acquisition_timer.start(max(0, self._slide_acquisition_settle_wait_ms))

    def _on_slide_acquisition_timer_timeout(self) -> None:
        if self._slide_acquisition_timer_phase == "settle":
            self._begin_digital_slide_post_settle_wait()
            return
        if self._slide_acquisition_timer_phase == "post_settle":
            self._begin_digital_slide_frame_wait()
            return
        self._capture_next_digital_slide_frame()

    def _begin_digital_slide_post_settle_wait(self) -> None:
        if self._slide_acquisition_store is None:
            return
        self._slide_acquisition_settle_ms = (
            (perf_counter() - self._slide_acquisition_settle_started_at) * 1000.0
            if self._slide_acquisition_settle_started_at > 0
            else 0.0
        )
        self._slide_acquisition_post_settle_started_at = perf_counter()
        self._slide_acquisition_timer_phase = "post_settle"
        self._set_digital_slide_timing(
            f"耗时: 移动 {self._slide_acquisition_move_ms:.0f} ms | "
            f"停稳 {self._slide_acquisition_settle_ms:.0f}/{self._slide_acquisition_settle_wait_ms} ms | "
            f"{self._slide_acquisition_wait_summary()}"
        )
        self._slide_acquisition_timer.start(max(0, self._slide_acquisition_post_wait_ms))

    def _begin_digital_slide_frame_wait(self) -> None:
        if self._slide_acquisition_store is None:
            return
        self._slide_acquisition_post_settle_ms = (
            (perf_counter() - self._slide_acquisition_post_settle_started_at) * 1000.0
            if self._slide_acquisition_post_settle_started_at > 0
            else 0.0
        )
        self._slide_acquisition_frame_marker = self._preview_frame_serial
        self._slide_acquisition_wait_started_at = perf_counter()
        self._slide_acquisition_required_discard_frames = (
            int(self._digital_slide_effective_settings().digital_slide_discard_frames)
        )
        self._slide_acquisition_timer_phase = "capture"
        self._set_digital_slide_timing(
            f"耗时: 移动 {self._slide_acquisition_move_ms:.0f} ms | "
            f"停稳 {self._slide_acquisition_settle_ms:.0f} ms | "
            f"后等 {self._slide_acquisition_post_settle_ms:.0f}/{self._slide_acquisition_post_wait_ms} ms | "
            f"丢弃 0/{self._slide_acquisition_required_discard_frames} 帧"
        )
        self._slide_acquisition_timer.start(0)

    def _capture_next_digital_slide_frame(self) -> None:
        store = self._slide_acquisition_store
        if store is None or self._slide_acquisition_index >= len(self._slide_acquisition_plan):
            return
        item = self._slide_acquisition_plan[self._slide_acquisition_index]
        wait_started_at = self._slide_acquisition_wait_started_at or perf_counter()
        frame_wait_ms = (perf_counter() - wait_started_at) * 1000.0
        new_frame_count = max(0, self._preview_frame_serial - self._slide_acquisition_frame_marker)
        required_discard_frames = max(0, int(self._slide_acquisition_required_discard_frames))
        if new_frame_count <= required_discard_frames and frame_wait_ms < 2000.0:
            self._set_digital_slide_timing(
                f"耗时: 移动 {self._slide_acquisition_move_ms:.0f} ms | "
                f"停稳 {self._slide_acquisition_settle_ms:.0f} ms | "
                f"后等 {self._slide_acquisition_post_settle_ms:.0f} ms | "
                f"丢弃 {min(new_frame_count, required_discard_frames)}/{required_discard_frames} 帧 | "
                f"等帧 {frame_wait_ms:.0f} ms"
            )
            self._slide_acquisition_timer.start(50)
            return
        if new_frame_count <= 0:
            self._fail_digital_slide_acquisition("采集失败：等待新视场图像超时")
            return
        frame = self._latest_preview_frame
        if frame is None or frame.isNull():
            self._fail_digital_slide_acquisition("采集失败：未获取到有效图像")
            return
        scale_started_at = perf_counter()
        scaled_frame, _capture_scale = self._scale_digital_slide_frame(frame)
        scale_ms = (perf_counter() - scale_started_at) * 1000.0
        expected_width, expected_height = self._slide_acquisition_viewport_size or (0, 0)
        if expected_width <= 0 or expected_height <= 0:
            manifest = store.read_manifest()
            expected_width, expected_height = manifest.viewport_width, manifest.viewport_height
        if scaled_frame.width() != expected_width or scaled_frame.height() != expected_height:
            self._fail_digital_slide_acquisition(
                f"采集帧尺寸变化：{scaled_frame.width()} x {scaled_frame.height()}，"
                f"预期 {expected_width} x {expected_height}"
            )
            return
        tile = DigitalSlideTile(
            z_index=item["z_index"],
            x=item["global_x"],
            y=item["global_y"],
            width=scaled_frame.width(),
            height=scaled_frame.height(),
            stage_x=item["stage_x"],
            stage_y=item["stage_y"],
            focus_z=item["focus_z"],
            status="ready",
        )
        self._enqueue_digital_slide_tile(
            tile,
            scaled_frame,
            {
                "settle_ms": self._slide_acquisition_settle_ms,
                "post_settle_ms": self._slide_acquisition_post_settle_ms,
                "discarded_frames": float(min(new_frame_count, required_discard_frames)),
                "frame_wait_ms": frame_wait_ms,
                "scale_ms": scale_ms,
            },
            item,
        )

    def _enqueue_digital_slide_tile(
        self,
        tile: DigitalSlideTile,
        image: QImage,
        timings: dict[str, float],
        item: dict[str, int],
    ) -> None:
        writer = self._slide_acquisition_writer
        if writer is None:
            self._fail_digital_slide_acquisition("写入切片失败：写入队列未启动")
            return
        enqueue_started_at = perf_counter()
        if not writer.enqueue(tile, image):
            self._slide_acquisition_pending_write = (tile, image, timings, item)
            self._set_digital_slide_progress("写入队列繁忙，等待后台写入释放空间...")
            self._update_digital_slide_eta()
            QTimer.singleShot(50, self._retry_pending_digital_slide_write)
            return
        timings["enqueue_ms"] = (perf_counter() - enqueue_started_at) * 1000.0
        self._slide_acquisition_pending_write = None
        self._slide_acquisition_index += 1
        self._set_digital_slide_progress(
            f"已排队 {self._slide_acquisition_index}/{len(self._slide_acquisition_plan)} 张 "
            f"(row {item['row'] + 1}, col {item['col'] + 1}, z {item['z_index'] + 1})"
        )
        self._set_digital_slide_progress_value(self._slide_acquisition_index, len(self._slide_acquisition_plan))
        self._set_digital_slide_timing(
            f"耗时: 移动 {self._slide_acquisition_move_ms:.0f} ms | "
            f"停稳 {timings.get('settle_ms', 0.0):.0f} ms | "
            f"后等 {timings.get('post_settle_ms', 0.0):.0f} ms | "
            f"丢弃 {timings.get('discarded_frames', 0.0):.0f} 帧 | "
            f"等帧 {timings.get('frame_wait_ms', 0.0):.0f} ms | "
            f"缩放 {timings.get('scale_ms', 0.0):.0f} ms | "
            f"入队 {timings.get('enqueue_ms', 0.0):.0f} ms | "
            f"写入 {self._slide_acquisition_last_write_ms:.0f} ms"
        )
        if self._slide_acquisition_finishing is not None:
            writer.finish()
            return
        self._schedule_next_digital_slide_move()

    def _retry_pending_digital_slide_write(self) -> None:
        pending = self._slide_acquisition_pending_write
        if (
            pending is None
            or self._slide_acquisition_store is None
            or self._slide_acquisition_discard_message is not None
        ):
            return
        self._slide_acquisition_pending_write = None
        tile, image, timings, item = pending
        self._enqueue_digital_slide_tile(tile, image, timings, item)

    def _on_digital_slide_tile_written(self, count: int, write_ms: float) -> None:
        self._slide_acquisition_last_write_ms = float(write_ms)
        self._set_digital_slide_timing(f"耗时: 写入 {write_ms:.0f} ms | 已写入 {count} 张")

    def _on_digital_slide_writer_failed(self, message: str) -> None:
        if self._slide_acquisition_store is None:
            return
        self._slide_acquisition_timer.stop()
        self._slide_acquisition_pending_write = None
        final_message = f"写入切片失败：{message}"
        if self._slide_acquisition_finishing is None and self._slide_acquisition_discard_message is None:
            self._slide_acquisition_finishing = ("failed", final_message)
            QMessageBox.warning(self, "数字化切片", final_message)
            self._set_digital_slide_progress(f"{final_message}，正在结束写入队列...")

    def _on_digital_slide_writer_drained(self) -> None:
        writer = self._slide_acquisition_writer
        if writer is not None:
            writer.wait(timeout_ms=100)
        self._slide_acquisition_writer = None
        if self._slide_acquisition_discard_message is not None:
            message = self._slide_acquisition_discard_message
            self._slide_acquisition_discard_message = None
            self._discard_digital_slide_acquisition(message=message)
            return
        finishing = self._slide_acquisition_finishing
        self._slide_acquisition_finishing = None
        if finishing is not None:
            status, message = finishing
            self._finish_digital_slide_acquisition(status=status, message=message)

    def _request_digital_slide_acquisition_finish(self, *, status: str, message: str) -> None:
        self._slide_acquisition_timer.stop()
        self._slide_acquisition_timer_phase = "idle"
        writer = self._slide_acquisition_writer
        if writer is not None and writer.is_running():
            self._slide_acquisition_finishing = (status, message)
            if self._slide_acquisition_pending_write is None:
                writer.finish()
            else:
                QTimer.singleShot(50, self._retry_pending_digital_slide_write)
            self._set_digital_slide_progress(f"{message}，等待写入队列完成...")
            return
        self._finish_digital_slide_acquisition(status=status, message=message)

    def _request_digital_slide_acquisition_discard(self, *, message: str) -> None:
        self._slide_acquisition_timer.stop()
        self._slide_acquisition_timer_phase = "idle"
        self._slide_acquisition_pending_write = None
        writer = self._slide_acquisition_writer
        if writer is not None and writer.is_running():
            self._slide_acquisition_discard_message = message
            writer.cancel()
            self._set_digital_slide_progress("正在停止写入队列并清理临时切片...")
            return
        self._discard_digital_slide_acquisition(message=message)

    def _stop_digital_slide_acquisition(self) -> None:
        if not self._slide_acquisition_active():
            return
        self._slide_acquisition_timer.stop()
        response = QMessageBox.question(
            self,
            "停止数字化切片",
            "是否保留当前已采集的图像？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if response == QMessageBox.StandardButton.Yes:
            self._request_digital_slide_acquisition_finish(
                status="interrupted",
                message="已保留中断的数字化切片",
            )
        else:
            self._request_digital_slide_acquisition_discard(message="已丢弃本次数字化切片采集")

    def _fail_digital_slide_acquisition(self, message: str) -> None:
        self._slide_acquisition_timer.stop()
        QMessageBox.warning(self, "数字化切片", message)
        self._request_digital_slide_acquisition_finish(status="failed", message=message)

    def _finish_digital_slide_acquisition(self, *, status: str, message: str) -> None:
        store = self._slide_acquisition_store
        path = self._slide_acquisition_path
        relative_path = self._slide_acquisition_document_path
        metadata = dict(self._slide_acquisition_metadata)
        elapsed_ms = (
            (perf_counter() - self._slide_acquisition_started_at) * 1000.0
            if self._slide_acquisition_started_at > 0
            else 0.0
        )
        self._slide_acquisition_store = None
        self._slide_acquisition_writer = None
        self._slide_acquisition_settings = None
        self._slide_acquisition_finishing = None
        self._slide_acquisition_discard_message = None
        self._slide_acquisition_pending_write = None
        self._slide_acquisition_viewport_size = None
        self._slide_acquisition_timer_phase = "idle"
        self._slide_acquisition_timer.stop()
        self._sync_digital_slide_task_state()
        if self._digital_slide_stop_button is not None:
            self._digital_slide_stop_button.setEnabled(False)
        if store is None or path is None:
            self._set_digital_slide_progress(message)
            return
        tile_count = store.tile_count()
        store.update_status(status)
        store.close()
        if tile_count <= 0:
            self._delete_slide_path(path)
            self._set_digital_slide_progress("没有采集到有效图像。")
            self._reset_digital_slide_eta()
            self._clear_digital_slide_output_path()
            return
        metadata.setdefault("digital_slide", {})
        if isinstance(metadata["digital_slide"], dict):
            metadata["digital_slide"]["capture_status"] = status
            metadata["digital_slide"]["working_path"] = str(path)
        self._add_digital_slide_document_from_path(
            path,
            document=None,
            source_type="filesystem",
            document_path=relative_path,
            metadata=metadata,
            tooltip=str(path),
        )
        self._slide_acquisition_index = tile_count
        self._set_digital_slide_progress(f"{message}，已生成 {tile_count} 张采集图像。")
        self._set_digital_slide_progress_value(tile_count, max(tile_count, len(self._slide_acquisition_plan)))
        self._set_digital_slide_finished_eta(elapsed_ms)
        if status in {"ready", "interrupted"}:
            self._clear_digital_slide_output_path()
        if status in {"ready", "interrupted"}:
            self._show_digital_slide_completion_dialog(
                status=status,
                message=message,
                path=path,
                tile_count=tile_count,
                elapsed_ms=elapsed_ms,
            )

    def _discard_digital_slide_acquisition(self, *, message: str) -> None:
        store = self._slide_acquisition_store
        path = self._slide_acquisition_path
        self._slide_acquisition_store = None
        self._slide_acquisition_writer = None
        self._slide_acquisition_settings = None
        self._slide_acquisition_finishing = None
        self._slide_acquisition_discard_message = None
        self._slide_acquisition_pending_write = None
        self._slide_acquisition_viewport_size = None
        self._slide_acquisition_timer_phase = "idle"
        self._slide_acquisition_timer.stop()
        if store is not None:
            store.close()
        if path is not None:
            self._delete_slide_path(path)
        self._sync_digital_slide_task_state()
        if self._digital_slide_stop_button is not None:
            self._digital_slide_stop_button.setEnabled(False)
        self._set_digital_slide_progress(message)
        self._set_digital_slide_progress_value(0, max(1, len(self._slide_acquisition_plan)))
        self._set_digital_slide_timing("耗时: -")
        self._reset_digital_slide_eta()
        self._clear_digital_slide_output_path()

    def _delete_slide_path(self, path: Path) -> None:
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            try:
                if candidate.exists():
                    candidate.unlink()
            except OSError:
                pass

    def _stop_digital_slide_writer(self, *, cancel: bool) -> None:
        writer = self._slide_acquisition_writer
        if writer is None:
            return
        if cancel:
            writer.cancel()
        else:
            writer.finish()
        writer.wait(timeout_ms=2000)
        self._slide_acquisition_writer = None
        if self._slide_acquisition_store is None:
            self._slide_acquisition_settings = None

    def _set_digital_slide_progress(self, message: str) -> None:
        if self._digital_slide_progress_label is not None:
            self._digital_slide_progress_label.setText(message)
        self.statusBar().showMessage(message, 4000)

    def _set_digital_slide_progress_value(self, value: int, total: int) -> None:
        if self._digital_slide_progress_bar is None:
            return
        total = max(1, int(total))
        value = max(0, min(int(value), total))
        self._digital_slide_progress_bar.setRange(0, total)
        self._digital_slide_progress_bar.setValue(value)
        self._digital_slide_progress_bar.setFormat(f"{value}/{total} 张")
        self._update_digital_slide_eta()

    def _estimate_digital_slide_total_ms(self, *, plan: list[dict[str, int]], settings: AppSettings) -> float:
        if not plan:
            return 0.0
        previous_x = int(self._slide_motion.relative_pos.get(AXIS_X, 0))
        previous_y = int(self._slide_motion.relative_pos.get(AXIS_Y, 0))
        previous_z = int(self._slide_motion.relative_pos.get(AXIS_Z, 0))
        total_ms = 0.0
        frame_wait_ms = 60.0 * (int(settings.digital_slide_discard_frames) + 1)
        per_tile_processing_ms = 80.0
        for item in plan:
            xy_moved = previous_x != int(item["stage_x"]) or previous_y != int(item["stage_y"])
            z_moved = previous_z != int(item["focus_z"])
            settle_ms = max(
                int(settings.digital_slide_xy_settle_ms) if xy_moved else 0,
                int(settings.digital_slide_z_settle_ms) if z_moved else 0,
            )
            post_ms = max(
                int(settings.digital_slide_xy_post_settle_ms) if xy_moved else 0,
                int(settings.digital_slide_z_post_settle_ms) if z_moved else 0,
            )
            total_ms += settle_ms + post_ms + frame_wait_ms + per_tile_processing_ms
            previous_x = int(item["stage_x"])
            previous_y = int(item["stage_y"])
            previous_z = int(item["focus_z"])
        return max(float(len(plan)) * 100.0, total_ms)

    def _update_digital_slide_eta(self) -> None:
        if (
            self._digital_slide_elapsed_label is None
            or self._digital_slide_remaining_label is None
            or self._digital_slide_eta_label is None
        ):
            return
        total = len(self._slide_acquisition_plan)
        completed = max(0, min(self._slide_acquisition_index, total))
        if total <= 0 or self._slide_acquisition_started_at <= 0:
            self._digital_slide_elapsed_label.setText("已用时: -")
            self._digital_slide_remaining_label.setText("预计剩余: -")
            self._digital_slide_eta_label.setText("预计完成: -")
            return
        elapsed_ms = max(0.0, (perf_counter() - self._slide_acquisition_started_at) * 1000.0)
        if completed > 0:
            estimated_total_ms = max(elapsed_ms, elapsed_ms * (total / max(1, completed)))
        else:
            estimated_total_ms = max(elapsed_ms, self._slide_acquisition_initial_estimated_total_ms)
        remaining_ms = max(0.0, estimated_total_ms - elapsed_ms)
        if not self._slide_acquisition_active() and completed >= total:
            remaining_ms = 0.0
        eta = datetime.now() + timedelta(milliseconds=remaining_ms)
        self._digital_slide_elapsed_label.setText(f"已用时: {self._format_duration_ms(elapsed_ms)}")
        self._digital_slide_remaining_label.setText(f"预计剩余: {self._format_duration_ms(remaining_ms)}")
        self._digital_slide_eta_label.setText(f"预计完成: {eta.strftime('%H:%M:%S')}")

    def _reset_digital_slide_eta(self) -> None:
        if self._digital_slide_elapsed_label is not None:
            self._digital_slide_elapsed_label.setText("已用时: -")
        if self._digital_slide_remaining_label is not None:
            self._digital_slide_remaining_label.setText("预计剩余: -")
        if self._digital_slide_eta_label is not None:
            self._digital_slide_eta_label.setText("预计完成: -")

    def _set_digital_slide_finished_eta(self, elapsed_ms: float) -> None:
        if self._digital_slide_elapsed_label is not None:
            self._digital_slide_elapsed_label.setText(f"已用时: {self._format_duration_ms(elapsed_ms)}")
        if self._digital_slide_remaining_label is not None:
            self._digital_slide_remaining_label.setText("预计剩余: 0秒")
        if self._digital_slide_eta_label is not None:
            self._digital_slide_eta_label.setText(f"完成时间: {datetime.now().strftime('%H:%M:%S')}")

    @staticmethod
    def _format_duration_ms(duration_ms: float) -> str:
        seconds = max(0, int(round(float(duration_ms) / 1000.0)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}时{minutes:02d}分{seconds:02d}秒"
        if minutes:
            return f"{minutes}分{seconds:02d}秒"
        return f"{seconds}秒"

    def _show_digital_slide_completion_dialog(
        self,
        *,
        status: str,
        message: str,
        path: Path,
        tile_count: int,
        elapsed_ms: float,
    ) -> None:
        title = "数字化切片采集完成" if status == "ready" else "数字化切片已保留"
        average_ms = elapsed_ms / max(1, tile_count)
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(message)
        box.setInformativeText(
            "\n".join(
                [
                    f"输出文件: {path}",
                    f"采集图像: {tile_count} 张",
                    f"总耗时: {self._format_duration_ms(elapsed_ms)}",
                    f"平均: {average_ms:.0f} ms/张",
                ]
            )
        )
        view_button = box.addButton("查看切片", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() == view_button:
            self.stop_live_preview()
            self._digital_slide_mode = False
            self._sync_digital_slide_mode_ui()
            if self._center_stack is not None:
                self._center_stack.setCurrentWidget(self.tab_widget)
            self._focus_digital_slide_path(path)

    def _focus_digital_slide_path(self, path: Path) -> None:
        target = str(path)
        try:
            target_resolved = str(path.resolve())
        except OSError:
            target_resolved = target
        for document in self.project.documents:
            candidates = {str(document.path)}
            try:
                candidates.add(str(Path(document.path).resolve()))
            except OSError:
                pass
            slide_meta = document.metadata.get("digital_slide") if isinstance(document.metadata, dict) else None
            if isinstance(slide_meta, dict):
                working_path = slide_meta.get("working_path")
                if working_path:
                    candidates.add(str(working_path))
            if target in candidates or target_resolved in candidates:
                self._set_current_document(document.id)
                return

    def _next_digital_slide_output_paths(self) -> tuple[str, Path]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        existing = {document.path for document in self.project.documents if document.is_project_asset()}
        counter = 1
        while True:
            relative = f"slides/slide_{stamp}_{counter:02d}{DIGITAL_SLIDE_SUFFIX}"
            if relative not in existing:
                break
            counter += 1
        if self._project_path is not None:
            output_path = project_slide_root(self._project_path) / Path(relative).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(tempfile.gettempdir()) / "fdm_digital_slides"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / Path(relative).name
        return relative, output_path

    def _apply_open_view_mode(self, canvas: DocumentCanvas | None) -> None:
        if canvas is None:
            return
        mode = self._app_settings.open_image_view_mode
        if mode == OpenImageViewMode.FIT:
            canvas.fit_to_view()
        elif mode == OpenImageViewMode.ACTUAL:
            canvas.actual_size()

    def _save_app_settings(self, *, context: str) -> bool:
        try:
            AppSettingsIO.save(self._app_settings)
        except OSError as exc:
            QMessageBox.warning(self, context, f"无法写入设置文件：\n{exc}")
            return False
        return True

    def _update_count_numbers_button(self) -> None:
        if self._count_numbers_button is None or self._measurement_tool_strip is None:
            return
        is_visible = self._tool_mode == "count" and not self._preview_active
        self._measurement_tool_strip.setCountContextVisible(is_visible)
        self._count_numbers_button.blockSignals(True)
        self._count_numbers_button.setChecked(bool(self._app_settings.show_count_numbers))
        self._count_numbers_button.setText("编号开" if self._app_settings.show_count_numbers else "编号关")
        has_document = bool(getattr(self, "_document_order", []))
        self._count_numbers_button.setEnabled(has_document and is_visible)
        self._count_numbers_button.blockSignals(False)
        if is_visible:
            self._measurement_tool_strip._refresh_context_visibility()  # noqa: SLF001
            layout = self._count_controls_widget.layout() if self._count_controls_widget is not None else None
            if layout is not None:
                layout.invalidate()
                layout.activate()
            if self._count_controls_widget is not None:
                self._count_controls_widget.updateGeometry()
                self._count_controls_widget.adjustSize()
            self._measurement_tool_strip.updateGeometry()

    def _toggle_count_numbers(self, checked: bool) -> None:
        self._app_settings.show_count_numbers = bool(checked)
        self._refresh_canvases_for_settings()
        self._save_app_settings(context="计数点编号")
        self._update_count_numbers_button()
        self.statusBar().showMessage("计数编号已显示" if checked else "计数编号已隐藏", 2000)

    def _restore_initial_window_geometry(self) -> None:
        geometry_token = str(self._app_settings.main_window_geometry or "").strip()
        restored = False
        if geometry_token:
            restored = self.restoreGeometry(QByteArray.fromBase64(geometry_token.encode("ascii")))
            if restored and not self._window_geometry_intersects_available_screen(self.frameGeometry()):
                restored = False
        if not restored:
            self._apply_default_window_geometry()
        if restored and self._app_settings.main_window_is_maximized:
            self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

    def _available_screens(self):
        return list(QGuiApplication.screens())

    def _window_geometry_intersects_available_screen(self, geometry) -> bool:
        for screen in self._available_screens():
            if geometry.intersects(screen.availableGeometry()):
                return True
        return False

    def _apply_default_window_geometry(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 860)
            return
        available = screen.availableGeometry()
        width = min(int(round(available.width() * 0.84)), 1600)
        height = min(int(round(available.height() * 0.84)), 1000)
        width = max(720, min(width, available.width()))
        height = max(520, min(height, available.height()))
        left = available.x() + max(0, (available.width() - width) // 2)
        top = available.y() + max(0, (available.height() - height) // 2)
        self.setGeometry(left, top, width, height)

    def _persist_window_geometry(self) -> None:
        self._app_settings.main_window_geometry = bytes(self.saveGeometry().toBase64()).decode("ascii")
        self._app_settings.main_window_is_maximized = bool(self.windowState() & Qt.WindowState.WindowMaximized)
        try:
            AppSettingsIO.save(self._app_settings)
        except OSError:
            return

    def _calibration_presets(self) -> list[CalibrationPreset]:
        return self._app_settings.calibration_presets

    def _selected_preset(self) -> tuple[int, CalibrationPreset] | None:
        preset_index = self.preset_combo.currentIndex()
        presets = self._calibration_presets()
        if preset_index < 0 or preset_index >= len(presets):
            return None
        return preset_index, presets[preset_index]

    def _project_snapshot(self) -> dict[str, object]:
        inherited_ids = sorted(
            document.id
            for document in self.project.documents
            if document.calibration is not None and document.calibration.mode == "project_default"
        )
        project_assets = sorted(
            (document.id, document.path)
            for document in self.project.documents
            if document.is_project_asset()
        )
        project_group_templates = [
            template.to_dict()
            for template in self.project.project_group_templates
            if normalize_group_label(template.label)
        ]
        return {
            "project_default_calibration": self.project.project_default_calibration.to_dict() if self.project.project_default_calibration else None,
            "project_default_document_ids": inherited_ids,
            "project_asset_documents": project_assets,
            "project_group_templates": project_group_templates,
        }

    def _mark_project_saved(self) -> None:
        self._project_clean_snapshot = self._project_snapshot()

    def _project_dirty(self) -> bool:
        return self._project_clean_snapshot is not None and self._project_snapshot() != self._project_clean_snapshot

    def _clone_preset(self, preset: CalibrationPreset, *, name: str | None = None) -> CalibrationPreset:
        return CalibrationPreset(
            name=preset.name if name is None else name,
            pixels_per_unit=preset.pixels_per_unit,
            unit=preset.unit,
            pixel_distance=preset.pixel_distance,
            actual_distance=preset.actual_distance,
            computed_pixels_per_unit=preset.computed_pixels_per_unit,
        )

    def _preset_content_equal(self, left: CalibrationPreset, right: CalibrationPreset, *, include_name: bool = True) -> bool:
        if include_name and left.name != right.name:
            return False
        return (
            abs(left.resolved_pixels_per_unit() - right.resolved_pixels_per_unit()) < 1e-9
            and left.unit == right.unit
            and left.pixel_distance == right.pixel_distance
            and left.actual_distance == right.actual_distance
        )

    def _find_matching_preset(self, calibration: Calibration | None) -> CalibrationPreset | None:
        if calibration is None:
            return None
        for preset in self._calibration_presets():
            if (
                preset.name == calibration.source_label
                and preset.unit == calibration.unit
                and abs(preset.resolved_pixels_per_unit() - calibration.pixels_per_unit) < 1e-9
            ):
                return preset
        for preset in self._calibration_presets():
            if (
                preset.unit == calibration.unit
                and abs(preset.resolved_pixels_per_unit() - calibration.pixels_per_unit) < 1e-9
            ):
                return preset
        return None

    def _default_preset_dialog_values(self, document: ImageDocument | None) -> tuple[float, float, str]:
        if document is None or document.calibration is None:
            return 100.0, 10.0, "um"
        calibration = document.calibration
        calibration_line = document.metadata.get("calibration_line")
        if calibration_line:
            line = calibration_line if isinstance(calibration_line, Line) else Line.from_dict(calibration_line)
            pixel_distance = max(line_length(line), 0.000001)
            actual_distance = calibration.px_to_unit(pixel_distance)
            if actual_distance > 0:
                return pixel_distance, actual_distance, calibration.unit
        preset = self._find_matching_preset(calibration)
        if preset is not None and preset.pixel_distance is not None and preset.actual_distance is not None:
            return preset.pixel_distance, preset.actual_distance, preset.unit
        return max(calibration.unit_to_px(1.0), 0.000001), 1.0, calibration.unit

    def _merge_imported_preset_batch(
        self,
        presets: list[CalibrationPreset],
        *,
        dedupe_by_content_only: bool,
    ) -> tuple[int, int, int]:
        plan = self._plan_imported_preset_batch(
            presets,
            dedupe_by_content_only=dedupe_by_content_only,
        )
        return self._apply_imported_preset_plan(plan)

    def _plan_imported_preset_batch(
        self,
        presets: list[CalibrationPreset],
        *,
        dedupe_by_content_only: bool,
    ) -> list[PresetImportPlanEntry]:
        plan: list[PresetImportPlanEntry] = []
        existing_presets = list(self._calibration_presets())
        for incoming_preset in presets:
            if any(self._preset_content_equal(item, incoming_preset, include_name=not dedupe_by_content_only) for item in existing_presets):
                plan.append(
                    PresetImportPlanEntry(
                        preset=self._clone_preset(incoming_preset),
                        action="skip",
                        final_name=incoming_preset.name,
                    )
                )
                continue
            candidate_name = incoming_preset.name
            action = "import"
            if any(item.name == candidate_name for item in existing_presets):
                candidate_name = f"{incoming_preset.name} (导入)"
                suffix = 2
                while any(item.name == candidate_name for item in existing_presets):
                    candidate_name = f"{incoming_preset.name} (导入 {suffix})"
                    suffix += 1
                action = "rename"
            planned_preset = self._clone_preset(incoming_preset, name=candidate_name)
            existing_presets.append(planned_preset)
            plan.append(
                PresetImportPlanEntry(
                    preset=planned_preset,
                    action=action,
                    final_name=candidate_name,
                )
            )
        return plan

    def _apply_imported_preset_plan(self, plan: list[PresetImportPlanEntry]) -> tuple[int, int, int]:
        imported_count = 0
        skipped_count = 0
        renamed_count = 0
        existing_presets = list(self._calibration_presets())
        for entry in plan:
            if entry.action == "skip":
                skipped_count += 1
                continue
            if entry.action == "rename":
                renamed_count += 1
            existing_presets.append(self._clone_preset(entry.preset, name=entry.final_name))
            imported_count += 1
        if imported_count:
            self._app_settings.calibration_presets = existing_presets
        return imported_count, skipped_count, renamed_count

    def _merge_legacy_calibration_presets(self, presets: list[CalibrationPreset]) -> int:
        imported_count, _, _ = self._merge_imported_preset_batch(
            presets,
            dedupe_by_content_only=False,
        )
        if imported_count:
            self._save_app_settings(context="导入旧项目预设")
            self._refresh_preset_combo()
        return imported_count

    def _format_calibration_mode(self, mode: str) -> str:
        return {
            "preset": "标定预设",
            "image_scale": "图内标定",
            "project_default": "项目统一比例尺",
            "none": "未标定",
        }.get(mode, mode or "未标定")

    def _set_document_project_default_calibration(self, document: ImageDocument) -> None:
        project_default = self.project.project_default_calibration
        if project_default is None:
            return
        document.calibration = project_default.clone()
        document.metadata.pop("calibration_line", None)
        document.recalculate_measurements()

    def _apply_project_default_calibration(self, calibration: Calibration, *, label: str) -> None:
        project_default = calibration.as_project_default()
        self.project.project_default_calibration = project_default.clone()
        for document in self.project.documents:
            before = document.snapshot_state()
            self._set_document_project_default_calibration(document)
            after = document.snapshot_state()
            if document.history is not None and before != after:
                document.history.push(label, before, after)
        self._update_ui_for_current_document()

    def _prompt_project_default_conflict(self, *, image_name: str, document_calibration: Calibration) -> bool:
        project_calibration = self.project.project_default_calibration
        if project_calibration is None:
            return False
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("标尺冲突")
        box.setText(f"{image_name} 同时存在图片标尺和项目统一比例尺。")
        box.setInformativeText(
            "图片标尺: "
            f"{document_calibration.source_label or self._format_calibration_mode(document_calibration.mode)}\n"
            "项目标尺: "
            f"{project_calibration.source_label or self._format_calibration_mode(project_calibration.mode)}"
        )
        image_button = box.addButton("使用图片标尺", QMessageBox.ButtonRole.AcceptRole)
        project_button = box.addButton("使用项目标尺", QMessageBox.ButtonRole.ActionRole)
        box.setDefaultButton(image_button)
        box.setEscapeButton(image_button)
        box.exec()
        return box.clickedButton() == project_button

    def _group_manager(self) -> GroupManager:
        return GroupManager(
            self.project,
            color_palette=self._color_palette,
            color_normalizer=lambda value, fallback="#1F7A8C": self._normalize_group_color(value, fallback=fallback),
        )

    def _project_group_template_for_label(self, label: str) -> ProjectGroupTemplate | None:
        return self._group_manager().project_group_template_for_label(label)

    def _next_group_color(self, document: ImageDocument) -> str:
        return self._group_manager().next_group_color(document)

    def _normalize_group_color(self, color_value: str, *, fallback: str = "#1F7A8C") -> str:
        color = QColor(str(color_value or "").strip())
        if color.isValid():
            return color.name()
        fallback_color = QColor(str(fallback or "").strip())
        if fallback_color.isValid():
            return fallback_color.name()
        return "#1f7a8c"

    def _ensure_project_group_template(self, *, label: str, color: str) -> bool:
        return self._group_manager().ensure_project_group_template(label=label, color=color)

    def _set_project_group_template_color(self, *, label: str, color: str) -> bool:
        return self._group_manager().set_project_group_template_color(label=label, color=color)

    def _apply_project_group_template_edit(self, *, original_label: str, target_label: str, color: str) -> bool:
        return self._group_manager().apply_project_group_template_edit(
            original_label=original_label,
            target_label=target_label,
            color=color,
        )

    def _apply_project_group_templates_to_document(
        self,
        document: ImageDocument,
        *,
        labels: set[str] | None = None,
    ) -> bool:
        return self._group_manager().apply_project_group_templates_to_document(document, labels=labels)

    def _sync_project_group_templates(self, *, label: str, labels: set[str] | None = None) -> bool:
        return self._group_manager().sync_project_group_templates(history_label=label, labels=labels)

    def _sync_project_group_template_edit_to_documents(
        self,
        *,
        original_label: str,
        target_label: str,
        color: str,
        history_label: str,
    ) -> bool:
        return self._group_manager().sync_project_group_template_edit_to_documents(
            original_label=original_label,
            target_label=target_label,
            color=color,
            history_label=history_label,
        )

    def _clear_group_suppression_when_present(self, document: ImageDocument, label: str) -> None:
        if document.find_group_by_label(label) is not None:
            document.unsuppress_project_group_label(label)

    def _ensure_document_named_group(
        self,
        document: ImageDocument,
        *,
        label: str,
        color: str,
        activate: bool,
        sync_color: bool = False,
    ) -> tuple[FiberGroup | None, bool]:
        return self._group_manager().ensure_document_named_group(
            document,
            label=label,
            color=color,
            activate=activate,
            sync_color=sync_color,
        )

    def _area_inference_group_color_for_label(self, label: str) -> str:
        return self._group_manager().area_inference_group_color_for_label(label)

    def _resolve_area_inference_group_colors(self, labels: list[str]) -> dict[str, str]:
        return self._group_manager().resolve_area_inference_group_colors(labels)

    def _resolved_area_inference_group_labels(
        self,
        model_name: str,
        *,
        update_project_group_templates: bool,
    ) -> list[str]:
        ordered_labels: list[str] = []
        seen_labels: set[str] = set()
        for template in self.project.project_group_templates:
            token = normalize_group_label(template.label)
            if not token or token in seen_labels:
                continue
            ordered_labels.append(token)
            seen_labels.add(token)
        for label in parse_area_model_labels(model_name):
            token = normalize_group_label(label)
            if not token or token in seen_labels:
                continue
            if update_project_group_templates and self._project_group_template_for_label(token) is None:
                self.project.project_group_templates.append(
                    ProjectGroupTemplate(
                        label=token,
                        color=self._area_inference_group_color_for_label(token),
                    )
                )
            ordered_labels.append(token)
            seen_labels.add(token)
        return ordered_labels

    def _area_inference_global_group_labels(self, model_name: str) -> list[str]:
        return self._resolved_area_inference_group_labels(
            model_name,
            update_project_group_templates=True,
        )

    def _normalize_document_groups_for_area_inference(
        self,
        document: ImageDocument,
        *,
        global_group_labels: list[str],
        recognized_labels: set[str],
        resolved_colors: dict[str, str] | None = None,
    ) -> bool:
        changed = False
        ordered_group_ids: list[str] = []
        seen_group_ids: set[str] = set()
        for label in global_group_labels:
            token = normalize_group_label(label)
            if not token:
                continue
            is_suppressed = document.is_project_group_label_suppressed(token)
            if is_suppressed and token not in recognized_labels:
                continue
            group, ensured_changed = self._ensure_document_named_group(
                document,
                label=token,
                color=(
                    resolved_colors[token]
                    if resolved_colors is not None and token in resolved_colors
                    else self._area_inference_group_color_for_label(token)
                ),
                activate=False,
                sync_color=self._project_group_template_for_label(token) is not None,
            )
            changed = ensured_changed or changed
            if group is None or group.id in seen_group_ids:
                continue
            ordered_group_ids.append(group.id)
            seen_group_ids.add(group.id)
        ordered_groups = [document.get_group(group_id) for group_id in ordered_group_ids]
        trailing_groups = [
            group
            for group in document.sorted_groups()
            if group.id not in seen_group_ids
        ]
        next_number = 1
        for group in [item for item in ordered_groups if item is not None] + trailing_groups:
            if group.number != next_number:
                group.number = next_number
                changed = True
            next_number += 1
        if changed:
            document.fiber_groups.sort(key=lambda group: group.number)
            document.rebuild_group_memberships()
        return changed

    def _sync_project_group_template_colors(
        self,
        color_by_label: dict[str, str],
        *,
        history_label: str,
    ) -> bool:
        labels_to_sync: set[str] = set()
        template_changed = False
        for raw_label, raw_color in color_by_label.items():
            token = normalize_group_label(raw_label)
            if not token or self._project_group_template_for_label(token) is None:
                continue
            template_changed = self._set_project_group_template_color(label=token, color=raw_color) or template_changed
            labels_to_sync.add(token)
        sync_changed = self._sync_project_group_templates(label=history_label, labels=labels_to_sync) if labels_to_sync else False
        return template_changed or sync_changed

    def _prompt_preset_apply_scope(self, preset: CalibrationPreset) -> str | None:
        if len(self.project.documents) <= 1:
            return "current"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("应用标定预设")
        box.setText(f"将预设“{preset.name}”应用到哪里？")
        project_button = box.addButton("项目所有图片", QMessageBox.ButtonRole.AcceptRole)
        current_button = box.addButton("当前图片", QMessageBox.ButtonRole.ActionRole)
        cancel_button = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(project_button)
        box.setEscapeButton(cancel_button)
        box.exec()
        if box.clickedButton() == project_button:
            return "project_all"
        if box.clickedButton() == current_button:
            return "current"
        return None

    def _build_cu_import_preview_text(
        self,
        records: list[object],
        plan: list[PresetImportPlanEntry],
        *,
        failures: list[str],
    ) -> str:
        lines = ["以下是本次解析到的 CU 标尺信息，请确认后再导入。"]
        for index, (record, entry) in enumerate(zip(records, plan), start=1):
            lines.append("")
            lines.append(f"{index}. {format_cu_scale_record_summary(record)}")
            if entry.action == "skip":
                lines.append("处理结果: 跳过，内容与现有预设重复")
            elif entry.action == "rename":
                lines.append(f"处理结果: 重命名导入为 {entry.final_name}")
            else:
                lines.append("处理结果: 直接导入")
        if failures:
            lines.append("")
            lines.append("以下文件解析失败，不会导入:")
            lines.extend(failures[:10])
        return "\n".join(lines)

    def _confirm_cu_import_preview(self, preview_text: str) -> bool:
        dialog = QDialog(self)
        dialog.setWindowTitle("确认导入CU标尺")
        dialog.resize(760, 520)
        layout = QVBoxLayout(dialog)
        description = QLabel("请核对下面的标尺名称和换算关系。确认后将写入全局标定预设。")
        description.setWordWrap(True)
        layout.addWidget(description)
        content = QPlainTextEdit()
        content.setReadOnly(True)
        content.setPlainText(preview_text)
        layout.addWidget(content, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_button is not None:
            ok_button.setText("确认导入")
        if cancel_button is not None:
            cancel_button.setText("取消")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        return dialog.exec() == QDialog.DialogCode.Accepted

    def open_images(self) -> None:
        self.stop_live_preview()
        paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", self.IMAGE_FILTER)
        if not paths:
            return
        self._open_image_requests(
            [(path, None) for path in paths],
            context_label="打开图片",
        )

    def open_folder(self) -> None:
        self.stop_live_preview()
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        image_paths = [
            item
            for item in sorted(Path(folder).iterdir(), key=lambda path: path.name.lower())
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_SUFFIXES
        ]
        if not image_paths:
            QMessageBox.information(self, "打开文件夹", "该文件夹中没有支持的图片。")
            return
        self._open_image_requests(
            [(str(image_path), None) for image_path in image_paths],
            context_label="打开文件夹",
        )

    def dragEnterEvent(self, event) -> None:
        if self._local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        if self._local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:
        paths = self._local_paths_from_mime(event.mimeData())
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self._open_dropped_paths(paths)

    def _local_paths_from_mime(self, mime_data) -> list[Path]:
        if mime_data is None or not mime_data.hasUrls():
            return []
        paths: list[Path] = []
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            local_path = Path(url.toLocalFile()).expanduser()
            if local_path.exists():
                paths.append(local_path)
        return paths

    def _folder_image_paths(self, folder: Path) -> list[Path]:
        return [
            item
            for item in sorted(folder.iterdir(), key=lambda path: path.name.lower())
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_SUFFIXES
        ]

    def _classify_dropped_paths(self, paths: list[Path]) -> tuple[list[Path], list[Path], int]:
        project_paths: list[Path] = []
        image_paths: list[Path] = []
        unsupported_count = 0
        seen_images: set[Path] = set()
        seen_projects: set[Path] = set()
        for path in paths:
            resolved = path.expanduser()
            if resolved.is_dir():
                folder_images = self._folder_image_paths(resolved)
                unsupported_count += 0 if folder_images else 1
                for image_path in folder_images:
                    image_key = image_path.resolve()
                    if image_key not in seen_images:
                        seen_images.add(image_key)
                        image_paths.append(image_path)
                continue
            if not resolved.is_file():
                unsupported_count += 1
                continue
            suffix = resolved.suffix.lower()
            if suffix == ".fdmproj":
                project_key = resolved.resolve()
                if project_key not in seen_projects:
                    seen_projects.add(project_key)
                    project_paths.append(resolved)
                continue
            if suffix in self.SUPPORTED_SUFFIXES:
                image_key = resolved.resolve()
                if image_key not in seen_images:
                    seen_images.add(image_key)
                    image_paths.append(resolved)
                continue
            unsupported_count += 1
        return project_paths, image_paths, unsupported_count

    def _open_dropped_paths(self, paths: list[Path]) -> None:
        settings_paths, remaining_paths = self._split_dropped_settings_paths(paths)
        if settings_paths:
            if len(settings_paths) > 1 or remaining_paths:
                QMessageBox.information(self, "导入设置", "settings.json 需单独拖入，一次只导入一个设置文件。")
                return
            self._import_settings_from_path(settings_paths[0])
            return
        project_paths, image_paths, unsupported_count = self._classify_dropped_paths(paths)
        if project_paths and (image_paths or len(project_paths) > 1):
            QMessageBox.information(self, "拖入打开", "项目文件需单独拖入，一次只打开一个项目。")
            return
        if project_paths:
            self._load_project_from_path(project_paths[0])
            return
        if image_paths:
            self.stop_live_preview()
            self._open_image_requests(
                [(str(path), None) for path in image_paths],
                context_label="拖入图片",
            )
            if unsupported_count:
                self.statusBar().showMessage(f"已忽略 {unsupported_count} 个不支持的拖入项目。", 5000)
            return
        if unsupported_count:
            QMessageBox.information(self, "拖入打开", "拖入内容中没有支持的图片或项目文件。")

    def _split_dropped_settings_paths(self, paths: list[Path]) -> tuple[list[Path], list[Path]]:
        settings_paths: list[Path] = []
        remaining_paths: list[Path] = []
        seen_settings: set[Path] = set()
        for path in paths:
            resolved = path.expanduser()
            if resolved.is_file() and resolved.name.lower() == "settings.json":
                settings_key = resolved.resolve()
                if settings_key not in seen_settings:
                    seen_settings.add(settings_key)
                    settings_paths.append(resolved)
                continue
            remaining_paths.append(path)
        return settings_paths, remaining_paths

    def _import_settings_from_path(self, path: Path) -> None:
        source_path = path.expanduser()
        target_path = settings_file_path()
        response = QMessageBox.question(
            self,
            "导入设置",
            f"是否使用拖入的 settings.json 覆盖当前软件设置？\n\n来源:\n{source_path}\n\n当前设置:\n{target_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        try:
            imported_settings, saved_path = AppSettingsIO.replace_with_file(source_path)
        except ValueError as exc:
            QMessageBox.warning(self, "导入设置", f"无法导入 settings.json：\n{exc}")
            return
        except OSError as exc:
            QMessageBox.warning(self, "导入设置", f"无法替换当前设置文件：\n{exc}")
            return
        self._apply_imported_app_settings(imported_settings)
        self.statusBar().showMessage(f"设置已从 {source_path.name} 导入", 5000)
        QMessageBox.information(self, "导入设置", f"设置已导入并覆盖当前 settings.json。\n\n位置:\n{saved_path}")

    def _normalize_image_path(self, path: str | Path) -> str:
        return str(Path(path).expanduser().resolve())

    def _prepare_image_load_requests(
        self,
        items: list[tuple[str, ImageDocument | None]],
    ) -> tuple[list[ImageLoadRequest], int, str | None]:
        open_documents: dict[str, ImageDocument] = {}
        for document in self.project.documents:
            if document.is_project_asset() and self._project_path is None:
                continue
            resolved_path = self._resolved_document_path(document)
            if not resolved_path:
                continue
            open_documents[self._normalize_image_path(resolved_path)] = document
        seen_paths: set[str] = set()
        requests: list[ImageLoadRequest] = []
        skipped_count = 0
        focus_document_id: str | None = None
        for raw_path, document in items:
            absolute_path = self._normalize_image_path(raw_path)
            existing_document = open_documents.get(absolute_path)
            if existing_document is not None:
                skipped_count += 1
                focus_document_id = existing_document.id
                continue
            if absolute_path in seen_paths:
                skipped_count += 1
                continue
            seen_paths.add(absolute_path)
            requests.append(ImageLoadRequest(path=absolute_path, document=document))
        return requests, skipped_count, focus_document_id

    def _open_image_requests(
        self,
        items: list[tuple[str, ImageDocument | None]],
        *,
        context_label: str,
        missing_paths: list[str] | None = None,
        repaired_paths: list[str] | None = None,
    ) -> None:
        if self.is_image_loading():
            QMessageBox.information(self, context_label, "当前仍有图片在加载，请稍候。")
            return
        requests, skipped_count, focus_document_id = self._prepare_image_load_requests(items)
        if not requests:
            if focus_document_id is not None:
                self._set_current_document(focus_document_id)
            self._show_batch_load_summary(
                BatchLoadState(
                    context_label=context_label,
                    total=0,
                    skipped_count=skipped_count,
                    loaded_count=0,
                    failed_count=0,
                    cancelled=False,
                    failures=[],
                    missing_paths=list(missing_paths or []),
                    repaired_paths=list(repaired_paths or []),
                )
            )
            return
        slide_requests = [request for request in requests if is_digital_slide_path(request.path)]
        if slide_requests:
            slide_state = BatchLoadState(
                context_label=context_label,
                total=len(slide_requests),
                skipped_count=0,
                failures=[],
                missing_paths=[],
                repaired_paths=[],
            )
            for request in slide_requests:
                self._load_digital_slide_request_sync(request, slide_state)
            requests = [request for request in requests if not is_digital_slide_path(request.path)]
            if not requests:
                slide_state.skipped_count = skipped_count
                self._show_batch_load_summary(slide_state)
                return
            if slide_state.loaded_count:
                self.statusBar().showMessage(f"已加载 {slide_state.loaded_count} 个数字化切片，继续加载普通图片。", 4000)
        if len(requests) == 1:
            state = BatchLoadState(
                context_label=context_label,
                total=1,
                skipped_count=skipped_count,
                failures=[],
                missing_paths=list(missing_paths or []),
                repaired_paths=list(repaired_paths or []),
            )
            self._load_single_request_sync(requests[0], state)
            self._show_batch_load_summary(state)
            return
        self._start_batch_image_load(
            requests,
            context_label=context_label,
            skipped_count=skipped_count,
            missing_paths=missing_paths,
            repaired_paths=repaired_paths,
        )

    def _load_single_request_sync(self, request: ImageLoadRequest, state: BatchLoadState) -> None:
        if is_digital_slide_path(request.path):
            self._load_digital_slide_request_sync(request, state)
            return
        reader = QImageReader(request.path)
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            reason = reader.errorString() or "无法读取图片"
            state.failed_count += 1
            if state.failures is not None:
                state.failures.append(f"{Path(request.path).name}: {reason}")
            return
        self._add_loaded_document(request, image)
        state.completed_count += 1
        state.loaded_count += 1

    def _load_digital_slide_request_sync(self, request: ImageLoadRequest, state: BatchLoadState) -> None:
        before_count = len(self.project.documents)
        self._add_digital_slide_document_from_path(
            request.path,
            document=request.document if isinstance(request.document, ImageDocument) else None,
            tooltip=request.path,
        )
        state.completed_count += 1
        if len(self.project.documents) > before_count:
            state.loaded_count += 1
        else:
            state.failed_count += 1
            if state.failures is not None:
                state.failures.append(f"{Path(request.path).name}: 无法读取数字化切片")

    def _start_batch_image_load(
        self,
        requests: list[ImageLoadRequest],
        *,
        context_label: str,
        skipped_count: int,
        missing_paths: list[str] | None = None,
        repaired_paths: list[str] | None = None,
    ) -> None:
        self.background_task_controller.start_batch_image_load(
            requests,
            context_label=context_label,
            skipped_count=skipped_count,
            missing_paths=missing_paths,
            repaired_paths=repaired_paths,
        )

    def _on_batch_load_progress(self, index: int, total: int, path: str) -> None:
        self.background_task_controller._on_batch_load_progress(index, total, path)

    def _on_batch_load_loaded(self, request: ImageLoadRequest, image: QImage) -> None:
        self.background_task_controller._on_batch_load_loaded(request, image)

    def _on_batch_load_failed(self, path: str, reason: str) -> None:
        self.background_task_controller._on_batch_load_failed(path, reason)

    def _on_batch_load_finished(self, cancelled: bool, loaded_count: int, skipped_count: int, failed_count: int) -> None:
        self.background_task_controller._on_batch_load_finished(cancelled, loaded_count, skipped_count, failed_count)

    def _show_batch_load_summary(self, state: BatchLoadState) -> None:
        summary_lines: list[str] = []
        if state.loaded_count:
            summary_lines.append(f"成功加载 {state.loaded_count} 张图片")
        if state.skipped_count:
            summary_lines.append(f"跳过重复图片 {state.skipped_count} 张")
        if state.failed_count:
            summary_lines.append(f"读取失败 {state.failed_count} 张")
        if state.missing_paths:
            summary_lines.append(f"未找到项目中的图片 {len(state.missing_paths)} 张")
        if state.repaired_paths:
            summary_lines.append(f"已自动修复 {len(state.repaired_paths)} 张图片路径")
        if state.cancelled:
            summary_lines.insert(0, "加载已取消，已保留已成功打开的图片。")
        if summary_lines:
            self.statusBar().showMessage("；".join(summary_lines), 6000)

        detail_lines = list(summary_lines)
        if state.failures:
            detail_lines.append("")
            detail_lines.append("失败明细:")
            detail_lines.extend(state.failures[:8])
        if state.missing_paths:
            detail_lines.append("")
            detail_lines.append("缺失图片:")
            detail_lines.extend(str(Path(path)) for path in state.missing_paths[:8])
        if state.repaired_paths:
            detail_lines.append("")
            detail_lines.append("原绝对路径已失效，已自动改用项目目录中的图片:")
            detail_lines.extend(state.repaired_paths[:8])

        has_warning = bool(state.failed_count or state.missing_paths or state.repaired_paths)
        if has_warning:
            QMessageBox.warning(self, state.context_label, "\n".join(detail_lines))
        elif state.cancelled or state.skipped_count:
            QMessageBox.information(self, state.context_label, "\n".join(detail_lines))

    def _start_area_inference_batch(
        self,
        requests: list[AreaInferenceRequest],
        *,
        model_name: str,
    ) -> None:
        self.background_task_controller.start_area_inference_batch(requests, model_name=model_name)

    def _ensure_prompt_segmentation_worker(self) -> None:
        self.background_task_controller.ensure_prompt_segmentation_worker()

    def _ensure_fiber_quick_geometry_worker(self) -> None:
        self.background_task_controller.ensure_fiber_quick_geometry_worker()

    def _ensure_fiber_quick_commit_geometry_worker(self) -> None:
        self.background_task_controller.ensure_fiber_quick_commit_geometry_worker()

    def _ensure_reference_instance_worker(self) -> None:
        self.background_task_controller.ensure_reference_instance_worker()

    def _clear_prompt_segmentation_cache(self) -> None:
        if self._prompt_seg_worker is None:
            pass
        else:
            try:
                self._prompt_seg_worker.clearRequested.emit()
            except Exception:
                pass
        if self._reference_instance_worker is not None:
            try:
                self._reference_instance_worker.clearRequested.emit()
            except Exception:
                pass
        for service in self._interactive_segmentation_services.values():
            try:
                service.clear_cache()
            except Exception:
                continue

    def _interactive_segmentation_service(self, model_variant: str):
        service = self._interactive_segmentation_services.get(model_variant)
        if service is None:
            service = create_interactive_segmentation_service(model_variant)
            self._interactive_segmentation_services[model_variant] = service
        return service

    def _on_area_inference_progress(self, index: int, total: int, path: str) -> None:
        self.background_task_controller._on_area_inference_progress(index, total, path)

    def _on_area_inference_succeeded(self, document_id: str, instances: object) -> None:
        self.background_task_controller._on_area_inference_succeeded(document_id, instances)

    def _on_area_inference_failed(self, document_id: str, path: str, reason: str) -> None:
        self.background_task_controller._on_area_inference_failed(document_id, path, reason)

    def _on_area_inference_finished(self, cancelled: bool, completed_count: int, failed_count: int) -> None:
        self.background_task_controller._on_area_inference_finished(cancelled, completed_count, failed_count)

    def _add_loaded_document(self, request: ImageLoadRequest, image: QImage) -> None:
        absolute_path = request.path
        target_document = request.document or ImageDocument(
            id=new_id("image"),
            path=absolute_path,
            image_size=(image.width(), image.height()),
        )
        target_document.image_size = (image.width(), image.height())
        if request.document is None:
            target_document.path = absolute_path
            target_document.source_type = "filesystem"
        elif target_document.uses_sidecar():
            target_document.sidecar_path = target_document.default_sidecar_path()
        target_document.initialize_runtime_state()
        if target_document.calibration is None:
            loaded_from_sidecar = target_document.uses_sidecar() and CalibrationSidecarIO.load_document(target_document)
            if self.project.project_default_calibration is not None:
                use_project_default = not loaded_from_sidecar
                if loaded_from_sidecar and target_document.calibration is not None:
                    use_project_default = self._prompt_project_default_conflict(
                        image_name=Path(absolute_path).name,
                        document_calibration=target_document.calibration,
                    )
                if use_project_default:
                    self._set_document_project_default_calibration(target_document)
                    target_document.mark_calibration_saved()
        else:
            target_document.mark_calibration_saved()
        self._apply_project_group_templates_to_document(target_document)
        target_document.mark_session_saved()

        self._mount_document(
            target_document,
            image,
            tooltip=absolute_path if request.document is None else self._document_tooltip(target_document),
        )

    def _mount_document(
        self,
        document: ImageDocument,
        image: QImage,
        *,
        tooltip: str,
    ) -> None:
        canvas = DocumentCanvas()
        canvas.set_document(document, image)
        canvas.set_settings(self._app_settings)
        canvas.set_tool_mode(self._tool_mode, overlay_kind=self._overlay_tool_kind)
        if is_magic_segment_tool_mode(self._tool_mode):
            self._sync_canvas_magic_subtract_input_mode(canvas)
        canvas.set_show_area_fill(self._show_area_fill)
        canvas.lineCommitted.connect(self._on_canvas_line_committed)
        canvas.measurementSelected.connect(self._on_canvas_measurement_selected)
        canvas.measurementEdited.connect(self._on_canvas_measurement_edited)
        canvas.pathSessionChanged.connect(self._on_canvas_path_session_changed)
        canvas.areaEditRejected.connect(self._on_canvas_area_edit_rejected)
        canvas.overlayCreateRequested.connect(self._on_canvas_overlay_create_requested)
        canvas.overlaySelected.connect(self._on_canvas_overlay_selected)
        canvas.overlayEdited.connect(self._on_canvas_overlay_edited)
        canvas.scaleAnchorPicked.connect(self._on_canvas_scale_anchor_picked)
        canvas.magicSegmentRequested.connect(self._on_canvas_magic_segment_requested)
        canvas.magicSegmentSessionChanged.connect(self._on_canvas_magic_segment_session_changed)

        self.project.documents.append(document)
        self._document_order.append(document.id)
        self._images[document.id] = image
        self._canvases[document.id] = canvas

        tab_index = self.tab_widget.addTab(canvas, self._document_display_name(document))
        self.tab_widget.setTabToolTip(tab_index, tooltip)
        list_item = QListWidgetItem(self._document_display_name(document))
        list_item.setData(Qt.ItemDataRole.UserRole, document.id)
        list_item.setToolTip(tooltip)
        self.image_list.addItem(list_item)
        self.tab_widget.setCurrentIndex(tab_index)
        self.image_list.setCurrentRow(tab_index)
        self._apply_open_view_mode(canvas)
        self._update_ui_for_current_document()

    def _add_digital_slide_document_from_path(
        self,
        path: str | Path,
        *,
        document: ImageDocument | None,
        source_type: str | None = None,
        document_path: str | None = None,
        metadata: dict[str, object] | None = None,
        tooltip: str | None = None,
    ) -> None:
        source_path = Path(path).expanduser().resolve()
        store = DigitalSlideStore(source_path)
        try:
            manifest = store.read_manifest()
        except Exception as exc:
            store.close()
            QMessageBox.warning(self, "打开数字化切片", f"无法读取数字化切片：\n{source_path}\n\n{exc}")
            return
        target_document = document or ImageDocument(
            id=new_id("slide"),
            path=document_path or str(source_path),
            image_size=(manifest.width, manifest.height),
            source_type=source_type or "filesystem",
            document_kind=DOCUMENT_KIND_DIGITAL_SLIDE,
            metadata=dict(metadata or {}),
        )
        target_document.document_kind = DOCUMENT_KIND_DIGITAL_SLIDE
        target_document.image_size = (manifest.width, manifest.height)
        if document is None:
            target_document.source_type = source_type or "filesystem"
            target_document.path = document_path or str(source_path)
            if target_document.source_type == "filesystem":
                target_document.absolute_path = str(source_path)
        if metadata:
            merged = dict(target_document.metadata)
            merged.update(metadata)
            target_document.metadata = merged
        slide_meta = dict(target_document.metadata.get("digital_slide", {})) if isinstance(target_document.metadata.get("digital_slide"), dict) else {}
        if target_document.source_type == "project_asset":
            slide_meta["working_path"] = str(source_path)
        target_document.metadata["digital_slide"] = slide_meta
        target_document.initialize_runtime_state()
        if target_document.calibration is None and self.project.project_default_calibration is not None:
            target_document.calibration = self._digital_slide_scaled_calibration(
                self.project.project_default_calibration,
                manifest,
            )
            target_document.metadata.pop("calibration_line", None)
            target_document.mark_calibration_saved()
        target_document.mark_session_saved()
        target_document.mark_calibration_saved()

        canvas = DigitalSlideCanvas()
        canvas.set_slide_document(target_document, store)
        canvas.set_settings(self._app_settings)
        canvas.set_tool_mode(self._tool_mode, overlay_kind=self._overlay_tool_kind)
        canvas.set_show_area_fill(self._show_area_fill)
        canvas.lineCommitted.connect(self._on_canvas_line_committed)
        canvas.measurementSelected.connect(self._on_canvas_measurement_selected)
        canvas.measurementEdited.connect(self._on_canvas_measurement_edited)
        canvas.pathSessionChanged.connect(self._on_canvas_path_session_changed)
        canvas.areaEditRejected.connect(self._on_canvas_area_edit_rejected)
        canvas.overlayCreateRequested.connect(self._on_canvas_overlay_create_requested)
        canvas.overlaySelected.connect(self._on_canvas_overlay_selected)
        canvas.overlayEdited.connect(self._on_canvas_overlay_edited)
        canvas.scaleAnchorPicked.connect(self._on_canvas_scale_anchor_picked)
        canvas.magicSegmentRequested.connect(self._on_canvas_magic_segment_requested)
        canvas.magicSegmentSessionChanged.connect(self._on_canvas_magic_segment_session_changed)
        canvas.viewportChanged.connect(self._on_digital_slide_viewport_changed)
        canvas.navigationModeChanged.connect(self._on_digital_slide_navigation_mode_changed)

        self.project.documents.append(target_document)
        self._document_order.append(target_document.id)
        self._canvases[target_document.id] = canvas
        self._slide_stores[target_document.id] = store

        tab_index = self.tab_widget.addTab(canvas, self._document_display_name(target_document))
        self.tab_widget.setTabToolTip(tab_index, tooltip or str(source_path))
        list_item = QListWidgetItem(self._document_display_name(target_document))
        list_item.setData(Qt.ItemDataRole.UserRole, target_document.id)
        list_item.setToolTip(tooltip or str(source_path))
        self.image_list.addItem(list_item)
        self.tab_widget.setCurrentIndex(tab_index)
        self.image_list.setCurrentRow(tab_index)
        canvas.schedule_initial_fit()
        self._update_ui_for_current_document()

    def _on_digital_slide_viewport_changed(self, x: int, y: int, focus_index: int) -> None:
        document = self.current_document()
        if document is None or not document.is_digital_slide():
            return
        self._update_image_resolution_label(document)

    def _current_digital_slide_canvas(self) -> DigitalSlideCanvas | None:
        document = self.current_document()
        if document is None or not document.is_digital_slide():
            return None
        canvas = self._canvases.get(document.id)
        return canvas if isinstance(canvas, DigitalSlideCanvas) else None

    def _set_current_digital_slide_smooth_navigation(self, checked: bool) -> None:
        canvas = self._current_digital_slide_canvas()
        if canvas is None:
            self._sync_digital_slide_navigation_action()
            return
        canvas.set_navigation_mode("smooth" if checked else "step")
        self._sync_digital_slide_navigation_action()
        self._update_image_resolution_label()

    def _on_digital_slide_navigation_mode_changed(self, mode: str) -> None:
        self._sync_digital_slide_navigation_action()
        self._update_image_resolution_label()

    def _sync_digital_slide_navigation_action(self) -> None:
        action = self.digital_slide_smooth_navigation_action
        if action is None:
            return
        canvas = self._current_digital_slide_canvas()
        enabled = canvas is not None and not self._preview_active
        action.blockSignals(True)
        action.setEnabled(enabled)
        action.setChecked(bool(canvas is not None and canvas.navigation_mode() == "smooth"))
        action.blockSignals(False)

    def save_project(self, path: str | None = None) -> bool:
        return self.project_session_controller.save_project(path)

    def load_project(self) -> None:
        self.project_session_controller.load_project()

    def _load_project_from_path(self, path: str | Path) -> None:
        self.project_session_controller.load_project_from_path(path)

    def export_results(self, preset: ExportSelection | None = None) -> None:
        self.export_controller.export_results(preset)

    def _prepare_raw_record_template_for_export(self, selection: ExportSelection) -> RawRecordTemplate | None:
        return self.export_controller.prepare_raw_record_template_for_export(selection)

    def _raw_record_template_for_path(self, template_path: str) -> RawRecordTemplate | None:
        return self.export_controller.raw_record_template_for_path(template_path)

    def fit_current_image(self) -> None:
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.fit_to_view()

    def actual_size_current_image(self) -> None:
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.actual_size()

    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(
            self._app_settings,
            document=self.current_document(),
            digital_slide_locked=self._slide_acquisition_active(),
            parent=self,
        )
        apply_button = dialog.button_box.button(QDialogButtonBox.StandardButton.Apply)
        if apply_button is not None:
            apply_button.clicked.connect(lambda: self._apply_settings_dialog(dialog, close_after=False))
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self._apply_settings_dialog(dialog, close_after=True)

    def _apply_settings_dialog(self, dialog: SettingsDialog, *, close_after: bool) -> None:
        new_settings = dialog.app_settings()
        if self._slide_acquisition_active():
            new_settings = self._settings_with_locked_digital_slide_values(new_settings)
        self._activate_app_settings(new_settings)
        refresh_widget_theme(dialog)
        self._save_app_settings(context="设置")

        document = self.current_document()
        if document is not None:
            group_colors = dialog.group_colors()
            if group_colors:
                local_group_colors: dict[str, str] = {}
                project_template_colors: dict[str, str] = {}
                for group in document.sorted_groups():
                    if group.id not in group_colors:
                        continue
                    target_color = self._normalize_group_color(group_colors[group.id], fallback=group.color)
                    label = normalize_group_label(group.label)
                    if label and self._project_group_template_for_label(label) is not None:
                        project_template_colors[label] = target_color
                    elif group.color != target_color:
                        local_group_colors[group.id] = target_color
                if local_group_colors:
                    def mutate_group_colors() -> None:
                        for group in document.sorted_groups():
                            if group.id in local_group_colors:
                                group.color = local_group_colors[group.id]

                    self._apply_document_change(document, "更新类别颜色", mutate_group_colors)
                if project_template_colors:
                    self._sync_project_group_template_colors(
                        project_template_colors,
                        history_label="同步项目全局类别颜色",
                    )

        should_pick_scale_anchor = dialog.wants_scale_anchor_pick()
        if should_pick_scale_anchor and self.current_document() is not None:
            if not close_after:
                dialog.accept()
                return
            self._begin_scale_anchor_pick(self.current_document())
        elif close_after:
            self.statusBar().showMessage("设置已更新", 3000)

    def _settings_with_locked_digital_slide_values(self, settings: AppSettings) -> AppSettings:
        preserved = settings.normalized_copy()
        current = self._app_settings.normalized_copy()
        for field_info in fields(AppSettings):
            if field_info.name.startswith("digital_slide_"):
                setattr(preserved, field_info.name, getattr(current, field_info.name))
        return preserved

    def _apply_imported_app_settings(self, settings: AppSettings) -> None:
        self._activate_app_settings(settings)

    def _activate_app_settings(self, settings: AppSettings) -> None:
        self._app_settings = settings
        self._apply_theme_mode()
        self._refresh_theme_sensitive_icons()
        if self._measurement_tool_strip is not None:
            self._measurement_tool_strip._apply_theme_styles()
        self._apply_tool_menu_stylesheets()
        self._magic_standard_add_roi_enabled = bool(settings.magic_segment_standard_add_roi_enabled)
        self._magic_standard_subtract_roi_enabled = bool(settings.magic_segment_standard_subtract_roi_enabled)
        self._magic_standard_subtract_input_mode = MagicSegmentSubtractInputMode.normalize(
            getattr(settings, "magic_segment_standard_subtract_input_mode", MagicSegmentSubtractInputMode.SMART)
        )
        self._fiber_quick_roi_enabled = bool(settings.fiber_quick_roi_enabled)
        if is_magic_segment_tool_mode(self._tool_mode):
            for canvas in self._canvases.values():
                self._sync_canvas_magic_subtract_input_mode(canvas)
        self._update_count_numbers_button()
        self._update_magic_segment_controls()
        if settings.selected_capture_device_id:
            self._capture_manager.set_selected_device(settings.selected_capture_device_id)
        self._apply_digital_slide_motion_settings()
        if self._digital_slide_xy_jog_step_spin is not None:
            self._digital_slide_xy_jog_step_spin.blockSignals(True)
            self._digital_slide_xy_jog_step_spin.setValue(settings.digital_slide_xy_jog_step)
            self._digital_slide_xy_jog_step_spin.blockSignals(False)
        if self._digital_slide_focus_jog_step_spin is not None:
            self._digital_slide_focus_jog_step_spin.blockSignals(True)
            self._digital_slide_focus_jog_step_spin.setValue(settings.digital_slide_z_jog_step)
            self._digital_slide_focus_jog_step_spin.blockSignals(False)
        if self._digital_slide_z_lower_edit is not None:
            self._digital_slide_z_lower_edit.blockSignals(True)
            self._set_optional_int_edit(self._digital_slide_z_lower_edit, settings.digital_slide_z_capture_lower)
            self._digital_slide_z_lower_edit.blockSignals(False)
        if self._digital_slide_z_upper_edit is not None:
            self._digital_slide_z_upper_edit.blockSignals(True)
            self._set_optional_int_edit(self._digital_slide_z_upper_edit, settings.digital_slide_z_capture_upper)
            self._digital_slide_z_upper_edit.blockSignals(False)
        if self._digital_slide_z_step_spin is not None:
            self._digital_slide_z_step_spin.blockSignals(True)
            self._digital_slide_z_step_spin.setValue(settings.digital_slide_z_capture_step)
            self._digital_slide_z_step_spin.blockSignals(False)
        self._sync_digital_slide_task_state()
        self._update_capture_device_ui()
        self._refresh_preset_combo()
        self._refresh_canvases_for_settings()

    def open_shortcut_help_dialog(self) -> None:
        dialog = ShortcutHelpDialog(self)
        dialog.exec()

    def _refresh_canvases_for_settings(self) -> None:
        for canvas in self._canvases.values():
            canvas.set_settings(self._app_settings)
            canvas.set_show_area_fill(self._show_area_fill)
        if self._preview_canvas is not None:
            self._preview_canvas.set_settings(self._app_settings)
            self._preview_canvas.set_show_area_fill(False)
        self._populate_group_list(self.current_document())
        self._populate_measurement_table(self.current_document())
        self._update_action_states()

    def _begin_scale_anchor_pick(self, document: ImageDocument | None) -> None:
        if document is None:
            return
        self._set_current_document(document.id)
        canvas = self._canvases.get(document.id)
        if canvas is None:
            return
        self.statusBar().showMessage("请在画布中单击比例尺起点位置。", 5000)
        canvas.begin_scale_anchor_pick()

    def import_cu_calibration_presets(self) -> None:
        if _CU_SCALE_IMPORT_ERROR is not None:
            QMessageBox.warning(self, "导入CU标尺", f"当前版本缺少 CU 标尺导入模块。\n{_CU_SCALE_IMPORT_ERROR}")
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "导入CU标尺", "", "CU 标尺 (*.scl)")
        if not paths:
            return
        parsed_records: list[object] = []
        failures: list[str] = []
        for path in paths:
            try:
                parsed_records.append(parse_cu_scale_file(path))
            except Exception as exc:
                failures.append(f"{Path(path).name}: {exc}")
        if not parsed_records:
            QMessageBox.warning(self, "导入CU标尺", "\n".join(failures) if failures else "没有可导入的 CU 标尺。")
            return
        parsed_presets = [record.preset for record in parsed_records]
        plan = self._plan_imported_preset_batch(parsed_presets, dedupe_by_content_only=True)
        preview_text = self._build_cu_import_preview_text(parsed_records, plan, failures=failures)
        if not self._confirm_cu_import_preview(preview_text):
            self.statusBar().showMessage("已取消导入 CU 标尺", 3000)
            return
        imported_count, skipped_count, renamed_count = self._apply_imported_preset_plan(plan)
        if imported_count:
            self._save_app_settings(context="导入CU标尺")
            self._refresh_preset_combo()
            self.statusBar().showMessage(f"已导入 {imported_count} 个 CU 标尺预设", 4000)
        if failures or skipped_count or renamed_count or imported_count == 0:
            summary_lines = [
                f"成功导入 {imported_count} 个预设",
                f"跳过重复 {skipped_count} 个",
                f"自动改名 {renamed_count} 个",
            ]
            if failures:
                summary_lines.append("")
                summary_lines.append("失败文件:")
                summary_lines.extend(failures[:8])
            QMessageBox.information(self, "导入CU标尺", "\n".join(summary_lines))

    def add_calibration_preset(self) -> None:
        pixel_distance, actual_distance, unit = self._default_preset_dialog_values(self.current_document())
        dialog = CalibrationPresetDialog(
            self,
            initial_pixel_distance=pixel_distance,
            initial_actual_distance=actual_distance,
            initial_unit=unit,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        name, pixel_distance, actual_distance, pixels_per_unit, unit = dialog.values()
        if not name:
            QMessageBox.warning(self, "新增预设", "预设名称不能为空。")
            return
        self._app_settings.calibration_presets.append(
            CalibrationPreset(
                name=name,
                pixels_per_unit=pixels_per_unit,
                unit=unit,
                pixel_distance=pixel_distance,
                actual_distance=actual_distance,
                computed_pixels_per_unit=pixels_per_unit,
            )
        )
        self._save_app_settings(context="新增预设")
        self._refresh_preset_combo(selected_name=name)
        self.statusBar().showMessage(f"已新增标定预设: {name}", 4000)

    def edit_selected_preset(self) -> None:
        selected = self._selected_preset()
        if selected is None:
            return
        preset_index, preset = selected
        initial_pixel_distance = preset.pixel_distance if preset.pixel_distance is not None else max(preset.resolved_pixels_per_unit(), 0.000001)
        initial_actual_distance = preset.actual_distance if preset.actual_distance is not None else 1.0
        dialog = CalibrationPresetDialog(
            self,
            title="编辑标定预设",
            initial_name=preset.name,
            initial_pixel_distance=initial_pixel_distance,
            initial_actual_distance=initial_actual_distance,
            initial_unit=preset.unit,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        name, pixel_distance, actual_distance, pixels_per_unit, unit = dialog.values()
        if not name:
            QMessageBox.warning(self, "编辑预设", "预设名称不能为空。")
            return
        self._app_settings.calibration_presets[preset_index] = CalibrationPreset(
            name=name,
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            pixel_distance=pixel_distance,
            actual_distance=actual_distance,
            computed_pixels_per_unit=pixels_per_unit,
        )
        self._save_app_settings(context="编辑预设")
        self._refresh_preset_combo(selected_name=name)
        self.statusBar().showMessage(f"已更新标定预设: {name}", 4000)

    def delete_selected_preset(self) -> None:
        selected = self._selected_preset()
        if selected is None:
            return
        preset_index, preset = selected
        result = QMessageBox.question(
            self,
            "删除预设",
            f"确定删除标定预设“{preset.name}”吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return
        del self._app_settings.calibration_presets[preset_index]
        self._save_app_settings(context="删除预设")
        self._refresh_preset_combo()
        self.statusBar().showMessage(f"已删除标定预设: {preset.name}", 4000)

    def apply_selected_preset(self) -> None:
        document = self.current_document()
        selected = self._selected_preset()
        if document is None or selected is None:
            return
        _, preset = selected
        scope = self._prompt_preset_apply_scope(preset)
        if scope is None:
            return

        if scope == "project_all":
            self._apply_project_default_calibration(preset.to_calibration(), label="应用项目统一标尺")
            self.statusBar().showMessage(f"已将标定预设应用到当前项目: {preset.name}", 4000)
            return

        def mutate() -> None:
            document.calibration = preset.to_calibration()
            document.recalculate_measurements()
            document.metadata.pop("calibration_line", None)

        self._apply_document_change(document, "应用标定预设", mutate, sync_sidecar=True)
        self.statusBar().showMessage(f"已应用标定预设: {preset.name}", 4000)

    def add_fiber_group(self) -> None:
        document = self.current_document()
        if document is None:
            return
        dialog = FiberGroupDialog(
            self,
            title="新增类别",
            initial_color=self._next_group_color(document),
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        label, selected_color, apply_to_project = dialog.values()
        token = normalize_group_label(label)
        if apply_to_project and not token:
            QMessageBox.warning(self, "新增类别", "应用到当前项目全局时，类别名称不能为空。")
            return

        template = self._project_group_template_for_label(token) if token else None
        existing_group = document.find_group_by_label(token) if token else None
        if template is not None:
            color = template.color
        elif apply_to_project:
            color = self._normalize_group_color(selected_color, fallback=self._next_group_color(document))
        elif existing_group is not None:
            color = existing_group.color
        else:
            color = self._normalize_group_color(selected_color, fallback=self._next_group_color(document))
        template_added = False
        if apply_to_project:
            template_added = self._ensure_project_group_template(label=token, color=color)

        def mutate() -> None:
            if token:
                self._ensure_document_named_group(
                    document,
                    label=token,
                    color=color,
                    activate=True,
                    sync_color=apply_to_project or template is not None,
                )
            else:
                group = document.create_group(
                    color=color,
                    label="",
                )
                document.set_active_group(group.id)

        current_changed = self._apply_document_change(document, "新增类别", mutate)
        sync_changed = self._sync_project_group_templates(label="同步项目全局类别") if apply_to_project else False
        self._update_ui_for_current_document()
        self._focus_current_canvas()

        if apply_to_project:
            if current_changed or sync_changed or template_added:
                self.statusBar().showMessage(f"已更新项目全局类别: {token}", 3000)
            else:
                self.statusBar().showMessage(f"项目全局类别已存在: {token}", 3000)
            return
        if token and not current_changed:
            self.statusBar().showMessage(f"同名类别已存在，已切换到现有类别: {token}", 3000)
            return
        self.statusBar().showMessage("已新增类别", 3000)

    def rename_active_group(self) -> None:
        document = self.current_document()
        if document is None:
            return
        group = document.get_group(document.active_group_id)
        if group is None:
            if document.active_group_id is None:
                self._rename_uncategorized_group(document)
            return
        dialog = FiberGroupDialog(
            self,
            title="编辑类别",
            initial_label=group.label,
            initial_color=group.color,
            apply_to_project_default=False,
            show_apply_to_project=True,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        label, selected_color, apply_to_project = dialog.values()
        target_label = normalize_group_label(label)
        if apply_to_project and not target_label:
            QMessageBox.warning(self, "编辑类别", "应用到当前项目全局时，类别名称不能为空。")
            return
        selected_qcolor = QColor(selected_color)
        target_color = selected_qcolor.name() if selected_qcolor.isValid() else selected_color.strip() or group.color
        current_label = normalize_group_label(group.label)
        current_qcolor = QColor(group.color)
        current_color = current_qcolor.name() if current_qcolor.isValid() else group.color
        if target_label == current_label and target_color == current_color and not apply_to_project:
            return
        merge_target = document.find_group_by_label(target_label) if target_label else None
        if merge_target is not None and merge_target.id != group.id:
            response = QMessageBox.question(
                self,
                "合并类别",
                f"当前图片中已存在类别“{target_label}”。\n\n确认后会将“{group.display_name()}”合并到该类别。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
            template_changed = (
                self._apply_project_group_template_edit(original_label=current_label, target_label=target_label, color=target_color)
                if apply_to_project
                else False
            )

            def mutate_merge() -> None:
                source = document.get_group(group.id)
                target = document.get_group(merge_target.id)
                if source is None or target is None:
                    return
                if apply_to_project or self._project_group_template_for_label(target_label) is not None:
                    target.color = target_color
                source_label = normalize_group_label(source.label)
                target_token = normalize_group_label(target.label)
                document.merge_group_into(source.id, target.id)
                if self._project_group_template_for_label(source_label) is not None:
                    document.suppress_project_group_label(source_label)
                if self._project_group_template_for_label(target_token) is not None:
                    document.unsuppress_project_group_label(target_token)

            changed = self._apply_document_change(document, "合并类别", mutate_merge)
            sync_changed = (
                self._sync_project_group_template_edit_to_documents(
                    original_label=current_label,
                    target_label=target_label,
                    color=target_color,
                    history_label="同步项目全局类别",
                )
                if apply_to_project
                else False
            )
            if apply_to_project:
                self._update_ui_for_current_document()
                self._focus_current_canvas()
                if changed or sync_changed or template_changed:
                    self.statusBar().showMessage(f"已更新项目全局类别: {target_label}", 3000)
                else:
                    self.statusBar().showMessage(f"项目全局类别已存在: {target_label}", 3000)
                return
            if changed:
                self.statusBar().showMessage("类别已合并", 3000)
            return

        template_changed = (
            self._apply_project_group_template_edit(original_label=current_label, target_label=target_label, color=target_color)
            if apply_to_project
            else False
        )

        def mutate_rename() -> None:
            target = document.get_group(group.id)
            if target is None:
                return
            original_label = normalize_group_label(target.label)
            target.label = target_label
            target.color = target_color
            if original_label and original_label != target_label and self._project_group_template_for_label(original_label) is not None:
                document.suppress_project_group_label(original_label)
            if self._project_group_template_for_label(target_label) is not None:
                document.unsuppress_project_group_label(target_label)

        changed = self._apply_document_change(document, "编辑类别", mutate_rename)
        sync_changed = (
            self._sync_project_group_template_edit_to_documents(
                original_label=current_label,
                target_label=target_label,
                color=target_color,
                history_label="同步项目全局类别",
            )
            if apply_to_project
            else False
        )
        if apply_to_project:
            self._update_ui_for_current_document()
            self._focus_current_canvas()
            if changed or sync_changed or template_changed:
                self.statusBar().showMessage(f"已更新项目全局类别: {target_label}", 3000)
            else:
                self.statusBar().showMessage(f"项目全局类别已存在: {target_label}", 3000)
            return
        if changed:
            self.statusBar().showMessage("类别已更新", 3000)

    def _rename_uncategorized_group(self, document: ImageDocument) -> None:
        dialog = FiberGroupDialog(
            self,
            title="编辑未分类",
            initial_label="",
            initial_color=self._app_settings.default_measurement_color,
            apply_to_project_default=False,
            show_apply_to_project=True,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        label, selected_color, apply_to_project = dialog.values()
        target_label = normalize_group_label(label)
        if apply_to_project and not target_label:
            QMessageBox.warning(self, "编辑未分类", "应用到当前项目全局时，类别名称不能为空。")
            return
        if not target_label:
            QMessageBox.warning(self, "编辑未分类", "类别名称不能为空。")
            return
        template = self._project_group_template_for_label(target_label)
        merge_target = document.find_group_by_label(target_label)
        if template is not None:
            target_color = template.color
        elif apply_to_project:
            target_color = self._normalize_group_color(selected_color, fallback=self._app_settings.default_measurement_color)
        elif merge_target is not None:
            target_color = merge_target.color
        else:
            target_color = self._normalize_group_color(selected_color, fallback=self._app_settings.default_measurement_color)
        template_added = False
        if merge_target is not None:
            response = QMessageBox.question(
                self,
                "合并类别",
                f"当前图片中已存在类别“{target_label}”。\n\n确认后会将未分类测量合并到该类别。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
            if apply_to_project:
                template_added = self._ensure_project_group_template(label=target_label, color=target_color)

            def mutate_merge() -> None:
                target = document.get_group(merge_target.id)
                if target is None:
                    return
                if apply_to_project or template is not None:
                    target.color = target_color
                document.move_uncategorized_measurements_to_group(target.id)
                target_token = normalize_group_label(target.label)
                if self._project_group_template_for_label(target_token) is not None:
                    document.unsuppress_project_group_label(target_token)

            changed = self._apply_document_change(document, "合并未分类", mutate_merge)
            sync_changed = (
                self._sync_project_group_templates(label="同步项目全局类别", labels={target_label})
                if apply_to_project
                else False
            )
            self._update_ui_for_current_document()
            self._focus_current_canvas()
            if apply_to_project:
                if changed or sync_changed or template_added:
                    self.statusBar().showMessage(f"已更新项目全局类别: {target_label}", 3000)
                else:
                    self.statusBar().showMessage(f"项目全局类别已存在: {target_label}", 3000)
                return
            if changed:
                self.statusBar().showMessage("未分类已合并到现有类别", 3000)
            return

        if apply_to_project:
            template_added = self._ensure_project_group_template(label=target_label, color=target_color)

        def mutate_create() -> None:
            group = document.create_group(color=target_color, label=target_label)
            document.move_uncategorized_measurements_to_group(group.id)
            document.set_active_group(group.id)
            if self._project_group_template_for_label(target_label) is not None:
                document.unsuppress_project_group_label(target_label)

        changed = self._apply_document_change(document, "编辑未分类", mutate_create)
        sync_changed = (
            self._sync_project_group_templates(label="同步项目全局类别", labels={target_label})
            if apply_to_project
            else False
        )
        self._update_ui_for_current_document()
        self._focus_current_canvas()
        if apply_to_project:
            if changed or sync_changed or template_added:
                self.statusBar().showMessage(f"已更新项目全局类别: {target_label}", 3000)
            else:
                self.statusBar().showMessage(f"项目全局类别已存在: {target_label}", 3000)
            return
        if changed:
            self.statusBar().showMessage("未分类已迁移到新类别", 3000)

    def delete_active_group(self) -> None:
        document = self.current_document()
        if document is None:
            return
        group = document.get_group(document.active_group_id)
        if group is not None:
            measurement_count = len(group.measurement_ids)
            message = f"确定删除类别“{group.display_name()}”吗？"
            if measurement_count:
                message += f"\n\n该类别下的 {measurement_count} 条测量会合并到未分类。"
            response = QMessageBox.question(
                self,
                "删除类别",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return

            def mutate() -> None:
                target = document.get_group(group.id)
                if target is None:
                    return
                template_label = normalize_group_label(target.label)
                document.remove_group_to_uncategorized(target.id)
                if self._project_group_template_for_label(template_label) is not None:
                    document.suppress_project_group_label(template_label)

            changed = self._apply_document_change(document, "删除类别", mutate)
            if changed:
                self.statusBar().showMessage("类别已删除", 3000)
            return

        if document.uncategorized_measurement_count() > 0:
            QMessageBox.information(
                self,
                "删除未分类",
                "未分类中仍有测量记录，请先将这些记录改到其它类别后再删除。",
            )
            return
        if not document.fiber_groups:
            QMessageBox.information(
                self,
                "删除未分类",
                "当前没有其它类别，未分类会作为默认入口保留。",
            )
            return
        response = QMessageBox.question(
            self,
            "删除未分类",
            "确定删除未分类入口吗？后续若再次出现未分类测量，它会自动恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return

        def mutate() -> None:
            document.hide_uncategorized_entry()

        self._apply_document_change(document, "删除未分类", mutate)
        self.statusBar().showMessage("未分类入口已隐藏", 3000)

    def delete_selected_measurement(self) -> None:
        document = self.current_document()
        if self._tool_mode == "calibration" or document is None:
            return
        selected_measurement_ids = self._selected_measurement_ids_from_table()
        if selected_measurement_ids:
            label = "删除测量" if len(selected_measurement_ids) == 1 else "批量删除测量"

            def mutate_rows() -> None:
                document.remove_measurements(selected_measurement_ids)

            self._apply_document_change(document, label, mutate_rows)
            self._focus_current_canvas()
            return
        if document.selected_overlay_id is not None:
            overlay_id = document.selected_overlay_id
            overlay = document.get_overlay_annotation(overlay_id)

            def mutate_overlay() -> None:
                document.remove_overlay_annotation(overlay_id)

            label = "删除标注"
            if overlay is not None and overlay.normalized_kind() == OverlayAnnotationKind.TEXT:
                label = "删除文字"
            self._apply_document_change(document, label, mutate_overlay)
            self._focus_current_canvas()
            return
        if document.view_state.selected_measurement_id is None:
            return
        measurement_id = document.view_state.selected_measurement_id

        def mutate() -> None:
            document.remove_measurement(measurement_id)

        self._apply_document_change(document, "删除测量", mutate)

    def delete_all_measurements(self) -> None:
        document = self.current_document()
        if document is None or self._tool_mode == "calibration":
            return
        if not any(item.measurements for item in self.project.documents):
            return
        selection = self._prompt_measurement_delete_options(
            title="删除全部测量",
            message="确认删除测量数据。你可以选择删除当前图片，或整个项目中的全部测量数据。",
        )
        if selection is None:
            return
        scope, _group_label = selection
        target_documents = [document] if scope == ExportScope.CURRENT else list(self.project.documents)
        removed_count = self._apply_documents_change(
            target_documents,
            "删除全部测量",
            lambda item: item.clear_measurements(),
        )
        if removed_count > 0:
            scope_label = "当前图片" if scope == ExportScope.CURRENT else "整个项目"
            self.statusBar().showMessage(f"已删除 {scope_label}中的 {removed_count} 条测量记录", 4000)
            self._focus_current_canvas()

    def delete_measurements_by_category(self) -> None:
        document = self.current_document()
        if document is None or self._tool_mode == "calibration":
            return
        group_labels = document.measurement_group_labels()
        if not group_labels:
            QMessageBox.information(self, "删除指定类别", "当前图片没有可删除的测量类别。")
            return
        selection = self._prompt_measurement_delete_options(
            title="删除指定类别",
            message="确认删除指定类别下的测量记录。类别定义、颜色模板和叠加标注不会被删除。",
            group_labels=group_labels,
        )
        if selection is None:
            return
        scope, group_label = selection
        if not group_label:
            return
        target_documents = [document] if scope == ExportScope.CURRENT else list(self.project.documents)
        removed_count = self._apply_documents_change(
            target_documents,
            "删除指定类别测量",
            lambda item, label=group_label: item.clear_measurements_by_group_label(label),
        )
        if removed_count > 0:
            scope_label = "当前图片" if scope == ExportScope.CURRENT else "整个项目"
            self.statusBar().showMessage(f"已删除“{group_label}”在{scope_label}中的 {removed_count} 条测量记录", 4000)
            self._focus_current_canvas()

    def _selected_measurement_ids_from_table(self) -> list[str]:
        selection_model = self.measurement_table.selectionModel()
        if selection_model is None:
            return []
        measurement_ids: list[str] = []
        seen: set[str] = set()
        for row_index in selection_model.selectedRows():
            item = self._measurement_id_item(row_index.row())
            if item is None:
                continue
            measurement_id = item.data(Qt.ItemDataRole.UserRole)
            if not measurement_id or measurement_id in seen:
                continue
            seen.add(measurement_id)
            measurement_ids.append(measurement_id)
        return measurement_ids

    def _prompt_measurement_delete_options(
        self,
        *,
        title: str,
        message: str,
        group_labels: list[str] | None = None,
    ) -> tuple[str, str | None] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(message, dialog))

        group_combo: QComboBox | None = None
        if group_labels:
            layout.addWidget(QLabel("删除类别：", dialog))
            group_combo = QComboBox(dialog)
            group_combo.addItems(group_labels)
            layout.addWidget(group_combo)

        layout.addWidget(QLabel("删除范围：", dialog))
        current_radio = QRadioButton("当前图片", dialog)
        current_radio.setChecked(True)
        project_radio = QRadioButton("整个项目", dialog)
        layout.addWidget(current_radio)
        layout.addWidget(project_radio)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog)
        ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("删除")
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != dialog.DialogCode.Accepted:
            return None
        scope = ExportScope.CURRENT if current_radio.isChecked() else ExportScope.ALL_OPEN
        return scope, group_combo.currentText() if group_combo is not None else None

    def run_area_auto_recognition(self) -> None:
        if not self.project.documents:
            QMessageBox.information(self, "面积自动识别", "请先打开图片。")
            return
        mappings = self._app_settings.area_model_mappings
        if not mappings:
            QMessageBox.information(self, "面积自动识别", "请先在设置中配置面积模型名称与权重文件映射。")
            return
        dialog = AreaAutoRecognitionDialog(
            mappings,
            allow_all_scope=len(self.project.documents) > 1,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        model_name, model_file, apply_all = dialog.values()
        if not model_name or not model_file:
            QMessageBox.warning(self, "面积自动识别", "请选择有效的模型配置。")
            return
        target_documents = self.project.documents if apply_all else ([self.current_document()] if self.current_document() else [])
        target_documents = [document for document in target_documents if document is not None]
        skipped_slides = [document for document in target_documents if document.is_digital_slide()]
        target_documents = [document for document in target_documents if not document.is_digital_slide()]
        if not target_documents:
            if skipped_slides:
                QMessageBox.information(self, "面积自动识别", "数字化切片文件会跳过面积自动识别。")
            return
        if skipped_slides:
            self.statusBar().showMessage(f"已跳过 {len(skipped_slides)} 个数字化切片文件。", 4000)
        requests = [
            AreaInferenceRequest(
                document_id=document.id,
                image_path=document.path,
                model_name=model_name,
                model_file=model_file,
            )
            for document in target_documents
        ]
        self._start_area_inference_batch(requests, model_name=model_name)

    def _apply_area_inference_result(
        self,
        document: ImageDocument,
        instances,
        *,
        global_group_labels: list[str] | None = None,
        model_name: str = "",
        update_project_group_templates: bool = True,
    ) -> None:
        if not instances:
            def clear_mutate() -> None:
                document.remove_auto_area_measurements()
                document.select_measurement(None)

            self._apply_document_change(document, "清除自动面积识别结果", clear_mutate)
            return

        if global_group_labels is None:
            resolved_global_group_labels = self._resolved_area_inference_group_labels(
                model_name,
                update_project_group_templates=update_project_group_templates,
            )
        elif global_group_labels:
            resolved_global_group_labels = list(global_group_labels)
        else:
            resolved_global_group_labels = self._resolved_area_inference_group_labels(
                model_name,
                update_project_group_templates=update_project_group_templates,
            )
            global_group_labels.extend(resolved_global_group_labels)
        inferred_label_order: list[str] = list(resolved_global_group_labels)
        for instance in instances:
            token = normalize_group_label(str(getattr(instance, "class_name", "")).strip() or UNCATEGORIZED_LABEL)
            if token and token not in inferred_label_order:
                inferred_label_order.append(token)
        resolved_colors = self._resolve_area_inference_group_colors(inferred_label_order)

        def mutate() -> None:
            document.remove_auto_area_measurements()
            recognized_labels = {
                normalize_group_label(str(getattr(instance, "class_name", "")).strip() or UNCATEGORIZED_LABEL)
                for instance in instances
            }
            self._normalize_document_groups_for_area_inference(
                document,
                global_group_labels=resolved_global_group_labels,
                recognized_labels=recognized_labels,
                resolved_colors=resolved_colors,
            )
            for instance in instances:
                class_name = str(instance.class_name).strip() or UNCATEGORIZED_LABEL
                group = document.ensure_group_for_label(
                    class_name,
                    color=resolved_colors.get(
                        normalize_group_label(class_name),
                        self._area_inference_group_color_for_label(class_name),
                    ),
                )
                measurement = Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id,
                    mode="auto_instance",
                    measurement_kind="area",
                    polygon_px=list(instance.polygon_px),
                    exact_area_px=float(instance.area_px),
                    confidence=float(instance.score),
                    status="auto_instance",
                )
                document.add_measurement(measurement)
            document.select_measurement(None)
            document.hide_uncategorized_entry()

        self._apply_document_change(document, "导入自动面积识别结果", mutate)

    def _on_canvas_magic_segment_requested(self, document_id: str, payload: object) -> None:
        canvas = self._canvases.get(document_id)
        document = self.project.get_document(document_id)
        if canvas is None or document is None or not isinstance(payload, dict):
            return
        image = self._images.get(document_id)
        request_id = int(payload.get("request_id", 0))
        tool_mode = str(payload.get("tool_mode", self._tool_mode) or self._tool_mode)
        if not is_magic_toolbar_tool_mode(tool_mode):
            tool_mode = MagicSegmentToolMode.STANDARD
        tool_label = self._magic_tool_label(tool_mode)
        if image is None or image.isNull():
            if is_reference_propagation_tool_mode(tool_mode):
                canvas.fail_reference_instance_result(request_id)
            elif is_fiber_quick_tool_mode(tool_mode):
                canvas.fail_fiber_quick_result(request_id)
            else:
                canvas.fail_magic_segment_result(request_id)
            self._update_magic_segment_controls()
            QMessageBox.warning(self, tool_label, "当前图片还未完成加载，暂时无法进行分割。")
            return
        cache_key = f"{document_id}:{int(image.cacheKey())}"
        requested_variant = self._app_settings.magic_segment_model_variant
        resolved_variant, _fallback_message = resolve_interactive_segmentation_backend(requested_variant)
        if not interactive_segmentation_models_ready(resolved_variant):
            if is_reference_propagation_tool_mode(tool_mode):
                canvas.fail_reference_instance_result(request_id)
            elif is_fiber_quick_tool_mode(tool_mode):
                canvas.fail_fiber_quick_result(request_id)
            else:
                canvas.fail_magic_segment_result(request_id)
            self._update_magic_segment_controls()
            model_paths = interactive_segmentation_model_paths(resolved_variant)
            runtime_root = interactive_segmentation_runtime_root(resolved_variant)
            if len(model_paths) == 1:
                missing_hint = f"请确认 {model_paths[0].as_posix()} 存在。"
            else:
                missing_hint = f"请确认 {runtime_root.as_posix()} 中存在所需模型文件。"
            QMessageBox.warning(
                self,
                tool_label,
                (
                    f"未找到 {interactive_segmentation_model_label(resolved_variant)} 模型文件，"
                    f"{missing_hint}"
                ),
            )
            return
        if is_reference_propagation_tool_mode(tool_mode):
            reference_box_payload = payload.get("reference_box")
            reference_box = None
            if isinstance(reference_box_payload, dict):
                start = reference_box_payload.get("start")
                end = reference_box_payload.get("end")
                if isinstance(start, Point) and isinstance(end, Point):
                    reference_box = (start, end)
            reference_measurement_id = str(payload.get("reference_measurement_id", "")).strip()
            reference_measurement = document.get_measurement(reference_measurement_id) if reference_measurement_id else None
            if reference_box is None and (
                reference_measurement is None
                or reference_measurement.measurement_kind != "area"
            ):
                canvas.fail_reference_instance_result(request_id)
                self._update_magic_segment_controls()
                self.statusBar().showMessage("同类扩选缺少有效参考实例", 4000)
                return
            self._ensure_reference_instance_worker()
            if self._reference_instance_worker is None:
                canvas.fail_reference_instance_result(request_id)
                self._update_magic_segment_controls()
                return
            self._reference_instance_worker.requested.emit(
                ReferenceInstancePropagationRequest(
                    document_id=document_id,
                    image=image,
                    cache_key=cache_key,
                    request_id=request_id,
                    model_variant=requested_variant,
                    reference_box=reference_box,
                    reference_polygon_px=list(reference_measurement.polygon_px) if reference_measurement is not None else [],
                    reference_area_rings_px=[list(ring) for ring in reference_measurement.area_rings_px] if reference_measurement is not None else [],
                )
            )
            self._update_magic_segment_controls()
            return
        positive_points = list(payload.get("positive_points", []))
        negative_points = list(payload.get("negative_points", []))
        active_stage = str(payload.get("active_stage", MagicSegmentOperationMode.ADD) or MagicSegmentOperationMode.ADD)
        roi_enabled = self._current_magic_roi_enabled(tool_mode, operation_mode=active_stage)
        roi_constraint_box = None
        small_object_enhancement_enabled = False
        small_object_workspace_box = None
        if (
            is_magic_segment_tool_mode(tool_mode)
            and active_stage == MagicSegmentOperationMode.SUBTRACT
            and roi_enabled
            and self._app_settings.magic_segment_restrict_subtract_roi_to_primary_bounds
        ):
            roi_constraint_box = canvas.magic_segment_primary_bounds()
            if roi_constraint_box is not None and any(
                not canvas.point_in_box(point, roi_constraint_box)
                for point in positive_points
            ):
                canvas.reject_magic_segment_subtract_points_outside_primary_bounds(request_id)
                self._update_magic_segment_controls()
                self.statusBar().showMessage("剔除模式 ROI 已限制在主体范围内，请在主体内部添加正采样点。", 5000)
                return
            if roi_constraint_box is not None and self._app_settings.magic_segment_small_object_subtract_enhancement_enabled:
                small_object_enhancement_enabled = True
                workspace_payload = payload.get("small_object_workspace_box")
                if isinstance(workspace_payload, (tuple, list)) and len(workspace_payload) == 4:
                    try:
                        small_object_workspace_box = tuple(int(round(float(value))) for value in workspace_payload)
                    except (TypeError, ValueError):
                        small_object_workspace_box = None
                if small_object_workspace_box is None:
                    small_object_workspace_box = canvas.magic_segment_small_object_workspace_box()
        if not positive_points:
            if is_fiber_quick_tool_mode(tool_mode):
                canvas.fail_fiber_quick_result(request_id)
            else:
                canvas.apply_magic_segment_result(request_id, None)
            self._update_magic_segment_controls()
            return
        if is_fiber_quick_tool_mode(tool_mode) and roi_enabled:
            pending_crop_box = initial_interactive_segmentation_crop_box(
                image_size=(image.height(), image.width()),
                positive_points=positive_points,
                negative_points=negative_points,
                tool_mode=tool_mode,
                roi_enabled=roi_enabled,
            )
            canvas.set_fiber_quick_pending_roi(request_id, pending_crop_box)
        if is_fiber_quick_tool_mode(tool_mode) and self._fiber_quick_geometry_worker is not None:
            self._fiber_quick_geometry_worker.cancel_document(document_id)
        self._ensure_prompt_segmentation_worker()
        if self._prompt_seg_worker is None:
            if is_fiber_quick_tool_mode(tool_mode):
                canvas.fail_fiber_quick_result(request_id)
            else:
                canvas.fail_magic_segment_result(request_id)
            self._update_magic_segment_controls()
            return
        self._prompt_request_tool_modes[(document_id, request_id)] = tool_mode
        self._prompt_seg_worker.register_request(document_id, request_id)
        self._prompt_seg_worker.requested.emit(
            PromptSegmentationRequest(
                document_id=document_id,
                image=image,
                cache_key=cache_key,
                request_id=request_id,
                positive_points=positive_points,
                negative_points=negative_points,
                tool_mode=tool_mode,
                active_stage=active_stage,
                model_variant=requested_variant,
                roi_enabled=roi_enabled,
                roi_constraint_box=roi_constraint_box,
                small_object_enhancement_enabled=small_object_enhancement_enabled,
                small_object_roi_area_threshold_px=self._app_settings.magic_segment_small_object_roi_area_threshold_px,
                small_object_workspace_box=small_object_workspace_box,
            )
        )
        self._update_magic_segment_controls()

    def _on_canvas_magic_segment_session_changed(self, document_id: str) -> None:
        current_document = self.current_document()
        if current_document is not None and current_document.id == document_id:
            self._update_magic_segment_controls()

    def _on_canvas_path_session_changed(self, document_id: str) -> None:
        current_document = self.current_document()
        if current_document is not None and current_document.id == document_id:
            self._update_path_drawing_controls()

    def _on_canvas_area_edit_rejected(self, document_id: str, reason: str) -> None:
        current_document = self.current_document()
        if current_document is not None and current_document.id == document_id:
            self.statusBar().showMessage(reason, 4000)
            self._update_path_drawing_controls()

    def _dispatch_pending_magic_segment_request(self, document_id: str, completed_request_id: int) -> bool:
        canvas = self._canvases.get(document_id)
        if canvas is None:
            return False
        payload = canvas.dequeue_pending_magic_segment_request(completed_request_id)
        if payload is None:
            return False
        self._on_canvas_magic_segment_requested(document_id, payload)
        return True

    def _dispatch_pending_fiber_quick_request(self, document_id: str, completed_request_id: int) -> bool:
        canvas = self._canvases.get(document_id)
        if canvas is None:
            return False
        payload = canvas.dequeue_pending_fiber_quick_request(completed_request_id)
        if payload is None:
            return False
        self._on_canvas_magic_segment_requested(document_id, payload)
        return True

    def _on_prompt_segmentation_succeeded(self, document_id: str, request_id: int, result: object) -> None:
        canvas = self._canvases.get(document_id)
        if canvas is None:
            return
        tool_mode = self._prompt_request_tool_modes.pop((document_id, request_id), None)
        if isinstance(result, PromptSegmentationResult):
            tool_mode = str(tool_mode or result.metadata.get("tool_mode", MagicSegmentToolMode.STANDARD) or MagicSegmentToolMode.STANDARD)
            if is_fiber_quick_tool_mode(tool_mode):
                debug_payload = {
                    "segmentation_roi_round": result.metadata.get("segmentation_roi_round"),
                    "segmentation_used_full_image": result.metadata.get("segmentation_used_full_image"),
                    "segmentation_crop_box": result.metadata.get("segmentation_crop_box"),
                    "component_area_px": result.metadata.get("component_area_px"),
                }
                if bool(debug_payload.get("segmentation_used_full_image")):
                    debug_payload.pop("segmentation_crop_box", None)
                apply_result = canvas.apply_fiber_quick_segmentation_result(
                    request_id,
                    mask=result.mask,
                    preview_polygon_points=result.polygon_px,
                    preview_area_rings_points=result.area_rings_px,
                    debug_payload=debug_payload,
                )
                if apply_result is None:
                    self._update_magic_segment_controls()
                    return
                if result.mask is None or len(result.polygon_px) < 3:
                    canvas.fail_fiber_quick_result(request_id, stage="segmentation")
                    self._dispatch_pending_fiber_quick_request(document_id, request_id)
                    self.statusBar().showMessage("快速测径失败: 未找到目标纤维区域。", 5000)
                    self._update_magic_segment_controls()
                    return
                if self._dispatch_pending_fiber_quick_request(document_id, request_id):
                    self.statusBar().showMessage("快速测径已更新分割结果，继续精修中。", 5000)
                    self._update_magic_segment_controls()
                    return
                self._ensure_fiber_quick_geometry_worker()
                if self._fiber_quick_geometry_worker is None:
                    canvas.fail_fiber_quick_result(request_id, stage="geometry")
                    self.statusBar().showMessage("快速测径失败: 几何线程初始化失败。", 5000)
                    self._update_magic_segment_controls()
                    return
                canvas.begin_fiber_quick_geometry(request_id)
                self._fiber_quick_geometry_request_ids.add((document_id, request_id))
                self._fiber_quick_geometry_worker.register_request(document_id, request_id)
                self._fiber_quick_geometry_worker.requested.emit(
                    FiberQuickGeometryRequest(
                        document_id=document_id,
                        request_id=request_id,
                        mask=result.mask,
                        preview_polygon_px=list(result.polygon_px),
                        preview_area_rings_px=[list(ring) for ring in result.area_rings_px],
                        positive_points=list(result.metadata.get("positive_points_px", []))
                        if isinstance(result.metadata.get("positive_points_px"), list)
                        else [],
                        negative_points=list(result.metadata.get("negative_points_px", []))
                        if isinstance(result.metadata.get("negative_points_px"), list)
                        else [],
                        edge_trim_enabled=bool(self._app_settings.fiber_quick_edge_trim_enabled),
                        line_extension_px=float(self._app_settings.fiber_quick_line_extension_px),
                        timeout_ms=DEFAULT_FIBER_QUICK_GEOMETRY_TIMEOUT_MS,
                    )
                )
                self.statusBar().showMessage("快速测径已完成分割，正在异步计算直径线。", 5000)
            else:
                small_object_used = bool(result.metadata.get("small_object_enhancement_used"))
                small_object_reject_reason = str(result.metadata.get("small_object_reject_reason", "") or "").strip()
                if small_object_used:
                    self._show_small_object_preview(canvas, result.metadata, result.polygon_px)
                else:
                    self._hide_small_object_preview()
                apply_result = canvas.apply_magic_segment_result(
                    request_id,
                    result.mask,
                    result.polygon_px,
                    result.area_rings_px,
                    {
                        "segmentation_roi_round": result.metadata.get("segmentation_roi_round"),
                        "segmentation_used_full_image": result.metadata.get("segmentation_used_full_image"),
                        "segmentation_crop_box": result.metadata.get("segmentation_crop_box"),
                        "component_area_px": result.metadata.get("component_area_px"),
                        "reason": result.metadata.get("reason"),
                        "small_object_enhancement_used": result.metadata.get("small_object_enhancement_used"),
                        "small_object_workspace_box": result.metadata.get("small_object_workspace_box"),
                        "small_object_scale": result.metadata.get("small_object_scale"),
                        "small_object_enhanced_size": result.metadata.get("small_object_enhanced_size"),
                        "small_object_reject_reason": result.metadata.get("small_object_reject_reason"),
                    },
                )
                if apply_result is None:
                    self._update_magic_segment_controls()
                    return
                if small_object_reject_reason:
                    self.statusBar().showMessage(
                        f"小目标增强未生成稳定剔除区域：{self._small_object_reject_label(small_object_reject_reason)}，请继续补点。",
                        5000,
                    )
                elif result.mask is None or not bool(apply_result.get("has_preview", False)):
                    self.statusBar().showMessage("魔棒分割失败: 未找到稳定目标区域。", 5000)
                self._dispatch_pending_magic_segment_request(document_id, request_id)
            if apply_result is None:
                self._update_magic_segment_controls()
                return
            fallback_message = str(result.metadata.get("model_fallback_message", "")).strip()
            if fallback_message:
                self.statusBar().showMessage(fallback_message, 5000)
        else:
            canvas.fail_magic_segment_result(request_id)
        self._update_magic_segment_controls()

    def _on_prompt_segmentation_failed(self, document_id: str, request_id: int, reason: str) -> None:
        canvas = self._canvases.get(document_id)
        if canvas is None:
            return
        tool_mode = self._prompt_request_tool_modes.pop((document_id, request_id), self._tool_mode)
        if is_fiber_quick_tool_mode(tool_mode):
            canvas.fail_fiber_quick_result(request_id, stage="segmentation")
            self._dispatch_pending_fiber_quick_request(document_id, request_id)
            self.statusBar().showMessage(f"快速测径失败: {reason}", 5000)
        else:
            canvas.fail_magic_segment_result(request_id)
            self._dispatch_pending_magic_segment_request(document_id, request_id)
            self.statusBar().showMessage(f"魔棒分割失败: {reason}", 5000)
        self._update_magic_segment_controls()

    def _on_fiber_quick_geometry_succeeded(self, document_id: str, request_id: int, result: object) -> None:
        self._fiber_quick_geometry_request_ids.discard((document_id, request_id))
        canvas = self._canvases.get(document_id)
        if canvas is None:
            return
        if hasattr(result, "line_px"):
            apply_result = canvas.apply_fiber_quick_geometry_result(
                request_id,
                preview_line=result.line_px if isinstance(result.line_px, Line) else None,
                confidence=float(getattr(result, "confidence", 0.0) or 0.0),
                debug_payload=dict(getattr(result, "debug_payload", {}))
                if isinstance(getattr(result, "debug_payload", {}), dict)
                else {},
            )
            if apply_result is None:
                self._update_magic_segment_controls()
                return
            if apply_result.get("has_preview"):
                if bool(canvas._fiber_quick.commit_pending):  # noqa: SLF001
                    commit_result = canvas.commit_fiber_quick_preview()
                    if bool(commit_result.get("committed", False)):
                        self.statusBar().showMessage("已创建快速测径线段", 4000)
                    else:
                        self.statusBar().showMessage("快速测径已生成代表线。按 Enter / F 确认。", 5000)
                else:
                    self.statusBar().showMessage("快速测径已生成代表线。按 Enter / F 确认。", 5000)
            else:
                self.statusBar().showMessage("快速测径失败: 未找到可靠直径线。", 5000)
        else:
            canvas.fail_fiber_quick_result(request_id, stage="geometry")
            self.statusBar().showMessage("快速测径失败: 未找到可靠直径线。", 5000)
        self._update_magic_segment_controls()

    def _on_fiber_quick_geometry_failed(self, document_id: str, request_id: int, reason: str) -> None:
        self._fiber_quick_geometry_request_ids.discard((document_id, request_id))
        canvas = self._canvases.get(document_id)
        if canvas is None:
            return
        if request_id != canvas._fiber_quick.request_id:  # noqa: SLF001
            return
        canvas.fail_fiber_quick_result(request_id, stage="geometry")
        self.statusBar().showMessage(f"快速测径失败: {reason}", 5000)
        self._update_magic_segment_controls()

    def _on_reference_instance_succeeded(self, document_id: str, request_id: int, result: object) -> None:
        canvas = self._canvases.get(document_id)
        if canvas is None:
            return
        if not isinstance(result, ReferenceInstancePropagationResult):
            canvas.fail_reference_instance_result(request_id)
            self._update_magic_segment_controls()
            return
        apply_result = canvas.apply_reference_instance_result(
            request_id,
            reference_polygon_points=result.reference_polygon_px,
            reference_area_rings_points=result.reference_area_rings_px,
            candidates=result.candidates,
        )
        if apply_result is None:
            self._update_magic_segment_controls()
            return
        fallback_message = str(result.metadata.get("model_fallback_message", "")).strip()
        if fallback_message:
            self.statusBar().showMessage(fallback_message, 5000)
        candidate_count = int(result.metadata.get("candidate_count", 0) or 0)
        if candidate_count > 0:
            self.statusBar().showMessage(f"已找到 {candidate_count} 个候选，按 Enter / F 加入当前类别。", 5000)
        else:
            self.statusBar().showMessage("未找到可用的同类候选。", 5000)
        self._update_magic_segment_controls()

    def _on_reference_instance_failed(self, document_id: str, request_id: int, reason: str) -> None:
        canvas = self._canvases.get(document_id)
        if canvas is None:
            return
        canvas.fail_reference_instance_result(request_id)
        self.statusBar().showMessage(f"同类扩选失败: {reason}", 5000)
        self._update_magic_segment_controls()

    def close_current_document(self) -> None:
        document = self.current_document()
        if document is None:
            return
        if not self._confirm_close_documents([document]):
            return
        self._remove_document(document.id)

    def close_all_documents(self) -> None:
        if not self.project.documents:
            return
        if not self._confirm_close_documents(self.project.documents):
            return
        self._reset_workspace()
        self._update_ui_for_current_document()

    def undo_current_document(self) -> None:
        document = self.current_document()
        if document is None or document.history is None or not document.history.undo(document):
            return
        if document.calibration is not None and (document.calibration.mode == "project_default" or not document.uses_sidecar()):
            document.mark_calibration_saved()
        else:
            CalibrationSidecarIO.save_document(document)
        self._update_ui_for_current_document()

    def redo_current_document(self) -> None:
        document = self.current_document()
        if document is None or document.history is None or not document.history.redo(document):
            return
        if document.calibration is not None and (document.calibration.mode == "project_default" or not document.uses_sidecar()):
            document.mark_calibration_saved()
        else:
            CalibrationSidecarIO.save_document(document)
        self._update_ui_for_current_document()

    def _confirm_close_documents(self, documents: list[ImageDocument]) -> bool:
        dirty_documents = [document for document in documents if self._document_has_unsaved_project_changes(document)]
        has_project_dirty = self._project_dirty()
        if not dirty_documents and not has_project_dirty:
            return True
        message_parts: list[str] = []
        if dirty_documents:
            if len(dirty_documents) == 1 and len(documents) == 1:
                message_parts.append(f"{Path(dirty_documents[0].path).name} 有未保存的项目改动。")
            else:
                message_parts.append(f"共有 {len(dirty_documents)} 张图片存在未保存的项目改动。")
        if has_project_dirty:
            message_parts.append("当前项目的统一比例尺、项目内图片、全局类别或继承关系有未保存改动。")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("未保存的改动")
        box.setText("\n".join(message_parts))
        save_button = box.addButton("保存", QMessageBox.ButtonRole.AcceptRole)
        discard_button = box.addButton("放弃", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked == cancel_button:
            return False
        if clicked == save_button:
            return self.save_project()
        return clicked == discard_button

    def _reset_workspace(self) -> None:
        self.stop_live_preview()
        self._hide_small_object_preview()
        self._clear_prompt_segmentation_cache()
        self.project = ProjectState.empty()
        self._project_path = None
        self._pending_project_load_snapshot = False
        self._document_order.clear()
        self._images.clear()
        for store in self._slide_stores.values():
            store.close()
        self._slide_stores.clear()
        self._canvases.clear()
        self.image_list.clear()
        self.tab_widget.clear()
        self._mark_project_saved()

    def _remove_document(self, document_id: str) -> None:
        if document_id not in self._document_order:
            return
        index = self._document_order.index(document_id)
        self._document_order.pop(index)
        self.project.documents = [document for document in self.project.documents if document.id != document_id]
        self._images.pop(document_id, None)
        store = self._slide_stores.pop(document_id, None)
        if store is not None:
            store.close()
        self._canvases.pop(document_id, None)
        self.tab_widget.removeTab(index)
        item = self.image_list.takeItem(index)
        del item
        self._clear_prompt_segmentation_cache()
        self._update_ui_for_current_document()

    def _apply_document_change(
        self,
        document: ImageDocument,
        label: str,
        mutator,
        *,
        sync_sidecar: bool = False,
    ) -> bool:
        before = document.snapshot_state()
        mutator()
        document.rebuild_group_memberships()
        document.mark_measurement_geometry_changed()
        document.refresh_dirty_flags()
        after = document.snapshot_state()
        changed = before != after
        if changed and document.history is not None:
            document.history.push(label, before, after)
        if sync_sidecar and document.uses_sidecar():
            CalibrationSidecarIO.save_document(document)
        elif sync_sidecar:
            document.mark_calibration_saved()
        self._update_ui_for_current_document()
        return changed

    def _append_new_measurement(self, document: ImageDocument, measurement: Measurement, *, label: str) -> None:
        previous_selected_measurement_id = document.view_state.selected_measurement_id
        previous_selected_overlay_id = document.selected_overlay_id
        measurement_index = len(document.measurements)
        document.insert_measurement_incremental(measurement)
        if document.history is not None:
            document.history.push_add_measurement(
                label,
                measurement_payload=measurement.to_dict(),
                index=measurement_index,
                previous_selected_measurement_id=previous_selected_measurement_id,
                previous_selected_overlay_id=previous_selected_overlay_id,
            )
        self._refresh_measurement_append_ui(document, measurement)

    def _refresh_measurement_append_ui(self, document: ImageDocument, measurement: Measurement) -> None:
        if document is not self.current_document():
            self._update_action_states()
            return
        self._refresh_group_list_counts(document)
        self._append_measurement_table_row(document, measurement)
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.update()
        self._update_action_states()

    def _apply_documents_change(
        self,
        documents: list[ImageDocument],
        label: str,
        mutator,
    ) -> int:
        total_removed = 0
        changed_any = False
        for document in documents:
            before = document.snapshot_state()
            removed_count = mutator(document)
            total_removed += int(removed_count or 0)
            document.rebuild_group_memberships()
            document.mark_measurement_geometry_changed()
            document.refresh_dirty_flags()
            after = document.snapshot_state()
            changed = before != after
            if changed and document.history is not None:
                document.history.push(label, before, after)
            changed_any = changed_any or changed
        if changed_any:
            self._update_ui_for_current_document()
        return total_removed

    def _on_tab_changed(self, index: int) -> None:
        if index < 0:
            return
        self.image_list.setCurrentRow(index)
        current_document = self.current_document()
        self._clear_magic_segment_sessions(except_document_id=current_document.id if current_document is not None else None)
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.set_tool_mode(self._tool_mode, overlay_kind=self._overlay_tool_kind)
            if is_magic_segment_tool_mode(self._tool_mode):
                self._sync_canvas_magic_subtract_input_mode(canvas)
            if current_document is None or not current_document.is_digital_slide():
                self._apply_open_view_mode(canvas)
            elif isinstance(canvas, DigitalSlideCanvas):
                canvas.schedule_initial_fit()
        self._update_ui_for_current_document()

    def _on_image_list_changed(self, row: int) -> None:
        if row >= 0 and row != self.tab_widget.currentIndex():
            self.tab_widget.setCurrentIndex(row)

    def _set_current_document(self, document_id: str) -> None:
        if document_id in self._document_order:
            index = self._document_order.index(document_id)
            self.tab_widget.setCurrentIndex(index)
            self.image_list.setCurrentRow(index)

    def _on_canvas_line_committed(self, document_id: str, mode: str, payload: object) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return
        if mode == "calibration":
            if isinstance(payload, Line):
                self._apply_calibration_line(document, payload)
            self._focus_current_canvas()
            return

        group = document.get_group(document.active_group_id)
        snap_result: SnapResult | None = None
        if mode == "snap":
            if not isinstance(payload, Line):
                self._focus_current_canvas()
                return
            snap_line = payload
            snap_offset = Point(0.0, 0.0)
            if document.is_digital_slide():
                snap_context = self._digital_slide_snap_context(document, payload)
                if snap_context is None:
                    self.statusBar().showMessage("当前数字化切片视场无法用于边缘吸附。", 4000)
                    self._focus_current_canvas()
                    return
                image, snap_line, snap_offset = snap_context
            else:
                image = self._images.get(document.id)
            if image is None or image.isNull():
                self.statusBar().showMessage("当前图片还未完成加载，暂时无法进行边缘吸附。", 4000)
                self._focus_current_canvas()
                return
            try:
                local_result = self.snap_service.snap_measurement(image, snap_line)
                if document.is_digital_slide():
                    snap_result = self._translate_digital_slide_snap_result(local_result, payload, snap_offset)
                else:
                    snap_result = local_result
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"边缘吸附失败: {exc}", 5000)
                self._focus_current_canvas()
                return

        if isinstance(payload, dict) and payload.get("measurement_kind") == "area":
            measurement = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=group.id if group else None,
                mode=mode,
                measurement_kind="area",
                polygon_px=list(payload.get("polygon_px", [])),
                area_rings_px=[list(ring) for ring in payload.get("area_rings_px", [])],
                exact_area_px=float(payload["exact_area_px"]) if payload.get("exact_area_px") is not None else None,
                confidence=1.0,
                status="manual" if mode != "auto_instance" else "auto_instance",
            )
            if mode == "magic_segment":
                ensure_measurement_display_geometry(measurement)
        elif mode == "continuous_manual" and isinstance(payload, dict):
            measurement = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=group.id if group else None,
                mode="continuous_manual",
                measurement_kind="polyline",
                polyline_px=list(payload.get("polyline_px", [])),
                confidence=1.0,
                status="continuous_manual",
            )
        elif mode == "count" and isinstance(payload, dict):
            measurement = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=group.id if group else None,
                mode="count",
                measurement_kind="count",
                point_px=payload.get("point_px"),
                confidence=1.0,
                status="count",
            )
        elif mode == "manual" and isinstance(payload, Line):
            measurement = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=group.id if group else None,
                mode="manual",
                line_px=payload,
                confidence=1.0,
                status="manual",
            )
        elif mode == "snap" and isinstance(payload, Line) and snap_result is not None:
            measurement = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=group.id if group else None,
                mode="snap",
                line_px=snap_result.original_line,
                snapped_line_px=snap_result.snapped_line,
                confidence=snap_result.confidence,
                status=snap_result.status,
                debug_payload=dict(snap_result.debug_payload),
            )
        elif mode == "fiber_quick" and isinstance(payload, dict) and isinstance(payload.get("line_px"), Line):
            measurement = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=group.id if group else None,
                mode="fiber_quick",
                line_px=payload["line_px"],
                confidence=float(payload.get("confidence", 0.0)),
                status=str(payload.get("status", "fiber_quick") or "fiber_quick"),
                debug_payload=dict(payload.get("debug_payload", {})),
            )
        else:
            self._focus_current_canvas()
            return

        self._append_new_measurement(document, measurement, label="新增测量")
        if snap_result is not None:
            self.statusBar().showMessage(self._edge_snap_status_message(snap_result), 4000)
        else:
            self.statusBar().showMessage("已新增测量", 2500)
        self._focus_current_canvas()

    def _digital_slide_snap_context(self, document: ImageDocument, line: Line) -> tuple[QImage, Line, Point] | None:
        canvas = self._canvases.get(document.id)
        store = self._slide_stores.get(document.id)
        if not isinstance(canvas, DigitalSlideCanvas) or store is None:
            return None
        width, height = document.image_size
        margin = max(12, int(getattr(self.snap_service, "profile_half_width_px", 2)) + 8)
        min_x = int(math.floor(min(line.start.x, line.end.x))) - margin
        min_y = int(math.floor(min(line.start.y, line.end.y))) - margin
        max_x = int(math.ceil(max(line.start.x, line.end.x))) + margin + 1
        max_y = int(math.ceil(max(line.start.y, line.end.y))) + margin + 1
        x = max(0, min(min_x, max(0, width - 1)))
        y = max(0, min(min_y, max(0, height - 1)))
        roi_width = max(1, min(width - x, max_x - x))
        roi_height = max(1, min(height - y, max_y - y))
        try:
            manifest = store.read_manifest()
            metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
            blend_width = int(metadata.get("blend_width", 0) or 0)
            image = store.render_viewport(
                x=x,
                y=y,
                width=roi_width,
                height=roi_height,
                z_index=canvas.focus_index(),
                blend_width=blend_width,
            )
        except Exception:
            return None
        if image.isNull():
            return None
        offset = Point(float(x), float(y))
        local_line = Line(
            start=Point(line.start.x - offset.x, line.start.y - offset.y),
            end=Point(line.end.x - offset.x, line.end.y - offset.y),
        )
        return image, local_line, offset

    def _translate_digital_slide_snap_result(self, result: SnapResult, original_line: Line, offset: Point) -> SnapResult:
        def translate_line(line: Line | None) -> Line | None:
            if line is None:
                return None
            return Line(
                start=Point(line.start.x + offset.x, line.start.y + offset.y),
                end=Point(line.end.x + offset.x, line.end.y + offset.y),
            )

        debug_payload = dict(result.debug_payload)
        debug_payload["digital_slide_roi_offset"] = (offset.x, offset.y)
        return SnapResult(
            status=result.status,
            original_line=original_line,
            snapped_line=translate_line(result.snapped_line),
            diameter_px=result.diameter_px,
            confidence=result.confidence,
            debug_payload=debug_payload,
        )

    def _on_canvas_measurement_selected(self, document_id: str, measurement_id: str | None) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return
        document.select_measurement(measurement_id or None)
        self._sync_measurement_table_selection(document)
        self._update_action_states()
        self._focus_current_canvas()

    def _on_canvas_measurement_edited(self, document_id: str, measurement_id: str, payload: object) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return

        def mutate() -> None:
            measurement = document.get_measurement(measurement_id)
            if measurement is None:
                return
            if isinstance(payload, dict) and payload.get("measurement_kind") == "area":
                measurement.polygon_px = list(payload.get("polygon_px", []))
                measurement.area_rings_px = [list(ring) for ring in payload.get("area_rings_px", [])]
                measurement.exact_area_px = float(payload["exact_area_px"]) if payload.get("exact_area_px") is not None else None
                measurement.measurement_kind = "area"
                payload_mode = payload.get("mode")
                if isinstance(payload_mode, str) and payload_mode:
                    measurement.mode = payload_mode
                invalidate_measurement_display_geometry(measurement)
            elif isinstance(payload, Line):
                measurement.snapped_line_px = payload
            else:
                return
            measurement.status = "edited"
            measurement.recalculate(document.calibration)
            if measurement.measurement_kind == "area" and measurement.mode == "magic_segment":
                ensure_measurement_display_geometry(measurement)
            document.select_measurement(measurement.id)

        self._apply_document_change(document, "编辑测量线", mutate)
        self._focus_current_canvas()

    def _on_canvas_overlay_create_requested(self, document_id: str, payload: object) -> None:
        document = self.project.get_document(document_id)
        if document is None or not isinstance(payload, dict):
            return
        kind = str(payload.get("kind", OverlayAnnotationKind.TEXT))
        if kind == OverlayAnnotationKind.TEXT:
            anchor = payload.get("anchor_px")
            if not isinstance(anchor, Point):
                self._focus_current_canvas()
                return
            content, ok = QInputDialog.getMultiLineText(self, "新增文字", "文字内容")
            if not ok:
                self._focus_current_canvas()
                return
            content = content.strip()
            if not content:
                self._focus_current_canvas()
                return

            def mutate_text() -> None:
                document.add_overlay_annotation(
                    OverlayAnnotation(
                        id=new_id("overlay"),
                        image_id=document.id,
                        kind=OverlayAnnotationKind.TEXT,
                        content=content,
                        anchor_px=anchor,
                    )
                )

            self._apply_document_change(document, "新增文字", mutate_text)
            self.statusBar().showMessage("已新增文字", 2500)
            self._focus_current_canvas()
            return
        start_point = payload.get("start_px")
        end_point = payload.get("end_px")
        if not isinstance(start_point, Point) or not isinstance(end_point, Point):
            self._focus_current_canvas()
            return

        def mutate_shape() -> None:
            document.add_overlay_annotation(
                OverlayAnnotation(
                    id=new_id("overlay"),
                    image_id=document.id,
                    kind=kind,
                    start_px=start_point,
                    end_px=end_point,
                )
            )

        self._apply_document_change(document, "新增标注", mutate_shape)
        self.statusBar().showMessage(f"已新增{self._overlay_tool_label(kind)}标注", 2500)
        self._focus_current_canvas()

    def _on_canvas_overlay_selected(self, document_id: str, overlay_id: str | None) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return
        document.select_overlay_annotation(overlay_id or None)
        if overlay_id:
            document.select_measurement(None)
        self._sync_measurement_table_selection(document)
        self._update_action_states()
        self._focus_current_canvas()

    def _on_canvas_overlay_edited(self, document_id: str, overlay_id: str, payload: object) -> None:
        document = self.project.get_document(document_id)
        if document is None or not isinstance(payload, OverlayAnnotation):
            return

        def mutate() -> None:
            current = document.get_overlay_annotation(overlay_id)
            if current is None:
                return
            document.replace_overlay_annotation(
                overlay_id,
                payload.clone(id=current.id, image_id=current.image_id, created_at=current.created_at),
            )

        label = "编辑标注"
        current = document.get_overlay_annotation(overlay_id)
        if current is not None and current.normalized_kind() == OverlayAnnotationKind.TEXT:
            label = "移动文字"
        self._apply_document_change(document, label, mutate)
        self._focus_current_canvas()

    def _on_canvas_scale_anchor_picked(self, document_id: str, anchor: Point) -> None:
        document = self.project.get_document(document_id)
        canvas = self._canvases.get(document_id)
        if document is None or canvas is None:
            return
        canvas.end_scale_anchor_pick()

        def mutate() -> None:
            document.scale_overlay_anchor = anchor

        self._apply_document_change(document, "设置比例尺位置", mutate)
        self.statusBar().showMessage("已更新当前图片的比例尺位置", 3000)
        self._focus_current_canvas()

    def _apply_calibration_line(self, document: ImageDocument, line: Line) -> None:
        dialog = CalibrationInputDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        actual_length, unit, apply_to_project = dialog.values()
        pixels_per_unit = line_length(line) / actual_length
        calibration = Calibration(
            mode="image_scale",
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            source_label=f"图内标定 {actual_length:g}{unit}",
        )

        if apply_to_project:
            self._apply_project_default_calibration(calibration, label="设置项目统一标尺")
            self.statusBar().showMessage("项目统一比例尺已更新", 4000)
            return

        def mutate() -> None:
            document.calibration = calibration
            document.metadata["calibration_line"] = line.to_dict()
            document.recalculate_measurements()

        self._apply_document_change(document, "图内标定", mutate, sync_sidecar=True)
        self.statusBar().showMessage("图内标尺标定已更新", 4000)

    def _refresh_preset_combo(self, *, selected_name: str | None = None) -> None:
        current_name = selected_name
        selected = self._selected_preset()
        if current_name is None and selected is not None:
            current_name = selected[1].name
        self.preset_combo.clear()
        target_index = -1
        for index, preset in enumerate(self._calibration_presets()):
            self.preset_combo.addItem(f"{preset.name} ({preset.resolved_pixels_per_unit():g} px/{preset.unit})")
            if current_name is not None and preset.name == current_name and target_index < 0:
                target_index = index
        if target_index >= 0:
            self.preset_combo.setCurrentIndex(target_index)
        elif self.preset_combo.count() > 0:
            self.preset_combo.setCurrentIndex(0)
        self._update_preset_combo_tooltip(self.preset_combo.currentText())
        has_preset = self.preset_combo.count() > 0
        if self._edit_preset_button is not None:
            self._edit_preset_button.setEnabled(has_preset)
        if self._delete_preset_button is not None:
            self._delete_preset_button.setEnabled(has_preset)
        if self._apply_preset_button is not None:
            self._apply_preset_button.setEnabled(has_preset and self.current_document() is not None)

    def _populate_group_list(self, document: ImageDocument | None) -> None:
        self._group_list_rebuilding = True
        self.group_list.clear()
        for row in self._group_manager().group_rows(
            document,
            default_uncategorized_color=self._app_settings.default_measurement_color,
        ):
            self._add_group_list_item(
                label=row.label,
                color=row.color,
                current_count=row.current_count,
                project_count=row.project_count,
                group_id=row.group_id,
                selected=row.selected,
            )
        self._group_list_rebuilding = False

    def _add_group_list_item(
        self,
        *,
        label: str,
        color: str,
        current_count: int,
        project_count: int,
        group_id: str | None,
        selected: bool,
    ) -> None:
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, group_id)
        item.setData(Qt.ItemDataRole.UserRole + 1, current_count)
        item.setData(Qt.ItemDataRole.UserRole + 3, project_count)
        item.setData(Qt.ItemDataRole.UserRole + 2, label)
        item.setSizeHint(QSize(0, FiberGroupListItemWidget.HEIGHT))
        self.group_list.addItem(item)
        widget = FiberGroupListItemWidget(
            label,
            current_count,
            project_count,
            color,
            selected=selected,
            parent=self.group_list,
        )
        self.group_list.setItemWidget(item, widget)
        if selected:
            item.setSelected(True)

    def _sync_group_list_item_widgets(self) -> None:
        for index in range(self.group_list.count()):
            item = self.group_list.item(index)
            widget = self.group_list.itemWidget(item)
            if isinstance(widget, FiberGroupListItemWidget):
                widget.setSelected(item.isSelected())

    def _refresh_group_list_counts(self, document: ImageDocument) -> None:
        rows = self._group_manager().group_rows(
            document,
            default_uncategorized_color=self._app_settings.default_measurement_color,
        )
        expected_group_ids = [row.group_id for row in rows]
        current_group_ids = [
            self.group_list.item(index).data(Qt.ItemDataRole.UserRole)
            for index in range(self.group_list.count())
        ]
        if current_group_ids != expected_group_ids:
            self._populate_group_list(document)
            return
        self._group_list_rebuilding = True
        try:
            for index, row in enumerate(rows):
                item = self.group_list.item(index)
                item.setData(Qt.ItemDataRole.UserRole + 1, row.current_count)
                item.setData(Qt.ItemDataRole.UserRole + 3, row.project_count)
                item.setSelected(row.selected)
                widget = self.group_list.itemWidget(item)
                if isinstance(widget, FiberGroupListItemWidget):
                    widget.setCounts(row.current_count, row.project_count)
                    widget.setSelected(row.selected)
        finally:
            self._group_list_rebuilding = False

    def _scroll_active_group_item_into_view(self) -> None:
        target_item = None
        selected_items = self.group_list.selectedItems()
        if selected_items:
            target_item = selected_items[0]
        elif self.group_list.count() > 0:
            document = self.current_document()
            active_group_id = document.active_group_id if document is not None else None
            for index in range(self.group_list.count()):
                item = self.group_list.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == active_group_id:
                    target_item = item
                    break
        if target_item is not None:
            self.group_list.scrollToItem(target_item, QAbstractItemView.ScrollHint.PositionAtCenter)

    def _documents_for_group_counts(self, current_document: ImageDocument | None) -> list[ImageDocument]:
        return self._group_manager().documents_for_group_counts(current_document)

    def _project_measurement_count_for_group_label(self, label: str, current_document: ImageDocument | None = None) -> int:
        return self._group_manager().project_measurement_count_for_group_label(label, current_document)

    def _project_uncategorized_measurement_count(self, current_document: ImageDocument | None = None) -> int:
        return self._group_manager().project_uncategorized_measurement_count(current_document)

    def _update_ui_for_current_document(self) -> None:
        document = self.current_document()
        self._populate_group_list(document)
        self._update_calibration_panel(document)
        self._populate_measurement_table(document)
        self._update_image_resolution_label(document)
        self._update_statusbar_aux_labels()
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.set_settings(self._app_settings)
            canvas.set_tool_mode("select" if self._preview_active and canvas is self._preview_canvas else self._tool_mode)
            canvas.set_show_area_fill(False if self._preview_active and canvas is self._preview_canvas else self._show_area_fill)
        self._update_action_states()

    def _update_calibration_panel(self, document: ImageDocument | None) -> None:
        if self._preview_active:
            self._set_calibration_status_card(
                title="实时预览中",
                summary="图片编辑与标定已暂停",
                status="preview",
                details="实时预览中，图片编辑、测量记录编辑与标定操作暂时不可用。",
                show_start_button=False,
            )
            return
        if document is None or document.calibration is None:
            self._set_calibration_status_card(
                title="未标定",
                summary="测量仅显示 px，无法输出真实尺寸",
                status="uncalibrated",
                details="请使用图内标定或应用标定预设后再进行真实尺寸测量。",
                show_start_button=True,
            )
            return
        calibration = document.calibration
        source_label = calibration.source_label or self._format_calibration_mode(calibration.mode)
        unit_per_px = 1.0 / calibration.pixels_per_unit if calibration.pixels_per_unit > 0 else 0.0
        details = [
            f"标定来源: {source_label}",
            f"标定模式: {self._format_calibration_mode(calibration.mode)}",
            f"换算关系: {calibration.pixels_per_unit:.4f} px/{calibration.unit}",
            f"像素尺寸: {unit_per_px:.6g} {calibration.unit}/px",
        ]
        if calibration.mode == "project_default" or not document.uses_sidecar():
            details.append("保存位置: 当前项目")
        else:
            details.append(f"侧车: {Path(document.sidecar_path or document.default_sidecar_path()).name}")
        self._set_calibration_status_card(
            title=f"已标定 · {source_label}",
            summary=f"{unit_per_px:.6g} {calibration.unit}/px",
            status="calibrated",
            details="\n".join(details),
            show_start_button=False,
        )

    def _populate_measurement_table(self, document: ImageDocument | None) -> None:
        self._table_rebuilding = True
        self.measurement_table.setRowCount(0)
        if document is not None:
            for row, measurement in enumerate(document.measurements):
                self.measurement_table.insertRow(row)
                self._set_measurement_table_row(row, document, measurement)
        self._table_rebuilding = False
        if document is not None:
            self._sync_measurement_table_selection(document)

    def _set_measurement_table_row(self, row: int, document: ImageDocument, measurement: Measurement) -> None:
        display_id = measurement.id.split("_")[-1]
        id_item = QTableWidgetItem(display_id)
        id_item.setData(Qt.ItemDataRole.UserRole, measurement.id)
        self.measurement_table.setCellWidget(row, self.TABLE_COL_GROUP, self._create_group_combo(document, measurement))
        self.measurement_table.setItem(row, self.TABLE_COL_KIND, QTableWidgetItem(self._format_measurement_kind(measurement)))
        self.measurement_table.setItem(row, self.TABLE_COL_RESULT, QTableWidgetItem(f"{measurement.display_value():.4f}"))
        self.measurement_table.setItem(row, self.TABLE_COL_UNIT, QTableWidgetItem(measurement.display_unit(document.calibration)))
        self.measurement_table.setItem(row, self.TABLE_COL_MODE, QTableWidgetItem(self._format_measurement_mode(measurement.mode)))
        self.measurement_table.setItem(row, self.TABLE_COL_CONFIDENCE, QTableWidgetItem(f"{measurement.confidence:.2f}"))
        self.measurement_table.setItem(row, self.TABLE_COL_STATUS, QTableWidgetItem(self._format_measurement_status(measurement.status)))
        self.measurement_table.setItem(row, self.TABLE_COL_ID, id_item)

    def _append_measurement_table_row(self, document: ImageDocument, measurement: Measurement) -> None:
        expected_existing_rows = max(0, len(document.measurements) - 1)
        if self.measurement_table.rowCount() != expected_existing_rows:
            self._populate_measurement_table(document)
            return
        self._table_rebuilding = True
        self.measurement_table.blockSignals(True)
        try:
            row = self.measurement_table.rowCount()
            self.measurement_table.insertRow(row)
            self._set_measurement_table_row(row, document, measurement)
            self.measurement_table.clearSelection()
            self.measurement_table.selectRow(row)
        finally:
            self.measurement_table.blockSignals(False)
            self._table_rebuilding = False

    def _format_measurement_kind(self, measurement: Measurement) -> str:
        return {
            "line": "线段",
            "polyline": "折线",
            "area": "面积",
            "count": "计数点",
        }.get(measurement.measurement_kind, measurement.measurement_kind)

    def _format_measurement_mode(self, mode: str) -> str:
        return {
            "manual": "手动线段",
            "continuous_manual": "连续测量",
            "count": "计数",
            "snap": "边缘吸附",
            "fiber_auto": "快速测径",
            "fiber_quick": "快速测径",
            "polygon_area": "多边形面积",
            "freehand_area": "自由形状面积",
            "magic_segment": "魔棒分割",
            "auto_instance": "实例分割",
            "reference_instance": "同类扩选",
        }.get(mode, mode)

    def _format_measurement_status(self, status: str) -> str:
        return {
            "manual": "手动测量",
            "continuous_manual": "连续测量",
            "ready": "已完成",
            "manual_review": "需人工复核",
            "snapped": "吸附成功",
            "edited": "已编辑",
            "line_too_short": "测量线过短",
            "profile_too_flat": "灰度变化不足",
            "edge_pair_not_found": "未找到有效边缘",
            "component_not_found": "未找到目标区域",
            "centerline_not_found": "未找到可靠中心线",
            "boundary_not_found": "未找到边界",
            "fiber_auto": "快速测径",
            "fiber_quick": "快速测径",
            "count": "计数",
            "auto_instance": "自动识别",
            "reference_instance": "同类扩选",
        }.get(status, status)

    def _edge_snap_status_message(self, result: SnapResult) -> str:
        return {
            "snapped": "边缘吸附成功",
            "manual_review": "边缘吸附完成，建议人工复核",
            "line_too_short": "测量线过短，已保留原线供人工修正",
            "profile_too_flat": "灰度变化不足，已保留原线供人工修正",
            "edge_pair_not_found": "未找到有效边缘，已保留原线供人工修正",
        }.get(result.status, "边缘吸附已完成")

    def _create_group_combo(self, document: ImageDocument, measurement: Measurement) -> QComboBox:
        combo = MeasurementGroupComboBox()
        combo.setProperty("measurement_id", measurement.id)
        combo.addItem(self._color_icon(self._app_settings.default_measurement_color), UNCATEGORIZED_LABEL, None)
        for group in document.sorted_groups():
            combo.addItem(self._color_icon(group.color), group.display_name(), group.id)
        current_index = combo.findData(measurement.fiber_group_id)
        combo.setCurrentIndex(0 if current_index < 0 else current_index)
        combo.currentIndexChanged.connect(lambda index, widget=combo: self._on_measurement_group_combo_changed(widget))
        return combo

    def _sync_measurement_table_selection(self, document: ImageDocument) -> None:
        target_id = document.view_state.selected_measurement_id
        self.measurement_table.blockSignals(True)
        self.measurement_table.clearSelection()
        if target_id is not None:
            for row in range(self.measurement_table.rowCount()):
                item = self._measurement_id_item(row)
                if item and item.data(Qt.ItemDataRole.UserRole) == target_id:
                    self.measurement_table.selectRow(row)
                    break
        self.measurement_table.blockSignals(False)

    def _on_measurement_selection_changed(self) -> None:
        if self._table_rebuilding:
            return
        document = self.current_document()
        canvas = self.current_canvas()
        if document is None or canvas is None:
            return
        selected_rows = self.measurement_table.selectionModel().selectedRows()
        if not selected_rows:
            self._update_action_states()
            return
        row = selected_rows[0].row()
        item = self._measurement_id_item(row)
        if item is None:
            return
        measurement_id = item.data(Qt.ItemDataRole.UserRole)
        document.select_measurement(measurement_id)
        canvas.set_selected_measurement(measurement_id)
        self._update_action_states()

    def _measurement_id_item(self, row: int) -> QTableWidgetItem | None:
        return self.measurement_table.item(row, self.TABLE_COL_ID)

    def _on_measurement_group_combo_changed(self, combo: QComboBox) -> None:
        if self._table_rebuilding:
            return
        document = self.current_document()
        if document is None:
            return
        measurement_id = combo.property("measurement_id")
        if not measurement_id:
            return
        target_group_id = combo.currentData()
        measurement = document.get_measurement(measurement_id)
        if measurement is None or measurement.fiber_group_id == target_group_id:
            return
        document.select_measurement(measurement_id)

        def mutate() -> None:
            document.set_measurement_group(measurement_id, target_group_id)

        self._apply_document_change(document, "修改测量分类", mutate)

    def _on_group_selection_changed(self) -> None:
        if self._group_list_rebuilding:
            return
        document = self.current_document()
        if document is None:
            return
        selected_items = self.group_list.selectedItems()
        if not selected_items:
            document.set_active_group(None)
            self._sync_group_list_item_widgets()
            self._update_action_states()
            return
        document.set_active_group(selected_items[0].data(Qt.ItemDataRole.UserRole))
        self._populate_group_list(document)
        self._update_action_states()

    def _focus_current_canvas(self) -> None:
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.focus_canvas()

    def _should_handle_group_hotkeys(self) -> bool:
        if QApplication.activeModalWidget() is not None:
            return False
        focus_widget = QApplication.focusWidget()
        if focus_widget is None:
            return True
        if isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return False
        if isinstance(focus_widget, QComboBox) and focus_widget.isEditable():
            return False
        return True

    def _should_handle_digital_slide_jog_hotkeys(self) -> bool:
        return bool(self._digital_slide_mode and self._preview_active and self._should_handle_group_hotkeys())

    def _switch_active_group_by_number(self, number: int) -> bool:
        document = self.current_document()
        if document is None:
            return False
        group = document.get_group_by_number(number)
        if group is None:
            return False
        document.set_active_group(group.id)
        self._populate_group_list(document)
        self._scroll_active_group_item_into_view()
        self._update_action_states()
        self._focus_current_canvas()
        return True

    def _create_progress_dialog(self, *, title: str, label_text: str, maximum: int) -> QProgressDialog:
        progress = QProgressDialog(label_text, "取消", 0, maximum, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.setMinimumWidth(420)
        return progress

    def _create_blocking_progress_dialog(self, *, title: str, label_text: str, maximum: int) -> QProgressDialog:
        progress = self._create_progress_dialog(title=title, label_text=label_text, maximum=maximum)
        progress.setCancelButton(None)
        progress.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        progress.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        return progress

    def _update_blocking_progress_dialog(
        self,
        progress: QProgressDialog,
        *,
        completed_steps: int,
        total_steps: int,
        label: str,
        path: Path | None,
    ) -> None:
        total = max(1, total_steps)
        progress.setMaximum(total)
        progress.setValue(max(0, min(completed_steps, total)))
        if path is not None:
            current_index = min(completed_steps + 1, total)
            progress.setLabelText(f"正在导出 ({current_index}/{total})\n{path.name}")
        elif label:
            progress.setLabelText(label)
        self._pump_modal_progress_events()

    def _pump_modal_progress_events(self) -> None:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def _close_progress_dialog(self, progress: QProgressDialog | None) -> None:
        if progress is None:
            return
        progress.close()
        progress.deleteLater()
        self._pump_modal_progress_events()

    def _is_export_file_busy_error(self, exc: Exception) -> bool:
        if isinstance(exc, PermissionError):
            return True
        if isinstance(exc, OSError) and getattr(exc, "errno", None) in {13, 16, 32}:
            return True
        lowered = str(exc).lower()
        return "permission denied" in lowered or "being used by another process" in lowered

    def _format_export_failure_message(self, exc: Exception, *, export_path: Path | None) -> str:
        failed_path = getattr(exc, "filename", None)
        resolved_path = Path(failed_path) if isinstance(failed_path, str) and failed_path else export_path
        if self._is_export_file_busy_error(exc):
            if resolved_path is not None:
                return (
                    "无法覆盖导出文件，文件可能正在被其他程序占用：\n"
                    f"{resolved_path}\n\n"
                    "请关闭占用该文件的程序后重试。"
                )
            return "无法覆盖导出文件，文件可能正在被其他程序占用。\n请关闭占用该文件的程序后重试。"
        if resolved_path is not None:
            return f"导出过程中写入文件失败：\n{resolved_path}\n\n{exc}"
        return f"导出过程中发生错误：\n{exc}"

    def _update_action_states(self) -> None:
        document = self.current_document()
        history = document.history if document is not None else None
        has_document = document is not None
        preview_active = self._preview_active
        selection_model = self.measurement_table.selectionModel() if hasattr(self, "measurement_table") else None
        has_selected_rows = bool(selection_model and selection_model.selectedRows())
        has_selected_object = bool(
            has_document
            and self._tool_mode != "calibration"
            and (
                has_selected_rows
                or
                document.view_state.selected_measurement_id is not None
                or document.selected_overlay_id is not None
            )
        )
        has_measurements = bool(document and document.measurements)
        has_measurement_groups = bool(document and document.measurement_group_labels())
        has_deletable_group_target = bool(
            document and (
                document.get_group(document.active_group_id) is not None
                or document.should_show_uncategorized_entry()
            )
        )
        has_named_active_group = bool(document and document.get_group(document.active_group_id) is not None)
        has_editable_active_group = bool(
            has_named_active_group
            or (
                document
                and document.active_group_id is None
                and document.should_show_uncategorized_entry()
            )
        )
        self.close_current_action.setEnabled(has_document)
        self.close_all_action.setEnabled(bool(self.project.documents))
        self.delete_measurement_action.setEnabled(has_selected_object and not preview_active)
        self.delete_measurement_button.setEnabled(has_selected_object and not preview_active)
        if self._delete_group_measurements_button is not None:
            self._delete_group_measurements_button.setEnabled(has_measurement_groups and not preview_active)
        if self._delete_all_measurements_button is not None:
            self._delete_all_measurements_button.setEnabled(has_measurements and not preview_active)
        self.add_group_action.setEnabled(has_document and not preview_active)
        self.rename_group_action.setEnabled(has_editable_active_group and not preview_active)
        self.delete_group_action.setEnabled(has_deletable_group_target and not preview_active)
        if self._add_group_button is not None:
            self._add_group_button.setEnabled(has_document and not preview_active)
        if self._rename_group_button is not None:
            self._rename_group_button.setEnabled(has_editable_active_group and not preview_active)
        if self.delete_group_button is not None:
            self.delete_group_button.setEnabled(has_deletable_group_target and not preview_active)
        has_preset = bool(self._calibration_presets())
        if self._add_preset_button is not None:
            self._add_preset_button.setEnabled(True)
        if self._edit_preset_button is not None:
            self._edit_preset_button.setEnabled(has_preset)
        if self._delete_preset_button is not None:
            self._delete_preset_button.setEnabled(has_preset)
        if self._import_cu_preset_button is not None:
            self._import_cu_preset_button.setEnabled(_CU_SCALE_IMPORT_ERROR is None)
        if self._apply_preset_button is not None:
            self._apply_preset_button.setEnabled(has_document and has_preset and not preview_active)
        if self._area_auto_button is not None:
            self._area_auto_button.setEnabled(
                has_document
                and bool(self._app_settings.area_model_mappings)
                and not preview_active
                and not (document is not None and document.is_digital_slide())
            )
        self.undo_action.setEnabled(bool(history and history.can_undo()) and not preview_active)
        self.redo_action.setEnabled(bool(history and history.can_redo()) and not preview_active)
        capture_feature_available = _CAPTURE_IMPORT_ERROR is None
        self.switch_capture_device_action.setEnabled(capture_feature_available)
        self.live_preview_action.setEnabled(capture_feature_available)
        if self.digital_slide_action is not None:
            self.digital_slide_action.setEnabled(capture_feature_available and not self._preview_analysis_finalizing)
            self.digital_slide_action.setToolTip("进入数字化切片采集工作台")
        can_optimize_signal = capture_feature_available and self._capture_manager.can_optimize_signal()
        analysis_active = self._preview_analysis_mode != "none"
        self.capture_frame_action.setEnabled(preview_active and self._capture_manager.can_capture_still() and not analysis_active and not self._digital_slide_mode)
        self.optimize_capture_signal_action.setVisible(can_optimize_signal)
        self.optimize_capture_signal_action.setEnabled(can_optimize_signal and not analysis_active and not self._digital_slide_mode)
        for mode, action in self._mode_actions.items():
            action.setEnabled(not preview_active or mode == "select")
        if self._manual_tool_button is not None:
            self._manual_tool_button.setEnabled(not preview_active)
        if self._area_tool_button is not None:
            self._area_tool_button.setEnabled(not preview_active)
        if self._magic_tool_button is not None:
            self._magic_tool_button.setEnabled(not preview_active)
        if self._overlay_tool_button is not None:
            self._overlay_tool_button.setEnabled(not preview_active)
        self._update_path_drawing_controls()
        self._update_magic_segment_controls()
        self._update_count_numbers_button()
        self._update_preview_analysis_controls()
        self._sync_manual_tool_button()
        self._sync_area_tool_button()
        self._sync_magic_tool_button()
        self._sync_overlay_tool_button()
        self._sync_digital_slide_navigation_action()

    def _magic_prompt_label_text(self, prompt_type: str) -> str:
        return "当前提示：负采样点" if prompt_type == "negative" else "当前提示：正采样点"

    def _magic_operation_label_text(self, operation_mode: str) -> str:
        if operation_mode == MagicSegmentOperationMode.SUBTRACT:
            return "当前编辑：剔除形状"
        return "当前编辑：主体"

    def _magic_operation_button_text(self, operation_mode: str) -> str:
        if operation_mode == MagicSegmentOperationMode.SUBTRACT:
            return "剔除(T)"
        return "添加(T)"

    def _magic_subtract_input_label(self, mode: str) -> str:
        return {
            MagicSegmentSubtractInputMode.POLYGON: "多边形剔除",
            MagicSegmentSubtractInputMode.FREEHAND: "自由圈选剔除",
        }.get(MagicSegmentSubtractInputMode.normalize(mode), "智能剔除")

    def _magic_subtract_input_tooltip(self, mode: str) -> str:
        return {
            MagicSegmentSubtractInputMode.POLYGON: "点击添加顶点，双击、靠近起点点击或 Enter 闭合剔除区域",
            MagicSegmentSubtractInputMode.FREEHAND: "按住拖拽圈出要剔除的区域，松手后生成剔除草稿",
        }.get(MagicSegmentSubtractInputMode.normalize(mode), "点选目标并由魔棒生成剔除草稿")

    def _remember_magic_subtract_input_mode(self, mode: str) -> str:
        normalized = MagicSegmentSubtractInputMode.normalize(mode)
        self._magic_standard_subtract_input_mode = normalized
        if getattr(self._app_settings, "magic_segment_standard_subtract_input_mode", None) != normalized:
            self._app_settings.magic_segment_standard_subtract_input_mode = normalized
            self._save_app_settings(context="剔除方式")
        return normalized

    def _sync_canvas_magic_subtract_input_mode(self, canvas: DocumentCanvas | None) -> None:
        if canvas is None or canvas.has_magic_manual_subtract_draft():
            return
        canvas.set_magic_subtract_input_mode(self._magic_standard_subtract_input_mode)

    def _confirm_discard_magic_manual_subtract_draft(self, title: str, text: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        discard_button = box.addButton("丢弃草稿并切换", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = box.addButton("取消切换", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_button)
        box.exec()
        return box.clickedButton() == discard_button

    def _set_magic_subtract_input_mode(self, mode: str) -> None:
        canvas = self.current_canvas()
        if canvas is None or not is_magic_segment_tool_mode(self._tool_mode):
            return
        normalized = MagicSegmentSubtractInputMode.normalize(mode)
        if canvas.current_magic_subtract_input_mode() == normalized:
            self._remember_magic_subtract_input_mode(normalized)
            self._update_magic_segment_controls()
            return
        if canvas.has_magic_manual_subtract_draft():
            if not self._confirm_discard_magic_manual_subtract_draft(
                "切换剔除方式",
                "当前剔除区域尚未闭合。要丢弃草稿并切换剔除方式吗？",
            ):
                self._update_magic_segment_controls()
                return
        normalized = self._remember_magic_subtract_input_mode(normalized)
        canvas.set_magic_subtract_input_mode(normalized)
        self.statusBar().showMessage(f"已切换为{self._magic_subtract_input_label(normalized)}", 2500)
        self._update_magic_segment_controls()

    def _current_magic_roi_enabled(self, tool_mode: str | None = None, *, operation_mode: str | None = None) -> bool:
        active_mode = str(tool_mode or self._tool_mode or "").strip()
        if is_fiber_quick_tool_mode(active_mode):
            return bool(self._fiber_quick_roi_enabled)
        if is_magic_segment_tool_mode(active_mode):
            active_operation = operation_mode
            if active_operation is None:
                canvas = self.current_canvas()
                active_operation = (
                    canvas.current_magic_segment_operation_mode()
                    if canvas is not None
                    else MagicSegmentOperationMode.ADD
                )
            if active_operation == MagicSegmentOperationMode.SUBTRACT:
                return bool(self._magic_standard_subtract_roi_enabled)
            return bool(self._magic_standard_add_roi_enabled)
        return False

    def _set_magic_roi_enabled(self, tool_mode: str, enabled: bool, *, operation_mode: str | None = None) -> None:
        if is_fiber_quick_tool_mode(tool_mode):
            self._fiber_quick_roi_enabled = bool(enabled)
        elif is_magic_segment_tool_mode(tool_mode):
            if operation_mode == MagicSegmentOperationMode.SUBTRACT:
                self._magic_standard_subtract_roi_enabled = bool(enabled)
            else:
                self._magic_standard_add_roi_enabled = bool(enabled)

    def _set_active_magic_roi_checked(self, checked: bool) -> None:
        if not (is_magic_segment_tool_mode(self._tool_mode) or is_fiber_quick_tool_mode(self._tool_mode)):
            return
        operation_mode = None
        if is_magic_segment_tool_mode(self._tool_mode):
            canvas = self.current_canvas()
            operation_mode = (
                canvas.current_magic_segment_operation_mode()
                if canvas is not None
                else MagicSegmentOperationMode.ADD
            )
        checked = bool(checked)
        if self._current_magic_roi_enabled(operation_mode=operation_mode) == checked:
            self._update_magic_segment_controls()
            return
        self._set_magic_roi_enabled(self._tool_mode, checked, operation_mode=operation_mode)
        state_text = "启用" if checked else "关闭"
        self.statusBar().showMessage(f"已{state_text}ROI局部分割", 2500)
        self._update_magic_segment_controls()

    def _toggle_active_magic_roi(self) -> None:
        if not (is_magic_segment_tool_mode(self._tool_mode) or is_fiber_quick_tool_mode(self._tool_mode)):
            return
        operation_mode = None
        if is_magic_segment_tool_mode(self._tool_mode):
            canvas = self.current_canvas()
            operation_mode = (
                canvas.current_magic_segment_operation_mode()
                if canvas is not None
                else MagicSegmentOperationMode.ADD
            )
        self._set_magic_roi_enabled(
            self._tool_mode,
            not self._current_magic_roi_enabled(operation_mode=operation_mode),
            operation_mode=operation_mode,
        )
        state_text = "启用" if self._current_magic_roi_enabled(operation_mode=operation_mode) else "关闭"
        self.statusBar().showMessage(f"已{state_text}ROI局部分割", 2500)
        self._update_magic_segment_controls()

    def _magic_small_object_enhancement_context_active(
        self,
        *,
        tool_mode: str | None = None,
        operation_mode: str | None = None,
        roi_enabled: bool | None = None,
    ) -> bool:
        active_tool = str(tool_mode or self._tool_mode or "").strip()
        if not is_magic_segment_tool_mode(active_tool):
            return False
        active_operation = operation_mode
        if active_operation is None:
            canvas = self.current_canvas()
            active_operation = (
                canvas.current_magic_segment_operation_mode()
                if canvas is not None
                else MagicSegmentOperationMode.ADD
            )
        if active_operation != MagicSegmentOperationMode.SUBTRACT:
            return False
        canvas = self.current_canvas()
        if (
            canvas is not None
            and is_magic_segment_tool_mode(active_tool)
            and canvas.current_magic_subtract_input_mode() != MagicSegmentSubtractInputMode.SMART
        ):
            return False
        active_roi = self._current_magic_roi_enabled(active_tool, operation_mode=active_operation) if roi_enabled is None else bool(roi_enabled)
        return bool(active_roi and self._app_settings.magic_segment_restrict_subtract_roi_to_primary_bounds)

    def _toggle_magic_small_object_enhancement(self, checked: bool) -> None:
        self._app_settings.magic_segment_small_object_subtract_enhancement_enabled = bool(checked)
        if not checked:
            self._hide_small_object_preview()
        self._save_app_settings(context="小洞")
        self._update_magic_segment_controls()
        self.statusBar().showMessage("小洞已开启" if checked else "小洞已关闭", 2500)

    def _update_magic_segment_controls(self) -> None:
        if self._magic_controls_widget is None or self._measurement_tool_strip is None:
            return
        is_visible = is_magic_toolbar_tool_mode(self._tool_mode) and not self._preview_active
        self._measurement_tool_strip.setMagicContextVisible(is_visible)
        if not is_visible:
            self._hide_small_object_preview()
            return
        canvas = self.current_canvas()
        has_document = canvas is not None and canvas.document_id is not None
        standard_mode = is_magic_segment_tool_mode(self._tool_mode)
        fiber_quick_mode = is_fiber_quick_tool_mode(self._tool_mode)
        if standard_mode:
            prompt_type = canvas.current_magic_segment_prompt_type() if canvas is not None else "positive"
            operation_mode = (
                canvas.current_magic_segment_operation_mode()
                if canvas is not None
                else MagicSegmentOperationMode.ADD
            )
            subtract_input_mode = (
                canvas.current_magic_subtract_input_mode()
                if canvas is not None
                else MagicSegmentSubtractInputMode.SMART
            )
            busy = bool(canvas and canvas.is_magic_segment_busy())
        elif fiber_quick_mode:
            prompt_type = canvas.current_fiber_quick_prompt_type() if canvas is not None else "positive"
            operation_mode = MagicSegmentOperationMode.ADD
            subtract_input_mode = MagicSegmentSubtractInputMode.SMART
            busy = bool(canvas and canvas.is_fiber_quick_busy())
        else:
            prompt_type = "positive"
            operation_mode = MagicSegmentOperationMode.ADD
            subtract_input_mode = MagicSegmentSubtractInputMode.SMART
            busy = bool(canvas and canvas.is_reference_instance_busy())
        if self._magic_prompt_label is not None:
            self._magic_prompt_label.setVisible(False)
            show_reference_prompt = self._measurement_tool_strip.isContextInline()
            if not standard_mode and not fiber_quick_mode and canvas is not None and canvas.has_reference_instance_preview():
                self._magic_prompt_label.setText("候选预览")
                self._magic_prompt_label.setVisible(show_reference_prompt)
            elif not standard_mode and not fiber_quick_mode:
                self._magic_prompt_label.setText("拖框或点已确认面积作为参考")
                self._magic_prompt_label.setVisible(show_reference_prompt)
        if self._magic_toggle_button is not None:
            show_prompt_toggle = (
                fiber_quick_mode
                or (
                    standard_mode
                    and (
                        operation_mode != MagicSegmentOperationMode.SUBTRACT
                        or subtract_input_mode == MagicSegmentSubtractInputMode.SMART
                    )
                )
            )
            self._magic_toggle_button.setVisible(show_prompt_toggle)
            if standard_mode:
                visual = magic_prompt_visual(prompt_type)
                self._magic_toggle_button.setText(f"{visual.button_label}(R)")
                self._magic_toggle_button.setProperty("magicPrompt", visual.prompt_type)
            else:
                self._magic_toggle_button.setText("正负(R)")
                self._magic_toggle_button.setProperty("magicPrompt", "")
            self._measurement_tool_strip._apply_button_palette(self._magic_toggle_button)
            refresh_widget_theme(self._magic_toggle_button)
            self._magic_toggle_button.setEnabled(
                has_document
                and show_prompt_toggle
                and (fiber_quick_mode or not busy)
            )
        roi_enabled = self._current_magic_roi_enabled(operation_mode=operation_mode)
        small_object_context = self._magic_small_object_enhancement_context_active(
            tool_mode=self._tool_mode,
            operation_mode=operation_mode,
            roi_enabled=roi_enabled,
        )
        small_object_enabled = bool(self._app_settings.magic_segment_small_object_subtract_enhancement_enabled)
        if not small_object_context or not small_object_enabled:
            self._hide_small_object_preview()
        if self._magic_operation_button is not None:
            self._magic_operation_button.setVisible(standard_mode)
            self._magic_operation_button.setText(self._magic_operation_button_text(operation_mode))
            self._magic_operation_button.setProperty(
                "magicPrompt",
                "negative" if operation_mode == MagicSegmentOperationMode.SUBTRACT else "",
            )
            self._measurement_tool_strip._apply_button_palette(self._magic_operation_button)
            refresh_widget_theme(self._magic_operation_button)
            self._magic_operation_button.setEnabled(has_document and standard_mode and not busy)
        if self._magic_subtract_mode_button is not None:
            self._magic_subtract_mode_button.setVisible(standard_mode and operation_mode == MagicSegmentOperationMode.SUBTRACT)
            self._magic_subtract_mode_button.setText(f"剔除方式：{self._magic_subtract_input_label(subtract_input_mode)}")
            self._magic_subtract_mode_button.setToolTip(self._magic_subtract_input_tooltip(subtract_input_mode))
            self._magic_subtract_mode_button.setEnabled(has_document and standard_mode and not busy)
        for mode, action in self._magic_subtract_mode_actions.items():
            action.setChecked(mode == subtract_input_mode)
            action.setEnabled(has_document and standard_mode and operation_mode == MagicSegmentOperationMode.SUBTRACT and not busy)
        if self._magic_options_button is not None:
            self._magic_options_button.setVisible(standard_mode or fiber_quick_mode)
            self._magic_options_button.setText("选项")
            self._magic_options_button.setToolTip(
                f"工具选项：ROI {'开启' if roi_enabled else '关闭'}；"
                f"小洞增强 {'开启' if small_object_enabled else '关闭'}"
            )
            self._magic_options_button.setEnabled(has_document and (standard_mode or fiber_quick_mode))
        if self._magic_roi_option_checkbox is not None:
            self._magic_roi_option_checkbox.blockSignals(True)
            self._magic_roi_option_checkbox.setChecked(roi_enabled)
            self._magic_roi_option_checkbox.setEnabled(has_document and (standard_mode or fiber_quick_mode))
            self._magic_roi_option_checkbox.blockSignals(False)
        if self._magic_small_object_option_checkbox is not None:
            self._magic_small_object_option_checkbox.blockSignals(True)
            self._magic_small_object_option_checkbox.setChecked(small_object_enabled)
            self._magic_small_object_option_checkbox.setEnabled(has_document and small_object_context and not busy)
            self._magic_small_object_option_checkbox.blockSignals(False)
        if self._magic_small_object_option_hint is not None:
            if small_object_context:
                self._magic_small_object_option_hint.setText("小洞增强已可用于当前智能剔除")
            else:
                self._magic_small_object_option_hint.setText("仅在标准魔棒的智能剔除、ROI 开启时可用")
        if self._magic_confirm_subtract_button is not None:
            self._magic_confirm_subtract_button.setVisible(standard_mode and operation_mode == MagicSegmentOperationMode.SUBTRACT)
            self._magic_confirm_subtract_button.setText("加洞(S)")
            self._magic_confirm_subtract_button.setEnabled(
                bool(canvas and canvas.can_confirm_current_magic_subtract_shape())
            )
        if self._magic_complete_button is not None:
            self._magic_complete_button.setText(
                "完成"
                if standard_mode
                else ("完成" if fiber_quick_mode else "加入")
            )
            self._magic_complete_button.setToolTip(
                "提交当前魔棒结果（Enter / F）"
                if standard_mode
                else ("确认快速测径结果（Enter / F）" if fiber_quick_mode else "加入参考实例结果（Enter / F）")
            )
            self._magic_complete_button.setEnabled(
                bool(
                    canvas
                    and (
                        (
                            canvas.has_magic_segment_preview()
                            if standard_mode
                            else (
                                canvas.has_fiber_quick_shape_preview()
                                if fiber_quick_mode
                                else canvas.has_reference_instance_preview()
                            )
                        )
                    )
                    and (
                        not busy
                        if not fiber_quick_mode
                        else not bool(canvas and canvas._fiber_quick.segmentation_busy)  # noqa: SLF001
                    )
                )
            )
        if self._magic_cancel_button is not None:
            self._magic_cancel_button.setText("取消")
            self._magic_cancel_button.setToolTip("取消当前魔棒会话或草稿（Esc）")
            self._magic_cancel_button.setEnabled(
                bool(
                    canvas
                    and (
                        canvas.has_magic_segment_session()
                        if standard_mode
                        else (
                            canvas.has_fiber_quick_session()
                            if fiber_quick_mode
                            else canvas.has_reference_instance_session()
                        )
                    )
                )
            )
        self._measurement_tool_strip._refresh_context_visibility()  # noqa: SLF001
        layout = self._magic_controls_widget.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        self._magic_controls_widget.updateGeometry()
        self._magic_controls_widget.adjustSize()
        self._measurement_tool_strip.updateGeometry()

    def _ensure_small_object_preview_window(self) -> SmallObjectEnhancementPreviewWindow:
        if self._small_object_preview_window is None:
            self._small_object_preview_window = SmallObjectEnhancementPreviewWindow(self)
        return self._small_object_preview_window

    def _hide_small_object_preview(self) -> None:
        if self._small_object_preview_window is not None:
            self._small_object_preview_window.hide()

    def _small_object_reject_label(self, reason: str) -> str:
        return {
            "empty_mask": "未找到目标",
            "mask_too_large": "结果过大",
            "touches_workspace_edges": "结果触及工作区边缘",
            "positive_outside_workspace": "正采样点超出工作区",
        }.get(reason, "置信度不足")

    def _show_small_object_preview(
        self,
        canvas: DocumentCanvas,
        metadata: dict[str, object],
        polygon_px: list[Point],
    ) -> None:
        if not self._magic_small_object_enhancement_context_active():
            self._hide_small_object_preview()
            return
        preview = self._ensure_small_object_preview_window()
        if not preview.set_preview(metadata, polygon_px):
            return
        box = metadata.get("small_object_workspace_box")
        if isinstance(box, (tuple, list)) and len(box) == 4:
            self._position_small_object_preview(preview, canvas, box)
        preview.show()
        preview.raise_()

    def _position_small_object_preview(
        self,
        preview: SmallObjectEnhancementPreviewWindow,
        canvas: DocumentCanvas,
        box: object,
    ) -> None:
        try:
            x0, y0, x1, y1 = [float(value) for value in box]  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        top_left = canvas.image_to_widget(Point(x0, y0))
        bottom_right = canvas.image_to_widget(Point(x1, y1))
        roi_rect = QRectF(top_left, bottom_right).normalized()
        roi_global_top_left = canvas.mapToGlobal(QPoint(int(roi_rect.left()), int(roi_rect.top())))
        roi_global = QRect(
            roi_global_top_left.x(),
            roi_global_top_left.y(),
            max(1, int(round(roi_rect.width()))),
            max(1, int(round(roi_rect.height()))),
        )
        margin = 12
        size = preview.size()
        screen = QGuiApplication.screenAt(roi_global.center()) or canvas.screen() or QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 720)
        candidates = [
            QRect(roi_global.right() + margin, roi_global.top(), size.width(), size.height()),
            QRect(roi_global.left() - size.width() - margin, roi_global.top(), size.width(), size.height()),
            QRect(roi_global.left(), roi_global.bottom() + margin, size.width(), size.height()),
            QRect(roi_global.left(), roi_global.top() - size.height() - margin, size.width(), size.height()),
        ]
        for candidate in candidates:
            if available.contains(candidate) and not candidate.intersects(roi_global):
                preview.move(candidate.topLeft())
                return
        x = min(max(available.left(), roi_global.right() + margin), max(available.left(), available.right() - size.width()))
        y = min(max(available.top(), roi_global.top()), max(available.top(), available.bottom() - size.height()))
        fallback = QRect(int(x), int(y), size.width(), size.height())
        if fallback.intersects(roi_global):
            x = max(available.left(), available.right() - size.width())
            y = available.top()
        preview.move(QPoint(int(x), int(y)))

    def _commit_active_path_drawing(self) -> bool:
        canvas = self.current_canvas()
        if canvas is None:
            return False
        committed = canvas.commit_pending_path()
        self._update_path_drawing_controls()
        return committed

    def _cycle_area_edit_operation_mode(self) -> bool:
        canvas = self.current_canvas()
        if canvas is None or not canvas.has_selected_area_measurement():
            return False
        if canvas.has_area_edit_draft():
            self.statusBar().showMessage("请先完成或取消当前绘制后再切换添加/剔除。", 3000)
            return False
        current_mode = canvas.current_area_edit_operation_mode()
        next_mode = (
            AreaEditOperationMode.SUBTRACT
            if current_mode == AreaEditOperationMode.ADD
            else AreaEditOperationMode.ADD
        )
        canvas.set_area_edit_operation_mode(next_mode)
        status_message = (
            "面积工具已切换为剔除"
            if next_mode == AreaEditOperationMode.SUBTRACT
            else "面积工具已切换为添加"
        )
        self.statusBar().showMessage(status_message, 2500)
        self._update_path_drawing_controls()
        return True

    def _cancel_active_path_drawing(self) -> bool:
        canvas = self.current_canvas()
        if canvas is None:
            return False
        cancelled = canvas.cancel_pending_path()
        self._update_path_drawing_controls()
        return cancelled

    def _update_path_drawing_controls(self) -> None:
        if self._path_controls_widget is None or self._measurement_tool_strip is None:
            return
        is_visible = (
            self._tool_mode
            in {"manual", "continuous_manual", "snap", "polygon_area", "freehand_area"}
            and not self._preview_active
        )
        self._measurement_tool_strip.setPathContextVisible(is_visible)
        if not is_visible:
            if self._area_operation_button is not None:
                self._area_operation_button.setVisible(False)
            return
        canvas = self.current_canvas()
        show_area_operation = bool(
            canvas
            and self._tool_mode in {"polygon_area", "freehand_area"}
            and canvas.has_selected_area_measurement()
        )
        if self._area_operation_button is not None:
            self._area_operation_button.setVisible(show_area_operation)
            area_operation_mode = (
                canvas.current_area_edit_operation_mode()
                if canvas is not None
                else AreaEditOperationMode.ADD
            )
            subtract_active = area_operation_mode == AreaEditOperationMode.SUBTRACT
            self._area_operation_button.setText("剔除(T)" if subtract_active else "添加(T)")
            self._area_operation_button.setToolTip(
                "当前会从选中面积中剔除绘制区域（T）"
                if subtract_active
                else "当前会绘制新的面积测量（T）"
            )
            self._area_operation_button.setProperty("magicPrompt", "negative" if subtract_active else "")
            self._area_operation_button.setEnabled(show_area_operation)
            self._measurement_tool_strip._apply_button_palette(self._area_operation_button)
            refresh_widget_theme(self._area_operation_button)
        if self._path_complete_button is not None:
            self._path_complete_button.setEnabled(bool(canvas and canvas.can_commit_pending_path()))
        if self._path_cancel_button is not None:
            self._path_cancel_button.setEnabled(bool(canvas and canvas.has_pending_path_drawing()))
        self._measurement_tool_strip._refresh_context_visibility()  # noqa: SLF001
        layout = self._path_controls_widget.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        self._path_controls_widget.updateGeometry()
        self._path_controls_widget.adjustSize()
        self._measurement_tool_strip.updateGeometry()

    def _preview_analysis_supported(self, mode: str | None = None) -> bool:
        selected = self._selected_capture_device()
        if not bool(
            self._preview_active
            and selected is not None
            and self._capture_manager.can_request_analysis_frame()
        ):
            return False
        if mode == "map_build":
            return self.MAP_BUILD_AVAILABLE
        return True

    def _sync_preview_analysis_buttons(self) -> None:
        if self._focus_stack_button is not None:
            self._focus_stack_button.blockSignals(True)
            self._focus_stack_button.setChecked(self._preview_analysis_mode == "focus_stack")
            self._focus_stack_button.blockSignals(False)
        if self._map_build_button is not None:
            self._map_build_button.blockSignals(True)
            self._map_build_button.setChecked(self._preview_analysis_mode == "map_build")
            self._map_build_button.blockSignals(False)

    def _update_preview_analysis_controls(self) -> None:
        if self._preview_analysis_widget is None or self._measurement_tool_strip is None:
            return
        is_visible = self._preview_active
        self._measurement_tool_strip.setPreviewContextVisible(is_visible)
        selected = self._selected_capture_device()
        focus_supported = self._preview_analysis_supported("focus_stack")
        map_supported = self._preview_analysis_supported("map_build")
        focus_tooltip = "实时预览分析：景深合成"
        map_tooltip = "实时预览分析：地图构建"
        if not self.MAP_BUILD_AVAILABLE:
            map_tooltip = "当前版本未启用地图构建。"
        focus_enabled = is_visible and focus_supported and not self._preview_analysis_finalizing and not self._digital_slide_mode
        map_enabled = is_visible and not self._preview_analysis_finalizing and not self._digital_slide_mode and (
            map_supported or not self.MAP_BUILD_AVAILABLE
        )
        if self._focus_stack_button is not None:
            self._focus_stack_button.setEnabled(focus_enabled)
            self._focus_stack_button.setToolTip(focus_tooltip)
        if self._map_build_button is not None:
            self._map_build_button.setEnabled(map_enabled)
            self._map_build_button.setToolTip(map_tooltip)
        self._sync_digital_slide_mode_ui()
        self._sync_preview_analysis_buttons()

    def _preview_analysis_intro_text(self, mode: str) -> str:
        if mode == "map_build":
            return "移动样品台到相邻视野，保持 20%-40% 重叠并等待静止；系统会先合成每个 tile，再拼接可靠地图。按 Enter 或 F 结束，Esc 取消。"
        return "尽量均匀地从一个焦距移动到另一个焦距，系统会持续采样并合成清晰图像。按 Enter 或 F 结束，Esc 取消。"

    def _analysis_mode_label(self, mode: str) -> str:
        return {
            "focus_stack": "景深合成",
            "map_build": "地图构建",
        }.get(mode, mode)

    def _preview_analysis_finalize_message(self, mode: str) -> str:
        if mode == "map_build":
            return "正在完成地图构建，请稍候…"
        return "正在完成景深合成，请稍候…"

    def _preview_analysis_interval_ms(self, mode: str) -> int:
        if mode == "map_build":
            return MAP_BUILD_ANALYSIS_INTERVAL_MS
        return self.PREVIEW_ANALYSIS_INTERVAL_MS

    def _map_build_state_banner(self, report: MapBuildReport) -> tuple[str, str, str]:
        state_map = {
            "moving": ("正在移动", "moving"),
            "settling": ("等待静止", "settling"),
            "sampling": ("正在采样", "sampling"),
            "tile_committed": ("已创建新 tile", "success"),
            "candidate_rejected": ("候选位置未通过", "warning"),
        }
        title, tone = state_map.get(report.motion_state, (report.motion_state, "neutral"))
        stable_count = min(report.stable_streak, MAP_BUILD_STABLE_REQUIRED_FRAMES)
        if report.motion_state == "moving":
            detail = f"请移动到相邻视野后停稳 | 位移 {report.translation_px:.1f}px | tile {report.tile_count}"
        elif report.motion_state == "settling":
            detail = f"保持不动，正在确认稳定 {stable_count}/{MAP_BUILD_STABLE_REQUIRED_FRAMES} | 位移 {report.translation_px:.1f}px"
        elif report.motion_state == "sampling":
            detail = f"当前视野已稳定，正在积累 tile | tile {report.tile_count} | 接受 {report.accepted_frames} 帧"
        elif report.motion_state == "tile_committed":
            detail = f"新 tile 已加入地图 | tile {report.tile_count} | 响应 {report.correlation_response:.2f}"
        elif report.motion_state == "candidate_rejected":
            detail = "当前位置匹配不可靠，请增加纹理或调整到 20%-40% 重叠后再停稳。"
        else:
            detail = report.message
        return title, detail, tone

    def _current_focus_stack_render_config(self) -> FocusStackRenderConfig:
        return FocusStackRenderConfig(
            profile=self._app_settings.focus_stack_profile or FocusStackProfile.BALANCED,
            sharpen_strength=self._app_settings.focus_stack_sharpen_strength,
        ).normalized_copy()

    def _toggle_preview_analysis_mode(self, mode: str, checked: bool) -> None:
        if not checked:
            if self._preview_analysis_mode == mode:
                self._cancel_preview_analysis_session(message=f"已取消{self._analysis_mode_label(mode)}")
            else:
                self._sync_preview_analysis_buttons()
            return
        if not self._preview_analysis_supported(mode):
            message = "该功能需要实时预览已提供可用分析帧。"
            if mode == "map_build":
                message = "地图构建需要实时预览已提供可用分析帧。"
            self._sync_preview_analysis_buttons()
            QMessageBox.information(self, self._analysis_mode_label(mode), message)
            return
        if self._preview_analysis_mode != "none":
            self._cancel_preview_analysis_session()
        self._start_preview_analysis_session(mode)

    def _create_preview_analysis_dialog(self, mode: str) -> PreviewAnalysisDialog:
        dialog = PreviewAnalysisDialog(
            self._analysis_mode_label(mode),
            intro_text=self._preview_analysis_intro_text(mode),
            compact=mode == "map_build",
            show_state_banner=mode == "map_build",
            parent=self,
        )
        dialog.finishRequested.connect(self._finalize_preview_analysis_session)
        dialog.cancelRequested.connect(lambda: self._cancel_preview_analysis_session())
        return dialog

    def _start_preview_analysis_session(self, mode: str) -> None:
        self.preview_analysis_task_controller.start_session(mode)

    def _teardown_preview_analysis_session(self, *, cancel_worker: bool, status_message: str | None = None) -> None:
        self.preview_analysis_task_controller.teardown(
            cancel_worker=cancel_worker,
            status_message=status_message,
        )

    def _cancel_preview_analysis_session(self, *, message: str | None = None) -> None:
        self.preview_analysis_task_controller.cancel(message=message)

    def _finalize_preview_analysis_session(self) -> None:
        self.preview_analysis_task_controller.finalize()

    def _request_preview_analysis_frame(self) -> None:
        self.preview_analysis_task_controller.request_frame()

    def _on_preview_analysis_frame_ready(self, request_id: int, image: object) -> None:
        self.preview_analysis_task_controller.on_frame_ready(request_id, image)

    def _on_preview_analysis_frame_failed(self, request_id: int, message: str) -> None:
        self.preview_analysis_task_controller.on_frame_failed(request_id, message)

    def _on_preview_analysis_worker_preview(self, payload: object) -> None:
        if self._preview_analysis_dialog is None:
            return
        if isinstance(payload, FocusStackReport):
            self._preview_analysis_dialog.set_result_image(payload.preview_image)
            self._preview_analysis_dialog.set_status(payload.message)
            self.statusBar().showMessage(payload.message, 2500)
            return
        if isinstance(payload, MapBuildReport):
            self._preview_analysis_dialog.set_result_image(payload.preview_image)
            title, detail, tone = self._map_build_state_banner(payload)
            self._preview_analysis_dialog.set_state_banner(title, detail, tone)
            self._preview_analysis_dialog.set_status(payload.message)
            self.statusBar().showMessage(payload.message, 2500)

    def _add_project_asset_image(self, image: QImage, *, metadata: dict[str, object] | None = None, status_message: str) -> None:
        document = ImageDocument(
            id=new_id("image"),
            path=self._next_project_capture_relative_path(),
            image_size=(image.width(), image.height()),
            source_type="project_asset",
            metadata=dict(metadata or {}),
        )
        document.initialize_runtime_state()
        if self.project.project_default_calibration is not None:
            self._set_document_project_default_calibration(document)
        document.mark_session_saved()
        document.mark_calibration_saved()
        self._mount_document(
            document,
            image,
            tooltip=self._document_tooltip(document),
        )
        self._clear_prompt_segmentation_cache()
        self.statusBar().showMessage(status_message, 4000)

    def _on_preview_analysis_worker_finished(self, payload: object) -> None:
        mode = self._preview_analysis_mode
        if mode == "none":
            return
        if isinstance(payload, FocusStackFinalResult):
            image = payload.image
            metadata = dict(payload.metadata)
            message = f"景深合成完成，已导入项目（采样 {payload.sampled_frames} / 接受 {payload.accepted_frames}）"
        elif isinstance(payload, MapBuildFinalResult):
            image = payload.image
            metadata = dict(payload.metadata)
            message = f"地图构建完成，已导入项目（tile {payload.tile_count}）"
        else:
            return
        self._teardown_preview_analysis_session(cancel_worker=False)
        if self._capture_manager.is_preview_active():
            self.stop_live_preview()
        self._add_project_asset_image(image, metadata=metadata, status_message=message)
        if mode != "none":
            self.statusBar().showMessage(message, 5000)

    def _on_preview_analysis_worker_failed(self, message: str) -> None:
        if self._preview_analysis_mode == "none":
            return
        title = self._analysis_mode_label(self._preview_analysis_mode)
        self._teardown_preview_analysis_session(cancel_worker=True)
        QMessageBox.warning(self, title, message)

    def _cycle_magic_segment_prompt_type(self) -> None:
        canvas = self.current_canvas()
        if canvas is None or not is_magic_segment_tool_mode(self._tool_mode) or canvas.is_magic_segment_busy():
            return
        if (
            canvas.current_magic_segment_operation_mode() == MagicSegmentOperationMode.SUBTRACT
            and canvas.current_magic_subtract_input_mode() != MagicSegmentSubtractInputMode.SMART
        ):
            return
        prompt_type = canvas.cycle_magic_segment_prompt_type()
        self.statusBar().showMessage(self._magic_prompt_label_text(prompt_type), 2500)
        self._update_magic_segment_controls()
        self._focus_current_canvas()

    def _cycle_active_magic_prompt_type(self) -> None:
        if is_magic_segment_tool_mode(self._tool_mode):
            self._cycle_magic_segment_prompt_type()
            return
        if is_fiber_quick_tool_mode(self._tool_mode):
            self._cycle_fiber_quick_prompt_type()

    def _cycle_fiber_quick_prompt_type(self) -> None:
        canvas = self.current_canvas()
        if canvas is None or not is_fiber_quick_tool_mode(self._tool_mode):
            return
        prompt_type = canvas.cycle_fiber_quick_prompt_type()
        self.statusBar().showMessage(self._magic_prompt_label_text(prompt_type), 2500)
        self._update_magic_segment_controls()
        self._focus_current_canvas()

    def _cycle_magic_segment_operation_mode(self) -> None:
        canvas = self.current_canvas()
        if canvas is None or not is_magic_segment_tool_mode(self._tool_mode) or canvas.is_magic_segment_busy():
            return
        if canvas.has_magic_manual_subtract_draft():
            if not self._confirm_discard_magic_manual_subtract_draft(
                "切换编辑状态",
                "当前剔除区域尚未闭合。要丢弃草稿并切换编辑状态吗？",
            ):
                self._update_magic_segment_controls()
                return
            canvas.cancel_magic_subtract_draft()
        before_mode = canvas.current_magic_segment_operation_mode()
        operation_mode = canvas.cycle_magic_segment_operation_mode()
        if operation_mode == MagicSegmentOperationMode.SUBTRACT:
            canvas.set_magic_subtract_input_mode(self._magic_standard_subtract_input_mode)
        if before_mode == MagicSegmentOperationMode.ADD and operation_mode == MagicSegmentOperationMode.ADD:
            self.statusBar().showMessage("请先完成第一个形状草稿", 2500)
        else:
            self.statusBar().showMessage(self._magic_operation_label_text(operation_mode), 2500)
        self._update_magic_segment_controls()
        self._focus_current_canvas()

    def _confirm_current_magic_subtract_shape(self) -> bool:
        canvas = self.current_canvas()
        if (
            canvas is None
            or not is_magic_segment_tool_mode(self._tool_mode)
            or canvas.is_magic_segment_busy()
            or canvas.current_magic_segment_operation_mode() != MagicSegmentOperationMode.SUBTRACT
        ):
            return False
        confirm_result = canvas.confirm_current_magic_subtract_shape()
        if not bool(confirm_result.get("confirmed", False)):
            return False
        self._hide_small_object_preview()
        count = int(confirm_result.get("count", 0) or 0)
        self.statusBar().showMessage(f"已确认剔除形状 {count} 块，可继续添加下一块。", 3000)
        self._update_magic_segment_controls()
        self._focus_current_canvas()
        return True

    def _commit_magic_segment_preview(self) -> bool:
        canvas = self.current_canvas()
        if canvas is None or not is_magic_segment_tool_mode(self._tool_mode) or canvas.is_magic_segment_busy():
            return False
        commit_result = canvas.commit_magic_segment_preview()
        committed = bool(commit_result.get("committed", False))
        messages: list[str] = []
        if committed:
            messages.append("已创建魔棒分割面积")
        elif bool(commit_result.get("result_empty", False)):
            messages.append("剔除后无剩余区域")
        elif str(commit_result.get("reason", "")) == "missing_primary":
            messages.append("请先完成第一个形状草稿")
        if bool(commit_result.get("discarded_fragments", False)):
            messages.append("结果裂成多个独立块，已按规则仅保留最大连通区域。")
        if messages:
            self.statusBar().showMessage(" ".join(messages), 4000)
        if committed:
            self._hide_small_object_preview()
        self._update_magic_segment_controls()
        self._focus_current_canvas()
        return committed

    def _commit_active_magic_preview(self) -> bool:
        if is_magic_segment_tool_mode(self._tool_mode):
            return self._commit_magic_segment_preview()
        if is_fiber_quick_tool_mode(self._tool_mode):
            return self._commit_fiber_quick_preview()
        if is_reference_propagation_tool_mode(self._tool_mode):
            return self._commit_reference_instance_preview()
        return False

    def _commit_reference_instance_preview(self) -> bool:
        canvas = self.current_canvas()
        document = self.current_document()
        if canvas is None or document is None or not is_reference_propagation_tool_mode(self._tool_mode) or canvas.is_reference_instance_busy():
            return False
        commit_result = canvas.commit_reference_instance_preview()
        candidates = list(commit_result.get("candidates", []))
        if not candidates:
            self.statusBar().showMessage("没有可加入当前类别的候选实例。", 4000)
            self._update_magic_segment_controls()
            self._focus_current_canvas()
            return False
        target_group = document.get_group(document.active_group_id) or document.ensure_default_group()
        added_count = 0
        skipped_count = 0

        def mutate() -> None:
            nonlocal target_group, added_count, skipped_count
            target_group = document.get_group(target_group.id) or document.ensure_default_group()
            existing_areas = [
                measurement
                for measurement in document.measurements
                if measurement.measurement_kind == "area" and measurement.fiber_group_id == target_group.id
            ]
            for candidate in candidates:
                polygon_px = list(candidate.get("polygon_px", []))
                area_rings_px = [list(ring) for ring in candidate.get("area_rings_px", [])]
                if len(polygon_px) < 3 and not area_rings_px:
                    continue
                overlaps_existing = any(
                    area_geometry_iou(
                        polygon_px,
                        area_rings_px,
                        measurement.polygon_px,
                        measurement.area_rings_px,
                    ) >= 0.7
                    for measurement in existing_areas
                )
                if overlaps_existing:
                    skipped_count += 1
                    continue
                measurement = Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=target_group.id,
                    mode="reference_instance",
                    measurement_kind="area",
                    polygon_px=polygon_px,
                    area_rings_px=area_rings_px,
                    confidence=float(candidate.get("confidence", 0.0)),
                    status="reference_instance",
                )
                document.add_measurement(measurement)
                existing_areas.append(measurement)
                added_count += 1
            document.select_overlay_annotation(None)

        self._apply_document_change(document, "导入同类扩选结果", mutate)
        if added_count > 0:
            message = f"已加入 {added_count} 个同类实例"
            if skipped_count > 0:
                message += f"，跳过 {skipped_count} 个重复候选"
            self.statusBar().showMessage(message, 5000)
        else:
            self.statusBar().showMessage("候选与当前类别结果重复，未新增实例。", 5000)
        self._update_magic_segment_controls()
        self._focus_current_canvas()
        return added_count > 0

    def _commit_fiber_quick_preview(self) -> bool:
        canvas = self.current_canvas()
        if canvas is None or not is_fiber_quick_tool_mode(self._tool_mode):
            return False
        if canvas._fiber_quick.segmentation_busy:  # noqa: SLF001
            self.statusBar().showMessage("分割尚未完成，请稍候。", 2500)
            self._update_magic_segment_controls()
            self._focus_current_canvas()
            return False
        commit_result = canvas.commit_fiber_quick_preview()
        committed = bool(commit_result.get("committed", False))
        pending = bool(commit_result.get("pending", False))
        if committed:
            self.statusBar().showMessage("已创建快速测径线段", 4000)
        elif pending:
            snapshot = commit_result.get("snapshot")
            if isinstance(snapshot, dict):
                self._enqueue_fiber_quick_background_job(canvas.document_id, snapshot)
            self.statusBar().showMessage("已确认当前分割，直径线计算完成后将自动写入。", 3000)
        else:
            self.statusBar().showMessage("当前没有可确认的快速测径结果。", 3000)
        self._update_magic_segment_controls()
        self._focus_current_canvas()
        return committed or pending

    def _enqueue_fiber_quick_background_job(self, document_id: str | None, snapshot: dict[str, object]) -> None:
        if not document_id:
            return
        document = self.project.get_document(document_id)
        if document is None:
            return
        if self._fiber_quick_geometry_worker is not None:
            self._fiber_quick_geometry_worker.cancel_document(document_id)
        self._ensure_fiber_quick_commit_geometry_worker()
        if self._fiber_quick_commit_geometry_worker is None:
            return
        self._fiber_quick_background_job_serial += 1
        job_id = self._fiber_quick_background_job_serial
        self._fiber_quick_background_jobs[(document_id, job_id)] = {
            "fiber_group_id": document.active_group_id,
            "debug_payload": dict(snapshot.get("debug_payload", {})),
        }
        self._fiber_quick_commit_geometry_worker.register_request(document_id, job_id)
        self._fiber_quick_commit_geometry_worker.requested.emit(
            FiberQuickGeometryRequest(
                document_id=document_id,
                request_id=job_id,
                mask=snapshot.get("mask"),
                preview_polygon_px=list(snapshot.get("polygon_px", [])) if isinstance(snapshot.get("polygon_px"), list) else [],
                preview_area_rings_px=[list(ring) for ring in snapshot.get("area_rings_px", [])] if isinstance(snapshot.get("area_rings_px"), list) else [],
                positive_points=list(snapshot.get("positive_points", [])) if isinstance(snapshot.get("positive_points"), list) else [],
                negative_points=list(snapshot.get("negative_points", [])) if isinstance(snapshot.get("negative_points"), list) else [],
                edge_trim_enabled=bool(self._app_settings.fiber_quick_edge_trim_enabled),
                line_extension_px=float(self._app_settings.fiber_quick_line_extension_px),
                timeout_ms=DEFAULT_FIBER_QUICK_GEOMETRY_TIMEOUT_MS,
            )
        )

    def _on_fiber_quick_commit_geometry_succeeded(self, document_id: str, request_id: int, result: object) -> None:
        job_meta = self._fiber_quick_background_jobs.pop((document_id, request_id), None)
        if job_meta is None or not hasattr(result, "line_px") or not isinstance(getattr(result, "line_px", None), Line):
            return
        document = self.project.get_document(document_id)
        if document is None:
            return
        merged_debug_payload = dict(job_meta.get("debug_payload", {}))
        if isinstance(getattr(result, "debug_payload", None), dict):
            merged_debug_payload.update(getattr(result, "debug_payload", {}))

        def mutate() -> None:
            measurement = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=job_meta.get("fiber_group_id"),
                mode="fiber_quick",
                line_px=result.line_px,
                confidence=float(getattr(result, "confidence", 0.0) or 0.0),
                status=str(getattr(result, "status", "fiber_quick") or "fiber_quick"),
                debug_payload=merged_debug_payload,
            )
            document.add_measurement(measurement)
            document.select_overlay_annotation(None)

        self._apply_document_change(document, "新增测量", mutate)
        self.statusBar().showMessage("快速测径已在后台完成并写入。", 3000)

    def _on_fiber_quick_commit_geometry_failed(self, document_id: str, request_id: int, reason: str) -> None:
        job_meta = self._fiber_quick_background_jobs.pop((document_id, request_id), None)
        if job_meta is None:
            return
        self.statusBar().showMessage(f"快速测径后台失败: {reason}", 4000)

    def _cancel_magic_segment_session(self) -> None:
        canvas = self.current_canvas()
        if canvas is None or not is_magic_segment_tool_mode(self._tool_mode):
            return
        if self._prompt_seg_worker is not None and canvas.document_id is not None:
            self._prompt_seg_worker.cancel_document(canvas.document_id)
        if canvas.cancel_magic_subtract_draft():
            self.statusBar().showMessage("已取消当前剔除草稿", 2500)
            self._update_magic_segment_controls()
            self._focus_current_canvas()
            return
        if canvas.has_magic_segment_session():
            canvas.clear_magic_segment_session()
            self._hide_small_object_preview()
            self.statusBar().showMessage("已放弃当前魔棒遮罩", 2500)
        self._update_magic_segment_controls()
        self._focus_current_canvas()

    def _cancel_active_magic_session(self) -> None:
        if is_magic_segment_tool_mode(self._tool_mode):
            self._cancel_magic_segment_session()
            return
        if is_fiber_quick_tool_mode(self._tool_mode):
            self._cancel_fiber_quick_session()
            return
        if is_reference_propagation_tool_mode(self._tool_mode):
            self._cancel_reference_instance_session()

    def _cancel_reference_instance_session(self) -> None:
        canvas = self.current_canvas()
        if canvas is None or not is_reference_propagation_tool_mode(self._tool_mode):
            return
        if canvas.has_reference_instance_session():
            canvas.clear_reference_instance_session()
            self.statusBar().showMessage("已放弃当前同类扩选", 2500)
        self._update_magic_segment_controls()
        self._focus_current_canvas()

    def _cancel_fiber_quick_session(self) -> None:
        canvas = self.current_canvas()
        if canvas is None or not is_fiber_quick_tool_mode(self._tool_mode):
            return
        if self._fiber_quick_geometry_worker is not None and canvas.document_id is not None:
            self._fiber_quick_geometry_worker.cancel_document(canvas.document_id)
        if self._prompt_seg_worker is not None and canvas.document_id is not None:
            self._prompt_seg_worker.cancel_document(canvas.document_id)
        if canvas.has_fiber_quick_session():
            canvas.clear_fiber_quick_session()
            self.statusBar().showMessage("已放弃当前快速测径", 2500)
        self._update_magic_segment_controls()
        self._focus_current_canvas()

    def _clear_magic_segment_sessions(self, *, except_document_id: str | None = None) -> None:
        should_hide_small_object_preview = except_document_id is None
        for document_id, canvas in self._canvases.items():
            if document_id == except_document_id:
                continue
            if canvas.has_magic_segment_session():
                if self._prompt_seg_worker is not None:
                    self._prompt_seg_worker.cancel_document(document_id)
                canvas.clear_magic_segment_session()
                should_hide_small_object_preview = True
            if canvas.has_reference_instance_session():
                canvas.clear_reference_instance_session()
            if canvas.has_fiber_quick_session():
                if self._fiber_quick_geometry_worker is not None:
                    self._fiber_quick_geometry_worker.cancel_document(document_id)
                if self._prompt_seg_worker is not None:
                    self._prompt_seg_worker.cancel_document(document_id)
                canvas.clear_fiber_quick_session()
        if should_hide_small_object_preview:
            self._hide_small_object_preview()

    def _overlay_metrics(self, width: int, height: int, render_mode: str) -> dict[str, float]:
        metrics = overlay_metrics(width, height, render_mode)
        return {
            "line_width": metrics.line_width,
            "endpoint_radius": metrics.endpoint_radius,
            "scale_bg_width": metrics.scale_bg_width,
            "scale_fg_width": metrics.scale_fg_width,
            "font_px": metrics.font_px,
        }

    def _create_export_surface(self, width: int, height: int) -> QImage:
        image = QImage(max(1, width), max(1, height), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor("#00000000"))
        return image

    def _render_overlay_image(
        self,
        document: ImageDocument,
        output_path: Path,
        *,
        include_measurements: bool,
        include_scale: bool,
        render_mode: str,
    ) -> None:
        if document.id not in self._images:
            return
        source_image = self._images[document.id]
        screen_scale = max(0.05, document.view_state.zoom or 1.0)

        if render_mode == ExportImageRenderMode.FULL_RESOLUTION:
            image = self._create_export_surface(source_image.width(), source_image.height())
            image_to_output_scale = 1.0

            def image_to_output(point) -> QPointF:
                return QPointF(point.x, point.y)
        elif render_mode == ExportImageRenderMode.CURRENT_VIEWPORT:
            canvas = self._canvases.get(document.id)
            viewport_width = max(200, canvas.width()) if canvas is not None else max(400, min(1400, source_image.width()))
            viewport_height = max(160, canvas.height()) if canvas is not None else max(300, min(900, source_image.height()))
            image = self._create_export_surface(viewport_width, viewport_height)
            image_to_output_scale = screen_scale

            def image_to_output(point) -> QPointF:
                return QPointF(
                    document.view_state.pan.x + (point.x * screen_scale),
                    document.view_state.pan.y + (point.y * screen_scale),
                )
        else:
            output_width = max(1, int(round(source_image.width() * screen_scale)))
            output_height = max(1, int(round(source_image.height() * screen_scale)))
            image = self._create_export_surface(output_width, output_height)
            image_to_output_scale = screen_scale

            def image_to_output(point) -> QPointF:
                return QPointF(point.x * screen_scale, point.y * screen_scale)

        painter = QPainter(image)
        if not painter.isActive():
            raise RuntimeError("无法创建可绘制的导出画布。")
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        if render_mode == ExportImageRenderMode.FULL_RESOLUTION:
            painter.drawImage(QPointF(0.0, 0.0), source_image)
        elif render_mode == ExportImageRenderMode.CURRENT_VIEWPORT:
            painter.fillRect(image.rect(), QColor("#101820"))
            target_rect = QRectF(
                document.view_state.pan.x,
                document.view_state.pan.y,
                source_image.width() * screen_scale,
                source_image.height() * screen_scale,
            )
            painter.drawImage(target_rect, source_image)
        else:
            painter.drawImage(
                QRectF(0.0, 0.0, image.width(), image.height()),
                source_image,
                QRectF(0.0, 0.0, source_image.width(), source_image.height()),
            )

        metrics = self._overlay_metrics(image.width(), image.height(), render_mode)
        line_width = metrics["line_width"]
        endpoint_radius = metrics["endpoint_radius"]
        scale_bg_width = metrics["scale_bg_width"]
        scale_fg_width = metrics["scale_fg_width"]
        font_px = metrics["font_px"]

        if include_measurements:
            draw_measurements(
                painter,
                document,
                image_to_output,
                self._app_settings,
                line_width=line_width,
                endpoint_radius=endpoint_radius,
                show_area_fill=self._show_area_fill,
            )

        if include_measurements or include_scale:
            draw_overlay_annotations(
                painter,
                document,
                image_to_output,
                self._app_settings,
                selected_overlay_id=None,
                render_mode=render_mode,
            )

        if include_scale:
            if (
                self._app_settings.scale_overlay_placement_mode == ScaleOverlayPlacementMode.MANUAL
                and document.scale_overlay_anchor is None
            ):
                self.statusBar().showMessage(
                    f"{Path(document.path).name} 尚未指定手动比例尺位置，已回退到左下角导出。",
                    5000,
                )
            draw_scale_overlay(
                painter,
                document,
                self._app_settings,
                image_width=image.width(),
                image_height=image.height(),
                image_to_output_scale=image_to_output_scale,
                scale_bg_width=scale_bg_width,
                scale_fg_width=scale_fg_width,
                font_px=font_px,
                render_mode=render_mode,
            )

        painter.end()
        if not image.save(str(output_path)):
            raise OSError(f"无法写入导出文件：{output_path}")

    def _color_icon(self, color_value: str, *, size: int = 12) -> QIcon:
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(color_value))
        return QIcon(pixmap)

    def _contrast_color(self, color_value: str) -> str:
        color = QColor(color_value)
        luminance = (0.299 * color.red()) + (0.587 * color.green()) + (0.114 * color.blue())
        return "#111111" if luminance > 186 else "#FFFFFF"

    def keyPressEvent(self, event) -> None:
        canvas = self.current_canvas()
        if event.key() == Qt.Key.Key_Space:
            if canvas is not None:
                canvas.set_temporary_grab_pressed(True)
            event.accept()
            return
        if (
            self._should_handle_digital_slide_jog_hotkeys()
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down)
        ):
            if not getattr(event, "isAutoRepeat", lambda: False)():
                mapping = {
                    Qt.Key.Key_Left: (AXIS_X, DIR_NEG),
                    Qt.Key.Key_Right: (AXIS_X, DIR_POS),
                    Qt.Key.Key_Up: (AXIS_Y, DIR_POS),
                    Qt.Key.Key_Down: (AXIS_Y, DIR_NEG),
                }
                axis, direction = mapping[event.key()]
                self._begin_digital_slide_jog(axis, direction)
            event.accept()
            return
        if self._preview_analysis_mode != "none" and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F):
                self._finalize_preview_analysis_session()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Escape:
                self._cancel_preview_analysis_session()
                event.accept()
                return
        if self._tool_mode in {"polygon_area", "freehand_area", "continuous_manual"} and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            if self._tool_mode in {"polygon_area", "freehand_area"} and event.key() == Qt.Key.Key_T:
                self._cycle_area_edit_operation_mode()
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F):
                if self._commit_active_path_drawing():
                    event.accept()
                    return
            if event.key() == Qt.Key.Key_Escape:
                if self._cancel_active_path_drawing():
                    event.accept()
                    return
        if is_magic_toolbar_tool_mode(self._tool_mode) and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            if is_magic_segment_tool_mode(self._tool_mode):
                if event.key() == Qt.Key.Key_R:
                    self._cycle_magic_segment_prompt_type()
                    event.accept()
                    return
                if event.key() == Qt.Key.Key_Y:
                    self._toggle_active_magic_roi()
                    event.accept()
                    return
                if event.key() == Qt.Key.Key_T:
                    self._cycle_magic_segment_operation_mode()
                    event.accept()
                    return
                if (
                    canvas is not None
                    and canvas.current_magic_segment_operation_mode() == MagicSegmentOperationMode.SUBTRACT
                    and event.key() in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3)
                ):
                    mode_by_key = {
                        Qt.Key.Key_1: MagicSegmentSubtractInputMode.SMART,
                        Qt.Key.Key_2: MagicSegmentSubtractInputMode.POLYGON,
                        Qt.Key.Key_3: MagicSegmentSubtractInputMode.FREEHAND,
                    }
                    self._set_magic_subtract_input_mode(mode_by_key[event.key()])
                    event.accept()
                    return
                if (
                    event.key() == Qt.Key.Key_S
                    and canvas is not None
                    and canvas.current_magic_segment_operation_mode() == MagicSegmentOperationMode.SUBTRACT
                ):
                    if self._confirm_current_magic_subtract_shape():
                        event.accept()
                        return
                    return
                if event.key() == Qt.Key.Key_S:
                    return
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F):
                    if canvas is not None and canvas.complete_magic_manual_subtract_draft():
                        self.statusBar().showMessage("已生成剔除草稿，可点击加入剔除继续", 2500)
                        self._update_magic_segment_controls()
                        event.accept()
                        return
                    self._commit_magic_segment_preview()
                    event.accept()
                    return
                if event.key() == Qt.Key.Key_Escape:
                    self._cancel_magic_segment_session()
                    event.accept()
                    return
            elif is_fiber_quick_tool_mode(self._tool_mode):
                if event.key() == Qt.Key.Key_R:
                    self._cycle_fiber_quick_prompt_type()
                    event.accept()
                    return
                if event.key() == Qt.Key.Key_Y:
                    self._toggle_active_magic_roi()
                    event.accept()
                    return
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F):
                    self._commit_fiber_quick_preview()
                    event.accept()
                    return
                if event.key() == Qt.Key.Key_Escape:
                    self._cancel_fiber_quick_session()
                    event.accept()
                    return
            elif is_reference_propagation_tool_mode(self._tool_mode):
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F):
                    self._commit_reference_instance_preview()
                    event.accept()
                    return
                if event.key() == Qt.Key.Key_Escape:
                    self._cancel_reference_instance_session()
                    event.accept()
                    return
        if event.modifiers() == Qt.KeyboardModifier.NoModifier and event.key() == Qt.Key.Key_A:
            if self._tool_mode == "select":
                if self._last_non_select_tool and self._last_non_select_tool != "select":
                    self.set_tool_mode(self._last_non_select_tool)
            else:
                self._last_non_select_tool = self._tool_mode
                self.set_tool_mode("select")
            event.accept()
            return
        if (
            event.modifiers() == Qt.KeyboardModifier.NoModifier
            and event.key() == Qt.Key.Key_V
            and self._should_handle_group_hotkeys()
        ):
            self._show_area_fill = not self._show_area_fill
            for canvas in self._canvases.values():
                canvas.set_show_area_fill(self._show_area_fill)
            self.statusBar().showMessage("面积填充已开启" if self._show_area_fill else "面积填充已关闭，仅显示轮廓", 3000)
            event.accept()
            return
        if (
            event.modifiers() == Qt.KeyboardModifier.NoModifier
            and Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9
            and self._should_handle_group_hotkeys()
        ):
            number = event.key() - Qt.Key.Key_0
            if self._switch_active_group_by_number(number):
                event.accept()
                return
        if self._tool_mode != "calibration" and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_measurement()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            canvas = self.current_canvas()
            if canvas is not None:
                canvas.set_temporary_grab_pressed(False)
            event.accept()
            return
        if (
            self._should_handle_digital_slide_jog_hotkeys()
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down)
        ):
            if not getattr(event, "isAutoRepeat", lambda: False)():
                self._end_digital_slide_jog()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _shutdown_background_threads(self) -> None:
        self._stop_digital_slide_writer(cancel=True)
        if self._preview_analysis_mode != "none":
            self._teardown_preview_analysis_session(cancel_worker=True)
        document_ids = list(self._canvases.keys())
        commit_document_ids = {
            document_id
            for document_id, _request_id in self._fiber_quick_background_jobs.keys()
        }
        commit_document_ids.update(document_ids)
        self.background_task_controller.shutdown_all(
            document_ids=document_ids,
            commit_document_ids=list(commit_document_ids),
        )
        self._fiber_quick_geometry_request_ids.clear()
        self._fiber_quick_background_jobs.clear()
        self._prompt_request_tool_modes.clear()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_close_documents(self.project.documents):
            event.ignore()
            return
        self._persist_window_geometry()
        self._hide_small_object_preview()
        self.stop_live_preview()
        self._clear_prompt_segmentation_cache()
        self._shutdown_background_threads()
        event.accept()
