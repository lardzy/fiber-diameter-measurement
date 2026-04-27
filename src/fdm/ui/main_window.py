from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QEventLoop, QPoint, QPointF, QRectF, QSize, Qt, QThread, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QCloseEvent, QGuiApplication, QIcon, QImage, QImageReader, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
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
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
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
)

from fdm import __version__
from fdm.area_display import ensure_measurement_display_geometry, invalidate_measurement_display_geometry
from fdm.content_experiment import (
    ContentExperimentRecord,
    ContentExperimentSession,
    ContentOverlayStyle,
    ContentRecordKind,
    ContentSelectionMode,
    content_session_stats,
    content_total_count,
    content_total_measured,
    session_from_project_metadata,
    write_session_to_project_metadata,
)
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
    project_assets_root,
    project_capture_root,
)
from fdm.project_io import ProjectIO
from fdm.settings import (
    AppSettings,
    AppSettingsIO,
    FocusStackProfile,
    MagicSegmentToolMode,
    OpenImageViewMode,
    ScaleOverlayPlacementMode,
    is_fiber_quick_tool_mode,
    is_magic_toolbar_tool_mode,
    is_magic_segment_tool_mode,
    is_reference_propagation_tool_mode,
)
from fdm.services.area_inference import AreaInferenceService, parse_area_model_labels
from fdm.services.export_service import ExportImageRenderMode, ExportScope, ExportSelection, ExportService
from fdm.services.fiber_quick_geometry import DEFAULT_FIBER_QUICK_GEOMETRY_TIMEOUT_MS
from fdm.services.content_workbook import ContentWorkbookService
from fdm.services.preview_analysis import (
    FocusStackFinalResult,
    FocusStackRenderConfig,
    FocusStackReport,
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
from fdm.ui.canvas import DocumentCanvas, MagicSegmentOperationMode
from fdm.ui.dialogs import (
    AreaAutoRecognitionDialog,
    CalibrationInputDialog,
    CalibrationPresetDialog,
    ContentFiberSelectionDialog,
    ExportOptionsDialog,
    FiberGroupDialog,
    SettingsDialog,
    ShortcutHelpDialog,
)
from fdm.ui.area_inference_worker import AreaBatchInferenceWorker, AreaInferenceRequest
from fdm.ui.icons import application_icon, themed_icon
from fdm.ui.image_loader import ImageBatchLoaderWorker, ImageLoadRequest
from fdm.ui.microview_preview_host import MicroviewPreviewHost
from fdm.ui.preview_analysis_dialog import PreviewAnalysisDialog
from fdm.ui.preview_analysis_worker import FocusStackSessionWorker, MapBuildSessionWorker
from fdm.ui.prompt_segmentation_worker import PromptSegmentationRequest, PromptSegmentationWorker
from fdm.ui.reference_instance_worker import (
    ReferenceInstancePropagationRequest,
    ReferenceInstancePropagationWorker,
)
from fdm.ui.fiber_quick_geometry_worker import FiberQuickGeometryRequest, FiberQuickGeometryWorker
from fdm.ui.rendering import draw_measurements, draw_overlay_annotations, draw_scale_overlay, overlay_metrics
from fdm.ui.theme import apply_application_theme, refresh_widget_theme
from fdm.ui.widgets import (
    ContentFiberListItemWidget,
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
class BatchLoadState:
    context_label: str
    total: int
    skipped_count: int = 0
    completed_count: int = 0
    loaded_count: int = 0
    failed_count: int = 0
    cancelled: bool = False
    failures: list[str] | None = None
    missing_paths: list[str] | None = None


@dataclass(slots=True)
class AreaInferenceBatchState:
    total: int
    model_name: str = ""
    update_project_group_templates: bool = True
    completed_count: int = 0
    failed_count: int = 0
    cancelled: bool = False
    failures: list[str] | None = None
    global_group_labels: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PresetImportPlanEntry:
    preset: CalibrationPreset
    action: str
    final_name: str


class MainWindow(QMainWindow):
    IMAGE_FILTER = "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    PROJECT_FILTER = "Fiber 项目 (*.fdmproj)"
    SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    MAP_BUILD_AVAILABLE = False
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
        self._magic_standard_roi_enabled = bool(self._app_settings.magic_segment_standard_roi_enabled)
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
        self._left_panel_splitter: QSplitter | None = None
        self._right_panel: QWidget | None = None
        self._group_header_labels: list[QLabel] = []
        self._load_thread: QThread | None = None
        self._load_worker: ImageBatchLoaderWorker | None = None
        self._load_progress_dialog: QProgressDialog | None = None
        self._load_state: BatchLoadState | None = None
        self._area_infer_thread: QThread | None = None
        self._area_infer_worker: AreaBatchInferenceWorker | None = None
        self._area_infer_progress_dialog: QProgressDialog | None = None
        self._area_infer_state: AreaInferenceBatchState | None = None
        self._prompt_seg_thread: QThread | None = None
        self._prompt_seg_worker: PromptSegmentationWorker | None = None
        self._fiber_quick_geometry_thread: QThread | None = None
        self._fiber_quick_geometry_worker: FiberQuickGeometryWorker | None = None
        self._fiber_quick_commit_geometry_thread: QThread | None = None
        self._fiber_quick_commit_geometry_worker: FiberQuickGeometryWorker | None = None
        self._reference_instance_thread: QThread | None = None
        self._reference_instance_worker: ReferenceInstancePropagationWorker | None = None
        self._prompt_request_tool_modes: dict[tuple[str, int], str] = {}
        self._fiber_quick_geometry_request_ids: set[tuple[str, int]] = set()
        self._fiber_quick_background_job_serial = 0
        self._fiber_quick_background_jobs: dict[tuple[str, int], dict[str, object]] = {}
        self._interactive_segmentation_services: dict[str, object] = {}
        self._show_area_fill = True
        self._content_mode_enabled = False
        self._area_model_box: QGroupBox | None = None
        self._area_auto_button: QPushButton | None = None
        self._magic_controls_widget: QWidget | None = None
        self._magic_prompt_label: QLabel | None = None
        self._magic_toggle_button: QToolButton | None = None
        self._magic_roi_button: QToolButton | None = None
        self._magic_operation_button: QToolButton | None = None
        self._magic_confirm_subtract_button: QToolButton | None = None
        self._magic_complete_button: QToolButton | None = None
        self._magic_cancel_button: QToolButton | None = None
        self._preview_analysis_widget: QWidget | None = None
        self._path_controls_widget: QWidget | None = None
        self._path_complete_button: QToolButton | None = None
        self._path_cancel_button: QToolButton | None = None
        self._focus_stack_button: QToolButton | None = None
        self._map_build_button: QToolButton | None = None
        self._map_build_status_label: QLabel | None = None
        self._add_preset_button: QPushButton | None = None
        self._edit_preset_button: QPushButton | None = None
        self._delete_preset_button: QPushButton | None = None
        self._import_cu_preset_button: QPushButton | None = None
        self._apply_preset_button: QPushButton | None = None
        self._add_group_button: QPushButton | None = None
        self._rename_group_button: QPushButton | None = None
        self.delete_group_button: QPushButton | None = None
        self._left_top_stack: QStackedWidget | None = None
        self._image_box: QGroupBox | None = None
        self._content_info_box: QGroupBox | None = None
        self._content_operator_edit: QLineEdit | None = None
        self._content_sample_id_edit: QLineEdit | None = None
        self._content_sample_name_edit: QLineEdit | None = None
        self._content_mode_preselect_radio: QRadioButton | None = None
        self._content_mode_postselect_radio: QRadioButton | None = None
        self._content_overlay_combo: QComboBox | None = None
        self._content_totals_label: QLabel | None = None
        self._content_box: QGroupBox | None = None
        self._content_start_button: QPushButton | None = None
        self._content_close_button: QPushButton | None = None
        self._content_save_excel_button: QPushButton | None = None
        self._content_status_label: QLabel | None = None
        self._content_record_table: QTableWidget | None = None
        self._content_delete_record_button: QPushButton | None = None
        self._delete_group_measurements_button: QPushButton | None = None
        self._delete_all_measurements_button: QPushButton | None = None
        self._center_stack: QStackedWidget | None = None
        self._preview_page: QWidget | None = None
        self._preview_display_stack: QStackedWidget | None = None
        self._preview_canvas: DocumentCanvas | None = None
        self._microview_preview_host: MicroviewPreviewHost | None = None
        self._microview_preview_scroll: QScrollArea | None = None
        self._content_frame_canvas: DocumentCanvas | None = None
        self._content_microview_preview_host: MicroviewPreviewHost | None = None
        self._content_overlay_canvas: DocumentCanvas | None = None
        self._content_native_preview_container: QWidget | None = None
        self._content_microview_preview_scroll: QScrollArea | None = None
        self._preview_status_label: QLabel | None = None
        self._image_resolution_label: QLabel | None = None
        self._preview_notice_label: QLabel | None = None
        self._calibration_label_scroll: QScrollArea | None = None
        self._version_label: QLabel | None = None
        self._preview_active = False
        self._preview_document: ImageDocument | None = None
        self._content_preview_active = False
        self._content_preview_document: ImageDocument | None = None
        self._content_analysis_frame: QImage | None = None
        self._capture_devices: list[CaptureDevice] = []
        self._content_capture_devices: list[CaptureDevice] = []
        self._microview_optimize_hints_shown: set[str] = set()
        self._preview_analysis_mode = "none"
        self._preview_analysis_dialog: PreviewAnalysisDialog | None = None
        self._preview_analysis_thread: QThread | None = None
        self._preview_analysis_worker: FocusStackSessionWorker | MapBuildSessionWorker | None = None
        self._preview_analysis_timer = QTimer(self)
        self._preview_analysis_timer.setInterval(300)
        self._preview_analysis_timer.timeout.connect(self._request_preview_analysis_frame)
        self._preview_analysis_request_id = 0
        self._preview_analysis_request_pending = False
        self._preview_analysis_finalizing = False
        self._content_session: ContentExperimentSession | None = None
        self._content_workbook_service = ContentWorkbookService()
        self._content_measure_start: Point | None = None
        self._content_measure_hover: Point | None = None
        self._content_fiber_menu_open = False
        self._content_pending_diameter_line: Line | None = None
        self._content_field_timer = QTimer(self)
        self._content_field_timer.setInterval(300)
        self._content_field_timer.timeout.connect(self._request_content_field_frame)
        self._content_native_overlay_timer = QTimer(self)
        self._content_native_overlay_timer.setInterval(33)
        self._content_native_overlay_timer.timeout.connect(self._refresh_content_native_overlay)
        self._content_field_request_id = -1
        self._content_field_request_pending = False
        self._content_field_baseline: object | None = None
        self._content_field_motion_hits = 0
        self._project_clean_snapshot: dict[str, object] | None = None
        self._pending_project_load_snapshot = False
        self._capture_manager = CaptureSessionManager(
            selected_device_id=self._app_settings.selected_capture_device_id,
            refresh_on_init=False,
        )
        self._content_capture_manager = CaptureSessionManager(
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

        self._build_ui()
        self._refresh_theme_sensitive_icons()
        self._capture_manager.devicesChanged.connect(self._on_capture_devices_changed)
        self._capture_manager.previewStateChanged.connect(self._on_live_preview_state_changed)
        self._capture_manager.frameReady.connect(self._on_live_preview_frame_ready)
        self._capture_manager.analysisFrameReady.connect(self._on_preview_analysis_frame_ready)
        self._capture_manager.analysisFrameFailed.connect(self._on_preview_analysis_frame_failed)
        self._capture_manager.errorOccurred.connect(self._on_capture_error)
        self._content_capture_manager.devicesChanged.connect(self._on_content_capture_devices_changed)
        self._content_capture_manager.previewStateChanged.connect(self._on_content_preview_state_changed)
        self._content_capture_manager.frameReady.connect(self._on_content_preview_frame_ready)
        self._content_capture_manager.analysisFrameReady.connect(self._on_content_analysis_frame_ready)
        self._content_capture_manager.analysisFrameFailed.connect(self._on_content_analysis_frame_failed)
        self._content_capture_manager.errorOccurred.connect(self._on_content_capture_error)
        self._capture_devices = self._capture_manager.devices()
        self._content_capture_devices = self._content_capture_manager.devices()
        self._refresh_preset_combo()
        self._update_capture_device_ui()
        self._restore_initial_window_geometry()
        self._update_ui_for_current_document()
        self._mark_project_saved()

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
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([280, 870, 360])
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

        self.content_experiment_action = QAction("含量试验", self)
        self.content_experiment_action.setCheckable(True)
        self.content_experiment_action.setIcon(themed_icon("area_auto", color="#F4D35E"))
        self.content_experiment_action.triggered.connect(self.toggle_content_experiment_mode)

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

        self.settings_action = QAction("设置", self)
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
            ("calibration", "比例尺标定"),
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

        edit_menu = self.menuBar().addMenu("编辑")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.delete_measurement_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.add_group_action)
        edit_menu.addAction(self.rename_group_action)
        edit_menu.addAction(self.delete_group_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.settings_action)

        view_menu = self.menuBar().addMenu("视图")
        for action in self._mode_actions.values():
            view_menu.addAction(action)
        view_menu.addSeparator()
        view_menu.addAction(self.fit_action)
        view_menu.addAction(self.actual_size_action)

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
        file_toolbar.addSeparator()
        file_toolbar.addAction(self.close_current_action)
        file_toolbar.addAction(self.close_all_action)
        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        file_toolbar.addWidget(spacer)
        file_toolbar.addSeparator()
        file_toolbar.addAction(self.switch_capture_device_action)
        file_toolbar.addAction(self.live_preview_action)
        file_toolbar.addAction(self.content_experiment_action)
        file_toolbar.addAction(self.capture_frame_action)
        file_toolbar.addAction(self.optimize_capture_signal_action)

        self._measure_toolbar = None
        self._measurement_tool_strip = self._build_measurement_tool_strip()

    def _build_measurement_tool_strip(self) -> MeasurementToolStrip:
        strip = MeasurementToolStrip(self)
        strip.addModeAction("select", self._mode_actions["select"])
        self._manual_tool_button = self._build_manual_tool_button()
        strip.addSplitModeButton("manual", self._manual_tool_button, aliases=["continuous_manual"])
        strip.addModeAction("count", self._mode_actions["count"])
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

    def _build_magic_segment_controls(self) -> QWidget:
        container = QWidget(self)
        layout = FlowLayout(container, h_spacing=6, v_spacing=6)
        container.setLayout(layout)
        self._magic_prompt_label = QLabel(container)
        self._magic_prompt_label.setProperty("contextLabel", True)
        layout.addWidget(self._magic_prompt_label)

        self._magic_toggle_button = QToolButton(container)
        self._magic_toggle_button.setProperty("contextTool", True)
        self._magic_toggle_button.setText("正负(R)")
        self._magic_toggle_button.clicked.connect(self._cycle_active_magic_prompt_type)
        layout.addWidget(self._magic_toggle_button)

        self._magic_roi_button = QToolButton(container)
        self._magic_roi_button.setProperty("contextTool", True)
        self._magic_roi_button.setCheckable(True)
        self._magic_roi_button.setText("ROI")
        self._magic_roi_button.clicked.connect(self._toggle_active_magic_roi)
        layout.addWidget(self._magic_roi_button)

        self._magic_operation_button = QToolButton(container)
        self._magic_operation_button.setProperty("contextTool", True)
        self._magic_operation_button.setText("添加(T)")
        self._magic_operation_button.clicked.connect(self._cycle_magic_segment_operation_mode)
        layout.addWidget(self._magic_operation_button)

        self._magic_confirm_subtract_button = QToolButton(container)
        self._magic_confirm_subtract_button.setProperty("contextTool", True)
        self._magic_confirm_subtract_button.setText("加块(S)")
        self._magic_confirm_subtract_button.setToolTip("确认当前剔除形状，并继续添加下一块剔除区域")
        self._magic_confirm_subtract_button.clicked.connect(self._confirm_current_magic_subtract_shape)
        layout.addWidget(self._magic_confirm_subtract_button)

        self._magic_complete_button = QToolButton(container)
        self._magic_complete_button.setProperty("contextTool", True)
        self._magic_complete_button.setText("完成")
        self._magic_complete_button.clicked.connect(self._commit_active_magic_preview)
        layout.addWidget(self._magic_complete_button)

        self._magic_cancel_button = QToolButton(container)
        self._magic_cancel_button.setProperty("contextTool", True)
        self._magic_cancel_button.setText("取消")
        self._magic_cancel_button.clicked.connect(self._cancel_active_magic_session)
        layout.addWidget(self._magic_cancel_button)

        return container

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

        header_button = QToolButton(container)
        header_button.setProperty("contextTool", True)
        header_button.setText("路径测量")
        header_button.setCursor(Qt.CursorShape.ArrowCursor)
        header_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        layout.addWidget(header_button)

        self._path_complete_button = QToolButton(container)
        self._path_complete_button.setProperty("contextTool", True)
        self._path_complete_button.setText("完成 (Enter / F)")
        self._path_complete_button.clicked.connect(self._commit_active_path_drawing)
        layout.addWidget(self._path_complete_button)

        self._path_cancel_button = QToolButton(container)
        self._path_cancel_button.setProperty("contextTool", True)
        self._path_cancel_button.setText("取消 (Esc)")
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
        if self._content_mode_enabled and tool_mode != "manual":
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
        if self._content_mode_enabled:
            self.set_tool_mode("select")
            return
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
        if self._content_mode_enabled and tool_mode != MagicSegmentToolMode.FIBER_QUICK:
            tool_mode = MagicSegmentToolMode.FIBER_QUICK
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
        if self._content_mode_enabled and active_mode != "manual":
            active_mode = "manual"
        icon = self._manual_tool_icon(active_mode, active=self._tool_mode in {mode for mode, _ in self._manual_tool_definitions()})
        tooltip = f"手动测量（当前：{self._manual_tool_label(active_mode)}）"
        if self._manual_tool_button is not None:
            self._manual_tool_button.blockSignals(True)
            self._manual_tool_button.setChecked(self._tool_mode in {mode for mode, _ in self._manual_tool_definitions()})
            self._manual_tool_button.setCurrentTool(active_mode, icon)
            self._manual_tool_button.setToolTip(tooltip)
            self._manual_tool_button.blockSignals(False)
        for tool_mode, action in self._manual_subtool_actions.items():
            action.setChecked(tool_mode == active_mode)
            action.setIcon(self._manual_tool_icon(tool_mode))
            action.setEnabled(not self._content_mode_enabled or tool_mode == "manual")

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
            self._area_tool_button.setChecked(self._tool_mode in {mode for mode, _ in self._area_tool_definitions()})
            self._area_tool_button.setCurrentTool(active_mode, icon)
            self._area_tool_button.setToolTip(tooltip)
            self._area_tool_button.blockSignals(False)
        for tool_mode, action in self._area_subtool_actions.items():
            action.setChecked(tool_mode == active_mode)
            action.setIcon(self._area_tool_icon(tool_mode))
            action.setEnabled(not self._content_mode_enabled)

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
        if self._content_mode_enabled and active_mode != MagicSegmentToolMode.FIBER_QUICK:
            active_mode = MagicSegmentToolMode.FIBER_QUICK
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
            action.setEnabled(not self._content_mode_enabled or tool_mode == MagicSegmentToolMode.FIBER_QUICK)

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
        if self._content_mode_enabled:
            self.set_tool_mode("select")
            return
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
            action.setEnabled(not self._content_mode_enabled)

    def _build_content_info_box(self) -> QGroupBox:
        box = QGroupBox("基础信息")
        layout = QVBoxLayout(box)
        form = QFormLayout()

        self._content_operator_edit = QLineEdit(self._app_settings.content_last_operator)
        self._content_sample_id_edit = QLineEdit()
        self._content_sample_name_edit = QLineEdit()
        for label, field_name, edit in [
            ("操作人", "operator", self._content_operator_edit),
            ("样品编号", "sample_id", self._content_sample_id_edit),
            ("样品名称", "sample_name", self._content_sample_name_edit),
        ]:
            edit.setReadOnly(True)
            edit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            edit.setPlaceholderText(f"点击“{label}”填写")
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, target=field_name: self._edit_content_basic_field(target))
            form.addRow(button, edit)

        self._content_mode_preselect_radio = QRadioButton("先选")
        self._content_mode_postselect_radio = QRadioButton("后选")
        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_group = QButtonGroup(mode_row)
        mode_group.addButton(self._content_mode_preselect_radio)
        mode_group.addButton(self._content_mode_postselect_radio)
        self._content_mode_preselect_radio.setChecked(True)
        self._content_mode_preselect_radio.toggled.connect(self._update_content_basic_info_from_ui)
        self._content_mode_postselect_radio.toggled.connect(self._update_content_basic_info_from_ui)
        mode_layout.addWidget(self._content_mode_preselect_radio)
        mode_layout.addWidget(self._content_mode_postselect_radio)
        mode_layout.addStretch(1)
        form.addRow("类别选择", mode_row)

        self._content_overlay_combo = QComboBox()
        self._content_overlay_combo.addItem("无", ContentOverlayStyle.NONE)
        self._content_overlay_combo.addItem("中心点", ContentOverlayStyle.CENTER_DOT)
        self._content_overlay_combo.addItem("横线", ContentOverlayStyle.HORIZONTAL)
        self._content_overlay_combo.addItem("竖线", ContentOverlayStyle.VERTICAL)
        self._content_overlay_combo.addItem("十字", ContentOverlayStyle.CROSS)
        self._content_overlay_combo.addItem("十字准星", ContentOverlayStyle.CROSSHAIR)
        self._content_overlay_combo.currentIndexChanged.connect(self._update_content_basic_info_from_ui)
        form.addRow("画布叠加", self._content_overlay_combo)

        layout.addLayout(form)
        self._content_totals_label = QLabel("总计数 0；实测 0")
        self._content_totals_label.setWordWrap(True)
        layout.addWidget(self._content_totals_label)
        hint = QLabel("含量试验中，数字键 1-8 直接给对应纤维计数；右键画布可切换当前纤维。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return box

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        self._left_panel = container
        container.setMinimumWidth(280)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        image_box = QGroupBox("已打开图片")
        self._image_box = image_box
        image_layout = QVBoxLayout(image_box)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_list_changed)
        image_layout.addWidget(self.image_list)
        content_info_box = self._build_content_info_box()
        self._content_info_box = content_info_box
        self._left_top_stack = QStackedWidget()
        self._left_top_stack.addWidget(image_box)
        self._left_top_stack.addWidget(content_info_box)

        group_box = QGroupBox("纤维类别")
        group_layout = QVBoxLayout(group_box)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(10, 0, FiberGroupListItemWidget.RIGHT_MARGIN, 0)
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
        self.group_list.setSpacing(4)
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
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._left_top_stack)
        splitter.addWidget(group_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([280, 420])
        layout.addWidget(splitter, 1)
        self._update_group_list_header_styles()
        return container

    def _connect_interactive_canvas(self, canvas: DocumentCanvas) -> None:
        canvas.lineCommitted.connect(self._on_canvas_line_committed)
        canvas.measurementSelected.connect(self._on_canvas_measurement_selected)
        canvas.measurementEdited.connect(self._on_canvas_measurement_edited)
        canvas.pathSessionChanged.connect(self._on_canvas_path_session_changed)
        canvas.overlayCreateRequested.connect(self._on_canvas_overlay_create_requested)
        canvas.overlaySelected.connect(self._on_canvas_overlay_selected)
        canvas.overlayEdited.connect(self._on_canvas_overlay_edited)
        canvas.scaleAnchorPicked.connect(self._on_canvas_scale_anchor_picked)
        canvas.magicSegmentRequested.connect(self._on_canvas_magic_segment_requested)
        canvas.magicSegmentSessionChanged.connect(self._on_canvas_magic_segment_session_changed)

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
        self._preview_display_stack.addWidget(self._microview_preview_scroll)

        self._content_frame_canvas = DocumentCanvas()
        self._content_frame_canvas.set_read_only(False)
        self._content_frame_canvas.set_fit_alignment("top_left")
        self._content_frame_canvas.set_show_area_fill(False)
        self._content_frame_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._content_frame_canvas.installEventFilter(self)
        self._connect_interactive_canvas(self._content_frame_canvas)
        self._preview_display_stack.addWidget(self._content_frame_canvas)

        self._content_native_preview_container = QWidget()
        self._content_native_preview_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._content_microview_preview_host = MicroviewPreviewHost(self._content_native_preview_container)
        self._content_microview_preview_host.set_prefer_gdi_preview(True)
        self._content_microview_preview_host.move(0, 0)
        self._content_microview_preview_host.installEventFilter(self)
        self._content_microview_preview_host.metricsChanged.connect(self._on_content_preview_host_metrics_changed)
        self._content_overlay_canvas = DocumentCanvas(self._content_native_preview_container)
        self._content_overlay_canvas.set_read_only(False)
        self._content_overlay_canvas.set_fit_alignment("top_left")
        self._content_overlay_canvas.set_show_area_fill(False)
        self._content_overlay_canvas.set_transparent_background(True)
        self._content_overlay_canvas.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        self._content_overlay_canvas.hide()
        self._content_overlay_canvas.installEventFilter(self)
        self._connect_interactive_canvas(self._content_overlay_canvas)
        self._content_microview_preview_scroll = QScrollArea()
        self._content_microview_preview_scroll.setWidget(self._content_native_preview_container)
        self._content_microview_preview_scroll.setWidgetResizable(False)
        self._content_microview_preview_scroll.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._content_microview_preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._preview_display_stack.addWidget(self._content_microview_preview_scroll)

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
        self._area_model_box = model_box
        model_layout = QVBoxLayout(model_box)
        self._area_auto_button = QPushButton("面积自动识别...")
        self._area_auto_button.setIcon(themed_icon("area_auto", color="#7BD389"))
        self._area_auto_button.clicked.connect(self.run_area_auto_recognition)
        model_layout.addWidget(self._area_auto_button)
        top_layout.addWidget(model_box)

        calibration_box = QGroupBox("标定")
        calibration_layout = QVBoxLayout(calibration_box)
        self._preview_notice_label = QLabel("实时预览中，图片编辑已禁用")
        self._preview_notice_label.setWordWrap(True)
        self._preview_notice_label.setStyleSheet("color: #F4D35E;")
        self._preview_notice_label.hide()
        calibration_layout.addWidget(self._preview_notice_label)
        self.calibration_label = QLabel("当前图片未标定")
        self.calibration_label.setWordWrap(True)
        self.calibration_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._calibration_label_scroll = QScrollArea()
        self._calibration_label_scroll.setWidget(self.calibration_label)
        self._calibration_label_scroll.setWidgetResizable(True)
        self._calibration_label_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._calibration_label_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._calibration_label_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._calibration_label_scroll.setMinimumHeight(88)
        self._calibration_label_scroll.setMaximumHeight(118)
        calibration_layout.addWidget(self._calibration_label_scroll)
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
        self._import_cu_preset_button = QPushButton("导入CU标尺")
        self._import_cu_preset_button.setIcon(themed_icon("preset_import", color="#D7E3FC"))
        self._import_cu_preset_button.clicked.connect(self.import_cu_calibration_presets)
        calibration_layout.addWidget(self._import_cu_preset_button)
        self._apply_preset_button = QPushButton("应用预设")
        self._apply_preset_button.setIcon(themed_icon("preset_apply", color="#D7E3FC"))
        self._apply_preset_button.clicked.connect(self.apply_selected_preset)
        calibration_layout.addWidget(self._apply_preset_button)
        top_layout.addWidget(calibration_box)

        content_box = QGroupBox("含量试验")
        self._content_box = content_box
        content_layout = QVBoxLayout(content_box)
        content_button_row = QHBoxLayout()
        self._content_start_button = QPushButton("开始/继续")
        self._content_start_button.setIcon(themed_icon("live_preview", color="#7BD389"))
        self._content_start_button.clicked.connect(self.start_or_continue_content_experiment)
        self._content_close_button = QPushButton("暂停")
        self._content_close_button.clicked.connect(self.close_content_experiment)
        self._content_save_excel_button = QPushButton("保存Excel")
        self._content_save_excel_button.clicked.connect(self.save_content_experiment_excel)
        content_button_row.addWidget(self._content_start_button)
        content_button_row.addWidget(self._content_close_button)
        content_button_row.addWidget(self._content_save_excel_button)
        content_layout.addLayout(content_button_row)
        self._content_status_label = QLabel("实时预览开启后可开始含量试验。")
        self._content_status_label.setWordWrap(True)
        content_layout.addWidget(self._content_status_label)
        top_layout.addWidget(content_box)

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
        layout.addWidget(right_splitter)

        return container

    def _make_export_action(self, label: str, selection: ExportSelection) -> QAction:
        action = QAction(label, self)
        action.triggered.connect(lambda checked=False, preset=selection: self.export_results(preset))
        return action

    def _content_allowed_tool_modes(self) -> set[str]:
        return {"select", "manual", "snap", MagicSegmentToolMode.FIBER_QUICK}

    def _coerce_content_tool_mode(self, mode: str) -> str:
        return mode if mode in self._content_allowed_tool_modes() else "select"

    def set_tool_mode(self, mode: str, *, overlay_kind: str | None = None) -> None:
        previous_mode = self._tool_mode
        if mode not in self._mode_actions:
            mode = "select"
        if self._content_mode_enabled:
            mode = self._coerce_content_tool_mode(mode)
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
        for canvas in [*self._canvases.values(), self._content_frame_canvas, self._content_overlay_canvas]:
            if canvas is None:
                continue
            canvas.set_tool_mode(mode, overlay_kind=self._overlay_tool_kind)
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
        self._update_path_drawing_controls()
        if self._content_mode_enabled:
            self._refresh_preview_surface()

    def current_document(self) -> ImageDocument | None:
        if self._preview_active or self._content_mode_enabled:
            return None
        index = self.tab_widget.currentIndex()
        if index < 0 or index >= len(self._document_order):
            return None
        return self.project.get_document(self._document_order[index])

    def current_canvas(self) -> DocumentCanvas | None:
        if self._content_mode_enabled:
            return self._content_current_canvas()
        if self._preview_active:
            if self._capture_manager.preview_kind() == "frame_stream":
                return self._preview_canvas
            return None
        document = self.current_document()
        if document is None:
            return None
        return self._canvases.get(document.id)

    def _content_current_canvas(self) -> DocumentCanvas | None:
        if self._content_preview_kind() == "native_embed":
            return self._content_frame_canvas if is_fiber_quick_tool_mode(self._tool_mode) else None
        return self._content_frame_canvas

    def _preview_kind(self) -> str:
        return self._capture_manager.preview_kind()

    def _is_native_preview(self) -> bool:
        return self._preview_kind() == "native_embed"

    def _content_preview_kind(self) -> str:
        return self._content_capture_manager.preview_kind()

    def _is_content_native_preview(self) -> bool:
        return self._content_preview_kind() == "native_embed"

    def _current_preview_target(self) -> object | None:
        if self._is_native_preview():
            return self._microview_preview_host
        return None

    def _current_content_preview_target(self) -> object | None:
        if self._is_content_native_preview():
            return self._content_microview_preview_host
        return None

    def _apply_preview_surface(self, preview_kind: str) -> None:
        if (
            self._preview_display_stack is None
            or self._preview_canvas is None
            or self._microview_preview_scroll is None
        ):
            return
        if self._content_mode_enabled:
            if (
                preview_kind == "native_embed"
                and self._content_microview_preview_scroll is not None
                and not is_fiber_quick_tool_mode(self._tool_mode)
            ):
                target_widget = self._content_microview_preview_scroll
            else:
                target_widget = self._content_frame_canvas
            if target_widget is None:
                return
        else:
            target_widget = self._microview_preview_scroll if preview_kind == "native_embed" else self._preview_canvas
        self._preview_display_stack.setCurrentWidget(target_widget)

    def _refresh_preview_surface(self) -> None:
        self._apply_preview_surface(self._content_preview_kind() if self._content_mode_enabled else self._preview_kind())

    def _on_preview_host_metrics_changed(self) -> None:
        if not self._preview_active or not self._is_native_preview() or self._microview_preview_host is None:
            return
        self._capture_manager.update_preview_target(self._microview_preview_host)

    def _on_content_preview_host_metrics_changed(self) -> None:
        if (
            not self._content_preview_active
            or not self._is_content_native_preview()
            or self._content_microview_preview_host is None
        ):
            return
        self._content_capture_manager.update_preview_target(self._content_microview_preview_host)

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
            if self._preview_notice_label is not None:
                warning = "#F4D35E" if self._is_dark_palette() else "#A66A00"
                self._preview_notice_label.setStyleSheet(f"color: {warning};")
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
        if self._preview_notice_label is not None:
            warning = "#F4D35E" if self._is_dark_palette() else "#A66A00"
            self._preview_notice_label.setStyleSheet(f"color: {warning};")
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

    def _set_calibration_label(self, text: str, *, status: str) -> None:
        color_key = {
            "uncalibrated": "danger",
            "calibrated": "info",
            "preview": "muted",
        }.get(status, "default")
        self.calibration_label.setText(text)
        self.calibration_label.setToolTip(text)
        self.calibration_label.setStyleSheet(f"color: {self._status_color(color_key)};")

    def _update_preset_combo_tooltip(self, text: str) -> None:
        self.preset_combo.setToolTip(text)

    def _update_image_resolution_label(self, document: ImageDocument | None = None) -> None:
        if self._image_resolution_label is None:
            return
        self._image_resolution_label.setStyleSheet(f"color: {self._status_color('muted')}; padding: 6px 2px 0 2px;")
        if self._content_mode_enabled:
            resolution = self._content_capture_manager.preview_resolution()
            if resolution is None and self._content_preview_document is not None:
                resolution = self._content_preview_document.image_size
            if resolution is None:
                self._image_resolution_label.setText("含量试验预览分辨率: -")
            else:
                self._image_resolution_label.setText(f"含量试验预览分辨率: {resolution[0]} x {resolution[1]} px")
            return
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

    def _apply_content_native_preview_resolution(self) -> None:
        if (
            self._content_microview_preview_host is None
            or self._content_overlay_canvas is None
            or self._content_native_preview_container is None
        ):
            return
        resolution = self._content_capture_manager.preview_resolution()
        if resolution is None:
            return
        width, height = resolution
        self._content_microview_preview_host.set_preview_resolution(width, height)
        self._content_overlay_canvas.setFixedSize(width, height)
        self._content_overlay_canvas.set_view_transform(zoom=1.0, pan=Point(0.0, 0.0))
        self._content_overlay_canvas.move(0, 0)
        self._content_overlay_canvas.hide()
        self._content_microview_preview_host.move(0, 0)
        self._content_native_preview_container.setFixedSize(width, height)
        if self._content_analysis_frame is None or self._content_analysis_frame.size() != QSize(width, height):
            placeholder = QImage(width, height, QImage.Format.Format_ARGB32)
            placeholder.fill(Qt.GlobalColor.transparent)
            self._update_content_preview_document(placeholder, native=True)
        if self._preview_status_label is not None:
            selected = self._selected_content_capture_device()
            label = selected.name if selected is not None else "采集设备"
            self._preview_status_label.setText(f"含量试验预览: {label}  ({width} x {height}, 原始分辨率)")
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
            raw_path = Path(str(document.path)).expanduser()
            if str(raw_path).strip():
                return raw_path.parent
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
            ".csv": "CSV 文件 (*.csv)",
        }.get(suffix, "所有文件 (*)")

    def _document_has_unsaved_project_changes(self, document: ImageDocument) -> bool:
        return document.dirty_flags.session_dirty or (not document.uses_sidecar() and document.dirty_flags.calibration_dirty)

    def _selected_capture_device(self) -> CaptureDevice | None:
        return self._capture_manager.selected_device()

    def _selected_content_capture_device(self) -> CaptureDevice | None:
        return self._content_capture_manager.selected_device()

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
        self._update_action_states()

    def _refresh_capture_devices(self) -> None:
        self._capture_devices = self._capture_manager.refresh_devices()
        selected = self._selected_capture_device()
        if selected is not None and not self._app_settings.selected_capture_device_id:
            self._app_settings.selected_capture_device_id = selected.id
        self._update_capture_device_ui()
        self._update_action_states()

    def _refresh_content_capture_devices(self) -> None:
        self._content_capture_devices = self._content_capture_manager.refresh_devices()
        selected_id = self._app_settings.selected_capture_device_id or self._capture_manager.selected_device_id()
        if selected_id:
            self._content_capture_manager.set_selected_device(selected_id)
        self._content_capture_devices = self._content_capture_manager.devices()

    def _set_selected_capture_device(self, device_id: str) -> None:
        restart_preview = self._capture_manager.is_preview_active()
        restart_content_preview = self._content_capture_manager.is_preview_active()
        if restart_preview:
            self.stop_live_preview()
        if restart_content_preview:
            self.stop_content_preview()
        if not self._capture_manager.set_selected_device(device_id):
            QMessageBox.warning(self, "切换采集设备", "无法切换到所选设备。")
            return
        if not self._content_capture_devices:
            self._content_capture_manager.refresh_devices()
        if self._content_capture_devices:
            self._content_capture_manager.set_selected_device(device_id)
        self._app_settings.selected_capture_device_id = device_id
        self._save_app_settings(context="切换采集设备")
        selected = self._selected_capture_device()
        if selected is not None:
            self.statusBar().showMessage(f"当前采集设备: {selected.name}", 4000)
        self._show_active_capture_warning()
        self._update_capture_device_ui()
        self._update_action_states()
        if restart_preview:
            self.start_live_preview()
        if restart_content_preview and self._content_mode_enabled:
            self.start_content_preview()

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
        if checked and self._content_mode_enabled:
            self._sync_live_preview_action()
            self.statusBar().showMessage("含量试验使用独立预览，不会选中顶部实时预览。", 4000)
            return
        if checked:
            self.start_live_preview()
            return
        self.stop_live_preview()

    def toggle_content_experiment_mode(self, checked: bool) -> None:
        if checked:
            self.enter_content_experiment()
            return
        self.close_content_experiment()

    def _sync_content_experiment_action(self) -> None:
        if not hasattr(self, "content_experiment_action"):
            return
        self.content_experiment_action.blockSignals(True)
        self.content_experiment_action.setChecked(self._content_mode_enabled)
        self.content_experiment_action.blockSignals(False)

    def enter_content_experiment(self) -> None:
        if self._preview_active or self._capture_manager.is_preview_active():
            self.stop_live_preview()
        self._content_mode_enabled = True
        self._sync_content_experiment_action()
        self._ensure_content_session()
        if self._tool_mode not in self._content_allowed_tool_modes():
            self.set_tool_mode("select")
        if self._center_stack is not None and self._preview_page is not None:
            self._center_stack.setCurrentWidget(self._preview_page)
        self._sync_live_preview_action()
        if not self._content_preview_active:
            self.start_content_preview()
        if not self._content_preview_active:
            self._update_content_ui()
            return
        session = self._ensure_content_session()
        if not session.fibers:
            self.edit_content_experiment_fibers()
            if not session.fibers:
                self._update_content_ui()
                self._focus_content_input_target()
                return
        self._activate_content_experiment_session()

    def start_content_preview(self) -> None:
        self._refresh_content_capture_devices()
        if not self._content_capture_devices:
            QMessageBox.information(self, "含量试验", self._content_capture_refresh_message())
            self._update_content_ui()
            return
        preview_kind = self._content_preview_kind()
        if self._center_stack is not None and self._preview_page is not None:
            self._center_stack.setCurrentWidget(self._preview_page)
        self._apply_preview_surface(preview_kind)
        if preview_kind == "native_embed" and self._content_microview_preview_host is not None:
            self._apply_content_native_preview_resolution()
            self._content_microview_preview_host.ensure_native_handle()
            QApplication.processEvents()
        preview_target = self._current_content_preview_target()
        if not self._content_capture_manager.start_preview(preview_target=preview_target):
            self._update_content_ui()
            return
        selected = self._selected_content_capture_device()
        if self._preview_status_label is not None:
            if preview_kind == "native_embed":
                self._preview_status_label.setText(
                    f"含量试验预览: {selected.name if selected is not None else '采集设备'}  (Microview 原生预览)"
                )
            else:
                self._preview_status_label.setText(f"含量试验预览: {selected.name if selected is not None else '采集设备'}")
        self.statusBar().showMessage("含量试验预览已启动", 3000)

    def stop_content_preview(self) -> None:
        if not self._content_capture_manager.is_preview_active():
            self._content_preview_active = False
            self._clear_content_preview_surface_state()
            self._update_ui_for_current_document()
            return
        self._content_capture_manager.stop_preview()
        self.statusBar().showMessage("含量试验预览已停止，记录已保留。", 3000)

    def _content_capture_refresh_message(self) -> str:
        lines = ["当前未检测到可用的采集设备。"]
        warnings = self._content_capture_manager.device_refresh_warnings()
        if warnings:
            lines.append("")
            lines.append("采集模块诊断:")
            lines.extend(warnings[:4])
        return "\n".join(lines)

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
        if self._preview_notice_label is not None:
            self._preview_notice_label.setVisible(active)
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
            self._clear_preview_surface_state()
            self._clear_prompt_segmentation_cache()
            self._invalidate_content_field("实时预览停止")
        else:
            self._invalidate_content_field("实时预览启动")
        self._sync_live_preview_action()
        self._update_ui_for_current_document()

    def _on_live_preview_frame_ready(self, image: object) -> None:
        if not self._preview_active or self._is_native_preview():
            return
        if not isinstance(image, QImage) or image.isNull() or self._preview_canvas is None:
            return
        self._update_preview_canvas_frame(image)
        self._update_action_states()

    def _update_preview_canvas_frame(self, image: QImage) -> None:
        if self._preview_canvas is None:
            return
        if (
            self._preview_document is None
            or self._preview_document.image_size != (image.width(), image.height())
        ):
            if self._preview_document is not None:
                self._invalidate_content_field("实时预览分辨率变化")
            self._preview_document = ImageDocument(
                id="preview_document",
                path="preview_frame.png",
                image_size=(image.width(), image.height()),
                source_type="project_asset",
            )
            self._preview_canvas.set_document(self._preview_document, image)
            self._preview_canvas.fit_to_view()
        else:
            self._preview_canvas.set_image(image)
        if self._preview_status_label is not None:
            selected = self._selected_capture_device()
            label = selected.name if selected is not None else "采集设备"
            self._preview_status_label.setText(f"正在预览: {label}  ({image.width()} x {image.height()})")
        if self._image_resolution_label is not None:
            self._image_resolution_label.setText(f"实时预览分辨率: {image.width()} x {image.height()} px")

    def _clear_preview_surface_state(self) -> None:
        self._preview_document = None
        self._apply_preview_surface("frame_stream")
        if self._preview_canvas is not None:
            self._preview_canvas.clear_document()
            self._preview_canvas.set_content_experiment_overlay()
        if self._preview_status_label is not None:
            self._preview_status_label.setText("请选择采集设备并开始实时预览")

    def _update_content_preview_document(self, image: QImage, *, native: bool) -> None:
        canvases = [canvas for canvas in (self._content_frame_canvas, self._content_overlay_canvas) if canvas is not None]
        if not canvases:
            return
        primary_canvas = self._content_frame_canvas if self._content_frame_canvas is not None else canvases[0]
        if (
            self._content_preview_document is None
            or self._content_preview_document.image_size != (image.width(), image.height())
            or primary_canvas.document_id is None
        ):
            if self._content_preview_document is not None:
                self._invalidate_content_field("含量试验预览分辨率变化")
            self._content_preview_document = ImageDocument(
                id="content_preview_document",
                path="content_preview_frame.png",
                image_size=(image.width(), image.height()),
                source_type="project_asset",
            )
            for canvas in canvases:
                canvas.set_document(self._content_preview_document, image)
                canvas.set_show_area_fill(False)
                canvas.set_view_transform(zoom=1.0, pan=Point(0.0, 0.0))
                if native:
                    canvas.setFixedSize(image.width(), image.height())
            if self._content_frame_canvas is not None:
                self._content_frame_canvas.setMinimumSize(image.width(), image.height())
        else:
            for canvas in canvases:
                canvas.set_image(image)
        if self._preview_status_label is not None:
            selected = self._selected_content_capture_device()
            label = selected.name if selected is not None else "采集设备"
            suffix = "Microview 原生预览" if native else "帧流预览"
            self._preview_status_label.setText(f"含量试验预览: {label}  ({image.width()} x {image.height()}, {suffix})")
        self._sync_content_preview_overlay()
        self._update_image_resolution_label()

    def _clear_content_preview_surface_state(self) -> None:
        self._content_preview_document = None
        self._content_analysis_frame = None
        if self._content_frame_canvas is not None:
            self._content_frame_canvas.clear_document()
            self._content_frame_canvas.set_content_experiment_overlay()
        if self._content_overlay_canvas is not None:
            self._content_overlay_canvas.clear_document()
            self._content_overlay_canvas.set_content_experiment_overlay()
        if self._preview_status_label is not None:
            self._preview_status_label.setText("含量试验预览未启动")

    def _on_capture_error(self, message: str) -> None:
        self._sync_live_preview_action()
        self._update_action_states()
        self.statusBar().showMessage(message, 5000)
        QMessageBox.warning(self, "实时预览", message)

    def _on_content_capture_devices_changed(self, devices: object) -> None:
        self._content_capture_devices = list(devices) if isinstance(devices, list) else []
        self._update_action_states()

    def _on_content_preview_state_changed(self, active: bool) -> None:
        self._content_preview_active = active
        if self._center_stack is not None and self._preview_page is not None:
            self._center_stack.setCurrentWidget(self._preview_page if self._content_mode_enabled else self.tab_widget)
        if active:
            self._apply_preview_surface(self._content_preview_kind())
            if self._is_content_native_preview() and self._content_microview_preview_host is not None:
                self._apply_content_native_preview_resolution()
                QApplication.processEvents()
                self._content_capture_manager.update_preview_target(self._content_microview_preview_host)
            warning = self._content_capture_manager.active_warning().strip()
            if warning:
                self.statusBar().showMessage(warning, 7000)
            self._invalidate_content_field("含量试验预览启动")
        else:
            self._clear_content_preview_surface_state()
            if self._content_mode_enabled:
                self._invalidate_content_field("含量试验预览停止")
        self._sync_live_preview_action()
        self._update_ui_for_current_document()

    def _on_content_preview_frame_ready(self, image: object) -> None:
        if not self._content_preview_active or self._is_content_native_preview():
            return
        if not isinstance(image, QImage) or image.isNull():
            return
        self._content_analysis_frame = image.copy()
        self._update_content_preview_document(image, native=False)
        self._update_action_states()

    def _on_content_analysis_frame_ready(self, request_id: int, image: object) -> None:
        if request_id != self._content_field_request_id:
            return
        self._on_content_field_frame_ready(image)

    def _on_content_analysis_frame_failed(self, request_id: int, message: str) -> None:
        if request_id != self._content_field_request_id:
            return
        self._on_content_field_frame_failed(message)

    def _on_content_capture_error(self, message: str) -> None:
        self._content_preview_active = False
        self._update_action_states()
        self._update_content_ui()
        self.statusBar().showMessage(message, 5000)
        QMessageBox.warning(self, "含量试验预览", message)

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
        for document in self.project.documents:
            if not document.is_project_asset():
                continue
            image = self._images.get(document.id)
            if image is None or image.isNull():
                QMessageBox.warning(self, "保存项目", f"无法找到项目内图片数据: {self._document_display_name(document)}")
                return False
            output_path = project_assets_root(target_path) / document.path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not image.save(str(output_path), "PNG"):
                QMessageBox.warning(self, "保存项目", f"写入项目内图片失败: {output_path}")
                return False
        return True

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
            message = f"无法写入设置文件：{exc}"
            if QApplication.platformName().lower() == "offscreen" or not self.isVisible():
                self.statusBar().showMessage(message, 5000)
            else:
                QMessageBox.warning(self, context, f"无法写入设置文件：\n{exc}")
            return False
        return True

    def _restore_initial_window_geometry(self) -> None:
        geometry_token = str(self._app_settings.main_window_geometry or "").strip()
        restored = False
        if geometry_token:
            restored = self.restoreGeometry(QByteArray.fromBase64(geometry_token.encode("ascii")))
            if restored and not self._window_geometry_intersects_available_screen(self.frameGeometry()):
                restored = False
        if not restored:
            self._apply_default_window_geometry()
        elif self.width() < 880:
            self.resize(880, max(self.height(), 520))
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
        width = max(880, min(width, available.width()))
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
            "metadata": dict(self.project.metadata),
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

    def _project_group_template_for_label(self, label: str) -> ProjectGroupTemplate | None:
        token = normalize_group_label(label)
        if not token:
            return None
        for template in self.project.project_group_templates:
            if normalize_group_label(template.label) == token:
                return template
        return None

    def _next_group_color(self, document: ImageDocument) -> str:
        return self._color_palette[(document.next_group_number() - 1) % len(self._color_palette)]

    def _normalize_group_color(self, color_value: str, *, fallback: str = "#1F7A8C") -> str:
        color = QColor(str(color_value or "").strip())
        if color.isValid():
            return color.name()
        fallback_color = QColor(str(fallback or "").strip())
        if fallback_color.isValid():
            return fallback_color.name()
        return "#1f7a8c"

    def _ensure_project_group_template(self, *, label: str, color: str) -> bool:
        token = normalize_group_label(label)
        if not token or self._project_group_template_for_label(token) is not None:
            return False
        self.project.project_group_templates.append(
            ProjectGroupTemplate(label=token, color=self._normalize_group_color(color)),
        )
        return True

    def _set_project_group_template_color(self, *, label: str, color: str) -> bool:
        template = self._project_group_template_for_label(label)
        if template is None:
            return False
        normalized_color = self._normalize_group_color(color, fallback=template.color)
        if template.color == normalized_color:
            return False
        template.color = normalized_color
        return True

    def _apply_project_group_templates_to_document(
        self,
        document: ImageDocument,
        *,
        labels: set[str] | None = None,
    ) -> bool:
        changed = False
        for template in self.project.project_group_templates:
            token = normalize_group_label(template.label)
            if (
                not token
                or (labels is not None and token not in labels)
                or document.is_project_group_label_suppressed(token)
            ):
                continue
            _group, ensured_changed = self._ensure_document_named_group(
                document,
                label=token,
                color=template.color,
                activate=False,
                sync_color=True,
            )
            changed = ensured_changed or changed
        if document.active_group_id is None and document.can_delete_uncategorized_entry():
            changed = document.hide_uncategorized_entry() or changed
        return changed

    def _sync_project_group_templates(self, *, label: str, labels: set[str] | None = None) -> bool:
        any_changed = False
        for document in self.project.documents:
            before = document.snapshot_state()
            changed = self._apply_project_group_templates_to_document(document, labels=labels)
            after = document.snapshot_state()
            if changed and document.history is not None and before != after:
                document.history.push(label, before, after)
                any_changed = True
            elif changed:
                any_changed = True
        return any_changed

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
        token = normalize_group_label(label)
        if not token:
            return None, False
        normalized_color = self._normalize_group_color(color)
        changed = False
        matches = document.groups_by_label(token)
        if matches:
            canonical = matches[0]
            for duplicate in matches[1:]:
                if document.merge_group_into(duplicate.id, canonical.id):
                    changed = True
            if sync_color and canonical.color != normalized_color:
                canonical.color = normalized_color
                changed = True
            if activate and document.active_group_id != canonical.id:
                document.set_active_group(canonical.id)
                changed = True
        else:
            active_group_id = document.active_group_id
            canonical = document.create_group(color=normalized_color, label=token)
            if activate or active_group_id is None:
                document.set_active_group(canonical.id)
            elif active_group_id != canonical.id:
                document.set_active_group(active_group_id)
            changed = True
        changed = document.unsuppress_project_group_label(token) or changed
        return canonical, changed

    def _area_inference_group_color_for_label(self, label: str) -> str:
        token = normalize_group_label(label)
        if not token:
            return self._color_palette[0]
        template = self._project_group_template_for_label(token)
        if template is not None:
            return template.color
        for document in self.project.documents:
            group = document.find_group_by_label(token)
            if group is not None:
                return group.color
        template_count = len(
            [
                template
                for template in self.project.project_group_templates
                if normalize_group_label(template.label)
            ]
        )
        return self._color_palette[template_count % len(self._color_palette)]

    def _resolve_area_inference_group_colors(self, labels: list[str]) -> dict[str, str]:
        resolved_colors: dict[str, str] = {}
        template_count = len(
            [
                template
                for template in self.project.project_group_templates
                if normalize_group_label(template.label)
            ]
        )
        fallback_offset = 0
        for label in labels:
            token = normalize_group_label(label)
            if not token or token in resolved_colors:
                continue
            template = self._project_group_template_for_label(token)
            if template is not None:
                resolved_colors[token] = template.color
                continue
            existing_color = None
            for document in self.project.documents:
                group = document.find_group_by_label(token)
                if group is not None:
                    existing_color = group.color
                    break
            if existing_color is not None:
                resolved_colors[token] = existing_color
                continue
            palette_index = (template_count + fallback_offset) % len(self._color_palette)
            resolved_colors[token] = self._color_palette[palette_index]
            fallback_offset += 1
        return resolved_colors

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
        current_button = box.addButton("当前图片", QMessageBox.ButtonRole.AcceptRole)
        project_button = box.addButton("项目所有图片", QMessageBox.ButtonRole.ActionRole)
        cancel_button = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(current_button)
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
    ) -> None:
        if self._load_thread is not None:
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
                )
            )
            return
        if len(requests) == 1:
            state = BatchLoadState(
                context_label=context_label,
                total=1,
                skipped_count=skipped_count,
                failures=[],
                missing_paths=list(missing_paths or []),
            )
            self._load_single_request_sync(requests[0], state)
            self._show_batch_load_summary(state)
            return
        self._start_batch_image_load(
            requests,
            context_label=context_label,
            skipped_count=skipped_count,
            missing_paths=missing_paths,
        )

    def _load_single_request_sync(self, request: ImageLoadRequest, state: BatchLoadState) -> None:
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

    def _start_batch_image_load(
        self,
        requests: list[ImageLoadRequest],
        *,
        context_label: str,
        skipped_count: int,
        missing_paths: list[str] | None = None,
    ) -> None:
        self._load_state = BatchLoadState(
            context_label=context_label,
            total=len(requests),
            skipped_count=skipped_count,
            failures=[],
            missing_paths=list(missing_paths or []),
        )
        progress = self._create_progress_dialog(
            title=context_label,
            label_text="准备加载图片...",
            maximum=len(requests),
        )
        self._load_progress_dialog = progress

        thread = QThread(self)
        worker = ImageBatchLoaderWorker(requests, skipped_count=skipped_count)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_batch_load_progress)
        worker.loaded.connect(self._on_batch_load_loaded)
        worker.failed.connect(self._on_batch_load_failed)
        worker.finished.connect(self._on_batch_load_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        progress.canceled.connect(worker.cancel)

        self._load_thread = thread
        self._load_worker = worker
        thread.start()
        progress.show()

    def _on_batch_load_progress(self, index: int, total: int, path: str) -> None:
        if self._load_progress_dialog is None:
            return
        completed = self._load_state.completed_count if self._load_state is not None else 0
        self._load_progress_dialog.setMaximum(total)
        self._load_progress_dialog.setValue(completed)
        self._load_progress_dialog.setLabelText(f"正在加载 ({index}/{total})\n{Path(path).name}")

    def _on_batch_load_loaded(self, request: ImageLoadRequest, image: QImage) -> None:
        state = self._load_state
        if state is not None:
            state.completed_count += 1
            state.loaded_count += 1
        self._add_loaded_document(request, image)
        if self._load_progress_dialog is not None and state is not None:
            self._load_progress_dialog.setValue(state.completed_count)

    def _on_batch_load_failed(self, path: str, reason: str) -> None:
        state = self._load_state
        if state is not None:
            state.completed_count += 1
            state.failed_count += 1
            if state.failures is not None:
                state.failures.append(f"{Path(path).name}: {reason}")
        if self._load_progress_dialog is not None and state is not None:
            self._load_progress_dialog.setValue(state.completed_count)

    def _on_batch_load_finished(self, cancelled: bool, loaded_count: int, skipped_count: int, failed_count: int) -> None:
        state = self._load_state
        if state is None:
            return
        state.cancelled = cancelled
        state.loaded_count = loaded_count
        state.skipped_count = skipped_count
        state.failed_count = failed_count
        state.completed_count = state.total
        if self._load_progress_dialog is not None:
            self._load_progress_dialog.setValue(state.total)
            self._load_progress_dialog.close()
            self._load_progress_dialog.deleteLater()
            self._load_progress_dialog = None
        self._show_batch_load_summary(state)
        if self._pending_project_load_snapshot:
            self._mark_project_saved()
            self._pending_project_load_snapshot = False
        self._load_thread = None
        self._load_worker = None
        self._load_state = None

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

        has_warning = bool(state.failed_count or state.missing_paths)
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
        self._area_infer_state = AreaInferenceBatchState(
            total=len(requests),
            model_name=model_name,
            update_project_group_templates=len(requests) > 1,
            failures=[],
        )
        progress = self._create_progress_dialog(
            title="面积自动识别",
            label_text=f"正在识别 (1/{len(requests)})\n{Path(requests[0].image_path).name}",
            maximum=len(requests),
        )
        self._area_infer_progress_dialog = progress

        thread = QThread(self)
        worker = AreaBatchInferenceWorker(requests, settings=self._app_settings)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_area_inference_progress)
        worker.succeeded.connect(self._on_area_inference_succeeded)
        worker.failed.connect(self._on_area_inference_failed)
        worker.finished.connect(self._on_area_inference_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        progress.canceled.connect(worker.cancel)

        self._area_infer_thread = thread
        self._area_infer_worker = worker
        thread.start()
        progress.show()

    def _ensure_prompt_segmentation_worker(self) -> None:
        if self._prompt_seg_thread is not None and self._prompt_seg_worker is not None:
            return
        thread = QThread(self)
        worker = PromptSegmentationWorker()
        worker.moveToThread(thread)
        worker.succeeded.connect(self._on_prompt_segmentation_succeeded)
        worker.failed.connect(self._on_prompt_segmentation_failed)
        thread.finished.connect(worker.deleteLater)
        thread.start()
        self._prompt_seg_thread = thread
        self._prompt_seg_worker = worker

    def _ensure_fiber_quick_geometry_worker(self) -> None:
        if self._fiber_quick_geometry_thread is not None and self._fiber_quick_geometry_worker is not None:
            return
        thread = QThread(self)
        worker = FiberQuickGeometryWorker()
        worker.moveToThread(thread)
        worker.succeeded.connect(self._on_fiber_quick_geometry_succeeded)
        worker.failed.connect(self._on_fiber_quick_geometry_failed)
        thread.finished.connect(worker.deleteLater)
        thread.start()
        self._fiber_quick_geometry_thread = thread
        self._fiber_quick_geometry_worker = worker

    def _ensure_fiber_quick_commit_geometry_worker(self) -> None:
        if self._fiber_quick_commit_geometry_thread is not None and self._fiber_quick_commit_geometry_worker is not None:
            return
        thread = QThread(self)
        worker = FiberQuickGeometryWorker(coalesce_latest=False)
        worker.moveToThread(thread)
        worker.succeeded.connect(self._on_fiber_quick_commit_geometry_succeeded)
        worker.failed.connect(self._on_fiber_quick_commit_geometry_failed)
        thread.finished.connect(worker.deleteLater)
        thread.start()
        self._fiber_quick_commit_geometry_thread = thread
        self._fiber_quick_commit_geometry_worker = worker

    def _ensure_reference_instance_worker(self) -> None:
        if self._reference_instance_thread is not None and self._reference_instance_worker is not None:
            return
        thread = QThread(self)
        worker = ReferenceInstancePropagationWorker()
        worker.moveToThread(thread)
        worker.succeeded.connect(self._on_reference_instance_succeeded)
        worker.failed.connect(self._on_reference_instance_failed)
        thread.finished.connect(worker.deleteLater)
        thread.start()
        self._reference_instance_thread = thread
        self._reference_instance_worker = worker

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
        if self._area_infer_progress_dialog is None:
            return
        completed = self._area_infer_state.completed_count if self._area_infer_state is not None else 0
        self._area_infer_progress_dialog.setMaximum(total)
        self._area_infer_progress_dialog.setValue(completed)
        self._area_infer_progress_dialog.setLabelText(f"正在识别 ({index}/{total})\n{Path(path).name}")

    def _on_area_inference_succeeded(self, document_id: str, instances: object) -> None:
        state = self._area_infer_state
        if state is not None:
            state.completed_count += 1
        document = self.project.get_document(document_id)
        if document is not None and isinstance(instances, list):
            self._apply_area_inference_result(
                document,
                instances,
                global_group_labels=state.global_group_labels if state is not None else None,
                model_name=state.model_name if state is not None else "",
                update_project_group_templates=bool(state.update_project_group_templates) if state is not None else True,
            )
        if self._area_infer_progress_dialog is not None and state is not None:
            self._area_infer_progress_dialog.setValue(state.completed_count)

    def _on_area_inference_failed(self, document_id: str, path: str, reason: str) -> None:
        del document_id
        state = self._area_infer_state
        if state is not None:
            state.completed_count += 1
            state.failed_count += 1
            if state.failures is not None:
                state.failures.append(f"{Path(path).name}: {reason}")
        if self._area_infer_progress_dialog is not None and state is not None:
            self._area_infer_progress_dialog.setValue(state.completed_count)

    def _on_area_inference_finished(self, cancelled: bool, completed_count: int, failed_count: int) -> None:
        state = self._area_infer_state
        if state is None:
            return
        state.cancelled = cancelled
        state.completed_count = completed_count
        state.failed_count = failed_count
        if self._area_infer_progress_dialog is not None:
            self._area_infer_progress_dialog.setValue(state.total)
            self._area_infer_progress_dialog.close()
            self._area_infer_progress_dialog.deleteLater()
            self._area_infer_progress_dialog = None

        if state.failures:
            QMessageBox.warning(self, "面积自动识别", "以下图片识别失败:\n" + "\n".join(state.failures[:10]))
        if completed_count > 0:
            self.statusBar().showMessage(
                f"面积自动识别已处理 {completed_count - failed_count} / {completed_count} 张图片",
                6000,
            )

        self._area_infer_thread = None
        self._area_infer_worker = None
        self._area_infer_state = None

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
        canvas.set_show_area_fill(self._show_area_fill)
        self._connect_interactive_canvas(canvas)

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

    def save_project(self, path: str | None = None) -> bool:
        if not self.project.documents and self._load_content_session_from_project() is None:
            QMessageBox.information(self, "保存项目", "请先打开图片。")
            return False
        target_path = Path(path) if path else self._project_path
        if target_path is None:
            default_dir = self._preferred_dialog_directory(recent_dir=self._app_settings.recent_project_dir)
            selected_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存项目",
                str(default_dir / "fiber_measurement.fdmproj"),
                self.PROJECT_FILTER,
            )
            if not selected_path:
                return False
            target_path = self._normalize_dialog_save_path(selected_path, "fiber_measurement.fdmproj")
        self.project.version = __version__
        self._sync_content_session_to_project()
        if self._content_session is not None:
            try:
                self._content_workbook_service.save_snapshot(self._content_session, target_path)
                self._sync_content_session_to_project()
            except Exception as exc:
                QMessageBox.warning(self, "保存项目", f"无法保存含量试验 Excel 快照：\n{exc}")
                return False
        if not self._persist_project_assets(target_path):
            return False
        ProjectIO.save(self.project, target_path)
        self._project_path = target_path
        self._remember_recent_directory(setting_name="recent_project_dir", directory=target_path.parent, context="保存项目")
        for document in self.project.documents:
            document.mark_session_saved()
            document.mark_calibration_saved()
        self._mark_project_saved()
        self._update_ui_for_current_document()
        self.statusBar().showMessage(f"项目已保存: {target_path}", 5000)
        return True

    def load_project(self) -> None:
        self.stop_content_preview()
        self.stop_live_preview()
        if self._load_thread is not None:
            QMessageBox.information(self, "打开项目", "当前仍有图片在加载，请稍候。")
            return
        if not self._confirm_close_documents(self.project.documents):
            return
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", self.PROJECT_FILTER)
        if not path:
            return
        project = ProjectIO.load(path)
        imported_count = self._merge_legacy_calibration_presets(project.calibration_presets)
        missing_paths = []
        self._reset_workspace()
        self._project_path = Path(path)
        self.project = ProjectState(
            version=project.version,
            documents=[],
            project_default_calibration=project.project_default_calibration,
            project_group_templates=list(project.project_group_templates),
        )
        self.project.metadata = project.metadata
        self._content_session = session_from_project_metadata(self.project.metadata)
        self._content_mode_enabled = bool(self._content_session and self._content_session.active)
        self._sync_content_experiment_action()
        self._refresh_preset_combo()
        load_items: list[tuple[str, ImageDocument | None]] = []
        for document in project.documents:
            resolved_path = document.resolved_path(self._project_path)
            if resolved_path.exists():
                load_items.append((str(resolved_path), document))
            else:
                missing_paths.append(str(resolved_path))
        self._pending_project_load_snapshot = True
        self._open_image_requests(load_items, context_label="打开项目", missing_paths=missing_paths)
        if self._load_thread is None:
            self._mark_project_saved()
            self._pending_project_load_snapshot = False
            self._update_ui_for_current_document()
        message = f"项目已加载: {path}"
        if imported_count:
            message += f"；已导入 {imported_count} 个旧版标定预设"
        self.statusBar().showMessage(message, 5000)
        if self._content_session is not None and self._content_session.active:
            QTimer.singleShot(0, self.enter_content_experiment)

    def export_results(self, preset: ExportSelection | None = None) -> None:
        if not self.project.documents:
            QMessageBox.information(self, "导出结果", "当前没有可导出的图片。")
            return
        preset = preset or ExportSelection.all_enabled(scope=ExportScope.ALL_OPEN)
        dialog = ExportOptionsDialog(
            preset,
            allow_all_scope=len(self.project.documents) > 1,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        selection = dialog.selection()
        if not selection.any_selected():
            QMessageBox.information(self, "导出结果", "请至少选择一种导出内容。")
            return
        target_documents = self.project.documents if selection.scope == ExportScope.ALL_OPEN else ([self.current_document()] if self.current_document() else [])
        target_documents = [document for document in target_documents if document is not None]
        planned_outputs = self.export_service.planned_outputs(target_documents, selection)
        if not planned_outputs:
            QMessageBox.information(self, "导出结果", "按当前导出内容设置，没有可生成的文件。")
            return
        default_dir = self._preferred_dialog_directory(recent_dir=self._app_settings.recent_export_dir)
        single_output_path: Path | None = None
        if len(planned_outputs) == 1:
            selected_path, _ = QFileDialog.getSaveFileName(
                self,
                "选择导出文件",
                str(default_dir / planned_outputs[0].filename),
                self._single_export_dialog_filter(planned_outputs[0].filename),
            )
            if not selected_path:
                return
            single_output_path = self._normalize_dialog_save_path(selected_path, planned_outputs[0].filename)
            output_dir = str(single_output_path.parent)
        else:
            output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", str(default_dir))
            if not output_dir:
                return
        progress = self._create_blocking_progress_dialog(
            title="导出结果",
            label_text="正在准备导出...",
            maximum=max(1, len(planned_outputs)),
        )
        current_output_path: Path | None = None

        def on_export_progress(completed_steps: int, total_steps: int, label: str, path: Path | None) -> None:
            nonlocal current_output_path
            if path is not None:
                current_output_path = path
            self._update_blocking_progress_dialog(
                progress,
                completed_steps=completed_steps,
                total_steps=total_steps,
                label=label,
                path=path,
            )

        progress.show()
        progress.raise_()
        progress.activateWindow()
        self._pump_modal_progress_events()
        try:
            outputs = self.export_service.export_project(
                self.project,
                output_dir,
                selection=selection,
                documents=target_documents,
                overlay_renderer=self._render_overlay_image,
                single_output_path=single_output_path,
                progress_callback=on_export_progress,
            )
        except Exception as exc:
            self._close_progress_dialog(progress)
            QMessageBox.warning(
                self,
                "导出失败",
                self._format_export_failure_message(exc, export_path=current_output_path),
            )
            return
        self._close_progress_dialog(progress)
        if not outputs:
            QMessageBox.information(self, "导出结果", "没有生成任何文件。")
            return
        export_root = single_output_path.parent if single_output_path is not None else Path(output_dir)
        self._remember_recent_directory(setting_name="recent_export_dir", directory=export_root, context="导出结果")
        output_labels = {
            "measurement_overlays": "测量叠加图",
            "scale_overlays": "比例尺叠加图",
            "combined_overlays": "测量+比例尺叠加图",
            "scale_jsons": "比例尺 JSON",
            "image_summary_csv": "图片汇总 CSV",
            "fiber_details_csv": "纤维种类汇总 CSV",
            "measurement_details_csv": "测量明细 CSV",
            "xlsx": "Excel 工作簿",
        }
        summary_lines = []
        for key, value in outputs.items():
            label = output_labels.get(key, key)
            if isinstance(value, list):
                summary_lines.append(f"{label}: {len(value)} 个文件")
            else:
                summary_lines.append(f"{label}: {value}")
        location_text = str(single_output_path) if single_output_path is not None else str(output_dir)
        QMessageBox.information(self, "导出完成", f"结果已导出到:\n{location_text}\n\n" + "\n".join(summary_lines))

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
        self._app_settings = new_settings
        self._apply_theme_mode()
        refresh_widget_theme(dialog)
        self._refresh_theme_sensitive_icons()
        if self._measurement_tool_strip is not None:
            self._measurement_tool_strip._apply_theme_styles()
        self._apply_tool_menu_stylesheets()
        self._magic_standard_roi_enabled = bool(new_settings.magic_segment_standard_roi_enabled)
        self._fiber_quick_roi_enabled = bool(new_settings.fiber_quick_roi_enabled)
        self._save_app_settings(context="设置")
        self._refresh_preset_combo()
        self._refresh_canvases_for_settings()

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
        for canvas in (self._content_frame_canvas, self._content_overlay_canvas):
            if canvas is not None:
                canvas.set_settings(self._app_settings)
                canvas.set_show_area_fill(False)
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
        if selected is None:
            return
        _, preset = selected
        if self._content_mode_enabled:
            session = self._ensure_content_session()
            session.calibration_name = preset.name
            session.calibration_pixels_per_unit = preset.resolved_pixels_per_unit()
            session.calibration_unit = preset.unit
            self._sync_content_session_to_project()
            self.statusBar().showMessage(f"含量试验已使用标尺: {preset.name}", 4000)
            self._update_calibration_panel(document)
            return
        if document is None:
            return
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
        if self._content_mode_enabled:
            self.edit_content_experiment_fibers()
            return
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
        if self._content_mode_enabled:
            self.edit_content_experiment_fibers()
            return
        document = self.current_document()
        if document is None:
            return
        group = document.get_group(document.active_group_id)
        if group is None:
            return
        dialog = FiberGroupDialog(
            self,
            title="编辑类别",
            initial_label=group.label,
            initial_color=group.color,
            apply_to_project_default=False,
            show_apply_to_project=False,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        label, selected_color, _apply_to_project = dialog.values()
        target_label = normalize_group_label(label)
        selected_qcolor = QColor(selected_color)
        target_color = selected_qcolor.name() if selected_qcolor.isValid() else selected_color.strip() or group.color
        current_label = normalize_group_label(group.label)
        current_qcolor = QColor(group.color)
        current_color = current_qcolor.name() if current_qcolor.isValid() else group.color
        if target_label == current_label and target_color == current_color:
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

            def mutate_merge() -> None:
                source = document.get_group(group.id)
                target = document.get_group(merge_target.id)
                if source is None or target is None:
                    return
                source_label = normalize_group_label(source.label)
                target_token = normalize_group_label(target.label)
                document.merge_group_into(source.id, target.id)
                if self._project_group_template_for_label(source_label) is not None:
                    document.suppress_project_group_label(source_label)
                if self._project_group_template_for_label(target_token) is not None:
                    document.unsuppress_project_group_label(target_token)

            changed = self._apply_document_change(document, "合并类别", mutate_merge)
            if target_label and self._project_group_template_for_label(target_label) is not None:
                changed = self._sync_project_group_template_colors(
                    {target_label: target_color},
                    history_label="同步项目全局类别颜色",
                ) or changed
            if changed:
                self.statusBar().showMessage("类别已合并", 3000)
            return

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
        if target_label and self._project_group_template_for_label(target_label) is not None:
            changed = self._sync_project_group_template_colors(
                {target_label: target_color},
                history_label="同步项目全局类别颜色",
            ) or changed
        if changed:
            self.statusBar().showMessage("类别已更新", 3000)

    def delete_active_group(self) -> None:
        if self._content_mode_enabled:
            self.remove_active_content_fiber()
            return
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
        if self._content_mode_enabled:
            self.delete_selected_content_records()
            return
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
        if not target_documents:
            return
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

    def _canvas_for_document_id(self, document_id: str | None) -> DocumentCanvas | None:
        if self._is_content_preview_document_id(document_id):
            return self._content_current_canvas()
        if document_id is None:
            return None
        return self._canvases.get(document_id)

    def _image_for_document_id(self, document_id: str | None) -> QImage | None:
        if self._is_content_preview_document_id(document_id):
            if self._content_analysis_frame is not None and not self._content_analysis_frame.isNull():
                return self._content_analysis_frame.copy()
            canvas = self._content_current_canvas()
            if canvas is not None and canvas._image is not None and not canvas._image.isNull():  # noqa: SLF001
                return canvas._image.copy()  # noqa: SLF001
            return None
        if document_id is None:
            return None
        return self._images.get(document_id)

    def _on_canvas_magic_segment_requested(self, document_id: str, payload: object) -> None:
        canvas = self._canvas_for_document_id(document_id)
        document = self.project.get_document(document_id)
        if canvas is None or not isinstance(payload, dict):
            return
        image = self._image_for_document_id(document_id)
        request_id = int(payload.get("request_id", 0))
        tool_mode = str(payload.get("tool_mode", self._tool_mode) or self._tool_mode)
        if not is_magic_toolbar_tool_mode(tool_mode):
            tool_mode = MagicSegmentToolMode.STANDARD
        tool_label = self._magic_tool_label(tool_mode)
        if self._is_content_preview_document_id(document_id) and not is_fiber_quick_tool_mode(tool_mode):
            canvas.fail_magic_segment_result(request_id)
            self._update_magic_segment_controls()
            self.statusBar().showMessage("含量试验中仅支持快速测径分割。", 4000)
            return
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
            if document is None:
                canvas.fail_reference_instance_result(request_id)
                self._update_magic_segment_controls()
                self.statusBar().showMessage("当前模式不支持同类扩选。", 4000)
                return
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
        roi_enabled = self._current_magic_roi_enabled(tool_mode)
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
            )
        )
        self._update_magic_segment_controls()

    def _on_canvas_magic_segment_session_changed(self, document_id: str) -> None:
        current_document = self.current_document()
        if self._is_content_preview_document_id(document_id) or (current_document is not None and current_document.id == document_id):
            self._update_magic_segment_controls()

    def _on_canvas_path_session_changed(self, document_id: str) -> None:
        current_document = self.current_document()
        if self._is_content_preview_document_id(document_id) or (current_document is not None and current_document.id == document_id):
            self._update_path_drawing_controls()

    def _dispatch_pending_magic_segment_request(self, document_id: str, completed_request_id: int) -> bool:
        canvas = self._canvas_for_document_id(document_id)
        if canvas is None:
            return False
        payload = canvas.dequeue_pending_magic_segment_request(completed_request_id)
        if payload is None:
            return False
        self._on_canvas_magic_segment_requested(document_id, payload)
        return True

    def _dispatch_pending_fiber_quick_request(self, document_id: str, completed_request_id: int) -> bool:
        canvas = self._canvas_for_document_id(document_id)
        if canvas is None:
            return False
        payload = canvas.dequeue_pending_fiber_quick_request(completed_request_id)
        if payload is None:
            return False
        self._on_canvas_magic_segment_requested(document_id, payload)
        return True

    def _on_prompt_segmentation_succeeded(self, document_id: str, request_id: int, result: object) -> None:
        canvas = self._canvas_for_document_id(document_id)
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
                    },
                )
                if apply_result is None:
                    self._update_magic_segment_controls()
                    return
                if result.mask is None or not bool(apply_result.get("has_preview", False)):
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
        canvas = self._canvas_for_document_id(document_id)
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
        canvas = self._canvas_for_document_id(document_id)
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
        canvas = self._canvas_for_document_id(document_id)
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
            message_parts.append("当前项目的统一比例尺、项目内图片、全局类别、含量试验或继承关系有未保存改动。")
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
        self.stop_content_preview()
        self.stop_live_preview()
        self._clear_prompt_segmentation_cache()
        self._content_workbook_service.close()
        self._content_session = None
        self._content_measure_start = None
        self._content_measure_hover = None
        self._content_mode_enabled = False
        self._content_preview_active = False
        self._content_preview_document = None
        self._content_analysis_frame = None
        self._sync_content_experiment_action()
        self._content_field_timer.stop()
        self._content_native_overlay_timer.stop()
        self._content_field_request_pending = False
        self._content_field_baseline = None
        self.project = ProjectState.empty()
        self._project_path = None
        self._pending_project_load_snapshot = False
        self._document_order.clear()
        self._images.clear()
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
            self._apply_open_view_mode(canvas)
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
        if self._is_content_preview_document_id(document_id):
            self._on_content_canvas_line_committed(mode, payload)
            return
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
            image = self._images.get(document.id)
            if image is None or image.isNull():
                self.statusBar().showMessage("当前图片还未完成加载，暂时无法进行边缘吸附。", 4000)
                self._focus_current_canvas()
                return
            try:
                snap_result = self.snap_service.snap_measurement(image, payload)
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"边缘吸附失败: {exc}", 5000)
                self._focus_current_canvas()
                return

        def mutate() -> None:
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
                return
            document.add_measurement(measurement)
            document.select_overlay_annotation(None)

        self._apply_document_change(document, "新增测量", mutate)
        if snap_result is not None:
            self.statusBar().showMessage(self._edge_snap_status_message(snap_result), 4000)
        else:
            self.statusBar().showMessage("已新增测量", 2500)
        self._focus_current_canvas()

    def _is_content_preview_document_id(self, document_id: str | None) -> bool:
        return bool(
            self._content_mode_enabled
            and self._content_preview_document is not None
            and document_id == self._content_preview_document.id
        )

    def _on_content_canvas_line_committed(self, mode: str, payload: object) -> None:
        if not self._content_experiment_is_active():
            self.statusBar().showMessage("请先开始含量试验并选择纤维类别。", 3000)
            self._focus_content_input_target()
            return
        if mode == "manual" and isinstance(payload, Line):
            if self._append_content_diameter_record(payload, source_mode="manual"):
                self.statusBar().showMessage("已记录手动线段直径", 2500)
            self._focus_content_input_target()
            return
        if mode == "snap" and isinstance(payload, Line):
            image = self._content_analysis_frame
            if image is None or image.isNull():
                self.statusBar().showMessage("正在获取分析帧，暂时无法进行边缘吸附。", 4000)
                self._focus_content_input_target()
                return
            try:
                snap_result = self.snap_service.snap_measurement(image, payload)
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"边缘吸附失败: {exc}", 5000)
                self._focus_content_input_target()
                return
            if snap_result.snapped_line is None:
                self.statusBar().showMessage(
                    f"边缘吸附未找到可靠边界，请重画线段或改用手动线段。状态: {self._format_measurement_status(snap_result.status)}",
                    5000,
                )
                self._focus_content_input_target()
                return
            if self._append_content_diameter_record(snap_result.snapped_line, source_mode="snap"):
                self.statusBar().showMessage(self._edge_snap_status_message(snap_result), 4000)
            self._focus_content_input_target()
            return
        if mode == "fiber_quick" and isinstance(payload, dict) and isinstance(payload.get("line_px"), Line):
            if self._append_content_diameter_record(payload["line_px"], source_mode="fiber_quick"):
                self.statusBar().showMessage("已记录快速测径直径", 3000)
            self._focus_content_input_target()

    def _append_content_diameter_record(self, line: Line | None, *, source_mode: str) -> bool:
        session = self._load_content_session_from_project()
        if session is None:
            return False
        if line is None:
            self._sync_content_preview_overlay()
            return False
        if not session.fibers:
            self.edit_content_experiment_fibers()
            return False
        if line_length(line) < 1.0:
            self._sync_content_preview_overlay()
            return False
        fiber_id = session.current_fiber_id
        if session.selection_mode == ContentSelectionMode.POSTSELECT:
            fiber_id = self._prompt_content_fiber(self.cursor().pos())
        fiber = session.fiber_by_id(fiber_id)
        if fiber is None:
            self._sync_content_preview_overlay()
            return False
        session.current_fiber_id = fiber.id
        diameter_px = line_length(line)
        calibration = self._current_preview_calibration()
        diameter_unit = calibration.px_to_unit(diameter_px) if calibration is not None else diameter_px
        diameter_unit_name = calibration.unit if calibration is not None else "px"
        record = ContentExperimentRecord(
            id=new_id("content_rec"),
            kind=ContentRecordKind.DIAMETER,
            fiber_id=fiber.id,
            source_mode=source_mode,
            field_id=session.current_field_id,
            line_px=line,
            diameter_px=diameter_px,
            diameter_unit=diameter_unit,
            diameter_unit_name=diameter_unit_name,
        )
        session.records.append(record)
        self._content_after_record_added(record, fiber)
        return True

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
            self._apply_preset_button.setEnabled(has_preset and (self.current_document() is not None or self._content_mode_enabled))

    def _populate_group_list(self, document: ImageDocument | None) -> None:
        self._group_list_rebuilding = True
        self.group_list.clear()
        if document is not None:
            if document.should_show_uncategorized_entry():
                self._add_group_list_item(
                    label=UNCATEGORIZED_LABEL,
                    color=self._app_settings.default_measurement_color,
                    current_count=document.uncategorized_measurement_count(),
                    project_count=self._project_uncategorized_measurement_count(document),
                    group_id=None,
                    selected=document.active_group_id is None,
                )
            for group in document.sorted_groups():
                self._add_group_list_item(
                    label=group.display_name(),
                    color=group.color,
                    current_count=len(group.measurement_ids),
                    project_count=self._project_measurement_count_for_group_label(group.label, document),
                    group_id=group.id,
                    selected=document.active_group_id == group.id,
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
            elif isinstance(widget, ContentFiberListItemWidget):
                widget.setSelected(item.isSelected())

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
        documents = list(self.project.documents)
        if current_document is not None and all(document.id != current_document.id for document in documents):
            documents.append(current_document)
        return documents

    def _project_measurement_count_for_group_label(self, label: str, current_document: ImageDocument | None = None) -> int:
        token = normalize_group_label(label)
        total = 0
        for document in self._documents_for_group_counts(current_document):
            if token:
                for group in document.groups_by_label(token):
                    total += len(group.measurement_ids)
            else:
                for group in document.sorted_groups():
                    if not normalize_group_label(group.label):
                        total += len(group.measurement_ids)
        return total

    def _project_uncategorized_measurement_count(self, current_document: ImageDocument | None = None) -> int:
        return sum(document.uncategorized_measurement_count() for document in self._documents_for_group_counts(current_document))

    def _load_content_session_from_project(self) -> ContentExperimentSession | None:
        if self._content_session is None:
            self._content_session = session_from_project_metadata(self.project.metadata)
        return self._content_session

    def _ensure_content_session(self) -> ContentExperimentSession:
        session = self._load_content_session_from_project()
        if session is None:
            session = ContentExperimentSession(
                operator=self._app_settings.content_last_operator,
                active=False,
            )
            self._content_session = session
            self._sync_content_session_to_project()
        return session

    def _sync_content_session_to_project(self) -> None:
        write_session_to_project_metadata(self.project.metadata, self._content_session)
        if self._content_session is not None and self._content_session.operator:
            if self._app_settings.content_last_operator != self._content_session.operator:
                self._app_settings.content_last_operator = self._content_session.operator
                self._save_app_settings(context="含量试验")

    def _content_fiber_color_map(self) -> dict[str, str]:
        session = self._load_content_session_from_project()
        if session is None:
            return {}
        return {fiber.id: fiber.color for fiber in session.fibers}

    def _content_workbook_mode_label(self, mode: str) -> str:
        return {
            "excel": "Excel",
            "xlsx": "xlsx快照",
        }.get(mode, "未打开")

    def _edit_content_basic_field(self, field_name: str) -> None:
        session = self._ensure_content_session()
        labels = {
            "operator": "操作人",
            "sample_id": "样品编号",
            "sample_name": "样品名称",
        }
        label = labels.get(field_name, "基础信息")
        current = str(getattr(session, field_name, "") or "")
        value, ok = QInputDialog.getText(self, f"填写{label}", f"请输入{label}", text=current)
        if not ok:
            return
        setattr(session, field_name, value.strip())
        self._sync_content_session_to_project()
        self._populate_content_basic_info()
        self._refresh_content_workbook()
        self._update_action_states()

    def _update_content_basic_info_from_ui(self) -> None:
        session = self._load_content_session_from_project()
        if session is None:
            return
        if self._content_operator_edit is not None:
            session.operator = self._content_operator_edit.text().strip()
        if self._content_sample_id_edit is not None:
            session.sample_id = self._content_sample_id_edit.text().strip()
        if self._content_sample_name_edit is not None:
            session.sample_name = self._content_sample_name_edit.text().strip()
        if self._content_mode_postselect_radio is not None and self._content_mode_postselect_radio.isChecked():
            session.selection_mode = ContentSelectionMode.POSTSELECT
        else:
            session.selection_mode = ContentSelectionMode.PRESELECT
        if self._content_overlay_combo is not None:
            session.overlay_style = str(self._content_overlay_combo.currentData() or ContentOverlayStyle.NONE)
        self._sync_content_session_to_project()
        self._sync_content_preview_overlay()
        self._refresh_content_workbook()

    def _populate_content_basic_info(self) -> None:
        session = self._load_content_session_from_project()
        if session is None:
            if self._content_operator_edit is not None:
                self._content_operator_edit.setText(self._app_settings.content_last_operator)
            if self._content_sample_id_edit is not None:
                self._content_sample_id_edit.clear()
            if self._content_sample_name_edit is not None:
                self._content_sample_name_edit.clear()
            if self._content_totals_label is not None:
                self._content_totals_label.setText("总计数 0；实测 0")
            return
        for edit, value in [
            (self._content_operator_edit, session.operator),
            (self._content_sample_id_edit, session.sample_id),
            (self._content_sample_name_edit, session.sample_name),
        ]:
            if edit is not None and edit.text() != value:
                edit.setText(value)
        if self._content_mode_preselect_radio is not None:
            self._content_mode_preselect_radio.setChecked(session.selection_mode != ContentSelectionMode.POSTSELECT)
        if self._content_mode_postselect_radio is not None:
            self._content_mode_postselect_radio.setChecked(session.selection_mode == ContentSelectionMode.POSTSELECT)
        if self._content_overlay_combo is not None:
            index = self._content_overlay_combo.findData(session.overlay_style)
            self._content_overlay_combo.setCurrentIndex(max(0, index))
        if self._content_totals_label is not None:
            self._content_totals_label.setText(
                f"总计数 {content_total_count(session)}；实测 {content_total_measured(session)}；当前视场 {session.current_field_id}"
            )

    def _populate_content_group_list(self) -> None:
        session = self._load_content_session_from_project()
        self._group_list_rebuilding = True
        self.group_list.clear()
        if session is not None:
            stats_by_id = {item.fiber.id: item for item in content_session_stats(session)}
            for index, fiber in enumerate(session.fibers, start=1):
                stats = stats_by_id[fiber.id]
                average_text = "-" if stats.average_diameter is None else f"{stats.average_diameter:.2f}"
                content_text = "-" if stats.content_percent is None else f"{stats.content_percent:.1f}%"
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, fiber.id)
                item.setData(Qt.ItemDataRole.UserRole + 2, fiber.name)
                item.setSizeHint(QSize(0, ContentFiberListItemWidget.HEIGHT))
                self.group_list.addItem(item)
                widget = ContentFiberListItemWidget(
                    f"{index}. {fiber.name}",
                    stats.count,
                    stats.measured,
                    average_text,
                    content_text,
                    fiber.color,
                    selected=session.current_fiber_id == fiber.id,
                    parent=self.group_list,
                )
                self.group_list.setItemWidget(item, widget)
                if session.current_fiber_id == fiber.id:
                    item.setSelected(True)
        self._group_list_rebuilding = False

    def _set_group_headers_for_content(self, enabled: bool) -> None:
        if len(self._group_header_labels) < 3:
            return
        if enabled:
            self._group_header_labels[0].setText("纤维类别")
            self._group_header_labels[0].setFixedWidth(104)
            self._group_header_labels[1].setText("计/测")
            self._group_header_labels[1].setFixedWidth(ContentFiberListItemWidget.COUNT_COLUMN_WIDTH)
            self._group_header_labels[1].setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._group_header_labels[2].setText("均径/含量")
            self._group_header_labels[2].setFixedWidth(
                ContentFiberListItemWidget.AVG_COLUMN_WIDTH + ContentFiberListItemWidget.CONTENT_COLUMN_WIDTH
            )
            self._group_header_labels[2].setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self._group_header_labels[0].setText("颜色")
            self._group_header_labels[0].setFixedWidth(36)
            self._group_header_labels[1].setText("类别")
            self._group_header_labels[1].setMinimumWidth(0)
            self._group_header_labels[1].setMaximumWidth(16777215)
            self._group_header_labels[1].setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._group_header_labels[2].setText("（当前/总数）")
            self._group_header_labels[2].setFixedWidth(FiberGroupListItemWidget.COUNT_COLUMN_WIDTH)
            self._group_header_labels[2].setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def _update_content_record_table(self) -> None:
        if self._content_record_table is None:
            return
        session = self._load_content_session_from_project()
        self._content_record_table.setRowCount(0)
        if session is None:
            return
        for row, record in enumerate(reversed(session.records)):
            fiber = session.fiber_by_id(record.fiber_id)
            self._content_record_table.insertRow(row)
            kind_label = "计数" if record.kind == ContentRecordKind.COUNT else "直径"
            self._content_record_table.setItem(row, 0, QTableWidgetItem(kind_label))
            self._content_record_table.setItem(row, 1, QTableWidgetItem(fiber.name if fiber is not None else ""))
            self._content_record_table.setItem(row, 2, QTableWidgetItem(record.display_value()))
            self._content_record_table.setItem(row, 3, QTableWidgetItem(str(record.field_id)))
            id_item = QTableWidgetItem(record.id.split("_")[-1])
            id_item.setData(Qt.ItemDataRole.UserRole, record.id)
            self._content_record_table.setItem(row, 4, id_item)

    def _update_content_controls(self) -> None:
        preview_active = self._content_preview_active
        content_visible = self._content_mode_enabled
        session = self._load_content_session_from_project()
        active = bool(content_visible and session and session.active)
        if self._left_top_stack is not None:
            self._left_top_stack.setCurrentWidget(self._content_info_box if content_visible else self._image_box)
        if self._area_model_box is not None:
            self._area_model_box.setVisible(not content_visible)
        if self._content_box is not None:
            self._content_box.setVisible(content_visible)
        if self._calibration_label_scroll is not None:
            if content_visible:
                self._calibration_label_scroll.setMinimumHeight(30)
                self._calibration_label_scroll.setMaximumHeight(42)
            else:
                self._calibration_label_scroll.setMinimumHeight(88)
                self._calibration_label_scroll.setMaximumHeight(118)
        for button in (self._add_preset_button, self._edit_preset_button, self._delete_preset_button, self._import_cu_preset_button):
            if button is not None:
                button.setVisible(not content_visible)
        if self._apply_preset_button is not None:
            self._apply_preset_button.setVisible(True)
        self._set_group_headers_for_content(content_visible)
        if self._content_start_button is not None:
            self._content_start_button.setEnabled(content_visible and preview_active)
            self._content_start_button.setText("继续" if session and not active and session.fibers else "开始")
        if self._content_close_button is not None:
            self._content_close_button.setText("暂停")
            self._content_close_button.setEnabled(content_visible)
        if self._content_save_excel_button is not None:
            self._content_save_excel_button.setEnabled(bool(session and session.records))
        if self._content_delete_record_button is not None:
            self._content_delete_record_button.setEnabled(bool(session and session.records))
        if self._content_status_label is not None:
            if not content_visible:
                self._content_status_label.setText("点击工具栏“含量试验”进入试验模式。")
            elif not preview_active:
                self._content_status_label.setText("含量试验已打开；正在启动试验预览或等待采集设备。")
            elif session is None:
                self._content_status_label.setText("正在准备含量试验。")
            elif not session.fibers:
                self._content_status_label.setText("请选择含量试验纤维类别后开始计数/测径。")
            elif not session.active:
                self._content_status_label.setText("含量试验已暂停；记录已保留，点击“开始”继续。")
            else:
                fiber = session.active_fiber()
                fiber_text = fiber.name if fiber is not None else "未选择"
                workbook_mode = self._content_workbook_mode_label(session.workbook_mode)
                status_text = f"{'进行中' if session.active else '已暂停'}；当前纤维: {fiber_text}；工作簿: {workbook_mode}"
                warning = self._content_workbook_service.last_warning.strip()
                if warning:
                    status_text += "（Excel COM失败，已记入日志）"
                    self._content_status_label.setToolTip(warning)
                else:
                    self._content_status_label.setToolTip("")
                self._content_status_label.setText(status_text)
        if content_visible and preview_active:
            self._content_field_timer.start()
            if self._is_content_native_preview() and not is_fiber_quick_tool_mode(self._tool_mode):
                self._content_native_overlay_timer.start()
            else:
                self._content_native_overlay_timer.stop()
        else:
            self._content_field_timer.stop()
            self._content_native_overlay_timer.stop()
            self._content_field_request_pending = False

    def _sync_content_preview_overlay(self) -> None:
        canvases = [
            canvas
            for canvas in (self._content_frame_canvas, self._content_overlay_canvas)
            if canvas is not None
        ]
        if not canvases:
            return
        if not self._content_mode_enabled:
            for canvas in canvases:
                canvas.set_content_experiment_overlay()
            if self._content_microview_preview_host is not None:
                self._content_microview_preview_host.set_content_experiment_overlay()
            return
        session = self._load_content_session_from_project()
        if session is None:
            for canvas in canvases:
                canvas.set_content_experiment_overlay()
            if self._content_microview_preview_host is not None:
                self._content_microview_preview_host.set_content_experiment_overlay()
            return
        visible_records = [
            record
            for record in session.records
            if record.field_id == session.current_field_id
        ]
        pending_line = None
        if self._content_measure_start is not None and self._content_measure_hover is not None:
            pending_line = Line(self._content_measure_start, self._content_measure_hover)
        for canvas in canvases:
            canvas.set_content_experiment_overlay(
                overlay_style=session.overlay_style,
                records=visible_records,
                fiber_colors=self._content_fiber_color_map(),
                pending_line=pending_line,
            )
        if self._content_microview_preview_host is not None:
            self._content_microview_preview_host.set_content_experiment_overlay(
                overlay_style=session.overlay_style,
                records=visible_records,
                fiber_colors=self._content_fiber_color_map(),
                pending_line=pending_line,
            )

    def _refresh_content_native_overlay(self) -> None:
        if (
            self._content_mode_enabled
            and self._content_preview_active
            and self._is_content_native_preview()
            and not is_fiber_quick_tool_mode(self._tool_mode)
            and self._content_microview_preview_host is not None
        ):
            self._content_microview_preview_host.draw_content_experiment_overlay()

    def _refresh_content_workbook(self) -> None:
        session = self._load_content_session_from_project()
        if session is None or not self._content_workbook_service.is_open():
            return
        try:
            self._content_workbook_service.sync_session(session)
        except Exception as exc:
            if self._content_status_label is not None:
                self._content_status_label.setText(f"工作簿同步失败：{exc}")

    def _update_content_ui(self) -> None:
        self._populate_content_basic_info()
        self._update_content_record_table()
        if self._content_mode_enabled and hasattr(self, "measurement_table"):
            self._populate_measurement_table(self.current_document())
        self._sync_content_preview_overlay()
        self._update_content_controls()

    def edit_content_experiment_fibers(self) -> None:
        session = self._ensure_content_session()
        dialog = ContentFiberSelectionDialog(
            self._app_settings.content_fiber_definitions,
            session.fibers,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        selected = dialog.selected_fibers()
        if not selected:
            QMessageBox.information(self, "含量试验", "请至少选择一种纤维。")
            return
        previous_ids = {fiber.id for fiber in session.fibers}
        session.set_fibers(selected)
        removed_ids = previous_ids - {fiber.id for fiber in session.fibers}
        if removed_ids:
            session.records = [record for record in session.records if record.fiber_id not in removed_ids]
        self._sync_content_session_to_project()
        self._refresh_content_workbook()
        self._update_content_ui()

    def remove_active_content_fiber(self) -> None:
        session = self._load_content_session_from_project()
        if session is None:
            return
        fiber = session.active_fiber()
        if fiber is None:
            return
        record_count = sum(1 for record in session.records if record.fiber_id == fiber.id)
        message = f"确定从本次含量试验中移除“{fiber.name}”吗？"
        if record_count:
            message += f"\n\n该纤维下的 {record_count} 条含量试验记录也会删除。"
        response = QMessageBox.question(
            self,
            "移除含量试验纤维",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        session.fibers = [item for item in session.fibers if item.id != fiber.id]
        session.records = [record for record in session.records if record.fiber_id != fiber.id]
        session.ensure_current_fiber()
        self._sync_content_session_to_project()
        self._refresh_content_workbook()
        self._update_content_ui()

    def start_or_continue_content_experiment(self) -> None:
        self.enter_content_experiment()

    def _activate_content_experiment_session(self) -> None:
        session = self._ensure_content_session()
        if not self._content_preview_active or not session.fibers:
            self._update_content_ui()
            return
        self._update_content_basic_info_from_ui()
        session.active = True
        session.ensure_current_fiber()
        self._sync_content_session_to_project()
        workbook_warning = ""
        if not self._content_workbook_service.is_open():
            try:
                mode = self._content_workbook_service.open_session(session, project_path=self._project_path)
                session.workbook_mode = mode
                self._sync_content_session_to_project()
                if self._content_workbook_service.last_warning:
                    workbook_warning = self._content_workbook_service.last_warning
            except Exception as exc:
                session.workbook_mode = ""
                self._sync_content_session_to_project()
                QMessageBox.warning(self, "含量试验", f"无法打开含量试验工作簿：\n{exc}\n\n记录会保留在项目中，保存项目时会尝试重建 Excel 快照。")
        self._invalidate_content_field("含量试验开始")
        self._update_content_ui()
        if workbook_warning:
            self.statusBar().showMessage(workbook_warning, 7000)
        self._focus_content_input_target()

    def close_content_experiment(self) -> None:
        session = self._load_content_session_from_project()
        if session is not None:
            session.active = False
        self._content_measure_start = None
        self._content_measure_hover = None
        self._content_pending_diameter_line = None
        self._content_field_baseline = None
        self._content_field_motion_hits = 0
        self._content_field_request_pending = False
        self._content_field_timer.stop()
        self._content_native_overlay_timer.stop()
        self._content_workbook_service.close()
        self._content_mode_enabled = False
        self._sync_content_experiment_action()
        if session is not None:
            self._sync_content_session_to_project()
        self.stop_content_preview()
        self._sync_content_preview_overlay()
        self._update_content_ui()
        self.statusBar().showMessage("含量试验已暂停，记录已保留；保存项目后可继续。", 5000)

    def save_content_experiment_excel(self) -> None:
        session = self._load_content_session_from_project()
        if session is None:
            return
        default_dir = self._preferred_dialog_directory(recent_dir=self._app_settings.recent_export_dir)
        filename = f"{session.sample_id or session.sample_name or 'content_experiment'}.xlsx"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存含量试验 Excel",
            str(default_dir / filename),
            "Excel 工作簿 (*.xlsx)",
        )
        if not output_path:
            return
        try:
            target = self._content_workbook_service.save_as(session, output_path)
        except Exception as exc:
            QMessageBox.warning(self, "保存含量试验 Excel", f"无法保存 Excel：\n{exc}")
            return
        self._remember_recent_directory(setting_name="recent_export_dir", directory=target.parent, context="保存含量试验 Excel")
        self.statusBar().showMessage(f"含量试验 Excel 已保存: {target}", 4000)

    def delete_selected_content_records(self) -> None:
        session = self._load_content_session_from_project()
        if session is None:
            return
        target_ids = set(self._selected_measurement_ids_from_table())
        if not target_ids:
            return
        session.records = [record for record in session.records if record.id not in target_ids]
        self._sync_content_session_to_project()
        self._refresh_content_workbook()
        self._update_content_ui()

    def _content_experiment_is_active(self) -> bool:
        session = self._load_content_session_from_project()
        return bool(self._content_mode_enabled and self._content_preview_active and session is not None and session.active)

    def eventFilter(self, watched, event) -> bool:
        if self._content_experiment_is_active() and self._is_content_interaction_widget(watched):
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.RightButton:
                    self._show_content_fiber_menu(self._event_global_position(event))
                    return True
                if watched is self._content_microview_preview_host and event.button() == Qt.MouseButton.LeftButton:
                    point = self._content_point_from_event(watched, event)
                    if point is not None and self._handle_content_native_tool_click(point):
                        return True
            if watched is self._content_microview_preview_host and event.type() == QEvent.Type.MouseMove:
                point = self._content_point_from_event(watched, event)
                if point is not None and self._content_measure_start is not None:
                    self._content_measure_hover = point
                    self._sync_content_preview_overlay()
                    return True
            if event.type() == QEvent.Type.KeyPress:
                if self._handle_content_key_event(event):
                    return True
        return super().eventFilter(watched, event)

    def _is_content_interaction_widget(self, watched: object) -> bool:
        return watched in {
            self._content_frame_canvas,
            self._content_overlay_canvas,
            self._content_microview_preview_host,
        }

    def _event_global_position(self, event) -> QPoint:
        if hasattr(event, "globalPosition"):
            point = event.globalPosition().toPoint()
            return QPoint(point.x(), point.y())
        if hasattr(event, "globalPos"):
            return event.globalPos()
        return self.cursor().pos()

    def _content_point_from_event(self, watched, event) -> Point | None:
        if watched in {self._content_frame_canvas, self._content_overlay_canvas}:
            canvas = watched if isinstance(watched, DocumentCanvas) else None
            if canvas is None:
                return None
            point = canvas.widget_to_image(event.position())
            document = self._content_preview_document
            if document is None:
                return None
            width, height = document.image_size
            if 0 <= point.x < width and 0 <= point.y < height:
                return point
            return None
        if watched is self._preview_canvas and self._preview_canvas is not None:
            point = self._preview_canvas.widget_to_image(event.position())
            document = self._preview_document
            if document is None:
                return None
            width, height = document.image_size
            if 0 <= point.x < width and 0 <= point.y < height:
                return point
            return None
        if watched in {self._microview_preview_host, self._content_microview_preview_host}:
            point = Point(event.position().x(), event.position().y())
            host = watched if isinstance(watched, MicroviewPreviewHost) else None
            if host is None:
                return None
            width, height = host.native_preview_size()
            if 0 <= point.x < width and 0 <= point.y < height:
                return point
        return None

    def _handle_content_native_tool_click(self, point: Point) -> bool:
        if self._tool_mode not in {"manual", "snap"}:
            return False
        if self._content_measure_start is None:
            self._content_measure_start = point
            self._content_measure_hover = point
            self._sync_content_preview_overlay()
            return True
        line = Line(self._content_measure_start, point)
        self._content_measure_start = None
        self._content_measure_hover = None
        self._sync_content_preview_overlay()
        if line_length(line) < 1.0:
            return True
        self._on_content_canvas_line_committed(self._tool_mode, line)
        return True

    def _handle_content_key_event(self, event) -> bool:
        if event.modifiers() != Qt.KeyboardModifier.NoModifier:
            return False
        if Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_8:
            number = event.key() - Qt.Key.Key_0
            if self._content_fiber_menu_open:
                return self._set_current_content_fiber_by_number(number)
            return self._add_content_count_by_number(number)
        if event.key() == Qt.Key.Key_Escape and self._content_measure_start is not None:
            self._content_measure_start = None
            self._content_measure_hover = None
            self._sync_content_preview_overlay()
            return True
        return False

    def _set_current_content_fiber_by_number(self, number: int) -> bool:
        session = self._load_content_session_from_project()
        if session is None or number < 1 or number > len(session.fibers):
            return False
        session.current_fiber_id = session.fibers[number - 1].id
        self._sync_content_session_to_project()
        self._update_content_ui()
        self._focus_content_input_target()
        return True

    def _add_content_count_by_number(self, number: int) -> bool:
        session = self._load_content_session_from_project()
        if session is None or number < 1 or number > len(session.fibers):
            return False
        fiber = session.fibers[number - 1]
        session.current_fiber_id = fiber.id
        record = ContentExperimentRecord(
            id=new_id("content_rec"),
            kind=ContentRecordKind.COUNT,
            fiber_id=fiber.id,
            field_id=session.current_field_id,
        )
        session.records.append(record)
        self._content_after_record_added(record, fiber)
        self._focus_content_input_target()
        return True

    def _handle_content_canvas_click(self, point: Point, global_pos: QPoint) -> None:
        session = self._load_content_session_from_project()
        if session is None:
            return
        if not session.fibers:
            self.edit_content_experiment_fibers()
            return
        if self._content_measure_start is None:
            self._content_measure_start = point
            self._content_measure_hover = point
            self._sync_content_preview_overlay()
            return
        line = Line(self._content_measure_start, point)
        self._content_measure_start = None
        self._content_measure_hover = None
        if line_length(line) < 1.0:
            self._sync_content_preview_overlay()
            return
        fiber_id = session.current_fiber_id
        if session.selection_mode == ContentSelectionMode.POSTSELECT:
            fiber_id = self._prompt_content_fiber(global_pos)
        fiber = session.fiber_by_id(fiber_id)
        if fiber is None:
            self._sync_content_preview_overlay()
            return
        session.current_fiber_id = fiber.id
        diameter_px = line_length(line)
        calibration = self._current_preview_calibration()
        diameter_unit = calibration.px_to_unit(diameter_px) if calibration is not None else diameter_px
        diameter_unit_name = calibration.unit if calibration is not None else "px"
        record = ContentExperimentRecord(
            id=new_id("content_rec"),
            kind=ContentRecordKind.DIAMETER,
            fiber_id=fiber.id,
            source_mode="manual",
            field_id=session.current_field_id,
            line_px=line,
            diameter_px=diameter_px,
            diameter_unit=diameter_unit,
            diameter_unit_name=diameter_unit_name,
        )
        session.records.append(record)
        self._content_after_record_added(record, fiber)
        self._focus_content_input_target()

    def _current_preview_calibration(self) -> Calibration | None:
        session = self._load_content_session_from_project()
        if (
            session is None
            or session.calibration_pixels_per_unit is None
            or session.calibration_pixels_per_unit <= 0
            or not session.calibration_unit
        ):
            return None
        return Calibration(
            mode="preset",
            pixels_per_unit=session.calibration_pixels_per_unit,
            unit=session.calibration_unit,
            source_label=session.calibration_name or "含量试验标尺",
        )

    def _prompt_content_fiber(self, global_pos: QPoint) -> str | None:
        session = self._load_content_session_from_project()
        if session is None:
            return None
        menu = self._build_content_fiber_menu()
        if menu is None:
            return None
        self._content_fiber_menu_open = True
        action = menu.exec(global_pos)
        self._content_fiber_menu_open = False
        if action is None:
            return None
        return action.data()

    def _show_content_fiber_menu(self, global_pos: QPoint) -> None:
        session = self._load_content_session_from_project()
        if session is None:
            return
        menu = self._build_content_fiber_menu()
        if menu is None:
            return
        self._content_fiber_menu_open = True
        action = menu.exec(global_pos)
        self._content_fiber_menu_open = False
        if action is None:
            return
        session.current_fiber_id = action.data()
        self._sync_content_session_to_project()
        self._update_content_ui()
        self._focus_content_input_target()

    def _build_content_fiber_menu(self) -> QMenu | None:
        session = self._load_content_session_from_project()
        if session is None or not session.fibers:
            return None
        menu = QMenu(self)
        for index, fiber in enumerate(session.fibers, start=1):
            action = menu.addAction(f"{index}. {fiber.name}")
            action.setData(fiber.id)
            action.setCheckable(True)
            action.setChecked(session.current_fiber_id == fiber.id)
            action.setIcon(self._color_icon(fiber.color))
            action.setShortcut(str(index))
        return menu

    def _content_after_record_added(self, record: ContentExperimentRecord, fiber) -> None:
        session = self._load_content_session_from_project()
        if session is None:
            return
        self._sync_content_session_to_project()
        self._check_content_reminders(record, fiber)
        self._refresh_content_workbook()
        self._update_content_ui()

    def _check_content_reminders(self, record: ContentExperimentRecord, fiber) -> None:
        session = self._load_content_session_from_project()
        if session is None:
            return
        if self._app_settings.content_count_reminder_count > 0:
            key = f"count_total_{self._app_settings.content_count_reminder_count}"
            if (
                key not in session.reminders_triggered
                and content_total_count(session) >= self._app_settings.content_count_reminder_count
            ):
                session.reminders_triggered.append(key)
                self._show_content_reminder("计数数量提醒", f"当前计数总数已达到 {content_total_count(session)}。")
        if self._app_settings.content_diameter_reminder_count > 0:
            key = f"diameter_total_{self._app_settings.content_diameter_reminder_count}"
            if (
                key not in session.reminders_triggered
                and content_total_measured(session) >= self._app_settings.content_diameter_reminder_count
            ):
                session.reminders_triggered.append(key)
                self._show_content_reminder("直径测量数量提醒", f"当前直径测量总数已达到 {content_total_measured(session)}。")
        if record.kind == ContentRecordKind.DIAMETER and record.diameter_unit is not None:
            low = fiber.diameter_min
            high = fiber.diameter_max
            if low is not None and record.diameter_unit < low:
                self._show_content_reminder(
                    "直径下限提醒",
                    f"“{fiber.name}”本次直径 {record.diameter_unit:.3f} 低于下限 {low:g}。",
                )
            if high is not None and record.diameter_unit > high:
                self._show_content_reminder(
                    "直径上限提醒",
                    f"“{fiber.name}”本次直径 {record.diameter_unit:.3f} 高于上限 {high:g}。",
                )

    def _show_content_reminder(self, title: str, message: str) -> None:
        self.statusBar().showMessage(message, 6000)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(title)
        box.setText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        box.open()

    def _invalidate_content_field(self, reason: str) -> None:
        session = self._load_content_session_from_project()
        if session is None:
            return
        session.current_field_id += 1
        self._content_measure_start = None
        self._content_measure_hover = None
        self._content_field_baseline = None
        self._content_field_motion_hits = 0
        for canvas in (self._content_frame_canvas, self._content_overlay_canvas):
            if canvas is not None:
                canvas.clear_content_transient_interactions()
        self._sync_content_session_to_project()
        self._sync_content_preview_overlay()
        if session.active:
            self.statusBar().showMessage(f"{reason}，已清空当前视场叠加标记。", 3000)

    def _request_content_field_frame(self) -> None:
        if not (self._content_mode_enabled and self._content_preview_active) or self._content_field_request_pending:
            return
        self._content_field_request_id -= 1
        if self._content_capture_manager.request_analysis_frame(self._content_field_request_id):
            self._content_field_request_pending = True

    def _on_content_field_frame_ready(self, image: object) -> None:
        self._content_field_request_pending = False
        if not (self._content_mode_enabled and self._content_preview_active) or not isinstance(image, QImage) or image.isNull():
            return
        self._content_analysis_frame = image.copy()
        if self._is_content_native_preview():
            self._update_content_preview_document(image, native=True)
        if not self._content_experiment_is_active():
            return
        signature = self._content_frame_signature(image)
        if signature is None:
            return
        if self._content_field_baseline is None:
            self._content_field_baseline = signature
            return
        try:
            import numpy as np

            baseline = self._content_field_baseline
            if baseline is None or getattr(baseline, "shape", None) != signature.shape:
                self._content_field_baseline = signature
                self._invalidate_content_field("实时画面尺寸变化")
                return
            baseline_f = baseline.astype("float32")
            signature_f = signature.astype("float32")
            diff = float(np.mean(np.abs(signature_f - baseline_f)))
            shift = 0.0
            phase_response = 0.0
            hist_delta = 0.0
            try:
                import cv2

                hanning = cv2.createHanningWindow((baseline.shape[1], baseline.shape[0]), cv2.CV_32F)
                (dx, dy), phase_response = cv2.phaseCorrelate(baseline_f, signature_f, hanning)
                shift = float((dx * dx + dy * dy) ** 0.5)
                hist_a, _ = np.histogram(baseline, bins=32, range=(0, 256), density=True)
                hist_b, _ = np.histogram(signature, bins=32, range=(0, 256), density=True)
                hist_delta = float(np.sum(np.abs(hist_a - hist_b)))
            except Exception:
                pass
        except Exception:
            self._content_field_baseline = signature
            return
        moved = diff > 18.0 or (phase_response > 0.15 and shift > 4.0) or hist_delta > 0.18
        if moved:
            self._content_field_motion_hits += 1
            if self._content_field_motion_hits >= 2:
                self._invalidate_content_field("检测到视场移动")
                self._content_field_baseline = signature
            return
        self._content_field_motion_hits = 0
        self._content_field_baseline = signature

    def _on_content_field_frame_failed(self, message: str) -> None:
        self._content_field_request_pending = False
        if message:
            self.statusBar().showMessage(message, 2500)

    def _content_frame_signature(self, image: QImage):
        try:
            import numpy as np

            gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
            gray = gray.scaled(160, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            height = gray.height()
            width = gray.width()
            bytes_per_line = gray.bytesPerLine()
            buffer = gray.constBits()
            array = np.frombuffer(buffer, dtype=np.uint8, count=bytes_per_line * height)
            array = array.reshape((height, bytes_per_line))[:, :width]
            return array.copy()
        except Exception:
            return None

    def _update_ui_for_current_document(self) -> None:
        document = self.current_document()
        self._load_content_session_from_project()
        if self._content_mode_enabled:
            self._populate_content_group_list()
        else:
            self._populate_group_list(document)
        self._update_calibration_panel(document)
        self._populate_measurement_table(document)
        self._update_image_resolution_label(document)
        self._update_statusbar_aux_labels()
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.set_settings(self._app_settings)
            active_mode = self._coerce_content_tool_mode(self._tool_mode) if self._content_mode_enabled else self._tool_mode
            if self._content_mode_enabled and active_mode != self._tool_mode:
                self._tool_mode = active_mode
                if active_mode in self._mode_actions:
                    self._mode_actions[active_mode].setChecked(True)
            canvas.set_tool_mode("select" if self._preview_active and canvas is self._preview_canvas else active_mode)
            canvas.set_show_area_fill(False if (self._preview_active and canvas is self._preview_canvas) or self._content_mode_enabled else self._show_area_fill)
        if self._content_mode_enabled:
            for content_canvas in (self._content_frame_canvas, self._content_overlay_canvas):
                if content_canvas is None or content_canvas is canvas:
                    continue
                content_canvas.set_settings(self._app_settings)
                content_canvas.set_tool_mode(self._coerce_content_tool_mode(self._tool_mode), overlay_kind=self._overlay_tool_kind)
                content_canvas.set_show_area_fill(False)
        self._update_content_ui()
        self._update_action_states()

    def _update_calibration_panel(self, document: ImageDocument | None) -> None:
        if self._content_mode_enabled:
            session = self._load_content_session_from_project()
            if session is not None and session.calibration_pixels_per_unit is not None and session.calibration_pixels_per_unit > 0:
                unit = session.calibration_unit or "unit"
                name = session.calibration_name or "含量试验标尺"
                self._set_calibration_label(
                    f"含量试验标尺: {name} ({session.calibration_pixels_per_unit:.4f} px/{unit})",
                    status="calibrated",
                )
            else:
                self._set_calibration_label("含量试验未选择标尺，直径按 px 记录", status="preview")
            return
        if self._preview_active:
            self._set_calibration_label("实时预览中", status="preview")
            return
        if document is None or document.calibration is None:
            self._set_calibration_label("当前图片未标定", status="uncalibrated")
            return
        calibration = document.calibration
        lines = [
            calibration.source_label or self._format_calibration_mode(calibration.mode),
            f"{self._format_calibration_mode(calibration.mode)}\n{calibration.pixels_per_unit:.4f} px/{calibration.unit}",
        ]
        if calibration.mode == "project_default" or not document.uses_sidecar():
            lines.append("保存位置: 当前项目")
        else:
            lines.append(f"侧车: {Path(document.sidecar_path or document.default_sidecar_path()).name}")
        self._set_calibration_label("\n".join(lines), status="calibrated")

    def _populate_measurement_table(self, document: ImageDocument | None) -> None:
        self._table_rebuilding = True
        self.measurement_table.setRowCount(0)
        if self._content_mode_enabled:
            self.measurement_table.setHorizontalHeaderLabels(["类别", "类型", "结果", "单位", "模式", "视场", "状态", "ID"])
            session = self._load_content_session_from_project()
            if session is not None:
                for row, record in enumerate(reversed(session.records)):
                    self.measurement_table.insertRow(row)
                    fiber = session.fiber_by_id(record.fiber_id)
                    display_id = record.id.split("_")[-1]
                    id_item = QTableWidgetItem(display_id)
                    id_item.setData(Qt.ItemDataRole.UserRole, record.id)
                    kind_label = "计数" if record.kind == ContentRecordKind.COUNT else "直径"
                    unit_text = "" if record.kind == ContentRecordKind.COUNT else (record.diameter_unit_name or "px")
                    self.measurement_table.setItem(row, self.TABLE_COL_GROUP, QTableWidgetItem(fiber.name if fiber is not None else ""))
                    self.measurement_table.setItem(row, self.TABLE_COL_KIND, QTableWidgetItem(kind_label))
                    self.measurement_table.setItem(row, self.TABLE_COL_RESULT, QTableWidgetItem(record.display_value()))
                    self.measurement_table.setItem(row, self.TABLE_COL_UNIT, QTableWidgetItem(unit_text))
                    self.measurement_table.setItem(row, self.TABLE_COL_MODE, QTableWidgetItem(self._format_content_source_mode(record)))
                    self.measurement_table.setItem(row, self.TABLE_COL_CONFIDENCE, QTableWidgetItem(str(record.field_id)))
                    self.measurement_table.setItem(row, self.TABLE_COL_STATUS, QTableWidgetItem("已记录"))
                    self.measurement_table.setItem(row, self.TABLE_COL_ID, id_item)
            self._table_rebuilding = False
            return
        self.measurement_table.setHorizontalHeaderLabels(["种类", "类型", "结果", "单位", "模式", "置信度", "状态", "ID"])
        if document is not None:
            for row, measurement in enumerate(document.measurements):
                self.measurement_table.insertRow(row)
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
        self._table_rebuilding = False
        if document is not None:
            self._sync_measurement_table_selection(document)

    def _format_content_source_mode(self, record: ContentExperimentRecord) -> str:
        if record.kind == ContentRecordKind.COUNT:
            return "数字计数"
        return {
            "manual": "手动线段",
            "snap": "边缘吸附",
            "fiber_quick": "快速测径",
        }.get(record.source_mode, "含量试验")

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
        if self._content_mode_enabled:
            self._update_action_states()
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
        if self._content_mode_enabled:
            session = self._load_content_session_from_project()
            if session is None:
                return
            selected_items = self.group_list.selectedItems()
            if selected_items:
                session.current_fiber_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
                self._sync_content_session_to_project()
            self._sync_group_list_item_widgets()
            self._update_content_ui()
            self._update_action_states()
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

    def _focus_content_input_target(self) -> None:
        if self._content_mode_enabled:
            canvas = self._content_current_canvas()
            if canvas is not None:
                canvas.focus_canvas()
            elif self._content_microview_preview_host is not None:
                self._content_microview_preview_host.setFocus(Qt.FocusReason.OtherFocusReason)

    def _should_handle_content_hotkeys(self) -> bool:
        if QApplication.activeModalWidget() is not None:
            return False
        focus_widget = QApplication.focusWidget()
        if focus_widget is None:
            return True
        if isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return False
        if isinstance(focus_widget, QComboBox):
            return False
        return True

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
        content_mode = self._content_mode_enabled
        selection_model = self.measurement_table.selectionModel() if hasattr(self, "measurement_table") else None
        has_selected_rows = bool(selection_model and selection_model.selectedRows())
        has_selected_content_rows = bool(content_mode and has_selected_rows)
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
        content_session = self._load_content_session_from_project() if content_mode else None
        has_content_fibers = bool(content_session and content_session.fibers)
        self.close_current_action.setEnabled(has_document)
        self.close_all_action.setEnabled(bool(self.project.documents))
        self.delete_measurement_action.setEnabled((has_selected_object and not preview_active and not content_mode) or has_selected_content_rows)
        self.delete_measurement_button.setEnabled((has_selected_object and not preview_active and not content_mode) or has_selected_content_rows)
        if self._delete_group_measurements_button is not None:
            self._delete_group_measurements_button.setEnabled(has_measurement_groups and not preview_active and not content_mode)
        if self._delete_all_measurements_button is not None:
            self._delete_all_measurements_button.setEnabled(has_measurements and not preview_active and not content_mode)
        self.add_group_action.setEnabled((has_document and not preview_active and not content_mode) or content_mode)
        self.rename_group_action.setEnabled((has_named_active_group and not preview_active and not content_mode) or (content_mode and has_content_fibers))
        self.delete_group_action.setEnabled((has_deletable_group_target and not preview_active and not content_mode) or (content_mode and has_content_fibers))
        if self._add_group_button is not None:
            self._add_group_button.setEnabled((has_document and not preview_active and not content_mode) or content_mode)
        if self._rename_group_button is not None:
            self._rename_group_button.setEnabled((has_named_active_group and not preview_active and not content_mode) or (content_mode and has_content_fibers))
        if self.delete_group_button is not None:
            self.delete_group_button.setEnabled((has_deletable_group_target and not preview_active and not content_mode) or (content_mode and has_content_fibers))
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
            self._apply_preset_button.setEnabled(has_preset and ((has_document and not preview_active and not content_mode) or content_mode))
        if self._area_auto_button is not None:
            self._area_auto_button.setEnabled(has_document and bool(self._app_settings.area_model_mappings) and not preview_active and not content_mode)
        self.undo_action.setEnabled(bool(history and history.can_undo()) and not preview_active and not content_mode)
        self.redo_action.setEnabled(bool(history and history.can_redo()) and not preview_active and not content_mode)
        capture_feature_available = _CAPTURE_IMPORT_ERROR is None
        self.switch_capture_device_action.setEnabled(capture_feature_available)
        self.live_preview_action.setEnabled(capture_feature_available and not content_mode)
        if hasattr(self, "content_experiment_action"):
            self.content_experiment_action.setEnabled(capture_feature_available)
        can_optimize_signal = capture_feature_available and self._capture_manager.can_optimize_signal()
        analysis_active = self._preview_analysis_mode != "none"
        self.capture_frame_action.setEnabled(preview_active and self._capture_manager.can_capture_still() and not analysis_active)
        self.optimize_capture_signal_action.setVisible(can_optimize_signal)
        self.optimize_capture_signal_action.setEnabled(can_optimize_signal and not analysis_active)
        for mode, action in self._mode_actions.items():
            if content_mode:
                action.setEnabled(mode in self._content_allowed_tool_modes())
            else:
                action.setEnabled(not preview_active or mode == "select")
        if self._manual_tool_button is not None:
            self._manual_tool_button.setEnabled((not preview_active and not content_mode) or content_mode)
        if self._area_tool_button is not None:
            self._area_tool_button.setEnabled(not preview_active and not content_mode)
        if self._magic_tool_button is not None:
            self._magic_tool_button.setEnabled((not preview_active and not content_mode) or content_mode)
        if self._overlay_tool_button is not None:
            self._overlay_tool_button.setEnabled(not preview_active and not content_mode)
        self._update_path_drawing_controls()
        self._update_magic_segment_controls()
        self._update_preview_analysis_controls()
        self._sync_manual_tool_button()
        self._sync_area_tool_button()
        self._sync_magic_tool_button()
        self._sync_overlay_tool_button()

    def _magic_prompt_label_text(self, prompt_type: str) -> str:
        return "当前提示：负采样点" if prompt_type == "negative" else "当前提示：正采样点"

    def _magic_operation_label_text(self, operation_mode: str) -> str:
        if operation_mode == MagicSegmentOperationMode.SUBTRACT:
            return "当前编辑：剔除形状"
        return "当前编辑：第一形状"

    def _magic_operation_button_text(self, operation_mode: str) -> str:
        if operation_mode == MagicSegmentOperationMode.SUBTRACT:
            return "剔除(T)"
        return "添加(T)"

    def _current_magic_roi_enabled(self, tool_mode: str | None = None) -> bool:
        active_mode = str(tool_mode or self._tool_mode or "").strip()
        if is_fiber_quick_tool_mode(active_mode):
            return bool(self._fiber_quick_roi_enabled)
        if is_magic_segment_tool_mode(active_mode):
            return bool(self._magic_standard_roi_enabled)
        return False

    def _set_magic_roi_enabled(self, tool_mode: str, enabled: bool) -> None:
        if is_fiber_quick_tool_mode(tool_mode):
            self._fiber_quick_roi_enabled = bool(enabled)
        elif is_magic_segment_tool_mode(tool_mode):
            self._magic_standard_roi_enabled = bool(enabled)

    def _toggle_active_magic_roi(self) -> None:
        if not (is_magic_segment_tool_mode(self._tool_mode) or is_fiber_quick_tool_mode(self._tool_mode)):
            return
        self._set_magic_roi_enabled(self._tool_mode, not self._current_magic_roi_enabled())
        state_text = "启用" if self._current_magic_roi_enabled() else "关闭"
        self.statusBar().showMessage(f"已{state_text}ROI局部分割", 2500)
        self._update_magic_segment_controls()

    def _update_magic_segment_controls(self) -> None:
        if self._magic_controls_widget is None or self._measurement_tool_strip is None:
            return
        is_visible = is_magic_toolbar_tool_mode(self._tool_mode) and (
            not self._preview_active
            or (self._content_mode_enabled and is_fiber_quick_tool_mode(self._tool_mode))
        )
        self._measurement_tool_strip.setMagicContextVisible(is_visible)
        if not is_visible:
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
            busy = bool(canvas and canvas.is_magic_segment_busy())
        elif fiber_quick_mode:
            prompt_type = canvas.current_fiber_quick_prompt_type() if canvas is not None else "positive"
            operation_mode = MagicSegmentOperationMode.ADD
            busy = bool(canvas and canvas.is_fiber_quick_busy())
        else:
            prompt_type = "positive"
            operation_mode = MagicSegmentOperationMode.ADD
            busy = bool(canvas and canvas.is_reference_instance_busy())
        if self._magic_prompt_label is not None:
            self._magic_prompt_label.setVisible(not standard_mode and not fiber_quick_mode)
            if not standard_mode and not fiber_quick_mode and canvas is not None and canvas.has_reference_instance_preview():
                self._magic_prompt_label.setText("候选预览")
            elif not standard_mode and not fiber_quick_mode:
                self._magic_prompt_label.setText("拖框或点已确认面积作为参考")
        if self._magic_toggle_button is not None:
            self._magic_toggle_button.setVisible(standard_mode or fiber_quick_mode)
            self._magic_toggle_button.setEnabled(
                has_document
                and (standard_mode or fiber_quick_mode)
                and (fiber_quick_mode or not busy)
            )
        if self._magic_roi_button is not None:
            self._magic_roi_button.setVisible(standard_mode or fiber_quick_mode)
            self._magic_roi_button.setChecked(self._current_magic_roi_enabled())
            self._magic_roi_button.setText("ROI")
            self._magic_roi_button.setEnabled(has_document and (standard_mode or fiber_quick_mode))
        if self._magic_operation_button is not None:
            self._magic_operation_button.setVisible(standard_mode)
            self._magic_operation_button.setText(self._magic_operation_button_text(operation_mode))
            self._magic_operation_button.setEnabled(has_document and standard_mode and not busy)
        if self._magic_confirm_subtract_button is not None:
            self._magic_confirm_subtract_button.setVisible(standard_mode and operation_mode == MagicSegmentOperationMode.SUBTRACT)
            self._magic_confirm_subtract_button.setEnabled(
                bool(canvas and canvas.can_confirm_current_magic_subtract_shape())
            )
        if self._magic_complete_button is not None:
            self._magic_complete_button.setText(
                "完成"
                if standard_mode
                else ("确认(F)" if fiber_quick_mode else "加入")
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
            self._magic_cancel_button.setText("取消(Esc)" if fiber_quick_mode else "取消")
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

    def _commit_active_path_drawing(self) -> bool:
        canvas = self.current_canvas()
        if canvas is None:
            return False
        committed = canvas.commit_pending_path()
        self._update_path_drawing_controls()
        return committed

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
        is_visible = self._tool_mode in {"polygon_area", "continuous_manual"} and not self._preview_active
        self._measurement_tool_strip.setPathContextVisible(is_visible)
        if not is_visible:
            return
        canvas = self.current_canvas()
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
            return self.MAP_BUILD_AVAILABLE and selected.backend_key == "microview"
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
        map_tooltip = "地图构建功能开发中，当前版本暂不可用。"
        if self.MAP_BUILD_AVAILABLE and selected is not None and selected.backend_key == "microview":
            map_tooltip = "实时预览分析：地图构建"
        focus_enabled = is_visible and focus_supported and not self._preview_analysis_finalizing
        map_enabled = is_visible and not self._preview_analysis_finalizing and (
            map_supported or not self.MAP_BUILD_AVAILABLE
        )
        if self._focus_stack_button is not None:
            self._focus_stack_button.setEnabled(focus_enabled)
            self._focus_stack_button.setToolTip(focus_tooltip)
        if self._map_build_button is not None:
            self._map_build_button.setEnabled(map_enabled)
            self._map_build_button.setToolTip(map_tooltip)
        self._sync_preview_analysis_buttons()

    def _preview_analysis_intro_text(self, mode: str) -> str:
        if mode == "map_build":
            return "移动样品台并适当切换焦距，系统会先对每个 tile 做景深合成，再实时拼接地图。按 Enter 或 F 结束，Esc 取消。"
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
                message = "地图构建功能开发中，当前版本暂不可用。"
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
            parent=self,
        )
        dialog.finishRequested.connect(self._finalize_preview_analysis_session)
        dialog.cancelRequested.connect(lambda: self._cancel_preview_analysis_session())
        return dialog

    def _start_preview_analysis_session(self, mode: str) -> None:
        if mode not in {"focus_stack", "map_build"}:
            return
        selected = self._selected_capture_device()
        if selected is None:
            return
        self._clear_magic_segment_sessions()
        self._preview_analysis_mode = mode
        self._preview_analysis_request_pending = False
        self._preview_analysis_finalizing = False
        self._preview_analysis_dialog = self._create_preview_analysis_dialog(mode)
        self._preview_analysis_dialog.show()
        self._preview_analysis_dialog.raise_()
        self._preview_analysis_dialog.activateWindow()

        thread = QThread(self)
        worker = (
            FocusStackSessionWorker(
                device_id=selected.id,
                device_name=selected.name,
                render_config=self._current_focus_stack_render_config(),
            )
            if mode == "focus_stack"
            else MapBuildSessionWorker(device_id=selected.id, device_name=selected.name)
        )
        worker.moveToThread(thread)
        worker.previewUpdated.connect(self._on_preview_analysis_worker_preview)
        worker.finished.connect(self._on_preview_analysis_worker_finished)
        worker.failed.connect(self._on_preview_analysis_worker_failed)
        thread.finished.connect(worker.deleteLater)
        thread.start()

        self._preview_analysis_thread = thread
        self._preview_analysis_worker = worker
        self._preview_analysis_timer.start()
        self._request_preview_analysis_frame()
        self.statusBar().showMessage(f"{self._analysis_mode_label(mode)}已启动", 3000)
        self._update_action_states()

    def _teardown_preview_analysis_session(self, *, cancel_worker: bool, status_message: str | None = None) -> None:
        self._preview_analysis_timer.stop()
        self._preview_analysis_request_pending = False
        self._preview_analysis_finalizing = False
        worker = self._preview_analysis_worker
        thread = self._preview_analysis_thread
        dialog = self._preview_analysis_dialog
        self._preview_analysis_worker = None
        self._preview_analysis_thread = None
        self._preview_analysis_dialog = None
        self._preview_analysis_mode = "none"
        if worker is not None and cancel_worker:
            try:
                worker.cancelRequested.emit()
            except Exception:
                pass
        if thread is not None:
            thread.quit()
            thread.wait(2000)
        if dialog is not None:
            close_silently = getattr(dialog, "close_silently", None)
            if callable(close_silently):
                close_silently()
            else:
                close = getattr(dialog, "close", None)
                if callable(close):
                    close()
        if status_message:
            self.statusBar().showMessage(status_message, 4000)
        self._update_action_states()

    def _cancel_preview_analysis_session(self, *, message: str | None = None) -> None:
        if self._preview_analysis_mode == "none":
            self._sync_preview_analysis_buttons()
            return
        self._teardown_preview_analysis_session(cancel_worker=True, status_message=message)

    def _finalize_preview_analysis_session(self) -> None:
        if self._preview_analysis_mode == "none" or self._preview_analysis_worker is None or self._preview_analysis_finalizing:
            return
        self._preview_analysis_finalizing = True
        self._preview_analysis_timer.stop()
        self._preview_analysis_request_pending = False
        if self._preview_analysis_dialog is not None:
            busy_message = self._preview_analysis_finalize_message(self._preview_analysis_mode)
            self._preview_analysis_dialog.set_status(busy_message)
            self._preview_analysis_dialog.set_busy(True, busy_message)
        self._preview_analysis_worker.finalizeRequested.emit()
        self._update_action_states()

    def _request_preview_analysis_frame(self) -> None:
        if (
            self._preview_analysis_mode == "none"
            or self._preview_analysis_worker is None
            or self._preview_analysis_request_pending
            or self._preview_analysis_finalizing
        ):
            return
        self._preview_analysis_request_id += 1
        request_id = self._preview_analysis_request_id
        if self._capture_manager.request_analysis_frame(request_id):
            self._preview_analysis_request_pending = True

    def _on_preview_analysis_frame_ready(self, request_id: int, image: object) -> None:
        if request_id == self._content_field_request_id:
            self._on_content_field_frame_ready(image)
            return
        if request_id != self._preview_analysis_request_id:
            return
        self._preview_analysis_request_pending = False
        if self._preview_analysis_mode == "none" or self._preview_analysis_worker is None or self._preview_analysis_finalizing:
            return
        if isinstance(image, QImage) and not image.isNull():
            self._preview_analysis_worker.frameSubmitted.emit(image.copy())

    def _on_preview_analysis_frame_failed(self, request_id: int, message: str) -> None:
        if request_id == self._content_field_request_id:
            self._on_content_field_frame_failed(message)
            return
        if request_id != self._preview_analysis_request_id:
            return
        self._preview_analysis_request_pending = False
        if self._preview_analysis_dialog is not None:
            self._preview_analysis_dialog.set_status(message)
        self.statusBar().showMessage(message, 4000)

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
        prompt_type = canvas.cycle_magic_segment_prompt_type()
        self.statusBar().showMessage(self._magic_prompt_label_text(prompt_type), 2500)
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
        before_mode = canvas.current_magic_segment_operation_mode()
        operation_mode = canvas.cycle_magic_segment_operation_mode()
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
        is_content_job = self._is_content_preview_document_id(document_id)
        document = None if is_content_job else self.project.get_document(document_id)
        if document is None and not is_content_job:
            return
        if self._fiber_quick_geometry_worker is not None:
            self._fiber_quick_geometry_worker.cancel_document(document_id)
        self._ensure_fiber_quick_commit_geometry_worker()
        if self._fiber_quick_commit_geometry_worker is None:
            return
        self._fiber_quick_background_job_serial += 1
        job_id = self._fiber_quick_background_job_serial
        self._fiber_quick_background_jobs[(document_id, job_id)] = {
            "content_mode": is_content_job,
            "fiber_group_id": document.active_group_id if document is not None else None,
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
        if bool(job_meta.get("content_mode")):
            if self._append_content_diameter_record(result.line_px, source_mode="fiber_quick"):
                self.statusBar().showMessage("快速测径已在后台完成并写入含量试验。", 3000)
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
        if canvas.has_magic_segment_session():
            canvas.clear_magic_segment_session()
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
        for document_id, canvas in self._canvases.items():
            if document_id == except_document_id:
                continue
            if canvas.has_magic_segment_session():
                if self._prompt_seg_worker is not None:
                    self._prompt_seg_worker.cancel_document(document_id)
                canvas.clear_magic_segment_session()
            if canvas.has_reference_instance_session():
                canvas.clear_reference_instance_session()
            if canvas.has_fiber_quick_session():
                if self._fiber_quick_geometry_worker is not None:
                    self._fiber_quick_geometry_worker.cancel_document(document_id)
                if self._prompt_seg_worker is not None:
                    self._prompt_seg_worker.cancel_document(document_id)
                canvas.clear_fiber_quick_session()

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
        if self._preview_analysis_mode != "none" and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F):
                self._finalize_preview_analysis_session()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Escape:
                self._cancel_preview_analysis_session()
                event.accept()
                return
        if self._tool_mode in {"polygon_area", "continuous_manual"} and event.modifiers() == Qt.KeyboardModifier.NoModifier:
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
                    event.key() == Qt.Key.Key_S
                    and canvas is not None
                    and canvas.current_magic_segment_operation_mode() == MagicSegmentOperationMode.SUBTRACT
                ):
                    if self._confirm_current_magic_subtract_shape():
                        event.accept()
                        return
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F):
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
            and self._content_experiment_is_active()
            and Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_8
            and self._should_handle_content_hotkeys()
        ):
            number = event.key() - Qt.Key.Key_0
            handled = (
                self._set_current_content_fiber_by_number(number)
                if self._content_fiber_menu_open
                else self._add_content_count_by_number(number)
            )
            if handled:
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
        try:
            super().keyPressEvent(event)
        except TypeError:
            return

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            canvas = self.current_canvas()
            if canvas is not None:
                canvas.set_temporary_grab_pressed(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow() and self._content_experiment_is_active():
            self._invalidate_content_field("窗口失焦")
        super().changeEvent(event)

    def _shutdown_background_threads(self) -> None:
        if self._preview_analysis_mode != "none":
            self._teardown_preview_analysis_session(cancel_worker=True)

        if self._load_worker is not None:
            try:
                self._load_worker.cancel()
            except Exception:
                pass
        if self._load_thread is not None:
            self._load_thread.quit()
            self._load_thread.wait(2000)
        if self._load_progress_dialog is not None:
            self._load_progress_dialog.close()
            self._load_progress_dialog.deleteLater()
        self._load_thread = None
        self._load_worker = None
        self._load_progress_dialog = None
        self._load_state = None

        if self._area_infer_worker is not None:
            try:
                self._area_infer_worker.cancel()
            except Exception:
                pass
        if self._area_infer_thread is not None:
            self._area_infer_thread.quit()
            self._area_infer_thread.wait(2000)
        if self._area_infer_progress_dialog is not None:
            self._area_infer_progress_dialog.close()
            self._area_infer_progress_dialog.deleteLater()
        self._area_infer_thread = None
        self._area_infer_worker = None
        self._area_infer_progress_dialog = None
        self._area_infer_state = None

        if self._fiber_quick_geometry_worker is not None:
            for document_id in list(self._canvases.keys()):
                self._fiber_quick_geometry_worker.cancel_document(document_id)
        if self._fiber_quick_geometry_thread is not None:
            self._fiber_quick_geometry_thread.quit()
            self._fiber_quick_geometry_thread.wait(2000)
        self._fiber_quick_geometry_thread = None
        self._fiber_quick_geometry_worker = None
        self._fiber_quick_geometry_request_ids.clear()

        if self._fiber_quick_commit_geometry_worker is not None:
            document_ids = {document_id for document_id, _request_id in self._fiber_quick_background_jobs.keys()}
            document_ids.update(self._canvases.keys())
            for document_id in document_ids:
                self._fiber_quick_commit_geometry_worker.cancel_document(document_id)
        if self._fiber_quick_commit_geometry_thread is not None:
            self._fiber_quick_commit_geometry_thread.quit()
            self._fiber_quick_commit_geometry_thread.wait(2000)
        self._fiber_quick_commit_geometry_thread = None
        self._fiber_quick_commit_geometry_worker = None
        self._fiber_quick_background_jobs.clear()

        if self._prompt_seg_worker is not None:
            for document_id in list(self._canvases.keys()):
                self._prompt_seg_worker.cancel_document(document_id)
        if self._prompt_seg_thread is not None:
            self._prompt_seg_thread.quit()
            self._prompt_seg_thread.wait(2000)
        self._prompt_seg_thread = None
        self._prompt_seg_worker = None
        self._prompt_request_tool_modes.clear()

        if self._reference_instance_thread is not None:
            self._reference_instance_thread.quit()
            self._reference_instance_thread.wait(1500)
        self._reference_instance_thread = None
        self._reference_instance_worker = None

    def closeEvent(self, event: QCloseEvent) -> None:
        offscreen_test_close = QApplication.platformName().lower() == "offscreen" and not self.isVisible()
        if not offscreen_test_close and not self._confirm_close_documents(self.project.documents):
            event.ignore()
            return
        self._persist_window_geometry()
        self.stop_content_preview()
        self.stop_live_preview()
        self._content_workbook_service.close()
        self._clear_prompt_segmentation_cache()
        self._shutdown_background_threads()
        event.accept()
