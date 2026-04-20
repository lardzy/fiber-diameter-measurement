from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QPointF, QRectF, QSize, Qt, QThread, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QCloseEvent, QGuiApplication, QIcon, QImage, QImageReader, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
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
from fdm.geometry import Line, Point, line_length
from fdm.models import (
    Calibration,
    CalibrationPreset,
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
        try:
            AppSettingsIO.save(self._app_settings)
        except OSError:
            pass
        self._document_order: list[str] = []
        self._images: dict[str, QImage] = {}
        self._canvases: dict[str, DocumentCanvas] = {}
        self._tool_mode = "select"
        self._last_non_select_tool: str | None = None
        self._overlay_tool_kind = OverlayAnnotationKind.TEXT
        self._group_list_rebuilding = False
        self._table_rebuilding = False
        self._file_toolbar: QToolBar | None = None
        self._measure_toolbar: QToolBar | None = None
        self._measurement_tool_strip: MeasurementToolStrip | None = None
        self._magic_tool_mode = MagicSegmentToolMode.STANDARD
        self._magic_tool_button: OverlayToolSplitButton | None = None
        self._magic_tool_menu: QMenu | None = None
        self._magic_subtool_actions: dict[str, QAction] = {}
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
        self._reference_instance_thread: QThread | None = None
        self._reference_instance_worker: ReferenceInstancePropagationWorker | None = None
        self._prompt_request_tool_modes: dict[tuple[str, int], str] = {}
        self._fiber_quick_geometry_request_ids: set[tuple[str, int]] = set()
        self._interactive_segmentation_services: dict[str, object] = {}
        self._show_area_fill = True
        self._area_auto_button: QPushButton | None = None
        self._magic_controls_widget: QWidget | None = None
        self._magic_prompt_label: QLabel | None = None
        self._magic_toggle_button: QToolButton | None = None
        self._magic_operation_button: QToolButton | None = None
        self._magic_complete_button: QToolButton | None = None
        self._magic_cancel_button: QToolButton | None = None
        self._preview_analysis_widget: QWidget | None = None
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
        self._center_stack: QStackedWidget | None = None
        self._preview_page: QWidget | None = None
        self._preview_display_stack: QStackedWidget | None = None
        self._preview_canvas: DocumentCanvas | None = None
        self._microview_preview_host: MicroviewPreviewHost | None = None
        self._microview_preview_scroll: QScrollArea | None = None
        self._preview_status_label: QLabel | None = None
        self._image_resolution_label: QLabel | None = None
        self._preview_notice_label: QLabel | None = None
        self._calibration_label_scroll: QScrollArea | None = None
        self._version_label: QLabel | None = None
        self._preview_active = False
        self._preview_document: ImageDocument | None = None
        self._capture_devices: list[CaptureDevice] = []
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

        self._build_ui()
        self._capture_manager.devicesChanged.connect(self._on_capture_devices_changed)
        self._capture_manager.previewStateChanged.connect(self._on_live_preview_state_changed)
        self._capture_manager.frameReady.connect(self._on_live_preview_frame_ready)
        self._capture_manager.analysisFrameReady.connect(self._on_preview_analysis_frame_ready)
        self._capture_manager.analysisFrameFailed.connect(self._on_preview_analysis_frame_failed)
        self._capture_manager.errorOccurred.connect(self._on_capture_error)
        self._capture_devices = self._capture_manager.devices()
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

        self.rename_group_action = QAction("重命名当前类别", self)
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
            ("manual", "手动测量"),
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
        file_toolbar.addAction(self.capture_frame_action)
        file_toolbar.addAction(self.optimize_capture_signal_action)

        self._measure_toolbar = None
        self._measurement_tool_strip = self._build_measurement_tool_strip()

    def _build_measurement_tool_strip(self) -> MeasurementToolStrip:
        strip = MeasurementToolStrip(self)
        for mode in [
            "select",
            "manual",
            "snap",
            "polygon_area",
            "freehand_area",
        ]:
            strip.addModeAction(mode, self._mode_actions[mode])
        self._magic_tool_button = self._build_magic_tool_button()
        strip.setMagicToolButton(self._magic_tool_button)
        strip.addModeAction("calibration", self._mode_actions["calibration"])
        self._overlay_tool_button = self._build_overlay_tool_button()
        strip.setOverlayButton(self._overlay_tool_button)
        self._magic_controls_widget = self._build_magic_segment_controls()
        strip.setMagicContextWidget(self._magic_controls_widget)
        self._preview_analysis_widget = self._build_preview_analysis_controls()
        strip.setPreviewContextWidget(self._preview_analysis_widget)
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
        self._magic_toggle_button.setText("切换正负 (R)")
        self._magic_toggle_button.clicked.connect(self._cycle_active_magic_prompt_type)
        layout.addWidget(self._magic_toggle_button)

        self._magic_operation_button = QToolButton(container)
        self._magic_operation_button.setProperty("contextTool", True)
        self._magic_operation_button.setText("模式：添加 (T)")
        self._magic_operation_button.clicked.connect(self._cycle_magic_segment_operation_mode)
        layout.addWidget(self._magic_operation_button)

        self._magic_complete_button = QToolButton(container)
        self._magic_complete_button.setProperty("contextTool", True)
        self._magic_complete_button.setText("完成 (Enter / F)")
        self._magic_complete_button.clicked.connect(self._commit_active_magic_preview)
        layout.addWidget(self._magic_complete_button)

        self._magic_cancel_button = QToolButton(container)
        self._magic_cancel_button.setProperty("contextTool", True)
        self._magic_cancel_button.setText("放弃 (Esc)")
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

    def _build_magic_tool_button(self) -> OverlayToolSplitButton:
        button = OverlayToolSplitButton(self)
        button.setText(self._magic_tool_label(self._magic_tool_mode))
        button.primaryTriggered.connect(lambda: self._activate_magic_tool(self._magic_tool_mode))

        menu = QMenu(self)
        menu.setObjectName("magicToolMenu")
        menu.setStyleSheet(
            """
            QMenu#magicToolMenu {
                background: #23282E;
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 12px;
                padding: 8px;
            }
            QMenu#magicToolMenu::item {
                min-height: 38px;
                margin: 2px 0;
                padding: 0 16px 0 12px;
                border-radius: 8px;
                color: #F3F4F6;
                font-weight: 600;
            }
            QMenu#magicToolMenu::item:selected {
                background: #2D343C;
            }
            QMenu#magicToolMenu::item:checked {
                background: rgba(217, 108, 117, 41);
            }
            QMenu#magicToolMenu::icon {
                padding-left: 2px;
            }
            QMenu#magicToolMenu::indicator {
                width: 0px;
                height: 0px;
            }
            """
        )
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
        menu.setStyleSheet(
            """
            QMenu#overlayToolMenu {
                background: #23282E;
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 12px;
                padding: 8px;
            }
            QMenu#overlayToolMenu::item {
                min-height: 38px;
                margin: 2px 0;
                padding: 0 16px 0 12px;
                border-radius: 8px;
                color: #F3F4F6;
                font-weight: 600;
            }
            QMenu#overlayToolMenu::item:selected {
                background: #2D343C;
            }
            QMenu#overlayToolMenu::item:checked {
                background: rgba(183, 154, 216, 41);
            }
            QMenu#overlayToolMenu::icon {
                padding-left: 2px;
            }
            QMenu#overlayToolMenu::indicator {
                width: 0px;
                height: 0px;
            }
            """
        )
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

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        self._left_panel = container
        container.setMinimumWidth(280)
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
        count_header = QLabel("当前/总数")
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
        self._rename_group_button = QPushButton("重命名")
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
        splitter.addWidget(image_box)
        splitter.addWidget(group_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([280, 420])
        layout.addWidget(splitter, 1)
        self._update_group_list_header_styles()
        return container

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
        self._preview_display_stack.addWidget(self._preview_canvas)
        self._microview_preview_host = MicroviewPreviewHost()
        self._microview_preview_host.metricsChanged.connect(self._on_preview_host_metrics_changed)
        self._microview_preview_scroll = QScrollArea()
        self._microview_preview_scroll.setWidget(self._microview_preview_host)
        self._microview_preview_scroll.setWidgetResizable(False)
        self._microview_preview_scroll.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._microview_preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
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

        measurement_box = QGroupBox("测量记录")
        measurement_layout = QVBoxLayout(measurement_box)
        self.measurement_table = QTableWidget(0, 8)
        self.measurement_table.setHorizontalHeaderLabels(["种类", "类型", "结果", "单位", "模式", "置信度", "状态", "ID"])
        header = self.measurement_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.measurement_table.setColumnWidth(self.TABLE_COL_GROUP, 100)
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
        self.measurement_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.measurement_table.itemSelectionChanged.connect(self._on_measurement_selection_changed)
        measurement_layout.addWidget(self.measurement_table)
        self.delete_measurement_button = QPushButton("删除选中对象")
        self.delete_measurement_button.setIcon(themed_icon("delete", color="#F28482"))
        self.delete_measurement_button.clicked.connect(self.delete_selected_measurement)
        measurement_layout.addWidget(self.delete_measurement_button)

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
        if mode != "select":
            self._last_non_select_tool = mode
        self._tool_mode = mode
        for canvas in self._canvases.values():
            canvas.set_tool_mode(mode, overlay_kind=self._overlay_tool_kind)
        if mode in self._mode_actions:
            self._mode_actions[mode].setChecked(True)
            self.statusBar().showMessage(f"当前工具: {self._mode_actions[mode].text()}", 3000)
        if self._measurement_tool_strip is not None:
            self._measurement_tool_strip.setActiveMode(mode)
        self._sync_magic_tool_button()
        self._sync_overlay_tool_button()
        self._update_magic_segment_controls()

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
        return self.palette().color(QPalette.ColorRole.Window).lightnessF() < 0.5

    def _status_color(self, kind: str) -> str:
        if kind == "danger":
            return "#FF7B72" if self._is_dark_palette() else "#C62828"
        if kind == "info":
            return "#79C0FF" if self._is_dark_palette() else "#1565C0"
        if kind == "muted":
            return "#C8D3DD" if self._is_dark_palette() else "#4E5969"
        return self.palette().color(QPalette.ColorRole.WindowText).name()

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
        self._update_action_states()

    def _refresh_capture_devices(self) -> None:
        self._capture_devices = self._capture_manager.refresh_devices()
        selected = self._selected_capture_device()
        if selected is not None and not self._app_settings.selected_capture_device_id:
            self._app_settings.selected_capture_device_id = selected.id
        self._update_capture_device_ui()
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
        self._sync_live_preview_action()
        self._update_ui_for_current_document()

    def _on_live_preview_frame_ready(self, image: object) -> None:
        if not self._preview_active or self._is_native_preview():
            return
        if not isinstance(image, QImage) or image.isNull() or self._preview_canvas is None:
            return
        if (
            self._preview_document is None
            or self._preview_document.image_size != (image.width(), image.height())
        ):
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
        self._update_action_states()

    def _clear_preview_surface_state(self) -> None:
        self._preview_document = None
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

    def _ensure_project_group_template(self, *, label: str, color: str) -> bool:
        token = normalize_group_label(label)
        if not token or self._project_group_template_for_label(token) is not None:
            return False
        self.project.project_group_templates.append(
            ProjectGroupTemplate(label=token, color=color),
        )
        return True

    def _apply_project_group_templates_to_document(self, document: ImageDocument) -> bool:
        changed = False
        for template in self.project.project_group_templates:
            token = normalize_group_label(template.label)
            if not token or document.is_project_group_label_suppressed(token):
                continue
            _group, ensured_changed = self._ensure_document_named_group(
                document,
                label=token,
                color=template.color,
                activate=False,
            )
            changed = ensured_changed or changed
        if document.active_group_id is None and document.can_delete_uncategorized_entry():
            changed = document.hide_uncategorized_entry() or changed
        return changed

    def _sync_project_group_templates(self, *, label: str) -> bool:
        any_changed = False
        for document in self.project.documents:
            before = document.snapshot_state()
            changed = self._apply_project_group_templates_to_document(document)
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
    ) -> tuple[object | None, bool]:
        token = normalize_group_label(label)
        if not token:
            return None, False
        changed = False
        matches = document.groups_by_label(token)
        if matches:
            canonical = matches[0]
            for duplicate in matches[1:]:
                if document.merge_group_into(duplicate.id, canonical.id):
                    changed = True
            if activate and document.active_group_id != canonical.id:
                document.set_active_group(canonical.id)
                changed = True
        else:
            active_group_id = document.active_group_id
            canonical = document.create_group(color=color, label=token)
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

    def _area_inference_global_group_labels(self, model_name: str) -> list[str]:
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
            self.project.project_group_templates.append(
                ProjectGroupTemplate(
                    label=token,
                    color=self._area_inference_group_color_for_label(token),
                )
            )
            ordered_labels.append(token)
            seen_labels.add(token)
        return ordered_labels

    def _normalize_document_groups_for_area_inference(
        self,
        document: ImageDocument,
        *,
        global_group_labels: list[str],
        recognized_labels: set[str],
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
                color=self._area_inference_group_color_for_label(token),
                activate=False,
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
        canvas.lineCommitted.connect(self._on_canvas_line_committed)
        canvas.measurementSelected.connect(self._on_canvas_measurement_selected)
        canvas.measurementEdited.connect(self._on_canvas_measurement_edited)
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

    def save_project(self, path: str | None = None) -> bool:
        if not self.project.documents:
            QMessageBox.information(self, "保存项目", "请先打开图片。")
            return False
        target_path = Path(path) if path else self._project_path
        if target_path is None:
            selected_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存项目",
                "fiber_measurement.fdmproj",
                self.PROJECT_FILTER,
            )
            if not selected_path:
                return False
            target_path = Path(selected_path)
        self.project.version = __version__
        if not self._persist_project_assets(target_path):
            return False
        ProjectIO.save(self.project, target_path)
        self._project_path = target_path
        for document in self.project.documents:
            document.mark_session_saved()
            document.mark_calibration_saved()
        self._mark_project_saved()
        self._update_ui_for_current_document()
        self.statusBar().showMessage(f"项目已保存: {target_path}", 5000)
        return True

    def load_project(self) -> None:
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
        message = f"项目已加载: {path}"
        if imported_count:
            message += f"；已导入 {imported_count} 个旧版标定预设"
        self.statusBar().showMessage(message, 5000)

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
        output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not output_dir:
            return
        target_documents = self.project.documents if selection.scope == ExportScope.ALL_OPEN else ([self.current_document()] if self.current_document() else [])
        outputs = self.export_service.export_project(
            self.project,
            output_dir,
            selection=selection,
            documents=[document for document in target_documents if document is not None],
            overlay_renderer=self._render_overlay_image,
        )
        if not outputs:
            QMessageBox.information(self, "导出结果", "没有生成任何文件。")
            return
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
        QMessageBox.information(self, "导出完成", f"结果已导出到:\n{output_dir}\n\n" + "\n".join(summary_lines))

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
        self._save_app_settings(context="设置")
        self._refresh_preset_combo()
        self._refresh_canvases_for_settings()

        document = self.current_document()
        if document is not None:
            group_colors = dialog.group_colors()
            if group_colors:
                def mutate_group_colors() -> None:
                    for group in document.sorted_groups():
                        if group.id in group_colors:
                            group.color = group_colors[group.id]

                self._apply_document_change(document, "更新类别颜色", mutate_group_colors)

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
        dialog = FiberGroupDialog(self, title="新增类别")
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        label, apply_to_project = dialog.values()
        token = normalize_group_label(label)
        if apply_to_project and not token:
            QMessageBox.warning(self, "新增类别", "应用到当前项目全局时，类别名称不能为空。")
            return

        existing_group = document.find_group_by_label(token) if token else None
        color = existing_group.color if existing_group is not None else self._next_group_color(document)
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
            return
        label, ok = QInputDialog.getText(self, "重命名类别", "类别名称（可留空）", text=group.label)
        if not ok:
            return
        target_label = normalize_group_label(label)
        current_label = normalize_group_label(group.label)
        if target_label == current_label:
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
            if changed:
                self.statusBar().showMessage("类别已合并", 3000)
            return

        def mutate_rename() -> None:
            target = document.get_group(group.id)
            if target is None:
                return
            original_label = normalize_group_label(target.label)
            target.label = target_label
            if original_label and original_label != target_label and self._project_group_template_for_label(original_label) is not None:
                document.suppress_project_group_label(original_label)
            if self._project_group_template_for_label(target_label) is not None:
                document.unsuppress_project_group_label(target_label)

        changed = self._apply_document_change(document, "重命名类别", mutate_rename)
        if changed:
            self.statusBar().showMessage("类别已重命名", 3000)

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
    ) -> None:
        if not instances:
            def clear_mutate() -> None:
                document.remove_auto_area_measurements()
                document.select_measurement(None)

            self._apply_document_change(document, "清除自动面积识别结果", clear_mutate)
            return

        if global_group_labels is None:
            resolved_global_group_labels = self._area_inference_global_group_labels(model_name)
        elif global_group_labels:
            resolved_global_group_labels = list(global_group_labels)
        else:
            resolved_global_group_labels = self._area_inference_global_group_labels(model_name)
            global_group_labels.extend(resolved_global_group_labels)

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
            )
            for instance in instances:
                class_name = str(instance.class_name).strip() or UNCATEGORIZED_LABEL
                group = document.ensure_group_for_label(
                    class_name,
                    color=self._area_inference_group_color_for_label(class_name),
                )
                measurement = Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id,
                    mode="auto_instance",
                    measurement_kind="area",
                    polygon_px=list(instance.polygon_px),
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
        if not positive_points:
            if is_fiber_quick_tool_mode(tool_mode):
                canvas.fail_fiber_quick_result(request_id)
            else:
                canvas.apply_magic_segment_result(request_id, None)
            self._update_magic_segment_controls()
            return
        if is_fiber_quick_tool_mode(tool_mode):
            pending_crop_box = initial_interactive_segmentation_crop_box(
                image_size=(image.height(), image.width()),
                positive_points=positive_points,
                negative_points=negative_points,
                tool_mode=tool_mode,
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
            )
        )
        self._update_magic_segment_controls()

    def _on_canvas_magic_segment_session_changed(self, document_id: str) -> None:
        current_document = self.current_document()
        if current_document is not None and current_document.id == document_id:
            self._update_magic_segment_controls()

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
            if apply_result is not None and apply_result.get("has_preview"):
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
        self._clear_prompt_segmentation_cache()
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
                    confidence=1.0,
                    status="manual" if mode != "auto_instance" else "auto_instance",
                )
                if mode == "magic_segment":
                    ensure_measurement_display_geometry(measurement)
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

    def _format_measurement_kind(self, measurement: Measurement) -> str:
        return "面积" if measurement.measurement_kind == "area" else "线段"

    def _format_measurement_mode(self, mode: str) -> str:
        return {
            "manual": "手动线段",
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

    def _update_action_states(self) -> None:
        document = self.current_document()
        history = document.history if document is not None else None
        has_document = document is not None
        preview_active = self._preview_active
        has_selected_object = bool(
            has_document
            and self._tool_mode != "calibration"
            and (
                document.view_state.selected_measurement_id is not None
                or document.selected_overlay_id is not None
            )
        )
        has_deletable_group_target = bool(
            document and (
                document.get_group(document.active_group_id) is not None
                or document.should_show_uncategorized_entry()
            )
        )
        has_named_active_group = bool(document and document.get_group(document.active_group_id) is not None)
        self.close_current_action.setEnabled(has_document)
        self.close_all_action.setEnabled(bool(self.project.documents))
        self.delete_measurement_action.setEnabled(has_selected_object and not preview_active)
        self.delete_measurement_button.setEnabled(has_selected_object and not preview_active)
        self.add_group_action.setEnabled(has_document and not preview_active)
        self.rename_group_action.setEnabled(has_named_active_group and not preview_active)
        self.delete_group_action.setEnabled(has_deletable_group_target and not preview_active)
        if self._add_group_button is not None:
            self._add_group_button.setEnabled(has_document and not preview_active)
        if self._rename_group_button is not None:
            self._rename_group_button.setEnabled(has_named_active_group and not preview_active)
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
            self._area_auto_button.setEnabled(has_document and bool(self._app_settings.area_model_mappings) and not preview_active)
        self.undo_action.setEnabled(bool(history and history.can_undo()) and not preview_active)
        self.redo_action.setEnabled(bool(history and history.can_redo()) and not preview_active)
        capture_feature_available = _CAPTURE_IMPORT_ERROR is None
        self.switch_capture_device_action.setEnabled(capture_feature_available)
        self.live_preview_action.setEnabled(capture_feature_available)
        can_optimize_signal = capture_feature_available and self._capture_manager.can_optimize_signal()
        analysis_active = self._preview_analysis_mode != "none"
        self.capture_frame_action.setEnabled(preview_active and self._capture_manager.can_capture_still() and not analysis_active)
        self.optimize_capture_signal_action.setVisible(can_optimize_signal)
        self.optimize_capture_signal_action.setEnabled(can_optimize_signal and not analysis_active)
        for mode, action in self._mode_actions.items():
            action.setEnabled(not preview_active or mode == "select")
        if self._magic_tool_button is not None:
            self._magic_tool_button.setEnabled(not preview_active)
        if self._overlay_tool_button is not None:
            self._overlay_tool_button.setEnabled(not preview_active)
        self._update_magic_segment_controls()
        self._update_preview_analysis_controls()
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
            return "编辑：剔除形状 (T)"
        return "编辑：第一形状 (T)"

    def _update_magic_segment_controls(self) -> None:
        if self._magic_controls_widget is None or self._measurement_tool_strip is None:
            return
        is_visible = is_magic_toolbar_tool_mode(self._tool_mode) and not self._preview_active
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
        if self._magic_operation_button is not None:
            self._magic_operation_button.setVisible(standard_mode)
            self._magic_operation_button.setText(self._magic_operation_button_text(operation_mode))
            self._magic_operation_button.setEnabled(has_document and standard_mode and not busy)
        if self._magic_complete_button is not None:
            self._magic_complete_button.setText(
                "完成 (Enter / F)"
                if standard_mode
                else ("确认线段 (Enter / F)" if fiber_quick_mode else "加入当前类别 (Enter / F)")
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
            dialog.close_silently()
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
        if request_id != self._preview_analysis_request_id:
            return
        self._preview_analysis_request_pending = False
        if self._preview_analysis_mode == "none" or self._preview_analysis_worker is None or self._preview_analysis_finalizing:
            return
        if isinstance(image, QImage) and not image.isNull():
            self._preview_analysis_worker.frameSubmitted.emit(image.copy())

    def _on_preview_analysis_frame_failed(self, request_id: int, message: str) -> None:
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
        opened_holes = int(commit_result.get("opened_holes", 0) or 0)
        if opened_holes > 0:
            messages.append(f"结果中检测到闭环区域，已自动开缝 {opened_holes} 处。")
        if bool(commit_result.get("bridge_fallback", False)):
            messages.append("部分闭环桥接已回退到 1px 保底开缝。")
        if bool(commit_result.get("discarded_fragments", False)):
            messages.append("结果中出现多个碎片，已仅保留最大连通区域。")
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
            self.statusBar().showMessage("已确认当前分割，直径线计算完成后将自动写入。", 3000)
        else:
            self.statusBar().showMessage("当前没有可确认的快速测径结果。", 3000)
        self._update_magic_segment_controls()
        self._focus_current_canvas()
        return committed or pending

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
        image.save(str(output_path))

    def _color_icon(self, color_value: str, *, size: int = 12) -> QIcon:
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(color_value))
        return QIcon(pixmap)

    def _contrast_color(self, color_value: str) -> str:
        color = QColor(color_value)
        luminance = (0.299 * color.red()) + (0.587 * color.green()) + (0.114 * color.blue())
        return "#111111" if luminance > 186 else "#FFFFFF"

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            canvas = self.current_canvas()
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
        if is_magic_toolbar_tool_mode(self._tool_mode) and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            if is_magic_segment_tool_mode(self._tool_mode):
                if event.key() == Qt.Key.Key_R:
                    self._cycle_magic_segment_prompt_type()
                    event.accept()
                    return
                if event.key() == Qt.Key.Key_T:
                    self._cycle_magic_segment_operation_mode()
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
        super().keyReleaseEvent(event)

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
        if not self._confirm_close_documents(self.project.documents):
            event.ignore()
            return
        self._persist_window_geometry()
        self.stop_live_preview()
        self._clear_prompt_segmentation_cache()
        self._shutdown_background_threads()
        event.accept()
