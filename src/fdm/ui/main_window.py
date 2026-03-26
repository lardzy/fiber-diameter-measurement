from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread
from PySide6.QtGui import QAction, QActionGroup, QColor, QCloseEvent, QIcon, QImage, QImageReader, QPainter, QPalette, QPixmap
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
from fdm.geometry import Line, line_length
from fdm.models import (
    Calibration,
    CalibrationPreset,
    ImageDocument,
    Measurement,
    ProjectGroupTemplate,
    ProjectState,
    TextAnnotation,
    UNCATEGORIZED_LABEL,
    normalize_group_label,
    new_id,
    project_assets_root,
    project_capture_root,
)
from fdm.project_io import ProjectIO
from fdm.settings import AppSettings, AppSettingsIO, OpenImageViewMode, ScaleOverlayPlacementMode
from fdm.services.area_inference import AreaInferenceService
from fdm.services.export_service import ExportImageRenderMode, ExportScope, ExportSelection, ExportService
from fdm.services.prompt_segmentation import PromptSegmentationService
from fdm.services.sidecar_io import CalibrationSidecarIO
from fdm.ui.canvas import DocumentCanvas
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
from fdm.ui.icons import themed_icon
from fdm.ui.image_loader import ImageBatchLoaderWorker, ImageLoadRequest
from fdm.ui.microview_preview_host import MicroviewPreviewHost
from fdm.ui.prompt_segmentation_worker import PromptSegmentationRequest, PromptSegmentationWorker
from fdm.ui.rendering import draw_measurements, draw_scale_overlay, draw_text_annotations, overlay_metrics
from fdm.ui.widgets import MeasurementGroupComboBox

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
    completed_count: int = 0
    failed_count: int = 0
    cancelled: bool = False
    failures: list[str] | None = None


@dataclass(slots=True)
class PresetImportPlanEntry:
    preset: CalibrationPreset
    action: str
    final_name: str


class MainWindow(QMainWindow):
    IMAGE_FILTER = "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    PROJECT_FILTER = "Fiber 项目 (*.fdmproj)"
    SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
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
        self.setWindowTitle("纤维直径测量")
        self.resize(1520, 940)

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
        self._group_list_rebuilding = False
        self._table_rebuilding = False
        self._file_toolbar: QToolBar | None = None
        self._measure_toolbar: QToolBar | None = None
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
        self._show_area_fill = True
        self._area_auto_button: QPushButton | None = None
        self._magic_controls_widget: QWidget | None = None
        self._magic_controls_action: QAction | None = None
        self._magic_prompt_label: QLabel | None = None
        self._magic_toggle_button: QToolButton | None = None
        self._magic_complete_button: QToolButton | None = None
        self._magic_cancel_button: QToolButton | None = None
        self._add_preset_button: QPushButton | None = None
        self._edit_preset_button: QPushButton | None = None
        self._delete_preset_button: QPushButton | None = None
        self._import_cu_preset_button: QPushButton | None = None
        self._apply_preset_button: QPushButton | None = None
        self._center_stack: QStackedWidget | None = None
        self._preview_page: QWidget | None = None
        self._preview_display_stack: QStackedWidget | None = None
        self._preview_canvas: DocumentCanvas | None = None
        self._microview_preview_host: MicroviewPreviewHost | None = None
        self._microview_preview_scroll: QScrollArea | None = None
        self._preview_status_label: QLabel | None = None
        self._image_resolution_label: QLabel | None = None
        self._preview_notice_label: QLabel | None = None
        self._preview_active = False
        self._preview_document: ImageDocument | None = None
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

        self._build_ui()
        self._capture_manager.devicesChanged.connect(self._on_capture_devices_changed)
        self._capture_manager.previewStateChanged.connect(self._on_live_preview_state_changed)
        self._capture_manager.frameReady.connect(self._on_live_preview_frame_ready)
        self._capture_manager.errorOccurred.connect(self._on_capture_error)
        self._capture_devices = self._capture_manager.devices()
        self._refresh_preset_combo()
        self._update_capture_device_ui()
        self._update_ui_for_current_document()
        self._mark_project_saved()

    def _build_ui(self) -> None:
        self.setStatusBar(QStatusBar())
        self._create_actions()
        self._build_menus()
        self._build_toolbar()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 930, 360])
        self.setCentralWidget(splitter)

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
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                ),
            )
        )
        self.export_actions.append(
            self._make_export_action(
                "导出比例尺图",
                ExportSelection(
                    include_scale_overlay=True,
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                ),
            )
        )
        self.export_actions.append(
            self._make_export_action(
                "导出测量 + 比例尺叠加图",
                ExportSelection(
                    include_combined_overlay=True,
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
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
                    render_mode=ExportImageRenderMode.SCREEN_SCALE_FULL_IMAGE,
                ),
            )
        )

        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        self._mode_actions: dict[str, QAction] = {}
        for mode, label in [
            ("select", "浏览"),
            ("manual", "手动测量"),
            ("polygon_area", "多边形面积"),
            ("freehand_area", "自由形状面积"),
            ("magic_segment", "魔棒分割"),
            ("calibration", "比例尺标定"),
            ("text", "文字"),
        ]:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, value=mode: self.set_tool_mode(value))
            self._mode_actions[mode] = action
            mode_group.addAction(action)
        self._mode_actions["select"].setChecked(True)
        self._mode_actions["select"].setIcon(themed_icon("select", color="#D4D8DD"))
        self._mode_actions["manual"].setIcon(themed_icon("manual", color="#F4D35E"))
        self._mode_actions["polygon_area"].setIcon(themed_icon("polygon_area", color="#7BD389"))
        self._mode_actions["freehand_area"].setIcon(themed_icon("freehand_area", color="#9C89B8"))
        self._mode_actions["magic_segment"].setIcon(themed_icon("magic_segment", color="#D96C75"))
        self._mode_actions["calibration"].setIcon(themed_icon("calibration", color="#FF7F50"))
        self._mode_actions["text"].setIcon(themed_icon("rename", color="#9C89B8"))

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

        self.addToolBarBreak()
        measure_toolbar = QToolBar("测量工具")
        measure_toolbar.setMovable(False)
        measure_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        measure_toolbar.setIconSize(QSize(16, 16))
        measure_toolbar.setStyleSheet(
            """
            QToolBar {
                spacing: 8px;
                padding: 4px 0;
            }
            QToolButton {
                min-height: 38px;
                padding: 6px 14px;
                border-radius: 8px;
                font-weight: 600;
            }
            QToolButton:checked {
                background: #12343B;
                color: #F7F4EA;
                border: 1px solid #2A9D8F;
            }
            """
        )
        self.addToolBar(measure_toolbar)
        self._measure_toolbar = measure_toolbar
        measure_toolbar.addAction(self._mode_actions["select"])
        measure_toolbar.addAction(self._mode_actions["manual"])
        measure_toolbar.addAction(self._mode_actions["polygon_area"])
        measure_toolbar.addAction(self._mode_actions["freehand_area"])
        measure_toolbar.addAction(self._mode_actions["magic_segment"])
        measure_toolbar.addAction(self._mode_actions["calibration"])
        measure_toolbar.addAction(self._mode_actions["text"])
        measure_toolbar.addSeparator()
        self._magic_controls_widget = self._build_magic_segment_controls()
        self._magic_controls_action = measure_toolbar.addWidget(self._magic_controls_widget)
        self._magic_controls_widget.setVisible(False)
        self._magic_controls_action.setVisible(False)

    def _build_magic_segment_controls(self) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(6)

        self._magic_prompt_label = QLabel("当前提示：正采样点")
        self._magic_prompt_label.setStyleSheet(
            "padding: 6px 10px; border-radius: 8px; background: #F6F1E8; color: #182430; font-weight: 600;"
        )
        layout.addWidget(self._magic_prompt_label)

        self._magic_toggle_button = QToolButton(container)
        self._magic_toggle_button.setText("切换正负 (R)")
        self._magic_toggle_button.clicked.connect(self._cycle_magic_segment_prompt_type)
        layout.addWidget(self._magic_toggle_button)

        self._magic_complete_button = QToolButton(container)
        self._magic_complete_button.setText("完成 (Enter / F)")
        self._magic_complete_button.clicked.connect(self._commit_magic_segment_preview)
        layout.addWidget(self._magic_complete_button)

        self._magic_cancel_button = QToolButton(container)
        self._magic_cancel_button.setText("放弃 (Esc)")
        self._magic_cancel_button.clicked.connect(self._cancel_magic_segment_session)
        layout.addWidget(self._magic_cancel_button)

        return container

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        container.setMinimumWidth(180)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(QLabel("已打开图片"))

        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_list_changed)
        layout.addWidget(self.image_list)
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
        calibration_layout.addWidget(self.calibration_label)
        self.preset_combo = QComboBox()
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

        group_box = QGroupBox("纤维类别")
        group_layout = QVBoxLayout(group_box)
        self.group_list = QListWidget()
        self.group_list.setViewMode(QListView.ViewMode.ListMode)
        self.group_list.setFlow(QListView.Flow.LeftToRight)
        self.group_list.setWrapping(True)
        self.group_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.group_list.setMovement(QListView.Movement.Static)
        self.group_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.group_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.group_list.setIconSize(QSize(14, 14))
        self.group_list.setSpacing(6)
        self.group_list.setMaximumHeight(140)
        self.group_list.setStyleSheet(
            """
            QListWidget {
                background: transparent;
                border: none;
            }
            QListWidget::item {
                background: #F6F1E8;
                color: #182430;
                border: 1px solid #D7CEC0;
                border-radius: 10px;
                padding: 6px 10px;
            }
            QListWidget::item:selected {
                background: #FFFDF8;
                color: #182430;
                border: 2px solid #12343B;
                outline: 0;
            }
            """
        )
        self.group_list.itemSelectionChanged.connect(self._on_group_selection_changed)
        group_layout.addWidget(self.group_list)
        group_button_row = QHBoxLayout()
        add_group_button = QPushButton("新增类别")
        add_group_button.setIcon(themed_icon("add", color="#7BD389"))
        add_group_button.clicked.connect(self.add_fiber_group)
        rename_group_button = QPushButton("重命名")
        rename_group_button.setIcon(themed_icon("rename", color="#D7E3FC"))
        rename_group_button.clicked.connect(self.rename_active_group)
        self.delete_group_button = QPushButton("删除")
        self.delete_group_button.setIcon(themed_icon("delete", color="#F28482"))
        self.delete_group_button.clicked.connect(self.delete_active_group)
        group_button_row.addWidget(add_group_button)
        group_button_row.addWidget(rename_group_button)
        group_button_row.addWidget(self.delete_group_button)
        group_layout.addLayout(group_button_row)
        top_layout.addWidget(group_box)

        measurement_box = QGroupBox("测量记录")
        measurement_layout = QVBoxLayout(measurement_box)
        self.measurement_table = QTableWidget(0, 8)
        self.measurement_table.setHorizontalHeaderLabels(["种类", "类型", "结果", "单位", "模式", "置信度", "状态", "ID"])
        header = self.measurement_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.measurement_table.setColumnWidth(self.TABLE_COL_GROUP, 180)
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
        right_splitter.setSizes([430, 360])
        layout.addWidget(right_splitter)

        return container

    def _make_export_action(self, label: str, selection: ExportSelection) -> QAction:
        action = QAction(label, self)
        action.triggered.connect(lambda checked=False, preset=selection: self.export_results(preset))
        return action

    def set_tool_mode(self, mode: str) -> None:
        if mode == "snap" or mode not in self._mode_actions:
            mode = "select"
        if mode != "select":
            self._last_non_select_tool = mode
        self._tool_mode = mode
        for canvas in self._canvases.values():
            canvas.set_tool_mode(mode)
        if mode in self._mode_actions:
            self._mode_actions[mode].setChecked(True)
            self.statusBar().showMessage(f"当前工具: {self._mode_actions[mode].text()}", 3000)
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

    def _set_calibration_label(self, text: str, *, status: str) -> None:
        color_key = {
            "uncalibrated": "danger",
            "calibrated": "info",
            "preview": "muted",
        }.get(status, "default")
        self.calibration_label.setText(text)
        self.calibration_label.setStyleSheet(f"color: {self._status_color(color_key)};")

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
        preview_target = self._current_preview_target()
        if self._center_stack is not None and self._preview_page is not None:
            self._center_stack.setCurrentWidget(self._preview_page)
        self._apply_preview_surface(preview_kind)
        if preview_kind == "native_embed" and self._microview_preview_host is not None:
            self._microview_preview_host.ensure_native_handle()
            QApplication.processEvents()
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
        if not self._capture_manager.is_preview_active():
            self._preview_active = False
            self._sync_live_preview_action()
            return
        self._capture_manager.stop_preview()
        self.statusBar().showMessage("实时预览已停止", 3000)

    def _on_live_preview_state_changed(self, active: bool) -> None:
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
            self._preview_document = None
            self._apply_preview_surface("frame_stream")
            if self._preview_status_label is not None:
                self._preview_status_label.setText("请选择采集设备并开始实时预览")
        self._sync_live_preview_action()
        self._update_ui_for_current_document()

    def _on_live_preview_frame_ready(self, image: object) -> None:
        if self._is_native_preview():
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
        if was_preview_active:
            self.stop_live_preview()
        document = ImageDocument(
            id=new_id("image"),
            path=self._next_project_capture_relative_path(),
            image_size=(frame.width(), frame.height()),
            source_type="project_asset",
        )
        document.initialize_runtime_state()
        if self.project.project_default_calibration is not None:
            self._set_document_project_default_calibration(document)
        document.mark_session_saved()
        document.mark_calibration_saved()
        self._mount_document(
            document,
            frame,
            tooltip=self._document_tooltip(document),
        )
        self.statusBar().showMessage("已采集当前画面到项目内存", 4000)

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
    ) -> None:
        self._area_infer_state = AreaInferenceBatchState(
            total=len(requests),
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
            self._apply_area_inference_result(document, instances)
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
        canvas.set_tool_mode(self._tool_mode)
        canvas.set_show_area_fill(self._show_area_fill)
        canvas.lineCommitted.connect(self._on_canvas_line_committed)
        canvas.measurementSelected.connect(self._on_canvas_measurement_selected)
        canvas.measurementEdited.connect(self._on_canvas_measurement_edited)
        canvas.textPlacementRequested.connect(self._on_canvas_text_placement_requested)
        canvas.textSelected.connect(self._on_canvas_text_selected)
        canvas.textMoved.connect(self._on_canvas_text_moved)
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
        if document.selected_text_id is not None:
            text_id = document.selected_text_id

            def mutate_text() -> None:
                document.remove_text_annotation(text_id)

            self._apply_document_change(document, "删除文字", mutate_text)
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
        self._start_area_inference_batch(requests)

    def _apply_area_inference_result(self, document: ImageDocument, instances) -> None:
        if not instances:
            def clear_mutate() -> None:
                document.remove_auto_area_measurements()
                document.select_measurement(None)

            self._apply_document_change(document, "清除自动面积识别结果", clear_mutate)
            return

        def mutate() -> None:
            document.remove_auto_area_measurements()
            for instance in instances:
                class_name = str(instance.class_name).strip() or UNCATEGORIZED_LABEL
                group = document.ensure_group_for_label(
                    class_name,
                    color=self._color_palette[(document.next_group_number() - 1) % len(self._color_palette)],
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
        positive_points = list(payload.get("positive_points", []))
        negative_points = list(payload.get("negative_points", []))
        if not positive_points:
            canvas.apply_magic_segment_result(request_id, [])
            self._update_magic_segment_controls()
            return
        if image is None or image.isNull():
            canvas.fail_magic_segment_result(request_id)
            self._update_magic_segment_controls()
            QMessageBox.warning(self, "魔棒分割", "当前图片还未完成加载，暂时无法进行魔棒分割。")
            return
        if not PromptSegmentationService.models_ready():
            canvas.fail_magic_segment_result(request_id)
            self._update_magic_segment_controls()
            QMessageBox.warning(
                self,
                "魔棒分割",
                "未找到 EdgeSAM 模型文件，请确认 runtime/segment-anything/edge_sam 中存在 encoder/decoder ONNX。",
            )
            return
        self._ensure_prompt_segmentation_worker()
        if self._prompt_seg_worker is None:
            canvas.fail_magic_segment_result(request_id)
            self._update_magic_segment_controls()
            return
        self._prompt_seg_worker.requested.emit(
            PromptSegmentationRequest(
                document_id=document_id,
                image=image,
                cache_key=f"{document_id}:{int(image.cacheKey())}",
                request_id=request_id,
                positive_points=positive_points,
                negative_points=negative_points,
            )
        )
        self._update_magic_segment_controls()

    def _on_canvas_magic_segment_session_changed(self, document_id: str) -> None:
        current_document = self.current_document()
        if current_document is not None and current_document.id == document_id:
            self._update_magic_segment_controls()

    def _on_prompt_segmentation_succeeded(self, document_id: str, request_id: int, polygon: object) -> None:
        canvas = self._canvases.get(document_id)
        if canvas is None:
            return
        canvas.apply_magic_segment_result(request_id, list(polygon) if isinstance(polygon, list) else [])
        self._update_magic_segment_controls()

    def _on_prompt_segmentation_failed(self, document_id: str, request_id: int, reason: str) -> None:
        canvas = self._canvases.get(document_id)
        if canvas is None:
            return
        canvas.fail_magic_segment_result(request_id)
        self.statusBar().showMessage(f"魔棒分割失败: {reason}", 5000)
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
            canvas.set_tool_mode(self._tool_mode)
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

        def mutate() -> None:
            if isinstance(payload, dict) and payload.get("measurement_kind") == "area":
                measurement = Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id if group else None,
                    mode=mode,
                    measurement_kind="area",
                    polygon_px=list(payload.get("polygon_px", [])),
                    confidence=1.0,
                    status="manual" if mode != "auto_instance" else "auto_instance",
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
            else:
                return
            document.add_measurement(measurement)
            document.select_text_annotation(None)

        self._apply_document_change(document, "新增测量", mutate)
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
                measurement.measurement_kind = "area"
            elif isinstance(payload, Line):
                measurement.snapped_line_px = payload
            else:
                return
            measurement.status = "edited"
            measurement.recalculate(document.calibration)
            document.select_measurement(measurement.id)

        self._apply_document_change(document, "编辑测量线", mutate)
        self._focus_current_canvas()

    def _on_canvas_text_placement_requested(self, document_id: str, anchor: Point) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return
        content, ok = QInputDialog.getMultiLineText(self, "新增文字", "文字内容")
        if not ok:
            self._focus_current_canvas()
            return
        content = content.strip()
        if not content:
            self._focus_current_canvas()
            return

        def mutate() -> None:
            document.add_text_annotation(
                TextAnnotation(
                    id=new_id("text"),
                    image_id=document.id,
                    content=content,
                    anchor_px=anchor,
                )
            )

        self._apply_document_change(document, "新增文字", mutate)
        self.statusBar().showMessage("已新增文字", 2500)
        self._focus_current_canvas()

    def _on_canvas_text_selected(self, document_id: str, text_id: str | None) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return
        document.select_text_annotation(text_id or None)
        if text_id:
            document.select_measurement(None)
        self._sync_measurement_table_selection(document)
        self._update_action_states()
        self._focus_current_canvas()

    def _on_canvas_text_moved(self, document_id: str, text_id: str, anchor: Point) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return

        def mutate() -> None:
            document.move_text_annotation(text_id, anchor)

        self._apply_document_change(document, "移动文字", mutate)
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
                ungrouped_item = QListWidgetItem(self._group_chip_label(UNCATEGORIZED_LABEL, selected=document.active_group_id is None))
                ungrouped_item.setData(Qt.ItemDataRole.UserRole, None)
                ungrouped_item.setIcon(self._color_icon(self._app_settings.default_measurement_color, size=14))
                ungrouped_item.setSizeHint(QSize(136, 36))
                font = ungrouped_item.font()
                font.setBold(document.active_group_id is None)
                ungrouped_item.setFont(font)
                self.group_list.addItem(ungrouped_item)
                if document.active_group_id is None:
                    ungrouped_item.setSelected(True)
            for group in document.sorted_groups():
                selected = document.active_group_id == group.id
                item = QListWidgetItem(self._group_chip_label(group.display_name(), selected=selected))
                item.setData(Qt.ItemDataRole.UserRole, group.id)
                item.setIcon(self._color_icon(group.color, size=14))
                item.setSizeHint(QSize(132, 36))
                font = item.font()
                font.setBold(selected)
                item.setFont(font)
                self.group_list.addItem(item)
                if selected:
                    item.setSelected(True)
        self._group_list_rebuilding = False

    def _update_ui_for_current_document(self) -> None:
        document = self.current_document()
        self._populate_group_list(document)
        self._update_calibration_panel(document)
        self._populate_measurement_table(document)
        self._update_image_resolution_label(document)
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
            "snap": "半自动吸附",
            "polygon_area": "多边形面积",
            "freehand_area": "自由形状面积",
            "magic_segment": "魔棒分割",
            "auto_instance": "实例分割",
        }.get(mode, mode)

    def _format_measurement_status(self, status: str) -> str:
        return {
            "manual": "手动测量",
            "ready": "已完成",
            "manual_review": "需人工复核",
            "snapped": "吸附成功",
            "edited": "已编辑",
            "line_too_short": "测量线过短",
            "component_not_found": "未找到目标区域",
            "boundary_not_found": "未找到边界",
            "auto_instance": "自动识别",
        }.get(status, status)

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
                or document.selected_text_id is not None
            )
        )
        has_deletable_group_target = bool(
            document and (
                document.get_group(document.active_group_id) is not None
                or document.should_show_uncategorized_entry()
            )
        )
        self.close_current_action.setEnabled(has_document)
        self.close_all_action.setEnabled(bool(self.project.documents))
        self.delete_measurement_action.setEnabled(has_selected_object and not preview_active)
        self.delete_measurement_button.setEnabled(has_selected_object and not preview_active)
        self.add_group_action.setEnabled(has_document and not preview_active)
        self.rename_group_action.setEnabled(has_document and not preview_active and document.get_group(document.active_group_id) is not None if document else False)
        self.delete_group_action.setEnabled(has_deletable_group_target and not preview_active)
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
        self.capture_frame_action.setEnabled(preview_active and self._capture_manager.can_capture_still())
        self.optimize_capture_signal_action.setVisible(can_optimize_signal)
        self.optimize_capture_signal_action.setEnabled(can_optimize_signal)
        for mode, action in self._mode_actions.items():
            action.setEnabled(not preview_active or mode == "select")
        self._update_magic_segment_controls()

    def _magic_prompt_label_text(self, prompt_type: str) -> str:
        return "当前提示：负采样点" if prompt_type == "negative" else "当前提示：正采样点"

    def _update_magic_segment_controls(self) -> None:
        if self._magic_controls_widget is None or self._magic_controls_action is None:
            return
        is_visible = self._tool_mode == "magic_segment" and not self._preview_active
        self._magic_controls_action.setVisible(is_visible)
        self._magic_controls_widget.setVisible(is_visible)
        if not is_visible:
            return
        canvas = self.current_canvas()
        has_document = canvas is not None and canvas.document_id is not None
        prompt_type = canvas.current_magic_segment_prompt_type() if canvas is not None else "positive"
        if self._magic_prompt_label is not None:
            self._magic_prompt_label.setText(self._magic_prompt_label_text(prompt_type))
        if self._magic_toggle_button is not None:
            self._magic_toggle_button.setEnabled(has_document)
        if self._magic_complete_button is not None:
            self._magic_complete_button.setEnabled(bool(canvas and canvas.has_magic_segment_preview()))
        if self._magic_cancel_button is not None:
            self._magic_cancel_button.setEnabled(bool(canvas and canvas.has_magic_segment_session()))

    def _cycle_magic_segment_prompt_type(self) -> None:
        canvas = self.current_canvas()
        if canvas is None or self._tool_mode != "magic_segment":
            return
        prompt_type = canvas.cycle_magic_segment_prompt_type()
        self.statusBar().showMessage(self._magic_prompt_label_text(prompt_type), 2500)
        self._focus_current_canvas()

    def _commit_magic_segment_preview(self) -> bool:
        canvas = self.current_canvas()
        if canvas is None or self._tool_mode != "magic_segment":
            return False
        committed = canvas.commit_magic_segment_preview()
        if committed:
            self.statusBar().showMessage("已创建魔棒分割面积", 2500)
        self._update_magic_segment_controls()
        self._focus_current_canvas()
        return committed

    def _cancel_magic_segment_session(self) -> None:
        canvas = self.current_canvas()
        if canvas is None or self._tool_mode != "magic_segment":
            return
        if canvas.has_magic_segment_session():
            canvas.clear_magic_segment_session()
            self.statusBar().showMessage("已放弃当前魔棒遮罩", 2500)
        self._update_magic_segment_controls()
        self._focus_current_canvas()

    def _clear_magic_segment_sessions(self, *, except_document_id: str | None = None) -> None:
        for document_id, canvas in self._canvases.items():
            if document_id == except_document_id:
                continue
            if canvas.has_magic_segment_session():
                canvas.clear_magic_segment_session()

    def _group_chip_label(self, text: str, *, selected: bool) -> str:
        return f"✓ {text}" if selected else text

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
            draw_text_annotations(
                painter,
                document,
                image_to_output,
                self._app_settings,
                selected_text_id=None,
            )

        if include_scale and document.calibration is not None:
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
                target_output_px=max(80.0, image.width() * 0.18),
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
        if self._tool_mode == "magic_segment" and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            if event.key() == Qt.Key.Key_R:
                self._cycle_magic_segment_prompt_type()
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

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_close_documents(self.project.documents):
            event.ignore()
            return
        self.stop_live_preview()
        if self._prompt_seg_thread is not None:
            self._prompt_seg_thread.quit()
            self._prompt_seg_thread.wait(1500)
        event.accept()
