from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread
from PySide6.QtGui import QAction, QActionGroup, QColor, QCloseEvent, QIcon, QImage, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
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
    QPushButton,
    QSplitter,
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
    ProjectState,
    TextAnnotation,
    UNCATEGORIZED_LABEL,
    new_id,
)
from fdm.project_io import ProjectIO
from fdm.raster import RasterImage
from fdm.settings import AppSettings, AppSettingsIO, OpenImageViewMode, ScaleOverlayPlacementMode
from fdm.services.area_inference import AreaInferenceService
from fdm.services.export_service import ExportImageRenderMode, ExportScope, ExportSelection, ExportService
from fdm.services.model_provider import NullModelProvider, OnnxModelProvider
from fdm.services.sidecar_io import CalibrationSidecarIO
from fdm.services.snap_service import SnapService
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.dialogs import (
    AreaAutoRecognitionDialog,
    CalibrationInputDialog,
    CalibrationPresetDialog,
    ExportOptionsDialog,
    SettingsDialog,
    TaskProgressDialog,
)
from fdm.ui.icons import themed_icon
from fdm.ui.image_loader import ImageBatchLoaderWorker, ImageLoadRequest, qimage_to_raster
from fdm.ui.rendering import draw_measurements, draw_scale_overlay, draw_text_annotations, overlay_metrics
from fdm.ui.widgets import MeasurementGroupComboBox


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
        self._rasters: dict[str, RasterImage] = {}
        self._canvases: dict[str, DocumentCanvas] = {}
        self._tool_mode = "select"
        self._last_non_select_tool: str | None = None
        self._group_list_rebuilding = False
        self._table_rebuilding = False
        self._file_toolbar: QToolBar | None = None
        self._measure_toolbar: QToolBar | None = None
        self._load_thread: QThread | None = None
        self._load_worker: ImageBatchLoaderWorker | None = None
        self._load_progress_dialog: TaskProgressDialog | None = None
        self._load_state: BatchLoadState | None = None
        self._show_area_fill = True
        self._area_auto_button: QPushButton | None = None
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
        self.snap_service = SnapService(model_provider=NullModelProvider())
        self.area_inference_service = AreaInferenceService()

        self._build_ui()
        self._refresh_preset_combo()
        self._update_model_status()
        self._update_ui_for_current_document()

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
            ("snap", "半自动吸附"),
            ("polygon_area", "多边形面积"),
            ("freehand_area", "自由形状面积"),
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
        self._mode_actions["snap"].setIcon(themed_icon("snap", color="#2A9D8F"))
        self._mode_actions["polygon_area"].setIcon(themed_icon("polygon_area", color="#7BD389"))
        self._mode_actions["freehand_area"].setIcon(themed_icon("freehand_area", color="#9C89B8"))
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
        measure_toolbar.addAction(self._mode_actions["snap"])
        measure_toolbar.addAction(self._mode_actions["polygon_area"])
        measure_toolbar.addAction(self._mode_actions["freehand_area"])
        measure_toolbar.addAction(self._mode_actions["calibration"])
        measure_toolbar.addAction(self._mode_actions["text"])

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
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
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tab_widget)
        return container

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)

        model_box = QGroupBox("模型状态")
        model_layout = QVBoxLayout(model_box)
        self.model_status_label = QLabel("未加载")
        model_layout.addWidget(self.model_status_label)
        load_model_button = QPushButton("加载 ONNX 模型")
        load_model_button.setIcon(themed_icon("model", color="#9AD1D4"))
        load_model_button.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_button)
        self._area_auto_button = QPushButton("面积自动识别...")
        self._area_auto_button.setIcon(themed_icon("area_auto", color="#7BD389"))
        self._area_auto_button.clicked.connect(self.run_area_auto_recognition)
        model_layout.addWidget(self._area_auto_button)
        top_layout.addWidget(model_box)

        calibration_box = QGroupBox("标定")
        calibration_layout = QVBoxLayout(calibration_box)
        self.calibration_label = QLabel("当前图片未标定")
        calibration_layout.addWidget(self.calibration_label)
        self.preset_combo = QComboBox()
        calibration_layout.addWidget(self.preset_combo)
        preset_row = QHBoxLayout()
        add_preset_button = QPushButton("新增预设")
        add_preset_button.setIcon(themed_icon("preset_add", color="#7BD389"))
        add_preset_button.clicked.connect(self.add_calibration_preset)
        apply_preset_button = QPushButton("应用预设")
        apply_preset_button.setIcon(themed_icon("preset_apply", color="#D7E3FC"))
        apply_preset_button.clicked.connect(self.apply_selected_preset)
        preset_row.addWidget(add_preset_button)
        preset_row.addWidget(apply_preset_button)
        calibration_layout.addLayout(preset_row)
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
        if mode != "select":
            self._last_non_select_tool = mode
        self._tool_mode = mode
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.set_tool_mode(mode)
        if mode in self._mode_actions:
            self._mode_actions[mode].setChecked(True)
            self.statusBar().showMessage(f"当前工具: {self._mode_actions[mode].text()}", 3000)

    def current_document(self) -> ImageDocument | None:
        index = self.tab_widget.currentIndex()
        if index < 0 or index >= len(self._document_order):
            return None
        return self.project.get_document(self._document_order[index])

    def current_canvas(self) -> DocumentCanvas | None:
        document = self.current_document()
        if document is None:
            return None
        return self._canvases.get(document.id)

    def _apply_open_view_mode(self, canvas: DocumentCanvas | None) -> None:
        if canvas is None:
            return
        mode = self._app_settings.open_image_view_mode
        if mode == OpenImageViewMode.FIT:
            canvas.fit_to_view()
        elif mode == OpenImageViewMode.ACTUAL:
            canvas.actual_size()

    def open_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", self.IMAGE_FILTER)
        if not paths:
            return
        self._open_image_requests(
            [(path, None) for path in paths],
            context_label="打开图片",
        )

    def open_folder(self) -> None:
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
        open_documents = {
            self._normalize_image_path(document.path): document
            for document in self.project.documents
        }
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
        raster = qimage_to_raster(image)
        self._add_loaded_document(request, image, raster)
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
        progress = TaskProgressDialog("准备加载图片...", "取消", 0, len(requests), self)
        progress.setWindowTitle(context_label)
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setValue(0)
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

    def _on_batch_load_loaded(self, request: ImageLoadRequest, image: QImage, raster: RasterImage) -> None:
        state = self._load_state
        if state is not None:
            state.completed_count += 1
            state.loaded_count += 1
        self._add_loaded_document(request, image, raster)
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

    def _add_loaded_document(self, request: ImageLoadRequest, image: QImage, raster: RasterImage) -> None:
        absolute_path = request.path
        target_document = request.document or ImageDocument(
            id=new_id("image"),
            path=absolute_path,
            image_size=(image.width(), image.height()),
        )
        target_document.path = absolute_path
        target_document.image_size = (image.width(), image.height())
        target_document.sidecar_path = target_document.default_sidecar_path()
        target_document.initialize_runtime_state()
        if target_document.calibration is None:
            CalibrationSidecarIO.load_document(target_document)
        else:
            target_document.mark_calibration_saved()
        target_document.mark_session_saved()

        canvas = DocumentCanvas()
        canvas.set_document(target_document, image)
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

        self.project.documents.append(target_document)
        self._document_order.append(target_document.id)
        self._images[target_document.id] = image
        self._rasters[target_document.id] = raster
        self._canvases[target_document.id] = canvas

        tab_index = self.tab_widget.addTab(canvas, Path(absolute_path).name)
        list_item = QListWidgetItem(Path(absolute_path).name)
        list_item.setData(Qt.ItemDataRole.UserRole, target_document.id)
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
        ProjectIO.save(self.project, target_path)
        self._project_path = target_path
        for document in self.project.documents:
            document.mark_session_saved()
            document.mark_calibration_saved()
        self._update_ui_for_current_document()
        self.statusBar().showMessage(f"项目已保存: {target_path}", 5000)
        return True

    def load_project(self) -> None:
        if self._load_thread is not None:
            QMessageBox.information(self, "打开项目", "当前仍有图片在加载，请稍候。")
            return
        if not self._confirm_close_documents(self.project.documents):
            return
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", self.PROJECT_FILTER)
        if not path:
            return
        project = ProjectIO.load(path)
        missing_paths = []
        self._reset_workspace()
        self._project_path = Path(path)
        self.project = ProjectState(version=project.version, documents=[], calibration_presets=project.calibration_presets)
        self.project.metadata = project.metadata
        self._refresh_preset_combo()
        load_items: list[tuple[str, ImageDocument | None]] = []
        for document in project.documents:
            if Path(document.path).exists():
                load_items.append((document.path, document))
            else:
                missing_paths.append(document.path)
        self._open_image_requests(load_items, context_label="打开项目", missing_paths=missing_paths)
        self.statusBar().showMessage(f"项目已加载: {path}", 5000)

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
        previous_settings = self._app_settings
        new_settings = dialog.app_settings()
        self._app_settings = new_settings
        AppSettingsIO.save(new_settings)
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

    def _refresh_canvases_for_settings(self) -> None:
        for canvas in self._canvases.values():
            canvas.set_settings(self._app_settings)
            canvas.set_show_area_fill(self._show_area_fill)
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

    def add_calibration_preset(self) -> None:
        dialog = CalibrationPresetDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        name, pixel_distance, actual_distance, pixels_per_unit, unit = dialog.values()
        if not name:
            QMessageBox.warning(self, "新增预设", "预设名称不能为空。")
            return
        self.project.calibration_presets.append(
            CalibrationPreset(
                name=name,
                pixels_per_unit=pixels_per_unit,
                unit=unit,
                pixel_distance=pixel_distance,
                actual_distance=actual_distance,
                computed_pixels_per_unit=pixels_per_unit,
            )
        )
        self._refresh_preset_combo()
        self.statusBar().showMessage(f"已新增标定预设: {name}", 4000)

    def apply_selected_preset(self) -> None:
        document = self.current_document()
        preset_index = self.preset_combo.currentIndex()
        if document is None or preset_index < 0 or preset_index >= len(self.project.calibration_presets):
            return
        preset = self.project.calibration_presets[preset_index]

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
        label, ok = QInputDialog.getText(self, "新增类别", "类别名称（可留空）")
        if not ok:
            return

        def mutate() -> None:
            group = document.create_group(
                color=self._color_palette[(document.next_group_number() - 1) % len(self._color_palette)],
                label=label.strip(),
            )
            document.set_active_group(group.id)

        self._apply_document_change(document, "新增类别", mutate)
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

        def mutate() -> None:
            target = document.get_group(group.id)
            if target is not None:
                target.label = label.strip()

        self._apply_document_change(document, "重命名类别", mutate)

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
                document.remove_group_to_uncategorized(group.id)

            self._apply_document_change(document, "删除类别", mutate)
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

    def load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX 模型", "", "ONNX 文件 (*.onnx)")
        if not path:
            return
        provider = OnnxModelProvider()
        try:
            provider.load(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "模型加载失败", str(exc))
            return
        self.snap_service.set_model_provider(provider)
        self._update_model_status()
        self.statusBar().showMessage(f"已加载模型: {path}", 5000)

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

        progress = TaskProgressDialog("正在执行面积自动识别...", "取消", 0, len(target_documents), self)
        progress.setWindowTitle("面积自动识别")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setValue(0)
        progress.show()

        completed = 0
        failures: list[str] = []
        for document in target_documents:
            if progress.wasCanceled():
                break
            progress.setLabelText(f"正在识别 ({completed + 1}/{len(target_documents)})\n{Path(document.path).name}")
            progress.setValue(completed)
            QApplication.processEvents()
            try:
                result = self.area_inference_service.infer_image(
                    image_path=document.path,
                    model_name=model_name,
                    model_file=model_file,
                    settings=self._app_settings,
                )
                self._apply_area_inference_result(document, result.instances)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{Path(document.path).name}: {exc}")
            completed += 1

        progress.setValue(len(target_documents))
        progress.close()
        progress.deleteLater()
        if failures:
            QMessageBox.warning(self, "面积自动识别", "以下图片识别失败:\n" + "\n".join(failures[:10]))
        if completed > 0:
            self.statusBar().showMessage(f"面积自动识别已处理 {completed - len(failures)} / {completed} 张图片", 6000)

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
        CalibrationSidecarIO.save_document(document)
        self._update_ui_for_current_document()

    def redo_current_document(self) -> None:
        document = self.current_document()
        if document is None or document.history is None or not document.history.redo(document):
            return
        CalibrationSidecarIO.save_document(document)
        self._update_ui_for_current_document()

    def _confirm_close_documents(self, documents: list[ImageDocument]) -> bool:
        dirty_documents = [document for document in documents if document.dirty_flags.session_dirty]
        if not dirty_documents:
            return True
        if len(dirty_documents) == 1 and len(documents) == 1:
            message = f"{Path(dirty_documents[0].path).name} 有未保存的会话改动。"
        else:
            message = f"共有 {len(dirty_documents)} 张图片存在未保存的会话改动。"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("未保存的改动")
        box.setText(message)
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
        self.project = ProjectState.empty()
        self._project_path = None
        self._document_order.clear()
        self._images.clear()
        self._rasters.clear()
        self._canvases.clear()
        self.image_list.clear()
        self.tab_widget.clear()

    def _remove_document(self, document_id: str) -> None:
        if document_id not in self._document_order:
            return
        index = self._document_order.index(document_id)
        self._document_order.pop(index)
        self.project.documents = [document for document in self.project.documents if document.id != document_id]
        self._images.pop(document_id, None)
        self._rasters.pop(document_id, None)
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
    ) -> None:
        before = document.snapshot_state()
        mutator()
        document.rebuild_group_memberships()
        document.refresh_dirty_flags()
        after = document.snapshot_state()
        if document.history is not None:
            document.history.push(label, before, after)
        if sync_sidecar:
            CalibrationSidecarIO.save_document(document)
        self._update_ui_for_current_document()

    def _on_tab_changed(self, index: int) -> None:
        if index < 0:
            return
        self.image_list.setCurrentRow(index)
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
            elif isinstance(payload, Line):
                snap_result = self.snap_service.snap_measurement(self._rasters[document.id], payload)
                measurement = Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id if group else None,
                    mode="snap",
                    line_px=payload,
                    snapped_line_px=snap_result.snapped_line or payload,
                    confidence=snap_result.confidence,
                    status=snap_result.status if snap_result.snapped_line is not None else "manual_review",
                    debug_payload=snap_result.debug_payload,
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
        actual_length, unit = dialog.values()
        pixels_per_unit = line_length(line) / actual_length

        def mutate() -> None:
            document.calibration = Calibration(
                mode="image_scale",
                pixels_per_unit=pixels_per_unit,
                unit=unit,
                source_label=f"图内标定 {actual_length:g}{unit}",
            )
            document.metadata["calibration_line"] = line.to_dict()
            document.recalculate_measurements()

        self._apply_document_change(document, "图内标定", mutate, sync_sidecar=True)
        self.statusBar().showMessage("图内标尺标定已更新", 4000)

    def _refresh_preset_combo(self) -> None:
        self.preset_combo.clear()
        for preset in self.project.calibration_presets:
            self.preset_combo.addItem(f"{preset.name} ({preset.resolved_pixels_per_unit():g} px/{preset.unit})")

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
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.set_settings(self._app_settings)
            canvas.set_tool_mode(self._tool_mode)
            canvas.set_show_area_fill(self._show_area_fill)
        self._update_action_states()

    def _update_calibration_panel(self, document: ImageDocument | None) -> None:
        if document is None or document.calibration is None:
            self.calibration_label.setText("当前图片未标定")
            return
        calibration = document.calibration
        sidecar_info = f"\n侧车: {Path(document.sidecar_path or document.default_sidecar_path()).name}"
        self.calibration_label.setText(
            f"{calibration.source_label}\n{calibration.pixels_per_unit:.4f} px/{calibration.unit}{sidecar_info}"
        )

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

    def _update_model_status(self) -> None:
        health = self.snap_service.model_provider.healthcheck()
        if health.get("ready"):
            self.model_status_label.setText(f"已加载\n{health.get('model_path', '')}")
        else:
            self.model_status_label.setText(f"未就绪\n{health.get('reason', '') or '将使用传统算法兜底'}")

    def _update_action_states(self) -> None:
        document = self.current_document()
        history = document.history if document is not None else None
        has_document = document is not None
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
        self.delete_measurement_action.setEnabled(has_selected_object)
        self.delete_measurement_button.setEnabled(has_selected_object)
        self.add_group_action.setEnabled(has_document)
        self.rename_group_action.setEnabled(has_document and document.get_group(document.active_group_id) is not None if document else False)
        self.delete_group_action.setEnabled(has_deletable_group_target)
        self.delete_group_button.setEnabled(has_deletable_group_target)
        if self._area_auto_button is not None:
            self._area_auto_button.setEnabled(has_document and bool(self._app_settings.area_model_mappings))
        self.undo_action.setEnabled(bool(history and history.can_undo()))
        self.redo_action.setEnabled(bool(history and history.can_redo()))

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
        event.accept()
