from __future__ import annotations

from pathlib import Path
import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QCloseEvent, QFont, QIcon, QImage, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
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
    FiberGroup,
    ImageDocument,
    Measurement,
    ProjectState,
    UNCATEGORIZED_COLOR,
    UNCATEGORIZED_LABEL,
    new_id,
)
from fdm.project_io import ProjectIO
from fdm.raster import RasterImage
from fdm.services.export_service import ExportImageRenderMode, ExportScope, ExportSelection, ExportService
from fdm.services.model_provider import NullModelProvider, OnnxModelProvider
from fdm.services.sidecar_io import CalibrationSidecarIO
from fdm.services.snap_service import SnapService
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.dialogs import CalibrationInputDialog, CalibrationPresetDialog, ExportOptionsDialog


class MainWindow(QMainWindow):
    IMAGE_FILTER = "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    PROJECT_FILTER = "Fiber 项目 (*.fdmproj)"
    SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("纤维直径测量")
        self.resize(1520, 940)

        self.project = ProjectState.empty()
        self._project_path: Path | None = None
        self._document_order: list[str] = []
        self._images: dict[str, QImage] = {}
        self._rasters: dict[str, RasterImage] = {}
        self._canvases: dict[str, DocumentCanvas] = {}
        self._tool_mode = "select"
        self._group_list_rebuilding = False
        self._table_rebuilding = False
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
        self.open_images_action.setShortcut("Ctrl+O")
        self.open_images_action.triggered.connect(self.open_images)

        self.open_folder_action = QAction("打开文件夹", self)
        self.open_folder_action.triggered.connect(self.open_folder)

        self.open_project_action = QAction("打开项目", self)
        self.open_project_action.triggered.connect(self.load_project)

        self.save_project_action = QAction("保存项目", self)
        self.save_project_action.setShortcut("Ctrl+S")
        self.save_project_action.triggered.connect(lambda: self.save_project())

        self.close_current_action = QAction("关闭当前图片", self)
        self.close_current_action.setShortcut("Ctrl+W")
        self.close_current_action.triggered.connect(self.close_current_document)

        self.close_all_action = QAction("关闭所有图片", self)
        self.close_all_action.setShortcut("Ctrl+Shift+W")
        self.close_all_action.triggered.connect(self.close_all_documents)

        self.undo_action = QAction("撤回", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo_current_document)

        self.redo_action = QAction("重做", self)
        self.redo_action.setShortcut("Ctrl+Shift+Z")
        self.redo_action.triggered.connect(self.redo_current_document)

        self.delete_measurement_action = QAction("删除选中测量", self)
        self.delete_measurement_action.setShortcut("Delete")
        self.delete_measurement_action.triggered.connect(self.delete_selected_measurement)

        self.add_group_action = QAction("新增类别", self)
        self.add_group_action.triggered.connect(self.add_fiber_group)

        self.rename_group_action = QAction("重命名当前类别", self)
        self.rename_group_action.triggered.connect(self.rename_active_group)

        self.fit_action = QAction("适应窗口", self)
        self.fit_action.triggered.connect(self.fit_current_image)

        self.actual_size_action = QAction("原始像素", self)
        self.actual_size_action.triggered.connect(self.actual_size_current_image)

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
            ("calibration", "比例尺标定"),
        ]:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, value=mode: self.set_tool_mode(value))
            self._mode_actions[mode] = action
            mode_group.addAction(action)
        self._mode_actions["select"].setChecked(True)

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

        view_menu = self.menuBar().addMenu("视图")
        for action in self._mode_actions.values():
            view_menu.addAction(action)
        view_menu.addSeparator()
        view_menu.addAction(self.fit_action)
        view_menu.addAction(self.actual_size_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(self.open_images_action)
        toolbar.addAction(self.open_folder_action)
        toolbar.addAction(self.open_project_action)
        toolbar.addAction(self.save_project_action)
        toolbar.addSeparator()

        export_button = QToolButton(self)
        export_button.setText("导出")
        export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QMenu(export_button)
        for action in self.export_actions:
            export_menu.addAction(action)
        export_button.setMenu(export_menu)
        toolbar.addWidget(export_button)
        toolbar.addSeparator()

        for action in self._mode_actions.values():
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.fit_action)
        toolbar.addAction(self.actual_size_action)
        toolbar.addSeparator()
        toolbar.addAction(self.close_current_action)

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

        model_box = QGroupBox("模型状态")
        model_layout = QVBoxLayout(model_box)
        self.model_status_label = QLabel("未加载")
        model_layout.addWidget(self.model_status_label)
        load_model_button = QPushButton("加载 ONNX 模型")
        load_model_button.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_button)
        layout.addWidget(model_box)

        calibration_box = QGroupBox("标定")
        calibration_layout = QVBoxLayout(calibration_box)
        self.calibration_label = QLabel("当前图片未标定")
        calibration_layout.addWidget(self.calibration_label)
        self.preset_combo = QComboBox()
        calibration_layout.addWidget(self.preset_combo)
        preset_row = QHBoxLayout()
        add_preset_button = QPushButton("新增预设")
        add_preset_button.clicked.connect(self.add_calibration_preset)
        apply_preset_button = QPushButton("应用预设")
        apply_preset_button.clicked.connect(self.apply_selected_preset)
        preset_row.addWidget(add_preset_button)
        preset_row.addWidget(apply_preset_button)
        calibration_layout.addLayout(preset_row)
        layout.addWidget(calibration_box)

        group_box = QGroupBox("纤维类别")
        group_layout = QVBoxLayout(group_box)
        self.group_list = QListWidget()
        self.group_list.setViewMode(QListView.ViewMode.IconMode)
        self.group_list.setFlow(QListView.Flow.LeftToRight)
        self.group_list.setWrapping(True)
        self.group_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.group_list.setMovement(QListView.Movement.Static)
        self.group_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.group_list.setSpacing(6)
        self.group_list.setMaximumHeight(140)
        self.group_list.setStyleSheet(
            """
            QListWidget {
                background: transparent;
                border: none;
            }
            QListWidget::item {
                border: 2px solid transparent;
                border-radius: 10px;
                padding: 6px 10px;
            }
            QListWidget::item:selected {
                border: 3px solid #F7F4EA;
                outline: 0;
            }
            """
        )
        self.group_list.itemSelectionChanged.connect(self._on_group_selection_changed)
        group_layout.addWidget(self.group_list)
        group_button_row = QHBoxLayout()
        add_group_button = QPushButton("新增类别")
        add_group_button.clicked.connect(self.add_fiber_group)
        rename_group_button = QPushButton("重命名")
        rename_group_button.clicked.connect(self.rename_active_group)
        group_button_row.addWidget(add_group_button)
        group_button_row.addWidget(rename_group_button)
        group_layout.addLayout(group_button_row)
        layout.addWidget(group_box)

        measurement_box = QGroupBox("测量记录")
        measurement_layout = QVBoxLayout(measurement_box)
        self.measurement_table = QTableWidget(0, 7)
        self.measurement_table.setHorizontalHeaderLabels(["ID", "种类", "模式", "结果", "单位", "置信度", "状态"])
        header = self.measurement_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.measurement_table.setColumnWidth(0, 110)
        self.measurement_table.setColumnWidth(1, 180)
        self.measurement_table.setColumnWidth(2, 90)
        self.measurement_table.setColumnWidth(3, 120)
        self.measurement_table.setColumnWidth(4, 70)
        self.measurement_table.setColumnWidth(5, 80)
        self.measurement_table.setColumnWidth(6, 110)
        self.measurement_table.verticalHeader().setVisible(False)
        self.measurement_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.measurement_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.measurement_table.itemSelectionChanged.connect(self._on_measurement_selection_changed)
        measurement_layout.addWidget(self.measurement_table)
        self.delete_measurement_button = QPushButton("删除选中测量")
        self.delete_measurement_button.clicked.connect(self.delete_selected_measurement)
        measurement_layout.addWidget(self.delete_measurement_button)
        layout.addWidget(measurement_box, 1)

        return container

    def _make_export_action(self, label: str, selection: ExportSelection) -> QAction:
        action = QAction(label, self)
        action.triggered.connect(lambda checked=False, preset=selection: self.export_results(preset))
        return action

    def set_tool_mode(self, mode: str) -> None:
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

    def open_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", self.IMAGE_FILTER)
        for path in paths:
            self._open_image(path)

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
        for image_path in image_paths:
            self._open_image(str(image_path))

    def _open_image(self, path: str, *, document: ImageDocument | None = None) -> None:
        absolute_path = str(Path(path).expanduser().resolve())
        for existing_document in self.project.documents:
            if Path(existing_document.path) == Path(absolute_path):
                self._set_current_document(existing_document.id)
                return

        reader = QImageReader(absolute_path)
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            QMessageBox.warning(self, "打开失败", f"无法读取图片:\n{absolute_path}")
            return

        target_document = document or ImageDocument(
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

        raster = self._qimage_to_raster(image)
        canvas = DocumentCanvas()
        canvas.set_document(target_document, image)
        canvas.set_tool_mode(self._tool_mode)
        canvas.lineCommitted.connect(self._on_canvas_line_committed)
        canvas.measurementSelected.connect(self._on_canvas_measurement_selected)
        canvas.measurementEdited.connect(self._on_canvas_measurement_edited)

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
        for document in project.documents:
            if Path(document.path).exists():
                self._open_image(document.path, document=document)
            else:
                missing_paths.append(document.path)
        self.project.metadata = project.metadata
        for document in self.project.documents:
            if document.calibration is None:
                CalibrationSidecarIO.load_document(document)
            else:
                document.mark_calibration_saved()
            document.mark_session_saved()
        self._refresh_preset_combo()
        if missing_paths:
            QMessageBox.warning(self, "部分图片缺失", "以下图片未找到:\n" + "\n".join(missing_paths))
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
        summary_lines = []
        for key, value in outputs.items():
            if isinstance(value, list):
                summary_lines.append(f"{key}: {len(value)} 个文件")
            else:
                summary_lines.append(f"{key}: {value}")
        QMessageBox.information(self, "导出完成", f"结果已导出到:\n{output_dir}\n\n" + "\n".join(summary_lines))

    def fit_current_image(self) -> None:
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.fit_to_view()

    def actual_size_current_image(self) -> None:
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.actual_size()

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

    def delete_selected_measurement(self) -> None:
        document = self.current_document()
        if self._tool_mode == "calibration" or document is None or document.view_state.selected_measurement_id is None:
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
        self._update_ui_for_current_document()

    def _on_image_list_changed(self, row: int) -> None:
        if row >= 0 and row != self.tab_widget.currentIndex():
            self.tab_widget.setCurrentIndex(row)

    def _set_current_document(self, document_id: str) -> None:
        if document_id in self._document_order:
            index = self._document_order.index(document_id)
            self.tab_widget.setCurrentIndex(index)
            self.image_list.setCurrentRow(index)

    def _on_canvas_line_committed(self, document_id: str, mode: str, line: Line) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return
        if mode == "calibration":
            self._apply_calibration_line(document, line)
            return

        group = document.get_group(document.active_group_id)

        def mutate() -> None:
            if mode == "manual":
                measurement = Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id if group else None,
                    mode="manual",
                    line_px=line,
                    confidence=1.0,
                    status="manual",
                )
            else:
                snap_result = self.snap_service.snap_measurement(self._rasters[document.id], line)
                measurement = Measurement(
                    id=new_id("meas"),
                    image_id=document.id,
                    fiber_group_id=group.id if group else None,
                    mode="snap",
                    line_px=line,
                    snapped_line_px=snap_result.snapped_line or line,
                    confidence=snap_result.confidence,
                    status=snap_result.status if snap_result.snapped_line is not None else "manual_review",
                    debug_payload=snap_result.debug_payload,
                )
            document.add_measurement(measurement)

        self._apply_document_change(document, "新增测量", mutate)
        self.statusBar().showMessage("已新增测量", 2500)

    def _on_canvas_measurement_selected(self, document_id: str, measurement_id: str) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return
        document.view_state.selected_measurement_id = measurement_id
        self._sync_measurement_table_selection(document)
        self._update_action_states()

    def _on_canvas_measurement_edited(self, document_id: str, measurement_id: str, line: Line) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return

        def mutate() -> None:
            measurement = document.get_measurement(measurement_id)
            if measurement is None:
                return
            measurement.snapped_line_px = line
            measurement.status = "edited"
            measurement.recalculate(document.calibration)
            document.view_state.selected_measurement_id = measurement.id

        self._apply_document_change(document, "编辑测量线", mutate)

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
            ungrouped_item = QListWidgetItem(self._group_chip_label(UNCATEGORIZED_LABEL, selected=document.active_group_id is None))
            ungrouped_item.setData(Qt.ItemDataRole.UserRole, None)
            ungrouped_item.setBackground(QColor(UNCATEGORIZED_COLOR))
            ungrouped_item.setForeground(QColor(self._contrast_color(UNCATEGORIZED_COLOR)))
            ungrouped_item.setSizeHint(QSize(128, 36))
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
                item.setBackground(QColor(group.color))
                item.setForeground(QColor(self._contrast_color(group.color)))
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
            canvas.set_tool_mode(self._tool_mode)
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
            unit = document.calibration.unit if document.calibration else "px"
            for row, measurement in enumerate(document.measurements):
                self.measurement_table.insertRow(row)
                display_id = measurement.id.split("_")[-1]
                id_item = QTableWidgetItem(display_id)
                id_item.setData(Qt.ItemDataRole.UserRole, measurement.id)
                self.measurement_table.setItem(row, 0, id_item)
                self.measurement_table.setCellWidget(row, 1, self._create_group_combo(document, measurement))
                self.measurement_table.setItem(row, 2, QTableWidgetItem("半自动" if measurement.mode == "snap" else "手动"))
                self.measurement_table.setItem(row, 3, QTableWidgetItem(f"{(measurement.diameter_unit or 0.0):.4f}"))
                self.measurement_table.setItem(row, 4, QTableWidgetItem(unit))
                self.measurement_table.setItem(row, 5, QTableWidgetItem(f"{measurement.confidence:.2f}"))
                self.measurement_table.setItem(row, 6, QTableWidgetItem(measurement.status))
        self._table_rebuilding = False
        if document is not None:
            self._sync_measurement_table_selection(document)

    def _create_group_combo(self, document: ImageDocument, measurement: Measurement) -> QComboBox:
        combo = QComboBox()
        combo.setProperty("measurement_id", measurement.id)
        combo.addItem(self._color_icon(UNCATEGORIZED_COLOR), UNCATEGORIZED_LABEL, None)
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
                item = self.measurement_table.item(row, 0)
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
        item = self.measurement_table.item(row, 0)
        if item is None:
            return
        measurement_id = item.data(Qt.ItemDataRole.UserRole)
        document.view_state.selected_measurement_id = measurement_id
        canvas.set_selected_measurement(measurement_id)
        self._update_action_states()

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
        document.view_state.selected_measurement_id = measurement_id

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
        has_selected_measurement = has_document and document.view_state.selected_measurement_id is not None and self._tool_mode != "calibration"
        self.close_current_action.setEnabled(has_document)
        self.close_all_action.setEnabled(bool(self.project.documents))
        self.delete_measurement_action.setEnabled(bool(has_selected_measurement))
        self.delete_measurement_button.setEnabled(bool(has_selected_measurement))
        self.add_group_action.setEnabled(has_document)
        self.rename_group_action.setEnabled(has_document and document.get_group(document.active_group_id) is not None if document else False)
        self.undo_action.setEnabled(bool(history and history.can_undo()))
        self.redo_action.setEnabled(bool(history and history.can_redo()))

    def _qimage_to_raster(self, image: QImage) -> RasterImage:
        grayscale = image.convertToFormat(QImage.Format.Format_Grayscale8)
        raster = RasterImage.blank(grayscale.width(), grayscale.height(), fill=255)
        for y in range(grayscale.height()):
            for x in range(grayscale.width()):
                raster.set(x, y, QColor(grayscale.pixel(x, y)).red())
        return raster

    def _group_chip_label(self, text: str, *, selected: bool) -> str:
        return f"✓ {text}" if selected else text

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
            image = source_image.copy()
            image_to_output_scale = 1.0

            def image_to_output(point) -> QPointF:
                return QPointF(point.x, point.y)
        elif render_mode == ExportImageRenderMode.CURRENT_VIEWPORT:
            canvas = self._canvases.get(document.id)
            viewport_width = max(200, canvas.width()) if canvas is not None else max(400, min(1400, source_image.width()))
            viewport_height = max(160, canvas.height()) if canvas is not None else max(300, min(900, source_image.height()))
            image = QImage(viewport_width, viewport_height, QImage.Format.Format_ARGB32)
            image.fill(QColor("#101820"))
            image_to_output_scale = screen_scale

            def image_to_output(point) -> QPointF:
                return QPointF(
                    document.view_state.pan.x + (point.x * screen_scale),
                    document.view_state.pan.y + (point.y * screen_scale),
                )
        else:
            output_width = max(1, int(round(source_image.width() * screen_scale)))
            output_height = max(1, int(round(source_image.height() * screen_scale)))
            image = source_image.scaled(
                output_width,
                output_height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            image_to_output_scale = screen_scale

            def image_to_output(point) -> QPointF:
                return QPointF(point.x * screen_scale, point.y * screen_scale)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        if render_mode == ExportImageRenderMode.CURRENT_VIEWPORT:
            target_rect = QRectF(
                document.view_state.pan.x,
                document.view_state.pan.y,
                source_image.width() * screen_scale,
                source_image.height() * screen_scale,
            )
            painter.drawImage(target_rect, source_image)

        long_edge = max(image.width(), image.height())
        if render_mode == ExportImageRenderMode.FULL_RESOLUTION:
            line_width = max(3.0, min(18.0, long_edge * 0.0025))
            endpoint_radius = max(5.0, line_width * 1.5)
            scale_bg_width = max(8.0, line_width * 2.1)
            scale_fg_width = max(4.0, line_width * 1.1)
            font_px = max(14.0, long_edge * 0.018)
        else:
            line_width = max(2.0, min(6.0, long_edge * 0.003))
            endpoint_radius = max(4.0, line_width * 1.6)
            scale_bg_width = max(6.0, line_width * 2.2)
            scale_fg_width = max(3.0, line_width * 1.1)
            font_px = max(12.0, long_edge * 0.022)

        if include_measurements:
            for measurement in document.measurements:
                group = document.get_group(measurement.fiber_group_id)
                color = QColor(group.color if group else UNCATEGORIZED_COLOR)
                painter.setPen(QPen(color, line_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                line = measurement.effective_line()
                painter.drawLine(image_to_output(line.start), image_to_output(line.end))
                painter.setBrush(color)
                painter.drawEllipse(image_to_output(line.start), endpoint_radius, endpoint_radius)
                painter.drawEllipse(image_to_output(line.end), endpoint_radius, endpoint_radius)

        if include_scale and document.calibration is not None:
            scale_value = self._nice_scale_value(
                document,
                target_output_px=max(80.0, image.width() * 0.18),
                image_to_output_scale=image_to_output_scale,
            )
            if scale_value is not None:
                value, bar_px = scale_value
                margin = max(24, int(round(min(image.width(), image.height()) * 0.04)))
                start_x = float(margin)
                start_y = float(image.height() - margin)
                painter.setPen(QPen(QColor("#111111"), scale_bg_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(QPointF(start_x, start_y), QPointF(start_x + bar_px, start_y))
                painter.setPen(QPen(QColor("#FFFFFF"), scale_fg_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(QPointF(start_x, start_y), QPointF(start_x + bar_px, start_y))
                font = QFont(painter.font())
                font.setPixelSize(int(round(font_px)))
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QPen(QColor("#FFFFFF"), 1))
                painter.drawText(QPointF(start_x, start_y - max(12.0, font_px * 0.55)), f"{value:g} {document.calibration.unit}")

        painter.end()
        image.save(str(output_path))

    def _nice_scale_value(
        self,
        document: ImageDocument,
        *,
        target_output_px: float,
        image_to_output_scale: float,
    ) -> tuple[float, float] | None:
        calibration = document.calibration
        if calibration is None:
            return None
        scaled_pixels_per_unit = calibration.pixels_per_unit * max(image_to_output_scale, 1e-9)
        raw_value = target_output_px / scaled_pixels_per_unit
        if raw_value <= 0:
            return None
        exponent = math.floor(math.log10(raw_value))
        base = raw_value / (10 ** exponent)
        if base < 2:
            nice_base = 1
        elif base < 5:
            nice_base = 2
        else:
            nice_base = 5
        nice_value = nice_base * (10 ** exponent)
        return nice_value, calibration.unit_to_px(nice_value) * image_to_output_scale

    def _color_icon(self, color_value: str) -> QIcon:
        pixmap = QPixmap(12, 12)
        pixmap.fill(QColor(color_value))
        return QIcon(pixmap)

    def _contrast_color(self, color_value: str) -> str:
        color = QColor(color_value)
        luminance = (0.299 * color.red()) + (0.587 * color.green()) + (0.114 * color.blue())
        return "#111111" if luminance > 186 else "#FFFFFF"

    def keyPressEvent(self, event) -> None:
        if event.modifiers() == Qt.KeyboardModifier.NoModifier and Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9:
            number = event.key() - Qt.Key.Key_0
            document = self.current_document()
            if document is not None:
                group = document.get_group_by_number(number)
                if group is not None:
                    document.set_active_group(group.id)
                    self._populate_group_list(document)
                    self._update_action_states()
                    return
        if self._tool_mode != "calibration" and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_measurement()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_close_documents(self.project.documents):
            event.ignore()
            return
        event.accept()
