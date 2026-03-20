from __future__ import annotations

from pathlib import Path
import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QImage, QImageReader, QPainter, QPen
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QComboBox,
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
    new_id,
)
from fdm.project_io import ProjectIO
from fdm.raster import RasterImage
from fdm.services.export_service import ExportService
from fdm.services.model_provider import NullModelProvider, OnnxModelProvider
from fdm.services.snap_service import SnapService
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.dialogs import CalibrationInputDialog, CalibrationPresetDialog


class MainWindow(QMainWindow):
    IMAGE_FILTER = "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    PROJECT_FILTER = "Fiber 项目 (*.fdmproj)"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("纤维直径测量")
        self.resize(1480, 920)

        self.project = ProjectState.empty()
        self._document_order: list[str] = []
        self._images: dict[str, QImage] = {}
        self._rasters: dict[str, RasterImage] = {}
        self._canvases: dict[str, DocumentCanvas] = {}
        self._tool_mode = "select"
        self._color_palette = [
            "#1F7A8C",
            "#E07A5F",
            "#81B29A",
            "#3D405B",
            "#F2CC8F",
            "#6D597A",
        ]

        self.export_service = ExportService()
        self.snap_service = SnapService(model_provider=NullModelProvider())
        self._build_ui()
        self._refresh_preset_combo()
        self._update_model_status()
        self._update_ui_for_current_document()

    def _build_ui(self) -> None:
        self.setStatusBar(QStatusBar())
        self._build_toolbar()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 900, 320])
        self.setCentralWidget(splitter)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("打开图片", self)
        open_action.triggered.connect(self.open_images)
        toolbar.addAction(open_action)

        load_project_action = QAction("打开项目", self)
        load_project_action.triggered.connect(self.load_project)
        toolbar.addAction(load_project_action)

        save_project_action = QAction("保存项目", self)
        save_project_action.triggered.connect(self.save_project)
        toolbar.addAction(save_project_action)

        export_action = QAction("导出结果", self)
        export_action.triggered.connect(self.export_results)
        toolbar.addAction(export_action)
        toolbar.addSeparator()

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
            toolbar.addAction(action)
            mode_group.addAction(action)
            self._mode_actions[mode] = action
        self._mode_actions["select"].setChecked(True)

        toolbar.addSeparator()
        fit_action = QAction("适应窗口", self)
        fit_action.triggered.connect(self.fit_current_image)
        toolbar.addAction(fit_action)

        actual_size_action = QAction("原始像素", self)
        actual_size_action.triggered.connect(self.actual_size_current_image)
        toolbar.addAction(actual_size_action)

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        label = QLabel("已打开图片")
        layout.addWidget(label)

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
        preset_button_row = QHBoxLayout()
        add_preset_button = QPushButton("新增预设")
        add_preset_button.clicked.connect(self.add_calibration_preset)
        apply_preset_button = QPushButton("应用预设")
        apply_preset_button.clicked.connect(self.apply_selected_preset)
        preset_button_row.addWidget(add_preset_button)
        preset_button_row.addWidget(apply_preset_button)
        calibration_layout.addLayout(preset_button_row)
        layout.addWidget(calibration_box)

        group_box = QGroupBox("纤维分组")
        group_layout = QVBoxLayout(group_box)
        self.group_combo = QComboBox()
        self.group_combo.currentIndexChanged.connect(self._on_group_combo_changed)
        group_layout.addWidget(self.group_combo)
        add_group_button = QPushButton("新增分组")
        add_group_button.clicked.connect(self.add_fiber_group)
        group_layout.addWidget(add_group_button)
        layout.addWidget(group_box)

        measurement_box = QGroupBox("测量记录")
        measurement_layout = QVBoxLayout(measurement_box)
        self.measurement_table = QTableWidget(0, 6)
        self.measurement_table.setHorizontalHeaderLabels(["ID", "模式", "结果", "单位", "置信度", "状态"])
        self.measurement_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.measurement_table.verticalHeader().setVisible(False)
        self.measurement_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.measurement_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.measurement_table.itemSelectionChanged.connect(self._on_measurement_selection_changed)
        measurement_layout.addWidget(self.measurement_table)
        delete_measurement_button = QPushButton("删除选中测量")
        delete_measurement_button.clicked.connect(self.delete_selected_measurement)
        measurement_layout.addWidget(delete_measurement_button)
        layout.addWidget(measurement_box, 1)

        return container

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
        document_id = self._document_order[index]
        return self.project.get_document(document_id)

    def current_canvas(self) -> DocumentCanvas | None:
        document = self.current_document()
        if document is None:
            return None
        return self._canvases.get(document.id)

    def open_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", self.IMAGE_FILTER)
        for path in paths:
            self._open_image(path)

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
        target_document.ensure_default_group()

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

    def save_project(self) -> None:
        if not self.project.documents:
            QMessageBox.information(self, "保存项目", "请先打开图片。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存项目",
            "fiber_measurement.fdmproj",
            self.PROJECT_FILTER,
        )
        if not path:
            return
        self.project.version = __version__
        ProjectIO.save(self.project, path)
        self.statusBar().showMessage(f"项目已保存: {path}", 5000)

    def load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", self.PROJECT_FILTER)
        if not path:
            return
        project = ProjectIO.load(path)
        missing_paths = []
        self._reset_workspace()
        self.project = ProjectState(version=project.version, documents=[], calibration_presets=project.calibration_presets)
        for document in project.documents:
            if Path(document.path).exists():
                self._open_image(document.path, document=document)
            else:
                missing_paths.append(document.path)
        self.project.metadata = project.metadata
        self._refresh_preset_combo()
        if missing_paths:
            QMessageBox.warning(
                self,
                "部分图片缺失",
                "以下图片未找到:\n" + "\n".join(missing_paths),
            )
        self.statusBar().showMessage(f"项目已加载: {path}", 5000)

    def export_results(self) -> None:
        if not self.project.documents:
            QMessageBox.information(self, "导出结果", "当前没有可导出的图片。")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not output_dir:
            return
        outputs = self.export_service.export_project(
            self.project,
            output_dir,
            overlay_renderer=self._render_overlay_image,
        )
        output_summary = "\n".join(f"{key}: {value}" for key, value in outputs.items())
        QMessageBox.information(self, "导出完成", f"结果已导出到:\n{output_dir}\n\n{output_summary}")

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
        name, pixels_per_unit, unit = dialog.values()
        if not name:
            QMessageBox.warning(self, "新增预设", "预设名称不能为空。")
            return
        self.project.calibration_presets.append(
            CalibrationPreset(name=name, pixels_per_unit=pixels_per_unit, unit=unit)
        )
        self._refresh_preset_combo()
        self.statusBar().showMessage(f"已新增标定预设: {name}", 4000)

    def apply_selected_preset(self) -> None:
        document = self.current_document()
        preset_index = self.preset_combo.currentIndex()
        if document is None or preset_index < 0 or preset_index >= len(self.project.calibration_presets):
            return
        preset = self.project.calibration_presets[preset_index]
        document.calibration = preset.to_calibration()
        document.recalculate_measurements()
        self._update_ui_for_current_document()
        self.statusBar().showMessage(f"已应用标定预设: {preset.name}", 4000)

    def add_fiber_group(self) -> None:
        document = self.current_document()
        if document is None:
            return
        name, ok = QInputDialog.getText(self, "新增纤维分组", "分组名称")
        if not ok or not name.strip():
            return
        color = self._color_palette[len(document.fiber_groups) % len(self._color_palette)]
        group = FiberGroup(
            id=new_id("group"),
            image_id=document.id,
            name=name.strip(),
            color=color,
        )
        document.fiber_groups.append(group)
        self._refresh_group_combo(document)
        self.statusBar().showMessage(f"已新增分组: {group.name}", 3000)

    def delete_selected_measurement(self) -> None:
        document = self.current_document()
        if document is None:
            return
        measurement_id = document.view_state.selected_measurement_id
        if measurement_id is None:
            return
        document.measurements = [
            measurement for measurement in document.measurements
            if measurement.id != measurement_id
        ]
        for group in document.fiber_groups:
            group.measurement_ids = [
                item for item in group.measurement_ids
                if item != measurement_id
            ]
        document.view_state.selected_measurement_id = None
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.set_selected_measurement(None)
        self._update_ui_for_current_document()

    def load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX 模型", "", "ONNX 文件 (*.onnx)")
        if not path:
            return
        provider = OnnxModelProvider()
        try:
            provider.load(path)
        except Exception as exc:  # noqa: BLE001 - surface exact runtime failure to the user
            QMessageBox.warning(self, "模型加载失败", str(exc))
            return
        self.snap_service.set_model_provider(provider)
        self._update_model_status()
        self.statusBar().showMessage(f"已加载模型: {path}", 5000)

    def _reset_workspace(self) -> None:
        self.project = ProjectState.empty()
        self._document_order.clear()
        self._images.clear()
        self._rasters.clear()
        self._canvases.clear()
        self.image_list.clear()
        self.tab_widget.clear()

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

        group = document.get_group(self.group_combo.currentData()) or document.ensure_default_group()
        if mode == "manual":
            measurement = Measurement(
                id=new_id("meas"),
                image_id=document.id,
                fiber_group_id=group.id,
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
                fiber_group_id=group.id,
                mode="snap",
                line_px=line,
                snapped_line_px=snap_result.snapped_line or line,
                confidence=snap_result.confidence,
                status=snap_result.status if snap_result.snapped_line is not None else "manual_review",
                debug_payload=snap_result.debug_payload,
            )
        document.add_measurement(measurement)
        canvas = self._canvases.get(document.id)
        if canvas is not None:
            canvas.set_selected_measurement(measurement.id)
        self._update_ui_for_current_document()

    def _on_canvas_measurement_selected(self, document_id: str, measurement_id: str) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return
        document.view_state.selected_measurement_id = measurement_id
        self._sync_measurement_table_selection(document)

    def _on_canvas_measurement_edited(self, document_id: str, measurement_id: str, line: Line) -> None:
        document = self.project.get_document(document_id)
        if document is None:
            return
        measurement = document.get_measurement(measurement_id)
        if measurement is None:
            return
        measurement.snapped_line_px = line
        measurement.status = "edited"
        measurement.recalculate(document.calibration)
        document.view_state.selected_measurement_id = measurement.id
        self._update_ui_for_current_document()

    def _apply_calibration_line(self, document: ImageDocument, line: Line) -> None:
        dialog = CalibrationInputDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        actual_length, unit = dialog.values()
        pixels_per_unit = line_length(line) / actual_length
        document.calibration = Calibration(
            mode="image_scale",
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            source_label=f"图内标定 {actual_length:g}{unit}",
        )
        document.metadata["calibration_line"] = line.to_dict()
        document.recalculate_measurements()
        self._update_ui_for_current_document()
        self.statusBar().showMessage("图内标尺标定已更新", 4000)

    def _refresh_preset_combo(self) -> None:
        self.preset_combo.clear()
        for preset in self.project.calibration_presets:
            self.preset_combo.addItem(f"{preset.name} ({preset.pixels_per_unit:g} px/{preset.unit})")

    def _refresh_group_combo(self, document: ImageDocument | None) -> None:
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        if document is None:
            self.group_combo.blockSignals(False)
            return
        for group in document.fiber_groups:
            self.group_combo.addItem(group.name, group.id)
        current_measurement = document.get_measurement(document.view_state.selected_measurement_id)
        if current_measurement and current_measurement.fiber_group_id:
            index = self.group_combo.findData(current_measurement.fiber_group_id)
            if index >= 0:
                self.group_combo.setCurrentIndex(index)
        elif self.group_combo.count() > 0:
            self.group_combo.setCurrentIndex(0)
        self.group_combo.blockSignals(False)

    def _update_ui_for_current_document(self) -> None:
        document = self.current_document()
        self._refresh_group_combo(document)
        self._update_calibration_panel(document)
        self._populate_measurement_table(document)
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.set_tool_mode(self._tool_mode)

    def _update_calibration_panel(self, document: ImageDocument | None) -> None:
        if document is None or document.calibration is None:
            self.calibration_label.setText("当前图片未标定")
            return
        calibration = document.calibration
        self.calibration_label.setText(
            f"{calibration.source_label}\n{calibration.pixels_per_unit:.4f} px/{calibration.unit}"
        )

    def _populate_measurement_table(self, document: ImageDocument | None) -> None:
        self.measurement_table.setRowCount(0)
        if document is None:
            return
        unit = document.calibration.unit if document.calibration else "px"
        for row, measurement in enumerate(document.measurements):
            self.measurement_table.insertRow(row)
            display_id = measurement.id.split("_")[-1]
            values = [
                display_id,
                "半自动" if measurement.mode == "snap" else "手动",
                f"{(measurement.diameter_unit or 0.0):.4f}",
                unit,
                f"{measurement.confidence:.2f}",
                measurement.status,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, measurement.id)
                self.measurement_table.setItem(row, column, item)
        self._sync_measurement_table_selection(document)

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
        self._refresh_group_combo(document)

    def _on_measurement_selection_changed(self) -> None:
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
        measurement = document.get_measurement(measurement_id)
        if measurement and measurement.fiber_group_id:
            combo_index = self.group_combo.findData(measurement.fiber_group_id)
            if combo_index >= 0:
                self.group_combo.setCurrentIndex(combo_index)

    def _on_group_combo_changed(self, index: int) -> None:
        if index < 0:
            return
        document = self.current_document()
        if document is None:
            return
        measurement = document.get_measurement(document.view_state.selected_measurement_id)
        if measurement is None:
            return
        new_group_id = self.group_combo.itemData(index)
        if measurement.fiber_group_id == new_group_id:
            return
        old_group = document.get_group(measurement.fiber_group_id)
        if old_group is not None:
            old_group.measurement_ids = [
                item for item in old_group.measurement_ids
                if item != measurement.id
            ]
        measurement.fiber_group_id = new_group_id
        new_group = document.get_group(new_group_id)
        if new_group is not None and measurement.id not in new_group.measurement_ids:
            new_group.measurement_ids.append(measurement.id)
        self._update_ui_for_current_document()

    def _update_model_status(self) -> None:
        health = self.snap_service.model_provider.healthcheck()
        if health.get("ready"):
            self.model_status_label.setText(f"已加载\n{health.get('model_path', '')}")
        else:
            self.model_status_label.setText(f"未就绪\n{health.get('reason', '') or '将使用传统算法兜底'}")

    def _qimage_to_raster(self, image: QImage) -> RasterImage:
        grayscale = image.convertToFormat(QImage.Format.Format_Grayscale8)
        raster = RasterImage.blank(grayscale.width(), grayscale.height(), fill=255)
        for y in range(grayscale.height()):
            for x in range(grayscale.width()):
                raster.set(x, y, QColor(grayscale.pixel(x, y)).red())
        return raster

    def _render_overlay_image(
        self,
        document: ImageDocument,
        output_path: Path,
        *,
        include_measurements: bool,
        include_scale: bool,
    ) -> None:
        if document.id not in self._images:
            return
        image = self._images[document.id].copy()
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if include_measurements:
            for measurement in document.measurements:
                group = document.get_group(measurement.fiber_group_id)
                color = QColor(group.color if group else "#1F7A8C")
                if measurement.mode == "snap":
                    color = QColor("#1DD3B0") if measurement.status == "snapped" else QColor("#FFB000")
                painter.setPen(QPen(color, 3))
                line = measurement.effective_line()
                painter.drawLine(
                    int(round(line.start.x)),
                    int(round(line.start.y)),
                    int(round(line.end.x)),
                    int(round(line.end.y)),
                )
                painter.setBrush(color)
                painter.drawEllipse(int(line.start.x) - 4, int(line.start.y) - 4, 8, 8)
                painter.drawEllipse(int(line.end.x) - 4, int(line.end.y) - 4, 8, 8)

        if include_scale and document.calibration is not None:
            scale_value = self._nice_scale_value(document)
            if scale_value is not None:
                value, bar_px = scale_value
                start_x = 24
                start_y = image.height() - 28
                painter.setPen(QPen(QColor("#111111"), 8))
                painter.drawLine(start_x, start_y, int(start_x + bar_px), start_y)
                painter.setPen(QPen(QColor("#FFFFFF"), 4))
                painter.drawLine(start_x, start_y, int(start_x + bar_px), start_y)
                painter.drawText(start_x, start_y - 10, f"{value:g} {document.calibration.unit}")

        painter.end()
        image.save(str(output_path))

    def _nice_scale_value(self, document: ImageDocument) -> tuple[float, float] | None:
        calibration = document.calibration
        if calibration is None:
            return None
        target_px = max(60.0, document.image_size[0] * 0.18)
        raw_value = target_px / calibration.pixels_per_unit
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
        return nice_value, calibration.unit_to_px(nice_value)
