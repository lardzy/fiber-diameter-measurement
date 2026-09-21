"""Channel selection and cancellable, background device-image ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
import json
import uuid

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from fdm.atomic_io import atomic_write_json
from fdm.raster import RasterPlane
from fdm.services.device_image_io import (
    CHANNEL_LABELS, COMMON_CHANNELS, DeviceChannel, DeviceReadCancelled,
    default_selection, inspect_source, read_channel,
)
from fdm.services.raster_io import (
    raster_plane_to_qimage, write_native_raster_asset, recommended_native_asset_suffix,
)


@dataclass(frozen=True)
class LoadedDeviceChannel:
    channel: DeviceChannel
    plane: RasterPlane
    image: QImage
    relative_path: str
    session_path: Path


class DeviceReadWorker(QObject):
    itemReady = Signal(object)
    progress = Signal(int, int, str)
    finished = Signal(bool, object)

    def __init__(self, operation, items, *, asset_root=None, export_path=None):
        super().__init__()
        self.operation = operation
        self.items = items
        self.asset_root = asset_root
        self.export_path = export_path
        self.cancelled = Event()

    def cancel(self):
        self.cancelled.set()

    @Slot()
    def run(self):
        errors = []
        for index, item in enumerate(self.items):
            if self.cancelled.is_set():
                break
            label = str(item) if self.operation == "scan" else item.display_name
            self.progress.emit(index, len(self.items), label)
            session_path = None
            try:
                if self.operation == "scan":
                    result = inspect_source(item, cancelled=self.cancelled.is_set)
                else:
                    plane = read_channel(item, cancelled=self.cancelled.is_set)
                    if self.operation == "export":
                        encoded = write_native_raster_asset(plane, self.export_path)
                        if not encoded:
                            raise ValueError(str(encoded.failure))
                        payload = item.source_metadata()
                        payload["calibration"] = item.calibration.to_dict() if item.calibration else None
                        payload["height_values"] = "raw_counts; absolute Z origin is not established"
                        atomic_write_json(Path(self.export_path).with_suffix(".json"), payload, ensure_ascii=False, indent=2)
                        result = self.export_path
                    else:
                        image = raster_plane_to_qimage(plane, display_transform=item.display_transform)
                        if self.operation == "preview":
                            result = (item, image)
                        else:
                            relative = f"imports/{uuid.uuid4().hex}{recommended_native_asset_suffix(plane.pixel_type)}"
                            session_path = Path(self.asset_root) / relative
                            session_path.parent.mkdir(parents=True, exist_ok=True)
                            encoded = write_native_raster_asset(plane, session_path)
                            if not encoded:
                                raise ValueError(str(encoded.failure))
                            result = LoadedDeviceChannel(item, plane, image, relative, session_path)
                if self.cancelled.is_set() and self.operation != "export":
                    if session_path is not None:
                        session_path.unlink(missing_ok=True)
                    break
                self.itemReady.emit(result)
            except DeviceReadCancelled:
                break
            except Exception as exc:  # isolate one source/channel and keep completed items
                if session_path is not None:
                    session_path.unlink(missing_ok=True)
                errors.append(f"{label}: {exc}")
        self.finished.emit(self.cancelled.is_set(), errors)


class DeviceImportDialog(QDialog):
    channelLoaded = Signal(object)
    focusDocument = Signal(str)
    calibrationSelected = Signal(object)

    def __init__(self, paths, *, asset_root, preferred=("intensity",), existing=None,
                 calibration_target_size=None, calibration_only=False,
                 expected_fingerprint=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("从设备文件读取标尺" if calibration_only else "选择设备图像")
        self.resize(920, 630)
        self._paths = paths
        self._asset_root = asset_root
        self._preferred = tuple(c for c in preferred if c in COMMON_CHANNELS)
        self._existing = dict(existing or {})
        self._calibration_only = calibration_only
        self._target_size = calibration_target_size
        self._expected_fingerprint = expected_fingerprint
        self._channels = []
        self._selected_channel = None
        self._thread = None
        self._worker = None
        self._closing = False
        self._errors = []
        self._cancelled = False
        self._operation = ""
        self.imported_count = 0
        self.imported_common = set()

        layout = QVBoxLayout(self)
        self.description = QLabel("正在读取点位、通道和内置标尺…")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        controls = QHBoxLayout()
        self.commonChecks = {}
        for kind in COMMON_CHANNELS:
            check = QCheckBox(CHANNEL_LABELS[kind])
            check.setChecked(kind in self._preferred)
            self.commonChecks[kind] = check
            controls.addWidget(check)
        apply = QPushButton("应用到已选点位")
        apply.clicked.connect(self._apply_common)
        controls.addWidget(apply)
        select_all = QPushButton("全选常用通道")
        select_all.clicked.connect(lambda: self._set_common(True))
        controls.addWidget(select_all)
        select_none = QPushButton("取消全选")
        select_none.clicked.connect(lambda: self._set_common(False))
        controls.addWidget(select_none)
        controls.addStretch()
        self.commonControls = QWidget()
        self.commonControls.setLayout(controls)
        self.commonControls.setVisible(not calibration_only)
        layout.addWidget(self.commonControls)
        self.tree = self._tree()
        layout.addWidget(self.tree, 1)

        self.advancedToggle = QCheckBox("高级：高度图（默认不导入）")
        layout.addWidget(self.advancedToggle)
        self.advanced = QWidget()
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        self.heightTree = self._tree()
        advanced_layout.addWidget(self.heightTree)
        height_actions = QHBoxLayout()
        for label, action in (("预览高度", "preview"), ("导出原始高度…", "export")):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, a=action: self._height_action(a))
            height_actions.addWidget(button)
        height_actions.addWidget(QLabel("勾选高度后点击下方“加入项目”；预览/导出不会加入项目。"))
        if not calibration_only:
            advanced_layout.addLayout(height_actions)
        else:
            for index in range(height_actions.count()):
                widget = height_actions.itemAt(index).widget()
                if widget is not None:
                    widget.deleteLater()
        self.advanced.setVisible(False)
        self.advancedToggle.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.advanced)

        self.confirmField = QCheckBox("确认当前图片与来源为同一完整视野，未裁切或缩放")
        self.confirmField.setVisible(calibration_only)
        self.confirmField.toggled.connect(self._refresh)
        layout.addWidget(self.confirmField)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)
        self.progressBar = QProgressBar()
        layout.addWidget(self.progressBar)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        copy = QPushButton("复制所选标尺")
        copy.clicked.connect(self._copy_calibration)
        buttons.addWidget(copy)
        self.relocateButton = QPushButton("重新定位原文件…")
        self.relocateButton.setVisible(expected_fingerprint is not None)
        self.relocateButton.clicked.connect(self._relocate)
        buttons.addWidget(self.relocateButton)
        buttons.addStretch()
        self.importButton = QPushButton("应用标尺" if calibration_only else "加入项目")
        self.importButton.clicked.connect(self._commit)
        self.cancelButton = QPushButton("取消")
        self.cancelButton.clicked.connect(self.reject)
        buttons.addWidget(self.importButton)
        buttons.addWidget(self.cancelButton)
        layout.addLayout(buttons)
        self._refresh()
        QTimer.singleShot(0, self, lambda: self._start("scan", self._paths))

    def _tree(self):
        tree = QTreeWidget()
        tree.setHeaderLabels(["点位 / 通道", "尺寸", "标尺（µm/px）", "状态"])
        tree.setColumnWidth(0, 330)
        tree.setColumnWidth(1, 125)
        tree.setColumnWidth(2, 250)
        tree.itemChanged.connect(self._refresh)
        tree.currentItemChanged.connect(self._selection_changed)
        return tree

    def _start(self, operation, items, export_path=None):
        if self._thread is not None:
            return
        self._operation = operation
        self._errors = []
        self._cancelled = False
        self._thread = QThread(self)
        self._worker = DeviceReadWorker(operation, items, asset_root=self._asset_root, export_path=export_path)
        self._worker.moveToThread(self._thread)
        self._worker.itemReady.connect(self._item_ready)
        self._worker.progress.connect(self._progress)
        self._worker.finished.connect(self._finished)
        self._worker.finished.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.started.connect(self._worker.run)
        self._thread.finished.connect(self._thread_finished)
        self._thread.start()
        self._refresh()

    @Slot(int, int, str)
    def _progress(self, current, total, label):
        self.progressBar.setRange(0, total)
        self.progressBar.setValue(current)
        self.status.setText(f"{current + 1}/{total} · {label}")

    @Slot(object)
    def _item_ready(self, result):
        if self._operation == "scan":
            known = {c.identity for c in self._channels}
            self._channels.extend(c for c in result if c.identity not in known)
        elif self._operation == "import":
            self.channelLoaded.emit(result)
            # The window registers the real document ID while handling the
            # signal. Standalone consumers may only need duplicate suppression.
            self._existing.setdefault(result.channel.identity, "")
            for item in self._leaves(self.heightTree if result.channel.kind == "height" else self.tree):
                if item.data(0, Qt.ItemDataRole.UserRole).identity == result.channel.identity:
                    item.setText(3, "已导入")
            self.imported_count += 1
            if result.channel.kind in COMMON_CHANNELS and result.channel.device == "OLS5000":
                self.imported_common.add(result.channel.kind)
        elif self._operation == "preview" and not self._closing:
            channel, image = result
            preview = QDialog(self)
            preview.setWindowTitle(channel.display_name + " · 高度预览")
            root = QVBoxLayout(preview)
            label = QLabel()
            label.setPixmap(QPixmap.fromImage(image).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            root.addWidget(label)
            root.addWidget(QLabel("显示为相对高度预览；原始计数和 Z 比例可通过“导出原始高度”获取。"))
            preview.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            preview.show()
        elif self._operation == "export":
            self.status.setText(f"已导出 {result} 和标定 JSON")

    @Slot(bool, object)
    def _finished(self, cancelled, errors):
        self._cancelled = cancelled
        self._errors = errors

    @Slot()
    def _thread_finished(self):
        # finished is emitted before native thread-local cleanup. Deleting a
        # PySide QThread while that cleanup still needs the GIL can deadlock
        # its destructor. wait() releases the GIL; never destroy it early.
        if not self._thread.wait(1):
            QTimer.singleShot(10, self, self._thread_finished)
            return
        self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self.progressBar.setValue(self.progressBar.maximum())
        if self._closing:
            super().reject()
            return
        if self._operation == "scan":
            if self._expected_fingerprint and any(c.source_sha256 != self._expected_fingerprint for c in self._channels):
                self._channels.clear()
                self._errors.append("所选文件与原来源指纹不符，请重新定位原文件。")
            self._populate()
        if self._errors:
            self.status.setText("\n".join(self._errors[:8]))
        elif self._operation == "scan":
            self.status.setText("选择需要的点位和通道。")
        elif self._operation == "import" and not self._cancelled:
            super().accept()
            return
        self._refresh()

    def _populate(self):
        self.tree.blockSignals(True)
        self.heightTree.blockSignals(True)
        self.tree.clear()
        self.heightTree.clear()
        defaults = {c.identity for c in default_selection(self._channels, self._preferred)}
        parents = {}
        for channel in self._channels:
            target = self.heightTree if channel.kind == "height" else self.tree
            key = (channel.source_sha256, channel.dataset_id)
            if channel.kind == "height":
                item = QTreeWidgetItem(target, [channel.display_name])
            else:
                if key not in parents:
                    parent = QTreeWidgetItem(target, [" · ".join(filter(None, [Path(channel.source_path).name, channel.dataset_label]))])
                    if not self._calibration_only:
                        parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
                    parents[key] = parent
                item = QTreeWidgetItem(parents[key], [CHANNEL_LABELS[channel.kind]])
            item.setData(0, Qt.ItemDataRole.UserRole, channel)
            item.setToolTip(0, channel.display_name)
            if not self._calibration_only:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked if channel.identity in defaults else Qt.CheckState.Unchecked)
            item.setText(1, f"{channel.width} × {channel.height}")
            cal = channel.calibration
            item.setText(2, f"X {cal.pixel_size_x:.9g} / Y {cal.pixel_size_y:.9g}" if cal else "标定待核验")
            item.setText(3, "已导入" if channel.identity in self._existing else channel.calibration_issue or "可用")
        for parent in parents.values():
            available = {parent.child(i).data(0, Qt.ItemDataRole.UserRole).kind
                         for i in range(parent.childCount())}
            for kind in COMMON_CHANNELS:
                if kind not in available:
                    missing = QTreeWidgetItem(parent, [CHANNEL_LABELS[kind], "", "", "此点位无此通道"])
                    missing.setDisabled(True)
        for tree in (self.tree, self.heightTree):
            tree.expandAll()
            tree.blockSignals(False)
        self.description.setText(f"{len(parents)} 个数据集；请选择所需通道。未选通道不会加入项目。")
        if not any(c.kind == "intensity" for c in self._channels):
            self.description.setText(self.description.text() + " 文件不包含激光强度。")
        for kind, check in self.commonChecks.items():
            check.setEnabled(any(c.kind == kind for c in self._channels))
        if self.tree.topLevelItemCount() and self.tree.topLevelItem(0).childCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0).child(0))

    def _leaves(self, tree):
        for i in range(tree.topLevelItemCount()):
            parent = tree.topLevelItem(i)
            if parent.childCount():
                for j in range(parent.childCount()):
                    item = parent.child(j)
                    if isinstance(item.data(0, Qt.ItemDataRole.UserRole), DeviceChannel):
                        yield item
            elif isinstance(parent.data(0, Qt.ItemDataRole.UserRole), DeviceChannel):
                yield parent

    def register_imported_document(self, identity: str, document_id: str) -> None:
        self._existing[identity] = document_id

    def selected_channels(self):
        return [item.data(0, Qt.ItemDataRole.UserRole)
                for tree in (self.tree, self.heightTree) for item in self._leaves(tree)
                if item.checkState(0) == Qt.CheckState.Checked]

    def _selection_changed(self, current, previous):
        self._selected_channel = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        self._refresh()

    def _current(self, *, height=False):
        if height:
            item = self.heightTree.currentItem()
            return item.data(0, Qt.ItemDataRole.UserRole) if item else None
        return self._selected_channel

    def _set_common(self, checked):
        for item in self._leaves(self.tree):
            item.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _apply_common(self):
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            if parent.checkState(0) == Qt.CheckState.Unchecked:
                continue
            for j in range(parent.childCount()):
                item = parent.child(j)
                channel = item.data(0, Qt.ItemDataRole.UserRole)
                if channel is not None:
                    item.setCheckState(0, Qt.CheckState.Checked if self.commonChecks[channel.kind].isChecked() else Qt.CheckState.Unchecked)

    def _toggle_advanced(self, checked):
        self.advanced.setVisible(checked)
        if not checked:
            for item in self._leaves(self.heightTree):
                item.setCheckState(0, Qt.CheckState.Unchecked)
        self._refresh()

    def _refresh(self, *_):
        if not hasattr(self, "importButton"):
            return
        busy = self._thread is not None
        self.tree.setEnabled(not busy)
        self.advanced.setEnabled(not busy)
        self.commonControls.setEnabled(not busy)
        self.advancedToggle.setEnabled(not busy)
        self.relocateButton.setEnabled(not busy)
        channel = self._current()
        if channel:
            cal = channel.calibration
            text = (f"X {cal.pixel_size_x:.12g} / Y {cal.pixel_size_y:.12g} µm/px · {cal.source_label}"
                    if cal else f"标定待核验：{channel.calibration_issue}")
            if self._calibration_only:
                text += f"\n来源 {channel.width} × {channel.height}；当前图片 {self._target_size or '未打开'}"
            self.details.setText(text)
        if self._calibration_only:
            self.importButton.setEnabled(not busy and channel is not None and channel.calibration is not None
                and self._target_size == (channel.width, channel.height) and self.confirmField.isChecked())
        else:
            selected = self.selected_channels()
            count = sum(channel.identity not in self._existing for channel in selected)
            self.importButton.setText(f"加入项目（{count} 张）" if count or not selected else f"定位已导入图片（{len(selected)} 张）")
            self.importButton.setEnabled(not busy and bool(selected))
        self.cancelButton.setText("取消任务并关闭" if busy else "关闭")

    def _commit(self):
        if self._calibration_only:
            channel = self._current()
            if channel and channel.calibration and self._target_size == (channel.width, channel.height) and self.confirmField.isChecked():
                self.calibrationSelected.emit(channel)
                self.accept()
            return
        channels = []
        for channel in self.selected_channels():
            if channel.identity in self._existing:
                if document_id := self._existing[channel.identity]:
                    self.focusDocument.emit(document_id)
                if channel.kind in COMMON_CHANNELS and channel.device == "OLS5000":
                    self.imported_common.add(channel.kind)
            else:
                channels.append(channel)
        if channels:
            self._start("import", channels)
        else:
            self.accept()

    def _height_action(self, action):
        if self._thread is not None:
            return
        channel = self._current(height=True)
        if channel is None:
            self.status.setText("请先在高级列表中选中一张高度图。")
            return
        destination = None
        if action == "export":
            destination, _ = QFileDialog.getSaveFileName(self, "导出原始高度", channel.display_name + ".tif", "TIFF (*.tif)")
            if not destination:
                return
            destination = str(Path(destination).with_suffix(".tif"))
        self._start(action, [channel], destination)

    def _copy_calibration(self):
        channel = self._current()
        if channel:
            payload = channel.source_metadata()
            payload["calibration"] = channel.calibration.to_dict() if channel.calibration else None
            QGuiApplication.clipboard().setText(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))

    def _relocate(self):
        path, _ = QFileDialog.getOpenFileName(self, "重新定位设备原文件", "", "设备图像 (*.dsx *.poir *.mpoir)")
        if path:
            self._channels.clear()
            self._selected_channel = None
            self.tree.clear()
            self.heightTree.clear()
            self._start("scan", [path])

    def reject(self):
        if self._thread is not None:
            self._closing = True
            self._worker.cancel()
            self.cancelButton.setEnabled(False)
            self.status.setText("正在取消；已完成的导入将保留。")
            return
        super().reject()

    def closeEvent(self, event):
        if self._thread is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
