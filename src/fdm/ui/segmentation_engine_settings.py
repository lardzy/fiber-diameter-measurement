from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from fdm.services.segmentation_engines import OfflineSegmentationEngineService
from fdm.settings import OfflineSegmentationEnginePack


class _EngineDiagnosticSignals(QObject):
    finished = Signal(str, object)


class _EngineDiagnosticTask(QRunnable):
    def __init__(
        self,
        service: OfflineSegmentationEngineService,
        record: OfflineSegmentationEnginePack,
    ) -> None:
        super().__init__()
        self._service = service
        self._record = record
        self.signals = _EngineDiagnosticSignals()

    def run(self) -> None:
        try:
            outcome: object = self._service.diagnose(self._record)
        except Exception as exc:  # noqa: BLE001 - surfaced in the manager UI
            outcome = exc
        self.signals.finished.emit(self._record.engine_id, outcome)


class OfflineSegmentationEngineDialog(QDialog):
    def __init__(
        self,
        records: list[OfflineSegmentationEnginePack],
        *,
        service: OfflineSegmentationEngineService,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("离线分割引擎管理")
        self.resize(820, 500)
        self._service = service
        self._records = [record.normalized_copy() for record in records]
        self._diagnostic_task: _EngineDiagnosticTask | None = None

        intro = QLabel(
            "这里仅管理可选的 SAM3 与 μSAM 本机引擎包，不会替换当前同类扩选、"
            "注册新分割工具或改变标准 EdgeSAM。引擎包必须提供纯 CPU 路径；"
            "仅在点击诊断时执行包内命令。"
        )
        intro.setWordWrap(True)

        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(("引擎", "名称", "版本", "设备", "路径 / 状态"))
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._sync_buttons)

        self._import_button = QPushButton("导入 ZIP…", self)
        self._import_button.clicked.connect(self._import_pack)
        self._link_button = QPushButton("关联现有目录…", self)
        self._link_button.clicked.connect(self._link_pack)
        self._diagnose_button = QPushButton("运行 CPU 诊断", self)
        self._diagnose_button.clicked.connect(self._diagnose_selected)
        self._remove_button = QPushButton("移除配置", self)
        self._remove_button.clicked.connect(self._remove_selected)
        self._delete_button = QPushButton("删除包文件…", self)
        self._delete_button.clicked.connect(self._delete_selected_pack)
        action_layout = QHBoxLayout()
        action_layout.addWidget(self._import_button)
        action_layout.addWidget(self._link_button)
        action_layout.addStretch(1)
        action_layout.addWidget(self._diagnose_button)
        action_layout.addWidget(self._remove_button)
        action_layout.addWidget(self._delete_button)

        self._details = QPlainTextEdit(self)
        self._details.setReadOnly(True)
        self._details.setMaximumBlockCount(400)
        self._details.setPlaceholderText("选择引擎并运行诊断后，在此显示版本、资源和运行结果。")
        self._details.setMaximumHeight(145)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("完成")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self._table, 1)
        layout.addLayout(action_layout)
        layout.addWidget(self._details)
        layout.addWidget(buttons)
        self._rebuild_table()

    def records(self) -> list[OfflineSegmentationEnginePack]:
        return [record.normalized_copy() for record in self._records]

    def _selected_row(self) -> int:
        ranges = self._table.selectedRanges()
        return ranges[0].topRow() if ranges else -1

    def _sync_buttons(self) -> None:
        selected = 0 <= self._selected_row() < len(self._records)
        idle = self._diagnostic_task is None
        self._import_button.setEnabled(idle)
        self._link_button.setEnabled(idle)
        self._diagnose_button.setEnabled(selected and idle)
        self._remove_button.setEnabled(selected and idle)
        managed = selected and self._records[self._selected_row()].managed
        self._delete_button.setEnabled(bool(managed and idle))

    def _rebuild_table(self, *, select_engine_id: str = "") -> None:
        self._table.setRowCount(len(self._records))
        selected_row = -1
        for row, record in enumerate(self._records):
            try:
                inspection = self._service.inspect(record.path, managed=record.managed)
                status = f"已验证 · {inspection.resource_count} 个资源 · {record.path}"
            except (OSError, ValueError) as exc:
                status = f"不可用：{exc}"
            values = (
                "SAM3" if record.engine_id == "sam3" else "μSAM",
                record.display_name,
                record.version,
                "CPU",
                status,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self._table.setItem(row, column, item)
            if record.engine_id == select_engine_id:
                selected_row = row
        self._table.resizeColumnsToContents()
        self._table.setColumnWidth(4, max(260, self._table.columnWidth(4)))
        if selected_row >= 0:
            self._table.selectRow(selected_row)
        elif self._records:
            self._table.selectRow(0)
        self._sync_buttons()

    def _install_record(self, record: OfflineSegmentationEnginePack) -> None:
        replaced = False
        for index, existing in enumerate(self._records):
            if existing.engine_id == record.engine_id:
                response = QMessageBox.question(
                    self,
                    "替换离线引擎",
                    f"已存在 {existing.display_name} {existing.version}。是否替换为 {record.display_name} {record.version}？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if response != QMessageBox.StandardButton.Yes:
                    return
                self._records[index] = record
                replaced = True
                break
        if not replaced:
            self._records.append(record)
        self._rebuild_table(select_engine_id=record.engine_id)

    def _import_pack(self) -> None:
        source, _filter = QFileDialog.getOpenFileName(
            self,
            "导入离线分割引擎 ZIP",
            str(Path.home()),
            "离线引擎包 (*.zip);;所有文件 (*)",
        )
        if not source:
            return
        try:
            inspection = self._service.import_package(source)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "导入离线引擎", f"引擎包检查失败：\n{exc}")
            return
        self._install_record(inspection.record)

    def _link_pack(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "关联现有离线引擎目录",
            str(Path.home()),
        )
        if not directory:
            return
        try:
            inspection = self._service.inspect(directory, managed=False)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "关联离线引擎", f"引擎目录检查失败：\n{exc}")
            return
        self._install_record(inspection.record)

    def _diagnose_selected(self) -> None:
        if self._diagnostic_task is not None:
            return
        row = self._selected_row()
        if not (0 <= row < len(self._records)):
            return
        record = self._records[row]
        self._details.setPlainText(
            "正在后台运行离线 CPU 诊断，请稍候…\n"
            "诊断只读取当前引擎目录，不会注册工具或上传图像。"
        )
        task = _EngineDiagnosticTask(self._service, record)
        task.signals.finished.connect(self._diagnostic_finished)
        self._diagnostic_task = task
        self._diagnose_button.setText("诊断中…")
        self._sync_buttons()
        QThreadPool.globalInstance().start(task)

    def _diagnostic_finished(self, engine_id: str, outcome: object) -> None:
        self._diagnostic_task = None
        self._diagnose_button.setText("运行 CPU 诊断")
        self._sync_buttons()
        if isinstance(outcome, Exception):
            self._details.setPlainText(f"诊断失败：{outcome}")
            return
        if not hasattr(outcome, "details"):
            self._details.setPlainText("诊断失败：引擎返回了无法识别的结果。")
            return
        result = outcome
        lines = [f"{engine_id} · {result.message}"]
        lines.extend(f"{key}: {value}" for key, value in sorted(result.details.items()))
        if result.stdout.strip():
            lines.extend(("", "stdout:", result.stdout.strip()))
        if result.stderr.strip():
            lines.extend(("", "stderr:", result.stderr.strip()))
        self._details.setPlainText("\n".join(lines))

    def _remove_selected(self) -> None:
        row = self._selected_row()
        if not (0 <= row < len(self._records)):
            return
        record = self._records[row]
        message = f"从当前配置草稿中移除 {record.display_name} {record.version}？"
        if record.managed:
            message += "\n\n已导入的包文件会保留，避免取消首选项时破坏原配置。"
        response = QMessageBox.question(
            self,
            "移除离线引擎",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        self._records.pop(row)
        self._rebuild_table()

    def _delete_selected_pack(self) -> None:
        row = self._selected_row()
        if not (0 <= row < len(self._records)):
            return
        record = self._records[row]
        if not record.managed:
            return
        response = QMessageBox.question(
            self,
            "删除离线引擎包",
            (
                f"立即删除 {record.display_name} {record.version} 的软件托管包文件？\n\n"
                "此操作会立即释放磁盘空间，无法通过取消首选项恢复。"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        try:
            self._service.remove_managed_pack(record)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "删除离线引擎包", f"无法删除引擎包：\n{exc}")
            return
        self._records.pop(row)
        self._details.setPlainText("已删除软件托管的离线引擎包文件。")
        self._rebuild_table()
