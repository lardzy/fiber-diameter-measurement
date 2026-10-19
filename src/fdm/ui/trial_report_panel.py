"""Optional report work, isolated from the measurement controller and GUI slot."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import threading
from PySide6.QtCore import QObject, QTimer, QUrl, QEvent, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget
from fdm.atomic_io import atomic_write_text
from fdm.ui.widgets import NoWheelComboBox
from fdm.live_experiment.excel_template import (
    ExportSnapshot, TemplateProfile, run_excel_request, template_request,
    template_preferences, save_template_preferences, working_template,
)


class TrialReportPanel(QObject):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace
        self.preferences = template_preferences()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-template-report")
        self.future = None
        self.cancel_event = threading.Event()
        self.diagnostics = {}
        self.last_report = None
        layout = workspace.report_layout
        self.path_label = QLabel(workspace)
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)
        row = QHBoxLayout()
        for title, callback in (("选择模板", self.choose), ("打开模板", self.open), ("所在位置", self.folder)):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, fn=callback: workspace._guard(fn))
            row.addWidget(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        for title, callback in (("读取模板标准", self.inspect), ("选择输入映射", self.choose_mapping)):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, fn=callback: workspace._guard(fn))
            row.addWidget(button)
        layout.addLayout(row)
        self.method = NoWheelComboBox(workspace)
        self.method.addItem("使用试验当前方法（按模板标签匹配）", None)
        self.method.currentIndexChanged.connect(self.method_changed)
        layout.addWidget(self.method)
        self.status = QLabel("生成后请复核报告方法与结果，再编辑或打印。", workspace)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.last_report_label = QLabel(workspace)
        self.last_report_label.setObjectName("templateLastSuccessfulReport")
        self.last_report_label.setWordWrap(True)
        self.last_report_label.hide()
        layout.addWidget(self.last_report_label)
        row = QHBoxLayout()
        self.cancel_button = QPushButton("取消模板任务", workspace)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        row.addWidget(self.cancel_button)
        self.diagnostics_toggle = QPushButton("展开诊断", workspace)
        self.diagnostics_toggle.setCheckable(True)
        self.diagnostics_toggle.setEnabled(False)
        self.diagnostics_toggle.toggled.connect(self.toggle_diagnostics)
        row.addWidget(self.diagnostics_toggle)
        layout.addLayout(row)
        self.diagnostics_widget = QWidget(workspace)
        self.diagnostics_widget.setObjectName("templateDiagnostics")
        details = QVBoxLayout(self.diagnostics_widget)
        details.setContentsMargins(0, 0, 0, 0)
        self.diagnostics_text = QPlainTextEdit(self.diagnostics_widget)
        self.diagnostics_text.setObjectName("templateDiagnosticsText")
        self.diagnostics_text.setReadOnly(True)
        self.diagnostics_text.setMaximumHeight(170)
        details.addWidget(self.diagnostics_text)
        row = QHBoxLayout()
        self.copy_diagnostics_button = QPushButton("复制诊断", self.diagnostics_widget)
        self.copy_diagnostics_button.clicked.connect(self.copy_diagnostics)
        row.addWidget(self.copy_diagnostics_button)
        self.export_diagnostics_button = QPushButton("导出诊断", self.diagnostics_widget)
        self.export_diagnostics_button.clicked.connect(lambda: workspace._guard(self.export_diagnostics))
        row.addWidget(self.export_diagnostics_button)
        details.addLayout(row)
        layout.addWidget(self.diagnostics_widget)
        self.diagnostics_widget.hide()
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.refresh_path()
        if hasattr(workspace, "report_status_label"):
            workspace.report_status_label.installEventFilter(self)
        self._sync_actions()

    def _sync_actions(self):
        workspace = self.workspace
        if hasattr(workspace, "export_button"):
            workspace.export_button.setEnabled(self.future is None)
            workspace.export_button.setText("正在导出…" if self.future is not None else "导出 Excel 报告")
        label = getattr(workspace, "report_status_label", None)
        if label is not None:
            text = self.status.text().splitlines()[0] if self.status.text() else ""
            label.setToolTip(self.status.text())
            label.setText(label.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, max(40, label.width())))
        for name in ("open_report_button", "report_folder_button"):
            button = getattr(workspace, name, None)
            if button is not None:
                button.setEnabled(self.last_report is not None)

    def _set_status(self, text):
        self.status.setText(text)
        self._sync_actions()

    def eventFilter(self, watched, event):
        if watched is getattr(self.workspace, "report_status_label", None) and event.type() == QEvent.Type.Resize:
            self._sync_actions()
        return False

    def path(self, *, create=False):
        return Path(self.preferences["template"]) if self.preferences.get("template") else working_template(create=create)

    def refresh_path(self):
        self.path_label.setText("报告模板：" + str(self.path()))

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(self.workspace, "选择可编辑报告模板", str(self.path()),
            "Excel 模板 (*.xlt *.xltm *.xltx);;Excel 工作簿 (*.xls *.xlsm *.xlsx)")
        if path:
            self.preferences["template"] = path
            self.preferences.pop("method_label", None)
            save_template_preferences(self.preferences)
            self.method.blockSignals(True)
            self.method.clear()
            self.method.addItem("使用试验当前方法（按模板标签匹配）", None)
            self.method.blockSignals(False)
            self.refresh_path()
            self.inspect()

    def open(self):
        path = self.path(create=True)
        if not path.is_file():
            raise FileNotFoundError("请先选择报告模板。")
        request = {"operation": "edit", "template": str(path),
                   "profile": asdict(TemplateProfile.load(self.preferences.get("mapping")))}
        self.start(lambda: run_excel_request(request, cancel_event=self.cancel_event))

    def folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path(create=True).parent)))

    def choose_mapping(self):
        path, _ = QFileDialog.getOpenFileName(self.workspace, "选择 CU 模板输入映射", "", "JSON (*.json)")
        if path:
            TemplateProfile.load(path)
            self.preferences["mapping"] = path
            save_template_preferences(self.preferences)

    def method_changed(self, _):
        self.preferences["method_label"] = self.method.currentData()
        save_template_preferences(self.preferences)

    def start(self, function):
        if self.future is not None:
            raise ValueError("已有模板任务正在处理，测量可以继续。")
        self.cancel_event.clear()
        self.set_diagnostics({})
        self.future = self.executor.submit(function)
        self.cancel_button.setEnabled(True)
        self._set_status("Excel 正在后台处理，实时采集与计数继续。")
        self.timer.start()

    def cancel(self):
        if self.future is not None:
            self.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self._set_status("正在取消模板任务并清理本次 Excel 实例，试验记录继续保留。")

    def toggle_diagnostics(self, checked):
        self.diagnostics_widget.setVisible(checked)
        self.diagnostics_toggle.setText("收起诊断" if checked else "展开诊断")

    def set_diagnostics(self, diagnostics):
        self.diagnostics = diagnostics
        self.diagnostics_text.setPlainText(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False))
        self.diagnostics_toggle.setEnabled(bool(diagnostics))
        if not diagnostics:
            self.diagnostics_toggle.setChecked(False)

    def copy_diagnostics(self):
        QApplication.clipboard().setText(self.diagnostics_text.toPlainText())

    def export_diagnostics(self):
        path, _ = QFileDialog.getSaveFileName(self.workspace, "导出模板报告诊断", "模板报告诊断.json", "JSON (*.json)")
        if path:
            atomic_write_text(path, self.diagnostics_text.toPlainText(), encoding="utf-8")

    def inspect(self):
        request = {"operation": "inspect", "template": str(self.path(create=True)),
                   "profile": asdict(TemplateProfile.load(self.preferences.get("mapping")))}
        self.start(lambda: run_excel_request(request, cancel_event=self.cancel_event))

    def export(self):
        if self.future is not None:
            raise ValueError("已有模板任务正在处理，测量可以继续。")
        flush = getattr(self.workspace, "commit_pending_edits", None)
        if flush is not None:
            flush()
        template = self.path(create=True)
        suffix = ".xlsx" if template.suffix.lower() == ".xltx" else ".xlsm" if template.suffix.lower() in {".xltm", ".xlsm"} else ".xls"
        path, _ = QFileDialog.getSaveFileName(self.workspace, "生成模板报告", "试验报告" + suffix,
                                             f"Excel (*{suffix})")
        if not path:
            return
        profile = TemplateProfile.load(self.preferences.get("mapping"))
        snapshot = ExportSnapshot.capture(self.workspace.session)
        label = self.preferences.get("method_label")
        self.start(lambda: run_excel_request(template_request(snapshot, template, path, profile,
                                                              method_label=label), cancel_event=self.cancel_event))

    def poll(self):
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        self.timer.stop()
        self.cancel_button.setEnabled(False)
        try:
            result = future.result()
        except Exception as exc:
            self.set_diagnostics(getattr(exc, "diagnostics", {"error_type": type(exc).__name__, "message": str(exc)}))
            self._set_status(str(exc))
            return
        self.set_diagnostics(result.get("diagnostics", {}))
        if result.get("operation") == "edit":
            self._set_status("模板原文件已在 Excel 中打开，可直接编辑并保存。")
            return
        if "methods" in result:
            self.method.blockSignals(True)
            self.method.clear()
            self.method.addItem("使用试验当前方法（按模板标签匹配）", None)
            for item in result["methods"]:
                self.method.addItem(item["label"], item["label"])
            self.method.setCurrentIndex(max(0, self.method.findData(self.preferences.get("method_label"))))
            self.method.blockSignals(False)
            self._set_status(f"已读取 {len(result['methods'])} 个模板方法；未改动模板文件。"
                                + ("\n" + "\n".join(result["warnings"]) if result.get("warnings") else ""))
            return
        lines = ["报告已生成：" + result["destination"], "报告方法：" + result["method"]]
        self.last_report = result
        self.last_report_label.setText("最近成功报告：" + result["destination"])
        self.last_report_label.show()
        lines.extend(result.get("warnings", []))
        for item in result.get("comparisons", []):
            local, report = item.get("local_content"), item.get("template_content")
            if isinstance(local, (int, float)) and isinstance(report, (int, float)) and abs(local - report) > .011:
                lines.append(f"{item['operator']}／{item['direction']}／{item['material']}：软件 {local:.3f}%，模板 {report:.3f}%")
        if result.get("status") != "ready":
            lines.insert(0, "模板计算需要复核；完整原始数据已写入。")
        self._set_status("\n".join(lines))
        self.workspace.statusBar().showMessage(lines[0], 8000)

    def close(self):
        self.timer.stop()
        self.cancel_event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
