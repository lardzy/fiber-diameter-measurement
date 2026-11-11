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
        self.trial_mode = workspace.session.workspace_options["mode"]
        self.preferences = template_preferences(self.trial_mode)
        self._live_manager = None
        self._methods_source = None
        self._task_source = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-template-report")
        self.future = None
        self.cancel_event = threading.Event()
        self.diagnostics = {}
        self.last_report = None
        layout = workspace.report_layout
        self.path_label = QLabel(workspace)
        self.path_label.setTextFormat(Qt.TextFormat.PlainText)
        self.path_label.setWordWrap(True)
        self.path_label.installEventFilter(self)
        layout.addWidget(self.path_label)
        self.live_template_label = QLabel(workspace)
        self.live_template_label.setTextFormat(Qt.TextFormat.PlainText)
        self.live_template_label.setWordWrap(True)
        layout.addWidget(self.live_template_label)
        row = QHBoxLayout()
        for title, callback in (("选择模板", self.choose), ("打开模板", self.open), ("所在位置", self.folder)):
            button = QPushButton(title)
            if callback == self.choose:
                button.setToolTip("为当前试验从所选文件新建实时工作簿，并作为后续试验模板。旧工作簿保留供您保存；同路径重选也会重新加载。")
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
        self.method.addItem("当前试验方法 → 自动匹配同名模板标准", None)
        self.method.currentIndexChanged.connect(self.method_changed)
        layout.addWidget(self.method)
        self.method.setVisible(self.trial_mode == "content")
        self.method.setToolTip("为当前试验方法指定模板中的同一标准标签。修改后将为当前试验新建实时工作簿，旧工作簿保留；后续试验使用此映射，其他已打开试验不变。")
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
        self._sync_actions()

    def _sync_actions(self):
        workspace = self.workspace
        if hasattr(workspace, "export_button"):
            workspace.export_button.setEnabled(self.future is None)
            workspace.export_button.setText("导出中…" if self.future is not None else "导出")
            workspace.export_button.setToolTip("另行导出一次性 Excel 报告；实时工作簿可在 Excel 中直接保存。")
        for name in ("open_report_button", "report_folder_button"):
            button = getattr(workspace, name, None)
            if button is not None:
                button.setEnabled(self.last_report is not None)

    def _set_status(self, text):
        self.status.setText(text)
        self._sync_actions()

    def eventFilter(self, watched, event):
        if watched is self.path_label and event.type() == QEvent.Type.Show:
            self.refresh_path()
        if watched is getattr(self.workspace, "report_status_label", None) and event.type() == QEvent.Type.Resize:
            self._sync_actions()
        return False

    def path(self, *, create=False):
        # Pages can outlive a selection made in another trial. Never write an
        # old page's cached preference dictionary back over the new selection.
        self.preferences = template_preferences(self.trial_mode)
        path = Path(self.preferences["template"]) if self.preferences.get("template") else working_template(create=create, trial_mode=self.trial_mode)
        source = self._template_source(self.preferences, path=path)
        if source != self._methods_source:
            self._reset_methods(source)
        return path

    def _template_source(self, preferences, *, path=None):
        """A standard list belongs to the file/mapping that produced it."""
        path = path or (Path(preferences["template"]) if preferences.get("template")
                        else working_template(trial_mode=self.trial_mode))
        mapping = preferences.get("mapping") or ""

        def identity(value):
            if not value:
                return "", None
            value = Path(value)
            try:
                stat = value.stat()
                stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
            except OSError:
                stamp = None
            return str(value), stamp

        return identity(path), identity(mapping)

    def _reset_methods(self, source):
        self._methods_source = source
        previous = self.method.blockSignals(True)
        self.method.clear()
        self.method.addItem("当前试验方法 → 自动匹配同名模板标准", None)
        self.method.blockSignals(previous)

    def refresh_path(self):
        path = self.path()
        label = "后续试验模板" if self.preferences.get("template") else "后续试验模板（个人工作副本）"
        self.path_label.setText(label + "：" + str(path))

    def bind_live_excel(self, manager):
        if self._live_manager is manager:
            return
        if self._live_manager is not None:
            self._live_manager.changed.disconnect(self._live_changed)
        self._live_manager = manager
        manager.changed.connect(self._live_changed)
        self._live_changed(self.workspace.session.id)

    def _live_changed(self, session_id):
        if session_id != self.workspace.session.id or self._live_manager is None:
            return
        state = self._live_manager.status(session_id)
        path = state.get("template_path")
        label = "当前实时工作簿模板" if state.get("connected") else "上次实时工作簿模板"
        lines = [label + "：" + path] if path else ["当前试验尚未建立实时工作簿。"]
        if state.get("pending_template_path"):
            prefix = "正在更换为：" if state.get("template_replacement_pending") else "更换未完成："
            lines.append(prefix + state["pending_template_path"])
        self.live_template_label.setText("\n".join(lines))

    def _replace_live_template(self):
        if self._live_manager is not None:
            self._live_manager.replace_template(self.workspace.session, preferences=self.preferences)

    def _require_idle_template_task(self):
        if self.future is not None:
            raise ValueError("已有模板任务正在处理，请完成或取消后再选择模板；测量可以继续。")

    def choose(self):
        self._require_idle_template_task()
        path, _ = QFileDialog.getOpenFileName(self.workspace, "选择可编辑报告模板", str(self.path()),
            "Excel 模板 (*.xlt *.xltm *.xltx);;Excel 工作簿 (*.xls *.xlsm *.xlsx)")
        if path:
            if not Path(path).is_file():
                raise FileNotFoundError("所选模板不存在，请重新选择；当前工作簿保留。")
            self.preferences = template_preferences(self.trial_mode)
            self.preferences["template"] = path
            self.preferences.pop("method_label", None)
            self.preferences.pop("method_labels", None)
            save_template_preferences(self.preferences, trial_mode=self.trial_mode)
            self._reset_methods(self._template_source(self.preferences))
            self.refresh_path()
            self._replace_live_template()
            self.inspect()

    def open(self):
        path = self.path(create=True)
        if not path.is_file():
            raise FileNotFoundError("请先选择报告模板。")
        request = {"operation": "edit", "template": str(path),
                   "profile": asdict(TemplateProfile.load(self.preferences.get("mapping"), trial_mode=self.trial_mode))}
        self.start(lambda: run_excel_request(request, cancel_event=self.cancel_event))

    def folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path(create=True).parent)))

    def choose_mapping(self):
        self._require_idle_template_task()
        path, _ = QFileDialog.getOpenFileName(self.workspace, "选择 CU 模板输入映射", "", "JSON (*.json)")
        if path:
            TemplateProfile.load(path, trial_mode=self.trial_mode)
            self.preferences = template_preferences(self.trial_mode)
            self.preferences["mapping"] = path
            save_template_preferences(self.preferences, trial_mode=self.trial_mode)
            self.refresh_path()
            self._replace_live_template()
            self._set_status("输入映射已更新，将为当前试验新建实时工作簿，旧工作簿保留；请重新读取模板标准。")

    def method_changed(self, _):
        self.preferences = template_preferences(self.trial_mode)
        source = self._template_source(self.preferences)
        if source != self._methods_source:
            # A hidden panel may still contain labels read from the old file.
            # Never attach such a label to another panel's new preference.
            self._reset_methods(source)
            self.refresh_path()
            self._set_status("模板或输入映射已改变，旧标准列表已清除；请重新读取模板标准后选择。")
            return
        self.preferences.pop("method_label", None)
        mappings = dict(self.preferences.get("method_labels") or {})
        method, label = self.workspace.session.method, self.method.currentData()
        if mappings.get(method) == label:
            return
        mappings[method] = label
        self.preferences["method_labels"] = mappings
        save_template_preferences(self.preferences, trial_mode=self.trial_mode)
        self._replace_live_template()
        self._set_status("标准映射已更新，将为当前试验新建实时工作簿；旧工作簿保留，其他已打开试验不变。")

    def start(self, function, *, source=None):
        if self.future is not None:
            raise ValueError("已有模板任务正在处理，测量可以继续。")
        self.cancel_event.clear()
        self.set_diagnostics({})
        self._task_source = source
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
                   "profile": asdict(TemplateProfile.load(self.preferences.get("mapping"), trial_mode=self.trial_mode))}
        self.start(lambda: run_excel_request(request, cancel_event=self.cancel_event), source=self._methods_source)

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
        profile = TemplateProfile.load(self.preferences.get("mapping"), trial_mode=self.trial_mode)
        snapshot = ExportSnapshot.capture(self.workspace.session)
        label = (self.preferences.get("method_labels") or {}).get(snapshot.method)
        self.start(lambda: run_excel_request(template_request(snapshot, template, path, profile,
                                                              method_label=label), cancel_event=self.cancel_event))

    def poll(self):
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        source, self._task_source = self._task_source, None
        self.timer.stop()
        self.cancel_button.setEnabled(False)
        if source is not None:
            self.refresh_path()
            if source != self._methods_source:
                self._set_status("模板或输入映射已改变，已忽略旧模板返回的标准列表；请重新读取模板标准。")
                return
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
            self.method.addItem("当前试验方法 → 自动匹配同名模板标准", None)
            for item in result["methods"]:
                self.method.addItem(item["label"], item["label"])
            self.method.setCurrentIndex(max(0, self.method.findData((self.preferences.get("method_labels") or {}).get(self.workspace.session.method))))
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
