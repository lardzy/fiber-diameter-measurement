"""Optional optical preparation tools with bounded background work.

Settings apply to the next analysis. Capturing or inspecting a blank never
replaces the camera image, ends a field, or mutates the experiment ledger.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget, QFileDialog,
)

from fdm.services.optical_config import IMAGING_MODE_CHOICES, OpticalAnalysisConfig


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


class OpticalSettingsDialog(QDialog):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace
        self.controller = workspace.controller
        self.setWindowTitle("普通光学设置与光照检查")
        self.setObjectName("opticalSettingsDialog")
        self.resize(640, 560)
        self._pool = None
        self._job = None
        self._cancel = None
        self._capture_frames = []
        self._capture_leases = []
        self._capture_context = None
        self._capture_started = 0.
        self._last_light = 0.
        self._last_frame_id = None
        self._closed = False
        self._config = OpticalAnalysisConfig.from_dict(
            self.controller.optical_analysis_config, algorithm="optical_adaptive").to_dict()
        self._last_diagnostics = {}
        self._analysis_display_key = None
        self._preview_image = None
        outer = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)

        settings = QWidget(self.tabs)
        form_layout = QVBoxLayout(settings)
        note = QLabel("普通光学适配面向明背景下的暗纤维及浅色双边纤维。候选仍由您复核；不会改变当前显微图像、标定或已经计入的数据。", settings)
        note.setWordWrap(True)
        form_layout.addWidget(note)
        form = QFormLayout()
        self.imaging_mode = QComboBox(settings)
        self.imaging_mode.setAccessibleName("纤维成像结构")
        for label, value in IMAGING_MODE_CHOICES:
            self.imaging_mode.addItem(label, value)
        self.imaging_mode.setCurrentIndex(max(0, self.imaging_mode.findData(self._config.get("imaging_mode"))))
        form.addRow("成像结构", self.imaging_mode)
        form_layout.addLayout(form)
        self.flat_enabled = QCheckBox("启用已采集的空白平场参考", settings)
        self.flat_enabled.setChecked(self._config.get("flat_enabled", False))
        form_layout.addWidget(self.flat_enabled)
        self.reference_status = QLabel(settings)
        self.reference_status.setWordWrap(True)
        form_layout.addWidget(self.reference_status)
        reference_actions = QHBoxLayout()
        self.capture_button = QPushButton("采集 / 重采空白参考（5 帧）", settings)
        self.capture_button.clicked.connect(self._start_capture)
        reference_actions.addWidget(self.capture_button)
        self.reference_preview_button = QPushButton("查看参考", settings)
        self.reference_preview_button.clicked.connect(self._load_reference_preview)
        reference_actions.addWidget(self.reference_preview_button)
        form_layout.addLayout(reference_actions)
        prep = QLabel("先移到同一载玻片无纤维的空白区，保持物镜、曝光、增益及光照一致，再采集。完成后移回样品。平场默认关闭；只影响以后普通光学分析的计算输入。", settings)
        prep.setWordWrap(True)
        form_layout.addWidget(prep)
        self.preview = QLabel(settings)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(240, 120)
        self.preview.setMaximumHeight(180)
        self.preview.setText("未打开参考预览")
        form_layout.addWidget(self.preview, 1)
        self.tabs.addTab(settings, "普通光学设置")

        diagnostics = QWidget(self.tabs)
        diagnostics_layout = QVBoxLayout(diagnostics)
        info = QLabel("光照检查仅提供准备建议，不阻止测量。下方数值描述图像，不代表纤维测量准确率。最多每秒检查两次；不会触发全图分析。", diagnostics)
        info.setWordWrap(True)
        diagnostics_layout.addWidget(info)
        self.light_status = QLabel("等待图像", diagnostics)
        self.light_status.setWordWrap(True)
        diagnostics_layout.addWidget(self.light_status)
        self.diagnostics_text = QPlainTextEdit(diagnostics)
        self.diagnostics_text.setReadOnly(True)
        self.diagnostics_text.setAccessibleName("光照与分析诊断详情，可选择复制")
        diagnostics_layout.addWidget(self.diagnostics_text, 1)
        self.export_button = QPushButton("导出离线光照诊断…", diagnostics)
        self.export_button.clicked.connect(self._export_diagnostics)
        diagnostics_layout.addWidget(self.export_button)
        self.tabs.addTab(diagnostics, "光照与诊断")

        self.status = QLabel(self)
        self.status.setWordWrap(True)
        self.status.setAccessibleName("光学设置状态")
        outer.addWidget(self.status)
        buttons = QDialogButtonBox(self)
        self.save_button = buttons.addButton("保存设置", QDialogButtonBox.ButtonRole.ApplyRole)
        self.refresh_button = buttons.addButton("应用并刷新候选", QDialogButtonBox.ButtonRole.ApplyRole)
        buttons.addButton("关闭", QDialogButtonBox.ButtonRole.RejectRole)
        self.save_button.clicked.connect(lambda: self._apply(False))
        self.refresh_button.clicked.connect(lambda: self._apply(True))
        buttons.rejected.connect(self.close)
        outer.addWidget(buttons)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._refresh_reference_label()

    def _raw_frame(self):
        return self.controller.latest_frame or (
            self.controller.presented.frame if self.controller.presented is not None else None)

    def _context(self):
        from fdm.services.optical_flat_field import reference_context
        frame = self._raw_frame()
        return reference_context(frame, self.controller.calibration) if frame is not None else None

    def _checked_context(self):
        try:
            return True, self._context()
        except (TypeError, ValueError, OverflowError) as exc:
            message = f"成像信息暂不可用，已暂停光学准备：{exc}"
            if self.status.text() != message:
                self.status.setText(message)
            if self._cancel is not None:
                self._cancel.set()
            self._release_capture()
            return False, None

    def reload_config(self):
        """Discard unsaved dialog edits when explicitly opening it again."""
        self._config = self.controller.optical_analysis_config
        self.imaging_mode.setCurrentIndex(max(0, self.imaging_mode.findData(self._config.get("imaging_mode"))))
        self.flat_enabled.setChecked(self._config.get("flat_enabled", False))
        self._refresh_reference_label()

    def _analysis_summary(self):
        from fdm.ui.optical_diagnostics import flat_field_status_text
        context = getattr(self.controller, "_automatic_context", None) or {}
        return flat_field_status_text(context.get("diagnostics", {}))

    def _refresh_diagnostic_display(self):
        context = getattr(self.controller, "_automatic_context", None) or {}
        analysis = context.get("diagnostics")
        key = (context.get("task_id"), context.get("completed"), id(analysis),
               id(self._last_diagnostics))
        if key == self._analysis_display_key:
            return
        self._analysis_display_key = key
        from fdm.ui.optical_diagnostics import illumination_summary
        summary = illumination_summary(self._last_diagnostics) if self._last_diagnostics else "等待图像光照检查。"
        if self._analysis_summary():
            summary = self._analysis_summary() + "\n" + summary
        self.light_status.setText(summary)
        self.diagnostics_text.setPlainText(_json({"光照检查": self._last_diagnostics,
            "最近全图分析": analysis or {}}))

    def _refresh_reference_label(self):
        reference = self._config.get("flat_reference")
        self.reference_status.setText("已采集参考；只有来源、分辨率、标尺和成像设置一致时才使用。" if reference else "尚无空白平场参考；关闭平场时使用局部背景校正。")
        self.reference_preview_button.setEnabled(bool(reference) and not self._job)

    def _apply(self, refresh):
        config = dict(self._config, imaging_mode=self.imaging_mode.currentData(),
                      flat_enabled=self.flat_enabled.isChecked())
        try:
            self.controller.configure_optical_analysis(config)
            self._config = self.controller.optical_analysis_config
            if refresh:
                self.workspace.algorithm.setCurrentIndex(self.workspace.algorithm.findData("optical_adaptive"))
            self.workspace._assistance_preferences_changed()
            if refresh:
                self.workspace._analyze_candidates()
            self.status.setText("已应用并请求刷新候选。" if refresh else "已保存，将用于下次普通光学分析；当前候选保持原参数。")
        except (ValueError, RuntimeError) as exc:
            self.status.setText(str(exc))

    def _start_capture(self):
        if self._job or self._capture_context is not None:
            return
        frame = self._raw_frame()
        if frame is None or not frame.live:
            self.status.setText("请先连接相机，移到空白区后采集连续 5 帧。")
            return
        valid, context = self._checked_context()
        if not valid:
            return
        self._capture_context = context
        self._capture_started = time.monotonic()
        self.capture_button.setEnabled(False)
        self.status.setText("正在采集空白参考 0 / 5；请保持光照与图像稳定。")

    def _release_capture(self):
        for lease in self._capture_leases:
            self.controller.budget.release(lease)
        self._capture_frames.clear()
        self._capture_leases.clear()
        self._capture_context = None
        self.capture_button.setEnabled(not bool(self._job))

    def _submit(self, kind, work, context, leases=(), scratch=None):
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optical-preparation")
        self._cancel = threading.Event()
        cancel = self._cancel
        budget = self.controller.budget
        def checked():
            if cancel.is_set():
                raise RuntimeError("操作已取消。")
        def run():
            try:
                checked()
                return work(checked)
            finally:
                for lease in leases:
                    budget.release(lease)
                if scratch is not None:
                    budget.release_scratch(scratch)
        try:
            future = self._pool.submit(run)
        except BaseException:
            for lease in leases:
                budget.release(lease)
            if scratch is not None:
                budget.release_scratch(scratch)
            raise
        self._job = (kind, future, context)
        self.capture_button.setEnabled(False)
        self.reference_preview_button.setEnabled(False)

    def _collect_frame(self):
        valid, context = self._checked_context()
        if not valid:
            return
        if self._capture_context != context:
            self.status.setText("来源、标尺或成像设置已改变，已取消空白采集。")
            self._release_capture()
            return
        if time.monotonic() - self._capture_started > 8.:
            self.status.setText("未收到足够的新帧，请检查采集后重试。")
            self._release_capture()
            return
        frame = self._raw_frame()
        if frame is None or any(frame.frame_id == item.frame_id for item in self._capture_frames):
            return
        try:
            lease = self.controller.budget.acquire(frame, keep_preview_slots=2)
            self._capture_leases.append(lease)
            self._capture_frames.append(frame)
            self.status.setText(f"正在采集空白参考 {len(self._capture_frames)} / 5")
            if len(self._capture_frames) < 5:
                return
            from fdm.services.optical_flat_field import (
                estimate_flat_field_scratch_bytes, prepare_flat_field_reference, save_flat_field_reference,
            )
            frames, context = list(self._capture_frames), self._capture_context
            # Scratch is charged separately from the five immutable frame leases.
            scratch_bytes = max(estimate_flat_field_scratch_bytes(frame.pixels.shape, operation=operation)
                                for operation in ("prepare", "save"))
            extra, ticket = self.controller.budget.acquire_with_scratch(frame,
                group="optical-preparation", scratch_bytes=scratch_bytes, keep_preview_slots=2)
            leases = [*self._capture_leases, extra]
            self._capture_frames, self._capture_leases, self._capture_context = [], [], None
            def prepare(check):
                result = prepare_flat_field_reference(frames, context,
                    memory_budget=scratch_bytes, checkpoint=check)
                if result.get("status") not in {"ok", "ready"}:
                    return result
                check()
                reference = save_flat_field_reference(result)
                return {"status": "ok", "reference": reference, "diagnostics": result.get("diagnostics", {})}
            self._submit("capture", prepare, context, leases, ticket)
            self.status.setText("正在检查空白质量并生成平场参考…")
        except (ValueError, RuntimeError, OSError) as exc:
            self.status.setText(str(exc))
            self._release_capture()

    def _load_reference_preview(self):
        if self._job or self._capture_context is not None:
            return
        from fdm.services.optical_flat_field import estimate_flat_field_scratch_bytes, load_flat_field_reference
        valid, context = self._checked_context()
        if not valid:
            return
        reference = self._config.get("flat_reference")
        frame = self._raw_frame()
        if reference and frame is not None:
            scratch = estimate_flat_field_scratch_bytes(frame.pixels.shape, operation="load")
            try:
                lease, ticket = self.controller.budget.acquire_with_scratch(frame,
                    group="optical-preparation", scratch_bytes=scratch, keep_preview_slots=2)
            except RuntimeError as exc:
                self.status.setText(str(exc))
                return
            def preview(check):
                import cv2
                result = load_flat_field_reference(reference, context, memory_budget=scratch)
                check()
                image = result.get("image")
                if image is not None:
                    ratio = min(1., 500 / image.shape[1], 160 / image.shape[0])
                    result["image"] = cv2.resize(image, (max(1, round(image.shape[1] * ratio)),
                                                         max(1, round(image.shape[0] * ratio))))
                return result
            self._submit("preview", preview, context, [lease], ticket)
        elif reference:
            self.status.setText("请先提供当前来源图像，以核对参考对应的成像条件。")

    def _export_diagnostics(self):
        if self._job or self._capture_context is not None:
            self.status.setText("正在检查图像，请稍后导出。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出离线光照诊断", "光照诊断.zip", "ZIP (*.zip)")
        if not path:
            return
        from copy import deepcopy
        from fdm.version import __version__
        valid, context = self._checked_context()
        if not valid:
            return
        config = dict(self._config, imaging_mode=self.imaging_mode.currentData(),
                      flat_enabled=self.flat_enabled.isChecked())
        analysis = getattr(self.controller, "_automatic_context", None) or {}
        payload = {"schema_version": 1, "application_version": __version__,
                   "config": deepcopy(config), "context": context,
                   "illumination": deepcopy(self._last_diagnostics),
                   "analysis": deepcopy(analysis.get("diagnostics", {})),
                   "real_hardware_validation": False}
        leases, ticket, reference_budget = [], None, None
        reference = config.get("flat_reference")
        if reference:
            try:
                from fdm.services.optical_flat_field import estimate_flat_field_scratch_bytes
                binding = reference.get("context") or {}
                height, width = int(binding.get("height", 0)), int(binding.get("width", 0))
                channels = {"GRAY": 1, "BGR": 3, "BGRA": 4}[binding.get("pixel_format")]
                shape = (height, width) if channels == 1 else (height, width, channels)
                reference_budget = estimate_flat_field_scratch_bytes(shape, operation="load")
                owner = self._raw_frame()
                if owner is None:
                    # The diagnostic exporter can be used after disconnecting
                    # the source. A one-byte owner lets the shared budget track
                    # reference decode scratch without holding an old frame.
                    import numpy as np
                    from fdm.live_experiment.frames import FrameHandle
                    owner = FrameHandle.snapshot(np.ones((1, 1), np.uint8),
                        source_id="optical-diagnostic-export", live=False)
                lease, ticket = self.controller.budget.acquire_with_scratch(owner,
                    group="optical-preparation", scratch_bytes=reference_budget, keep_preview_slots=2)
                leases.append(lease)
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                payload["flat_reference_error"] = str(exc)
                reference_budget = None
        def export(check):
            import os
            from pathlib import Path
            import tempfile
            import zipfile
            from fdm.services.optical_flat_field import flat_field_export_files
            target = Path(path)
            try:
                files = flat_field_export_files(reference, memory_budget=reference_budget) if reference and reference_budget else ()
            except (ValueError, OSError) as exc:
                # Broken references are precisely when offline feedback is
                # needed. Keep the descriptor and error, omit invalid pixels.
                payload["flat_reference_error"] = str(exc)
                files = ()
            fd, temporary = tempfile.mkstemp(prefix=".optical-diagnostics-", suffix=".zip", dir=target.parent)
            os.close(fd)
            try:
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("diagnostics.json", _json(payload))
                    for item in files:
                        check()
                        archive.write(item, f"flat-reference/{item.name}")
                check()
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return {"path": str(target), "reference_error": payload.get("flat_reference_error")}
        self._submit("export", export, context, leases, ticket)
        self.status.setText("正在导出诊断和经过校验的平场参考…")

    def _start_light(self, frame):
        from fdm.services.optical_flat_field import illumination_diagnostics
        valid, context = self._checked_context()
        if not valid:
            return
        scratch = frame.size[0] * frame.size[1] * 12
        try:
            lease, ticket = self.controller.budget.acquire_with_scratch(frame,
                group="optical-preparation", scratch_bytes=scratch, keep_preview_slots=2)
        except RuntimeError as exc:
            self.light_status.setText(str(exc))
            return
        self._last_light = time.monotonic()
        self._last_frame_id = frame.frame_id
        self._submit("light", lambda check: illumination_diagnostics(frame.pixels),
                     context, [lease], ticket)

    def _tick(self):
        if self._closed:
            return
        if self.tabs.currentIndex() == 1:
            # Reanalysis of a static image keeps its frame ID. Refresh just the
            # diagnostic text when the task changes, without rechecking light.
            self._refresh_diagnostic_display()
        valid, current_context = self._checked_context()
        if not valid:
            return
        if self._job is not None:
            kind, future, context = self._job
            if context != current_context:
                self._cancel.set()
            if not future.done():
                return
            self._job = None
            self.capture_button.setEnabled(True)
            self._refresh_reference_label()
            try:
                result = future.result()
                if context != current_context or self._cancel.is_set():
                    self.status.setText("来源或成像设置已改变，未应用刚才的检查结果。")
                    return
                self._completed(kind, result)
            except Exception as exc:
                self.status.setText(f"光学准备未完成：{exc}")
        if self._capture_context is not None:
            self._collect_frame()
        elif self._job is None and self.tabs.currentIndex() == 1:
            frame = self._raw_frame()
            if frame is not None and frame.frame_id != self._last_frame_id and time.monotonic() - self._last_light >= .5:
                self._start_light(frame)

    def _completed(self, kind, result):
        if kind == "export":
            self.status.setText(f"已导出离线光照诊断：{result['path']}")
            if result.get("reference_error"):
                self.status.setText(self.status.text() + "\n参考不可用，已记录原因；未附加无效图片。")
        elif kind == "capture":
            if result.get("status") == "ok" and result.get("reference"):
                self._config["flat_reference"] = result["reference"]
                self._refresh_reference_label()
                self.status.setText("空白参考已采集。可查看参考，再启用并保存设置；不会自动改变当前候选。")
                suggestions = (result.get("diagnostics") or {}).get("suggestions") or []
                if suggestions:
                    self.status.setText(self.status.text() + "\n" + "\n".join(suggestions[:3]))
            else:
                self.status.setText(result.get("message") or "空白质量不适合平场，请移到均匀无纤维区域重采。")
            self.diagnostics_text.setPlainText(_json(result.get("diagnostics", result)))
            self._analysis_display_key = None
        elif kind == "preview":
            image = result.get("image")
            if image is None:
                self.status.setText(result.get("message") or "参考文件不可用。")
                return
            import numpy as np
            array = np.asarray(image)
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            array = np.ascontiguousarray(array)
            if array.ndim == 3 and array.shape[2] == 4:
                import cv2
                array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
                fmt = QImage.Format.Format_RGBA8888
            else:
                fmt = QImage.Format.Format_Grayscale8 if array.ndim == 2 else QImage.Format.Format_BGR888
            qimage = QImage(array.data, array.shape[1], array.shape[0], array.strides[0], fmt)
            self._preview_image = qimage.scaled(500, 160, Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation).copy()
            self.preview.setPixmap(QPixmap.fromImage(self._preview_image))
            self.status.setText("平场参考预览；不是测量证据或当前相机画面。")
        else:
            self._last_diagnostics = result
            self._refresh_diagnostic_display()

    def showEvent(self, event):
        self._closed = False
        if self._analysis_summary():
            self.status.setText(self._analysis_summary())
        self._timer.start()
        super().showEvent(event)

    def event(self, event):
        if event.type() == QEvent.Type.ShortcutOverride and event.key() == Qt.Key.Key_Escape:
            # The owning trial has an Escape action for its drawing tools.
            # Keep Escape local so closing this dialog cannot cancel a draft.
            event.accept()
            return True
        return super().event(event)

    def _stop_tools(self):
        """All closing paths, including Escape/reject, release owned work."""
        self._closed = True
        self._timer.stop()
        if self._cancel is not None:
            self._cancel.set()
        self._release_capture()
        if self._pool is not None:
            # A running task releases its own frame/scratch leases in finally.
            self._pool.shutdown(wait=False, cancel_futures=False)
            self._pool = None
        # Keep the running future until it exits. Reopening this dialog must
        # not start a second executor while the cancelled task still owns frames.
        self._preview_image = None
        self.preview.clear()

    def hideEvent(self, event):
        # QDialog.reject()/done() hide the window without sending closeEvent.
        self._stop_tools()
        super().hideEvent(event)

    def closeEvent(self, event):
        self._stop_tools()
        super().closeEvent(event)
