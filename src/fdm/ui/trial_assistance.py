"""Trial-only interaction glue; display preferences never enter the ledger."""
from __future__ import annotations

import json
import math

from PySide6.QtCore import QItemSelectionModel, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCursor
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QLabel, QMenu, QSlider,
    QToolButton, QWidget, QWidgetAction, QCheckBox, QSizePolicy, QDoubleSpinBox,
)


class TrialAssistanceMixin:
    def _init_assistance(self):
        from fdm.ui.icons import themed_icon
        from fdm.ui.measurement_evidence_view import EvidenceReviewController
        self.evidence_review = EvidenceReviewController(self)
        self.candidate_model.session = self.session
        self._candidate_linking = False
        self._assistance_loading = True
        self._assistance_state_key = None
        self.measure_mode.setToolTip("选择手工测径算法。默认同时支持候选点选、两次点击和拖线；高级选项可调整交互模式。")
        row = self.view_controls.layout()
        row.setSpacing(4)
        row.itemAt(0).widget().hide()
        # Let the secondary view selector shrink to its font/style-aware
        # content width when the records dock shares a narrow viewport. A
        # fixed 108px floor can force the whole main window to grow.
        self.display_choice.setMinimumWidth(0)
        self.display_choice.setMaximumWidth(150)
        self.analyze_button = QToolButton(self.view_controls)
        self.analyze_button.setObjectName("trialAnalyze")
        self.analyze_button.setText("全图分析")
        self.analyze_button.setIcon(themed_icon("statistics"))
        self.analyze_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.analyze_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.analyze_button.setAccessibleName("全图分析或刷新候选")
        self.analyze_button.clicked.connect(lambda: self._guard(self._analyze_candidates))
        analyze_menu = QMenu(self.analyze_button)
        algorithm_group = QActionGroup(analyze_menu)
        self.algorithm_actions = {}
        for label, algorithm in (("传统全图", "traditional"), ("多尺度全图", "multiscale")):
            action = analyze_menu.addAction(label)
            action.setCheckable(True)
            algorithm_group.addAction(action)
            action.triggered.connect(lambda checked=False, name=algorithm: self.algorithm.setCurrentIndex(self.algorithm.findData(name)))
            self.algorithm_actions[algorithm] = action
        analyze_menu.addSeparator()
        analyze_menu.addAction("取消分析并清除未接受候选", self._cancel_candidate_analysis)
        analyze_menu.addAction("导出离线性能诊断…", self._export_assistance_diagnostics)
        self.analyze_button.setMenu(analyze_menu)
        row.insertWidget(0, self.analyze_button)
        self.auto_analysis = QCheckBox("自动", self.view_controls)
        self.auto_analysis.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.auto_analysis.setAccessibleName("稳定后自动分析")
        self.auto_analysis.setToolTip("视场连续稳定 300 毫秒后分析一次；候选仍需人工点选。")
        row.insertWidget(1, self.auto_analysis)
        self.candidate_display_button = QToolButton(self.view_controls)
        self.candidate_display_button.setText("候选")
        self.candidate_display_button.setAccessibleName("候选显示与不透明度")
        self.candidate_display_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.candidate_display_menu = QMenu(self.candidate_display_button)
        self.show_candidates_action = self.candidate_display_menu.addAction("显示候选")
        self.show_candidates_action.setCheckable(True)
        self.show_candidates_action.setChecked(True)
        opacity_panel = QWidget(self.candidate_display_menu)
        opacity_row = QHBoxLayout(opacity_panel)
        opacity_row.addWidget(QLabel("不透明度", opacity_panel))
        self.candidate_opacity = QSlider(Qt.Orientation.Horizontal, opacity_panel)
        self.candidate_opacity.setRange(20, 100)
        self.candidate_opacity.setValue(60)
        self.candidate_opacity.setAccessibleName("候选不透明度")
        self.candidate_opacity.setMinimumWidth(100)
        opacity_row.addWidget(self.candidate_opacity)
        self.candidate_opacity_label = QLabel("60%", opacity_panel)
        opacity_row.addWidget(self.candidate_opacity_label)
        slider_action = QWidgetAction(self.candidate_display_menu)
        slider_action.setDefaultWidget(opacity_panel)
        self.candidate_display_menu.addAction(slider_action)
        retention_panel = QWidget(self.candidate_display_menu)
        retention_row = QHBoxLayout(retention_panel)
        retention_row.addWidget(QLabel("换场后保留已点选标记", retention_panel))
        self.selection_retention_seconds = QDoubleSpinBox(retention_panel)
        self.selection_retention_seconds.setObjectName("trialSelectionRetentionSeconds")
        self.selection_retention_seconds.setAccessibleName("换场后保留已点选标记")
        self.selection_retention_seconds.setRange(0., 10.)
        self.selection_retention_seconds.setDecimals(1)
        self.selection_retention_seconds.setSingleStep(.1)
        self.selection_retention_seconds.setSuffix(" 秒")
        self.selection_retention_seconds.setSpecialValueText("关闭")
        self.selection_retention_seconds.setValue(1.)
        self.selection_retention_seconds.setToolTip("仅保留画布点选且成功计入的标记；修改时长影响之后的换场，关闭会立即清除。")
        retention_row.addWidget(self.selection_retention_seconds)
        retention_action = QWidgetAction(self.candidate_display_menu)
        retention_action.setDefaultWidget(retention_panel)
        self.candidate_display_menu.addAction(retention_action)
        self.candidate_display_button.setMenu(self.candidate_display_menu)
        row.insertWidget(2, self.candidate_display_button)
        row.insertStretch(3, 1)
        self.evidence_button = QToolButton(self)
        self.evidence_button.setObjectName("trialEvidence")
        self.evidence_button.setAccessibleName("证据相册及存档设置")
        self.evidence_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.evidence_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.evidence_button.setIcon(themed_icon("open_images"))
        self.evidence_button.setIconSize(QSize(16, 16))
        self.evidence_button.setFixedSize(40, 22)
        self.evidence_button.clicked.connect(lambda: self._guard(self._show_evidence_gallery))
        evidence_menu = QMenu(self.evidence_button)
        evidence_menu.addAction("查看上一条测量证据", lambda: self._guard(self._show_latest_evidence))
        evidence_menu.addAction("固定上一条为复核小窗", lambda: self._guard(self._pin_latest_evidence))
        evidence_menu.addSeparator()
        group = QActionGroup(evidence_menu)
        self.evidence_mode_actions = {}
        for title, mode in (("不保存测量图片", "disabled"), ("局部 JPEG · 质量 95", "crop_jpeg"), ("完整无损 PNG", "full_png")):
            action = evidence_menu.addAction(title)
            action.setCheckable(True)
            group.addAction(action)
            action.triggered.connect(lambda checked=False, name=mode: self._guard(lambda: self._configure(evidence_options={"version": 1, "mode": name})))
            self.evidence_mode_actions[mode] = action
        self.evidence_button.setMenu(evidence_menu)
        # Reuse the existing outcome row, rather than consuming category space.
        latest_layout = self.latest_value.parentWidget().layout()
        def find_layout(layout):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.layout() is not None:
                    result = find_layout(item.layout())
                    if result is not None:
                        return result
                elif item.widget() is self.measurement_outcome:
                    return layout
            return None
        latest_layout = find_layout(latest_layout)
        latest_layout.removeWidget(self.measurement_outcome)
        outcome = QHBoxLayout()
        outcome.setSpacing(2)
        outcome.addWidget(self.measurement_outcome, 1)
        outcome.addWidget(self.evidence_button)
        latest_layout.addLayout(outcome)
        self.latest_value.setCursor(Qt.CursorShape.PointingHandCursor)
        self.latest_value.setToolTip("点击查看上一条已接受测量的证据")
        self.latest_value.installEventFilter(self)
        self.canvas.candidateClicked.connect(self._candidate_clicked)
        self.canvas.candidateAmbiguous.connect(self._candidate_ambiguous)
        self.canvas.candidateHovered.connect(self._candidate_hovered)
        self.candidate_table.selectionModel().currentRowChanged.connect(self._candidate_row_selected)
        self.candidate_model.dataChanged.connect(lambda *_: self.canvas.update())
        self.candidate_model.exclusionChanged.connect(self._candidate_excluded)
        self.controller.candidateStateChanged.connect(self._refresh_assistance_state)
        self.controller.recordAccepted.connect(self._evidence_record_accepted)
        self.controller.recordsAccepted.connect(lambda records: self._evidence_record_accepted(records[-1]) if records else None)
        self.measure_mode.currentIndexChanged.connect(self._candidate_mode_changed)
        self.input_mode.currentIndexChanged.connect(self._candidate_mode_changed)
        self.auto_analysis.toggled.connect(self._assistance_preferences_changed)
        self.algorithm.currentIndexChanged.connect(self._assistance_preferences_changed)
        self.show_candidates_action.toggled.connect(self._assistance_preferences_changed)
        self.candidate_opacity.valueChanged.connect(self._assistance_preferences_changed)
        self.selection_retention_seconds.valueChanged.connect(self._assistance_preferences_changed)
        try:
            preferences = json.loads(self._view_preferences_path.read_text(encoding="utf-8")).get("assistance", {})
        except (OSError, ValueError, TypeError, AttributeError):
            preferences = {}
        # A second trial can be constructed before the 250ms preference write.
        # Use the host's shared current choice, not a stale disk snapshot.
        preferences = getattr(self.host, "_trial_assistance_preferences", preferences)
        self._apply_assistance_preferences(preferences)
        self._assistance_loading = False
        self._refresh_assistance_state()

    def _assistance_preferences(self):
        return {"version": 3, "automatic": self.auto_analysis.isChecked(),
                "algorithm": self.algorithm.currentData(), "visible": self.show_candidates_action.isChecked(),
                "opacity": self.candidate_opacity.value(), "input_mode": self.input_mode.currentData(),
                "manual_algorithm": self.measure_mode.currentData(),
                "retention_seconds": self.selection_retention_seconds.value()}

    def _apply_assistance_preferences(self, preferences):
        loading, self._assistance_loading = self._assistance_loading, True
        try:
            self.auto_analysis.setChecked(preferences.get("automatic") is True)
            self.show_candidates_action.setChecked(preferences.get("visible", True) is not False)
            try:
                opacity = int(preferences.get("opacity", 60))
            except (TypeError, ValueError, OverflowError):
                opacity = 60
            self.candidate_opacity.setValue(max(20, min(100, opacity)))
            try:
                retention = float(preferences.get("retention_seconds", 1.))
                if not math.isfinite(retention):
                    retention = 1.
            except (TypeError, ValueError, OverflowError):
                retention = 1.
            self.selection_retention_seconds.setValue(max(0., min(10., retention)))
            self.algorithm.setCurrentIndex(max(0, self.algorithm.findData(preferences.get("algorithm", "traditional"))))
            # V1 combined the manual algorithm with candidate-only input. The
            # upgrade opts into mixed input without changing the manual algorithm.
            modern = preferences.get("version") in (2, 3)
            manual = preferences.get("manual_algorithm", "assisted") if modern else preferences.get("input_mode", "assisted")
            interaction = preferences.get("input_mode", "mixed") if modern else "mixed"
            self.measure_mode.setCurrentIndex(max(0, self.measure_mode.findData(manual)))
            self.input_mode.setCurrentIndex(max(0, self.input_mode.findData(interaction)))
            self._apply_assistance_controls()
        finally:
            self._assistance_loading = loading

    def _apply_assistance_controls(self):
        self.canvas.set_candidate_presentation(self.show_candidates_action.isChecked(), self.candidate_opacity.value())
        self.controller.selection_retention.set_duration(self.selection_retention_seconds.value())
        interaction = self.input_mode.currentData()
        picking = interaction == "candidate"
        self.canvas.set_input_mode(interaction)
        self.canvas.set_manual_gesture("drag" if self.gesture.currentIndex() else "click")
        self.gesture.setEnabled(interaction == "manual")
        self.gesture.setToolTip("混合模式同时支持两点和拖线。" if interaction == "mixed" else "仅手工模式使用此手势。")
        self.classification_mode.setEnabled(not picking)
        self.classification_mode.setToolTip("候选点选始终按点击时的当前类别计入；手工模式保留先选／后选。" if picking else "先选类别用于下一根；后选修改最近测量。")
        self.candidate_opacity_label.setText(f"{self.candidate_opacity.value()}%")
        for algorithm, action in self.algorithm_actions.items():
            action.setChecked(algorithm == self.algorithm.currentData())
        self.controller.configure_automatic(enabled=self.auto_analysis.isChecked(), algorithm=self.algorithm.currentData())

    def _candidate_mode_changed(self, *_):
        if hasattr(self, "auto_analysis") and not self._assistance_loading:
            self.canvas.cancel_pointer_gesture()
            self.controller.cancel_draft()
            self.controller.cancel_candidate_submissions()
            self._assistance_preferences_changed()

    def _assistance_preferences_changed(self, *_):
        if self._assistance_loading:
            return
        self._apply_assistance_controls()
        preferences = self._assistance_preferences()
        self.host._trial_assistance_preferences = preferences
        for workspace in getattr(self.host, "_live_experiment_workspaces", {}).values():
            if workspace is not self and hasattr(workspace, "auto_analysis") and not workspace._closed:
                workspace._apply_assistance_preferences(preferences)
        self._view_preferences_timer.start()
        self._refresh_assistance_state()

    def _refresh_assistance_state(self):
        if getattr(self, "_assistance_refreshing", False):
            return
        self._assistance_refreshing = True
        try:
            self._update_assistance_state()
        finally:
            self._assistance_refreshing = False

    def _update_assistance_state(self):
        if not hasattr(self, "auto_analysis") or self._closed:
            return
        controller = self.controller
        frame = controller.latest_frame
        prepared = bool(controller.calibration and controller.calibration.isotropic
            and self.session.active_operator_id in self.session.operators and self.material.count())
        controller.configure_automatic(active=self.isVisible() and bool(frame and frame.live), prepared=prepared)
        context = controller._automatic_context
        running = bool(context and not context.get("completed", False))
        available = sum(item.get("state", "available") == "available" for item in controller.candidates)
        pending = controller.candidate_pending_count
        key = (running, available, pending, controller.candidate_overlay_ready,
               bool(frame), controller.state, bool(controller.calibration),
               self.session.effective_evidence_options["mode"])
        if key != self._assistance_state_key:
            self._assistance_state_key = key
            self.analyze_button.setText("分析中…" if running else "刷新候选" if controller.candidates else "全图分析")
            self.analyze_button.setEnabled(bool(frame and controller.calibration and controller.state == "ready"))
            self.analyze_button.setToolTip("重新分析固定源帧，替换未接受候选；已接受记录保留。" + (f" 正在核验 {pending} 条，刷新将在核验后执行。" if pending else ""))
            self.candidate_display_button.setText(f"候选 {available}")
            self.workflow_tabs.setTabText(1, f"审核 {available}" if available else "审核")
            mode = self.session.effective_evidence_options["mode"]
            self.evidence_button.setToolTip("浏览当前试验全部操作者的图片；箭头查看上一条、固定小窗或选择存档模式。" + {"disabled": "当前不保存测量图片。", "crop_jpeg": "当前保存局部 JPEG，质量 95；有透明信息时使用 PNG。", "full_png": "当前保存完整无损 PNG。"}[mode])
            for name, action in self.evidence_mode_actions.items():
                action.setChecked(name == mode)
        self._queue_evidence_refresh()

    def _evidence_record_accepted(self, record):
        # The latest-evidence action is immediately usable. Decoding and
        # repainting the optional review/gallery leave the acceptance path.
        self.evidence_review.accepted(record, refresh=False)
        self._queue_evidence_refresh()

    def _queue_evidence_refresh(self):
        if not hasattr(self, "_evidence_refresh_timer"):
            self._evidence_refresh_timer = QTimer(self)
            self._evidence_refresh_timer.setSingleShot(True)
            self._evidence_refresh_timer.timeout.connect(self._flush_evidence_refresh)
        if not self._evidence_refresh_timer.isActive():
            self._evidence_refresh_timer.start(0)

    def _flush_evidence_refresh(self):
        if self._closed:
            return
        self.evidence_review.refresh()
        gallery = getattr(self.host, "_evidence_gallery", None)
        if gallery is not None and gallery.workspace is self:
            gallery.refresh()

    def _analyze_candidates(self):
        self.controller.analyze(self.algorithm.currentData())
        self._refresh_assistance_state()

    def _cancel_candidate_analysis(self):
        self.controller.discard_candidates()
        self._refresh_assistance_state()

    def _cancel_trial_gesture(self):
        self.canvas.cancel_pointer_gesture()
        self.controller.cancel_draft()
        self.controller.cancel_candidate_submissions()

    def _candidate_index(self, candidate_id):
        return next((i for i, item in enumerate(self.controller.candidates)
                     if item.get("candidate_id") == candidate_id), None)

    def _candidate_excluded(self, candidate_id, excluded):
        if candidate_id is not None:
            self._guard(lambda: self.controller.set_candidate_excluded(candidate_id, excluded))
        self._refresh_assistance_state()

    def _candidate_clicked(self, candidate_id):
        index = self._candidate_index(candidate_id)
        if index is None:
            return
        self.controller.candidate_interaction_started()
        material = self.material.currentData()
        if material is None:
            self._refresh_measurement_readiness()
            self.statusBar().showMessage("请先选择当前纤维类别，再点击候选。", 8000)
            self.choose_categories_button.setFocus()
            return
        self.candidate_model.setData(self.candidate_model.index(index, 3), material)
        try:
            self.controller.submit_candidate(candidate_id, material)
        except (ValueError, RuntimeError) as exc:
            self.statusBar().showMessage(str(exc), 10000)
            self._candidate_hovered(candidate_id)
            self.workflow_tabs.setCurrentIndex(1)
        self._refresh_assistance_state()

    def _candidate_ambiguous(self, candidate_ids):
        menu = QMenu(self.canvas)
        for candidate_id in candidate_ids:
            index = self._candidate_index(candidate_id)
            if index is None:
                continue
            item = self.controller.candidates[index]
            scale = self.controller.calibration.x_um if self.controller.calibration else 1.
            action = menu.addAction(f"C{item.get('ordinal', index + 1)} · {item['width_px'] * scale:.3f} μm")
            action.triggered.connect(lambda checked=False, key=candidate_id: self._candidate_clicked(key))
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        menu.popup(QCursor.pos())

    def _candidate_hovered(self, candidate_id):
        if self._candidate_linking or candidate_id is None:
            return
        row = self._candidate_index(candidate_id)
        if row is None:
            return
        self._candidate_linking = True
        try:
            index = self.candidate_model.index(row, 1)
            self.candidate_table.selectionModel().setCurrentIndex(index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
            self.candidate_table.scrollTo(index, QAbstractItemView.ScrollHint.EnsureVisible)
        finally:
            self._candidate_linking = False

    def _candidate_row_selected(self, current, previous):
        if self._candidate_linking:
            return
        item = self.candidate_model.candidates[current.row()] if current.isValid() else {}
        self.canvas.set_selected_candidate(item.get("candidate_id"))

    def _show_latest_evidence(self):
        key = self.evidence_review.latest_record_id
        record = self.session.measurements.get(key)
        if record is None:
            raise ValueError("尚无已接受的测量证据。")
        self.evidence_review.show_record(record, latest=True)

    def _pin_latest_evidence(self):
        self._show_latest_evidence()
        self.evidence_review.dialog.pin.setChecked(True)

    def _show_evidence_gallery(self):
        from fdm.ui.measurement_evidence_gallery import EvidenceGalleryController
        gallery = getattr(self.host, "_evidence_gallery", None)
        if gallery is None:
            gallery = EvidenceGalleryController(self.host)
            self.host._evidence_gallery = gallery
        gallery.open(self)

    def _export_assistance_diagnostics(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出离线性能诊断", "试验性能诊断.json", "JSON (*.json)")
        if path:
            from fdm.atomic_io import atomic_write_text
            from fdm.version import __version__
            import platform
            data = {"version": __version__, "platform": platform.platform(),
                "real_hardware_validation": False, "preferences": self._assistance_preferences(),
                "frame_size": self.controller.latest_frame.size if self.controller.latest_frame else None,
                "performance": getattr(self.controller, "performance_diagnostics", lambda: {})()}
            data["performance_schema"] = self.controller.performance_diagnostics_schema()
            data["frame_sampling"] = self.controller.performance_frame_context()
            atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            self.statusBar().showMessage("已导出当前运行诊断；该文件不等同于真实设备验收。", 8000)
