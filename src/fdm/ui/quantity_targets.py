"""Visible trial goals and an explicit opt-in editor for legacy projects."""
from dataclasses import replace

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
    QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget, QHBoxLayout, QToolButton, QSizePolicy)

from fdm.live_experiment.reminders import ReminderPolicy, reminder_progress, effective_content_total_target
from fdm.live_experiment.trial_defaults import save_reminder_defaults
from fdm.ui.icons import themed_icon
from fdm.ui.widgets import NoWheelSpinBox


class QuantityTargets(QWidget):
    def __init__(self, session, configure, parent=None):
        super().__init__(parent)
        self.session, self.configure = session, configure
        self.setObjectName("quantityTargets")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.progress = QLabel(self)
        self.progress.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.progress.setMinimumWidth(0)
        self.progress.setAccessibleName("数量目标进度")
        layout.addWidget(self.progress, 1)
        self.settings_button = QToolButton(self)
        self.settings_button.setFixedSize(26, 26)
        self.settings_button.setStyleSheet("QToolButton { min-height: 22px; padding: 0; }")
        self.settings_button.setAccessibleName("数量目标")
        self.settings_button.setToolTip("设置数量目标；每类进度显示在类别列表中。")
        self.settings_button.setIcon(themed_icon("count"))
        self.settings_button.clicked.connect(self.edit_targets)
        layout.addWidget(self.settings_button)
        self.refresh()

    def refresh(self):
        if self.session.active_operator_id not in self.session.operators:
            self.progress.setText("选择操作者后显示数量目标")
            return
        rows = reminder_progress(self.session)
        total = next((row for row in rows if row.kind in ("total", "diameter_measured")), None)
        if total is not None:
            title = "有效测量" if total.kind == "diameter_measured" else "本方向总根数"
            self.progress.setText(f"{'✓ ' if total.reached else ''}{title} {total.count} / {total.target}")
        else:
            self.progress.setText("配置类别后显示数量目标")
        details = []
        for row in rows:
            title = (row.material_name + "实测" if row.kind == "measured" else
                     row.material_name + "补计／计数" if row.kind == "counted" else
                     "本方向总根数（实测＋补计）" if row.kind == "total" else "当前操作者有效测量")
            details.append(f"{'✓ ' if row.reached else ''}{title} {row.count} / {row.target}")
        if self.session.reminder_policy is None:
            details.append("沿用旧试验数量规则 · 可在数量目标中切换")
        self.progress.setToolTip("\n".join(details))

    def edit_targets(self):
        dialog = QuantityTargetDialog(self.session, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.configure(**dialog.changes())
            self.refresh()


class QuantityTargetDialog(QDialog):
    def __init__(self, session, parent=None, *, defaults_file=None):
        super().__init__(parent)
        self.session = session
        self.defaults_file = defaults_file
        self.setWindowTitle("数量目标")
        self.resize(450, 460)
        layout = QVBoxLayout(self)
        note = QLabel("达到目标时提示一次，可以继续测量。每类填 0 可恢复类别参数或试验默认目标。"
                      "目标是试验配置；默认值不代表标准强制要求。", self)
        note.setWordWrap(True)
        layout.addWidget(note)
        self.enable_new = QCheckBox("启用按试验类型的数量目标", self)
        legacy = session.reminder_policy is None
        self.enable_new.setChecked(not legacy)
        self.enable_new.setVisible(legacy)
        layout.addWidget(self.enable_new)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        panel = QWidget(scroll)
        form = QFormLayout(panel)
        self.policy = session.reminder_policy or ReminderPolicy(
            content_measured_target=session.measured_reminder,
            content_total_target=session.total_reminder,
            diameter_measured_target=session.measured_reminder)
        self.editors = {}

        def spin(key, label, value, *, inherit=False):
            widget = NoWheelSpinBox(panel)
            widget.setRange(0 if inherit else 1, max(100000000, value))
            widget.setValue(value)
            widget.setAccessibleName(label)
            if inherit:
                widget.setToolTip("输入 0 恢复使用默认目标；输入正整数为此类别单独设置目标。")
            form.addRow(label, widget)
            self.editors[key] = widget
            return widget

        self.legacy_measured = spin("legacy_measured", "原每类实测目标", session.measured_reminder)
        self.legacy_total = spin("legacy_total", "原本方向总目标", session.total_reminder)
        self.diameter = spin("diameter", "当前操作者有效测量", self.policy.diameter_measured_target)
        self.default = spin("default", "未单独设置类别的目标", self.policy.content_measured_target)
        self.total = spin("total", "本方向总根数（实测＋补计）", self.policy.content_total_target)
        self._shared_total = getattr(session, "_content_total_target", None)
        if self._shared_total is not None:
            for editor in (self.total, self.legacy_total):
                editor.setMaximum(max(editor.maximum(), effective_content_total_target(session)))
                editor.setValue(effective_content_total_target(session))
                editor.setEnabled(False)
                editor.setToolTip("所有含量试验共用。请在设置 → 采集与试验中调整总根数提醒。")
            shared_note = QLabel("总根数提醒已统一到设置 → 采集与试验，修改后作用于所有含量试验。", panel)
            shared_note.setWordWrap(True)
            form.addRow(shared_note)
        materials = {m.id: m for op in session.operators.values() for group in op.materials.values() for m in group}
        overrides = dict(self.policy.material_targets)
        self.material_editors = {key: spin("material:" + key, m.name + "实测", overrides.get(key, 0), inherit=True)
                                 for key, m in materials.items()}

        def refresh_inherited_text():
            for key, editor in self.material_editors.items():
                material_default = materials[key].measured_target
                if material_default is not None:
                    editor.setSpecialValueText(f"使用类别参数（{material_default} 根）")
                else:
                    editor.setSpecialValueText(f"使用默认（{self.default.value()} 根）")

        self.default.valueChanged.connect(refresh_inherited_text)
        refresh_inherited_text()
        scroll.setWidget(panel)
        layout.addWidget(scroll, 1)

        def update_rows():
            new = self.enable_new.isChecked()
            diameter = session.workspace_options["mode"] == "diameter"
            for widget in (self.legacy_measured, self.legacy_total):
                form.setRowVisible(widget, not new)
            form.setRowVisible(self.diameter, new and diameter)
            for widget in (self.default, self.total, *self.material_editors.values()):
                form.setRowVisible(widget, new and not diameter)

        self.enable_new.toggled.connect(update_rows)
        update_rows()
        mode_label = "直径" if session.workspace_options["mode"] == "diameter" else "含量"
        self.save_default = QPushButton(f"设为新{mode_label}试验默认", self)
        self.save_default.clicked.connect(self._save_default)
        layout.addWidget(self.save_default)
        self.default_status = QLabel(self)
        self.default_status.setWordWrap(True)
        layout.addWidget(self.default_status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def changes(self):
        if not self.enable_new.isChecked():
            return dict(measured_reminder=self.legacy_measured.value(),
                        total_reminder=self.session.total_reminder if self._shared_total is not None else self.legacy_total.value())
        if self.session.workspace_options["mode"] == "diameter":
            policy = replace(self.policy, diameter_measured_target=self.diameter.value())
        else:
            targets = dict(self.policy.material_targets)
            for key, editor in self.material_editors.items():
                if editor.value() == 0:
                    targets.pop(key, None)
                else:
                    targets[key] = editor.value()
            policy = replace(self.policy, content_measured_target=self.default.value(),
                             content_total_target=self.policy.content_total_target if self._shared_total is not None else self.total.value(),
                             material_targets=targets)
        return {"reminder_policy": policy}

    def _save_default(self):
        mode = self.session.workspace_options["mode"]
        changes = self.changes()
        policy = changes.get("reminder_policy") or replace(self.policy,
            content_measured_target=self.legacy_measured.value(),
            content_total_target=self.policy.content_total_target if self._shared_total is not None else self.legacy_total.value(),
            diameter_measured_target=self.legacy_measured.value())
        if self._shared_total is not None:
            # Saving other defaults must not restore the old trial snapshot's
            # total over the application's shared value.
            policy = replace(policy, content_total_target=effective_content_total_target(self.session))
        try:
            save_reminder_defaults(mode, policy, path=self.defaults_file)
            self.default_status.setText("已保存为新试验默认，其他已有试验不改变。")
        except (OSError, ValueError) as error:
            self.default_status.setText(f"默认值未保存：{error}")
