"""Editing goals preserves inheritance, stable overrides and no-op history."""
from dataclasses import replace

import pytest
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from fdm.live_experiment.models import ExperimentSession, default_materials
from fdm.live_experiment.reminders import ReminderPolicy
from fdm.ui.quantity_targets import QuantityTargetDialog, QuantityTargets


def make_session(mode="content", policy=ReminderPolicy()):
    session = ExperimentSession(workspace_options={"mode": mode}, reminder_policy=policy)
    session.add_operator("测试员")
    session.set_materials(default_materials())
    session.establish_history_baseline()
    return session


@pytest.mark.parametrize("mode", ["content", "diameter"])
@pytest.mark.parametrize("policy", [None, ReminderPolicy(),
    ReminderPolicy(material_targets={"wool": 21, "hidden-material": 88})])
def test_accept_unchanged_dialog_preserves_exact_configuration_and_revision(mode, policy):
    session = make_session(mode, policy)
    before = session.to_dict()
    dialog = QuantityTargetDialog(session)
    try:
        dialog.accept()
        session.configure(**dialog.changes())
        assert session.to_dict() == before
        assert not session.can_undo
    finally:
        dialog.deleteLater()


def test_default_edit_updates_inherited_materials_but_preserves_all_explicit_overrides():
    session = make_session(policy=ReminderPolicy(material_targets={"wool": 21, "hidden-material": 88}))
    dialog = QuantityTargetDialog(session)
    try:
        assert dialog.material_editors["wool"].value() == 21
        inherited = dialog.material_editors["cashmere"]
        assert inherited.value() == 0 and inherited.text() == "使用默认（300 根）"
        dialog.default.setValue(500)
        assert inherited.text() == "使用默认（500 根）"
        policy = dialog.changes()["reminder_policy"]
        assert policy.material_target("cashmere") == 500
        assert dict(policy.material_targets) == {"wool": 21, "hidden-material": 88}
        session.configure(reminder_policy=policy)
        restored = ExperimentSession.from_dict(session.to_dict())
        assert restored.reminder_policy == policy
    finally:
        dialog.deleteLater()


def test_explicit_override_can_be_restored_to_default_with_real_keyboard_input():
    original = ReminderPolicy(material_targets={"wool": 21, "hidden-material": 88})
    session = make_session(policy=original)
    dialog = QuantityTargetDialog(session)
    try:
        dialog.show()
        QApplication.processEvents()
        editor = dialog.material_editors["wool"]
        editor.setFocus()
        editor.selectAll()
        QTest.keyClicks(editor, "0")
        QTest.keyClick(editor, Qt.Key.Key_Tab)
        assert editor.value() == 0 and editor.text() == "使用默认（300 根）"
        policy = dialog.changes()["reminder_policy"]
        assert dict(policy.material_targets) == {"hidden-material": 88}
        assert policy.material_target("wool") == 300
        session.configure(reminder_policy=policy)
        session.undo()
        assert session.reminder_policy == original
    finally:
        dialog.close()
        dialog.deleteLater()


def test_equal_numeric_override_stays_explicit_when_default_later_changes():
    session = make_session(policy=ReminderPolicy(material_targets={"wool": 300}))
    dialog = QuantityTargetDialog(session)
    try:
        dialog.default.setValue(500)
        policy = dialog.changes()["reminder_policy"]
        assert policy.material_target("wool") == 300
        assert policy.material_target("cashmere") == 500
        assert dict(policy.material_targets) == {"wool": 300}
    finally:
        dialog.deleteLater()


def test_visible_settings_button_applies_one_change_and_refreshes_progress():
    session = make_session()
    widget = QuantityTargets(session, session.configure)
    widget.show()
    before = session.revision
    def accept_edit():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QuantityTargetDialog)
        dialog.default.setValue(500)
        dialog.material_editors["wool"].setValue(17)
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
    QTimer.singleShot(0, accept_edit)
    try:
        widget.settings_button.click()
        assert session.revision == before + 1
        assert session.reminder_policy.content_measured_target == 500
        assert dict(session.reminder_policy.material_targets) == {"wool": 17}
        assert "0 / 17" in widget.progress.toolTip() and "0 / 500" in widget.progress.toolTip()
    finally:
        widget.close()
        widget.deleteLater()


def test_legacy_opt_in_keeps_legacy_targets_without_creating_material_overrides():
    session = make_session(policy=None)
    session.configure(measured_reminder=17, total_reminder=29)
    before = session.to_dict()
    dialog = QuantityTargetDialog(session)
    try:
        dialog.enable_new.setChecked(True)
        policy = dialog.changes()["reminder_policy"]
        assert policy == replace(ReminderPolicy(), content_measured_target=17,
                                content_total_target=29, diameter_measured_target=17)
        dialog.reject()
        assert session.to_dict() == before
    finally:
        dialog.deleteLater()
