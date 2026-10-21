"""Incremental candidate presentation and compact evidence entry contracts."""
from unittest.mock import Mock

from PySide6.QtCore import QItemSelectionModel, QSize, Qt, QTimer
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionToolButton

from test_live_experiment_content_workspace import workspace
from test_trial_assistance_ui import candidates


def test_same_batch_changes_only_target_row_and_keeps_selection(workspace):
    ids = candidates(workspace)
    model = workspace.candidate_model
    model.setData(model.index(2, 3), "cashmere")
    selection = workspace.candidate_table.selectionModel()
    selection.setCurrentIndex(model.index(2, 3), QItemSelectionModel.SelectionFlag.ClearAndSelect)
    reset, changed = QSignalSpy(model.modelReset), QSignalSpy(model.dataChanged)
    workspace.controller.candidates[0]["state"] = "confirming"
    workspace.controller.candidatesChanged.emit(workspace.controller.candidates)
    assert reset.count() == 0
    assert changed.count() == 1
    assert changed.at(0)[0].row() == changed.at(0)[1].row() == 0
    assert selection.currentIndex() == model.index(2, 3)
    assert model.assignments == {2: "cashmere"}
    assert [item["candidate_id"] for item in model.candidates] == ids
    workspace.controller.candidatesChanged.emit(workspace.controller.candidates)
    assert changed.count() == 1


def test_new_batch_resets_model_but_same_batch_reclassification_keeps_rows(workspace):
    candidates(workspace)
    model = workspace.candidate_model
    reset = QSignalSpy(model.modelReset)
    new = [dict(item, candidate_id="replacement:" + str(i))
           for i, item in enumerate(workspace.controller.candidates)]
    model.set_candidates(new, model.scale, model.materials)
    assert reset.count() == 1
    assert model.assignments == {}
    changed = QSignalSpy(model.dataChanged)
    model.set_candidates(new, 2, model.materials)
    assert reset.count() == 1
    assert changed.count() == len(new)


def test_evidence_button_is_compact_icon_with_separate_menu(workspace, monkeypatch):
    from fdm.ui.live_experiment_view import apply_trial_theme
    apply_trial_theme(workspace)
    QApplication.processEvents()
    button = workspace.evidence_button
    assert button.text() == ""
    assert not button.icon().isNull()
    assert button.iconSize() == QSize(16, 16)
    assert button.size() == QSize(40, 22)
    assert button.accessibleName() == "证据相册及存档设置"
    assert "当前" in button.toolTip()
    option = QStyleOptionToolButton()
    button.initStyleOption(option)
    style = button.style()
    main = style.subControlRect(QStyle.ComplexControl.CC_ToolButton, option,
                                QStyle.SubControl.SC_ToolButton, button)
    arrow = style.subControlRect(QStyle.ComplexControl.CC_ToolButton, option,
                                 QStyle.SubControl.SC_ToolButtonMenu, button)
    assert main.width() >= 26 and arrow.width() >= 10
    assert arrow.left() >= 26
    opened = Mock()
    monkeypatch.setattr(workspace, "_show_evidence_gallery", opened)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=main.center())
    assert opened.call_count == 1
    menu_shown = QSignalSpy(button.menu().aboutToShow)
    QTimer.singleShot(30, button.menu().close)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=arrow.center())
    assert menu_shown.count() == 1 and opened.call_count == 1


def test_candidate_notifications_coalesce_secondary_views(workspace, monkeypatch):
    QApplication.processEvents()
    notify = Mock()
    monkeypatch.setattr(workspace, "_notify_records", notify)
    review = Mock()
    monkeypatch.setattr(workspace.evidence_review, "refresh", review)
    for value in (20, 21, 22):
        record = workspace.session.add_measurement(value, "wool",
            field_id=workspace.controller.field_id)
        workspace._record_accepted(record)
        workspace._changed()
        workspace._refresh_assistance_state()
        workspace._changed()
    assert workspace.latest_value.text() == "22.000 μm"
    assert len(workspace.session.measurements) == 3
    assert notify.call_count == 0 and review.call_count == 0
    QApplication.processEvents()
    assert notify.call_count == 1 and review.call_count == 1
