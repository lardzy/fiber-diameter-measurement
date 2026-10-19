"""Third-round behavior through actual navigation and domain-owned records."""
from pathlib import Path
import sys
import threading

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/validation"))
from validate_workflow_redesign import WorkflowHarness, settle
from fdm.live_experiment import default_materials
from fdm.ui.live_experiment_workspace import _new_trial_session


@pytest.fixture
def harness(tmp_path):
    with WorkflowHarness(tmp_path / "profile", size=(1600, 900)) as h:
        h.enter()
        yield h


def configure_materials(page):
    for direction in ("warp", "weft"):
        page.session.set_materials(default_materials(), direction=direction)
    page._refresh_materials()
    page._changed()


def test_new_content_does_not_preselect_materials_but_diameter_keeps_defaults():
    content, diameter = _new_trial_session("content"), _new_trial_session("diameter")
    assert all(not items for op in content.operators.values() for items in op.materials.values())
    assert all(len(items) == 2 for op in diameter.operators.values() for items in op.materials.values())


def test_empty_entry_return_and_preflight_do_not_dirty_project(harness):
    page = harness.select(1)
    snapshot = page.session.to_dict()
    assert page.material.count() == 0 and page.empty_categories.isVisible()
    page.ruler_choice.setCurrentIndex(1)
    page.ruler_choice.activated.emit(1)
    page._press(QPointF(100, 100))
    assert page.controller.draft is None
    assert "类别" in page.statusBar().currentMessage()
    assert page.session.to_dict() == snapshot
    assert not harness.window.project.live_experiments
    starts = harness.backend.starts
    page._return_to_capture()
    settle()
    assert harness.hub.pages.currentWidget() is harness.hub.capture_page
    assert harness.backend.starts == starts
    assert not harness.window.project.live_experiments
    assert not harness.modal_messages


def test_new_button_preserves_existing_trial_and_creates_empty_draft(harness):
    page = harness.select(1)
    configure_materials(page)
    record = page.session.add_measurement(20, "wool", field_id=page.controller.field_id)
    page._changed()
    previous = page.session
    page._new_experiment()
    settle()
    fresh = harness.hub.pages.currentWidget()
    assert fresh is not page and fresh.session is not previous
    assert fresh.material.count() == 0
    assert previous in harness.window.project.live_experiments
    assert record.id in previous.measurements
    assert fresh.session not in harness.window.project.live_experiments
    assert harness.hub.navigation.currentIndex() == 1


def test_switch_new_path_and_cancel_category_selection_leave_empty_draft(harness, monkeypatch):
    from fdm.ui import live_experiment_workspace as ui
    page = harness.select(1)
    previous = page.session
    monkeypatch.setattr(ui.QInputDialog, "getItem", lambda *_: ("＋ 新建试验", True))
    QTest.mouseClick(page.switch_trial_button, Qt.MouseButton.LeftButton)
    settle()
    page = harness.hub.pages.currentWidget()
    assert page.session is not previous and page.material.count() == 0
    before = page.session.to_dict()
    dialogs = []
    def cancel_categories():
        dialog = QApplication.activeModalWidget()
        dialogs.append(dialog.windowTitle())
        assert isinstance(dialog, QDialog)
        dialog.reject()
    QTimer.singleShot(0, cancel_categories)
    QTest.mouseClick(page.choose_categories_button, Qt.MouseButton.LeftButton)
    assert dialogs and "类别" in dialogs[0]
    assert page.session.to_dict() == before
    assert not harness.window.project.live_experiments


def test_return_with_unaccepted_results_uses_real_cancel_and_discard_dialog(harness):
    page = harness.select(1)
    page.controller.candidates = [{"width_px": 20, "start": [30, 30], "end": [30, 50]}]
    for leave in (False, True):
        def respond(leave=leave):
            dialog = QApplication.activeModalWidget()
            assert isinstance(dialog, QMessageBox)
            button = next(button for button in dialog.buttons()
                          if ("放弃" if leave else "留在") in button.text())
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        QTimer.singleShot(0, respond)
        QTest.mouseClick(page.return_capture_button, Qt.MouseButton.LeftButton)
        settle()
        assert (harness.hub.pages.currentWidget() is harness.hub.capture_page) is leave
        assert bool(page.controller.candidates) is not leave
    assert not harness.window.project.live_experiments


def test_navigation_can_cancel_discard_of_unaccepted_work(harness, monkeypatch):
    page = harness.select(1)
    monkeypatch.setattr(page, "_confirm_trial_navigation", lambda: False)
    page._return_to_capture()
    assert harness.hub.pages.currentWidget() is page
    assert harness.hub.navigation.currentIndex() == 1
    assert not harness.modal_messages


def test_metadata_expansion_is_application_preference(harness):
    page = harness.select(1)
    before = page.session.to_dict()
    page.sample_details_toggle.setChecked(True)
    page._persist_view_preferences()
    page.sample_details_toggle.setChecked(False)
    page._restore_view_preferences()
    assert page.sample_details_toggle.isChecked()
    assert page.sample_details.isVisible()
    assert page.session.to_dict() == before
    assert not harness.window.project.live_experiments


def test_navigation_retains_page_when_pending_edits_cannot_commit(harness, monkeypatch):
    page = harness.select(1)
    def rejected():
        raise ValueError("组距设置尚未应用，请等待或取消重算后保存。")
    with monkeypatch.context() as patch:
        patch.setattr(page, "commit_pending_edits", rejected)
        QTest.mouseClick(harness.hub.navigation, Qt.MouseButton.LeftButton,
                         pos=harness.hub.navigation.tabRect(0).center())
        assert harness.hub.pages.currentWidget() is page
        assert harness.hub.navigation.currentIndex() == 1
        assert "组距设置尚未应用" in page.statusBar().currentMessage()
        assert not harness.slot_errors


def test_fixed_export_flushes_metadata_and_takes_full_snapshot(harness, tmp_path, monkeypatch):
    import fdm.ui.trial_report_panel as reports
    page = harness.select(1)
    configure_materials(page)
    for name in ("甲", "乙", "丙"):
        if name == "甲":
            page.session.rename_operator(name)
        else:
            page.session.add_operator(name)
        page.session.add_measurement(20, "wool", field_id="export-test")
    page._refresh_operators()
    page._changed()
    page.sample_number.setText("0000123")
    page.sample.setText("尚未失焦的名称")
    template = tmp_path / "template.xlt"
    template.write_bytes(b"test template; COM is mocked")
    destination = tmp_path / "report.xls"
    monkeypatch.setattr(page.report_panel, "path", lambda **_: template)
    monkeypatch.setattr(reports.QFileDialog, "getSaveFileName", lambda *_: (str(destination), ""))
    captured = []
    capture = reports.ExportSnapshot.capture
    monkeypatch.setattr(reports.ExportSnapshot, "capture", lambda session: captured.append(capture(session)) or captured[-1])
    complete = threading.Event()
    def export(request, **_):
        assert complete.wait(10)
        destination.write_bytes(b"test successful report")
        return {"operation": "export", "destination": str(destination), "method": "测试方法", "status": "ready"}
    monkeypatch.setattr(reports, "run_excel_request", export)
    try:
        QTest.mouseClick(page.export_button, Qt.MouseButton.LeftButton)
        settle()
        assert not page.export_button.isEnabled()
        snapshot = captured[0]
        assert snapshot.number == "0000123" and snapshot.name == "尚未失焦的名称"
        assert len(snapshot.operators) == 3 and len(snapshot.measurements) == 3
        page.session.add_measurement(30, "wool", field_id="after-snapshot")
        assert len(snapshot.measurements) == 3
    finally:
        complete.set()
    page.report_panel.future.result(timeout=10)
    page.report_panel.poll()
    assert page.export_button.isEnabled() and page.open_report_button.isEnabled()
    assert "报告已生成" in page.report_status_label.toolTip()
