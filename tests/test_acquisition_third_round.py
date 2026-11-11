"""Third-round behavior through actual navigation and domain-owned records."""
from pathlib import Path
import hashlib
import json
import sys
import threading
import time

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/validation"))
from validate_workflow_redesign import WorkflowHarness, settle, synthetic_frame
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
    monkeypatch.setattr(ui.QInputDialog, "getItem", lambda *_: ("＋ 新建并保留当前", True))
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
    assert dialogs == ["选择本次纤维"]
    assert page.session.to_dict() == before
    assert not harness.window.project.live_experiments


def test_finish_content_replaces_only_confirmed_sample_and_keeps_camera(harness, tmp_path):
    page = harness.select(1)
    configure_materials(page)
    page.sample_number.setText("A-12")
    page.sample.setText("待换样")
    page.commit_pending_edits()
    session = page.session
    session.add_measurement(20, "wool", field_id=page.controller.field_id)
    session.adjust_count("wool", 3)
    page._changed()
    others = [_new_trial_session("content"), _new_trial_session("diameter")]
    project = harness.window.project
    project.live_experiments.extend(others)
    before = [other.to_dict() for other in others]
    harness.window._mark_project_saved()
    starts, stops = harness.backend.starts, harness.backend.stops
    assert page.finish_trial_button.property("emphasis") == "warning"
    assert not hasattr(page, "return_capture_button") and not hasattr(page, "new_trial_button")
    for confirm in (False, True):
        def choose():
            dialog = QApplication.activeModalWidget()
            try:
                assert "将移除当前含量试验，并开始空白试验" in dialog.text()
                assert all(s in dialog.informativeText() for s in ("A-12", "待换样", "实测：1", "补计：3"))
                button = next(b for b in dialog.buttons() if b.text() == ("移除并开始" if confirm else "取消"))
                if confirm:
                    assert button.property("emphasis") == "destructive"
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            finally:
                if dialog.isVisible():
                    dialog.reject()
        QTimer.singleShot(0, choose)
        QTest.mouseClick(page.finish_trial_button, Qt.MouseButton.LeftButton)
        settle()
        if not confirm:
            assert harness.hub.pages.currentWidget() is page and session in project.live_experiments
            assert not harness.window._project_dirty()
    fresh = harness.hub.pages.currentWidget()
    assert fresh is not page and fresh.session.id != session.id
    assert project.live_experiments == others and before == [other.to_dict() for other in others]
    assert harness.window._project_dirty() and fresh.save_button.isEnabled()
    assert fresh.session not in project.live_experiments and not fresh.session.measurements
    assert not fresh.session.count_adjustments and not fresh.session.can_undo
    assert not fresh.session.sample_name and not fresh.session.sample_number
    assert not fresh.material.count() and fresh.workflow_tabs.currentIndex() == 0
    assert harness.hub.records.model.session is fresh.session
    assert harness.backend.starts == starts and harness.backend.stops == stops
    page._changed()  # A late old-page notification cannot re-register its sample.
    assert project.live_experiments == others
    harness.backend.tick()
    assert fresh.controller.latest_frame is not None
    path = tmp_path / "换样.fdmproj"
    assert harness.window.project_session_controller.save_project(str(path)).success
    assert [s["id"] for s in json.loads(path.read_text())["live_experiments"]] == [s.id for s in others]
    assert not harness.slot_errors


def test_last_trial_removal_is_saved_and_empty_draft_is_not_a_project_edit(harness, tmp_path, monkeypatch):
    page = harness.select(1)
    configure_materials(page)
    raw = tmp_path / "原图.png"
    assert synthetic_frame((120, 90)).save(str(raw))
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    record = page.session.add_measurement(20, "wool", evidence={
        "state": "saved", "path": str(raw), "sha256": digest, "frame_id": "1"})
    page._changed()
    report = tmp_path / "报告.csv"
    from fdm.live_experiment.exports import export_csv
    export_csv(page.session, report)
    report_digest = hashlib.sha256(report.read_bytes()).digest()
    path = tmp_path / "唯一试验.fdmproj"
    assert harness.window.project_session_controller.save_project(str(path)).success
    asset = Path(page.session.measurements[record.id].evidence["path"])
    assert asset != raw and asset.is_file()
    assert json.loads(path.read_text())["live_experiments"][0]["id"] == page.session.id
    monkeypatch.setattr(page, "_confirm_sample_removal", lambda: True)
    page._finish_content_trial()
    settle()
    fresh = harness.hub.pages.currentWidget()
    assert not harness.window.project.live_experiments and fresh.save_button.isEnabled()
    assert harness.window.project_session_controller.save_project(str(path)).success
    assert not json.loads(path.read_text()).get("live_experiments")
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p in (raw, asset))
    assert hashlib.sha256(report.read_bytes()).digest() == report_digest
    assert not harness.window._project_dirty()
    monkeypatch.setattr(fresh, "_confirm_sample_removal", lambda: True)
    fresh._finish_content_trial()
    settle()
    assert harness.hub.pages.currentWidget().session.id != fresh.session.id
    assert not harness.window._project_dirty()


def test_rejected_page_replacement_keeps_old_session_source_and_records(harness, monkeypatch):
    page = harness.select(1)
    configure_materials(page)
    record = page.session.add_measurement(21, "wool", field_id=page.controller.field_id)
    page._changed()
    project = harness.window.project
    before = page.session.to_dict()
    source = page.source
    harness.window._mark_project_saved()
    monkeypatch.setattr(page, "_confirm_sample_removal", lambda: True)
    monkeypatch.setattr(page, "prepare_close", lambda: False)
    page._finish_content_trial()
    settle()
    assert project.live_experiments == [page.session]
    assert page.session.to_dict() == before
    assert not harness.window._project_dirty() and not project.has_removed_live_trial
    assert harness.hub.pages.currentWidget() is page
    assert harness.hub.records.model.session is page.session
    assert page.source is source and not page.controller.hold_measurement_results
    assert record.id in page.session.measurements and not harness.slot_errors


@pytest.mark.parametrize("outcome", ["cancel", "saved", "failed"])
def test_sample_change_waits_for_evidence_without_mixing_results(harness, monkeypatch, outcome):
    page = harness.select(1)
    configure_materials(page)
    harness.backend.tick()
    controller = page.controller
    frame = controller.latest_frame
    store = controller.evidence
    release = threading.Event()
    writer = store._write
    if outcome == "failed":
        def write_error(*args, **kwargs):
            raise OSError("模拟原图写入失败")
        monkeypatch.setattr("fdm.live_experiment.evidence.write_png_stream", write_error)
    def delayed(task):
        if not release.wait(5):
            raise RuntimeError("test evidence barrier was not released")
        return writer(task)
    monkeypatch.setattr(store, "_write", delayed)
    evidence = store.reserve(frame, options={"mode": "full_png"})
    record = page.session.add_measurement(20, "wool", field_id=controller.field_id, evidence=dict(evidence))
    controller._evidence_records[evidence["evidence_id"]] = [record.id]
    page._changed()
    monkeypatch.setattr(page, "_confirm_sample_removal", lambda: True)
    waited = []
    def while_waiting():
        waiting = QApplication.activeModalWidget()
        waited.append(waiting.text() if isinstance(waiting, QMessageBox) else "")
        if outcome == "cancel" and isinstance(waiting, QMessageBox):
            QTest.mouseClick(waiting.defaultButton(), Qt.MouseButton.LeftButton)
        release.set()
    QTimer.singleShot(20, while_waiting)
    try:
        page._finish_content_trial()
    finally:
        release.set()
    assert waited == ["正在完成原图写入…"]
    if outcome == "saved":
        fresh = harness.hub.pages.currentWidget()
        assert fresh is not page and not fresh.session.measurements
        assert Path(evidence["path"]).is_file()
        assert page.session not in harness.window.project.live_experiments
    else:
        assert harness.hub.pages.currentWidget() is page
        assert record.id in page.session.measurements and page.session in harness.window.project.live_experiments
        assert not controller.hold_measurement_results
        if outcome == "failed":
            assert "模拟原图写入失败" in page.statusBar().currentMessage()
        # Cancellation returns before the original evidence writer finishes.
        # Let the ordinary completion reach the retained session before teardown.
        deadline = time.monotonic() + 3
        while store.pending and time.monotonic() < deadline:
            QTest.qWait(5)
        assert not store.pending
    assert not harness.slot_errors


def test_navigation_with_unaccepted_results_uses_real_cancel_and_discard_dialog(harness):
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
        QTest.mouseClick(harness.hub.navigation, Qt.MouseButton.LeftButton,
                         pos=harness.hub.navigation.tabRect(0).center())
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
    # The compact status now belongs to live Excel synchronization. The
    # independent export keeps its own completion and destination details.
    assert page.report_panel.last_report["destination"] == str(destination)
    assert "报告已生成" in page.statusBar().currentMessage()
