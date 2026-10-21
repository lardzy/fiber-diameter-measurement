"""Optical controls preserve interaction state and own bounded raw-frame work."""
from concurrent.futures import Future
from copy import deepcopy
import json
import time
import zipfile

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from fdm.live_experiment.frames import FrameHandle
from fdm.services.optical_config import ALGORITHM_CHOICES
from fdm.ui.optical_diagnostics import candidate_diagnostic_text
from test_live_experiment_content_workspace import workspace
from test_trial_assistance_ui import candidates


def open_dialog(view):
    view._show_optical_settings()
    dialog = view.optical_settings_dialog
    dialog._timer.stop()
    return dialog


def complete(dialog):
    deadline = time.monotonic() + 3
    while dialog._job is not None and not dialog._job[1].done() and time.monotonic() < deadline:
        time.sleep(.002)
    assert dialog._job is None or dialog._job[1].done()
    dialog._tick()


def blank(sequence, source="camera"):
    return FrameHandle.snapshot(np.full((120, 160, 3), 210, np.uint8),
                                source_id=source, sequence=sequence, live=True)


def test_registry_has_three_algorithms_and_optical_tools_share_existing_toolbar(workspace):
    assert [(workspace.algorithm.itemText(i), workspace.algorithm.itemData(i))
            for i in range(workspace.algorithm.count())] == list(ALGORITHM_CHOICES)
    assert workspace.algorithm.currentData() == "traditional"
    QApplication.processEvents()
    before = workspace.view_controls.height()
    actions = workspace.analyze_button.menu().actions()
    action = next(item for item in actions if item.text() == "普通光学设置…")
    action.trigger()
    QApplication.processEvents()
    assert workspace.optical_settings_dialog.isVisible()
    assert workspace.view_controls.height() == before


@pytest.mark.parametrize("version", [2, 3, 4])
def test_preference_upgrade_keeps_algorithm_interaction_manual_and_gesture(workspace, version):
    workspace.gesture.setCurrentIndex(1)
    before = workspace.session.to_dict()
    workspace._apply_assistance_preferences({"version": version, "algorithm": "multiscale",
        "input_mode": "manual", "manual_algorithm": "free", "automatic": True,
        "optical_analysis": {"imaging_mode": "double_edge"}})
    assert workspace.algorithm.currentData() == "multiscale"
    assert workspace.input_mode.currentData() == "manual"
    assert workspace.measure_mode.currentData() == "free"
    assert workspace.gesture.currentIndex() == 1
    assert workspace._assistance_preferences()["version"] == 4
    assert workspace.controller.optical_analysis_config["imaging_mode"] == "double_edge"
    assert workspace.session.to_dict() == before


def test_save_applies_only_next_analysis_and_explicit_refresh_is_separate(workspace, monkeypatch):
    candidates(workspace)
    controller = workspace.controller
    batch = controller._automatic_context
    controller.start((20., 20.))
    draft = controller.draft
    before = workspace.session.to_dict()
    analyses = []
    monkeypatch.setattr(workspace, "_analyze_candidates", lambda: analyses.append(workspace.algorithm.currentData()))
    dialog = open_dialog(workspace)
    dialog.imaging_mode.setCurrentIndex(dialog.imaging_mode.findData("double_edge"))
    dialog.save_button.click()
    assert controller.optical_analysis_config["imaging_mode"] == "double_edge"
    assert controller.draft is draft and controller._automatic_context is batch
    assert workspace.session.to_dict() == before
    assert analyses == []
    dialog.refresh_button.click()
    assert analyses == ["optical_adaptive"]


def test_quality_tooltips_do_not_change_candidate_availability(workspace):
    candidates(workspace)
    item = workspace.controller.candidates[0]
    item.update(structure="double_edge", warning_codes=["ambiguous_pair"],
        conflicts_with=["other"], quality_features={"edge_snr": 2.3, "width_cv": .25,
            "edge_parallel_degrees": 3.1, "valid_sections": 5, "total_sections": 7})
    workspace.controller.candidatesChanged.emit(workspace.controller.candidates)
    text = workspace.candidate_model.data(workspace.candidate_model.index(0, 2), Qt.ItemDataRole.ToolTipRole)
    assert "透亮双边" in text and "5 / 7" in text
    assert "不代表测量准确率" in text and "提示核对是否重复" in text
    assert workspace.candidate_model.available(0)
    workspace._candidate_hovered(item["candidate_id"])
    assert "透亮双边" in workspace.canvas.toolTip()
    workspace._candidate_hovered(None)
    assert workspace.canvas.toolTip() == ""
    assert "nan" not in candidate_diagnostic_text({"quality_features": {"edge_snr": float("nan")}})


def test_failed_flat_correction_is_explicit_without_changing_candidate_or_layout(workspace):
    candidates(workspace)
    context = workspace.controller._automatic_context
    context["diagnostics"] = {"flat_field": {"status": "not_applied", "applied": False,
        "reason": "context_mismatch", "message": "设备或成像设置已变化。"}}
    workspace._refresh_assistance_state()
    assert "校正未应用" in workspace.analyze_button.toolTip()
    assert workspace.controller.candidate_overlay_ready
    dialog = open_dialog(workspace)
    assert "校正未应用" in dialog.status.text()


def test_blank_capture_owns_five_distinct_frames_and_only_saves_reference(workspace, monkeypatch):
    dialog = open_dialog(workspace)
    current = [blank(0)]
    monkeypatch.setattr(dialog, "_raw_frame", lambda: current[0])
    before, used = workspace.session.to_dict(), workspace.controller.budget.used_bytes
    dialog._start_capture()
    dialog._collect_frame()
    dialog._collect_frame()
    assert len(dialog._capture_frames) == 1
    for sequence in range(1, 5):
        current[0] = blank(sequence)
        dialog._collect_frame()
    assert dialog._job is not None
    complete(dialog)
    assert dialog._config["flat_reference"]
    assert not dialog.flat_enabled.isChecked()
    assert workspace.controller.optical_analysis_config["flat_reference"] is None
    assert workspace.controller.budget.used_bytes == used
    assert workspace.session.to_dict() == before
    dialog._load_reference_preview()
    complete(dialog)
    assert dialog._preview_image is not None, dialog.status.text()
    assert dialog._preview_image.width() <= 500
    assert workspace.controller.budget.used_bytes == used


def test_capture_context_change_and_close_release_frame_leases(workspace, monkeypatch):
    dialog = open_dialog(workspace)
    current = [blank(0)]
    monkeypatch.setattr(dialog, "_raw_frame", lambda: current[0])
    used = workspace.controller.budget.used_bytes
    dialog._start_capture()
    dialog._collect_frame()
    current[0] = blank(1, source="other")
    dialog._collect_frame()
    assert dialog._capture_context is None
    assert workspace.controller.budget.used_bytes == used
    dialog._start_capture()
    dialog._collect_frame()
    dialog.close()
    assert dialog._capture_context is None
    assert workspace.controller.budget.used_bytes == used


def test_escape_mid_capture_releases_frames_and_stops_hidden_light_checks(workspace, monkeypatch):
    workspace.controller.start((20., 20.))
    draft = workspace.controller.draft
    dialog = open_dialog(workspace)
    monkeypatch.setattr(dialog, "_raw_frame", lambda: blank(0))
    used = workspace.controller.budget.used_bytes
    dialog._start_capture()
    dialog._collect_frame()
    assert len(dialog._capture_frames) == 1
    assert workspace.controller.budget.used_bytes > used
    dialog.tabs.setCurrentIndex(1)
    started = []
    monkeypatch.setattr(dialog, "_start_light", lambda frame: started.append(frame.frame_id))
    dialog._timer.start()
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    QApplication.processEvents()
    assert not dialog.isVisible()
    assert dialog._closed and not dialog._timer.isActive()
    assert dialog._capture_context is None
    assert workspace.controller.budget.used_bytes == used
    QTest.qWait(550)
    dialog._tick()
    assert started == []
    assert workspace.controller.draft is draft


def test_escape_running_job_reopen_keeps_single_inflight_until_cancel_finishes(workspace, monkeypatch):
    import threading
    dialog = open_dialog(workspace)
    entered, release = threading.Event(), threading.Event()
    frame = blank(0)
    monkeypatch.setattr(dialog, "_raw_frame", lambda: frame)
    used = workspace.controller.budget.used_bytes
    lease = workspace.controller.budget.acquire(frame)
    def work(check):
        entered.set()
        release.wait(timeout=3)
        check()
        return {}
    dialog._submit("light", work, dialog._context(), [lease])
    try:
        assert entered.wait(timeout=1)
        original_future = dialog._job[1]
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        QApplication.processEvents()
        assert dialog._closed and dialog._cancel.is_set()
        workspace._show_optical_settings(diagnostics=True)
        dialog._timer.stop()
        started = []
        monkeypatch.setattr(dialog, "_start_light", lambda frame: started.append(frame.frame_id))
        dialog._tick()
        assert dialog._job[1] is original_future
        assert started == []
    finally:
        release.set()
        complete(dialog)
    assert workspace.controller.budget.used_bytes == used


def test_light_diagnostics_single_inflight_and_at_most_two_hz(workspace, monkeypatch):
    import fdm.ui.optical_settings_dialog as module
    dialog = open_dialog(workspace)
    dialog.tabs.setCurrentIndex(1)
    now = [1.]
    current = [blank(0)]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(dialog, "_raw_frame", lambda: current[0])
    tasks = []
    def submit(kind, work, context, leases=(), scratch=None):
        tasks.append(kind)
        for lease in leases:
            workspace.controller.budget.release(lease)
        workspace.controller.budget.release_scratch(scratch)
    monkeypatch.setattr(dialog, "_submit", submit)
    dialog._tick()
    current[0] = blank(1)
    now[0] = 1.49
    dialog._tick()
    assert tasks == ["light"]
    now[0] = 1.5
    dialog._tick()
    assert tasks == ["light", "light"]
    pending = Future()
    dialog._job = ("light", pending, dialog._context())
    now[0] = 3.
    current[0] = blank(2)
    dialog._tick()
    assert len(tasks) == 2
    dialog._job = None


def test_light_result_has_readable_chinese_metrics_separate_from_json(workspace):
    from fdm.services.optical_flat_field import illumination_diagnostics
    dialog = open_dialog(workspace)
    diagnostic = illumination_diagnostics(blank(0).pixels)
    dialog._completed("light", diagnostic)
    text = dialog.light_status.text()
    assert "背景亮度 210.0 / 255" in text
    assert "近饱和 0.0%" in text and "清晰度指标" in text
    assert "brightness_background" not in text
    assert "brightness_background" in dialog.diagnostics_text.toPlainText()


def test_invalid_metadata_is_quiet_and_never_leaks_frame_or_scratch(workspace, monkeypatch):
    dialog = open_dialog(workspace)
    bad = FrameHandle.snapshot(np.full((40, 60), 210, np.uint8),
        source_id="camera", live=True, metadata={"exposure": float("nan")})
    monkeypatch.setattr(dialog, "_raw_frame", lambda: bad)
    used = workspace.controller.budget.used_bytes
    dialog._start_light(bad)
    assert workspace.controller.budget.used_bytes == used
    assert dialog._job is None
    dialog._start_capture()
    assert dialog._capture_context is None
    dialog.tabs.setCurrentIndex(1)
    for _ in range(5):
        dialog._tick()
    assert workspace.controller.budget.used_bytes == used
    assert "成像信息暂不可用" in dialog.status.text()


def test_static_reanalysis_refreshes_diagnostics_without_rechecking_light(workspace, monkeypatch):
    candidates(workspace)
    from fdm.services.optical_flat_field import illumination_diagnostics
    dialog = open_dialog(workspace)
    dialog.tabs.setCurrentIndex(1)
    dialog._completed("light", illumination_diagnostics(dialog._raw_frame().pixels))
    dialog._last_frame_id = dialog._raw_frame().frame_id
    starts = []
    monkeypatch.setattr(dialog, "_start_light", lambda frame: starts.append(frame.frame_id))
    context = workspace.controller._automatic_context
    context["task_id"] = "second-analysis-same-frame"
    context["diagnostics"] = {"flat_field": {"status": "not_applied", "message": "参考已失效"}}
    dialog._tick()
    assert "参考已失效" in dialog.diagnostics_text.toPlainText()
    assert "校正未应用" in dialog.light_status.text()
    assert starts == []


def test_offline_export_contains_verified_reference_and_diagnostic_snapshot(workspace, tmp_path, monkeypatch):
    from fdm.services.optical_flat_field import prepare_flat_field_reference, save_flat_field_reference
    dialog = open_dialog(workspace)
    current = blank(0)
    monkeypatch.setattr(dialog, "_raw_frame", lambda: current)
    prepared = prepare_flat_field_reference([blank(i) for i in range(5)], dialog._context())
    reference = save_flat_field_reference(prepared, tmp_path / "reference")
    dialog._config["flat_reference"] = reference
    output = tmp_path / "diagnostic.zip"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(output), "ZIP"))
    dialog._export_diagnostics()
    complete(dialog)
    with zipfile.ZipFile(output) as archive:
        payload = json.loads(archive.read("diagnostics.json"))
        assert payload["application_version"] == "0.4.10"
        assert payload["config"]["flat_reference"]["sha256"] == reference["sha256"]
        assert len(archive.namelist()) == 3
    dialog._config["flat_reference"] = dict(reference, sha256="0" * 64)
    dialog._export_diagnostics()
    complete(dialog)
    with zipfile.ZipFile(output) as archive:
        payload = json.loads(archive.read("diagnostics.json"))
        assert payload["flat_reference_error"]
        assert archive.namelist() == ["diagnostics.json"]
    assert "参考不可用" in dialog.status.text()
