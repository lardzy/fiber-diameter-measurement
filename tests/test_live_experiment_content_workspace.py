"""Interaction and result-publication regression tests for the live workspace."""
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QAction, QCloseEvent, QImage, QKeyEvent
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QMainWindow, QMessageBox, QTableWidget

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameHandle, PresentedFrameToken, FrameBudget, FrameBudgetExceeded
from fdm.models import Calibration, ImageDocument, ProjectState
from fdm.raster import RasterPixelType, RasterPlane
from fdm.ui.live_experiment_workspace import open_live_experiment


def make_session():
    s = ExperimentSession(save_images=False)
    s.add_operator("甲")
    for direction in ("warp", "weft"):
        s.set_materials(default_materials(), direction=direction)
    return s


@pytest.fixture
def workspace():
    class Host(QMainWindow):
        def current_document(self):
            return self.project.documents[0]
    host = Host()
    host.project = ProjectState.empty()
    host.project.documents.append(ImageDocument(id="source", path="sample.png", image_size=(320, 240),
        calibration=Calibration("manual", 1, "um", "test")))
    host.project.live_experiments.append(make_session())
    image = QImage(320, 240, QImage.Format.Format_RGB32)
    image.fill(0xffffffff)
    host._images = {"source": image}
    host.save_project_action = QAction(host)
    view = open_live_experiment(host)
    view.activateWindow()
    QApplication.processEvents()
    yield view
    view.controller.poll()
    view.close()
    host.close()
    QApplication.processEvents()


def present(controller, sequence, live=True):
    frame = FrameHandle.snapshot(np.full((240, 320, 3), 240, np.uint8), source_id="test",
                                 sequence=sequence, live=live)
    controller.receive_frame(frame)
    controller.mark_presented(PresentedFrameToken(frame, controller.field_epoch, 0, 1, (0, 0)))
    return frame


def ready_result(controller):
    controller.set_calibration(CalibrationSnapshot(1, 1))
    present(controller, 1)
    controller.state = "ready"
    controller._motion_future = (controller.field_epoch, Future())
    controller.start((30, 30))
    controller.finish((30, 50), "wool", method="free")
    deadline = time.perf_counter() + 1
    while not controller._jobs[0][1].done() and time.perf_counter() < deadline:
        time.sleep(.001)
    assert controller._jobs[0][1].done()


def test_submission_barrier_covers_latest_presented_without_extending_timeout(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        ready_result(controller)
        present(controller, 10)
        controller.poll()
        intent = controller._jobs[0][0]
        assert intent.submission_barrier_sequence == 10
        started = intent.submission_barrier_at
        present(controller, 11)
        controller._motion_sequence = 10
        controller.poll()
        assert not controller.session.measurements
        assert intent.submission_barrier_sequence == 11
        assert intent.submission_barrier_at == started
        controller._motion_sequence = 11
        controller.poll()
        assert len(controller.session.measurements) == 1
        record = next(iter(controller.session.measurements.values()))
        assert record.provenance["submission_barrier_sequence"] == 11
        assert record.provenance["submission_barrier_initial_sequence"] == 10
        assert record.provenance["submission_barrier_previous_sequence"] == 10
    finally:
        controller.close()


def test_submission_barrier_times_out_with_explicit_backpressure(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        ready_result(controller)
        controller.poll()
        intent = controller._jobs[0][0]
        intent.submission_barrier_at -= 3
        rejected = QSignalSpy(controller.resultRejected)
        controller.poll()
        assert not controller._jobs and not controller.session.measurements
        assert rejected.count() == 1 and "两秒" in rejected.at(0)[0]
    finally:
        controller.close()


def test_detected_move_invalidates_submission_barrier(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        ready_result(controller)
        controller.poll()
        controller.end_field("moved")
        controller._motion_sequence = 100
        controller.poll()
        assert not controller.session.measurements
    finally:
        controller.close()


def test_frame_budget_keeps_shared_headroom_until_last_reserving_lease():
    frame = FrameHandle.snapshot(np.zeros((4, 5), np.uint8), source_id="test")
    other = FrameHandle.snapshot(frame.pixels, source_id="test", sequence=1)
    third = FrameHandle.snapshot(frame.pixels, source_id="test", sequence=2)
    budget = FrameBudget(100)
    first = budget.acquire(frame, reserve_bytes=60)
    second = budget.acquire(other, reserve_bytes=10)
    assert budget.used_bytes == budget.peak_bytes == 100
    with pytest.raises(FrameBudgetExceeded):
        budget.acquire(third)
    budget.release(first)
    assert budget.used_bytes == 30
    budget.release(second)
    assert budget.used_bytes == 0


def test_mailbox_coalesces_producer_frames_and_blocks_previous_source(workspace):
    emitted = QSignalSpy(workspace.frameArrived)
    callback, error = workspace._source_callbacks()
    template = workspace.controller.latest_frame
    def produce():
        for sequence in range(1, 200):
            callback(replace(template, sequence=sequence))
    producer = threading.Thread(target=produce)
    producer.start(); producer.join()
    assert emitted.count() == 1
    assert workspace._frame_mailbox[1].sequence == 199
    QApplication.processEvents()
    assert workspace.controller.latest_frame.sequence == 199
    workspace._stop_source()
    callback(replace(template, sequence=999))
    error("stale source failure")
    QApplication.processEvents()
    assert workspace.controller.latest_frame.sequence == 199
    assert workspace.controller.state != "unavailable"


def test_numeric_count_works_from_combo_and_table_but_not_text_or_diameter(workspace):
    workspace.keyboard_mode.setCurrentIndex(1)
    workspace.material.setFocus()
    QTest.keyClick(workspace.material, Qt.Key.Key_2)
    assert len(workspace.session.count_adjustments) == 1
    assert next(iter(workspace.session.count_adjustments.values())).material_id == "cashmere"
    workspace.history_table.setFocus()
    QTest.keyClick(workspace.history_table, Qt.Key.Key_1)
    assert len(workspace.session.count_adjustments) == 2
    workspace.advanced_toggle.setChecked(True)
    workspace.settings_dialog.activateWindow()
    QApplication.processEvents()
    workspace.sample.setFocus()
    QTest.keyClicks(workspace.sample, "2")
    assert workspace.sample.text() == "2"
    assert len(workspace.session.count_adjustments) == 2
    workspace.mode.setCurrentIndex(1)
    workspace.canvas.setFocus()
    QTest.keyClick(workspace.canvas, Qt.Key.Key_2)
    assert len(workspace.session.count_adjustments) == 2
    assert workspace.count_add_button.isHidden() and workspace.weights_button.isHidden()


def test_direction_and_undo_settings_restore_visible_controls(workspace):
    workspace.session.active_direction = "weft"
    workspace._refresh_operators()
    assert workspace.direction.currentData() == "weft"
    workspace.gesture.setCurrentIndex(1)
    assert workspace.session.workspace_options["gesture"] == "drag"
    workspace._history_action("undo")
    assert workspace.gesture.currentIndex() == 0
    workspace._history_action("redo")
    assert workspace.gesture.currentIndex() == 1
    assert workspace.direction.currentData() == "weft"


def test_record_redo_restores_current_field_overlay_ids(workspace):
    s, controller = workspace.session, workspace.controller
    record = s.add_measurement(20, "wool", field_id=controller.field_id, endpoints=((10, 10), (10, 30)))
    controller.current_records.append(record.id)
    workspace._record_accepted(record)
    workspace._history_action("undo")
    assert workspace.current_model.rowCount() == 0
    workspace._history_action("redo")
    assert workspace.current_model.ids == [record.id] and controller.current_records == [record.id]


@pytest.mark.parametrize("pixel_type,channels", [(RasterPixelType.GRAY8, 1), (RasterPixelType.RGB8, 3), (RasterPixelType.RGBA8, 4)])
def test_static_source_uses_original_raster_and_correct_channel_order(workspace, pixel_type, channels):
    shape = (240, 320) if channels == 1 else (240, 320, channels)
    pixels = np.zeros(shape, np.uint8)
    if channels == 1:
        pixels[:] = 17
    else:
        pixels[:] = [11, 22, 33, 255][:channels]
    plane = RasterPlane(320, 240, pixel_type, pixels.tobytes())
    workspace.host._rasters = {"source": plane}
    workspace._load_static()
    captured = workspace.controller.latest_frame.pixels
    expected = pixels if channels == 1 else pixels[..., [2, 1, 0] if channels == 3 else [2, 1, 0, 3]]
    np.testing.assert_array_equal(captured, expected)
    assert not captured.flags.writeable
    if channels == 1:
        assert not captured.flags.owndata
    assert workspace.controller.latest_frame.metadata["raster_sha256"] == plane.sha256()


def test_non8bit_raster_never_falls_back_to_display_image(workspace):
    plane = RasterPlane(320, 240, RasterPixelType.GRAY16, np.full((240, 320), 12345, "<u2").tobytes())
    workspace.host._rasters = {"source": plane}
    with pytest.raises(ValueError, match="8 位图像副本"):
        workspace._load_static()


def test_status_label_updates_without_statistics_revision(workspace):
    workspace.controller.set_calibration(CalibrationSnapshot(2, 3))
    workspace._refresh_summary()
    assert "X 2 / Y 3" in workspace.live_state_label.text()
    workspace.controller.unavailable("lost")
    workspace._refresh_summary()
    assert "来源不可用" in workspace.live_state_label.text() and "未标定" in workspace.live_state_label.text()


def test_current_summary_labels_actual_grouped_mean(workspace):
    workspace.session.add_measurement(20, "wool")
    workspace._refresh_summary()
    assert "计算均径 21.250" in workspace.summary.text()


def test_material_editor_initial_values_and_stable_identity(workspace, monkeypatch):
    before = workspace.session.operators[workspace.session.active_operator_id].materials["warp"]
    def accept(dialog):
        table = dialog.findChild(QTableWidget)
        assert table.cellWidget(0, 1).text() == "绵羊毛"
        assert table.cellWidget(1, 4).value() == 30
        table.cellWidget(0, 2).setValue(1.32)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", accept)
    workspace._materials()
    after = workspace.session.operators[workspace.session.active_operator_id].materials["warp"]
    assert after[0].id == before[0].id and after[0].density == 1.32
    assert after[1].id == before[1].id and after[1].upper_limit == 30


def test_batch_atomic_provenance_and_possible_duplicate_hint(workspace):
    controller = workspace.controller
    controller._automatic = SimpleNamespace(cancel=lambda: None)
    frame = controller.latest_frame
    controller._automatic_context = {"token": controller.presented, "calibration": controller.calibration,
        "algorithm": "traditional", "operator_id": workspace.session.active_operator_id,
        "direction": "warp", "lease": controller.budget.acquire(frame), "diagnostics": {"revision": 1}}
    candidate = {"width_px": 20, "start": [30, 30], "end": [30, 50], "path": [[0, 0], [1, 1]]}
    controller.candidates = [candidate]
    record = workspace.session.add_measurement(20, "wool", field_id=controller.field_id, endpoints=((30, 50), (30, 30)))
    controller.current_records.append(record.id)
    assert controller.possible_duplicate_candidates([0]) == [0]
    controller.accept_candidates([(0, "wool")])
    added = list(workspace.session.measurements.values())[-1]
    assert added.provenance["algorithm_revision"] == "cu-automatic-traditional-v1"
    batch = workspace.session.batches[added.provenance["batch_id"]]
    assert batch["provenance"]["original_candidates"] == [candidate]
    controller._automatic = None


def test_frame_references_are_counted_once_and_released_on_close(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        controller.set_calibration(CalibrationSnapshot(1, 1))
        first = present(controller, 1, live=False)
        assert controller.budget.used_bytes == first.nbytes
        controller.start((10, 10))
        second = present(controller, 2, live=False)
        assert controller.budget.used_bytes == first.nbytes + second.nbytes
        third = FrameHandle.snapshot(first.pixels, source_id="test", sequence=3, live=False)
        controller.receive_frame(third)
        assert controller.budget.used_bytes == 3 * first.nbytes
        controller.cancel_draft()
        assert controller.budget.used_bytes == 2 * first.nbytes
        controller.mark_presented(PresentedFrameToken(third, controller.field_epoch, 0, 1, (0, 0)))
        assert controller.budget.used_bytes == first.nbytes
    finally:
        controller.close()
    assert controller.budget.used_bytes == 0


def test_field_metadata_created_only_after_acceptance_and_closed_on_transition(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        controller.end_field("empty")
        assert controller.session.fields == {}
        ready_result(controller)
        controller._motion_sequence = 1
        controller.poll()
        field_id = controller.field_id
        record = next(iter(controller.session.measurements.values()))
        field = controller.session.fields[field_id]
        assert field["first_frame_id"] == record.frame_id
        assert field["calibration"]["x_um"] == 1
        controller.end_field("source moved")
        assert controller.session.fields[field_id]["end_reason"] == "source moved"
        assert "ended_at" in controller.session.fields[field_id]
        assert len(controller.session.fields) == 1
    finally:
        controller.close()


def test_motion_releases_full_frame_lease_before_fft(tmp_path, monkeypatch):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    frame = FrameHandle.snapshot(np.zeros((240, 320), np.uint8), source_id="test")
    lease = controller.budget.acquire(frame)
    holder = [frame, lease]
    original = controller._motion.compare_reduced
    def check(*args):
        assert controller.budget.used_bytes == 0
        assert holder == []
        return original(*args)
    monkeypatch.setattr(controller._motion, "compare_reduced", check)
    try:
        assert controller._motion_work(holder, True).stable
        assert controller._motion.retained_bytes < frame.nbytes
    finally:
        controller.close()


def test_old_motion_error_cannot_disconnect_new_field(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        present(controller, 1, live=False)
        old_epoch = controller.field_epoch
        failed = Future()
        failed.set_exception(RuntimeError("old source failed"))
        controller._motion_future = (old_epoch, failed)
        controller.end_field("new source")
        controller.poll()
        assert controller.state == "ready" and controller.latest_frame is not None
    finally:
        controller.close()


def test_unavailable_source_releases_canvas_reference(workspace):
    assert workspace.canvas.frame is not None
    workspace.controller.unavailable("device lost")
    assert workspace.canvas.frame is None and workspace.canvas.image is None


def test_text_undo_does_not_undo_measurement(workspace):
    record = workspace.session.add_measurement(20, "wool")
    workspace.advanced_toggle.setChecked(True)
    workspace.settings_dialog.activateWindow()
    QApplication.processEvents()
    workspace.sample.setFocus()
    QTest.keyClicks(workspace.sample, "abc")
    QTest.keyClick(workspace.sample, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert record.id in workspace.session.measurements
    assert workspace.sample.text() == ""


def test_first_live_source_requires_stability_even_if_calibrated_before_arrival(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        controller.set_calibration(CalibrationSnapshot(1, 1))
        assert controller.state == "unavailable"
        present(controller, 0)
        assert controller.state == "stabilizing" and controller._stable_count == 0
        with pytest.raises(ValueError, match="稳定"):
            controller.start((10, 10))
        controller.state = "ready"
        controller._stable_count = 2
        changed = FrameHandle.snapshot(np.zeros((240, 320), np.uint8), source_id="other", sequence=0)
        controller.receive_frame(changed)
        assert controller.state == "stabilizing" and controller._stable_count == 0
    finally:
        controller.close()


def pending_overrange(controller):
    ready_result(controller)
    intent, future = controller._jobs[0]
    intent.material_id = "cashmere"
    result = dict(future.result(), diameter_um=40, width_px=40, end=[30, 70])
    done = Future()
    done.set_result(result)
    controller._jobs[0] = (intent, done)
    controller._motion_sequence = 1
    controller.poll()
    assert len(controller.pending_classification) == 1


def candidates_ready(controller):
    controller.set_calibration(CalibrationSnapshot(1, 1))
    frame = present(controller, 1)
    controller.state = "ready"
    controller._motion_future = (controller.field_epoch, Future())
    controller._automatic = SimpleNamespace(cancel=lambda: None, close=lambda: None, poll=lambda: [])
    controller._automatic_context = {"token": controller.presented, "calibration": controller.calibration,
        "algorithm": "traditional", "operator_id": controller.session.active_operator_id,
        "direction": "warp", "lease": controller.budget.acquire(frame), "diagnostics": {"revision": 1}}
    controller.candidates = [{"width_px": 20, "start": [30, 30], "end": [30, 50]}]


def queue_live_publication(controller, kind):
    if kind == "normal":
        ready_result(controller)
    elif kind == "manual":
        pending_overrange(controller)
    else:
        candidates_ready(controller)
        controller.candidates.append({"width_px": 25, "start": [60, 30], "end": [60, 55]})
    present(controller, 10)
    if kind == "normal":
        controller.poll()
    elif kind == "manual":
        controller.resolve_pending()
    else:
        controller.accept_candidates([(0, "wool"), (1, "wool")])


@pytest.mark.parametrize("kind", ["normal", "manual", "automatic"])
@pytest.mark.parametrize("latest_stable", [False, True])
def test_late_displayed_frame_must_be_checked_before_any_ledger_entry(tmp_path, monkeypatch, kind, latest_stable):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    pending_motion = []
    def submit_motion(function, holder, reset):
        sequence = holder[0].sequence
        controller.budget.release(holder[1])
        holder.clear()
        future = Future()
        pending_motion.append((sequence, future))
        return future
    monkeypatch.setattr(controller._motion_pool, "submit", submit_motion)
    try:
        queue_live_publication(controller, kind)
        present(controller, 12)
        controller._motion_future[1].set_result(SimpleNamespace(stable=True, sequence=10))
        controller.poll()
        assert not controller.session.measurements
        # A completed stable frame 10 cannot validate already-displayed frame 12.
        assert pending_motion[-1][0] == 12
        assert controller._motion_future[1] is pending_motion[-1][1]
        pending_motion[-1][1].set_result(SimpleNamespace(stable=latest_stable, sequence=12))
        controller.poll()
        assert len(controller.session.measurements) == ((2 if kind == "automatic" else 1) if latest_stable else 0)
        if latest_stable:
            assert all(record.provenance["submission_barrier_sequence"] == 12
                       for record in controller.session.measurements.values())
        else:
            assert not controller.pending_classification and controller._confirmation is None
    finally:
        controller.close()
    assert controller.budget.used_bytes == 0


@pytest.mark.parametrize("kind", ["normal", "manual", "automatic"])
def test_advancing_display_does_not_extend_publication_deadline(tmp_path, kind):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        queue_live_publication(controller, kind)
        if kind == "normal":
            controller._jobs[0][0].submission_barrier_at -= 3
        else:
            controller._confirmation["started"] -= 3
        present(controller, 20)
        # Even a detector catching up after the deadline must not revive it.
        controller._motion_sequence = 20
        rejected = QSignalSpy(controller.resultRejected)
        controller.poll()
        assert not controller.session.measurements and controller._confirmation is None
        assert not controller._jobs
        assert rejected.count() == 1 and "两秒" in rejected.at(0)[0]
    finally:
        controller.close()


@pytest.mark.parametrize("kind", ["manual", "automatic"])
def test_confirmation_covers_latest_display_frame_then_commits_automatically(tmp_path, kind):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        (pending_overrange if kind == "manual" else candidates_ready)(controller)
        present(controller, 10)
        if kind == "manual":
            controller.resolve_pending()
        else:
            selected = [(0, "wool")]
            controller.accept_candidates(selected)
            selected[0] = (0, "cashmere")
            controller.candidates[0]["start"][0] = 100
        assert controller._confirmation["sequence"] == 10
        started = controller._confirmation["started"]
        assert not controller.session.measurements
        present(controller, 11)
        controller._motion_sequence = 9
        controller.poll()
        assert not controller.session.measurements
        controller._motion_sequence = 10
        controller.poll()
        assert not controller.session.measurements
        assert controller._confirmation["sequence"] == 11
        assert controller._confirmation["started"] == started
        controller._motion_sequence = 11
        controller.poll()
        record = next(iter(controller.session.measurements.values()))
        assert record.provenance["submission_barrier_sequence"] == 11
        assert record.provenance["submission_barrier_initial_sequence"] == 10
        assert record.endpoints[0] == (30, 30)
        assert record.material_id == ("cashmere" if kind == "manual" else "wool")
        assert controller._confirmation is None
    finally:
        controller.close()


@pytest.mark.parametrize("kind", ["manual", "automatic"])
def test_field_change_cancels_queued_confirmation(tmp_path, kind):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        (pending_overrange if kind == "manual" else candidates_ready)(controller)
        present(controller, 10)
        if kind == "manual":
            controller.resolve_pending()
        else:
            controller.accept_candidates([(0, "wool")])
        assert controller._confirmation
        controller.end_field("moved")
        controller._motion_sequence = 100
        controller.poll()
        assert controller._confirmation is None and not controller.session.measurements
    finally:
        controller.close()
    assert controller.budget.used_bytes == 0


def test_second_candidate_confirmation_gets_fresh_display_barrier_after_timeout(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        candidates_ready(controller)
        present(controller, 35)
        controller.accept_candidates([(0, "wool")])
        controller._confirmation["started"] -= 3
        rejected = QSignalSpy(controller.resultRejected)
        controller.poll()
        assert controller._confirmation is None and rejected.count() == 1
        assert "两秒" in rejected.at(0)[0] and controller.candidates
        controller.state = "ready"
        controller._motion_sequence = 35
        present(controller, 85)
        controller.accept_candidates([(0, "wool")])
        assert controller._confirmation["sequence"] == 85
        assert not controller.session.measurements
        controller._motion_sequence = 85
        controller.poll()
        record = next(iter(controller.session.measurements.values()))
        assert record.provenance["submission_barrier_sequence"] == 85
    finally:
        controller.close()


def show_candidates(workspace):
    candidates = [{"width_px": 20, "start": [30, 30], "end": [30, 50]},
                  {"width_px": 40, "start": [60, 30], "end": [60, 70]}]
    workspace.controller.candidates = candidates
    workspace.controller.candidatesChanged.emit(candidates)
    return workspace.candidate_model


def test_candidates_require_explicit_category_and_keep_per_row_stable_ids(workspace, monkeypatch):
    model = show_candidates(workspace)
    accepted = []
    monkeypatch.setattr(workspace.controller, "accept_candidates", lambda selections, **kwargs: accepted.append(selections))
    assert model.columnCount() == 6
    assert model.data(model.index(0, 3)).startswith("未赋类")
    with pytest.raises(ValueError, match="未赋类"):
        workspace._accept_candidates()
    assert not accepted
    model.setData(model.index(0, 3), "cashmere", Qt.ItemDataRole.EditRole)
    model.setData(model.index(1, 3), "wool", Qt.ItemDataRole.EditRole)
    workspace._accept_candidates()
    assert accepted == [[(0, "cashmere"), (1, "wool")]]
    assert model.data(model.index(0, 3), Qt.ItemDataRole.UserRole) == "cashmere"
    assert not model.setData(model.index(0, 3), "unknown", Qt.ItemDataRole.EditRole)


def test_candidates_delegate_and_batch_assignment_preserve_excluded_rows(workspace):
    model = show_candidates(workspace)
    index = model.index(0, 3)
    delegate = workspace.candidate_table.itemDelegateForColumn(3)
    editor = delegate.createEditor(workspace.candidate_table, None, index)
    try:
        delegate.setEditorData(editor, index)
        assert editor.currentData() is None
        editor.setCurrentIndex(editor.findData("cashmere"))
        delegate.setModelData(editor, model, index)
        assert model.assignments[0] == "cashmere"
        model.setData(model.index(0, 0), Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        workspace.material.setCurrentIndex(workspace.material.findData("wool"))
        workspace._assign_candidate_material()
        assert model.assignments == {0: "cashmere", 1: "wool"}
        model.setData(model.index(1, 3), "cashmere", Qt.ItemDataRole.EditRole)
        assert model.range_status(1) == "越限，需核对"
    finally:
        editor.deleteLater()


def test_confirming_candidates_explicitly_excludes_unchecked_unassigned_rows(workspace, monkeypatch):
    model = show_candidates(workspace)
    model.setData(model.index(0, 3), "wool", Qt.ItemDataRole.EditRole)
    model.setData(model.index(1, 0), Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    accepted = []
    monkeypatch.setattr(workspace.controller, "accept_candidates", lambda selections, **kwargs: accepted.append(selections))
    workspace._accept_candidates()
    assert accepted == [[(0, "wool")]]


def test_candidate_range_consent_is_visible_nonmodal_and_reset_by_category_or_bounds(workspace, monkeypatch):
    model = show_candidates(workspace)
    model.setData(model.index(0, 0), Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    model.setData(model.index(1, 3), "cashmere", Qt.ItemDataRole.EditRole)
    assert "30" in model.data(model.index(1, 4)) and "40" in model.data(model.index(1, 2))
    accepted = []
    monkeypatch.setattr(workspace.controller, "accept_candidates", lambda selections, **kwargs: accepted.append((selections, kwargs)))
    monkeypatch.setattr(QMessageBox, "question", lambda *_: pytest.fail("越限确认不应弹出模态窗口"))
    with pytest.raises(ValueError, match="越限"):
        workspace._accept_candidates()
    model.setData(model.index(1, 5), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
    workspace._accept_candidates()
    assert accepted == [([(1, "cashmere")], {"confirmed_out_of_range": {1}})]
    model.setData(model.index(1, 3), "wool", Qt.ItemDataRole.EditRole)
    model.setData(model.index(1, 3), "cashmere", Qt.ItemDataRole.EditRole)
    assert not model.range_confirmed(1)
    model.setData(model.index(1, 5), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
    model.materials = tuple(replace(m, upper_limit=35) if m.id == "cashmere" else m for m in model.materials)
    assert not model.range_confirmed(1)


def test_candidate_out_of_range_override_is_explicit_snapshotted_and_auditable(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        candidates_ready(controller)
        controller.candidates[0]["width_px"] = 40
        present(controller, 10)
        with pytest.raises(ValueError, match="确认保留"):
            controller.accept_candidates([(0, "cashmere")])
        assert controller._confirmation is None and not controller.session.measurements
        controller.accept_candidates([(0, "cashmere")], confirmed_out_of_range={0})
        controller._motion_sequence = 10
        controller.poll()
        record = next(iter(controller.session.measurements.values()))
        assert record.provenance["range_confirmation"] == {"material_id": "cashmere", "material_name": "山羊绒",
            "lower_limit_um": None, "upper_limit_um": 30.0, "diameter_um": 40.0}
    finally:
        controller.close()


def test_changed_material_bounds_invalidate_queued_range_consent(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    try:
        candidates_ready(controller)
        controller.candidates[0]["width_px"] = 40
        present(controller, 10)
        controller.accept_candidates([(0, "cashmere")], confirmed_out_of_range={0})
        materials = controller.session.operators[controller.session.active_operator_id].materials["warp"]
        controller.session.set_materials([replace(m, upper_limit=50) if m.id == "cashmere" else m for m in materials])
        rejected = QSignalSpy(controller.resultRejected)
        controller._motion_sequence = 10
        controller.poll()
        assert not controller.session.measurements and rejected.count() == 1
        assert "范围已变化" in rejected.at(0)[0]
    finally:
        controller.close()


def test_new_workspace_default_setup_is_undo_baseline():
    host = QMainWindow()
    host.project = ProjectState.empty()
    host.current_document = lambda: None
    host.save_project_action = QAction(host)
    fresh = open_live_experiment(host)
    try:
        assert not fresh.session.can_undo
        assert len(fresh.session.operators) == 1 and fresh.material.count() == 2
        record = fresh.session.add_measurement(20, "wool")
        fresh._history_action("undo")
        assert record.id not in fresh.session.measurements
        fresh._history_action("undo")
        assert len(fresh.session.operators) == 1 and fresh.material.count() == 2
    finally:
        fresh.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        host.close()


def small_live(controller, sequence, *, reserve=367):
    frame = FrameHandle.snapshot(np.zeros((10, 10), np.uint8), source_id="budget",
        sequence=sequence, metadata={"acquisition_reserved_bytes": reserve})
    assert controller.receive_frame(frame)
    controller.mark_presented(PresentedFrameToken(frame, controller.field_epoch, 0, 1, (0, 0)))
    controller.state = "ready"
    return frame


def test_new_draft_leaves_room_for_future_preview_and_waits_for_old_evidence(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    motion_gate = threading.Event()
    # Arrival now starts reduction immediately. Keep that genuine frame owner
    # alive deterministically instead of depending on the thread's speed.
    controller._motion_pool.submit(lambda: motion_gate.wait(5))
    try:
        controller.budget.limit_bytes = 700
        controller.set_calibration(CalibrationSnapshot(1, 1))
        old = small_live(controller, 0)
        old_evidence = controller.budget.acquire(old)
        small_live(controller, 1)
        epoch = controller.field_epoch
        with pytest.raises(FrameBudgetExceeded, match="暂停新增"):
            controller.start((2, 2))
        assert controller.draft is None and controller.field_epoch == epoch
        controller.budget.release(old_evidence)
        with pytest.raises(FrameBudgetExceeded, match="暂停新增"):
            controller.start((2, 2))
        assert controller.draft is None and controller.field_epoch == epoch
        motion_gate.set()
        controller._motion_future[1].result(timeout=2)
        controller.start((2, 2))
        small_live(controller, 2)
        small_live(controller, 3)
        assert controller.draft is not None and controller.field_epoch == epoch
        assert controller.budget.peak_bytes <= 700
    finally:
        motion_gate.set()
        controller.close()


def test_finish_budget_backpressure_preserves_draft_and_existing_overlay(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    blocker = None
    try:
        controller.budget.limit_bytes = 700
        controller.set_calibration(CalibrationSnapshot(1, 1))
        small_live(controller, 0)
        controller.start((2, 2))
        draft = controller.draft
        record = controller.session.add_measurement(20, "wool", field_id=controller.field_id)
        controller.current_records.append(record.id)
        extra = FrameHandle.snapshot(np.zeros((10, 10), np.uint8), source_id="writer")
        blocker = controller.budget.acquire(extra)
        epoch = controller.field_epoch
        with pytest.raises(FrameBudgetExceeded, match="暂停新增"):
            controller.finish((2, 7), "wool", method="free")
        assert controller.draft is draft and controller.field_epoch == epoch
        assert controller.current_records == [record.id] and not controller._jobs
        controller.budget.release(blocker)
        blocker = None
        controller.finish((2, 7), "wool", method="free")
        assert controller.draft is None and len(controller._jobs) == 1
    finally:
        if blocker:
            controller.budget.release(blocker)
        controller.close()


def test_preview_allocation_failure_does_not_fake_field_movement(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    blocker = None
    try:
        controller.budget.limit_bytes = 600
        old = small_live(controller, 0)
        blocker = controller.budget.acquire(FrameHandle.snapshot(np.zeros((10, 10), np.uint8), source_id="writer"))
        record = controller.session.add_measurement(20, "wool", field_id=controller.field_id)
        controller.current_records.append(record.id)
        epoch = controller.field_epoch
        frame = FrameHandle.snapshot(np.zeros((10, 10), np.uint8), source_id="budget", sequence=1,
            metadata={"acquisition_reserved_bytes": 367})
        assert not controller.receive_frame(frame)
        assert controller.latest_frame is old and controller.presented.frame is old
        assert controller.field_epoch == epoch and controller.current_records == [record.id]
        assert controller.budget.used_bytes == 567
        controller.budget.release(blocker)
        blocker = None
        assert controller.receive_frame(frame)
    finally:
        if blocker:
            controller.budget.release(blocker)
        controller.close()


def test_mailbox_owns_a_budget_lease_and_releases_it_on_stop(workspace):
    base = workspace.controller.budget.used_bytes
    callback, _ = workspace._source_callbacks()
    frame = FrameHandle.snapshot(np.zeros((10, 10), np.uint8), source_id="mailbox", live=False)
    callback(frame)
    assert workspace.controller.budget.used_bytes == base + frame.nbytes
    workspace._stop_source()
    assert workspace.controller.budget.used_bytes == base


def test_unchanged_sample_metadata_does_not_hide_last_measurement_in_undo(workspace):
    record = workspace.session.add_measurement(20, "wool")
    revision = workspace.session.revision
    workspace._sample_changed()
    assert workspace.session.revision == revision
    workspace._history_action("undo")
    assert record.id not in workspace.session.measurements


def test_close_runs_queued_motion_finally_to_release_its_frame(tmp_path):
    controller = LiveExperimentController(make_session(), evidence_directory=tmp_path)
    gate = threading.Event()
    controller._motion_pool.submit(lambda: gate.wait(.05))
    frame = FrameHandle.snapshot(np.zeros((64, 64), np.uint8), source_id="queued-motion")
    lease = controller.budget.acquire(frame)
    future = controller._motion_pool.submit(controller._motion_work, [frame, lease], True)
    controller._motion_future = (controller.field_epoch, future)
    controller.close()
    assert future.done() and not future.cancelled()
    assert controller.budget.used_bytes == 0


def test_candidate_tab_record_actions_cannot_mutate_hidden_table_selection(workspace, monkeypatch):
    record = workspace.session.add_measurement(20, "wool", field_id=workspace.controller.field_id)
    workspace._rebuild_records()
    workspace.current_table.setCurrentIndex(workspace.current_model.index(0, 0))
    show_candidates(workspace)
    workspace.candidate_table.setCurrentIndex(workspace.candidate_model.index(0, 0))
    monkeypatch.setattr(QMessageBox, "question", lambda *_: pytest.fail("候选页面不应操作隐藏的测量记录"))
    for action in (workspace._delete_record, workspace._edit_record, workspace._review_record, workspace._discard_failed_evidence):
        workspace._guard(action)
        assert "先在当前视场或试验历史" in workspace.statusBar().currentMessage()
        assert record.id in workspace.session.measurements


def test_candidate_editor_digit_keys_do_not_trigger_extra_counts(workspace):
    model = show_candidates(workspace)
    workspace.keyboard_mode.setCurrentIndex(1)
    workspace.material.setCurrentIndex(0)
    delegate = workspace.candidate_table.itemDelegateForColumn(3)
    editor = delegate.createEditor(workspace.candidate_table.viewport(), None, model.index(0, 3))
    try:
        editor.show()
        editor.setFocus()
        QApplication.processEvents()
        QTest.keyClick(editor, Qt.Key.Key_2)
        assert workspace.material.currentData() == "wool"
        assert not workspace.session.count_adjustments
    finally:
        editor.deleteLater()


def test_rejected_larger_source_metadata_invalidates_old_ruler_and_field(workspace):
    controller = workspace.controller
    controller.start((30, 30))
    record = workspace.session.add_measurement(20, "wool", field_id=controller.field_id)
    controller.current_records.append(record.id)
    epoch = controller.field_epoch
    controller.budget.limit_bytes = controller.budget.used_bytes
    callback, _ = workspace._source_callbacks()
    old_frame = controller.latest_frame
    bigger = FrameHandle.snapshot(np.zeros((480, 640, 3), np.uint8), source_id=old_frame.source_id,
        generation=old_frame.generation + 1, sequence=0, live=True)
    callback(bigger)
    assert workspace._frame_mailbox is None
    QApplication.processEvents()
    assert controller.field_epoch > epoch and controller.state == "unavailable"
    assert controller.calibration is None and controller.draft is None and controller.presented is None
    assert controller.current_records == [] and record.id in workspace.session.measurements
    assert workspace.canvas.frame is None


def test_rejected_new_source_does_not_republish_old_pending_mailbox_frame(workspace):
    callback, _ = workspace._source_callbacks()
    old_frame = workspace.controller.latest_frame
    pending = FrameHandle.snapshot(np.zeros((240, 320, 4), np.uint8), source_id=old_frame.source_id,
        generation=old_frame.generation, sequence=1, live=False)
    workspace.controller.budget.limit_bytes = workspace.controller.budget.used_bytes + pending.nbytes
    callback(pending)
    bigger = FrameHandle.snapshot(np.zeros((960, 1280, 3), np.uint8), source_id=old_frame.source_id,
        generation=old_frame.generation + 1, sequence=0, live=True)
    callback(bigger)
    QApplication.processEvents()
    assert workspace.controller.latest_frame is None and workspace.canvas.frame is None
    assert workspace.controller.calibration is None and workspace.controller.state == "unavailable"


def cu_three_record_sample(workspace):
    records = [workspace.session.add_measurement(value, material, field_id=workspace.controller.field_id)
        for value, material in ((15, "wool"), (30, "wool"), (18, "cashmere"))]
    workspace._rebuild_records()
    workspace._refresh_summary()
    return records


def test_material_board_plus_minus_preserves_current_class_and_raw_mean(workspace):
    records = cu_three_record_sample(workspace)
    assert workspace.material_rows["wool"]["mean"].text() == "22.500"
    assert workspace.material_rows["wool"]["measured_count"].text() == "2"
    assert workspace.material.currentData() == "wool"
    cashmere = workspace.material_rows["cashmere"]
    QTest.mouseClick(cashmere["add"], Qt.MouseButton.LeftButton)
    workspace._refresh_summary()
    assert workspace.material.currentData() == "wool" and cashmere["extra_count"].text() == "1"
    assert cashmere["mean"].text() == "18.000" and cashmere["measured_count"].text() == "1"
    assert len(workspace.session.measurements) == 3
    QTest.mouseClick(cashmere["subtract"], Qt.MouseButton.LeftButton)
    workspace._refresh_summary()
    assert cashmere["extra_count"].text() == "0" and not cashmere["subtract"].isEnabled()
    QTest.mouseClick(cashmere["subtract"], Qt.MouseButton.LeftButton)
    assert len(workspace.session.measurements) == 3 and all(record.id in workspace.session.measurements for record in records)


def test_cu_before_after_classification_reassigns_only_last_valid_current_record(workspace, monkeypatch):
    records = cu_three_record_sample(workspace)
    QTest.mouseClick(workspace.material_rows["wool"]["select"], Qt.MouseButton.LeftButton)
    assert workspace.session.measurements[records[-1].id].material_id == "cashmere"
    workspace.classification_mode.setCurrentIndex(1)
    assert workspace.session.workspace_options["classification"] == "after"
    monkeypatch.setattr(workspace.history_model, "reset_ids", lambda *_: pytest.fail("高频后选不能重建全部历史"))
    workspace.controller.current_records.append("deleted-id")
    QTest.mouseClick(workspace.material_rows["wool"]["select"], Qt.MouseButton.LeftButton)
    workspace._refresh_summary()
    assert workspace.session.measurements[records[-1].id].material_id == "wool"
    assert workspace.material_rows["wool"]["mean"].text() == "21.000"
    assert workspace.material_rows["wool"]["measured_count"].text() == "3"
    assert workspace.material_rows["cashmere"]["measured_count"].text() == "0"
    QTest.mouseClick(workspace.material_rows["cashmere"]["select"], Qt.MouseButton.LeftButton)
    workspace._refresh_summary()
    assert workspace.material_rows["wool"]["mean"].text() == "22.500"
    assert workspace.material_rows["cashmere"]["mean"].text() == "18.000"
    assert workspace.canvas.active_material_id == "cashmere"


def test_refresh_materials_blocks_selection_signals_and_does_not_reclassify(workspace):
    record = cu_three_record_sample(workspace)[-1]
    workspace.classification_mode.setCurrentIndex(1)
    workspace.material.setCurrentIndex(workspace.material.findData("wool"))
    revision = workspace.session.revision
    changed = QSignalSpy(workspace.material.currentIndexChanged)
    workspace._refresh_materials()
    assert changed.count() == 0 and workspace.session.revision == revision
    assert workspace.session.measurements[record.id].material_id == "cashmere"


def test_after_selection_does_not_cross_field_operator_or_direction(workspace):
    session = workspace.session
    current_operator = session.active_operator_id
    foreign_operator = session.add_operator("乙")
    session.active_operator_id = current_operator
    records = [session.add_measurement(20, "wool", field_id="old-field"),
               session.add_measurement(20, "wool", field_id=workspace.controller.field_id, direction="weft"),
               session.add_measurement(20, "wool", field_id=workspace.controller.field_id, operator_id=foreign_operator)]
    workspace.controller.current_records[:] = [record.id for record in records]
    workspace.classification_mode.setCurrentIndex(1)
    workspace._select_material("cashmere")
    assert all(session.measurements[record.id].material_id == "wool" for record in records)


def test_classification_and_sample_number_roundtrip_and_undo(workspace):
    workspace.advanced_toggle.setChecked(True)
    workspace.sample_number.setText("CU-2026-015")
    workspace.sample_number.editingFinished.emit()
    workspace.classification_mode.setCurrentIndex(1)
    restored = ExperimentSession.from_dict(workspace.session.to_dict())
    assert restored.workspace_options["classification"] == "after" and restored.sample_number == "CU-2026-015"
    workspace._history_action("undo")
    assert workspace.classification_mode.currentIndex() == 0 and not workspace.after_select.isChecked()
    workspace._history_action("redo")
    assert workspace.classification_mode.currentIndex() == 1 and workspace.after_select.isChecked()


def test_history_labels_operator_for_all_operator_view(workspace):
    first = workspace.session.add_measurement(20, "wool")
    second_operator = workspace.session.add_operator("乙")
    second = workspace.session.add_measurement(30, "wool", operator_id=second_operator)
    workspace._rebuild_records()
    assert "全部操作者" in workspace.tabs.tabText(1)
    model = workspace.history_model
    assert model.data(model.index(model.rows_by_id[first.id], 1)) == "甲"
    assert model.data(model.index(model.rows_by_id[second.id], 1)) == "乙"


class DelayedStoppingSource:
    def __init__(self, error_callback=lambda *_: None):
        self.release = threading.Event()
        self.thread = threading.Thread(target=self.release.wait, daemon=True)
        self.thread.start()
        self.error_callback = error_callback
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        self.thread.join(.005)
        if self.thread.is_alive():
            self.error_callback("SDK still releasing")
            return False
        return True

    def finish(self):
        self.release.set()
        self.thread.join(1)


def test_retiring_source_blocks_all_replacements_until_confirmed_stopped(workspace, monkeypatch):
    from fdm.live_experiment import sources
    old_frame = workspace.controller.latest_frame
    late_frame, late_error = workspace._source_callbacks()
    old = DelayedStoppingSource(late_error)
    workspace.source = old
    new = SimpleNamespace(start=lambda: None, stop=lambda: True)
    constructed = []
    monkeypatch.setattr(sources, "CameraFrameSource", lambda *_: constructed.append("camera") or new)
    generation = workspace._static_generation
    try:
        assert workspace._start_camera() is False
        assert workspace._load_static() is False
        assert not constructed and workspace.source is old and workspace._source_retiring
        assert workspace._static_generation == generation
        assert workspace.controller.state == "unavailable"
        assert "仍在释放" in workspace.statusBar().currentMessage()
        late_frame(old_frame)
        late_error("old callback must stay ignored")
        QApplication.processEvents()
        assert workspace.controller.latest_frame is None
        old.finish()
        workspace._load_static()
        assert constructed == [] and workspace.source is None
        assert not workspace._source_retiring
    finally:
        old.finish()


def test_failed_source_stop_prevents_window_and_host_transition(workspace):
    old = DelayedStoppingSource()
    workspace.source = old
    try:
        assert workspace.close() is False
        assert not workspace._closed and not workspace.controller._closed
        assert workspace._refresh_timer.isActive()
        assert workspace.host._live_experiment_workspace is workspace
        assert workspace.source is old and workspace.isVisible()
        old.finish()
        event = QCloseEvent()
        workspace.closeEvent(event)
        assert event.isAccepted() and workspace._closed and workspace.controller._closed
        assert workspace.source is None and workspace.host._live_experiment_workspace is None
    finally:
        old.finish()


def test_stop_exception_preserves_owned_source_for_retry(workspace):
    def failing_stop():
        raise OSError("release failed")
    old = SimpleNamespace(stop=failing_stop)
    workspace.source = old
    assert workspace._stop_source() is False
    assert workspace.source is old and workspace._source_retiring
    assert "release failed" in workspace.statusBar().currentMessage()
    old.stop = lambda: True
    assert workspace._stop_source() is True
    assert workspace.source is None and not workspace._source_retiring


def test_edit_record_uses_unique_numbered_material_choices(workspace, monkeypatch):
    operator = workspace.session.operators[workspace.session.active_operator_id]
    operator.materials["warp"] = tuple(replace(material, name="同名材料") for material in operator.materials["warp"])
    record = workspace.session.add_measurement(20, "wool", field_id=workspace.controller.field_id)
    workspace._rebuild_records()
    workspace.current_table.setCurrentIndex(workspace.current_model.index(0, 0))
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *_: (21, True))
    def choose(*args):
        choices = args[3]
        assert choices == ["1 · 同名材料", "2 · 同名材料"]
        return choices[1], True
    monkeypatch.setattr(QInputDialog, "getItem", choose)
    workspace._edit_record()
    assert workspace.session.measurements[record.id].material_id == "cashmere"


def _wait_statistics_job(workspace):
    deadline = time.perf_counter() + 3
    while workspace._statistics_job is not None and time.perf_counter() < deadline:
        workspace._poll_statistics_rebuild()
        QTest.qWait(5)
    assert workspace._statistics_job is None


@pytest.mark.parametrize("grouped", [True, False])
def test_statistics_dialog_preserves_loaded_custom_width(workspace, monkeypatch, grouped):
    from PySide6.QtWidgets import QComboBox
    workspace.session.configure(group_width=1.75, grouped=grouped)
    def accept(dialog):
        combos = dialog.findChildren(QComboBox)
        grouping = combos[1]
        assert grouping.findData(1.75) >= 0
        assert grouping.currentData() == (1.75 if grouped else None)
        combos[0].setCurrentText("SN/T 1062-2002")
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", accept)
    workspace._statistics_settings()
    assert workspace.session.group_width == 1.75 and workspace.session.grouped is grouped
    assert workspace.session.method == "SN/T 1062-2002"
    assert workspace._statistics_job is None


def test_width_rebuild_runs_off_thread_while_new_frames_are_painted(workspace, monkeypatch):
    from PySide6.QtCore import QObject
    from fdm.live_experiment.models import AggregateRebuildRequest
    class PaintCounter(QObject):
        count = 0
        def eventFilter(self, watched, event):
            if event.type() == QEvent.Type.Paint:
                self.count += 1
            return False
    observer = PaintCounter(workspace.canvas)
    workspace.canvas.installEventFilter(observer)
    workspace.session.add_measurement(15, "wool")
    started, release = threading.Event(), threading.Event()
    owner = threading.get_ident()
    build = AggregateRebuildRequest.build
    def delayed(request, cancelled):
        assert threading.get_ident() != owner
        started.set()
        assert release.wait(2)
        return build(request, cancelled)
    monkeypatch.setattr(AggregateRebuildRequest, "build", delayed)
    try:
        workspace._configure(group_width=2)
        assert started.wait(1)
        assert workspace.session.group_width == 2.5
        assert workspace.statistics_progress.isVisible()
        first = workspace.controller.latest_frame
        before = observer.count
        for sequence in (1, 2, 3):
            frame = FrameHandle.snapshot(np.full((240, 320, 3), 230 + sequence, np.uint8),
                source_id=first.source_id, generation=first.generation, sequence=sequence, live=False)
            workspace._receive_frame(frame)
            QTest.qWait(20)
        assert observer.count > before
        assert workspace.controller.presented.frame.sequence == 3
    finally:
        release.set()
        _wait_statistics_job(workspace)
    assert workspace.session.group_width == 2


def test_rebuild_stale_result_and_second_setting_do_not_change_committed_controls(workspace, monkeypatch):
    from fdm.live_experiment.models import AggregateRebuildRequest
    started, release = threading.Event(), threading.Event()
    build = AggregateRebuildRequest.build
    def delayed(request, cancelled):
        started.set()
        assert release.wait(2)
        return build(request, cancelled)
    monkeypatch.setattr(AggregateRebuildRequest, "build", delayed)
    workspace.session.add_measurement(15, "wool")
    try:
        workspace._configure(group_width=2)
        assert started.wait(1)
        old_images = workspace.session.save_images
        workspace.keep_images.setChecked(not old_images)
        assert workspace.session.save_images == old_images
        assert workspace.keep_images.isChecked() == old_images
        with pytest.raises(ValueError, match="重算尚未结束"):
            workspace._history_action("undo")
        workspace.session.add_measurement(22, "wool")
    finally:
        release.set()
        _wait_statistics_job(workspace)
    assert workspace.session.group_width == 2.5
    assert len(workspace.session.measurements) == 2
    assert "数据已变化" in workspace.statusBar().currentMessage()


def test_width_undo_and_redo_use_background_rebuild(workspace, monkeypatch):
    from fdm.live_experiment.models import AggregateRebuildRequest
    workspace.session.add_measurement(15, "wool")
    owner = threading.get_ident()
    workers = []
    build = AggregateRebuildRequest.build
    def trace(request, cancelled):
        workers.append((request.action, threading.get_ident()))
        return build(request, cancelled)
    monkeypatch.setattr(AggregateRebuildRequest, "build", trace)
    workspace._configure(group_width=2)
    _wait_statistics_job(workspace)
    workspace._history_action("undo")
    _wait_statistics_job(workspace)
    assert workspace.session.group_width == 2.5
    workspace._history_action("redo")
    _wait_statistics_job(workspace)
    assert workspace.session.group_width == 2
    assert [action for action, _ in workers] == ["configure", "undo", "redo"]
    assert all(thread != owner for _, thread in workers)


def test_close_cancels_width_worker_without_losing_workspace_owner(workspace, monkeypatch):
    from concurrent.futures import CancelledError
    from fdm.live_experiment.models import AggregateRebuildRequest
    started, release = threading.Event(), threading.Event()
    def delayed(request, cancelled):
        started.set()
        assert release.wait(2)
        assert cancelled()
        raise CancelledError()
    monkeypatch.setattr(AggregateRebuildRequest, "build", delayed)
    workspace.session.add_measurement(15, "wool")
    try:
        workspace._configure(group_width=2)
        assert started.wait(1)
        event = QCloseEvent()
        workspace.closeEvent(event)
        assert not event.isAccepted()
        assert not workspace._closed and not workspace.controller._closed
        assert workspace.host._live_experiment_workspace is workspace
        assert workspace._refresh_timer.isActive()
    finally:
        release.set()
        _wait_statistics_job(workspace)
    assert workspace.session.group_width == 2.5
    assert "取消" in workspace.statusBar().currentMessage()
    event = QCloseEvent()
    workspace.closeEvent(event)
    assert event.isAccepted() and workspace._closed
    assert workspace._statistics_pool is None


def test_failed_width_rebuild_keeps_settings_and_shows_reason(workspace):
    for _ in range(10):
        workspace.session.add_measurement(15, "wool")
    workspace._configure(group_width=1e154)
    _wait_statistics_job(workspace)
    assert workspace.session.group_width == 2.5
    assert len(workspace.session.measurements) == 10
    assert "重算未应用" in workspace.statusBar().currentMessage()
    assert not workspace.statistics_progress.isVisible()
    assert not workspace.statistics_cancel.isVisible()


def test_pending_rebuild_cannot_open_stale_second_settings_dialog(workspace, monkeypatch):
    from fdm.live_experiment.models import AggregateRebuildRequest
    started, release = threading.Event(), threading.Event()
    build = AggregateRebuildRequest.build
    def delayed(request, cancelled):
        started.set()
        assert release.wait(2)
        return build(request, cancelled)
    def forbidden_dialog(_):
        raise AssertionError("a second dialog captured stale settings during rebuild")
    monkeypatch.setattr(AggregateRebuildRequest, "build", delayed)
    monkeypatch.setattr(QDialog, "exec", forbidden_dialog)
    workspace.session.add_measurement(15, "wool")
    try:
        workspace._configure(group_width=2)
        assert started.wait(1)
        assert not workspace.statistics_button.isEnabled()
        with pytest.raises(ValueError, match="再打开计算设置"):
            workspace._statistics_settings()
        assert workspace.session.group_width == 2.5
    finally:
        release.set()
        _wait_statistics_job(workspace)
    assert workspace.session.group_width == 2
    assert workspace.statistics_button.isEnabled()


@pytest.mark.parametrize("control", ["gesture", "classification_mode", "mode"])
def test_busy_configuration_rejects_mode_switch_without_cancelling_draft(workspace, monkeypatch, control):
    from fdm.live_experiment.models import AggregateRebuildRequest
    started, release = threading.Event(), threading.Event()
    build = AggregateRebuildRequest.build
    def delayed(request, cancelled):
        started.set()
        assert release.wait(2)
        return build(request, cancelled)
    monkeypatch.setattr(AggregateRebuildRequest, "build", delayed)
    workspace.session.add_measurement(15, "wool")
    workspace.canvas.repaint()
    QApplication.processEvents()
    try:
        workspace._configure(group_width=2)
        assert started.wait(1)
        workspace.controller.start((30, 30))
        draft = workspace.controller.draft
        target = workspace.controller._classification_target
        assert draft is not None
        before = dict(workspace.session.workspace_options)
        getattr(workspace, control).setCurrentIndex(1)
        assert workspace.controller.draft is draft
        assert workspace.controller._classification_target is target
        assert workspace.session.workspace_options == before
        assert getattr(workspace, control).currentIndex() == 0
        assert not workspace.after_select.isChecked()
        assert workspace.direction.isVisible()
    finally:
        workspace.controller.cancel_draft()
        release.set()
        _wait_statistics_job(workspace)


def test_accepted_gesture_switch_still_cancels_previous_draft(workspace):
    workspace.canvas.repaint()
    QApplication.processEvents()
    workspace.controller.start((30, 30))
    assert workspace.controller.draft is not None
    workspace.gesture.setCurrentIndex(1)
    assert workspace.session.workspace_options["gesture"] == "drag"
    assert workspace.gesture.currentIndex() == 1
    assert workspace.controller.draft is None
