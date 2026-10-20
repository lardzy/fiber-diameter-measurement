"""Stored crops remain reviewable after motion, without consuming live frames."""
import json
import time

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QVBoxLayout, QLineEdit, QWidget

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.ui.measurement_evidence_view import EvidenceReviewController


@pytest.fixture
def review(tmp_path):
    workspace = QWidget()
    layout = QVBoxLayout(workspace)
    editor = QLineEdit()
    layout.addWidget(editor)
    workspace.session = ExperimentSession(save_images=False)
    workspace.session.add_operator("测试")
    workspace.session.set_materials(default_materials())
    workspace.show()
    result = EvidenceReviewController(workspace, preferences_path=tmp_path / "review.json")
    yield result
    result.close()
    workspace.close()
    workspace.deleteLater()


def record(review, tmp_path, *, number=12, state="saved", with_original=True, crop=True):
    path = tmp_path / f"evidence-{number}.png"
    image = QImage(256, 256, QImage.Format.Format_RGB32)
    image.fill(QColor("#c5ced8"))
    assert image.save(str(path))
    evidence = dict(state=state, path=str(path), error="磁盘已满" if state == "failed" else None)
    if crop:
        evidence.update(crop=[100, 200, 256, 256], mode="crop_jpeg", codec="png")
    return review.workspace.session.add_measurement(20, "wool", endpoints=((120, 230), (150, 250)),
        evidence=evidence, provenance={"measurement_number": number,
                                      **({"original_diameter_um": 20} if with_original else {})})


def wait_image(review):
    deadline = time.monotonic() + 2
    while review.dialog.image.image_size is None and time.monotonic() < deadline:
        review.refresh()
        QTest.qWait(5)
    assert review.dialog.image.image_size is not None, review.dialog.status.text()


def test_crop_coordinates_and_saved_pixels_are_preserved(review, tmp_path):
    item = record(review, tmp_path)
    original_bytes = (tmp_path / "evidence-12.png").read_bytes()
    review.show_record(item, latest=True)
    wait_image(review)
    line = review.dialog.image.overlay_items[1].line()
    assert line.p1() == QPointF(20, 30)
    assert line.p2() == QPointF(50, 50)
    assert "#12" in review.dialog.details.text() and "原测值" in review.dialog.details.text()
    QTest.mouseClick(review.dialog.annotations, Qt.MouseButton.LeftButton)
    assert all(not item.isVisible() for item in review.dialog.image.overlay_items)
    assert (tmp_path / "evidence-12.png").read_bytes() == original_bytes


def test_pending_and_failure_do_not_use_any_current_camera_frame(review, tmp_path):
    item = record(review, tmp_path, state="pending")
    review.show_record(item, latest=True)
    assert review.dialog.image.image_size is None
    assert "正在写入" in review.dialog.status.text()
    session = review.workspace.session
    session.set_evidence(item.id, {**item.evidence, "state": "saved"})
    wait_image(review)
    session.set_evidence(item.id, {**item.evidence, "state": "failed", "error": "磁盘已满"})
    review.refresh()
    assert review.dialog.image.image_size is None and "磁盘已满" in review.dialog.status.text()


def test_history_locks_while_latest_follow_can_be_reenabled(review, tmp_path):
    first = record(review, tmp_path, number=1)
    second = record(review, tmp_path, number=2)
    review.accepted(first)
    assert not review.dialog.isVisible()
    review.show_record(first)
    assert not review.dialog.follow.isChecked()
    review.accepted(second)
    assert review.record_id == first.id and review.latest_record_id == second.id
    QTest.mouseClick(review.dialog.follow, Qt.MouseButton.LeftButton)
    assert review.record_id == second.id
    wait_image(review)


def test_revision_distinguishes_original_geometry_and_record_value(review, tmp_path):
    item = record(review, tmp_path)
    review.show_record(item)
    wait_image(review)
    review.workspace.session.edit_measurement(item.id, diameter_um=24)
    review.refresh()
    assert "24.0000" in review.dialog.details.text()
    assert "已修订" in review.dialog.details.text()
    assert "原测值 20.0000" in review.dialog.details.text()
    assert review.dialog.image.overlay_items[1].line().p1() == QPointF(20, 30)


def test_legacy_full_frame_uses_zero_origin_and_does_not_invent_original_value(review, tmp_path):
    item = record(review, tmp_path, crop=False, with_original=False)
    review.show_record(item)
    wait_image(review)
    assert review.dialog.image.overlay_items[1].line().p1() == QPointF(120, 230)
    assert "旧记录未留存原测值" in review.dialog.details.text()
    assert "完整无损 PNG" in review.dialog.details.text()


def test_pinning_uses_owned_tool_without_global_topmost_and_preserves_suspend_preferences(review, tmp_path):
    item = record(review, tmp_path)
    review.show_record(item, latest=True)
    normal_width = review.dialog.width()
    QTest.mouseClick(review.dialog.pin, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert review.dialog.windowType() == Qt.WindowType.Tool
    assert not (review.dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
    assert review.dialog.parentWidget() is review.workspace.window()
    assert review.dialog.width() < normal_width
    assert review.dialog.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    revision = review.workspace.session.revision
    review.suspend()
    assert not review.dialog.isVisible() and review.preferences["open"]
    review.resume()
    assert review.dialog.isVisible() and review.workspace.session.revision == revision
    QTest.mouseClick(review.dialog.collapse, Qt.MouseButton.LeftButton)
    assert not review.preferences["open"]
    review.resume()
    assert not review.dialog.isVisible()


def test_preferences_restore_visible_pinned_window_and_clamp_offscreen_geometry(review, tmp_path):
    item = record(review, tmp_path)
    review.show_record(item, latest=True)
    QTest.mouseClick(review.dialog.pin, Qt.MouseButton.LeftButton)
    review.preferences["pinned_geometry"] = [90000, -90000, 450, 350]
    review.flush_preferences()
    review.close()
    restored = EvidenceReviewController(review.workspace, preferences_path=review.preferences_path)
    try:
        assert restored.preferences["pinned"] and restored.preferences["open"]
        restored.resume()
        available = restored.dialog.screen().availableGeometry()
        assert available.contains(restored.dialog.geometry())
        assert restored.latest_record_id == item.id
    finally:
        restored.close()


def test_untouched_trial_does_not_overwrite_another_trials_latest_preferences(review, tmp_path):
    untouched = EvidenceReviewController(review.workspace, preferences_path=review.preferences_path)
    try:
        item = record(review, tmp_path)
        review.show_record(item, latest=True)
        QTest.mouseClick(review.dialog.pin, Qt.MouseButton.LeftButton)
        review.flush_preferences()
        expected = json.loads(review.preferences_path.read_text(encoding="utf-8"))
        untouched.close()
        assert json.loads(review.preferences_path.read_text(encoding="utf-8")) == expected
    finally:
        untouched.close()


def test_two_used_trials_persist_last_user_choice_regardless_of_flush_order(review, tmp_path):
    other = EvidenceReviewController(review.workspace, preferences_path=review.preferences_path)
    try:
        item = record(review, tmp_path)
        review.show_record(item, latest=True)
        QTest.mouseClick(review.dialog.pin, Qt.MouseButton.LeftButton)
        review.suspend()
        other.resume()
        assert other.dialog.pin.isChecked() and other.dialog.windowType() == Qt.WindowType.Tool
        QTest.mouseClick(other.dialog.pin, Qt.MouseButton.LeftButton)
        QTest.mouseClick(other.dialog.annotations, Qt.MouseButton.LeftButton)
        other.flush_preferences()
        review.flush_preferences()
        values = json.loads(review.preferences_path.read_text(encoding="utf-8"))
        assert not values["pinned"] and not values["annotations"]
        other.suspend()
        review.resume()
        assert not review.dialog.pin.isChecked() and not review.dialog.annotations.isChecked()
        QTest.mouseClick(review.dialog.pin, Qt.MouseButton.LeftButton)
        other.close()
        assert json.loads(review.preferences_path.read_text(encoding="utf-8"))["pinned"]
    finally:
        other.close()


def test_evidence_zoom_and_mouse_pan_do_not_edit_measurement_or_image(review, tmp_path):
    item = record(review, tmp_path)
    review.show_record(item)
    wait_image(review)
    view = review.dialog.image
    view.actual_pixels()
    before = view.transform().m11()
    event = QWheelEvent(QPointF(80, 80), QPointF(80, 80), QPoint(), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
    QApplication.sendEvent(view.viewport(), event)
    assert view.transform().m11() > before
    view.zoom(3)
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    QTest.mouseMove(view.viewport(), QPoint(150, 140))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(150, 140))
    assert review.workspace.session.measurements[item.id] == item


def test_inconsistent_crop_dimensions_are_not_shown_with_wrong_overlay(review, tmp_path):
    item = record(review, tmp_path)
    review.workspace.session.set_evidence(item.id, {**item.evidence, "crop": [100, 200, 100, 100]})
    review.show_record(review.workspace.session.measurements[item.id])
    for _ in range(30):
        review.refresh()
        QTest.qWait(5)
    assert review.dialog.image.image_size is None
    assert "尺寸与裁剪元数据不一致" in review.dialog.status.text()


def test_stale_decode_cannot_replace_newly_selected_record(review, tmp_path, monkeypatch):
    from concurrent.futures import Future
    first = record(review, tmp_path, number=1)
    second = record(review, tmp_path, number=2)
    future = Future()
    calls = []
    def submit(function, path, expected_size):
        calls.append(path)
        return future
    monkeypatch.setattr(review._decoder, "submit", submit)
    review.show_record(first)
    review.show_record(second)
    assert len(calls) == 1  # no unbounded decoder work queue
    image = QImage(first.evidence["path"])
    future.set_result((image, None))
    review.refresh()
    assert review.dialog.image.image_size is None
    assert review._loading_key[0] == second.id and len(calls) == 2


def test_follow_latest_after_undo_uses_remaining_record(review, tmp_path):
    first = record(review, tmp_path, number=1)
    second = record(review, tmp_path, number=2)
    review.show_record(second, latest=True)
    review.workspace.session.undo()
    review.refresh()
    assert review.latest_record_id == first.id and review.record_id == first.id


def test_escape_collapses_without_restoring_on_workspace_return(review, tmp_path):
    item = record(review, tmp_path)
    review.show_record(item)
    QTest.keyClick(review.dialog, Qt.Key.Key_Escape)
    assert not review.preferences["open"] and not review.dialog.isVisible()
    review.suspend()
    review.resume()
    assert not review.dialog.isVisible()
