"""The album follows accepted records without decoding the entire project."""
from dataclasses import replace
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest
from PySide6.QtCore import QEventLoop, QMargins, QPoint, QRect, QTimer, Qt
from PySide6.QtGui import QColor, QImage, QWindowStateChangeEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.ui.measurement_evidence_gallery import (
    EvidenceGalleryController, _ThumbnailPlan, _decode_thumbnail, _probe_thumbnail,
    _fit_client_geometry, _windows_frame_margins,
)


def wait_events(ms=15):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        wait_events()
    assert predicate()


def freeze_measurement_clock(monkeypatch):
    import fdm.live_experiment.models as models
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(2026, 10, 20, 2, 0, 0, 123000, tzinfo=timezone.utc)
            return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)
    # MeasurementRecord captured _now as its dataclass default factory, so
    # control that factory's clock instead of replacing the function name.
    monkeypatch.setattr(models, "datetime", FrozenDateTime)


def session():
    value = ExperimentSession(evidence_options={"version": 1, "mode": "crop_jpeg"})
    value.add_operator("操作者一")
    value.set_materials(default_materials(), direction="warp")
    value.set_materials(default_materials(), direction="weft")
    return value


def workspace(value=None):
    result = QWidget()
    result.session = value or session()
    result.reviews = []
    result.evidence_review = SimpleNamespace(show_record=lambda record, **kwargs: result.reviews.append((record.id, kwargs)))
    result.show()
    return result


@pytest.fixture
def gallery(tmp_path):
    host = QWidget()
    host.show()
    work = workspace()
    result = EvidenceGalleryController(host, preferences_path=tmp_path / "gallery.json")
    result.bind(work)
    yield result
    result.close()
    work.close()
    work.deleteLater()
    host.close()
    host.deleteLater()
    wait_events()


def measurement(gallery, tmp_path, *, number=1, state="saved", path=None, color="#e0c5b2", **kwargs):
    path = path or tmp_path / f"image-{number}.png"
    if not Path(path).exists() and state == "saved":
        image = QImage(256, 256, QImage.Format.Format_RGB32)
        image.fill(QColor(color))
        assert image.save(str(path))
    evidence = dict(state=state, path=str(path), evidence_id=f"evidence-{number}",
                    crop=[100, 200, 256, 256], source_size=[768, 576], mode="crop_jpeg", codec="png")
    return gallery.workspace.session.add_measurement(20 + number / 100, "wool",
        endpoints=((120, 230), (150, 250)), evidence=evidence,
        provenance={"measurement_number": number}, **kwargs)


def load(gallery, item):
    gallery.refresh()
    gallery.open()
    wait_for(lambda: gallery.thumbnail(item)[0] is not None)
    return gallery.thumbnail(item)[0]


def controlled_decoder_and_paints(gallery, monkeypatch):
    """Advance real file stages explicitly and observe actual Qt card paints."""
    jobs, painted = [], []
    gallery._timer.stop()
    def submit(function, *args):
        future = Future()
        jobs.append((future, function, args))
        return future
    monkeypatch.setattr(gallery._decoder, "submit", submit)
    delegate = type(gallery.dialog.view.itemDelegate())
    original = delegate.paint
    def paint(self, painter, option, index):
        if self.controller is gallery:
            record = self.controller.proxy.record(index)
            if record is not None:
                painted.append(record.id)
        return original(self, painter, option, index)
    monkeypatch.setattr(delegate, "paint", paint)
    return jobs, painted


def complete_stage(gallery, job):
    future, function, args = job
    future.set_result(function(*args))
    gallery._tick()
    wait_events(25)


def test_album_defaults_to_all_people_directions_and_newest_first(gallery, tmp_path):
    first = measurement(gallery, tmp_path, number=1)
    value = gallery.workspace.session
    second_operator = value.add_operator("操作者二")
    value.active_operator_id = second_operator
    value.set_materials(default_materials(), direction="warp")
    value.set_materials(default_materials(), direction="weft")
    value.active_direction = "weft"
    second = measurement(gallery, tmp_path, number=2)
    value.adjust_count("wool", 1)
    measurement(gallery, tmp_path, number=3, state="disabled")
    gallery.open()
    assert gallery.proxy.rowCount() == 2
    assert gallery.proxy.record(gallery.proxy.index(0, 0)).id == second.id
    assert gallery.proxy.record(gallery.proxy.index(1, 0)).id == first.id
    assert gallery.dialog.operator.currentData() is None
    assert gallery.dialog.direction.currentData() is None
    gallery.dialog.direction.setCurrentIndex(gallery.dialog.direction.findData("warp"))
    assert gallery.proxy.rowCount() == 1
    assert gallery.proxy.record(gallery.proxy.index(0, 0)).id == first.id
    gallery.dialog.direction.setCurrentIndex(0)
    gallery.dialog.operator.setCurrentIndex(gallery.dialog.operator.findData(second_operator))
    assert gallery.proxy.rowCount() == 1


def test_identical_timestamps_use_persisted_accept_order_across_field_number_reset(gallery, tmp_path, monkeypatch):
    from fdm.live_experiment.persistence import load_session, save_session
    freeze_measurement_clock(monkeypatch)
    first = measurement(gallery, tmp_path, number=80, id="z-earlier", field_id="old-field")
    gallery.open()
    second = measurement(gallery, tmp_path, number=1, id="a-later", field_id="new-field")
    gallery.refresh()
    assert first.created_at == second.created_at
    assert [gallery.proxy.record(gallery.proxy.index(row, 0)).id for row in range(2)] == [second.id, first.id]
    saved = save_session(gallery.workspace.session, tmp_path / "same-time.fdmlive")
    gallery.workspace.session = load_session(saved)
    gallery.refresh()
    assert [gallery.proxy.record(gallery.proxy.index(row, 0)).id for row in range(2)] == [second.id, first.id]
    third = measurement(gallery, tmp_path, number=1, id="m-newest", field_id="third-field")
    gallery.refresh()
    assert [gallery.proxy.record(gallery.proxy.index(row, 0)).id for row in range(3)] == [third.id, second.id, first.id]


def test_sort_comparisons_do_not_walk_the_measurement_ledger(gallery, tmp_path, monkeypatch):
    freeze_measurement_clock(monkeypatch)
    measurement(gallery, tmp_path, number=1, id="z-earlier")
    measurement(gallery, tmp_path, number=2, id="a-later")
    gallery.open()
    ranks = gallery.proxy._accepted_rank
    class NoIteration(dict):
        def __iter__(self):
            raise AssertionError("A comparison must not rebuild insertion ranks")
    gallery.workspace.session.measurements = NoIteration(gallery.workspace.session.measurements)
    for _ in range(100):
        gallery.proxy.lessThan(gallery.model.index(0, 0), gallery.model.index(1, 0))
        gallery.proxy.refresh_order(gallery.workspace.session)
    assert gallery.proxy._accepted_rank is ranks


def test_shared_full_image_keeps_every_record_and_opens_locked_review(gallery, tmp_path):
    first = measurement(gallery, tmp_path, number=1)
    second = measurement(gallery, tmp_path, number=2, path=Path(first.evidence["path"]))
    gallery.open()
    assert gallery.proxy.rowCount() == 2
    gallery._review(gallery.proxy.index(1, 0))
    assert gallery.workspace.reviews == [(first.id, {"latest": False})]
    assert first.evidence["path"] == second.evidence["path"]
    gallery.dialog.view.setCurrentIndex(gallery.proxy.index(0, 0))
    QTest.keyClick(gallery.dialog.view, Qt.Key.Key_Return)
    assert gallery.workspace.reviews[-1] == (second.id, {"latest": False})


def test_actual_item_click_selects_and_double_click_opens_history_once(gallery, tmp_path):
    item = measurement(gallery, tmp_path)
    gallery.open()
    wait_events(50)
    view = gallery.dialog.view
    index = gallery.proxy.index(0, 0)
    position = view.visualRect(index).center()
    assert view.viewport().rect().contains(position)
    # QTest.mouseDClick emits only the double-click event, unlike an OS which
    # delivers the preceding first click. Reproduce that complete sequence.
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert view.currentIndex() == index
    assert gallery.workspace.reviews == []
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert gallery.workspace.reviews == [(item.id, {"latest": False})]


def test_crop_thumbnail_uses_saved_image_only_and_keeps_clean_bytes(gallery, tmp_path):
    item = measurement(gallery, tmp_path)
    original = Path(item.evidence["path"]).read_bytes()
    image = load(gallery, item)
    assert image.width() <= round(gallery.preferences["thumbnail_width"] * gallery.dialog.devicePixelRatioF())
    assert image.text("fdm_source_size") == "256,256"
    assert gallery.memory_bytes <= gallery.MEMORY_LIMIT
    gallery.dialog.annotations.setChecked(False)
    gallery.dialog.view.viewport().repaint()
    assert Path(item.evidence["path"]).read_bytes() == original


def test_header_is_visually_unchanged_and_shared_file_decodes_repaint_only_their_own_card(gallery, tmp_path, monkeypatch):
    first = measurement(gallery, tmp_path, number=1)
    second = measurement(gallery, tmp_path, number=2, path=Path(first.evidence["path"]))
    jobs, painted = controlled_decoder_and_paints(gallery, monkeypatch)
    gallery.open()
    wait_events(60)
    assert set(painted) == {first.id, second.id}
    painted.clear()
    complete_stage(gallery, jobs[0])  # Reading the header only starts decoding.
    assert painted == []
    assert gallery.thumbnail(second) == (None, None)
    complete_stage(gallery, jobs[1])
    assert painted and set(painted) == {second.id}
    assert gallery.thumbnail(second)[0] is not None
    assert gallery.thumbnail(first)[0] is None
    painted.clear()
    complete_stage(gallery, jobs[2])
    assert painted == []
    complete_stage(gallery, jobs[3])
    assert painted and set(painted) == {first.id}
    assert gallery.thumbnail(first)[0] is not None


def test_preloaded_offscreen_thumbnail_does_not_repaint_visible_cards(gallery, tmp_path, monkeypatch):
    first = measurement(gallery, tmp_path)
    value = gallery.workspace.session
    for index in range(50):
        value.measurements[f"preload-{index:03}"] = replace(first, id=f"preload-{index:03}")
    value.revision += 1
    _jobs, painted = controlled_decoder_and_paints(gallery, monkeypatch)
    gallery.open()
    wait_events(60)
    index = gallery.proxy.index(35, 0)
    assert not gallery.dialog.view.visualRect(index).intersects(gallery.dialog.view.viewport().rect())
    record = gallery.proxy.record(index)
    image = QImage(224, 168, QImage.Format.Format_RGB32)
    image.fill(QColor("#e0c5b2"))
    painted.clear()
    gallery._cache_result(gallery._key(record), image, None)
    wait_events(25)
    assert painted == []
    assert gallery.thumbnail(record)[0] is not None


@pytest.mark.parametrize("completed_state", ["saved", "failed"])
def test_evidence_state_changes_update_cards_and_filter_membership_via_model(gallery, tmp_path, monkeypatch, completed_state):
    first = measurement(gallery, tmp_path, number=1, state="pending")
    second = measurement(gallery, tmp_path, number=2, state="pending")
    _jobs, painted = controlled_decoder_and_paints(gallery, monkeypatch)
    gallery.open()
    gallery.dialog.state.setCurrentIndex(gallery.dialog.state.findData("pending"))
    wait_events(60)
    assert gallery.proxy.rowCount() == 2
    assert set(painted) == {first.id, second.id}
    if completed_state == "saved":
        image = QImage(256, 256, QImage.Format.Format_RGB32)
        image.fill(QColor("#d0e8d8"))
        assert image.save(first.evidence["path"])
    gallery.workspace.session.set_evidence(first.id, {**first.evidence, "state": completed_state})
    gallery.refresh()
    wait_events(25)
    assert gallery.proxy.rowCount() == 1
    assert gallery.proxy.record(gallery.proxy.index(0, 0)).id == second.id
    gallery.dialog.state.setCurrentIndex(gallery.dialog.state.findData(completed_state))
    painted.clear()
    wait_events(25)
    assert gallery.proxy.rowCount() == 1
    index = gallery.proxy.index(0, 0)
    assert gallery.proxy.record(index).id == first.id
    assert first.id in painted
    text = gallery.proxy.data(index)
    assert "写入中" not in text
    assert ("写入失败" in text) == (completed_state == "failed")


def test_header_error_repaints_card_and_read_error_filter_reorders_correctly(gallery, tmp_path, monkeypatch):
    older = measurement(gallery, tmp_path, number=1)
    missing = measurement(gallery, tmp_path, number=2)
    Path(missing.evidence["path"]).unlink()
    jobs, painted = controlled_decoder_and_paints(gallery, monkeypatch)
    gallery.open()
    wait_events(60)
    painted.clear()
    complete_stage(gallery, jobs[0])
    assert painted and set(painted) == {missing.id}
    assert gallery.thumbnail(missing)[1] is not None
    gallery.dialog.state.setCurrentIndex(gallery.dialog.state.findData("read_error"))
    wait_events(25)
    assert gallery.proxy.rowCount() == 1
    assert gallery.proxy.record(gallery.proxy.index(0, 0)).id == missing.id
    # Adding an older checked error inserts it after the existing newest one;
    # resolving the newest error removes it through proxy structural signals.
    gallery._cache_result(gallery._key(older), None, "测试缺图")
    wait_events(25)
    assert [gallery.proxy.record(gallery.proxy.index(i, 0)).id for i in range(2)] == [missing.id, older.id]
    recovered = QImage(224, 168, QImage.Format.Format_RGB32)
    recovered.fill(QColor("#e0c5b2"))
    gallery._cache_result(gallery._key(missing), recovered, None)
    painted.clear()
    wait_events(25)
    assert gallery.proxy.rowCount() == 1
    assert gallery.proxy.record(gallery.proxy.index(0, 0)).id == older.id
    assert older.id in painted


def test_pending_failed_disabled_and_missing_states_do_not_decode_live_frames(gallery, tmp_path):
    pending = measurement(gallery, tmp_path, number=1, state="pending")
    failed = measurement(gallery, tmp_path, number=2, state="failed")
    measurement(gallery, tmp_path, number=3, state="disabled")
    missing = measurement(gallery, tmp_path, number=4)
    Path(missing.evidence["path"]).unlink()
    gallery.open()
    wait_for(lambda: gallery.thumbnail(missing)[1] is not None)
    assert gallery.proxy.rowCount() == 3
    assert gallery.thumbnail(pending) == (None, None)
    assert gallery.thumbnail(failed) == (None, None)
    assert "缺失" in gallery.thumbnail(missing)[1]
    assert gallery.decode_count == 0
    gallery.dialog.state.setCurrentIndex(gallery.dialog.state.findData("failed"))
    assert gallery.proxy.rowCount() == 1
    assert gallery.proxy.record(gallery.proxy.index(0, 0)).id == failed.id


def test_read_error_filter_only_contains_already_checked_files(gallery, tmp_path):
    missing = measurement(gallery, tmp_path, number=1)
    pending = measurement(gallery, tmp_path, number=2, state="pending")
    Path(missing.evidence["path"]).unlink()
    revision = gallery.workspace.session.revision
    gallery.open()
    wait_for(lambda: gallery.thumbnail(missing)[1] is not None)
    gallery.dialog.state.setCurrentIndex(gallery.dialog.state.findData("read_error"))
    assert gallery.proxy.rowCount() == 1
    assert gallery.proxy.record(gallery.proxy.index(0, 0)).id == missing.id
    assert "未扫描全部" in gallery.dialog.status.text()
    assert gallery.decode_count == 0
    assert gallery.workspace.session.revision == revision
    assert pending.id not in gallery._known_read_errors


def test_save_as_drops_read_error_from_old_path_without_a_domain_revision(gallery, tmp_path):
    missing = measurement(gallery, tmp_path)
    data = Path(missing.evidence["path"]).read_bytes()
    Path(missing.evidence["path"]).unlink()
    gallery.open()
    wait_for(lambda: gallery.thumbnail(missing)[1] is not None)
    gallery.dialog.state.setCurrentIndex(gallery.dialog.state.findData("read_error"))
    assert gallery.proxy.rowCount() == 1
    value = gallery.workspace.session
    revision = value.revision
    new_path = tmp_path / "recovered.png"
    new_path.write_bytes(data)
    value.measurements[missing.id] = replace(missing, evidence={**missing.evidence, "path": str(new_path)})
    gallery.refresh()
    assert gallery.proxy.rowCount() == 0
    assert missing.id not in gallery._known_read_errors
    assert value.revision == revision


def test_hidden_gallery_never_starts_file_decoding(gallery, tmp_path):
    measurement(gallery, tmp_path)
    gallery.refresh()
    wait_events(100)
    assert gallery._future is None and gallery.decode_count == 0
    assert gallery.memory_bytes == 0


def test_async_evidence_completion_and_undo_redo_update_album(gallery, tmp_path):
    item = measurement(gallery, tmp_path, state="pending")
    gallery.open()
    assert gallery.proxy.rowCount() == 1
    image = QImage(256, 256, QImage.Format.Format_RGB32)
    image.fill(QColor("#bde1c0"))
    assert image.save(item.evidence["path"])
    value = gallery.workspace.session
    value.set_evidence(item.id, {**item.evidence, "state": "saved"})
    gallery.refresh()
    current = value.measurements[item.id]
    wait_for(lambda: gallery.thumbnail(current)[0] is not None)
    value.undo()
    gallery.refresh()
    assert gallery.proxy.rowCount() == 0
    value.redo()
    gallery.refresh()
    assert gallery.proxy.rowCount() == 1
    assert gallery.proxy.record(gallery.proxy.index(0, 0)).evidence["state"] == "saved"


def test_save_as_rebase_without_revision_reloads_current_path(gallery, tmp_path):
    item = measurement(gallery, tmp_path)
    load(gallery, item)
    before = gallery.decode_count
    path = tmp_path / "saved-as.png"
    image = QImage(256, 256, QImage.Format.Format_RGB32)
    image.fill(QColor("#88ddaa"))
    assert image.save(str(path))
    value = gallery.workspace.session
    revision = value.revision
    changed = replace(item, evidence={**item.evidence, "path": str(path)})
    value.measurements[item.id] = changed  # Persistence-only rebase has no domain revision.
    gallery.refresh()
    wait_for(lambda: gallery.decode_count > before and gallery.thumbnail(changed)[0] is not None)
    assert value.revision == revision
    assert gallery.thumbnail(changed)[0].pixelColor(20, 20) == QColor("#88ddaa")


def test_ten_thousand_records_only_request_visible_and_adjacent_rows(gallery, tmp_path):
    first = measurement(gallery, tmp_path)
    value = gallery.workspace.session
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    value.measurements = {f"large-{index:05}": replace(first, id=f"large-{index:05}",
        created_at=(base + timedelta(seconds=index)).isoformat(),
        provenance={"measurement_number": index + 1}) for index in range(10000)}
    value.revision += 1
    gallery.open()
    wait_for(lambda: gallery.decode_count >= 3)
    assert gallery.proxy.rowCount() == 10000
    assert gallery.decode_count < 100
    assert len(gallery._queue) <= gallery.MAX_QUEUED
    assert len(gallery._cache) < 100
    assert gallery.memory_bytes <= gallery.MEMORY_LIMIT
    gallery.dialog.view.scrollTo(gallery.proxy.index(5000, 0))
    wait_events(100)
    assert len(gallery._queue) <= gallery.MAX_QUEUED
    assert gallery.decode_count < 150
    assert gallery.memory_bytes <= gallery.MEMORY_LIMIT


def test_new_measurements_do_not_steal_history_position(gallery, tmp_path):
    first = measurement(gallery, tmp_path)
    value = gallery.workspace.session
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(80):
        key = f"history-{index:03}"
        value.measurements[key] = replace(first, id=key, created_at=(base + timedelta(seconds=index)).isoformat())
    value.revision += 1
    gallery.open()
    wait_events(100)
    gallery.dialog.view.scrollTo(gallery.proxy.index(45, 0), gallery.dialog.view.ScrollHint.PositionAtTop)
    wait_events()
    anchor = gallery._anchor()
    assert gallery.dialog.view.verticalScrollBar().value() > 0
    measurement(gallery, tmp_path, number=2)
    gallery.refresh()
    wait_events(50)
    source_row = gallery.model.rows_by_key[anchor[0]]
    anchored = gallery.proxy.mapFromSource(gallery.model.index(source_row, 0))
    # One inserted item can change the column in a wrapping grid; retain the
    # same historical record's vertical position rather than the leftmost ID.
    assert abs(gallery.dialog.view.visualRect(anchored).top() - anchor[1]) <= 1
    assert gallery._new_items == 1
    assert "新增 1 项" in gallery.dialog.latest.text()
    gallery._go_latest()
    assert gallery.dialog.view.verticalScrollBar().value() == 0 and gallery._new_items == 0


def test_switch_trial_resets_filters_and_discards_old_decode(gallery, tmp_path, monkeypatch):
    import fdm.ui.measurement_evidence_gallery as module
    started, release = threading.Event(), threading.Event()
    original = module._decode_thumbnail
    def held(plan):
        started.set()
        release.wait(3)
        return original(plan)
    monkeypatch.setattr(module, "_decode_thumbnail", held)
    item = measurement(gallery, tmp_path)
    gallery.open()
    wait_for(started.is_set)
    previous_generation = gallery._generation
    other = workspace()
    try:
        gallery.dialog.direction.setCurrentIndex(1)
        gallery.bind(other)
        assert gallery._generation > previous_generation
        assert gallery.dialog.direction.currentData() is None
        assert gallery.proxy.rowCount() == 0
        release.set()
        wait_for(lambda: gallery._future is None)
        assert not gallery._cache and gallery.memory_bytes == 0
    finally:
        release.set()
        other.close()
        other.deleteLater()


def test_oversized_image_rejected_before_decode_and_cache_evicted_for_reservation(gallery, tmp_path, monkeypatch):
    import fdm.ui.measurement_evidence_gallery as module
    item = measurement(gallery, tmp_path)
    monkeypatch.setattr(module, "_probe_thumbnail", lambda *args: (
        _ThumbnailPlan(args[0], (20000, 20000), (224, 224), gallery.MEMORY_LIMIT + 1), None))
    gallery.open()
    wait_for(lambda: gallery.thumbnail(item)[1] is not None)
    assert gallery.decode_count == 0 and "较大" in gallery.thumbnail(item)[1]
    image = QImage(1024, 1024, QImage.Format.Format_RGB32)
    for index in range(20):
        gallery._cache_result(("test", index), image, None)
    assert gallery.memory_bytes <= gallery.MEMORY_LIMIT
    assert gallery._reserve(48 * 1024 * 1024)
    assert gallery.memory_bytes <= gallery.MEMORY_LIMIT
    assert gallery._cache_bytes <= 16 * 1024 * 1024


def test_probe_validates_dimensions_and_decoder_returns_bounded_image(tmp_path):
    path = tmp_path / "full.png"
    image = QImage(768, 576, QImage.Format.Format_RGB32)
    image.fill(QColor("#bcddee"))
    assert image.save(str(path))
    plan, error = _probe_thumbnail(str(path), (768, 576), (224, 168))
    assert error is None and plan.reserved_bytes >= 768 * 576 * 4
    thumb, error = _decode_thumbnail(plan)
    assert error is None and (thumb.width(), thumb.height()) == (224, 168)
    plan, error = _probe_thumbnail(str(path), (256, 256), (224, 168))
    assert plan is None and "不一致" in error


def test_high_bit_depth_temporary_decode_memory_is_reserved(tmp_path):
    path = tmp_path / "sixteen-bit.png"
    image = QImage(768, 576, QImage.Format.Format_RGBA64)
    image.fill(QColor("#bcddee"))
    assert image.save(str(path))
    plan, error = _probe_thumbnail(str(path), (768, 576), (224, 168))
    assert error is None and plan.pixel_bytes == 8
    assert plan.reserved_bytes >= (768 * 576 + 224 * 168 * 2) * 8
    thumb, error = _decode_thumbnail(plan)
    assert error is None and thumb.sizeInBytes() == 224 * 168 * 8


def test_suspend_resume_user_close_and_preferences_do_not_modify_trial(gallery, tmp_path):
    item = measurement(gallery, tmp_path)
    revision = gallery.workspace.session.revision
    gallery.open()
    assert gallery.dialog.parentWidget() is gallery.host.window()
    assert not gallery.dialog.isModal()
    assert gallery.dialog.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
    assert not gallery.dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    gallery.dialog.size_slider.setValue(288)
    gallery.dialog.annotations.setChecked(False)
    gallery.dialog.move(45, 55)
    gallery.suspend()
    assert gallery.is_open and not gallery.dialog.isVisible()
    gallery.resume()
    assert gallery.dialog.isVisible()
    gallery.dialog.close()
    assert not gallery.is_open
    gallery.resume()
    assert not gallery.dialog.isVisible()
    gallery.flush_preferences()
    restored = EvidenceGalleryController(gallery.host, preferences_path=gallery.preferences_path)
    try:
        assert restored.preferences["thumbnail_width"] == 288
        assert not restored.preferences["annotations"]
        restored.open(gallery.workspace)
        wait_events()
        assert restored.dialog.screen().availableGeometry().contains(restored.dialog.frameGeometry())
    finally:
        restored.close()
    assert gallery.workspace.session.revision == revision


@pytest.mark.parametrize("margins", (QMargins(1, 24, 1, 1), QMargins(8, 58, 8, 8),
                                     QMargins(12, 84, 12, 12)))
@pytest.mark.parametrize("area", (QRect(0, 0, 1600, 860), QRect(-1600, 80, 1600, 860)))
@pytest.mark.parametrize("rect", (QRect(8, 8, 1040, 720), QRect(9000, 9000, 4000, 3000)))
def test_geometry_fit_accounts_for_all_native_edges_and_shrinks_client(margins, area, rect):
    fitted = _fit_client_geometry(rect, area, margins)
    assert area.adjusted(8, 8, -8, -8).contains(fitted.marginsAdded(margins))
    assert fitted.width() <= area.width() - 16 - margins.left() - margins.right()
    assert fitted.height() <= area.height() - 16 - margins.top() - margins.bottom()


@pytest.mark.parametrize("dpr,expected", ((1., (13, 58, 13, 13)),
                                         (1.25, (11, 47, 11, 11)),
                                         (2., (7, 29, 7, 7))))
@pytest.mark.parametrize("origin", ((8, 66), (-2500, -240)))
def test_native_resize_border_uses_current_pixel_scale_and_rounds_outwards(dpr, expected, origin):
    x, y = origin
    bounds = ((x - 13, y - 58, x + 1040 + 13, y + 720 + 13), origin, (1040, 720))
    margins = _windows_frame_margins(123, dpr, read_bounds=lambda hwnd: bounds)
    assert margins == QMargins(*expected)
    # Cover a negative-coordinate monitor without confusing pixels and logical units.
    area = QRect(-2000, -400, 1600, 1000)
    fitted = _fit_client_geometry(QRect(-2500, -1000, 3000, 2000), area, margins)
    assert area.adjusted(8, 8, -8, -8).contains(fitted.marginsAdded(margins))


def test_native_query_failure_preserves_qt_frame_margins_and_retries(gallery, monkeypatch):
    from fdm.ui import measurement_evidence_gallery as module

    gallery.open()
    wait_events()
    qt = gallery.dialog.windowHandle().frameMargins()
    calls = []
    def unavailable(hwnd):
        calls.append(hwnd)
        raise OSError("HWND is not mapped yet")
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(module, "_read_windows_window_bounds", unavailable)
    assert gallery._frame_margins() == qt
    assert gallery._frame_margins() == qt
    assert len(calls) == 2
    assert all(value == int(gallery.dialog.winId()) for value in calls)
    monkeypatch.setattr(module, "_read_windows_window_bounds",
                        lambda hwnd: ((-13, -58, 1053, 733), (0, 0), (1040, 720)))
    native = _windows_frame_margins(int(gallery.dialog.winId()), gallery.dialog.devicePixelRatioF())
    assert gallery._frame_margins() == QMargins(max(qt.left(), native.left()), max(qt.top(), native.top()),
                                               max(qt.right(), native.right()), max(qt.bottom(), native.bottom()))


@pytest.mark.parametrize("dpr", (0., -1., float("nan"), float("inf")))
def test_native_query_rejects_invalid_pixel_scale(dpr):
    assert _windows_frame_margins(123, dpr, read_bounds=lambda hwnd: None) == QMargins()


@pytest.mark.parametrize("stored", ([8, 8, 1040, 720], [-5000, -5000, 3000, 2000],
                                   [9000, 9000, 1040, 720]))
def test_legacy_client_preferences_restore_real_native_frame_inside_screen(gallery, tmp_path, stored):
    path = tmp_path / "legacy-gallery.json"
    path.write_text(json.dumps({"version": 1, "geometry": stored}), encoding="utf-8")
    revision = gallery.workspace.session.revision
    restored = EvidenceGalleryController(gallery.host, preferences_path=path)
    try:
        restored.open(gallery.workspace)
        wait_events()
        assert restored.dialog.windowHandle() is not None
        assert restored.dialog.screen().availableGeometry().contains(restored.dialog.frameGeometry())
        restored.flush_preferences()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["version"] == 1
        assert QRect(*saved["geometry"]) == restored.dialog.geometry()
        assert gallery.workspace.session.revision == revision
    finally:
        restored.close()


def test_geometry_corrects_resume_and_removed_screen_without_snapping_user_moves(gallery, monkeypatch):
    revision = gallery.workspace.session.revision
    gallery.open()
    wait_events()
    applied = []
    original = gallery._apply_geometry
    monkeypatch.setattr(gallery, "_apply_geometry", lambda rect: (applied.append(rect), original(rect))[-1])
    gallery.dialog.move(-300, -200)
    wait_events()
    assert not applied  # No correction loop; native frame rounding can adjust coordinates.
    assert gallery.dialog.frameGeometry().left() < 0
    gallery.suspend()
    gallery.resume()
    wait_events()
    assert gallery.dialog.screen().availableGeometry().contains(gallery.dialog.frameGeometry())
    gallery.preferences["geometry"] = [9000, 9000, 3000, 2000]
    gallery._screen_removed()
    wait_events()
    assert gallery.dialog.screen().availableGeometry().contains(gallery.dialog.frameGeometry())
    assert gallery.workspace.session.revision == revision


@pytest.mark.parametrize("state", ("maximized", "fullscreen"))
def test_geometry_skips_native_managed_states_and_refits_normal_restore(gallery, monkeypatch, state):
    revision = gallery.workspace.session.revision
    gallery.open()
    wait_events()
    normal_size = gallery.dialog.size()
    gallery.dialog.move(-300, -200)
    (gallery.dialog.showMaximized if state == "maximized" else gallery.dialog.showFullScreen)()
    wait_events()
    managed_geometry = gallery.dialog.geometry()
    assert QRect(*gallery.preferences["geometry"]).size() == normal_size
    applied = []
    original = gallery._apply_geometry
    monkeypatch.setattr(gallery, "_apply_geometry", lambda rect: (applied.append(rect), original(rect))[-1])
    gallery._restore_geometry(create_native=True)
    gallery._clamp_visible_geometry()
    gallery._screen_removed()
    wait_events()
    assert not applied
    assert gallery.dialog.geometry() == managed_geometry
    gallery.dialog.showNormal()
    wait_events()
    assert applied
    assert gallery.dialog.size() == normal_size
    assert gallery.dialog.screen().availableGeometry().contains(gallery.dialog.frameGeometry())
    assert gallery.workspace.session.revision == revision


def test_normal_transition_does_not_save_intermediate_maximized_client_size(gallery):
    gallery.open()
    wait_events()
    normal = QRect(gallery.dialog.geometry())
    revision = gallery.workspace.session.revision
    # Windows may report "normal" while the resize events still carry the
    # previous maximized client size. Deliver that exact event order.
    QApplication.sendEvent(gallery.dialog, QWindowStateChangeEvent(Qt.WindowState.WindowMaximized))
    gallery.dialog.resize(normal.width() + 200, normal.height() + 200)
    assert QRect(*gallery.preferences["geometry"]).size() == normal.size()
    assert gallery._normal_restore_rect == normal
    wait_events()
    assert gallery.dialog.size() == normal.size()
    assert gallery._normal_restore_rect is None
    assert gallery.dialog.screen().availableGeometry().contains(gallery.dialog.frameGeometry())
    assert gallery.workspace.session.revision == revision


@pytest.mark.parametrize("native_state,qt_state", (
    ((True, False), Qt.WindowState.WindowMaximized),
    ((False, True), Qt.WindowState.WindowMinimized),
))
def test_native_state_precedes_qt_resize_and_move_without_overwriting_normal_geometry(
        gallery, monkeypatch, native_state, qt_state):
    gallery.open()
    wait_events()
    normal = QRect(gallery.dialog.geometry())
    revision = gallery.workspace.session.revision
    state = {"native": (False, False), "qt": Qt.WindowState.WindowNoState}
    monkeypatch.setattr(gallery, "_native_managed_state", lambda: state["native"])
    monkeypatch.setattr(gallery.dialog, "isMaximized",
                        lambda: state["qt"] == Qt.WindowState.WindowMaximized)
    monkeypatch.setattr(gallery.dialog, "isMinimized",
                        lambda: state["qt"] == Qt.WindowState.WindowMinimized)
    # The observed WM/UIA order has an unreliable normalGeometry during the
    # first Resize/Move. It becomes authoritative at WindowStateChange only.
    monkeypatch.setattr(gallery.dialog, "normalGeometry",
                        lambda: normal if state["qt"] != Qt.WindowState.WindowNoState
                        else gallery.dialog.geometry())
    for _ in range(2):
        state["native"] = native_state
        gallery.dialog.resize(normal.width() + 200, normal.height() + 200)
        gallery.dialog.move(normal.x() + 30, normal.y() + 30)
        assert QRect(*gallery.preferences["geometry"]) == normal
        assert gallery.dialog.normalGeometry() != normal
        gallery._clamp_visible_geometry()
        assert gallery.dialog.size() != normal.size()
        state["qt"] = qt_state
        QApplication.sendEvent(gallery.dialog, QWindowStateChangeEvent(Qt.WindowState.WindowNoState))
        gallery.flush_preferences()
        assert QRect(*json.loads(gallery.preferences_path.read_text())["geometry"]) == normal
        assert gallery.preferences["maximized"] == native_state[0]
        state["native"], state["qt"] = (False, False), Qt.WindowState.WindowNoState
        QApplication.sendEvent(gallery.dialog, QWindowStateChangeEvent(qt_state))
        wait_events()
        assert gallery.dialog.geometry() == normal
        assert gallery.dialog.screen().availableGeometry().contains(gallery.dialog.frameGeometry())
    assert gallery.workspace.session.revision == revision


def test_qt_transition_recovers_normal_geometry_when_native_state_query_fails(gallery, monkeypatch):
    from fdm.ui import measurement_evidence_gallery as module

    gallery.open()
    wait_events()
    normal = QRect(gallery.dialog.geometry())
    revision = gallery.workspace.session.revision
    def unavailable(hwnd):
        raise OSError("Native state query failed")
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(module, "_read_windows_managed_state", unavailable)
    assert gallery._native_managed_state() == (False, False)
    # If the OS query fails before WM state arrives, use Qt's stable normal
    # geometry at the state transition instead of retaining the transient size.
    gallery.preferences["geometry"] = [0, 0, normal.width() + 200, normal.height() + 200]
    gallery.dialog.showMaximized()
    wait_events()
    assert QRect(*gallery.preferences["geometry"]) == normal
    # Return to the actual host platform for native frame queries on restore.
    monkeypatch.undo()
    gallery.dialog.showNormal()
    wait_events()
    assert gallery.dialog.size() == normal.size()
    assert gallery.dialog.screen().availableGeometry().contains(gallery.dialog.frameGeometry())
    assert gallery.workspace.session.revision == revision


def test_reopening_retries_a_restored_missing_image(gallery, tmp_path):
    item = measurement(gallery, tmp_path)
    data = Path(item.evidence["path"]).read_bytes()
    Path(item.evidence["path"]).unlink()
    gallery.open()
    wait_for(lambda: gallery.thumbnail(item)[1] is not None)
    gallery.dialog.close()
    Path(item.evidence["path"]).write_bytes(data)
    gallery.open()
    wait_for(lambda: gallery.thumbnail(item)[0] is not None)


def test_close_is_idempotent_and_empty_album_never_changes_domain(gallery):
    revision = gallery.workspace.session.revision
    gallery.open()
    assert gallery.proxy.rowCount() == 0
    assert "只影响之后" in gallery.dialog.status.text()
    assert "未保存的图片无法补回" in gallery.dialog.status.text()
    assert gallery.workspace.session.revision == revision
    gallery.close()
    gallery.close()
    assert gallery.workspace is None and gallery.memory_bytes == 0
