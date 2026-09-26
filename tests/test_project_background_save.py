from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise
from threading import Event, get_ident
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest

from fdm.geometry import Line, Point
from fdm.models import (
    Calibration,
    ImageDocument,
    Measurement,
    OverlayAnnotation,
    ProjectState,
    project_assets_root,
)
from fdm.project_io import ProjectIO
from fdm.services.raster_io import numpy_to_raster_plane
from fdm.ui import project_save_coordinator as saving
from fdm.ui.image_loader import ImageLoadRequest
from fdm.ui.main_window import MainWindow
from fdm.watermark import WatermarkSpec


def wait_for(app, condition, timeout=6):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            raise AssertionError("Qt condition timed out")
        app.processEvents()
        time.sleep(0.003)
    app.processEvents()


def add_image(window, path):
    image = QImage(120, 90, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    image.save(str(path))
    document = ImageDocument(id=path.stem, path=str(path), image_size=(120, 90))
    window._add_loaded_document(
        ImageLoadRequest(path=str(path), document=document), image
    )
    return document


@pytest.fixture
def window(tmp_path, desktop_application):
    window = MainWindow()
    add_image(window, tmp_path / "first.png")
    window._project_path = tmp_path / "项目.fdmproj"
    window._mark_project_saved()
    yield window
    wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    window.project_save_coordinator.cancel_transition()
    window._reset_workspace()
    window.close()


@contextmanager
def paused_writer(monkeypatch, app):
    entered, release = Event(), Event()
    original = saving.write_project_save
    calls = []

    def write(request):
        calls.append((request, get_ident()))
        entered.set()
        assert release.wait(5), "test must release writer"
        return original(request)

    with monkeypatch.context() as patcher:
        patcher.setattr(saving, "write_project_save", write)
        try:
            yield entered, release, calls
        finally:
            release.set()
            # Ensure completed QThreads drain before the patch is removed.
            for _ in range(10):
                app.processEvents()
                time.sleep(0.003)


def test_slow_save_keeps_events_alive_and_preserves_newer_edit(
    window, monkeypatch, desktop_application
):
    doc = window.current_document()
    doc.watermark = WatermarkSpec(enabled=True, text="before")
    doc.mark_session_dirty()
    saved_stamp = doc.state_stamp
    beats = []
    timer = QTimer(window)
    timer.setInterval(20)
    timer.timeout.connect(lambda: beats.append(time.monotonic()))
    timer.start()
    with paused_writer(monkeypatch, desktop_application) as (entered, release, calls):
        window.save_project_action.trigger()
        wait_for(desktop_application, entered.is_set)
        wait_for(desktop_application, lambda: len(beats) >= 100)
        assert window._save_status_indicator._spin.isActive()
        assert calls[0][1] != get_ident()
        doc.add_measurement(
            Measurement(
                "new", doc.id, None, "manual", line_px=Line(Point(1, 1), Point(20, 1))
            )
        )
        doc.overlay_annotations.append(
            OverlayAnnotation("note", doc.id, "text", "new annotation")
        )
        doc.watermark = replace(doc.watermark, text="after")
        doc.calibration = Calibration("manual", 12, "nm", "new")
        doc.metadata["new_metadata"] = "keep me"
        doc.mark_calibration_dirty()
        window.current_canvas().set_view_zoom(2)
        release.set()
        wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    restored = ProjectIO.load(window._project_path).documents[0]
    assert restored.watermark.text == "before" and not restored.measurements
    assert doc.watermark.text == "after" and doc.measurements[0].id == "new"
    assert doc.metadata["new_metadata"] == "keep me" and doc.calibration.unit == "nm"
    assert doc.dirty_flags.session_dirty and doc.dirty_flags.calibration_dirty
    assert doc.saved_state_stamp == saved_stamp
    assert window.project_save_coordinator.status.phase == "saved_newer"
    assert not window._save_status_indicator._spin.isActive()
    assert max(b - a for a, b in pairwise(beats)) < 0.25
    doc.restore_state_stamp(saved_stamp)
    assert not doc.dirty_flags.session_dirty
    timer.stop()


def test_pending_save_uses_last_keypress_snapshot_only(
    window, monkeypatch, desktop_application
):
    doc = window.current_document()
    with paused_writer(monkeypatch, desktop_application) as (entered, release, calls):
        assert window.request_save_project()
        wait_for(desktop_application, entered.is_set)
        assert window.request_save_project()
        assert window.project_save_coordinator.pending is None
        for text in ("second", "third"):
            doc.watermark = WatermarkSpec(enabled=True, text=text)
            doc.mark_session_dirty()
            assert window.request_save_project()
        doc.watermark = replace(doc.watermark, text="not requested")
        doc.mark_session_dirty()
        release.set()
        wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
        assert len(calls) == 2
    assert ProjectIO.load(window._project_path).documents[0].watermark.text == "third"
    assert doc.watermark.text == "not requested" and doc.dirty_flags.session_dirty


def test_added_image_during_first_save_stays_unsaved(
    window, tmp_path, monkeypatch, desktop_application
):
    window._project_path = None
    path = tmp_path / "首次.fdmproj"
    monkeypatch.setattr(window, "_select_project_save_path", lambda _: str(path))
    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        assert window.request_save_project()
        wait_for(desktop_application, entered.is_set)
        second = add_image(window, tmp_path / "second.png")
        release.set()
        wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert len(ProjectIO.load(path).documents) == 1
    assert window.current_document() is second and window._project_dirty()
    assert window.request_save_project()
    wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert len(ProjectIO.load(path).documents) == 2 and not window._project_dirty()


def test_first_save_cancel_restores_status(window, monkeypatch):
    window._project_path = None
    window.project_save_coordinator.refresh_dirty()
    previous = window.project_save_coordinator.status
    monkeypatch.setattr(window, "_select_project_save_path", lambda _: "")
    assert not window.request_save_project()
    assert window.project_save_coordinator.status == previous
    assert not window.project_save_coordinator.busy


def test_failure_drops_pending_preserves_old_bytes_and_retries(
    window, monkeypatch, desktop_application
):
    assert window.save_project()
    path = window._project_path
    previous = path.read_bytes()
    doc = window.current_document()
    doc.mark_session_dirty()
    transitions = []
    with paused_writer(monkeypatch, desktop_application) as (entered, release, calls):
        with patch.object(ProjectIO, "save_payload", side_effect=OSError("disk full")):
            assert window.request_save_project()
            wait_for(desktop_application, entered.is_set)
            doc.mark_session_dirty()
            window.request_save_project()
            window._defer_for_project_save(lambda: transitions.append(True), "关闭软件")
            release.set()
            wait_for(
                desktop_application, lambda: not window.project_save_coordinator.busy
            )
        assert len(calls) == 1
    assert path.read_bytes() == previous and doc.dirty_flags.session_dirty
    assert not transitions and window.project_save_coordinator.pending is None
    assert window.project_save_coordinator.status.phase == "failed"
    assert "disk full" in window._save_status_indicator.toolTip()
    window._save_status_indicator.retryRequested.emit()
    wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert (
        window.project_save_coordinator.status.phase == "saved"
        and not doc.dirty_flags.session_dirty
    )


@pytest.mark.parametrize("cancel", [True, False])
def test_close_waits_and_can_be_cancelled(
    window, monkeypatch, desktop_application, cancel
):
    doc_id = window.current_document().id
    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        window.request_save_project()
        wait_for(desktop_application, entered.is_set)
        window.close_current_document()
        assert window.project.get_document(doc_id) is not None
        assert window.project_save_coordinator.status.transition
        if cancel:
            window._save_status_indicator.cancelTransitionRequested.emit()
        release.set()
        wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
        wait_for(
            desktop_application,
            lambda: window.project_save_coordinator._transition is None,
        )
    desktop_application.processEvents()
    assert (window.project.get_document(doc_id) is not None) == cancel


def test_close_rechecks_changes_made_after_snapshot(
    window, monkeypatch, desktop_application
):
    called = []
    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        window.request_save_project()
        wait_for(desktop_application, entered.is_set)
        window.close_current_document()
        window.current_document().mark_session_dirty()
        monkeypatch.setattr(
            window,
            "_confirm_close_documents",
            lambda docs, **kwargs: (
                (called.append(docs[0].dirty_flags.session_dirty) or False)
                if docs
                else True
            ),
        )
        release.set()
        wait_for(desktop_application, lambda: bool(called))
    assert called == [True] and window.current_document() is not None


def test_late_result_cannot_update_new_project(
    window, monkeypatch, desktop_application
):
    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        window.request_save_project()
        wait_for(desktop_application, entered.is_set)
        window.project = ProjectState.empty()
        window._project_path = None
        window.project_save_coordinator.reset()
        release.set()
        wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert window._project_path is None and not window.project.documents
    assert window.project_save_coordinator.last_metrics["stale"]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_native_asset_background_save_reuses_encoding(
    window, tmp_path, desktop_application, dtype
):
    doc = window.current_document()
    doc.source_type = "project_asset"
    doc.path = "imports/native.png" if dtype != np.float32 else "imports/native.tif"
    plane = numpy_to_raster_plane(np.arange(120 * 90, dtype=dtype).reshape(90, 120))
    doc.raster_pixel_type = plane.pixel_type
    window._rasters[doc.id] = plane
    window._raster_metadata[doc.id] = None
    assert window.request_save_project()
    wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert window.project_save_coordinator.status.phase == "saved"
    asset = project_assets_root(window._project_path) / doc.path
    before = asset.read_bytes(), asset.stat().st_mtime_ns
    with patch(
        "fdm.ui.project_session_controller.write_native_raster_asset",
        side_effect=AssertionError("must reuse"),
    ):
        assert window.request_save_project()
        wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert window.project_save_coordinator.status.phase == "saved"
    assert (asset.read_bytes(), asset.stat().st_mtime_ns) == before
    assert not window._project_dirty()


def test_indicator_width_focus_and_settled_state(window, desktop_application):
    indicator = window._save_status_indicator
    desktop_application.processEvents()
    width = indicator.width()
    assert indicator.focusPolicy() == Qt.FocusPolicy.NoFocus
    window.request_save_project()
    wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert indicator.width() == width
    window.project_save_coordinator._settle()
    assert window.project_save_coordinator.status.phase == "clean"
    indicator.set_compact(True)
    assert indicator.width() == 28 and "项目.fdmproj" in indicator.toolTip()
    window.current_document().mark_session_dirty()
    window._update_project_navigation_summary()
    assert window.project_save_coordinator.status.phase == "dirty"


def test_save_shortcut_routes_to_background(window, monkeypatch, desktop_application):
    window.show()
    window.activateWindow()
    window.current_canvas().setFocus()
    desktop_application.processEvents()
    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        QTest.keyClick(
            window.current_canvas(), Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier
        )
        wait_for(desktop_application, entered.is_set)
        assert window.project_save_coordinator.busy
        release.set()
        wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)


def test_analysis_and_logo_changes_are_not_overwritten(
    window, tmp_path, monkeypatch, desktop_application
):
    from test_project_analysis_asset_save import _artifact, _reference

    from fdm.ui.watermark_rendering import import_logo

    doc = window.current_document()
    logo_path = tmp_path / "logo.png"
    image = QImage(12, 8, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.red)
    assert image.save(str(logo_path))
    digest, data = import_logo(logo_path)
    doc.watermark = WatermarkSpec(enabled=True, kind="logo", logo_sha256=digest)
    doc.watermark_assets[digest] = data
    reference = _reference(tmp_path / "source.npz")
    original = _artifact(doc.id, reference)
    window.project.analysis_artifacts = [original]
    window._session_analysis_assets[reference.path] = tmp_path / "source.npz"
    window.project.mark_extension_changed()
    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        window.request_save_project()
        wait_for(desktop_application, entered.is_set)
        logo_path.unlink()
        doc.watermark = WatermarkSpec(enabled=True, text="new text")
        doc.mark_session_dirty()
        window.project.analysis_artifacts = [original.mark_stale("changed during save")]
        window.project.mark_extension_changed()
        release.set()
        wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    stored = ProjectIO.load(window._project_path)
    assert stored.documents[0].watermark.kind == "logo"
    assert stored.documents[0].watermark_assets[digest] == data
    assert stored.analysis_artifacts[0].is_current
    current = window.project.analysis_artifacts[0]
    assert not current.is_current and current.stale_reason == "changed during save"
    assert ".rev-" in current.assets[0].path
    assert (
        doc.watermark.kind == "text"
        and doc.dirty_flags.session_dirty
        and window._project_dirty()
    )


def test_digital_slide_background_backup_owns_connection(
    window, tmp_path, monkeypatch, desktop_application
):
    from test_associated_file_open import _create_slide

    from fdm.services.digital_slide_store import DigitalSlideStore

    source = tmp_path / "source.fdmslide"
    _create_slide(source)
    document = ImageDocument(
        "slide",
        "slides/slide.fdmslide",
        (64, 48),
        source_type="project_asset",
        document_kind="digital_slide",
        metadata={"digital_slide": {"working_path": str(source)}},
    )
    window._open_image_requests([(str(source), document)], context_label="test")
    wait_for(desktop_application, lambda: not window.is_image_loading())
    assert window.project.get_document(document.id) is not None
    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        window.request_save_project()
        wait_for(desktop_application, entered.is_set)
        document.metadata["new_note"] = "retain"
        document.mark_session_dirty()
        release.set()
        wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert window.project_save_coordinator.status.phase == "saved_newer"
    target = project_assets_root(window._project_path) / document.path
    assert target.is_file() and source.is_file()
    assert DigitalSlideStore.read_manifest_read_only(target).width == 64
    assert document.metadata["new_note"] == "retain"
    assert document.metadata["digital_slide"]["working_path"] == str(target)


def test_close_save_choice_resumes_asynchronously(
    window, monkeypatch, desktop_application
):
    from PySide6.QtWidgets import QApplication

    window.current_document().mark_session_dirty()

    def select_save():
        box = QApplication.activeModalWidget()
        next(button for button in box.buttons() if button.text() == "保存").click()

    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        QTimer.singleShot(0, select_save)
        window.close_current_document()
        wait_for(desktop_application, entered.is_set)
        assert window.current_document() is not None
        release.set()
        wait_for(desktop_application, lambda: not window.project.documents)


def test_switch_project_waits_for_save(
    window, tmp_path, monkeypatch, desktop_application
):
    other = tmp_path / "other.fdmproj"
    ProjectIO.save(
        ProjectState(
            version="test",
            documents=[ImageDocument("other", str(tmp_path / "first.png"), (120, 90))],
        ),
        other,
    )
    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        window.request_save_project()
        wait_for(desktop_application, entered.is_set)
        window._load_project_from_path(other)
        assert window._project_path != other
        release.set()
        wait_for(desktop_application, lambda: window._project_path == other)
        wait_for(desktop_application, lambda: not window.is_image_loading())
    assert window.project.documents[0].id == "other"
    assert not window.project_save_coordinator.busy


def test_newer_sidecar_savepoint_is_preserved(window, monkeypatch, desktop_application):
    doc = window.current_document()
    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        window.request_save_project()
        wait_for(desktop_application, entered.is_set)
        doc.calibration = Calibration("manual", 2, "um", "new")
        doc.mark_calibration_dirty()
        doc.mark_calibration_saved()
        release.set()
        wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert not doc.dirty_flags.calibration_dirty and doc.dirty_flags.session_dirty
    assert doc._saved_calibration_signature == doc.calibration_signature()


def test_post_commit_ui_error_never_leaves_writer_busy(
    window, monkeypatch, desktop_application
):
    monkeypatch.setattr(
        window,
        "_refresh_project_save_state",
        lambda: (_ for _ in ()).throw(RuntimeError("UI update failed")),
    )
    window.request_save_project()
    wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert window._project_path.is_file()
    assert window.project_save_coordinator.status.text == "已写入，状态异常"
    assert "文件已写入" in window.project_save_coordinator.status.detail


def test_old_request_result_is_ignored(window, monkeypatch, desktop_application):
    from fdm.ui.project_session_controller import ProjectSaveResult, ProjectWriteResult

    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        window.request_save_project()
        wait_for(desktop_application, entered.is_set)
        coordinator = window.project_save_coordinator
        coordinator._receive(
            coordinator._request_number - 1,
            coordinator._epoch,
            ProjectWriteResult(ProjectSaveResult(False, message="obsolete")),
        )
        assert coordinator._outcome is None
        release.set()
        wait_for(desktop_application, lambda: not coordinator.busy)
    assert coordinator.status.phase == "saved"


def test_repeated_saves_release_qt_threads_and_snapshots(window, desktop_application):
    from PySide6.QtCore import QCoreApplication, QEvent, QThread

    coordinator = window.project_save_coordinator
    for _ in range(8):
        assert window.request_save_project()
        wait_for(desktop_application, lambda: not coordinator.busy)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert (
            coordinator.active is None
            and coordinator.pending is None
            and coordinator._outcome is None
        )
        assert not coordinator.findChildren(QThread)


def test_window_close_is_deferred_until_thread_finishes(
    window, monkeypatch, desktop_application
):
    window.show()
    desktop_application.processEvents()
    with paused_writer(monkeypatch, desktop_application) as (entered, release, _):
        window.request_save_project()
        wait_for(desktop_application, entered.is_set)
        assert window.close() is False
        assert window.isVisible()
        release.set()
        wait_for(desktop_application, lambda: not window.isVisible())
    assert window.project_save_coordinator._thread is None
    assert window._application_key_filter_installed is False


def test_unexpected_later_conversion_failure_rolls_back_earlier_asset(
    window, tmp_path, monkeypatch, desktop_application
):
    first = window.current_document()
    first.source_type = "project_asset"
    first.path = "imports/first.png"
    plane = numpy_to_raster_plane(np.zeros((90, 120), dtype=np.uint8))
    first.raster_pixel_type = plane.pixel_type
    first.mark_session_dirty()
    window._rasters[first.id] = plane
    window._raster_metadata[first.id] = None
    second = add_image(window, tmp_path / "fallback.png")
    second.source_type = "project_asset"
    second.path = "imports/second.png"
    window._rasters.pop(second.id, None)

    def fail_conversion(_image):
        raise RuntimeError("conversion failed after first resource")

    monkeypatch.setattr(
        "fdm.ui.project_session_controller.qimage_to_raster_plane", fail_conversion
    )
    window.request_save_project()
    wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert window.project_save_coordinator.status.phase == "failed"
    assert not window._project_path.exists()
    assert not any(
        path.is_file() for path in project_assets_root(window._project_path).rglob("*")
    )
    assert first.dirty_flags.session_dirty


def test_save_as_retry_keeps_failed_destination(window, tmp_path, desktop_application):
    assert window.save_project()
    original_path = window._project_path
    original_bytes = original_path.read_bytes()
    target = tmp_path / "different" / "另存为.fdmproj"
    window.current_document().mark_session_dirty()
    with patch.object(
        ProjectIO,
        "save_payload",
        side_effect=OSError("destination temporarily unavailable"),
    ):
        assert window.request_save_project(str(target))
        wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert window._project_path == original_path
    assert window.project_save_coordinator.status.path == str(target)
    window._save_status_indicator.retryRequested.emit()
    wait_for(desktop_application, lambda: not window.project_save_coordinator.busy)
    assert target.is_file() and window._project_path == target
    assert original_path.read_bytes() == original_bytes
