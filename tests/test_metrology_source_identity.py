from __future__ import annotations

from dataclasses import replace
import hashlib
from threading import Event, get_ident
import time
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QImage

from fdm.geometry import Point
from fdm.metrology import MeasurementDefinition, MeasurementFeature
from fdm.models import ImageDocument
from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore, DigitalSlideTile, DigitalSlideTileDescriptor
from fdm.services.metrology_service import evaluate_document
from fdm.services.metrology_source_identity import ProfessionalSourceRevisionResult, ProfessionalSourceRevisionWorker, professional_slide_source_revision
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.professional_measurement import ProfessionalWorkspace


def wait_until(app, condition, timeout=4):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert condition()


def create_slide(path):
    store = DigitalSlideStore.create(path, DigitalSlideManifest(1, 80, 40, 40, 40, [-2, 0, 2]))
    image = QImage(40, 40, QImage.Format.Format_RGB32)
    image.fill(0xff887766)
    for focus in range(3):
        for x in (0, 40):
            store.write_tile(DigitalSlideTile(focus, x, 0, 40, 40, stage_x=x, focus_z=focus * 2 - 2), image)
    manifest = store.read_manifest()
    descriptors = [item for focus in range(3) for item in store.list_tile_descriptors(z_index=focus)]
    revision = professional_slide_source_revision(manifest, descriptors)
    store.close()
    return manifest, revision


def make_document(path, identity="image"):
    document = ImageDocument(identity, str(path), (80, 40), document_kind="digital_slide")
    document.initialize_runtime_state()
    return document


def test_source_hash_is_order_independent_and_identifies_registration_not_pixel_codec():
    manifest = DigitalSlideManifest(1, 100, 100, 20, 20, [-1, 0, 1], created_at="fixed")
    first = DigitalSlideTileDescriptor(1, 0, 10, 20, 20, 20, 50, 60, -1)
    second = DigitalSlideTileDescriptor(2, 1, 30, 20, 20, 20, 70, 60, 0)
    baseline = professional_slide_source_revision(manifest, [first, second])
    assert baseline == professional_slide_source_revision(manifest, [second, first])
    assert baseline == professional_slide_source_revision(manifest, [replace(first, sharpness=99, status="reviewed"), second])
    for changed in (replace(first, x=11), replace(first, tile_id=4), replace(first, focus_z=-2), replace(first, stage_x=51)):
        assert professional_slide_source_revision(manifest, [changed, second]) != baseline
    changed_manifest = DigitalSlideManifest.from_dict(manifest.to_dict())
    changed_manifest.metadata["registration"] = {"dx": 2.5}
    assert professional_slide_source_revision(changed_manifest, [first, second]) != baseline


def test_mount_computes_identity_off_gui_without_pixel_reads_or_persistent_changes(tmp_path, monkeypatch, desktop_application):
    path = tmp_path / "source.fdmslide"
    _manifest, expected = create_slide(path)
    before_bytes = hashlib.sha256(path.read_bytes()).hexdigest()
    store = DigitalSlideStore(path); store.open_read_only()
    document = make_document(path)
    before_payload = document.to_dict()
    before_stamp = document.state_stamp
    gui_thread = get_ident(); descriptor_threads = []
    original = DigitalSlideStore.list_tile_descriptors

    def descriptors(self, *, z_index):
        descriptor_threads.append(get_ident())
        return original(self, z_index=z_index)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Source identity must not read any image bytes")

    monkeypatch.setattr(DigitalSlideStore, "list_tile_descriptors", descriptors)
    monkeypatch.setattr(DigitalSlideStore, "read_tile_image", forbidden)
    canvas = DigitalSlideCanvas()
    try:
        canvas.set_slide_document(document, store, defer_rendering=True)
        assert document.professional_source_revision_pending
        wait_until(desktop_application, lambda: not document.professional_source_revision_pending)
        assert document.professional_source_revision == expected
        assert document.professional_source_revision_error is None
        assert descriptor_threads and all(identity != gui_thread for identity in descriptor_threads)
        assert document.to_dict() == before_payload
        assert document.state_stamp == before_stamp
        assert not document.dirty_flags.session_dirty
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before_bytes
    finally:
        canvas.shutdown(); canvas.close(); store.close()


def test_reopen_same_source_is_stable_and_late_generation_cannot_overwrite_new_document(tmp_path, monkeypatch, desktop_application):
    path_a, path_b = tmp_path / "first.fdmslide", tmp_path / "second.fdmslide"
    _manifest_a, expected_a = create_slide(path_a)
    _manifest_b, expected_b = create_slide(path_b)
    blocked, release = Event(), Event()
    original = DigitalSlideStore.list_tile_descriptors

    def delayed(self, *, z_index):
        if self.path == path_a and not release.is_set():
            blocked.set(); release.wait(3)
        return original(self, z_index=z_index)

    monkeypatch.setattr(DigitalSlideStore, "list_tile_descriptors", delayed)
    first_store, second_store = DigitalSlideStore(path_a), DigitalSlideStore(path_b)
    first_store.open_read_only(); second_store.open_read_only()
    first, second = make_document(path_a, "same-id"), make_document(path_b, "same-id")
    canvas = DigitalSlideCanvas()
    try:
        canvas.set_slide_document(first, first_store, defer_rendering=True)
        wait_until(desktop_application, blocked.is_set)
        old_generation = canvas._professional_source_generation
        canvas.set_slide_document(second, second_store, defer_rendering=True)
        release.set()
        wait_until(desktop_application, lambda: not second.professional_source_revision_pending)
        assert second.professional_source_revision == expected_b
        canvas._on_professional_source_revision(ProfessionalSourceRevisionResult(old_generation, second.id, "wrong-old-revision"))
        assert second.professional_source_revision == expected_b
        assert first.professional_source_revision is None
        assert "取消" in first.professional_source_revision_error
        canvas.set_slide_document(second, second_store, defer_rendering=True)
        wait_until(desktop_application, lambda: not second.professional_source_revision_pending)
        assert second.professional_source_revision == expected_b
        current_generation = canvas._professional_source_generation
        canvas.shutdown()
        canvas._on_professional_source_revision(ProfessionalSourceRevisionResult(current_generation, second.id, "late"))
        assert second.professional_source_revision == expected_b
    finally:
        release.set(); canvas.shutdown(); canvas.close(); first_store.close(); second_store.close()


def test_source_worker_reports_unavailable_source_without_creating_file(tmp_path):
    path = tmp_path / "missing.fdmslide"
    finished = Event(); results = []
    worker = ProfessionalSourceRevisionWorker(path, DigitalSlideManifest(1, 10, 10, 10, 10, [0]), generation=1, document_id="missing", callback=lambda result: (results.append(result), finished.set()))
    try:
        assert finished.wait(2)
        assert results[0].revision is None and "无法核对" in results[0].error
        assert not path.exists()
    finally:
        worker.close()


def test_source_pending_or_mismatch_is_invalid_and_rechecks_cache_without_dirtying():
    document = make_document("/tmp/source.fdmslide")
    feature = MeasurementFeature("f", document.id, "line", (Point(0, 0), Point(3, 4)), focus_index=0, source_revision="expected")
    definition = MeasurementDefinition("m", document.id, "distance", (feature.id,), focus_index=0)
    document.apply_professional_state([feature], [definition], mark_dirty=False)
    assert "尚未核对" in evaluate_document(document)[definition.id].reason
    document.professional_source_revision_pending = True
    assert "正在核对" in evaluate_document(document)[definition.id].reason
    document.professional_source_revision_pending = False
    document.professional_source_revision = "expected"
    assert evaluate_document(document)[definition.id].valid
    document.professional_source_revision = "changed"
    assert "版本已改变" in evaluate_document(document)[definition.id].reason
    assert not document.dirty_flags.session_dirty
    assert "professional_source_revision" not in document.to_dict()
    assert "professional_source_revision" not in document.snapshot_state()


def test_verified_source_does_not_silently_attach_an_unversioned_historical_feature():
    document = make_document("/tmp/source.fdmslide")
    document.professional_source_revision = "verified-current"
    feature = MeasurementFeature("unversioned", document.id, "line", (Point(0, 0), Point(3, 4)), focus_index=0)
    definition = MeasurementDefinition("m", document.id, "distance", (feature.id,), focus_index=0)
    document.apply_professional_state([feature], [definition], mark_dirty=False)
    result = evaluate_document(document)[definition.id]
    assert not result.valid and "缺少切片来源版本" in result.reason
    assert document.professional_features[0].source_revision is None
    assert document.professional_measurements == [definition]


@pytest.mark.parametrize("pending,error", [(True, None), (False, "读取失败"), (False, None)])
def test_new_command_is_blocked_until_source_is_verified(pending, error):
    document = make_document("/tmp/source.fdmslide")
    document.professional_source_revision_pending = pending
    document.professional_source_revision_error = error
    controller = SimpleNamespace(canvas=SimpleNamespace(_document=document, _read_only=False))
    messages = []
    workspace = SimpleNamespace(controller=lambda: controller, message=messages.append)
    ProfessionalWorkspace.begin(workspace)
    assert messages
