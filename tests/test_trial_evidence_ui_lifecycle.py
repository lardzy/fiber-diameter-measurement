"""Actual MainWindow transitions retain/retire only their managed trial images."""
from pathlib import Path
import time

import pytest
from PySide6.QtWidgets import QMessageBox

from test_acquisition_workflow_redesign import workflow
from fdm.live_experiment import default_materials
from fdm.live_experiment import evidence_resources
from fdm.live_experiment.evidence_resources import EvidenceResources
from fdm.project_io import ProjectIO
from fdm.ui.trial_preferences_binding import trial_preferences_binding


@pytest.fixture
def harness(tmp_path, monkeypatch):
    resources = EvidenceResources(tmp_path / 'owned-images')
    monkeypatch.setattr(evidence_resources, '_resources', resources)
    try:
        with workflow.WorkflowHarness(tmp_path / 'profile') as h:
            h.resources = resources
            yield h
    finally:
        resources.shutdown()


def wait_until(predicate, timeout=7):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'Qt operation timed out'
        workflow.settle()
        time.sleep(.003)
    workflow.settle()


def collect(h):
    future = h.resources.request_cleanup()
    wait_until(future.done)
    return future.result()


def choose(monkeypatch, label):
    def run(dialog):
        next(button for button in dialog.buttons() if button.text() == label).click()
        return 0
    monkeypatch.setattr(QMessageBox, 'exec', run)


def create_measurement(h):
    h.enter()
    page = h.select(1)
    page.session.set_materials(default_materials())
    page._refresh_materials()
    page.ruler_choice.setCurrentIndex(1)
    page.ruler_choice.activated.emit(1)
    trial_preferences_binding(h.window).update(evidence_enabled=True)
    frame = page.controller.latest_frame
    entry = page.controller.evidence.reserve(frame, options={'mode': 'crop_jpeg'},
                                             points=((30, 30), (30, 70)))
    page.controller.evidence.flush()
    assert entry['state'] == 'saved'
    record = page.session.add_measurement(20, 'wool', field_id=page.controller.field_id,
        endpoints=((30, 30), (30, 70)), evidence=entry)
    page.session.update_field(page.controller.field_id, first_frame_id=frame.frame_id)
    page.controller.current_records.append(record.id)
    page._record_accepted(record)
    page._changed()
    assert page.session in h.window.project.live_experiments
    return page, record.id, Path(entry['path']), frame


def test_cancel_close_keeps_image_then_discard_close_collects(harness, monkeypatch):
    h = harness
    page, record_id, image, _ = create_measurement(h)
    choose(monkeypatch, '取消')
    assert not h.window.close()
    collect(h)
    assert image.exists() and not page._closed
    assert record_id in page.session.measurements
    choose(monkeypatch, '放弃')
    assert h.window.close()
    collect(h)
    assert not image.exists()
    assert not h.window.isVisible()
    assert not h.slot_errors


def test_page_switch_and_existing_trial_switch_keep_image(harness):
    h = harness
    page, record_id, image, _ = create_measurement(h)
    h.select(0)
    h.select(2)
    h.select(1)
    collect(h)
    assert image.exists() and record_id in page.session.measurements
    from fdm.ui.live_experiment_workspace import _new_trial_session
    other = _new_trial_session('content')
    page._replace_experiment(other)
    replacement = h.window._live_experiment_workspaces['content']
    assert replacement is not page and page._closed
    collect(h)
    assert image.exists()
    replacement._replace_experiment(page.session)
    restored = h.window._live_experiment_workspaces['content']
    collect(h)
    assert record_id in restored.session.measurements and image.exists()
    assert not h.slot_errors


def test_save_publishes_image_collects_temporary_and_dedup_stays_usable(harness, tmp_path):
    h = harness
    page, record_id, temporary, frame = create_measurement(h)
    target = tmp_path / 'saved.fdmproj'
    assert h.window.request_save_project(str(target))
    wait_until(lambda: not h.window.project_save_coordinator.busy)
    assert h.window.project_save_coordinator.last_metrics['success']
    published = Path(page.session.measurements[record_id].evidence['path'])
    collect(h)
    assert published.exists() and published != temporary
    assert not temporary.exists()
    duplicate = page.controller.evidence.reserve(frame, options={'mode': 'crop_jpeg'},
                                                points=((30, 30), (30, 70)))
    assert duplicate['path'] == str(published)
    page.session.delete_measurement(record_id)
    assert page.session.undo()
    assert Path(page.session.measurements[record_id].evidence['path']).exists()
    restored = ProjectIO.load(target)
    assert Path(restored.live_experiments[0].measurements[record_id].evidence['path']).exists()
    assert not h.slot_errors


def test_failed_save_and_cancel_exit_keep_only_copy(harness, monkeypatch, tmp_path):
    h = harness
    page, record_id, image, _ = create_measurement(h)
    def fail(*args, **kwargs):
        raise OSError('validation disk full')
    monkeypatch.setattr('fdm.project_io.atomic_write_json', fail)
    target = tmp_path / 'failed.fdmproj'
    assert h.window.request_save_project(str(target))
    wait_until(lambda: not h.window.project_save_coordinator.busy)
    assert not h.window.project_save_coordinator.last_metrics['success']
    choose(monkeypatch, '取消')
    assert not h.window.close()
    collect(h)
    assert image.exists() and not target.exists()
    assert Path(page.session.measurements[record_id].evidence['path']) == image
    assert not h.slot_errors


def test_finish_sample_cancel_then_success_only_retires_removed_sample(harness, monkeypatch, tmp_path):
    h = harness
    page, record_id, image, _ = create_measurement(h)
    from fdm.ui.live_experiment_workspace import _new_trial_session
    other = _new_trial_session('diameter')
    h.window.project.live_experiments.append(other)
    exported = tmp_path / 'user-original.png'
    exported.write_bytes(image.read_bytes())
    choose(monkeypatch, '取消')
    page._finish_content_trial()
    collect(h)
    assert image.exists() and page.session in h.window.project.live_experiments
    choose(monkeypatch, '移除并开始')
    page._finish_content_trial()
    collect(h)
    assert not image.exists()
    assert other in h.window.project.live_experiments
    assert page.session not in h.window.project.live_experiments
    assert not h.window._live_experiment_workspaces['content'].session.measurements
    assert exported.exists()
    assert not h.slot_errors


def test_open_new_project_cancel_then_discard_cleans_old_temporary(harness, monkeypatch, tmp_path):
    h = harness
    page, record_id, image, _ = create_measurement(h)
    from fdm.models import ProjectState
    next_project = tmp_path / 'next.fdmproj'
    ProjectIO.save(ProjectState.empty(), next_project)
    choose(monkeypatch, '取消')
    assert h.window._load_project_from_path(next_project).cancelled
    collect(h)
    assert image.exists()
    choose(monkeypatch, '放弃')
    assert h.window._load_project_from_path(next_project).success
    collect(h)
    assert not image.exists()
    assert next_project.exists()
    assert not h.slot_errors


def test_save_choice_during_window_close_keeps_published_assets(harness, monkeypatch, tmp_path):
    h = harness
    page, record_id, temporary, _ = create_measurement(h)
    target = tmp_path / 'save-on-close.fdmproj'
    monkeypatch.setattr(h.window, '_select_project_save_path', lambda _: str(target))
    choose(monkeypatch, '保存')
    assert not h.window.close()  # actual close waits for background publication
    wait_until(lambda: not h.window.isVisible())
    collect(h)
    assert not temporary.exists()
    restored = ProjectIO.load(target)
    published = Path(restored.live_experiments[0].measurements[record_id].evidence['path'])
    assert published.exists()
    assert not h.slot_errors


def test_close_during_real_review_read_keeps_file_until_decoder_finishes(harness, monkeypatch):
    from threading import Event
    h = harness
    page, record_id, image, _ = create_measurement(h)
    entered, release = Event(), Event()
    review = page.evidence_review
    decoder = review._decode
    def delayed(path, expected_size):
        entered.set()
        assert release.wait(5)
        assert Path(path).exists()
        return decoder(path, expected_size)
    monkeypatch.setattr(review, '_decode', delayed)
    try:
        review.show_record(page.session.measurements[record_id], latest=False)
        wait_until(entered.is_set)
        choose(monkeypatch, '放弃')
        assert h.window.close()
        collect(h)
        assert image.exists()
    finally:
        release.set()
    wait_until(lambda: not image.exists())
    assert not h.slot_errors


def test_cleanup_ui_waits_for_automatic_continuations_and_sums_deleted_bytes(harness, monkeypatch):
    from threading import Event
    from fdm.live_experiment.frames import FrameHandle
    from fdm.ui.trial_evidence_lifecycle import trial_evidence_cleanup
    import numpy as np
    h = harness
    page, record_id, first, _ = create_measurement(h)
    another_frame = FrameHandle.snapshot(np.full((14, 18), 113, np.uint8), source_id='second-image')
    second_entry = page.controller.evidence.reserve(another_frame)
    page.controller.evidence.flush()
    second = Path(second_entry['path'])
    expected_bytes = first.stat().st_size + second.stat().st_size
    states = []
    cleanup = trial_evidence_cleanup(h.window)
    cleanup.statusChanged.connect(lambda message, busy: states.append((message, busy)))
    activity = h.resources.activity()
    next_activity = []
    paused_after_one = Event()
    original_delete = h.resources._delete
    def pause_once(directory, relative, identity, report, **kwargs):
        original_delete(directory, relative, identity, report, **kwargs)
        if report.deleted_files == 1 and not paused_after_one.is_set():
            next_activity.append(h.resources.activity())
            paused_after_one.set()
    monkeypatch.setattr(h.resources, '_delete', pause_once)
    requests = []
    original_request = h.resources.request_cleanup
    def counted_request():
        requests.append(True)
        return original_request()
    monkeypatch.setattr(h.resources, 'request_cleanup', counted_request)
    try:
        choose(monkeypatch, '移除并开始')
        page._finish_content_trial()
        cleanup.request()
        wait_until(lambda: cleanup.future.done())
        cleanup._completed()
        assert cleanup.future is not None and states[-1][1]
        before = len(requests)
        for _ in range(10):
            cleanup._completed()
        assert len(requests) == before  # Observation never busy-polls new work.
        activity.close()
        wait_until(paused_after_one.is_set)
        wait_until(lambda: '已清理 1 个' in cleanup.last_message)
        assert cleanup.future is not None and states[-1][1]
        next_activity[0].close()
        wait_until(lambda: cleanup.future is None)
        assert not states[-1][1]
        assert '已清理 2 个' in cleanup.last_message
        assert f'{expected_bytes / 1048576:.2f} MiB' in cleanup.last_message
        assert '使用中 0 个' in cleanup.last_message
        assert '稍后继续' not in cleanup.last_message
        assert not first.exists() and not second.exists()
        assert not h.slot_errors
    finally:
        activity.close()
        for lease in next_activity:
            lease.close()
