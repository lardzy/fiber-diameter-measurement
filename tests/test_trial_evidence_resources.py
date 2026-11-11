"""Real-file lifecycle checks: collection never treats an arbitrary path as owned."""
from pathlib import Path

import numpy as np
import pytest

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.evidence import EvidenceStore
from fdm.live_experiment.evidence_resources import EvidenceResources
from fdm.live_experiment.frames import FrameHandle
from fdm.live_experiment import evidence_resources
from fdm.live_experiment.project_assets import rebase_experiment_assets
from fdm.models import ProjectState, project_assets_root
from fdm.project_io import ProjectIO


@pytest.fixture
def managed(tmp_path, monkeypatch):
    manager = EvidenceResources(tmp_path / 'managed')
    monkeypatch.setattr(evidence_resources, '_resources', manager)
    yield manager
    manager.shutdown()


def frame():
    return FrameHandle.snapshot(np.full((18, 24, 3), 71, np.uint8), source_id='lifecycle')


def session():
    result = ExperimentSession()
    result.add_operator('A')
    result.set_materials(default_materials())
    return result


def saved_store(manager, project=None, trial=None):
    store = EvidenceStore()
    if trial is not None:
        store.attach_owner(project, trial)
    entry = store.reserve(frame())
    store.flush()
    return store, entry


def collect(manager):
    return manager.request_cleanup().result(timeout=5)


def test_disabled_store_creates_no_directory_or_worker(managed):
    store = EvidenceStore()
    assert not store.directory.exists()
    assert store.reserve(frame(), options={'mode': 'disabled'}) == {'state': 'disabled'}
    store.close()
    assert not managed.root.exists()


def test_active_trial_keeps_images_across_view_close_and_manual_cleanup(managed):
    trial, project = session(), object()
    store, entry = saved_store(managed, project, trial)
    record = trial.add_measurement(20, 'wool', evidence=entry)
    path = Path(entry['path'])
    store.close()
    assert collect(managed).deleted_files == 0
    assert path.exists()
    trial.delete_measurement(record.id)
    assert collect(managed).deleted_files == 0
    assert trial.undo()
    assert Path(trial.measurements[record.id].evidence['path']).exists()
    managed.discard_project(project)
    collect(managed)
    assert not path.exists()


def test_discard_one_trial_preserves_other_trial_and_user_files(managed, tmp_path):
    project = object()
    first, second = session(), session()
    a, ea = saved_store(managed, project, first)
    b, eb = saved_store(managed, project, second)
    first.add_measurement(20, 'wool', evidence=ea)
    second.add_measurement(21, 'wool', evidence=eb)
    a.close(); b.close()
    report = tmp_path / 'report.xlsx'
    report.write_bytes(b'user file')
    managed.discard_session(first)
    collect(managed)
    assert not Path(ea['path']).exists()
    assert Path(eb['path']).exists()
    assert report.read_bytes() == b'user file'


def test_explicit_store_directory_is_not_owned(managed, tmp_path):
    directory = tmp_path / 'user-export'
    store = EvidenceStore(directory)
    entry = store.reserve(frame())
    store.close()
    collect(managed)
    assert Path(entry['path']).exists()
    assert not managed.root.exists()


def test_save_rebases_cache_and_keeps_prepared_snapshot_until_released(managed, tmp_path):
    trial, project = session(), ProjectState.empty()
    project.live_experiments.append(trial)
    store, entry = saved_store(managed, project, trial)
    source_frame = frame()
    # Use the same frame identity throughout the dedup check.
    cached = store.reserve(source_frame)
    store.flush()
    record = trial.add_measurement(20, 'wool', evidence=cached)
    old_path = Path(cached['path'])
    snapshot = project.to_dict()
    pin = managed.pin_payload(snapshot)
    target = tmp_path / 'published.fdmproj'
    ProjectIO.save(project, target)
    rebase_experiment_assets(project.live_experiments, snapshot, project_assets_root(target))
    new_path = Path(trial.measurements[record.id].evidence['path'])
    assert new_path != old_path and new_path.exists()
    assert store.reserve(source_frame)['path'] == str(new_path)
    collect(managed)
    assert old_path.exists()
    pin.close()
    collect(managed)
    assert not old_path.exists()
    assert new_path.exists()
    assert store.reserve(source_frame)['path'] == str(new_path)
    store.close()


def test_undo_only_reference_not_in_save_snapshot_is_preserved(managed, tmp_path):
    trial, project = session(), ProjectState.empty()
    project.live_experiments.append(trial)
    store, entry = saved_store(managed, project, trial)
    one = trial.add_measurement(20, 'wool', evidence=entry)
    two = trial.add_measurement(21, 'wool', evidence=entry)
    trial.delete_measurement(two.id)
    snapshot = project.to_dict()
    target = tmp_path / 'published.fdmproj'
    ProjectIO.save(project, target)
    rebase_experiment_assets([trial], snapshot, project_assets_root(target))
    collect(managed)
    # Same image is still needed by a deleted, undoable record not in the save.
    assert Path(entry['path']).exists()
    assert trial.undo()
    assert Path(trial.measurements[two.id].evidence['path']).exists()
    store.close()


def test_reader_lease_preserves_image_until_async_read_finishes(managed):
    trial, project = session(), object()
    store, entry = saved_store(managed, project, trial)
    store.close()
    path = Path(entry['path'])
    lease = managed.pin_paths([path])
    managed.discard_session(trial)
    collect(managed)
    assert path.exists()
    lease.close()
    collect(managed)
    assert not path.exists()


def test_another_live_instance_is_never_collected(managed):
    store, entry = saved_store(managed)
    store.close()
    other = EvidenceResources(managed.root)
    try:
        assert collect(other).in_use >= 1
        assert Path(entry['path']).exists()
    finally:
        other.shutdown()


def test_next_start_cleans_dead_instance_and_partial_write_receipt(managed):
    store, entry = saved_store(managed)
    store.close()
    path = Path(entry['path'])
    pending = path.with_suffix('.pending')
    path.rename(pending)  # Simulate termination after encoder close, before rename.
    managed.shutdown()
    other = EvidenceResources(managed.root)
    try:
        result = collect(other)
        assert result.deleted_files == 1
        assert not pending.exists()
        assert not managed.directory.exists()
    finally:
        other.shutdown()


def test_unknown_contents_and_legacy_directory_not_removed(managed, tmp_path):
    store, entry = saved_store(managed)
    store.close()
    unknown = store.directory / 'user-photo.png'
    unknown.write_bytes(b'not managed')
    legacy = managed.root / 'fdm-live-evidence-old'
    legacy.mkdir()
    (legacy / 'photo.png').write_bytes(b'legacy')
    managed.shutdown()
    other = EvidenceResources(managed.root)
    try:
        collect(other)
        assert not Path(entry['path']).exists()
        assert unknown.read_bytes() == b'not managed'
        assert (legacy / 'photo.png').exists()
    finally:
        other.shutdown()


def test_replaced_file_and_symlink_are_not_deleted(managed, tmp_path):
    trial = session()
    store, entry = saved_store(managed, object(), trial)
    store.close()
    original = Path(entry['path'])
    destination = tmp_path / 'important.png'
    original.rename(destination)
    try:
        original.symlink_to(destination)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows 普通用户未获创建符号链接权限；目录联接另有独立验证")
        raise
    managed.discard_session(trial)
    result = collect(managed)
    assert original.is_symlink() and destination.exists()
    assert result.skipped >= 1


def test_replaced_regular_file_inode_mismatch_is_preserved(managed, tmp_path):
    trial = session()
    store, entry = saved_store(managed, object(), trial)
    store.close()
    path = Path(entry['path'])
    path.rename(tmp_path / 'original.png')
    path.write_bytes(b'user replacement')
    managed.discard_session(trial)
    collect(managed)
    assert path.read_bytes() == b'user replacement'


def test_busy_activity_defers_cleanup_without_waiting(managed):
    trial = session()
    store, entry = saved_store(managed, object(), trial)
    store.close()
    lease = managed.activity()
    managed.discard_session(trial)
    result = collect(managed)
    assert result.deferred and Path(entry['path']).exists()
    lease.close()
    collect(managed)
    assert not Path(entry['path']).exists()


def test_cleanup_denied_reports_and_can_retry(managed, monkeypatch):
    trial = session()
    store, entry = saved_store(managed, object(), trial)
    store.close()
    target = Path(entry['path'])
    original = Path.unlink
    def denied(path, *args, **kwargs):
        if path == target:
            raise PermissionError('fixture in use')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', denied)
    managed.discard_session(trial)
    assert collect(managed).errors
    assert target.exists()
    monkeypatch.setattr(Path, 'unlink', original)
    collect(managed)
    assert not target.exists()


def test_existing_pending_file_is_not_deleted_on_refused_write(managed):
    store = EvidenceStore()
    managed._ensure_directory(store._resource_key)
    source = frame()
    import hashlib
    pending = store.directory / (hashlib.sha256(source.frame_id.encode()).hexdigest() + '.pending')
    pending.write_bytes(b'user pending file')
    entry = store.reserve(source)
    store.close()
    assert entry['state'] == 'failed'
    assert pending.read_bytes() == b'user pending file'
    collect(managed)
    assert pending.exists()


def test_slow_file_check_does_not_block_reader_and_reader_cancels_retirement(managed, monkeypatch):
    import threading
    from time import perf_counter
    trial = session()
    store, entry = saved_store(managed, object(), trial)
    store.close()
    path = Path(entry['path'])
    entered, resume = threading.Event(), threading.Event()
    ordinary = evidence_resources._ordinary
    def delayed(candidate, **kwargs):
        if candidate == path:
            entered.set()
            assert resume.wait(3)
        return ordinary(candidate, **kwargs)
    monkeypatch.setattr(evidence_resources, '_ordinary', delayed)
    managed.discard_session(trial)
    assert entered.wait(3)
    started = perf_counter()
    lease = managed.pin_paths([path])
    activity = managed.activity()
    assert perf_counter() - started < .05
    assert lease.resolve(path) == str(path)
    activity.close()
    resume.set()
    collect(managed)
    assert path.exists()
    lease.close()
    collect(managed)
    assert not path.exists()


def test_slow_unlink_does_not_block_ui_and_stale_read_is_rejected(managed, monkeypatch):
    import threading
    from time import perf_counter
    trial = session()
    store, entry = saved_store(managed, object(), trial)
    store.close()
    path = Path(entry['path'])
    entered, resume = threading.Event(), threading.Event()
    unlink = Path.unlink
    def delayed(candidate, *args, **kwargs):
        if candidate == path:
            entered.set()
            assert resume.wait(3)
        return unlink(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', delayed)
    managed.discard_session(trial)
    assert entered.wait(3)
    started = perf_counter()
    lease = managed.pin_paths([path])
    activity = managed.activity()
    assert perf_counter() - started < .05
    assert lease.resolve(path) is None
    activity.close()
    lease.close()
    resume.set()
    collect(managed)
    assert not path.exists()
    later = managed.pin_paths([path])
    assert later.resolve(path) is None
    later.close()


def test_busy_mid_collection_resumes_at_journal_offset(managed, monkeypatch):
    trial = session()
    store = EvidenceStore()
    store.attach_owner(object(), trial)
    entries = []
    for index in range(20):
        source = FrameHandle.snapshot(np.full((4, 4), index, np.uint8),
                                     source_id='continuation', sequence=index)
        entries.append(store.reserve(source))
    store.close()
    original = managed._delete
    visits, held = [], []
    def paused(directory, relative, identity, report, **kwargs):
        original(directory, relative, identity, report, **kwargs)
        visits.append(relative)
        if len(visits) == 7:
            held.append(managed.activity())
    monkeypatch.setattr(managed, '_delete', paused)
    managed.discard_session(trial)
    first = collect(managed)
    assert first.deferred and held
    assert managed._collection_offsets
    held[0].close()
    collect(managed)
    assert all(not Path(entry['path']).exists() for entry in entries)
    # Recheck only the operation racing the pause, not every earlier journal row.
    assert visits.count(visits[0]) == 1
    assert not managed._collection_offsets


def test_lease_release_between_worker_return_and_future_completion_is_not_lost(managed, monkeypatch):
    import threading
    import time
    trial = session()
    store, entry = saved_store(managed, object(), trial)
    store.close()
    path = Path(entry['path'])
    blocked = managed.activity()
    entered, complete = threading.Event(), threading.Event()
    original = managed._collect
    calls = []
    def delayed_return():
        report = original()
        calls.append(report)
        if len(calls) == 1:
            entered.set()
            assert complete.wait(3)
        return report
    monkeypatch.setattr(managed, '_collect', delayed_return)
    managed.discard_session(trial)
    assert entered.wait(3)
    assert calls[0].deferred
    blocked.close()  # Future is running although its collection pass has ended.
    complete.set()
    deadline = time.monotonic() + 4
    while path.exists() and time.monotonic() < deadline:
        time.sleep(.003)
    assert not path.exists()  # No manual cleanup request was needed.


def test_stop_cleanup_is_nonblocking_keeps_instance_lock_and_defers_remaining_images(managed, monkeypatch):
    import threading
    import time
    trial = session()
    store = EvidenceStore()
    store.attach_owner(object(), trial)
    entries = []
    for index in range(20):
        source = FrameHandle.snapshot(np.full((4, 4), index, np.uint8),
                                     source_id='stop-cleanup', sequence=index)
        entries.append(store.reserve(source))
    store.close()
    entered, release = threading.Event(), threading.Event()
    target = Path(entries[0]['path'])
    unlink = Path.unlink
    def delayed(path, *args, **kwargs):
        if path == target:
            entered.set()
            assert release.wait(4)
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', delayed)
    managed.discard_session(trial)
    running = managed._future
    assert entered.wait(3)
    began = time.perf_counter()
    managed.stop_cleanup()
    assert time.perf_counter() - began < .05
    assert managed._instance_lock is not None and managed._instance_lock.fd is not None
    other = EvidenceResources(managed.root)
    try:
        assert collect(other).in_use >= 1
        release.set()
        result = running.result(timeout=3)
        assert result.stopped
        remaining = [Path(item['path']) for item in entries if Path(item['path']).exists()]
        assert remaining
        assert managed.request_cleanup().result().stopped
        # Stopping collection does not prematurely advertise a dead process.
        assert collect(other).in_use >= 1
        assert all(path.exists() for path in remaining)
        managed.shutdown()  # Test-only analogue of the OS closing process FDs.
        collect(other)
        assert all(not path.exists() for path in remaining)
    finally:
        release.set()
        other.shutdown()


def test_windows_junction_does_not_follow_moved_store_into_user_directory(managed, tmp_path):
    import os
    import subprocess
    if os.name != 'nt':
        pytest.skip('Windows directory junction boundary')
    trial = session()
    store, entry = saved_store(managed, object(), trial)
    store.close()
    original = Path(entry['path'])
    moved = tmp_path / 'moved-store'
    store.directory.rename(moved)
    user_file = moved / 'user-photo.txt'
    user_file.write_text('must survive', encoding='utf-8')
    relative = store.directory.relative_to(tmp_path)
    # Relative generated names avoid shell parsing of the user's profile path.
    result = subprocess.run(['cmd.exe', '/d', '/c', 'mklink', '/J', str(relative), 'moved-store'],
                            cwd=tmp_path, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    try:
        assert store.directory.lstat().st_file_attributes & 0x400
        managed.discard_session(trial)
        result = collect(managed)
        assert result.skipped >= 1
        assert (moved / original.name).is_file()
        assert user_file.read_text(encoding='utf-8') == 'must survive'
    finally:
        # Remove only the junction itself. Never recursively remove its target.
        store.directory.rmdir()
