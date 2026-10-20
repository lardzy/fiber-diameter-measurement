"""A successful Save As must retire the live dependency on the old PNG path."""
import hashlib
from pathlib import Path
import time

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QGraphicsPixmapItem

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.project_assets import rebase_experiment_assets
from fdm.models import ProjectState, project_assets_root
from fdm.project_io import ProjectIO
from fdm.ui.project_session_controller import ProjectSessionController, write_project_save


def saved_project(tmp_path):
    source = tmp_path / "session.png"
    image = QImage(4, 3, QImage.Format.Format_RGB32)
    image.fill(QColor(17, 83, 159))
    assert image.save(str(source))
    evidence = {"state": "saved", "path": str(source), "frame_id": "frame-1",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "error": None}
    session = ExperimentSession()
    session.add_operator("A")
    session.set_materials(default_materials())
    record = session.add_measurement(20, "wool", evidence=evidence)
    project = ProjectState.empty()
    project.live_experiments.append(session)
    first = tmp_path / "A" / "original.fdmproj"
    ProjectIO.save(project, first)
    source.unlink()
    return first, ProjectIO.load(first), record.id


def wait_saved(app, window):
    deadline = time.monotonic() + 8
    while window.project_save_coordinator.busy:
        assert time.monotonic() < deadline, "background save did not finish"
        app.processEvents()
        time.sleep(.003)
    app.processEvents()
    assert window.project_save_coordinator.last_metrics["success"]


@pytest.mark.parametrize("route", ["sync", "background"])
def test_save_as_move_original_then_review_and_save_current_project(tmp_path, desktop_application, route):
    from fdm.ui.main_window import MainWindow
    from fdm.ui.live_experiment_workspace import LiveExperimentWorkspace

    first, project, record_id = saved_project(tmp_path)
    window = MainWindow()
    window.project = project
    window._project_path = first
    window._mark_project_saved()
    session = project.live_experiments[0]
    second = tmp_path / "B" / "current.fdmproj"
    view = None
    try:
        def save(path):
            if route == "sync":
                result = window.save_project(str(path))
                assert result.success, result.message
            else:
                assert window.request_save_project(str(path))
                wait_saved(desktop_application, window)

        revision = session.revision
        save(second)
        assert session.revision == revision
        expected = project_assets_root(second).resolve() / "experiments" / f"{session.measurements[record_id].evidence['sha256']}.png"
        assert Path(session.measurements[record_id].evidence["path"]) == expected
        assert not window._project_dirty()
        first.parent.rename(tmp_path / "A-moved-away")

        # Exercise the real historical-image review, not only a path assertion.
        view = LiveExperimentWorkspace(window, session)
        window._live_experiment_workspace = view
        view.tabs.setCurrentIndex(1)
        view.history_table.selectRow(0)
        view._review_record()
        review = view.evidence_review
        assert review.dialog.isVisible()
        deadline = time.monotonic() + 3
        while review.dialog.image.image_size is None and time.monotonic() < deadline:
            desktop_application.processEvents()
            review.refresh()
            time.sleep(.003)
        assert review.dialog.image.image_size is not None, review.dialog.status.text()
        pixels = [item for item in review.dialog.image.scene().items() if isinstance(item, QGraphicsPixmapItem)]
        assert len(pixels) == 1
        assert pixels[0].pixmap().toImage().pixelColor(0, 0) == QColor(17, 83, 159)
        assert QImage(str(expected)).pixelColor(0, 0) == QColor(17, 83, 159)
        review.dialog.close()

        session.edit_measurement(record_id, diameter_um=21, material_id="cashmere")
        assert session.undo() and session.redo()
        assert Path(session.measurements[record_id].evidence["path"]) == expected
        view._changed()
        save(second)
        restored = ProjectIO.load(second).live_experiments[0].measurements[record_id]
        assert restored.diameter_um == 21 and restored.material_id == "cashmere"
        assert Path(restored.evidence["path"]) == expected
        assert not window._project_dirty()
    finally:
        if view is not None:
            view.close()
        window._reset_workspace()
        window.close()


def test_rebase_preserves_newer_content_and_updates_undo_only_record(tmp_path):
    first, project, record_id = saved_project(tmp_path)
    session = project.live_experiments[0]
    original = session.measurements[record_id].evidence
    changed = session.add_measurement(22, "wool", evidence=original)
    deleted = session.add_measurement(24, "wool", evidence=original)
    snapshot = project.to_dict()
    second = tmp_path / "B" / "current.fdmproj"
    ProjectIO.save(project, second)
    session.edit_measurement(record_id, diameter_um=23, material_id="cashmere")
    session.set_evidence(changed.id, {"state": "disabled", "reason": "explicit user decision"})
    session.delete_measurement(deleted.id)
    added = session.add_measurement(26, "wool", evidence=original)
    before_revision, before_undo = session.revision, list(session._undo)
    rebase_experiment_assets(project.live_experiments, snapshot, project_assets_root(second))
    assert session.revision == before_revision and session._undo == before_undo
    expected = str(project_assets_root(second).resolve() / "experiments" / f"{original['sha256']}.png")
    assert session.measurements[record_id].diameter_um == 23
    assert session.measurements[record_id].material_id == "cashmere"
    assert session.measurements[record_id].evidence["path"] == expected
    assert session.measurements[changed.id].evidence["state"] == "disabled"
    assert session.measurements[added.id].evidence == original
    assert deleted.id not in session.measurements
    assert session.undo()  # new record
    assert session.undo()  # deletion
    assert session.measurements[deleted.id].evidence["path"] == expected
    first.parent.rename(tmp_path / "A-moved-away")
    ProjectIO.save(project, second)


def test_two_prepared_save_as_snapshots_rebase_to_latest_success(tmp_path):
    from test_project_export_controllers import _ProjectHost

    first, project, record_id = saved_project(tmp_path)
    host = _ProjectHost(tmp_path)
    host.project, host._project_path = project, first
    controller = ProjectSessionController(host)
    second, third = tmp_path / "B" / "second.fdmproj", tmp_path / "C" / "third.fdmproj"
    # Both queued plans captured A, before the first successful acknowledgement.
    prepared_second, prepared_third = controller.prepare_save(str(second)), controller.prepare_save(str(third))
    for request in (prepared_second, prepared_third):
        written = write_project_save(request)
        assert written.result.success, written.result.message
        assert controller.apply_save(request, written).success
    record = project.live_experiments[0].measurements[record_id]
    assert Path(record.evidence["path"]).is_relative_to(project_assets_root(third).resolve())
    first.parent.rename(tmp_path / "A-moved-away")
    second.parent.rename(tmp_path / "B-moved-away")
    assert not QImage(record.evidence["path"]).isNull()
    assert controller.save_project(str(third)).success


def test_failed_publication_does_not_rebase_live_evidence(tmp_path, monkeypatch):
    from test_project_export_controllers import _ProjectHost

    first, project, record_id = saved_project(tmp_path)
    host = _ProjectHost(tmp_path)
    host.project, host._project_path = project, first
    controller = ProjectSessionController(host)
    session = project.live_experiments[0]
    before = session.measurements[record_id].evidence.copy()
    second = tmp_path / "B" / "failed.fdmproj"
    prepared = controller.prepare_save(str(second))
    def fail(*args, **kwargs):
        raise OSError("test disk publication failure")
    monkeypatch.setattr("fdm.project_io.atomic_write_json", fail)
    written = write_project_save(prepared)
    assert not written.result.success
    assert not controller.apply_save(prepared, written).success
    assert session.measurements[record_id].evidence == before
    assert host._project_path == first
    assert not second.exists()
    assert not list(project_assets_root(second).rglob("*.png"))
