"""Real Qt worker / persistence smoke probe for the packaged runtime."""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PySide6.QtCore import QTimer

from fdm.models import ImageDocument, ProjectState, project_assets_root
from fdm.project_io import ProjectIO
from fdm.services.raster_io import numpy_to_raster_plane
from fdm.ui.project_save_coordinator import ProjectSaveCoordinator
from fdm.ui.project_session_controller import ProjectSessionController
from fdm.ui.qt_raster_runtime import ensure_raster_application


class _ProbeHost:
    def __init__(self):
        self.plane = numpy_to_raster_plane(np.arange(64, dtype=np.uint16).reshape(8, 8))
        document = ImageDocument(
            "probe",
            "imports/probe.png",
            (8, 8),
            source_type="project_asset",
            raster_pixel_type=self.plane.pixel_type,
        )
        document.initialize_runtime_state()
        self.project = ProjectState(version="probe", documents=[document])
        self._project_path = None
        self.project_session_controller = ProjectSessionController(self)

    def _project_dirty(self):
        return False

    def _document_has_unsaved_project_changes(self, document):
        return document.dirty_flags.session_dirty

    def _project_asset_raster_for_save(self, document):
        return self.plane, None

    def _project_asset_image_for_save(self, document):
        return None

    def _mark_project_saved(self):
        pass

    def _update_ui_for_current_document(self):
        pass

    def _show_status_message(self, message, timeout_ms=0):
        pass

    def _remember_recent_directory(self, **kwargs):
        pass


def run_project_save_self_check() -> dict:
    app = ensure_raster_application("project-save-self-check")
    cases = {}
    with TemporaryDirectory(prefix="fdm-save-probe-") as temporary:
        root = Path(temporary)
        host = _ProbeHost()
        coordinator = ProjectSaveCoordinator(host)
        document = host.project.documents[0]
        path = root / "后台保存.fdmproj"
        ticks = []
        timer = QTimer(coordinator)
        timer.setInterval(0)
        timer.timeout.connect(lambda: ticks.append(True) if not ticks else None)
        timer.start()

        def finish():
            deadline = time.monotonic() + 30
            while coordinator.busy:
                app.processEvents()
                if time.monotonic() > deadline:
                    raise TimeoutError("background save did not complete")
                time.sleep(0.001)
            app.processEvents()

        cases["background_completion"] = (
            coordinator.request_save(str(path)) and coordinator.busy
        )
        # This edit occurs after the writer has received its private snapshot.
        document.metadata["after_request"] = True
        document.mark_session_dirty()
        finish()
        cases["background_completion"] &= (
            path.is_file() and coordinator.status.phase == "saved_newer"
        )
        restored = ProjectIO.load(path).documents[0]
        cases["event_dispatch"] = bool(ticks)
        cases["snapshot_isolation"] = "after_request" not in restored.metadata
        cases["preserves_newer_edits"] = (
            document.dirty_flags.session_dirty and document.metadata["after_request"]
        )
        receipt = host.project_session_controller._raster_asset_receipts[document.id]
        coordinator.request_save()
        finish()
        cases["repeat_save"] = (
            not document.dirty_flags.session_dirty
            and host.project_session_controller._raster_asset_receipts[document.id]
            is receipt
            and ProjectIO.load(path).documents[0].metadata.get("after_request") is True
        )
        old_bytes = path.read_bytes()
        blocked = root / "directory.fdmproj"
        blocked.mkdir()
        document.mark_session_dirty()
        coordinator.request_save(str(blocked))
        finish()
        cases["failure_rollback"] = (
            coordinator.status.phase == "failed"
            and path.read_bytes() == old_bytes
            and host._project_path == path
            and document.dirty_flags.session_dirty
            and not any(p.is_file() for p in project_assets_root(blocked).rglob("*"))
        )
        cases["thread_idle"] = coordinator._thread is None and not coordinator.busy
        timer.stop()
        coordinator.deleteLater()
        app.processEvents()
    return {"ok": all(cases.values()), "revision": 1, "cases": cases}
