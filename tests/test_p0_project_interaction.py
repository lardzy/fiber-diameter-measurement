from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement, ProjectState
from fdm.ui.project_session_controller import ProjectSessionController

try:
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from fdm.ui.canvas import CanvasSelectionRef, DocumentCanvas
    from fdm.ui.main_window import MainWindow
    from fdm.area_display import area_derived_geometry_service

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


class _ProjectHost:
    def __init__(self, documents: list[ImageDocument]) -> None:
        self.project = ProjectState(version="test", documents=documents)


class _FakeMouseEvent:
    def __init__(self, position: QPointF) -> None:
        self._position = position

    def position(self) -> QPointF:
        return self._position

    @staticmethod
    def button():
        return Qt.MouseButton.LeftButton

    @staticmethod
    def modifiers():
        return Qt.KeyboardModifier.NoModifier


class ProjectPersistenceSnapshotTests(unittest.TestCase):
    def test_persistence_snapshot_uses_ordered_references_without_deepcopy(self) -> None:
        first = ImageDocument(id="first", path="first.png", image_size=(10, 10))
        missing = ImageDocument(id="missing", path="missing.png", image_size=(10, 10))
        last = ImageDocument(id="last", path="last.png", image_size=(10, 10))
        added = ImageDocument(id="added", path="added.png", image_size=(10, 10))
        host = _ProjectHost([first, last, added])
        controller = ProjectSessionController(host)  # type: ignore[arg-type]
        controller._begin_project_load([first, missing, last])  # noqa: SLF001
        controller.register_unresolved_document(
            missing,
            attempted_path="/missing.png",
            reason="not found",
            original_index=1,
        )

        with patch("fdm.ui.project_session_controller.copy.deepcopy") as deepcopy_mock:
            snapshot = controller.persistence_snapshot()

        deepcopy_mock.assert_not_called()
        self.assertEqual(
            [identity.document_id for identity in snapshot.documents],
            ["first", "missing", "last", "added"],
        )
        self.assertEqual([identity.document_id for identity in snapshot.unresolved], ["missing"])

    def test_live_document_wins_over_unresolved_copy_with_same_id(self) -> None:
        unresolved = ImageDocument(id="same", path="old.png", image_size=(10, 10))
        live = ImageDocument(id="same", path="relocated.png", image_size=(10, 10))
        host = _ProjectHost([])
        controller = ProjectSessionController(host)  # type: ignore[arg-type]
        controller._begin_project_load([unresolved])  # noqa: SLF001
        controller.register_unresolved_document(
            unresolved,
            attempted_path="/old.png",
            reason="not found",
            original_index=0,
        )
        host.project.documents.append(live)

        snapshot = controller.persistence_snapshot()

        self.assertEqual(snapshot.documents[0].path, "relocated.png")


@unittest.skipUnless(PYSIDE_AVAILABLE, "requires PySide6")
class ProjectInteractionP0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _canvas_with_selected_measurement() -> tuple[ImageDocument, DocumentCanvas]:
        document = ImageDocument(id="image", path="image.png", image_size=(160, 120))
        document.initialize_runtime_state()
        measurement = Measurement(
            id="measurement",
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            line_px=Line(Point(20, 30), Point(100, 30)),
        )
        document.add_measurement(measurement)
        document.select_measurement(measurement.id)
        image = QImage(160, 120, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.white)
        canvas = DocumentCanvas()
        canvas.resize(320, 240)
        canvas.set_document(document, image)
        canvas.set_tool_mode("polygon_area")
        return document, canvas

    def test_polygon_vertices_emit_one_clear_selection_transition(self) -> None:
        document, canvas = self._canvas_with_selected_measurement()
        unified: list[CanvasSelectionRef] = []
        legacy_measurement: list[object] = []
        canvas.objectSelectionChanged.connect(lambda _document_id, ref: unified.append(ref))
        canvas.measurementSelected.connect(
            lambda _document_id, measurement_id: legacy_measurement.append(measurement_id)
        )

        canvas.mousePressEvent(
            _FakeMouseEvent(canvas.image_to_widget(Point(25, 25)))  # type: ignore[arg-type]
        )
        canvas.mousePressEvent(
            _FakeMouseEvent(canvas.image_to_widget(Point(80, 25)))  # type: ignore[arg-type]
        )

        self.assertIsNone(document.view_state.selected_measurement_id)
        self.assertEqual(unified, [CanvasSelectionRef.none()])
        self.assertEqual(legacy_measurement, [""])

    def test_project_dirty_does_not_cross_save_deepcopy_boundary(self) -> None:
        window = MainWindow()
        try:
            document = ImageDocument(id="dense", path="dense.png", image_size=(100, 80))
            document.initialize_runtime_state()
            window.project.documents.append(document)
            window._project_path = Path("/tmp/p0-project.fdmproj")
            window._mark_project_saved()

            with patch(
                "fdm.ui.project_session_controller.copy.deepcopy",
                side_effect=AssertionError("dirty checks must not deepcopy documents"),
            ):
                self.assertFalse(window._project_dirty())
        finally:
            with patch.object(window, "_confirm_close_documents", return_value=True):
                window.close()

    def test_current_document_refresh_updates_project_summary_once(self) -> None:
        window = MainWindow()
        try:
            original = window._update_project_navigation_summary
            with patch.object(
                window,
                "_update_project_navigation_summary",
                wraps=original,
            ) as summary_mock:
                window._update_ui_for_current_document()
            self.assertEqual(summary_mock.call_count, 1)
        finally:
            with patch.object(window, "_confirm_close_documents", return_value=True):
                window.close()

    def test_closing_document_releases_area_geometry_cache_owners(self) -> None:
        window = MainWindow()
        try:
            document = ImageDocument(id="cached", path="cached.png", image_size=(100, 80))
            ring = [Point(10, 10), Point(90, 10), Point(90, 70), Point(10, 70)]
            measurement = Measurement(
                id="cached-area",
                image_id=document.id,
                fiber_group_id=None,
                mode="polygon_area",
                measurement_kind="area",
                polygon_px=list(ring),
                area_rings_px=[list(ring)],
            )
            document.add_measurement(measurement)
            document.mark_session_saved()
            image = QImage(100, 80, QImage.Format.Format_RGB32)
            image.fill(Qt.GlobalColor.white)
            window._mount_document(document, image, tooltip=document.path)
            area_derived_geometry_service.raw_geometry(measurement)
            area_derived_geometry_service.nearest_vertex(measurement, Point(10, 10), 1.0)
            identity = (id(measurement), measurement.id)

            window._remove_document(document.id)

            for cache in (
                area_derived_geometry_service._bounds,
                area_derived_geometry_service._moments,
                area_derived_geometry_service._hole_areas,
                area_derived_geometry_service._raw_paths,
                area_derived_geometry_service._proxies,
                area_derived_geometry_service._hit_indexes,
            ):
                self.assertTrue(all(key[:2] != identity for key in cache))
        finally:
            with patch.object(window, "_confirm_close_documents", return_value=True):
                window.close()


if __name__ == "__main__":
    unittest.main()
