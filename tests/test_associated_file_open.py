from __future__ import annotations

import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

from fdm.application_launch import build_application_open_request
from fdm.lifecycle import AcquisitionDisposition, TransitionIntent
from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore
from fdm.ui.associated_file_controller import AssociatedSlideDisposition
from fdm.ui.main_window import MainWindow


def _create_slide(path: Path) -> None:
    store = DigitalSlideStore.create(
        path,
        DigitalSlideManifest(
            version=1,
            width=64,
            height=48,
            viewport_width=32,
            viewport_height=24,
            focus_levels=[0],
        ),
    )
    store.close()


class AssociatedFileOpenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_slide_is_added_to_saved_project_and_marks_it_dirty(self) -> None:
        window = MainWindow()
        try:
            with TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                slide_path = root / "加入项目.fdmslide"
                second_slide_path = root / "第二个切片.fdmslide"
                _create_slide(slide_path)
                _create_slide(second_slide_path)
                window._project_path = root / "current.fdmproj"
                window._mark_project_saved()

                with patch.object(
                    window.associated_file_open_controller,
                    "_choose_slide_disposition",
                    return_value=AssociatedSlideDisposition.ADD_TO_CURRENT,
                ) as choice_mock:
                    window.associated_file_open_controller._open_digital_slides(
                        [slide_path, second_slide_path]
                    )

                choice_mock.assert_called_once()
                self.assertEqual(len(window.project.documents), 2)
                self.assertTrue(all(item.is_digital_slide() for item in window.project.documents))
                self.assertEqual(window._project_path, root / "current.fdmproj")
                self.assertTrue(window._project_dirty())
        finally:
            window._reset_workspace()
            window.close()

    def test_slide_can_replace_project_with_standalone_workspace(self) -> None:
        window = MainWindow()
        try:
            with TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                slide_path = root / "独立工作区.fdmslide"
                _create_slide(slide_path)
                window._project_path = root / "current.fdmproj"
                window._mark_project_saved()

                with patch.object(
                    window.associated_file_open_controller,
                    "_choose_slide_disposition",
                    return_value=AssociatedSlideDisposition.STANDALONE_WORKSPACE,
                ):
                    window.associated_file_open_controller._open_digital_slides([slide_path])

                self.assertIsNone(window._project_path)
                self.assertEqual(len(window.project.documents), 1)
                self.assertEqual(Path(window.project.documents[0].path), slide_path.resolve())
        finally:
            window._reset_workspace()
            window.close()

    def test_standalone_slide_respects_cancelled_close_confirmation(self) -> None:
        window = MainWindow()
        try:
            with TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                slide_path = root / "取消独立.fdmslide"
                _create_slide(slide_path)
                project_path = root / "current.fdmproj"
                window._project_path = project_path

                with (
                    patch.object(
                        window.associated_file_open_controller,
                        "_choose_slide_disposition",
                        return_value=AssociatedSlideDisposition.STANDALONE_WORKSPACE,
                    ),
                    patch.object(window, "_confirm_close_documents", return_value=False),
                    patch.object(window, "stop_live_preview") as stop_preview,
                    patch.object(window, "_prepare_transition") as prepare_transition,
                    patch.object(window, "_reset_workspace") as reset_mock,
                    patch.object(window, "_open_image_requests") as open_mock,
                ):
                    window.associated_file_open_controller._open_digital_slides([slide_path])

                stop_preview.assert_not_called()
                prepare_transition.assert_not_called()
                reset_mock.assert_not_called()
                open_mock.assert_not_called()
                self.assertEqual(window._project_path, project_path)
        finally:
            window._reset_workspace()
            window.close()

    def test_active_acquisition_is_not_stopped_when_standalone_confirmation_is_cancelled(self) -> None:
        window = MainWindow()
        try:
            with TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                slide_path = root / "采集中取消独立.fdmslide"
                _create_slide(slide_path)
                window._project_path = root / "current.fdmproj"
                call_order: list[str] = []

                with (
                    patch.object(
                        window.associated_file_open_controller,
                        "_choose_slide_disposition",
                        return_value=AssociatedSlideDisposition.STANDALONE_WORKSPACE,
                    ),
                    patch.object(window, "_slide_acquisition_active", return_value=True),
                    patch.object(
                        window,
                        "_preflight_acquisition_disposition",
                        side_effect=lambda _intent: call_order.append("acquisition")
                        or AcquisitionDisposition.KEEP_PARTIAL,
                    ),
                    patch.object(
                        window,
                        "_confirm_close_documents",
                        side_effect=lambda _documents: call_order.append("confirm") or False,
                    ),
                    patch.object(window, "_prepare_transition") as prepare_transition,
                    patch.object(window, "_reset_workspace") as reset_workspace,
                    patch.object(window, "_open_image_requests") as open_requests,
                ):
                    window.associated_file_open_controller._open_digital_slides([slide_path])

                self.assertEqual(call_order, ["acquisition", "confirm"])
                prepare_transition.assert_not_called()
                reset_workspace.assert_not_called()
                open_requests.assert_not_called()
        finally:
            window._reset_workspace()
            window.close()

    def test_slide_open_cancel_and_duplicate_do_not_change_project(self) -> None:
        window = MainWindow()
        try:
            with TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                slide_path = root / "重复.fdmslide"
                _create_slide(slide_path)
                project_path = root / "current.fdmproj"
                window._project_path = project_path
                window._mark_project_saved()

                with patch.object(
                    window.associated_file_open_controller,
                    "_choose_slide_disposition",
                    return_value=AssociatedSlideDisposition.CANCEL,
                ):
                    window.associated_file_open_controller._open_digital_slides([slide_path])
                self.assertEqual(window.project.documents, [])
                self.assertEqual(window._project_path, project_path)

                window._project_path = None
                window.associated_file_open_controller._open_digital_slides([slide_path])
                window._project_path = project_path
                window._mark_project_saved()
                with patch.object(
                    window.associated_file_open_controller,
                    "_choose_slide_disposition",
                ) as choice_mock:
                    window.associated_file_open_controller._open_digital_slides([slide_path])

                choice_mock.assert_not_called()
                self.assertEqual(len(window.project.documents), 1)
        finally:
            window._reset_workspace()
            window.close()

    def test_invalid_slide_and_cancelled_acquisition_preserve_workspace(self) -> None:
        window = MainWindow()
        try:
            with TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                invalid_path = root / "损坏.fdmslide"
                invalid_path.write_bytes(b"not sqlite")
                with patch.object(QMessageBox, "warning") as warning_mock:
                    window.associated_file_open_controller._open_digital_slides([invalid_path])
                warning_mock.assert_called_once()
                self.assertEqual(window.project.documents, [])

                slide_path = root / "采集中.fdmslide"
                _create_slide(slide_path)
                with (
                    patch.object(window, "_slide_acquisition_active", return_value=True),
                    patch.object(
                        window,
                        "_preflight_acquisition_disposition",
                        return_value=AcquisitionDisposition.CANCEL,
                    ) as preflight_mock,
                    patch.object(window, "_prepare_transition") as transition_mock,
                    patch.object(window, "_open_image_requests") as open_mock,
                ):
                    window.associated_file_open_controller._open_digital_slides([slide_path])

                preflight_mock.assert_called_once_with(TransitionIntent.OPEN_DOCUMENT)
                transition_mock.assert_not_called()
                open_mock.assert_not_called()
                self.assertEqual(window.project.documents, [])
        finally:
            window._reset_workspace()
            window.close()

    def test_slide_preflight_does_not_modify_unrelated_sqlite_file(self) -> None:
        window = MainWindow()
        try:
            with TemporaryDirectory() as tmpdir:
                invalid_path = Path(tmpdir) / "其他数据.fdmslide"
                connection = sqlite3.connect(invalid_path)
                connection.execute("CREATE TABLE sentinel(value TEXT NOT NULL)")
                connection.execute("INSERT INTO sentinel(value) VALUES('keep')")
                connection.commit()
                connection.close()
                original_bytes = invalid_path.read_bytes()

                with patch.object(QMessageBox, "warning") as warning_mock:
                    window.associated_file_open_controller._open_digital_slides([invalid_path])

                warning_mock.assert_called_once()
                self.assertEqual(invalid_path.read_bytes(), original_bytes)
                self.assertFalse(Path(f"{invalid_path}-wal").exists())
                self.assertFalse(Path(f"{invalid_path}-shm").exists())
                self.assertEqual(window.project.documents, [])
        finally:
            window._reset_workspace()
            window.close()

    def test_external_request_waits_while_image_loading(self) -> None:
        window = MainWindow()
        try:
            with TemporaryDirectory() as tmpdir:
                slide_path = Path(tmpdir) / "排队.fdmslide"
                _create_slide(slide_path)
                request = build_application_open_request(
                    [slide_path],
                    source="test",
                    request_id="queued-request",
                )
                repeated_request = build_application_open_request(
                    [slide_path],
                    source="test",
                    request_id="repeated-path-request",
                )
                with patch.object(window, "is_image_loading", return_value=True):
                    window.enqueue_application_open_request(request)
                    window.enqueue_application_open_request(repeated_request)
                    self.app.processEvents()
                    self.assertEqual(
                        window.associated_file_open_controller.pending_request_count(),
                        1,
                    )
                window.associated_file_open_controller._dispatch_next()

                self.assertEqual(
                    window.associated_file_open_controller.pending_request_count(),
                    0,
                )
                self.assertEqual(len(window.project.documents), 1)
        finally:
            window._reset_workspace()
            window.close()


if __name__ == "__main__":
    unittest.main()
