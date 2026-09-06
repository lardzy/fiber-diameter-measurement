from threading import Event
from unittest.mock import patch

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage

from fdm.geometry import Point, Line
from fdm.models import ImageDocument
from fdm.project_io import ProjectIO
from fdm.ui.main_window import MainWindow


@pytest.fixture
def workspace(tmp_path):
    window = MainWindow()
    documents = []
    for index in range(2):
        path = tmp_path / f"image-{index}.png"
        image = QImage(128, 128, QImage.Format.Format_RGB32)
        image.fill(0xFFFFFFFF)
        assert image.save(str(path))
        document = ImageDocument(id=f"doc-{index}", path=str(path), image_size=(128, 128))
        document.create_group(color="#ff0000", label="棉")
        document.create_group(color="#00ff00", label="麻")
        document.initialize_runtime_state()
        window._mount_document(document, image, tooltip=str(path))
        documents.append(document)
    window._set_current_document(documents[0].id)
    yield window, documents
    queue = getattr(window, "_measurement_commit_queue", None)
    if queue is not None:
        queue.cancel_all()
    with patch.object(window, "_confirm_close_documents", return_value=True):
        window.close()


def area_payload():
    points = [Point(10, 10), Point(80, 10), Point(80, 80), Point(10, 80)]
    return dict(
        measurement_kind="area", polygon_px=points, area_rings_px=[points], exact_area_px=4900
    )


def delayed(gate, value):
    def compute():
        if not gate.wait(5):
            raise TimeoutError("test worker was not released")
        return value

    return compute


def test_confirm_switch_category_switch_image_then_save_keeps_order_and_origin(workspace, tmp_path):
    window, (first, second) = workspace
    gate = Event()
    original_group = first.active_group_id
    window._queue_measurement_commit(
        first, delayed(gate, area_payload()), mode="magic_segment", group_id=original_group
    )
    assert window._switch_active_group_by_number(2)
    window._on_canvas_line_committed(first.id, "manual", Line(Point(15, 15), Point(25, 25)))
    window._set_current_document(second.id)
    assert not first.measurements  # Switching did not wait for finalization.
    QTimer.singleShot(0, gate.set)
    with patch.object(window, "_focus_current_canvas") as focus:
        result = window.project_session_controller.save_project(str(tmp_path / "accepted.fdmproj"))
    assert result.success
    focus.assert_not_called()  # A late completion cannot steal the new canvas focus.
    assert [item.measurement_kind for item in first.measurements] == ["area", "line"]
    assert [item.fiber_group_id for item in first.measurements] == [
        original_group,
        first.fiber_groups[1].id,
    ]
    assert second.measurements == []
    loaded = ProjectIO.load(tmp_path / "accepted.fdmproj")
    assert [item.to_dict() for item in loaded.documents[0].measurements] == [
        item.to_dict() for item in first.measurements
    ]
    window._set_current_document(first.id)
    window.undo_current_document()
    assert len(first.measurements) == 1
    window.redo_current_document()
    assert len(first.measurements) == 2


def test_undo_cancels_accepted_pending_operation_and_duplicate_publication(workspace):
    window, (document, _) = workspace
    gate = Event()
    window._queue_measurement_commit(
        document,
        delayed(gate, area_payload()),
        mode="magic_segment",
        group_id=document.active_group_id,
    )
    window.undo_current_document()
    gate.set()
    window._flush_pending_measurements()
    assert document.measurements == []
    assert not document.history.can_undo()


def test_uncategorized_acceptance_stays_uncategorized_through_redo(workspace):
    window, (document, _) = workspace
    gate = Event()
    window._queue_measurement_commit(
        document, delayed(gate, area_payload()), mode="magic_segment", group_id=None
    )
    document.set_active_group(document.fiber_groups[1].id)
    gate.set()
    window._flush_pending_measurements()
    assert document.measurements[0].fiber_group_id is None
    window.undo_current_document()
    window.redo_current_document()
    assert document.measurements[0].fiber_group_id is None


def test_changed_pixels_reject_result_and_do_not_silently_export_partial_snapshot(workspace):
    window, (document, _) = workspace
    gate = Event()
    window._queue_measurement_commit(
        document,
        delayed(gate, area_payload()),
        mode="magic_segment",
        group_id=document.active_group_id,
    )
    replacement = QImage(128, 128, QImage.Format.Format_RGB32)
    replacement.fill(0xFF000000)
    window._images[document.id] = replacement
    gate.set()
    with pytest.raises(RuntimeError, match="图片内容已变化"):
        window._flush_pending_measurements(for_snapshot=True)
    assert document.measurements == []


def test_area_geometry_edit_updates_record_without_resetting_table(workspace):
    window, (document, _) = workspace
    window._on_canvas_line_committed(document.id, "magic_segment", area_payload())
    measurement = document.measurements[0]
    model = window._records_controller.source_model
    resets = []
    model.modelReset.connect(lambda: resets.append(True))
    window._on_canvas_measurement_edited(
        document.id,
        measurement.id,
        dict(
            measurement_kind="area",
            polygon_px=[Point(10, 10), Point(60, 10), Point(60, 60)],
            exact_area_px=1250,
        ),
    )
    assert not resets
    assert measurement.area_px == 1250
    assert model.rowCount() == 1


def test_actual_magic_confirm_deduplicates_and_exports_final_geometry(workspace, tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from PySide6.QtWidgets import QDialog
    from fdm.services.mask_region import mask_region
    from fdm.services.prompt_segmentation import magic_mask_to_geometry
    from fdm.services.export_service import ExportSelection, ExportScope
    from fdm.settings import MagicSegmentToolMode

    window, (document, _) = workspace
    canvas = window.current_canvas()
    window._tool_mode = MagicSegmentToolMode.STANDARD
    canvas.set_tool_mode(MagicSegmentToolMode.STANDARD)
    primary = np.zeros((128, 128), bool)
    primary[10:90, 10:90] = True
    subtract = np.zeros_like(primary)
    subtract[30:50, 30:50] = True
    mask, rings, polygon, _ = magic_mask_to_geometry(mask_region(primary))
    session = canvas._magic_segment
    session.primary_mask, session.primary_rings, session.primary_polygon = mask, rings, polygon
    session.confirmed_subtract_masks.append(mask_region(subtract))
    assert window._commit_magic_segment_preview()
    assert not window._commit_magic_segment_preview()
    selection = ExportSelection(scope=ExportScope.CURRENT, include_measurement_overlay=True)
    dialog = SimpleNamespace(
        DialogCode=QDialog.DialogCode,
        exec=lambda: QDialog.DialogCode.Accepted,
        selection=lambda: selection,
    )
    output = tmp_path / "accepted-overlay.png"
    with (
        patch.object(window, "_create_export_options_dialog", return_value=dialog),
        patch.object(window, "_select_export_save_path", return_value=str(output)),
        patch.object(window, "_show_export_information"),
        patch.object(window, "_show_export_warning") as warning,
    ):
        window.export_results(selection)
    warning.assert_not_called()
    assert len(document.measurements) == 1
    measurement = document.measurements[0]
    assert measurement.exact_area_px == 6000
    assert len(measurement.area_rings_px) == 2
    exported = QImage(str(output))
    assert not exported.isNull()
    assert exported.pixelColor(40, 40) == QImage(document.path).pixelColor(40, 40)
    assert exported.pixelColor(20, 20) != QImage(document.path).pixelColor(20, 20)


def test_insertion_history_is_immutable_after_runtime_point_mutation(workspace):
    window, (document, _) = workspace
    window._on_canvas_line_committed(document.id, "magic_segment", area_payload())
    original = document.measurements[0].to_dict()
    document.measurements[0].polygon_px[0].x = 777
    window.undo_current_document()
    window.redo_current_document()
    assert document.measurements[0].to_dict() == original
