from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QDialogButtonBox
import pytest

from fdm.construction_geometry import (
    CircleCenterRadiusDefinition,
    ConstructionEntity,
    ConstructionStyle,
    LineDefinition,
)
from fdm.geometry import Point
from fdm.models import ImageDocument, ProjectState
from fdm.services.digital_slide_store import (
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTile,
)
from fdm.services.export_service import (
    ExportImageRenderMode,
    ExportOptionsSnapshot,
    ExportSelection,
    ExportService,
    RenderedExport,
)
from fdm.ui.dialogs import ExportOptionsDialog
from fdm.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _document_with_constructions() -> ImageDocument:
    document = ImageDocument(
        id="construction-export-document",
        path="/tmp/construction-export.png",
        image_size=(120, 100),
    )
    document.initialize_runtime_state()
    document.add_construction_entity(
        ConstructionEntity(
            id="visible-line",
            name="DO_NOT_EXPORT_CONSTRUCTION",
            definition=LineDefinition(Point(10, 20), Point(110, 20)),
            style=ConstructionStyle(
                stroke_color="#FF0000",
                stroke_width=1.0,
                dashed=False,
                opacity=1.0,
            ),
        ),
        mark_dirty=False,
    )
    document.add_construction_entity(
        ConstructionEntity(
            id="hidden-line",
            name="hidden construction",
            definition=LineDefinition(Point(10, 50), Point(110, 50)),
            visible=False,
            style=ConstructionStyle(
                stroke_color="#0000FF",
                dashed=False,
            ),
        ),
        mark_dirty=False,
    )
    # The definition can be persisted, but resolves to an explicit invalid
    # result because a geometric circle cannot have zero radius.
    document.add_construction_entity(
        ConstructionEntity(
            id="invalid-circle",
            name="invalid construction",
            definition=CircleCenterRadiusDefinition(Point(60, 80), 0.0),
        ),
        mark_dirty=False,
    )
    return document


def _different_pixel_count(left: QImage, right: QImage, y0: int, y1: int) -> int:
    assert left.size() == right.size()
    return sum(
        left.pixel(x, y) != right.pixel(x, y)
        for y in range(y0, y1)
        for x in range(left.width())
    )


def test_construction_overlay_is_opt_in_and_not_a_standalone_export() -> None:
    default = ExportSelection.all_enabled()
    assert default.include_construction_geometry is False

    modifier_only = ExportSelection(include_construction_geometry=True)
    assert modifier_only.any_selected() is False

    snapshot = ExportOptionsSnapshot.from_selection(
        ExportSelection(
            include_measurement_overlay=True,
            include_construction_geometry=True,
        )
    )
    assert snapshot.include_construction_geometry is True
    assert snapshot.to_selection().include_construction_geometry is True


def test_excel_and_csv_never_serialize_construction_geometry() -> None:
    document = _document_with_constructions()
    marker = b"DO_NOT_EXPORT_CONSTRUCTION"
    selection = ExportSelection(
        include_excel=True,
        include_csv=True,
        # Even an explicit image modifier must have no effect on tables.
        include_construction_geometry=True,
    )

    with TemporaryDirectory() as tmp_dir:
        result = ExportService().export_project(
            ProjectState(version="test", documents=[document]),
            tmp_dir,
            selection=selection,
        )

        csv_paths = (
            result["image_summary_csv"],
            result["fiber_details_csv"],
            result["measurement_details_csv"],
        )
        assert all(marker not in path.read_bytes() for path in csv_paths)
        with zipfile.ZipFile(result["xlsx"]) as workbook:
            assert all(marker not in workbook.read(name) for name in workbook.namelist())
        assert not any(key.endswith("overlays") for key in result.outputs)


def test_export_service_only_passes_construction_flag_for_explicit_result_image() -> None:
    document = _document_with_constructions()
    project = ProjectState(version="test", documents=[document])
    calls: list[dict[str, object]] = []

    def renderer(_document, output_path, **kwargs):
        calls.append(kwargs)
        output_path.write_bytes(b"rendered")
        return RenderedExport(output_path, 120, 100)

    with TemporaryDirectory() as tmp_dir:
        ExportService().export_project(
            project,
            tmp_dir,
            selection=ExportSelection(include_measurement_overlay=True),
            overlay_renderer=renderer,
        )
        assert "include_construction_geometry" not in calls[-1]

        calls.clear()
        ExportService().export_project(
            project,
            tmp_dir,
            selection=ExportSelection(
                include_measurement_overlay=True,
                include_scale_overlay=True,
                include_combined_overlay=True,
                include_construction_geometry=True,
            ),
            overlay_renderer=renderer,
        )

    assert len(calls) == 3
    assert all(call["include_construction_geometry"] is True for call in calls)


def test_export_dialog_exposes_explicit_image_only_modifier(
    app: QApplication,
) -> None:
    dialog = ExportOptionsDialog(
        ExportSelection(
            include_measurement_overlay=True,
            include_construction_geometry=True,
        ),
        allow_all_scope=False,
    )
    dialog.show()
    app.processEvents()
    try:
        assert dialog._construction_geometry.isEnabled()
        assert dialog._construction_geometry.isChecked()
        assert dialog.selection().include_construction_geometry is True
        assert "包含辅助几何" in dialog._export_summary_label.text()

        dialog._measurement_overlay.setChecked(False)
        assert not dialog._construction_geometry.isEnabled()
        assert not dialog._button_box.button(
            QDialogButtonBox.StandardButton.Ok
        ).isEnabled()
    finally:
        dialog.close()
        app.processEvents()


def test_main_window_result_image_draws_only_visible_resolved_constructions(
    app: QApplication,
) -> None:
    window = MainWindow()
    try:
        source = QImage(120, 100, QImage.Format.Format_RGB32)
        source.fill(QColor("#FFFFFF"))
        document = _document_with_constructions()
        document.selected_construction_id = "visible-line"
        window._images[document.id] = source

        with TemporaryDirectory() as tmp_dir:
            baseline_path = Path(tmp_dir) / "baseline.png"
            selected_path = Path(tmp_dir) / "selected.png"
            unselected_path = Path(tmp_dir) / "unselected.png"
            window._render_overlay_image(
                document,
                baseline_path,
                include_measurements=False,
                include_scale=False,
                render_mode=ExportImageRenderMode.FULL_RESOLUTION,
            )
            window._render_overlay_image(
                document,
                selected_path,
                include_measurements=False,
                include_scale=False,
                include_construction_geometry=True,
                render_mode=ExportImageRenderMode.FULL_RESOLUTION,
            )
            document.selected_construction_id = None
            window._render_overlay_image(
                document,
                unselected_path,
                include_measurements=False,
                include_scale=False,
                include_construction_geometry=True,
                render_mode=ExportImageRenderMode.FULL_RESOLUTION,
            )

            baseline = QImage(str(baseline_path))
            selected = QImage(str(selected_path))
            unselected = QImage(str(unselected_path))

        assert _different_pixel_count(baseline, selected, 17, 24) > 0
        assert _different_pixel_count(baseline, selected, 47, 54) == 0
        assert _different_pixel_count(baseline, selected, 74, 87) == 0
        # Export deliberately ignores UI selection, so no control handles or
        # selected-object emphasis can leak into the result image.
        assert _different_pixel_count(selected, unselected, 0, 100) == 0
    finally:
        window.close()
        app.processEvents()


def test_digital_slide_construction_export_maps_global_coordinates(
    app: QApplication,
) -> None:
    window = MainWindow()
    try:
        with TemporaryDirectory() as tmp_dir:
            slide_path = Path(tmp_dir) / "construction-viewport.fdmslide"
            store = DigitalSlideStore.create(
                slide_path,
                DigitalSlideManifest(
                    version=1,
                    width=500,
                    height=400,
                    viewport_width=20,
                    viewport_height=16,
                    focus_levels=[0],
                ),
            )
            tile = QImage(20, 16, QImage.Format.Format_RGB32)
            tile.fill(QColor("#FFFFFF"))
            store.write_tile(
                DigitalSlideTile(
                    z_index=0,
                    x=100,
                    y=200,
                    width=20,
                    height=16,
                    stage_x=0,
                    stage_y=0,
                    focus_z=0,
                ),
                tile,
            )
            store.close()
            document = ImageDocument(
                id="construction-slide-export",
                path=str(slide_path),
                image_size=(500, 400),
                document_kind="digital_slide",
                metadata={
                    "digital_slide": {
                        "viewport_origin": [100, 200],
                        "focus_index": 0,
                    }
                },
            )
            document.initialize_runtime_state()
            window._add_digital_slide_document_from_path(
                slide_path,
                document=document,
            )

            with patch(
                "fdm.ui.main_window.draw_construction_entities"
            ) as draw:
                window._render_overlay_image(
                    document,
                    Path(tmp_dir) / "viewport.png",
                    include_measurements=False,
                    include_scale=False,
                    include_construction_geometry=True,
                    render_mode=ExportImageRenderMode.CURRENT_VIEWPORT,
                )

            mapper = draw.call_args.args[2]
            visible_rect = draw.call_args.kwargs["visible_image_rect"]
            assert (mapper(Point(100, 200)).x(), mapper(Point(100, 200)).y()) == (
                0.0,
                0.0,
            )
            assert (mapper(Point(110, 207)).x(), mapper(Point(110, 207)).y()) == (
                10.0,
                7.0,
            )
            assert (visible_rect.left(), visible_rect.top()) == (100.0, 200.0)
            assert (visible_rect.width(), visible_rect.height()) == (20.0, 16.0)
    finally:
        window._reset_workspace()
        window.close()
        app.processEvents()
