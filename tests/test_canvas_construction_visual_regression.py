from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from fdm.construction_geometry import (
    CircleCenterRadiusDefinition,
    ConstructionEntity,
    FreePointDefinition,
    LineDefinition,
    LineExtent,
)
from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.services.object_snap_service import SnapKind
from fdm.settings import AppSettings
from fdm.ui.canvas import DocumentCanvas


class _MouseMoveEvent:
    def __init__(self, position: QPointF) -> None:
        self._position = QPointF(position)

    def position(self) -> QPointF:
        return QPointF(self._position)

    def modifiers(self) -> Qt.KeyboardModifier:
        return Qt.KeyboardModifier.NoModifier


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def construction_canvas(
    app: QApplication,
) -> tuple[ImageDocument, DocumentCanvas]:
    image = QImage(640, 400, QImage.Format.Format_RGB32)
    image.fill(QColor("#3C4650"))
    document = ImageDocument(
        id="construction-visual-regression",
        path="/tmp/construction-visual-regression.png",
        image_size=(image.width(), image.height()),
    )
    document.initialize_runtime_state()
    entities = (
        ConstructionEntity(
            id="point",
            name="自由点",
            definition=FreePointDefinition(Point(175.0, 95.0)),
        ),
        ConstructionEntity(
            id="line",
            name="有限线段",
            definition=LineDefinition(
                Point(100.0, 290.0),
                Point(315.0, 290.0),
                LineExtent.SEGMENT,
            ),
        ),
        ConstructionEntity(
            id="circle",
            name="圆",
            definition=CircleCenterRadiusDefinition(Point(455.0, 190.0), 72.0),
        ),
    )
    for entity in entities:
        document.add_construction_entity(entity, select=False, mark_dirty=False)

    settings = AppSettings()
    settings.object_snap_enabled = True
    settings.object_snap_kinds = [
        "point",
        "endpoint",
        "midpoint",
        "center",
        "quadrant",
        "intersection",
    ]
    settings.object_snap_aperture_px = 10.0

    canvas = DocumentCanvas()
    canvas.resize(800, 520)
    canvas.set_settings(settings)
    canvas.set_document(document, image)
    canvas.set_tool_mode("select")
    canvas.fit_to_view()
    canvas.show()
    app.processEvents()
    yield document, canvas
    canvas.close()
    app.processEvents()


def _render(canvas: DocumentCanvas, app: QApplication) -> QImage:
    canvas.repaint()
    app.processEvents()
    return canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)


def _pixels_with_color(image: QImage, color: str) -> tuple[tuple[int, int], ...]:
    rgb = QColor(color).rgb()
    return tuple(
        (x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixel(x, y) == rgb
    )


def _assert_color_is_local_to_controls(
    canvas: DocumentCanvas,
    image: QImage,
    *,
    color: str,
    control_points: tuple[Point, ...],
    radius_screen_px: float = 10.0,
) -> None:
    pixels = _pixels_with_color(image, color)
    assert len(pixels) >= 20, "test scene must contain a visible interaction marker"
    mapped = tuple(canvas.image_to_widget(point) for point in control_points)
    escaped = tuple(
        (x, y)
        for x, y in pixels
        if all(
            abs(x - center.x()) > radius_screen_px
            or abs(y - center.y()) > radius_screen_px
            for center in mapped
        )
    )
    assert escaped == (), (
        f"interaction color {color} escaped its local controls; "
        f"first pixels={escaped[:10]}"
    )


def _reset_interaction(canvas: DocumentCanvas, document: ImageDocument) -> None:
    document.select_construction(None)
    canvas._hovered_construction_id = None  # noqa: SLF001 - visual state fixture
    canvas._hovered_construction_handle = None  # noqa: SLF001
    canvas._object_snap_engine.clear_hysteresis()  # noqa: SLF001
    canvas._set_active_snap_candidate(None, repaint=False)  # noqa: SLF001
    canvas.update()


def test_endpoint_hover_marker_does_not_paint_towards_canvas_origin(
    construction_canvas: tuple[ImageDocument, DocumentCanvas],
    app: QApplication,
) -> None:
    document, canvas = construction_canvas
    _reset_interaction(canvas, document)
    start = Point(100.0, 290.0)
    endpoint = Point(315.0, 290.0)

    canvas.mouseMoveEvent(_MouseMoveEvent(canvas.image_to_widget(endpoint)))

    assert canvas._active_snap_candidate is not None  # noqa: SLF001
    assert canvas._active_snap_candidate.kind is SnapKind.ENDPOINT  # noqa: SLF001
    _assert_color_is_local_to_controls(
        canvas,
        _render(canvas, app),
        color="#F4D35E",
        control_points=(start, endpoint),
    )


def test_quadrant_hover_marker_does_not_paint_towards_canvas_origin(
    construction_canvas: tuple[ImageDocument, DocumentCanvas],
    app: QApplication,
) -> None:
    document, canvas = construction_canvas
    _reset_interaction(canvas, document)
    center = Point(455.0, 190.0)
    radius_handle = Point(527.0, 190.0)
    quadrant = Point(455.0, 118.0)

    canvas.mouseMoveEvent(_MouseMoveEvent(canvas.image_to_widget(quadrant)))

    assert canvas._active_snap_candidate is not None  # noqa: SLF001
    assert canvas._active_snap_candidate.kind is SnapKind.QUADRANT  # noqa: SLF001
    _assert_color_is_local_to_controls(
        canvas,
        _render(canvas, app),
        color="#F4D35E",
        control_points=(center, radius_handle, quadrant),
    )


@pytest.mark.parametrize(
    ("entity_id", "control_points"),
    [
        ("point", (Point(175.0, 95.0),)),
        ("line", (Point(100.0, 290.0), Point(315.0, 290.0))),
        ("circle", (Point(455.0, 190.0), Point(527.0, 190.0))),
    ],
)
def test_selected_construction_handles_do_not_paint_towards_canvas_origin(
    construction_canvas: tuple[ImageDocument, DocumentCanvas],
    app: QApplication,
    entity_id: str,
    control_points: tuple[Point, ...],
) -> None:
    document, canvas = construction_canvas
    _reset_interaction(canvas, document)

    canvas.set_selected_construction(entity_id)

    _assert_color_is_local_to_controls(
        canvas,
        _render(canvas, app),
        color="#58C4C7",
        control_points=control_points,
    )
