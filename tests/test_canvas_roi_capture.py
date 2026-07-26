from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.project_roi import (
    EllipseRoiGeometry,
    FreehandRoiGeometry,
    PolygonRoiGeometry,
    ProjectRoi,
    ProjectRoiKind,
    RectangleRoiGeometry,
    RoiBooleanExpression,
    RoiBooleanOperator,
    RoiPoint,
)
from fdm.ui.canvas import DocumentCanvas, RoiGeometryCommit


@pytest.fixture(scope="module", autouse=True)
def _application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas():
    widget = DocumentCanvas()
    widget.resize(160, 120)
    image = QImage(80, 60, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    document = ImageDocument(
        id="roi-document",
        path="/tmp/roi-canvas.png",
        image_size=(80, 60),
    )
    widget.set_document(document, image)
    widget._zoom = 2.0  # noqa: SLF001
    widget._pan = Point(9.0, 7.0)  # noqa: SLF001
    yield widget
    widget.clear_document()
    widget.close()


def _mouse_event(
    canvas: DocumentCanvas,
    event_type: QEvent.Type,
    image_point: Point,
    *,
    button: Qt.MouseButton,
    buttons: Qt.MouseButton,
) -> QMouseEvent:
    widget_point = canvas.image_to_widget(image_point)
    return QMouseEvent(
        event_type,
        QPointF(widget_point),
        QPointF(widget_point),
        QPointF(widget_point),
        button,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )


def _press(canvas: DocumentCanvas, point: Point) -> None:
    canvas.mousePressEvent(
        _mouse_event(
            canvas,
            QEvent.Type.MouseButtonPress,
            point,
            button=Qt.MouseButton.LeftButton,
            buttons=Qt.MouseButton.LeftButton,
        )
    )


def _move(canvas: DocumentCanvas, point: Point) -> None:
    canvas.mouseMoveEvent(
        _mouse_event(
            canvas,
            QEvent.Type.MouseMove,
            point,
            button=Qt.MouseButton.NoButton,
            buttons=Qt.MouseButton.LeftButton,
        )
    )


def _release(canvas: DocumentCanvas, point: Point) -> None:
    canvas.mouseReleaseEvent(
        _mouse_event(
            canvas,
            QEvent.Type.MouseButtonRelease,
            point,
            button=Qt.MouseButton.LeftButton,
            buttons=Qt.MouseButton.NoButton,
        )
    )


@pytest.mark.parametrize(
    ("kind", "geometry_type"),
    [
        (ProjectRoiKind.RECTANGLE, RectangleRoiGeometry),
        (ProjectRoiKind.ELLIPSE, EllipseRoiGeometry),
    ],
)
def test_drag_roi_uses_original_image_coordinates_and_restores_tool(
    canvas: DocumentCanvas,
    kind: ProjectRoiKind,
    geometry_type: type,
) -> None:
    commits: list[RoiGeometryCommit] = []
    canvas.roiGeometryCommitted.connect(commits.append)
    canvas.set_tool_mode("polygon_area")
    _press(canvas, Point(2.0, 2.0))
    assert canvas._drawing_polygon_points  # noqa: SLF001

    expected_request_id = f"request-{kind.value}"
    assert canvas.begin_roi_capture(kind, request_id=expected_request_id)
    assert not canvas._drawing_polygon_points  # noqa: SLF001
    _press(canvas, Point(11.25, 8.5))
    _move(canvas, Point(36.75, 29.0))
    _release(canvas, Point(36.75, 29.0))

    assert len(commits) == 1
    commit = commits[0]
    assert commit.document_id == "roi-document"
    assert commit.kind is kind
    assert isinstance(commit.geometry, geometry_type)
    assert commit.geometry.x == pytest.approx(11.25)
    assert commit.geometry.y == pytest.approx(8.5)
    assert commit.geometry.width == pytest.approx(25.5)
    assert commit.geometry.height == pytest.approx(20.5)
    assert commit.request_id == expected_request_id
    assert canvas._tool_mode == "polygon_area"  # noqa: SLF001
    assert not canvas._document.measurements  # noqa: SLF001
    assert not canvas._document.overlay_annotations  # noqa: SLF001


def test_polygon_roi_supports_double_click_and_enter(canvas: DocumentCanvas) -> None:
    commits: list[RoiGeometryCommit] = []
    canvas.roiGeometryCommitted.connect(commits.append)

    assert canvas.begin_roi_capture(ProjectRoiKind.POLYGON)
    _press(canvas, Point(5.0, 5.0))
    _press(canvas, Point(30.0, 5.0))
    double_click = _mouse_event(
        canvas,
        QEvent.Type.MouseButtonDblClick,
        Point(20.0, 25.0),
        button=Qt.MouseButton.LeftButton,
        buttons=Qt.MouseButton.LeftButton,
    )
    canvas.mouseDoubleClickEvent(double_click)

    assert len(commits) == 1
    geometry = commits[-1].geometry
    assert isinstance(geometry, PolygonRoiGeometry)
    assert [(point.x, point.y) for point in geometry.rings[0]] == [
        (5.0, 5.0),
        (30.0, 5.0),
        (20.0, 25.0),
    ]

    assert canvas.begin_roi_capture("polygon")
    _press(canvas, Point(7.0, 9.0))
    _press(canvas, Point(25.0, 9.0))
    _press(canvas, Point(18.0, 28.0))
    canvas.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Return,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert len(commits) == 2
    assert isinstance(commits[-1].geometry, PolygonRoiGeometry)


def test_freehand_roi_commits_raw_trace_without_creating_measurement(
    canvas: DocumentCanvas,
) -> None:
    commits: list[RoiGeometryCommit] = []
    canvas.roiGeometryCommitted.connect(commits.append)

    assert canvas.begin_roi_capture(ProjectRoiKind.FREEHAND)
    _press(canvas, Point(4.0, 8.0))
    _move(canvas, Point(28.0, 8.0))
    _move(canvas, Point(28.0, 24.0))
    _release(canvas, Point(4.0, 24.0))

    assert len(commits) == 1
    geometry = commits[0].geometry
    assert isinstance(geometry, FreehandRoiGeometry)
    assert [(point.x, point.y) for point in geometry.rings[0]] == [
        (4.0, 8.0),
        (28.0, 8.0),
        (28.0, 24.0),
        (4.0, 24.0),
    ]
    assert not canvas._document.measurements  # noqa: SLF001
    assert not canvas._document.overlay_annotations  # noqa: SLF001


def test_escape_cancels_roi_and_restores_previous_tool(
    canvas: DocumentCanvas,
) -> None:
    commits: list[RoiGeometryCommit] = []
    canvas.roiGeometryCommitted.connect(commits.append)
    canvas.set_tool_mode("count")
    assert canvas.begin_roi_capture(ProjectRoiKind.POLYGON)
    _press(canvas, Point(5.0, 5.0))

    canvas.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Escape,
            Qt.KeyboardModifier.NoModifier,
        )
    )

    assert not commits
    assert canvas._roi_capture is None  # noqa: SLF001
    assert canvas._tool_mode == "count"  # noqa: SLF001


def _square(
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[RoiPoint, ...]:
    return (
        RoiPoint(left, top),
        RoiPoint(right, top),
        RoiPoint(right, bottom),
        RoiPoint(left, bottom),
    )


def test_project_roi_display_preserves_holes_and_composite_difference(
    canvas: DocumentCanvas,
) -> None:
    canvas._zoom = 1.0  # noqa: SLF001
    canvas._pan = Point(0.0, 0.0)  # noqa: SLF001
    canvas.resize(80, 60)
    polygon = ProjectRoi(
        id="donut",
        document_id="roi-document",
        name="含孔洞 ROI",
        geometry=PolygonRoiGeometry(
            (
                _square(4.0, 4.0, 34.0, 34.0),
                _square(14.0, 14.0, 24.0, 24.0),
            )
        ),
        color="#E76F51",
    )
    left = ProjectRoi(
        id="left",
        document_id="roi-document",
        name="组合主体",
        geometry=RectangleRoiGeometry(40.0, 4.0, 32.0, 32.0),
        visible=False,
    )
    cutout = ProjectRoi(
        id="cutout",
        document_id="roi-document",
        name="组合孔洞",
        geometry=EllipseRoiGeometry(50.0, 14.0, 12.0, 12.0),
        visible=False,
    )
    composite = ProjectRoi(
        id="difference",
        document_id="roi-document",
        name="差集",
        geometry=RoiBooleanExpression(
            RoiBooleanOperator.DIFFERENCE,
            ("left", "cutout"),
        ),
        color="#2A9D8F",
    )
    lookup = {
        roi.id: roi for roi in (polygon, left, cutout, composite)
    }
    canvas.set_project_rois(tuple(lookup.values()), lookup)

    rendered = QImage(
        canvas.size(),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    rendered.fill(Qt.GlobalColor.transparent)
    canvas.render(rendered)

    white = QColor(Qt.GlobalColor.white).name()
    assert QColor(rendered.pixel(9, 9)).name() != white
    assert QColor(rendered.pixel(19, 19)).name() == white
    assert QColor(rendered.pixel(44, 9)).name() != white
    assert QColor(rendered.pixel(56, 20)).name() == white
