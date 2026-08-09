from __future__ import annotations

from dataclasses import replace
import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from fdm.construction_geometry import (
    ArraySide,
    CircleCenterRadiusDefinition,
    CircleThreePointDefinition,
    ConstructionResolver,
    ConstructionEntity,
    FreePointDefinition,
    LineDefinition,
    LineAxisConstraint,
    LineExtent,
    LiveFeatureRef,
    FrozenFeatureSnapshot,
    MidpointDefinition,
    ParallelArrayDefinition,
    ParallelThroughPointDefinition,
    PerpendicularDefinition,
    PointCircleTangentDefinition,
    ResolvedCircle,
    ResolvedLine,
    ResolvedPoint,
    SourceObjectKind,
    TangentTangentRadiusCircleDefinition,
)
from fdm.construction_document import ResolvedSourceCandidate, make_construction_resolver
from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement
from fdm.services.object_snap_service import SnapKind
from fdm.settings import AppSettings
from fdm.ui.canvas import DocumentCanvas

class _MouseEvent:
    def __init__(
        self,
        position: QPointF,
        *,
        button: Qt.MouseButton = Qt.MouseButton.LeftButton,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        self._position = QPointF(position)
        self._button = button
        self._modifiers = modifiers
        self.accepted = False

    def position(self) -> QPointF:
        return QPointF(self._position)

    def button(self) -> Qt.MouseButton:
        return self._button

    def modifiers(self) -> Qt.KeyboardModifier:
        return self._modifiers

    def accept(self) -> None:
        self.accepted = True


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas_fixture(app: QApplication) -> tuple[ImageDocument, QImage, DocumentCanvas]:
    image = QImage(200, 120, QImage.Format.Format_RGB32)
    image.fill(QColor("#FFFFFF"))
    document = ImageDocument(
        id="canvas-construction",
        path="/tmp/canvas-construction.png",
        image_size=(image.width(), image.height()),
    )
    document.initialize_runtime_state()
    canvas = DocumentCanvas()
    canvas.resize(320, 240)
    canvas.set_document(document, image)
    yield document, image, canvas
    canvas.close()
    app.processEvents()


def _click_image(canvas: DocumentCanvas, point: Point) -> None:
    position = canvas.image_to_widget(point)
    canvas.mousePressEvent(_MouseEvent(position))
    canvas.mouseReleaseEvent(_MouseEvent(position))


def _render_canvas(canvas: DocumentCanvas, app: QApplication) -> QImage:
    canvas.show()
    canvas.repaint()
    app.processEvents()
    return canvas.grab().toImage()


def _different_pixel_count(left: QImage, right: QImage) -> int:
    assert left.size() == right.size()
    return sum(
        left.pixel(x, y) != right.pixel(x, y)
        for y in range(left.height())
        for x in range(left.width())
    )


def test_source_candidate_label_respects_kind_when_ids_overlap(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    shared_id = "same-source-label-id"
    document.add_construction_entity(
        ConstructionEntity(
            id=shared_id,
            name="同 ID 辅助对象",
            definition=LineDefinition(Point(10.0, 10.0), Point(30.0, 10.0)),
        ),
        select=False,
        mark_dirty=False,
    )
    document.add_measurement(
        Measurement(
            id=shared_id,
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(10.0, 20.0), Point(30.0, 20.0)),
        )
    )
    candidate = ResolvedSourceCandidate(
        object_id=shared_id,
        object_kind=SourceObjectKind.MEASUREMENT,
        feature="geometry",
        geometry=ResolvedLine(Point(10.0, 20.0), Point(30.0, 20.0)),
        distance_px=0.0,
    )

    label = canvas._construction_source_candidate_label(candidate)  # noqa: SLF001

    assert label.startswith("测量对象 · ")
    assert "同 ID 辅助对象" not in label

    positive = canvas._construction_source_candidate_label(  # noqa: SLF001
        ResolvedSourceCandidate(
            object_id=shared_id,
            object_kind=SourceObjectKind.CONSTRUCTION,
            feature="line:+1",
            geometry=ResolvedLine(Point(10.0, 20.0), Point(30.0, 20.0)),
            distance_px=0.0,
        )
    )
    negative = canvas._construction_source_candidate_label(  # noqa: SLF001
        ResolvedSourceCandidate(
            object_id=shared_id,
            object_kind=SourceObjectKind.CONSTRUCTION,
            feature="line:-1",
            geometry=ResolvedLine(Point(10.0, 20.0), Point(30.0, 20.0)),
            distance_px=0.0,
        )
    )
    assert positive.endswith("偏移线 +1")
    assert negative.endswith("偏移线 -1")


def test_construction_entry_creates_free_point_and_finite_line(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    created: list[ConstructionEntity] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )

    canvas.set_tool_mode("construction", construction_kind="point")
    _click_image(canvas, Point(30.0, 25.0))

    assert len(created) == 1
    assert isinstance(created[0].definition, FreePointDefinition)
    assert created[0].definition.point == Point(30.0, 25.0)

    canvas.set_tool_mode("construction", construction_kind="segment")
    _click_image(canvas, Point(40.0, 45.0))
    assert len(created) == 1
    _click_image(canvas, Point(110.0, 65.0))

    assert len(created) == 2
    definition = created[1].definition
    assert isinstance(definition, LineDefinition)
    assert definition.extent is LineExtent.SEGMENT
    assert definition.start == Point(40.0, 45.0)
    assert definition.end == Point(110.0, 65.0)


@pytest.mark.parametrize(
    ("tool", "constraint"),
    [
        ("horizontal_line", LineAxisConstraint.HORIZONTAL),
        ("vertical_line", LineAxisConstraint.VERTICAL),
    ],
)
def test_axis_line_creation_and_drag_preserve_orientation(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
    tool: str,
    constraint: LineAxisConstraint,
) -> None:
    document, _image, canvas = canvas_fixture
    created: list[ConstructionEntity] = []
    edits: list[ConstructionEntity] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.constructionEdited.connect(
        lambda document_id, _entity_id, entity: (
            document_id == document.id and edits.append(entity)
        )
    )
    canvas.set_tool_mode("construction", construction_kind=tool)
    _click_image(canvas, Point(45.0, 35.0))

    assert len(created) == 1
    entity = created[0]
    assert isinstance(entity.definition, LineDefinition)
    assert entity.definition.axis_constraint is constraint
    document.add_construction_entity(entity, select=False, mark_dirty=False)

    canvas.set_tool_mode("select")
    _click_image(canvas, entity.definition.start)
    start = canvas.image_to_widget(entity.definition.start)
    destination = canvas.image_to_widget(Point(80.0, 70.0))
    canvas.mousePressEvent(_MouseEvent(start))
    canvas.mouseMoveEvent(_MouseEvent(destination))
    canvas.mouseReleaseEvent(_MouseEvent(destination))

    assert len(edits) == 1
    moved = edits[0].definition
    assert isinstance(moved, LineDefinition)
    assert moved.axis_constraint is constraint
    if constraint is LineAxisConstraint.HORIZONTAL:
        assert moved.start.y == pytest.approx(moved.end.y)
        assert moved.start.y == pytest.approx(70.0)
    else:
        assert moved.start.x == pytest.approx(moved.end.x)
        assert moved.start.x == pytest.approx(80.0)


def test_degenerate_three_point_circle_stays_in_session_until_solvable(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    created: list[ConstructionEntity] = []
    states: list[dict[str, object]] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.constructionCommandChanged.connect(
        lambda document_id, payload: (
            document_id == document.id and states.append(dict(payload))
        )
    )
    canvas.set_tool_mode("construction", construction_kind="circle_3p")

    _click_image(canvas, Point(20.0, 20.0))
    _click_image(canvas, Point(50.0, 20.0))
    _click_image(canvas, Point(80.0, 20.0))

    assert created == []
    assert states
    assert "共线" in str(states[-1]["invalid_reason"])
    assert states[-1]["point_count"] == 2

    _click_image(canvas, Point(50.0, 55.0))

    assert len(created) == 1
    assert isinstance(created[0].definition, CircleThreePointDefinition)


def test_construction_right_click_steps_back_without_starting_pan(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    states: list[dict[str, object]] = []
    canvas.constructionCommandChanged.connect(
        lambda document_id, payload: (
            document_id == document.id and states.append(dict(payload))
        )
    )
    canvas.set_tool_mode("construction", construction_kind="segment")
    _click_image(canvas, Point(40.0, 35.0))
    assert states and states[-1]["point_count"] == 1

    position = canvas.image_to_widget(Point(40.0, 35.0))
    event = _MouseEvent(position, button=Qt.MouseButton.RightButton)
    canvas.mousePressEvent(event)

    assert event.accepted
    assert states[-1]["point_count"] == 0
    assert not canvas._panning  # noqa: SLF001 - right-click command contract


def test_circle_can_finish_from_mouse_first_numeric_radius(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    created: list[ConstructionEntity] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.set_tool_mode(
        "construction",
        construction_kind="circle_center_radius",
    )
    canvas.set_construction_parameter("distance", 12.5)
    _click_image(canvas, Point(75.0, 45.0))

    assert canvas.finish_construction_command()
    assert len(created) == 1
    definition = created[0].definition
    assert isinstance(definition, CircleCenterRadiusDefinition)
    assert definition.center == Point(75.0, 45.0)
    assert definition.radius == pytest.approx(12.5)


def test_point_circle_tangent_requires_explicit_solution_branch(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    point_entity = ConstructionEntity(
        id="tangent-source-point",
        name="切线点",
        definition=FreePointDefinition(Point(25.0, 60.0)),
    )
    circle_entity = ConstructionEntity(
        id="tangent-source-circle",
        name="切线圆",
        definition=CircleCenterRadiusDefinition(Point(115.0, 60.0), 22.0),
    )
    document.add_construction_entity(point_entity, select=False, mark_dirty=False)
    document.add_construction_entity(circle_entity, select=False, mark_dirty=False)
    created: list[ConstructionEntity] = []
    states: list[dict[str, object]] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.constructionCommandChanged.connect(
        lambda document_id, payload: (
            document_id == document.id and states.append(dict(payload))
        )
    )
    canvas.set_tool_mode(
        "construction",
        construction_kind="tangent_point_circle",
    )

    _click_image(canvas, Point(25.0, 60.0))
    _click_image(canvas, Point(115.0, 38.0))

    assert created == []
    assert states and "多个解" in str(states[-1]["invalid_reason"])
    _click_image(canvas, Point(70.0, 43.0))

    assert len(created) == 1
    definition = created[0].definition
    assert isinstance(definition, PointCircleTangentDefinition)
    assert definition.point_source.object_id == point_entity.id
    assert definition.circle_source.object_id == circle_entity.id
    assert definition.branch in {0, 1}


def test_point_circle_tangent_accepts_measurement_endpoint_feature_directly(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    endpoint = Point(25.0, 60.0)
    measurement = Measurement(
        id="tangent-measurement-line",
        image_id=document.id,
        fiber_group_id=None,
        mode="manual",
        measurement_kind="line",
        line_px=Line(endpoint, Point(70.0, 60.0)),
    )
    circle = ConstructionEntity(
        id="tangent-endpoint-circle",
        name="切线圆",
        definition=CircleCenterRadiusDefinition(Point(130.0, 60.0), 20.0),
    )
    document.add_measurement(measurement)
    document.add_construction_entity(circle, select=False, mark_dirty=False)
    created: list[ConstructionEntity] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.set_tool_mode("construction", construction_kind="tangent_point_circle")

    _click_image(canvas, endpoint)
    _click_image(canvas, Point(130.0, 40.0))
    _click_image(canvas, Point(75.0, 43.0))

    assert len(created) == 1
    definition = created[0].definition
    assert isinstance(definition, PointCircleTangentDefinition)
    assert isinstance(definition.point_source, LiveFeatureRef)
    assert definition.point_source.object_id == measurement.id
    assert definition.point_source.object_kind is SourceObjectKind.MEASUREMENT
    assert definition.point_source.feature == "start"


def test_point_circle_tangent_accepts_arbitrary_frozen_point(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    circle = ConstructionEntity(
        id="tangent-free-point-circle",
        name="切线圆",
        definition=CircleCenterRadiusDefinition(Point(125.0, 65.0), 18.0),
    )
    document.add_construction_entity(circle, select=False, mark_dirty=False)
    created: list[ConstructionEntity] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.set_tool_mode("construction", construction_kind="tangent_point_circle")

    _click_image(canvas, Point(25.0, 25.0))
    _click_image(canvas, Point(125.0, 47.0))
    _click_image(canvas, Point(70.0, 40.0))

    assert len(created) == 1
    definition = created[0].definition
    assert isinstance(definition, PointCircleTangentDefinition)
    assert isinstance(definition.point_source, FrozenFeatureSnapshot)
    assert definition.point_source.geometry.point == Point(25.0, 25.0)


def test_perpendicular_command_temporarily_snaps_to_projected_foot(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    source = ConstructionEntity(
        id="perpendicular-snap-source",
        name="垂足来源",
        definition=LineDefinition(
            Point(20.0, 55.0),
            Point(180.0, 55.0),
            LineExtent.SEGMENT,
        ),
    )
    document.add_construction_entity(source, select=False, mark_dirty=False)
    settings = AppSettings()
    settings.object_snap_enabled = True
    settings.object_snap_kinds = []
    settings.object_snap_aperture_px = 10.0
    canvas.set_settings(settings)
    created: list[ConstructionEntity] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.set_tool_mode("construction", construction_kind="perpendicular")

    _click_image(canvas, Point(75.0, 55.0))
    candidate = canvas._query_object_snap(Point(82.0, 58.0))  # noqa: SLF001

    assert candidate is not None
    assert candidate.kind is SnapKind.PERPENDICULAR
    assert candidate.point_px == Point(82.0, 55.0)
    assert candidate.label == "垂足"

    _click_image(canvas, Point(82.0, 58.0))
    assert len(created) == 1
    definition = created[0].definition
    assert isinstance(definition, PerpendicularDefinition)
    assert definition.point == Point(82.0, 55.0)


def test_point_circle_tangent_command_exposes_temporary_tangent_snap(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    point = ConstructionEntity(
        id="tangent-snap-point",
        name="切线点",
        definition=FreePointDefinition(Point(25.0, 60.0)),
    )
    circle = ConstructionEntity(
        id="tangent-snap-circle",
        name="切线圆",
        definition=CircleCenterRadiusDefinition(Point(115.0, 60.0), 20.0),
    )
    document.add_construction_entity(point, select=False, mark_dirty=False)
    document.add_construction_entity(circle, select=False, mark_dirty=False)
    settings = AppSettings()
    settings.object_snap_enabled = True
    settings.object_snap_kinds = []
    settings.object_snap_aperture_px = 10.0
    canvas.set_settings(settings)
    canvas.set_tool_mode("construction", construction_kind="tangent_point_circle")

    _click_image(canvas, Point(25.0, 60.0))
    _click_image(canvas, Point(115.0, 40.0))
    session = canvas._construction_session  # noqa: SLF001
    assert session is not None and len(session.sources) == 2
    solutions = canvas._advanced_solution_candidates(session)  # noqa: SLF001
    assert len(solutions) == 2
    line = solutions[0][1]
    assert isinstance(line, ResolvedLine)
    tangent = line.point_at(line.project_parameter(Point(115.0, 60.0)))

    candidate = canvas._query_object_snap(tangent)  # noqa: SLF001

    assert candidate is not None
    assert candidate.kind is SnapKind.TANGENT
    assert candidate.point_px == tangent
    assert candidate.label == "切点"


def test_tangent_radius_circle_exposes_all_line_side_solution_families(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    horizontal = ConstructionEntity(
        id="ttr-horizontal",
        name="水平来源",
        definition=LineDefinition(
            Point(20.0, 60.0),
            Point(180.0, 60.0),
            LineExtent.INFINITE,
        ),
    )
    vertical = ConstructionEntity(
        id="ttr-vertical",
        name="垂直来源",
        definition=LineDefinition(
            Point(100.0, 10.0),
            Point(100.0, 110.0),
            LineExtent.INFINITE,
        ),
    )
    document.add_construction_entity(horizontal, select=False, mark_dirty=False)
    document.add_construction_entity(vertical, select=False, mark_dirty=False)
    created: list[ConstructionEntity] = []
    states: list[dict[str, object]] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.constructionCommandChanged.connect(
        lambda document_id, payload: (
            document_id == document.id and states.append(dict(payload))
        )
    )
    canvas.set_tool_mode("construction", construction_kind="tangent_circle_ttr")
    canvas.set_construction_parameter("distance", 15.0)

    _click_image(canvas, Point(40.0, 60.0))
    _click_image(canvas, Point(100.0, 25.0))

    assert created == []
    assert states and "多个解" in str(states[-1]["invalid_reason"])
    session = canvas._construction_session  # noqa: SLF001 - solution-family contract
    assert session is not None
    solutions = canvas._advanced_solution_candidates(session)  # noqa: SLF001
    assert len(solutions) == 4

    # Choose a point on the desired circle, not its centre (all equal-radius
    # solution circles are one radius away from their own centres).
    _click_image(canvas, Point(130.0, 75.0))
    assert len(created) == 1
    definition = created[0].definition
    assert isinstance(definition, TangentTangentRadiusCircleDefinition)
    all_entities = [horizontal, vertical, created[0]]
    resolved = ConstructionResolver(document.id, all_entities).resolve(created[0])
    assert resolved.valid and isinstance(resolved.geometry, ResolvedCircle)
    assert resolved.geometry.center.x == pytest.approx(115.0)
    assert resolved.geometry.center.y == pytest.approx(75.0)


def test_large_document_regular_preview_reuses_revision_scoped_resolver(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    source = ConstructionEntity(
        id="preview-cache-source",
        name="预览来源线",
        definition=LineDefinition(
            Point(20.0, 50.0),
            Point(180.0, 50.0),
            LineExtent.INFINITE,
        ),
    )
    unrelated = [
        ConstructionEntity(
            id=f"preview-cache-unrelated-{index}",
            name="无关辅助点",
            definition=FreePointDefinition(
                Point(2_000.0 + index, 2_000.0)
            ),
        )
        for index in range(10_000)
    ]
    document.construction_entities = [source, *unrelated]
    document.mark_construction_geometry_changed()
    canvas.notify_document_visual_changed()
    assert canvas._ensure_construction_spatial_index() is not None  # noqa: SLF001
    canvas.set_tool_mode("construction", construction_kind="parallel_through")
    session = canvas._ensure_construction_session()  # noqa: SLF001
    session.sources = [LiveFeatureRef(document.id, source.id)]
    surface = QImage(320, 240, QImage.Format.Format_ARGB32_Premultiplied)
    surface.fill(QColor(0, 0, 0, 0))
    painter = QPainter(surface)
    try:
        with patch(
            "fdm.ui.canvas.make_construction_resolver",
            wraps=make_construction_resolver,
        ) as resolver_factory:
            for point in (Point(45.0, 70.0), Point(65.0, 75.0), Point(85.0, 80.0)):
                canvas.mouseMoveEvent(_MouseEvent(canvas.image_to_widget(point)))
                assert session.invalid_reason == ""
                canvas._draw_construction_preview(painter)  # noqa: SLF001
            assert resolver_factory.call_count == 1

            document.mark_measurement_geometry_changed()
            canvas.mouseMoveEvent(
                _MouseEvent(canvas.image_to_widget(Point(105.0, 85.0)))
            )
            assert session.invalid_reason == ""
            canvas._draw_construction_preview(painter)  # noqa: SLF001
            assert resolver_factory.call_count == 2

            document.mark_construction_geometry_changed()
            canvas.mouseMoveEvent(
                _MouseEvent(canvas.image_to_widget(Point(125.0, 90.0)))
            )
            assert session.invalid_reason == ""
            canvas._draw_construction_preview(painter)  # noqa: SLF001
            assert resolver_factory.call_count == 3
    finally:
        painter.end()


def test_advanced_multi_solution_preview_draws_cached_geometry_directly(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    first = ConstructionEntity(
        id="preview-solution-first-circle",
        name="第一来源圆",
        definition=CircleCenterRadiusDefinition(Point(55.0, 60.0), 18.0),
    )
    second = ConstructionEntity(
        id="preview-solution-second-circle",
        name="第二来源圆",
        definition=CircleCenterRadiusDefinition(Point(145.0, 60.0), 18.0),
    )
    document.construction_entities = [first, second]
    document.mark_construction_geometry_changed()
    canvas.notify_document_visual_changed()
    canvas.set_tool_mode(
        "construction",
        construction_kind="common_tangent_external",
    )
    session = canvas._ensure_construction_session()  # noqa: SLF001
    session.sources = [
        LiveFeatureRef(document.id, first.id),
        LiveFeatureRef(document.id, second.id),
    ]
    session.hover_point = Point(100.0, 30.0)
    surface = QImage(320, 240, QImage.Format.Format_ARGB32_Premultiplied)
    surface.fill(QColor(0, 0, 0, 0))
    painter = QPainter(surface)
    try:
        with (
            patch(
                "fdm.ui.canvas.make_construction_resolver",
                wraps=make_construction_resolver,
            ) as resolver_factory,
            patch.object(
                canvas,
                "_draw_preview_definition",
                side_effect=AssertionError(
                    "resolved advanced solutions must not be resolved again"
                ),
            ) as definition_preview,
            patch("fdm.ui.canvas.draw_construction_entities") as draw_entities,
        ):
            canvas.mouseMoveEvent(
                _MouseEvent(canvas.image_to_widget(Point(100.0, 30.0)))
            )
            canvas._draw_construction_preview(painter)  # noqa: SLF001
            canvas.mouseMoveEvent(
                _MouseEvent(canvas.image_to_widget(Point(105.0, 32.0)))
            )
            canvas._draw_construction_preview(painter)  # noqa: SLF001

        definition_preview.assert_not_called()
        assert resolver_factory.call_count == 1
        assert len(session.advanced_solution_cache) == 2
        assert draw_entities.call_count == 2
        for call in draw_entities.call_args_list:
            entries = tuple(call.args[1])
            assert len(entries) == 2
            assert [resolved.geometry for _entity, resolved in entries] == [
                geometry for _definition, geometry in session.advanced_solution_cache
            ]
    finally:
        painter.end()


def test_coincident_derived_sources_show_identity_menu_and_use_selection(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    first = ConstructionEntity(
        id="coincident-first",
        name="重合线一",
        definition=LineDefinition(Point(20.0, 45.0), Point(150.0, 45.0)),
    )
    second = ConstructionEntity(
        id="coincident-second",
        name="重合线二",
        definition=LineDefinition(Point(20.0, 45.0), Point(150.0, 45.0)),
    )
    document.add_construction_entity(first, select=False, mark_dirty=False)
    document.add_construction_entity(second, select=False, mark_dirty=False)
    created: list[ConstructionEntity] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.set_tool_mode("construction", construction_kind="midpoint")

    with patch.object(
        canvas,
        "_choose_construction_source_candidate",
        side_effect=lambda candidates, **_kwargs: candidates[1],
    ) as choose_source:
        _click_image(canvas, Point(80.0, 45.0))

    choose_source.assert_called_once()
    assert len(choose_source.call_args.args[0]) == 2
    assert len(created) == 1
    definition = created[0].definition
    assert isinstance(definition, MidpointDefinition)
    # Candidate order follows top-most document order (second first); choosing
    # the second menu row must therefore preserve the first line's identity.
    assert definition.source.object_id == first.id


def test_parallel_through_command_preserves_snapped_point_feature_reference(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    source_line = ConstructionEntity(
        id="parallel-source-line",
        name="方向线",
        definition=LineDefinition(Point(20.0, 25.0), Point(150.0, 25.0)),
    )
    through_point = ConstructionEntity(
        id="parallel-through-point",
        name="通过点",
        definition=FreePointDefinition(Point(85.0, 75.0)),
    )
    document.add_construction_entity(source_line, select=False, mark_dirty=False)
    document.add_construction_entity(through_point, select=False, mark_dirty=False)
    created: list[ConstructionEntity] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.set_tool_mode("construction", construction_kind="parallel_through")

    _click_image(canvas, Point(60.0, 25.0))
    _click_image(canvas, Point(85.0, 75.0))

    assert len(created) == 1
    definition = created[0].definition
    assert isinstance(definition, ParallelThroughPointDefinition)
    assert isinstance(definition.point_source, LiveFeatureRef)
    assert definition.source.object_id == source_line.id
    assert definition.point_source.object_id == through_point.id
    assert definition.point_source.feature == "geometry"


@pytest.mark.parametrize(
    ("tool_kind", "definition_type"),
    [
        ("parallel_through", ParallelThroughPointDefinition),
        ("perpendicular", PerpendicularDefinition),
    ],
)
def test_through_point_commands_disambiguate_coincident_point_identity(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
    tool_kind: str,
    definition_type: type,
) -> None:
    document, _image, canvas = canvas_fixture
    source_line = ConstructionEntity(
        id=f"{tool_kind}-identity-line",
        name="方向线",
        definition=LineDefinition(Point(20.0, 25.0), Point(150.0, 25.0)),
    )
    first = ConstructionEntity(
        id=f"{tool_kind}-identity-first",
        name="重合点一",
        definition=FreePointDefinition(Point(85.0, 75.0)),
    )
    second = ConstructionEntity(
        id=f"{tool_kind}-identity-second",
        name="重合点二",
        definition=FreePointDefinition(Point(85.0, 75.0)),
    )
    for entity in (source_line, first, second):
        document.add_construction_entity(entity, select=False, mark_dirty=False)
    created: list[ConstructionEntity] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.set_tool_mode("construction", construction_kind=tool_kind)

    def choose(candidates, **_kwargs):
        if isinstance(candidates[0].geometry, ResolvedPoint):
            assert len(candidates) == 2
            return candidates[1]
        return candidates[0]

    with patch.object(
        canvas,
        "_choose_construction_source_candidate",
        side_effect=choose,
    ) as choose_source:
        _click_image(canvas, Point(60.0, 25.0))
        with patch(
            "fdm.ui.canvas.resolved_construction_entries",
            side_effect=AssertionError("identity menu must reuse the spatial index"),
        ) as resolve_entries:
            _click_image(canvas, Point(85.0, 75.0))

    resolve_entries.assert_not_called()

    assert choose_source.call_count == 2
    assert len(created) == 1
    definition = created[0].definition
    assert isinstance(definition, definition_type)
    assert isinstance(definition.point_source, LiveFeatureRef)
    # Candidate order is top-most first; choosing row two preserves the first
    # point's identity instead of silently accepting the snap priority winner.
    assert definition.point_source.object_id == first.id


def test_through_point_identity_menu_excludes_merely_nearby_points(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    source_line = ConstructionEntity(
        id="near-point-identity-line",
        name="方向线",
        definition=LineDefinition(Point(20.0, 25.0), Point(150.0, 25.0)),
    )
    exact = ConstructionEntity(
        id="near-point-exact",
        name="准确点",
        definition=FreePointDefinition(Point(85.0, 75.0)),
    )
    nearby = ConstructionEntity(
        id="near-point-other",
        name="亚像素邻近点",
        definition=FreePointDefinition(Point(85.5, 75.0)),
    )
    for entity in (source_line, exact, nearby):
        document.add_construction_entity(entity, select=False, mark_dirty=False)
    created: list[ConstructionEntity] = []
    canvas.constructionCreateRequested.connect(
        lambda document_id, entity: (
            document_id == document.id and created.append(entity)
        )
    )
    canvas.set_tool_mode("construction", construction_kind="parallel_through")

    point_candidate_counts: list[int] = []

    def choose(candidates, **_kwargs):
        if isinstance(candidates[0].geometry, ResolvedPoint):
            point_candidate_counts.append(len(candidates))
        return candidates[0]

    with patch.object(
        canvas,
        "_choose_construction_source_candidate",
        side_effect=choose,
    ):
        _click_image(canvas, Point(60.0, 25.0))
        _click_image(canvas, Point(85.0, 75.0))

    assert point_candidate_counts == [1]
    assert len(created) == 1
    definition = created[0].definition
    assert isinstance(definition, ParallelThroughPointDefinition)
    assert isinstance(definition.point_source, LiveFeatureRef)
    assert definition.point_source.object_id == exact.id


def test_persistent_construction_geometry_is_rendered_by_document_canvas(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
    app: QApplication,
) -> None:
    document, _image, canvas = canvas_fixture
    baseline = _render_canvas(canvas, app)
    document.add_construction_entity(
        ConstructionEntity(
            id="persistent-line",
            name="持久辅助线",
            definition=LineDefinition(
                Point(15.0, 50.0),
                Point(170.0, 50.0),
                LineExtent.SEGMENT,
            ),
        ),
        select=False,
        mark_dirty=False,
    )

    rendered = _render_canvas(canvas, app)

    assert _different_pixel_count(baseline, rendered) > 30


def test_manual_measurement_anchor_uses_construction_object_snap(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    snap_point = Point(50.0, 40.0)
    document.add_construction_entity(
        ConstructionEntity(
            id="snap-point",
            name="捕捉点",
            definition=FreePointDefinition(snap_point),
        ),
        select=False,
        mark_dirty=False,
    )
    commits: list[tuple[str, str, object]] = []
    canvas.lineCommitted.connect(
        lambda document_id, mode, line: commits.append((document_id, mode, line))
    )
    canvas.set_tool_mode("manual")
    start = canvas.image_to_widget(Point(56.0, 40.0))
    end = canvas.image_to_widget(Point(125.0, 75.0))

    canvas.mousePressEvent(_MouseEvent(start))
    canvas.mouseMoveEvent(_MouseEvent(end))
    canvas.mouseReleaseEvent(_MouseEvent(end))

    assert len(commits) == 1
    assert commits[0][0:2] == (document.id, "manual")
    committed = commits[0][2]
    assert committed.start == snap_point
    assert committed.end == Point(125.0, 75.0)


def test_measurement_source_revision_rebuilds_derived_construction_snap_index(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    measurement = Measurement(
        id="source-measurement",
        image_id=document.id,
        fiber_group_id=None,
        mode="manual",
        measurement_kind="line",
        line_px=Line(Point(20.0, 30.0), Point(60.0, 30.0)),
    )
    document.add_measurement(measurement)
    document.add_construction_entity(
        ConstructionEntity(
            id="measurement-midpoint",
            name="测量来源中点",
            definition=MidpointDefinition(
                LiveFeatureRef(
                    document.id,
                    measurement.id,
                    object_kind="measurement",
                )
            ),
        ),
        select=False,
        mark_dirty=False,
    )
    settings = AppSettings()
    settings.object_snap_kinds = ["midpoint"]
    canvas.set_settings(settings)
    canvas.set_tool_mode("manual")

    original = canvas._query_object_snap(Point(40.0, 30.0))  # noqa: SLF001
    assert original is not None
    assert original.source_id == "measurement-midpoint"

    measurement.replace_line_geometry(
        line_px=Line(Point(100.0, 70.0), Point(140.0, 70.0)),
    )
    document.mark_measurement_geometry_changed()
    canvas.notify_document_visual_changed()
    canvas._object_snap_engine.clear_hysteresis()  # noqa: SLF001

    assert canvas._query_object_snap(Point(40.0, 30.0)) is None  # noqa: SLF001
    moved = canvas._query_object_snap(Point(120.0, 70.0))  # noqa: SLF001
    assert moved is not None
    assert moved.source_id == "measurement-midpoint"


def test_current_construction_snap_index_does_not_reresolve_graph_per_move(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    point = Point(65.0, 45.0)
    document.add_construction_entity(
        ConstructionEntity(
            id="cached-snap-point",
            name="缓存捕捉点",
            definition=FreePointDefinition(point),
        ),
        select=False,
        mark_dirty=False,
    )
    canvas.set_tool_mode("manual")
    assert canvas._query_object_snap(point) is not None  # noqa: SLF001

    with patch(
        "fdm.ui.canvas.resolved_construction_entries",
        side_effect=AssertionError("current index must be reused"),
    ) as resolve_entries:
        assert canvas._query_object_snap(Point(66.0, 45.0)) is not None  # noqa: SLF001

    resolve_entries.assert_not_called()


def test_construction_metadata_change_reuses_geometry_index_and_snap_flags(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    point = Point(65.0, 45.0)
    entity = ConstructionEntity(
        id="metadata-index-point",
        name="元数据索引点",
        definition=FreePointDefinition(point),
    )
    document.add_construction_entity(entity, select=False, mark_dirty=False)
    settings = AppSettings()
    settings.object_snap_enabled = True
    settings.object_snap_kinds = ["point"]
    canvas.set_settings(settings)
    canvas.set_tool_mode("manual")
    assert canvas._query_object_snap(point) is not None  # noqa: SLF001
    original_index = canvas._construction_spatial_index  # noqa: SLF001
    geometry_revision = document.construction_geometry_revision

    assert document.replace_construction_entity(
        entity.id,
        replace(entity, snap_enabled=False),
        mark_dirty=False,
    )
    with patch(
        "fdm.ui.canvas.resolved_construction_entries",
        side_effect=AssertionError("metadata edit must not rebuild geometry index"),
    ) as resolve_entries:
        canvas.notify_document_visual_changed()
        candidate = canvas._query_object_snap(point)  # noqa: SLF001

    resolve_entries.assert_not_called()
    assert candidate is None
    assert canvas._construction_spatial_index is original_index  # noqa: SLF001
    assert document.construction_geometry_revision == geometry_revision


def test_construction_paint_reuses_index_and_queries_only_visible_objects(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    visible = ConstructionEntity(
        id="visible-paint-point",
        name="视口内点",
        definition=FreePointDefinition(Point(65.0, 45.0)),
    )
    offscreen = [
        ConstructionEntity(
            id=f"offscreen-paint-point-{index}",
            name="视口外点",
            definition=FreePointDefinition(Point(10_000.0 + index, 10_000.0)),
        )
        for index in range(500)
    ]
    document.construction_entities = [visible, *offscreen]
    document.mark_construction_geometry_changed()
    canvas.notify_document_visual_changed()
    context = canvas._paint_context()  # noqa: SLF001

    with patch("fdm.ui.canvas.draw_construction_entities") as draw_entities:
        canvas._draw_constructions(None, context)  # type: ignore[arg-type]  # noqa: SLF001
    entries = tuple(draw_entities.call_args.args[1])
    assert [entity.id for entity, _resolved in entries] == [visible.id]

    with (
        patch(
            "fdm.ui.canvas.resolved_construction_entries",
            side_effect=AssertionError("current paint index must be reused"),
        ) as resolve_entries,
        patch("fdm.ui.canvas.draw_construction_entities"),
    ):
        canvas._draw_constructions(None, context)  # type: ignore[arg-type]  # noqa: SLF001
    resolve_entries.assert_not_called()


def test_construction_hit_test_queries_only_local_candidates_at_ten_thousand_scale(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    far_points = [
        ConstructionEntity(
            id=f"far-hit-point-{index}",
            name="远处辅助点",
            definition=FreePointDefinition(
                Point(
                    1_000.0 + (index % 100) * 300.0,
                    1_000.0 + (index // 100) * 300.0,
                )
            ),
        )
        for index in range(10_000)
    ]
    earlier = ConstructionEntity(
        id="local-hit-earlier",
        name="较早同位点",
        definition=FreePointDefinition(Point(65.0, 45.0)),
    )
    latest = ConstructionEntity(
        id="local-hit-latest",
        name="较晚同位点",
        definition=FreePointDefinition(Point(65.0, 45.0)),
    )
    document.construction_entities = [*far_points, earlier, latest]
    document.mark_construction_geometry_changed()
    canvas.notify_document_visual_changed()
    assert canvas._ensure_construction_spatial_index() is not None  # noqa: SLF001

    exact_distance = canvas._distance_to_resolved_geometry  # noqa: SLF001
    with (
        patch(
            "fdm.ui.canvas.resolved_construction_entries",
            side_effect=AssertionError("current spatial index must be reused"),
        ),
        patch.object(
            canvas,
            "_distance_to_resolved_geometry",
            wraps=exact_distance,
        ) as exact_checks,
    ):
        hit = canvas._hit_test_construction(Point(65.0, 45.0))  # noqa: SLF001

    assert hit == latest.id
    assert exact_checks.call_count == 2


def test_construction_hit_test_keeps_unbounded_line_and_array_semantics(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    unbounded = ConstructionEntity(
        id="hit-infinite-line",
        name="无限辅助线",
        definition=LineDefinition(
            Point(20.0, 20.0),
            Point(40.0, 20.0),
            LineExtent.INFINITE,
        ),
    )
    array_source = ConstructionEntity(
        id="hit-array-source",
        name="阵列来源",
        definition=LineDefinition(
            Point(20.0, 60.0),
            Point(40.0, 60.0),
            LineExtent.INFINITE,
        ),
        visible=False,
    )
    array = ConstructionEntity(
        id="hit-parametric-array",
        name="参数化阵列",
        definition=ParallelArrayDefinition(
            LiveFeatureRef(document.id, array_source.id),
            spacing=10.0,
            count=3,
            side=ArraySide.POSITIVE,
            extent=LineExtent.INFINITE,
        ),
    )
    document.construction_entities = [unbounded, array_source, array]
    document.mark_construction_geometry_changed()
    canvas.notify_document_visual_changed()

    assert canvas._hit_test_construction(Point(180.0, 20.0)) == unbounded.id  # noqa: SLF001
    assert canvas._hit_test_construction(Point(180.0, 70.0)) == array.id  # noqa: SLF001


def test_construction_drag_resolves_affected_closure_and_repaints_locally(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    source = ConstructionEntity(
        id="drag-cached-source",
        name="拖动来源线",
        definition=LineDefinition(Point(40.0, 40.0), Point(60.0, 40.0)),
    )
    dependent = ConstructionEntity(
        id="drag-cached-midpoint",
        name="拖动关联中点",
        definition=MidpointDefinition(LiveFeatureRef(document.id, source.id)),
    )
    unrelated = [
        ConstructionEntity(
            id=f"drag-unrelated-{index}",
            name="无关辅助点",
            definition=FreePointDefinition(Point(5_000.0 + index, 5_000.0)),
        )
        for index in range(2_000)
    ]
    document.construction_entities = [source, dependent, *unrelated]
    document.mark_construction_geometry_changed()
    canvas.notify_document_visual_changed()
    canvas.set_tool_mode("select")
    canvas._begin_construction_drag((source.id, 0))  # noqa: SLF001
    drag_resolver = canvas._drag_construction_resolver  # noqa: SLF001
    assert drag_resolver is not None
    assert canvas._drag_construction_affected_ids == {  # noqa: SLF001
        source.id,
        dependent.id,
    }

    original_resolve = drag_resolver.resolve
    with (
        patch.object(drag_resolver, "resolve", wraps=original_resolve) as resolve,
        patch.object(canvas, "update") as update,
        patch(
            "fdm.ui.canvas.resolved_construction_entries",
            side_effect=AssertionError("drag frame must reuse the base index"),
        ),
    ):
        canvas.mouseMoveEvent(
            _MouseEvent(canvas.image_to_widget(Point(50.0, 40.0)))
        )

    assert resolve.call_count <= 4
    assert update.call_args_list
    assert all(call.args for call in update.call_args_list)
    assert all(call.args[0] != canvas.rect() for call in update.call_args_list)

    context = canvas._paint_context()  # noqa: SLF001
    with (
        patch(
            "fdm.ui.canvas.make_construction_resolver",
            side_effect=AssertionError("paint must reuse drag caches"),
        ),
        patch("fdm.ui.canvas.draw_construction_entities") as draw_entities,
    ):
        canvas._draw_constructions(None, context)  # type: ignore[arg-type]  # noqa: SLF001
    entries = tuple(draw_entities.call_args.args[1])
    assert [entity.id for entity, _resolved in entries] == [source.id, dependent.id]
    resolved_by_id = {entity.id: resolved for entity, resolved in entries}
    assert resolved_by_id[source.id].geometry == ResolvedLine(
        Point(50.0, 40.0),
        Point(60.0, 40.0),
    )
    assert resolved_by_id[dependent.id].geometry == ResolvedPoint(Point(55.0, 40.0))


def test_select_then_drag_construction_handle_emits_edited_entity(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    original = ConstructionEntity(
        id="drag-point",
        name="可拖动点",
        definition=FreePointDefinition(Point(45.0, 35.0)),
    )
    document.add_construction_entity(original, select=False, mark_dirty=False)
    selections: list[object] = []
    edits: list[tuple[str, str, ConstructionEntity]] = []
    canvas.objectSelectionChanged.connect(
        lambda document_id, selection: selections.append(selection)
    )
    canvas.constructionEdited.connect(
        lambda document_id, construction_id, entity: edits.append(
            (document_id, construction_id, entity)
        )
    )
    canvas.set_tool_mode("select")

    _click_image(canvas, Point(45.0, 35.0))

    assert document.selected_construction_id == original.id
    assert selections
    start = canvas.image_to_widget(Point(45.0, 35.0))
    # A short move remains inside the object-snap aperture; the edited object
    # itself must be excluded or this handle would stick to its old point.
    destination = canvas.image_to_widget(Point(50.0, 35.0))
    canvas.mousePressEvent(_MouseEvent(start))
    canvas.mouseMoveEvent(_MouseEvent(destination))
    canvas.mouseReleaseEvent(_MouseEvent(destination))

    assert len(edits) == 1
    assert edits[0][0:2] == (document.id, original.id)
    edited = edits[0][2]
    assert isinstance(edited.definition, FreePointDefinition)
    assert edited.definition.point == Point(50.0, 35.0)
    assert document.get_construction_entity(original.id) is original


def test_measurement_endpoint_drag_does_not_snap_back_to_its_own_geometry(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    measurement = Measurement(
        id="self-snap-line",
        image_id=document.id,
        fiber_group_id=None,
        mode="manual",
        measurement_kind="line",
        line_px=Line(Point(30.0, 55.0), Point(130.0, 55.0)),
    )
    document.add_measurement(measurement)
    edits: list[Line] = []
    canvas.measurementEdited.connect(
        lambda document_id, measurement_id, line: (
            document_id == document.id
            and measurement_id == measurement.id
            and isinstance(line, Line)
            and edits.append(line)
        )
    )
    canvas.set_tool_mode("select")
    start = canvas.image_to_widget(Point(30.0, 55.0))
    destination = canvas.image_to_widget(Point(35.0, 55.0))

    canvas.mousePressEvent(_MouseEvent(start))
    canvas.mouseMoveEvent(_MouseEvent(destination))
    canvas.mouseReleaseEvent(_MouseEvent(destination))

    assert len(edits) == 1
    assert edits[0].start == Point(35.0, 55.0)
    assert edits[0].end == Point(130.0, 55.0)


def test_measurement_drag_reuses_object_snap_dependency_exclusion(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
) -> None:
    document, _image, canvas = canvas_fixture
    measurement = Measurement(
        id="exclusion-cache-measurement",
        image_id=document.id,
        fiber_group_id=None,
        mode="manual",
        measurement_kind="line",
        line_px=Line(Point(20.0, 40.0), Point(80.0, 40.0)),
    )
    document.add_measurement(measurement)
    canvas._dragging_handle = (measurement.id, "start")  # noqa: SLF001

    with patch(
        "fdm.ui.canvas.construction_transitive_dependents",
        return_value=("dependent-construction",),
    ) as closure:
        first = canvas._object_snap_excluded_sources()  # noqa: SLF001
        second = canvas._object_snap_excluded_sources()  # noqa: SLF001

    closure.assert_called_once()
    assert first == second
    assert first == ({"dependent-construction"}, {measurement.id})


@pytest.mark.parametrize("zoom", [0.5, 1.0, 4.0])
def test_object_snap_aperture_remains_screen_stable_across_zoom(
    canvas_fixture: tuple[ImageDocument, QImage, DocumentCanvas],
    zoom: float,
) -> None:
    document, _image, canvas = canvas_fixture
    snap_point = Point(50.0, 40.0)
    document.add_construction_entity(
        ConstructionEntity(
            id=f"snap-point-{zoom}",
            name="缩放捕捉点",
            definition=FreePointDefinition(snap_point),
        ),
        select=False,
        mark_dirty=False,
    )
    settings = AppSettings()
    settings.object_snap_enabled = True
    settings.object_snap_kinds = ["point"]
    settings.object_snap_aperture_px = 10.0
    canvas.set_settings(settings)
    canvas.set_tool_mode("manual")
    canvas._zoom = zoom  # noqa: SLF001 - verifies logical-screen aperture
    target_widget = canvas.image_to_widget(snap_point)

    inside_image = canvas.widget_to_image(target_widget + QPointF(8.0, 0.0))
    inside = canvas._query_object_snap(inside_image)  # noqa: SLF001
    assert inside is not None
    assert inside.point_px == snap_point

    canvas._object_snap_engine.clear_hysteresis()  # noqa: SLF001
    canvas._set_active_snap_candidate(None, repaint=False)  # noqa: SLF001
    outside_image = canvas.widget_to_image(target_widget + QPointF(12.0, 0.0))
    outside = canvas._query_object_snap(outside_image)  # noqa: SLF001
    assert outside is None
