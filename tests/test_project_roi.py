from __future__ import annotations

import json
import math

import numpy as np
import pytest

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
    rasterize_roi_mask,
    roi_bounds,
)


def _roi(
    roi_id: str,
    geometry,
    *,
    document_id: str = "image_1",
    revision: int = 0,
) -> ProjectRoi:
    return ProjectRoi(
        id=roi_id,
        document_id=document_id,
        name=roi_id,
        geometry=geometry,
        revision=revision,
    )


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


def test_rectangle_uses_pixel_centres_and_returns_read_only_mask() -> None:
    roi = _roi("rect", RectangleRoiGeometry(1.0, 1.0, 2.0, 2.0))

    mask = rasterize_roi_mask(roi, 5, 5)

    expected = np.zeros((5, 5), dtype=np.bool_)
    expected[1:3, 1:3] = True
    np.testing.assert_array_equal(mask, expected)
    assert not mask.flags.writeable
    assert roi.bounds() == (1.0, 1.0, 3.0, 3.0)


def test_ellipse_is_sampled_at_pixel_centres() -> None:
    roi = _roi("ellipse", EllipseRoiGeometry(0.0, 0.0, 4.0, 4.0))

    mask = roi.rasterize_mask(4, 4)

    expected = np.array(
        [
            [False, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
            [False, True, True, False],
        ],
        dtype=np.bool_,
    )
    np.testing.assert_array_equal(mask, expected)


def test_polygon_rings_use_odd_even_independent_of_winding() -> None:
    outer = _square(0.0, 0.0, 6.0, 6.0)
    hole = _square(2.0, 2.0, 4.0, 4.0)
    forward = _roi("donut", PolygonRoiGeometry((outer, hole)))
    reversed_hole = _roi(
        "donut_reversed",
        PolygonRoiGeometry((outer, tuple(reversed(hole)))),
    )

    first = forward.rasterize_mask(6, 6)
    second = reversed_hole.rasterize_mask(6, 6)

    np.testing.assert_array_equal(first, second)
    assert int(first.sum()) == 32
    assert not first[2:4, 2:4].any()


def test_self_intersecting_polygon_uses_odd_even_fill() -> None:
    bow_tie = (
        RoiPoint(0.0, 0.0),
        RoiPoint(4.0, 4.0),
        RoiPoint(0.0, 4.0),
        RoiPoint(4.0, 0.0),
    )
    roi = _roi("bow_tie", PolygonRoiGeometry((bow_tie,)))

    mask = roi.rasterize_mask(4, 4)

    assert int(mask.sum()) == 12
    assert mask[0, 0]
    assert mask[0, 3]
    assert mask[3, 0]
    assert mask[3, 3]


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        (RoiBooleanOperator.UNION, 14),
        (RoiBooleanOperator.INTERSECTION, 4),
        (RoiBooleanOperator.DIFFERENCE, 5),
        (RoiBooleanOperator.XOR, 10),
    ],
)
def test_boolean_expression_rasterization(
    operator: RoiBooleanOperator,
    expected: int,
) -> None:
    left = _roi("left", RectangleRoiGeometry(0.0, 0.0, 3.0, 3.0))
    right = _roi("right", RectangleRoiGeometry(1.0, 1.0, 3.0, 3.0))
    composite = _roi(
        "combined",
        RoiBooleanExpression(operator, ("left", "right")),
    )
    lookup = {item.id: item for item in (left, right, composite)}

    mask = composite.rasterize_mask(5, 5, roi_lookup=lookup)

    assert int(mask.sum()) == expected


def test_composite_bounds_follow_boolean_semantics() -> None:
    left = _roi("left", RectangleRoiGeometry(0.0, 0.0, 3.0, 4.0))
    right = _roi("right", RectangleRoiGeometry(2.0, 1.0, 4.0, 5.0))
    lookup = {"left": left, "right": right}

    union = _roi(
        "union",
        RoiBooleanExpression(RoiBooleanOperator.UNION, ("left", "right")),
    )
    intersection = _roi(
        "intersection",
        RoiBooleanExpression(
            RoiBooleanOperator.INTERSECTION,
            ("left", "right"),
        ),
    )
    difference = _roi(
        "difference",
        RoiBooleanExpression(
            RoiBooleanOperator.DIFFERENCE,
            ("left", "right"),
        ),
    )

    assert roi_bounds(union, roi_lookup=lookup) == (0.0, 0.0, 6.0, 6.0)
    assert roi_bounds(intersection, roi_lookup=lookup) == (2.0, 1.0, 3.0, 4.0)
    assert roi_bounds(difference, roi_lookup=lookup) == (0.0, 0.0, 3.0, 4.0)


def test_composite_rejects_missing_cross_document_and_cycle_references() -> None:
    member = _roi("member", RectangleRoiGeometry(0, 0, 1, 1))
    missing = _roi(
        "missing",
        RoiBooleanExpression(RoiBooleanOperator.UNION, ("member", "absent")),
    )
    with pytest.raises(KeyError, match="不存在"):
        missing.rasterize_mask(2, 2, roi_lookup={"member": member})

    foreign = _roi(
        "foreign",
        RectangleRoiGeometry(0, 0, 1, 1),
        document_id="image_2",
    )
    cross_document = _roi(
        "cross",
        RoiBooleanExpression(RoiBooleanOperator.UNION, ("member", "foreign")),
    )
    with pytest.raises(ValueError, match="其他文档"):
        cross_document.rasterize_mask(
            2,
            2,
            roi_lookup={"member": member, "foreign": foreign},
        )

    a = _roi(
        "a",
        RoiBooleanExpression(RoiBooleanOperator.UNION, ("member", "b")),
    )
    b = _roi(
        "b",
        RoiBooleanExpression(RoiBooleanOperator.UNION, ("member", "a")),
    )
    with pytest.raises(ValueError, match="循环"):
        a.rasterize_mask(
            2,
            2,
            roi_lookup={"member": member, "a": a, "b": b},
        )


def test_roundtrip_preserves_all_roi_kinds_and_strict_json() -> None:
    geometries = (
        RectangleRoiGeometry(1, 2, 3, 4),
        EllipseRoiGeometry(1, 2, 3, 4),
        PolygonRoiGeometry((_square(0, 0, 2, 2),)),
        FreehandRoiGeometry((_square(0, 0, 2, 2),)),
        RoiBooleanExpression(RoiBooleanOperator.XOR, ("left", "right")),
    )

    for index, geometry in enumerate(geometries):
        original = ProjectRoi(
            id=f"roi_{index}",
            document_id="image_1",
            name=f"区域 {index}",
            group="纤维",
            geometry=geometry,
            visible=index % 2 == 0,
            locked=index % 2 == 1,
            color="#aabbcc",
            revision=index,
        )

        payload = original.to_dict()
        restored = ProjectRoi.from_dict(payload)

        assert restored == original
        assert restored.kind is ProjectRoiKind(geometry.kind)
        assert restored.color == "#AABBCC"
        json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_geometry_revision_only_advances_when_geometry_changes() -> None:
    roi = _roi("rect", RectangleRoiGeometry(0, 0, 2, 2), revision=7)

    renamed = roi.with_metadata(name="新名称", visible=False)
    unchanged = roi.replace_geometry(roi.geometry)
    changed = roi.replace_geometry(RectangleRoiGeometry(1, 0, 2, 2))

    assert renamed.revision == 7
    assert unchanged is roi
    assert changed.revision == 8
    assert roi.geometry == RectangleRoiGeometry(0, 0, 2, 2)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.__setitem__("unknown", True),
        lambda payload: payload.__setitem__("revision", True),
        lambda payload: payload.__setitem__("color", "red"),
        lambda payload: payload.__setitem__("kind", "triangle"),
        lambda payload: payload["geometry"].__setitem__("x", math.nan),
    ],
)
def test_from_dict_rejects_malformed_or_unknown_fields(mutator) -> None:
    payload = _roi("rect", RectangleRoiGeometry(0, 0, 2, 2)).to_dict()
    mutator(payload)

    with pytest.raises((TypeError, ValueError)):
        ProjectRoi.from_dict(payload)


def test_boolean_schema_version_is_not_accepted_as_integer_one() -> None:
    payload = _roi("rect", RectangleRoiGeometry(0, 0, 2, 2)).to_dict()
    payload["schema_version"] = True

    with pytest.raises(ValueError, match="schema_version"):
        ProjectRoi.from_dict(payload)


def test_geometry_rejects_degenerate_and_non_finite_coordinates() -> None:
    with pytest.raises(ValueError):
        RectangleRoiGeometry(0, 0, 0, 2)
    with pytest.raises(ValueError):
        EllipseRoiGeometry(0, 0, 2, math.inf)
    with pytest.raises(ValueError):
        PolygonRoiGeometry(
            ((RoiPoint(0, 0), RoiPoint(1, 1), RoiPoint(0, 0)),)
        )
    with pytest.raises(ValueError):
        RoiBooleanExpression(RoiBooleanOperator.UNION, ("same", "same"))
