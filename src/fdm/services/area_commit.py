"""Geometry-only finalization of an accepted magic-wand measurement."""

from fdm.geometry import Point
from fdm.services.prompt_segmentation import (
    finalize_magic_subtraction_mask,
    magic_mask_to_geometry,
    magic_mask_area_px,
)


def finalize_area_commit(snapshot):
    mask = snapshot["mask"]
    stats = {}
    if snapshot["subtract_masks"]:
        mask, stats = finalize_magic_subtraction_mask(mask, snapshot["subtract_masks"])
    if mask is None:
        raise ValueError("剔除后无剩余区域，未新增测量")
    mask, rings, polygon, geometry_stats = magic_mask_to_geometry(
        mask, select_prompt_component=False
    )
    if mask is None or len(polygon) < 3:
        raise ValueError("剔除结果无有效面积，未新增测量")
    ox, oy = snapshot["origin"]
    return dict(
        measurement_kind="area",
        polygon_px=[Point(p.x + ox, p.y + oy) for p in polygon],
        area_rings_px=[[Point(p.x + ox, p.y + oy) for p in ring] for ring in rings],
        exact_area_px=magic_mask_area_px(mask),
        debug_payload=snapshot["debug_payload"],
        display_preview=snapshot.get("display_preview", ()),
        commit_stats={**stats, **geometry_stats},
    )
