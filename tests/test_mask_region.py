import numpy as np
import pytest
from fdm.geometry import Point
from fdm.services.mask_region import mask_region, rasterize_rings_region
from fdm.services.prompt_segmentation import (
    magic_mask_to_geometry,
    finalize_magic_subtraction_mask,
    fill_magic_draft_internal_holes,
)


@pytest.mark.parametrize("offset", [(0, 0), (97, 83), (102, 100)])
def test_local_geometry_matches_full_mask_including_holes_and_prompt_clamping(offset):
    x, y = offset
    mask = np.zeros((256, 256), bool)
    mask[y : y + 90, x : x + 110] = True
    mask[y + 20 : y + 35, x + 20 : x + 35] = False
    mask[200:211, 211:221] = True
    region = mask_region(mask)
    for points in ([Point(x + 3, y + 3)], [Point(-10, -10)], [Point(210, 200)], [Point(5.5, 6.5)]):
        args = dict(positive_points=points, negative_points=[Point(220, 205)])
        full, rings, polygon, stats = magic_mask_to_geometry(mask, **args)
        local, lr, lp, ls = magic_mask_to_geometry(region, **args)
        assert np.array_equal(full, local.to_full_mask())
        assert (rings, polygon, stats) == (lr, lp, ls)
    assert not region.data.flags.writeable


@pytest.mark.parametrize("split", [False, True])
def test_subtraction_and_fill_are_pixel_exact(split):
    primary = np.zeros((1024, 1536), bool)
    primary[311:811, 433:933] = True
    primary[420:470, 550:600] = False
    subtract = np.zeros_like(primary)
    subtract[560:590, 400 : 1000 if split else 600] = True
    full, stats = finalize_magic_subtraction_mask(primary, [subtract])
    local, ls = finalize_magic_subtraction_mask(mask_region(primary), [mask_region(subtract)])
    assert stats == ls
    assert np.array_equal(full, local.to_full_mask())
    assert np.array_equal(
        fill_magic_draft_internal_holes(primary),
        fill_magic_draft_internal_holes(mask_region(primary)).to_full_mask(),
    )


def test_regional_rasterization_retains_round_then_clamp():
    import cv2

    points = [Point(-5.5, 20.5), Point(90.5, -2), Point(110, 80), Point(2, 100)]
    full = np.zeros((96, 96), np.uint8)
    contour = np.array(
        [[min(max(round(p.x), 0), 95), min(max(round(p.y), 0), 95)] for p in points], np.int32
    )
    cv2.fillPoly(full, [contour], 1)
    assert np.array_equal(
        full.astype(bool), rasterize_rings_region([points], extent=(96, 96)).to_full_mask()
    )


@pytest.mark.parametrize("size", [2048, 4096, 8192])
def test_large_source_local_result_matches_full_pixels_at_edges_and_with_fragments(size):
    primary = np.zeros((size, size), bool)
    primary[-512:, -512:] = True
    primary[-300:-250, -300:-250] = False
    subtract = np.zeros_like(primary)
    subtract[-250:-245, -512:] = True
    full, expected_stats = finalize_magic_subtraction_mask(primary, [subtract])
    region, stats = finalize_magic_subtraction_mask(mask_region(primary), [mask_region(subtract)])
    assert stats == expected_stats
    assert np.array_equal(full, region.to_full_mask())
    assert region.data.size <= 512 * 512
    empty, stats = finalize_magic_subtraction_mask(mask_region(primary), [mask_region(primary)])
    assert empty is None and stats["result_empty"]


def test_slide_origin_and_accepted_snapshot_own_their_pixels():
    from fdm.services.area_commit import finalize_area_commit

    pixels = np.ones((50, 60), bool)
    region = mask_region(pixels, origin=(100, 200), extent=(1024, 1024))
    pixels[:] = False
    snapshot = dict(mask=region, subtract_masks=(), origin=(4000, 7000), debug_payload={})
    payload = finalize_area_commit(snapshot)
    assert payload["exact_area_px"] == 3000
    assert min(p.x for p in payload["polygon_px"]) == 4100
    assert min(p.y for p in payload["polygon_px"]) == 7200
