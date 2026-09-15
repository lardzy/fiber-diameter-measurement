from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fdm.cancellation import CancellationError, CancellationTokenSource
from fdm.services.contour_comparison import ContourAxis, ContourFrame
from fdm.services.contour_comparison_wand import ContourWandService, apply_wand_mask
from fdm.services.mask_region import mask_region


def sample():
    image = np.full((80, 100, 4), 255, np.uint8)
    mask = np.zeros((80, 100), bool)
    mask[10:60, 20:80] = True
    return ContourFrame('cloth', image, mask, ContourAxis((50, 5), (48, 60)), .2, axis_confirmed=True)


def test_native_wand_algebra_retains_holes_disjoint_regions_and_source():
    frame = sample()
    mask = np.zeros_like(frame.mask)
    mask[20:40, 30:60] = True
    mask[25:30, 40:45] = False
    for operation, expected in (('replace', mask), ('add', mask | frame.mask), ('remove', frame.mask & ~mask)):
        result = apply_wand_mask(frame, mask_region(mask), operation=operation)
        np.testing.assert_array_equal(result.mask, expected)
        assert result.rgba is frame.rgba and result.axis == frame.axis
        assert result.mm_per_pixel == .2 and result.axis_confirmed and result.edited
        assert not result.mask.flags.writeable
    assert frame.mask[20:40, 30:60].all()
    cut = np.zeros_like(mask)
    cut[:, 45:55] = True
    result = apply_wand_mask(frame, cut, operation='remove')
    assert result.mask[30, 30] and result.mask[30, 70] and not result.mask[30, 50]


def test_refining_add_or_remove_uses_original_base_not_last_prediction():
    frame = sample()
    base = frame.mask
    whole = np.zeros_like(base); whole[15:55, 5:45] = True
    refinement = np.zeros_like(base); refinement[25:45, 5:35] = True
    for operation in ('add', 'remove'):
        first = apply_wand_mask(frame, whole, operation=operation)
        second = apply_wand_mask(first, refinement, operation=operation, base_mask=base)
        expected = base | refinement if operation == 'add' else base & ~refinement
        np.testing.assert_array_equal(second.mask, expected)


def test_wrong_size_empty_result_and_transparent_pixels_cannot_become_contour():
    frame = sample()
    for invalid in (None, np.zeros_like(frame.mask), np.ones((20, 30), bool)):
        with pytest.raises(ValueError):
            apply_wand_mask(frame, invalid)
    rgba = frame.rgba.copy(); rgba[:, :10, 3] = 0
    result = apply_wand_mask(replace(frame, rgba=rgba), np.ones_like(frame.mask))
    assert not result.mask[:, :10].any()
    assert any('截断' in warning for warning in result.warnings)


def test_backend_uses_native_image_coordinates_and_source_identity_not_mask():
    service = ContourWandService()
    frame = sample()
    calls = []
    def predict(**kwargs):
        assert kwargs['image'].size().width() == 100
        assert kwargs['image'].size().height() == 80
        assert not kwargs['roi_enabled']
        calls.append(kwargs)
        return SimpleNamespace(mask=frame.mask)
    service._backend = SimpleNamespace(predict_polygon=predict)
    first = service.predict(frame, [(44.5, 31.25)], [(12, 8)])
    service.predict(first, [(46, 32)], [])
    assert calls[0]['cache_key'] == calls[1]['cache_key']
    assert calls[0]['positive_points'][0].x == 44.5
    assert calls[0]['negative_points'][0].y == 8
    service.predict(replace(frame, rgba=frame.rgba.copy()), [(45, 30)], [])
    assert calls[2]['cache_key'] != calls[0]['cache_key']
    for _ in range(4):
        service.predict(replace(frame, rgba=frame.rgba.copy()), [(45, 30)], [])
    assert len(service._sources) <= 2


def test_cancel_after_inference_and_invalid_prompts_do_not_publish():
    service = ContourWandService()
    frame = sample()
    source = CancellationTokenSource()
    def predict(**kwargs):
        source.cancel()
        return SimpleNamespace(mask=frame.mask)
    service._backend = SimpleNamespace(predict_polygon=predict)
    for positive in ([], [(-1, 10)], [(100, 30)], [(float('nan'), 4)]):
        with pytest.raises(ValueError):
            service.predict(frame, positive, [])
    with pytest.raises(CancellationError):
        service.predict(frame, [(45, 30)], [], token=source.token)
