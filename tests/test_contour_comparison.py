from dataclasses import replace
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from openpyxl import load_workbook
import pytest

from fdm.cancellation import CancellationError, CancellationTokenSource
from fdm.services.contour_comparison import (
    ContourAxis, ContourFrame, automatic_mask, boundary_segments,
    compare_contours, edit_mask, export_comparison_excel, frame_from_rgba,
    load_comparison, load_frame, save_comparison,
)


def rectangle(box=(10, 10, 90, 110), *, scale=.5, shape=(140, 120), axis=None):
    mask = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    rgba = np.full((*shape, 4), 255, np.uint8)
    rgba[mask, :3] = (25, 65, 115)
    return ContourFrame("reference", rgba, mask, axis or ContourAxis((50, 10), (50, 110)), scale)


def test_known_dimensions_and_signs_do_not_normalize_length():
    before = rectangle()
    after = rectangle((8, 12, 93, 108))
    result = compare_contours(before, after, .5)
    row = result.sections[20]
    assert (row.left_before, row.right_before, row.span_before) == (20, 20, 40)
    assert (row.left_change, row.right_change, row.span_change) == (1, 1.5, 2.5)
    assert result.summary["上端向外变化"] == -1
    assert result.summary["下端向外变化"] == -1
    assert result.summary["纵向总长变化"] == -2
    assert result.before_area == 2000
    assert result.after_area == 2040
    assert result.sections[0].status == "仅处理前有轮廓"
    assert result.sections[0].left_change is None
    assert result.sections[-1].span_change is None


def test_direction_point_length_is_not_a_scale():
    frame = rectangle()
    second = replace(frame, axis=ContourAxis((50, 10), (50, 20)))
    result = compare_contours(frame, second, 2)
    assert all(row.span_change == 0 for row in result.sections)
    assert result.summary["纵向总长变化"] == 0


def test_rigid_camera_axis_rotation_round_trip_and_scale():
    before = rectangle()
    rgba = np.rot90(before.rgba, -1).copy()
    mask = np.rot90(before.mask, -1).copy()
    after = ContourFrame("rotated", rgba, mask, ContourAxis((130, 50), (30, 50)), .5)
    result = compare_contours(before, after, 2)
    assert result.before_bounds == result.after_bounds
    assert all(row.span_change == 0 for row in result.sections)
    p = np.array([[23, 56], [-3, 40]])
    np.testing.assert_allclose(after.axis.to_image(after.axis.to_world(p, .5), .5), p)


def test_image_size_change_requires_calibration_and_uses_physical_height():
    before = rectangle(scale=1)
    mask = cv2.resize(before.mask.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST).astype(bool)
    rgba = cv2.resize(before.rgba, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    after = ContourFrame("2x", rgba, mask, ContourAxis((100, 20), (100, 220)), .5)
    result = compare_contours(before, after, 1)
    assert result.before_area == result.after_area
    assert all(row.span_change == 0 for row in result.sections)
    with pytest.raises(ValueError, match="尺寸不同"):
        compare_contours(replace(before, mm_per_pixel=None), replace(after, mm_per_pixel=None), 1)
    with pytest.raises(ValueError, match="两张图片"):
        compare_contours(before, replace(after, mm_per_pixel=None), 1)


def test_holes_and_leg_gaps_are_intervals_not_extra_fabric():
    frame = rectangle(scale=1)
    mask = frame.mask.copy()
    mask[50:110, 40:60] = False
    mask[20:25, 20:25] = False
    pants = replace(frame, mask=mask)
    result = compare_contours(pants, frame, 1)
    row = next(row for row in result.sections if row.height == 70.5)
    assert row.before_intervals == ((-40, -10), (10, 40))
    assert row.span_before == 80
    assert row.span_change == 0
    assert row.status.startswith("分段数改变")
    assert result.before_area == 8000 - 1200 - 25
    assert len(boundary_segments(mask)) > 360
    assert row.boundary_changes == ()


def test_inner_leg_edges_are_reported_when_interval_structure_agrees():
    before = rectangle(scale=1)
    b = before.mask.copy()
    b[50:110, 40:60] = False
    a = before.mask.copy()
    a[50:110, 38:62] = False
    r = compare_contours(replace(before, mask=b), replace(before, mask=a), 1)
    row = next(row for row in r.sections if row.height == 70.5)
    assert row.boundary_changes == ((0, 2), (2, 0))
    assert row.span_change == 0


def test_reference_origin_is_not_rebased_after_shape_edit():
    frame = rectangle(scale=None)
    moved = rectangle((12, 10, 92, 110), scale=None)
    result = compare_contours(frame, moved, 5)
    assert (result.sections[0].left_change, result.sections[0].right_change) == (-2, 2)
    assert result.unit == "px"
    assert result.warnings[0].startswith("未标定")


@pytest.mark.parametrize("angle", [0, .01, 17, 45, 80, 90, 135, 179])
def test_exact_boundary_sweep_is_closed_at_vertices(angle):
    f = rectangle(scale=1)
    angle = np.deg2rad(angle)
    f = replace(f, axis=ContourAxis((50.5, 10.5), (50.5 + 100*np.sin(angle), 10.5 + 100*np.cos(angle))))
    r = compare_contours(f, f, .5)
    assert all(row.span_change == 0 for row in r.sections if row.before_intervals)
    assert any(row.before_intervals for row in r.sections)


def test_large_sampling_step_still_samples_thin_object():
    f = rectangle((20, 12, 50, 13), scale=1)
    r = compare_contours(f, f, 100)
    assert len(r.sections) == 1
    assert r.sections[0].span_before == 30


@pytest.mark.parametrize("method", ["auto", "dark", "light"])
def test_segmentation_native_pixel_dimensions_and_leg_gap(method):
    rgba = np.full((180, 120, 4), 255, np.uint8)
    expected = np.zeros((180, 120), bool)
    expected[10:170, 20:100] = True
    expected[70:170, 50:70] = False
    rgba[expected, :3] = 40
    if method == "light":
        rgba[:, :, :3] = 255 - rgba[:, :, :3]
    mask, _ = automatic_mask(rgba, method=method)
    np.testing.assert_array_equal(mask, expected)
    assert not mask[150, 60]


def test_auto_fills_enclosed_hole_but_manual_subtract_preserves_it():
    frame = rectangle()
    rgba = frame.rgba.copy()
    rgba[30:35, 30:35, :3] = 255
    mask, _ = automatic_mask(rgba)
    assert mask[32, 32]
    edited = edit_mask(replace(frame, mask=mask), [(30, 30), (35, 30), (35, 35), (30, 35)], add=False, polygon=True)
    assert not edited.mask[32, 32]
    assert frame.mask[32, 32]
    assert edited.axis == frame.axis
    assert edited.mask.flags.writeable is False


def test_auto_transparency_roi_and_blank_failure():
    rgba = rectangle().rgba.copy()
    rgba[:, :, 3] = rectangle().mask.astype(np.uint8) * 255
    mask, _ = automatic_mask(rgba)
    np.testing.assert_array_equal(mask, rectangle().mask)
    blank = np.full((40, 50, 4), 255, np.uint8)
    m, warnings = automatic_mask(blank)
    assert not m.any() and warnings
    cropped, _ = automatic_mask(rectangle().rgba, roi=(0, 0, 95, 120))
    np.testing.assert_array_equal(cropped, rectangle().mask)
    with pytest.raises(ValueError, match="过小"):
        automatic_mask(rgba, roi=(-20, -20, 0, 0))


def test_save_reopen_independent_from_sources_and_excel_units(tmp_path):
    before = rectangle()
    after = replace(rectangle((8, 12, 93, 108)), label="=SUM(A1)")
    session = tmp_path / "portable.fdmcompare"
    save_comparison(session, before, after, .5)
    b, a, step, shared = load_comparison(session)
    assert shared and step == .5
    assert b.axis == before.axis
    np.testing.assert_array_equal(a.mask, after.mask)
    np.testing.assert_array_equal(a.rgba, after.rgba)
    result = compare_contours(b, a, step)
    export = tmp_path / "results.xlsx"
    export_comparison_excel(export, b, a, result)
    wb = load_workbook(export, data_only=False)
    assert wb["汇总与口径"]["B3"].data_type == "s"
    assert wb["汇总与口径"]["B3"].value == "=SUM(A1)"
    assert wb["逐高度外缘"]["D2"].value is None
    assert wb["逐高度外缘"]["D22"].value == 1
    assert "mm" in wb["逐高度外缘"]["A1"].value
    wb.close()


def test_failed_save_does_not_replace_previous_and_version_refused(tmp_path):
    path = tmp_path / "state.fdmcompare"
    before = rectangle()
    save_comparison(path, before, None, 10)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        save_comparison(path, before, None, float("nan"))
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".state*"))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"schema": "fdm.contour-comparison", "version": 999}))
    with pytest.raises(ValueError, match="版本"):
        load_comparison(path)


def test_cancel_and_sampling_limits():
    source = CancellationTokenSource()
    source.cancel()
    f = rectangle()
    with pytest.raises(CancellationError):
        compare_contours(f, f, 1, token=source.token)
    with pytest.raises(ValueError, match="5000"):
        compare_contours(f, f, .001)
    with pytest.raises(ValueError, match="有效轮廓"):
        compare_contours(f, replace(f, mask=np.zeros(f.mask.shape, bool)), 1)


def test_brush_keeps_full_motion_path_and_source_immutable():
    f = rectangle()
    points = [(20, 20), (80, 90), (25, 100)]
    edited = edit_mask(f, points, add=False, radius=2)
    assert not edited.mask[55, 50]
    assert f.mask[55, 50]
    assert edited.rgba is f.rgba


def test_file_orientation_and_multiframe_guard(tmp_path):
    from PIL import Image
    image = Image.fromarray(rectangle().rgba[:, :, :3])
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "oriented.jpg"
    image.save(path, exif=exif, quality=100)
    loaded = load_frame(path)
    assert loaded.mask.shape == (120, 140)
    stack = tmp_path / "stack.tif"
    image.save(stack, save_all=True, append_images=[image])
    with pytest.raises(ValueError, match="堆栈"):
        load_frame(stack)


def test_roi_touching_foreground_is_not_silently_complete():
    f = rectangle()
    _, warnings = automatic_mask(f.rgba, method="dark", roi=(20, 0, 95, 120))
    assert any("框选边界" in warning for warning in warnings)
