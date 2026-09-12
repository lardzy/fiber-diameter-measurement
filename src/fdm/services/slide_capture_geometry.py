"""Calibrated physical stride; never reinterpret an existing acquisition."""
from __future__ import annotations

from dataclasses import replace
import math


def scaled_capture_size(source_width: int, source_height: int, max_width: int | None) -> tuple[int, int, float]:
    """Use the same saved-pixel dimensions in planning and settings guidance."""
    source_width, source_height = max(1, int(source_width)), max(1, int(source_height))
    if not max_width or source_width <= max_width:
        return source_width, source_height, 1.0
    scale = max_width / source_width
    return int(max_width), max(1, int(source_height * scale)), scale


def calibration_signature(settings, frame_size):
    return {
        "camera": settings.selected_capture_device_id,
        "profile": settings.digital_slide_active_profile_id,
        "size": list(frame_size),
        "reverse": [settings.digital_slide_reverse_x_axis, settings.digital_slide_reverse_y_axis],
        "stage_signs": [1 if settings.digital_slide_x_stage_step >= 0 else -1, 1 if settings.digital_slide_y_stage_step >= 0 else -1],
    }


def calibrated_capture_settings(settings, frame_size, *, source_frame_size=None):
    if settings.digital_slide_pixel_stride_mode != "calibrated_overlap":
        return settings
    profile = settings.digital_slide_xy_calibration
    if profile.get("signature") != calibration_signature(settings, frame_size):
        raise ValueError("设备／采集分辨率与校准档案不一致，请完成 X、Y 校准或选择原有步距模式。")
    if source_frame_size is not None and profile.get("capture_frame_size") and list(source_frame_size) != profile["capture_frame_size"]:
        raise ValueError("相机原始分辨率与校准档案不一致，不能复用物理步距校准。")
    vectors = []
    values = {}
    for index, axis in enumerate(("x", "y")):
        evidence = profile.get(axis, {})
        primary = float(evidence.get("pixels_per_step", 0))
        cross = float(evidence.get("cross_per_step", 0))
        uncertainty = float(evidence.get("uncertainty_px", 0))
        if not evidence.get("reliable") or not all(math.isfinite(v) for v in (primary, cross, uncertainty)) or primary <= 0:
            raise ValueError(f"{axis.upper()} 校准证据不足，不能按重叠率改变电机步距。")
        desired = frame_size[index] * (1 - settings.digital_slide_overlap_percent / 100)
        step = max(1, round(desired / primary))
        old_step = getattr(settings, f"digital_slide_{axis}_stage_step")
        values[f"digital_slide_{axis}_stage_step"] = step if old_step >= 0 else -step
        values[f"digital_slide_{axis}_pixel_stride"] = max(1, round(step * primary))
        vectors.append([step * primary, step * cross] if index == 0 else [step * cross, step * primary])
    values["digital_slide_pixel_stride_mode"] = "manual_pixels"
    values["digital_slide_xy_calibration"] = {**profile, "applied_vectors": vectors}
    # Profiles are not re-normalized here: that would overwrite the frozen stride.
    return replace(settings, **values)


def calibrated_plan_coordinates(plan, settings, frame_size):
    vectors = settings.digital_slide_xy_calibration.get("applied_vectors")
    if not vectors or not plan:
        return None
    positions = [(item["col"] * vectors[0][0] + item["row"] * vectors[1][0],
                  item["col"] * vectors[0][1] + item["row"] * vectors[1][1]) for item in plan]
    ox = min(0, math.floor(min(p[0] for p in positions)))
    oy = min(0, math.floor(min(p[1] for p in positions)))
    # Tile storage uses integer positions; refined subpixel shifts belong to the layout.
    for item, (x, y) in zip(plan, positions):
        item["global_x"], item["global_y"] = round(x - ox), round(y - oy)
    return (max(item["global_x"] for item in plan) + frame_size[0],
            max(item["global_y"] for item in plan) + frame_size[1])
