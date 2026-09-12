"""Read-only guidance for acquisition settings; never opens a source or moves hardware."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from fdm.services.slide_capture_geometry import (
    calibrated_capture_settings, calibration_signature, scaled_capture_size,
)

if TYPE_CHECKING:
    from fdm.settings import AppSettings


@dataclass(frozen=True)
class CaptureSettingsGuidance:
    calibration: str
    context: str
    result: str
    notice: str


def _size(value: object) -> tuple[int, int] | None:
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            result = tuple(int(v) for v in value)
            return result if min(result) > 0 else None
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def _axis_ready(value: object) -> bool:
    if not isinstance(value, dict) or not value.get("reliable"):
        return False
    try:
        primary = float(value.get("pixels_per_step", 0))
        return primary > 0 and all(math.isfinite(float(value.get(k, 0))) for k in (
            "pixels_per_step", "cross_per_step", "uncertainty_px"))
    except (ValueError, TypeError, OverflowError):
        return False


def capture_settings_guidance(
    settings: AppSettings, source_frame_size: tuple[int, int] | None = None,
) -> CaptureSettingsGuidance:
    profile = settings.digital_slide_xy_calibration
    signature = profile.get("signature", {})
    signature = signature if isinstance(signature, dict) else {}
    live_size = _size(source_frame_size)
    recorded_source = _size(profile.get("capture_frame_size"))
    recorded_size = _size(signature.get("size"))
    source_size = live_size or recorded_source
    if source_size:
        width, height, _ = scaled_capture_size(*source_size, settings.digital_slide_capture_max_width)
        frame_size = (width, height)
    else:
        # Old profiles may lack the original camera size. Only reuse their exact
        # saved width; do not invent a new original resolution or aspect ratio.
        frame_size = recorded_size if recorded_size and settings.digital_slide_capture_max_width == recorded_size[0] else None

    mismatches = []
    if signature:
        expected = calibration_signature(settings, frame_size or recorded_size or (0, 0))
        for key, label in (("camera", "相机"), ("profile", "采集配置"), ("reverse", "坐标方向"), ("stage_signs", "电机步距方向")):
            if signature.get(key) != expected[key]:
                mismatches.append(label)
        if frame_size != recorded_size:
            mismatches.append("保存尺寸")
        if live_size and recorded_source and live_size != recorded_source:
            mismatches.append("相机原始尺寸")
    axis_ready = [_axis_ready(profile.get(axis)) for axis in ("x", "y")]
    if mismatches:
        calibration = "校准需更新：" + "、".join(mismatches) + "已变化"
    else:
        calibration = " · ".join(f"{axis} {'已有校准' if ready and signature else '待校准'}" for axis, ready in zip(("X", "Y"), axis_ready))
    if frame_size:
        context = f"保存视场 {frame_size[0]} × {frame_size[1]} px"
        context += " · 当前相机帧" if live_size else " · 按历史档案估算，采集前核对"
    else:
        context = "连接相机并启动预览后，显示保存尺寸与预计间距。"

    mode = settings.digital_slide_pixel_stride_mode
    if mode == "manual_pixels":
        return CaptureSettingsGuidance(calibration, context, "电机按填写的步距移动，图像按填写的像素间距排布。",
            "此模式不使用重叠百分比。图像间距不能证明实际采到了多少公共内容。")
    if mode != "calibrated_overlap":
        result = "连接相机后自动计算图像排布间距。"
        if frame_size:
            strides = [max(1, round(v * (1 - settings.digital_slide_overlap_percent / 100))) for v in frame_size]
            result = f"图像排布间距：X {strides[0]} px · Y {strides[1]} px"
        return CaptureSettingsGuidance(calibration, context, result,
            "排布重叠只改变图像摆放，不改变电机步距。要采集真实重叠，请完成 XY 校准并选择校准联动。")

    if mismatches or not signature or not all(axis_ready) or frame_size is None:
        return CaptureSettingsGuidance(calibration, context, "校准完成后自动显示 X、Y 电机步距和图像间距。",
            "当前不能按目标重叠采集。请更新 XY 校准；也可切回原有步距模式。")
    try:
        effective = calibrated_capture_settings(settings, frame_size, source_frame_size=live_size)
    except (ValueError, TypeError, OverflowError):
        return CaptureSettingsGuidance("校准档案需更新", context, "请重新核对 XY 步距校准。",
            "现有档案不能用于联动采集；旧模式的采集参数仍保留。")
    from fdm.services.slide_registration import RegistrationConfig
    config = RegistrationConfig()
    lines, insufficient, usable_percent = [], [], []
    for index, axis in enumerate(("x", "y")):
        step = getattr(effective, f"digital_slide_{axis}_stage_step")
        stride = getattr(effective, f"digital_slide_{axis}_pixel_stride")
        extent = frame_size[index]
        overlap = max(0, extent - stride)
        uncertainty = max(2, extent * config.uncertainty_fraction, float(profile[axis].get("uncertainty_px", 0)))
        usable = max(0, overlap - uncertainty)
        usable_percent.append(f"{axis.upper()} {100 * usable / extent:.1f}%")
        if usable < max(config.min_strip_pixels, extent * config.min_overlap):
            insufficient.append(axis.upper())
        lines.append(f"{axis.upper()}：{step} steps → {stride} px；预计重叠 {100 * overlap / extent:.1f}%")
    lines.append("扣除定位余量后：" + " · ".join(usable_percent))
    notice = "建议从 20% 开始；开始采集前会核对设备，并按范围重算张数、时间和容量。"
    if insufficient:
        notice = "预计有效重叠不足（" + "、".join(insufficient) + "）：扣除定位余量后，至少需 10% 且 64 px。建议提高目标重叠或检查校准。"
    elif settings.digital_slide_overlap_percent >= 25:
        increase = ((.8 / (1 - settings.digital_slide_overlap_percent / 100)) ** 2 - 1) * 100
        notice = f"同范围大网格相较 20% 约多 {increase:.0f}% 视场；实际张数和耗时以采集计划为准。"
    if not live_size:
        notice += " 当前为档案估算，尚未核对实时相机。"
    return CaptureSettingsGuidance(calibration, context, "\n".join(lines), notice)
