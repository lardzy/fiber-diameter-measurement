"""Human-readable, non-blocking optical analysis evidence."""
from __future__ import annotations

import math


_WARNINGS = {
    "low_edge_support": "边缘支持较弱，请核对两侧位置",
    "variable_width": "沿线宽度变化较大",
    "ambiguous_pair": "存在其他可能的边缘配对",
    "near_crossing": "靠近交叉位置，请核对",
    "low_contrast": "纤维与背景对比较低",
    "uneven_illumination": "背景光照不均",
    "clipped_highlights": "部分区域过曝",
    "clipped_shadows": "部分区域欠曝",
    "flat_field_unavailable": "平场参考不可用，本次未使用",
    "flat_field_mismatch": "平场参考与当前来源不一致，本次未使用",
}


def candidate_diagnostic_text(candidate):
    """Quality features describe evidence; they are never an accuracy score."""
    diagnostic = candidate.get("diagnostics") or {}
    features = candidate.get("quality_features") or diagnostic.get("quality_features") or {}
    structure = candidate.get("structure") or diagnostic.get("structure")
    flags = candidate.get("warning_codes") or diagnostic.get("warning_codes") or []
    lines = []
    if structure:
        lines.append("识别结构：" + {"solid": "实心纤维", "double_edge": "透亮双边纤维"}.get(structure, str(structure)))
    for key, label, suffix in (("edge_snr", "边缘信噪比", ""),
            ("width_cv", "宽度变异系数", ""),
            ("edge_parallel_degrees", "两侧方向差", "°"),
            ("background_contrast", "背景对比", "")):
        value = features.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            lines.append(f"{label}：{value:.3g}{suffix}")
    if "valid_sections" in features and "total_sections" in features:
        lines.append(f"有效截面：{features['valid_sections']} / {features['total_sections']}")
    lines.extend(_WARNINGS.get(code, f"诊断：{code}") for code in flags)
    if candidate.get("conflicts_with"):
        lines.append("与另一种候选配对存在冲突；再次接受相关候选时将提示核对是否重复。")
    if lines:
        lines.append("这些指标用于复核，不代表测量准确率；可继续人工确认。")
    return "\n".join(lines)


def record_diagnostic_text(record):
    provenance = record.provenance or {}
    details = dict(provenance.get("candidate_diagnostics") or {})
    for key in ("structure", "quality_features", "warning_codes", "conflicts_with"):
        if key in provenance:
            details[key] = provenance[key]
    return candidate_diagnostic_text(details)


def flat_field_status_text(diagnostics):
    flat = (diagnostics or {}).get("flat_field") or {}
    if flat.get("applied") is True:
        return "最近分析：已应用空白平场校正。"
    if flat and flat.get("status") != "disabled":
        return "最近分析：空白平场校正未应用。" + str(flat.get("message") or "请检查参考是否与当前成像条件一致。")
    return ""


def illumination_summary(diagnostics):
    """Readable image metrics, separate from copyable machine diagnostics."""
    def number(key, *, percent=False):
        value = diagnostics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return "—"
        return f"{value * 100:.1f}%" if percent else f"{value:.1f}"
    lines = [f"背景亮度 {number('brightness_background')} / 255 · 近饱和 {number('saturated_high_fraction', percent=True)} · 近黑 {number('saturated_low_fraction', percent=True)}",
             f"背景不均 {number('background_unevenness', percent=True)} · 图像对比 {number('robust_contrast', percent=True)} · 清晰度指标 {number('focus_laplacian_variance')}"]
    lines.extend((diagnostics.get("suggestions") or [])[:2])
    return "\n".join(lines)
