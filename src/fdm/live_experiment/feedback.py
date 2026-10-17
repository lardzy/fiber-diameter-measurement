"""Localized, frame-free measurement feedback shared by controllers and views."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


FeedbackSeverity = Literal["info", "success", "warning", "error"]

# Codes are retained for diagnostics; users get an actionable Chinese message.
REJECTION_MESSAGES = {
    "line_too_short": "测量线过短，请在纤维两侧背景分别选择起点和终点。",
    "endpoints_not_background": "起点或终点落在纤维上，请将两端放在纤维外的背景处。",
    "no_complete_crossing": "未识别到完整的纤维两侧边缘。请在纤维两侧背景分别落点，使测量线完整跨过纤维。",
    "edge_tracking_failed": "未能连续追踪纤维边缘，请选择边缘清晰的位置重试。",
    "edge_span_too_short": "可用边缘过短，请选择更完整、清晰的纤维区域。",
    "edges_not_parallel": "两侧边缘不够平行，请避开交叉或弯曲处重新测量。",
    "crossing_too_oblique": "测量线与纤维的夹角过小，请沿纤维的垂直方向选择两端。",
    "invalid_width": "未得到有效直径，请重新选择纤维两侧的测量点。",
    "zero_length": "两个测量点重合，请选择不同的起点和终点。",
    "edge_not_found": "未找到有效的两侧边缘，请调整两个测量点的位置后重试。",
    "verification_timeout": "视场核验超过两秒，本次结果未入账；请等待视场稳定后重试。",
    "confirmation_timeout": "视场核验超过两秒，本次确认未入账；请等待视场稳定后重试。",
    "frame_unavailable": "当前没有完整可用的图像，请等待画面恢复后重试。",
    "resource_limit": "图像缓冲暂满，请等待当前任务完成后重试。",
    "no_valid_fibers": "未检测到符合当前算法规则的纤维，请调整视场或分析参数。",
    "algorithm_error": "图像分析未能完成，请检查当前图像并重试。",
    "worker_crashed": "分析进程意外退出，本次结果未入账；请重试。",
    "timeout": "全图分析超过时限，本次结果未入账；请缩小分析范围后重试。",
    "cancelled": "本次测量已取消。",
    "measurement_failed": "本次测量未能完成，请调整测量位置后重试。",
}

_ALIASES = {"零长度测量": "zero_length", "未找到有效的两侧边缘": "edge_not_found",
            "全图分析失败": "algorithm_error", "测量已取消": "cancelled"}


@dataclass(frozen=True, slots=True)
class MeasurementFeedback:
    """One presentation event, never retained in projects or image exports.

    Context contains scalar identities only. Keeping a feedback card visible
    cannot retain a full frame, pending intent or native acquisition buffer.
    """

    code: str
    message: str
    severity: FeedbackSeverity = "warning"
    intent_id: str | None = None
    field_id: str | None = None
    field_epoch: int | None = None
    source_id: str | None = None
    source_generation: int | None = None
    frame_id: str | None = None
    detail: str = ""

    def __post_init__(self):
        if self.severity not in {"info", "success", "warning", "error"}:
            raise ValueError("反馈级别无效。")


def feedback_for_reason(reason, *, code=None, severity=None, **context) -> MeasurementFeedback:
    """Translate known kernel/worker codes, preserving Chinese domain guidance.

    Unknown English exception text remains in ``detail`` for diagnostics and
    never becomes the visible card. Views can also use this for synchronous
    input validation errors caught by their existing operation guards.
    """
    if isinstance(reason, dict):
        reason = reason.get("message") or reason.get("reason") or reason.get("type") or ""
    detail = str(reason or "").strip()
    key = code or _ALIASES.get(detail, detail)
    if key in REJECTION_MESSAGES:
        message = REJECTION_MESSAGES[key]
    elif re.search(r"[\u3400-\u9fff]", detail):
        key, message = code or "measurement_rejected", detail
    else:
        key = code or (detail if re.fullmatch(r"[a-z][a-z0-9_]*", detail) else "measurement_failed")
        message = REJECTION_MESSAGES["measurement_failed"]
    level = severity or ("info" if key == "cancelled" else "warning")
    return MeasurementFeedback(code=key, message=message, severity=level, detail=detail, **context)
