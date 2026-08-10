from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import PureWindowsPath
from typing import Callable, Mapping, Sequence

from fdm.platform.windows_window_locator import (
    PhysicalRect,
    WindowRecord,
    WindowSnapshot,
    enumerate_windows,
)


CU5_PROCESS_NAMES = frozenset({"cu-5.exe", "cu5.exe"})
_RESOURCE_TEXT_TOKENS = (
    "实时预览",
    "视频预览",
    "图像预览",
    "preview",
    "microview",
    "video",
)
_MINIMUM_SCORE = 75.0
_DEFAULT_AMBIGUITY_MARGIN = 12.0


class Cu5PreviewLocatorError(RuntimeError):
    code = "locator_error"


class Cu5PreviewNotFoundError(Cu5PreviewLocatorError):
    code = "not_found"


class Cu5PreviewUnavailableError(Cu5PreviewLocatorError):
    code = "unavailable"


class Cu5PreviewAmbiguousError(Cu5PreviewLocatorError):
    code = "ambiguous"

    def __init__(self, message: str, *, candidates: tuple["Cu5PreviewCandidate", ...]) -> None:
        super().__init__(message)
        self.candidates = candidates


def _canonical_class_name(value: object) -> str:
    """Return a restart-stable Win32 class token.

    MFC sometimes embeds the module base address in top-level class names
    (``Afx:00400000:...``).  Persisting that whole value would be almost as
    brittle as persisting an HWND, so those generated names collapse to the
    stable ``afx`` family token.
    """

    token = str(value or "").strip().casefold()
    if token.startswith("afx:"):
        return "afx"
    return token


@dataclass(frozen=True, slots=True)
class Cu5PreviewSelector:
    """Restart-stable hints for selecting CU-5's rendered preview child.

    Deliberately absent are HWNDs, PIDs and absolute desktop coordinates.  The
    optional size is only a weak hint because CU-5 can be resized.
    """

    class_name: str = ""
    control_id: int | None = None
    process_name: str = ""
    width: int | None = None
    height: int | None = None
    ancestor_classes: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: object) -> "Cu5PreviewSelector":
        if isinstance(value, cls):
            return value.normalized()
        if not isinstance(value, Mapping):
            return cls()
        raw_size = value.get("size")
        size = raw_size if isinstance(raw_size, Mapping) else {}
        raw_ancestors = value.get("ancestor_classes", value.get("ancestors", ()))
        if isinstance(raw_ancestors, str):
            raw_ancestors = (raw_ancestors,)
        if not isinstance(raw_ancestors, (tuple, list)):
            raw_ancestors = ()
        return cls(
            class_name=str(value.get("class_name", value.get("class", "")) or ""),
            control_id=_optional_positive_int(value.get("control_id")),
            process_name=str(
                value.get("process_name", value.get("process", "")) or ""
            ),
            width=_optional_positive_int(size.get("width", value.get("width"))),
            height=_optional_positive_int(size.get("height", value.get("height"))),
            ancestor_classes=tuple(str(item or "") for item in raw_ancestors),
        ).normalized()

    @classmethod
    def from_record(
        cls,
        record: WindowRecord,
        snapshot: WindowSnapshot | None = None,
    ) -> "Cu5PreviewSelector":
        ancestor_classes: list[str] = []
        if snapshot is not None:
            for hwnd in record.ancestor_hwnds:
                ancestor = snapshot.by_hwnd.get(hwnd)
                if ancestor is None:
                    continue
                token = _canonical_class_name(ancestor.class_name)
                if token and token not in ancestor_classes:
                    ancestor_classes.append(token)
        return cls(
            class_name=record.class_name,
            control_id=record.control_id,
            process_name=record.process_name,
            width=record.rect.width,
            height=record.rect.height,
            ancestor_classes=tuple(ancestor_classes),
        ).normalized()

    def normalized(self) -> "Cu5PreviewSelector":
        ancestors: list[str] = []
        for value in self.ancestor_classes:
            token = _canonical_class_name(value)
            if token and token not in ancestors:
                ancestors.append(token)
        return Cu5PreviewSelector(
            class_name=_canonical_class_name(self.class_name),
            control_id=_optional_positive_int(self.control_id),
            process_name=PureWindowsPath(str(self.process_name or "")).name.casefold(),
            width=_optional_positive_int(self.width),
            height=_optional_positive_int(self.height),
            ancestor_classes=tuple(ancestors),
        )

    @property
    def active(self) -> bool:
        return bool(
            self.class_name
            or self.control_id is not None
            or self.process_name
            or self.width is not None
            or self.height is not None
            or self.ancestor_classes
        )

    def to_dict(self) -> dict[str, object]:
        selector = self.normalized()
        payload: dict[str, object] = {}
        if selector.process_name:
            payload["process_name"] = selector.process_name
        if selector.class_name:
            payload["class_name"] = selector.class_name
        if selector.control_id is not None:
            payload["control_id"] = selector.control_id
        if selector.width is not None and selector.height is not None:
            payload["size"] = {
                "width": selector.width,
                "height": selector.height,
            }
        if selector.ancestor_classes:
            payload["ancestor_classes"] = list(selector.ancestor_classes)
        return payload


def _optional_positive_int(value: object) -> int | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


@dataclass(frozen=True, slots=True)
class Cu5PreviewCandidate:
    record: WindowRecord
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Cu5PreviewMatch:
    record: WindowRecord
    score: float
    reasons: tuple[str, ...]
    runner_up_score: float | None = None
    selector: Cu5PreviewSelector | None = None

    @property
    def hwnd(self) -> int:
        return self.record.hwnd

    @property
    def rect(self) -> PhysicalRect:
        return self.record.rect


def _normalized_class(record: WindowRecord) -> str:
    return record.class_name.strip().casefold()


def _contains_resource_text(value: str) -> bool:
    normalized = str(value or "").strip().casefold()
    return any(token.casefold() in normalized for token in _RESOURCE_TEXT_TOKENS)


def _looks_like_cu5_root(
    record: WindowRecord,
    selector: Cu5PreviewSelector | None = None,
) -> bool:
    process_name = PureWindowsPath(record.process_path).name.casefold()
    if selector is not None and selector.process_name:
        return record.parent_hwnd is None and process_name == selector.process_name
    if process_name in CU5_PROCESS_NAMES:
        return True
    compact_title = record.title.casefold().replace(" ", "").replace("_", "")
    return record.parent_hwnd is None and ("cu-5" in compact_title or "cu5" in compact_title)


def _is_sdk_class(record: WindowRecord) -> bool:
    class_name = _normalized_class(record).replace("_", "")
    return class_name == "cwndforsdk" or "cwndforsdk" in class_name


def _is_excluded_control(record: WindowRecord) -> bool:
    if _is_sdk_class(record):
        return False
    class_name = _normalized_class(record)
    return class_name in {
        "button",
        "static",
        "edit",
        "combobox",
        "toolbarwindow32",
        "rebarwindow32",
        "statusbar",
        "msctls_statusbar32",
        "mdiclient",
    }


def _candidate_resource_score(
    candidate: WindowRecord,
    records: Sequence[WindowRecord],
) -> tuple[float, tuple[str, ...]]:
    if _contains_resource_text(candidate.title):
        return 48.0, ("窗口文本指向实时预览",)
    related: list[tuple[int, WindowRecord]] = []
    candidate_ancestors = set(candidate.ancestor_hwnds)
    for record in records:
        if record.pid != candidate.pid or not _contains_resource_text(record.title):
            continue
        if record.hwnd in candidate_ancestors:
            related.append((0, record))
        elif record.parent_hwnd == candidate.parent_hwnd and record.parent_hwnd is not None:
            related.append((1, record))
        elif record.root_hwnd == candidate.root_hwnd:
            related.append((2, record))
    if not related:
        return 0.0, ()
    relationship = min(item[0] for item in related)
    if relationship == 0:
        return 40.0, ("预览资源文本位于祖先窗口",)
    if relationship == 1:
        return 30.0, ("同级资源文本指向实时预览",)
    return 10.0, ("CU-5 视图包含预览资源文本",)


def _geometry_score(
    candidate: WindowRecord,
    container: WindowRecord,
) -> tuple[float, tuple[str, ...]]:
    width = candidate.rect.width
    height = candidate.rect.height
    score = 0.0
    reasons: list[str] = []
    if (width, height) == (768, 576):
        score += 48.0
        reasons.append("精确匹配 768x576 视频区域")
    ratio = width / height if height else 0.0
    ratio_error = abs(ratio - (4.0 / 3.0))
    if ratio_error <= 0.01:
        score += 28.0
        reasons.append("匹配 4:3 视频比例")
    elif ratio_error <= 0.04:
        score += 16.0
        reasons.append("接近 4:3 视频比例")
    if width >= 320 and height >= 240:
        score += 6.0

    if container.rect.contains_rect(candidate.rect) and container.rect.area > 0:
        candidate_x, candidate_y = candidate.rect.center
        container_x, container_y = container.rect.center
        half_width = max(1.0, container.rect.width / 2.0)
        half_height = max(1.0, container.rect.height / 2.0)
        normalized_distance = math.hypot(
            (candidate_x - container_x) / half_width,
            (candidate_y - container_y) / half_height,
        ) / math.sqrt(2.0)
        central_score = 22.0 * max(0.0, 1.0 - normalized_distance)
        score += central_score
        if central_score >= 14.0:
            reasons.append("位于 MDI 工作区中央")
        area_fraction = candidate.rect.area / container.rect.area
        if 0.10 <= area_fraction <= 0.90:
            score += 8.0
    return score, tuple(reasons)


def _score_candidate(
    candidate: WindowRecord,
    *,
    snapshot: WindowSnapshot,
    process_records: Sequence[WindowRecord],
    selector: Cu5PreviewSelector | None = None,
) -> Cu5PreviewCandidate | None:
    if (
        candidate.parent_hwnd is None
        or not candidate.visible
        or candidate.minimized
        or candidate.cloaked
        or candidate.rect.width < 160
        or candidate.rect.height < 120
        or _is_excluded_control(candidate)
    ):
        return None

    score = 0.0
    reasons: list[str] = []
    if _is_sdk_class(candidate):
        score += 145.0
        reasons.append("匹配 CWndForSDK 视频宿主类")

    ancestors = [
        snapshot.by_hwnd[hwnd]
        for hwnd in candidate.ancestor_hwnds
        if hwnd in snapshot.by_hwnd
    ]
    mdi_ancestors = [
        record
        for record in ancestors
        if _normalized_class(record) == "mdiclient"
    ]
    if mdi_ancestors:
        score += 32.0
        reasons.append("位于 MDIClient 层级")
    afx_count = sum(
        1
        for record in ancestors
        if _normalized_class(record).startswith("afx")
        or "afx" in _normalized_class(record)
    )
    if afx_count:
        score += min(20.0, 6.0 * afx_count)
        reasons.append("位于 MFC Afx 视图层级")

    resource_score, resource_reasons = _candidate_resource_score(
        candidate,
        process_records,
    )
    score += resource_score
    reasons.extend(resource_reasons)

    root = snapshot.by_hwnd.get(candidate.root_hwnd)
    container = mdi_ancestors[-1] if mdi_ancestors else root
    if container is not None:
        geometry_score, geometry_reasons = _geometry_score(candidate, container)
        score += geometry_score
        reasons.extend(geometry_reasons)

    if candidate.control_id not in (None, 0, -1):
        score += 3.0
    selector_score, selector_reasons = _selector_score(candidate, ancestors, selector)
    score += selector_score
    reasons.extend(selector_reasons)
    if score < _MINIMUM_SCORE:
        return None
    return Cu5PreviewCandidate(candidate, score, tuple(reasons))


def _selector_score(
    candidate: WindowRecord,
    ancestors: Sequence[WindowRecord],
    selector: Cu5PreviewSelector | None,
) -> tuple[float, tuple[str, ...]]:
    if selector is None or not selector.active:
        return 0.0, ()
    score = 0.0
    reasons: list[str] = []
    if selector.process_name:
        if candidate.process_name.casefold() == selector.process_name:
            score += 35.0
            reasons.append("匹配已记忆的 CU-5 进程名")
        else:
            score -= 180.0
    if selector.class_name:
        if _canonical_class_name(candidate.class_name) == selector.class_name:
            score += 110.0
            reasons.append("匹配已记忆的视频宿主类")
        else:
            score -= 65.0
    if selector.control_id is not None:
        if candidate.control_id == selector.control_id:
            score += 95.0
            reasons.append("匹配已记忆的控件 ID")
        else:
            score -= 35.0
    if selector.width is not None and selector.height is not None:
        width_error = abs(candidate.rect.width - selector.width) / max(1, selector.width)
        height_error = abs(candidate.rect.height - selector.height) / max(1, selector.height)
        size_error = max(width_error, height_error)
        if size_error <= 0.01:
            score += 28.0
            reasons.append("匹配已记忆的视频区域尺寸")
        elif size_error <= 0.10:
            score += 12.0
    if selector.ancestor_classes:
        actual = {_canonical_class_name(item.class_name) for item in ancestors}
        matched = sum(1 for item in selector.ancestor_classes if item in actual)
        if matched:
            score += min(36.0, matched * 12.0)
            reasons.append("匹配已记忆的祖先窗口层级")
    return score, tuple(reasons)


def rank_cu5_preview_candidates(
    source: WindowSnapshot | Sequence[WindowRecord],
    *,
    selector: Cu5PreviewSelector | Mapping[str, object] | None = None,
) -> tuple[Cu5PreviewCandidate, ...]:
    snapshot = (
        source
        if isinstance(source, WindowSnapshot)
        else WindowSnapshot.from_records(source)
    )
    stable_selector = Cu5PreviewSelector.from_value(selector)
    roots = [
        record
        for record in snapshot.roots()
        if _looks_like_cu5_root(record, stable_selector)
    ]
    if not roots:
        return ()
    available_roots = [
        record
        for record in roots
        if record.visible and not record.minimized and not record.cloaked
    ]
    if not available_roots:
        return ()
    available_root_ids = {record.hwnd for record in available_roots}
    available_pids = {record.pid for record in available_roots}
    process_records = [
        record
        for record in snapshot.records
        if record.pid in available_pids and record.root_hwnd in available_root_ids
    ]
    ranked = [
        scored
        for record in process_records
        if (scored := _score_candidate(
            record,
            snapshot=snapshot,
            process_records=process_records,
            selector=stable_selector,
        ))
        is not None
    ]
    ranked.sort(key=lambda item: (-item.score, item.record.hwnd))
    return tuple(ranked)


def locate_cu5_preview(
    source: WindowSnapshot | Sequence[WindowRecord],
    *,
    ambiguity_margin: float = _DEFAULT_AMBIGUITY_MARGIN,
    selector: Cu5PreviewSelector | Mapping[str, object] | None = None,
) -> Cu5PreviewMatch:
    snapshot = (
        source
        if isinstance(source, WindowSnapshot)
        else WindowSnapshot.from_records(source)
    )
    stable_selector = Cu5PreviewSelector.from_value(selector)
    roots = [
        record
        for record in snapshot.roots()
        if _looks_like_cu5_root(record, stable_selector)
    ]
    if not roots:
        raise Cu5PreviewNotFoundError("未找到正在运行的 CU-5.exe 主窗口。")
    available_roots = [
        record
        for record in roots
        if record.visible and not record.minimized and not record.cloaked
    ]
    if not available_roots:
        states: list[str] = []
        if any(record.minimized for record in roots):
            states.append("已最小化")
        if any(record.cloaked for record in roots):
            states.append("被系统隐藏")
        if any(not record.visible for record in roots):
            states.append("不可见")
        detail = "、".join(states) or "不可用"
        raise Cu5PreviewUnavailableError(
            f"CU-5 主窗口当前{detail}，无法可靠截取实时预览。"
        )

    ranked = rank_cu5_preview_candidates(snapshot, selector=stable_selector)
    if not ranked:
        raise Cu5PreviewNotFoundError(
            "已找到 CU-5，但未识别到可靠的实时预览视频区域。"
        )
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = max(0.0, float(ambiguity_margin))
    if runner_up is not None and best.score - runner_up.score < margin:
        raise Cu5PreviewAmbiguousError(
            "CU-5 中存在多个相近的预览区域，已停止自动截图以避免截错。",
            candidates=ranked[:4],
        )
    return Cu5PreviewMatch(
        record=best.record,
        score=best.score,
        reasons=best.reasons,
        runner_up_score=runner_up.score if runner_up is not None else None,
        selector=Cu5PreviewSelector.from_record(best.record, snapshot),
    )


class Cu5PreviewLocator:
    """Locates the already-rendered CU-5 video child window.

    This service only inspects the Win32 window tree. It deliberately never
    imports or opens Microview, because CU-5 may own the capture board.
    """

    def __init__(
        self,
        *,
        enumerate_snapshot: Callable[[], WindowSnapshot] = enumerate_windows,
        ambiguity_margin: float = _DEFAULT_AMBIGUITY_MARGIN,
        selector: Cu5PreviewSelector | Mapping[str, object] | None = None,
    ) -> None:
        self._enumerate_snapshot = enumerate_snapshot
        self._ambiguity_margin = max(0.0, float(ambiguity_margin))
        self._selector = Cu5PreviewSelector.from_value(selector)

    @property
    def selector(self) -> Cu5PreviewSelector:
        return self._selector

    def set_selector(
        self,
        selector: Cu5PreviewSelector | Mapping[str, object] | None,
    ) -> None:
        self._selector = Cu5PreviewSelector.from_value(selector)

    def locate(
        self,
        source: WindowSnapshot | Sequence[WindowRecord] | None = None,
    ) -> Cu5PreviewMatch:
        snapshot = self._enumerate_snapshot() if source is None else source
        return locate_cu5_preview(
            snapshot,
            ambiguity_margin=self._ambiguity_margin,
            selector=self._selector,
        )


__all__ = [
    "CU5_PROCESS_NAMES",
    "Cu5PreviewAmbiguousError",
    "Cu5PreviewCandidate",
    "Cu5PreviewLocator",
    "Cu5PreviewLocatorError",
    "Cu5PreviewMatch",
    "Cu5PreviewNotFoundError",
    "Cu5PreviewSelector",
    "Cu5PreviewUnavailableError",
    "locate_cu5_preview",
    "rank_cu5_preview_candidates",
]
