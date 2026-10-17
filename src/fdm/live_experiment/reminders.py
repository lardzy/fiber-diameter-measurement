"""Quantity goals and session-local notifications, independent of widgets.

Goals describe accepted business records, never candidates or failed attempts.
The policy is project data; a tracker's delivery history is deliberately not.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ExperimentSession


def _target(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("数量目标必须是正整数")
    return value


@dataclass(frozen=True)
class ReminderPolicy:
    """Version 1 goals. Material overrides use stable IDs across owners/directions.

    ``material_targets`` accepts a mapping on construction and is normalized to
    an immutable tuple, so configuration history cannot be edited accidentally.
    """

    version: int = 1
    content_measured_target: int = 300
    content_total_target: int = 1500
    diameter_measured_target: int = 300
    material_targets: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version != 1:
            raise ValueError("不支持的数量目标配置版本")
        for name in ("content_measured_target", "content_total_target", "diameter_measured_target"):
            _target(getattr(self, name))
        raw = self.material_targets
        pairs = raw.items() if isinstance(raw, Mapping) else raw
        try:
            values = list(pairs)
            if any(len(item) != 2 for item in values):
                raise ValueError
            normalized = []
            for material_id, target in values:
                if not isinstance(material_id, str) or not material_id.strip():
                    raise ValueError
                normalized.append((material_id, _target(target)))
            if len({key for key, _ in normalized}) != len(normalized):
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError("类别数量目标必须使用唯一的材料 ID 和正整数") from error
        object.__setattr__(self, "material_targets", tuple(sorted(normalized)))

    def material_target(self, material_id: str) -> int:
        return dict(self.material_targets).get(material_id, self.content_measured_target)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version,
                "content_measured_target": self.content_measured_target,
                "content_total_target": self.content_total_target,
                "diameter_measured_target": self.diameter_measured_target,
                "material_targets": dict(self.material_targets)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReminderPolicy":
        if not isinstance(value, dict):
            raise ValueError("数量目标配置必须是对象")
        allowed = {"version", "content_measured_target", "content_total_target",
                   "diameter_measured_target", "material_targets"}
        if value.keys() - allowed or "version" not in value:
            raise ValueError("数量目标配置缺少版本或包含未知字段")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError("数量目标配置无效") from error


def normalize_policy(value: ReminderPolicy | dict[str, Any] | None) -> ReminderPolicy | None:
    if value is None or isinstance(value, ReminderPolicy):
        return value
    return ReminderPolicy.from_dict(value)


@dataclass(frozen=True)
class ReminderProgress:
    session_id: str
    kind: str  # measured, total, or diameter_measured
    operator_id: str
    direction: str | None
    material_id: str | None
    count: int
    target: int
    operator_name: str
    material_name: str = ""
    legacy: bool = False

    @property
    def reached(self) -> bool:
        return self.count >= self.target

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.session_id, self.legacy, self.kind, self.operator_id,
                self.direction, self.material_id, self.target)


@dataclass(frozen=True)
class ReminderNotice:
    """One delivery for all targets crossed by a business operation or batch."""

    crossings: tuple[ReminderProgress, ...]

    @property
    def text(self) -> str:
        messages = []
        for item in self.crossings:
            prefix = item.operator_name
            if item.direction is not None:
                prefix += "／" + ("经向" if item.direction == "warp" else "纬向")
            subject = (f"{item.material_name}实测" if item.kind == "measured" else
                       "有效测量" if item.kind == "diameter_measured" else "总根数")
            messages.append(f"{prefix} {subject}已达到 {item.target} 根（当前 {item.count} 根）")
        return "；".join(messages)


def reminder_progress(session: "ExperimentSession", *, operator_id: str | None = None,
                      direction: str | None = None) -> tuple[ReminderProgress, ...]:
    """Current-owner display values; diameter goals combine both directions."""
    owner, direct = session._owner(operator_id, direction)
    return tuple(item for item in all_reminder_progress(session)
                 if item.operator_id == owner and item.direction in (None, direct))


def all_reminder_progress(session: "ExperimentSession") -> tuple[ReminderProgress, ...]:
    """Use incremental aggregates; cost is bounded by owners and material slots."""
    policy = session.reminder_policy
    diameter = policy is not None and session.workspace_options["mode"] == "diameter"
    rows = []
    for owner, operator in session.operators.items():
        if diameter:
            count = sum(a.raw.count for (op, _, _), a in session._aggregates.items() if op == owner)
            rows.append(ReminderProgress(session.id, "diameter_measured", owner, None, None,
                                         count, policy.diameter_measured_target, operator.name))
            continue
        for direct in ("warp", "weft"):
            total = 0
            for material in operator.materials[direct]:
                aggregate = session._aggregates.get((owner, direct, material.id))
                count = aggregate.raw.count if aggregate is not None else 0
                total += count + (aggregate.extra_count if aggregate is not None else 0)
                target = policy.material_target(material.id) if policy else session.measured_reminder
                rows.append(ReminderProgress(session.id, "measured", owner, direct, material.id,
                                             count, target, operator.name, material.name, policy is None))
            target = policy.content_total_target if policy else session.total_reminder
            rows.append(ReminderProgress(session.id, "total", owner, direct, None,
                                         total, target, operator.name, legacy=policy is None))
    return tuple(rows)


def reminder_state(session: "ExperimentSession", *, operator_id: str | None = None,
                   direction: str | None = None) -> dict[str, Any]:
    """Compatibility dictionary plus explicit per-material and scope metadata."""
    progress = reminder_progress(session, operator_id=operator_id, direction=direction)
    measured_rows = [item for item in progress if item.kind == "measured"]
    total = next(item for item in progress if item.kind != "measured")
    policy = session.reminder_policy
    return {"measured": {item.material_id: item.count for item in measured_rows},
            "measured_targets": {item.material_id: item.target for item in measured_rows},
            "measured_target": policy.content_measured_target if policy else session.measured_reminder,
            "measured_reached": [item.material_id for item in measured_rows if item.reached],
            "total": total.count, "total_target": total.target, "total_reached": total.reached,
            "mode": session.workspace_options["mode"], "legacy": policy is None,
            "policy_version": policy.version if policy else None,
            "total_scope": "operator_measured" if total.kind == "diameter_measured" else "operator_direction_total"}


class ReminderTracker:
    """Notify once per stable scope/target during this open session.

    Construction/open, policy changes, and explicit ``seed`` never notify.
    Undo/redo are observed through the session's transient mutation origin and
    only update the baseline; already delivered targets stay acknowledged after
    deletion, reclassification, history replay, renaming, or reordering.
    """

    def __init__(self, session: "ExperimentSession") -> None:
        self._session_id: str | None = None
        self._seen: set[tuple[Any, ...]] = set()
        self._counts: dict[tuple[Any, ...], int] = {}
        self._policy: Any = None
        self.seed(session)

    @staticmethod
    def _policy_key(session: "ExperimentSession") -> tuple[Any, ...]:
        return (session.workspace_options["mode"], session.reminder_policy,
                session.measured_reminder if session.reminder_policy is None else None,
                session.total_reminder if session.reminder_policy is None else None)

    def seed(self, session: "ExperimentSession") -> None:
        if self._session_id != session.id:
            self._seen.clear()
        self._session_id = session.id
        self._policy = self._policy_key(session)
        progress = all_reminder_progress(session)
        self._counts = {item.key: item.count for item in progress}
        self._seen.update(item.key for item in progress if item.reached)

    def update(self, session: "ExperimentSession") -> ReminderNotice | None:
        if self._session_id != session.id or self._policy != self._policy_key(session):
            self.seed(session)
            return None
        progress = all_reminder_progress(session)
        history = session._reminder_mutation_origin in ("undo", "redo", "load")
        crossings = []
        for item in progress:
            if item.reached and self._counts.get(item.key, 0) < item.target and item.key not in self._seen:
                self._seen.add(item.key)
                if not history:
                    crossings.append(item)
        self._counts = {item.key: item.count for item in progress}
        return ReminderNotice(tuple(crossings)) if crossings else None
