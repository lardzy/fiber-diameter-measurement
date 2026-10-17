"""Independent, reversible live-experiment domain; no camera, Qt or Excel calls."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import math
from typing import Any, Callable
from uuid import uuid4

from .reminders import ReminderPolicy, normalize_policy


def _id() -> str:
    return uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是有限数值")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label}必须是{'正' if positive else ''}有限数值")
    return value


def _direction(value: str) -> str:
    if value not in ("warp", "weft"):
        raise ValueError("方向必须为 warp 或 weft")
    return value


def _workspace_options(value: Any, previous: dict[str, str] | None = None) -> dict[str, str]:
    if not isinstance(value, dict) or value.keys() - {"gesture", "mode", "classification"}:
        raise ValueError("工作区选项仅支持 gesture、mode 和 classification")
    result = {"gesture": "click", "mode": "content", "classification": "before", **(previous or {}), **value}
    if (result["gesture"] not in ("click", "drag") or result["mode"] not in ("content", "diameter")
            or result["classification"] not in ("before", "after")):
        raise ValueError("工作区操作方式或试验类型无效")
    return result


_FAMILIES = {"绵羊毛": "wool", "羊毛": "wool", "山羊绒": "cashmere", "羊绒": "cashmere",
             "棉": "cotton", "棉纤维": "cotton", "亚麻": "flax", "大麻": "hemp",
             "苎麻": "ramie", "罗布麻": "apocynum"}


@dataclass(frozen=True)
class MaterialSnapshot:
    id: str
    name: str
    density: float | None = None
    family: str = ""
    lower_limit: float | None = None
    upper_limit: float | None = None
    lower_suggestion_id: str | None = None
    upper_suggestion_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.name.strip():
            raise ValueError("材料必须有稳定 ID 和名称")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "family", self.family or _FAMILIES.get(self.name.strip(), "other"))
        if self.density is not None:
            value = _finite(self.density, "密度")
            if value < 0:
                raise ValueError("密度不能为负数")
            object.__setattr__(self, "density", value)
        for name in ("lower_limit", "upper_limit"):
            value = getattr(self, name)
            if value == -1:  # Accept an explicitly disabled CU limit, never a diameter.
                object.__setattr__(self, name, None)
            elif value is not None:
                object.__setattr__(self, name, _finite(value, "直径限值", positive=True))
        if self.lower_limit is not None and self.upper_limit is not None and self.lower_limit > self.upper_limit:
            raise ValueError("直径下限不能大于上限")
        object.__setattr__(self, "provenance", deepcopy(self.provenance))

    def range_status(self, diameter_um: float) -> str:
        value = _finite(diameter_um, "直径", positive=True)
        if self.upper_limit is not None and value > self.upper_limit:
            return "above"
        if self.lower_limit is not None and value < self.lower_limit:
            return "below"
        return "within"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MaterialSnapshot":
        return cls(**deepcopy(payload))


# Recovered definition data only; the two unknown trailing doubles are omitted.
# The first 37 names are shared by CU5/CU6/MT; the final three are MT additions.
_BUILT_IN_MATERIAL_ROWS = (
    ('wool', '绵羊毛', 1.31, 'wool', None, None, None, None),
    ('cashmere', '山羊绒', 1.3, 'cashmere', None, 30.0, None, 'goat_hair'),
    ('goat_hair', '山羊毛', 1.3, 'other', 30.0, None, 'cashmere_alias', None),
    ('alpaca', '羊驼毛', 1.3, 'other', None, None, None, None),
    ('rabbit', '兔毛', 1.1, 'other', None, 30.0, None, 'coarse_rabbit'),
    ('coarse_rabbit', '粗兔毛', 0.95, 'other', 30.0, None, 'rabbit', None),
    ('yak_down', '牦牛绒', 1.32, 'other', None, 35.0, None, 'yak_hair'),
    ('yak_hair', '牦牛毛', 1.32, 'other', 35.0, None, 'yak_down', None),
    ('camel_down', '驼绒', 1.31, 'other', None, 40.0, None, 'camel_hair'),
    ('camel_hair', '驼毛', 1.31, 'other', 40.0, None, 'camel_down', None),
    ('mohair', '马海毛', 1.32, 'other', None, None, None, None),
    ('mulberry_silk', '桑蚕丝', 1.36, 'other', None, None, None, None),
    ('cotton', '棉', 1.54, 'cotton', None, None, None, None),
    ('nylon', '锦纶', 1.14, 'other', None, None, None, None),
    ('acrylic', '腈纶', 1.18, 'other', None, None, None, None),
    ('polyester', '涤纶', 1.38, 'other', None, None, None, None),
    ('viscose', '粘胶', 1.51, 'other', None, None, None, None),
    ('flax', '亚麻', 1.5, 'flax', None, None, None, None),
    ('ramie', '苎麻', 1.51, 'ramie', None, None, None, None),
    ('apocynum', '罗布麻', 1.5, 'apocynum', None, None, None, None),
    ('hemp', '大麻', 1.48, 'hemp', None, None, None, None),
    ('tencel', '天丝', 1.51, 'other', None, None, None, None),
    ('polyethylene', '乙纶', 0.96, 'other', None, None, None, None),
    ('polypropylene', '丙纶', 0.91, 'other', None, None, None, None),
    ('diacetate', '二醋酯', 1.32, 'other', None, None, None, None),
    ('triacetate', '三醋酯', 1.3, 'other', None, None, None, None),
    ('cupro', '铜氨', 1.52, 'other', None, None, None, None),
    ('camel_down_alt', '骆驼绒', 1.31, 'other', None, 40.0, None, 'camel_hair_alt'),
    ('camel_hair_alt', '骆驼毛', 1.31, 'other', 40.0, None, 'camel_down_alt', None),
    ('coarse_cashmere', '粗山羊绒', 1.3, 'other', None, None, None, None),
    ('coarse_yak_down', '粗牦牛绒', 1.32, 'other', None, None, None, None),
    ('coarse_camel_down', '粗驼绒', 1.31, 'other', None, None, None, None),
    ('coarse_medullated_hair', '粗腔毛', 1.32, 'other', None, None, None, None),
    ('wool_alias', '羊毛', 1.31, 'wool', None, None, None, None),
    ('cashmere_alias', '羊绒', 1.3, 'cashmere', None, 30.0, None, 'goat_hair'),
    ('modal', '莫代尔纤维', 1.52, 'other', None, None, None, None),
    ('lyocell', '莱赛尔纤维', 1.52, 'other', None, None, None, None),
    ('brown_cotton', '棕色棉', 0.0, 'cotton', None, None, None, None),
    ('green_cotton', '绿色棉', 0.0, 'cotton', None, None, None, None),
    ('white_cotton', '白棉', 0.0, 'cotton', None, None, None, None),
)


def built_in_materials(profile: str = "all") -> tuple[MaterialSnapshot, ...]:
    """Return fresh editable snapshots, preserving recovered densities and limits.

    All 40 definitions are available by default. ``normal`` selects the shared
    37-definition library; ``mt`` selects all 40. A zero recovered density stays
    zero and prevents calculation until the operator confirms a positive value.
    These observations are not an independent certification of any standard.
    """
    if profile not in ("all", "normal", "cu5", "cu6", "mt", "cu6mt", "decolorization_free"):
        raise ValueError("未知内置类别库版本")
    rows = _BUILT_IN_MATERIAL_ROWS[:37] if profile in ("normal", "cu5", "cu6") else _BUILT_IN_MATERIAL_ROWS
    result = []
    for index, (id_, name, density, family, lower, upper, below, above) in enumerate(rows):
        result.append(MaterialSnapshot(id_, name, density, family, lower, upper, below, above,
            provenance={"source": "CU TYPES.LIST recovered definition", "library_revision": "cu-types-1",
                        "source_versions": ["cu5", "cu6", "cu6mt"] if index < 37 else ["cu6mt"],
                        "source_sha256": "743261d2cf246b6108c1db3ef85babc46402d3ae7578030ed43540ef537f6d50",
                        "source_record_offset": index * 176,
                        "verified_as_standard": False}))
    return tuple(result)


def default_materials() -> tuple[MaterialSnapshot, ...]:
    return built_in_materials("normal")[:2]


@dataclass
class Operator:
    id: str
    name: str
    warp_weight: float = 0.0
    weft_weight: float = 0.0
    materials: dict[str, tuple[MaterialSnapshot, ...]] = field(default_factory=lambda: {"warp": (), "weft": ()})

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "warp_weight": self.warp_weight,
                "weft_weight": self.weft_weight,
                "materials": {d: [m.to_dict() for m in self.materials[d]] for d in ("warp", "weft")}}


@dataclass(frozen=True)
class MeasurementRecord:
    id: str
    operator_id: str
    direction: str
    material_id: str
    diameter_um: float
    field_id: str = ""
    frame_id: int | str | None = None
    endpoints: tuple[tuple[float, float], tuple[float, float]] | None = None
    measurement_method: str = "assisted"
    evidence: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CountAdjustment:
    id: str
    operator_id: str
    direction: str
    material_id: str
    delta: int
    field_id: str = ""
    note: str = ""
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Moment:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    square_sum: float = 0.0

    def update(self, value: float, sign: int) -> None:
        n = self.count + sign
        if n <= 0:
            self.count, self.mean, self.m2, self.square_sum = 0, 0.0, 0.0, 0.0
            return
        delta = value - self.mean
        new_mean = self.mean + sign * delta / n
        self.m2 = max(0.0, self.m2 + sign * delta * (value - new_mean))
        self.mean = new_mean
        self.square_sum = math.fsum((self.square_sum, sign * value * value))
        self.count = n
        if n == 1:
            self.m2 = 0.0

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0


def _ramie_factor(diameter: float) -> float:
    return 0.3202 if diameter < 15 else 0.2652 if diameter < 40 else 0.221


@dataclass
class _Aggregate:
    raw: _Moment = field(default_factory=_Moment)
    grouped: _Moment = field(default_factory=_Moment)
    extra_count: int = 0
    count_events: int = 0
    bins: Counter[int] = field(default_factory=Counter)
    ramie_mid_square_sum: float = 0.0
    ramie_upper_square_sum: float = 0.0

    def measure(self, value: float, width: float, sign: int) -> None:
        bucket = math.floor(value / width)
        midpoint = (bucket + 0.5) * width
        upper = (bucket + 1) * width
        self.raw.update(value, sign)
        self.grouped.update(midpoint, sign)
        self.bins[bucket] += sign
        if not self.bins[bucket]:
            del self.bins[bucket]
        self.ramie_mid_square_sum = math.fsum((self.ramie_mid_square_sum, sign * midpoint * midpoint * _ramie_factor(midpoint)))
        self.ramie_upper_square_sum = math.fsum((self.ramie_upper_square_sum, sign * midpoint * midpoint * _ramie_factor(upper)))
        if not self.raw.count:
            self.ramie_mid_square_sum = self.ramie_upper_square_sum = 0.0


@dataclass(frozen=True)
class _Change:
    collection: str
    key: Any
    before: Any
    after: Any


@dataclass(frozen=True)
class AggregateRebuildRequest:
    """An immutable record snapshot, built off-thread and committed by its owner.

    Records are frozen objects. Only scalar identity/diameter/count fields are
    read, never their mutable provenance/evidence dictionaries or the session.
    """
    session_id: str
    revision: int
    action: str
    change: _Change
    group_width: float
    measurements: tuple[MeasurementRecord, ...]
    counts: tuple[CountAdjustment, ...]

    def build(self, cancelled: Callable[[], bool] = lambda: False) -> dict[tuple[str, str, str], _Aggregate]:
        aggregates: dict[tuple[str, str, str], _Aggregate] = {}
        for index, record in enumerate(self.measurements):
            if index % 256 == 0 and cancelled():
                raise CancelledError("组距重算已取消")
            value = ExperimentSession._validate_diameter(record.diameter_um, self.group_width)
            key = (record.operator_id, record.direction, record.material_id)
            aggregates.setdefault(key, _Aggregate()).measure(value, self.group_width, 1)
        for index, event in enumerate(self.counts):
            if index % 256 == 0 and cancelled():
                raise CancelledError("组距重算已取消")
            key = (event.operator_id, event.direction, event.material_id)
            aggregate = aggregates.setdefault(key, _Aggregate())
            aggregate.extra_count += event.delta
            aggregate.count_events += 1
        if cancelled():
            raise CancelledError("组距重算已取消")
        # A changed width can overflow cumulative squares even when each
        # diameter is individually representable. Never publish partial state.
        for aggregate in aggregates.values():
            if not all(math.isfinite(value) for value in (
                    aggregate.raw.square_sum, aggregate.grouped.square_sum,
                    aggregate.raw.m2, aggregate.grouped.m2,
                    aggregate.ramie_mid_square_sum, aggregate.ramie_upper_square_sum)):
                raise ValueError("直径累计超出可计算范围，未更改组距")
        return aggregates


class ExperimentSession:
    """One experiment, up to three operators, independently configured directions.

    Mutate through methods: aggregates update per record, and summaries are cached by
    revision. UI focus/active direction changes do not rewrite accepted records.
    """

    SCHEMA_VERSION = 1

    def __init__(self, *, id: str | None = None, name: str = "含量试验", sample_number: str = "",
                 sample_name: str = "", method: str = "GB/T 16988-2013", group_width: float = 2.5,
                 grouped: bool = True, profile: str = "normal", save_images: bool = False,
                 workspace_options: dict[str, str] | None = None,
                 reminder_policy: ReminderPolicy | dict[str, Any] | None = ReminderPolicy()) -> None:
        from .statistics import resolve_method, resolve_profile
        self.id = id or _id()
        self.name, self.sample_number, self.sample_name = name, sample_number, sample_name
        self.method = resolve_method(method).label
        self.group_width = _finite(group_width, "组距", positive=True)
        self.grouped, self.profile, self.save_images = bool(grouped), resolve_profile(profile), bool(save_images)
        self.workspace_options = _workspace_options({} if workspace_options is None else workspace_options)
        self.created_at = _now()
        self.operators: dict[str, Operator] = {}
        self.measurements: dict[str, MeasurementRecord] = {}
        self.count_adjustments: dict[str, CountAdjustment] = {}
        self.fields: dict[str, dict[str, Any]] = {}
        self.batches: dict[str, dict[str, Any]] = {}
        self.active_operator_id: str | None = None
        self.active_direction = "warp"
        self.measured_reminder = 300
        self.total_reminder = 1500
        self.reminder_policy = normalize_policy(reminder_policy)
        # Runtime history origin is not project content or notification history.
        self._reminder_mutation_origin = "load"
        self.revision = 0
        self._aggregates: dict[tuple[str, str, str], _Aggregate] = {}
        self._undo: list[_Change] = []
        self._redo: list[_Change] = []
        self._summary_cache: tuple[Any, dict[str, Any]] | None = None
        self._evidence_overrides: dict[str, dict[str, Any] | None] = {}
        self._known_measurement_ids: set[str] = set()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def _touch(self) -> None:
        self.revision += 1
        self._summary_cache = None

    def _owner(self, operator_id: str | None, direction: str | None) -> tuple[str, str]:
        op = operator_id or self.active_operator_id
        if op not in self.operators:
            raise ValueError("请先选择操作者")
        return op, _direction(direction or self.active_direction)

    def material(self, material_id: str, *, operator_id: str | None = None,
                 direction: str | None = None) -> MaterialSnapshot:
        op, direct = self._owner(operator_id, direction)
        for item in self.operators[op].materials[direct]:
            if item.id == material_id:
                return item
        raise ValueError("当前操作者和方向未配置此材料")

    def add_operator(self, name: str, *, id: str | None = None) -> str:
        name = name.strip()
        if not name:
            raise ValueError("操作者姓名不能为空")
        for operator in self.operators.values():
            if operator.name == name:
                self.active_operator_id = operator.id
                return operator.id
        if len(self.operators) >= 3:
            raise ValueError("一次试验最多包含三位操作者")
        key = id or _id()
        if key in self.operators:
            raise ValueError("操作者 ID 重复")
        first = next(iter(self.operators.values()), None)
        operator = Operator(key, name)
        if first is not None:
            operator.warp_weight, operator.weft_weight = first.warp_weight, first.weft_weight
            operator.materials = deepcopy(first.materials)
        self._commit(_Change("operators", key, None, operator))
        self.active_operator_id = key
        self.active_direction = "warp"
        return key

    def set_materials(self, materials: list[MaterialSnapshot] | tuple[MaterialSnapshot, ...], *,
                      operator_id: str | None = None, direction: str | None = None) -> None:
        op, direct = self._owner(operator_id, direction)
        items = tuple(deepcopy(materials))
        if len(items) > 9 or len({m.id for m in items}) != len(items):
            raise ValueError("每个方向最多九类，材料 ID 不能重复")
        if len({m.name for m in items}) != len(items):
            raise ValueError("同一方向材料名称不能重复")
        old = self.operators[op].materials[direct]
        for material in old:
            aggregate = self._aggregates.get((op, direct, material.id))
            if aggregate and (aggregate.raw.count or aggregate.extra_count or aggregate.count_events) and material.id not in {m.id for m in items}:
                raise ValueError("不能移除已有数据的材料，请先删除或迁移记录和计数")
        self._commit(_Change("materials", (op, direct), old, items))

    def set_weights(self, warp: float, weft: float, *, operator_id: str | None = None) -> None:
        op, _ = self._owner(operator_id, "warp")
        values = (_finite(warp, "经重"), _finite(weft, "纬重"))
        if min(values) < 0:
            raise ValueError("经纬重量不能为负数")
        old = (self.operators[op].warp_weight, self.operators[op].weft_weight)
        self._commit(_Change("weights", op, old, values))

    def rename_operator(self, name: str, *, operator_id: str | None = None) -> None:
        op, _ = self._owner(operator_id, "warp")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("操作者姓名不能为空")
        name = name.strip()
        if any(other.id != op and other.name == name for other in self.operators.values()):
            raise ValueError("操作者姓名不能重复")
        old = self.operators[op].name
        if name != old:
            self._commit(_Change("operator_name", op, old, name))

    def configure(self, **changes: Any) -> None:
        """Synchronous domain API; GUI callers must use configure_deferred."""
        request = self.configure_deferred(**changes)
        if request is not None:
            self.apply_rebuild(request, request.build())

    def configure_deferred(self, **changes: Any) -> AggregateRebuildRequest | None:
        """Apply small changes now, or return a side-effect-free width rebuild.

        The caller runs request.build() in a worker, then calls apply_rebuild on
        the session's owner thread. Old settings remain active until that commit.
        """
        from .statistics import resolve_method, resolve_profile
        allowed = {"method", "group_width", "grouped", "profile", "save_images", "measured_reminder", "total_reminder",
                   "name", "sample_name", "sample_number", "workspace_options", "reminder_policy"}
        if changes.keys() - allowed:
            raise ValueError("未知试验配置")
        new = dict(changes)
        if "workspace_options" in new:
            new["workspace_options"] = _workspace_options(new["workspace_options"], self.workspace_options)
        for key in ("name", "sample_name", "sample_number"):
            if key in new and not isinstance(new[key], str):
                raise ValueError("试验名称和样品信息必须为文本")
        if "method" in new:
            new["method"] = resolve_method(new["method"]).label
        if "profile" in new:
            new["profile"] = resolve_profile(new["profile"])
        if "group_width" in new:
            new["group_width"] = _finite(new["group_width"], "组距", positive=True)
        for key in ("measured_reminder", "total_reminder"):
            if key in new and (isinstance(new[key], bool) or not isinstance(new[key], int) or new[key] < 1):
                raise ValueError("提醒数量必须是正整数")
        if "reminder_policy" in new:
            new["reminder_policy"] = normalize_policy(new["reminder_policy"])
        elif self.reminder_policy is not None:
            # Keep callers of the earlier configuration API meaningful for new
            # content sessions, without migrating legacy projects implicitly.
            targets = {target: new[old] for old, target in (
                ("measured_reminder", "content_measured_target"),
                ("total_reminder", "content_total_target")) if old in new}
            if targets:
                new["reminder_policy"] = replace(self.reminder_policy, **targets)
        for key in ("grouped", "save_images"):
            if key in new and not isinstance(new[key], bool):
                raise ValueError("开关必须为布尔值")
        new = {key: value for key, value in new.items() if getattr(self, key) != value}
        if new:
            change = _Change("config", None, {k: getattr(self, k) for k in new}, new)
            request = self._prepare_rebuild(change, "configure")
            if request is not None:
                return request
            self._commit(change)
        return None

    def _prepare_rebuild(self, change: _Change, action: str) -> AggregateRebuildRequest | None:
        if change.collection != "config":
            return None
        values = change.before if action == "undo" else change.after
        width = values.get("group_width", self.group_width)
        if width == self.group_width or not (self.measurements or self.count_adjustments):
            return None
        return AggregateRebuildRequest(self.id, self.revision, action, change, width,
                                       tuple(self.measurements.values()), tuple(self.count_adjustments.values()))

    def history_deferred(self, action: str) -> AggregateRebuildRequest | None:
        """Execute ordinary undo/redo, returning width changes for a worker."""
        if action not in ("undo", "redo"):
            raise ValueError("历史操作必须为 undo 或 redo")
        stack = self._undo if action == "undo" else self._redo
        if stack:
            request = self._prepare_rebuild(stack[-1], action)
            if request is not None:
                return request
            getattr(self, action)()
        return None

    def apply_rebuild(self, request: AggregateRebuildRequest,
                      aggregates: dict[tuple[str, str, str], _Aggregate]) -> None:
        """Atomically publish a worker result only for the unchanged snapshot."""
        if request.session_id != self.id or request.revision != self.revision:
            raise ValueError("重算期间试验数据已变化，已保留当前记录和原设置；请重新提交组距设置。")
        action, change = request.action, request.change
        if action == "configure":
            self._apply(change, True, rebuilt=aggregates)
            self._undo.append(change)
            self._redo.clear()
        elif action in ("undo", "redo"):
            source, target = (self._undo, self._redo) if action == "undo" else (self._redo, self._undo)
            if not source or source[-1] is not change:
                raise ValueError("重算期间撤销历史已变化，未应用过期结果；请重试。")
            self._apply(change, action == "redo", rebuilt=aggregates)
            source.pop()
            target.append(change)
        else:
            raise ValueError("无效的组距重算操作")
        self._reminder_mutation_origin = action

    @staticmethod
    def _validate_diameter(value: float, width: float) -> float:
        value = _finite(value, "直径", positive=True)
        ratio = value / width
        if not math.isfinite(ratio) or not math.isfinite((value + width) * (value + width)):
            raise ValueError("直径或组距超出可计算范围")
        return value

    def _prepare_measurement(self, diameter_um: float, material_id: str, *, operator_id: str | None = None,
                        direction: str | None = None, field_id: str = "", frame_id: int | str | None = None,
                        endpoints: Any = None, evidence: dict[str, Any] | None = None,
                        provenance: dict[str, Any] | None = None, measurement_method: str = "assisted",
                        id: str | None = None) -> MeasurementRecord:
        op, direct = self._owner(operator_id, direction)
        self.material(material_id, operator_id=op, direction=direct)
        value = self._validate_diameter(diameter_um, self.group_width)
        points = None
        if endpoints is not None:
            if len(endpoints) != 2 or any(len(point) != 2 for point in endpoints):
                raise ValueError("测量端点必须为两个二维点")
            points = tuple(tuple(_finite(c, "端点") for c in point) for point in endpoints)
        record = MeasurementRecord(id or _id(), op, direct, material_id, value, str(field_id), frame_id,
                                   points, measurement_method, deepcopy(evidence), deepcopy(provenance or {}))
        if record.id in self._known_measurement_ids:
            raise ValueError("测量记录 ID 重复")
        return record

    def _preflight_measurements(self, records: list[MeasurementRecord], replacing: MeasurementRecord | None = None) -> None:
        """Reject arithmetic overflow before touching any record or accumulator."""
        deltas: dict[tuple[str, str, str], list[tuple[float, int]]] = {}
        if replacing is not None:
            deltas.setdefault((replacing.operator_id, replacing.direction, replacing.material_id), []).append((replacing.diameter_um, -1))
        for record in records:
            deltas.setdefault((record.operator_id, record.direction, record.material_id), []).append((record.diameter_um, 1))
        for key, values in deltas.items():
            aggregate = self._aggregates.get(key, _Aggregate())
            raw = [aggregate.raw.square_sum]
            grouped = [aggregate.grouped.square_sum]
            for value, sign in values:
                midpoint = (math.floor(value / self.group_width) + 0.5) * self.group_width
                raw.append(sign * value * value)
                grouped.append(sign * midpoint * midpoint)
            try:
                totals = (math.fsum(raw), math.fsum(grouped))
            except OverflowError as error:
                raise ValueError("直径累计超出可计算范围") from error
            if not all(math.isfinite(total) for total in totals):
                raise ValueError("直径累计超出可计算范围")

    def add_measurement(self, diameter_um: float, material_id: str, **kwargs: Any) -> MeasurementRecord:
        record = self._prepare_measurement(diameter_um, material_id, **kwargs)
        self._preflight_measurements([record])
        self._commit(_Change("measurements", record.id, None, record))
        self._known_measurement_ids.add(record.id)
        return record

    def add_measurements_batch(self, entries: list[dict[str, Any]], *,
                               provenance: dict[str, Any] | None = None) -> list[MeasurementRecord]:
        """Validate the entire batch, then publish it as one reversible operation."""
        if not entries:
            return []
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("批次来源必须为字典")
        batch_id = _id()
        records = []
        for index, entry in enumerate(entries):
            values = deepcopy(entry)
            source = values.pop("provenance", {}) or {}
            if not isinstance(source, dict):
                raise ValueError("测量来源必须为字典")
            source.update({"batch_id": batch_id, "batch_index": index, "batch_size": len(entries)})
            records.append(self._prepare_measurement(**values, provenance=source))
        if len({record.id for record in records}) != len(records):
            raise ValueError("批次中测量记录 ID 重复")
        self._preflight_measurements(records)
        metadata = {"id": batch_id, "record_ids": [record.id for record in records],
                    "created_at": _now(), "provenance": deepcopy(provenance or {})}
        changes = tuple([_Change("measurements", record.id, None, record) for record in records]
                        + [_Change("batches", batch_id, None, metadata)])
        self._commit(_Change("batch", batch_id, None, changes))
        self._known_measurement_ids.update(record.id for record in records)
        return records

    def set_evidence(self, record_id: str, evidence: dict[str, Any] | None) -> bool:
        """Publish asynchronous image status without adding a measurement undo step.

        A completed write for an undone record remains available if it is redone.
        Unknown IDs are ignored, so an expired callback cannot create a record.
        """
        if record_id not in self._known_measurement_ids:
            return False
        value = deepcopy(evidence)
        self._evidence_overrides[record_id] = value
        if record_id in self.measurements:
            self.measurements[record_id] = replace(self.measurements[record_id], evidence=value)
        self._touch()
        return True

    def update_field(self, field_id: str, **metadata: Any) -> None:
        """Persist accepted-field lifecycle independently of measurement undo."""
        if not isinstance(field_id, str) or not field_id:
            raise ValueError("视场 ID 不能为空")
        value = {**deepcopy(self.fields.get(field_id, {})), **deepcopy(metadata), "id": field_id}
        if value != self.fields.get(field_id):
            self.fields[field_id] = value
            self._touch()

    def edit_measurement(self, record_id: str, *, diameter_um: float | None = None,
                         material_id: str | None = None, evidence: dict[str, Any] | None = None,
                         provenance: dict[str, Any] | None = None) -> MeasurementRecord:
        old = self.measurements[record_id]
        selected = material_id if material_id is not None else old.material_id
        self.material(selected, operator_id=old.operator_id, direction=old.direction)
        value = old.diameter_um if diameter_um is None else self._validate_diameter(diameter_um, self.group_width)
        new = replace(old, diameter_um=value, material_id=selected,
                      evidence=deepcopy(evidence) if evidence is not None else old.evidence,
                      provenance=deepcopy(provenance) if provenance is not None else old.provenance)
        self._preflight_measurements([new], replacing=old)
        self._commit(_Change("measurements", old.id, old, new))
        return new

    def delete_measurement(self, record_id: str) -> None:
        self._commit(_Change("measurements", record_id, self.measurements[record_id], None))

    def adjust_count(self, material_id: str, delta: int, *, operator_id: str | None = None,
                     direction: str | None = None, field_id: str = "", note: str = "") -> CountAdjustment:
        op, direct = self._owner(operator_id, direction)
        self.material(material_id, operator_id=op, direction=direct)
        if isinstance(delta, bool) or not isinstance(delta, int) or not delta:
            raise ValueError("计数变化必须为非零整数")
        current = self._aggregates.get((op, direct, material_id), _Aggregate()).extra_count
        if current + delta < 0:
            raise ValueError("额外计数不能小于零，实测记录不受计数减号影响")
        event = CountAdjustment(_id(), op, direct, material_id, delta, field_id, note)
        self._commit(_Change("count_adjustments", event.id, None, event))
        return event

    def _record_delta(self, record: MeasurementRecord | CountAdjustment, sign: int) -> None:
        key = (record.operator_id, record.direction, record.material_id)
        aggregate = self._aggregates.setdefault(key, _Aggregate())
        if isinstance(record, MeasurementRecord):
            aggregate.measure(record.diameter_um, self.group_width, sign)
        else:
            aggregate.extra_count += sign * record.delta
            aggregate.count_events += sign

    def _apply(self, change: _Change, forward: bool, *, touch: bool = True,
               rebuilt: dict[tuple[str, str, str], _Aggregate] | None = None) -> None:
        old, new = (change.before, change.after) if forward else (change.after, change.before)
        name, key = change.collection, change.key
        if name == "batch":
            changes = change.after if forward else reversed(change.after)
            for child in changes:
                self._apply(child, forward, touch=False)
        elif name == "batches":
            if new is None:
                self.batches.pop(key, None)
            else:
                self.batches[key] = deepcopy(new)
        elif name in ("measurements", "count_adjustments"):
            if old is not None:
                self._record_delta(old, -1)
            if new is not None:
                self._record_delta(new, 1)
            collection = getattr(self, name)
            if new is None:
                collection.pop(key, None)
            else:
                if name == "measurements" and key in self._evidence_overrides:
                    new = replace(new, evidence=deepcopy(self._evidence_overrides[key]))
                collection[key] = new
        elif name == "operators":
            if new is None:
                self.operators.pop(key, None)
                if self.active_operator_id == key:
                    self.active_operator_id = next(iter(self.operators), None)
            else:
                self.operators[key] = deepcopy(new)
        elif name == "materials":
            self.operators[key[0]].materials[key[1]] = tuple(new)
        elif name == "operator_name":
            self.operators[key].name = new
        elif name == "weights":
            self.operators[key].warp_weight, self.operators[key].weft_weight = new
        elif name == "config":
            rebuild = new.get("group_width", self.group_width) != self.group_width
            for setting, value in new.items():
                setattr(self, setting, deepcopy(value))
            if rebuild:
                if rebuilt is not None:
                    self._aggregates = rebuilt
                else:
                    self._rebuild_aggregates()
        if touch:
            self._touch()

    def _commit(self, change: _Change) -> None:
        self._apply(change, True)
        self._undo.append(change)
        self._redo.clear()
        self._reminder_mutation_origin = "commit"

    def undo(self) -> bool:
        if not self._undo:
            return False
        change = self._undo.pop()
        self._apply(change, False)
        self._redo.append(change)
        self._reminder_mutation_origin = "undo"
        return True

    def establish_history_baseline(self) -> None:
        """Keep initial defaults while starting a fresh user-edit history.

        Call only after creating a new experiment's initial operator/materials;
        ongoing experiments retain their complete undo and redo history.
        """
        self._undo.clear()
        self._redo.clear()

    def redo(self) -> bool:
        if not self._redo:
            return False
        change = self._redo.pop()
        self._apply(change, True)
        self._undo.append(change)
        self._reminder_mutation_origin = "redo"
        return True

    def _rebuild_aggregates(self) -> None:
        self._aggregates.clear()
        for record in self.measurements.values():
            self._record_delta(record, 1)
        for event in self.count_adjustments.values():
            self._record_delta(event, 1)

    def reminder_state(self, *, operator_id: str | None = None, direction: str | None = None) -> dict[str, Any]:
        from .reminders import reminder_state
        return reminder_state(self, operator_id=operator_id, direction=direction)

    def to_dict(self) -> dict[str, Any]:
        payload = {"schema_version": self.SCHEMA_VERSION, "id": self.id, "name": self.name,
                "sample_number": self.sample_number, "sample_name": self.sample_name,
                "method": self.method, "group_width": self.group_width, "grouped": self.grouped,
                "profile": self.profile, "save_images": self.save_images, "created_at": self.created_at,
                "workspace_options": dict(self.workspace_options),
                "active_operator_id": self.active_operator_id, "active_direction": self.active_direction,
                "measured_reminder": self.measured_reminder, "total_reminder": self.total_reminder,
                "revision": self.revision, "operators": [op.to_dict() for op in self.operators.values()],
                "measurements": [r.to_dict() for r in self.measurements.values()],
                "count_adjustments": [r.to_dict() for r in self.count_adjustments.values()],
                "fields": deepcopy(self.fields), "batches": deepcopy(self.batches)}
        if self.reminder_policy is not None:
            payload["reminder_policy"] = self.reminder_policy.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentSession":
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("不支持的实时试验格式版本")
        session = cls(reminder_policy=payload.get("reminder_policy"),
                      **{key: payload[key] for key in ("id", "name", "sample_number", "sample_name", "method",
                        "group_width", "grouped", "profile", "save_images", "workspace_options") if key in payload})
        for item in payload.get("operators", []):
            if item["id"] in session.operators or any(op.name == item["name"] for op in session.operators.values()):
                raise ValueError("重复操作者")
            key = session.add_operator(item["name"], id=item["id"])
            session.set_weights(item.get("warp_weight", 0), item.get("weft_weight", 0), operator_id=key)
            for direct in ("warp", "weft"):
                session.set_materials([MaterialSnapshot.from_dict(m) for m in item["materials"][direct]],
                                      operator_id=key, direction=direct)
        for item in payload.get("measurements", []):
            values = deepcopy(item)
            created = values.pop("created_at", _now())
            record = session.add_measurement(**values)
            session.measurements[record.id] = replace(record, created_at=created)
        for item in payload.get("count_adjustments", []):
            event = CountAdjustment(**deepcopy(item))
            if event.id in session.count_adjustments:
                raise ValueError("重复计数事件")
            session.material(event.material_id, operator_id=event.operator_id, direction=event.direction)
            if isinstance(event.delta, bool) or not isinstance(event.delta, int) or not event.delta:
                raise ValueError("无效的计数事件")
            aggregate = session._aggregates.get((event.operator_id, event.direction, event.material_id), _Aggregate())
            if aggregate.extra_count + event.delta < 0:
                raise ValueError("计数事件导致额外计数为负")
            session.count_adjustments[event.id] = event
            session._record_delta(event, 1)
        active = payload.get("active_operator_id")
        if active is not None and active not in session.operators:
            raise ValueError("活动操作者不存在")
        session.active_operator_id = active
        session.active_direction = _direction(payload.get("active_direction", "warp"))
        session.created_at = str(payload.get("created_at", session.created_at))
        session.fields = deepcopy(payload.get("fields", {}))
        session.batches = deepcopy(payload.get("batches", {}))
        session.configure(measured_reminder=payload.get("measured_reminder", 300),
                          total_reminder=payload.get("total_reminder", 1500),
                          reminder_policy=session.reminder_policy)
        session.revision = max(0, int(payload.get("revision", 0)))
        session._undo.clear()
        session._redo.clear()
        session._summary_cache = None
        session._reminder_mutation_origin = "load"
        return session
