from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from fdm.geometry import Line, Point
from fdm.models import new_id, utc_now_iso


CONTENT_EXPERIMENT_METADATA_KEY = "content_experiment"
MAX_CONTENT_FIBERS = 8


class ContentRecordKind:
    COUNT = "count"
    DIAMETER = "diameter"


class ContentSelectionMode:
    PRESELECT = "preselect"
    POSTSELECT = "postselect"


class ContentOverlayStyle:
    NONE = "none"
    CENTER_DOT = "center_dot"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    CROSS = "cross"
    CROSSHAIR = "crosshair"


@dataclass(slots=True)
class ContentFiberDefinition:
    id: str
    name: str
    color: str = "#1F7A8C"
    builtin: bool = False
    diameter_min: float | None = None
    diameter_max: float | None = None
    density: float | None = None

    def clone(self, **changes) -> "ContentFiberDefinition":
        return replace(self, **changes)

    def normalized_name(self) -> str:
        return str(self.name or "").strip()

    def resolved_density(self) -> float:
        return float(self.density) if self.density is not None and self.density > 0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.normalized_name(),
            "color": self.color,
            "builtin": bool(self.builtin),
            "diameter_min": self.diameter_min,
            "diameter_max": self.diameter_max,
            "density": self.density,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContentFiberDefinition":
        return cls(
            id=str(payload.get("id") or new_id("content_fiber")),
            name=str(payload.get("name", "")).strip(),
            color=str(payload.get("color", "#1F7A8C")),
            builtin=bool(payload.get("builtin", False)),
            diameter_min=_optional_float(payload.get("diameter_min")),
            diameter_max=_optional_float(payload.get("diameter_max")),
            density=_optional_float(payload.get("density")),
        )


@dataclass(slots=True)
class ContentExperimentRecord:
    id: str
    kind: str
    fiber_id: str
    field_id: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    line_px: Line | None = None
    diameter_px: float | None = None
    diameter_unit: float | None = None

    def display_value(self) -> str:
        if self.kind == ContentRecordKind.COUNT:
            return "1"
        value = self.diameter_unit if self.diameter_unit is not None else self.diameter_px
        if value is None:
            return "-"
        return f"{value:.4f}".rstrip("0").rstrip(".")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "fiber_id": self.fiber_id,
            "field_id": self.field_id,
            "created_at": self.created_at,
            "line_px": self.line_px.to_dict() if self.line_px else None,
            "diameter_px": self.diameter_px,
            "diameter_unit": self.diameter_unit,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContentExperimentRecord":
        line_payload = payload.get("line_px")
        kind = str(payload.get("kind", ContentRecordKind.COUNT)).strip()
        if kind not in {ContentRecordKind.COUNT, ContentRecordKind.DIAMETER}:
            kind = ContentRecordKind.COUNT
        return cls(
            id=str(payload.get("id") or new_id("content_rec")),
            kind=kind,
            fiber_id=str(payload.get("fiber_id", "")),
            field_id=int(payload.get("field_id", 0) or 0),
            created_at=str(payload.get("created_at", utc_now_iso())),
            line_px=Line.from_dict(line_payload) if isinstance(line_payload, dict) else None,
            diameter_px=_optional_float(payload.get("diameter_px")),
            diameter_unit=_optional_float(payload.get("diameter_unit")),
        )


@dataclass(slots=True)
class ContentFiberStats:
    fiber: ContentFiberDefinition
    count: int = 0
    measured: int = 0
    average_diameter: float | None = None
    mean_diameter_squared: float | None = None
    content_percent: float | None = None

    @property
    def total_roots(self) -> int:
        return self.count + self.measured


@dataclass(slots=True)
class ContentExperimentSession:
    id: str = field(default_factory=lambda: new_id("content_session"))
    active: bool = False
    operator: str = ""
    sample_id: str = ""
    sample_name: str = ""
    selection_mode: str = ContentSelectionMode.PRESELECT
    overlay_style: str = ContentOverlayStyle.NONE
    current_fiber_id: str | None = None
    fibers: list[ContentFiberDefinition] = field(default_factory=list)
    records: list[ContentExperimentRecord] = field(default_factory=list)
    current_field_id: int = 0
    workbook_snapshot_relpath: str = ""
    workbook_mode: str = ""
    reminders_triggered: list[str] = field(default_factory=list)

    def active_fiber(self) -> ContentFiberDefinition | None:
        return self.fiber_by_id(self.current_fiber_id)

    def fiber_by_id(self, fiber_id: str | None) -> ContentFiberDefinition | None:
        if not fiber_id:
            return None
        for fiber in self.fibers:
            if fiber.id == fiber_id:
                return fiber
        return None

    def ensure_current_fiber(self) -> ContentFiberDefinition | None:
        if self.current_fiber_id and self.fiber_by_id(self.current_fiber_id) is not None:
            return self.fiber_by_id(self.current_fiber_id)
        self.current_fiber_id = self.fibers[0].id if self.fibers else None
        return self.active_fiber()

    def set_fibers(self, fibers: list[ContentFiberDefinition]) -> None:
        self.fibers = [fiber.clone() for fiber in fibers[:MAX_CONTENT_FIBERS] if fiber.normalized_name()]
        if self.current_fiber_id is None or self.fiber_by_id(self.current_fiber_id) is None:
            self.current_fiber_id = self.fibers[0].id if self.fibers else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "id": self.id,
            "active": self.active,
            "operator": self.operator,
            "sample_id": self.sample_id,
            "sample_name": self.sample_name,
            "selection_mode": self.selection_mode,
            "overlay_style": self.overlay_style,
            "current_fiber_id": self.current_fiber_id,
            "fibers": [fiber.to_dict() for fiber in self.fibers],
            "records": [record.to_dict() for record in self.records],
            "current_field_id": self.current_field_id,
            "workbook_snapshot_relpath": self.workbook_snapshot_relpath,
            "workbook_mode": self.workbook_mode,
            "reminders_triggered": list(self.reminders_triggered),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContentExperimentSession":
        session = cls(
            id=str(payload.get("id") or new_id("content_session")),
            active=bool(payload.get("active", False)),
            operator=str(payload.get("operator", "")),
            sample_id=str(payload.get("sample_id", "")),
            sample_name=str(payload.get("sample_name", "")),
            selection_mode=normalize_selection_mode(payload.get("selection_mode")),
            overlay_style=normalize_overlay_style(payload.get("overlay_style")),
            current_fiber_id=payload.get("current_fiber_id"),
            fibers=[
                ContentFiberDefinition.from_dict(item)
                for item in payload.get("fibers", [])
                if isinstance(item, dict)
            ],
            records=[
                ContentExperimentRecord.from_dict(item)
                for item in payload.get("records", [])
                if isinstance(item, dict)
            ],
            current_field_id=int(payload.get("current_field_id", 0) or 0),
            workbook_snapshot_relpath=str(payload.get("workbook_snapshot_relpath", "")),
            workbook_mode=str(payload.get("workbook_mode", "")),
            reminders_triggered=[
                str(item)
                for item in payload.get("reminders_triggered", [])
                if str(item).strip()
            ],
        )
        session.records = [record for record in session.records if session.fiber_by_id(record.fiber_id) is not None]
        session.ensure_current_fiber()
        return session


def default_content_fiber_definitions() -> list[ContentFiberDefinition]:
    """CU-5 默认类型库。密度值来自参考模板中的“比重”行。"""

    rows: list[tuple[str, float | None, float | None, float | None]] = [
        ("绵羊毛", None, None, 1.31),
        ("山羊绒", None, 30.0, 1.30),
        ("山羊毛", 30.0, None, 1.30),
        ("羊驼毛", None, None, 1.30),
        ("兔毛", None, 30.0, 1.10),
        ("粗兔毛", 30.0, None, 0.95),
        ("牦牛绒", None, 35.0, 1.32),
        ("牦牛毛", 35.0, None, 1.32),
        ("驼绒", None, 40.0, 1.31),
        ("驼毛", 40.0, None, 1.31),
        ("马海毛", None, None, 1.32),
        ("桑蚕丝", None, None, 1.36),
        ("棉", None, None, 1.54),
        ("锦纶", None, None, 1.14),
        ("腈纶", None, None, 1.18),
        ("涤纶", None, None, 1.38),
        ("粘胶", None, None, 1.51),
        ("亚麻", None, None, 1.50),
        ("苎麻", None, None, 1.51),
        ("罗布麻", None, None, 1.50),
        ("大麻", None, None, 1.48),
        ("天丝", None, None, 1.51),
        ("乙纶", None, None, 0.96),
        ("丙纶", None, None, 0.91),
        ("二醋酯", None, None, 1.32),
        ("三醋酯", None, None, 1.30),
        ("铜氨", None, None, 1.52),
        ("骆驼绒", None, 40.0, 1.31),
        ("骆驼毛", 40.0, None, 1.31),
        ("粗山羊绒", None, None, 1.30),
        ("粗牦牛绒", None, None, 1.32),
        ("粗驼绒", None, None, 1.31),
        ("粗腔毛", None, None, 1.32),
        ("羊毛", None, None, 1.31),
        ("羊绒", None, 30.0, 1.30),
        ("莫代尔纤维", None, None, 1.52),
        ("莱赛尔纤维", None, None, 1.52),
    ]
    colors = [
        "#1F7A8C",
        "#E07A5F",
        "#81B29A",
        "#3D405B",
        "#F2CC8F",
        "#6D597A",
        "#227C9D",
        "#FF7C43",
    ]
    return [
        ContentFiberDefinition(
            id=f"builtin_{index:02d}",
            name=name,
            color=colors[(index - 1) % len(colors)],
            builtin=True,
            diameter_min=diameter_min,
            diameter_max=diameter_max,
            density=density,
        )
        for index, (name, diameter_min, diameter_max, density) in enumerate(rows, start=1)
    ]


def normalize_selection_mode(value: object) -> str:
    token = str(value or "").strip()
    if token in {ContentSelectionMode.PRESELECT, ContentSelectionMode.POSTSELECT}:
        return token
    return ContentSelectionMode.PRESELECT


def normalize_overlay_style(value: object) -> str:
    token = str(value or "").strip()
    if token in {
        ContentOverlayStyle.NONE,
        ContentOverlayStyle.CENTER_DOT,
        ContentOverlayStyle.HORIZONTAL,
        ContentOverlayStyle.VERTICAL,
        ContentOverlayStyle.CROSS,
        ContentOverlayStyle.CROSSHAIR,
    }:
        return token
    return ContentOverlayStyle.NONE


def normalized_content_fiber_definitions(
    definitions: list[ContentFiberDefinition] | None,
) -> list[ContentFiberDefinition]:
    builtins = default_content_fiber_definitions()
    builtin_by_id = {fiber.id: fiber for fiber in builtins}
    normalized: list[ContentFiberDefinition] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()

    for fiber in definitions or []:
        if not fiber.normalized_name():
            continue
        if fiber.id in seen_ids:
            continue
        if fiber.builtin and fiber.id in builtin_by_id:
            base = builtin_by_id[fiber.id]
            merged = base.clone(
                color=fiber.color or base.color,
                diameter_min=fiber.diameter_min if fiber.diameter_min is not None else base.diameter_min,
                diameter_max=fiber.diameter_max if fiber.diameter_max is not None else base.diameter_max,
                density=fiber.density if fiber.density is not None else base.density,
            )
            normalized.append(merged)
            seen_ids.add(merged.id)
            seen_names.add(merged.normalized_name())
            continue
        name = fiber.normalized_name()
        if name in seen_names:
            continue
        normalized.append(fiber.clone(id=fiber.id or new_id("content_fiber"), name=name, builtin=False))
        seen_ids.add(normalized[-1].id)
        seen_names.add(name)

    for fiber in builtins:
        if fiber.id not in seen_ids and fiber.normalized_name() not in seen_names:
            normalized.append(fiber)
            seen_ids.add(fiber.id)
            seen_names.add(fiber.normalized_name())
    return normalized


def session_from_project_metadata(metadata: dict[str, Any]) -> ContentExperimentSession | None:
    payload = metadata.get(CONTENT_EXPERIMENT_METADATA_KEY)
    if not isinstance(payload, dict):
        return None
    return ContentExperimentSession.from_dict(payload)


def write_session_to_project_metadata(metadata: dict[str, Any], session: ContentExperimentSession | None) -> None:
    if session is None:
        metadata.pop(CONTENT_EXPERIMENT_METADATA_KEY, None)
        return
    metadata[CONTENT_EXPERIMENT_METADATA_KEY] = session.to_dict()


def content_session_stats(session: ContentExperimentSession) -> list[ContentFiberStats]:
    stats_by_id = {
        fiber.id: ContentFiberStats(fiber=fiber)
        for fiber in session.fibers
    }
    diameter_values: dict[str, list[float]] = {fiber.id: [] for fiber in session.fibers}
    for record in session.records:
        stats = stats_by_id.get(record.fiber_id)
        if stats is None:
            continue
        if record.kind == ContentRecordKind.COUNT:
            stats.count += 1
            continue
        if record.kind == ContentRecordKind.DIAMETER:
            value = record.diameter_unit if record.diameter_unit is not None else record.diameter_px
            if value is None:
                continue
            stats.measured += 1
            diameter_values.setdefault(record.fiber_id, []).append(float(value))

    mass_by_id: dict[str, float] = {}
    for fiber_id, stats in stats_by_id.items():
        values = diameter_values.get(fiber_id, [])
        if values:
            stats.average_diameter = sum(values) / len(values)
            stats.mean_diameter_squared = sum(value * value for value in values) / len(values)
        if stats.mean_diameter_squared is not None and stats.total_roots > 0:
            mass_by_id[fiber_id] = stats.total_roots * stats.mean_diameter_squared * stats.fiber.resolved_density()

    total_mass = sum(mass_by_id.values())
    if total_mass > 0:
        for fiber_id, mass in mass_by_id.items():
            stats_by_id[fiber_id].content_percent = (mass / total_mass) * 100.0
    return [stats_by_id[fiber.id] for fiber in session.fibers]


def content_total_count(session: ContentExperimentSession) -> int:
    return sum(1 for record in session.records if record.kind == ContentRecordKind.COUNT)


def content_total_measured(session: ContentExperimentSession) -> int:
    return sum(1 for record in session.records if record.kind == ContentRecordKind.DIAMETER)


def next_content_color(index: int) -> str:
    palette = [
        "#1F7A8C",
        "#E07A5F",
        "#81B29A",
        "#3D405B",
        "#F2CC8F",
        "#6D597A",
        "#227C9D",
        "#FF7C43",
    ]
    return palette[index % len(palette)]


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
