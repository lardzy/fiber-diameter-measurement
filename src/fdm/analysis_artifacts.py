"""Immutable, versioned records for analysis results.

Analysis artifacts are intentionally separate from measurement records.
Histograms, profiles, texture tables, masks and graph results therefore do not
pretend to be length/area/count measurements.  Source revisions make staleness
explicit while retaining the original result for audit and export.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from typing import TypeAlias


ANALYSIS_ARTIFACT_SCHEMA_VERSION = 1
_ID_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_TOOL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ASSET_SCHEMA_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UNSAFE_SERIALIZED_ASSET_SUFFIXES = frozenset({".npy", ".pickle", ".pkl"})
_MAX_INLINE_TABLE_CELLS = 100_000
_MAX_INLINE_CURVE_POINTS = 100_000
_MAX_INLINE_JSON_BYTES = 1_048_576
_UNSET = object()

JsonScalar: TypeAlias = str | int | float | bool | None


class AnalysisArtifactStatus(StrEnum):
    CURRENT = "current"
    STALE = "stale"


class AnalysisObjectKind(StrEnum):
    ROI = "roi"
    MEASUREMENT = "measurement"


class AnalysisAssetKind(StrEnum):
    TABLE = "table"
    CURVE = "curve"
    MASK = "mask"
    LABEL_IMAGE = "label_image"
    GRAPH = "graph"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class AnalysisObjectReference:
    kind: AnalysisObjectKind
    object_id: str
    revision: int

    def __post_init__(self) -> None:
        try:
            kind = AnalysisObjectKind(self.kind)
        except (TypeError, ValueError) as error:
            raise ValueError(f"不支持的分析对象引用类型: {self.kind!r}") from error
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "object_id",
            _required_id(self.object_id, field_name="object_id"),
        )
        object.__setattr__(
            self,
            "revision",
            _non_negative_int(self.revision, field_name="revision"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "object_id": self.object_id,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "AnalysisObjectReference":
        _require_mapping(payload, field_name="source_reference")
        _require_exact_keys(
            payload,
            required={"kind", "object_id", "revision"},
            field_name="source_reference",
        )
        return cls(
            kind=payload["kind"],  # type: ignore[arg-type]
            object_id=payload["object_id"],  # type: ignore[arg-type]
            revision=payload["revision"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AnalysisTable:
    name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[JsonScalar, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _required_text(self.name, field_name="table.name", maximum_length=256),
        )
        columns = tuple(
            _required_text(
                column,
                field_name=f"table.columns[{index}]",
                maximum_length=256,
            )
            for index, column in enumerate(self.columns)
        )
        if not columns:
            raise ValueError("table.columns 不能为空")
        if len(set(columns)) != len(columns):
            raise ValueError("table.columns 不能包含重复列名")
        rows: list[tuple[JsonScalar, ...]] = []
        for row_index, row in enumerate(self.rows):
            frozen_row = tuple(
                _normalize_json_scalar(
                    value,
                    field_name=f"table.rows[{row_index}][{column_index}]",
                )
                for column_index, value in enumerate(row)
            )
            if len(frozen_row) != len(columns):
                raise ValueError(
                    f"table.rows[{row_index}] 的列数必须为 {len(columns)}"
                )
            rows.append(frozen_row)
        if len(columns) * len(rows) > _MAX_INLINE_TABLE_CELLS:
            raise ValueError(
                "table 过大，必须写入 analysis/ 下的安全资产并通过 assets 引用"
            )
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rows", tuple(rows))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AnalysisTable":
        _require_mapping(payload, field_name="table")
        _require_exact_keys(
            payload,
            required={"name", "columns", "rows"},
            field_name="table",
        )
        columns = payload["columns"]
        rows = payload["rows"]
        if not isinstance(columns, list):
            raise TypeError("table.columns 必须是列表")
        if not isinstance(rows, list) or any(
            not isinstance(row, list) for row in rows
        ):
            raise TypeError("table.rows 必须是二维列表")
        return cls(
            name=payload["name"],  # type: ignore[arg-type]
            columns=tuple(columns),  # type: ignore[arg-type]
            rows=tuple(tuple(row) for row in rows),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AnalysisCurve:
    name: str
    x: tuple[float, ...]
    y: tuple[float | None, ...]
    x_unit: str = ""
    y_unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _required_text(self.name, field_name="curve.name", maximum_length=256),
        )
        x = tuple(
            _finite_number(value, field_name=f"curve.x[{index}]")
            for index, value in enumerate(self.x)
        )
        y = tuple(
            None
            if value is None
            else _finite_number(value, field_name=f"curve.y[{index}]")
            for index, value in enumerate(self.y)
        )
        if not x:
            raise ValueError("curve.x 不能为空")
        if len(x) != len(y):
            raise ValueError("curve.x 与 curve.y 的长度必须一致")
        if len(x) > _MAX_INLINE_CURVE_POINTS:
            raise ValueError(
                "curve 过大，必须写入 analysis/ 下的安全资产并通过 assets 引用"
            )
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(
            self,
            "x_unit",
            _optional_text(self.x_unit, field_name="curve.x_unit", maximum_length=64),
        )
        object.__setattr__(
            self,
            "y_unit",
            _optional_text(self.y_unit, field_name="curve.y_unit", maximum_length=64),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "x": list(self.x),
            "y": list(self.y),
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AnalysisCurve":
        _require_mapping(payload, field_name="curve")
        _require_exact_keys(
            payload,
            required={"name", "x", "y", "x_unit", "y_unit"},
            field_name="curve",
        )
        x = payload["x"]
        y = payload["y"]
        if not isinstance(x, list) or not isinstance(y, list):
            raise TypeError("curve.x 和 curve.y 必须是列表")
        return cls(
            name=payload["name"],  # type: ignore[arg-type]
            x=tuple(x),  # type: ignore[arg-type]
            y=tuple(y),  # type: ignore[arg-type]
            x_unit=payload["x_unit"],  # type: ignore[arg-type]
            y_unit=payload["y_unit"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True, init=False)
class AnalysisAssetReference:
    kind: AnalysisAssetKind
    path: str
    sha256: str
    media_type: str
    _metadata_json: str = field(repr=False)

    def __init__(
        self,
        *,
        kind: AnalysisAssetKind | str,
        path: str,
        sha256: str,
        media_type: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        try:
            normalized_kind = AnalysisAssetKind(kind)
        except (TypeError, ValueError) as error:
            raise ValueError(f"不支持的分析资产类型: {kind!r}") from error
        normalized_path = _relative_asset_path(path)
        normalized_sha = str(sha256 or "").strip().lower()
        if not _SHA256_PATTERN.fullmatch(normalized_sha):
            raise ValueError("asset.sha256 必须是 64 位小写十六进制")
        normalized_media_type = _required_text(
            media_type,
            field_name="asset.media_type",
            maximum_length=128,
        )
        _validate_safe_asset_encoding(
            path=normalized_path,
            media_type=normalized_media_type,
        )
        metadata_json = _canonical_json_object(
            metadata or {},
            field_name="asset.metadata",
        )
        metadata_payload = json.loads(metadata_json)
        schema = metadata_payload.get("schema")
        if not isinstance(schema, str) or not _ASSET_SCHEMA_PATTERN.fullmatch(schema):
            raise ValueError(
                "asset.metadata.schema 必须是安全、版本化的非空 schema 标识"
            )
        if metadata_payload.get("allow_pickle") not in (None, False):
            raise ValueError("分析资产禁止启用 pickle")
        dtype = str(metadata_payload.get("dtype", "") or "").strip().lower()
        if dtype in {"object", "object_", "o"}:
            raise ValueError("分析资产禁止使用 object dtype")
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "path", normalized_path)
        object.__setattr__(self, "sha256", normalized_sha)
        object.__setattr__(self, "media_type", normalized_media_type)
        object.__setattr__(self, "_metadata_json", metadata_json)

    @property
    def metadata(self) -> dict[str, object]:
        return json.loads(self._metadata_json)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "path": self.path,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "AnalysisAssetReference":
        _require_mapping(payload, field_name="asset")
        _require_exact_keys(
            payload,
            required={"kind", "path", "sha256", "media_type", "metadata"},
            field_name="asset",
        )
        metadata = payload["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("asset.metadata 必须是对象")
        return cls(
            kind=payload["kind"],  # type: ignore[arg-type]
            path=payload["path"],  # type: ignore[arg-type]
            sha256=payload["sha256"],  # type: ignore[arg-type]
            media_type=payload["media_type"],  # type: ignore[arg-type]
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True, init=False)
class AnalysisArtifact:
    id: str
    source_document_id: str
    source_pixel_revision: int
    source_reference: AnalysisObjectReference | None
    tool_id: str
    tool_version: str
    calibration_signature: str | None
    tables: tuple[AnalysisTable, ...]
    curves: tuple[AnalysisCurve, ...]
    assets: tuple[AnalysisAssetReference, ...]
    status: AnalysisArtifactStatus
    stale_reason: str | None
    created_at: str
    _parameters_json: str = field(repr=False)
    _scalars_json: str = field(repr=False)

    def __init__(
        self,
        *,
        id: str,
        source_document_id: str,
        source_pixel_revision: int,
        tool_id: str,
        tool_version: str,
        parameters: Mapping[str, object] | None = None,
        calibration_signature: str | None = None,
        source_reference: AnalysisObjectReference | None = None,
        scalars: Mapping[str, JsonScalar] | None = None,
        tables: Iterable[AnalysisTable] = (),
        curves: Iterable[AnalysisCurve] = (),
        assets: Iterable[AnalysisAssetReference] = (),
        status: AnalysisArtifactStatus | str = AnalysisArtifactStatus.CURRENT,
        stale_reason: str | None = None,
        created_at: str | None = None,
    ) -> None:
        normalized_id = _required_id(id, field_name="id")
        document_id = _required_id(
            source_document_id,
            field_name="source_document_id",
        )
        pixel_revision = _non_negative_int(
            source_pixel_revision,
            field_name="source_pixel_revision",
        )
        normalized_tool_id = str(tool_id or "").strip().lower()
        if not _TOOL_ID_PATTERN.fullmatch(normalized_tool_id):
            raise ValueError(
                "tool_id 必须是小写字母或数字开头，且仅包含 "
                "a-z、0-9、点、下划线或连字符"
            )
        normalized_tool_version = _required_text(
            tool_version,
            field_name="tool_version",
            maximum_length=128,
        )
        parameters_json = _canonical_json_object(
            parameters or {},
            field_name="parameters",
        )
        scalars_json = _canonical_scalar_mapping(
            scalars or {},
            field_name="scalars",
        )
        if source_reference is not None and not isinstance(
            source_reference,
            AnalysisObjectReference,
        ):
            raise TypeError("source_reference 必须是 AnalysisObjectReference")
        normalized_signature = (
            None
            if calibration_signature is None
            else _required_text(
                calibration_signature,
                field_name="calibration_signature",
                maximum_length=256,
            )
        )
        frozen_tables = tuple(tables)
        frozen_curves = tuple(curves)
        frozen_assets = tuple(assets)
        if any(not isinstance(item, AnalysisTable) for item in frozen_tables):
            raise TypeError("tables 必须全部是 AnalysisTable")
        if any(not isinstance(item, AnalysisCurve) for item in frozen_curves):
            raise TypeError("curves 必须全部是 AnalysisCurve")
        if any(
            not isinstance(item, AnalysisAssetReference)
            for item in frozen_assets
        ):
            raise TypeError("assets 必须全部是 AnalysisAssetReference")
        if (
            sum(len(table.columns) * len(table.rows) for table in frozen_tables)
            > _MAX_INLINE_TABLE_CELLS
        ):
            raise ValueError(
                "分析表格总量过大，必须写入 analysis/ 下的安全资产并通过 assets 引用"
            )
        if (
            sum(len(curve.x) for curve in frozen_curves)
            > _MAX_INLINE_CURVE_POINTS
        ):
            raise ValueError(
                "分析曲线总量过大，必须写入 analysis/ 下的安全资产并通过 assets 引用"
            )
        try:
            normalized_status = AnalysisArtifactStatus(status)
        except (TypeError, ValueError) as error:
            raise ValueError(f"不支持的 analysis status: {status!r}") from error
        normalized_reason = (
            None
            if stale_reason is None
            else _required_text(
                stale_reason,
                field_name="stale_reason",
                maximum_length=1024,
            )
        )
        if normalized_status is AnalysisArtifactStatus.CURRENT:
            if normalized_reason is not None:
                raise ValueError("current 状态不能包含 stale_reason")
        elif normalized_reason is None:
            raise ValueError("stale 状态必须包含 stale_reason")
        timestamp = _normalize_timestamp(created_at or _utc_now_iso())

        object.__setattr__(self, "id", normalized_id)
        object.__setattr__(self, "source_document_id", document_id)
        object.__setattr__(self, "source_pixel_revision", pixel_revision)
        object.__setattr__(self, "source_reference", source_reference)
        object.__setattr__(self, "tool_id", normalized_tool_id)
        object.__setattr__(self, "tool_version", normalized_tool_version)
        object.__setattr__(self, "calibration_signature", normalized_signature)
        object.__setattr__(self, "tables", frozen_tables)
        object.__setattr__(self, "curves", frozen_curves)
        object.__setattr__(self, "assets", frozen_assets)
        object.__setattr__(self, "status", normalized_status)
        object.__setattr__(self, "stale_reason", normalized_reason)
        object.__setattr__(self, "created_at", timestamp)
        object.__setattr__(self, "_parameters_json", parameters_json)
        object.__setattr__(self, "_scalars_json", scalars_json)

    @property
    def parameters(self) -> dict[str, object]:
        return json.loads(self._parameters_json)

    @property
    def scalars(self) -> dict[str, JsonScalar]:
        return json.loads(self._scalars_json)

    @property
    def is_current(self) -> bool:
        return self.status is AnalysisArtifactStatus.CURRENT

    def mark_stale(self, reason: str) -> "AnalysisArtifact":
        if self.status is AnalysisArtifactStatus.STALE:
            return self
        return self._with_status(AnalysisArtifactStatus.STALE, reason)

    def _with_status(
        self,
        status: AnalysisArtifactStatus,
        stale_reason: str | None,
    ) -> "AnalysisArtifact":
        return AnalysisArtifact(
            id=self.id,
            source_document_id=self.source_document_id,
            source_pixel_revision=self.source_pixel_revision,
            source_reference=self.source_reference,
            tool_id=self.tool_id,
            tool_version=self.tool_version,
            parameters=self.parameters,
            calibration_signature=self.calibration_signature,
            scalars=self.scalars,
            tables=self.tables,
            curves=self.curves,
            assets=self.assets,
            status=status,
            stale_reason=stale_reason,
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": ANALYSIS_ARTIFACT_SCHEMA_VERSION,
            "id": self.id,
            "source_document_id": self.source_document_id,
            "source_pixel_revision": self.source_pixel_revision,
            "tool_id": self.tool_id,
            "tool_version": self.tool_version,
            "parameters": self.parameters,
            "scalars": self.scalars,
            "tables": [table.to_dict() for table in self.tables],
            "curves": [curve.to_dict() for curve in self.curves],
            "assets": [asset.to_dict() for asset in self.assets],
            "status": self.status.value,
            "created_at": self.created_at,
        }
        if self.source_reference is not None:
            payload["source_reference"] = self.source_reference.to_dict()
        if self.calibration_signature is not None:
            payload["calibration_signature"] = self.calibration_signature
        if self.stale_reason is not None:
            payload["stale_reason"] = self.stale_reason
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AnalysisArtifact":
        _require_mapping(payload, field_name="AnalysisArtifact")
        _require_exact_keys(
            payload,
            required={
                "schema_version",
                "id",
                "source_document_id",
                "source_pixel_revision",
                "tool_id",
                "tool_version",
                "parameters",
                "scalars",
                "tables",
                "curves",
                "assets",
                "status",
                "created_at",
            },
            optional={
                "source_reference",
                "calibration_signature",
                "stale_reason",
            },
            field_name="AnalysisArtifact",
        )
        schema_version = payload["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != ANALYSIS_ARTIFACT_SCHEMA_VERSION
        ):
            raise ValueError(
                "不支持的 AnalysisArtifact schema_version: "
                f"{schema_version!r}"
            )
        parameters = payload["parameters"]
        scalars = payload["scalars"]
        tables = payload["tables"]
        curves = payload["curves"]
        assets = payload["assets"]
        if not isinstance(parameters, Mapping):
            raise TypeError("AnalysisArtifact.parameters 必须是对象")
        if not isinstance(scalars, Mapping):
            raise TypeError("AnalysisArtifact.scalars 必须是对象")
        if not isinstance(tables, list) or any(
            not isinstance(item, Mapping) for item in tables
        ):
            raise TypeError("AnalysisArtifact.tables 必须是对象列表")
        if not isinstance(curves, list) or any(
            not isinstance(item, Mapping) for item in curves
        ):
            raise TypeError("AnalysisArtifact.curves 必须是对象列表")
        if not isinstance(assets, list) or any(
            not isinstance(item, Mapping) for item in assets
        ):
            raise TypeError("AnalysisArtifact.assets 必须是对象列表")
        source_reference_payload = payload.get("source_reference")
        if source_reference_payload is not None and not isinstance(
            source_reference_payload,
            Mapping,
        ):
            raise TypeError("AnalysisArtifact.source_reference 必须是对象")
        return cls(
            id=payload["id"],  # type: ignore[arg-type]
            source_document_id=payload["source_document_id"],  # type: ignore[arg-type]
            source_pixel_revision=payload["source_pixel_revision"],  # type: ignore[arg-type]
            source_reference=(
                None
                if source_reference_payload is None
                else AnalysisObjectReference.from_dict(source_reference_payload)
            ),
            tool_id=payload["tool_id"],  # type: ignore[arg-type]
            tool_version=payload["tool_version"],  # type: ignore[arg-type]
            parameters=parameters,
            calibration_signature=payload.get("calibration_signature"),  # type: ignore[arg-type]
            scalars=scalars,  # type: ignore[arg-type]
            tables=(
                AnalysisTable.from_dict(item)
                for item in tables
            ),
            curves=(
                AnalysisCurve.from_dict(item)
                for item in curves
            ),
            assets=(
                AnalysisAssetReference.from_dict(item)
                for item in assets
            ),
            status=payload["status"],  # type: ignore[arg-type]
            stale_reason=payload.get("stale_reason"),  # type: ignore[arg-type]
            created_at=payload["created_at"],  # type: ignore[arg-type]
        )


def calibration_signature_from_values(
    *,
    pixels_per_unit: float | None,
    unit: str | None,
) -> str | None:
    """Build a stable signature without exposing calibration formatting details."""

    if pixels_per_unit is None:
        if unit not in (None, "", "px"):
            raise ValueError("未标定状态不能带有物理单位")
        return None
    normalized_scale = _finite_number(
        pixels_per_unit,
        field_name="pixels_per_unit",
    )
    if normalized_scale <= 0.0:
        raise ValueError("pixels_per_unit 必须大于 0")
    normalized_unit = _required_text(
        unit,
        field_name="unit",
        maximum_length=64,
    )
    payload = json.dumps(
        {
            "pixels_per_unit": normalized_scale,
            "unit": normalized_unit,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def refresh_artifact_validity(
    artifact: AnalysisArtifact,
    *,
    source_document_exists: bool = True,
    current_pixel_revision: int | None = None,
    current_calibration_signature: str | None | object = _UNSET,
    roi_revisions: Mapping[str, int] | None = None,
    measurement_revisions: Mapping[str, int] | None = None,
) -> AnalysisArtifact:
    """Mark a current artifact stale when any authoritative source changed.

    A stale artifact is never automatically revived.  Recalculation must create
    a new current artifact so the old result remains auditable.
    """

    if not artifact.is_current:
        return artifact
    if not isinstance(source_document_exists, bool):
        raise TypeError("source_document_exists 必须是布尔值")
    if not source_document_exists:
        return artifact.mark_stale("来源文档已不存在")
    if current_pixel_revision is not None:
        normalized_revision = _non_negative_int(
            current_pixel_revision,
            field_name="current_pixel_revision",
        )
        if normalized_revision != artifact.source_pixel_revision:
            return artifact.mark_stale("来源图片像素已变化")
    if current_calibration_signature is not _UNSET:
        normalized_signature = (
            None
            if current_calibration_signature is None
            else _required_text(
                current_calibration_signature,
                field_name="current_calibration_signature",
                maximum_length=256,
            )
        )
        if normalized_signature != artifact.calibration_signature:
            return artifact.mark_stale("标定已变化")

    reference = artifact.source_reference
    if reference is None:
        return artifact
    revisions = (
        roi_revisions
        if reference.kind is AnalysisObjectKind.ROI
        else measurement_revisions
    )
    source_label = (
        "ROI"
        if reference.kind is AnalysisObjectKind.ROI
        else "测量对象"
    )
    if revisions is None or reference.object_id not in revisions:
        return artifact.mark_stale(f"引用的{source_label}已不存在")
    current_revision = _non_negative_int(
        revisions[reference.object_id],
        field_name=f"{source_label} revision",
    )
    if current_revision != reference.revision:
        return artifact.mark_stale(f"引用的{source_label}几何已变化")
    return artifact


def refresh_artifacts_validity(
    artifacts: Iterable[AnalysisArtifact],
    *,
    document_id: str,
    source_document_exists: bool = True,
    current_pixel_revision: int | None = None,
    current_calibration_signature: str | None | object = _UNSET,
    roi_revisions: Mapping[str, int] | None = None,
    measurement_revisions: Mapping[str, int] | None = None,
) -> tuple[AnalysisArtifact, ...]:
    """Refresh only artifacts belonging to one document, preserving order."""

    normalized_document_id = _required_id(
        document_id,
        field_name="document_id",
    )
    refreshed: list[AnalysisArtifact] = []
    for artifact in artifacts:
        if not isinstance(artifact, AnalysisArtifact):
            raise TypeError("artifacts 必须全部是 AnalysisArtifact")
        if artifact.source_document_id != normalized_document_id:
            refreshed.append(artifact)
            continue
        refreshed.append(
            refresh_artifact_validity(
                artifact,
                source_document_exists=source_document_exists,
                current_pixel_revision=current_pixel_revision,
                current_calibration_signature=current_calibration_signature,
                roi_revisions=roi_revisions,
                measurement_revisions=measurement_revisions,
            )
        )
    return tuple(refreshed)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _normalize_timestamp(value: object) -> str:
    normalized = _required_text(
        value,
        field_name="created_at",
        maximum_length=128,
    )
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at 必须是 ISO 8601 时间") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at 必须包含时区")
    return parsed.isoformat()


def _canonical_json_object(
    value: Mapping[str, object],
    *,
    field_name: str,
) -> str:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} 必须是对象")
    normalized = _normalize_json_value(value, field_name=field_name, depth=0)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} 必须是对象")
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > _MAX_INLINE_JSON_BYTES:
        raise ValueError(
            f"{field_name} 过大，必须写入 analysis/ 下的安全资产并通过 assets 引用"
        )
    return encoded


def _canonical_scalar_mapping(
    value: Mapping[str, JsonScalar],
    *,
    field_name: str,
) -> str:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} 必须是对象")
    normalized: dict[str, JsonScalar] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} 的键必须是非空字符串")
        normalized[key] = _normalize_json_scalar(
            item,
            field_name=f"{field_name}.{key}",
        )
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > _MAX_INLINE_JSON_BYTES:
        raise ValueError(
            f"{field_name} 过大，必须写入 analysis/ 下的安全资产并通过 assets 引用"
        )
    return encoded


def _normalize_json_value(
    value: object,
    *,
    field_name: str,
    depth: int,
) -> object:
    if depth > 64:
        raise ValueError(f"{field_name} 嵌套层级过深")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} 不能包含 NaN 或 Inf")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} 的键必须是字符串")
            normalized[key] = _normalize_json_value(
                item,
                field_name=f"{field_name}.{key}",
                depth=depth + 1,
            )
        return normalized
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _normalize_json_value(
                item,
                field_name=f"{field_name}[{index}]",
                depth=depth + 1,
            )
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{field_name} 只能包含 JSON 对象、列表、字符串、有限数、布尔值或 null"
    )


def _normalize_json_scalar(
    value: object,
    *,
    field_name: str,
) -> JsonScalar:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} 不能是 NaN 或 Inf")
        return value
    raise TypeError(f"{field_name} 必须是 JSON 标量")


def _relative_asset_path(value: object) -> str:
    normalized = _required_text(
        value,
        field_name="asset.path",
        maximum_length=1024,
    ).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", normalized):
        raise ValueError("asset.path 必须是安全的相对路径")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("asset.path 必须是安全的相对路径")
    if not path.parts or path.parts[0].casefold() != "analysis":
        raise ValueError("asset.path 必须位于项目 analysis/ 资产目录")
    if any(any(character in '<>:"|?*' for character in part) for part in path.parts):
        raise ValueError("asset.path 包含 Windows 不支持的字符")
    return path.as_posix()


def _validate_safe_asset_encoding(*, path: str, media_type: str) -> None:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _UNSAFE_SERIALIZED_ASSET_SUFFIXES:
        raise ValueError("分析资产禁止使用 NPY 或 pickle 序列化")
    normalized_media_type = media_type.strip().lower()
    if "pickle" in normalized_media_type or "python-serialize" in normalized_media_type:
        raise ValueError("分析资产禁止使用 pickle 媒体类型")


def _require_mapping(
    payload: object,
    *,
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_name} 必须是对象")
    if any(not isinstance(key, str) for key in payload):
        raise TypeError(f"{field_name} 的键必须是字符串")
    return payload


def _require_exact_keys(
    payload: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] | None = None,
    field_name: str,
) -> None:
    actual = set(payload)
    missing = required - actual
    unknown = actual - required - (optional or set())
    if missing:
        raise ValueError(f"{field_name} 缺少字段: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{field_name} 包含未知字段: {', '.join(sorted(unknown))}")


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} 必须是数值")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} 必须是数值") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} 必须是有限数")
    return normalized


def _required_text(
    value: object,
    *,
    field_name: str,
    maximum_length: int,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    if len(normalized) > maximum_length:
        raise ValueError(f"{field_name} 不能超过 {maximum_length} 个字符")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{field_name} 不能包含控制字符")
    return normalized


def _optional_text(
    value: object,
    *,
    field_name: str,
    maximum_length: int,
) -> str:
    if value in (None, ""):
        return ""
    return _required_text(
        value,
        field_name=field_name,
        maximum_length=maximum_length,
    )


def _required_id(value: object, *, field_name: str) -> str:
    normalized = _required_text(value, field_name=field_name, maximum_length=256)
    if not _ID_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} 包含无效字符")
    return normalized


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} 必须是整数")
    if value < 0:
        raise ValueError(f"{field_name} 不能小于 0")
    return value


__all__ = [
    "ANALYSIS_ARTIFACT_SCHEMA_VERSION",
    "AnalysisArtifact",
    "AnalysisArtifactStatus",
    "AnalysisAssetKind",
    "AnalysisAssetReference",
    "AnalysisCurve",
    "AnalysisObjectKind",
    "AnalysisObjectReference",
    "AnalysisTable",
    "JsonScalar",
    "calibration_signature_from_values",
    "refresh_artifact_validity",
    "refresh_artifacts_validity",
]
