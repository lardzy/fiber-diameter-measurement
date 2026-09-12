"""Immutable, non-destructive coordinates for a digital slide.

Coordinates and thresholds are in *stored* image pixels.  Source identity is
content based; paths and timestamps are deliberately not identities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
from typing import Callable

from fdm.atomic_io import atomic_write_json

LAYOUT_METADATA_KEY = "stitch_layout"
LAYOUT_VERSION = 1


def read_connection(path: str | Path) -> sqlite3.Connection:
    # Do not use immutable=1: capture may still be committing into a WAL.
    connection = sqlite3.connect(Path(path).absolute().as_uri() + "?mode=ro", uri=True, timeout=2)
    connection.row_factory = sqlite3.Row
    return connection


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, slots=True)
class LayoutTile:
    tile_id: int
    fov_id: str
    z_index: int
    focus_z: int
    nominal_x: float
    nominal_y: float
    x: float
    y: float
    width: int
    height: int
    stage_x: int
    stage_y: int


@dataclass(frozen=True, slots=True)
class LayerRegistrationEvidence:
    z_index: int
    focus_z: int
    accepted: bool
    reason: str
    dx: float = 0.0
    dy: float = 0.0
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class PairRegistrationResult:
    first: str
    second: str
    axis: str
    accepted: bool
    reason: str
    dx: float = 0.0
    dy: float = 0.0
    confidence: float = 0.0
    residual: float = 0.0
    verified_focus: tuple[int, ...] = ()
    evidence_count: int = 0
    layers: tuple[LayerRegistrationEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class SlideLayoutSnapshot:
    source_digest: str
    tiles: tuple[LayoutTile, ...]
    pairs: tuple[PairRegistrationResult, ...]
    width: int
    height: int
    layout_id: str = ""
    version: int = LAYOUT_VERSION
    sampling: str = "single-source-linear-v1"

    def to_dict(self) -> dict:
        return asdict(self)

    def sealed(self) -> "SlideLayoutSnapshot":
        payload = self.to_dict()
        payload.pop("layout_id")
        return replace(self, layout_id=sha256(canonical_bytes(payload)).hexdigest())

    @classmethod
    def from_dict(cls, payload: dict) -> "SlideLayoutSnapshot":
        if payload.get("version") != LAYOUT_VERSION or payload.get("sampling") != "single-source-linear-v1":
            raise ValueError("不支持的拼接布局版本")
        tiles = tuple(LayoutTile(**item) for item in payload["tiles"])
        pairs = tuple(PairRegistrationResult(**{**item,
            "verified_focus": tuple(item.get("verified_focus", ())),
            "layers": tuple(LayerRegistrationEvidence(**layer) for layer in item.get("layers", ()))})
            for item in payload["pairs"])
        result = cls(source_digest=str(payload["source_digest"]), tiles=tiles, pairs=pairs,
                     width=int(payload["width"]), height=int(payload["height"]), layout_id=str(payload["layout_id"]))
        ids = {tile.tile_id for tile in tiles}
        fovs = {tile.fov_id for tile in tiles}
        if len(ids) != len(tiles) or min(result.width, result.height) <= 0:
            raise ValueError("拼接布局的图块或尺寸无效")
        for tile in tiles:
            if min(tile.width, tile.height) <= 0 or not all(math.isfinite(v) for v in (tile.x, tile.y, tile.nominal_x, tile.nominal_y)):
                raise ValueError("拼接布局包含无效坐标")
            if tile.x < 0 or tile.y < 0 or tile.x + tile.width > result.width + 1e-6 or tile.y + tile.height > result.height + 1e-6:
                raise ValueError("拼接图块超出布局范围")
        by_fov = {}
        for tile in tiles:
            point = by_fov.setdefault(tile.fov_id, (tile.x, tile.y))
            if point != (tile.x, tile.y):
                raise ValueError("同一视场各焦层不能使用不同 XY 位置")
        if any(p.axis not in {"x", "y"} or not all(math.isfinite(v) for v in (p.dx, p.dy, p.confidence, p.residual)) for p in pairs):
            raise ValueError("拼接接缝证据无效")
        if any(not all(math.isfinite(v) for v in (layer.dx, layer.dy, layer.score)) for p in pairs for layer in p.layers):
            raise ValueError("拼接焦层证据无效")
        if any(pair.first not in fovs or pair.second not in fovs for pair in pairs):
            raise ValueError("拼接连接引用不存在的视场")
        if result.sealed().layout_id != result.layout_id:
            raise ValueError("拼接布局校验失败")
        return result

    @property
    def accepted_count(self) -> int:
        return sum(pair.accepted for pair in self.pairs)

    def summary(self) -> str:
        ineligible = sum(not p.accepted and p.reason in {"overlap", "focus_mismatch"} for p in self.pairs)
        partial = sum(p.accepted and p.reason == "partial_focus" for p in self.pairs)
        detail = f"（{partial} 处仅部分焦层）" if partial else ""
        return f"接缝 {len(self.pairs)} 处 · 已修复 {self.accepted_count}{detail} · 待复核 {len(self.pairs) - self.accepted_count - ineligible} · 不适用 {ineligible}"


def read_source_tiles(connection: sqlite3.Connection) -> tuple[LayoutTile, ...]:
    rows = connection.execute("SELECT id,z_index,focus_z,x,y,width,height,stage_x,stage_y FROM tiles ORDER BY id")
    return tuple(LayoutTile(
        tile_id=int(r["id"]), fov_id=f'{r["x"]}:{r["y"]}:{r["stage_x"]}:{r["stage_y"]}',
        z_index=int(r["z_index"]), focus_z=int(r["focus_z"]), nominal_x=float(r["x"]), nominal_y=float(r["y"]),
        x=float(r["x"]), y=float(r["y"]), width=int(r["width"]), height=int(r["height"]),
        stage_x=int(r["stage_x"]), stage_y=int(r["stage_y"])) for r in rows)


def source_digest(connection: sqlite3.Connection, cancelled: Callable[[], bool] = lambda: False) -> str:
    digest = sha256()
    manifest = json.loads(connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
    digest.update(canonical_bytes({k: manifest.get(k) for k in ("version", "width", "height", "viewport_width", "viewport_height", "focus_levels")}))
    metadata = manifest.get("metadata", {})
    digest.update(canonical_bytes({k: metadata.get(k) for k in ("capture_session_id", "position_kind", "focus_basis", "z_approach", "z_backlash_steps", "xy_calibration")}))
    for row in connection.execute("SELECT id,z_index,focus_z,x,y,width,height,stage_x,stage_y,codec,image_png FROM tiles ORDER BY id"):
        if cancelled():
            raise InterruptedError("拼接检查已暂停")
        digest.update(canonical_bytes(list(row)[:-1]))
        digest.update(sha256(bytes(row["image_png"])).digest())
    return digest.hexdigest()


def validate_source(path: str | Path, layout: SlideLayoutSnapshot, cancelled: Callable[[], bool] = lambda: False) -> None:
    connection = read_connection(path)
    try:
        connection.execute("BEGIN")
        if source_digest(connection, cancelled) != layout.source_digest:
            raise ValueError("切片内容已变化，不能应用此拼接版本")
        expected = {t.tile_id: t for t in read_source_tiles(connection)}
        if len(layout.tiles) != len(expected) or any(t.tile_id not in expected or replace(t, x=t.nominal_x, y=t.nominal_y) != expected[t.tile_id] for t in layout.tiles):
            raise ValueError("拼接布局与源图块几何不一致")
    finally:
        connection.close()


def sidecar_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".fdmstitch")


def save_layout(path: str | Path, layout: SlideLayoutSnapshot) -> None:
    # Atomic JSON holds only coordinates/evidence, never the source image data.
    atomic_write_json(Path(path), layout.to_dict())


def load_layout(path: str | Path) -> SlideLayoutSnapshot:
    return SlideLayoutSnapshot.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def local_result_path(source_path: str | Path) -> Path:
    # Durable, small recovery metadata, separate from disposable image caches.
    from fdm.settings import settings_directory
    key = sha256(str(Path(source_path).absolute()).encode("utf-8")).hexdigest()
    return settings_directory() / "stitch-results" / f"{key}.fdmstitch"
