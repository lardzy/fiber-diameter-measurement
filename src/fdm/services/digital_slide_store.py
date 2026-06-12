from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter


DIGITAL_SLIDE_SUFFIX = ".fdmslide"
DOCUMENT_KIND_IMAGE = "image"
DOCUMENT_KIND_DIGITAL_SLIDE = "digital_slide"


@dataclass(slots=True)
class DigitalSlideManifest:
    version: int
    width: int
    height: int
    viewport_width: int
    viewport_height: int
    focus_levels: list[int]
    tile_count: int = 0
    status: str = "ready"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "width": self.width,
            "height": self.height,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "focus_levels": list(self.focus_levels),
            "tile_count": int(self.tile_count),
            "status": self.status,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DigitalSlideManifest":
        return cls(
            version=int(payload.get("version", 1)),
            width=max(1, int(payload.get("width", payload.get("image_width", 1)))),
            height=max(1, int(payload.get("height", payload.get("image_height", 1)))),
            viewport_width=max(1, int(payload.get("viewport_width", payload.get("tile_width", 1)))),
            viewport_height=max(1, int(payload.get("viewport_height", payload.get("tile_height", 1)))),
            focus_levels=[int(item) for item in payload.get("focus_levels", [0])],
            tile_count=max(0, int(payload.get("tile_count", 0))),
            status=str(payload.get("status", "ready")),
            created_at=str(payload.get("created_at", "")) or datetime.now().isoformat(timespec="seconds"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class DigitalSlideTile:
    z_index: int
    x: int
    y: int
    width: int
    height: int
    stage_x: int = 0
    stage_y: int = 0
    focus_z: int = 0
    sharpness: float = 0.0
    status: str = "ready"


def is_digital_slide_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == DIGITAL_SLIDE_SUFFIX


def qimage_to_png_bytes(image: QImage) -> bytes:
    if image.isNull():
        raise ValueError("cannot encode null image")
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise RuntimeError("无法编码切片图像。")
    buffer.close()
    return bytes(data)


def png_bytes_to_qimage(payload: bytes) -> QImage:
    image = QImage()
    image.loadFromData(payload, "PNG")
    return image


class DigitalSlideStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> "DigitalSlideStore":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @classmethod
    def create(cls, path: str | Path, manifest: DigitalSlideManifest) -> "DigitalSlideStore":
        store = cls(path)
        store.path.parent.mkdir(parents=True, exist_ok=True)
        if store.path.exists():
            store.path.unlink()
        store.open()
        store._initialize_schema()
        store.write_manifest(manifest)
        return store

    def open(self) -> None:
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    def _initialize_schema(self) -> None:
        conn = self._connection()
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                z_index INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                stage_x INTEGER NOT NULL DEFAULT 0,
                stage_y INTEGER NOT NULL DEFAULT 0,
                focus_z INTEGER NOT NULL DEFAULT 0,
                sharpness REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'ready',
                image_png BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tiles_view ON tiles(z_index, x, y, width, height);
            """
        )
        conn.commit()

    def write_manifest(self, manifest: DigitalSlideManifest) -> None:
        conn = self._connection()
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('manifest', ?)",
            (json.dumps(manifest.to_dict(), ensure_ascii=False),),
        )
        conn.commit()

    def read_manifest(self) -> DigitalSlideManifest:
        conn = self._connection()
        self._initialize_schema()
        row = conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if row is None:
            raise ValueError(f"数字化切片缺少 manifest: {self.path}")
        manifest = DigitalSlideManifest.from_dict(json.loads(str(row["value"])))
        manifest.tile_count = self.tile_count()
        return manifest

    def update_status(self, status: str) -> None:
        manifest = self.read_manifest()
        manifest.status = status
        manifest.tile_count = self.tile_count()
        self.write_manifest(manifest)

    def tile_count(self) -> int:
        conn = self._connection()
        self._initialize_schema()
        row = conn.execute("SELECT COUNT(*) AS total FROM tiles").fetchone()
        return int(row["total"] if row is not None else 0)

    def write_tile(self, tile: DigitalSlideTile, image: QImage) -> None:
        if image.isNull():
            raise ValueError("cannot write null tile image")
        conn = self._connection()
        self._initialize_schema()
        payload = qimage_to_png_bytes(image)
        conn.execute(
            """
            INSERT INTO tiles(
                z_index, x, y, width, height, stage_x, stage_y, focus_z,
                sharpness, status, image_png
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(tile.z_index),
                int(tile.x),
                int(tile.y),
                int(tile.width),
                int(tile.height),
                int(tile.stage_x),
                int(tile.stage_y),
                int(tile.focus_z),
                float(tile.sharpness),
                str(tile.status),
                payload,
            ),
        )
        manifest = self.read_manifest()
        manifest.tile_count = self.tile_count()
        self.write_manifest(manifest)
        conn.commit()

    def read_tiles_for_viewport(self, *, x: int, y: int, width: int, height: int, z_index: int) -> list[tuple[DigitalSlideTile, QImage]]:
        conn = self._connection()
        self._initialize_schema()
        x2 = int(x) + int(width)
        y2 = int(y) + int(height)
        rows = conn.execute(
            """
            SELECT z_index, x, y, width, height, stage_x, stage_y, focus_z, sharpness, status, image_png
            FROM tiles
            WHERE z_index = ?
              AND x < ?
              AND y < ?
              AND (x + width) > ?
              AND (y + height) > ?
            ORDER BY id ASC
            """,
            (int(z_index), x2, y2, int(x), int(y)),
        ).fetchall()
        tiles: list[tuple[DigitalSlideTile, QImage]] = []
        for row in rows:
            image = png_bytes_to_qimage(bytes(row["image_png"]))
            if image.isNull():
                continue
            tiles.append(
                (
                    DigitalSlideTile(
                        z_index=int(row["z_index"]),
                        x=int(row["x"]),
                        y=int(row["y"]),
                        width=int(row["width"]),
                        height=int(row["height"]),
                        stage_x=int(row["stage_x"]),
                        stage_y=int(row["stage_y"]),
                        focus_z=int(row["focus_z"]),
                        sharpness=float(row["sharpness"]),
                        status=str(row["status"]),
                    ),
                    image,
                )
            )
        return tiles

    def render_viewport(self, *, x: int, y: int, width: int, height: int, z_index: int) -> QImage:
        output = QImage(max(1, int(width)), max(1, int(height)), QImage.Format.Format_RGB32)
        output.fill(QColor("#101820"))
        painter = QPainter(output)
        if not painter.isActive():
            return output
        for tile, image in self.read_tiles_for_viewport(x=x, y=y, width=width, height=height, z_index=z_index):
            source_left = max(0, int(x) - tile.x)
            source_top = max(0, int(y) - tile.y)
            source_right = min(tile.width, int(x) + int(width) - tile.x)
            source_bottom = min(tile.height, int(y) + int(height) - tile.y)
            if source_right <= source_left or source_bottom <= source_top:
                continue
            target_left = tile.x + source_left - int(x)
            target_top = tile.y + source_top - int(y)
            painter.drawImage(
                QRectF(target_left, target_top, source_right - source_left, source_bottom - source_top),
                image,
                QRectF(source_left, source_top, source_right - source_left, source_bottom - source_top),
            )
        painter.end()
        return output


def copy_slide_file(source: str | Path, target: str | Path) -> Path:
    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == target_path.resolve():
        return target_path
    shutil.copy2(source_path, target_path)
    return target_path
