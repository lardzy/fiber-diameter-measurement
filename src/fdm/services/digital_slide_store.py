from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter, QRegion

from fdm.atomic_io import atomic_replace_file, staged_path_for
from fdm.services.digital_slide_cache import is_network_file_path


DIGITAL_SLIDE_SUFFIX = ".fdmslide"
DOCUMENT_KIND_IMAGE = "image"
DOCUMENT_KIND_DIGITAL_SLIDE = "digital_slide"
DIGITAL_SLIDE_TILE_CODEC_PNG = "png"
DIGITAL_SLIDE_TILE_CODEC_JPEG = "jpeg"
DIGITAL_SLIDE_OVERVIEW_MAX_EDGE = 256
SUPPORTED_DIGITAL_SLIDE_TILE_CODECS = {
    DIGITAL_SLIDE_TILE_CODEC_PNG,
    DIGITAL_SLIDE_TILE_CODEC_JPEG,
}


def _sqlite_sidecar_paths(path: Path) -> tuple[Path, Path]:
    return Path(f"{path}-wal"), Path(f"{path}-shm")


def _remove_sqlite_sidecars(path: Path) -> None:
    for sidecar in _sqlite_sidecar_paths(path):
        sidecar.unlink(missing_ok=True)


def _quick_check_connection(connection: sqlite3.Connection) -> None:
    rows = connection.execute("PRAGMA quick_check").fetchall()
    messages = [str(row[0]) for row in rows]
    if messages != ["ok"]:
        detail = "; ".join(messages) or "no result"
        raise sqlite3.DatabaseError(f"SQLite quick_check failed: {detail}")


def _connect_sqlite_read_only(path: str | Path) -> sqlite3.Connection:
    """Open an existing database read-only, including Windows UNC paths.

    SQLite URI filenames reject a non-local ``file://server/...`` authority in
    standard builds.  ``Path.as_uri()`` produces exactly that form for a UNC
    path, so network files use the native filename plus ``query_only`` instead.
    An existence check guards the ordinary filename connection from being used
    as a create path, while the pragma prevents SQL writes during the short
    manifest validation connection.
    """

    slide_path = Path(path).expanduser()
    if not slide_path.is_file():
        raise FileNotFoundError(slide_path)
    if is_network_file_path(slide_path):
        connection = sqlite3.connect(str(slide_path))
        try:
            connection.execute("PRAGMA query_only=ON")
        except Exception:
            connection.close()
            raise
        return connection
    resolved = slide_path.resolve(strict=True)
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)


def _make_staged_database_standalone(path: Path) -> None:
    connection = sqlite3.connect(str(path))
    try:
        mode_row = connection.execute("PRAGMA journal_mode").fetchone()
        mode = str(mode_row[0]).lower() if mode_row is not None else ""
        if mode == "wal":
            checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is not None and int(checkpoint[0]) != 0:
                raise sqlite3.OperationalError("staged SQLite database is busy during checkpoint")
        changed_mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
        if changed_mode is None or str(changed_mode[0]).lower() != "delete":
            raise sqlite3.OperationalError("could not switch staged SQLite database to DELETE journal mode")
        connection.commit()
        _quick_check_connection(connection)
    finally:
        connection.close()
    _remove_sqlite_sidecars(path)


def _assert_existing_sqlite_target_is_standalone(path: Path) -> None:
    """Reject ambiguous live/stale WAL targets without touching old bytes.

    Replacing only the main database while an existing ``-wal``/``-shm`` pair
    is present is not an atomic SQLite update.  More importantly, checkpointing
    or changing journal mode before ``os.replace`` would mutate the old project
    even when the later replacement fails.  Callers must close/recover that
    database first; a normal closed target has no WAL sidecars.
    """

    sidecars = [sidecar for sidecar in _sqlite_sidecar_paths(path) if sidecar.exists()]
    if sidecars:
        names = ", ".join(sidecar.name for sidecar in sidecars)
        raise sqlite3.OperationalError(
            f"target SQLite database has active or stale WAL sidecars ({names}); "
            "close or recover it before replacement"
        )


def _atomic_backup_connection(connection: sqlite3.Connection, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with staged_path_for(target, suffix=".sqlite.tmp") as staged_path:
        destination = sqlite3.connect(str(staged_path))
        try:
            connection.backup(destination)
            destination.commit()
        finally:
            destination.close()
        _make_staged_database_standalone(staged_path)
        _assert_existing_sqlite_target_is_standalone(target)
        atomic_replace_file(staged_path, target)
    return target


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


@dataclass(frozen=True, slots=True)
class DigitalSlideTileDescriptor:
    tile_id: int
    z_index: int
    x: int
    y: int
    width: int
    height: int
    stage_x: int
    stage_y: int
    focus_z: int
    sharpness: float = 0.0
    status: str = "ready"


class DigitalSlideOverviewAccumulator:
    """Build small per-focus overviews while decoded capture tiles are available."""

    def __init__(
        self,
        manifest: DigitalSlideManifest,
        *,
        maximum_edge: int = DIGITAL_SLIDE_OVERVIEW_MAX_EDGE,
        focus_indices: set[int] | None = None,
    ) -> None:
        max_edge = max(1, min(int(maximum_edge), 1024))
        self._scale = min(
            1.0,
            max_edge / max(1, int(manifest.width)),
            max_edge / max(1, int(manifest.height)),
        )
        self._width = max(1, int(round(manifest.width * self._scale)))
        self._height = max(1, int(round(manifest.height * self._scale)))
        self._images: dict[int, QImage] = {}
        self._focus_indices = (
            {int(index) for index in focus_indices}
            if focus_indices is not None
            else None
        )

    def add_tile(self, tile: DigitalSlideTile, image: QImage) -> None:
        if image.isNull():
            return
        if self._focus_indices is not None and int(tile.z_index) not in self._focus_indices:
            return
        target = QRectF(
            float(tile.x) * self._scale,
            float(tile.y) * self._scale,
            float(tile.width) * self._scale,
            float(tile.height) * self._scale,
        )
        if target.isEmpty() or not target.intersects(
            QRectF(0.0, 0.0, float(self._width), float(self._height))
        ):
            return
        focus_index = int(tile.z_index)
        overview = self._images.get(focus_index)
        if overview is None:
            overview = QImage(
                self._width,
                self._height,
                QImage.Format.Format_RGB32,
            )
            overview.fill(QColor("#101820"))
            self._images[focus_index] = overview
        painter = QPainter(overview)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(target, image)
        painter.end()

    def images(self) -> dict[int, QImage]:
        return {
            focus_index: QImage(image)
            for focus_index, image in self._images.items()
        }


def is_digital_slide_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == DIGITAL_SLIDE_SUFFIX


def normalize_tile_codec(codec: str | None) -> str:
    token = str(codec or DIGITAL_SLIDE_TILE_CODEC_PNG).strip().lower()
    if token in {"jpg", "jpeg"}:
        return DIGITAL_SLIDE_TILE_CODEC_JPEG
    if token in SUPPORTED_DIGITAL_SLIDE_TILE_CODECS:
        return token
    return DIGITAL_SLIDE_TILE_CODEC_PNG


def normalize_jpeg_quality(quality: int | None) -> int:
    try:
        value = int(quality if quality is not None else 90)
    except (TypeError, ValueError):
        value = 90
    return max(70, min(value, 95))


def qimage_to_image_bytes(image: QImage, *, codec: str = DIGITAL_SLIDE_TILE_CODEC_PNG, quality: int | None = None) -> bytes:
    if image.isNull():
        raise ValueError("cannot encode null image")
    normalized_codec = normalize_tile_codec(codec)
    image_format = "JPG" if normalized_codec == DIGITAL_SLIDE_TILE_CODEC_JPEG else "PNG"
    image_to_save = image
    if normalized_codec == DIGITAL_SLIDE_TILE_CODEC_JPEG and image.hasAlphaChannel():
        image_to_save = image.convertToFormat(QImage.Format.Format_RGB888)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    save_quality = normalize_jpeg_quality(quality) if normalized_codec == DIGITAL_SLIDE_TILE_CODEC_JPEG else -1
    if not image_to_save.save(buffer, image_format, save_quality):
        raise RuntimeError("无法编码切片图像。")
    buffer.close()
    return bytes(data)


def qimage_to_png_bytes(image: QImage) -> bytes:
    return qimage_to_image_bytes(image, codec=DIGITAL_SLIDE_TILE_CODEC_PNG)


def image_bytes_to_qimage(payload: bytes, *, codec: str | None = None) -> QImage:
    image = QImage()
    normalized_codec = normalize_tile_codec(codec)
    image_format = "JPG" if normalized_codec == DIGITAL_SLIDE_TILE_CODEC_JPEG else "PNG"
    image.loadFromData(payload, image_format)
    if image.isNull():
        image.loadFromData(payload)
    return image


def png_bytes_to_qimage(payload: bytes) -> QImage:
    return image_bytes_to_qimage(payload, codec=DIGITAL_SLIDE_TILE_CODEC_PNG)


class DigitalSlideStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._read_only = False
        self._schema_initialized = False
        self._image_cache: OrderedDict[tuple[int, str], QImage] = OrderedDict()
        self._image_cache_limit = 64
        self._image_cache_byte_limit = 256 * 1024 * 1024
        self._image_cache_bytes = 0

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
        _remove_sqlite_sidecars(store.path)
        store.open()
        store._initialize_schema()
        store.write_manifest(manifest)
        return store

    def open(self) -> None:
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._read_only = False
        self._schema_initialized = False

    def open_read_only(self) -> None:
        if self._conn is not None:
            return
        self._conn = _connect_sqlite_read_only(self.path)
        self._conn.row_factory = sqlite3.Row
        self._read_only = True
        self._schema_initialized = True

    def is_open(self) -> bool:
        return self._conn is not None

    def set_decoded_image_cache_budget(
        self,
        *,
        max_images: int,
        max_bytes: int,
    ) -> None:
        """Configure the store-local decode cache.

        Long-lived renderers own a shared byte-LRU and set this budget to zero
        so the same decoded tile is not retained by two independent caches.
        """

        self._image_cache_limit = max(0, int(max_images))
        self._image_cache_byte_limit = max(0, int(max_bytes))
        while (
            self._image_cache
            and (
                len(self._image_cache) > self._image_cache_limit
                or self._image_cache_bytes > self._image_cache_byte_limit
            )
        ):
            _, removed = self._image_cache.popitem(last=False)
            self._image_cache_bytes = max(
                0,
                self._image_cache_bytes - max(0, int(removed.sizeInBytes())),
            )

    @staticmethod
    def read_manifest_read_only(path: str | Path) -> DigitalSlideManifest:
        """Validate an existing slide without creating or migrating its schema."""

        slide_path = Path(path).expanduser()
        connection = _connect_sqlite_read_only(slide_path)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='manifest'"
            ).fetchone()
            if row is None:
                raise ValueError(f"数字化切片缺少 manifest: {slide_path}")
            manifest = DigitalSlideManifest.from_dict(json.loads(str(row["value"])))
            tile_row = connection.execute("SELECT COUNT(*) AS total FROM tiles").fetchone()
            manifest.tile_count = int(tile_row["total"] if tile_row is not None else 0)
            return manifest
        finally:
            connection.close()

    def close(self) -> None:
        connection = self._conn
        first_error: Exception | None = None
        closed = connection is None
        if connection is not None:
            if not self._read_only:
                try:
                    connection.commit()
                except Exception as exc:  # noqa: BLE001 - still attempt physical close
                    first_error = exc
            try:
                connection.close()
                closed = True
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = exc
        if closed:
            self._conn = None
            self._read_only = False
            self._schema_initialized = False
        self._image_cache.clear()
        self._image_cache_bytes = 0
        if first_error is not None:
            raise first_error

    def backup_to(self, target: str | Path) -> Path:
        target_path = Path(target)
        try:
            if self.path.resolve() == target_path.resolve():
                return target_path
        except OSError:
            pass
        connection = self._connection()
        connection.commit()
        return _atomic_backup_connection(connection, target_path)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    def _initialize_schema(self) -> None:
        if self._schema_initialized:
            return
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
                image_png BLOB NOT NULL,
                codec TEXT NOT NULL DEFAULT 'png',
                quality INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_tiles_view ON tiles(z_index, x, y, width, height);
            CREATE TABLE IF NOT EXISTS focus_overviews (
                z_index INTEGER PRIMARY KEY,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                image_png BLOB NOT NULL,
                codec TEXT NOT NULL DEFAULT 'png',
                updated_at TEXT NOT NULL
            );
            """
        )
        self._ensure_tile_codec_columns(conn)
        conn.commit()
        self._schema_initialized = True

    def _ensure_tile_codec_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(tiles)").fetchall()
        columns = {str(row["name"]) if isinstance(row, sqlite3.Row) else str(row[1]) for row in rows}
        if "codec" not in columns:
            conn.execute("ALTER TABLE tiles ADD COLUMN codec TEXT NOT NULL DEFAULT 'png'")
        if "quality" not in columns:
            conn.execute("ALTER TABLE tiles ADD COLUMN quality INTEGER")

    def write_manifest(self, manifest: DigitalSlideManifest) -> None:
        conn = self._connection()
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('manifest', ?)",
            (json.dumps(manifest.to_dict(), ensure_ascii=False, allow_nan=False),),
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

    def list_tile_descriptors(self, *, z_index: int) -> list[DigitalSlideTileDescriptor]:
        conn = self._connection()
        self._initialize_schema()
        rows = conn.execute(
            """
            SELECT id, z_index, x, y, width, height, stage_x, stage_y,
                   focus_z, sharpness, status
            FROM tiles
            WHERE z_index = ?
            ORDER BY y ASC, x ASC, id ASC
            """,
            (int(z_index),),
        ).fetchall()
        return [
            DigitalSlideTileDescriptor(
                tile_id=int(row["id"]),
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
            )
            for row in rows
        ]

    def list_tile_descriptors_in_rect(
        self,
        *,
        z_index: int,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> list[DigitalSlideTileDescriptor]:
        """Return descriptor-only rows intersecting one source rectangle.

        Interactive rendering uses this query so visiting a new focus plane
        does not materialize and sort every descriptor in that plane merely to
        display one native field.  Tile BLOBs are deliberately excluded.
        """

        if float(width) <= 0.0 or float(height) <= 0.0:
            return []
        conn = self._connection()
        self._initialize_schema()
        x2 = float(x) + float(width)
        y2 = float(y) + float(height)
        rows = conn.execute(
            """
            SELECT id, z_index, x, y, width, height, stage_x, stage_y,
                   focus_z, sharpness, status
            FROM tiles
            WHERE z_index = ?
              AND x < ?
              AND y < ?
              AND (x + width) > ?
              AND (y + height) > ?
            ORDER BY y ASC, x ASC, id ASC
            """,
            (int(z_index), x2, y2, float(x), float(y)),
        )
        return [
            DigitalSlideTileDescriptor(
                tile_id=int(row["id"]),
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
            )
            for row in rows
        ]

    def read_tile_image(self, tile_id: int) -> QImage:
        conn = self._connection()
        self._initialize_schema()
        row = conn.execute(
            "SELECT image_png, codec FROM tiles WHERE id = ?",
            (int(tile_id),),
        ).fetchone()
        if row is None:
            return QImage()
        return self._decode_tile_image(
            int(tile_id),
            bytes(row["image_png"]),
            str(row["codec"] or DIGITAL_SLIDE_TILE_CODEC_PNG),
        )

    def read_tile_image_scaled(
        self,
        tile_id: int,
        *,
        width: int,
        height: int,
    ) -> QImage:
        """Decode a display-only tile near its requested LOD dimensions."""

        conn = self._connection()
        self._initialize_schema()
        row = conn.execute(
            "SELECT image_png, codec FROM tiles WHERE id = ?",
            (int(tile_id),),
        ).fetchone()
        if row is None:
            return QImage()
        payload = bytes(row["image_png"])
        codec = normalize_tile_codec(
            str(row["codec"] or DIGITAL_SLIDE_TILE_CODEC_PNG)
        )
        image_format = (
            b"JPG" if codec == DIGITAL_SLIDE_TILE_CODEC_JPEG else b"PNG"
        )
        buffer = QBuffer()
        buffer.setData(QByteArray(payload))
        if buffer.open(QIODevice.OpenModeFlag.ReadOnly):
            reader = QImageReader(buffer, image_format)
            reader.setAutoTransform(True)
            reader.setScaledSize(
                QSize(max(1, int(width)), max(1, int(height)))
            )
            image = reader.read()
            buffer.close()
            if not image.isNull():
                return image
        image = image_bytes_to_qimage(payload, codec=codec)
        if image.isNull():
            return image
        return image.scaled(
            max(1, int(width)),
            max(1, int(height)),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

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

    def read_focus_overview(
        self,
        z_index: int,
        *,
        maximum_edge: int = DIGITAL_SLIDE_OVERVIEW_MAX_EDGE,
    ) -> QImage:
        conn = self._connection()
        self._initialize_schema()
        row = conn.execute(
            "SELECT image_png, codec FROM focus_overviews WHERE z_index = ?",
            (int(z_index),),
        ).fetchone()
        if row is None:
            return QImage()
        image = image_bytes_to_qimage(
            bytes(row["image_png"]),
            codec=str(row["codec"] or DIGITAL_SLIDE_TILE_CODEC_PNG),
        )
        return _bounded_overview_image(image, maximum_edge=maximum_edge)

    def write_focus_overviews(
        self,
        overviews: Mapping[int, QImage],
    ) -> int:
        conn = self._connection()
        self._initialize_schema()
        updated_at = datetime.now().isoformat(timespec="seconds")
        rows: list[tuple[int, int, int, bytes, str, str]] = []
        for focus_index, source in sorted(overviews.items()):
            image = _bounded_overview_image(
                source,
                maximum_edge=DIGITAL_SLIDE_OVERVIEW_MAX_EDGE,
            )
            if image.isNull():
                continue
            rows.append(
                (
                    int(focus_index),
                    int(image.width()),
                    int(image.height()),
                    qimage_to_image_bytes(
                        image,
                        codec=DIGITAL_SLIDE_TILE_CODEC_PNG,
                    ),
                    DIGITAL_SLIDE_TILE_CODEC_PNG,
                    updated_at,
                )
            )
        if not rows:
            return 0
        conn.executemany(
            """
            INSERT OR REPLACE INTO focus_overviews(
                z_index, width, height, image_png, codec, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return len(rows)

    def write_tile(
        self,
        tile: DigitalSlideTile,
        image: QImage,
        *,
        codec: str = DIGITAL_SLIDE_TILE_CODEC_PNG,
        quality: int | None = None,
        update_manifest: bool = True,
    ) -> None:
        if image.isNull():
            raise ValueError("cannot write null tile image")
        conn = self._connection()
        self._initialize_schema()
        normalized_codec = normalize_tile_codec(codec)
        normalized_quality = normalize_jpeg_quality(quality) if normalized_codec == DIGITAL_SLIDE_TILE_CODEC_JPEG else None
        payload = qimage_to_image_bytes(image, codec=normalized_codec, quality=normalized_quality)
        conn.execute(
            "DELETE FROM focus_overviews WHERE z_index = ?",
            (int(tile.z_index),),
        )
        conn.execute(
            """
            INSERT INTO tiles(
                z_index, x, y, width, height, stage_x, stage_y, focus_z,
                sharpness, status, image_png, codec, quality
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                normalized_codec,
                normalized_quality,
            ),
        )
        if update_manifest:
            manifest = self.read_manifest()
            manifest.tile_count = self.tile_count()
            self.write_manifest(manifest)
        conn.commit()

    def _decode_tile_image(self, tile_id: int, payload: bytes, codec: str | None) -> QImage:
        normalized_codec = normalize_tile_codec(codec)
        key = (int(tile_id), normalized_codec)
        cached = self._image_cache.get(key)
        if cached is not None:
            self._image_cache.move_to_end(key)
            return cached
        image = image_bytes_to_qimage(payload, codec=normalized_codec)
        if image.isNull():
            return image
        self._image_cache[key] = image
        self._image_cache_bytes += max(0, int(image.sizeInBytes()))
        self._image_cache.move_to_end(key)
        while (
            len(self._image_cache) > self._image_cache_limit
            or self._image_cache_bytes > self._image_cache_byte_limit
        ):
            _, removed = self._image_cache.popitem(last=False)
            self._image_cache_bytes = max(
                0,
                self._image_cache_bytes - max(0, int(removed.sizeInBytes())),
            )
        return image

    def read_tiles_for_viewport(
        self,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        z_index: int,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> list[tuple[DigitalSlideTile, QImage]]:
        conn = self._connection()
        self._initialize_schema()
        x2 = int(x) + int(width)
        y2 = int(y) + int(height)
        rows = conn.execute(
            """
            SELECT id, z_index, x, y, width, height, stage_x, stage_y, focus_z, sharpness, status, image_png, codec, quality
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
            if cancellation_requested is not None and cancellation_requested():
                break
            image = self._decode_tile_image(int(row["id"]), bytes(row["image_png"]), str(row["codec"] or "png"))
            if cancellation_requested is not None and cancellation_requested():
                break
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

    def viewport_coverage_mask(
        self,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        z_index: int,
    ):
        """Return valid stored-pixel coverage without decoding tile payloads."""

        import numpy as np

        width = max(1, int(width))
        height = max(1, int(height))
        x = int(x)
        y = int(y)
        x2 = x + width
        y2 = y + height
        conn = self._connection()
        self._initialize_schema()
        rows = conn.execute(
            """
            SELECT x, y, width, height
            FROM tiles
            WHERE z_index = ?
              AND x < ?
              AND y < ?
              AND (x + width) > ?
              AND (y + height) > ?
            ORDER BY id ASC
            """,
            (int(z_index), x2, y2, x, y),
        ).fetchall()
        coverage = np.zeros((height, width), dtype=bool)
        for row in rows:
            left = max(0, int(row["x"]) - x)
            top = max(0, int(row["y"]) - y)
            right = min(width, int(row["x"]) + int(row["width"]) - x)
            bottom = min(height, int(row["y"]) + int(row["height"]) - y)
            if right > left and bottom > top:
                coverage[top:bottom, left:right] = True
        return coverage

    def iter_tiles(self) -> Iterator[tuple[DigitalSlideTile, QImage, str, int | None]]:
        conn = self._connection()
        self._initialize_schema()
        rows = conn.execute(
            """
            SELECT id, z_index, x, y, width, height, stage_x, stage_y, focus_z, sharpness, status, image_png, codec, quality
            FROM tiles
            ORDER BY id ASC
            """
        )
        for row in rows:
            codec = normalize_tile_codec(str(row["codec"] or "png"))
            image = self._decode_tile_image(int(row["id"]), bytes(row["image_png"]), codec)
            if image.isNull():
                continue
            quality = row["quality"]
            yield (
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
                codec,
                int(quality) if quality is not None else None,
            )

    def render_viewport(
        self,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        z_index: int,
        blend_width: int = 0,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> QImage:
        output = QImage(max(1, int(width)), max(1, int(height)), QImage.Format.Format_RGB32)
        output.fill(QColor("#101820"))
        painter = QPainter(output)
        if not painter.isActive():
            return output
        covered_regions: list[QRect] = []
        blend_width = max(0, int(blend_width))

        def draw_sub_rect(target_rect: QRect, *, source_left: int, source_top: int, target_left: int, target_top: int) -> None:
            if target_rect.isEmpty():
                return
            painter.drawImage(
                QRectF(target_rect),
                image,
                QRectF(
                    source_left + target_rect.left() - target_left,
                    source_top + target_rect.top() - target_top,
                    target_rect.width(),
                    target_rect.height(),
                ),
            )

        for tile, image in self.read_tiles_for_viewport(
            x=x,
            y=y,
            width=width,
            height=height,
            z_index=z_index,
            cancellation_requested=cancellation_requested,
        ):
            if cancellation_requested is not None and cancellation_requested():
                break
            source_left = max(0, int(x) - tile.x)
            source_top = max(0, int(y) - tile.y)
            source_right = min(tile.width, int(x) + int(width) - tile.x)
            source_bottom = min(tile.height, int(y) + int(height) - tile.y)
            if source_right <= source_left or source_bottom <= source_top:
                continue
            target_left = tile.x + source_left - int(x)
            target_top = tile.y + source_top - int(y)
            target_rect = QRect(
                int(target_left),
                int(target_top),
                int(source_right - source_left),
                int(source_bottom - source_top),
            )
            if blend_width <= 0 or not covered_regions:
                draw_sub_rect(
                    target_rect,
                    source_left=source_left,
                    source_top=source_top,
                    target_left=target_left,
                    target_top=target_top,
                )
                covered_regions.append(target_rect)
                continue

            overlap_region = QRegion()
            for covered_rect in covered_regions:
                intersection = target_rect.intersected(covered_rect)
                if not intersection.isEmpty():
                    overlap_region = overlap_region.united(QRegion(intersection))
            if overlap_region.isEmpty():
                draw_sub_rect(
                    target_rect,
                    source_left=source_left,
                    source_top=source_top,
                    target_left=target_left,
                    target_top=target_top,
                )
                covered_regions.append(target_rect)
                continue

            edge_region = QRegion()
            edge_width = min(blend_width, max(1, target_rect.width()))
            edge_height = min(blend_width, max(1, target_rect.height()))
            edge_region = edge_region.united(QRegion(QRect(target_rect.left(), target_rect.top(), edge_width, target_rect.height())))
            edge_region = edge_region.united(QRegion(QRect(target_rect.right() - edge_width + 1, target_rect.top(), edge_width, target_rect.height())))
            edge_region = edge_region.united(QRegion(QRect(target_rect.left(), target_rect.top(), target_rect.width(), edge_height)))
            edge_region = edge_region.united(QRegion(QRect(target_rect.left(), target_rect.bottom() - edge_height + 1, target_rect.width(), edge_height)))
            blend_region = overlap_region.intersected(edge_region)
            opaque_region = QRegion(target_rect).subtracted(blend_region)
            painter.setOpacity(1.0)
            for rect in opaque_region:
                draw_sub_rect(
                    rect,
                    source_left=source_left,
                    source_top=source_top,
                    target_left=target_left,
                    target_top=target_top,
                )
            if not blend_region.isEmpty():
                painter.setOpacity(0.5)
                for rect in blend_region:
                    draw_sub_rect(
                        rect,
                        source_left=source_left,
                        source_top=source_top,
                        target_left=target_left,
                        target_top=target_top,
                    )
                painter.setOpacity(1.0)
            covered_regions.append(target_rect)
        painter.end()
        return output

    def render_overview(
        self,
        *,
        z_index: int,
        maximum_edge: int = 256,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> QImage:
        """Load a stored focus preview, generating it once for legacy slides."""

        max_edge = max(1, min(int(maximum_edge), 1024))
        if cancellation_requested is not None and cancellation_requested():
            return QImage()
        stored = self.read_focus_overview(
            z_index,
            maximum_edge=max_edge,
        )
        if not stored.isNull():
            return stored

        # Legacy slides do not contain focus_overviews. Generate a canonical
        # preview once, persist it when possible, and retain bounded streaming
        # as the compatibility path for existing files.
        rendered = self._render_overview_from_tiles(
            z_index=z_index,
            maximum_edge=max(max_edge, DIGITAL_SLIDE_OVERVIEW_MAX_EDGE),
            cancellation_requested=cancellation_requested,
        )
        if rendered.isNull():
            return rendered
        if cancellation_requested is not None and cancellation_requested():
            return QImage()
        try:
            self.write_focus_overviews({int(z_index): rendered})
        except Exception:  # noqa: BLE001 - optional migration must not hide preview
            pass
        return _bounded_overview_image(rendered, maximum_edge=max_edge)

    def _render_overview_from_tiles(
        self,
        *,
        z_index: int,
        maximum_edge: int,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> QImage:
        """Compose a bounded whole-slide preview without a full-size raster.

        Tile payloads are streamed from SQLite and painted directly into the
        small output image.  Memory use therefore depends on one decoded tile
        plus the overview, rather than on the dimensions of the complete
        digital slide.  Callers can cancel between tile decodes when a tab is
        closed or the requested focus plane changes.
        """

        manifest = self.read_manifest()
        max_edge = max(1, min(int(maximum_edge), 1024))
        scale = min(
            1.0,
            max_edge / max(1, int(manifest.width)),
            max_edge / max(1, int(manifest.height)),
        )
        output_width = max(1, int(round(manifest.width * scale)))
        output_height = max(1, int(round(manifest.height * scale)))
        if cancellation_requested is not None and cancellation_requested():
            return QImage()

        output = QImage(
            output_width,
            output_height,
            QImage.Format.Format_RGB32,
        )
        output.fill(QColor("#101820"))
        painter = QPainter(output)
        if not painter.isActive():
            return QImage()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        output_bounds = QRectF(output.rect())

        conn = self._connection()
        self._initialize_schema()
        rows = conn.execute(
            """
            SELECT x, y, width, height, image_png, codec
            FROM tiles
            WHERE z_index = ?
            ORDER BY id ASC
            """,
            (int(z_index),),
        )
        drawn_tiles = 0
        cancelled = False
        try:
            for row in rows:
                if cancellation_requested is not None and cancellation_requested():
                    cancelled = True
                    break
                image = image_bytes_to_qimage(
                    bytes(row["image_png"]),
                    codec=str(row["codec"] or "png"),
                )
                if cancellation_requested is not None and cancellation_requested():
                    cancelled = True
                    break
                if image.isNull():
                    continue
                target = QRectF(
                    float(row["x"]) * scale,
                    float(row["y"]) * scale,
                    float(row["width"]) * scale,
                    float(row["height"]) * scale,
                )
                if target.isEmpty() or not target.intersects(output_bounds):
                    continue
                painter.drawImage(target, image)
                drawn_tiles += 1
        finally:
            painter.end()

        if cancelled or drawn_tiles == 0:
            return QImage()
        return output


def _bounded_overview_image(image: QImage, *, maximum_edge: int) -> QImage:
    if image.isNull():
        return QImage()
    max_edge = max(1, min(int(maximum_edge), 1024))
    if image.width() <= max_edge and image.height() <= max_edge:
        return QImage(image)
    return image.scaled(
        max_edge,
        max_edge,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def copy_slide_file(source: str | Path, target: str | Path) -> Path:
    source_path = Path(source).expanduser()
    target_path = Path(target).expanduser()
    if source_path.resolve() == target_path.resolve():
        return target_path
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    store = DigitalSlideStore(source_path)
    try:
        store.open()
        return store.backup_to(target_path)
    finally:
        store.close()


def compress_slide_file(
    source: str | Path,
    target: str | Path,
    *,
    codec: str = DIGITAL_SLIDE_TILE_CODEC_JPEG,
    quality: int | None = 90,
    dynamic_focus_overview_enabled: bool = True,
    progress_callback: Any | None = None,
) -> Path:
    source_path = Path(source).expanduser()
    target_path = Path(target).expanduser()
    if source_path.resolve() == target_path.resolve():
        raise ValueError("压缩目标不能与源文件相同，请选择另存副本。")
    normalized_codec = normalize_tile_codec(codec)
    normalized_quality = normalize_jpeg_quality(quality) if normalized_codec == DIGITAL_SLIDE_TILE_CODEC_JPEG else None
    source_store = DigitalSlideStore(source_path)
    try:
        source_store.open()
        source_manifest = source_store.read_manifest()
        total = source_store.tile_count()
        metadata = dict(source_manifest.metadata)
        metadata["tile_codec"] = normalized_codec
        metadata["tile_quality"] = normalized_quality if normalized_codec == DIGITAL_SLIDE_TILE_CODEC_JPEG else None
        metadata["compressed_from"] = str(source_path)
        metadata["compressed_at"] = datetime.now().isoformat(timespec="seconds")
        metadata["compression_note"] = (
            "JPEG 压缩可能引入伪影；精确测量建议保留 PNG 无损切片。"
            if normalized_codec == DIGITAL_SLIDE_TILE_CODEC_JPEG
            else "PNG 无损重编码；不能恢复源文件中已经由 JPEG 丢失的细节。"
        )
        target_manifest = DigitalSlideManifest(
            version=source_manifest.version,
            width=source_manifest.width,
            height=source_manifest.height,
            viewport_width=source_manifest.viewport_width,
            viewport_height=source_manifest.viewport_height,
            focus_levels=list(source_manifest.focus_levels),
            tile_count=0,
            status=source_manifest.status,
            created_at=source_manifest.created_at,
            metadata=metadata,
        )
        overview_focus_indices = None
        if not dynamic_focus_overview_enabled and target_manifest.focus_levels:
            overview_focus_indices = {len(target_manifest.focus_levels) // 2}
        overview_accumulator = DigitalSlideOverviewAccumulator(
            target_manifest,
            focus_indices=overview_focus_indices,
        )
        with staged_path_for(target_path, suffix=".fdmslide.tmp") as staged_path:
            target_store: DigitalSlideStore | None = None
            try:
                target_store = DigitalSlideStore.create(staged_path, target_manifest)
                completed = 0
                for tile, image, _old_codec, _old_quality in source_store.iter_tiles():
                    target_store.write_tile(
                        tile,
                        image,
                        codec=normalized_codec,
                        quality=normalized_quality,
                        update_manifest=False,
                    )
                    overview_accumulator.add_tile(tile, image)
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, total)
                target_manifest.tile_count = target_store.tile_count()
                target_manifest.status = source_manifest.status
                target_store.write_manifest(target_manifest)
                target_store.write_focus_overviews(overview_accumulator.images())
                target_store.close()
                target_store = None
                _make_staged_database_standalone(staged_path)
                _assert_existing_sqlite_target_is_standalone(target_path)
                atomic_replace_file(staged_path, target_path)
            finally:
                if target_store is not None:
                    try:
                        target_store.close()
                    except Exception:
                        pass
                _remove_sqlite_sidecars(staged_path)
        return target_path
    finally:
        source_store.close()
