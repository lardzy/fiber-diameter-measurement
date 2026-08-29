from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PySide6.QtGui import QImage

from fdm.geometry import Line, Point
from fdm.models import ImageDocument
from fdm.services.digital_slide_store import DigitalSlideStore


def _qimage_content_version(image: QImage) -> str:
    normalized = image.convertToFormat(QImage.Format.Format_RGBA8888)
    digest = hashlib.sha256()
    digest.update(f"{normalized.width()}x{normalized.height()}:rgba8888:".encode("ascii"))
    digest.update(normalized.constBits())
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class SegmentationSourceSnapshot:
    """Immutable, native-pixel input used by interactive segmentation.

    Segmentation always runs in the local coordinate system of ``image``.
    ``origin_px`` is the only translation boundary between that local raster
    and the document coordinate system.  Keeping this mapping explicit avoids
    both full-slide allocations and accidental double translations.
    """

    document_id: str
    image: QImage
    origin_px: Point
    focus_index: int | None
    source_kind: str
    source_identity: str
    source_version: str
    valid_coverage: np.ndarray | None

    def __post_init__(self) -> None:
        image = QImage(self.image)
        if image.isNull():
            raise ValueError("分割来源图像为空。")
        object.__setattr__(self, "image", image)
        if self.valid_coverage is not None:
            coverage = np.asarray(self.valid_coverage, dtype=bool)
            expected = (image.height(), image.width())
            if coverage.shape != expected:
                raise ValueError(
                    f"分割来源覆盖掩码尺寸不匹配：{coverage.shape} != {expected}。"
                )
            frozen_coverage = np.ascontiguousarray(coverage)
            frozen_coverage.setflags(write=False)
            object.__setattr__(self, "valid_coverage", frozen_coverage)

    @property
    def width(self) -> int:
        return int(self.image.width())

    @property
    def height(self) -> int:
        return int(self.image.height())

    @property
    def cache_key(self) -> str:
        focus = "image" if self.focus_index is None else f"z{self.focus_index}"
        return (
            f"{self.document_id}:{self.source_kind}:{self.source_identity}:"
            f"{self.source_version}:{focus}:"
            f"{int(round(self.origin_px.x))},{int(round(self.origin_px.y))}:"
            f"{self.width}x{self.height}:{int(self.image.cacheKey())}"
        )

    @property
    def global_bounds(self) -> tuple[int, int, int, int]:
        x0 = int(round(self.origin_px.x))
        y0 = int(round(self.origin_px.y))
        return x0, y0, x0 + self.width, y0 + self.height

    def to_local_point(self, point: Point) -> Point:
        return Point(point.x - self.origin_px.x, point.y - self.origin_px.y)

    def to_global_point(self, point: Point) -> Point:
        return Point(point.x + self.origin_px.x, point.y + self.origin_px.y)

    def to_local_points(self, points: Iterable[Point]) -> list[Point]:
        return [self.to_local_point(point) for point in points]

    def to_global_points(self, points: Iterable[Point]) -> list[Point]:
        return [self.to_global_point(point) for point in points]

    def to_global_rings(self, rings: Iterable[Iterable[Point]]) -> list[list[Point]]:
        return [self.to_global_points(ring) for ring in rings]

    def to_local_box(
        self,
        box: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        if box is None:
            return None
        x0, y0, x1, y1 = box
        origin_x = int(round(self.origin_px.x))
        origin_y = int(round(self.origin_px.y))
        local = (
            max(0, min(self.width, int(x0) - origin_x)),
            max(0, min(self.height, int(y0) - origin_y)),
            max(0, min(self.width, int(x1) - origin_x)),
            max(0, min(self.height, int(y1) - origin_y)),
        )
        if local[2] <= local[0] or local[3] <= local[1]:
            return None
        return local

    def contains_global_point(self, point: Point, *, require_coverage: bool = True) -> bool:
        local = self.to_local_point(point)
        if not (0.0 <= local.x < self.width and 0.0 <= local.y < self.height):
            return False
        x = int(math.floor(local.x))
        y = int(math.floor(local.y))
        if require_coverage and self.valid_coverage is not None:
            return bool(self.valid_coverage[y, x])
        return True

    def source_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": self.source_kind,
            "identity": self.source_identity,
            "version": self.source_version,
            "focus_index": self.focus_index,
            "origin_px": [float(self.origin_px.x), float(self.origin_px.y)],
            "size_px": [self.width, self.height],
            "cache_key": self.cache_key,
            "coverage_complete": (
                True
                if self.valid_coverage is None
                else bool(self.valid_coverage.all())
            ),
        }

    def translate_line_to_global(self, line: Line | None) -> Line | None:
        if line is None:
            return None
        return Line(
            start=self.to_global_point(line.start),
            end=self.to_global_point(line.end),
        )


def image_segmentation_snapshot(
    document: ImageDocument,
    image: QImage,
    *,
    source_version: str = "",
) -> SegmentationSourceSnapshot:
    if image is None or image.isNull():
        raise ValueError("当前图片尚未完成加载。")
    identity = str(document.resolved_path()) if document.path else document.id
    version = str(source_version or _qimage_content_version(image))
    return SegmentationSourceSnapshot(
        document_id=document.id,
        image=QImage(image),
        origin_px=Point(0.0, 0.0),
        focus_index=None,
        source_kind="image",
        source_identity=identity,
        source_version=version,
        valid_coverage=None,
    )


def digital_slide_segmentation_snapshot(
    document: ImageDocument,
    store: DigitalSlideStore,
    *,
    origin_px: Point,
    width: int,
    height: int,
    focus_index: int,
) -> SegmentationSourceSnapshot:
    manifest = store.read_manifest()
    metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
    try:
        blend_width = int(metadata.get("blend_width", 0) or 0)
    except (TypeError, ValueError):
        blend_width = 0
    x = int(round(origin_px.x))
    y = int(round(origin_px.y))
    if not (0 <= x < int(manifest.width) and 0 <= y < int(manifest.height)):
        raise ValueError("分割视野原点超出数字切片范围。")
    if not (0 <= int(focus_index) < max(1, len(manifest.focus_levels))):
        raise ValueError("分割焦层超出数字切片范围。")
    width = max(1, min(int(width), max(1, int(manifest.width) - x)))
    height = max(1, min(int(height), max(1, int(manifest.height) - y)))
    image = store.render_viewport(
        x=x,
        y=y,
        width=width,
        height=height,
        z_index=int(focus_index),
        blend_width=blend_width,
    )
    coverage = store.viewport_coverage_mask(
        x=x,
        y=y,
        width=width,
        height=height,
        z_index=int(focus_index),
    )
    if not bool(coverage.any()):
        raise ValueError("当前焦层与视野没有可用于分割的有效图块。")
    version = _qimage_content_version(image)
    return SegmentationSourceSnapshot(
        document_id=document.id,
        image=image,
        origin_px=Point(float(x), float(y)),
        focus_index=int(focus_index),
        source_kind="digital_slide_viewport",
        source_identity=str(Path(store.path)),
        source_version=version,
        valid_coverage=coverage,
    )
