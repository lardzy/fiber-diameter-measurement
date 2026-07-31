from __future__ import annotations

from dataclasses import dataclass
import json

from fdm.models import ImageDocument
from fdm.raster import RasterPlane
from fdm.services.raster_io import RasterMetadata


@dataclass(frozen=True, slots=True)
class ImageInformationSnapshot:
    document_id: str
    display_name: str
    source_path: str
    width: int
    height: int
    pixel_type: str
    channel_count: int
    has_alpha: bool
    byte_count: int
    pixel_sha256: str
    source_type: str
    source_format: str = ""
    source_mode: str = ""
    icc_profile_bytes: int = 0
    icc_profile_sha256: str = ""
    dpi_x: float | None = None
    dpi_y: float | None = None
    calibration_mode: str = ""
    pixels_per_unit: float | None = None
    calibration_unit: str = ""
    derivation_source_document_id: str = ""
    derivation_step_count: int = 0
    derivation_result_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "display_name": self.display_name,
            "source_path": self.source_path,
            "image_size": [self.width, self.height],
            "pixel_type": self.pixel_type,
            "channel_count": self.channel_count,
            "has_alpha": self.has_alpha,
            "byte_count": self.byte_count,
            "pixel_sha256": self.pixel_sha256,
            "source_type": self.source_type,
            "source_format": self.source_format,
            "source_mode": self.source_mode,
            "icc_profile_bytes": self.icc_profile_bytes,
            "icc_profile_sha256": self.icc_profile_sha256,
            "dpi_x": self.dpi_x,
            "dpi_y": self.dpi_y,
            "calibration_mode": self.calibration_mode,
            "pixels_per_unit": self.pixels_per_unit,
            "calibration_unit": self.calibration_unit,
            "derivation_source_document_id": self.derivation_source_document_id,
            "derivation_step_count": self.derivation_step_count,
            "derivation_result_sha256": self.derivation_result_sha256,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )


def build_image_information_snapshot(
    document: ImageDocument,
    plane: RasterPlane,
    *,
    display_name: str,
    metadata: RasterMetadata | None = None,
) -> ImageInformationSnapshot:
    if not isinstance(document, ImageDocument):
        raise TypeError("document 必须是 ImageDocument")
    if not isinstance(plane, RasterPlane):
        raise TypeError("plane 必须是 RasterPlane")
    calibration = document.calibration
    derivation = document.derivation
    return ImageInformationSnapshot(
        document_id=document.id,
        display_name=str(display_name or "").strip() or document.id,
        source_path=str(document.path or ""),
        width=plane.width,
        height=plane.height,
        pixel_type=plane.pixel_type.value,
        channel_count=plane.pixel_type.channel_count,
        has_alpha=plane.pixel_type.has_alpha,
        byte_count=plane.byte_count,
        pixel_sha256=plane.sha256(),
        source_type=str(document.source_type or ""),
        source_format=(metadata.source_format if metadata is not None else ""),
        source_mode=(metadata.source_mode if metadata is not None else ""),
        icc_profile_bytes=(
            len(metadata.icc_profile or b"") if metadata is not None else 0
        ),
        icc_profile_sha256=(
            metadata.icc_profile_sha256 if metadata is not None else ""
        ),
        dpi_x=(metadata.dpi_x if metadata is not None else None),
        dpi_y=(metadata.dpi_y if metadata is not None else None),
        calibration_mode=(calibration.mode if calibration is not None else ""),
        pixels_per_unit=(
            float(calibration.pixels_per_unit)
            if calibration is not None
            else None
        ),
        calibration_unit=(calibration.unit if calibration is not None else ""),
        derivation_source_document_id=(
            derivation.source_document_id if derivation is not None else ""
        ),
        derivation_step_count=(
            len(derivation.recipe.operations) if derivation is not None else 0
        ),
        derivation_result_sha256=(
            derivation.result_sha256 or "" if derivation is not None else ""
        ),
    )


__all__ = [
    "ImageInformationSnapshot",
    "build_image_information_snapshot",
]
