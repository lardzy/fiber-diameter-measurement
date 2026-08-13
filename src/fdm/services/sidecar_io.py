from __future__ import annotations

from pathlib import Path
import copy
import json
from dataclasses import dataclass

from fdm.atomic_io import atomic_write_json
from fdm.geometry import Line
from fdm.models import CalibrationSidecar, ImageDocument


@dataclass(frozen=True, slots=True)
class SidecarSaveResult:
    success: bool
    path: Path | None = None
    action: str = "skipped"
    message: str = ""

    def __bool__(self) -> bool:
        return self.success


class CalibrationSidecarIO:
    @staticmethod
    def sidecar_path_for_image(image_path: str | Path) -> Path:
        return Path(f"{Path(image_path)}.fdm.json")

    @classmethod
    def build_sidecar(cls, document: ImageDocument) -> CalibrationSidecar | None:
        if document.calibration is None:
            return None
        calibration_line = document.metadata.get("calibration_line")
        document.sidecar_path = str(cls.sidecar_path_for_image(document.path))
        return CalibrationSidecar(
            image_path=document.path,
            calibration=document.calibration,
            calibration_line=calibration_line if isinstance(calibration_line, Line) else Line.from_dict(calibration_line) if calibration_line else None,
        )

    @classmethod
    def save_document(cls, document: ImageDocument) -> SidecarSaveResult:
        # Legacy integrations may still assign ``document.calibration``
        # directly.  Detect that small scalar change at the persistence edge so
        # a failed write cannot incorrectly leave the document clean.
        document.ensure_external_calibration_change_is_dirty()
        if not document.uses_sidecar():
            document.mark_calibration_saved()
            return SidecarSaveResult(True, action="not_applicable")
        sidecar = cls.build_sidecar(document)
        output_path = Path(document.sidecar_path or cls.sidecar_path_for_image(document.path))
        if sidecar is None:
            try:
                if output_path.exists():
                    output_path.unlink()
            except OSError as exc:
                document.refresh_dirty_flags()
                return SidecarSaveResult(False, output_path, "delete_failed", str(exc))
            document.sidecar_path = output_path.as_posix()
            document.mark_calibration_saved()
            return SidecarSaveResult(True, output_path, "deleted")
        try:
            atomic_write_json(output_path, sidecar.to_dict(), ensure_ascii=False, indent=2)
        except (OSError, TypeError, ValueError) as exc:
            document.refresh_dirty_flags()
            return SidecarSaveResult(False, output_path, "write_failed", str(exc))
        document.sidecar_path = output_path.as_posix()
        document.mark_calibration_saved()
        return SidecarSaveResult(True, output_path, "written")

    @classmethod
    def load_document(cls, document: ImageDocument) -> bool:
        if not document.uses_sidecar():
            document.mark_calibration_saved()
            return False
        input_path = Path(document.sidecar_path or cls.sidecar_path_for_image(document.path))
        document.sidecar_path = input_path.as_posix()
        if not input_path.exists():
            return False
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            sidecar = CalibrationSidecar.from_dict(payload)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            document.calibration = None
            document.calibration_load_error = str(exc)
            document.calibration_load_payload = None
            return False
        except (KeyError, TypeError, ValueError) as exc:
            document.calibration = None
            document.calibration_load_error = str(exc)
            document.calibration_load_payload = copy.deepcopy(payload) if isinstance(payload, dict) else {"payload": payload}
            return False
        document.calibration_load_error = None
        document.calibration_load_payload = None
        document.calibration = sidecar.calibration
        if sidecar.calibration_line is not None:
            document.metadata["calibration_line"] = sidecar.calibration_line.to_dict()
        else:
            document.metadata.pop("calibration_line", None)
        document.mark_calibration_saved()
        return True

    @classmethod
    def export_document(cls, document: ImageDocument, output_path: str | Path) -> SidecarSaveResult:
        sidecar = cls.build_sidecar(document)
        if sidecar is None:
            return SidecarSaveResult(True, action="not_available")
        export_path = Path(output_path)
        try:
            atomic_write_json(export_path, sidecar.to_dict(), ensure_ascii=False, indent=2)
        except (OSError, TypeError, ValueError) as exc:
            return SidecarSaveResult(False, export_path, "write_failed", str(exc))
        return SidecarSaveResult(True, export_path, "written")
