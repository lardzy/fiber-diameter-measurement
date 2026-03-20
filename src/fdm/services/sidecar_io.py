from __future__ import annotations

from pathlib import Path
import json

from fdm.geometry import Line
from fdm.models import CalibrationSidecar, ImageDocument


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
    def save_document(cls, document: ImageDocument) -> Path | None:
        sidecar = cls.build_sidecar(document)
        output_path = Path(document.sidecar_path or cls.sidecar_path_for_image(document.path))
        if sidecar is None:
            if output_path.exists():
                output_path.unlink()
            document.sidecar_path = output_path.as_posix()
            document.mark_calibration_saved()
            return None
        output_path.write_text(
            json.dumps(sidecar.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        document.sidecar_path = output_path.as_posix()
        document.mark_calibration_saved()
        return output_path

    @classmethod
    def load_document(cls, document: ImageDocument) -> bool:
        input_path = Path(document.sidecar_path or cls.sidecar_path_for_image(document.path))
        document.sidecar_path = input_path.as_posix()
        if not input_path.exists():
            return False
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        sidecar = CalibrationSidecar.from_dict(payload)
        document.calibration = sidecar.calibration
        if sidecar.calibration_line is not None:
            document.metadata["calibration_line"] = sidecar.calibration_line.to_dict()
        else:
            document.metadata.pop("calibration_line", None)
        document.mark_calibration_saved()
        return True

    @classmethod
    def export_document(cls, document: ImageDocument, output_path: str | Path) -> Path | None:
        sidecar = cls.build_sidecar(document)
        if sidecar is None:
            return None
        export_path = Path(output_path)
        export_path.write_text(
            json.dumps(sidecar.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return export_path
