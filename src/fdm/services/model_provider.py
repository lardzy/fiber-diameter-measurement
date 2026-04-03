from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fdm.raster import RasterImage, RotatedROI


@dataclass(slots=True)
class ModelResult:
    mask: RasterImage
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelProvider:
    def load(self, model_path: str | Path) -> None:
        raise NotImplementedError

    def infer_roi(self, roi: RotatedROI) -> ModelResult | None:
        raise NotImplementedError

    def healthcheck(self) -> dict[str, Any]:
        raise NotImplementedError


class NullModelProvider(ModelProvider):
    def __init__(self) -> None:
        self.reason = "No ONNX model loaded."

    def load(self, model_path: str | Path) -> None:
        self.reason = f"Model loading is disabled: {model_path}"

    def infer_roi(self, roi: RotatedROI) -> ModelResult | None:
        return None

    def healthcheck(self) -> dict[str, Any]:
        return {"ready": False, "reason": self.reason}


class OnnxModelProvider(ModelProvider):
    """ONNX Runtime backed binary segmentation provider."""

    def __init__(self) -> None:
        self.model_path: Path | None = None
        self._session: Any | None = None
        self._input_name: str | None = None
        self._last_error: str | None = None

    def load(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            self._last_error = str(exc)
            raise RuntimeError(
                "onnxruntime and numpy are required to use ONNX models."
            ) from exc

        self.model_path = model_path
        self._session = ort.InferenceSession(
            model_path.as_posix(),
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model does not expose any input tensor.")
        self._input_name = inputs[0].name
        self._last_error = None

    def infer_roi(self, roi: RotatedROI) -> ModelResult | None:
        if self._session is None or self._input_name is None:
            return None
        try:
            import numpy as np
        except ImportError:
            return None

        width = roi.image.width
        height = roi.image.height
        array = np.asarray(roi.image.pixels, dtype=np.float32).reshape((1, 1, height, width))
        array /= 255.0
        outputs = self._session.run(None, {self._input_name: array})
        if not outputs:
            return None
        prediction = outputs[0]
        if prediction.ndim == 4:
            prediction = prediction[0, 0]
        elif prediction.ndim == 3:
            prediction = prediction[0]
        flat = prediction.reshape((height * width,))
        confidence = float(flat.mean())
        mask_pixels = np.where(flat >= 0.5, 255, 0).astype(int).tolist()
        mask = RasterImage(width=width, height=height, pixels=mask_pixels)
        return ModelResult(
            mask=mask,
            confidence=confidence,
            metadata={"source": "onnx", "model_path": self.model_path.as_posix() if self.model_path else ""},
        )

    def healthcheck(self) -> dict[str, Any]:
        return {
            "ready": self._session is not None,
            "reason": self._last_error or "",
            "model_path": self.model_path.as_posix() if self.model_path else "",
        }
