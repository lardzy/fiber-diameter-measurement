from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from fdm.geometry import Point
from fdm.settings import bundle_resource_root


EDGE_SAM_ENCODER_FILENAME = "edge_sam_encoder.onnx"
EDGE_SAM_DECODER_FILENAME = "edge_sam_decoder.onnx"
EDGE_SAM_TARGET_LENGTH = 1024


@dataclass(slots=True)
class PromptSegmentationResult:
    polygon_px: list[Point]
    area_px: float
    metadata: dict[str, object]


@dataclass(slots=True)
class _EmbeddingEntry:
    image_embeddings: object
    original_size: tuple[int, int]


def edge_sam_runtime_root() -> Path:
    return bundle_resource_root() / "runtime" / "segment-anything" / "edge_sam"


def edge_sam_model_paths() -> tuple[Path, Path]:
    runtime_root = edge_sam_runtime_root()
    return (
        runtime_root / EDGE_SAM_ENCODER_FILENAME,
        runtime_root / EDGE_SAM_DECODER_FILENAME,
    )


class PromptSegmentationService:
    def __init__(
        self,
        *,
        encoder_path: str | Path | None = None,
        decoder_path: str | Path | None = None,
        target_length: int = EDGE_SAM_TARGET_LENGTH,
    ) -> None:
        default_encoder, default_decoder = edge_sam_model_paths()
        self._encoder_path = Path(encoder_path) if encoder_path is not None else default_encoder
        self._decoder_path = Path(decoder_path) if decoder_path is not None else default_decoder
        self._target_length = target_length
        self._encoder_session = None
        self._decoder_session = None
        self._encoder_input_name = ""
        self._decoder_input_names: dict[str, str] = {}
        self._embedding_cache: dict[str, _EmbeddingEntry] = {}

    @staticmethod
    def models_ready() -> bool:
        encoder_path, decoder_path = edge_sam_model_paths()
        return encoder_path.exists() and decoder_path.exists()

    def clear_cache(self) -> None:
        self._embedding_cache.clear()

    def predict_polygon(
        self,
        *,
        image,
        cache_key: str,
        positive_points: list[Point],
        negative_points: list[Point],
    ) -> PromptSegmentationResult:
        if not positive_points:
            return PromptSegmentationResult(
                polygon_px=[],
                area_px=0.0,
                metadata={"reason": "missing_positive_prompt"},
            )
        embedding = self._embedding_for_image(image, cache_key=cache_key)
        mask = self._predict_mask_from_embedding(
            embedding,
            positive_points=positive_points,
            negative_points=negative_points,
        )
        polygon = self._mask_to_polygon(mask)
        return PromptSegmentationResult(
            polygon_px=polygon,
            area_px=float(cv2.contourArea(self._points_to_contour(polygon))) if polygon else 0.0,
            metadata={
                "positive_points": len(positive_points),
                "negative_points": len(negative_points),
                "cache_size": len(self._embedding_cache),
                "cache_key": cache_key,
            },
        )

    def _ensure_sessions(self) -> None:
        if self._encoder_session is not None and self._decoder_session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for the magic segmentation tool.") from exc

        if not self._encoder_path.exists() or not self._decoder_path.exists():
            raise FileNotFoundError(
                "未找到 EdgeSAM 模型文件，请确认 runtime/segment-anything/edge_sam 中存在 encoder/decoder ONNX。"
            )

        self._encoder_session = ort.InferenceSession(
            self._encoder_path.as_posix(),
            providers=["CPUExecutionProvider"],
        )
        self._decoder_session = ort.InferenceSession(
            self._decoder_path.as_posix(),
            providers=["CPUExecutionProvider"],
        )
        inputs = self._encoder_session.get_inputs()
        if not inputs:
            raise RuntimeError("EdgeSAM encoder 未暴露输入张量。")
        self._encoder_input_name = inputs[0].name
        self._decoder_input_names = {item.name: item.name for item in self._decoder_session.get_inputs()}

    def _embedding_for_image(self, image, *, cache_key: str) -> _EmbeddingEntry:
        key = str(cache_key)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        cv_image = self._image_to_rgb_array(image)
        image_embeddings, original_size = self._run_encoder(cv_image)
        cached = _EmbeddingEntry(image_embeddings=image_embeddings, original_size=original_size)
        self._embedding_cache[key] = cached
        return cached

    def _image_to_rgb_array(self, image):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
        from PySide6.QtGui import QImage

        if image is None or not hasattr(image, "isNull") or image.isNull():
            raise RuntimeError("无法读取图片: 当前图像为空。")
        rgb = image.convertToFormat(QImage.Format.Format_RGB888)
        buffer = rgb.constBits()
        array = np.frombuffer(buffer, dtype=np.uint8, count=rgb.sizeInBytes())
        array = array.reshape((rgb.height(), rgb.bytesPerLine()))
        return array[:, : rgb.width() * 3].reshape((rgb.height(), rgb.width(), 3)).copy()

    def _run_encoder(self, cv_image):
        self._ensure_sessions()
        original_size = tuple(int(value) for value in cv_image.shape[:2])
        target_h, target_w = self._get_preprocess_shape(original_size[0], original_size[1], self._target_length)
        resized = cv2.resize(cv_image, (target_w, target_h))
        transformed = resized.transpose((2, 0, 1))
        transformed = transformed.astype("uint8", copy=False)
        transformed = transformed[None, ...]
        image_embeddings = self._encoder_session.run(
            None,
            {self._encoder_input_name: transformed},
        )[0]
        return image_embeddings, original_size

    def _predict_mask_from_embedding(
        self,
        embedding: _EmbeddingEntry,
        *,
        positive_points: list[Point],
        negative_points: list[Point],
    ):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
        self._ensure_sessions()
        prompt_points = [[point.x, point.y] for point in positive_points + negative_points]
        prompt_labels = [1.0] * len(positive_points) + [0.0] * len(negative_points)
        point_coords = np.array(prompt_points, dtype=np.float32)
        point_labels = np.array(prompt_labels, dtype=np.float32)
        point_coords = self._apply_coords(point_coords, embedding.original_size).astype(np.float32)
        point_coords = np.expand_dims(point_coords, axis=0)
        point_labels = np.expand_dims(point_labels, axis=0)
        input_dict = {
            self._decoder_input_names.get("image_embeddings", "image_embeddings"): embedding.image_embeddings,
            self._decoder_input_names.get("point_coords", "point_coords"): point_coords,
            self._decoder_input_names.get("point_labels", "point_labels"): point_labels,
        }
        outputs = self._decoder_session.run(None, input_dict)
        masks = None
        for output in outputs:
            if getattr(output, "ndim", 0) >= 3:
                masks = output
                break
        if masks is None:
            raise RuntimeError("EdgeSAM decoder 未返回掩码张量。")
        scores = self._calculate_stability_score(masks[0], 0.0, 1.0)
        max_score_index = int(scores.argmax())
        mask = masks[0, max_score_index]
        input_size = self._get_preprocess_shape(*embedding.original_size, self._target_length)
        return self._postprocess_masks(mask, input_size=input_size, original_size=embedding.original_size) > 0.0

    def _mask_to_polygon(self, mask) -> list[Point]:
        mask_uint8 = mask.astype("uint8") * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 9.0:
            return []
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            approx = contour
        polygon = [Point(float(point[0][0]), float(point[0][1])) for point in approx]
        if len(polygon) < 3:
            return []
        return polygon

    @staticmethod
    def _points_to_contour(points: list[Point]):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
        if not points:
            return np.zeros((0, 1, 2), dtype=np.float32)
        return np.array([[[point.x, point.y]] for point in points], dtype=np.float32)

    @staticmethod
    def _get_preprocess_shape(old_h: int, old_w: int, long_side_length: int) -> tuple[int, int]:
        scale = long_side_length * 1.0 / max(old_h, old_w)
        new_h, new_w = old_h * scale, old_w * scale
        return int(new_h + 0.5), int(new_w + 0.5)

    def _apply_coords(self, coords, original_size: tuple[int, int]):
        old_h, old_w = original_size
        new_h, new_w = self._get_preprocess_shape(old_h, old_w, self._target_length)
        adjusted = coords.copy().astype(float)
        adjusted[..., 0] = adjusted[..., 0] * (new_w / old_w)
        adjusted[..., 1] = adjusted[..., 1] * (new_h / old_h)
        return adjusted

    @staticmethod
    def _calculate_stability_score(masks, mask_threshold: float, threshold_offset: float):
        high_threshold_mask = masks > (mask_threshold + threshold_offset)
        low_threshold_mask = masks > (mask_threshold - threshold_offset)
        intersections = high_threshold_mask & low_threshold_mask
        unions = high_threshold_mask | low_threshold_mask
        return intersections.sum(axis=(-1, -2), dtype="int32") / unions.sum(axis=(-1, -2), dtype="int32")

    def _postprocess_masks(
        self,
        mask,
        *,
        input_size: tuple[int, int],
        original_size: tuple[int, int],
    ):
        img_size = self._target_length
        resized = cv2.resize(mask, (img_size, img_size))
        cropped = resized[..., : input_size[0], : input_size[1]]
        return cv2.resize(cropped, original_size[::-1])
