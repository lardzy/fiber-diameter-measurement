from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2

from fdm.geometry import Point
from fdm.runtime_logging import append_runtime_log
from fdm.settings import MagicSegmentModelVariant, bundle_resource_root


EDGE_SAM_ENCODER_FILENAME = "edge_sam_encoder.onnx"
EDGE_SAM_DECODER_FILENAME = "edge_sam_decoder.onnx"
EDGE_SAM_3X_ENCODER_FILENAME = "edge_sam_3x_encoder.onnx"
EDGE_SAM_3X_DECODER_FILENAME = "edge_sam_3x_decoder.onnx"
EDGE_SAM_TARGET_LENGTH = 1024
EDGE_SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
EDGE_SAM_PIXEL_STD = (58.395, 57.12, 57.375)


@dataclass(slots=True)
class PromptSegmentationResult:
    mask: object | None
    polygon_px: list[Point]
    area_px: float
    metadata: dict[str, object]


@dataclass(slots=True)
class _EmbeddingEntry:
    image_embeddings: object
    original_size: tuple[int, int]


def _normalize_model_variant(model_variant: str | None) -> str:
    token = str(model_variant or "").strip()
    if token in {
        MagicSegmentModelVariant.EDGE_SAM,
        MagicSegmentModelVariant.EDGE_SAM_3X,
    }:
        return token
    return MagicSegmentModelVariant.EDGE_SAM


def edge_sam_runtime_root(model_variant: str = MagicSegmentModelVariant.EDGE_SAM) -> Path:
    normalized = _normalize_model_variant(model_variant)
    folder = "edge_sam_3x" if normalized == MagicSegmentModelVariant.EDGE_SAM_3X else "edge_sam"
    return bundle_resource_root() / "runtime" / "segment-anything" / folder


def edge_sam_model_paths(
    model_variant: str = MagicSegmentModelVariant.EDGE_SAM,
) -> tuple[Path, Path]:
    normalized = _normalize_model_variant(model_variant)
    runtime_root = edge_sam_runtime_root(normalized)
    if normalized == MagicSegmentModelVariant.EDGE_SAM_3X:
        return (
            runtime_root / EDGE_SAM_3X_ENCODER_FILENAME,
            runtime_root / EDGE_SAM_3X_DECODER_FILENAME,
        )
    return (
        runtime_root / EDGE_SAM_ENCODER_FILENAME,
        runtime_root / EDGE_SAM_DECODER_FILENAME,
    )


def magic_segment_model_label(model_variant: str) -> str:
    normalized = _normalize_model_variant(model_variant)
    if normalized == MagicSegmentModelVariant.EDGE_SAM_3X:
        return "EdgeSAM-3x"
    return "EdgeSAM"


def resolve_magic_segment_model_variant(
    requested_variant: str | None,
) -> tuple[str, str | None]:
    normalized = _normalize_model_variant(requested_variant)
    requested_paths = edge_sam_model_paths(normalized)
    if all(path.exists() for path in requested_paths):
        return normalized, None
    if normalized != MagicSegmentModelVariant.EDGE_SAM:
        fallback_paths = edge_sam_model_paths(MagicSegmentModelVariant.EDGE_SAM)
        if all(path.exists() for path in fallback_paths):
            return (
                MagicSegmentModelVariant.EDGE_SAM,
                "未找到 EdgeSAM-3x 模型文件，已自动回退到标准 EdgeSAM。",
            )
    return normalized, None


def magic_mask_area_px(mask) -> float:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    if mask is None:
        return 0.0
    return float(np.count_nonzero(mask))


def magic_mask_to_polygon(mask) -> list[Point]:
    if mask is None:
        return []
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


def normalize_magic_draft_mask(mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    if mask is None:
        return None
    working = np.asarray(mask, dtype=bool)
    if not np.any(working):
        return None
    return working.copy()


def finalize_magic_subtraction_mask(primary_mask, subtract_mask) -> tuple[object | None, dict[str, object]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    primary = normalize_magic_draft_mask(primary_mask)
    stats = {
        "opened_holes": 0,
        "discarded_fragments": False,
        "result_empty": False,
        "had_intersection": False,
    }
    if primary is None:
        stats["result_empty"] = True
        return None, stats
    subtract = normalize_magic_draft_mask(subtract_mask)
    if subtract is None:
        return primary.copy(), stats
    intersection = primary & subtract
    if not np.any(intersection):
        return primary.copy(), stats
    stats["had_intersection"] = True
    result = primary & ~subtract
    if not np.any(result):
        stats["result_empty"] = True
        return None, stats
    subtract_inside_primary = not np.any(subtract & ~primary)
    if subtract_inside_primary:
        result, opened_holes = _open_internal_holes(result)
        stats["opened_holes"] = opened_holes
    result, discarded_fragments = _keep_largest_component(result)
    stats["discarded_fragments"] = discarded_fragments
    if not np.any(result):
        stats["result_empty"] = True
        return None, stats
    return result, stats


def _open_internal_holes(mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    mask_uint8 = np.asarray(mask, dtype=np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask_uint8.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return mask_uint8.astype(bool), 0
    opened_holes = 0
    for index, relation in enumerate(hierarchy[0]):
        parent_index = int(relation[3])
        if parent_index < 0:
            continue
        hole = contours[index].reshape(-1, 2)
        outer = contours[parent_index].reshape(-1, 2)
        if hole.size == 0 or outer.size == 0:
            continue
        start, end = _nearest_contour_pair(hole, outer)
        cv2.line(mask_uint8, start, end, 0, 1)
        opened_holes += 1
    return mask_uint8.astype(bool), opened_holes


def _nearest_contour_pair(inner_points, outer_points) -> tuple[tuple[int, int], tuple[int, int]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    best_distance = None
    best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    outer = np.asarray(outer_points, dtype=np.int32)
    for inner in np.asarray(inner_points, dtype=np.int32):
        deltas = outer - inner
        distances = np.einsum("ij,ij->i", deltas, deltas)
        nearest_index = int(distances.argmin())
        candidate_distance = int(distances[nearest_index])
        if best_distance is None or candidate_distance < best_distance:
            best_distance = candidate_distance
            best_pair = (
                (int(inner[0]), int(inner[1])),
                (int(outer[nearest_index][0]), int(outer[nearest_index][1])),
            )
    if best_pair is None:
        return (0, 0), (0, 0)
    return best_pair


def _keep_largest_component(mask):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by the app
        raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if component_count <= 2:
        return mask_uint8.astype(bool), False
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(areas.argmax()) + 1
    return labels == largest_label, True


class PromptSegmentationService:
    def __init__(
        self,
        *,
        encoder_path: str | Path | None = None,
        decoder_path: str | Path | None = None,
        model_variant: str = MagicSegmentModelVariant.EDGE_SAM,
        target_length: int = EDGE_SAM_TARGET_LENGTH,
        max_cache_entries: int = 2,
    ) -> None:
        normalized_variant = _normalize_model_variant(model_variant)
        default_encoder, default_decoder = edge_sam_model_paths(normalized_variant)
        self._model_variant = normalized_variant
        self._encoder_path = Path(encoder_path) if encoder_path is not None else default_encoder
        self._decoder_path = Path(decoder_path) if decoder_path is not None else default_decoder
        self._target_length = target_length
        self._max_cache_entries = max(1, int(max_cache_entries))
        self._encoder_session = None
        self._decoder_session = None
        self._encoder_input_name = ""
        self._encoder_input_type = "tensor(uint8)"
        self._encoder_input_shape: tuple[object, ...] = ()
        self._decoder_input_names: dict[str, str] = {}
        self._embedding_cache: OrderedDict[str, _EmbeddingEntry] = OrderedDict()

    @staticmethod
    def models_ready(model_variant: str = MagicSegmentModelVariant.EDGE_SAM) -> bool:
        encoder_path, decoder_path = edge_sam_model_paths(model_variant)
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
                mask=None,
                polygon_px=[],
                area_px=0.0,
                metadata={"reason": "missing_positive_prompt"},
            )
        cv_image = self._image_to_rgb_array(image)
        embedding = self._embedding_for_rgb_array(cv_image, cache_key=cache_key)
        mask = self._predict_mask_from_embedding(
            embedding,
            positive_points=positive_points,
            negative_points=negative_points,
        )
        polygon = magic_mask_to_polygon(mask)
        return PromptSegmentationResult(
            mask=mask.copy() if mask is not None else None,
            polygon_px=polygon,
            area_px=magic_mask_area_px(mask),
            metadata={
                "positive_points": len(positive_points),
                "negative_points": len(negative_points),
                "cache_size": len(self._embedding_cache),
                "cache_key": cache_key,
                "model_variant": self._model_variant,
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
                f"未找到 {magic_segment_model_label(self._model_variant)} 模型文件，请确认对应 runtime 目录中存在 encoder/decoder ONNX。"
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
            raise RuntimeError(f"{magic_segment_model_label(self._model_variant)} encoder 未暴露输入张量。")
        self._encoder_input_name = inputs[0].name
        self._encoder_input_type = str(getattr(inputs[0], "type", "") or "tensor(uint8)")
        self._encoder_input_shape = tuple(getattr(inputs[0], "shape", ()) or ())
        self._decoder_input_names = {item.name: item.name for item in self._decoder_session.get_inputs()}

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

    def _embedding_for_image(self, image, *, cache_key: str) -> _EmbeddingEntry:
        key = str(cache_key)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            self._embedding_cache.move_to_end(key)
            return cached
        cv_image = self._image_to_rgb_array(image)
        return self._embedding_for_rgb_array(cv_image, cache_key=cache_key)

    def _embedding_for_rgb_array(self, cv_image, *, cache_key: str) -> _EmbeddingEntry:
        key = str(cache_key)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            self._embedding_cache.move_to_end(key)
            return cached
        started_at = perf_counter()
        image_embeddings, original_size = self._run_encoder(cv_image)
        cached = _EmbeddingEntry(image_embeddings=image_embeddings, original_size=original_size)
        self._embedding_cache[key] = cached
        self._embedding_cache.move_to_end(key)
        while len(self._embedding_cache) > self._max_cache_entries:
            self._embedding_cache.popitem(last=False)
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        if elapsed_ms >= 80.0:
            append_runtime_log(
                "Magic segmentation preprocess",
                (
                    f"elapsed_ms={elapsed_ms:.2f}, "
                    f"cache_size={len(self._embedding_cache)}, "
                    f"image_size={original_size[1]}x{original_size[0]}, "
                    f"model_variant={self._model_variant}"
                ),
            )
        return cached

    def _run_encoder(self, cv_image):
        self._ensure_sessions()
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for the magic segmentation tool.") from exc
        original_size = tuple(int(value) for value in cv_image.shape[:2])
        target_length = self._effective_target_length()
        target_h, target_w = self._get_preprocess_shape(original_size[0], original_size[1], target_length)
        resized = cv2.resize(cv_image, (target_w, target_h))
        transformed = resized.transpose((2, 0, 1))
        if self._requires_external_resize_norm_pad():
            transformed = self._normalize_and_pad_encoder_input(
                transformed,
                np,
                target_size=target_length,
                input_size=(target_h, target_w),
            )
        else:
            transformed = self._cast_encoder_input(transformed, np)
        transformed = transformed[None, ...]
        image_embeddings = self._encoder_session.run(
            None,
            {self._encoder_input_name: transformed},
        )[0]
        return image_embeddings, original_size

    def _effective_target_length(self) -> int:
        fixed_square_size = self._fixed_square_encoder_size()
        if fixed_square_size is not None:
            return fixed_square_size
        return self._target_length

    def _fixed_square_encoder_size(self) -> int | None:
        if len(self._encoder_input_shape) < 4:
            return None
        height = self._encoder_input_shape[2]
        width = self._encoder_input_shape[3]
        if isinstance(height, int) and isinstance(width, int) and height > 0 and height == width:
            return int(height)
        return None

    def _requires_external_resize_norm_pad(self) -> bool:
        fixed_square_size = self._fixed_square_encoder_size()
        input_type = str(self._encoder_input_type or "").strip().lower()
        return fixed_square_size is not None and input_type in {
            "tensor(float)",
            "tensor(float16)",
            "tensor(float32)",
            "tensor(double)",
        }

    def _cast_encoder_input(self, transformed, np):
        input_type = str(self._encoder_input_type or "").strip().lower()
        if input_type == "tensor(uint8)":
            return np.ascontiguousarray(transformed.astype(np.uint8, copy=False))
        if input_type in {"tensor(float)", "tensor(float32)"}:
            return np.ascontiguousarray(transformed.astype(np.float32, copy=False))
        if input_type == "tensor(float16)":
            return np.ascontiguousarray(transformed.astype(np.float16, copy=False))
        if input_type == "tensor(double)":
            return np.ascontiguousarray(transformed.astype(np.float64, copy=False))
        if input_type == "tensor(int64)":
            return np.ascontiguousarray(transformed.astype(np.int64, copy=False))
        raise RuntimeError(
            f"{magic_segment_model_label(self._model_variant)} encoder 输入类型暂不支持: {self._encoder_input_type}"
        )

    def _normalize_and_pad_encoder_input(
        self,
        transformed,
        np,
        *,
        target_size: int,
        input_size: tuple[int, int],
    ):
        normalized = transformed.astype(np.float32, copy=False)
        mean = np.asarray(EDGE_SAM_PIXEL_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(EDGE_SAM_PIXEL_STD, dtype=np.float32).reshape(3, 1, 1)
        normalized = (normalized - mean) / std
        padded = np.zeros((3, target_size, target_size), dtype=np.float32)
        padded[:, : input_size[0], : input_size[1]] = normalized
        return self._cast_encoder_input(padded, np)

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
            raise RuntimeError(f"{magic_segment_model_label(self._model_variant)} decoder 未返回掩码张量。")
        scores = self._calculate_stability_score(masks[0], 0.0, 1.0)
        max_score_index = int(scores.argmax())
        mask = masks[0, max_score_index]
        input_size = self._get_preprocess_shape(*embedding.original_size, self._effective_target_length())
        return self._postprocess_masks(mask, input_size=input_size, original_size=embedding.original_size) > 0.0

    def _mask_to_polygon(self, mask) -> list[Point]:
        return magic_mask_to_polygon(mask)

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
        new_h, new_w = self._get_preprocess_shape(old_h, old_w, self._effective_target_length())
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
        img_size = self._effective_target_length()
        resized = cv2.resize(mask, (img_size, img_size))
        cropped = resized[..., : input_size[0], : input_size[1]]
        return cv2.resize(cropped, original_size[::-1])
