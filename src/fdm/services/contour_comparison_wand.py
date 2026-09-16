"""Standard magic-wand inference for the standalone contour workspace.

One instance belongs to its serial background worker. Only two source-image
embeddings are retained; changing a mask never changes the source cache key.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from uuid import uuid4
import weakref

import numpy as np

from fdm.services.contour_comparison import ContourFrame, mask_warnings


def apply_wand_mask(frame: ContourFrame, prediction, *, operation="replace", base_mask=None):
    """Combine native masks, retaining holes and disconnected manual regions."""
    from fdm.services.mask_region import MaskRegion

    if isinstance(prediction, MaskRegion):
        if prediction.extent != frame.mask.shape:
            raise ValueError("魔棒结果与原照片尺寸不一致。")
        prediction = prediction.to_full_mask()
    if prediction is None:
        raise ValueError("此处没有识别到轮廓，请在物体内部重新点选。")
    selected = np.asarray(prediction, dtype=bool)
    if selected.shape != frame.mask.shape:
        raise ValueError("魔棒结果与原照片尺寸不一致。")
    selected = selected & (frame.rgba[:, :, 3] > 0)
    if not selected.any():
        raise ValueError("此处没有识别到轮廓，请在物体内部重新点选。")
    base = frame.mask if base_mask is None else base_mask
    if base.shape != frame.mask.shape or base.dtype != np.bool_:
        raise ValueError("魔棒修正的底稿与照片不一致。")
    if operation == "replace":
        mask = selected
    elif operation == "add":
        mask = base | selected
    elif operation == "remove":
        mask = base & ~selected
    else:
        raise ValueError("未知的魔棒轮廓操作。")
    warnings = list(mask_warnings(mask))
    warnings.append("魔棒为建议轮廓，请复核标尺遮挡、毛边及漏选部分。")
    # Model masks can stop a few pixels short of a cropped photograph's border;
    # this is a review hint, never a reason to extrapolate or extend geometry.
    margin = max(2, min(32, round(min(mask.shape) * .01)))
    if not any("截断" in item for item in warnings) and any(edge.any() for edge in (mask[:margin], mask[-margin:], mask[:, :margin], mask[:, -margin:])):
        warnings.append("轮廓接近照片边界，可能截断；端部结果需复核。")
    return replace(frame, mask=mask, edited=True, warnings=tuple(warnings))


class ContourWandService:
    def __init__(self, model_variant="edge_sam_3x"):
        self.model_variant = model_variant
        self._backend = None
        self._sources = OrderedDict()

    def _source_key(self, pixels):
        identity = id(pixels)
        previous = self._sources.get(identity)
        if previous is None or previous[0]() is not pixels:
            previous = (weakref.ref(pixels), "contour-comparison:" + uuid4().hex)
            self._sources[identity] = previous
        self._sources.move_to_end(identity)
        while len(self._sources) > 2:
            self._sources.popitem(last=False)
        return previous[1]

    def predict(self, frame, positive, negative, *, operation="replace", base_mask=None, token=None):
        # Lazy import/model loading keeps ordinary comparison startup lightweight.
        from PySide6.QtGui import QImage
        from fdm.geometry import Point
        from fdm.services.prompt_segmentation import (
            create_interactive_segmentation_service, resolve_magic_segment_model_variant,
        )

        if token is not None:
            token.raise_if_cancelled()
        h, w = frame.mask.shape
        for points in (positive, negative):
            for x, y in points:
                if not np.isfinite((x, y)).all() or not (0 <= x < w and 0 <= y < h):
                    raise ValueError("魔棒提示点必须位于照片内。")
        if not positive:
            raise ValueError("请先在要识别的区域内点一下，再点选需要排除的背景。")
        if self._backend is None:
            variant = resolve_magic_segment_model_variant(self.model_variant)
            self._backend = create_interactive_segmentation_service(variant)
        image = QImage(frame.rgba.data, w, h, frame.rgba.strides[0], QImage.Format.Format_RGBA8888)
        result = self._backend.predict_polygon(
            image=image, cache_key=self._source_key(frame.rgba),
            positive_points=[Point(*point) for point in positive],
            negative_points=[Point(*point) for point in negative],
            # The target can span the whole photo. Do not use the microscopic
            # small-object ROI, nor downsample the authoritative output mask.
            roi_enabled=False,
            cancel_check=(lambda: token.is_cancelled) if token is not None else None,
        )
        if token is not None:
            token.raise_if_cancelled()
        return apply_wand_mask(frame, result.mask, operation=operation, base_mask=base_mask)
