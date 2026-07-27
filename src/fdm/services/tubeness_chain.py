"""Auditable derivation of binary masks from persisted Tubeness responses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fdm.analysis_artifacts import AnalysisArtifact, AnalysisAssetReference
from fdm.services.analysis_asset_io import validate_analysis_asset_reference


TUBENESS_ASSET_SCHEMA = "fdm.tubeness.v1"
TUBENESS_THRESHOLD_MASK_SCHEMA = "fdm.tubeness-threshold-mask.v1"


class TubenessChainError(ValueError):
    """Raised when a Tubeness artifact cannot safely feed the analysis chain."""


@dataclass(frozen=True, slots=True)
class TubenessThresholdMask:
    """One immutable threshold result plus its auditable parent asset identity."""

    mask: NDArray[np.bool_]
    threshold: float
    maximum_response: float
    foreground_pixel_count: int
    included_pixel_count: int
    best_scale_minimum: float
    best_scale_maximum: float
    response_asset_sha256: str

    def __post_init__(self) -> None:
        mask = np.ascontiguousarray(np.asarray(self.mask, dtype=bool)).copy()
        if mask.ndim != 2 or min(mask.shape) < 1:
            raise ValueError("Tubeness 阈值掩膜必须是非空二维数组")
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)


def tubeness_response_reference(
    artifact: AnalysisArtifact,
) -> AnalysisAssetReference:
    """Return the unique persisted Tubeness response asset."""

    if not isinstance(artifact, AnalysisArtifact):
        raise TypeError("artifact 必须是 AnalysisArtifact")
    if artifact.tool_id != "fdm.tubeness":
        raise TubenessChainError("当前结果不是 Tubeness 分析结果。")
    matches = tuple(
        reference
        for reference in artifact.assets
        if reference.metadata.get("schema") == TUBENESS_ASSET_SCHEMA
    )
    if not matches:
        raise TubenessChainError(
            "该 Tubeness 结果缺少 response / best_scale 安全资产；"
            "旧结果需要重新计算后才能生成阈值掩膜。"
        )
    if len(matches) != 1:
        raise TubenessChainError(
            "该 Tubeness 结果包含多个响应资产，无法确定唯一来源；"
            "请重新计算后再试。"
        )
    return matches[0]


def build_tubeness_threshold_mask(
    artifact: AnalysisArtifact,
    asset_path: str | Path,
    *,
    threshold: float,
) -> TubenessThresholdMask:
    """Validate a persisted Tubeness NPZ and threshold its response.

    The source archive is always validated against the project reference before
    opening it with ``allow_pickle=False``.  The returned mask is derived from
    the stored response, never from a screen preview.
    """

    reference = tubeness_response_reference(artifact)
    source = Path(asset_path)
    validate_analysis_asset_reference(source, reference)
    if isinstance(threshold, bool):
        raise TubenessChainError("Tubeness 阈值必须是有限正数。")
    resolved_threshold = float(threshold)
    if not math.isfinite(resolved_threshold) or resolved_threshold <= 0.0:
        raise TubenessChainError("Tubeness 阈值必须是有限正数。")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if "response" not in archive.files or "best_scale" not in archive.files:
                raise TubenessChainError(
                    "Tubeness 资产缺少 response 或 best_scale 成员；"
                    "请重新计算该结果。"
                )
            response = np.asarray(archive["response"])
            best_scale = np.asarray(archive["best_scale"])
    except TubenessChainError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise TubenessChainError(f"无法读取 Tubeness 响应资产：{exc}") from exc
    if (
        response.ndim != 2
        or min(response.shape) < 1
        or best_scale.shape != response.shape
    ):
        raise TubenessChainError(
            "Tubeness response / best_scale 必须是尺寸一致的非空二维数组。"
        )
    if response.dtype.kind not in "fiu" or best_scale.dtype.kind not in "fiu":
        raise TubenessChainError("Tubeness 响应资产包含不支持的数据类型。")
    response_values = np.asarray(response, dtype=np.float64)
    scale_values = np.asarray(best_scale, dtype=np.float64)
    if not np.all(np.isfinite(response_values)) or not np.all(
        np.isfinite(scale_values)
    ):
        raise TubenessChainError(
            "Tubeness 响应资产包含 NaN 或 Inf，不能生成可审计掩膜。"
        )
    maximum = float(np.max(response_values))
    if maximum <= 0.0:
        raise TubenessChainError(
            "该 Tubeness 结果没有正响应，无法生成二值掩膜。"
        )
    if resolved_threshold > maximum:
        raise TubenessChainError(
            f"阈值 {resolved_threshold:g} 高于最大响应 {maximum:g}。"
        )
    mask = np.asarray(response_values >= resolved_threshold, dtype=bool)
    foreground = int(np.count_nonzero(mask))
    if foreground <= 0:
        raise TubenessChainError("当前阈值没有选中任何响应像素。")
    selected_scales = scale_values[mask]
    return TubenessThresholdMask(
        mask=mask,
        threshold=resolved_threshold,
        maximum_response=maximum,
        foreground_pixel_count=foreground,
        included_pixel_count=int(mask.size),
        best_scale_minimum=float(np.min(selected_scales)),
        best_scale_maximum=float(np.max(selected_scales)),
        response_asset_sha256=reference.sha256,
    )


__all__ = [
    "TUBENESS_ASSET_SCHEMA",
    "TUBENESS_THRESHOLD_MASK_SCHEMA",
    "TubenessChainError",
    "TubenessThresholdMask",
    "build_tubeness_threshold_mask",
    "tubeness_response_reference",
]
