"""Source implementations of CU's two automatic diameter algorithms.

Inputs are uint8 GRAY/BGR/BGRA arrays in original image coordinates. No resizing,
file exchange, vendor executable, or fallback algorithm is used. The returned
candidates are proposals; callers must confirm categories before committing them.
"""
from __future__ import annotations

from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from fdm._native import require_native

DEFAULT_MEMORY_BUDGET = 2 * 1024**3
ALGORITHM_REVISION = 1


class CuAutomaticCancelled(RuntimeError):
    """Cooperative cancellation, also used by the isolated worker."""


def analyze_frame(
    image,
    algorithm: str = "traditional",
    checkpoint: Callable[[], object] | None = None,
    max_candidates: int = 10000,
    memory_budget: int = DEFAULT_MEMORY_BUDGET,
    *,
    preset: str = "fiber1",
    config: dict | None = None,
    _debug: bool = False,
) -> dict:
    """Analyze one immutable frame; preserve native result order and topology.

    A checkpoint may raise its caller's cancellation/timeout exception, which is
    propagated unchanged. A truthy return also requests cancellation. Resource
    exhaustion returns ``resource_limit`` with no partial candidates. Incorrect
    input and a missing/stale native extension are explicit exceptions.
    """
    started = perf_counter()
    from fdm.services.optical_config import ALGORITHM_IDS
    if algorithm not in ALGORITHM_IDS:
        raise ValueError("未知的全图分析算法。")
    if algorithm == "optical_adaptive":
        from fdm.services.optical_automatic import analyze_optical_frame
        return analyze_optical_frame(image, config=config, checkpoint=checkpoint,
                                     max_candidates=max_candidates, memory_budget=memory_budget)
    if preset not in {"fiber0", "fiber1", "constructor"}:
        raise ValueError("传统测径参数必须为 fiber0、fiber1 或 constructor。")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 1_000_000:
        raise ValueError("max_candidates 必须是 1 到 1000000 之间的整数。")
    if isinstance(memory_budget, bool) or not isinstance(memory_budget, int) or memory_budget <= 0:
        raise ValueError("memory_budget 必须为正整数字节数。")
    # Differences between historical cv primitives are explicitly validated
    # against this supported batch-primitives version, not silently upgraded.
    if tuple(cv2.__version__.split(".")[:2]) != ("4", "13"):
        raise RuntimeError("CU 全图测径要求 OpenCV 4.13，请使用已验证的依赖版本。")

    def check() -> None:
        if checkpoint is not None and checkpoint():
            raise CuAutomaticCancelled("全图分析已取消。")

    check()
    source = np.asarray(image)
    if source.dtype != np.uint8:
        raise TypeError("全图分析图像必须为 uint8。")
    if source.ndim not in (2, 3) or (source.ndim == 3 and source.shape[2] not in (3, 4)):
        raise ValueError("全图分析图像必须为灰度、BGR 或 BGRA。")
    if min(source.shape[:2]) < 16:
        raise ValueError("全图分析图像的宽和高至少为 16 像素。")
    if max(source.shape[:2]) > 32766:
        raise ValueError("全图分析图像尺寸超出定点绘线坐标范围。")
    diagnostics = {"algorithm": algorithm, "revision": ALGORITHM_REVISION, "input_shape": tuple(source.shape)}
    # Only the unavoidable input copies and cv2 transient allowance belong in
    # this preflight. Native owning images and actual vector capacities are
    # charged as allocated, including the exact traditional C squared table.
    pixel_count = int(source.shape[0]) * int(source.shape[1])
    prepared_channels = 1 if source.ndim == 2 else 3
    required = source.nbytes + pixel_count * (2 * prepared_channels + 16)
    if required > memory_budget:
        return {"status": "resource_limit", "reason": "automatic_memory_budget", "candidates": [],
                "diagnostics": {**diagnostics, "minimum_reserved_bytes": required, "memory_budget": memory_budget}}
    native = require_native()
    if source.ndim == 3 and source.shape[2] == 4:
        pixels = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
    else:
        # Writable/noncontiguous views must be owned before releasing the GIL.
        pixels = np.array(source, dtype=np.uint8, order="C", copy=True)
    pixels.setflags(write=False)
    prepared = perf_counter()
    check()
    try:
        result = native._cu_automatic(pixels, algorithm, check, max_candidates,
                                      memory_budget - pixels.nbytes - source.nbytes, preset, _debug)
    except MemoryError as exc:
        result = {"status": "resource_limit", "reason": str(exc) or "automatic_memory_budget",
                  "candidates": [], "diagnostics": {}}
    result["diagnostics"].update(diagnostics)
    result["diagnostics"].update(
        preparation_ms=(prepared - started) * 1000,
        total_ms=(perf_counter() - started) * 1000,
        memory_budget=memory_budget,
        backend="native_cu_source",
    )
    return result
