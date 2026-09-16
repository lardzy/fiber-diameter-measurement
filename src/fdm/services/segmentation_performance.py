"""Request-local segmentation diagnostics; never part of saved measurements."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from time import perf_counter

from fdm.runtime_logging import append_runtime_log


@dataclass
class SegmentationPerformance:
    stages_ms: dict[str, float] = field(default_factory=dict)
    encoder_calls: int = 0
    decoder_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    roi_crops: list[tuple[int, int, int, int]] = field(default_factory=list)
    stop_reason: str = "completed"

    @contextmanager
    def stage(self, name: str):
        started = perf_counter()
        try:
            yield
        finally:
            self.stages_ms[name] = self.stages_ms.get(name, 0.0) + (perf_counter() - started) * 1000.0

    def snapshot(self, *, total_ms: float, cache_bytes: int) -> dict[str, object]:
        stages = {name: 0.0 for name in (
            "image_prepare_ms", "session_init_ms", "encoder_prepare_ms", "encoder_ms",
            "decoder_ms", "mask_postprocess_ms", "geometry_ms",
        )}
        stages.update(self.stages_ms)
        return {
            "stages_ms": stages,
            "service_ms": total_ms,
            "encoder_calls": self.encoder_calls,
            "decoder_calls": self.decoder_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "roi_crops": list(self.roi_crops),
            "expansion_count": max(0, len(self.roi_crops) - 1),
            "stop_reason": self.stop_reason,
            "cache_bytes": cache_bytes,
        }


def log_slow_segmentation(performance: dict[str, object], **context: object) -> None:
    elapsed = float(performance.get("total_ms", performance.get("service_ms", 0.0)))
    if elapsed < 100.0:
        return
    append_runtime_log(
        "Magic segmentation request",
        json.dumps({**context, **performance}, ensure_ascii=False, allow_nan=False),
    )
