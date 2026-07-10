from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import os
import sys
from threading import Lock
from time import monotonic


RUNTIME_LOG_MAX_BYTES = 10 * 1024 * 1024
RUNTIME_LOG_BACKUP_COUNT = 5
_LOG_LOCK = Lock()


@dataclass(slots=True)
class _MetricBucket:
    started_at: float
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    last_detail: str = ""


_METRIC_BUCKETS: dict[str, _MetricBucket] = {}


def runtime_log_path() -> Path:
    local_app_data = Path.home()
    if sys.platform.startswith("win"):
        app_data = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or str(Path.home())
        )
        local_app_data = app_data
    log_dir = local_app_data / "FiberDiameterMeasurement" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "startup.log"


def append_runtime_log(title: str, details: str = "") -> None:
    try:
        log_path = runtime_log_path()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = f"[{timestamp}] {title}\n"
        if details:
            record += details.rstrip() + "\n"
        record += "\n"
        with _LOG_LOCK:
            _rotate_runtime_log_if_needed(
                log_path,
                incoming_bytes=len(record.encode("utf-8")),
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(record)
    except OSError:
        return


def aggregate_runtime_metric(
    title: str,
    value: float,
    *,
    detail: str = "",
    interval_s: float = 5.0,
    now: float | None = None,
) -> bool:
    """Aggregate high-frequency metrics and emit at most once per interval."""

    timestamp = monotonic() if now is None else float(now)
    summary: str | None = None
    with _LOG_LOCK:
        bucket = _METRIC_BUCKETS.get(title)
        if bucket is None:
            bucket = _MetricBucket(started_at=timestamp)
            _METRIC_BUCKETS[title] = bucket
        numeric_value = float(value)
        bucket.count += 1
        bucket.total += numeric_value
        bucket.minimum = min(bucket.minimum, numeric_value)
        bucket.maximum = max(bucket.maximum, numeric_value)
        if detail:
            bucket.last_detail = str(detail)
        if timestamp - bucket.started_at >= max(0.1, float(interval_s)):
            average = bucket.total / max(1, bucket.count)
            summary = (
                f"window_s={timestamp - bucket.started_at:.2f}, count={bucket.count}, "
                f"avg_ms={average:.2f}, min_ms={bucket.minimum:.2f}, max_ms={bucket.maximum:.2f}"
            )
            if bucket.last_detail:
                summary += f", last=({bucket.last_detail})"
            _METRIC_BUCKETS[title] = _MetricBucket(started_at=timestamp)
    if summary is None:
        return False
    append_runtime_log(title, summary)
    return True


def flush_runtime_metrics() -> int:
    """Flush incomplete aggregation windows, for orderly task/app shutdown."""

    pending: list[tuple[str, str]] = []
    timestamp = monotonic()
    with _LOG_LOCK:
        for title, bucket in _METRIC_BUCKETS.items():
            if bucket.count <= 0:
                continue
            elapsed = max(0.0, timestamp - bucket.started_at)
            summary = (
                f"window_s={elapsed:.2f}, count={bucket.count}, "
                f"avg_ms={bucket.total / bucket.count:.2f}, "
                f"min_ms={bucket.minimum:.2f}, max_ms={bucket.maximum:.2f}, flushed=true"
            )
            if bucket.last_detail:
                summary += f", last=({bucket.last_detail})"
            pending.append((title, summary))
        _METRIC_BUCKETS.clear()
    for title, summary in pending:
        append_runtime_log(title, summary)
    return len(pending)


def _reset_runtime_metric_buckets_for_tests() -> None:
    with _LOG_LOCK:
        _METRIC_BUCKETS.clear()


def _rotate_runtime_log_if_needed(
    log_path: Path,
    *,
    incoming_bytes: int,
    max_bytes: int = RUNTIME_LOG_MAX_BYTES,
    backup_count: int = RUNTIME_LOG_BACKUP_COUNT,
) -> None:
    if max_bytes <= 0 or backup_count <= 0:
        return
    try:
        current_size = log_path.stat().st_size
    except FileNotFoundError:
        return
    if current_size + max(0, incoming_bytes) <= max_bytes:
        return
    oldest = log_path.with_name(f"{log_path.name}.{backup_count}")
    if oldest.exists():
        oldest.unlink()
    for index in range(backup_count - 1, 0, -1):
        source = log_path.with_name(f"{log_path.name}.{index}")
        if source.exists():
            source.replace(log_path.with_name(f"{log_path.name}.{index + 1}"))
    log_path.replace(log_path.with_name(f"{log_path.name}.1"))
