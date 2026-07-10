from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fdm.runtime_logging import (
    _reset_runtime_metric_buckets_for_tests,
    _rotate_runtime_log_if_needed,
    aggregate_runtime_metric,
    flush_runtime_metrics,
)


def test_runtime_log_rotation_keeps_bounded_backups() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "startup.log"
        path.write_text("current", encoding="utf-8")
        path.with_name("startup.log.1").write_text("one", encoding="utf-8")
        path.with_name("startup.log.2").write_text("two", encoding="utf-8")

        _rotate_runtime_log_if_needed(
            path,
            incoming_bytes=10,
            max_bytes=8,
            backup_count=2,
        )

        assert not path.exists()
        assert path.with_name("startup.log.1").read_text(encoding="utf-8") == "current"
        assert path.with_name("startup.log.2").read_text(encoding="utf-8") == "one"


def test_high_frequency_metrics_emit_one_five_second_summary() -> None:
    _reset_runtime_metric_buckets_for_tests()
    with patch("fdm.runtime_logging.append_runtime_log") as append:
        assert not aggregate_runtime_metric("frame", 10.0, now=100.0)
        assert not aggregate_runtime_metric("frame", 20.0, now=102.0)
        assert aggregate_runtime_metric("frame", 30.0, detail="request=3", now=105.0)
        assert not aggregate_runtime_metric("frame", 40.0, now=106.0)

    append.assert_called_once()
    title, summary = append.call_args.args
    assert title == "frame"
    assert "count=3" in summary
    assert "avg_ms=20.00" in summary
    assert "request=3" in summary


def test_flush_emits_incomplete_metric_window() -> None:
    _reset_runtime_metric_buckets_for_tests()
    with patch("fdm.runtime_logging.append_runtime_log") as append:
        assert not aggregate_runtime_metric("short-task", 12.0, detail="frame=1", now=100.0)
        assert flush_runtime_metrics() == 1

    title, summary = append.call_args.args
    assert title == "short-task"
    assert "count=1" in summary
    assert "flushed=true" in summary
    assert flush_runtime_metrics() == 0
