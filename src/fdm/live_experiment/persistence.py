"""Atomic JSON persistence for this application's experiment format only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fdm.atomic_io import atomic_write_json
from .models import ExperimentSession


def save_session(session: ExperimentSession, path: str | Path) -> Path:
    return atomic_write_json(path, session.to_dict())


def _reject_constant(value: str) -> Any:
    raise ValueError(f"试验文件包含非法数值：{value}")


def load_session(path: str | Path) -> ExperimentSession:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("试验文件必须是 JSON 对象")
    return ExperimentSession.from_dict(payload)
