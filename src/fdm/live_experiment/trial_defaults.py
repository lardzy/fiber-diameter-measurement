"""Explicit defaults for subsequently created trials, stored outside projects."""
from __future__ import annotations

import json
from pathlib import Path

from fdm.atomic_io import atomic_write_json
from .models import DiameterRangePolicy
from .reminders import ReminderPolicy


def defaults_path() -> Path:
    from fdm.settings import settings_directory
    return settings_directory() / "trial-defaults.json"


def _read(path=None):
    path = Path(path or defaults_path())
    if not path.exists():
        return {"version": 1}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("试验默认配置格式无效")
    for mode in ("content", "diameter"):
        if mode in payload and not isinstance(payload[mode], dict):
            raise ValueError("试验默认参数必须是对象")
    return payload


def new_trial_defaults(mode: str, *, path=None) -> dict:
    if mode not in ("content", "diameter"):
        raise ValueError("未知试验类型")
    data = _read(path).get(mode, {})
    result = {"reminder_policy": ReminderPolicy.from_dict(data["reminder_policy"])
              if "reminder_policy" in data else ReminderPolicy()}
    if mode == "diameter":
        result["diameter_range_policy"] = DiameterRangePolicy.from_value(data.get("diameter_range_policy"))
    return result


def save_reminder_defaults(mode: str, policy: ReminderPolicy, *, path=None) -> Path:
    if mode not in ("content", "diameter"):
        raise ValueError("未知试验类型")
    policy = ReminderPolicy.from_dict(policy.to_dict())
    payload = _read(path)
    payload.setdefault(mode, {})["reminder_policy"] = policy.to_dict()
    return atomic_write_json(path or defaults_path(), payload)


def save_diameter_range_default(policy: DiameterRangePolicy, *, path=None) -> Path:
    policy = DiameterRangePolicy.from_value(policy)
    payload = _read(path)
    payload.setdefault("diameter", {})["diameter_range_policy"] = policy.to_dict()
    return atomic_write_json(path or defaults_path(), payload)
