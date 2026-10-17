"""Remember an explicitly selected ruler for a device and its pixel geometry.

Acquisition generations are deliberately excluded: stopping the same camera
does not change its optical calibration. Different devices/sizes require a
separate selection. Values are snapshots, never guessed from image content.
"""
from dataclasses import asdict
import json

from fdm.live_experiment.controller import CalibrationSnapshot


def source_key(frame):
    return json.dumps([frame.source_id, *frame.size], ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _path():
    from fdm.settings import settings_file_path
    return settings_file_path().parent / "capture-rulers.json"


def load_rulers():
    path = _path()
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("rulers"), dict):
        raise ValueError("已保存的采集标尺配置无法读取，请重新选择标尺。")
    return {key: CalibrationSnapshot(**value) for key, value in data["rulers"].items()}


def save_ruler(key, calibration):
    from fdm.atomic_io import atomic_write_text
    rulers = load_rulers()
    rulers[key] = calibration
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps({"version": 1, "rulers": {
        name: asdict(value) for name, value in rulers.items()}}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
