"""Versioned, immutable automatic-analysis settings shared by UI and workers.

These are application preferences and task snapshots, not project migrations.
The reference descriptor contains only resource identities; pixels never enter
the JSON payload or measurement provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Mapping


ALGORITHM_CHOICES = (
    ("传统全图（CU 兼容）", "traditional"),
    ("玻纤多尺度（CU 兼容）", "multiscale"),
    ("普通光学适配", "optical_adaptive"),
)
AUTOMATIC_ALGORITHMS = ALGORITHM_CHOICES
ALGORITHM_IDS = frozenset(algorithm for _, algorithm in ALGORITHM_CHOICES)
ALGORITHM_REVISIONS = {
    "traditional": "cu-automatic-traditional-v1",
    "multiscale": "cu-automatic-multiscale-v1",
    "optical_adaptive": "optical-adaptive-v1",
}
IMAGING_MODES = ("mixed", "solid", "double_edge")
IMAGING_MODE_CHOICES = (("深浅混合", "mixed"), ("实心纤维", "solid"),
                        ("透亮双边纤维", "double_edge"))


def automatic_algorithm_label(algorithm: str) -> str:
    return dict((value, label) for label, value in ALGORITHM_CHOICES).get(str(algorithm), str(algorithm))


def measurement_method_label(method: str) -> str:
    value = str(method)
    if value.startswith(("automatic_", "automatic-")):
        return automatic_algorithm_label(value[len("automatic_"):])
    return {"assisted": "贴边测径", "free": "自由测量", "manual": "手工测量",
            **dict((algorithm, label) for label, algorithm in ALGORITHM_CHOICES)}.get(value, value)


def _canonical_json(value: Any) -> str:
    """Reject non-JSON or non-finite values before crossing the worker boundary."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _context(value: Any, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}必须是上下文字典。")
    try:
        return json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须只包含有限的 JSON 值。") from exc


def validate_reference_descriptor(value: Any) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("空白视场参考必须是资源描述。")
    path = value.get("path")
    if not isinstance(path, str) or not path or not (Path(path).is_absolute() or PureWindowsPath(path).is_absolute()):
        raise ValueError("空白视场参考需要绝对资源路径。")
    digest = value.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError("空白视场参考缺少有效的 SHA256。")
    result = {"path": path, "sha256": digest.lower(), "context": _context(value.get("context"), "参考上下文")}
    # Diagnostic metadata is useful in offline exports, but is never required
    # to determine which pixels to use.
    if "diagnostics" in value:
        result["diagnostics"] = _context(value["diagnostics"], "参考诊断")
    return result


@dataclass(frozen=True, slots=True)
class OpticalAnalysisConfig:
    version: int = 1
    algorithm: str = "traditional"
    imaging_mode: str = "mixed"
    background_mode: str = "local"
    flat_enabled: bool = False
    _reference_json: str = "null"
    _context_json: str = "{}"

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("不支持的全图分析配置版本。")
        if not isinstance(self.algorithm, str) or self.algorithm not in ALGORITHM_IDS:
            raise ValueError("未知的全图分析算法。")
        if self.imaging_mode not in IMAGING_MODES:
            raise ValueError("未知的光学成像模式。")
        if self.background_mode != "local":
            raise ValueError("未知的背景处理方式。")
        if type(self.flat_enabled) is not bool:
            raise ValueError("空白视场校正开关必须为布尔值。")
        # Direct construction is as strict as from_dict; canonical immutable
        # strings prevent nested mutations after a task is queued.
        reference = validate_reference_descriptor(json.loads(self._reference_json))
        context = _context(json.loads(self._context_json), "分析帧上下文")
        object.__setattr__(self, "_reference_json", _canonical_json(reference))
        object.__setattr__(self, "_context_json", _canonical_json(context))

    @classmethod
    def from_dict(cls, payload: Mapping | "OpticalAnalysisConfig" | None = None,
                  *, algorithm: str | None = None) -> "OpticalAnalysisConfig":
        if isinstance(payload, cls):
            payload = payload.to_dict()
        if payload is not None and not isinstance(payload, Mapping):
            raise ValueError("全图分析配置必须是字典。")
        values = dict(payload or {})
        chosen = algorithm if algorithm is not None else values.get("algorithm", "traditional")
        if not isinstance(chosen, str) or chosen not in ALGORITHM_IDS:
            raise ValueError("未知的全图分析算法。")
        revision = values.get("algorithm_revision")
        # An explicit algorithm override is used when the user changes the
        # selector. It replaces the previous algorithm's revision too.
        if algorithm is None and revision is not None and revision != ALGORITHM_REVISIONS.get(chosen):
            raise ValueError("全图分析算法修订不受当前版本支持。")
        return cls(version=values.get("version", 1), algorithm=chosen,
                   imaging_mode=values.get("imaging_mode", "mixed"),
                   background_mode=values.get("background_mode", "local"),
                   flat_enabled=values.get("flat_enabled", False),
                   _reference_json=_canonical_json(validate_reference_descriptor(values.get("flat_reference"))),
                   _context_json=_canonical_json(_context(values.get("reference_context"), "分析帧上下文")))

    @property
    def algorithm_revision(self) -> str:
        return ALGORITHM_REVISIONS[self.algorithm]

    @property
    def flat_reference(self) -> dict | None:
        return json.loads(self._reference_json)

    @property
    def reference_context(self) -> dict:
        return json.loads(self._context_json)

    def to_dict(self) -> dict:
        return {"version": self.version, "algorithm": self.algorithm,
                "algorithm_revision": self.algorithm_revision,
                "imaging_mode": self.imaging_mode, "background_mode": self.background_mode,
                "flat_enabled": self.flat_enabled, "flat_reference": self.flat_reference,
                "reference_context": self.reference_context}

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


AnalysisConfig = OpticalAnalysisConfig


def normalize_analysis_config(config=None, algorithm: str = "optical_adaptive") -> dict:
    return OpticalAnalysisConfig.from_dict(config, algorithm=algorithm).to_dict()
