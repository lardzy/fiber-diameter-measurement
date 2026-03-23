from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import sys

from fdm.geometry import Point
from fdm.settings import AppSettings

LABEL_ALIAS: dict[str, str] = {
    "粘": "粘纤",
    "莱": "莱赛尔",
    "莫": "莫代尔",
}

LABEL_SWAP_BY_MODEL: dict[str, dict[str, str]] = {
    "棉-莱赛尔": {
        "棉": "莱赛尔",
        "莱赛尔": "棉",
    },
    "粘纤-莱赛尔": {
        "粘纤": "莱赛尔",
        "莱赛尔": "粘纤",
    },
}


def normalize_area_label(label: str) -> str:
    token = str(label or "").strip()
    if not token:
        return "未分类"
    return LABEL_ALIAS.get(token, token)


def parse_area_model_labels(model_name: str) -> list[str]:
    labels: list[str] = []
    for item in str(model_name or "").split("-"):
        normalized = normalize_area_label(item)
        if normalized not in labels:
            labels.append(normalized)
    return labels or ["未分类"]


def normalize_area_model_name(model_name: str) -> str:
    return str(model_name or "").replace(" ", "").strip()


def normalize_area_result_label(model_name: str, label: str) -> str:
    normalized_label = normalize_area_label(label)
    swap_mapping = LABEL_SWAP_BY_MODEL.get(normalize_area_model_name(model_name), {})
    return swap_mapping.get(normalized_label, normalized_label)


@dataclass(slots=True)
class AreaInstanceResult:
    class_name: str
    score: float
    bbox: list[int]
    polygon_px: list[Point]
    area_px: float


@dataclass(slots=True)
class AreaInferenceResult:
    instances: list[AreaInstanceResult]
    engine_meta: dict[str, object]


class AreaInferenceService:
    def __init__(self) -> None:
        self._worker_path = Path(__file__).resolve().parents[1] / "workers" / "area_worker.py"

    def _worker_command(self, settings: AppSettings) -> list[str]:
        if getattr(sys, "frozen", False):
            executable = Path(sys.executable).resolve()
            sibling_worker = executable.with_name("FiberAreaWorker.exe")
            if sibling_worker.exists():
                return [str(sibling_worker)]
        configured = settings.resolved_area_worker_program()
        if configured:
            configured_path = Path(configured)
            if configured_path.exists() and configured_path.name.lower().startswith("fiberareaworker"):
                return [str(configured_path)]
            return [configured, str(self._worker_path)]
        return [sys.executable, str(self._worker_path)]

    def _friendly_failure_message(self, message: str, *, worker_command: list[str]) -> str:
        token = str(message or "").strip()
        missing_module = None
        if "No module named 'PIL'" in token or 'No module named "PIL"' in token:
            missing_module = "Pillow(PIL)"
        elif "No module named 'torchvision'" in token or 'No module named "torchvision"' in token:
            missing_module = "torchvision"
        elif "No module named 'torch'" in token or 'No module named "torch"' in token:
            missing_module = "torch"
        if missing_module is not None:
            command_hint = worker_command[0] if worker_command else "当前 Worker"
            return (
                f"面积识别运行环境缺少 {missing_module}。"
                f"\n当前使用的 Worker: {command_hint}"
                "\n如果你在源码环境运行，请安装面积识别依赖；"
                "如果你在打包后的程序中运行，请检查设置里的 Worker 是否仍指向外部 Python，建议留空使用自动模式。"
            )
        return token or "面积识别失败"

    def infer_image(
        self,
        *,
        image_path: str,
        model_name: str,
        model_file: str,
        settings: AppSettings,
        inference_options: dict[str, object] | None = None,
    ) -> AreaInferenceResult:
        if not self._worker_path.exists() and not getattr(sys, "frozen", False):
            raise RuntimeError(f"未找到面积识别 worker: {self._worker_path}")

        resolved_weights_dir = settings.resolved_area_weights_dir()
        resolved_vendor_root = settings.resolved_area_vendor_root()
        payload = {
            "image_path": str(Path(image_path).expanduser().resolve()),
            "model_name": model_name,
            "model_file": model_file,
            "weights_dir": str(resolved_weights_dir),
            "vendor_root": str(resolved_vendor_root),
            "inference_options": dict(inference_options or {}),
        }
        worker_command = self._worker_command(settings)
        result = subprocess.run(
            worker_command,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if not stdout:
            raise RuntimeError(stderr or "面积识别 worker 没有返回结果。")
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"面积识别返回了无法解析的数据: {stdout[:300]}") from exc

        if result.returncode != 0:
            message = self._friendly_failure_message(
                str(response.get("error") or stderr or "面积识别失败"),
                worker_command=worker_command,
            )
            raise RuntimeError(message)
        if not isinstance(response, dict):
            raise RuntimeError("面积识别返回格式无效。")

        instances: list[AreaInstanceResult] = []
        for item in response.get("instances", []):
            if not isinstance(item, dict):
                continue
            polygon_px: list[Point] = []
            for point in item.get("polygon", []):
                if isinstance(point, dict):
                    polygon_px.append(Point(x=float(point.get("x", 0.0)), y=float(point.get("y", 0.0))))
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    polygon_px.append(Point(x=float(point[0]), y=float(point[1])))
            if len(polygon_px) < 3:
                continue
            instances.append(
                AreaInstanceResult(
                    class_name=normalize_area_result_label(model_name, str(item.get("class_name", ""))),
                    score=float(item.get("score", 0.0)),
                    bbox=[int(value) for value in item.get("bbox", [0, 0, 0, 0])][:4],
                    polygon_px=polygon_px,
                    area_px=float(item.get("area_px", 0.0)),
                )
            )
        return AreaInferenceResult(
            instances=instances,
            engine_meta=dict(response.get("engine_meta", {})),
        )
