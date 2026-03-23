from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess

from fdm.geometry import Point
from fdm.settings import AppSettings

LABEL_ALIAS: dict[str, str] = {
    "粘": "粘纤",
    "莱": "莱赛尔",
    "莫": "莫代尔",
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

    def infer_image(
        self,
        *,
        image_path: str,
        model_name: str,
        model_file: str,
        settings: AppSettings,
        inference_options: dict[str, object] | None = None,
    ) -> AreaInferenceResult:
        worker_python = str(settings.area_worker_python or "").strip()
        if not worker_python:
            raise RuntimeError("未配置面积识别 Worker Python。")
        if not self._worker_path.exists():
            raise RuntimeError(f"未找到面积识别 worker: {self._worker_path}")

        payload = {
            "image_path": str(Path(image_path).expanduser().resolve()),
            "model_name": model_name,
            "model_file": model_file,
            "weights_dir": settings.area_weights_dir,
            "vendor_root": settings.area_vendor_root,
            "inference_options": dict(inference_options or {}),
        }
        result = subprocess.run(
            [worker_python, str(self._worker_path)],
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
            message = str(response.get("error") or stderr or "面积识别失败")
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
                    class_name=normalize_area_label(str(item.get("class_name", ""))),
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
