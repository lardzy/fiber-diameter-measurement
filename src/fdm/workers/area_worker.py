from __future__ import annotations

from pathlib import Path
import base64
import importlib.util
import json
import sys
import traceback


def _error(message: str, *, details: str | None = None) -> int:
    payload = {"error": message}
    if details:
        payload["details"] = details
    print(json.dumps(payload, ensure_ascii=False))
    return 1


def _load_engine_module(vendor_root: Path):
    area_infer_root = vendor_root.parent.parent
    engine_path = area_infer_root / "app" / "engine.py"
    if not engine_path.exists():
        raise RuntimeError(f"未找到参考 area engine: {engine_path}")
    spec = importlib.util.spec_from_file_location("fdm_area_ref_engine", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 area engine: {engine_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        return _error("输入 JSON 无法解析", details=str(exc))

    image_path = Path(str(payload.get("image_path", "")).strip()).expanduser()
    model_name = str(payload.get("model_name", "")).strip()
    model_file = str(payload.get("model_file", "")).strip()
    vendor_root = Path(str(payload.get("vendor_root", "")).strip()).expanduser()
    weights_dir = Path(str(payload.get("weights_dir", "")).strip()).expanduser()
    inference_options = payload.get("inference_options", {})

    if not image_path.exists():
        return _error(f"未找到图片: {image_path}")
    if not vendor_root.exists():
        return _error(f"未找到 YOLACT vendor 目录: {vendor_root}")
    if not weights_dir.exists():
        return _error(f"未找到权重目录: {weights_dir}")
    if not model_name or not model_file:
        return _error("模型名称或权重文件名为空。")

    try:
        raw = image_path.read_bytes()
        engine_module = _load_engine_module(vendor_root.resolve())
        engine = engine_module.AreaNativeEngine(
            weights_dir=str(weights_dir.resolve()),
            vendor_root=str(vendor_root.resolve()),
        )
        result = engine.infer(
            model_name=model_name,
            model_file=model_file,
            image_bytes_b64=base64.b64encode(raw).decode("ascii"),
            inference_options=inference_options if isinstance(inference_options, dict) else None,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), details=traceback.format_exc(limit=8))


if __name__ == "__main__":
    raise SystemExit(main())
