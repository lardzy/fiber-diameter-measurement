from __future__ import annotations

import base64
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime
import gc
import hashlib
import io
import json
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.model_metadata import (
    MODEL_METADATA_VERSION,
    MODEL_SPECS,
    find_model_spec,
    parse_model_classes,
    resolve_model_classes,
)


AREA_WORKER_DIAGNOSTICS_ENV = "FDM_AREA_WORKER_DIAGNOSTICS"
AREA_WORKER_LOG_PATH_ENV = "FDM_AREA_WORKER_LOG_PATH"
AREA_WORKER_REQUEST_ID_ENV = "FDM_AREA_WORKER_REQUEST_ID"


def _trace_area_stage(stage: str, **details: object) -> None:
    enabled = str(os.environ.get(AREA_WORKER_DIAGNOSTICS_ENV, "")).strip().lower()
    log_token = str(os.environ.get(AREA_WORKER_LOG_PATH_ENV, "")).strip()
    if enabled not in {"1", "true", "yes", "on"} or not log_token:
        return
    payload = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "pid": os.getpid(),
        "request_id": str(os.environ.get(AREA_WORKER_REQUEST_ID_ENV, "")),
        "stage": str(stage),
        "details": {key: value for key, value in details.items()},
    }
    try:
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        log_path = Path(log_token)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except (OSError, TypeError, ValueError):
        return


PALETTE: list[tuple[int, int, int]] = [
    (255, 87, 34),
    (30, 136, 229),
    (67, 160, 71),
    (142, 36, 170),
    (255, 179, 0),
    (0, 172, 193),
    (94, 53, 177),
    (216, 27, 96),
]

class InferServiceError(RuntimeError):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code

@dataclass
class _ModelRuntime:
    model_name: str
    model_file: str
    class_names: tuple[str, ...]
    device: str
    cfg_name: str
    cfg_obj: Any
    net: Any
    loaded_at: float
    trusted_metadata: bool


class AreaNativeEngine:
    def __init__(
        self,
        *,
        weights_dir: str,
        vendor_root: str,
        default_cfg_name: str = "yolact_base_config",
        infer_device: str = "auto",
        gpu_policy: str = "warn_continue",
        max_cached_models: int = 2,
        require_trusted_weights: bool = True,
        verify_trusted_weights: bool = True,
        required_model_files: tuple[str, ...] = (),
        max_image_bytes: int = 48 * 1024 * 1024,
        max_image_pixels: int = 50_000_000,
        max_mask_working_bytes: int = 1536 * 1024 * 1024,
    ) -> None:
        self.weights_dir = Path(weights_dir).resolve()
        self.vendor_root = Path(vendor_root).resolve()
        self.default_cfg_name = default_cfg_name
        self._requested_device = self._normalize_infer_device(infer_device)
        self._gpu_policy = self._normalize_gpu_policy(gpu_policy)
        self._max_cached_models = max(1, min(int(max_cached_models), 2))
        self._require_trusted_weights = bool(require_trusted_weights)
        # "Trusted" must mean both allowlisted and byte-for-byte verified.  Do
        # not permit a caller to accidentally weaken that promise by combining
        # require_trusted_weights=True with verify_trusted_weights=False.
        self._verify_trusted_weights = bool(verify_trusted_weights) or self._require_trusted_weights
        self._required_model_files = tuple(Path(item).name for item in required_model_files if Path(item).name)
        self._max_image_bytes = max(1, int(max_image_bytes))
        self._max_image_pixels = max(1, int(max_image_pixels))
        self._max_mask_working_bytes = max(16 * 1024 * 1024, int(max_mask_working_bytes))
        self._lock = threading.RLock()
        self._runtime_loaded = False
        self._cache: OrderedDict[tuple[str, tuple[str, ...], str], _ModelRuntime] = OrderedDict()
        self._verified_weights: dict[Path, tuple[int, int, int, str]] = {}
        self._cache_evictions = 0

        self._torch = None
        self._cfg = None
        self._yolact_cls = None
        self._fast_transform_cls = None
        self._postprocess_fn = None
        self._effective_device = None
        self._effective_device_key = "cpu"
        self._device_warning: str | None = None
        self._cuda_available = False
        self._gpu_count = 0
        self._gpu_name: str | None = None

    def _normalize_infer_device(self, value: str | None) -> str:
        token = str(value or "").strip().lower()
        if token == "cuda":
            return "cuda:0"
        if token in {"cpu", "cuda:0", "auto"}:
            return token
        return "cpu"

    def _normalize_gpu_policy(self, value: str | None) -> str:
        token = str(value or "").strip().lower()
        if token in {"warn_continue", "fail"}:
            return token
        return "warn_continue"

    def _set_device_warning(self, warning: str | None) -> None:
        token = str(warning or "").strip()
        self._device_warning = token or None

    def _runtime_device_payload(self) -> dict[str, Any]:
        return {
            "requested_device": self._requested_device,
            "effective_device": "cuda:0" if self._effective_device_key == "cuda" else "cpu",
            "cuda_available": bool(self._cuda_available),
            "gpu_name": self._gpu_name,
            "gpu_count": int(self._gpu_count),
            "device_warning": self._device_warning,
            "gpu_policy": self._gpu_policy,
        }

    def _fallback_to_cpu(self, reason: str) -> None:
        if self._effective_device_key != "cuda":
            return
        self._clear_model_cache()
        self._effective_device = self._torch.device("cpu")
        self._effective_device_key = "cpu"
        self._set_device_warning(reason)

    def _resolve_runtime_device(self) -> None:
        cuda_available = False
        gpu_count = 0
        gpu_name: str | None = None
        try:
            cuda_available = bool(self._torch.cuda.is_available())
            if cuda_available:
                gpu_count = int(self._torch.cuda.device_count())
                if gpu_count > 0:
                    gpu_name = str(self._torch.cuda.get_device_name(0))
        except Exception:
            cuda_available = False
            gpu_count = 0
            gpu_name = None

        self._cuda_available = cuda_available
        self._gpu_count = gpu_count
        self._gpu_name = gpu_name
        self._set_device_warning(None)

        if self._requested_device == "cpu":
            self._effective_device = self._torch.device("cpu")
            self._effective_device_key = "cpu"
            return

        if cuda_available:
            self._effective_device = self._torch.device("cuda:0")
            self._effective_device_key = "cuda"
            return

        if self._requested_device == "cuda:0" and self._gpu_policy == "fail":
            raise InferServiceError("infer_service_unavailable", "cuda_requested_but_unavailable")

        if self._requested_device == "cuda:0":
            self._set_device_warning("cuda_requested_but_unavailable_fallback_to_cpu")
        else:
            self._set_device_warning("cuda_unavailable_fallback_to_cpu")
        self._effective_device = self._torch.device("cpu")
        self._effective_device_key = "cpu"

    def _ensure_runtime(self) -> None:
        if self._runtime_loaded:
            _trace_area_stage("runtime_import_cached")
            return

        if not self.vendor_root.exists():
            raise InferServiceError("infer_service_unavailable", f"vendor_root_not_found:{self.vendor_root}")

        vendor_str = str(self.vendor_root)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)

        _trace_area_stage("runtime_import_started", vendor_root=str(self.vendor_root))
        try:
            import torch
            from data import cfg
            from data.config import yolact_base_config
            from yolact import Yolact
            from utils.augmentations import FastBaseTransform
            from layers.output_utils import postprocess
        except Exception as exc:
            raise InferServiceError("infer_service_unavailable", f"runtime_import_failed:{exc}") from exc

        self._torch = torch
        self._cfg = cfg
        self._yolact_cls = Yolact
        self._fast_transform_cls = FastBaseTransform
        self._postprocess_fn = postprocess
        self._yolact_base_config = yolact_base_config
        self._resolve_runtime_device()
        self._runtime_loaded = True
        _trace_area_stage(
            "runtime_import_completed",
            torch_version=str(getattr(self._torch, "__version__", "unknown")),
            effective_device=self._effective_device_key,
        )

    def _build_cfg(self, class_names: list[str]) -> Any:
        dataset = self._yolact_base_config.dataset.copy(
            {
                "name": "TextileFiber",
                "class_names": tuple(class_names),
                "label_map": {idx + 1: idx + 1 for idx in range(len(class_names))},
            }
        )
        return self._yolact_base_config.copy(
            {
                "name": self.default_cfg_name.replace("_config", ""),
                "dataset": dataset,
                "num_classes": len(class_names) + 1,
            }
        )

    def _apply_cfg(self, cfg_obj: Any) -> None:
        self._cfg.replace(cfg_obj)
        self._cfg.name = getattr(cfg_obj, "name", None) or self.default_cfg_name.replace("_config", "")
        if not hasattr(self._cfg, "mask_proto_debug"):
            self._cfg.mask_proto_debug = False
        if not hasattr(self._cfg, "rescore_bbox"):
            self._cfg.rescore_bbox = False
        if not hasattr(self._cfg, "eval_mask_branch"):
            self._cfg.eval_mask_branch = True

    def _normalize_options(self, inference_options: dict[str, Any] | None) -> dict[str, Any]:
        options = dict(inference_options or {})
        include_overlay = options.get("include_overlay", True)
        if not isinstance(include_overlay, bool):
            raise InferServiceError("infer_bad_response", "include_overlay_must_be_boolean")
        normalized = {
            "score_threshold": float(options.get("score_threshold", 0.15) or 0.15),
            "top_k": int(options.get("top_k", 200) or 200),
            "nms_top_k": int(options.get("nms_top_k", 200) or 200),
            "nms_conf_thresh": float(options.get("nms_conf_thresh", 0.05) or 0.05),
            "nms_thresh": float(options.get("nms_thresh", 0.5) or 0.5),
            "overlay_alpha": float(options.get("overlay_alpha", 0.45) or 0.45),
            "include_overlay": include_overlay,
        }
        normalized["score_threshold"] = max(0.0, min(1.0, normalized["score_threshold"]))
        normalized["nms_conf_thresh"] = max(0.0, min(1.0, normalized["nms_conf_thresh"]))
        normalized["nms_thresh"] = max(0.0, min(1.0, normalized["nms_thresh"]))
        normalized["top_k"] = max(1, min(200, normalized["top_k"]))
        normalized["nms_top_k"] = max(1, min(400, normalized["nms_top_k"]))
        normalized["overlay_alpha"] = max(0.05, min(0.95, normalized["overlay_alpha"]))
        return normalized

    def _decode_image(self, image_bytes_b64: str) -> np.ndarray:
        _trace_area_stage("image_decode_started", encoded_chars=len(image_bytes_b64))
        try:
            raw = base64.b64decode(image_bytes_b64, validate=True)
            if len(raw) > self._max_image_bytes:
                raise InferServiceError(
                    "infer_request_too_large",
                    f"decoded_image_bytes_exceeded:{len(raw)}>{self._max_image_bytes}",
                )
            try:
                with Image.open(io.BytesIO(raw)) as header_image:
                    width, height = header_image.size
            except Exception as exc:
                raise InferServiceError("infer_bad_response", f"invalid_image_header:{exc}") from exc
            pixels = int(width) * int(height)
            if width <= 0 or height <= 0 or pixels > self._max_image_pixels:
                raise InferServiceError(
                    "infer_request_too_large",
                    f"encoded_image_pixels_exceeded:{pixels}>{self._max_image_pixels}",
                )
            arr = np.frombuffer(raw, dtype=np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except InferServiceError:
            raise
        except Exception as exc:
            raise InferServiceError("infer_bad_response", f"invalid_image_bytes:{exc}") from exc

        if image_bgr is None:
            raise InferServiceError("infer_bad_response", "invalid_image_decode")
        height, width = image_bgr.shape[:2]
        pixels = int(height) * int(width)
        if pixels > self._max_image_pixels:
            raise InferServiceError(
                "infer_request_too_large",
                f"decoded_image_pixels_exceeded:{pixels}>{self._max_image_pixels}",
            )
        _trace_area_stage(
            "image_decode_completed",
            width=int(width),
            height=int(height),
            pixels=pixels,
        )
        return image_bgr

    def _validate_mask_working_budget(self, *, pixels: int, top_k: int) -> None:
        estimated_bytes = max(1, int(pixels)) * max(1, int(top_k)) * 4
        if estimated_bytes > self._max_mask_working_bytes:
            raise InferServiceError(
                "infer_request_too_large",
                "mask_working_set_exceeded:"
                f"{estimated_bytes}>{self._max_mask_working_bytes} "
                f"(pixels={pixels},top_k={top_k})",
            )

    def _mask_to_polygon(self, mask_bool: np.ndarray | None) -> list[list[int]]:
        if not isinstance(mask_bool, np.ndarray):
            return []
        if mask_bool.dtype != np.uint8:
            mask_u8 = mask_bool.astype(np.uint8) * 255
        else:
            mask_u8 = mask_bool
        if mask_u8.ndim != 2:
            return []
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        if largest is None or len(largest) < 3:
            return []
        perimeter = cv2.arcLength(largest, True)
        epsilon = max(1.0, 0.0035 * perimeter)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        points = approx[:, 0, :] if approx.ndim == 3 and approx.shape[1] == 1 else approx
        polygon: list[list[int]] = []
        if not isinstance(points, np.ndarray):
            return []
        for p in points.tolist():
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                continue
            polygon.append([int(p[0]), int(p[1])])
        if len(polygon) < 3:
            return []
        return polygon

    def _resolve_weight_path(self, model_file: str) -> Path:
        model_key = Path(str(model_file or "").strip()).name
        if not model_key:
            raise InferServiceError("infer_model_load_failed", "invalid_model_file")
        path = self.weights_dir / model_key
        if not path.exists() or not path.is_file():
            raise InferServiceError("infer_model_load_failed", f"weight_not_found:{path}")
        return path

    def _verify_weight_path(self, *, model_name: str, model_file: str, path: Path) -> bool:
        spec = find_model_spec(model_file=model_file)
        if spec is None:
            if self._require_trusted_weights:
                raise InferServiceError(
                    "infer_model_load_failed",
                    (
                        f"untrusted_model_file:{path.name}:custom area checkpoints are disabled by default; "
                        "source development may opt in with FDM_ALLOW_UNTRUSTED_AREA_MODELS=1, "
                        "but frozen/full releases only accept model_metadata.py allowlisted hashes"
                    ),
                )
            return False
        expected_sha256 = str(spec.get("sha256") or "").strip().lower()
        if len(expected_sha256) != 64:
            raise InferServiceError("infer_model_load_failed", f"missing_trusted_sha256:{path.name}")
        if not self._verify_trusted_weights:
            return True

        stat = path.stat()
        cached = self._verified_weights.get(path)
        fingerprint = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
            expected_sha256,
        )
        if cached == fingerprint:
            _trace_area_stage(
                "weight_hash_cached",
                model_file=path.name,
                size_bytes=int(stat.st_size),
            )
            return True
        digest = hashlib.sha256()
        _trace_area_stage(
            "weight_hash_started",
            model_file=path.name,
            size_bytes=int(stat.st_size),
        )
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise InferServiceError("infer_model_load_failed", f"weight_hash_failed:{path.name}:{exc}") from exc
        actual_sha256 = digest.hexdigest()
        if not secrets.compare_digest(actual_sha256, expected_sha256):
            raise InferServiceError(
                "infer_model_load_failed",
                f"weight_sha256_mismatch:{path.name}:{actual_sha256}",
            )
        self._verified_weights[path] = fingerprint
        _trace_area_stage("weight_hash_completed", model_file=path.name)
        return True

    def _load_state_dict_safely(self, path: Path) -> OrderedDict[str, Any]:
        """Load a tensor-only checkpoint without invoking arbitrary pickle code."""

        _trace_area_stage("checkpoint_load_started", model_file=path.name)
        try:
            payload = self._torch.load(
                str(path),
                map_location=self._torch.device("cpu"),
                weights_only=True,
            )
        except TypeError as exc:
            raise InferServiceError(
                "infer_model_load_failed",
                f"safe_weights_only_unsupported:{path.name}:PyTorch 2.4 or newer is required",
            ) from exc
        except Exception as exc:
            raise InferServiceError(
                "infer_model_load_failed",
                (
                    f"safe_weights_only_load_failed:{path.name}:{exc}; "
                    "the checkpoint must be a compatible tensor state_dict"
                ),
            ) from exc

        # Some training pipelines wrap the actual state dict in a single
        # "state_dict" entry.  This remains safe because weights_only already
        # restricted deserialization to tensors and simple containers.
        if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
            payload = payload["state_dict"]
        if not isinstance(payload, Mapping) or not payload:
            raise InferServiceError(
                "infer_model_load_failed",
                f"safe_weights_only_invalid_state_dict:{path.name}:expected a non-empty mapping",
            )
        if any(not isinstance(key, str) for key in payload):
            raise InferServiceError(
                "infer_model_load_failed",
                f"safe_weights_only_invalid_state_dict:{path.name}:all parameter keys must be strings",
            )
        _trace_area_stage(
            "checkpoint_load_completed",
            model_file=path.name,
            parameter_count=len(payload),
        )
        return OrderedDict(payload.items())

    def _evict_runtime(self, runtime: _ModelRuntime) -> None:
        try:
            del runtime.net
        except Exception:
            pass
        gc.collect()
        if self._effective_device_key == "cuda" and self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

    def _clear_model_cache(self) -> None:
        runtimes = list(self._cache.values())
        self._cache.clear()
        for runtime in runtimes:
            self._evict_runtime(runtime)

    def _enforce_cache_limit(self) -> None:
        while len(self._cache) > self._max_cached_models:
            _, runtime = self._cache.popitem(last=False)
            self._cache_evictions += 1
            self._evict_runtime(runtime)

    def _model_cache_key(self, model_file: str, class_names: tuple[str, ...]) -> tuple[str, tuple[str, ...], str]:
        return (Path(model_file).name, class_names, self._effective_device_key)

    def _load_model(self, *, model_name: str, model_file: str) -> _ModelRuntime:
        _trace_area_stage(
            "model_load_started",
            model_name=model_name,
            model_file=Path(model_file).name,
        )
        weight_path = self._resolve_weight_path(model_file)
        trusted_metadata = self._verify_weight_path(
            model_name=model_name,
            model_file=model_file,
            path=weight_path,
        )
        self._ensure_runtime()

        classes, metadata_mapping = resolve_model_classes(model_name=model_name, model_file=model_file)
        trusted_metadata = trusted_metadata and metadata_mapping
        cache_key = self._model_cache_key(model_file=model_file, class_names=classes)
        if cache_key in self._cache:
            runtime = self._cache.pop(cache_key)
            self._cache[cache_key] = runtime
            _trace_area_stage("model_load_cached", model_file=runtime.model_file)
            return runtime

        cfg_obj = self._build_cfg(list(classes))
        self._apply_cfg(cfg_obj)

        try:
            net = self._yolact_cls()
            state_dict = self._load_state_dict_safely(weight_path)

            # Preserve the compatibility cleanup from YOLACT.load_weights,
            # while keeping deserialization inside the safe loader above.
            for key in list(state_dict.keys()):
                if key.startswith("backbone.layer") and not key.startswith("backbone.layers"):
                    del state_dict[key]
                elif key.startswith("fpn.downsample_layers."):
                    layer_index = int(key.split(".")[2])
                    if self._cfg.fpn is not None and layer_index >= self._cfg.fpn.num_downsample:
                        del state_dict[key]
            net.load_state_dict(state_dict)
            try:
                net = net.to(self._effective_device)
            except Exception as exc:
                if self._effective_device_key == "cuda" and self._gpu_policy == "warn_continue":
                    self._fallback_to_cpu(f"model_to_cuda_failed:{exc}")
                    return self._load_model(model_name=model_name, model_file=model_file)
                raise InferServiceError("infer_service_unavailable", f"model_to_device_failed:{exc}") from exc
            net.eval()
            net.detect.use_fast_nms = True
            net.detect.use_cross_class_nms = False
        except InferServiceError:
            raise
        except Exception as exc:
            raise InferServiceError("infer_model_load_failed", f"load_model_failed:{exc}") from exc

        runtime = _ModelRuntime(
            model_name=model_name,
            model_file=Path(model_file).name,
            class_names=classes,
            device=self._effective_device_key,
            cfg_name=getattr(cfg_obj, "name", "yolact_base"),
            cfg_obj=cfg_obj,
            net=net,
            loaded_at=time.time(),
            trusted_metadata=trusted_metadata,
        )
        self._cache[cache_key] = runtime
        self._enforce_cache_limit()
        _trace_area_stage(
            "model_load_completed",
            model_file=runtime.model_file,
            device=runtime.device,
            class_count=len(runtime.class_names),
        )
        return runtime

    def health(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_runtime()
            runtime_info = self._runtime_device_payload()
            return {
                "status": "ok",
                "weights_dir": str(self.weights_dir),
                "vendor_root": str(self.vendor_root),
                "cached_models": [
                    {
                        "model_file": item.model_file,
                        "class_names": list(item.class_names),
                        "cfg_name": item.cfg_name,
                        "device": item.device,
                        "loaded_at": item.loaded_at,
                        "trusted_metadata": item.trusted_metadata,
                    }
                    for item in self._cache.values()
                ],
                "model_cache": {
                    "size": len(self._cache),
                    "max_size": self._max_cached_models,
                    "evictions": self._cache_evictions,
                },
                "weight_policy": {
                    "require_trusted": self._require_trusted_weights,
                    "verify_sha256": self._verify_trusted_weights,
                    "metadata_version": MODEL_METADATA_VERSION,
                },
                "runtime": {
                    "torch_version": getattr(self._torch, "__version__", "unknown"),
                    **runtime_info,
                },
            }

    def readiness(self) -> dict[str, Any]:
        with self._lock:
            if not self.vendor_root.is_dir():
                raise InferServiceError("infer_service_unavailable", f"vendor_root_not_found:{self.vendor_root}")
            if not self.weights_dir.is_dir():
                raise InferServiceError("infer_service_unavailable", f"weights_dir_not_found:{self.weights_dir}")
            required_files = self._required_model_files
            explicit_requirements = bool(required_files)
            if not required_files:
                required_files = tuple(
                    str(spec["model_file"])
                    for spec in MODEL_SPECS
                    if (self.weights_dir / str(spec["model_file"])).is_file()
                )
            if not required_files:
                raise InferServiceError("infer_service_unavailable", "no_loadable_model_candidates")
            verified_models: list[str] = []
            loadable_model = ""
            failures: list[str] = []
            for model_file in required_files:
                try:
                    path = self._resolve_weight_path(model_file)
                    spec = find_model_spec(model_file=model_file)
                    model_name = str(spec.get("model_name") or "") if spec is not None else ""
                    if self._verify_weight_path(model_name=model_name, model_file=model_file, path=path):
                        verified_models.append(path.name)
                    if not loadable_model:
                        runtime = self._load_model(model_name=model_name, model_file=model_file)
                        loadable_model = runtime.model_file
                except InferServiceError as exc:
                    failures.append(f"{model_file}:{exc.message}")
                    if explicit_requirements:
                        raise
            if not loadable_model:
                raise InferServiceError(
                    "infer_service_unavailable",
                    "no_loadable_model:" + ";".join(failures),
                )
            return {
                "status": "ready",
                "required_models": list(required_files),
                "verified_models": verified_models,
                "loadable_model": loadable_model,
                "device": self._runtime_device_payload(),
                "metadata_version": MODEL_METADATA_VERSION,
            }

    def warmup(self, *, model_name: str, model_file: str) -> dict[str, Any]:
        with self._lock:
            runtime = self._load_model(model_name=model_name, model_file=model_file)
            return {
                "status": "ok",
                "model_file": runtime.model_file,
                "class_names": list(runtime.class_names),
                "cfg_name": runtime.cfg_name,
                "device": runtime.device,
                "class_mapping": "trusted_metadata_v1" if runtime.trusted_metadata else "model_name_fallback",
                **self._runtime_device_payload(),
            }

    def _render_overlay(
        self,
        image_bgr: np.ndarray,
        instances: list[dict[str, Any]],
        class_names: tuple[str, ...],
        alpha: float,
    ) -> str:
        overlay = image_bgr.astype(np.float32).copy()
        class_to_color: dict[str, np.ndarray] = {
            name: np.array(PALETTE[idx % len(PALETTE)], dtype=np.float32)
            for idx, name in enumerate(class_names)
        }

        for item in instances:
            cls_name = str(item.get("class_name") or "未分类")
            mask = item.get("mask")
            if not isinstance(mask, np.ndarray):
                continue
            ys, xs = np.where(mask)
            if ys.size <= 0:
                continue
            color = class_to_color.get(cls_name, np.array(PALETTE[0], dtype=np.float32))
            overlay[ys, xs] = overlay[ys, xs] * (1.0 - alpha) + color * alpha

        out = np.clip(overlay, 0, 255).astype(np.uint8)

        class_code_map = {name: f"C{idx + 1}" for idx, name in enumerate(class_names)}
        for item in instances[:120]:
            x1, y1, x2, y2 = item.get("bbox", [0, 0, 0, 0])
            cls_name = str(item.get("class_name") or "未分类")
            color = class_to_color.get(cls_name, np.array(PALETTE[0], dtype=np.float32))
            cv_color = tuple(int(v) for v in color.tolist())
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), cv_color, 1)
            code = class_code_map.get(cls_name, "C1")
            cv2.putText(out, code, (int(x1) + 2, int(y1) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        ok, encoded = cv2.imencode(".png", out)
        if not ok:
            raise InferServiceError("infer_bad_response", "overlay_encode_failed")
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    def _run_model_infer(self, *, net: Any, image_bgr: np.ndarray, options: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        h, w = image_bgr.shape[:2]
        transform = self._fast_transform_cls()
        try:
            transform = transform.to(self._effective_device)
        except Exception:
            transform = transform

        frame = self._torch.from_numpy(image_bgr).to(self._effective_device).float().unsqueeze(0)
        batch = transform(frame)
        _trace_area_stage(
            "model_forward_started",
            width=int(w),
            height=int(h),
            device=self._effective_device_key,
        )
        with self._torch.no_grad():
            preds = net(batch)
        _trace_area_stage("model_forward_completed")
        _trace_area_stage("postprocess_started")
        result = self._postprocess_fn(
            preds,
            w,
            h,
            score_threshold=float(options["score_threshold"]),
            crop_masks=True,
        )
        _trace_area_stage("postprocess_completed")
        return result

    def infer(
        self,
        *,
        model_name: str,
        model_file: str,
        image_bytes_b64: str,
        inference_options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        t0 = time.time()
        _trace_area_stage(
            "engine_infer_started",
            model_name=model_name,
            model_file=Path(model_file).name,
        )
        with self._lock:
            runtime = self._load_model(model_name=model_name, model_file=model_file)
            options = self._normalize_options(inference_options)
            self._apply_cfg(runtime.cfg_obj)

            self._cfg.nms_top_k = options["nms_top_k"]
            self._cfg.nms_conf_thresh = options["nms_conf_thresh"]
            self._cfg.nms_thresh = options["nms_thresh"]
            self._cfg.max_num_detections = max(1, int(options["top_k"]))

            net = runtime.net
            net.detect.top_k = int(options["nms_top_k"])
            net.detect.conf_thresh = float(options["nms_conf_thresh"])
            net.detect.nms_thresh = float(options["nms_thresh"])
            net.detect.use_fast_nms = True
            net.detect.use_cross_class_nms = False

            image_bgr = self._decode_image(image_bytes_b64)
            self._validate_mask_working_budget(
                pixels=int(image_bgr.shape[0]) * int(image_bgr.shape[1]),
                top_k=min(int(options["top_k"]), 64),
            )

            try:
                classes_t, scores_t, boxes_t, masks_t = self._run_model_infer(
                    net=net,
                    image_bgr=image_bgr,
                    options=options,
                )
            except InferServiceError:
                raise
            except Exception as exc:
                if self._effective_device_key == "cuda" and self._gpu_policy == "warn_continue":
                    self._fallback_to_cpu(f"infer_on_cuda_failed:{exc}")
                    runtime = self._load_model(model_name=model_name, model_file=model_file)
                    self._apply_cfg(runtime.cfg_obj)
                    net = runtime.net
                    net.detect.top_k = int(options["nms_top_k"])
                    net.detect.conf_thresh = float(options["nms_conf_thresh"])
                    net.detect.nms_thresh = float(options["nms_thresh"])
                    net.detect.use_fast_nms = True
                    net.detect.use_cross_class_nms = False
                    try:
                        classes_t, scores_t, boxes_t, masks_t = self._run_model_infer(
                            net=net,
                            image_bgr=image_bgr,
                            options=options,
                        )
                    except Exception as retry_exc:
                        raise InferServiceError("infer_bad_response", f"runtime_infer_failed:{retry_exc}") from retry_exc
                else:
                    raise InferServiceError("infer_bad_response", f"runtime_infer_failed:{exc}") from exc

            instances: list[dict[str, Any]] = []
            per_class_area_px: dict[str, int] = {name: 0 for name in runtime.class_names}

            if hasattr(scores_t, "numel") and int(scores_t.numel()) > 0:
                self._validate_mask_working_budget(
                    pixels=int(image_bgr.shape[0]) * int(image_bgr.shape[1]),
                    top_k=int(scores_t.numel()),
                )
                scores_np = scores_t.detach().cpu().numpy()
                order = np.argsort(-scores_np)
                order = order[: max(1, int(options["top_k"]))]

                classes_np = classes_t.detach().cpu().numpy()
                boxes_np = boxes_t.detach().cpu().numpy()
                for i in order.tolist():
                    cls_idx = int(classes_np[i])
                    score = float(scores_np[i])
                    box = boxes_np[i].tolist()
                    if len(box) != 4:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]

                    if 0 <= cls_idx < len(runtime.class_names):
                        cls_name = runtime.class_names[cls_idx]
                    else:
                        cls_name = "未分类"

                    raw_mask = masks_t[i].detach().cpu().numpy()
                    mask_bool = (raw_mask > 0.5) if isinstance(raw_mask, np.ndarray) else None
                    area_px = int(mask_bool.sum()) if isinstance(mask_bool, np.ndarray) else max(0, (x2 - x1 + 1) * (y2 - y1 + 1))
                    polygon = self._mask_to_polygon(mask_bool)

                    per_class_area_px[cls_name] = per_class_area_px.get(cls_name, 0) + area_px
                    instances.append(
                        {
                            "class_name": cls_name,
                            "score": score,
                            "bbox": [x1, y1, x2, y2],
                            "area_px": area_px,
                            "polygon": polygon,
                            "mask": mask_bool,
                        }
                    )

            overlay_png_b64 = None
            if options["include_overlay"]:
                overlay_png_b64 = self._render_overlay(
                    image_bgr=image_bgr,
                    instances=instances,
                    class_names=runtime.class_names,
                    alpha=float(options["overlay_alpha"]),
                )
            for item in instances:
                item.pop("mask", None)

            response = {
                "instances": instances,
                "per_class_area_px": per_class_area_px,
                "engine_meta": {
                    "engine": "linux_native_yolact",
                    "cfg_name": runtime.cfg_name,
                    "model_file": runtime.model_file,
                    "class_names": list(runtime.class_names),
                    "class_mapping": "trusted_metadata_v1" if runtime.trusted_metadata else "model_name_fallback",
                    "model_metadata_version": MODEL_METADATA_VERSION,
                    **self._runtime_device_payload(),
                    "elapsed_ms": round((time.time() - t0) * 1000.0, 2),
                    "instance_count": len(instances),
                },
            }
            if overlay_png_b64 is not None:
                response["overlay_png_b64"] = overlay_png_b64
            _trace_area_stage(
                "engine_infer_completed",
                elapsed_ms=float(response["engine_meta"]["elapsed_ms"]),
                instance_count=len(instances),
            )
            return response


DEFAULT_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "yolact"
DEFAULT_WEIGHTS_DIR = os.environ.get("AREA_WEIGHTS_DIR", "/opt/area_weights")
DEFAULT_INFER_DEVICE = os.environ.get("AREA_INFER_DEVICE", "auto")
DEFAULT_GPU_POLICY = os.environ.get("AREA_INFER_GPU_POLICY", "warn_continue")


def _env_bool(name: str, default: bool) -> bool:
    token = str(os.environ.get(name, "")).strip().lower()
    if not token:
        return default
    return token in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _required_models_from_env() -> tuple[str, ...]:
    return tuple(
        Path(item.strip()).name
        for item in str(os.environ.get("AREA_REQUIRED_MODELS", "")).split(",")
        if Path(item.strip()).name
    )

engine = AreaNativeEngine(
    weights_dir=DEFAULT_WEIGHTS_DIR,
    vendor_root=os.environ.get("AREA_YOLACT_VENDOR_ROOT", str(DEFAULT_VENDOR_ROOT)),
    infer_device=DEFAULT_INFER_DEVICE,
    gpu_policy=DEFAULT_GPU_POLICY,
    max_cached_models=_env_int("AREA_MAX_CACHED_MODELS", 2, minimum=1, maximum=2),
    require_trusted_weights=_env_bool("AREA_REQUIRE_TRUSTED_WEIGHTS", True),
    verify_trusted_weights=_env_bool("AREA_VERIFY_TRUSTED_WEIGHTS", True),
    required_model_files=_required_models_from_env(),
    max_image_bytes=_env_int(
        "AREA_MAX_IMAGE_BYTES",
        48 * 1024 * 1024,
        minimum=1024,
        maximum=256 * 1024 * 1024,
    ),
    max_image_pixels=_env_int(
        "AREA_MAX_IMAGE_PIXELS",
        50_000_000,
        minimum=1_000_000,
        maximum=200_000_000,
    ),
    max_mask_working_bytes=_env_int(
        "AREA_MAX_MASK_WORKING_BYTES",
        1536 * 1024 * 1024,
        minimum=16 * 1024 * 1024,
        maximum=4 * 1024 * 1024 * 1024,
    ),
)
