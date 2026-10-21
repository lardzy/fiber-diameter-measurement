"""Opt-in blank-field resources and advisory illumination diagnostics.

No camera or worker is created here. Callers capture five owned frames and
schedule these bounded operations away from the UI. Reference correction is
only for a detection copy; measurements and evidence keep the original frame.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import uuid

import cv2
import numpy as np

from fdm.atomic_io import atomic_write_bytes, atomic_write_json
from fdm.services.optical_config import OpticalAnalysisConfig, validate_reference_descriptor


DEFAULT_MEMORY_BUDGET = 256 * 1024**2
REFERENCE_FRAME_COUNT = 5
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_MANIFEST_LIMIT = 256 * 1024


class OpticalTaskCancelled(RuntimeError):
    pass


def _checkpoint(checkpoint):
    if checkpoint is not None and checkpoint():
        raise OpticalTaskCancelled("光学分析已取消。")


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _image(value):
    array = np.asarray(getattr(value, "pixels", value))
    if (array.dtype != np.uint8 or array.ndim not in (2, 3)
            or array.ndim == 3 and array.shape[2] not in (3, 4)
            or min(array.shape[:2]) <= 0):
        raise ValueError("需要非空 uint8 灰度、BGR 或 BGRA 图像。")
    return array


def _pixel_format(image):
    return "GRAY" if image.ndim == 2 else "BGR" if image.shape[2] == 3 else "BGRA"


def reference_context(frame, calibration=None, *, orientation=None, imaging_settings=None) -> dict:
    """Stable binding, intentionally excluding sequence/connection generation.

    Unreadable camera settings are represented as unknown (None). A later known
    value cannot silently use a reference captured with an unknown value.
    """
    pixels = _image(frame)
    metadata = getattr(frame, "metadata", {}) or {}
    if orientation is None:
        orientation = metadata.get("orientation")
        if orientation is None:
            orientation = {key: metadata[key] for key in ("rotation", "flip_x", "flip_y", "mirror_x", "mirror_y") if key in metadata} or None
    if imaging_settings is None:
        imaging_settings = metadata.get("imaging_settings")
        if imaging_settings is None:
            imaging_settings = {key: metadata[key] for key in ("exposure", "exposure_us", "gain", "gamma", "white_balance", "brightness", "contrast", "sdk_color", "sdk_skip") if key in metadata} or None
    if calibration is None:
        calibration_value = None
    elif isinstance(calibration, Mapping):
        calibration_value = {key: calibration[key] for key in ("x_um", "y_um", "source") if key in calibration}
    else:
        calibration_value = {key: getattr(calibration, key) for key in ("x_um", "y_um", "source") if hasattr(calibration, key)}
    result = {"version": 1, "source_id": str(getattr(frame, "source_id", metadata.get("source_id", ""))),
              "device_id": metadata.get("device_id"), "width": int(pixels.shape[1]), "height": int(pixels.shape[0]),
              "pixel_format": _pixel_format(pixels), "orientation": orientation,
              "calibration": calibration_value, "imaging_settings": imaging_settings}
    return _json_copy(result)


snapshot_context = reference_context


def _context_mismatches(expected: Mapping, actual: Mapping) -> list[str]:
    # Compare every saved or currently readable field, not just a hand-picked
    # subset that could miss a new camera property supplied by an adapter.
    return sorted(key for key in set(expected) | set(actual)
                  if expected.get(key) != actual.get(key))


def estimate_flat_field_scratch_bytes(shape, *, operation="prepare") -> int:
    """Scratch reservation; separately leased input frames are not included."""
    dimensions = tuple(int(value) for value in shape)
    if (len(dimensions) not in (2, 3) or min(dimensions) <= 0
            or len(dimensions) == 3 and dimensions[2] not in (3, 4)):
        raise ValueError("无效图像尺寸。")
    # Python integers cannot overflow at the native 32-bit size boundary.
    size = math.prod(dimensions)
    if operation == "prepare":
        return size * 8 + 1024**2
    if operation in ("apply", "load", "save"):
        row_bytes = math.prod(dimensions[1:]) * min(64, dimensions[0])
        return size * 5 + row_bytes * 40 + 1024**2
    raise ValueError("未知的平场操作。")


def _budget(memory_budget) -> int:
    if type(memory_budget) is not int or memory_budget <= 0:
        raise ValueError("内存预算必须是正整数字节数。")
    return memory_budget


def illumination_diagnostics(image) -> dict:
    """Advisory statistics on a bounded grayscale sample, never a pass/fail gate."""
    source = _image(image)
    height, width = source.shape[:2]
    # Uniform subsampling keeps source pixels and bounds the workspace; no
    # interpolation is used for saturation statistics.
    step = max(1, math.ceil(max(height, width) / 512))
    sampled = np.ascontiguousarray(source[::step, ::step])
    gray = sampled if sampled.ndim == 2 else cv2.cvtColor(sampled, cv2.COLOR_BGR2GRAY if sampled.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    p5, p50, p95 = (float(v) for v in np.percentile(gray, (5, 50, 95)))
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bright = gray[gray > threshold]
    background = float(np.mean(bright)) if bright.size else p50
    tiles = [float(np.percentile(tile, 80)) for row in np.array_split(gray, min(4, gray.shape[0]), axis=0)
             for tile in np.array_split(row, min(4, gray.shape[1]), axis=1)]
    unevenness = float(np.std(tiles) / max(1., background))
    high = float(np.mean(gray >= 254))
    low = float(np.mean(gray <= 1))
    focus = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    contrast = (p95 - p5) / 255.
    suggestions = []
    if high > .01:
        suggestions.append("部分区域接近饱和，可适当降低照明或曝光后复核边缘。")
    if low > .01:
        suggestions.append("部分区域接近全黑，可检查照明与曝光；深色纤维也可能造成此现象。")
    if unevenness > .08:
        suggestions.append("背景明暗不均，可检查照明，或在相同成像条件下采集空白视场参考。")
    if contrast < .08:
        suggestions.append("图像整体对比度偏低；可先检查照明与对焦，按实际轮廓选择实心或透亮双边结构，并人工复核候选。")
    if focus < 15:
        suggestions.append("边缘细节较少，可检查对焦；空白视场和低纹理图像也会产生此提示。")
    return {"version": 1, "advisory_only": True, "sample_step": step,
            "brightness_background": background, "bright_class_threshold": float(threshold),
            "saturated_high_fraction": high, "saturated_low_fraction": low,
            "background_unevenness": unevenness, "robust_contrast": contrast,
            "focus_laplacian_variance": focus, "gray_p05": p5, "gray_p50": p50, "gray_p95": p95,
            "suggestions": suggestions,
            "limitations": "光照和清晰度指标仅作建议，不是准确率，也不判定测量是否有效。"}


def prepare_flat_field_reference(frames: Sequence, context: Mapping, *,
                                 memory_budget=DEFAULT_MEMORY_BUDGET, checkpoint=None) -> dict:
    try:
        if len(frames) != REFERENCE_FRAME_COUNT:
            raise ValueError("空白视场参考需要恰好 5 张连续、不同的原始帧。")
        arrays = [_image(frame) for frame in frames]
        if any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("采集空白视场期间图像尺寸或格式发生了变化。")
        binding = _json_copy(dict(context))
        if int(binding.get("width", 0)) != arrays[0].shape[1] or int(binding.get("height", 0)) != arrays[0].shape[0] or binding.get("pixel_format") != _pixel_format(arrays[0]):
            raise ValueError("参考上下文与采集图像尺寸或格式不一致。")
        handles = [frame for frame in frames if hasattr(frame, "frame_id")]
        if handles and (len(handles) != len(frames) or len({frame.frame_id for frame in handles}) != len(handles)):
            raise ValueError("空白视场参考需要 5 张不同的采集帧。")
        if handles and len({frame.generation for frame in handles}) != 1:
            raise ValueError("采集空白视场期间连接发生了变化。")
        for frame in handles:
            if str(frame.source_id) != str(binding.get("source_id", "")):
                raise ValueError("采集空白视场期间设备发生了变化。")
            actual = reference_context(frame, calibration=binding.get("calibration"))
            for key in ("device_id", "orientation", "imaging_settings"):
                # The caller may supply a known setting absent from frame
                # metadata. Readable per-frame changes must still cancel.
                if actual.get(key) is not None and actual[key] != binding.get(key):
                    raise ValueError("采集空白视场期间成像设置发生了变化。")
        required = estimate_flat_field_scratch_bytes(arrays[0].shape)
        if required > _budget(memory_budget):
            return {"status": "resource_limit", "image": None, "message": "空白视场参考超过可用内存预算。", "required_bytes": required}
        _checkpoint(checkpoint)
        stack = np.stack(arrays)
        stack.partition(2, axis=0)
        reference = np.array(stack[2], copy=True, order="C")
        del stack
        _checkpoint(checkpoint)
        color = reference[..., :3] if reference.ndim == 3 and reference.shape[2] == 4 else reference
        if np.any(color <= 0):
            raise ValueError("空白视场参考包含零亮度像素，无法安全校正；请检查照明后重采。")
        reference.setflags(write=False)
        return {"status": "ready", "image": reference, "context": binding,
                "diagnostics": illumination_diagnostics(reference), "message": "已生成 5 帧中位数空白视场参考。",
                "frame_count": REFERENCE_FRAME_COUNT, "required_bytes": required}
    except OpticalTaskCancelled:
        return {"status": "cancelled", "image": None, "message": "空白视场采集已取消。"}
    except MemoryError:
        return {"status": "resource_limit", "image": None, "message": "空白视场参考内存分配失败。"}
    except (TypeError, ValueError, OverflowError, cv2.error) as exc:
        return {"status": "invalid", "image": None, "message": str(exc)}


def save_flat_field_reference(prepared: Mapping, directory=None) -> dict:
    if prepared.get("status") != "ready":
        raise ValueError(prepared.get("message", "没有可保存的空白视场参考。"))
    image = _image(prepared["image"])
    if directory is None:
        from fdm.settings import settings_file_path
        directory = settings_file_path().parent / "optical-flat-fields"
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    reference_id = uuid.uuid4().hex
    image_path = target / f"{reference_id}.png"
    manifest_path = target / f"{reference_id}.json"
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("空白视场 PNG 编码失败。")
    content = encoded.tobytes()
    digest = hashlib.sha256(content).hexdigest()
    metadata = {"version": 1, "reference_id": reference_id, "image_file": image_path.name,
                "sha256": digest, "shape": list(image.shape), "dtype": "uint8",
                "frame_count": REFERENCE_FRAME_COUNT, "created_at": datetime.now(timezone.utc).isoformat(),
                "context": _json_copy(prepared["context"]), "diagnostics": _json_copy(prepared["diagnostics"])}
    # Unique PNG first; the atomically written JSON is the publication point.
    # A failed new resource never overwrites a previously working reference.
    atomic_write_bytes(image_path, content)
    try:
        atomic_write_json(manifest_path, metadata)
    except BaseException:
        image_path.unlink(missing_ok=True)
        raise
    return {"path": str(manifest_path), "sha256": digest, "context": metadata["context"], "diagnostics": metadata["diagnostics"]}


def capture_blank_reference(frames, context, storage_root=None, *, memory_budget=DEFAULT_MEMORY_BUDGET, checkpoint=None) -> dict:
    prepared = prepare_flat_field_reference(frames, context, memory_budget=memory_budget, checkpoint=checkpoint)
    if prepared["status"] != "ready":
        raise ValueError(prepared["message"])
    _checkpoint(checkpoint)
    return save_flat_field_reference(prepared, storage_root)


def load_flat_field_reference(descriptor, context=None, *, memory_budget=DEFAULT_MEMORY_BUDGET) -> dict:
    """A missing, corrupt or incompatible reference is a recoverable result."""
    try:
        descriptor = validate_reference_descriptor(descriptor)
        if descriptor is None:
            return {"status": "missing", "image": None, "message": "未选择空白视场参考。"}
        budget = _budget(memory_budget)
        manifest_path = Path(descriptor["path"])
        if manifest_path.stat().st_size > _MANIFEST_LIMIT:
            raise ValueError("空白视场参考清单过大。")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("version") != 1 or metadata.get("dtype") != "uint8" or metadata.get("frame_count") != REFERENCE_FRAME_COUNT:
            raise ValueError("空白视场参考清单格式不受支持。")
        binding = metadata.get("context")
        if not isinstance(binding, dict) or _context_mismatches(descriptor["context"], binding):
            raise ValueError("空白视场参考上下文与配置不一致。")
        if context is not None:
            mismatch = _context_mismatches(binding, dict(context))
            if mismatch:
                return {"status": "context_mismatch", "image": None, "metadata": metadata, "mismatches": mismatch,
                        "message": "设备、尺寸或成像设置已变化，空白视场校正未应用。"}
        filename = metadata.get("image_file")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename or "\\" in filename:
            raise ValueError("空白视场参考图片路径无效。")
        image_path = manifest_path.parent / filename
        size = image_path.stat().st_size
        if size > budget:
            return {"status": "resource_limit", "image": None, "message": "空白视场参考文件超过内存预算。"}
        with image_path.open("rb") as stream:
            header = stream.read(33)
            if len(header) < 33 or header[:8] != _PNG_MAGIC or header[12:16] != b"IHDR":
                raise ValueError("空白视场参考不是有效 PNG。")
            width, height = struct.unpack(">II", header[16:24])
            shape = metadata.get("shape")
            if (not isinstance(shape, list) or len(shape) not in (2, 3)
                    or type(width) is not int or width < 1 or height < 1
                    or shape[:2] != [height, width] or header[24] != 8
                    or header[25] not in (0, 2, 6)):
                raise ValueError("空白视场参考 PNG 尺寸或像素格式无效。")
            channels = {0: 1, 2: 3, 6: 4}[header[25]]
            if shape != ([height, width] if channels == 1 else [height, width, channels]):
                raise ValueError("空白视场参考通道数与清单不一致。")
            if (binding.get("width") != width or binding.get("height") != height
                    or binding.get("pixel_format") != {1: "GRAY", 3: "BGR", 4: "BGRA"}[channels]):
                raise ValueError("空白视场参考图片与成像上下文不一致。")
            required = estimate_flat_field_scratch_bytes(shape, operation="load")
            if required > budget or size > math.prod(shape) * 2 + 65536:
                return {"status": "resource_limit", "image": None, "message": "空白视场参考解码超过内存预算。", "required_bytes": required}
            stream.seek(0)
            content = stream.read(size + 1)
        if hashlib.sha256(content).hexdigest() != descriptor["sha256"] or metadata.get("sha256") != descriptor["sha256"]:
            raise ValueError("空白视场参考哈希校验失败。")
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or list(image.shape) != shape or image.dtype != np.uint8:
            raise ValueError("空白视场参考图片损坏。")
        color = image[..., :3] if image.ndim == 3 and image.shape[2] == 4 else image
        if np.any(color <= 0):
            raise ValueError("空白视场参考包含零亮度像素。")
        image.setflags(write=False)
        return {"status": "ready", "image": image, "metadata": metadata,
                "manifest_path": str(manifest_path), "image_path": str(image_path), "message": "空白视场参考校验通过。"}
    except FileNotFoundError:
        return {"status": "missing", "image": None, "message": "空白视场参考文件缺失，校正未应用。"}
    except MemoryError:
        return {"status": "resource_limit", "image": None, "message": "空白视场参考解码内存不足。"}
    except (OSError, TypeError, ValueError, OverflowError, KeyError, cv2.error) as exc:
        return {"status": "invalid", "image": None, "message": f"空白视场参考不可用，校正未应用：{exc}"}


def flat_field_export_files(descriptor, *, memory_budget=DEFAULT_MEMORY_BUDGET) -> tuple[Path, Path]:
    loaded = load_flat_field_reference(descriptor, memory_budget=memory_budget)
    if loaded["status"] != "ready":
        raise ValueError(loaded["message"])
    return Path(loaded["manifest_path"]), Path(loaded["image_path"])


def apply_flat_field(image, descriptor, context, *, enabled=True,
                     memory_budget=DEFAULT_MEMORY_BUDGET, checkpoint=None) -> dict:
    source = _image(image)
    required = estimate_flat_field_scratch_bytes(source.shape, operation="apply") if enabled else int(source.nbytes)
    if required > _budget(memory_budget):
        return {"status": "resource_limit", "applied": False, "image": None,
                "reason": "flat_field_memory_budget", "diagnostics": {"required_bytes": required}}
    try:
        output = np.array(source, copy=True, order="C")
    except MemoryError:
        return {"status": "resource_limit", "applied": False, "image": None,
                "reason": "flat_field_memory_budget", "diagnostics": {"required_bytes": required}}
    if not enabled:
        return {"status": "disabled", "applied": False, "image": output,
                "reason": "flat_field_disabled", "diagnostics": {"message": "空白视场校正已关闭。"}}
    loaded = load_flat_field_reference(descriptor, context, memory_budget=memory_budget)
    if loaded["status"] != "ready":
        return {"status": "not_applied", "applied": False, "image": output,
                "reason": loaded["status"], "diagnostics": {key: value for key, value in loaded.items() if key not in ("image", "metadata")}}
    reference = loaded["image"]
    if reference.shape != source.shape:
        return {"status": "not_applied", "applied": False, "image": output,
                "reason": "shape_mismatch", "diagnostics": {"message": "参考与图像尺寸或格式不同，校正未应用。"}}
    color = reference[..., :3] if reference.ndim == 3 and reference.shape[2] == 4 else reference
    levels = np.mean(color, axis=(0, 1), dtype=np.float64)
    try:
        for start in range(0, source.shape[0], 64):
            _checkpoint(checkpoint)
            stop = min(source.shape[0], start + 64)
            original = source[start:stop, ..., :3] if source.ndim == 3 and source.shape[2] == 4 else source[start:stop]
            flat = color[start:stop]
            corrected = original.astype(np.float64)
            corrected *= levels
            corrected /= flat
            np.rint(corrected, out=corrected)
            np.clip(corrected, 0, 255, out=corrected)
            if source.ndim == 3 and source.shape[2] == 4:
                output[start:stop, ..., :3] = corrected.astype(np.uint8)
            else:
                output[start:stop] = corrected.astype(np.uint8)
        return {"status": "applied", "applied": True, "image": output, "reason": "",
                "diagnostics": {"message": "已在分析副本应用空白视场校正。", "reference_sha256": descriptor["sha256"],
                                "levels": np.asarray(levels).tolist()}}
    except OpticalTaskCancelled:
        return {"status": "cancelled", "applied": False, "image": None, "reason": "cancelled", "diagnostics": {}}
    except MemoryError:
        return {"status": "resource_limit", "applied": False, "image": None,
                "reason": "flat_field_memory_budget", "diagnostics": {"required_bytes": required}}


def prepare_detection_image(image, config, *, memory_budget=DEFAULT_MEMORY_BUDGET, checkpoint=None) -> dict:
    snapshot = OpticalAnalysisConfig.from_dict(config, algorithm="optical_adaptive")
    result = apply_flat_field(image, snapshot.flat_reference, snapshot.reference_context,
                             enabled=snapshot.flat_enabled, memory_budget=memory_budget, checkpoint=checkpoint)
    result["diagnostics"] = {"illumination": illumination_diagnostics(image),
                             "flat_field": {"status": result["status"], "applied": result["applied"],
                                            "reason": result["reason"], **result["diagnostics"]}}
    return result
