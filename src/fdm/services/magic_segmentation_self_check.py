"""Exercise the shipped ONNX models and the production ROI cache."""

from pathlib import Path


def run_magic_segmentation_self_check(resource_root: str | Path) -> dict[str, object]:
    import cv2
    import numpy as np
    import onnxruntime as ort
    from PySide6.QtGui import QImage

    from fdm.geometry import Point
    from fdm.services.prompt_segmentation import PromptSegmentationService
    from fdm.settings import MagicSegmentToolMode

    root = Path(resource_root)
    if (root / "_internal" / "runtime" / "segment-anything").is_dir():
        root = root / "_internal"
    pixels = np.clip(np.random.default_rng(613).normal(200, 4, (512, 512, 3)), 0, 255).astype(np.uint8)
    cv2.circle(pixels, (256, 256), 45, (60, 70, 90), -1)
    image = QImage(pixels.data, 512, 512, pixels.strides[0], QImage.Format.Format_RGB888).copy()
    models: dict[str, object] = {}
    for variant in ("edge_sam", "edge_sam_3x"):
        folder = root / "runtime" / "segment-anything" / variant
        service = PromptSegmentationService(
            model_variant=variant,
            encoder_path=folder / f"{variant}_encoder.onnx",
            decoder_path=folder / f"{variant}_decoder.onnx",
        )
        service.local_masks = True
        service.warmup()
        args = dict(image=image, cache_key="release-roi-probe", positive_points=[Point(256, 256)],
                    negative_points=[], tool_mode=MagicSegmentToolMode.STANDARD, roi_enabled=True)
        first = service.predict_polygon(**args)
        second = service.predict_polygon(**args, roi_workspace_box=first.metadata.get("segmentation_crop_box"))
        first_perf = first.metadata["segmentation_performance"]
        second_perf = second.metadata["segmentation_performance"]
        equal = bool(first.mask is not None and second.mask is not None
                     and first.mask.origin == second.mask.origin
                     and np.array_equal(first.mask.data, second.mask.data))
        providers = service._encoder_session.get_providers()
        record = {
            "encoder_calls_first": first_perf["encoder_calls"],
            "encoder_calls_repeat": second_perf["encoder_calls"],
            "decoder_calls_repeat": second_perf["decoder_calls"],
            "mask_equal": equal,
            "area_px": first.area_px,
            "providers": providers,
            "first_ms": first_perf["service_ms"],
            "repeat_ms": second_perf["service_ms"],
            "cache_bytes": second_perf["cache_bytes"],
        }
        record["ok"] = bool(
            equal and 0 < first.area_px < 256 * 256 and len(first.polygon_px) >= 3
            and first_perf["encoder_calls"] == 1 and second_perf["encoder_calls"] == 0
            and second_perf["decoder_calls"] == 1 and "CPUExecutionProvider" in providers
        )
        models[variant] = record
        # Release the first model's sessions before probing the next variant.
        del service
    return {"ok": all(item["ok"] for item in models.values()), "backend": "onnxruntime",
            "backend_version": ort.__version__, "models": models}
