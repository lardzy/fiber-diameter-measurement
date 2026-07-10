from __future__ import annotations

import base64
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from app.engine import AreaNativeEngine, InferServiceError, _ModelRuntime
from app.model_metadata import MODEL_SPECS, find_model_spec, resolve_model_classes


def _write_unsafe_load_marker(path: str) -> None:
    Path(path).write_text("unsafe checkpoint code executed", encoding="utf-8")


class _ExecutableCheckpointPayload:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    def __reduce__(self):
        return _write_unsafe_load_marker, (str(self.marker_path),)


class EngineSafetyTests(unittest.TestCase):
    def test_runtime_asset_hashes_match_model_metadata_authority(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        with (project_root / "runtime_assets.toml").open("rb") as stream:
            assets = tomllib.load(stream)
        configured = {
            str(asset["target"]): str(asset["sha256"])
            for asset in assets["assets"]
            if asset.get("group") == "area-models"
        }
        metadata = {
            f"runtime/area-models/{spec['model_file']}": spec["sha256"]
            for spec in MODEL_SPECS
        }

        self.assertEqual(configured, metadata)

    def test_shipped_checkpoint_load_calls_are_weights_only(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        paths = (
            project_root / "runtime/area-infer/vendor/yolact/yolact.py",
            project_root / "runtime/area-infer/vendor/yolact/backbone.py",
            project_root / "src/fdm/services/prompt_segmentation.py",
            project_root / "src/fdm/_vendor/efficient_sam/efficient_sam.py",
        )
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            load_calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ]
            self.assertTrue(load_calls, path.name)
            for call in load_calls:
                weights_only = next(
                    (keyword.value for keyword in call.keywords if keyword.arg == "weights_only"),
                    None,
                )
                self.assertIsInstance(weights_only, ast.Constant, path.name)
                self.assertIs(weights_only.value, True, path.name)

    def test_trusted_metadata_defines_actual_reversed_class_order(self) -> None:
        classes, trusted = resolve_model_classes(model_name="棉-莱赛尔", model_file="b_c1_1.3.pth")

        self.assertTrue(trusted)
        self.assertEqual(classes, ("莱赛尔", "棉"))
        self.assertIsNotNone(find_model_spec(model_file="b_c1_1.3.pth"))

        custom_classes, custom_trusted = resolve_model_classes(
            model_name="棉-莱赛尔",
            model_file="custom.pth",
        )
        self.assertFalse(custom_trusted)
        self.assertEqual(custom_classes, ("棉", "莱赛尔"))

    def test_unknown_model_is_rejected_when_trusted_weights_are_required(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight_path = root / "unknown.pth"
            weight_path.write_bytes(b"untrusted")
            engine = AreaNativeEngine(
                weights_dir=str(root),
                vendor_root=str(root),
                require_trusted_weights=True,
            )

            with self.assertRaises(InferServiceError) as ctx:
                engine._verify_weight_path(
                    model_name="custom",
                    model_file=weight_path.name,
                    path=weight_path,
                )

        self.assertEqual(ctx.exception.code, "infer_model_load_failed")
        self.assertIn("untrusted_model_file", ctx.exception.message)

    def test_known_model_sha_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight_path = root / "b_c1_1.3.pth"
            weight_path.write_bytes(b"tampered")
            engine = AreaNativeEngine(
                weights_dir=str(root),
                vendor_root=str(root),
                require_trusted_weights=True,
                verify_trusted_weights=True,
            )

            with self.assertRaises(InferServiceError) as ctx:
                engine._verify_weight_path(
                    model_name="棉-莱赛尔",
                    model_file=weight_path.name,
                    path=weight_path,
                )

        self.assertEqual(ctx.exception.code, "infer_model_load_failed")
        self.assertIn("weight_sha256_mismatch", ctx.exception.message)

    def test_trusted_policy_cannot_disable_sha_verification(self) -> None:
        engine = AreaNativeEngine(
            weights_dir="/tmp",
            vendor_root="/tmp",
            require_trusted_weights=True,
            verify_trusted_weights=False,
        )

        self.assertTrue(engine._verify_trusted_weights)

    def test_checkpoint_loader_always_uses_weights_only(self) -> None:
        captured: dict[str, object] = {}

        class _FakeTorch:
            @staticmethod
            def device(token: str) -> str:
                return token

            @staticmethod
            def load(path: str, **kwargs):
                captured["path"] = path
                captured.update(kwargs)
                return {"layer.weight": object()}

        engine = AreaNativeEngine(weights_dir="/tmp", vendor_root="/tmp")
        engine._torch = _FakeTorch()
        state_dict = engine._load_state_dict_safely(Path("/tmp/model.pth"))

        self.assertEqual(list(state_dict), ["layer.weight"])
        self.assertIs(captured["weights_only"], True)
        self.assertEqual(captured["map_location"], "cpu")

    def test_checkpoint_loader_never_falls_back_after_safe_load_failure(self) -> None:
        class _FakeTorch:
            @staticmethod
            def device(token: str) -> str:
                return token

            @staticmethod
            def load(path: str, **kwargs):
                del path, kwargs
                raise RuntimeError("unsupported pickle global")

        engine = AreaNativeEngine(weights_dir="/tmp", vendor_root="/tmp")
        engine._torch = _FakeTorch()

        with self.assertRaises(InferServiceError) as ctx:
            engine._load_state_dict_safely(Path("/tmp/model.pth"))

        self.assertIn("safe_weights_only_load_failed", ctx.exception.message)
        self.assertIn("compatible tensor state_dict", ctx.exception.message)

    def test_executable_pickle_checkpoint_is_rejected_without_execution(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            marker = root / "executed.txt"
            checkpoint = root / "malicious.pth"
            torch.save(_ExecutableCheckpointPayload(marker), checkpoint)
            engine = AreaNativeEngine(weights_dir=str(root), vendor_root=str(root))
            engine._torch = torch

            with self.assertRaises(InferServiceError) as ctx:
                engine._load_state_dict_safely(checkpoint)

            self.assertFalse(marker.exists())
            self.assertIn("safe_weights_only_load_failed", ctx.exception.message)

    def test_model_cache_is_lru_bounded_to_two_entries(self) -> None:
        engine = AreaNativeEngine(weights_dir="/tmp", vendor_root="/tmp", max_cached_models=2)
        for index in range(3):
            runtime = _ModelRuntime(
                model_name=f"model-{index}",
                model_file=f"model-{index}.pth",
                class_names=(f"class-{index}",),
                device="cpu",
                cfg_name="cfg",
                cfg_obj=object(),
                net=object(),
                loaded_at=float(index),
                trusted_metadata=False,
            )
            engine._cache[(runtime.model_file, runtime.class_names, "cpu")] = runtime
        engine._enforce_cache_limit()

        self.assertEqual(len(engine._cache), 2)
        self.assertNotIn(("model-0.pth", ("class-0",), "cpu"), engine._cache)
        self.assertEqual(engine._cache_evictions, 1)

    def test_decoded_image_pixel_and_byte_limits_are_enforced(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")

        pixel_limited = AreaNativeEngine(
            weights_dir="/tmp",
            vendor_root="/tmp",
            max_image_pixels=3,
        )
        with patch("app.engine.cv2.imdecode") as decode:
            with self.assertRaises(InferServiceError) as pixel_ctx:
                pixel_limited._decode_image(payload)
        decode.assert_not_called()
        self.assertEqual(pixel_ctx.exception.code, "infer_request_too_large")

        byte_limited = AreaNativeEngine(
            weights_dir="/tmp",
            vendor_root="/tmp",
            max_image_bytes=2,
        )
        with self.assertRaises(InferServiceError) as byte_ctx:
            byte_limited._decode_image(payload)
        self.assertEqual(byte_ctx.exception.code, "infer_request_too_large")

    def test_pixels_times_top_k_mask_working_budget_is_enforced(self) -> None:
        engine = AreaNativeEngine(
            weights_dir="/tmp",
            vendor_root="/tmp",
            max_mask_working_bytes=64 * 1024 * 1024,
        )

        engine._validate_mask_working_budget(pixels=1_000_000, top_k=10)
        with self.assertRaises(InferServiceError) as ctx:
            engine._validate_mask_working_budget(pixels=5_000_000, top_k=10)

        self.assertEqual(ctx.exception.code, "infer_request_too_large")
        self.assertIn("mask_working_set_exceeded", ctx.exception.message)


if __name__ == "__main__":
    unittest.main()
