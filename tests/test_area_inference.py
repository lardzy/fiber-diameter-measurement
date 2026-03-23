from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import PIL  # noqa: F401
    import torch  # noqa: F401
    import torchvision  # noqa: F401

    AREA_RUNTIME_DEPS_AVAILABLE = True
except Exception:
    AREA_RUNTIME_DEPS_AVAILABLE = False

from fdm.services.area_inference import AreaInferenceService
from fdm.settings import AppSettings, application_root
from fdm.workers.area_worker import _load_engine_module


class AreaInferenceTests(unittest.TestCase):
    def test_load_engine_module_handles_dataclass_module_registration(self) -> None:
        vendor_root = application_root() / "runtime" / "area-infer" / "vendor" / "yolact"
        if not vendor_root.exists():
            self.skipTest(f"vendor root not found: {vendor_root}")

        module = _load_engine_module(vendor_root.resolve())

        self.assertTrue(hasattr(module, "AreaNativeEngine"))

    def test_area_inference_service_runs_cpu_inference_on_demo_image(self) -> None:
        if not AREA_RUNTIME_DEPS_AVAILABLE:
            self.skipTest("area inference runtime dependencies are not installed")

        image_path = application_root() / "sample_data" / "readme-demo" / "演示图片.jpg"
        weights_path = application_root() / "runtime" / "area-models" / "b_c1_1.3.pth"
        vendor_root = application_root() / "runtime" / "area-infer" / "vendor" / "yolact"
        if not image_path.exists():
            self.skipTest(f"demo image not found: {image_path}")
        if not weights_path.exists():
            self.skipTest(f"weight not found: {weights_path}")
        if not vendor_root.exists():
            self.skipTest(f"vendor root not found: {vendor_root}")

        service = AreaInferenceService()
        result = service.infer_image(
            image_path=str(image_path),
            model_name="棉-莱赛尔",
            model_file="b_c1_1.3.pth",
            settings=AppSettings(),
        )

        self.assertGreater(len(result.instances), 0)
        self.assertEqual(result.engine_meta.get("effective_device"), "cpu")
        self.assertEqual(result.engine_meta.get("requested_device"), "cpu")

