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

try:
    from fdm.ui.area_inference_worker import AreaBatchInferenceWorker, AreaInferenceRequest

    QT_AREA_WORKER_AVAILABLE = True
except Exception:
    AreaBatchInferenceWorker = object  # type: ignore[assignment]
    AreaInferenceRequest = object  # type: ignore[assignment]
    QT_AREA_WORKER_AVAILABLE = False

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

    @unittest.skipUnless(QT_AREA_WORKER_AVAILABLE, "requires Qt area worker")
    def test_area_batch_inference_worker_emits_progress_and_success(self) -> None:
        emitted_progress: list[tuple[int, int, str]] = []
        emitted_success: list[tuple[str, object]] = []
        emitted_finished: list[tuple[bool, int, int]] = []

        worker = AreaBatchInferenceWorker(
            [
                AreaInferenceRequest(
                    document_id="doc-1",
                    image_path="/tmp/fake-image.png",
                    model_name="棉-莱赛尔",
                    model_file="b_c1_1.3.pth",
                )
            ],
            settings=AppSettings(),
        )
        worker.progress.connect(lambda index, total, path: emitted_progress.append((index, total, path)))
        worker.succeeded.connect(lambda document_id, instances: emitted_success.append((document_id, instances)))
        worker.finished.connect(lambda cancelled, completed, failed: emitted_finished.append((cancelled, completed, failed)))

        class _FakeResult:
            def __init__(self) -> None:
                self.instances = ["ok"]

        from unittest.mock import patch

        with patch("fdm.ui.area_inference_worker.AreaInferenceService.infer_image", return_value=_FakeResult()):
            worker.run()

        self.assertEqual(emitted_progress, [(1, 1, "/tmp/fake-image.png")])
        self.assertEqual(emitted_success, [("doc-1", ["ok"])])
        self.assertEqual(emitted_finished, [(False, 1, 0)])
