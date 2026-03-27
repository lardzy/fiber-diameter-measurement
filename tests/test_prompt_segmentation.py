from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Point

try:
    from fdm.services.prompt_segmentation import PromptSegmentationService, edge_sam_model_paths

    PROMPT_SEGMENTATION_AVAILABLE = True
except ModuleNotFoundError:
    PromptSegmentationService = object  # type: ignore[assignment]
    edge_sam_model_paths = object  # type: ignore[assignment]
    PROMPT_SEGMENTATION_AVAILABLE = False


@unittest.skipUnless(PROMPT_SEGMENTATION_AVAILABLE, "requires prompt segmentation runtime dependencies")
class PromptSegmentationTests(unittest.TestCase):
    def test_edge_sam_model_paths_point_to_runtime_directory(self) -> None:
        encoder_path, decoder_path = edge_sam_model_paths()

        self.assertEqual(encoder_path.name, "edge_sam_encoder.onnx")
        self.assertEqual(decoder_path.name, "edge_sam_decoder.onnx")
        self.assertIn("runtime/segment-anything/edge_sam", encoder_path.as_posix())
        self.assertIn("runtime/segment-anything/edge_sam", decoder_path.as_posix())

    def test_embedding_cache_reuses_encoder_result_for_same_image(self) -> None:
        service = PromptSegmentationService(encoder_path="/tmp/encoder.onnx", decoder_path="/tmp/decoder.onnx")
        fake_embeddings = object()
        image = object()

        with (
            patch.object(service, "_image_to_rgb_array", return_value="demo-image") as load_mock,
            patch.object(service, "_run_encoder", return_value=(fake_embeddings, (120, 200))) as run_mock,
        ):
            first = service._embedding_for_image(image, cache_key="prompt-segmentation")
            second = service._embedding_for_image(image, cache_key="prompt-segmentation")

        self.assertIs(first, second)
        self.assertIs(first.image_embeddings, fake_embeddings)
        self.assertEqual(first.original_size, (120, 200))
        load_mock.assert_called_once()
        run_mock.assert_called_once()

    def test_embedding_cache_uses_small_lru_window(self) -> None:
        service = PromptSegmentationService(
            encoder_path="/tmp/encoder.onnx",
            decoder_path="/tmp/decoder.onnx",
            max_cache_entries=2,
        )

        with (
            patch.object(service, "_image_to_rgb_array", return_value="demo-image"),
            patch.object(
                service,
                "_run_encoder",
                side_effect=[
                    (object(), (120, 200)),
                    (object(), (120, 200)),
                    (object(), (120, 200)),
                ],
            ),
        ):
            service._embedding_for_image(object(), cache_key="first")
            service._embedding_for_image(object(), cache_key="second")
            service._embedding_for_image(object(), cache_key="third")

        self.assertEqual(list(service._embedding_cache.keys()), ["second", "third"])

    def test_predict_polygon_without_positive_points_returns_empty_result(self) -> None:
        service = PromptSegmentationService(encoder_path="/tmp/encoder.onnx", decoder_path="/tmp/decoder.onnx")

        result = service.predict_polygon(
            image=object(),
            cache_key="demo",
            positive_points=[],
            negative_points=[Point(12, 18)],
        )

        self.assertEqual(result.polygon_px, [])
        self.assertEqual(result.area_px, 0.0)
        self.assertEqual(result.metadata["reason"], "missing_positive_prompt")

    def test_run_encoder_uses_uint8_input_tensor(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable: {exc}")

        class FakeEncoderSession:
            def __init__(self) -> None:
                self.received = None

            def run(self, _outputs, inputs):
                self.received = next(iter(inputs.values()))
                return [object()]

        service = PromptSegmentationService(encoder_path="/tmp/encoder.onnx", decoder_path="/tmp/decoder.onnx")
        service._encoder_session = FakeEncoderSession()
        service._decoder_session = object()
        service._encoder_input_name = "input_image"
        service._decoder_input_names = {}

        image = np.zeros((24, 48, 3), dtype=np.uint8)
        _embeddings, original_size = service._run_encoder(image)

        self.assertEqual(original_size, (24, 48))
        self.assertIsNotNone(service._encoder_session.received)
        self.assertEqual(service._encoder_session.received.dtype.name, "uint8")
        self.assertEqual(service._encoder_session.received.shape[0], 1)

    def test_mask_to_polygon_prefers_largest_contour(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable: {exc}")

        service = PromptSegmentationService(encoder_path="/tmp/encoder.onnx", decoder_path="/tmp/decoder.onnx")
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[4:12, 4:12] = 1
        mask[25:68, 40:88] = 1

        polygon = service._mask_to_polygon(mask)

        self.assertGreaterEqual(len(polygon), 4)
        xs = [point.x for point in polygon]
        ys = [point.y for point in polygon]
        self.assertGreaterEqual(min(xs), 39.0)
        self.assertGreaterEqual(min(ys), 24.0)
        self.assertLessEqual(max(xs), 88.0)
        self.assertLessEqual(max(ys), 68.0)


if __name__ == "__main__":
    unittest.main()
