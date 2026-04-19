from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Point
from fdm.settings import ComplexMagicSegmentModelVariant, MagicSegmentModelVariant

try:
    from fdm.services.prompt_segmentation import (
        PromptSegmentationService,
        edge_sam_model_paths,
        finalize_magic_subtraction_mask,
        fill_magic_draft_internal_holes,
        interactive_segmentation_model_label,
        interactive_segmentation_model_paths,
        interactive_segmentation_runtime_root,
        magic_mask_to_geometry,
        normalize_magic_draft_mask,
        resolve_interactive_segmentation_backend,
        resolve_magic_segment_model_variant,
    )

    PROMPT_SEGMENTATION_AVAILABLE = True
except ModuleNotFoundError:
    PromptSegmentationService = object  # type: ignore[assignment]
    edge_sam_model_paths = object  # type: ignore[assignment]
    finalize_magic_subtraction_mask = object  # type: ignore[assignment]
    fill_magic_draft_internal_holes = object  # type: ignore[assignment]
    interactive_segmentation_model_label = object  # type: ignore[assignment]
    interactive_segmentation_model_paths = object  # type: ignore[assignment]
    interactive_segmentation_runtime_root = object  # type: ignore[assignment]
    magic_mask_to_geometry = object  # type: ignore[assignment]
    normalize_magic_draft_mask = object  # type: ignore[assignment]
    resolve_interactive_segmentation_backend = object  # type: ignore[assignment]
    resolve_magic_segment_model_variant = object  # type: ignore[assignment]
    PROMPT_SEGMENTATION_AVAILABLE = False


@unittest.skipUnless(PROMPT_SEGMENTATION_AVAILABLE, "requires prompt segmentation runtime dependencies")
class PromptSegmentationTests(unittest.TestCase):
    def test_edge_sam_model_paths_point_to_runtime_directory(self) -> None:
        encoder_path, decoder_path = edge_sam_model_paths()
        encoder_3x_path, decoder_3x_path = edge_sam_model_paths(MagicSegmentModelVariant.EDGE_SAM_3X)

        self.assertEqual(encoder_path.name, "edge_sam_encoder.onnx")
        self.assertEqual(decoder_path.name, "edge_sam_decoder.onnx")
        self.assertIn("runtime/segment-anything/edge_sam", encoder_path.as_posix())
        self.assertIn("runtime/segment-anything/edge_sam", decoder_path.as_posix())
        self.assertEqual(encoder_3x_path.name, "edge_sam_3x_encoder.onnx")
        self.assertEqual(decoder_3x_path.name, "edge_sam_3x_decoder.onnx")
        self.assertIn("runtime/segment-anything/edge_sam_3x", encoder_3x_path.as_posix())
        self.assertIn("runtime/segment-anything/edge_sam_3x", decoder_3x_path.as_posix())

    def test_interactive_segmentation_paths_cover_complex_backends(self) -> None:
        light_hq_path = interactive_segmentation_model_paths(ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM)
        efficientsam_path = interactive_segmentation_model_paths(ComplexMagicSegmentModelVariant.EFFICIENTSAM_S)

        self.assertEqual(len(light_hq_path), 1)
        self.assertEqual(light_hq_path[0].name, "sam_hq_vit_tiny.pth")
        self.assertIn("runtime/segment-anything/light_hq_sam", light_hq_path[0].as_posix())
        self.assertEqual(len(efficientsam_path), 1)
        self.assertEqual(efficientsam_path[0].name, "efficient_sam_vits.pt")
        self.assertIn("runtime/segment-anything/efficient_sam_s", efficientsam_path[0].as_posix())
        self.assertIn("runtime/segment-anything/light_hq_sam", interactive_segmentation_runtime_root(ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM).as_posix())
        self.assertEqual(interactive_segmentation_model_label(ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM), "Light HQ-SAM")
        self.assertEqual(interactive_segmentation_model_label(ComplexMagicSegmentModelVariant.EFFICIENTSAM_S), "EfficientSAM-S")

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

        self.assertIsNone(result.mask)
        self.assertEqual(result.polygon_px, [])
        self.assertEqual(result.area_rings_px, [])
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

    def test_run_encoder_uses_float32_input_tensor_when_model_requires_float(self) -> None:
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
        service._encoder_input_type = "tensor(float)"
        service._decoder_input_names = {}

        image = np.zeros((24, 48, 3), dtype=np.uint8)
        _embeddings, original_size = service._run_encoder(image)

        self.assertEqual(original_size, (24, 48))
        self.assertIsNotNone(service._encoder_session.received)
        self.assertEqual(service._encoder_session.received.dtype.name, "float32")
        self.assertEqual(service._encoder_session.received.shape[0], 1)

    def test_run_encoder_normalizes_and_pads_fixed_square_float_model_input(self) -> None:
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
        service._encoder_input_name = "image"
        service._encoder_input_type = "tensor(float)"
        service._encoder_input_shape = (1, 3, 1024, 1024)
        service._decoder_input_names = {}

        image = np.zeros((24, 48, 3), dtype=np.uint8)
        _embeddings, original_size = service._run_encoder(image)

        self.assertEqual(original_size, (24, 48))
        self.assertIsNotNone(service._encoder_session.received)
        self.assertEqual(service._encoder_session.received.dtype.name, "float32")
        self.assertEqual(service._encoder_session.received.shape, (1, 3, 1024, 1024))
        top_left = service._encoder_session.received[0, :, 0, 0]
        self.assertAlmostEqual(float(top_left[0]), -2.1179, places=3)
        self.assertAlmostEqual(float(top_left[1]), -2.0357, places=3)
        self.assertAlmostEqual(float(top_left[2]), -1.8044, places=3)
        self.assertTrue(np.allclose(service._encoder_session.received[0, :, 700, 100], 0.0))

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

    def test_resolve_magic_segment_model_variant_falls_back_to_standard_model(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            runtime_root = Path(tmp_dir)
            standard_encoder = runtime_root / "edge_sam_encoder.onnx"
            standard_decoder = runtime_root / "edge_sam_decoder.onnx"
            standard_encoder.write_bytes(b"demo")
            standard_decoder.write_bytes(b"demo")

            def fake_model_paths(model_variant: str = MagicSegmentModelVariant.EDGE_SAM):
                if model_variant == MagicSegmentModelVariant.EDGE_SAM_3X:
                    return (runtime_root / "missing_3x_encoder.onnx", runtime_root / "missing_3x_decoder.onnx")
                return (standard_encoder, standard_decoder)

            with patch("fdm.services.prompt_segmentation.interactive_segmentation_model_paths", side_effect=fake_model_paths):
                resolved_variant, fallback_message = resolve_magic_segment_model_variant(
                    MagicSegmentModelVariant.EDGE_SAM_3X
                )

        self.assertEqual(resolved_variant, MagicSegmentModelVariant.EDGE_SAM)
        self.assertIn("回退到标准 EdgeSAM", str(fallback_message))

    def test_resolve_interactive_segmentation_backend_does_not_fallback_complex_models(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            runtime_root = Path(tmp_dir)
            missing_light_hq = runtime_root / "sam_hq_vit_tiny.pth"

            def fake_model_paths(model_variant: str = MagicSegmentModelVariant.EDGE_SAM):
                if model_variant == ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM:
                    return (missing_light_hq,)
                return (runtime_root / "edge_sam_encoder.onnx", runtime_root / "edge_sam_decoder.onnx")

            with patch("fdm.services.prompt_segmentation.interactive_segmentation_model_paths", side_effect=fake_model_paths):
                resolved_variant, fallback_message = resolve_interactive_segmentation_backend(
                    ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM
                )

        self.assertEqual(resolved_variant, ComplexMagicSegmentModelVariant.LIGHT_HQ_SAM)
        self.assertIsNone(fallback_message)

    def test_normalize_magic_draft_mask_returns_none_for_empty_input(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable: {exc}")

        self.assertIsNone(normalize_magic_draft_mask(None))

        mask = np.zeros((120, 140), dtype=bool)
        self.assertIsNone(normalize_magic_draft_mask(mask))

    def test_finalize_magic_subtraction_mask_preserves_internal_hole_for_bridge_stage(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable: {exc}")

        primary = np.zeros((120, 140), dtype=bool)
        primary[10:100, 12:118] = True
        subtract = np.zeros((120, 140), dtype=bool)
        subtract[36:74, 42:84] = True

        finalized_mask, stats = finalize_magic_subtraction_mask(primary, subtract)
        _selected_mask, area_rings, polygon, geometry_stats = magic_mask_to_geometry(finalized_mask)

        self.assertIsNotNone(finalized_mask)
        self.assertEqual(int(stats["opened_holes"]), 0)
        self.assertFalse(bool(stats["discarded_fragments"]))
        self.assertEqual(len(area_rings), 2)
        self.assertGreaterEqual(len(polygon), 4)
        self.assertGreater(int(geometry_stats["opened_holes"]), 0)

    def test_finalize_magic_subtraction_mask_discards_smaller_fragments_after_partial_overlap(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable: {exc}")

        primary = np.zeros((120, 140), dtype=bool)
        primary[10:100, 12:118] = True
        subtract = np.zeros((120, 140), dtype=bool)
        subtract[10:100, 70:74] = True

        finalized_mask, stats = finalize_magic_subtraction_mask(primary, subtract)

        self.assertIsNotNone(finalized_mask)
        self.assertEqual(int(stats["opened_holes"]), 0)
        self.assertTrue(bool(stats["discarded_fragments"]))

    def test_finalize_magic_subtraction_mask_returns_empty_when_subtract_covers_primary(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable: {exc}")

        primary = np.zeros((80, 90), dtype=bool)
        primary[10:50, 12:60] = True
        subtract = np.zeros((80, 90), dtype=bool)
        subtract[6:56, 8:66] = True

        finalized_mask, stats = finalize_magic_subtraction_mask(primary, subtract)

        self.assertIsNone(finalized_mask)
        self.assertTrue(bool(stats["result_empty"]))

    def test_finalize_magic_subtraction_mask_keeps_primary_when_masks_do_not_overlap(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable: {exc}")

        primary = np.zeros((80, 90), dtype=bool)
        primary[10:40, 12:44] = True
        subtract = np.zeros((80, 90), dtype=bool)
        subtract[48:70, 56:82] = True

        finalized_mask, stats = finalize_magic_subtraction_mask(primary, subtract)

        self.assertIsNotNone(finalized_mask)
        self.assertFalse(bool(stats["had_intersection"]))
        self.assertFalse(bool(stats["discarded_fragments"]))

    def test_magic_mask_to_geometry_prefers_component_hit_by_positive_prompt(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable: {exc}")

        mask = np.zeros((120, 160), dtype=bool)
        mask[10:92, 10:108] = True
        mask[36:52, 128:140] = True

        selected_mask, area_rings, polygon, stats = magic_mask_to_geometry(
            mask,
            positive_points=[Point(133, 44)],
            negative_points=[],
        )

        self.assertIsNotNone(selected_mask)
        self.assertEqual(int(stats["selected_positive_hits"]), 1)
        self.assertEqual(len(area_rings), 1)
        self.assertGreaterEqual(len(polygon), 3)
        self.assertTrue(bool(selected_mask[44, 133]))
        self.assertFalse(bool(selected_mask[30, 30]))

    def test_fill_magic_draft_internal_holes_fills_hole_without_cutting_to_boundary(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable: {exc}")

        mask = np.zeros((80, 90), dtype=bool)
        mask[10:70, 12:74] = True
        mask[28:52, 34:54] = False

        filled = fill_magic_draft_internal_holes(mask)

        self.assertIsNotNone(filled)
        self.assertTrue(bool(filled[40, 44]))
        self.assertTrue(bool(filled[20, 20]))


if __name__ == "__main__":
    unittest.main()
