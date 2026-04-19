from __future__ import annotations

from pathlib import Path
import sys
import unittest

try:
    from PySide6.QtGui import QColor, QImage, QPainter

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Point
from fdm.services.prompt_segmentation import PromptSegmentationResult
from fdm.services.reference_instance_propagation import (
    MatcherCandidateFinder,
    ReferenceCandidateProposal,
    ReferenceInstancePropagationService,
    TemplateMatchCandidateFinder,
    area_geometry_iou,
)
from fdm.settings import MagicSegmentModelVariant


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is required for propagation tests")
class ReferenceInstancePropagationTests(unittest.TestCase):
    def _image_with_two_similar_rectangles(self) -> QImage:
        image = QImage(128, 96, QImage.Format.Format_RGB32)
        image.fill(QColor("#F0F0F0"))
        painter = QPainter(image)
        painter.fillRect(18, 20, 32, 28, QColor("#505050"))
        painter.fillRect(74, 20, 32, 28, QColor("#505050"))
        painter.end()
        return image

    def _rect_mask(self, width: int, height: int, rect: tuple[int, int, int, int]):
        import numpy as np

        x0, y0, x1, y1 = rect
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask

    def test_template_match_candidate_finder_keeps_reference_candidate(self) -> None:
        import numpy as np

        gray = np.full((80, 100), 220, dtype=np.uint8)
        gray[20:44, 18:42] = 64
        mask = np.zeros((80, 100), dtype=bool)
        mask[20:44, 18:42] = True

        finder = TemplateMatchCandidateFinder()
        candidates = finder.find_candidates(
            gray_image=gray,
            reference_mask=mask,
            reference_bbox=(18, 20, 42, 44),
        )

        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0].bbox, (18, 20, 42, 44))
        self.assertAlmostEqual(candidates[0].confidence, 1.0)

    def test_matcher_candidate_finder_is_reserved_for_phase_two(self) -> None:
        import numpy as np

        finder = MatcherCandidateFinder()
        gray = np.zeros((8, 8), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)

        with self.assertRaises(NotImplementedError):
            finder.find_candidates(
                gray_image=gray,
                reference_mask=mask,
                reference_bbox=(1, 1, 4, 4),
            )

    def test_reference_instance_propagation_service_refines_candidates_from_existing_area_reference(self) -> None:
        image = self._image_with_two_similar_rectangles()
        candidate_mask = self._rect_mask(image.width(), image.height(), (74, 20, 106, 48))

        class StubFinder:
            def find_candidates(self, *, gray_image, reference_mask, reference_bbox):
                del gray_image, reference_mask, reference_bbox
                return [
                    ReferenceCandidateProposal(
                        center_px=Point(90, 34),
                        bbox=(74, 20, 106, 48),
                        confidence=0.92,
                    )
                ]

        class FakePromptService:
            def __init__(self, mask) -> None:
                self._model_variant = MagicSegmentModelVariant.EDGE_SAM_3X
                self._mask = mask

            def clear_cache(self) -> None:
                return

            def predict_polygon(self, *, image, cache_key, positive_points, negative_points):
                del image, cache_key, positive_points, negative_points
                return PromptSegmentationResult(
                    mask=self._mask.copy(),
                    polygon_px=[Point(74, 20), Point(106, 20), Point(106, 48), Point(74, 48)],
                    area_rings_px=[],
                    area_px=float(self._mask.sum()),
                    metadata={},
                )

        service = ReferenceInstancePropagationService(
            model_variant=MagicSegmentModelVariant.EDGE_SAM_3X,
            candidate_finder=StubFinder(),
        )
        service._prompt_service = FakePromptService(candidate_mask)  # noqa: SLF001

        result = service.propagate_from_reference(
            image=image,
            cache_key="image:1",
            reference_polygon_px=[Point(18, 20), Point(50, 20), Point(50, 48), Point(18, 48)],
            reference_area_rings_px=[],
        )

        self.assertEqual(result.reference_polygon_px, [Point(18, 20), Point(50, 20), Point(50, 48), Point(18, 48)])
        self.assertEqual(len(result.candidates), 1)
        self.assertGreater(result.candidates[0].confidence, 0.48)
        self.assertEqual(result.metadata["finder"], "StubFinder")

    def test_area_geometry_iou_treats_distinct_instances_as_non_overlapping(self) -> None:
        left = [Point(10, 10), Point(30, 10), Point(30, 30), Point(10, 30)]
        right = [Point(40, 10), Point(60, 10), Point(60, 30), Point(40, 30)]

        self.assertAlmostEqual(area_geometry_iou(left, [], right, []), 0.0)

