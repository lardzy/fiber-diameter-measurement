from __future__ import annotations

from unittest.mock import patch

import numpy as np
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from fdm.geometry import Point
from fdm.models import ImageDocument, Measurement
from fdm.services.prompt_segmentation import PromptSegmentationResult
from fdm.settings import MagicSegmentToolMode
from fdm.ui.canvas import DocumentCanvas, MagicSegmentOperationMode
from fdm.ui.prompt_segmentation_worker import (
    PromptSegmentationRequest,
    PromptSegmentationWorker,
)


class _OffsetCanvas(DocumentCanvas):
    def mounted_image_origin(self) -> Point:
        return Point(100.0, 200.0)


def _image(width: int, height: int) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("#ffffff"))
    return image


def test_magic_mask_geometry_is_translated_once_to_slide_coordinates() -> None:
    app = QApplication.instance() or QApplication([])
    image = _image(40, 30)
    document = ImageDocument(
        id="slide",
        path="/tmp/test.fdmslide",
        image_size=(400, 300),
        document_kind="digital_slide",
    )
    document.initialize_runtime_state()
    canvas = DocumentCanvas()
    canvas.set_document(document, image)
    canvas.set_tool_mode(MagicSegmentToolMode.STANDARD)
    canvas._magic_segment.request_id = 1  # noqa: SLF001
    canvas._magic_segment.pending_stage = MagicSegmentOperationMode.ADD  # noqa: SLF001
    mask = np.zeros((30, 40), dtype=bool)
    mask[4:16, 6:22] = True
    commits: list[dict[str, object]] = []
    canvas.lineCommitted.connect(lambda _doc, _mode, payload: commits.append(payload))

    canvas.apply_magic_segment_result(
        1,
        mask,
        debug_payload={
            "segmentation_source": {
                "origin_px": [100, 200],
                "focus_index": 2,
            }
        },
    )
    result = canvas.commit_magic_segment_preview()

    assert result["committed"]
    assert len(commits) == 1
    points = commits[0]["polygon_px"]
    assert min(point.x for point in points) >= 106
    assert min(point.y for point in points) >= 204
    assert max(point.x for point in points) < 122
    assert max(point.y for point in points) < 216
    assert commits[0]["debug_payload"]["segmentation_source"]["focus_index"] == 2
    canvas.close()
    app.processEvents()


def test_prompt_worker_clips_model_mask_to_stored_slide_coverage() -> None:
    worker = PromptSegmentationWorker()
    raw_mask = np.ones((10, 12), dtype=bool)

    class FakeService:
        def predict_polygon(self, **_kwargs):
            return PromptSegmentationResult(
                mask=raw_mask,
                polygon_px=[],
                area_rings_px=[],
                area_px=float(raw_mask.size),
                metadata={},
            )

    worker._services["edge_sam"] = FakeService()  # type: ignore[assignment]  # noqa: SLF001
    coverage = np.zeros_like(raw_mask)
    coverage[:, :7] = True
    results: list[PromptSegmentationResult] = []
    failures: list[str] = []
    worker.succeeded.connect(lambda _doc, _request, result: results.append(result))
    worker.failed.connect(lambda _doc, _request, reason: failures.append(reason))
    request = PromptSegmentationRequest(
        document_id="slide",
        image=_image(12, 10),
        cache_key="snapshot",
        request_id=1,
        positive_points=[Point(2, 2)],
        negative_points=[],
        tool_mode=MagicSegmentToolMode.STANDARD,
        active_stage=MagicSegmentOperationMode.ADD,
        model_variant="edge_sam",
        roi_enabled=False,
        source_token="source-token",
        valid_coverage=coverage,
    )

    with patch(
        "fdm.ui.prompt_segmentation_worker.resolve_interactive_segmentation_backend",
        return_value=("edge_sam", None),
    ):
        worker.infer(request)

    assert failures == []
    assert len(results) == 1
    result = results[0]
    assert result.mask is not None
    assert not np.any(result.mask.to_full_mask()[:, 7:])
    assert result.metadata["coverage_clipped"] is True
    assert result.metadata["source_token"] == "source-token"


def test_magic_manual_subtract_preview_keeps_frozen_slide_coordinates() -> None:
    app = QApplication.instance() or QApplication([])
    document = ImageDocument(
        id="slide",
        path="/tmp/test.fdmslide",
        image_size=(400, 300),
        document_kind="digital_slide",
    )
    document.initialize_runtime_state()
    canvas = _OffsetCanvas()
    canvas.set_document(document, _image(40, 30))
    canvas.set_tool_mode(MagicSegmentToolMode.STANDARD)
    primary_mask = np.zeros((30, 40), dtype=bool)
    primary_mask[2:26, 2:36] = True
    canvas._magic_segment.primary_mask = primary_mask  # noqa: SLF001
    canvas._magic_segment.primary_polygon = [  # noqa: SLF001
        Point(102, 202),
        Point(135, 202),
        Point(135, 225),
        Point(102, 225),
    ]
    canvas._magic_segment.primary_debug_payload = {  # noqa: SLF001
        "segmentation_source": {"origin_px": [100, 200], "focus_index": 1}
    }

    assert canvas._complete_magic_manual_subtract_polygon(  # noqa: SLF001
        [Point(110, 208), Point(120, 208), Point(120, 216), Point(110, 216)]
    )

    preview = canvas._magic_segment.subtract_polygon  # noqa: SLF001
    assert min(point.x for point in preview) >= 110
    assert min(point.y for point in preview) >= 208
    assert canvas.commit_magic_segment_preview()["committed"]
    canvas.close()
    app.processEvents()


def test_area_subtract_round_trips_through_mounted_slide_origin() -> None:
    app = QApplication.instance() or QApplication([])
    document = ImageDocument(
        id="slide",
        path="/tmp/test.fdmslide",
        image_size=(400, 300),
        document_kind="digital_slide",
    )
    document.initialize_runtime_state()
    measurement = Measurement(
        id="area",
        image_id=document.id,
        fiber_group_id=None,
        measurement_kind="area",
        mode="polygon_area",
        polygon_px=[
            Point(102, 202),
            Point(132, 202),
            Point(132, 226),
            Point(102, 226),
        ],
    )
    measurement.recalculate(None)
    document.add_measurement(measurement)
    document.select_measurement(measurement.id)
    canvas = _OffsetCanvas()
    canvas.set_document(document, _image(40, 30))
    edits: list[dict[str, object]] = []
    canvas.measurementEdited.connect(
        lambda _document_id, _measurement_id, payload: edits.append(payload)
    )

    assert canvas._complete_area_subtract_polygon(  # noqa: SLF001
        [Point(110, 208), Point(120, 208), Point(120, 216), Point(110, 216)]
    )

    assert len(edits) == 1
    result_points = edits[0]["polygon_px"]
    assert min(point.x for point in result_points) >= 102
    assert min(point.y for point in result_points) >= 202
    assert max(point.x for point in result_points) <= 132
    assert max(point.y for point in result_points) <= 226
    canvas.close()
    app.processEvents()
