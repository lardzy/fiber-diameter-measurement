from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement, new_id
from fdm.screenshot_settings import ScreenshotSettings
from fdm.services.digital_slide_store import (
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTile,
)
from fdm.settings import AppSettings, MagicSegmentToolMode
from fdm.services.segmentation_source import digital_slide_segmentation_snapshot
from fdm.services.prompt_segmentation import PromptSegmentationResult
from fdm.ui.canvas import MagicSegmentOperationMode
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.main_window import MainWindow


def test_training_export_rebuilds_verified_slide_roi_at_recorded_focus_and_origin(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    slide_path = tmp_path / "training.fdmslide"
    store = DigitalSlideStore.create(
        slide_path,
        DigitalSlideManifest(
            version=1,
            width=80,
            height=60,
            viewport_width=40,
            viewport_height=30,
            focus_levels=[0, 100],
        ),
    )
    tile = QImage(40, 30, QImage.Format.Format_RGB32)
    tile.fill(QColor("#00ff00"))
    store.write_tile(
        DigitalSlideTile(z_index=1, x=20, y=10, width=40, height=30),
        tile,
    )
    recorded_version = digital_slide_segmentation_snapshot(
        ImageDocument(
            id="version-probe",
            path=str(slide_path),
            image_size=(80, 60),
            document_kind="digital_slide",
        ),
        store,
        origin_px=Point(20, 10),
        width=40,
        height=30,
        focus_index=1,
    ).source_version
    document = ImageDocument(
        id="slide-training",
        path=str(slide_path),
        absolute_path=str(slide_path),
        image_size=(80, 60),
        document_kind="digital_slide",
    )
    document.initialize_runtime_state()
    group = document.create_group(color="#00aa88", label="棉")
    document.add_measurement(
        Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=group.id,
            mode="magic_segment",
            measurement_kind="area",
            polygon_px=[
                Point(25, 15),
                Point(42, 15),
                Point(42, 28),
                Point(25, 28),
            ],
            debug_payload={
                "segmentation_source": {
                    "kind": "digital_slide_viewport",
                    "focus_index": 1,
                    "origin_px": [20, 10],
                    "size_px": [40, 30],
                    "version": recorded_version,
                }
            },
        )
    )
    canvas = DigitalSlideCanvas()
    canvas._focus_index = 0  # noqa: SLF001 - different from recorded object on purpose
    canvas._viewport_origin = Point(0, 0)  # noqa: SLF001

    with (
        patch("fdm.ui.main_window.AppSettingsIO.load", return_value=AppSettings()),
        patch("fdm.ui.main_window.AppSettingsIO.save", return_value=tmp_path / "settings.json"),
        patch(
            "fdm.ui.main_window.ScreenshotSettingsIO.load",
            return_value=ScreenshotSettings(),
        ),
    ):
        window = MainWindow()
    try:
        window.project.documents.append(document)
        window._slide_stores[document.id] = store  # noqa: SLF001
        window._canvases[document.id] = canvas  # noqa: SLF001

        class Requested:
            payload = None

            def emit(self, payload) -> None:
                self.payload = payload

        class Worker:
            def __init__(self) -> None:
                self.requested = Requested()

            def register_request(self, _document_id: str, _request_id: int) -> None:
                pass

        worker = Worker()
        window._prompt_seg_worker = worker  # type: ignore[assignment]  # noqa: SLF001
        canvas._focus_index = 1  # noqa: SLF001
        canvas._viewport_origin = Point(20, 10)  # noqa: SLF001
        with (
            patch.object(window, "_ensure_prompt_segmentation_worker"),
            patch(
                "fdm.ui.main_window.resolve_interactive_segmentation_backend",
                return_value=("edge_sam", None),
            ),
            patch(
                "fdm.ui.main_window.interactive_segmentation_models_ready",
                return_value=True,
            ),
        ):
            window._on_canvas_magic_segment_requested(  # noqa: SLF001
                document.id,
                {
                    "request_id": 7,
                    "positive_points": [Point(25, 15)],
                    "negative_points": [Point(30, 20)],
                    "tool_mode": MagicSegmentToolMode.STANDARD,
                    "active_stage": MagicSegmentOperationMode.ADD,
                },
            )
        assert worker.requested.payload is not None
        assert worker.requested.payload.positive_points == [Point(5, 5)]
        assert worker.requested.payload.negative_points == [Point(10, 10)]
        assert worker.requested.payload.image.size().toTuple() == (40, 30)
        assert worker.requested.payload.valid_coverage is not None
        assert worker.requested.payload.source_token

        class GeometryWorker:
            def __init__(self) -> None:
                self.requested = Requested()

            def cancel_document(self, _document_id: str) -> None:
                pass

            def register_request(self, _document_id: str, _request_id: int) -> None:
                pass

        geometry_worker = GeometryWorker()
        window._fiber_quick_geometry_worker = geometry_worker  # type: ignore[assignment]  # noqa: SLF001
        canvas._fiber_quick.request_id = 8  # noqa: SLF001
        with (
            patch.object(window, "_ensure_prompt_segmentation_worker"),
            patch(
                "fdm.ui.main_window.resolve_interactive_segmentation_backend",
                return_value=("edge_sam", None),
            ),
            patch(
                "fdm.ui.main_window.interactive_segmentation_models_ready",
                return_value=True,
            ),
        ):
            window._on_canvas_magic_segment_requested(  # noqa: SLF001
                document.id,
                {
                    "request_id": 8,
                    "positive_points": [Point(28, 18)],
                    "negative_points": [],
                    "tool_mode": MagicSegmentToolMode.FIBER_QUICK,
                    "active_stage": MagicSegmentOperationMode.ADD,
                },
            )
        quick_request = worker.requested.payload
        assert quick_request.positive_points == [Point(8, 8)]
        mask = np.zeros((30, 40), dtype=bool)
        mask[4:18, 5:24] = True
        with patch.object(window, "_ensure_fiber_quick_geometry_worker"):
            window._on_prompt_segmentation_succeeded(  # noqa: SLF001
                document.id,
                8,
                PromptSegmentationResult(
                    mask=mask,
                    polygon_px=[Point(5, 4), Point(23, 4), Point(23, 17), Point(5, 17)],
                    area_rings_px=[],
                    area_px=float(mask.sum()),
                    metadata={
                        "tool_mode": MagicSegmentToolMode.FIBER_QUICK,
                        "source_token": quick_request.source_token,
                        "positive_points_px": [Point(8, 8)],
                        "negative_points_px": [],
                    },
                ),
            )
        assert geometry_worker.requested.payload.positive_points == [Point(8, 8)]
        window._on_fiber_quick_geometry_succeeded(  # noqa: SLF001
            document.id,
            8,
            SimpleNamespace(
                line_px=Line(Point(6, 9), Point(20, 9)),
                confidence=0.9,
                debug_payload={},
            ),
        )
        assert canvas._fiber_quick.preview_line == Line(  # noqa: SLF001
            Point(26, 19),
            Point(40, 19),
        )

        issues = []
        samples = window._training_dataset_slide_samples(  # noqa: SLF001
            document,
            annotation_complete=True,
            issues=issues,
        )

        assert len(samples) == 1
        sample = samples[0]
        assert sample.focus_index == 1
        assert sample.origin_px == (20, 10)
        assert sample.image.shape[:2] == (30, 40)
        assert sample.image[0, 0, 1] > 240
        assert sample.valid_coverage is not None and bool(sample.valid_coverage.all())
        assert sample.instances[0].rings_px[0][0] == (5.0, 5.0)
        assert sample.instances[0].source_verified

        replacement = QImage(40, 30, QImage.Format.Format_RGB32)
        replacement.fill(QColor("#ff0000"))
        store.write_tile(
            DigitalSlideTile(z_index=1, x=20, y=10, width=40, height=30),
            replacement,
        )
        changed_issues = []
        changed_samples = window._training_dataset_slide_samples(  # noqa: SLF001
            document,
            annotation_complete=True,
            issues=changed_issues,
        )
        assert not changed_samples[0].instances[0].source_verified
        assert "slide_source_changed" in {issue.code for issue in changed_issues}
        app.processEvents()
    finally:
        window._slide_stores.pop(document.id, None)  # noqa: SLF001
        window._canvases.pop(document.id, None)  # noqa: SLF001
        canvas.shutdown()
        canvas.close()
        store.close()
        window._reset_workspace()  # noqa: SLF001
        window.close()
