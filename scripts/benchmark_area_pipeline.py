"""Compare local postprocessing and measure real MainWindow insertion callbacks.

This bypasses model inference and does not read or modify the user's settings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from fdm.geometry import Point
from fdm.models import ImageDocument, Measurement
from fdm.services.mask_region import mask_region
from fdm.services.prompt_segmentation import finalize_magic_subtraction_mask, magic_mask_to_geometry
from fdm.ui.main_window import MainWindow


def timed(callback):
    start = time.perf_counter()
    result = callback()
    return (time.perf_counter() - start) * 1000, result


def summary(samples):
    return dict(
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        max_ms=max(samples),
        samples=len(samples),
    )


def postprocessing():
    results = []

    def finish(primary, subtract):
        result, _ = finalize_magic_subtraction_mask(primary, subtract)
        return magic_mask_to_geometry(result, select_prompt_component=False)

    for size in (2048, 4096, 8192):
        primary = np.zeros((size, size), bool)
        primary[900:1412, 900:1412] = True
        primary[1000:1040, 1000:1040] = False
        subtract = np.zeros_like(primary)
        subtract[1100:1200, 1100:1200] = True
        region = mask_region(primary)
        remove = mask_region(subtract)
        full_times = []
        local_times = []
        for _ in range(12):
            elapsed, full = timed(lambda: finish(primary, [subtract]))
            full_times.append(elapsed)
            elapsed, local = timed(lambda: finish(region, [remove]))
            local_times.append(elapsed)
        assert np.array_equal(full[0], local[0].to_full_mask())
        assert full[1:] == local[1:]
        results.append(
            dict(
                source_size=size,
                full=summary(full_times),
                local=summary(local_times),
                full_mask_bytes=primary.nbytes + subtract.nbytes,
                local_mask_bytes=region.data.nbytes + remove.data.nbytes,
                pixel_equal=True,
            )
        )
    return results


def points(count):
    return [
        Point(
            500 + 200 * math.cos(i * math.tau / count), 500 + 200 * math.sin(i * math.tau / count)
        )
        for i in range(count)
    ]


def insertions():
    _application = QApplication.instance() or QApplication([])
    with (
        TemporaryDirectory() as temporary,
        patch("fdm.settings.settings_file_path", return_value=Path(temporary) / "settings.json"),
    ):
        window = MainWindow()
        document = ImageDocument(id="benchmark", path="synthetic.png", image_size=(2048, 2048))
        document.create_group(color="#008800", label="棉")
        document.initialize_runtime_state()
        image = QImage(2048, 2048, QImage.Format.Format_RGB32)
        image.fill(0xFFFFFFFF)
        window._mount_document(document, image, tooltip="benchmark")
        for index in range(50):
            ring = points(1000)
            document.insert_measurement_incremental(
                Measurement(
                    str(index),
                    document.id,
                    document.active_group_id,
                    "magic_segment",
                    measurement_kind="area",
                    polygon_px=ring,
                    area_rings_px=[ring],
                    exact_area_px=10000,
                )
            )
        window._update_ui_for_current_document()
        results = []
        try:
            for vertices in (1000, 10000, 50000):
                samples = []
                for _ in range(8):
                    ring = points(vertices)
                    payload = dict(
                        measurement_kind="area",
                        polygon_px=ring,
                        area_rings_px=[ring],
                        exact_area_px=10000,
                    )
                    elapsed, _ = timed(
                        lambda: window._on_canvas_line_committed(
                            document.id, "magic_segment", payload
                        )
                    )
                    samples.append(elapsed)
                results.append(dict(vertices=vertices, ui_install=summary(samples)))
        finally:
            with patch.object(window, "_confirm_close_documents", return_value=True):
                window.close()
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(postprocessing=postprocessing(), insertions=insertions())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
