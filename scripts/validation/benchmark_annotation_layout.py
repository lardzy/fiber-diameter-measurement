"""Isolated Qt annotation layout benchmark, not capture performance acceptance.

Runs the current canvas and an explicit full-reflow LRU baseline on the same
synthetic geometry, adding 200 accepted records and toggling the candidate gate.
No camera, alignment worker, automatic analyzer or JPEG writer is exercised.
Source test fixtures are required; no user settings or projects are opened.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PySide6.QtWidgets import QApplication

from fdm.live_experiment.frames import FrameHandle
from fdm.ui.live_measurement_annotations import Annotation, layout_annotations
from fdm.ui.live_measurement_canvas import LiveMeasurementCanvas


class FullReflowLRU(LiveMeasurementCanvas):
    """Explicit test baseline: same four entries, no incremental placement."""

    def _arrange_labels(self, items, protected):
        metrics = self.fontMetrics()
        annotations = [Annotation(item["key"], item["start"], item["end"],
                                  metrics.horizontalAdvance(item["label"]) + 10.,
                                  metrics.height() + 4., item["priority"]) for item in items]
        viewport = self._visible_viewport()
        key = (tuple((a.key, a.start, a.end, a.width, a.height, a.priority) for a in annotations),
               tuple(protected), (viewport.x(), viewport.y(), viewport.width(), viewport.height()))
        if key not in self._annotation_layout_cache:
            self._annotation_layout_cache[key] = layout_annotations(
                annotations, viewport, protected_segments=protected)
            while len(self._annotation_layout_cache) > 4:
                self._annotation_layout_cache.popitem(last=False)
        else:
            self._annotation_layout_cache.move_to_end(key)
        self._annotation_cache_key = key
        self._label_placements = self._annotation_layout_cache[key]
        return self._label_placements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "canvas_fixtures", ROOT / "tests/test_live_measurement_candidate_canvas.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    app = QApplication([])
    report = dict(
        scope="Isolated QWidget; one new accepted record per step; candidate gate toggled; "
              "no capture/alignment/JPEG load",
        runtime=platform.platform(), qt_platform=app.platformName(), cases=[],
        source_fingerprints={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                             for path in ("src/fdm/ui/live_measurement_canvas.py",
                                          "src/fdm/ui/live_measurement_annotations.py")})
    for canvas_type in (FullReflowLRU, LiveMeasurementCanvas):
        controller = fixture.CanvasController()
        controller.candidates = [dict(candidate_id="c0", ordinal=1, state="available",
                                     start=(300, 200), end=(300, 220))]
        canvas = canvas_type(controller)
        canvas.resize(768, 576)
        canvas.set_frame(FrameHandle.snapshot(np.full((576, 768, 3), 180, np.uint8),
                                              source_id="benchmark", live=False))
        canvas.show()
        app.processEvents()
        rows = []
        for index in range(200):
            x, y = 40 + (index % 20) * 24, 350 + ((index // 20) % 6) * 25
            record = controller.session.add_measurement(
                20, "wool", endpoints=((x, y), (x, y + 20)),
                provenance={"measurement_number": index + 1})
            controller.current_records.append(record.id)
            samples = []
            for visible in (True, False, True, False):
                controller.candidate_overlay_ready = visible
                started = time.perf_counter()
                canvas.repaint()
                samples.append((time.perf_counter() - started) * 1000)
            rows.append(dict(records=index + 1, cold_visible_ms=samples[0],
                             cold_hidden_ms=samples[1], warm_visible_ms=samples[2],
                             warm_hidden_ms=samples[3], label_count=len(canvas._label_placements)))
        report["cases"].append(dict(implementation=canvas_type.__name__, steps=rows))
        canvas.close()
        canvas.deleteLater()
        app.processEvents()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for case in report["cases"]:
        for start, end in ((0, 50), (50, 100), (100, 200)):
            rows = case["steps"][start:end]
            timings = {key: round(float(np.percentile([row[key] for row in rows], 95)), 3)
                       for key in ("cold_visible_ms", "cold_hidden_ms", "warm_visible_ms", "warm_hidden_ms")}
            print(case["implementation"], start, end, timings)


if __name__ == "__main__":
    main()
