"""Prepare and independently verify an actual frozen-EXE precision UI probe.

The EXE is controlled externally by probe_professional_precision_frozen.ps1.
This helper never injects Python into that process. Its optional source-render
reference runs after the EXE exits, against its saved project/PNG/CSV artifacts.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_document(document) -> dict:
    return {"calibration": document.calibration.to_dict() if document.calibration else None,
            "features": [item.to_dict() for item in document.professional_features],
            "measurements": [item.to_dict() for item in document.professional_measurements]}


def prepare(output: Path) -> dict:
    from validate_professional_measurement_frozen import prepare as prepare_image
    from fdm.geometry import Point
    from fdm.measurement_annotation import MeasurementAnnotationLayout
    from fdm.metrology import MeasurementDefinition, MeasurementFeature
    from fdm.models import Calibration, ProjectState
    from fdm.project_io import ProjectIO
    from fdm.services.metrology_service import evaluate_document
    from fdm.version import __version__
    fixture = prepare_image(output)
    project = Path(fixture["project"])
    document = ProjectIO.load(project).documents[0]
    document.calibration = Calibration("manual", 2.0, "µm", "Synthetic precision reference", 2.0)
    features, measurements = [], []
    for index, (a, b) in enumerate(((Point(200, 300), Point(733.10196, 300)),
                                  (Point(400, 550), Point(450, 550))), 1):
        feature = MeasurementFeature(f"precision-feature-{index}", document.id, "line", (a, b))
        features.append(feature)
        measurements.append(MeasurementDefinition(f"precision-measurement-{index}", document.id,
            "distance", (feature.id,), annotation_layout=MeasurementAnnotationLayout(
                placement_mode="manual", offset_px=Point(0, -60), leader_style="dual")))
    document.apply_professional_state(features, measurements, mark_dirty=False)
    ProjectIO.save(ProjectState(__version__, [document]), project)
    expected = evaluate_document(document)
    fixture.update(version=__version__, project_sha256=sha(project),
                   measurement_values=[expected[item.id].primary.value for item in measurements],
                   semantic_document=semantic_document(document),
                   raw_project=json.loads(project.read_text(encoding="utf-8")),
                   scope="Synthetic calibrated fixed dual dimensions; actual frozen UI input and offline artifact verification, no device accuracy claim")
    (output / "fixture.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return fixture


def verify(path: Path, *, compare_reference: bool = True) -> dict:
    from fdm.project_io import ProjectIO
    from fdm.services.metrology_service import evaluate_document
    result = json.loads(path.read_text(encoding="utf-8-sig")); parent = path.parent
    checks: list[dict] = []
    def require(name, ok, **details):
        checks.append(dict(name=name, passed=bool(ok), **details))
        if not ok:
            raise AssertionError(name)
    require("external_probe_completed", result.get("passed") is True and not result.get("errors"))
    process_ids = result.get("process_ids", [])
    require("two_normal_process_exits", len(process_ids) == 2 and len(set(process_ids)) == 2
            and [row["pid"] for row in result.get("exits", [])] == process_ids
            and all(row["code"] == 0 for row in result["exits"]))
    require("no_forced_termination", result.get("forced_termination") is False and result.get("graceful_exit") is True)
    require("source_pixels_unchanged", result["source_sha256_before"].lower() == result["source_sha256_after"].lower())
    require("cancel_preserves_applied_setting", result.get("cancel_preserved_zero") is True)
    require("restore_default_is_draft_until_apply", result.get("restore_default_draft") is True)
    require("restart_preserves_four_decimals", result.get("reopened_decimals") == 4)
    def artifact(record):
        target = (parent / record["file"]).resolve()
        require("owned_artifact:"+record["file"], target.parent == parent.resolve())
        require("artifact_hash:"+record["file"], sha(target) == record["sha256"].lower())
        return target
    def forbidden_precision_fields(value, location="$", found=None):
        found = [] if found is None else found
        if isinstance(value, dict):
            for key, child in value.items():
                here = f"{location}.{key}"
                if key == "professional_measurement_label_decimals":
                    found.append(here)
                forbidden_precision_fields(child, here, found)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                forbidden_precision_fields(child, f"{location}[{index}]", found)
        return found
    def raw_without_view_values(payload):
        # Preserve every key (including unknown fields). Only viewport values
        # can legitimately change after real window placement/reopening.
        value = copy.deepcopy(payload)
        for document in value["documents"]:
            view = document.get("view_state", {})
            for key in ("zoom", "selected_measurement_id"):
                if key in view:
                    view[key] = "<view-value>"
            for key in ("x", "y"):
                if key in view.get("pan", {}):
                    view["pan"][key] = "<view-value>"
        return value
    documents, raw_projects = {}, {}
    for row in result["project_snapshots"]:
        target = artifact(row)
        require("this_save_published:"+row["phase"],
                int(row["published_last_write_utc_ticks"]) > int(row["before_last_write_utc_ticks"])
                and row["source_sha256"].lower() == row["sha256"].lower())
        raw = json.loads(target.read_text(encoding="utf-8"))
        require("no_precision_field_in_raw_project:"+row["phase"], not forbidden_precision_fields(raw))
        raw_projects[row["phase"]] = raw_without_view_values(raw)
        documents[row["phase"]] = ProjectIO.load(target).documents[0]
    require("all_project_phases", set(documents) == {"baseline", "after-settings", "reopened"})
    baseline = documents["baseline"]
    fixture = json.loads((Path(result["project"]).parent / "fixture.json").read_text(encoding="utf-8"))
    require("fixture_geometry_survives_initial_open", semantic_document(baseline) == fixture["semantic_document"])
    require("fixture_source_identity", result["source_sha256_before"].lower() == fixture["image_sha256"].lower())
    require("fixture_has_no_precision_field", not forbidden_precision_fields(fixture["raw_project"]))
    for phase, raw in raw_projects.items():
        require("complete_persisted_fields_unchanged:"+phase,
                raw == raw_without_view_values(fixture["raw_project"]),
                ignored_value_paths=["documents[*].view_state.zoom", "documents[*].view_state.pan.x",
                                     "documents[*].view_state.pan.y", "documents[*].view_state.selected_measurement_id"])

    for phase, document in documents.items():
        require("semantic_geometry_unchanged:"+phase, semantic_document(document) == semantic_document(baseline))
        require("calibrated_dual_dimensions:"+phase, len(document.professional_measurements) == 2
                and all(d.annotation_layout.leader_style == "dual" for d in document.professional_measurements))
    expected = evaluate_document(baseline)
    require("fractional_fixture_value", math.isclose(expected[baseline.professional_measurements[0].id].primary.value, 266.55098, rel_tol=1e-13))
    csv_rows = []
    for row in result["csv"]:
        with artifact(row).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        require("csv_six_metric_rows:"+row["phase"], len(rows) == 6)
        csv_rows.append(rows)
    require("csv_precision_independent", len(csv_rows) == 2 and csv_rows[0] == csv_rows[1])
    require("csv_preserves_fractional_value", math.isclose(float(csv_rows[0][0]["数值"]), 266.55098, rel_tol=1e-13))
    require("all_csv_phases", [row["phase"] for row in result["csv"]] == ["two", "four"])
    expected_csv = {(definition.id, metric.label): (metric.value, metric.unit)
                    for definition in baseline.professional_measurements
                    for metric in expected[definition.id].metrics}
    for phase, rows in zip(("two", "four"), csv_rows):
        actual_keys = [(row["测量ID"], row["指标"]) for row in rows]
        require("csv_metric_identities:"+phase, len(set(actual_keys)) == 6 and set(actual_keys) == set(expected_csv))
        for row in rows:
            value, unit = expected_csv[row["测量ID"], row["指标"]]
            require("csv_raw_metric:"+phase+":"+row["测量ID"]+":"+row["指标"],
                    math.isclose(float(row["数值"]), value, rel_tol=1e-13, abs_tol=1e-13)
                    and row["单位"] == unit and row["图像ID"] == baseline.id and row["类型"] == "distance")
    require("all_settings_phases", len(result["settings"]) == 3 and
            {row["phase"]: row["decimals"] for row in result["settings"]} == {"zero": 0, "restored": 2, "four": 4})
    for row in result["settings"]:
        saved = json.loads(artifact(row).read_text(encoding="utf-8-sig"))
        require("stored_precision:"+row["phase"], saved["professional_measurement_label_decimals"] == row["decimals"])
    for row in result["snapshots"]:
        artifact(row)
    exports = {row["decimals"]: artifact(row) for row in result["images"]}
    require("three_export_precisions", set(exports) == {0, 2, 4})
    import numpy as np
    from PIL import Image
    pixels = {value: np.array(Image.open(image).convert("RGBA")) for value, image in exports.items()}
    require("full_resolution_exports", all(image.shape == (800, 1280, 4) for image in pixels.values()))
    for decimals in (0, 4):
        changed = np.any(pixels[2] != pixels[decimals], axis=2)
        require("export_label_changes:"+str(decimals), bool(changed.any()), changed_pixels=int(changed.sum()))
        # Labels and their short dimension rails are centered near y=240/490.
        outside = changed.copy(); outside[200:280, 100:900] = False; outside[450:530, 100:900] = False
        require("only_annotation_region_changes:"+str(decimals), not outside.any())
    if compare_reference:
        os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtWidgets import QApplication
        from fdm.ui.professional_rendering import draw_professional_measurements, result_caption
        from fdm.ui.theme import apply_application_theme
        app = QApplication.instance() or QApplication([])
        apply_application_theme(app, "light")
        source = QImage(result["image"])
        require("reference_source_available", not source.isNull())
        fixed_captions = {0: ["P1 · 距离 267 µm", "P2 · 距离 25 µm"],
                          2: ["P1 · 距离 266.55 µm", "P2 · 距离 25.00 µm"],
                          4: ["P1 · 距离 266.5510 µm", "P2 · 距离 25.0000 µm"]}
        for decimals, target in exports.items():
            captions = [result_caption(d, expected[d.id], i, decimals=decimals)
                        for i, d in enumerate(baseline.professional_measurements, 1)]
            require("fixed_precision_caption_rule:"+str(decimals), captions == fixed_captions[decimals])
            reference = QImage(1280, 800, QImage.Format.Format_ARGB32_Premultiplied)
            reference.fill(Qt.GlobalColor.transparent)
            painter = QPainter(reference)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.drawImage(QPointF(0, 0), source)
            draw_professional_measurements(painter, baseline, lambda p: QPointF(p.x, p.y), label_decimals=decimals)
            painter.end()
            reference_path = parent / f"reference-{decimals}.png"; reference.save(str(reference_path))
            actual = QImage(str(target)).convertToFormat(QImage.Format.Format_RGBA8888)
            reference = reference.convertToFormat(QImage.Format.Format_RGBA8888)
            actual_bytes = bytes(actual.constBits()); reference_bytes = bytes(reference.constBits())
            require("actual_export_matches_independent_reference:"+str(decimals), actual_bytes == reference_bytes,
                    expected_captions=captions,
                    reference_file=reference_path.name, reference_sha256=sha(reference_path))
    verified = dict(schema_version=1, passed=True, probe_sha256=sha(path), checks=checks,
                    reference_comparison=compare_reference,
                    scope="Actual frozen EXE UIA input and exported artifacts; independent source rendering is a separate post-run reference, not injected code; no installation or physical metrology acceptance")
    (parent / "verified.json").write_text(json.dumps(verified, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify"); check.add_argument("--result", type=Path, required=True)
    check.add_argument("--skip-reference", action="store_true", help="Record weaker artifact-only verification; do not claim pixel-identical source comparison")
    args = parser.parse_args()
    result = prepare(args.output.resolve()) if args.command == "prepare" else verify(args.result.resolve(), compare_reference=not args.skip_reference)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
