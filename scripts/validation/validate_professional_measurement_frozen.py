"""Prepare/verify an external UIA smoke test of the extracted Windows EXE.

Run ``prepare --output <fresh-directory>`` with the existing source Python,
then ``probe_professional_measurement_frozen.ps1`` as the interactive ordinary
Windows user. Finally run ``verify --result <probe-run/result.json>``. No code
is injected into the EXE. The verifier reads independently saved project/CSV
files and checks the expected geometric invariants and UIA evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path) -> dict:
    from fdm.models import ImageDocument, ProjectState
    from fdm.project_io import ProjectIO
    from fdm.version import __version__
    output.mkdir(parents=True, exist_ok=False)
    width, height = 1280, 800
    stride = (width * 3 + 3) & ~3
    pixels = bytearray()
    for y in range(height - 1, -1, -1):
        row = bytearray()
        for x in range(width):
            value = round((45 if y < 400 else 205) + 16 * math.sin(x / 17) + 4 * math.sin(y / 31))
            row.extend((value, value, value))
        row.extend(b"\0" * (stride - width * 3)); pixels.extend(row)
    image = output / "synthetic-edge.bmp"
    header = struct.pack("<2sIHHI", b"BM", 54 + len(pixels), 0, 0, 54)
    dib = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    image.write_bytes(header + dib + pixels)
    document = ImageDocument("professional-frozen-smoke", str(image), (width, height))
    document.initialize_runtime_state()
    project = output / "professional-smoke.fdmproj"
    ProjectIO.save(ProjectState(__version__, [document]), project)
    result = {"schema_version": 1, "version": __version__, "project": str(project), "image": str(image),
              "image_sha256": sha(image), "image_size": [width, height],
              "edge_y_px": 399.5, "scope": "Synthetic external-EXE UI smoke; no device or physical precision claim"}
    (output / "fixture.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result


def verify(path: Path) -> dict:
    from fdm.project_io import ProjectIO
    from fdm.services.metrology_service import evaluate_document
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    parent = path.parent
    checks = []
    def require(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})
        if not passed:
            raise AssertionError(name)
    require("external_probe_completed", result.get("passed") is True and result.get("graceful_exit") is True,
            errors=result.get("errors"))
    require("raw_source_unchanged", result.get("source_sha256_before", "").lower() == result.get("source_sha256_after", "").lower())
    require("two_independent_exe_runs", len(result.get("process_ids", [])) == 2 and all(e["code"] == 0 for e in result.get("exits", [])))
    documents = {}
    for record in result["project_snapshots"]:
        target = (parent / record["file"]).resolve()
        require("owned_project_snapshot", target.parent == parent.resolve())
        require("saved_project_hash:" + record["phase"], sha(target) == record["sha256"].lower())
        document = ProjectIO.load(target).documents[0]
        documents[record["phase"]] = document
        require("saved_count:" + record["phase"], len(document.professional_measurements) == record["count"])
    created, moved, undone, fitted = [documents[name] for name in (
        "02-created-distance", "03-dragged-label", "04-undo-label", "06-confirmed-caliper")]
    definition = created.professional_measurements[0]
    feature = created.get_professional_feature(definition.feature_ids[0])
    first, second = feature.points_px
    expected = math.hypot(second.x - first.x, second.y - first.y)
    require("manual_distance_has_two_distinct_points", expected > 10)
    require("manual_distance_defaults_dual", definition.annotation_layout.leader_style == "dual")
    require("required_dual_feature", "measurement-dual-dimensions/v1" in json.loads((parent/"02-created-distance.fdmproj").read_text())["required_features"])
    original_result = evaluate_document(created)[definition.id]
    require("saved_distance_matches_points", math.isclose(original_result.primary.value, expected, rel_tol=1e-12))
    require("label_drag_changed_layout", moved.professional_measurements[0].annotation_layout != definition.annotation_layout)
    require("label_drag_preserved_source_geometry", moved.professional_features == created.professional_features)
    require("label_drag_preserved_value", evaluate_document(moved)[definition.id] == original_result)
    require("undo_restored_layout", undone.professional_measurements[0].annotation_layout == definition.annotation_layout)
    require("undo_preserved_source_geometry", undone.professional_features == created.professional_features)
    require("cancelled_preview_did_not_add_result", result.get("caliper_cancelled_without_commit") is True and len(fitted.professional_measurements) == 2)
    caliper = fitted.professional_measurements[1]
    source = fitted.get_professional_feature(caliper.feature_ids[0])
    evaluated = evaluate_document(fitted)[caliper.id]
    require("confirmed_caliper_retains_evidence", source.kind == "caliper_line" and bool(source.quality.get("profiles")) and evaluated.valid)
    # The fixture's sole strong horizontal transition lies halfway between two
    # adjacent pixel rows; this is a software synthetic edge, not real accuracy.
    mean_y = sum(p.y for p in source.points_px) / len(source.points_px)
    require("caliper_matches_synthetic_edge", abs(mean_y - 399.5) <= 1, mean_y=mean_y)
    csv_path = parent / result["csv"]["file"]
    require("csv_hash", sha(csv_path) == result["csv"]["sha256"].lower())
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require("independent_metric_csv", len(rows) == 4 and len(rows) == result["csv"]["rows"], rows=len(rows))
    require("reopen_visible_results", result.get("reopened") is True)
    for snapshot in result["snapshots"]:
        target = parent / snapshot["file"]
        require("screenshot_hash:"+snapshot["phase"], sha(target) == snapshot["sha256"].lower())
    verified = {"schema_version": 1, "passed": True, "probe_result": str(path), "probe_sha256": sha(path),
                "checks": checks, "scope": "Actual frozen EXE UIA input and persisted project/CSV verification; source evaluator independently checks saved geometry; no installation, full regression or real sample accuracy claim"}
    (parent / "verified.json").write_text(json.dumps(verified, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").add_argument("--output", type=Path, required=True)
    sub.add_parser("verify").add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output.resolve()) if args.command == "prepare" else verify(args.result.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
