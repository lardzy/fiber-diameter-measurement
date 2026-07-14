from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.project_io import _sanitize_invalid_calibration_payloads


def _valid_calibration(label: str = "40x") -> dict[str, object]:
    return {
        "mode": "preset",
        "pixels_per_unit": 12.5,
        "unit": "um",
        "source_label": label,
    }


def _valid_preset(name: str = "40x") -> dict[str, object]:
    return {
        "name": name,
        "pixels_per_unit": 12.5,
        "unit": "um",
        "pixel_distance": 250.0,
        "actual_distance": 20.0,
        "computed_pixels_per_unit": 12.5,
    }


class ProjectIOCopyOnWriteTests(unittest.TestCase):
    def test_valid_project_reuses_all_non_issue_payload_branches(self) -> None:
        rings = [
            [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
            [[25.0, 25.0], [25.0, 75.0], [75.0, 75.0], [75.0, 25.0]],
        ]
        measurements = [{"id": "area-1", "area_rings_px": rings}]
        document = {
            "id": "document-1",
            "calibration": _valid_calibration(),
            "measurements": measurements,
        }
        documents = [document]
        presets = [_valid_preset()]
        project_calibration = _valid_calibration("project")
        load_issues: list[object] = []
        payload = {
            "project_default_calibration": project_calibration,
            "documents": documents,
            "calibration_presets": presets,
            "load_issues": load_issues,
        }

        with patch(
            "fdm.project_io.copy.deepcopy",
            side_effect=AssertionError("valid project geometry must not be deep-copied"),
        ):
            sanitized, issues = _sanitize_invalid_calibration_payloads(payload)

        self.assertIsNot(sanitized, payload)
        self.assertIs(sanitized["project_default_calibration"], project_calibration)
        self.assertIs(sanitized["documents"], documents)
        self.assertIs(sanitized["documents"][0], document)
        self.assertIs(sanitized["documents"][0]["measurements"], measurements)
        self.assertIs(sanitized["documents"][0]["measurements"][0]["area_rings_px"], rings)
        self.assertIs(sanitized["calibration_presets"], presets)
        self.assertNotIn("load_issues", sanitized)
        self.assertIs(payload["load_issues"], load_issues)
        self.assertEqual(issues, [])

    def test_invalid_document_calibration_clones_only_document_container(self) -> None:
        rings = [
            [[0.0, 0.0], [80.0, 0.0], [80.0, 80.0], [0.0, 80.0]],
            [[20.0, 20.0], [20.0, 60.0], [60.0, 60.0], [60.0, 20.0]],
        ]
        measurements = [{"id": "area-invalid-scale", "area_rings_px": rings}]
        nested_legacy_payload = {"operator": "legacy"}
        invalid_calibration = {
            "mode": "preset",
            "pixels_per_unit": 0.0,
            "unit": "um",
            "source_label": "invalid",
            "legacy": nested_legacy_payload,
        }
        invalid_document = {
            "id": "invalid-document",
            "calibration": invalid_calibration,
            "measurements": measurements,
        }
        valid_document = {
            "id": "valid-document",
            "calibration": _valid_calibration(),
            "measurements": [],
        }
        documents = [invalid_document, valid_document]
        payload = {
            "project_default_calibration": _valid_calibration("project"),
            "documents": documents,
            "calibration_presets": [_valid_preset()],
        }

        sanitized, issues = _sanitize_invalid_calibration_payloads(payload)

        self.assertIsNot(sanitized["documents"], documents)
        self.assertIsNot(sanitized["documents"][0], invalid_document)
        self.assertIsNone(sanitized["documents"][0]["calibration"])
        self.assertIs(sanitized["documents"][0]["measurements"], measurements)
        self.assertIs(sanitized["documents"][0]["measurements"][0]["area_rings_px"], rings)
        self.assertIs(sanitized["documents"][1], valid_document)

        self.assertIs(invalid_document["calibration"], invalid_calibration)
        self.assertEqual(invalid_calibration["pixels_per_unit"], 0.0)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["kind"], "document_calibration")
        self.assertEqual(issues[0]["document_id"], "invalid-document")
        self.assertEqual(issues[0]["raw_payload"], invalid_calibration)
        self.assertIsNot(issues[0]["raw_payload"], invalid_calibration)
        self.assertIsNot(issues[0]["raw_payload"]["legacy"], nested_legacy_payload)

    def test_invalid_project_default_and_presets_clone_only_changed_branches(self) -> None:
        project_nested = {"source": "old-project"}
        invalid_project_calibration = {
            "mode": "project_default",
            "pixels_per_unit": float("inf"),
            "unit": "um",
            "source_label": "invalid-project",
            "legacy": project_nested,
        }
        valid_preset = _valid_preset("valid")
        invalid_preset_nested = {"source": "old-settings"}
        invalid_preset = {
            "name": "invalid",
            "pixels_per_unit": -1.0,
            "unit": "um",
            "legacy": invalid_preset_nested,
        }
        presets = [valid_preset, invalid_preset]
        documents: list[object] = []
        payload = {
            "project_default_calibration": invalid_project_calibration,
            "documents": documents,
            "calibration_presets": presets,
        }

        sanitized, issues = _sanitize_invalid_calibration_payloads(payload)

        self.assertIsNone(sanitized["project_default_calibration"])
        self.assertIs(sanitized["documents"], documents)
        self.assertIsNot(sanitized["calibration_presets"], presets)
        self.assertEqual(sanitized["calibration_presets"], [valid_preset])
        self.assertIs(sanitized["calibration_presets"][0], valid_preset)
        self.assertIs(payload["project_default_calibration"], invalid_project_calibration)
        self.assertIs(payload["calibration_presets"], presets)

        self.assertEqual(
            [issue["kind"] for issue in issues],
            ["project_default_calibration", "calibration_preset"],
        )
        self.assertIsNot(issues[0]["raw_payload"], invalid_project_calibration)
        self.assertIsNot(issues[0]["raw_payload"]["legacy"], project_nested)
        self.assertIsNot(issues[1]["raw_payload"], invalid_preset)
        self.assertIsNot(issues[1]["raw_payload"]["legacy"], invalid_preset_nested)

    def test_legacy_issue_registry_copies_only_raw_calibration_payload(self) -> None:
        raw_nested = {"source": "legacy-sidecar"}
        raw_payload = {
            "mode": "preset",
            "pixels_per_unit": 0.0,
            "unit": "um",
            "source_label": "invalid",
            "legacy": raw_nested,
        }
        shared_metadata = {"display": "keep-shared"}
        issue = {
            "kind": "document_calibration",
            "document_id": "document-1",
            "message": "invalid",
            "raw_payload": raw_payload,
            "metadata": shared_metadata,
        }
        payload = {"documents": [], "load_issues": [issue]}

        sanitized, issues = _sanitize_invalid_calibration_payloads(payload)

        self.assertNotIn("load_issues", sanitized)
        self.assertIn("load_issues", payload)
        self.assertEqual(len(issues), 1)
        self.assertIsNot(issues[0], issue)
        self.assertIsNot(issues[0]["raw_payload"], raw_payload)
        self.assertIsNot(issues[0]["raw_payload"]["legacy"], raw_nested)
        self.assertIs(issues[0]["metadata"], shared_metadata)


if __name__ == "__main__":
    unittest.main()
