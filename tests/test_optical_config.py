from dataclasses import FrozenInstanceError
import json

import pytest

from fdm.services.optical_config import (
    ALGORITHM_CHOICES, ALGORITHM_IDS, ALGORITHM_REVISIONS, AUTOMATIC_ALGORITHMS,
    OpticalAnalysisConfig, automatic_algorithm_label, measurement_method_label,
    normalize_analysis_config,
)


def test_registry_keeps_legacy_default_and_explicit_optical():
    assert OpticalAnalysisConfig().algorithm == "traditional"
    assert not OpticalAnalysisConfig().flat_enabled
    assert ALGORITHM_CHOICES is AUTOMATIC_ALGORITHMS
    assert ALGORITHM_IDS == {"traditional", "multiscale", "optical_adaptive"}
    assert automatic_algorithm_label("multiscale") == "玻纤多尺度（CU 兼容）"
    assert normalize_analysis_config()["algorithm"] == "optical_adaptive"
    for algorithm in ALGORITHM_IDS:
        assert measurement_method_label(f"automatic-{algorithm}") == automatic_algorithm_label(algorithm)
        assert measurement_method_label(f"automatic_{algorithm}") == automatic_algorithm_label(algorithm)
    assert measurement_method_label("assisted") == "贴边测径"
    assert measurement_method_label("free") == "自由测量"
    assert measurement_method_label("future") == "future"


def test_snapshot_nested_resources_are_immutable_and_json_safe(tmp_path):
    descriptor = {"path": str(tmp_path / "blank.json"), "sha256": "a" * 64,
                  "context": {"source_id": "camera", "settings": {"exposure": 15}}}
    payload = {"algorithm": "optical_adaptive", "flat_enabled": True,
               "flat_reference": descriptor, "reference_context": {"width": 768}}
    snapshot = OpticalAnalysisConfig.from_dict(payload)
    digest = snapshot.digest
    descriptor["context"]["settings"]["exposure"] = 99
    payload["reference_context"]["width"] = 1024
    mutable = snapshot.to_dict()
    mutable["flat_reference"]["context"]["settings"]["exposure"] = 23
    assert snapshot.flat_reference["context"]["settings"]["exposure"] == 15
    assert snapshot.reference_context["width"] == 768
    assert snapshot.digest == digest
    assert OpticalAnalysisConfig.from_dict(json.loads(json.dumps(snapshot.to_dict()))) == snapshot
    with pytest.raises(FrozenInstanceError):
        snapshot.algorithm = "traditional"


@pytest.mark.parametrize("invalid", [
    {"version": 2}, {"version": True}, {"algorithm": "optical"},
    {"imaging_mode": "wool"}, {"background_mode": "gaussian"},
    {"flat_enabled": "false"}, {"flat_reference": {}},
    {"reference_context": {"exposure": float("nan")}},
    {"algorithm_revision": "future-v2"},
])
def test_invalid_config_does_not_silently_change_algorithm(invalid):
    with pytest.raises((TypeError, ValueError)):
        OpticalAnalysisConfig.from_dict(invalid)


def test_explicit_algorithm_change_replaces_only_algorithm_revision():
    old = OpticalAnalysisConfig.from_dict({"imaging_mode": "double_edge"})
    new = OpticalAnalysisConfig.from_dict(old, algorithm="optical_adaptive")
    assert new.algorithm_revision == ALGORITHM_REVISIONS["optical_adaptive"]
    assert new.imaging_mode == "double_edge"
    assert new.digest != old.digest
