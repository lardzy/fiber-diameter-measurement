from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdm.services.analysis_profiles import (
    ANALYSIS_PROFILE_STORE_SCHEMA_VERSION,
    AnalysisMeasurementProfile,
    AnalysisMeasurementProfileStore,
    analysis_measurement_profiles_path,
)


def _profile(**changes: object) -> AnalysisMeasurementProfile:
    values: dict[str, object] = {
        "profile_id": "fiber-histogram",
        "name": "纤维直方图",
        "tool_id": "fdm.histogram",
        "tool_version": "2",
        "parameters": {"bins": 64, "log_counts": True},
        "created_at": "2026-07-27T08:00:00+00:00",
        "updated_at": "2026-07-27T08:00:00+00:00",
    }
    values.update(changes)
    return AnalysisMeasurementProfile(**values)  # type: ignore[arg-type]


def test_profile_roundtrip_is_strict_and_parameters_are_defensive() -> None:
    parameters = {"nested": [1, 2]}
    profile = _profile(parameters=parameters)
    parameters["nested"].append(3)

    restored = AnalysisMeasurementProfile.from_dict(profile.to_dict())
    returned = restored.parameters
    returned["nested"].append(4)

    assert restored == profile
    assert restored.parameters == {"nested": [1, 2]}
    payload = profile.to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="未知"):
        AnalysisMeasurementProfile.from_dict(payload)
    with pytest.raises(ValueError):
        _profile(parameters={"bad": float("nan")})
    with pytest.raises(TypeError, match="对象键"):
        _profile(parameters={"nested": {1: "not-json"}})


def test_profile_roundtrip_persists_output_fields_and_rejects_unknown() -> None:
    profile = _profile(
        tool_id="fdm.intensity",
        parameters={"channel": "luminance"},
        output_fields=("central_tendency", "percentiles"),
    )

    restored = AnalysisMeasurementProfile.from_dict(profile.to_dict())

    assert restored.output_fields == ("central_tendency", "percentiles")
    returned = restored.to_dict()
    returned["output_fields"].append("range")  # type: ignore[union-attr]
    assert restored.output_fields == ("central_tendency", "percentiles")
    with pytest.raises(ValueError, match="未知输出字段"):
        _profile(
            tool_id="fdm.intensity",
            output_fields=("does_not_exist",),
        )


def test_profile_store_is_atomic_bounded_and_supports_upsert_delete(
    tmp_path: Path,
) -> None:
    store = AnalysisMeasurementProfileStore(tmp_path / "profiles.json")
    first = _profile()
    second = _profile(
        profile_id="particle-default",
        name="粒子默认",
        tool_id="fdm.particles",
        parameters={"min_area_px": 10},
    )

    assert store.load() == ()
    assert store.save((first, second)) == (first, second)
    assert store.load() == (first, second)
    assert json.loads(store.path.read_text(encoding="utf-8"))[
        "schema_version"
    ] == ANALYSIS_PROFILE_STORE_SCHEMA_VERSION

    updated = first.with_updates(parameters={"bins": 128})
    profiles = store.upsert(updated)
    assert next(item for item in profiles if item.profile_id == first.profile_id).parameters == {
        "bins": 128
    }
    assert store.delete(second.profile_id) == (updated,)

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        store.load()


def test_profile_store_loads_schema_v1_without_output_fields_as_all_outputs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles-v1.json"
    legacy = _profile().to_dict()
    legacy.pop("output_fields", None)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "profiles": [legacy]},
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    store = AnalysisMeasurementProfileStore(path)
    loaded = store.load()

    assert len(loaded) == 1
    assert loaded[0].output_fields is None
    store.save(loaded)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == ANALYSIS_PROFILE_STORE_SCHEMA_VERSION
    assert "output_fields" not in saved["profiles"][0]


def test_profile_names_are_unique_per_tool_and_implementation_version(
    tmp_path: Path,
) -> None:
    store = AnalysisMeasurementProfileStore(tmp_path / "profiles.json")
    v1 = _profile(
        profile_id="intensity-v1",
        name="常用强度",
        tool_id="fdm.intensity",
        tool_version="1",
    )
    v2 = _profile(
        profile_id="intensity-v2",
        name="常用强度",
        tool_id="fdm.intensity",
        tool_version="2",
    )

    assert store.save((v1, v2)) == (v1, v2)
    with pytest.raises(ValueError, match="同一分析工具及版本"):
        store.save(
            (
                v2,
                _profile(
                    profile_id="intensity-v2-copy",
                    name="常用强度",
                    tool_id="fdm.intensity",
                    tool_version="2",
                ),
            )
        )


def test_default_store_path_is_next_to_application_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(
        "fdm.services.analysis_profiles.settings_file_path",
        lambda: settings_path,
    )

    assert analysis_measurement_profiles_path() == (
        tmp_path / "analysis-measurement-profiles.json"
    )
    assert AnalysisMeasurementProfileStore().path == (
        tmp_path / "analysis-measurement-profiles.json"
    )
