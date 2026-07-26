from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest import mock
import warnings
import zipfile

import numpy as np
import pytest

from fdm.analysis_artifacts import (
    AnalysisAssetKind,
    AnalysisAssetReference,
)
from fdm.services.analysis_asset_io import (
    ANALYSIS_NPZ_FORMAT,
    ANALYSIS_NPZ_MANIFEST_MEMBER,
    copy_verified_analysis_asset,
    inspect_safe_analysis_npz,
    validate_analysis_asset_reference,
    write_safe_analysis_npz,
)


def _reference_metadata(
    arrays: dict[str, np.ndarray],
    *,
    schema: str = "fdm.test-array.v1",
) -> dict[str, object]:
    return {
        "schema": schema,
        "allow_pickle": False,
        "members": {
            name: {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            for name, array in arrays.items()
        },
    }


def _reference(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    schema: str = "fdm.test-array.v1",
) -> AnalysisAssetReference:
    info = inspect_safe_analysis_npz(path)
    return AnalysisAssetReference(
        kind=AnalysisAssetKind.OTHER,
        path="analysis/result/data.npz",
        sha256=info.sha256,
        media_type="application/x-npz",
        metadata=_reference_metadata(arrays, schema=schema),
    )


def test_safe_npz_roundtrip_embeds_manifest_and_preserves_values() -> None:
    arrays = {
        "mask": np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
        "values": np.asarray([1.25, 2.5, 5.0], dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "analysis.npz"

        info = write_safe_analysis_npz(
            target,
            schema="fdm.test-array.v1",
            arrays=arrays,
            metadata={"unit": "µm", "allow_pickle": True},
        )

        assert info.path == target
        assert info.schema == "fdm.test-array.v1"
        assert info.byte_count > 0
        assert info.uncompressed_byte_count == sum(
            array.nbytes for array in arrays.values()
        )
        assert info.members == (
            ("mask", "uint8", (2, 2)),
            ("values", "float32", (3,)),
        )
        with np.load(target, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["mask"], arrays["mask"])
            np.testing.assert_allclose(archive["values"], arrays["values"])
            manifest = json.loads(
                archive[ANALYSIS_NPZ_MANIFEST_MEMBER]
                .astype(np.uint8, copy=False)
                .tobytes()
                .decode("utf-8")
            )
        assert manifest["format"] == ANALYSIS_NPZ_FORMAT
        assert manifest["allow_pickle"] is False
        assert manifest["metadata"] == {"unit": "µm"}


@pytest.mark.parametrize(
    ("schema", "arrays", "match"),
    [
        ("含空格 schema", {"values": np.arange(3)}, "schema"),
        ("fdm.test.v1", {}, "至少需要一个"),
        ("fdm.test.v1", {"../values": np.arange(3)}, "成员名称"),
        (
            "fdm.test.v1",
            {"values": np.asarray([{"unsafe": True}], dtype=object)},
            "不安全 dtype",
        ),
        (
            "fdm.test.v1",
            {"values": np.asarray(["text"], dtype="U4")},
            "不安全 dtype",
        ),
    ],
)
def test_writer_rejects_unsafe_schema_names_and_dtypes(
    schema: str,
    arrays: dict[str, np.ndarray],
    match: str,
) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        with pytest.raises((TypeError, ValueError), match=match):
            write_safe_analysis_npz(
                Path(temporary) / "result.npz",
                schema=schema,
                arrays=arrays,
            )


def test_writer_requires_npz_and_failure_preserves_old_target_bytes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with pytest.raises(ValueError, match=r"\.npz"):
            write_safe_analysis_npz(
                root / "result.npy",
                schema="fdm.test.v1",
                arrays={"values": np.arange(3)},
            )

        target = root / "result.npz"
        target.write_bytes(b"old-project-analysis-asset")

        def fail_after_partial_write(stream, **_arrays) -> None:
            stream.write(b"partial")
            raise OSError("injected encoder failure")

        with mock.patch(
            "fdm.services.analysis_asset_io.np.savez_compressed",
            side_effect=fail_after_partial_write,
        ):
            with pytest.raises(OSError, match="injected"):
                write_safe_analysis_npz(
                    target,
                    schema="fdm.test.v1",
                    arrays={"values": np.arange(3)},
                )

        assert target.read_bytes() == b"old-project-analysis-asset"
        assert not list(root.glob(".result.npz.*"))


def test_reference_validation_checks_hash_schema_and_members() -> None:
    arrays = {"values": np.arange(12, dtype=np.float32).reshape(3, 4)}
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "result.npz"
        write_safe_analysis_npz(
            target,
            schema="fdm.test-array.v1",
            arrays=arrays,
        )
        reference = _reference(target, arrays)

        validated = validate_analysis_asset_reference(target, reference)

        assert validated.sha256 == reference.sha256
        wrong_hash = AnalysisAssetReference(
            kind=reference.kind,
            path=reference.path,
            sha256="0" * 64,
            media_type=reference.media_type,
            metadata=reference.metadata,
        )
        with pytest.raises(ValueError, match="SHA256"):
            validate_analysis_asset_reference(target, wrong_hash)
        wrong_schema = AnalysisAssetReference(
            kind=reference.kind,
            path=reference.path,
            sha256=reference.sha256,
            media_type=reference.media_type,
            metadata=_reference_metadata(arrays, schema="fdm.other.v1"),
        )
        with pytest.raises(ValueError, match="schema"):
            validate_analysis_asset_reference(target, wrong_schema)
        wrong_members = AnalysisAssetReference(
            kind=reference.kind,
            path=reference.path,
            sha256=reference.sha256,
            media_type=reference.media_type,
            metadata={
                "schema": "fdm.test-array.v1",
                "members": {
                    "values": {"dtype": "float64", "shape": [3, 4]},
                },
            },
        )
        with pytest.raises(ValueError, match="成员描述"):
            validate_analysis_asset_reference(target, wrong_members)


def test_verified_copy_is_atomic_and_revalidates_destination() -> None:
    arrays = {"values": np.arange(8, dtype=np.uint16)}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.npz"
        target = root / "nested" / "target.npz"
        write_safe_analysis_npz(
            source,
            schema="fdm.test-array.v1",
            arrays=arrays,
        )
        reference = _reference(source, arrays)

        output = copy_verified_analysis_asset(source, target, reference)

        assert output.path == target
        assert output.sha256 == reference.sha256
        assert target.read_bytes() == source.read_bytes()
        same = copy_verified_analysis_asset(target, target, reference)
        assert same.sha256 == reference.sha256


def test_inspector_rejects_pickle_object_archive_and_corrupt_zip() -> None:
    manifest = {
        "format": ANALYSIS_NPZ_FORMAT,
        "schema": "fdm.unsafe.v1",
        "allow_pickle": False,
        "members": {
            "values": {"dtype": "object", "shape": [1]},
        },
        "metadata": {},
    }
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        unsafe = root / "unsafe.npz"
        np.savez(
            unsafe,
            values=np.asarray([{"pickle": True}], dtype=object),
            **{
                ANALYSIS_NPZ_MANIFEST_MEMBER: np.frombuffer(
                    manifest_bytes,
                    dtype=np.uint8,
                )
            },
        )
        with pytest.raises(ValueError, match="不安全 dtype"):
            inspect_safe_analysis_npz(unsafe)

        corrupt = root / "corrupt.npz"
        corrupt.write_bytes(b"not a zip")
        with pytest.raises(ValueError, match="合法 NPZ"):
            inspect_safe_analysis_npz(corrupt)


def test_inspector_rejects_duplicate_zip_members_before_numpy_load() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "duplicate.npz"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(target, "w") as archive:
                archive.writestr("values.npy", b"first")
                archive.writestr("values.npy", b"second")

        with pytest.raises(ValueError, match="重复成员"):
            inspect_safe_analysis_npz(target)
