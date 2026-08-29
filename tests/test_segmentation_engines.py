from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import pytest

from fdm.services.segmentation_engines import OfflineSegmentationEngineService
from fdm.settings import AppSettings, OfflineSegmentationEnginePack


def _write_pack(root: Path, *, engine_id: str = "sam3") -> Path:
    root.mkdir(parents=True)
    python_link = root / "python"
    python_link.write_text(
        f"#!/bin/sh\nexec {json.dumps(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    python_link.chmod(0o755)
    resource = root / "weights.bin"
    resource.write_bytes(b"offline-weights")
    diagnostic = root / "diagnostic.py"
    diagnostic.write_text(
        "import json\nprint(json.dumps({'ok': True, 'device': 'cpu'}))\n",
        encoding="utf-8",
    )
    payload = {
        "kind": "fdm.offline_segmentation_engine",
        "schema_version": 1,
        "engine_id": engine_id,
        "display_name": "SAM3 Offline" if engine_id == "sam3" else "μSAM Offline",
        "version": "1.2.3",
        "device": "cpu",
        "python": "python",
        "diagnostic": ["@diagnostic.py"],
        "resources": [
            {
                "path": "weights.bin",
                "sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "engine.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_inspect_and_diagnose_cpu_engine_pack(tmp_path: Path) -> None:
    service = OfflineSegmentationEngineService(tmp_path / "managed")
    inspection = service.inspect(_write_pack(tmp_path / "pack"), managed=False)

    assert inspection.record.engine_id == "sam3"
    assert inspection.record.device == "cpu"
    assert inspection.resource_count == 1
    result = service.diagnose(inspection.record, timeout_seconds=10)
    assert result.ok
    assert result.details["device"] == "cpu"


def test_resource_checksum_failure_is_rejected(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "bad")
    (pack / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="校验失败"):
        OfflineSegmentationEngineService(tmp_path / "managed").inspect(pack)


def test_resource_without_checksum_is_rejected(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "unsigned")
    manifest_path = pack / "engine.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"][0].pop("sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="缺少有效 SHA-256"):
        OfflineSegmentationEngineService(tmp_path / "managed").inspect(pack)


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "bad")
    service = OfflineSegmentationEngineService(tmp_path / "managed")
    with pytest.raises(ValueError, match="不安全"):
        service.import_package(archive)
    assert not (tmp_path / "outside.txt").exists()


def test_only_managed_pack_directory_can_be_deleted(tmp_path: Path) -> None:
    source = _write_pack(tmp_path / "source")
    service = OfflineSegmentationEngineService(tmp_path / "managed")
    imported = service.import_package(source).record

    assert imported.managed
    assert Path(imported.path).is_dir()
    assert service.remove_managed_pack(imported)
    assert not Path(imported.path).exists()
    linked = service.inspect(source, managed=False).record
    assert not service.remove_managed_pack(linked)
    assert source.is_dir()


def test_engine_pack_settings_round_trip() -> None:
    pack = OfflineSegmentationEnginePack(
        engine_id="micro_sam",
        display_name="μSAM",
        version="1",
        path="/tmp/micro-sam",
        manifest_sha256="a" * 64,
        managed=False,
    )
    restored = AppSettings.from_dict(
        AppSettings(offline_segmentation_engine_packs=[pack]).to_dict()
    )
    assert len(restored.offline_segmentation_engine_packs) == 1
    assert restored.offline_segmentation_engine_packs[0].engine_id == "micro_sam"
