"""Reproduce this release snapshot including current private runtime assets."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
import sync_cu_live_windows as sync

source_snapshot = sync.snapshot


def release_snapshot(path, manual_reference=None, automatic_reference=None):
    manifest = source_snapshot(path, manual_reference, automatic_reference)
    raw_source = path.with_suffix(".sources.zip")
    path.rename(raw_source)
    with zipfile.ZipFile(raw_source) as source, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for entry in source.infolist():
            if entry.filename != ".fdm_snapshot.json":
                archive.writestr(entry.filename, source.read(entry))
        for directory in ("camera", "content-templates", "area-infer", "segment-anything"):
            for item in sorted((ROOT / "runtime" / directory).rglob("*")):
                if not item.is_file() or item.is_symlink() or any(part.startswith(".") or part == "__pycache__" for part in item.relative_to(ROOT).parts) or item.suffix == ".pyc":
                    continue
                relative = item.relative_to(ROOT).as_posix()
                data = item.read_bytes()
                manifest["files"].append({"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                archive.writestr(relative, data)
        manifest["release_version"] = "0.4.9"
        manifest["excluded_runtime"] = ["area-models"]
        archive.writestr(".fdm_snapshot.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    raw_source.unlink()
    return manifest


if __name__ == "__main__":
    sync.snapshot = release_snapshot
    sync.main()
