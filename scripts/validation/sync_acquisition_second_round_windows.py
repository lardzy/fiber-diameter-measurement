"""Sync a hash-checked isolated Windows candidate including private templates."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import sync_cu_live_windows as sync

source_snapshot = sync.snapshot


def release_snapshot(path, manual_reference=None, automatic_reference=None):
    manifest = source_snapshot(path, manual_reference, automatic_reference)
    source_path = path.with_suffix(".sources.zip")
    path.rename(source_path)
    with zipfile.ZipFile(source_path) as source, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for entry in source.infolist():
            if entry.filename != ".fdm_snapshot.json":
                archive.writestr(entry.filename, source.read(entry))
        for directory in ("camera", "content-templates", "area-infer", "segment-anything"):
            for item in sorted((ROOT / "runtime" / directory).rglob("*")):
                if not item.is_file() or item.is_symlink() or any(part.startswith(".") or part == "__pycache__" for part in item.relative_to(ROOT).parts) or item.suffix == ".pyc":
                    continue
                name = item.relative_to(ROOT).as_posix()
                data = item.read_bytes()
                manifest["files"].append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                archive.writestr(name, data)
        manifest["excluded_runtime"] = ["area-models"]
        archive.writestr(".fdm_snapshot.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    source_path.unlink()
    return manifest


if __name__ == "__main__":
    sync.snapshot = release_snapshot
    sync.main()
