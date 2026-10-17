"""Verify the actual candidate installer payload without modifying installed apps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--installer", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    args = parser.parse_args()
    root, installer = args.root, args.installer
    digest = sha(installer)
    out = root / "build/second-round/package"
    out.mkdir(parents=True, exist_ok=True)
    destination = root / "dist/windows" / ("installer-payload-" + digest[:12])
    assert sha(args.extractor) == "1c8453c198dd3fe7947f3705e59d2294ddce3dfa9f96a737e94449b119f44181"
    if not destination.exists():
        with (out / "extract.log").open("wb") as log:
            subprocess.run([str(args.extractor), "-x", "-b", "-a", "-u", "-o", "-d" + str(destination), str(installer)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    app = destination / "{app}"
    manifest = json.loads((app / "release-manifest.json").read_text(encoding="utf-8"))
    source = json.loads((root / ".fdm_snapshot.json").read_text(encoding="utf-8"))
    sources = {row["path"]: row for row in source["files"]}
    assert manifest["excluded_components"] == ["area-models"]
    assert "content-templates" in manifest["included_components"]
    for row in manifest["files"]:
        item = app / row["path"]
        assert item.stat().st_size == row["size"], row["path"]
        expected = row["sha256"]
        if expected is None:
            assert row.get("editable") and row["path"].startswith("runtime/content-templates/")
            expected = sources[row["path"]]["sha256"]
        assert sha(item) == expected, row["path"]
        assert "test-sdk" not in row["path"] and "runtime/area-models/" not in row["path"]
    for name in ("MVAPI.dll", "saa7130.dll"):
        relative = "runtime/camera/microview/x64/" + name
        assert sha(app / relative) == sources[relative]["sha256"]
    raw_path = out / "extracted-selfcheck.raw.stdout"
    with raw_path.open("wb") as stdout, (out / "selfcheck.stderr.log").open("wb") as stderr:
        subprocess.run([str(app / "FiberDiameterMeasurement.exe"), "--self-check", "--json"],
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"), stdout=stdout, stderr=stderr,
                       check=True, timeout=900)
    raw = raw_path.read_bytes()
    try:
        decoded, encoding = raw.decode("utf-8", errors="strict"), "utf-8"
    except UnicodeDecodeError:
        decoded, encoding = raw.decode("cp936", errors="strict"), "cp936"
    assert decoded.encode(encoding) == raw
    check = json.loads(decoded)
    assert check["ok"] and not check["errors"] and check["build_id"] == manifest["build_id"]
    (out / "extracted-selfcheck.json").write_text(json.dumps(check, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for row in source["files"]:
        if row["path"].startswith(("src/", "native/", "runtime/")):
            assert sha(root / row["path"]) == row["sha256"], row["path"]
    report = dict(passed=True, version=manifest["version"], build_id=manifest["build_id"],
                  source_commit=manifest["source_commit"], installer=str(installer), sha256=digest,
                  bytes=installer.stat().st_size, payload=str(app), files_verified=len(manifest["files"]),
                  source_snapshot_sha256=sha(root / ".fdm_snapshot.json"), selfcheck_encoding=encoding,
                  excluded_components=manifest["excluded_components"], actual_installation_tested=False,
                  hardware_capture_tested=False)
    (out / "installer-identity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    shutil.copy2(app / "release-manifest.json", out / "release-manifest.json")
    shutil.copy2(root / ".fdm_snapshot.json", out / "source-manifest.json")
    print(json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
