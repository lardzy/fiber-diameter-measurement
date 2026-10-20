"""Verify an extracted fourth-round installer without changing installed apps."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


EXPECTED_ROOT = r"C:\tmp\fdm-cu-live-fourth-round-20261019"
EXTRACTOR_SHA256 = "1c8453c198dd3fe7947f3705e59d2294ddce3dfa9f96a737e94449b119f44181"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--installer", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    args = parser.parse_args()
    root, installer = args.root.resolve(), args.installer.resolve()
    require(str(root).casefold() == EXPECTED_ROOT.casefold(), "Use the isolated fourth-round checkout")
    require(installer.is_relative_to(root / "dist/installer"), "Installer must belong to the candidate checkout")
    require(sha(args.extractor) == EXTRACTOR_SHA256, "Unexpected extraction tool")
    digest = sha(installer)
    out = root / "build/fourth-round/package"
    out.mkdir(parents=True, exist_ok=True)
    destination = root / "dist/windows" / ("installer-payload-" + digest[:12])
    if not destination.exists():
        with (out / "extract.log").open("wb") as log:
            subprocess.run([str(args.extractor), "-x", "-b", "-a", "-u", "-o", "-d" + str(destination), str(installer)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    app = destination / "{app}"
    manifest = json.loads((app / "release-manifest.json").read_text(encoding="utf-8"))
    source = json.loads((root / ".fdm_snapshot.json").read_text(encoding="utf-8"))
    sources = {row["path"]: row for row in source["files"]}
    require(manifest["version"] == "0.4.10", "Product version must remain 0.4.10")
    require(manifest["excluded_components"] == ["area-models"], "Unexpected internal release profile")
    require("content-templates" in manifest["included_components"], "CU templates missing")
    for row in manifest["files"]:
        item = app / row["path"]
        require(item.stat().st_size == row["size"], "Payload size mismatch: " + row["path"])
        expected = row["sha256"]
        if expected is None:
            require(row.get("editable") and row["path"].startswith("runtime/content-templates/"),
                    "Unhashed non-template payload: " + row["path"])
            expected = sources[row["path"]]["sha256"]
        require(sha(item) == expected, "Payload hash mismatch: " + row["path"])
        require("test-sdk" not in row["path"] and "runtime/area-models/" not in row["path"],
                "Unexpected test SDK or area model: " + row["path"])
    for name in ("MVAPI.dll", "saa7130.dll"):
        relative = "runtime/camera/microview/x64/" + name
        require(sha(app / relative) == sources[relative]["sha256"], "Camera runtime mismatch: " + relative)
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
    check = json.loads(decoded)
    require(check["ok"] and not check["errors"] and check["build_id"] == manifest["build_id"],
            "Extracted application self-check failed")
    write_json(out / "extracted-selfcheck.json", check)
    verified_sources = 0
    for row in source["files"]:
        if row["path"].startswith(("src/", "native/", "runtime/")):
            require(sha(root / row["path"]) == row["sha256"], "Source changed during build: " + row["path"])
            verified_sources += 1
    report = dict(passed=True, version=manifest["version"], build_id=manifest["build_id"],
                  source_commit=manifest["source_commit"], source_git_baseline=source["head"],
                  installer=str(installer), sha256=digest, bytes=installer.stat().st_size,
                  payload=str(app), files_verified=len(manifest["files"]), production_sources_verified=verified_sources,
                  source_snapshot_sha256=sha(root / ".fdm_snapshot.json"), selfcheck_encoding=encoding,
                  excluded_components=manifest["excluded_components"], actual_installation_tested=False,
                  hardware_capture_tested=False)
    write_json(out / "installer-identity.json", report)
    shutil.copy2(app / "release-manifest.json", out / "release-manifest.json")
    shutil.copy2(root / ".fdm_snapshot.json", out / "source-manifest.json")
    print(json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
