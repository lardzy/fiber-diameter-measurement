"""Verify the isolated brightness installer by extraction, never by installation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import uuid


EXPECTED_ROOT = r"C:\tmp\fdm-brightness-release-20261023"
EXPECTED_VERSION = "0.4.11"
EXTRACTOR_SHA256 = "1c8453c198dd3fe7947f3705e59d2294ddce3dfa9f96a737e94449b119f44181"
CAMERA_SHA256 = {
    "runtime/camera/microview/x64/MVAPI.dll": "1d22ce270ea4aa37fb1073943a707c82f44f255164e480e1877d606b531933ea",
    "runtime/camera/microview/x64/saa7130.dll": "cbb60864f7c2526ce32515132bc04b89eb2c5e177ec5af62804e8b68e7cb6315",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def manifest_files(manifest: dict) -> dict[str, dict]:
    rows = manifest.get("files")
    require(isinstance(rows, list) and bool(rows), "Missing file inventory")
    result = {}
    for row in rows:
        token = row.get("path", "")
        require(isinstance(token, str) and bool(token) and "\\" not in token and ":" not in token,
                f"Invalid manifest path: {token!r}")
        relative = PurePosixPath(token)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.as_posix() == token, f"Unsafe manifest path: {token!r}")
        require(token.casefold() not in result, f"Duplicate case-insensitive manifest path: {token}")
        result[token.casefold()] = row
    return result


def contained_file(root: Path, token: str) -> Path:
    path = root / token
    require(path.resolve().is_relative_to(root.resolve()) and path.is_file(), f"Missing/escaped file: {token}")
    return path


def verify_sources(root: Path, sources: dict[str, dict]) -> int:
    verified = 0
    for row in sources.values():
        token = row["path"]
        if token.startswith(("src/", "native/", "runtime/")):
            item = contained_file(root, token)
            require(item.stat().st_size == row["size"] and sha(item) == row["sha256"],
                    "Production source changed during build: " + token)
            verified += 1
    require(verified > 0, "No production sources verified")
    for name in ("src/fdm/ui/slide_brightness.py", "src/fdm/ui/slide_brightness_control.py"):
        require(name in sources, "Brightness module missing from source snapshot: " + name)
    return verified


def require_worker_selfcheck(report: dict) -> tuple[dict, dict]:
    live = report.get("functional_checks", {}).get("cu_live_measurement", {})
    require(live.get("ok") is True and live.get("revision") == 6, "Actual CU worker probe revision 6 required")
    required = {"candidate_alignment_worker", "candidate_alignment_same", "candidate_alignment_noisy",
                "candidate_alignment_drift", "candidate_alignment_reference_reset", "candidate_alignment_closed",
                "candidate_alignment_child_process", "optical_adaptive", "optical_empty", "optical_worker"}
    cases = live.get("cases", {})
    require(all(cases.get(name) is True for name in required), "Alignment/optical self-check cases failed")
    alignment = live.get("candidate_alignment_worker", {})
    runtime = alignment.get("worker_runtime", {})
    require(alignment.get("ok") is True and alignment.get("closed") is True
            and alignment.get("child_process") is True, "Actual alignment subprocess and closure not verified")
    require(isinstance(runtime.get("pid"), int) and runtime["pid"] > 0
            and isinstance(runtime.get("opencv_threads"), int) and runtime["opencv_threads"] >= 1,
            "Missing alignment worker runtime evidence")
    optical = live.get("optical_worker", {})
    require(optical.get("ok") is True and optical.get("child_process") is True
            and isinstance(optical.get("worker_pid"), int) and optical["worker_pid"] > 0,
            "Actual optical worker subprocess not verified")
    return alignment, optical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--installer", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    args = parser.parse_args()
    require(os.name == "nt", "Run this verifier on Windows")
    root, installer, extractor = args.root.resolve(), args.installer.resolve(), args.extractor.resolve()
    require(str(root).casefold() == EXPECTED_ROOT.casefold(), "Use the isolated brightness-release checkout")
    require(installer.is_relative_to(root / "dist/installer") and installer.is_file(), "Installer is outside the candidate checkout")
    require(sha(extractor) == EXTRACTOR_SHA256, "Unexpected Inno extraction tool")
    out = (root / "build/package").resolve()
    require(out.is_relative_to(root), "Package evidence directory escapes the isolated checkout")
    run = out / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    report = dict(passed=False, version=EXPECTED_VERSION, installer=str(installer), sha256=sha(installer),
                  bytes=installer.stat().st_size, extractor_sha256=EXTRACTOR_SHA256,
                  run_directory=str(run), actual_installation_tested=False, hardware_capture_tested=False,
                  registry_or_file_associations_modified=False, errors=[])
    try:
        source_path = root / ".fdm_snapshot.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        sources = manifest_files(source)
        report["source_snapshot_sha256"] = sha(source_path)
        report["source_git_baseline"] = source["head"]
        report["production_sources_verified"] = verify_sources(root, sources)
        shutil.copy2(source_path, out / "source-manifest.json")
        destination = root / "dist/windows" / ("installer-payload-" + report["sha256"][:12])
        require(destination.resolve().is_relative_to(root), "Extraction directory escapes the isolated checkout")
        marker = destination / "extraction-identity.json"
        if destination.exists():
            require(marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == {
                "installer_sha256": report["sha256"], "extractor_sha256": EXTRACTOR_SHA256},
                "Existing payload is incomplete or belongs to another extraction; preserve it for inspection")
        else:
            with (run / "extract.log").open("wb") as log:
                subprocess.run([str(extractor), "-x", "-b", "-a", "-u", "-o", "-d" + str(destination), str(installer)],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
            write_json(marker, dict(installer_sha256=report["sha256"], extractor_sha256=EXTRACTOR_SHA256))
        app = destination / "{app}"
        report["payload"] = str(app)
        manifest_path = contained_file(app, "release-manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inventory = manifest_files(manifest)
        require(manifest["version"] == EXPECTED_VERSION, "Wrong product version")
        if manifest["source_commit"] == "unknown":
            require(not (root / ".git").exists(), "Unknown package commit despite a Git checkout")
            report["provenance_mode"] = "hashed-source-snapshot-without-git"
        else:
            require(manifest["source_commit"] == source["head"], "Package commit does not match source snapshot")
            report["provenance_mode"] = "git-commit-and-hashed-source-snapshot"
        report["manifest_source_dirty"] = manifest.get("source_dirty")
        report["manifest_source_dirty_entries"] = manifest.get("source_dirty_entries", [])
        require(manifest["profile"] == "full" and manifest["clean_build"] is True, "Clean full-profile build required")
        require(manifest["excluded_components"] == ["area-models"]
                and "content-templates" in manifest["included_components"], "Wrong internal release profile")
        actual = {p.relative_to(app).as_posix().casefold() for p in app.rglob("*")
                  if p.is_file() and p != manifest_path}
        require(actual == set(inventory), "Extracted payload differs from manifest inventory")
        templates = []
        for key, row in inventory.items():
            item = contained_file(app, row["path"])
            require(item.stat().st_size == row["size"], "Payload size mismatch: " + row["path"])
            require("test-sdk" not in key.replace("_", "-") and "runtime/area-models/" not in key,
                    "Unexpected test SDK or area model: " + row["path"])
            expected = row["sha256"]
            if key.startswith("runtime/content-templates/"):
                require(key in sources, "Internal template not in source snapshot: " + row["path"])
                require(sha(item) == sources[key]["sha256"], "Internal template differs from source: " + row["path"])
                templates.append(row["path"])
            if expected is None:
                require(row.get("editable") is True and key.startswith("runtime/content-templates/"),
                        "Unhashed non-template payload: " + row["path"])
                expected = sources[key]["sha256"]
            require(sha(item) == expected, "Payload hash mismatch: " + row["path"])
        expected_templates = {key for key in sources if key.startswith("runtime/content-templates/")}
        require(bool(templates) and {name.casefold() for name in templates} == expected_templates,
                "Internal template inventory differs from source snapshot")
        for token, original_sha in CAMERA_SHA256.items():
            require(sources[token.casefold()]["sha256"] == original_sha and sha(contained_file(app, token)) == original_sha,
                    "Original camera runtime changed: " + token)
        report.update(build_id=manifest["build_id"], source_commit=manifest["source_commit"],
                      release_manifest_sha256=sha(manifest_path), files_verified=len(inventory),
                      internal_templates_verified=templates, camera_sha256=CAMERA_SHA256,
                      excluded_components=manifest["excluded_components"])
        shutil.copy2(manifest_path, out / "release-manifest.json")
        config, temporary = run / "isolated-config", run / "temp"
        config.mkdir()
        temporary.mkdir()
        env = dict(os.environ, APPDATA=str(config), LOCALAPPDATA=str(config), TEMP=str(temporary), TMP=str(temporary),
                   PYTHONIOENCODING="utf-8", QT_QPA_PLATFORM="windows")
        for key in ("PYTHONPATH", "PYTHONHOME", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "PSExecutionPolicyPreference"):
            env.pop(key, None)
        raw_path = run / "extracted-selfcheck.raw.stdout"
        with raw_path.open("wb") as stdout, (run / "selfcheck.stderr.log").open("wb") as stderr:
            completed = subprocess.run([str(app / "FiberDiameterMeasurement.exe"), "--self-check", "--json"],
                                       cwd=app, env=env, stdout=stdout, stderr=stderr, timeout=900)
        report["selfcheck_exit_code"] = completed.returncode
        raw = raw_path.read_bytes()
        try:
            decoded, encoding = raw.decode("utf-8-sig", errors="strict"), "utf-8"
        except UnicodeDecodeError:
            decoded, encoding = raw.decode("cp936", errors="strict"), "cp936"
        check = json.loads(decoded)
        write_json(run / "extracted-selfcheck.json", check)
        write_json(out / "extracted-selfcheck.json", check)
        require(completed.returncode == 0 and check.get("ok") is True and check.get("errors") == []
                and check.get("build_id") == manifest["build_id"] and check.get("version") == EXPECTED_VERSION,
                "Actual extracted EXE self-check failed")
        alignment, optical = require_worker_selfcheck(check)
        require(verify_sources(root, sources) == report["production_sources_verified"], "Production source verification changed")
        report.update(passed=True, selfcheck_encoding=encoding, selfcheck_environment=str(config),
                      alignment_worker_verified=True, alignment_worker_runtime=alignment["worker_runtime"],
                      optical_worker_verified=True, optical_worker_pid=optical["worker_pid"])
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run / "installer-identity.json", report)
    write_json(out / "installer-identity.json", report)
    print(json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
