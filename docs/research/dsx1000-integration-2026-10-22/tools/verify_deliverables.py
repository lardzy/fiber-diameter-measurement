"""Verify static research artifacts; never loads vendor code or contacts a device."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[2]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    errors: list[str] = []
    evidence: dict[str, dict] = {}
    archive_results: list[dict] = []
    json_count = 0
    for path in sorted(BASE.rglob("*.json")):
        if "collector" in path.relative_to(BASE).parts:
            continue  # Collector validation intentionally includes damaged session fixtures.
        if path.name != "verification.json":
            json.loads(path.read_text(encoding="utf-8"))
            json_count += 1
    for path in sorted(BASE.rglob("*.json.gz")):
        compressed = path.read_bytes()
        raw = gzip.decompress(compressed)
        payload = json.loads(raw)
        # REA analysis snapshots wrap the retained evidence bundle. Count its
        # records, not the outer provider/target metadata, as evidence.
        if isinstance(payload, dict) and "evidence_bundle" in payload:
            archive_kind = "analysis_snapshot"
            records = payload["evidence_bundle"]["records"]
        elif isinstance(payload, dict) and "records" in payload:
            archive_kind = "evidence_bundle"
            records = payload["records"]
        else:
            archive_kind = "evidence_record"
            records = [payload]
        for record in records:
            if not isinstance(record, dict) or not re.fullmatch(r"ev_[0-9a-f]{64}", record.get("evidence_id", "")):
                errors.append(f"Malformed evidence record in {path.relative_to(BASE)}")
                continue
            identity = record["evidence_id"]
            evidence[identity] = record
        archive_results.append({
            "path": str(path.relative_to(BASE)), "kind": archive_kind, "records": len(records),
            "compressed_bytes": len(compressed), "compressed_sha256": sha(compressed),
            "uncompressed_bytes": len(raw), "uncompressed_sha256": sha(raw),
        })
    archive_map = {entry["path"]: entry for entry in archive_results}

    def check_archive(path: str, compressed_sha: str, raw_sha: str) -> None:
        got = archive_map[path]
        if got["compressed_sha256"] != compressed_sha or got["uncompressed_sha256"] != raw_sha:
            errors.append(f"Archive manifest mismatch: {path}")

    for group in ("identity", "acquisition"):
        manifest = json.loads((BASE / group / "evidence-manifest.json").read_text())
        if group == "identity":
            check_archive(group + "/" + manifest["compressed_path"],
                          manifest["compressed_sha256"], manifest["uncompressed_sha256"])
        else:
            for item in manifest["archives"]:
                check_archive(group + "/" + item["archive"],
                              item["compressed_sha256"], item["uncompressed_sha256"])
    entry = json.loads((BASE / "entrypoints/evidence-bundle-manifest.json").read_text())
    check_archive("entrypoints/" + entry["compressed_path"], entry["gzip_sha256"], entry["raw_sha256"])
    motion = json.loads((BASE / "motion/rea-evidence-index.json").read_text())
    check_archive("motion/" + motion["bundle"], motion["compressed_sha256"], motion["uncompressed_sha256"])
    motion_manifest = json.loads((BASE / "motion/motion-manifest.json").read_text())
    for item in motion_manifest["files"]:
        path = BASE / "motion" / item["path"]
        if sha(path.read_bytes()) != item["sha256"]:
            errors.append(f"Motion selected artifact mismatch: {item['path']}")

    referenced_ids: set[str] = set()
    link_count = 0
    for path in sorted(BASE.rglob("*.md")):
        text = path.read_text()
        referenced_ids.update(re.findall(r"ev_[0-9a-f]{64}", text))
        # Array/math notation inside code spans is not a Markdown link.
        prose = re.sub(r"```[\s\S]*?```|`[^`\n]+`", "", text)
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", prose):
            target = target.strip("<>")
            if "://" in target or target.startswith("#"):
                continue
            target = unquote(target.split("#", 1)[0])
            target_path = (path.parent / target).resolve()
            if target_path.name == "verification.json":
                continue  # This file is the output of the current check.
            link_count += 1
            if not target_path.exists():
                errors.append(f"Broken document link: {path.name} -> {target}")
    for missing in sorted(referenced_ids - evidence.keys()):
        errors.append(f"Missing retained evidence: {missing}")

    source_identities: dict[str, str] = {}
    for record in evidence.values():
        subject = record.get("subject") or {}
        path = subject.get("local_path")
        digest = (subject.get("digest") or {}).get("sha256")
        if path and digest and Path(path).is_file():
            source_identities[path] = digest
    source_results = []
    for path, expected in sorted(source_identities.items()):
        actual = sha(Path(path).read_bytes())
        source_results.append({"path": path, "sha256": actual, "matches_evidence": actual == expected})
        if actual != expected:
            errors.append(f"Vendor source changed: {path}")
    fdm = json.loads((BASE / "identity/fdm-source.json").read_text())
    for item in fdm["files"]:
        if sha((REPO / item["path"]).read_bytes()) != item["sha256"]:
            errors.append(f"FDM source changed: {item['path']}")
    product_diff = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", "src", "native", "tests", "pyproject.toml"],
        cwd=REPO, text=True,
    ).splitlines()
    if product_diff:
        errors.append("Product paths have changed relative to HEAD")
    inventory = json.loads((BASE / "identity/inventory.json").read_text())
    whole_file = [item for item in inventory["files"] if item.get("hash_scope") == "whole-file"]
    if len(whole_file) != 530 or len(inventory["files"]) != 3037:
        errors.append("Unexpected identity inventory scope")

    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Static artifacts only; not runtime, hardware, deployment or performance acceptance.",
        "status": "passed" if not errors else "failed", "errors": errors,
        "json_files_parsed": json_count, "relative_document_links_checked": link_count,
        "unique_retained_evidence_records": len(evidence),
        "document_referenced_evidence_ids": len(referenced_ids),
        "archives": archive_results, "vendor_source_identities_rechecked": source_results,
        "fdm_source_hashes_rechecked": len(fdm["files"]), "product_diff_paths": product_diff,
        "inventory": {"files": len(inventory["files"]), "whole_file_hashes": len(whole_file),
                      "note": "Inventory hashes were built during analysis; this verification rehashes evidence subjects, not all backup files."},
        "vendor_execution": False, "device_commands_sent": False,
        "hardware_tests_run": False, "performance_measurements_taken": False,
    }
    (BASE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "errors": errors,
                      "unique_evidence_records": len(evidence), "vendor_files_rechecked": len(source_results),
                      "archives": len(archive_results), "links_checked": link_count}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
