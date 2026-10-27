"""Index retained REA evidence and hash deliverables. No vendor execution."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def files():
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.name != ".DS_Store"
                  and p != ROOT / "SHA256SUMS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify existing manifest without changing files")
    args = parser.parse_args()
    manifest = ROOT / "SHA256SUMS"
    if args.check:
        expected = {}
        for line in manifest.read_text().splitlines():
            digest, path = line.split("  ", 1)
            expected[path] = digest
        actual = {str(p.relative_to(ROOT)): sha(p) for p in files()}
        errors = [p for p in expected.keys() | actual.keys() if expected.get(p) != actual.get(p)]
        print(json.dumps(dict(status="failed" if errors else "passed", files=len(actual), mismatches=sorted(errors))))
        return int(bool(errors))
    records = {}
    archives = []
    for p in sorted(ROOT.rglob("*.json.gz")):
        raw = gzip.decompress(p.read_bytes())
        payload = json.loads(raw)
        bundle = payload.get("evidence_bundle", payload)
        entries = bundle.get("records", [bundle])
        rel = str(p.relative_to(ROOT))
        for entry in entries:
            identity = entry["evidence_id"]
            row = records.setdefault(identity, dict(evidence_id=identity, subject=entry.get("subject"),
                                                     operation=entry.get("operation"), archives=[]))
            row["archives"].append(rel)
        archives.append(dict(path=rel, records=len(entries), sha256=sha(p),
                             raw_sha256=hashlib.sha256(raw).hexdigest()))
    index = dict(schema="dsx-algorithms-evidence-index-v1", unique_records=len(records), archives=archives,
                 note="Some managed records are intentionally duplicated across bundles; IDs are deduplicated here. Earlier acquisition/motion evidence remains in parent research folders.",
                 records=[records[k] for k in sorted(records)])
    (ROOT / "algorithm-evidence-index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    paths = files()
    manifest.write_text("".join(f"{sha(p)}  {p.relative_to(ROOT)}\n" for p in paths))
    print(json.dumps(dict(status="written", files=len(paths), unique_evidence_records=len(records), archives=len(archives))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
