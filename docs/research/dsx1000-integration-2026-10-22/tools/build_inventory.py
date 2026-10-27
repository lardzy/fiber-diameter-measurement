"""Read the supplied Olympus copy; write a reproducible investigation inventory.

Never loads a DLL, starts vendor code, opens a device, or changes the source tree.
PE fields are declarations; they do not prove runtime architecture or behavior.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path


def pe_identity(path: Path) -> dict:
    if path.suffix.lower() not in {".dll", ".exe", ".sys"}:
        return {}
    with path.open("rb") as f:
        head = f.read(64)
        if head[:2] != b"MZ" or len(head) < 64:
            return {}
        off = struct.unpack_from("<I", head, 60)[0]
        f.seek(off)
        head = f.read(264)
        if head[:4] != b"PE\0\0" or len(head) < 26:
            return {"format": "MZ-without-PE"}
        machine, sections = struct.unpack_from("<HH", head, 4)
        opt_size = struct.unpack_from("<H", head, 20)[0]
        magic = struct.unpack_from("<H", head, 24)[0]
        directory = 24 + (112 if magic == 0x20B else 96)
        cli_rva, cli_size = (0, 0)
        if len(head) >= directory + 120:
            cli_rva, cli_size = struct.unpack_from("<II", head, directory + 112)
        result = {"format": "PE", "machine": hex(machine),
                  "pe_architecture": {0x14C: "x86", 0x8664: "x64", 0xAA64: "arm64"}.get(machine, "unknown"),
                  "cli_declared": bool(cli_rva and cli_size)}
        if cli_rva and cli_size:
            f.seek(off + 24 + opt_size)
            table = f.read(sections * 40)
            for i in range(sections):
                if len(table) < (i + 1) * 40:
                    break
                virtual_size, va, raw_size, raw_offset = struct.unpack_from("<IIII", table, i * 40 + 8)
                if va <= cli_rva < va + max(virtual_size, raw_size):
                    f.seek(raw_offset + cli_rva - va + 16)
                    flags = f.read(4)
                    if len(flags) == 4:
                        result["cli_flags"] = hex(struct.unpack("<I", flags)[0])
                    break
        return result


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise SystemExit("Output must be outside the supplied software tree")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.name.startswith("._"):
            continue
        relative = path.relative_to(root).as_posix()
        primary = relative.startswith("OLYMPUS/DSX-BSW/")
        installer = relative.startswith("backup/DSX10-BSW_") and path.suffix.lower() == ".msi" and "DSX" in path.name.upper()
        record = {"path": relative, "size": path.stat().st_size,
                  "group": "installed-dsx-copy" if primary else "backup" if relative.startswith("backup/") else "other-olympus",
                  "sha256": digest(path) if primary or installer else None,
                  "hash_scope": "whole-file" if primary or installer else "not-hashed"}
        if primary:
            record.update(pe_identity(path))
        records.append(record)
    summary = {"source_root": str(root), "source_executed": False,
               "scope": "All regular files inventoried; AppleDouble sidecars and symlinks excluded. Full SHA256 for every installed DSX copy file and named DSX backup MSI only.",
               "files": len(records), "groups": dict(Counter(x["group"] for x in records)),
               "full_hash_files": sum(x["sha256"] is not None for x in records),
               "pe_architectures": dict(Counter(x["pe_architecture"] for x in records if "pe_architecture" in x)),
               "identity_caveat": "A sequential filesystem observation, not an atomic device snapshot. PE x86 plus IL-only CLI may be AnyCPU; consult REA managed classification."}
    (output / "inventory.json").write_text(json.dumps({"summary": summary, "files": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "file-identities.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["path", "size", "group", "sha256", "hash_scope", "format", "machine", "pe_architecture", "cli_declared", "cli_flags"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
