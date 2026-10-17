#!/usr/bin/env python3
"""Exercise missing-capture rebuild only inside a disposable Windows snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import sysconfig
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from native_build import check_native_build_dependencies, ensure_native_kernels


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32" or not ROOT.name.startswith("fdm-cu-live-") or ROOT.parent != Path(r"C:\tmp"):
        parser.error("This destructive fixture is restricted to C:\\tmp\\fdm-cu-live-* snapshots")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    paths = [ROOT / "src/fdm/_native" / (name + suffix) for name in ("_kernels", "_microview")]
    before = {path.name: digest(path) for path in paths}
    errors = []
    with tempfile.TemporaryDirectory(prefix="rebuild-fixture-", dir=ROOT / "build") as directory:
        backups = Path(directory)
        for path in paths:
            if path.exists():
                shutil.copy2(path, backups / path.name)
        paths[1].unlink(missing_ok=True)
        try:
            missing_errors = check_native_build_dependencies(root=ROOT)
            errors = ensure_native_kernels(ROOT)
            errors.extend(check_native_build_dependencies(root=ROOT))
            if not missing_errors:
                errors.append("The missing capture extension was not rejected")
        except BaseException:
            for path in paths:
                if (backups / path.name).exists():
                    shutil.copy2(backups / path.name, path)
            raise
        if errors:
            for path in paths:
                if (backups / path.name).exists():
                    shutil.copy2(backups / path.name, path)
        report = {"ok": not errors, "missing_capture_rebuilt": not errors and paths[1].is_file(),
                  "before_sha256": before, "after_sha256": {path.name: digest(path) for path in paths},
                  "missing_probe_errors": missing_errors, "errors": errors}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"ok": report["ok"], "errors": errors}, allow_nan=False))
        return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
