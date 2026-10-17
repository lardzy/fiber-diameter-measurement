"""Repeat real background writes, newer edits, receipt reuse and rollback.

Run separately from timing benchmarks. Uses temporary projects only; each round
executes three writes through the production coordinator and frozen self-check.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PySide6 import __version__ as qt_binding_version
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from fdm.ui.project_save_self_check import run_project_save_self_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    app = QApplication.instance() or QApplication([])
    report = {"python": sys.version, "platform": platform.platform(),
        "pyside6": qt_binding_version, "rounds": [], "ok": False,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("src/fdm/ui/project_save_coordinator.py",
                         "src/fdm/ui/project_save_self_check.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.rounds):
        started = time.perf_counter()
        result = run_project_save_self_check()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect()
        app.processEvents()
        report["rounds"].append({"round": index + 1,
            "elapsed_ms": (time.perf_counter() - started) * 1000, **result})
        report["ok"] = all(row["ok"] for row in report["rounds"])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not result["ok"]:
            raise RuntimeError(f"save round {index + 1} failed: {result}")
        if (index + 1) % 20 == 0:
            print(f"{index + 1}/{args.rounds} rounds passed", flush=True)
    print(f"PASS: {args.rounds} rounds, {args.rounds * 3} background writes", flush=True)


if __name__ == "__main__":
    main()
