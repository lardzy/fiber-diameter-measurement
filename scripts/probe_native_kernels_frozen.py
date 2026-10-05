"""Freeze this small probe to verify native loading without a desktop console.

The first argument is the JSON report path. This validates the kernel bundle,
not the complete Windows application or its installer.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from fdm._native import selected_backend
from fdm.services.native_kernels_self_check import run_native_kernels_self_check


def main() -> int:
    output = Path(sys.argv[1])
    os.environ["FDM_NATIVE_BACKEND"] = "python"
    sys.stdin = sys.stdout = sys.stderr = None
    try:
        report = run_native_kernels_self_check()
        report["frozen"] = bool(getattr(sys, "frozen", False))
        report["forced_python_still_selects_native"] = selected_backend() is not None
        report["stdio_none"] = all(stream is None for stream in (sys.stdin, sys.stdout, sys.stderr))
        report["ok"] = all(report.get(key) is True for key in (
            "ok", "frozen", "forced_python_still_selects_native", "stdio_none"
        ))
    except Exception as exc:
        report = {"ok": False, "error": str(exc), "type": type(exc).__name__}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
