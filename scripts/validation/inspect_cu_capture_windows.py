#!/usr/bin/env python3
"""Read existing CU window identity and capture without starting/driving CU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from fdm.platform.windows_window_locator import enumerate_windows
from fdm.services.cu5_preview_locator import Cu5PreviewLocator, matches_cu_family_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = enumerate_windows()
    report = {"schema_version": 1,
        "top_level_windows": [asdict(row) for row in snapshot.roots() if row.visible],
        "cu_windows": [asdict(row) for row in snapshot.records
                       if matches_cu_family_identity(row.process_path, row.title)],
        "captured": False, "frames": [],
        "limits": ["No window is moved, restored, started or sent a command.",
                   "Rendered pixels alone cannot certify a live physical camera feed."]}
    capture = None
    try:
        match = Cu5PreviewLocator().locate(snapshot)
        report["match"] = asdict(match.record)
        from fdm._native import _windows_capture
        capture = _windows_capture.WindowCapture(match.record.root_hwnd, match.hwnd)
        report["identity"] = dict(capture.identity())
        deadline = time.perf_counter() + 3
        while time.perf_counter() < deadline and len(report["frames"]) < 10:
            packet = capture.poll(100)
            if packet is not None:
                pixels = packet["pixels"]
                report["frames"].append({"shape": list(pixels.shape),
                    "sequence": packet["sequence"], "generation": packet["generation"],
                    "timestamp_100ns": packet["source_timestamp_100ns"],
                    "sha256": hashlib.sha256(pixels).hexdigest(),
                    "minimum": int(pixels[..., :3].min()), "maximum": int(pixels[..., :3].max())})
        report["captured"] = bool(report["frames"])
    except Exception as exc:
        report["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if capture is not None:
            capture.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in
          ("captured", "unavailable_reason")}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
