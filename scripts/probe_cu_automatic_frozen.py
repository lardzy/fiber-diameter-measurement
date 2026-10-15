"""Standalone real spawn/shared-memory probe for the frozen CU automatic worker."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time


def main() -> int:
    import numpy as np
    from fdm._native import build_info
    from fdm.services.cu_automatic_worker import CuAutomaticWorker

    image = np.full((240, 320, 3), 220, np.uint8)
    image[114:126, 40:280] = 40
    rows = []
    with CuAutomaticWorker() as worker:
        for algorithm, expected in [("traditional", 11.0), ("multiscale", 10.997208727965706)]:
            identifier = worker.submit(image, algorithm)
            start = time.monotonic()
            response = None
            while time.monotonic() - start < 35:
                messages = worker.poll()
                if messages:
                    response = messages[0]
                    break
                time.sleep(0.005)
            if response is None:
                raise RuntimeError("frozen automatic worker did not respond")
            result = response["result"]
            ok = (response["task_id"] == identifier and response["error"] is None
                  and result["status"] == "ok" and len(result["candidates"]) == 1
                  and abs(result["candidates"][0]["width_px"] - expected) < 1e-8)
            rows.append({"algorithm": algorithm, "ok": ok, "pid": worker._process.pid,
                         "elapsed_ms": (time.monotonic() - start) * 1000, "result": result})
    output = {"ok": all(row["ok"] for row in rows) and rows[0]["pid"] == rows[1]["pid"],
              "frozen": bool(getattr(sys, "frozen", False)), "native_build_info": build_info(), "cases": rows}
    data = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(data, encoding="utf-8")
    else:
        print(data)
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    raise SystemExit(main())
