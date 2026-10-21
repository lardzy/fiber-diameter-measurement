"""Independent synthetic optical validation; never substitutes for camera metrology.

Parameters are not fitted here. Raster phases, angles and noise differ from the
unit-test development set. Preserve the output directory for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def strip(width, angle, structure, seed, shape=(288, 416)):
    factor = 4
    h, w = shape
    yy, xx = np.mgrid[0:h * factor, 0:w * factor].astype(np.float32)
    xx = (xx + .5) / factor - .5
    yy = (yy + .5) / factor - .5
    theta = math.radians(angle)
    along = (xx - w / 2 - .27) * math.cos(theta) + (yy - h / 2 + .31) * math.sin(theta)
    across = -(xx - w / 2 - .27) * math.sin(theta) + (yy - h / 2 + .31) * math.cos(theta)
    field = 205 + 24 * xx / w + 8 * yy / h
    body = (np.abs(across) < width / 2) & (np.abs(along) < 105)
    signal = field.copy()
    signal[body] -= 145 if structure == "dark" else 15 if structure == "pale" else 125
    if structure == "rim":
        core = body & (np.abs(across) < width / 2 - min(3, width * .2))
        signal[core] += 113
    signal = cv2.resize(signal, (w, h), interpolation=cv2.INTER_AREA)
    signal = cv2.GaussianBlur(signal, (5, 5), .8)
    signal += np.random.default_rng(seed).normal(0, .65, shape)
    return np.clip(np.rint(signal), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angles", type=int, nargs="+", default=[12, 39, 77, 118])
    parser.add_argument("--seed-base", type=int, default=4871)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Preserve previous validation; choose another output directory")
    from fdm.services.cu_automatic_measurement import analyze_frame
    from fdm.services.optical_automatic import PARAMETERS
    from dataclasses import asdict

    cv2.setNumThreads(1)
    cases, errors, durations = [], [], []
    for structure in ("dark", "pale", "rim"):
        for width in (8, 12, 20, 32, 48, 80):
            for angle in args.angles:
                source = strip(width, angle, structure, args.seed_base + width * 131 + angle)
                started = time.perf_counter()
                result = analyze_frame(source, "optical_adaptive")
                elapsed = (time.perf_counter() - started) * 1000
                candidates = result["candidates"]
                actual = [c["width_px"] for c in candidates]
                error = abs(actual[0] - width) if len(actual) == 1 else None
                if error is not None:
                    errors.append(error)
                durations.append(elapsed)
                cases.append(dict(structure=structure, width=width, angle=angle,
                    source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                    status=result["status"], count=len(actual), actual=actual,
                    error_px=error, elapsed_ms=elapsed))
                if width == 32 and angle == 39:
                    rendered = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
                    for candidate in candidates:
                        a, b = (tuple(np.rint(candidate[key]).astype(int)) for key in ("start", "end"))
                        cv2.line(rendered, a, b, (25, 190, 240), 1, cv2.LINE_AA)
                        cv2.putText(rendered, f"{candidate['width_px']:.2f} px", (b[0]+10, b[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, .4, (30, 70, 190), 1, cv2.LINE_AA)
                    cv2.imwrite(str(output / f"{structure}.png"), rendered)
    report = dict(schema_version=1, scope="Independent synthetic validation; no camera or physical accuracy acceptance",
        runtime=platform.platform(), opencv=cv2.__version__, version="0.4.10",
        parameters=asdict(PARAMETERS), angles=args.angles, seed_base=args.seed_base, cases=cases,
        summary=dict(cases=len(cases), correct_single_candidates=len(errors),
            width_error_p50_px=float(np.percentile(errors, 50)) if errors else None,
            width_error_p95_px=float(np.percentile(errors, 95)) if errors else None,
            elapsed_p50_ms=float(np.percentile(durations, 50)),
            elapsed_p95_ms=float(np.percentile(durations, 95))),
        source_sha256=hashlib.sha256((ROOT / "src/fdm/services/optical_automatic.py").read_bytes()).hexdigest())
    report["passed"] = len(errors) == len(cases) and bool(errors) and float(np.percentile(errors, 95)) <= 1
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], **report["summary"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
