"""Reproducible offline pair benchmark; never writes the source slide.

PYTHONPATH=src python scripts/benchmark_slide_stitching.py --output /tmp/stitch.json
Add --slide /path/to/local.fdmslide for a read-only applicability report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from fdm.services.slide_registration import estimate_pair, register_slide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slide", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048,4096,8192])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    rows = []
    for size in args.sizes:
        rng = np.random.default_rng(71+size)
        scene = cv2.GaussianBlur(rng.integers(0,256,(size+32,2*size),np.uint8),(0,0),1)
        nominal = round(.8*size)
        elapsed, errors, rejected = [], [], []
        for _ in range(args.repeats):
            actual = nominal + rng.uniform(-10,10)
            vertical = rng.uniform(-4,4)
            xx, yy = np.meshgrid(np.arange(size,dtype=np.float32)+16+actual,np.arange(size,dtype=np.float32)+16+vertical)
            second = cv2.remap(scene,xx,yy,cv2.INTER_LINEAR)
            del xx, yy
            start = perf_counter()
            result = estimate_pair(scene[16:size+16,16:size+16],second,dx=nominal,dy=0,axis="x")
            elapsed.append((perf_counter()-start)*1000)
            if result.accepted:
                errors.append(float(np.hypot(result.dx-actual,result.dy-vertical)))
            else:
                rejected.append(result.reason)
        rows.append({"size":size,"runs":args.repeats,"accepted":len(errors),
                     "ms_p50":float(np.percentile(elapsed,50)),"ms_p95":float(np.percentile(elapsed,95)),
                     "error_px_p95":float(np.percentile(errors,95)) if errors else None,"rejected":rejected})
        del scene, second
    output = {"kind":"synthetic_cpu_registration_not_input_latency","pairs":rows}
    if args.slide:
        before = args.slide.stat()
        start = perf_counter()
        layout = register_slide(args.slide)
        after = args.slide.stat()
        output["existing_slide"] = {"tile_count":len(layout.tiles),"seams":len(layout.pairs),
            "accepted":layout.accepted_count,"reasons":dict(Counter(p.reason for p in layout.pairs)),
            "seconds":perf_counter()-start,"source_digest":layout.source_digest,
            "source_stat_unchanged":(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
