#!/usr/bin/env python3
"""Paired real-file evidence encoding; decode/hash checks run outside timers.

Supply a preserved reference png_stream.py (for example from commit 5049422).
Inputs and alternating order are fixed. Both writers include flush, fsync and
atomic replacement. No existing evidence is opened or changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.benchmark_cu_live_measurement import distribution, environment, ResourceSampler
from scripts.validation.benchmark_cu_png import inputs
from fdm.live_experiment import png_stream
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sizes', nargs='+', type=int, default=[2048, 4096])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--samples', type=int, default=30)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('cu_png_preserved_reference', args.reference)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    implementations = [('r4_reference', reference), ('production', png_stream)]
    report = {'environment': environment(), 'warmup': args.warmup, 'samples': args.samples,
        'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': [],
        'limits': ['Synthetic pixels; ratios are not microscopy storage estimates.',
                   'Timing includes encoding, hash, file writes, fsync and atomic replace; excludes decode verification.',
                   'Sampled RSS includes outside-timer decoding and retained fixture images, not codec-only RSS.']}
    with tempfile.TemporaryDirectory(prefix='fdm-png-pair-') as temporary:
        root = Path(temporary)
        for width in args.sizes:
            for scenario, pixels in inputs(width):
                pixels.setflags(write=False)
                trials, outputs, exact = [], {}, {name: 0 for name, _ in implementations}
                with ResourceSampler() as resources:
                    for step in range(args.warmup + args.samples):
                        order = implementations if step % 2 == 0 else implementations[::-1]
                        row = {'warmup': step < args.warmup, 'order': [name for name, _ in order]}
                        for name, module in order:
                            pending, final = root / 'image.pending', root / 'image.png'
                            began = time.perf_counter_ns()
                            with pending.open('wb', buffering=0) as stream:
                                result = module.write_png_stream(pixels, stream)
                                stream.flush()
                                os.fsync(stream.fileno())
                            os.replace(pending, final)
                            row[name + '_ms'] = (time.perf_counter_ns() - began) / 1e6
                            payload = final.read_bytes()
                            assert hashlib.sha256(payload).hexdigest() == result.sha256
                            decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
                            assert np.array_equal(decoded, pixels), (scenario, name)
                            exact[name] += 1
                            outputs[name] = {'bytes': result.size_bytes, 'sha256': result.sha256,
                                'workspace_bytes': result.workspace_bytes, 'encoder': module.ENCODER_REVISION}
                            payload = decoded = None
                        trials.append(row)
                case = {'width': width, 'scenario': scenario, 'shape': list(pixels.shape),
                    'input_sha256': hashlib.sha256(memoryview(pixels).cast('B')).hexdigest(),
                    'production_compression_level': png_stream._compression_level(pixels),
                    'outputs': outputs, 'exact_roundtrips': exact, 'trials': trials,
                    'metrics': {name: distribution([r[name + '_ms'] for r in trials if not r['warmup']])
                                for name, _ in implementations}, 'resources': resources.result}
                report['cases'].append(case)
                print(width, scenario, {name: case['metrics'][name]['p95_ms'] for name, _ in implementations}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
