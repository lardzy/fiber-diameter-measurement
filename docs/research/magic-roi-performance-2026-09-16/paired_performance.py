"""Same-machine, isolated-process comparison with the pre-change service.

Run with uv run --no-sync python <this file>. Each version gets its own process
and the exact same deterministic pixels/prompts, shipped model and CPU settings.
Only service latency is timed; QImage creation, RSS sampling and mask export are
outside the measured interval. Desktop queue/delivery is covered by runtime logs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PySide6.QtGui import QImage

from fdm.geometry import Point

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def qimage(pixels):
    return QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                  QImage.Format.Format_RGB888).convertToFormat(QImage.Format.Format_RGB32)


def rss_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        if ctypes.windll.psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize
        return None
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) * 1024


def load_service(version, baseline):
    if version == "baseline":
        name = "fdm.services._roi_benchmark_baseline"
        spec = importlib.util.spec_from_file_location(name, baseline)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        from fdm.services import prompt_segmentation as module
    # The old service emits a separate preprocess record on misses. File I/O is
    # disabled for both sides so this is an inference/cache comparison.
    module.append_runtime_log = lambda *args, **kwargs: None

    class Probe(module.PromptSegmentationService):
        def __init__(self):
            folder = ROOT / "runtime/segment-anything/edge_sam_3x"
            super().__init__(model_variant="edge_sam_3x", encoder_path=folder / "edge_sam_3x_encoder.onnx",
                             decoder_path=folder / "edge_sam_3x_decoder.onnx")
            self.local_masks = True
            self.encodes = self.decodes = 0

        def _run_encoder(self, *args, **kwargs):
            self.encodes += 1
            return super()._run_encoder(*args, **kwargs)

        def _predict_mask_candidates_from_embedding(self, *args, **kwargs):
            self.decodes += 1
            return super()._predict_mask_candidates_from_embedding(*args, **kwargs)

    return module, Probe()


def request(service, version, img, pos, neg=(), workspace=None, source="benchmark", **kwargs):
    before_e, before_d = service.encodes, service.decodes
    extra = {"roi_workspace_box": workspace} if version == "current" else {}
    start = time.perf_counter()
    cpu_start = time.process_time()
    result = service.predict_polygon(image=img, cache_key=source, positive_points=list(pos),
        negative_points=list(neg), tool_mode="magic_segment", roi_enabled=True, **extra, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    row = {"ms": elapsed, "process_cpu_ms": (time.process_time() - cpu_start) * 1000,
           "encodes": service.encodes - before_e, "decodes": service.decodes - before_d,
           "crop": result.metadata.get("segmentation_crop_box"), "area": result.area_px,
           "cache_bytes": sum(e.image_embeddings.nbytes for e in service._embedding_cache.values())}
    return result, row


def workspace_for(result):
    return result.metadata.get("segmentation_crop_box") if result.metadata.get("segmentation_workspace_reusable", True) else None


def grid_picture(width=2048, height=1536):
    pixels = np.full((height, width, 3), 200, np.uint8)
    centers = [Point(200 + 300 * x, 180 + 280 * y) for y in range(5) for x in range(6)]
    for p in centers:
        cv2.circle(pixels, (int(p.x), int(p.y)), 45, (60, 70, 90), -1)
    return qimage(pixels), centers


def run_child(args):
    module, service = load_service(args.version, args.baseline)
    service._ensure_sessions()
    tiny = np.full((512, 512, 3), 200, np.uint8)
    cv2.circle(tiny, (256, 256), 45, (60, 70, 90), -1)
    for _ in range(2):
        request(service, args.version, qimage(tiny), [Point(256, 256)], source="warmup")
        service.clear_cache()
    output = {"version": args.version, "platform": platform.platform(), "cpu": platform.processor(),
              "started_at_unix": time.time(), "service_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              "ort": ort.__version__, "providers": service._encoder_session.get_providers(),
              "model": "edge_sam_3x", "samples_per_scenario": args.samples, "scenarios": {}}
    if platform.system() == "Darwin":
        output["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    if args.replay_only:
        output = json.loads(args.output.read_text())
    cases = [] if args.replay_only else ["new_roi", "same_roi", "continuous_prompts", "expanded_repeat", "edge_target", "large_image"]
    for name in cases:
        service.clear_cache()
        img, centers = grid_picture(6000, 4000) if name == "large_image" else grid_picture()
        workspace = None
        pos, neg = [centers[0]], []
        if name == "expanded_repeat":
            pixels = np.full((1536, 2048, 3), 205, np.uint8)
            cv2.rectangle(pixels, (0, 730), (2047, 806), (55, 65, 85), -1)
            img, pos = qimage(pixels), [Point(1024, 768)]
        elif name == "edge_target":
            pixels = np.full((1536, 2048, 3), 200, np.uint8)
            cv2.circle(pixels, (30, 512), 45, (60, 70, 90), -1)
            img, pos = qimage(pixels), [Point(30, 512)]
        initial = None
        if name not in ("new_roi", "large_image"):
            result, initial = request(service, args.version, img, pos, source=name)
            workspace = workspace_for(result)
        rows = []
        for i in range(args.samples):
            if name in ("new_roi", "large_image"):
                # A fresh key for cycles beyond the 30 spatially distinct ROIs.
                pos = [centers[i % len(centers)]]
                source = f"{name}-{i // len(centers)}"
            else:
                source = name
            if name == "continuous_prompts":
                pos.append(Point(centers[0].x - 15 + i % 30, centers[0].y))
                neg = [Point(centers[0].x + 60, centers[0].y + i % 3)]
            result, row = request(service, args.version, img, pos, neg, workspace, source)
            if name not in ("new_roi", "large_image"):
                workspace = workspace_for(result)
            row["rss_bytes"] = rss_bytes()
            rows.append(row)
        values = [r["ms"] for r in rows]
        record = {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
                  "encoder_calls": sum(r["encodes"] for r in rows),
                  "peak_cache_bytes": max(r["cache_bytes"] for r in rows),
                  "peak_sampled_rss_bytes": max(r["rss_bytes"] or 0 for r in rows),
                  "initial": initial, "samples": rows}
        output["scenarios"][name] = record
        print(json.dumps({"version": args.version, "scenario": name,
                          **{k: v for k, v in record.items() if k not in ("samples", "initial")}}), flush=True)
        del img, result
        gc.collect()

    # A request cancelled before starting must avoid both encoding and decoding.
    cancelled = []
    cancel_image = qimage(tiny)
    for i in range(args.samples):
        before_e, before_d = service.encodes, service.decodes
        started = time.perf_counter()
        try:
            service.predict_polygon(image=cancel_image, cache_key="cancel", positive_points=[Point(256, 256)],
                negative_points=[], tool_mode="magic_segment", roi_enabled=True, cancel_check=lambda: True)
            raise AssertionError("cancelled request unexpectedly completed")
        except RuntimeError as exc:
            assert "取消" in str(exc)
        cancelled.append({"ms": (time.perf_counter() - started) * 1000,
                          "encodes": service.encodes - before_e, "decodes": service.decodes - before_d})
    output["cancelled_before_start"] = cancelled

    # Fixed-crop tests compare raw model masks for exactly identical encoder input
    # and prompt coordinates. No scheduling differences are involved here.
    fixed = {}
    holes = tiny.copy()
    cv2.circle(holes, (256, 256), 16, (200, 200, 200), -1)
    fixtures = [("circle", tiny, [Point(230, 256)], []),
                ("hole", holes, [Point(230, 256)], [Point(256, 256)]),
                ("border", tiny[200:312, 235:340], [Point(10, 56)], [])]
    photo_path = ROOT / ".tmp/fiberseg/image/58.jpg"
    if photo_path.exists():
        photo = cv2.cvtColor(cv2.imread(str(photo_path)), cv2.COLOR_BGR2RGB)
        fixtures.append(("microscopy_fragment", photo[20:310, 1040:1330], [Point(140, 145)], []))
    for name, pixels, positives, negatives in fixtures:
        entry = service._embedding_for_rgb_array(pixels, cache_key=f"exact-{name}")
        mask = service._predict_mask_from_embedding(entry, positive_points=positives, negative_points=negatives)
        fixed[name] = {"sha256": hashlib.sha256(mask.tobytes()).hexdigest(), "area": int(mask.sum())}
        result = service._predict_polygon_for_rgb_array(pixels, cache_key=f"exact-{name}",
            positive_points=positives, negative_points=negatives, metadata_extra={})
        rings = [[[p.x, p.y] for p in ring] for ring in result.area_rings_px]
        fixed[name]["final_mask_sha256"] = hashlib.sha256(result.mask.tobytes()).hexdigest() if result.mask is not None else None
        fixed[name]["final_area"] = result.area_px
        fixed[name]["rings_sha256"] = hashlib.sha256(json.dumps(rings, sort_keys=True).encode()).hexdigest()
    output["fixed_crop_masks"] = fixed

    # A replay includes changes to the crop context. Export compact masks for the
    # comparison process, and keep its quality metrics separate from speed claims.
    replay_cases = [("circle", qimage(tiny), [Point(246, 256), Point(256, 256), Point(264, 256)], []),
                    ("hole", qimage(holes), [Point(230, 256), Point(231, 248), Point(230, 260)], [Point(256, 256)])]
    if photo_path.exists():
        replay_cases += [("microscopy_fragment", qimage(photo), [Point(1180, 165), Point(1184, 169), Point(1178, 163)], []),
                         ("microscopy_background_gap", qimage(photo), [Point(430, 430), Point(431, 438), Point(429, 446)], []),
                         ("microscopy_fiber", qimage(photo), [Point(803, 850), Point(804, 858), Point(803, 866)], [])]
    output["replays"] = {}
    for name, img, positives, negatives in replay_cases:
        service.clear_cache()
        workspace = None
        replay = []
        for step in range(len(positives)):
            result, row = request(service, args.version, img, positives[:step + 1], negatives,
                                  workspace, source=f"replay-{name}")
            workspace = workspace_for(result)
            mask = result.mask.to_full_mask() if result.mask is not None else np.zeros((img.height(), img.width()), bool)
            target = args.output.parent / f"{args.version}-{name}-{step}.npz"
            np.savez_compressed(target, mask=mask)
            replay.append({**row, "mask_file": target.name, "rings": len(result.area_rings_px)})
        output["replays"][name] = replay
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--version", choices=("baseline", "current"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-ref", default="d9a0dd7")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--replay-only", action="store_true", help="Refresh quality replays while retaining the measured timing samples")
    parser.add_argument("--order", choices=("baseline-first", "current-first"), default="baseline-first")
    parser.add_argument("--output", type=Path, default=HERE / "implementation-runs/paired.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.version:
        run_child(args)
        return
    assert args.samples >= 30, "Acceptance requires at least 30 samples per scenario."
    baseline = args.output.parent / "baseline_service.py"
    baseline.write_bytes(subprocess.check_output(["git", "show", f"{args.baseline_ref}:src/fdm/services/prompt_segmentation.py"], cwd=ROOT))
    reports = {}
    for version in (("baseline", "current") if args.order == "baseline-first" else ("current", "baseline")):
        output = args.output.parent / f"{version}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--version", version, "--baseline", str(baseline),
                   "--samples", str(args.samples), "--output", str(output)]
        if args.replay_only:
            command.append("--replay-only")
        subprocess.run(command, check=True, cwd=ROOT)
        reports[version] = json.loads(output.read_text())
    baseline.unlink()
    comparison = {"baseline_ref": args.baseline_ref, "execution_order": args.order, "versions": reports, "comparison": {}, "quality": {}}
    for name, old in reports["baseline"]["scenarios"].items():
        new = reports["current"]["scenarios"][name]
        comparison["comparison"][name] = {"p50_reduction_percent": 100 * (1 - new["p50_ms"] / old["p50_ms"]),
                                         "p95_change_percent": 100 * (new["p95_ms"] / old["p95_ms"] - 1)}
    comparison["fixed_crop_pixel_equal"] = reports["baseline"]["fixed_crop_masks"] == reports["current"]["fixed_crop_masks"]
    for name, old_steps in reports["baseline"]["replays"].items():
        rows = []
        for old, new in zip(old_steps, reports["current"]["replays"][name]):
            a = np.load(args.output.parent / old["mask_file"])["mask"]
            b = np.load(args.output.parent / new["mask_file"])["mask"]
            union = np.count_nonzero(a | b)
            rows.append({"iou": float(np.count_nonzero(a & b) / union) if union else 1.0,
                         "pixel_equal": bool(np.array_equal(a, b)), "old_area": int(a.sum()), "new_area": int(b.sum()),
                         "old_rings": old["rings"], "new_rings": new["rings"]})
        comparison["quality"][name] = rows
    args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in comparison.items() if k != "versions"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
