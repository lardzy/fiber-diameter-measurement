"""Sample benchmark process memory outside its Python interpreter/GIL.

The sampler is excluded from child totals. Sampling still costs host CPU; this
is not a replacement for separate no-load runs or real-device measurements.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def sample_process(pid, *, exclude_pid, parent_only=False):
    import psutil
    process = psutil.Process(pid)
    with process.oneshot():
        memory = process.memory_info()
        result = {"process_cpu_seconds": sum(process.cpu_times()[:2]),
            "rss_bytes": memory.rss, "native_threads": process.num_threads(),
            "resource_provider": "psutil in isolated sampler process"}
        if hasattr(memory, "peak_wset"):
            result["process_lifetime_peak_rss_bytes"] = memory.peak_wset
    children = []
    for child in (() if parent_only else process.children(recursive=True)):
        if child.pid == exclude_pid:
            continue
        try:
            with child.oneshot():
                children.append({"pid": child.pid, "name": child.name(),
                    "rss_bytes": child.memory_info().rss,
                    "cpu_seconds": sum(child.cpu_times()[:2]),
                    "native_threads": child.num_threads()})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    result.update(children=children, children_rss_bytes=sum(c["rss_bytes"] for c in children),
        children_cpu_seconds=sum(c["cpu_seconds"] for c in children))
    result["total_rss_bytes"] = result["rss_bytes"] + result["children_rss_bytes"]
    return result


def collect(pid, directory, *, parent_only=False):
    sample_count = 0
    maxima, child_peaks = {}, {}
    started = time.perf_counter()

    def record():
        nonlocal sample_count
        sample = sample_process(pid, exclude_pid=os.getpid(), parent_only=parent_only)
        for key in ("rss_bytes", "children_rss_bytes", "total_rss_bytes", "native_threads", "process_lifetime_peak_rss_bytes"):
            if key in sample:
                maxima["sampled_max_" + key] = max(maxima.get("sampled_max_" + key, 0), sample[key])
        for child in sample["children"]:
            peak = child_peaks.setdefault(child["pid"], {"pid": child["pid"], "name": child["name"],
                "sampled_peak_rss_bytes": 0, "maximum_observed_cpu_seconds": 0, "sampled_peak_native_threads": 0})
            for dest, source in (("sampled_peak_rss_bytes", "rss_bytes"),
                                 ("maximum_observed_cpu_seconds", "cpu_seconds"),
                                 ("sampled_peak_native_threads", "native_threads")):
                peak[dest] = max(peak[dest], child[source])
        sample_count += 1
        return sample

    before = record()
    (directory / "ready").touch()
    while not (directory / "stop").exists():
        time.sleep(.1)
        record()
    after = record()
    report = {"wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": after["process_cpu_seconds"] - before["process_cpu_seconds"],
        "before": before, "after": after, "sampling_interval_ms": 100,
        "sample_count": sample_count, "children_sampled_peaks": list(child_peaks.values()),
        "sampler_pid": os.getpid(), "sampling_mode": "isolated_process",
        "parent_only": parent_only,
        "peak_scope": ("Sampled target-process RSS only; child processes excluded. Sub-100 ms peaks may be missed."
                       if parent_only else "Sampled RSS of target and its workers; excludes sampler. Sub-100 ms peaks may be missed."), **maxima}
    (directory / "result.json").write_text(json.dumps(report), encoding="utf-8")


class ProcessResourceSampler:
    def __init__(self, *, parent_only=False):
        self.parent_only = parent_only

    def __enter__(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="fdm-gui-resources-")
        self._directory = Path(self._temporary.name)
        self._process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            "--pid", str(os.getpid()), "--directory", str(self._directory),
            *(["--parent-only"] if self.parent_only else [])],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        deadline = time.perf_counter() + 15
        while not (self._directory / "ready").exists():
            if self._process.poll() is not None or time.perf_counter() > deadline:
                self._process.kill()
                _, error = self._process.communicate()
                self._temporary.cleanup()
                raise RuntimeError("Resource sampler failed to start: " + error.decode(errors="replace"))
            time.sleep(.01)
        return self

    def __exit__(self, *_):
        try:
            (self._directory / "stop").touch()
            try:
                _, error = self._process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.communicate()
                raise RuntimeError("Resource sampler did not stop")
            if self._process.returncode:
                raise RuntimeError("Resource sampler failed: " + error.decode(errors="replace"))
            self.result = json.loads((self._directory / "result.json").read_text(encoding="utf-8"))
        finally:
            self._temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--parent-only", action="store_true", help="Skip process-tree enumeration; report only target-process resources")
    arguments = parser.parse_args()
    collect(arguments.pid, arguments.directory, parent_only=arguments.parent_only)
