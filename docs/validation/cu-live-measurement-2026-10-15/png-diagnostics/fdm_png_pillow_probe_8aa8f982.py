#!/usr/bin/env python3
"""Standalone diagnostic; never imports or changes production EvidenceStore.

Run only during an explicitly allocated measurement window. Defaults are five
warmups and thirty paired samples for 2K/4K/6K and four deterministic images.
Timing includes conversion, encoding, file flush/fsync, SHA256, and replacement.
Source fixture allocation and verification reads are outside timed intervals.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tempfile
import threading
import time
import weakref
import zlib

import cv2
import numpy as np
import PIL
from PIL import Image, features


class Holder:
    """Executor argument contains no immutable, permanently pinned frame tuple."""
    def __init__(self, pixels):
        self.pixels = pixels

    def take(self):
        pixels, self.pixels = self.pixels, None
        return pixels


class InjectedWriteFailure(OSError):
    pass


class Writer:
    """Real file stream with chunked digest and optional late IDAT write fault."""
    def __init__(self, handle, fail_write=False):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes_written = 0
        self.fail_write = fail_write
        self.idat_seen = False

    def write(self, data):
        # PNG writes an IDAT length/type header immediately before its data.
        # Fail the data write, after the encoder has produced its first block.
        if self.fail_write and self.idat_seen:
            raise InjectedWriteFailure("injected late IDAT write failure")
        self.idat_seen |= len(data) == 8 and bytes(data[-4:]) == b"IDAT"
        written = self.handle.write(data)
        if written != len(data):
            raise OSError("short file write")
        self.digest.update(data)
        self.bytes_written += written
        return written

    def flush(self):
        self.handle.flush()


class DeferredErrorWriter(Writer):
    """Candidate: let ZIP reach its normal end after a file-write OSError.

    This deliberately finishes the CPU encoding after an I/O failure, matching
    imencode's existing encode-before-write behavior. It cannot repair internal
    codec errors, MemoryError, process termination, or an underlying hung write.
    No failed output is committed. This is an unvalidated diagnostic candidate.
    """
    def __init__(self, handle, fail_write=False):
        super().__init__(handle, fail_write=fail_write)
        self.error = None
        self.discarded_bytes = 0

    def _remember(self, error):
        if self.error is None:
            # Do not retain the exception traceback's encoder/frame references.
            self.error = error.with_traceback(None)

    def write(self, data):
        if self.error is not None:
            self.discarded_bytes += len(data)
            return len(data)
        try:
            return super().write(data)
        except OSError as error:
            self._remember(error)
            self.discarded_bytes += len(data)
            return len(data)

    def flush(self):
        if self.error is None:
            try:
                super().flush()
            except OSError as error:
                self._remember(error)

    def finish(self):
        if self.error is not None:
            raise self.error


def source(size, scenario):
    h, w = size * 2 // 3, size
    rng = np.random.default_rng(20261015 + size)
    if scenario == "incompressible_uint8":
        pixels = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        pixels.flags.writeable = False
        return pixels
    row = np.full((w, 3), 232, dtype=np.uint8)
    for index, x in enumerate(np.linspace(w // 12, w * 11 // 12, 8).astype(int)):
        row[max(0, x - 5):x + 6] = (24 + index * 3, 39, 61)
    pixels = np.broadcast_to(row, (h, w, 3)).copy()
    if scenario == "one_bit_noise":
        pixels ^= rng.integers(0, 2, pixels.shape, dtype=np.uint8)
    elif scenario == "gradient_noise":
        gradient = np.linspace(32, 220, w, dtype=np.uint8)
        pixels[:] = gradient[None, :, None]
        # Fixed natural-looking low amplitude texture, not high entropy white noise.
        noise = rng.integers(0, 16, pixels.shape, dtype=np.uint8)
        np.add(pixels, noise, out=pixels)
    pixels.flags.writeable = False
    return pixels


def distribution(values):
    return {**{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)},
            "max": float(max(values)), "samples": values}


def png_chunk(writer, kind, data):
    """One PNG chunk, without concatenating a compressed file in memory."""
    writer.write(struct.pack(">I", len(data)) + kind)
    writer.write(data)
    writer.write(struct.pack(">I", zlib.crc32(data, zlib.crc32(kind)) & 0xffffffff))


def row_png(pixels, writer):
    """Experimental uint8 PNG: fixed SUB filter, one continuous zlib stream.

    Holds one color-reordered row and one filtered row; compressor output is
    emitted immediately in <=64KiB IDAT chunks. No whole-image RGB/PNG buffer.
    Stdlib zlib owns cleanup even when Python file I/O interrupts this function.
    """
    if pixels.dtype != np.uint8 or pixels.ndim not in (2, 3):
        raise ValueError("row PNG expects uint8 GRAY, BGR or BGRA")
    height, width = pixels.shape[:2]
    channels = 1 if pixels.ndim == 2 else pixels.shape[2]
    if channels not in (1, 3, 4) or width < 1 or height < 1:
        raise ValueError("invalid row PNG shape")
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    writer.write(b"\x89PNG\r\n\x1a\n")
    png_chunk(writer, b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
    raw = np.empty(width * channels, dtype=np.uint8)
    filtered = np.empty(raw.size + 1, dtype=np.uint8)
    colors = raw.reshape(width, channels)
    filtered[0] = 1  # SUB; byte distance is channels for 8-bit samples.
    compressor = zlib.compressobj(level=1, method=zlib.DEFLATED, wbits=15,
                                  memLevel=8, strategy=zlib.Z_RLE)

    def emit(data):
        # Returned compressor bytes are bounded by row input + zlib's internal
        # pending block, not by the entire image. Slices borrow those bytes.
        view = memoryview(data)
        for offset in range(0, len(view), 65536):
            png_chunk(writer, b"IDAT", view[offset:offset + 65536])

    for y in range(height):
        if channels == 1:
            np.copyto(raw, pixels[y].reshape(-1))
        else:
            colors[:, 0] = pixels[y, :, 2]
            colors[:, 1] = pixels[y, :, 1]
            colors[:, 2] = pixels[y, :, 0]
            if channels == 4:
                colors[:, 3] = pixels[y, :, 3]
        filtered[1:1 + channels] = raw[:channels]
        np.subtract(raw[channels:], raw[:-channels], out=filtered[1 + channels:])
        emit(compressor.compress(memoryview(filtered)))
    emit(compressor.flush(zlib.Z_FINISH))
    png_chunk(writer, b"IEND", b"")


def write_png(holder, path, backend, *, failure=None, released=None,
              error_policy="direct", diagnostic=None):
    if error_policy not in ("direct", "deferred"):
        raise ValueError("unknown error policy")
    started = time.perf_counter_ns()
    pixels = holder.take()
    temporary = path.with_suffix(".pending")
    prepared = encoded_at = finished = None
    pil = encoded = None
    writer = None
    encoding_completed = False
    try:
        if backend == "pillow_stream":
            h, w = pixels.shape[:2]
            # BGR is not a Pillow mmap mode: this creates one 4-byte/pixel RGB
            # storage image, without a full RGB ndarray or bytes intermediate.
            pil = Image.frombuffer("RGB", (w, h), pixels, "raw", "BGR", pixels.strides[0], 1)
            del pixels
            if released is not None:
                released()
            prepared = time.perf_counter_ns()
            with temporary.open("wb", buffering=0) as handle:
                writer_type = DeferredErrorWriter if error_policy == "deferred" else Writer
                writer = writer_type(handle, fail_write=failure == "write")
                pil.save(writer, format="PNG", compress_level=1,
                         compress_type=zlib.Z_RLE, optimize=False)
                encoding_completed = True
                encoded_at = time.perf_counter_ns()
                pil.close()
                pil = None
                if isinstance(writer, DeferredErrorWriter):
                    writer.finish()
                if failure == "fsync":
                    raise OSError("injected fsync failure")
                handle.flush()
                os.fsync(handle.fileno())
                digest = writer.digest.hexdigest()
                byte_count = writer.bytes_written
        elif backend == "row_stream":
            prepared = time.perf_counter_ns()
            with temporary.open("wb", buffering=0) as handle:
                writer = Writer(handle, fail_write=failure == "write")
                row_png(pixels, writer)
                encoding_completed = True
                encoded_at = time.perf_counter_ns()
                del pixels
                if released is not None:
                    released()
                if failure == "fsync":
                    raise OSError("injected fsync failure")
                handle.flush()
                os.fsync(handle.fileno())
                digest = writer.digest.hexdigest()
                byte_count = writer.bytes_written
        elif backend == "opencv_imencode":
            prepared = time.perf_counter_ns()
            ok, encoded = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1,
                cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE])
            if not ok:
                raise OSError("imencode returned false")
            encoding_completed = True
            encoded_at = time.perf_counter_ns()
            # Preserve the current production raw lifetime through file commit.
            with temporary.open("wb", buffering=0) as handle:
                if failure == "write":
                    raise InjectedWriteFailure("injected post-imencode write failure")
                if handle.write(memoryview(encoded)) != encoded.nbytes:
                    raise OSError("short file write")
                handle.flush()
                if failure == "fsync":
                    raise OSError("injected fsync failure")
                os.fsync(handle.fileno())
            digest = hashlib.sha256(memoryview(encoded)).hexdigest()
            byte_count = encoded.nbytes
        else:
            raise ValueError("unknown backend")
        if failure == "replace":
            raise OSError("injected replace failure")
        os.replace(temporary, path)
        finished = time.perf_counter_ns()
        return {"state": "saved", "sha256": digest, "bytes": byte_count,
                "total_ms": (finished - started) / 1e6,
                "prepare_ms": (prepared - started) / 1e6,
                "encode_ms": (encoded_at - prepared) / 1e6,
                "commit_ms": (finished - encoded_at) / 1e6}
    finally:
        if diagnostic is not None:
            diagnostic.update({"error_policy": error_policy,
                "encoding_completed_before_raise": encoding_completed,
                "discarded_bytes": getattr(writer, "discarded_bytes", 0),
                "first_deferred_error_type": type(writer.error).__name__
                    if isinstance(writer, DeferredErrorWriter) and writer.error is not None else None,
                "first_deferred_error_message": str(writer.error)
                    if isinstance(writer, DeferredErrorWriter) and writer.error is not None else None})
        if pil is not None:
            pil.close()
        temporary.unlink(missing_ok=True)


def verify(path, expected, digest):
    actual_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            actual_digest.update(chunk)
    # Python Path.open/read supports the Unicode path; decode tests pixel BGR.
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    return (actual_digest.hexdigest() == digest and decoded is not None
            and decoded.shape == expected.shape and np.array_equal(decoded, expected))


class MemorySampler:
    def __init__(self):
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            self.process = None
        self.stop = threading.Event()
        self.thread = None
        self.baseline = self.peak = self.final = None

    def __enter__(self):
        if self.process is not None:
            self.baseline = self.peak = self.process.memory_info().rss
            def sample():
                while not self.stop.wait(.002):
                    self.peak = max(self.peak, self.process.memory_info().rss)
            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_):
        if self.thread is not None:
            self.stop.set()
            self.thread.join()
            self.final = self.process.memory_info().rss
            self.peak = max(self.peak, self.final)

    def as_dict(self):
        return {"baseline_rss": self.baseline, "sampled_peak_rss": self.peak,
                "final_rss": self.final, "interval_ms": 2,
                "note": "Diagnostic sampled process RSS, not a proven allocation bound; excludes capture GPU."}


def check_failures(directory, pixels, backend, iterations, error_policy="direct"):
    rows = []
    for failure in ("write", "fsync", "replace"):
        caught = 0
        diagnostics = []
        path = directory / f"故障-{backend}-{error_policy}-{failure}.png"
        with MemorySampler() as memory:
            for _ in range(iterations):
                diagnostic = {}
                try:
                    write_png(Holder(pixels), path, backend, failure=failure,
                              error_policy=error_policy, diagnostic=diagnostic)
                except OSError as error:
                    caught += 1
                    diagnostic.update({"raised_type": type(error).__name__, "raised_message": str(error)})
                else:
                    raise AssertionError("injected failure was not raised")
                if path.exists() or path.with_suffix(".pending").exists():
                    raise AssertionError("failure left a committed or temporary image")
                if backend == "pillow_stream" and failure == "write":
                    completed = diagnostic["encoding_completed_before_raise"]
                    if completed != (error_policy == "deferred"):
                        raise AssertionError("Pillow write-failure policy did not match encoder completion")
                    if error_policy == "deferred" and (diagnostic["discarded_bytes"] <= 0
                            or diagnostic["raised_type"] != diagnostic["first_deferred_error_type"]
                            or diagnostic["raised_message"] != diagnostic["first_deferred_error_message"]):
                        raise AssertionError("deferred writer did not preserve the first error")
                diagnostics.append(diagnostic)
        rows.append({"failure": failure, "error_policy": error_policy,
                     "caught": caught, "attempts": iterations, "diagnostics": diagnostics,
                     "memory": memory.as_dict()})
    # Actual Executor WorkItem owns Holder only. Source owner must die as soon as
    # frombuffer has copied it, while the PNG save is still about to begin.
    owned = pixels.copy()
    reference = weakref.ref(owned)
    holder = Holder(owned)
    del owned
    released = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        path = directory / "持有权转换-中文.png"
        result = pool.submit(write_png, holder, path, "pillow_stream",
                             released=lambda: released.append(reference() is None)).result()
    if released != [True] or holder.pixels is not None or not verify(path, pixels, result["sha256"]):
        raise AssertionError("executor holder transfer or pixel check failed")
    path.unlink()
    return rows


def check_row_formats(directory):
    rng = np.random.default_rng(20261015)
    values = {"GRAY": rng.integers(0, 256, (171, 256), dtype=np.uint8),
              "BGR": rng.integers(0, 256, (171, 256, 3), dtype=np.uint8),
              "BGRA": rng.integers(0, 256, (171, 256, 4), dtype=np.uint8)}
    rows = []
    for name, pixels in values.items():
        pixels.flags.writeable = False
        path = directory / f"逐行格式核对-{name}.png"
        result = write_png(Holder(pixels), path, "row_stream")
        exact = verify(path, pixels, result["sha256"])
        if not exact:
            raise AssertionError(f"row PNG pixel/alpha round trip failed: {name}")
        rows.append({"format": name, "shape": list(pixels.shape), "pixels_exact": exact,
                     "bytes": result["bytes"]})
        path.unlink()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096, 6144])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--failure-iterations", type=int, default=8)
    parser.add_argument("--opencv-threads", type=int)
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--child-memory", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.failure_iterations < 1:
        parser.error("positive samples/failure-iterations and nonnegative warmup required")
    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    if args.child_memory:
        size, scenario, backend = args.child_memory
        pixels = source(int(size), scenario)
        with tempfile.TemporaryDirectory(prefix="PNG独立内存-", dir=args.work_directory) as temporary:
            path = Path(temporary) / "原图.png"
            with MemorySampler() as memory:
                result = write_png(Holder(pixels), path, backend)
            if not verify(path, pixels, result["sha256"]):
                raise AssertionError("isolated memory probe round trip failed")
        report = memory.as_dict()
        report.update(isolated_process=True, pixel_sha256=hashlib.sha256(pixels).hexdigest(),
                      process_id=os.getpid(), opencv_threads=cv2.getNumThreads())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
        return
    records = []
    failure_rows = []
    with tempfile.TemporaryDirectory(prefix="PNG配对-中文路径-", dir=args.work_directory) as temporary:
        directory = Path(temporary)
        for size in args.sizes:
            for scenario in ("stripes", "one_bit_noise", "gradient_noise", "incompressible_uint8"):
                pixels = source(size, scenario)
                timings = {name: [] for name in ("opencv_imencode", "pillow_stream", "row_stream")}
                for step in range(args.warmup + args.samples):
                    # Rotate the first backend, reversing each complete cycle;
                    # every backend gets all three positions in six trials.
                    order = list(timings)
                    offset = step % len(order)
                    order = order[offset:] + order[:offset]
                    if (step // len(order)) % 2:
                        order.reverse()
                    for backend in order:
                        path = directory / f"原图-{size}-{scenario}-{backend}.png"
                        result = write_png(Holder(pixels), path, backend)
                        if not verify(path, pixels, result["sha256"]):
                            raise AssertionError(f"lossless round trip failed: {path}")
                        path.unlink()
                        if step >= args.warmup:
                            timings[backend].append(result)
                row = {"size": [pixels.shape[1], pixels.shape[0]], "scenario": scenario,
                       "raw_bytes": pixels.nbytes, "pil_rgb_payload_bytes": pixels.shape[0] * pixels.shape[1] * 4,
                       "source_sha256": hashlib.sha256(pixels).hexdigest(), "backends": {}}
                for backend, samples in timings.items():
                    # Fresh interpreter prevents earlier backend allocator
                    # caches from hiding this encoder's RSS growth.
                    memory_path = directory / f"内存-{backend}.json"
                    command = [sys.executable, str(Path(__file__).resolve()),
                               "--child-memory", str(size), scenario, backend,
                               "--output", str(memory_path)]
                    if args.opencv_threads is not None:
                        command.extend(["--opencv-threads", str(args.opencv_threads)])
                    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    memory_report = json.loads(memory_path.read_text(encoding="utf-8"))
                    memory_path.unlink()
                    if memory_report["pixel_sha256"] != row["source_sha256"]:
                        raise AssertionError("isolated memory fixture mismatch")
                    row["backends"][backend] = {"bytes": samples[-1]["bytes"],
                        "lossless_verified_samples": args.warmup + args.samples + 1,
                        "timings": {key: distribution([sample[key] for sample in samples])
                                    for key in ("total_ms", "prepare_ms", "encode_ms", "commit_ms")},
                        "memory": memory_report}
                records.append(row)
                print(json.dumps({"completed": len(records), "size": size,
                    "scenario": scenario, "total_p95_ms": {name: value["timings"]["total_ms"]["p95"]
                        for name, value in row["backends"].items()}}, allow_nan=False), flush=True)
        # Repeated late-write failure probes use small input so they diagnose
        # ownership/cleanup without adding expensive high-resolution workload.
        pixels = source(512, "incompressible_uint8")
        for backend, error_policy in (("opencv_imencode", "direct"),
                ("pillow_stream", "direct"), ("pillow_stream", "deferred"),
                ("row_stream", "direct")):
            failure_rows.append({"backend": backend, "error_policy": error_policy,
                "results": check_failures(directory, pixels, backend,
                                          args.failure_iterations, error_policy)})
        row_format_checks = check_row_formats(directory)
    report = {"schema_version": 4, "platform": platform.platform(), "python": platform.python_version(),
              "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
              "numpy": np.__version__, "pillow": PIL.__version__,
              "pillow_zlib": features.version_codec("zlib"), "python_zlib": zlib.ZLIB_RUNTIME_VERSION,
              "pillow_environment": {key: os.environ.get(key) for key in
                  ("PILLOW_ALIGNMENT", "PILLOW_BLOCK_SIZE", "PILLOW_BLOCKS_MAX")},
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "samples": args.samples, "warmup": args.warmup, "records": records,
              "failures": failure_rows, "row_format_checks": row_format_checks,
              "row_backend_definition": {"filter": "SUB", "level": 1, "strategy": "Z_RLE",
                  "wbits": 15, "memLevel": 8, "idat_chunk_limit": 65536,
                  "whole_image_conversion": False, "whole_png_buffer": False},
              "limits": ["Synthetic PNG/file-I/O comparison, not UI or camera acceptance.",
                  "Each backend RSS probe runs in a fresh interpreter after fixture allocation; 2ms sampling is still not a proven allocation upper bound.",
                  "Fixture stays alive throughout paired sampling; holder transfer is verified separately by weak reference.",
                  "Pillow RGB is 4 bytes/pixel; no zero-copy claim for BGR input.",
                  "Pillow hashes streaming writes; OpenCV hashes encoded output after fsync, matching its current lifetime.",
                  "row_stream is an experimental standard PNG writer, retaining raw through row encoding; throughput/compression differ from adaptive Pillow and require measured evidence.",
                  "Compressor bytes, two scanlines and hash/CRC/file state still need explicit workspace accounting; this probe does not claim an absolute allocator/RSS bound.",
                  "Write fault injection checks reported failure and file cleanup; sampled RSS cannot prove absence of a small native leak.",
                  "Deferred-error sink is an experimental I/O-failure candidate: it finishes encoding to a discard sink then rethrows the first error; no claim covers native encoder failure or MemoryError.",
                  "Full-range deterministic random uint8 is an additional incompressible boundary input, not a replacement for the three image-like scenarios.",
                  "WGC 264MiB plus three raw72MiB plus Pillow96MiB exceeds512MiB; this probe does not waive source admission."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
