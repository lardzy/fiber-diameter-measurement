"""Production-path probes for the recovered live measurement feature."""
from __future__ import annotations

import math
import sys


CU_LIVE_PROBE_REVISION = 6

_ALIGNMENT_WORKER_CASES = {
    "ok": "candidate_alignment_worker",
    "same": "candidate_alignment_same",
    "noisy": "candidate_alignment_noisy",
    "drift": "candidate_alignment_drift",
    "reference_reset": "candidate_alignment_reference_reset",
    "closed": "candidate_alignment_closed",
    "child_process": "candidate_alignment_child_process",
}


def _optical_worker_probe():
    """Verify the packaged optical imports through the actual spawn dispatcher."""
    import os
    import time
    import numpy as np
    from fdm.services.cu_automatic_worker import CuAutomaticWorker
    pixels = np.full((160, 240), 220, np.uint8)
    pixels[70:90, 30:210] = 195
    try:
        with CuAutomaticWorker() as worker:
            task = worker.submit(pixels, "optical_adaptive", config={"imaging_mode": "mixed"})
            deadline = time.monotonic() + 32
            while time.monotonic() < deadline:
                responses = worker.poll()
                if responses:
                    response = responses[0]
                    result = response.get("result") or {}
                    candidates = result.get("candidates") or []
                    pid = worker._process.pid if worker._process is not None else None
                    return {"ok": bool(response.get("task_id") == task and not response.get("error")
                        and result.get("status") == "ok" and len(candidates) == 1
                        and abs(candidates[0]["width_px"] - 20) <= 1
                        and pid is not None and pid != os.getpid()),
                        "worker_pid": pid, "child_process": pid is not None and pid != os.getpid(),
                        "response": response}
                time.sleep(.005)
        return {"ok": False, "error": "optical worker timed out"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _candidate_alignment_worker_probe():
    """Run the actual spawn path and retain failure/runtime details for offline checks."""
    try:
        from fdm.live_experiment.candidate_alignment_worker import run_alignment_worker_self_check

        result = run_alignment_worker_self_check()
        if not isinstance(result, dict):
            result = {"ok": False, "error": "Invalid alignment worker self-check result"}
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    cases = {name: result.get(field) is True for field, name in _ALIGNMENT_WORKER_CASES.items()}
    return cases, result


def _field_preparation_probe():
    """Run the split production preparation and compare its ordered decisions."""
    import cv2
    import numpy as np
    from fdm.live_experiment.frames import FrameHandle
    from fdm.live_experiment.motion import FieldMotionDetector, MotionPreparer, reduced_gray

    rng = np.random.default_rng(312)
    image = rng.integers(30, 220, (240, 320, 3), dtype=np.uint8)
    image[:, :80] = 0  # An asymmetric dark mass detects an inverted weighting.
    preparer = MotionPreparer()
    serial, split = FieldMotionDetector(), FieldMotionDetector()
    for sequence, pixels in enumerate((image, image ^ 1, np.roll(image, 30, axis=0), image)):
        frame = FrameHandle.snapshot(pixels, source_id="preparation-probe", sequence=sequence, live=True)
        prepared = preparer.prepare(frame)
        expected = serial.compare(frame)
        actual = split.compare_prepared(prepared, (frame.source_id, frame.generation, frame.size),
                                        frame.frame_id, frame.sequence)
        # Independent integer moments also exercise the compiled centroid path.
        gray = 255 - prepared.gray.astype(np.uint64)
        mass = int(gray.sum())
        center = (int((gray.sum(axis=0) * np.arange(gray.shape[1], dtype=np.uint64)).sum()) // mass,
                  int((gray.sum(axis=1) * np.arange(gray.shape[0], dtype=np.uint64)).sum()) // mass)
        if (actual != expected or prepared.centroid != center
                or prepared.gray.flags.writeable or prepared.spectrum.flags.writeable):
            return False
    # Exercise the contiguous large-frame view in the frozen application too;
    # serial and prepared paths alone would otherwise share the same reducer.
    for channels in (3, 4):
        large = rng.integers(0, 256, (2049, 2049, channels), dtype=np.uint8)
        large.flags.writeable = False
        code = cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY
        expected = cv2.resize(cv2.cvtColor(large, code), (683, 683), interpolation=cv2.INTER_LINEAR)
        if not np.array_equal(reduced_gray(large), expected):
            return False
    return True


def _content_configuration_probe():
    """Exercise the real deferred aggregate transaction and its width history.

    The tiny probe runs the build synchronously; GUI scheduling and cancellation
    are covered by workspace tests. Independent midpoint values detect a stale
    aggregate even if configuration serialization itself appears correct.
    """
    from fdm.live_experiment import ExperimentSession, default_materials, summarize

    session = ExperimentSession(save_images=False)
    session.add_operator("configuration-probe")
    session.set_materials(default_materials())
    session.add_measurement(15, "wool")
    session.add_measurement(30, "wool")
    session.adjust_count("wool", 1)
    baseline = session.to_dict()
    checks = {"content_deferred_rebuild": False, "content_width_undo": False,
              "content_width_redo": False}

    def matches(expected_width, expected_mean, expected_variance):
        row = next(row for row in summarize(session)["operators"][session.active_operator_id]["warp"]["categories"]
                   if row["id"] == "wool")
        state = session.to_dict()
        return (session.group_width == expected_width and row["mean"] == 22.5
                and row["calculation_mean"] == expected_mean
                and math.isclose(row["calculation_stddev"], math.sqrt(expected_variance), rel_tol=1e-12)
                and row["extra_count"] == 1 and row["total_count"] == 3
                and state["measurements"] == baseline["measurements"]
                and state["count_adjustments"] == baseline["count_adjustments"])

    request = session.configure_deferred(group_width=1.75)
    if request is None:
        return checks
    unchanged = session.to_dict() == baseline
    rebuilt = request.build()
    unchanged = unchanged and session.to_dict() == baseline
    session.apply_rebuild(request, rebuilt)
    # For 1.75 μm bins, 15 -> 14.875 and 30 -> 30.625; sample variance 124.03125.
    changed = session.to_dict()
    loaded = ExperimentSession.from_dict(changed)
    checks["content_deferred_rebuild"] = (unchanged and matches(1.75, 22.75, 124.03125)
                                          and loaded.group_width == 1.75 and loaded.to_dict() == changed)
    for action, width, mean, variance in (("undo", 2.5, 23.75, 112.5),
                                          ("redo", 1.75, 22.75, 124.03125)):
        before = session.to_dict()
        request = session.history_deferred(action)
        if request is None:
            return checks
        unchanged = session.to_dict() == before
        rebuilt = request.build()
        unchanged = unchanged and session.to_dict() == before
        session.apply_rebuild(request, rebuilt)
        checks[f"content_width_{action}"] = unchanged and matches(width, mean, variance)
    return checks


def _project_asset_rebase_probe(root, evidence, pixels):
    """Commit a real save, then retire the old evidence path and exercise history."""
    from pathlib import Path
    import cv2
    import numpy as np
    from fdm.live_experiment import ExperimentSession, default_materials
    from fdm.live_experiment.project_assets import rebase_experiment_assets
    from fdm.models import ProjectState, project_assets_root
    from fdm.project_io import ProjectIO

    session = ExperimentSession(save_images=True)
    session.add_operator("asset-rebase-probe")
    session.set_materials(default_materials())
    active = session.add_measurement(15, "wool", evidence=evidence)
    absent = session.add_measurement(30, "wool", evidence=evidence)
    project = ProjectState.empty()
    project.live_experiments.append(session)
    saved_snapshot = project.to_dict()
    target = root / "saved-project" / "probe.fdmproj"
    ProjectIO.save(project, target)
    session.delete_measurement(absent.id)
    before = (session.revision, list(session._undo), list(session._redo))
    rebase_experiment_assets(project.live_experiments, saved_snapshot, project_assets_root(target))
    expected = project_assets_root(target).resolve() / "experiments" / f"{evidence['sha256']}.png"
    if (before != (session.revision, session._undo, session._redo)
            or Path(session.measurements[active.id].evidence["path"]) != expected):
        return False
    # This is exclusively the probe's temporary source, never a user image.
    Path(evidence["path"]).unlink()
    if not session.undo() or Path(session.measurements[absent.id].evidence["path"]) != expected:
        return False
    if not session.redo() or absent.id in session.measurements or not session.undo():
        return False
    if Path(session.measurements[absent.id].evidence["path"]) != expected:
        return False
    ProjectIO.save(project, target)
    loaded = ProjectIO.load(target).live_experiments[0]
    for record_id in (active.id, absent.id):
        restored = loaded.measurements[record_id].evidence
        if restored["state"] != "saved" or Path(restored["path"]) != expected:
            return False
    decoded = cv2.imdecode(np.frombuffer(expected.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return decoded is not None and np.array_equal(decoded, pixels)


def _evidence_persistence_probe():
    """Execute the production evidence writer and verified asset publication."""
    import hashlib
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import cv2
    import numpy as np
    from fdm.live_experiment.evidence import EvidenceStore
    from fdm.live_experiment.frames import FrameBudget, FrameHandle

    cases = {name: False for name in ("evidence_png_gray", "evidence_png_bgr",
                                     "evidence_png_bgra", "evidence_leases_released",
                                     "project_asset_rebase")}
    values = np.arange(17 * 23 * 4, dtype=np.uint32).reshape(17, 23, 4)
    # Distinct channels and varying alpha detect channel swaps and alpha loss.
    rgba_pattern = ((values * 37 + values // 13) % 256).astype(np.uint8)
    images = {"gray": rgba_pattern[:, :, 0].copy(), "bgr": rgba_pattern[:, :, :3].copy(),
              "bgra": rgba_pattern.copy()}
    with TemporaryDirectory(prefix="fdm-cu-evidence-probe-") as temporary:
        root = Path(temporary)
        budget = FrameBudget(limit_bytes=16 * 1024**2)
        store = EvidenceStore(root / "original-evidence", budget=budget)
        entries = {}
        try:
            for sequence, (name, pixels) in enumerate(images.items()):
                frame = FrameHandle.snapshot(pixels, source_id="evidence-probe", sequence=sequence,
                                             pixel_format=name.upper(), live=False)
                entries[name] = store.reserve(frame)
        finally:
            store.close()
        cases["evidence_leases_released"] = (not store.pending and budget.used_bytes == 0
                                             and not budget._leases and not budget._scratch
                                             and not budget._frames)
        for name, pixels in images.items():
            entry = entries[name]
            path = Path(entry["path"])
            if entry["state"] != "saved" or not path.is_file():
                continue
            payload = path.read_bytes()
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            cases[f"evidence_png_{name}"] = (entry.get("encoder") == "png-block-fast-v2"
                                             and hashlib.sha256(payload).hexdigest() == entry["sha256"]
                                             and decoded is not None and decoded.dtype == np.uint8
                                             and np.array_equal(decoded, pixels))
        if cases["evidence_png_bgr"]:
            cases["project_asset_rebase"] = _project_asset_rebase_probe(root, entries["bgr"], images["bgr"])
    return cases


def run_cu_live_self_check(*, validate_capture=True):
    import numpy as np
    from fdm._native import require_native
    from fdm.services.cu_manual_measurement import measure_manual
    from fdm.services.cu_automatic_measurement import analyze_frame
    from fdm.live_experiment import ExperimentSession, default_materials, summarize
    from fdm.live_experiment.frames import FrameHandle
    from fdm.live_experiment.motion import FieldMotionDetector, sobel_fraction

    native = require_native()
    image = np.full((240, 320, 3), 220, np.uint8)
    image[114:126, 40:280] = 40
    manual = measure_manual(image, (160, 90), (160, 150))
    traditional = analyze_frame(image, "traditional")
    multiscale = analyze_frame(image, "multiscale")
    optical = analyze_frame(image, "optical_adaptive")
    optical_empty = analyze_frame(np.full_like(image, 220), "optical_adaptive")
    session = ExperimentSession(save_images=False)
    session.add_operator("probe")
    session.set_materials(default_materials())
    session.add_measurement(15, "wool")
    session.add_measurement(30, "wool")
    session.adjust_count("wool", 1)
    session.add_measurement(18, "cashmere")
    categories = summarize(session)["operators"][session.active_operator_id]["warp"]["categories"]
    wool = next(row for row in categories if row["id"] == "wool")
    detector = FieldMotionDetector()
    first = FrameHandle.snapshot(image, source_id="probe", live=True)
    second = FrameHandle.snapshot(np.roll(image, 30, axis=0), source_id="probe", sequence=1, live=True)
    cases = {
        "manual_native": manual["status"] == "ok" and math.isclose(manual["width_px"], 11.0, abs_tol=1e-10),
        "traditional_native": traditional["status"] == "ok" and len(traditional["candidates"]) == 1 and math.isclose(traditional["candidates"][0]["width_px"], 11.0, abs_tol=1e-10),
        "multiscale_native": multiscale["status"] == "ok" and math.isclose(multiscale["candidates"][0]["width_px"], 10.997208727965706, abs_tol=1e-8),
        "optical_adaptive": (optical["status"] == "ok" and len(optical["candidates"]) == 1
                             and abs(optical["candidates"][0]["width_px"] - 12.0) <= 1),
        "optical_empty": optical_empty["status"] == "empty" and not optical_empty["candidates"],
        "content_formula": math.isclose(wool["content"], 85.33239059467755, abs_tol=1e-10),
        "field_motion": detector.compare(first).stable and not detector.compare(second).stable,
        "field_preparation": _field_preparation_probe(),
        "project_roundtrip": ExperimentSession.from_dict(session.to_dict()).to_dict() == session.to_dict(),
    }
    cases.update(_content_configuration_probe())
    cases.update(_evidence_persistence_probe())
    alignment_cases, alignment_report = _candidate_alignment_worker_probe()
    cases.update(alignment_cases)
    optical_worker = _optical_worker_probe()
    cases["optical_worker"] = optical_worker.get("ok") is True
    if sys.platform == "win32" and validate_capture:
        from fdm._native import _microview
        from fdm.services.microview_transport import bridge_build_matches
        cases["microview_bridge_import"] = callable(_microview.FrameTap)
        cases["microview_bridge_transport"] = all(_microview.self_check().values())
        capture_info = dict(_microview.build_info())
        kernel_info = dict(native.build_info())
        cases["microview_bridge_build"] = bridge_build_matches(capture_info, kernel_info)
    return {"ok": all(cases.values()), "revision": CU_LIVE_PROBE_REVISION, "cases": cases,
            "candidate_alignment_worker": alignment_report,
            "optical_worker": optical_worker,
            "native_build_info": dict(native.build_info()), "hardware_capture_tested": False}


def cu_live_probe_passed(report):
    required = {"manual_native", "traditional_native", "multiscale_native", "optical_adaptive", "optical_empty", "optical_worker", "content_formula",
                "field_motion", "field_preparation", "project_roundtrip", "content_deferred_rebuild",
                "content_width_undo", "content_width_redo", "evidence_png_gray",
                "evidence_png_bgr", "evidence_png_bgra", "evidence_leases_released",
                "project_asset_rebase"}
    required.update(_ALIGNMENT_WORKER_CASES.values())
    if sys.platform == "win32":
        required.update({"microview_bridge_import", "microview_bridge_build", "microview_bridge_transport"})
    return (isinstance(report, dict) and report.get("ok") is True and report.get("revision") == CU_LIVE_PROBE_REVISION
            and isinstance(report.get("cases"), dict)
            and all(report["cases"].get(name) is True for name in required))
