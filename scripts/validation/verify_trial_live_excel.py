"""Windows Qt coordinator -> real persistent PowerShell -> desktop Excel probe.

Only disposable workbooks created by this probe are used. Their Excel PIDs are
recorded for explicit local cleanup; templates are never saved by the probe.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-template", type=Path, required=True)
    parser.add_argument("--diameter-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This probe requires Windows desktop Excel")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from fdm import settings
    from fdm.live_experiment import ExperimentSession, MaterialSnapshot
    from fdm.live_experiment.excel_template import save_template_preferences
    from fdm.ui.live_excel import LiveExcelManager
    settings.settings_file_path = lambda: output / "profile" / "settings.json"
    app = QApplication.instance() or QApplication([])
    manager = LiveExcelManager()
    report = {"platform": sys.platform, "source": "native_desktop_excel", "cases": {}, "events": []}
    template_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.content_template, args.diameter_template)}
    report["templates"] = template_hashes
    manager.changed.connect(lambda sid: report["events"].append({"time": time.perf_counter(), "session": sid, **manager.status(sid)}))

    def wait(session, predicate, timeout=60):
        until = time.monotonic() + timeout
        while not predicate() and time.monotonic() < until:
            QTest.qWait(5)
            status = manager.status(session.id)
            if status["state"] in ("error", "unsupported", "stopped"):
                raise RuntimeError(json.dumps(status, ensure_ascii=False))
        assert predicate(), manager.status(session.id)

    def metrics(values):
        ordered = sorted(values)
        return {"samples": len(values), "p50_ms": statistics.median(values),
                "p95_ms": ordered[math.ceil(len(ordered)*.95)-1], "max_ms": max(values)}

    try:
        sessions = []
        for mode, template in (("content", args.content_template), ("diameter", args.diameter_template)):
            save_template_preferences({"template": str(template.resolve())}, trial_mode=mode)
            session = ExperimentSession(workspace_options={"mode": mode}, save_images=False)
            session.add_operator("验证员甲")
            session.set_materials([MaterialSnapshot("one", "绵羊毛", 1.31), MaterialSnapshot("two", "山羊绒", 1.31)])
            session.configure(sample_name="FDM 实时同步验证", sample_number=mode + "-20261111")
            sessions.append(session)
            started = time.perf_counter()
            manager.ensure(session)
            wait(session, lambda: manager.flush(session))
            case = report["cases"][mode] = {"startup_ms": (time.perf_counter()-started)*1000}
            raw_latency, final_latency, on_interaction, off_interaction = [], [], [], []
            baseline = ExperimentSession.from_dict(session.to_dict())
            for index in range(20):
                started = time.perf_counter()
                baseline.add_measurement(20 + index / 100, "one")
                off_interaction.append((time.perf_counter()-started)*1000)
                started = time.perf_counter()
                session.add_measurement(20 + index / 100, "one")
                revision = session.revision
                manager.update(session)
                on_interaction.append((time.perf_counter()-started)*1000)
                wait(session, lambda: manager.status(session.id).get("inputs_written_revision") == revision)
                raw_latency.append((time.perf_counter()-started)*1000)
                wait(session, lambda: manager.flush(session))
                final_latency.append((time.perf_counter()-started)*1000)
            # Inputs-written precedes calculation/repaint. Only final ACK is
            # used as the conservative visible-workbook latency boundary.
            case.update(inputs_written=metrics(raw_latency), report_ready_and_visible=metrics(final_latency),
                        interaction_off=metrics(off_interaction), interaction_on=metrics(on_interaction))
            identity = next(iter(session.measurements))
            session.edit_measurement(identity, diameter_um=25.125, material_id="two")
            manager.update(session)
            wait(session, lambda: manager.flush(session))
            session.delete_measurement(identity)
            manager.update(session)
            wait(session, lambda: manager.flush(session))
            session.undo()
            manager.update(session)
            wait(session, lambda: manager.flush(session))
            case["final"] = manager.status(session.id)
            case["record_count"] = len(session.measurements)
            assert not case["final"].get("warnings"), case["final"]
            case["rebuilds"] = []
            for target in (300, 1502 if mode == "content" else 2977):
                started = time.perf_counter()
                session.add_measurements_batch([{"diameter_um": 20 + index % 50 / 10, "material_id": "one"}
                    for index in range(target - len(session.measurements))])
                manager.update(session)
                wait(session, lambda: manager.flush(session))
                result = manager.status(session.id)
                case["rebuilds"].append({"records": len(session.measurements),
                    "total_ms": (time.perf_counter()-started)*1000, "status": result})
                if target > 300:
                    assert result.get("warnings"), "Capacity overflow must remain visible"
                else:
                    assert not result.get("warnings"), result
            session.undo()
            session.undo()
            manager.update(session)
            wait(session, lambda: manager.flush(session))
            assert len(session.measurements) == 20
            assert not manager.status(session.id).get("warnings")
        # A second trial must not reopen/rewrite the first workbook.
        manager.ensure(sessions[0])
        wait(sessions[0], lambda: manager.flush(sessions[0]))
        sessions[0].add_measurement(22.5, "one")
        manager.update(sessions[0])
        wait(sessions[0], lambda: manager.flush(sessions[0]))
        report["content_update_while_diameter_is_open"] = manager.status(sessions[0].id)
        for session in sessions:
            wait(session, lambda session=session: manager.detach(session))
        until = time.monotonic()+30
        while not manager.close() and time.monotonic()<until:
            QTest.qWait(10)
        assert manager.close(), "worker did not detach/shutdown"
        assert template_hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.content_template, args.diameter_template)}
        report["passed"] = True
    except Exception as exc:
        report["passed"], report["error"] = False, str(exc)
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "cases": report["cases"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
