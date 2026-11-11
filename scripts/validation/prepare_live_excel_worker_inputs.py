"""Generate synthetic inputs for the two Windows live-Excel worker probes.

This does not open Excel, read templates or modify application preferences.
Template names are relative to the private directory passed to the PS probes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


def build_inputs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from fdm.live_experiment import ExperimentSession, MaterialSnapshot
    from fdm.live_experiment.excel_live import input_patch, project_inputs, report_update
    from fdm.live_experiment.excel_template import ExportSnapshot, TemplateProfile

    result = {}
    for key, mode, filename in (
        ("content", "content", "sheet.xlt"),
        ("custom", "content", "sheet11.xlt"),
        ("diameter", "diameter", "sheet_dia.xlt"),
    ):
        session = ExperimentSession(sample_name="独立验证样品", sample_number="00123",
                                    workspace_options={"mode": mode}, save_images=False)
        session.add_operator("操作员甲")
        session.set_materials([MaterialSnapshot("wool", "羊毛", 1.31),
                               MaterialSnapshot("cotton", "棉", 1.54)])
        for value in (12.5, 15.25, 18.75):
            session.add_measurement(value, "wool")
        # A fixed local date reproduces the recorded acceptance input without
        # depending on today's date or the machine's timezone.
        snapshot = replace(ExportSnapshot.capture(session), created_at="2026-11-11T12:00:00")
        profile = TemplateProfile.load(trial_mode=mode)
        projection = project_inputs(snapshot, filename, profile)
        payload = input_patch(None, projection)
        payload.update(report_update(projection, information=True, settings=True))
        # Deterministic per synthetic case; worker binds it to a workbook generation.
        payload['batch_id'] = 'validation-initialize-' + key
        # project_inputs resolves paths for production. The probes intentionally
        # resolve these three filenames beneath their private -Root instead.
        payload["template"] = filename
        result[key] = payload
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New requests.json path; must not exist")
    args = parser.parse_args()
    payload = build_inputs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Generated {len(payload)} synthetic cases: {args.output.resolve()}")


if __name__ == "__main__":
    main()
