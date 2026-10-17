"""Template input is a contract, never a byte fingerprint or a second ledger."""
import json
from dataclasses import replace

import pytest
from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.models import MaterialSnapshot
from fdm.live_experiment.excel_template import ExportSnapshot, TemplateProfile, template_request, working_template


def session():
    trial = ExperimentSession()
    trial.sample_name, trial.sample_number = "样品", "00123"
    for name in ("甲", "乙", "丙"):
        operator = trial.add_operator(name)
        for direction in ("warp", "weft"):
            trial.set_materials([MaterialSnapshot(f"m{i}", f"纤维{i}",1.3) for i in range(9)],
                                operator_id=operator, direction=direction)
    return trial


@pytest.mark.parametrize("count", [0,1,1488,1500,2000])
def test_all_diameters_written_without_touching_formula_rows(count):
    trial = session()
    for i in range(count): trial.add_measurement(12 + i / 10000, "m0")
    trial.adjust_count("m1", 3)
    request = template_request(ExportSnapshot.capture(trial), "sheet.xlt", "report.xls", show=False)
    assert sum(len(block["values"]) for block in request["blocks"]) == count
    assert all(int(''.join(c for c in block['cell'] if c.isdigit())) == 25 for block in request['blocks'])
    assert not any(cell["cell"] == "C10" for cell in request["cells"])
    assert request["profile"]["method_index_cell"] == "D10"
    for clear in request["clears"]:
        first = clear.split(":")[0]
        row = int(''.join(c for c in first if c.isdigit()))
        assert row < 9 or row >= 25
    assert len(request["checks"]) == 54
    assert bool(request["warnings"]) == (count > 1488)
    cells = {item["cell"]: item["value"] for item in request["cells"]}
    assert [cells[x] for x in ("D3","V3","AN3")] == ["甲","乙","丙"]
    assert all(cells[x] == "00123" for x in ("G1","Y1","AQ1"))
    assert all(cells[f"{x}6"] == "纤维0" for x in ("D","M","V","AE","AN","AW"))


def test_snapshot_does_not_follow_later_edits_or_counts():
    trial = session()
    trial.add_measurement(12,"m0")
    snap = ExportSnapshot.capture(trial)
    trial.add_measurement(18,"m0")
    trial.adjust_count("m0",2)
    request = template_request(snap,"sheet.xlt","report.xls")
    assert sum(len(b["values"]) for b in request["blocks"]) == 1
    assert not any(c["value"] for c in request["cells"] if c["cell"].endswith('8'))


def test_existing_user_template_survives_updated_default(tmp_path,monkeypatch):
    from fdm import settings
    from fdm.live_experiment import excel_template as module
    source = tmp_path/'runtime'/'content-templates'/'sheet.xlt'
    source.parent.mkdir(parents=True); source.write_bytes(b'first default')
    monkeypatch.setattr(settings,'runtime_directory',lambda:tmp_path/'runtime')
    monkeypatch.setattr(settings,'settings_directory',lambda:tmp_path/'user')
    path = working_template(create=True)
    path.write_bytes(b'user changed formulas and macros')
    source.write_bytes(b'new default')
    assert working_template(create=True).read_bytes() == b'user changed formulas and macros'


def test_mapping_validates_location_not_template_bytes(tmp_path):
    p=tmp_path/'mapping.json'
    p.write_text(json.dumps({'data_sheet':'用户输入','recalculate_macro':'更新分组'}))
    assert TemplateProfile.load(p).data_sheet == '用户输入'
    p.write_text(json.dumps({'first_data_row':9}))
    with pytest.raises(ValueError,match='覆盖'): TemplateProfile.load(p)


def test_save_images_defaults_off_but_explicit_setting_persists():
    trial=ExperimentSession()
    assert not trial.save_images
    trial.save_images=True
    assert ExperimentSession.from_dict(trial.to_dict()).save_images


def test_windows_powershell_worker_has_utf8_bom_for_non_ascii_literals():
    from pathlib import Path
    from fdm.live_experiment import excel_template
    assert Path(excel_template.__file__).with_suffix('.ps1').read_bytes().startswith(b'\xef\xbb\xbf')
