"""A selected template remains the source, including in the global settings draft."""
import hashlib
import json
from pathlib import Path

import pytest

from fdm import settings
from fdm.live_experiment.excel_template import (
    ExcelTemplateError, default_template_path, selected_template_preferences,
    save_template_preferences, save_template_preferences_batch, template_identity, template_preferences,
    template_preferences_path, working_template,
)


@pytest.fixture
def isolated_sources(tmp_path, monkeypatch):
    root = tmp_path / 'runtime' / 'content-templates'
    root.mkdir(parents=True)
    for name in ('sheet.xlt', 'sheet_dia.xlt'):
        (root / name).write_bytes(b'program template')
    monkeypatch.setattr(settings, 'runtime_directory', lambda: tmp_path / 'runtime')
    monkeypatch.setattr(settings, 'settings_directory', lambda: tmp_path / 'profile')
    return root


@pytest.mark.parametrize('mode', ['content', 'diameter'])
def test_default_resolves_runtime_without_creating_user_files(isolated_sources, mode):
    assert working_template(create=True, trial_mode=mode) == default_template_path(mode)
    assert not settings.settings_directory().exists()


def test_bom_is_accepted_but_malformed_preferences_never_select_default(isolated_sources):
    path = template_preferences_path()
    path.parent.mkdir()
    path.write_text(json.dumps({'template': '/selected/sheet.xlt'}), encoding='utf-8-sig')
    assert working_template() == Path('/selected/sheet.xlt')
    for invalid in ('{broken', '[]', '{"template": 123}', '{"method_labels": []}'):
        path.write_text(invalid, encoding='utf-8')
        with pytest.raises(ExcelTemplateError) as error:
            working_template()
        assert error.value.code == 'template_preferences'


def test_deliberate_selection_repairs_preferences_and_drops_another_files_mapping(isolated_sources, tmp_path):
    source = tmp_path / '中文模板.xlt'
    source.write_bytes(b'custom report sheets')
    path = template_preferences_path()
    path.parent.mkdir()
    path.write_text('{broken')
    selected = selected_template_preferences(source)
    assert selected['template'] == str(source.resolve())
    assert selected['template_identity']['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    save_template_preferences(selected)
    assert working_template() == source.resolve()
    other = tmp_path / 'other.xlt'
    other.write_bytes(b'other report')
    new = selected_template_preferences(other, current={**selected, 'mapping': 'old-only.json', 'method_labels': {'x':'y'}})
    assert 'mapping' not in new and 'method_labels' not in new
    source.write_bytes(b'same name new bytes')
    assert template_identity(source)['sha256'] != selected['template_identity']['sha256']


def test_settings_selection_is_draft_only_and_default_is_explicit(isolated_sources, tmp_path, monkeypatch, desktop_application):
    from fdm.ui.trial_settings_page import TrialTemplateSettings
    from PySide6.QtWidgets import QFileDialog
    selected = tmp_path / '报告模板.xlt'
    selected.write_bytes(b'new report')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(selected), ''))
    page = TrialTemplateSettings()
    try:
        assert page.changes() == {}
        page.choose('content')
        assert template_preferences() == {}
        changes = page.changes()
        assert list(changes) == ['content']
        assert changes['content']['template'] == str(selected.resolve())
        save_template_preferences(changes['content'])
        page.mark_saved()
        assert page.changes() == {}
        assert template_preferences()['template'] == str(selected.resolve())
        page.use_default('content')
        assert page.changes() == {'content': {}}
        # Discarding this draft must retain the selected source on disk.
        assert template_preferences()['template'] == str(selected.resolve())
    finally:
        page.deleteLater()


def test_broken_configuration_does_not_prevent_settings_recovery(isolated_sources, desktop_application):
    from fdm.ui.trial_settings_page import TrialTemplateSettings
    path = template_preferences_path()
    path.parent.mkdir()
    path.write_text('null')
    page = TrialTemplateSettings()
    try:
        assert '格式无效' in page.labels['content'].text()
        assert page.changes() == {}
        page.use_default('content')
        assert page.changes() == {'content': {}}
        assert path.read_text() == 'null'
    finally:
        page.deleteLater()


def test_two_mode_settings_roll_back_first_file_if_second_save_fails(isolated_sources, monkeypatch):
    from fdm.live_experiment import excel_template
    save_template_preferences({'template': '/old-content.xlt'}, 'content')
    save_template_preferences({'template': '/old-diameter.xlt'}, 'diameter')
    original = excel_template.atomic_write_text

    def write(path, text, **kwargs):
        if path == template_preferences_path('diameter'):
            raise PermissionError('simulated locked destination')
        return original(path, text, **kwargs)

    monkeypatch.setattr(excel_template, 'atomic_write_text', write)
    with pytest.raises(ExcelTemplateError) as error:
        save_template_preferences_batch({'content': {'template': '/new-content.xlt'},
                                         'diameter': {'template': '/new-diameter.xlt'}})
    assert error.value.code == 'template_preferences_save'
    assert error.value.diagnostics['rollback_errors'] == []
    assert template_preferences('content')['template'] == '/old-content.xlt'
    assert template_preferences('diameter')['template'] == '/old-diameter.xlt'
