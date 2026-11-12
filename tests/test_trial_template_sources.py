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


def test_save_during_diagnostic_hash_does_not_reject_template_choice(isolated_sources, tmp_path, monkeypatch):
    from fdm.live_experiment import excel_template
    source = tmp_path / '编辑中的模板.xlt'
    source.write_bytes(b'original template')
    sha256 = hashlib.sha256

    class SavingDigest:
        def __init__(self):
            self.digest = sha256()

        def update(self, data):
            self.digest.update(data)
            source.write_bytes(b'new template saved while reading diagnostic hash')

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(excel_template.hashlib, 'sha256', SavingDigest)
    selected = selected_template_preferences(source)
    assert selected['template'] == str(source.resolve())
    assert selected['template_identity']['sha256'] is None
    assert selected['template_identity']['hash_status'] == 'changed_during_read'
    save_template_preferences(selected)
    assert working_template() == source


def test_diagnostic_hash_read_failure_does_not_preempt_excel_open(isolated_sources, tmp_path, monkeypatch):
    source = tmp_path / '暂时被其他程序写入.xlt'
    source.write_bytes(b'valid path')
    original_open = Path.open

    def locked_hash(path, *args, **kwargs):
        if path == source and args == ('rb',):
            raise PermissionError('diagnostic reader cannot acquire this file')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', locked_hash)
    selected = selected_template_preferences(source)
    assert selected['template_identity']['sha256'] is None
    assert selected['template_identity']['hash_status'] == 'unavailable'
    assert 'diagnostic reader' in selected['template_identity']['hash_error']
    save_template_preferences(selected)
    assert working_template() == source


@pytest.mark.parametrize('directory', [False, True])
def test_template_choice_still_requires_an_existing_file(isolated_sources, tmp_path, directory):
    source = tmp_path / 'missing-or-directory.xlt'
    if directory:
        source.mkdir()
    with pytest.raises(FileNotFoundError):
        selected_template_preferences(source)


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
