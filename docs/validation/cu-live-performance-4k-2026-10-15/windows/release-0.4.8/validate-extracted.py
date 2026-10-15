"""Run inside the independent Windows task directory; never installs an app."""
from pathlib import Path
import hashlib, json, sys, winreg
ROOT = Path(r'C:\tmp\fdm-cu-live-opt4k-20261015')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts.build_windows_onedir import run_packaged_self_check
OUT = ROOT / 'build/opt4k/release-0.4.8'
DEST = Path(r'C:\tmp\fdm-cu-live-unpacked-0.4.8-20261015')

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def installed_record():
    name = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{F0F6A8B5-4838-4DF0-B2C1-18F1D5AA4A66}_is1'
    records = []
    for hive_name, hive in [('HKLM', winreg.HKEY_LOCAL_MACHINE), ('HKCU', winreg.HKEY_CURRENT_USER)]:
        for view_name, view in [('64', winreg.KEY_WOW64_64KEY), ('32', winreg.KEY_WOW64_32KEY)]:
            try:
                with winreg.OpenKey(hive, name, 0, winreg.KEY_READ | view) as key:
                    values = {}
                    for field in ['DisplayVersion', 'InstallLocation', 'UninstallString']:
                        try: values[field] = winreg.QueryValueEx(key, field)[0]
                        except FileNotFoundError: pass
                    records.append({'hive': hive_name, 'view': view_name, 'values': values})
            except FileNotFoundError: pass
    return records

if sys.argv[1:] == ['before']:
    (OUT / 'existing-install-record-before.json').write_text(json.dumps(installed_record(), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    raise SystemExit(0)

apps = list(DEST.rglob('FiberDiameterMeasurement.exe'))
assert len(apps) == 1, apps
app = apps[0].parent
manifest_path = app / 'release-manifest.json'
source_manifest = ROOT / 'dist' / 'windows' / 'FiberDiameterMeasurement' / 'release-manifest.json'
assert digest(manifest_path) == digest(source_manifest), 'Extracted manifest differs from final onedir'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
assert manifest['profile'] == 'full' and manifest['version'] == '0.4.8'
assert manifest['excluded_components'] == ['area-models']
assert 'content-templates' in manifest['included_components']
failures = []
for row in manifest['files']:
    target = app / row['path']
    if not target.is_file() or target.stat().st_size != row['size'] or digest(target) != row['sha256']:
        failures.append(row['path'])
assert not failures, failures
from PyInstaller.archive.readers import CArchiveReader
import types
archive = CArchiveReader(str(app / 'FiberDiameterMeasurement.exe'))
pyz_name = next(name for name, row in archive.toc.items() if row[-1] == 'z')
code = archive.open_embedded_archive(pyz_name).extract('fdm.ui.live_experiment_workspace')
def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, types.CodeType):
        for item in value.co_consts: yield from strings(item)
    elif isinstance(value, (tuple, list)):
        for item in value: yield from strings(item)
compiled_strings = set(strings(code))
wording_checks = {
    'latest_measurement_label': '后选：选类修改最新测量' in compiled_strings,
    'pending_and_failed_operation_tooltip': '后选跟随本场最新测量，计算和待确认期间也可选类；取消或失败后只用于下一次，不改上一条。' in compiled_strings,
    'obsolete_last_valid_record_absent': not any('本场最后有效记录' in text for text in compiled_strings),
}
assert all(wording_checks.values()), wording_checks
report = OUT / 'extracted-installer-selfcheck-final.json' 
errors = run_packaged_self_check(app, report_path=report)
assert not errors, errors
selfcheck = json.loads(report.read_text(encoding='utf-8'))
live = selfcheck['functional_checks']['cu_live_measurement']
assert live['revision'] == 4 and len(live['cases']) == 17 and all(live['cases'].values()), live
before = json.loads((OUT / 'existing-install-record-before.json').read_text(encoding='utf-8'))
after = installed_record()
assert after == before, 'Existing installation record changed'
private_templates = sorted(str(p.relative_to(app)) for p in (app / 'runtime' / 'content-templates').rglob('*') if p.is_file())
assert len(private_templates) == 4, private_templates
installer = ROOT / 'dist' / 'installer' / 'fiber-diameter-measurement-setup-0.4.8-no-area-models.exe'
result = {'ok': True, 'validation_kind': 'same-installer-extracted-payload-not-registered-install',
          'installer': str(installer), 'installer_sha256': digest(installer),
          'extracted_app': str(app), 'manifest_sha256': digest(manifest_path),
          'files_verified': len(manifest['files']), 'frozen_final_wording_checks': wording_checks, 'private_content_templates': private_templates,
          'excluded_components': manifest['excluded_components'],
          'selfcheck_report': str(report), 'selfcheck_sha256': digest(report),
          'existing_install_record_before': before, 'existing_install_record_after': after,
          'existing_installation_unchanged': before == after}
(OUT / 'extracted-installer-validation-final.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, allow_nan=False))
