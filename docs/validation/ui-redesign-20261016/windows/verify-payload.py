"""Verify the exact installer payload without installing or changing associations."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

root = Path(r'C:\tmp\fdm-cu-live-ui-redesign-20261016')
out = root / ('build/ui-redesign-final' if '--final' in sys.argv else 'build/ui-redesign')
installer = root / 'dist/installer/fiber-diameter-measurement-setup-0.4.9-no-area-models.exe'
extractor = Path(r'C:\tmp\fdm-cu-live-final-r4-20261015\build\validation-tools\innounp.exe')
destination = root / 'dist/windows/installer-payload'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

assert sha(extractor) == '1c8453c198dd3fe7947f3705e59d2294ddce3dfa9f96a737e94449b119f44181'
assert installer.is_file()
if not destination.exists():
    print('EXTRACTING_INSTALLER', flush=True)
    with (out / 'extract-installer.log').open('w', encoding='utf-8') as log:
        subprocess.run([str(extractor), '-x', '-b', '-a', '-u', '-o', '-d' + str(destination), str(installer)], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
else:
    print('VERIFYING_PREVIOUSLY_EXTRACTED_SAME_INSTALLER', flush=True)
app = destination / '{app}'
manifest = json.loads((app / 'release-manifest.json').read_text(encoding='utf-8'))
assert manifest['version'] == '0.4.9'
assert manifest['excluded_components'] == ['area-models']
assert 'content-templates' in manifest['included_components']
files = manifest['files']
source = json.loads((root / '.fdm_snapshot.json').read_text(encoding='utf-8'))
source_files = {row['path']: row for row in source['files']}
for row in files:
    path = app / row['path']
    assert path.stat().st_size == row['size'], row['path']
    expected = row['sha256']
    if expected is None:
        # Editable report templates intentionally have no release integrity
        # hash. For delivery, still require their exact frozen-source bytes.
        assert row.get('editable') is True and row['path'].startswith('runtime/content-templates/')
        expected = source_files[row['path']]['sha256']
    assert sha(path) == expected, row['path']
assert not any('runtime/area-models/' in row['path'].replace('\\', '/') for row in files)
test_sdk = sha(root / 'build/test-sdk/MVAPI.dll')
assert not any(row['sha256'] == test_sdk or 'test-sdk' in row['path'] for row in files)
expected_sdk = {'MVAPI.dll': '1d22ce270ea4aa37fb1073943a707c82f44f255164e480e1877d606b531933ea', 'saa7130.dll': 'cbb60864f7c2526ce32515132bc04b89eb2c5e177ec5af62804e8b68e7cb6315'}
for name, digest in expected_sdk.items():
    assert sha(app / 'runtime/camera/microview/x64' / name) == digest
runtime = json.loads((root / 'build/self-check/packaged-runtime.json').read_text(encoding='utf-8'))
assert runtime['ok'] and not runtime['errors']
print('EXTRACTED_FILES_VERIFIED', len(files), flush=True)
raw_output = out / 'extracted-selfcheck.raw.stdout'
existing_output = out / 'extracted-selfcheck.json'
if not raw_output.exists() and existing_output.exists():
    # Preserve the completed exit-0 self-check from this exact extracted EXE.
    shutil.copy2(existing_output, raw_output)
if not raw_output.exists():
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    with raw_output.open('wb') as stdout, (out / 'extracted-selfcheck.stderr.log').open('wb') as stderr:
        subprocess.run([str(app / 'FiberDiameterMeasurement.exe'), '--self-check', '--json'], stdout=stdout, stderr=stderr, env=env, check=True, timeout=900)
raw = raw_output.read_bytes()
try:
    decoded = raw.decode('utf-8', errors='strict')
    output_encoding = 'utf-8'
except UnicodeDecodeError:
    # The frozen dependency stack restored a locale stream on this Chinese
    # Windows image. Its complete JSON is CP936, including Chinese summaries.
    # Never replace or discard a byte; confirm the expected localized text.
    decoded = raw.decode('cp936', errors='strict')
    assert decoded.encode('cp936', errors='strict') == raw
    assert '批处理完成：成功 2 张，失败 0 张，共 2 张。' in decoded
    output_encoding = 'cp936'
check = json.loads(decoded)
existing_output.write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding='utf-8')
(out / 'selfcheck-encoding.json').write_text(json.dumps({
    'raw_file': raw_output.name, 'raw_bytes': len(raw),
    'raw_sha256': sha(raw_output), 'detected_encoding': output_encoding,
    'strict_decode': True, 'roundtrip_exact': decoded.encode(output_encoding) == raw,
    'normalized_file': existing_output.name, 'normalized_encoding': 'utf-8',
    'first_non_utf8_byte_offset': 1334 if output_encoding == 'cp936' else None,
    'localized_field': 'functional_checks.analysis_pipeline.batch.summary',
    'mixed_log_output': False,
}, ensure_ascii=False, indent=2), encoding='utf-8')
assert check['ok'] and not check['errors']
assert check['build_id'] == manifest['build_id']
source_count = 0
for row in source['files']:
    if row['path'].startswith(('src/', 'native/', 'runtime/')):
        assert sha(root / row['path']) == row['sha256'], row['path']
        source_count += 1
templates = [{**row, 'source_sha256': source_files[row['path']]['sha256']} for row in files if 'runtime/content-templates/' in row['path'].replace('\\', '/')]
report = {'passed': True, 'version': manifest['version'], 'build_id': manifest['build_id'], 'source_commit': manifest['source_commit'], 'source_files_verified': source_count, 'installer': str(installer), 'bytes': installer.stat().st_size, 'sha256': sha(installer), 'payload': str(app), 'payload_files_verified': len(files), 'test_sdk_excluded': True, 'official_sdk_sha256': expected_sdk, 'templates': templates, 'excluded_components': manifest['excluded_components'], 'included_components': manifest['included_components'], 'extracted_selfcheck_passed': True, 'actual_installation_tested': False, 'hardware_capture_tested': False}
(out / 'installer-identity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
shutil.copy2(app / 'release-manifest.json', out / 'release-manifest.json')
shutil.copy2(root / 'build/self-check/packaged-runtime.json', out / 'packaged-runtime.json')
shutil.copy2(root / '.fdm_snapshot.json', out / 'source-manifest.json')
print(json.dumps(report, ensure_ascii=False), flush=True)
