"""Package only a verified Windows build; does not build or execute vendor code."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = 'DsxEvidenceCollector-1.0.0-win-x64'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    runs = sorted(ROOT.glob('validation/run-*/validation-result.json'))
    run = runs[-1].parent
    outcome = read(run / 'validation-result.json')
    tests = read(run / 'tests/self-tests.json')
    identity = read(run / 'dist/build-identity.json')
    assert outcome['self_test_exit'] == outcome['ui_exit'] == 0
    assert outcome['fallback_passed'] and not outcome['validation_process_elevated']
    assert len(tests) == 35 and all(t['passed'] for t in tests)
    assert digest(run / 'dist/DsxEvidenceCollector.exe') == identity['exe_sha256']
    for source in identity['sources']:
        assert digest(ROOT / 'src' / source['name']) == source['sha256'], source['name']

    latest = ROOT / 'validation/latest'
    latest.mkdir(exist_ok=True)
    for source, dest in [(run/'tests/self-tests.json', 'self-tests.json'),
                         (run/'validation-result.json', 'validation-result.json'),
                         (run/'dist/build-identity.json', 'build-identity.json')]:
        shutil.copy2(source, latest / dest)
    shutil.copytree(run/'ui', latest/'ui', dirs_exist_ok=True)

    target = ROOT / 'release' / NAME
    # A packaging error cannot replace an already delivered package unnoticed.
    target.mkdir(parents=True, exist_ok=False)
    for path in (run/'dist').iterdir():
        if path.is_file(): shutil.copy2(path, target/path.name)
    for name in ['README.md', '中文现场操作卡.md', '验证摘要.md', 'Collect-Basic.ps1', 'Collect-Basic.cmd']:
        shutil.copy2(ROOT/name, target/name)
    shutil.copytree(latest, target/'validation/latest')
    shutil.copytree(ROOT/'src', target/'source/src')
    for name in ['build.ps1', 'validate.ps1', 'Collect-Basic.ps1', 'baseline.json']:
        shutil.copy2(ROOT/name, target/'source'/name)
    # Plain text is directly viewable without an installed Markdown application.
    (target/'先读我-现场操作.txt').write_bytes(('\ufeff'+(ROOT/'中文现场操作卡.md').read_text()).encode('utf-8'))
    for path in target.rglob('*.ps1'):
        # Windows PowerShell 5.1 handles UTF-8 reliably when the BOM is explicit.
        text = path.read_text(encoding='utf-8-sig')
        path.write_bytes(b'\xef\xbb\xbf'+text.encode('utf-8'))
    files = [{'path':p.relative_to(target).as_posix(), 'bytes':p.stat().st_size, 'sha256':digest(p)}
             for p in sorted(target.rglob('*')) if p.is_file()]
    (target/'SHA256SUMS.txt').write_text(''.join(f"{f['sha256']}  {f['path']}\n" for f in files))
    (target/'release-manifest.json').write_text(json.dumps({
        'name': NAME, 'validation_run':run.name, 'exe_sha256':identity['exe_sha256'],
        'files':files, 'self_exclusion':['SHA256SUMS.txt','release-manifest.json'],
        'real_hardware_tested':False, 'windows_1909_tested':False,
        'reference_pack_48':identity['reference_pack_48'],
        'ps1_packaging':'UTF-8 BOM added for Windows PowerShell 5.1; script text unchanged'
    }, ensure_ascii=False, indent=2)+'\n')
    zip_path = Path(str(target)+'.zip')
    with zipfile.ZipFile(zip_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(target.rglob('*')):
            if path.is_file():archive.write(path, NAME+'/'+path.relative_to(target).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        for item in files:
            assert hashlib.sha256(archive.read(NAME+'/'+item['path'])).hexdigest() == item['sha256']
    zip_sha = digest(zip_path)
    Path(str(zip_path)+'.sha256').write_text(zip_sha+'  '+zip_path.name+'\n')
    report = {'status':'passed','zip':str(zip_path.relative_to(ROOT)), 'zip_sha256':zip_sha,
              'bytes':zip_path.stat().st_size,'manifest_files_verified':len(files),
              'source_files_matched_build':len(identity['sources']), 'tests_passed':len(tests),
              'validation_run':run.name,'vendor_binaries_included':False,'fdm_product_modified':False}
    (ROOT/'release/release-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))

if __name__ == '__main__':main()
