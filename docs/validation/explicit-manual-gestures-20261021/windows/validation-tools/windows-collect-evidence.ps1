$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='C:\tmp\fdm-cu-live-explicit-gestures-20261021'
$out=Join-Path $root 'build\seventh-round'
$python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$source=@'
from pathlib import Path
import hashlib,json,zipfile
from datetime import datetime,timezone
root=Path(r'C:\tmp\fdm-cu-live-explicit-gestures-20261021')
out=root/'build/seventh-round'
files=['windows-tests.log','windows-tests-status.txt','test-selection.json','source-quick.log','source-quick-status.txt','source-quick/result.json','source-quick/explicit-gestures.png','installer-build-status.txt','package-verify-status.txt','build-validation-gate.json','package/installer-identity.json','package/extracted-selfcheck.json','package/release-manifest.json','package/source-manifest.json','actual-installation.json','installed-selfcheck/result.json','installed-selfcheck/selfcheck.json','explorer-start/result.json','explorer-start/installed-window-printwindow.png','explorer-start/installed-window-screen.png','installation-backup/restored.json']
stages=[]
for phase,log,status in [('native_extensions',root/'build/cu-live-validation/native-build.log',None),('targeted_pytest',out/'windows-tests.log',out/'windows-tests-status.txt'),('source_quick',out/'source-quick.log',out/'source-quick-status.txt'),('installer_build',out/'installer-build.log',out/'installer-build-status.txt'),('package_verification',out/'package-verify.log',out/'package-verify-status.txt'),('ordinary_installed_selfcheck',out/'installed-selfcheck-launch.log',None),('explorer_launch',out/'explorer-launch.log',None)]:
 st=log.stat(); start=getattr(st,'st_birthtime',st.st_ctime); end=status.stat().st_mtime if status else st.st_mtime
 stages.append(dict(phase=phase,start_utc=datetime.fromtimestamp(start,timezone.utc).isoformat(),end_utc=datetime.fromtimestamp(end,timezone.utc).isoformat(),elapsed_seconds=round(end-start,3),timing_basis='log creation to status write or final log write'))
(out/'phase-timings.json').write_text(json.dumps(stages,indent=2))
files.append('phase-timings.json')
archive=out/'explicit-gestures-validation-evidence.zip'
if archive.exists(): raise RuntimeError('Preserve existing evidence archive')
for name in files:
 if not (out/name).is_file(): raise RuntimeError('Missing evidence: '+name)
identity=json.loads((out/'package/installer-identity.json').read_text(encoding='utf-8'))
manifest=json.loads((out/'package/source-manifest.json').read_text(encoding='utf-8'))
canvas=next(x for x in manifest['files'] if x['path']=='src/fdm/ui/live_measurement_canvas.py')
assert identity['version']=='0.4.10' and identity['passed'] is True
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(out/name,name)
print(json.dumps(dict(evidence=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),build_id=identity['build_id'],installer_sha256=identity['sha256'],installer_bytes=identity['bytes'],canvas=canvas,real_microview_tested=False),ensure_ascii=True))
'@
$path=Join-Path $out 'collect-explicit-gestures-evidence.py'
[IO.File]::WriteAllText($path,$source,[Text.UTF8Encoding]::new($false))
& $python $path
exit $LASTEXITCODE
