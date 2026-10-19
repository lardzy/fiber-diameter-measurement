"""Archive only third-round evidence, excluding live OS caches in test profiles."""
from pathlib import Path
import hashlib,json,zipfile
root=Path(r'C:\tmp\fdm-cu-live-third-round-20261019')
source=root/'build/third-round'
destination=root/'build/third-round-runtime-evidence-complete.zip'
if destination.exists():
    raise RuntimeError('Evidence archive already exists; preserve before retrying')
files=[]
for folder in (source/'frozen-gui',source/'explorer-start'):
    for p in folder.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in ('.png','.json','.xls','.xlt','.fdmproj','.log','.txt'):
            continue
        if 'isolated-config' in p.parts and 'FiberDiameterMeasurement' not in p.parts:
            continue
        files.append(p)
for name in ('actual-installation.log','actual-installation.json','installer-build.log','package-verify.log','test-selection.json'):
    files.append(source/name)
with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(set(files)):
        archive.write(p,p.relative_to(source).as_posix())
with destination.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
print(json.dumps({'path':str(destination),'files':len(set(files)),'bytes':destination.stat().st_size,'sha256':digest}))
