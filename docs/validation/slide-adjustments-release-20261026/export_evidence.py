from pathlib import Path
import hashlib,json,zipfile,shutil
r=Path(r'C:\tmp\fdm-slide-adjustments-release-20261026');s=Path(r'\\Mac\Home\Downloads\fdm-slide-adjustments-release-transfer-20261026')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
id=json.loads((r/'build/package/installer-identity.json').read_text(encoding='utf-8'));assert id['passed']
runs=list((r/'build/frozen-gui').glob('*/result.json'));assert runs
gui=json.loads(max(runs,key=lambda p:p.stat().st_mtime_ns).read_text(encoding='utf-8-sig'))
assert gui['passed'] and gui['graceful_exit'] and gui['source_unchanged']
assert len(gui['exits'])==3 and all(p['code']==0 for p in gui['exits'])
exe=Path(id['installer']);assert sha(exe)==id['sha256']
name=f"FDM-{id['version']}-slide-adjustments-{id['build_id'][:12]}.exe"
if not (s/name).exists():shutil.copy2(exe,s/name)
assert sha(s/name)==id['sha256']
files=[]
for group in ['package','frozen-gui']:
 for p in (r/'build'/group).rglob('*'):
  if p.is_file() and not any(part in ['isolated-config','temp'] for part in p.relative_to(r/'build').parts):files.append(p)
for name2 in ['release-build.log','release-status.json','verify.log','verify.stderr.log','verify-status.json','gui.log','gui.stderr.log','gui-status.json','initial-gui.log','initial-gui.stderr.log','initial-gui-status.json']:
 p=r/'build'/name2
 if p.is_file():files.append(p)
rows=[]
with zipfile.ZipFile(s/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(files):
  b=p.read_bytes();key=p.relative_to(r/'build').as_posix();z.writestr(key,b);rows.append(dict(path=key,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
(s/'transfer.json').write_text(json.dumps(dict(installer=name,sha256=id['sha256'],bytes=id['bytes'],evidence_sha256=sha(s/'evidence.zip'),evidence_bytes=(s/'evidence.zip').stat().st_size,files=rows),indent=2)+'\n')
print(json.dumps(dict(installer=name,sha256=id['sha256'],evidence_files=len(files))))
