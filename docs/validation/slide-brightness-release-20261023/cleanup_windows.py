from pathlib import Path
import hashlib,json,shutil
ROOT=Path(r'C:\tmp\fdm-brightness-release-20261023')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
identity=json.loads((ROOT/'build/package/installer-identity.json').read_text(encoding='utf-8'))
assert identity['passed'] and sha(Path(identity['installer']))==identity['sha256']
rows=[];targets=[]
archives={
 r'C:\tmp\fdm-artifact-997f7787dfcf45dc828e10e4fb743528.zip':'e1128827e4c650be38f1d6f4a8e082ab6ca932e845274ddc73195ad730a61126',
 r'C:\tmp\fdm-artifact-dc47187d3d06498aa348ce45bd7aa7eb.zip':'763578595883e3c903bce789b6c975f113b013edc3eb9a92da4086ae038699eb'}
for token,digest in archives.items():
 p=Path(token);assert not p.is_symlink() and sha(p)==digest
 rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest,reason='Old single-installer transfer archive; historical installer identity confirmed'));targets.append(p)
roots=['fdm-cu-live-20261015','fdm-cu-live-eighth-round-20261021','fdm-cu-live-explicit-gestures-20261021','fdm-cu-live-fifth-round-20261020','fdm-cu-live-fourth-round-20261019','fdm-cu-live-microview-return-20261015','fdm-cu-live-mixed-frame-fix-20261021','fdm-cu-live-opt4k-20261015','fdm-cu-live-rebuild-20261015','fdm-cu-live-second-round-20261017','fdm-cu-live-seventh-round-20261021','fdm-cu-live-sixth-round-20261020','fdm-cu-live-third-round-20261019','fdm-cu-live-ui-redesign-20261016']
for name in roots:
 root=Path('C:/tmp')/name;p=root/'build/lib.win-amd64-cpython-312'
 assert not any(x.is_symlink() for x in [p,*p.rglob('*')])
 files=[x for x in p.rglob('*') if x.is_file()];assert len(files)==2 and all(x.suffix=='.pyd' for x in files)
 proofs=[]
 for f in files:
  preserved=root/'src'/f.relative_to(p);digest=sha(f);assert sha(preserved)==digest
  proofs.append(dict(path=str(f),preserved=str(preserved),sha256=digest,bytes=f.stat().st_size))
 rows.append(dict(path=str(p),bytes=sum(x.stat().st_size for x in files),duplicates=proofs,reason='Native build duplicates byte-identical to preserved source-tree binaries'));targets.append(p)
out=ROOT/'build/windows-old-cleanup.json'
report=dict(deleted=False,total_bytes=sum(x['bytes'] for x in rows),rows=rows,new_installer_sha256=identity['sha256'])
out.write_text(json.dumps(report,indent=2)+'\n')
for p in targets:
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
assert all(not p.exists() for p in targets)
report['deleted']=True;out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(deleted_entries=len(rows),deleted_bytes=report['total_bytes'])))
