"""Delete only reviewed previous installers and reproducible build outputs."""
from pathlib import Path
import hashlib,json,os,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/validation/slide-brightness-release-20261023'
identity=json.loads((OUT/'package/installer-identity.json').read_text())
assert identity['passed'] is True
new=next((ROOT/'dist/internal/slide-brightness-20261023').glob('*.exe'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(new)==identity['sha256']
old={
 'dist/internal/optical-adaptive-20261021/FDM-0.4.10-optical-adaptive-daa32894eef7.exe':'ee7ec2619ce1252074789305f7ab4d739d5642b461cf08f93c2e89cc99ae9c8c',
 'dist/internal/explicit-manual-gestures-20261021/FDM-0.4.10-explicit-gestures-2527fbdb485c.exe':'fd631229a9e74b61cf654d4e96271d50f482e02d425c4c48f48fd5397a17c833'}
paths=[]
for name,digest in old.items():
 p=ROOT/name;assert p.is_file() and sha(p)==digest;paths.append(p)
for name in ['build/native-kernel-migration/installed-native','build/native-kernel-migration/installed-python','build/cu-live-optimization/uv-cache','build/ui-modernization-review/baseline/.pytest_cache']:
 p=ROOT/name
 if p.exists():paths.append(p)
for directory,dirs,files in os.walk(ROOT/'build',followlinks=False):
 dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
 for name in files:
  p=Path(directory)/name
  if p.suffix=='.pyc' and not p.is_symlink() and not any(p.is_relative_to(x) for x in paths):paths.append(p)
for group in ['build/cu-live-windows-final-r2','build/cu-live-windows-final-r3','build/cu-live-windows-final-r4']:
 paths+=list((ROOT/group).glob('transfer-*.zip'))
rows=[]
for p in paths:
 assert p.is_relative_to(ROOT) and not p.is_symlink()
 files=[p] if p.is_file() else [f for f in p.rglob('*') if f.is_file() and not f.is_symlink()]
 rows.append(dict(path=str(p.relative_to(ROOT)),bytes=sum(f.stat().st_size for f in files),files=len(files),sha256=sha(p) if p.is_file() else None))
receipt=dict(scope='Reviewed prior installers and rebuildable local build outputs only',new_installer=str(new.relative_to(ROOT)),new_installer_sha256=sha(new),rows=rows,total_bytes=sum(r['bytes'] for r in rows),deleted=False)
(OUT/'local-cleanup.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
for p in paths:
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
assert all(not p.exists() for p in paths)
assert sha(new)==identity['sha256']
receipt['deleted']=True
(OUT/'local-cleanup.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(deleted_entries=len(rows),deleted_bytes=receipt['total_bytes'],new_installer_verified=True)))
