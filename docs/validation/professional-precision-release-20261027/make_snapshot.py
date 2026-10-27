from pathlib import Path
import hashlib,json,subprocess,zipfile,importlib.util
root=Path.cwd();out=root/'build/professional-precision-release-20261027';rows=[]
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
names=subprocess.check_output(['git','ls-files','-z']).decode().split('\0')
paths={n:root/n for n in names if n and not n.startswith(('docs/','.git/','.github/','build/','dist/','runtime/')) and (root/n).is_file() and not (root/n).is_symlink()}
for group in ['camera','content-templates','area-infer','segment-anything']:
 for p in (root/'runtime'/group).rglob('*'):
  if p.is_file() and not p.is_symlink() and not any(x.startswith('.') or x=='__pycache__' for x in p.relative_to(root).parts) and p.suffix!='.pyc':paths[p.relative_to(root).as_posix()]=p
vendor=Path(importlib.util.find_spec('pybind11').origin).parent
for p in vendor.rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':paths['validation_vendor/pybind11/'+p.relative_to(vendor).as_posix()]=p
with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for name,p in sorted(paths.items()):
  b=p.read_bytes();rows.append(dict(path=name,size=len(b),sha256=hashlib.sha256(b).hexdigest()));z.writestr(name,b)
 m=dict(format=1,source=str(root),head=head,files=rows,excluded_runtime=['area-models'],scope='Committed tracked source plus internal runtime assets and existing pybind11 headers; docs excluded')
 z.writestr('.fdm_snapshot.json',json.dumps(m,ensure_ascii=False,indent=2))
(out/'source-manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
b=(out/'source.zip').read_bytes();print(json.dumps(dict(head=head,files=len(rows),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))
