"""Verify preserved content and every original-path mapping on the host."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'plan.json').read_text(encoding='utf-8-sig'))
mappings=json.loads((ROOT/'archive-mappings.json').read_text(encoding='utf-8-sig'))
sha=lambda p: hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert sha(ROOT/'archive-mappings.json')==plan['archive_mappings_sha256']
assert sha(ROOT/'preserved.zip')==plan['archive_sha256']==mappings['archive_sha256']
files={x['FullPath'].casefold():x for t in plan['targets'] for x in t['files']}
expected={x['path'].casefold() for x in plan['archive_reasons']}
seen=set()
with zipfile.ZipFile(ROOT/'preserved.zip') as z:
 manifest=json.loads(z.read('manifest.json'))
 content={x['ArchiveEntry']:x for x in manifest['Files']}
 assert len(content)==len(manifest['Files'])
 for entry,f in content.items():
  with z.open(entry) as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
  assert actual==f['Sha256'] and z.getinfo(entry).file_size==f['Length'],entry
 for m in mappings['entries']:
  path=m['original_full_path'].casefold()
  assert path not in seen and path in expected and path in files,path
  seen.add(path);f=files[path];representative=content[m['archive_entry']]
  assert m['sha256']==f['Sha256']==representative['Sha256'],path
  assert m['length']==f['Length']==representative['Length'],path
 assert seen==expected
result={'status':'passed','original_paths':len(seen),'unique_contents':len(content),'archive_bytes':(ROOT/'preserved.zip').stat().st_size,'archive_sha256':sha(ROOT/'preserved.zip'),'mapping_sha256':sha(ROOT/'archive-mappings.json'),'all_unique_payloads_sha256_verified':True,'all_original_paths_reconstructable':True}
(ROOT/'archive-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
