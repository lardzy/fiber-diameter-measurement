"""Compress own REA bundles, prove supersets, and index immutable source identities."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SOURCES = Path('/Volumes/LARD/2.中转/OLYMPUS/OLYMPUS/DSX-BSW/DSX-BSW-CTRL')
CANONICAL = ['phase-correlation-evidence.json','solver-evidence.json']


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    records = {}
    sources = {}
    bundles=[]
    for name in CANONICAL:
        plain = ROOT/name
        compressed=ROOT/(name+'.gz')
        b = plain.read_bytes() if plain.exists() else gzip.decompress(compressed.read_bytes())
        d=json.loads(b)
        for r in d['records']:
            records[r['evidence_id']]={"evidence_id":r['evidence_id'],"bundle":compressed.name,
                "operation":r['operation'],"parameters":r['parameters'],"subject":r['subject'],
                "confidence":r['confidence'],"limitations":r.get('limitations',[])}
            s=r['subject']
            sources[s['local_path']]=s
        compressed.write_bytes(gzip.compress(b,compresslevel=9,mtime=0))
        assert gzip.decompress(compressed.read_bytes())==b
        bundles.append({"file":compressed.name,"records":len(d['records']),
                        "uncompressed_sha256":sha(b),"sha256":sha(compressed.read_bytes())})
    # Only remove older own exports when every evidence ID survives in canonical packages.
    for name in ['managed-evidence.json','display-evidence.json']:
        p=ROOT/name
        if p.exists():
            old=json.loads(p.read_bytes())
            assert set(r['evidence_id'] for r in old['records']) <= records.keys()
            p.unlink()
    for name in CANONICAL:
        (ROOT/name).unlink(missing_ok=True)
    identities=[]
    for path,s in sorted(sources.items()):
        p=Path(path)
        b=p.read_bytes()
        assert sha(b)==s['digest']['sha256'],path
        identities.append({**s,"size_bytes":len(b),"source_rehashed":True,
                           "architecture_note":"PE machine value alone does not establish managed process bitness"})
    ini=SOURCES/'OlyNmv.ini'
    b=ini.read_bytes()
    (ROOT/'OlyNmv.ini.observed.txt').write_bytes(b)
    identities.append({"name":ini.name,"local_path":str(ini),"sha256":sha(b),
                       "size_bytes":len(b),"content_utf8":b.decode('utf-8'),"source_rehashed":True})
    (ROOT/'source-identities.json').write_text(json.dumps(identities,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'evidence-index.json').write_text(json.dumps({"bundles":bundles,"unique_records":len(records),
                                "records":list(records.values())},ensure_ascii=False,indent=2)+'\n')
    manifest=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts:
            manifest.append(f'{sha(p.read_bytes())}  {p.relative_to(ROOT)}')
    (ROOT/'SHA256SUMS').write_text('\n'.join(manifest)+'\n')
    print(json.dumps({"bundles":bundles,"unique_records":len(records),"sources":len(identities)}))


if __name__=='__main__':
    main()
