from pathlib import Path
import gzip
import hashlib
import json
import struct

ROOT=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    plain=ROOT/'hamal-evidence.json';compressed=ROOT/'hamal-evidence.json.gz'
    b=plain.read_bytes() if plain.exists() else gzip.decompress(compressed.read_bytes())
    d=json.loads(b)
    ids={e['evidence_id'] for e in d['records']}
    old=ROOT/'managed-evidence.json'
    if old.exists():
        assert {e['evidence_id'] for e in json.loads(old.read_bytes())['records']}<=ids
        old.unlink()
    compressed.write_bytes(gzip.compress(b,compresslevel=9,mtime=0))
    assert gzip.decompress(compressed.read_bytes())==b
    plain.unlink(missing_ok=True)
    sources={};records=[];tables={}
    table_layout={'0x180010600':'h','0x180011680':'i','0x180011920':'i','0x180011bc0':'i','0x180014800':'i'}
    for e in d['records']:
        s=e['subject'];sources[s['local_path']]=s
        records.append({k:e[k] for k in ['evidence_id','operation','parameters','subject','limitations']})
        if e['operation']=='procedure_pseudo_code':
            (ROOT/('Hamal.'+e['parameters']['procedure']+'.pseudo.c')).write_text('// '+e['evidence_id']+'\n'+str(e['normalized_result']))
        if e['operation']=='read_bytes' and e['parameters']['address'] in table_layout:
            a=e['parameters']['address'];data=bytes.fromhex(e['normalized_result']['bytes_hex']);fmt=table_layout[a]
            tables[a]={'evidence_id':e['evidence_id'],'values':struct.unpack('<'+str(len(data)//struct.calcsize(fmt))+fmt,data)}
    identities=[]
    for p,s in sorted(sources.items()):
        data=Path(p).read_bytes();assert sha(data)==s['digest']['sha256']
        identities.append({**s,'size_bytes':len(data),'source_rehashed':True})
    (ROOT/'source-identities.json').write_text(json.dumps(identities,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'evidence-index.json').write_text(json.dumps({'bundle':compressed.name,'records':records,
        'count':len(records),'uncompressed_sha256':sha(b),'sha256':sha(compressed.read_bytes())},ensure_ascii=False,indent=2)+'\n')
    (ROOT/'native-tables.json').write_text(json.dumps(tables,indent=2)+'\n')
    print({'evidence_count':len(records),'sources':len(identities),'compressed_bytes':compressed.stat().st_size})


if __name__=='__main__':main()
