"""Read-only selected ECMA-335 attribute decoder; never loads the assembly."""
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
record = json.loads((ROOT / 'rdk-attribute-rows.json').read_text())
data = Path(record['artifact']['path']).read_bytes()
assert hashlib.sha256(data).hexdigest() == record['artifact']['sha256']
u16 = lambda p: struct.unpack_from('<H', data, p)[0]
u32 = lambda p: struct.unpack_from('<I', data, p)[0]
pe = u32(0x3c)
opt = pe + 24
section = opt + u16(pe + 20)
sections = [(u32(section+i*40+12), max(u32(section+i*40+8), u32(section+i*40+16)), u32(section+i*40+20)) for i in range(u16(pe+6))]
def off(rva):
    return next(raw + rva - va for va, size, raw in sections if va <= rva < va + size)
dd = opt + (96 if u16(opt) == 0x10b else 112)
cli = off(u32(dd + 14*8))
meta = off(u32(cli + 8))
pos = meta + 16 + ((u32(meta+12)+3)//4*4)
count = u16(pos+2)
pos += 4
streams = {}
for _ in range(count):
    rel, size = u32(pos), u32(pos+4)
    end = data.index(0, pos+8)
    name = data[pos+8:end].decode('ascii')
    streams[name] = (meta+rel, size)
    pos += 8 + ((end-(pos+8)+1+3)//4*4)
assert not data[streams['#~'][0]+6] & 4, 'this recorded small assembly uses 2-byte blob indexes'
names = {f['token']: f['name'] for f in record['fields']}
keys = ['command_type', 'parameter_count', 'remote_mode_only', 'lens_setting_required', 'cannot_execute_while_live_stopped', 'cannot_execute_during_shooting']
output = []
for a in record['attributes']:
    if not (a['type_name'] or '').endswith('.RdkCommandAttribute'):
        continue
    idx = u16(a['row_offset']+4)
    start = streams['#Blob'][0]+idx
    length = data[start]
    assert length < 128, 'selected attribute blobs use one-byte length'
    blob = data[start+1:start+1+length]
    assert hashlib.sha256(blob).hexdigest() == a['value_sha256']
    assert length == 16 and blob[:2] == b'\x01\x00' and blob[-2:] == b'\x00\x00'
    values = list(struct.unpack_from('<iiBBBB', blob, 2))
    values[2:] = [bool(v) for v in values[2:]]
    name = names[a['parent_token']]
    output.append({'name': name, 'command_tag': name+('=' if values[0] == 0 else '?'), 'field_token': a['parent_token'], 'attribute_token': a['token'], 'attribute_file_offset': start+1, 'attribute_sha256': a['value_sha256'], **dict(zip(keys, values))})
print(json.dumps({'authority': 'analyst-derived from verified static bytes', 'parent_evidence_id': record['evidence_id'], 'artifact': record['artifact'], 'decode': '<prolog:u16><command_type:i32><parameter_count:i32><four bool:u8><named_count:u16>', 'commands': output}, ensure_ascii=False, indent=2))
