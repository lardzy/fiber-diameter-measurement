"""Read REA CIL evidence and print selected build-local anchors; never load DLLs."""
import argparse
import json
import gzip
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('assembly')
parser.add_argument('names', nargs='*')
parser.add_argument('--list', action='store_true')
args = parser.parse_args()
p = Path(__file__).parent / 'raw' / (args.assembly + '.dll.members.json')
o = json.loads(p.read_text() if p.exists() else gzip.open(str(p) + '.gz', 'rt').read())
n = o['normalized_result']
lookup = {}
for col in ['methods', 'fields', 'member_refs', 'types']:
    for v in n[col]:
        lookup[v['token']] = v.get('full_name') or ((v.get('declaring_type') or v.get('parent_name') or '') + '.' + v.get('name', ''))
print('EVIDENCE', o['evidence_id'], o['subject'])
for m in n['methods']:
    if args.names and not any(q in m['name'] or q in m['declaring_type'] for q in args.names):
        continue
    if args.list:
        print(m['token'], m['declaring_type'], m['name'])
        continue
    print('\nMETHOD', m['token'], m['declaring_type'], m['name'], 'flags', m['flags'], 'signature', m['signature'])
    b = m.get('body') or {}
    print('BODY', {k: v for k, v in b.items() if k not in ('anchors', 'opcode_counts')})
    for a in b.get('anchors', []):
        print(f" IL_{a['il_offset']:04x}", a['opcode'], a.get('operand'), lookup.get(a.get('operand'), ''))

