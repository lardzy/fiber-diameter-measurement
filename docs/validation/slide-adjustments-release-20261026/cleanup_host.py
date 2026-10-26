"""One-time cleanup of this release's own temporary directories after archival."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path('/Users/lishuyang/PycharmProjects/fiber-diameter-measurement')
OUT=ROOT/'docs/validation/slide-adjustments-release-20261026'
SHARE=Path('/Users/lishuyang/Downloads/fdm-slide-adjustments-release-transfer-20261026')
BUILD=ROOT/'build/slide-adjustments-release-20261026'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
retained=json.loads((OUT/'host-retained-installer.json').read_text())
assert retained['verified'] and sha(Path(retained['path']))==retained['sha256']
transfer=json.loads((OUT/'transfer.json').read_text())
assert sha(SHARE/transfer['installer'])==retained['sha256']
assert sha(SHARE/'evidence.zip')==transfer['evidence_sha256']
assert json.loads((OUT/'windows-cleanup.json').read_text())['deleted'] is True
for f in transfer['files']:assert sha(OUT/f['path'])==f['sha256']
rows=[]
for root in (SHARE,BUILD):
 assert root.is_dir() and not root.is_symlink()
 entries=list(root.rglob('*'));assert all(not p.is_symlink() for p in entries)
 files=[p for p in entries if p.is_file()]
 rows.append({'path':str(root),'bytes':sum(p.stat().st_size for p in files),'files':len(files)})
report={'deleted':False,'rows':rows,'bytes':sum(r['bytes'] for r in rows),'retained_installer':retained,'reason':'Own transfer and rebuildable scratch directories; installer and evidence archived and verified'}
p=OUT/'host-cleanup.json';p.write_text(json.dumps(report,indent=2)+'\n')
for root in (SHARE,BUILD):shutil.rmtree(root)
assert not SHARE.exists() and not BUILD.exists()
report['deleted']=True;p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
