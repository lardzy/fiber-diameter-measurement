"""Preserve the audit leaf index locally before removing this task's guest scratch."""
from pathlib import Path
import json,hashlib,zipfile
from vm_cleanup import ps,ROOT
source=r"C:\tmp\windows11-space-audit-20261022\out\scan\all-files-index-local-only.csv"
shared=Path.home()/"Downloads"/"fdm-space-cleanup-transfer-20261022"
part=shared/"audit-full-index.csv.part"
remote=r"\\Mac\Home\Downloads\fdm-space-cleanup-transfer-20261022\audit-full-index.csv.part"
if part.exists():raise RuntimeError("Inspect existing transfer before retry")
meta=json.loads(ps("$p='"+source+"';$f=Get-Item -LiteralPath $p;Copy-Item -LiteralPath $p -Destination '"+remote+"';@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $p).Hash.ToLowerInvariant()}|ConvertTo-Json",timeout=300))
with part.open("rb") as f:actual=hashlib.file_digest(f,"sha256").hexdigest()
assert part.stat().st_size==meta["bytes"] and actual==meta["sha256"]
target=ROOT/"audit-full-index.zip"
with zipfile.ZipFile(target,"x",zipfile.ZIP_DEFLATED) as z:z.write(part,"all-files-index.csv")
with zipfile.ZipFile(target) as z:
 assert z.testzip() is None
 with z.open("all-files-index.csv") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==actual
with target.open("rb") as f:archive_hash=hashlib.file_digest(f,"sha256").hexdigest()
receipt={"source":source,"source_bytes":meta["bytes"],"source_sha256":actual,"archive":target.name,"archive_bytes":target.stat().st_size,"archive_sha256":archive_hash,"status":"verified"}
(ROOT/"audit-index-preserved.json").write_text(json.dumps(receipt,indent=2)+"\n")
part.unlink()
print(json.dumps(receipt))
