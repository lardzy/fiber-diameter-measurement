"""Package local static evidence and checksums; never executes original binaries."""
from pathlib import Path
import gzip
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    # REA exports are compressed only after verifying exact decompressed bytes.
    for f in sorted(ROOT.glob("*evidence.json")):
        raw=f.read_bytes(); target=f.with_suffix(f.suffix+".gz")
        target.write_bytes(gzip.compress(raw,mtime=0))
        assert gzip.decompress(target.read_bytes())==raw
        f.unlink()
    records={}
    for f in sorted(ROOT.glob("*evidence.json.gz")):
        for e in json.loads(gzip.decompress(f.read_bytes()))["records"]:
            records[e["evidence_id"]]=(f.name,e)
    index=[]
    for eid,(bundle,e) in records.items():
        s=e.get("subject",{}); source=s.get("local_path","")
        tag=s.get("name","unknown")
        tag=("APP_" if "/DSX-BSW-APP/" in source else "CTRL_" if "/DSX-BSW-CTRL/" in source else "")+tag
        if "/OLSStitching/" in source: tag="OLS_"+tag
        if e["operation"] in ("procedure_pseudo_code","procedure_assembly") and isinstance(e.get("normalized_result"),str):
            suffix="asm" if e["operation"]=="procedure_assembly" else "c"
            path=ROOT/"decompiled"/f"{tag}_{e['parameters']['procedure']}.{suffix}.txt"
            path.write_text(f"Evidence: {eid}\nSubject SHA256: {s.get('digest',{}).get('sha256')}\nBundle: {bundle}\n\n"+e["normalized_result"])
        index.append({"id":eid,"bundle":bundle,"subject":s,"operation":e["operation"],"parameters":e["parameters"]})
    # Earlier extracts without an engine tag duplicate the same retained evidence.
    for f in (ROOT/"decompiled").glob("*.txt"):
        if not f.name.startswith(("APP_","CTRL_","OLS_")): f.unlink()
    (ROOT/"evidence-index.json").write_text(json.dumps(index,ensure_ascii=False,indent=2)+"\n")
    rows=["# REA 证据定位表","","证据是静态反编译/IL，不是原厂源代码，也不是实机结果。原始记录见对应 gzip 包；可读片段见 decompiled/。","","| 文件 | 函数 / 操作 | 证据 ID | 包 |","|---|---|---|---|"]
    for e in index:
        if e["operation"] not in ("procedure_pseudo_code","procedure_assembly","inspect_managed_members","inspect_managed_native_boundaries"):continue
        s=e["subject"];path=s.get("local_path","");name=s.get("name","")
        tag=("OLS/" if "/OLSStitching/" in path else "")+("APP/" if "/DSX-BSW-APP/" in path else "CTRL/" if "/DSX-BSW-CTRL/" in path else "")+name
        rows.append(f"| {tag} | `{e['parameters'].get('procedure',e['operation'])}` | `{e['id']}` | {e['bundle']} |")
    (ROOT/"evidence-map.md").write_text("\n".join(rows)+"\n")
    entries=sorted(ROOT.glob("*evidence.json.gz"))+[ROOT/"evidence-index.json",ROOT/"evidence-map.md",ROOT/"identity-manifest.json"]
    with zipfile.ZipFile(ROOT/"rea-evidence.zip","w",zipfile.ZIP_DEFLATED) as z:
        for path in entries:
            info=zipfile.ZipInfo(path.name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,path.read_bytes())
    sums=[]
    for f in sorted(ROOT.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts or f.name in ("SHA256SUMS.txt",".DS_Store"):continue
        sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(ROOT)}")
    (ROOT/"SHA256SUMS.txt").write_text("\n".join(sums)+"\n")
    print(f"{len(records)} unique REA evidence; {len(sums)} deliverable files; checksums refreshed")


if __name__=="__main__":main()
