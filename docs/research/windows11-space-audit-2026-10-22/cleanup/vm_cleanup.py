"""Scoped transport for user-authorized Windows space cleanup.
No mutation is automatically retried. Guest jobs have explicit receipts.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent
DEST=r'C:\tmp\fdm-space-cleanup-20261022'

def ps(code,timeout=90):
    code="$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"+code
    p=subprocess.run(['/Applications/Parallels Desktop.app/Contents/MacOS/prlctl','exec','Windows 11','powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-EncodedCommand',base64.b64encode(code.encode('utf-16le')).decode()],capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stdout.decode('utf-8','replace')+p.stderr.decode('utf-8','replace'))
    return p.stdout.decode('utf-8-sig').strip()

def push():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'tools').iterdir()):
            if p.suffix in ('.cs','.ps1','.json'):
                payload=p.read_bytes()
                if p.suffix=='.ps1':payload=b'\xef\xbb\xbf'+payload.removeprefix(b'\xef\xbb\xbf')
                z.writestr(p.name,payload)
    raw=buf.getvalue()
    shared=Path.home()/'Downloads'/'fdm-space-cleanup-transfer-20261022'
    shared.mkdir(exist_ok=True)
    (shared/'host-marker.txt').write_text('fdm-space-cleanup-20261022-verified-transfer')
    (shared/'input.zip').write_bytes(raw)
    remote_shared=r'\\Mac\Home\Downloads\fdm-space-cleanup-transfer-20261022'
    return ps("$p='"+DEST+"';$d='"+remote_shared+"';if([IO.File]::ReadAllText($d+'\\host-marker.txt') -ne 'fdm-space-cleanup-20261022-verified-transfer'){throw 'Shared folder identity mismatch'};[IO.Directory]::CreateDirectory($p)|Out-Null;Copy-Item -LiteralPath ($d+'\\input.zip') -Destination ($p+'\\input.zip') -Force;if((Get-FileHash ($p+'\\input.zip')).Hash.ToLowerInvariant() -ne '"+hashlib.sha256(raw).hexdigest()+"'){throw 'hash mismatch'};Expand-Archive -LiteralPath ($p+'\\input.zip') -DestinationPath $p -Force;'uploaded'")

def reset_failed_preflight():
    receipt=json.loads(status('preflight'))
    if receipt.get('status')!='error':raise RuntimeError('Only a failed preflight may be reset')
    dispatch=json.loads((ROOT/'preflight-dispatch.json').read_text())
    code="$p='"+DEST+"';if(Get-Process -Id "+str(dispatch['pid'])+" -ErrorAction SilentlyContinue){throw 'Preflight still running'};"
    code+="$a=Join-Path $p ('failed-preflight-'+[DateTime]::UtcNow.ToString('yyyyMMddHHmmss'));[IO.Directory]::CreateDirectory($a)|Out-Null;"
    code+="foreach($n in @('preflight-started.json','preflight-result.json','preflight.stdout.txt','preflight.stderr.txt','progress.json')){if([IO.File]::Exists((Join-Path $p $n))){Move-Item -LiteralPath (Join-Path $p $n) -Destination $a}};'Failed preflight receipts preserved'"
    (ROOT/'preflight-failure.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    return ps(code)

def start(stage):
    if stage not in ('preflight','delete','maintenance','verify','cache','cache-resume','finish-preflight'):raise ValueError(stage)
    script={'cache':'CleanBrowserCaches.ps1','cache-resume':'ResumeBrowserCaches.ps1','finish-preflight':'FinishPreflight.ps1'}.get(stage,'Cleanup.ps1')
    arguments='' if stage.startswith('cache') or stage=='finish-preflight' else ' -Stage '+stage
    code="$p='"+DEST+"';$flag=$p+'\\"+(stage+'-launch' if stage.startswith('cache') or stage=='finish-preflight' else stage)+"-started.json';if([IO.File]::Exists($flag)){throw 'Stage already started; inspect receipt'};"
    code+="[IO.File]::WriteAllText($flag,'{}');$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "+DEST+"\\"+script+arguments+"'};if($r.ReturnValue -ne 0){throw ('Dispatch failed '+$r.ReturnValue)};$j=[ordered]@{pid=$r.ProcessId;stage='"+stage+"';utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json;[IO.File]::WriteAllText($flag,$j);$j"
    out=ps(code);(ROOT/(stage+'-dispatch.json')).write_text(out+'\n');return out

def status(stage):
    if stage not in ('preflight','delete','maintenance','verify','cache','cache-resume','finish-preflight'):raise ValueError(stage)
    progress=stage+'-progress.json' if stage.startswith('cache') or stage=='finish-preflight' else 'progress.json'
    return ps("$p='"+DEST+"';if([IO.File]::Exists($p+'\\ "+stage+"-result.json'.Replace(' ',''))){[IO.File]::ReadAllText($p+'\\ "+stage+"-result.json'.Replace(' ',''))}elseif([IO.File]::Exists($p+'\\"+progress+"')){[IO.File]::ReadAllText($p+'\\"+progress+"')}else{'No receipt yet'}")

def pull(name):
    allowed=('preserved.zip','preflight-evidence.zip','plan.json','protected-before.json','protected-after.json','preflight-result.json','delete-result.json','maintenance-result.json','verify-result.json','maintenance.stdout.txt','maintenance.stderr.txt','system-before.txt','system-after.txt','dism-cleanup.txt','uv-prune.txt','npm-verify.txt','guest-before.json','guest-after.json','cache-result.json','cache-resume-result.json','finish-preflight-result.json','final-environment.json','final-evidence.zip')
    if name not in allowed:raise ValueError(name)
    remote=DEST+'\\'+name
    meta=json.loads(ps("$f=[IO.FileInfo]::new('"+remote+"');[ordered]@{length=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName).Hash}|ConvertTo-Json"))
    target=ROOT/name;part=ROOT/(name+'.part')
    if not part.exists():
        shared=Path.home()/'Downloads'/'fdm-space-cleanup-transfer-20261022'
        shared.mkdir(exist_ok=True)
        marker=shared/'host-marker.txt'
        marker.write_text('fdm-space-cleanup-20261022-verified-transfer')
        remote_shared=r'\\Mac\Home\Downloads\fdm-space-cleanup-transfer-20261022'
        ps("$d='"+remote_shared+"';if([IO.File]::ReadAllText($d+'\\host-marker.txt') -ne 'fdm-space-cleanup-20261022-verified-transfer'){throw 'Shared folder identity mismatch'};Copy-Item -LiteralPath '"+remote+"' -Destination ($d+'\\"+name+".part') -ErrorAction Stop",timeout=300)
        transferred=shared/(name+'.part')
        if transferred.stat().st_size!=meta['length'] or hashlib.sha256(transferred.read_bytes()).hexdigest()!=meta['sha256'].lower():raise RuntimeError('Shared transfer SHA mismatch')
        transferred.replace(part)
    offset=part.stat().st_size if part.exists() else 0
    with part.open('ab') as out:
        while offset<meta['length']:
            count=min(131072,meta['length']-offset)
            data=ps("$f=[IO.File]::OpenRead('"+remote+"');try{$f.Position="+str(offset)+";$b=New-Object byte[] "+str(count)+";$n=0;while($n -lt $b.Length){$r=$f.Read($b,$n,$b.Length-$n);if(!$r){throw 'short read'};$n+=$r};[Convert]::ToBase64String($b)}finally{$f.Dispose()}")
            raw=base64.b64decode(data,validate=True)
            if len(raw)!=count:raise RuntimeError('Truncated block; not appended')
            out.write(raw);out.flush();offset+=len(raw)
    actual=hashlib.sha256(part.read_bytes()).hexdigest()
    if actual!=meta['sha256'].lower():raise RuntimeError('Full-file SHA mismatch')
    part.replace(target)
    if name=='preserved.zip':
        with zipfile.ZipFile(target) as z:
            if z.testzip() is not None:raise RuntimeError('Archive CRC mismatch')
        ps("[IO.File]::WriteAllText('"+DEST+"\\host-archive-verified.sha256','"+actual+"')")
    if name=='preflight-evidence.zip':
        with zipfile.ZipFile(target) as z:
            if z.testzip() is not None:raise RuntimeError('Receipt archive CRC mismatch')
            by_name={p.rsplit('/',1)[-1].rsplit('\\',1)[-1]:p for p in z.namelist()}
            receipt=json.loads(z.read(by_name['preflight-result.json']))
            for file,key in [('plan.json','plan_sha256'),('protected-before.json','protected_sha256'),('archive-mappings.json','archive_mappings_sha256')]:
                payload=z.read(by_name[file])
                if hashlib.sha256(payload).hexdigest()!=receipt[key]:raise RuntimeError('Plan receipt mismatch')
                (ROOT/file).write_bytes(payload)
            (ROOT/'preflight-result.json').write_bytes(z.read(by_name['preflight-result.json']))
            (ROOT/'guest-before.json').write_bytes(z.read(by_name['guest-before.json']))
        ack=json.dumps({k:receipt[k] for k in ('plan_sha256','protected_sha256','archive_mappings_sha256')},separators=(',',':'))
        ps("[IO.File]::WriteAllText('"+DEST+"\\host-plan-verified.json','"+ack+"')")
    if name=='final-evidence.zip':
        with zipfile.ZipFile(target) as z:
            if z.testzip() is not None:raise RuntimeError('Final evidence CRC mismatch')
            verification=json.loads(z.read('verify-result.json'))
            if verification['status']!='passed':raise RuntimeError('Protected verification did not pass')
        ps("[IO.File]::WriteAllText('"+DEST+"\\host-final-evidence-verified.sha256','"+actual+"')")
    return {'file':name,'bytes':meta['length'],'sha256':actual}

def pack_final():
    return ps("$p='"+DEST+"';$v=[IO.File]::ReadAllText($p+'\\verify-result.json')|ConvertFrom-Json;if($v.status -ne 'passed'){throw 'Verification not passed'};$paths=@(Get-ChildItem -LiteralPath $p -Force|Where-Object{$_.Name -notin @('preserved.zip','input.zip','final-evidence.zip') -and $_.Name -notlike 'part-*.txt' -and $_.Name -notlike '*.tmp' -and $_.Name -notlike '*.incomplete.zip'}|ForEach-Object{$_.FullName});Compress-Archive -LiteralPath $paths -DestinationPath ($p+'\\final-evidence.zip');'Final evidence packed'",timeout=300)

def finalize_scratch():
    out=ps("$s=[IO.File]::ReadAllText('"+DEST+"\\FinalizeScratch.ps1');& ([ScriptBlock]::Create($s))",timeout=600)
    result=json.loads(out)
    (ROOT/'scratch-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return out

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['push','start','status','pull','probe','reset-failed-preflight','pull-maintenance','pack-final','finalize-scratch']);a.add_argument('value',nargs='?');args=a.parse_args()
    if args.action=='push':print(push())
    elif args.action=='reset-failed-preflight':print(reset_failed_preflight())
    elif args.action=='start':print(start(args.value))
    elif args.action=='status':print(status(args.value))
    elif args.action=='pull':print(json.dumps(pull(args.value)))
    elif args.action=='pack-final':print(pack_final())
    elif args.action=='finalize-scratch':print(finalize_scratch())
    elif args.action=='pull-maintenance':
        for n in ('maintenance-result.json','system-before.txt','system-after.txt','dism-cleanup.txt'):print(json.dumps(pull(n)))
    else:print(ps("[ordered]@{utc=[DateTime]::UtcNow.ToString('o');identity=[Security.Principal.WindowsIdentity]::GetCurrent().Name;free=(Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='C:'\").FreeSpace}|ConvertTo-Json"))
