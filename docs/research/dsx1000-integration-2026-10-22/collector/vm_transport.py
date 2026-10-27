"""Transport only this collector's files to/from its isolated Windows build directory."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
VM = 'Windows 11'
DEST = r'C:\tmp\dsx-evidence-collector-20261022'

def ps(code, timeout=60):
    pre = "$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
    encoded = base64.b64encode((pre + code).encode('utf-16le')).decode()
    for attempt in range(5):
        proc = subprocess.run(['prlctl','exec',VM,'--current-user','powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-EncodedCommand',encoded], capture_output=True, timeout=timeout)
        out = proc.stdout.decode('utf-8', 'replace'); err = proc.stderr.decode('utf-8', 'replace')
        if not proc.returncode:return out.strip()
        if not ('PrlJob_Get' in err and 'Invalid argument' in err) or attempt == 4:raise RuntimeError(out + err)
        time.sleep(.3) # Only this task's idempotent file transport and read-only queries use ps().

def inspect():
    return ps("[ordered]@{os=[Environment]::OSVersion.VersionString;csc64=(Test-Path ($env:WINDIR+'\\Microsoft.NET\\Framework64\\v4.0.30319\\csc.exe'));csc32=(Test-Path ($env:WINDIR+'\\Microsoft.NET\\Framework\\v4.0.30319\\csc.exe'));refs=(Test-Path 'C:\\Program Files (x86)\\Reference Assemblies\\Microsoft\\Framework\\.NETFramework\\v4.8');runtime=[Environment]::Version.ToString();is64=[Environment]::Is64BitProcess} | ConvertTo-Json")

def push():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(ROOT.rglob('*')):
            if path.is_file() and path.suffix in ('.cs','.manifest','.config','.json','.ps1') and not {'validation','release','dist'}.intersection(path.relative_to(ROOT).parts):
                z.writestr(str(path.relative_to(ROOT)).replace('\\','/'),path.read_bytes())
    raw=buf.getvalue(); chunks=base64.b64encode(raw).decode()
    ps("[IO.Directory]::CreateDirectory('"+DEST+"')|Out-Null")
    for start in range(0,len(chunks),500):
        code="[IO.File]::WriteAllText('"+DEST+"\\chunk-"+str(start//500)+".b64','"+chunks[start:start+500]+"')"
        for attempt in range(3):
            try: ps(code);break
            except RuntimeError:
                if attempt==2:raise
                time.sleep(.3) # Retrying overwrite of our single chunk is idempotent.
    count=(len(chunks)+499)//500
    result=ps("$p='"+DEST+"';$s='';0.."+str(count-1)+"|ForEach-Object{$s += [IO.File]::ReadAllText($p+'\\chunk-'+$_+'.b64')};[IO.File]::WriteAllBytes($p+'\\input.zip',[Convert]::FromBase64String($s));$h=(Get-FileHash ($p+'\\input.zip')).Hash.ToLowerInvariant();if($h -ne '"+hashlib.sha256(raw).hexdigest()+"'){throw 'hash mismatch'};Expand-Archive -LiteralPath ($p+'\\input.zip') -DestinationPath $p -Force;$h")
    return {'uploaded_bytes':len(raw),'sha256':result}

def run(name):
    if name not in ('build.ps1','validate.ps1'): raise ValueError('Only collector build/validation scripts may run')
    proc=subprocess.run(['prlctl','exec',VM,'--current-user','powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',DEST+'\\'+name],capture_output=True,timeout=240)
    out=proc.stdout.decode('utf-8','replace'); err=proc.stderr.decode('utf-8','replace')
    print(out,flush=True)
    if proc.returncode: raise RuntimeError(err or ('Exit '+str(proc.returncode)))
    return {'script':name,'exit_code':proc.returncode}

def pull(name):
    if name not in ('result.zip',): raise ValueError('Only result.zip may be retrieved')
    remote=DEST+'\\'+name
    info=json.loads(ps("Get-Item -LiteralPath '"+remote+"' | Select-Object Length | ConvertTo-Json"))
    expected=ps("(Get-FileHash -LiteralPath '"+remote+"').Hash.ToLowerInvariant()")
    partial=ROOT/'validation'/('transfer-'+expected+'.part');partial.parent.mkdir(exist_ok=True)
    result=bytearray(partial.read_bytes() if partial.exists() else b'')
    for pos in range(len(result),info['Length'],24000):
        data=ps("$f=[IO.File]::OpenRead('"+remote+"');try{$f.Position="+str(pos)+";$b=New-Object byte[] 24000;$n=$f.Read($b,0,$b.Length);[Convert]::ToBase64String($b,0,$n)}finally{$f.Dispose()}")
        result.extend(base64.b64decode(data))
        partial.write_bytes(result)
    if hashlib.sha256(result).hexdigest()!=expected:raise RuntimeError('Download hash mismatch')
    path=ROOT/'validation'/'windows-result.zip';path.parent.mkdir(exist_ok=True);path.write_bytes(result)
    partial.unlink()
    with zipfile.ZipFile(io.BytesIO(result)) as z:
        for member in z.infolist():
            target=(path.parent/member.filename.replace('\\','/')).resolve()
            if path.parent.resolve() not in target.parents:raise RuntimeError('Invalid archive path')
            if member.is_dir() or member.filename.endswith('\\'):target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(z.read(member))
    return {'downloaded_bytes':len(result),'sha256':expected,'path':str(path)}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inspect','push','run','pull']);parser.add_argument('name',nargs='?');a=parser.parse_args()
    value={'inspect':inspect,'push':push,'run':lambda:run(a.name),'pull':lambda:pull(a.name)}[a.action]()
    print(json.dumps(value,ensure_ascii=False))
