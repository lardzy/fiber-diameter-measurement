"""Read-only guest inspection and isolated audit artifacts.

The explicitly named stop-own-orphan action mitigates this audit's recorded
timed-out process only, guarded by its PID and decoded command identity.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=r'C:\tmp\windows11-space-audit-20261022'

def ps(code,timeout=90):
    prefix="$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
    encoded=base64.b64encode((prefix+code).encode('utf-16le')).decode()
    for attempt in range(5):
        p=subprocess.run(['prlctl','exec','Windows 11','powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-EncodedCommand',encoded],capture_output=True,timeout=timeout)
        out=p.stdout.decode('utf-8','replace');err=p.stderr.decode('utf-8','replace')
        if p.returncode==0:return out.strip()
        if not ('PrlJob_Get' in err and 'Invalid argument' in err) or attempt==4:raise RuntimeError(out+err)
        time.sleep(.3)

def push():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT/'tools').iterdir():
            if p.suffix in ('.cs','.ps1'):z.writestr(p.name,p.read_bytes())
    raw=buffer.getvalue();data=base64.b64encode(raw).decode()
    ps("[IO.Directory]::CreateDirectory('"+DEST+"')|Out-Null")
    for pos in range(0,len(data),500):ps("[IO.File]::WriteAllText('"+DEST+"\\input-"+str(pos//500)+".b64','"+data[pos:pos+500]+"')")
    count=(len(data)+499)//500
    return ps("$p='"+DEST+"';$s='';0.."+str(count-1)+"|ForEach-Object{$s += [IO.File]::ReadAllText($p+'\\input-'+$_+'.b64')};[IO.File]::WriteAllBytes($p+'\\input.zip',[Convert]::FromBase64String($s));if((Get-FileHash ($p+'\\input.zip')).Hash.ToLowerInvariant() -ne '"+hashlib.sha256(raw).hexdigest()+"'){throw 'hash mismatch'};Expand-Archive -LiteralPath ($p+'\\input.zip') -DestinationPath $p -Force;'uploaded'")

def pull(name='results.zip'):
    if name!='results.zip':raise ValueError('Only audit results may be retrieved')
    remote=DEST+'\\'+name
    info=json.loads(ps("Get-Item -LiteralPath '"+remote+"'|Select-Object Length|ConvertTo-Json"))
    expected=ps("(Get-FileHash -LiteralPath '"+remote+"').Hash.ToLowerInvariant()")
    partial=ROOT/'guest'/('transfer-'+expected+'.part');partial.parent.mkdir(exist_ok=True)
    result=bytearray(partial.read_bytes() if partial.exists() else b'')
    for pos in range(len(result),info['Length'],24000):
        chunk=ps("$f=[IO.File]::OpenRead('"+remote+"');try{$f.Position="+str(pos)+";$b=New-Object byte[] 24000;$n=$f.Read($b,0,$b.Length);[Convert]::ToBase64String($b,0,$n)}finally{$f.Dispose()}")
        result.extend(base64.b64decode(chunk));partial.write_bytes(result)
    assert hashlib.sha256(result).hexdigest()==expected
    target=ROOT/'guest'/name;target.write_bytes(result);partial.unlink()
    with zipfile.ZipFile(io.BytesIO(result)) as z:
        for member in z.infolist():
            dest=(target.parent/member.filename.replace('\\','/')).resolve()
            if target.parent.resolve() not in dest.parents:raise ValueError('Archive path outside audit')
            if member.is_dir() or member.filename.endswith('\\'):dest.mkdir(parents=True,exist_ok=True)
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(member))
    return {'sha256':expected,'bytes':len(result),'path':str(target)}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['push','pull','inventory','scan','status','system-summary','final-info','delta-info','audit-processes','stop-own-orphan','candidate-identities']);a=p.parse_args()
    if a.action=='push':print(push())
    elif a.action=='pull':print(json.dumps(pull()))
    elif a.action=='candidate-identities':
        targets=json.loads((ROOT/'retention/guest-findings.json').read_text())['small_identity_reads_requested']['distribution_roots']
        code='$targets=@('+','.join("'"+t.replace("'","''")+"'" for t in targets)+');'+r"""
$rows=@();foreach($target in $targets){
 $files=@();$dirs=@();$skipped=@();$errors=@();$stack=[Collections.Generic.Stack[string]]::new()
 if([IO.Directory]::Exists($target)){$stack.Push($target)}elseif([IO.File]::Exists($target)){$files+=Get-Item -LiteralPath $target}else{$errors+='Target missing'}
 while($stack.Count){$dir=$stack.Pop();try{foreach($entry in [IO.Directory]::EnumerateFileSystemEntries($dir)){
  $a=[IO.File]::GetAttributes($entry);if(($a -band [IO.FileAttributes]::ReparsePoint) -ne 0){$skipped+=$entry;continue}
  if(($a -band [IO.FileAttributes]::Directory) -ne 0){$stack.Push($entry);if([IO.Path]::GetFileName($entry) -match '^(data|userdata|user_data|projects|outputs|config|configs|profiles|logs)$'){$dirs+=$entry}}
  elseif([IO.Path]::GetExtension($entry) -match '^\.(fdmproj|fdmslide|db|sqlite|json|exe)$'){$files += [IO.FileInfo]::new($entry)}
 }}catch{$errors+=$_.Exception.Message}}
 $manifests=@();foreach($f in $files|Where-Object{$_.Name -in @('release-manifest.json','candidate-manifest.json','source-manifest.json')}){
  $fields=@{};try{$m=[IO.File]::ReadAllText($f.FullName)|ConvertFrom-Json;foreach($key in @('version','app_version','git_commit','commit','source_commit','build_profile','profile','created_at','built_at')){if($m.$key -is [string]){$fields[$key]=([string]$m.$key).Substring(0,[Math]::Min(200,([string]$m.$key).Length))}}}catch{}
  $manifests += [ordered]@{path=$f.FullName;length=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName).Hash;fields=$fields}
 }
 $data=@($files|Where-Object{$_.Extension -in @('.fdmproj','.fdmslide','.db','.sqlite')}|Select-Object FullName,Length)
 $exe=@($files|Where-Object{$_.Extension -eq '.exe' -and ($_.DirectoryName -eq $target -or $_.FullName -eq $target)}|ForEach-Object{$v=[Diagnostics.FileVersionInfo]::GetVersionInfo($_.FullName);[ordered]@{path=$_.FullName;length=$_.Length;file_version=$v.FileVersion;product_version=$v.ProductVersion}})
 $rows += [ordered]@{target=$target;manifests=$manifests;top_exe=$exe;potential_project_databases=$data;named_data_config_dirs=$dirs;skipped_reparse=$skipped;errors=$errors}
}
$rows|ConvertTo-Json -Depth 7
"""
        # Long EncodedCommand arguments can fail at the Parallels transport layer.
        # Stage this bounded read-only script only in our own audit directory.
        payload=base64.b64encode(code.encode('utf-8')).decode()
        for pos in range(0,len(payload),500):
            ps("[IO.File]::WriteAllText('"+DEST+"\\candidate-part-"+str(pos//500)+".txt','"+payload[pos:pos+500]+"')")
        count=(len(payload)+499)//500
        prepare="$p='"+DEST+"';$s='';0.."+str(count-1)+"|ForEach-Object{$s += [IO.File]::ReadAllText($p+'\\candidate-part-'+$_+'.txt')};[IO.File]::WriteAllBytes($p+'\\candidate-identities.ps1',[Convert]::FromBase64String($s));if((Get-FileHash ($p+'\\candidate-identities.ps1')).Hash.ToLowerInvariant() -ne '"+hashlib.sha256(code.encode('utf-8')).hexdigest()+"'){throw 'hash mismatch'};& ($p+'\\candidate-identities.ps1')"
        raw=ps(prepare,timeout=180);(ROOT/'guest/candidate-identities.json').write_text(raw+'\n')
        rows=json.loads(raw);print(json.dumps({'targets':len(rows),'manifests':sum(len(r['manifests']) for r in rows),'potential_project_databases':sum(len(r['potential_project_databases']) for r in rows),'errors':[r for r in rows if r['errors']]},ensure_ascii=False))
    elif a.action=='stop-own-orphan':
        code=r"""
$p=Get-CimInstance Win32_Process -Filter 'ProcessId=7912'
if(!$p){'Already exited';exit}
if($p.Name -ne 'powershell.exe' -or $p.CommandLine -notmatch '(?i)-EncodedCommand\s+([^\s"]+)'){throw 'Ownership not proved'}
$decoded=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($matches[1]))
$sha=[Security.Cryptography.SHA256]::Create();$hash=[BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($decoded))).Replace('-','').ToLowerInvariant();$sha.Dispose()
if($hash -ne '6a91fb4cfcb1eea242f277fbf5bc6549a27f8aa8b8a5bdadb537cbec451958bf'){throw 'Command identity changed'}
Stop-Process -Id 7912 -Force
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');stopped_pid=7912;decoded_sha256=$hash;reason='Own timed-out audit summary process; raw command not collected'}|ConvertTo-Json
"""
        raw=ps(code);(ROOT/'guest/orphan-stopped.json').write_text(raw+'\n');print(raw)
    elif a.action=='audit-processes':
        code=r"""
$rows=@();foreach($p in @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'")) {
 $decoded='';if($p.CommandLine -match '(?i)-EncodedCommand\s+([^\s"]+)'){try{$decoded=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($matches[1]))}catch{}}
 $owned=$decoded.Contains('C:\tmp\windows11-space-audit-20261022')
 $sha=[Security.Cryptography.SHA256]::Create();$hash=[BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($decoded))).Replace('-','').ToLowerInvariant();$sha.Dispose()
 $rows += [ordered]@{pid=$p.ProcessId;parent_pid=$p.ParentProcessId;created=$p.CreationDate;contains_our_audit_root=$owned;is_scan=$decoded.Contains('RunScan.ps1');is_system_summary=($owned -and $decoded.Contains('component-store.txt') -and $decoded.Contains('ConvertTo-Json -Depth 8'));decoded_sha256=$hash;working_set=$p.WorkingSetSize}
}
$rows|ConvertTo-Json -Depth 4
"""
        raw=ps(code);(ROOT/'guest/audit-processes.json').write_text(raw+'\n');print(raw)
    elif a.action=='delta-info':
        code=r"""
$rootFiles=@([IO.Directory]::EnumerateFiles('C:\')|ForEach-Object{ $f=[IO.FileInfo]::new($_); [ordered]@{path=$f.FullName;length=$f.Length} })
$sizes=@();foreach($p in @('C:\Windows\SoftwareDistribution','C:\Windows\SystemTemp','C:\Windows\Temp','C:\ProgramData\Microsoft\Windows Defender','C:\ProgramData\Microsoft\Windows\DeliveryOptimization')) {
  $errors=@();$m=Get-ChildItem -LiteralPath $p -File -Recurse -Force -ErrorAction SilentlyContinue -ErrorVariable +errors|Measure-Object Length -Sum
  $sizes += [ordered]@{path=$p;length=$m.Sum;count=$m.Count;errors=@($errors|ForEach-Object{$_.Exception.Message})}
}
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');root_files=$rootFiles;sizes=$sizes;shadows=@(Get-CimInstance Win32_ShadowStorage|Select-Object UsedSpace,AllocatedSpace,MaxSpace);volumes=@(Get-CimInstance Win32_LogicalDisk|Select-Object DeviceID,Size,FreeSpace)}|ConvertTo-Json -Depth 6
"""
        raw=ps(code,timeout=180)
        (ROOT/'guest/delta-info.json').write_text(raw+'\n');print(raw)
    elif a.action=='final-info':
        code = r"""
$entries=@([IO.Directory]::EnumerateFileSystemEntries('C:\'))+@([IO.Directory]::EnumerateFiles('C:\Windows'))
$special=@(); foreach($path in @('C:\pagefile.sys','C:\hiberfil.sys','C:\swapfile.sys','C:\Windows\MEMORY.DMP','C:\Windows.old')) {
  $exists=$entries -contains $path
  $length=if($exists -and $path -ne 'C:\Windows.old'){[IO.FileInfo]::new($path).Length}else{$null}
  $special += [ordered]@{path=$path;exists=$exists;length=$length;method='directory_enumeration_and_FileInfo'}
}
$audit=Get-ChildItem -LiteralPath 'C:\tmp\windows11-space-audit-20261022' -File -Recurse -Force
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');special_files=$special;pagefiles=@(Get-CimInstance Win32_PageFileUsage|Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage);top_memory_processes=@(Get-Process|Sort-Object WorkingSet64 -Descending|Select-Object -First 8 Id,ProcessName,WorkingSet64,PrivateMemorySize64);volumes=@(Get-CimInstance Win32_LogicalDisk|Select-Object DeviceID,DriveType,Size,FreeSpace);audit_files=@($audit|Select-Object FullName,Length);diskfree=(& fsutil volume diskfree C:|Out-String)}|ConvertTo-Json -Depth 5
"""
        raw=ps(code,timeout=180)
        (ROOT/'guest/final-info.json').write_text(raw+'\n')
        data=json.loads(raw)
        print(json.dumps({k:v for k,v in data.items() if k!='audit_files'},ensure_ascii=False,indent=2))
        print(json.dumps({'audit_logical_bytes':sum(f['Length'] for f in data['audit_files']),'audit_files':len(data['audit_files'])}))
    elif a.action=='system-summary':
        # Reuse the verified archive locally. Never deeply serialize PowerShell
        # Get-Content objects, which carry provider metadata and caused runaway
        # memory use in the original guest-side summary attempt.
        source=ROOT/'guest/transfer'
        data={'environment':json.loads((source/'environment.json').read_text(encoding='utf-8-sig'))}
        for key,name in {'component':'component-store.txt','vss':'shadowstorage.txt','shadows':'shadows.txt','diskfree':'diskfree.txt','trim':'trim-state.txt'}.items():
            data[key]=(source/name).read_text(encoding='utf-8-sig')
        raw=json.dumps(data,ensure_ascii=False,indent=2)
        (ROOT/'guest/system-summary.json').write_text(raw+'\n')
        data=json.loads(raw)
        print(json.dumps({k:v for k,v in data.items() if k!='environment'},ensure_ascii=False,indent=2))
        print(json.dumps({'special':data['environment']['special_files'],'folders':data['environment']['user_shell_folders'],'pagefile':data['environment']['pagefiles']},ensure_ascii=False,indent=2))
    elif a.action in ('inventory','scan'):
        name='Inventory.ps1' if a.action=='inventory' else 'RunScan.ps1'
        print(ps("& '"+DEST+'\\'+name+"'",timeout=1800))
    else:
        print(ps("$p='"+DEST+"\\out\\scan\\progress.json';if([IO.File]::Exists($p)){$f=[IO.File]::Open($p,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]7);try{$s=[IO.StreamReader]::new($f);try{$s.ReadToEnd()}finally{$s.Dispose()}}finally{$f.Dispose()}}"))
