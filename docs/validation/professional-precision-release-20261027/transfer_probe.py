"""Transfer only external validation helpers, retaining every probe revision."""
from pathlib import Path
import base64,hashlib,json,sys
from datetime import datetime,timezone
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts/validation'))
import fifth_round_remote as remote
remote.WINDOWS_ROOT=r'C:\tmp\fdm-cu-live-professional-20261027'
OUT=ROOT/'docs/validation/professional-precision-release-20261027'
files=['scripts/validation/probe_professional_precision_frozen.ps1','scripts/validation/validate_professional_precision_frozen.py','docs/validation/professional-precision-release-20261027/upload_results.py']
code="$ErrorActionPreference='Stop'; $r='C:\\tmp\\fdm-professional-precision-release-20261027'\n"
rows=[]
for name in files:
 data=(ROOT/name).read_bytes(); target=name if name.startswith('scripts/') else 'build/upload_results.py'
 digest=hashlib.sha256(data).hexdigest();encoded=base64.b64encode(data).decode()
 code+=f"$p=Join-Path $r '{target}'; $archive=Join-Path $r 'build/frozen-gui/probe-revisions'; [void](New-Item -ItemType Directory -Force $archive); if(Test-Path $p){{$old=(Get-FileHash $p).Hash.ToLowerInvariant();Copy-Item $p (Join-Path $archive ($old+'-'+[IO.Path]::GetFileName($p))) -Force}}\n"
 code+=f"[IO.File]::WriteAllBytes($p,[Convert]::FromBase64String('{encoded}')); if((Get-FileHash $p).Hash.ToLowerInvariant() -ne '{digest}'){{throw 'Probe hash mismatch'}}; Copy-Item $p (Join-Path $archive ('{digest}-'+[IO.Path]::GetFileName($p))) -Force\n"
 rows.append(dict(path=target,bytes=len(data),sha256=digest))
code+="$tokens=$null;$errors=$null;[void][Management.Automation.Language.Parser]::ParseFile((Join-Path $r 'scripts/validation/probe_professional_precision_frozen.ps1'),[ref]$tokens,[ref]$errors);if($errors.Count){throw ($errors|Out-String)}\n"
result=remote.run_powershell('Windows 11',code,timeout=120)
folder=OUT/'probe-transfers';folder.mkdir(exist_ok=True)
(folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
print('Transferred and parsed external validation helpers; previous revisions retained')
