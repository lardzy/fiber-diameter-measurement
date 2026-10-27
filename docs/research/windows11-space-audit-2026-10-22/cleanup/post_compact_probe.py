"""Read-only boot and retained-environment smoke test after disk compaction."""
import json
from vm_cleanup import ps,ROOT
code=r"""
$os=Get-CimInstance Win32_OperatingSystem
$d=Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
$checks=@{}
foreach($p in @('C:\Users\lishuyang\Downloads\fdm-native-validation-installed-20261013','C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe','C:\tmp\dsx-evidence-collector-20261022')){$checks[$p]=(Test-Path -LiteralPath $p)}
$absent=@{}
foreach($p in @('C:\tmp\windows11-space-audit-20261022','C:\tmp\fdm-space-cleanup-20261022')){$absent[$p]=!(Test-Path -LiteralPath $p)}
$py='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$out=& $py -B -c 'import json,sys,numpy; from PySide6 import QtCore; print(json.dumps(dict(python=sys.version.split()[0],numpy=numpy.__version__,qt=QtCore.qVersion())))' 2>&1
$exit=$LASTEXITCODE
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');os=$os.Caption;build=$os.BuildNumber;boot_utc=$os.LastBootUpTime.ToUniversalTime().ToString('o');free_bytes=$d.FreeSpace;size_bytes=$d.Size;retained_paths=$checks;scratch_absent=$absent;python_exit=$exit;python_output=($out -join "`n")}|ConvertTo-Json -Depth 6 -Compress
"""
r=json.loads(ps(code));r['status']='passed' if r['python_exit']==0 and all(r['retained_paths'].values()) and all(r['scratch_absent'].values()) else 'failed'
(ROOT/'post-compact-boot-check.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(r,ensure_ascii=False))
if r['status']!='passed':raise SystemExit(1)
