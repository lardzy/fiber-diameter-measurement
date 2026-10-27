"""Remove only this release's reproducible build copies after host verification."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path.cwd();OUT=ROOT/'docs/validation/professional-measurement-release-20261027'
sys.path.insert(0,str(ROOT/'scripts/validation'))
import fifth_round_remote as remote
remote.WINDOWS_ROOT=r'C:\tmp\fdm-cu-live-professional-20261027'
def sha(path):
 with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
retained=json.loads((OUT/'host-retained-installer.json').read_text())
path=Path(retained['path']);assert path.is_relative_to(ROOT/'dist/internal/professional-measurement-20261027')
assert path.stat().st_size==retained['bytes'] and sha(path)==retained['sha256']
transfer=json.loads((OUT/'transfer.json').read_text())
for row in transfer['files']:
 p=(OUT/row['path']).resolve();assert p.is_relative_to(OUT.resolve())
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
identity=json.loads((OUT/'package/installer-identity.json').read_text())
assert identity['passed'] and identity['sha256']==retained['sha256']
code=r"""$ErrorActionPreference='Stop'
$r='C:\tmp\fdm-professional-measurement-release-20261027'
$id=Get-Content ($r+'\build\package\installer-identity.json') -Raw -Encoding UTF8|ConvertFrom-Json
$gui=Get-Content ($r+'\build\gui-status.json') -Raw -Encoding UTF8|ConvertFrom-Json
if(-not $id.passed -or $id.source_git_baseline -ne 'cbfffe9c6a9c7b97fa51543695a283027a2c77c3' -or $id.sha256 -ne '330c059e82e61c4faa4d413e7820e7a78ce0bdc3f32ef840953658ed747c5d0c'){throw 'Owned build identity mismatch'}
if($gui.exit_code -ne 0 -or -not $gui.independent_verification){throw 'GUI validation incomplete'}
$active=@(Get-CimInstance Win32_Process|Where-Object{$_.ProcessId -ne $PID -and (($_.ExecutablePath -and $_.ExecutablePath.StartsWith($r+'\',[StringComparison]::OrdinalIgnoreCase)) -or ($_.CommandLine -and $_.CommandLine.Contains($r)))})
if($active.Count){throw 'A process still references the owned release directory; no deletion'}
$items=@(Get-ChildItem -LiteralPath $r -Recurse -Force)
if(@($items|Where-Object{($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0}).Count){throw 'Unexpected reparse point; no deletion'}
$files=@($items|Where-Object{-not $_.PSIsContainer})
$before=([IO.DriveInfo]::new('C')).AvailableFreeSpace
$bytes=($files|Measure-Object Length -Sum).Sum
Remove-Item -LiteralPath $r -Recurse -Force
if(Test-Path -LiteralPath $r){throw 'Owned build directory remains'}
@{root=$r;removed=$true;file_count=$files.Count;bytes=$bytes;free_before=$before;free_after=([IO.DriveInfo]::new('C')).AvailableFreeSpace;at=[DateTime]::UtcNow.ToString('o');previous_builds_preserved=$true}|ConvertTo-Json
"""
response=remote.run_powershell('Windows 11',code,timeout=240)
receipt=json.loads(response.stdout);assert receipt['removed'] is True
removed=[]
stage=ROOT/'build/professional-measurement-release-20261027'
for rel in ['source.zip','transfer/installer.exe','transfer/evidence.zip','transfer/transfer.json']:
 p=stage/rel
 if p.exists():
  assert p.is_file() and not p.is_symlink();removed.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)));p.unlink()
receipt['host_temporary_files_removed']=removed
receipt['retained_installer']=retained
(OUT/'cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
