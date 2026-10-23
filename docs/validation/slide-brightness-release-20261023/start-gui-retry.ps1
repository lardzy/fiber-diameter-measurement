$ProgressPreference='SilentlyContinue';$ErrorActionPreference='Stop'
$r='C:\tmp\fdm-brightness-release-20261023';$s='\\Mac\Home\Downloads\fdm-brightness-release-transfer-20261023'
$live=@(Get-Process -Name FiberDiameterMeasurement -ErrorAction SilentlyContinue | Where-Object {$_.Path -like ($r+'\*')})
if($live.Count){throw 'An owned test EXE is still running; inspect before relaunch'}
foreach($n in @('probe.ps1','run-gui.ps1','export_evidence.py','cleanup_windows.py')){Copy-Item (Join-Path $s $n) $r -Force}
$tokens=$null;$errors=$null
[void][Management.Automation.Language.Parser]::ParseFile(($r+'\probe.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors | Out-String)}
$p=Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',($r+'\run-gui.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput ($r+'\build\gui-retry.log') -RedirectStandardError ($r+'\build\gui-retry.stderr.log')
[pscustomobject]@{gui_retry_pid=$p.Id;parse_errors=0}|ConvertTo-Json
