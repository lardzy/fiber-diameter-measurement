$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$r='C:\tmp\fdm-slide-adjustments-release-20261026'
$s='\\Mac\Home\Downloads\fdm-slide-adjustments-release-transfer-20261026'
$p=Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',($r+'\run-gui.ps1')) -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput ($r+'\build\gui.log') -RedirectStandardError ($r+'\build\gui.stderr.log')
[IO.File]::WriteAllText(($r+'\build\gui-status.json'),(@{exit_code=$p.ExitCode;finished=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json))
exit $p.ExitCode
