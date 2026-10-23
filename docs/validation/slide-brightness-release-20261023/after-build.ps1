$ProgressPreference='SilentlyContinue';$ErrorActionPreference='Stop'
$r='C:\tmp\fdm-brightness-release-20261023';$s='\\Mac\Home\Downloads\fdm-brightness-release-transfer-20261023'
$deadline=(Get-Date).AddMinutes(15)
while(-not(Test-Path ($r+'\build\repackage-exit.txt'))){if((Get-Date) -gt $deadline){throw 'Installer completion timeout'};Start-Sleep -Seconds 2}
if([int](Get-Content ($r+'\build\repackage-exit.txt') -Raw) -ne 0){throw 'Installer compilation failed'}
Copy-Item ($r+'\build\repackage.log'),($r+'\build\repackage-exit.txt') $s -Force
$env:PYTHONUTF8='1'
$py='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$args=@(($r+'\verify_package.py'),'--root',$r,'--installer',($r+'\dist\installer\fiber-diameter-measurement-setup-0.4.11-no-area-models.exe'),'--extractor','C:\tmp\fdm-cu-live-final-r4-20261015\build\validation-tools\innounp.exe')
$p=Start-Process $py -ArgumentList $args -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput ($r+'\build\verify.log') -RedirectStandardError ($r+'\build\verify.stderr.log')
[pscustomobject]@{exit_code=$p.ExitCode;finished=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content ($s+'\verify-status.json') -Encoding UTF8
Copy-Item ($r+'\build\verify.log'),($r+'\build\verify.stderr.log'),($r+'\build\package\installer-identity.json') $s -Force
if($p.ExitCode -ne 0){exit $p.ExitCode}
$id=Get-Content ($r+'\build\package\installer-identity.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$exe=Join-Path $id.payload 'FiberDiameterMeasurement.exe'
$args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',($r+'\probe.ps1'),'-Executable',$exe,'-Slide',($r+'\synthetic.fdmslide'),'-OutputRoot',($r+'\build\frozen-gui'))
$p=Start-Process powershell.exe -ArgumentList $args -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput ($r+'\build\gui.log') -RedirectStandardError ($r+'\build\gui.stderr.log')
[pscustomobject]@{exit_code=$p.ExitCode;finished=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content ($s+'\gui-status.json') -Encoding UTF8
Copy-Item ($r+'\build\gui.log'),($r+'\build\gui.stderr.log') $s -Force
exit $p.ExitCode
