$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$r='C:\tmp\fdm-professional-precision-release-20261027'
$deadline=[DateTime]::UtcNow.AddMinutes(20)
while(-not(Test-Path ($r+'\build\release-status.json'))){if([DateTime]::UtcNow -gt $deadline){throw 'Build completion timeout'};Start-Sleep -Seconds 2}
$status=Get-Content ($r+'\build\release-status.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($status.exit_code -ne 0){throw 'Build did not pass'}
$env:PYTHONUTF8='1'
$py='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$arguments=@(($r+'\scripts\validation\verify_professional_precision_installer.py'),'--root',$r,'--installer',($r+'\dist\installer\fiber-diameter-measurement-setup-0.4.14-no-area-models.exe'),'--extractor','C:\tmp\fdm-cu-live-final-r4-20261015\build\validation-tools\innounp.exe')
$p=Start-Process $py -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput ($r+'\build\verify.log') -RedirectStandardError ($r+'\build\verify.stderr.log')
[IO.File]::WriteAllText(($r+'\build\verify-status.json'),(@{exit_code=$p.ExitCode;finished=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json))
exit $p.ExitCode
