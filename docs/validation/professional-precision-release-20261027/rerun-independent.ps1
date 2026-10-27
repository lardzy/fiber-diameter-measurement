$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$r='C:\tmp\fdm-professional-precision-release-20261027'
$status=Get-Content ($r+'\build\gui-status.json') -Raw -Encoding UTF8|ConvertFrom-Json
$result=Get-Content $status.result -Raw -Encoding UTF8|ConvertFrom-Json
if(-not $result.passed -or -not $result.graceful_exit -or $result.forced_termination){throw 'Actual GUI probe must already pass and exit normally'}
$folder=Split-Path $status.result
if(-not(Test-Path ($folder+'\initial-independent-status.json'))){Copy-Item ($r+'\build\gui-status.json') ($folder+'\initial-independent-status.json')}
if((Test-Path ($folder+'\reference-2.png')) -and -not(Test-Path ($folder+'\reference-initial-2.png'))){Copy-Item ($folder+'\reference-2.png') ($folder+'\reference-initial-2.png')}
if((Test-Path ($folder+'\independent-verify.log')) -and -not(Test-Path ($folder+'\independent-verify-initial.log'))){Copy-Item ($folder+'\independent-verify.log') ($folder+'\independent-verify-initial.log')}
$env:PYTHONUTF8='1'
$py='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$arguments=@(($r+'\scripts\validation\validate_professional_precision_frozen.py'),'verify','--result',$status.result)
$p=Start-Process $py -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput ($folder+'\independent-verify.log') -RedirectStandardError ($folder+'\independent-verify.stderr.log')
$code=$p.ExitCode
$status.exit_code=$code;$status.independent_verification=($code -eq 0);$status.finished=[DateTime]::UtcNow.ToString('o')
[IO.File]::WriteAllText(($r+'\build\gui-status.json'),($status|ConvertTo-Json))
$status|ConvertTo-Json
exit $code
