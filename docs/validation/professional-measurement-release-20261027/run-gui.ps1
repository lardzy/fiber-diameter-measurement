$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$r='C:\tmp\fdm-professional-measurement-release-20261027'
$identity=Get-Content ($r+'\build\package\installer-identity.json') -Raw -Encoding UTF8|ConvertFrom-Json
if(-not $identity.passed){throw 'Installer verification must pass first'}
$exe=Join-Path $identity.payload 'FiberDiameterMeasurement.exe'
$py='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$output=$r+'\build\frozen-gui'
$fixture=$r+'\build\professional-ui-fixtures\'+[Guid]::NewGuid().ToString('N')
$env:PYTHONUTF8='1'
& $py ($r+'\scripts\validation\validate_professional_measurement_frozen.py') prepare --output $fixture
if($LASTEXITCODE -ne 0){throw 'Fresh fixture preparation failed'}
$arguments=@('-NoProfile','-ExecutionPolicy','Bypass','-File',($r+'\scripts\validation\probe_professional_measurement_frozen.ps1'),'-Executable',$exe,'-Project',($fixture+'\professional-smoke.fdmproj'),'-Image',($fixture+'\synthetic-edge.bmp'),'-OutputRoot',$output)
$p=Start-Process powershell -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput ($r+'\build\gui.log') -RedirectStandardError ($r+'\build\gui.stderr.log')
$code=$p.ExitCode
$result=Get-ChildItem $output -Filter result.json -Recurse|Sort-Object LastWriteTimeUtc -Descending|Select-Object -First 1
$status=@{exit_code=$code;finished=[DateTime]::UtcNow.ToString('o');independent_verification=$false}
if($result){$status.result=$result.FullName}
if($code -eq 0 -and $result){
    $env:PYTHONUTF8='1'
    & $py ($r+'\scripts\validation\validate_professional_measurement_frozen.py') verify --result $result.FullName > ($result.DirectoryName+'\independent-verify.log')
    $code=$LASTEXITCODE
    $status.independent_verification=($code -eq 0)
    $status.exit_code=$code
}
[IO.File]::WriteAllText(($r+'\build\gui-status.json'),($status|ConvertTo-Json))
$status|ConvertTo-Json
exit $code
