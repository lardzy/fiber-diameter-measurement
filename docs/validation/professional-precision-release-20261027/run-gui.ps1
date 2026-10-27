$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$r='C:\tmp\fdm-professional-precision-release-20261027'
$source=Get-Content ($r+'\build\source-validation\status.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($source.exit_code -ne 0){throw 'Source and UI validation must pass before actual GUI probe'}
$identity=Get-Content ($r+'\build\package\installer-identity.json') -Raw -Encoding UTF8|ConvertFrom-Json
if(-not $identity.passed){throw 'Installer verification must pass first'}
$exe=Join-Path $identity.payload 'FiberDiameterMeasurement.exe'
$py='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$output=$r+'\build\frozen-gui'
if(Test-Path ($r+'\build\gui-status.json')){
 $old=Get-Content ($r+'\build\gui-status.json') -Raw -Encoding UTF8|ConvertFrom-Json
 if($old.result -and (Test-Path $old.result)){
  $archive=Split-Path $old.result
  foreach($name in @('gui.log','gui.stderr.log','gui-status.json')){if(Test-Path ($r+'\build\'+$name)){Copy-Item ($r+'\build\'+$name) (Join-Path $archive ('runner-'+$name))}}
 }
}
$fixture=$r+'\build\precision-ui-fixtures\'+[Guid]::NewGuid().ToString('N')
$env:PYTHONUTF8='1'
$fixtureLog=Join-Path $output ('prepare-'+(Split-Path $fixture -Leaf)+'.log')
& $py ($r+'\scripts\validation\validate_professional_precision_frozen.py') prepare --output $fixture > $fixtureLog
if($LASTEXITCODE -ne 0){throw 'Fresh fixture preparation failed'}
$arguments=@('-STA','-NoProfile','-ExecutionPolicy','Bypass','-File',($r+'\scripts\validation\probe_professional_precision_frozen.ps1'),'-Executable',$exe,'-Project',($fixture+'\professional-smoke.fdmproj'),'-Image',($fixture+'\synthetic-edge.bmp'),'-OutputRoot',$output)
$started=[DateTime]::UtcNow
$p=Start-Process powershell -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput ($r+'\build\gui.log') -RedirectStandardError ($r+'\build\gui.stderr.log')
$nativeHandle=$p.Handle
if(-not $p.WaitForExit(900000)){throw 'External UI probe exceeded 15 minutes; preserve state for diagnosis'}
$code=$p.ExitCode
if($null -eq $code){$code=-1}
$result=Get-ChildItem $output -Filter result.json -Recurse|Where-Object {$_.LastWriteTimeUtc -ge $started}|Sort-Object LastWriteTimeUtc -Descending|Select-Object -First 1
$status=@{exit_code=$code;finished=[DateTime]::UtcNow.ToString('o');independent_verification=$false}
if($result){$status.result=$result.FullName}
if($code -eq 0 -and $result){
    $env:PYTHONUTF8='1'
    $verifyArguments=@(($r+'\scripts\validation\validate_professional_precision_frozen.py'),'verify','--result',$result.FullName)
    $check=Start-Process $py -ArgumentList $verifyArguments -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput ($result.DirectoryName+'\independent-verify.log') -RedirectStandardError ($result.DirectoryName+'\independent-verify.stderr.log')
    $code=$check.ExitCode
    $status.independent_verification=($code -eq 0)
    $status.exit_code=$code
}
[IO.File]::WriteAllText(($r+'\build\gui-status.json'),($status|ConvertTo-Json))
$status|ConvertTo-Json
exit $code
