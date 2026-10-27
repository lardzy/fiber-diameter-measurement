$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$r='C:\tmp\fdm-professional-precision-release-20261027'
$deadline=[DateTime]::UtcNow.AddMinutes(25)
while(-not(Test-Path ($r+'\build\release-status.json'))){if([DateTime]::UtcNow -gt $deadline){throw 'Build timeout'};Start-Sleep -Seconds 2}
$build=Get-Content ($r+'\build\release-status.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($build.exit_code -ne 0){throw 'Build failed'}
$py='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$out=$r+'\build\source-validation'
New-Item -ItemType Directory ($out+'\isolated-config'),($out+'\temp') -Force|Out-Null
$env:APPDATA=$out+'\isolated-config';$env:LOCALAPPDATA=$env:APPDATA;$env:TEMP=$out+'\temp';$env:TMP=$env:TEMP
$env:PYTHONUTF8='1';$env:PYTHONDONTWRITEBYTECODE='1';$env:QT_QPA_PLATFORM='windows';$env:PYTHONPATH=$r+'\src'
Remove-Item Env:QT_SCALE_FACTOR -ErrorAction SilentlyContinue
Remove-Item Env:QT_SCREEN_SCALE_FACTORS -ErrorAction SilentlyContinue
Remove-Item Env:QT_FONT_DPI -ErrorAction SilentlyContinue
Set-Location $r
$tests=@(Get-Content ($r+'\build\test-scope.json') -Raw -Encoding UTF8|ConvertFrom-Json)
$testArguments=@('-m','pytest','-q',('--junitxml='+$out+'\windows-regression.xml'))+$tests
$p=Start-Process $py -ArgumentList $testArguments -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput ($out+'\windows-regression.log') -RedirectStandardError ($out+'\windows-regression.stderr.log')
$code=$p.ExitCode
$ui=@()
if($code -eq 0){
 foreach($scale in @(1,1.5,2)){
  $tag=[int]($scale*100)
  $uiArguments=@(($r+'\scripts\validation\validate_professional_precision_ui.py'),'--platform','windows','--scale',([string]$scale),'--output',($out+'\windows-'+$tag))
  $p=Start-Process $py -ArgumentList $uiArguments -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput ($out+'\ui-'+$tag+'.log') -RedirectStandardError ($out+'\ui-'+$tag+'.stderr.log')
  $ui+=@{scale=$scale;exit_code=$p.ExitCode}
  if($p.ExitCode -ne 0){$code=$p.ExitCode;break}
 }
}
@{exit_code=$code;ui=$ui;finished=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Depth 4|Set-Content ($out+'\status.json') -Encoding UTF8
exit $code
