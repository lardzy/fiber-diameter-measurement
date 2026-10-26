$ErrorActionPreference='Stop'
$root='C:\tmp\fdm-slide-adjustments-20261026\r2'
$repeat='C:\tmp\fdm-slide-adjustments-20261026\r2-repeat-4k'
while(-not (Test-Path ($root+'\result.json'))){Start-Sleep -Seconds 2}
if(Test-Path $repeat){throw 'Repeat output already exists'}
Set-Location -LiteralPath $root
$out=$repeat+'\out'
New-Item -ItemType Directory -Path $out|Out-Null
foreach($name in @('native-reuse.json','source-manifest.json')){Copy-Item -LiteralPath ($root+'\'+$name) -Destination $repeat}
Copy-Item -LiteralPath $PSCommandPath -Destination ($repeat+'\run-windows.ps1')
$env:PYTHONPATH=$root+'\src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:QT_QPA_PLATFORM='windows'
$env:TEMP=$repeat+'\temp'
$env:TMP=$env:TEMP
$env:APPDATA=$repeat+'\profile\Roaming'
$env:LOCALAPPDATA=$repeat+'\profile\Local'
New-Item -ItemType Directory -Path $env:TEMP,$env:APPDATA,$env:LOCALAPPDATA -Force|Out-Null
$python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$results=@()
foreach($i in 1..3){
 [IO.File]::WriteAllText(($repeat+'\progress.txt'),[DateTime]::UtcNow.ToString('o')+' trial-'+$i)
 $p=Start-Process -FilePath $python -ArgumentList @('-B','scripts/validate_slide_brightness.py','--output',($out+'\trial-'+$i),'--platform','windows','--width','3840','--height','2160','--scale','2','--theme','light','--adjustment-mode','combined','--samples','41','--require-performance') -NoNewWindow -Wait -PassThru -RedirectStandardOutput ($out+'\trial-'+$i+'.log') -RedirectStandardError ($out+'\trial-'+$i+'.stderr.log')
 $results+=@{trial=$i;exit_code=$p.ExitCode}
}
[IO.File]::WriteAllText(($repeat+'\result.json'),(@{fixed_trial_count=3;source_root=$root;results=$results}|ConvertTo-Json -Depth 5))
