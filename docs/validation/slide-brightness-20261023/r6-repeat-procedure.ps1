$ErrorActionPreference='Stop'
$root='C:\tmp\fdm-slide-brightness-20261023\r6'
Set-Location -LiteralPath $root
$out=$root+'\out-repeat'
if(Test-Path $out){throw 'Repeat already exists'}
New-Item -ItemType Directory $out|Out-Null
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH=$root+'\src'
$env:TEMP=$root+'\temp'
$env:TMP=$env:TEMP
$env:APPDATA=$root+'\profile\Roaming'
$env:LOCALAPPDATA=$root+'\profile\Local'
$r=@()
foreach($i in 1..3){
 [IO.File]::WriteAllText(($out+'\progress.txt'),[string]$i)
 $ErrorActionPreference='Continue'
 & 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe' -B scripts/validate_slide_brightness.py --output ($out+'\trial-'+$i) --platform windows --width 1920 --height 1080 --scale 1.5 --theme dark --require-performance *> ($out+'\trial-'+$i+'.log')
 $code=$LASTEXITCODE
 $ErrorActionPreference='Stop'
 $r+=@{trial=$i;exit_code=$code}
}
[IO.File]::WriteAllText(($out+'\result.json'),(@{fixed_trial_count=3;results=$r}|ConvertTo-Json -Depth 5))
