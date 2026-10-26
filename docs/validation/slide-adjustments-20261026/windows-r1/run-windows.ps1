$ErrorActionPreference='Stop'
$root='C:\tmp\fdm-slide-adjustments-20261026\r1'
Set-Location -LiteralPath $root
$out=$root+'\out'
if(Test-Path $out){throw 'Output already exists'}
New-Item -ItemType Directory -Path $out|Out-Null
[IO.File]::WriteAllText(($out+'\execution-context.json'),(@{identity=[Security.Principal.WindowsIdentity]::GetCurrent().Name;session=(Get-Process -Id $PID).SessionId;utc=[DateTime]::UtcNow.ToString('o');windows=[Environment]::OSVersion.VersionString}|ConvertTo-Json))
$env:PYTHONPATH=$root+'\src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:QT_QPA_PLATFORM='windows'
$env:TEMP=$root+'\temp'
$env:TMP=$env:TEMP
$env:APPDATA=$root+'\profile\Roaming'
$env:LOCALAPPDATA=$root+'\profile\Local'
New-Item -ItemType Directory -Path $env:TEMP,$env:APPDATA,$env:LOCALAPPDATA -Force|Out-Null
$python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$results=@()
function Progress($s){[IO.File]::WriteAllText(($root+'\progress.txt'),[DateTime]::UtcNow.ToString('o')+' '+$s)}
function RunPython($name,[string[]]$arguments){
 $p=Start-Process -FilePath $python -ArgumentList $arguments -NoNewWindow -Wait -PassThru -RedirectStandardOutput ($out+'\'+$name+'.log') -RedirectStandardError ($out+'\'+$name+'.stderr.log')
 return $p.ExitCode
}
try{
 Progress 'pytest'
 Remove-Item Env:QT_SCALE_FACTOR -ErrorAction SilentlyContinue
 Remove-Item Env:QT_SCREEN_SCALE_FACTORS -ErrorAction SilentlyContinue
 $baseDpr=& $python -B -c 'from PySide6.QtWidgets import QApplication; a=QApplication([]); print(a.primaryScreen().devicePixelRatio())'
 $baseDpr=[double]::Parse($baseDpr.Trim(),[Globalization.CultureInfo]::InvariantCulture)
 $env:QT_SCALE_FACTOR=(1.0/$baseDpr).ToString([Globalization.CultureInfo]::InvariantCulture)
 [IO.File]::WriteAllText(($out+'\pytest-dpr.json'),(@{baseline_dpr=$baseDpr;qt_scale_factor=$env:QT_SCALE_FACTOR;target_dpr=1}|ConvertTo-Json))
 $tests=@('tests/test_slide_brightness.py','tests/test_slide_brightness_ui.py','tests/test_slide_brightness_boundaries.py','tests/test_slide_adjustments_settings.py','tests/test_slide_adjustments_popup.py','tests/test_theme_and_fluent_icons.py','tests/test_digital_slide_browse_renderer.py','tests/test_digital_slide_overlay_lifecycle.py','tests/test_digital_slide_annotation_display.py','tests/test_main_window_display_adjustment.py','tests/test_display_adjustment_dialog.py','tests/test_scale_overlay_editor.py','tests/test_export_service.py','tests/test_canvas_navigator.py','tests/test_digital_slide_configuration.py','tests/test_settings_dialog_navigation.py')
 $code=RunPython 'pytest' (@('-B','-m','pytest','-q','-p','no:cacheprovider')+$tests+@('--junitxml',($out+'\pytest.xml')))
 $results+=@{name='pytest';exit_code=$code}
 foreach($mode in @('brightness','exposure','combined')){
  foreach($case in @(@{w=1366;h=768;s='1'},@{w=1920;h=1080;s='1.5'},@{w=3840;h=2160;s='2'})){
   foreach($theme in @('light','dark')){
    $name="$mode-$($case.w)x$($case.h)-$($case.s)-$theme"
    Progress $name
    $code=RunPython $name @('-B','scripts/validate_slide_brightness.py','--output',($out+'\'+$name),'--platform','windows','--width',$case.w,'--height',$case.h,'--scale',$case.s,'--theme',$theme,'--adjustment-mode',$mode,'--samples','41','--require-performance')
    $results+=@{name=$name;exit_code=$code}
   }
  }
 }
 $status=if(@($results|Where-Object{$_.exit_code -ne 0}).Count){'failed'}else{'passed'}
 [IO.File]::WriteAllText(($root+'\result.json'),(@{status=$status;utc=[DateTime]::UtcNow.ToString('o');results=$results}|ConvertTo-Json -Depth 8))
 Progress $status
}catch{
 [IO.File]::WriteAllText(($root+'\result.json'),(@{status='error';error=$_.Exception.Message;results=$results}|ConvertTo-Json -Depth 8))
}
