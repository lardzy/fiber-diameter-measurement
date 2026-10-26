# Windows PowerShell 5.1; UTF-8 BOM required for the Chinese UIA names.
# Run as the ordinary interactive user in Session 1. This tests only the
# extracted EXE and a synthetic slide; it never enters capture/device controls.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][string]$Slide,
    [Parameter(Mandatory=$true)][string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
$allowedRoot = 'C:\tmp\fdm-slide-adjustments-release-20261026\'
function Assert-TaskPath([string]$Path) {
    if (-not [IO.Path]::IsPathRooted($Path)) { throw 'Absolute task paths are required.' }
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the independent validation directory: $full"
    }
    # Reject existing junction/symlink ancestors as well as textual escapes.
    $ancestor = $full
    while ($ancestor) {
        if (Test-Path -LiteralPath $ancestor) {
            $item = Get-Item -LiteralPath $ancestor -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse points are not allowed: $ancestor"
            }
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
    return $full
}
$Executable = Assert-TaskPath $Executable
$Slide = Assert-TaskPath $Slide
$OutputRoot = Assert-TaskPath $OutputRoot
$exe = Get-Item -LiteralPath $Executable
$slideFile = Get-Item -LiteralPath $Slide
if ($exe.PSIsContainer -or $exe.Name -ne 'FiberDiameterMeasurement.exe' -or
    $slideFile.PSIsContainer -or $slideFile.Extension -ne '.fdmslide') {
    throw 'An extracted FiberDiameterMeasurement.exe and a synthetic .fdmslide are required.'
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if ([Diagnostics.Process]::GetCurrentProcess().SessionId -ne 1 -or
    $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run as the non-elevated interactive user in Session 1, not SYSTEM or an administrator token.'
}

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type -ReferencedAssemblies System.Drawing,System.Core -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
public sealed class BrightnessWindow { public long Handle; public long Owner; }
public sealed class BrightnessCapture {
    public int Width; public int Height; public double CenterMean; public int CenterColors;
}
public static class BrightnessProbeNative {
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    private delegate bool EnumProc(IntPtr hwnd, IntPtr state);
    [DllImport("user32.dll")] private static extern bool EnumWindows(EnumProc callback, IntPtr state);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint pid);
    [DllImport("user32.dll")] private static extern IntPtr GetWindow(IntPtr hwnd, uint command);
    [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll")] private static extern bool GetWindowRect(IntPtr hwnd, out Rect rect);
    [DllImport("user32.dll")] private static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
    [DllImport("user32.dll")] private static extern bool PostMessage(IntPtr hwnd, uint msg, IntPtr wp, IntPtr lp);
    [DllImport("user32.dll")] private static extern bool MoveWindow(IntPtr hwnd, int x, int y, int w, int h, bool repaint);
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern uint GetDpiForSystem();
    private static void RequireOwner(uint pid, long handle) {
        uint actual; GetWindowThreadProcessId(new IntPtr(handle), out actual);
        if (actual != pid) throw new InvalidOperationException("Window PID mismatch.");
    }
    public static BrightnessWindow[] Windows(uint pid) {
        var result = new List<BrightnessWindow>();
        EnumWindows(delegate(IntPtr h, IntPtr unused) {
            uint actual; GetWindowThreadProcessId(h, out actual);
            if (actual == pid && IsWindowVisible(h))
                result.Add(new BrightnessWindow {Handle=h.ToInt64(),Owner=GetWindow(h,4).ToInt64()});
            return true;
        }, IntPtr.Zero);
        return result.ToArray();
    }
    public static bool Close(uint pid, long handle) {
        RequireOwner(pid,handle);
        return PostMessage(new IntPtr(handle),0x0010,IntPtr.Zero,IntPtr.Zero);
    }
    public static void Place(uint pid, long handle, int width, int height) {
        RequireOwner(pid,handle);
        if (!MoveWindow(new IntPtr(handle),10,10,width,height,true))
            throw new InvalidOperationException("Could not place owned window.");
    }
    public static BrightnessCapture Capture(uint pid, long handle, string path) {
        RequireOwner(pid,handle);
        Rect r;
        if (!GetWindowRect(new IntPtr(handle),out r) || r.Right<=r.Left || r.Bottom<=r.Top)
            throw new InvalidOperationException("No window rectangle.");
        using (var bitmap = new Bitmap(r.Right-r.Left,r.Bottom-r.Top)) {
            using (var graphics = Graphics.FromImage(bitmap)) {
                var hdc = graphics.GetHdc();
                try {
                    if (!PrintWindow(new IntPtr(handle),hdc,2))
                        throw new InvalidOperationException("PrintWindow failed; no desktop capture fallback.");
                } finally { graphics.ReleaseHdc(hdc); }
            }
            bitmap.Save(path,ImageFormat.Png);
            // The center of the synthetic-slide viewport, excluding toolbar,
            // navigator, popup and side panels. Also save this explicit ROI.
            var roi = new Rectangle(bitmap.Width*45/100,bitmap.Height*45/100,
                                    bitmap.Width/10,bitmap.Height/5);
            using (var center = bitmap.Clone(roi,PixelFormat.Format32bppArgb)) {
                center.Save(System.IO.Path.ChangeExtension(path,".center.png"),ImageFormat.Png);
                var colors = new HashSet<int>(); double sum=0; int count=0;
                for (int y=0;y<center.Height;y+=3) for (int x=0;x<center.Width;x+=3) {
                    var c=center.GetPixel(x,y); colors.Add(c.ToArgb());
                    sum+=(c.R+c.G+c.B)/3.0; count++;
                }
                return new BrightnessCapture {Width=bitmap.Width,Height=bitmap.Height,
                    CenterMean=sum/count,CenterColors=colors.Count};
            }
        }
    }
}
'@
[void][BrightnessProbeNative]::SetProcessDPIAware()
$runDirectory = Join-Path $OutputRoot ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N'))
$config = Join-Path $runDirectory 'isolated-config'
$temporary = Join-Path $runDirectory 'temp'
$appConfig = Join-Path $config 'FiberDiameterMeasurement'
[void](New-Item -ItemType Directory -Path $appConfig,$temporary -Force)
$utf8 = [Text.UTF8Encoding]::new($false)
function Write-Json($Path, $Value) {
    [IO.File]::WriteAllText($Path,($Value | ConvertTo-Json -Depth 14),$utf8)
}
Write-Json (Join-Path $appConfig 'settings.json') ([ordered]@{
    theme_mode='light'; selected_capture_device_id='brightness-probe-unavailable'
    digital_slide_motor_output_enabled=$false; digital_slide_render_cache_gib=0
})
$result = [ordered]@{
    schema_version=1; passed=$false; started_at=(Get-Date).ToUniversalTime().ToString('o')
    scope='Frozen EXE UI smoke; synthetic source; no hardware, installation or performance acceptance'
    executable=$Executable; executable_sha256=(Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
    slide=$Slide; source_sha256_before=(Get-FileHash -LiteralPath $Slide -Algorithm SHA256).Hash
    source_sha256_after=$null; source_unchanged=$false; run_directory=$runDirectory
    session_id=1; process_id=$null; controls=@(); snapshots=@(); popup_candidates=@(); errors=@()
    graceful_exit=$false; exit_code=$null; process_still_running=$false; forced_termination=$false
}
$process = $null
$mainHandle = 0L
function Save-Result { Write-Json (Join-Path $runDirectory 'result.json') $result }
function Get-OwnedRoots {
    foreach ($window in [BrightnessProbeNative]::Windows([uint32]$process.Id)) {
        try {
            $element = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$window.Handle)
            if ($element.Current.ProcessId -eq $process.Id) {
                [pscustomobject]@{ window=$window; element=$element }
            }
        } catch { }
    }
}
function Find-Owned([string]$Name) {
    $condition = [System.Windows.Automation.AndCondition]::new(
        [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ProcessIdProperty,[int]$process.Id),
        [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::NameProperty,$Name))
    $seen=@{}
    foreach ($root in @(Get-OwnedRoots)) {
        foreach ($element in $root.element.FindAll([System.Windows.Automation.TreeScope]::Subtree,$condition)) {
            $key=[string]::Join(',',$element.GetRuntimeId())
            if (-not $seen.ContainsKey($key) -and $element.Current.IsEnabled -and -not $element.Current.IsOffscreen) {
                $seen[$key]=$true; $element
            }
        }
    }
}
function Wait-Control([string]$Name,[int]$Seconds=20) {
    $deadline=[DateTime]::UtcNow.AddSeconds($Seconds)
    do {
        if ($process.HasExited) { throw "EXE exited before control appeared: $Name" }
        $found=@(Find-Owned $Name)
        if ($found.Count -eq 1) { return $found[0] }
        if ($found.Count -gt 1) { throw "Ambiguous PID-owned control: $Name" }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Timed out waiting for PID-owned control: $Name"
}
function Record-Control([string]$Phase,$Element) {
    if ($Element.Current.ProcessId -ne $process.Id) { throw 'Foreign control.' }
    $result.controls += [ordered]@{phase=$Phase; name=$Element.Current.Name;
        process_id=$Element.Current.ProcessId; class=$Element.Current.ClassName;
        automation_id=$Element.Current.AutomationId; control_type=$Element.Current.ControlType.ProgrammaticName;
        patterns=@($Element.GetSupportedPatterns() | ForEach-Object {$_.ProgrammaticName})}
    Save-Result
}
function Invoke-Owned($Element) {
    if ($Element.Current.ProcessId -ne $process.Id -or -not $Element.Current.IsEnabled -or $Element.Current.IsOffscreen) {
        throw 'Refusing absent, foreign, disabled or offscreen control.'
    }
    $pattern=$null
    if ($Element.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern,[ref]$pattern)) {
        $pattern.Invoke(); return
    }
    foreach ($supported in $Element.GetSupportedPatterns()) {
        if ($supported.ProgrammaticName -eq 'LegacyIAccessiblePatternIdentifiers.Pattern') {
            $Element.GetCurrentPattern($supported).DoDefaultAction(); return
        }
    }
    throw 'No supported UIA invocation pattern.'
}
function Set-BrightnessValue($Element,[double]$Value) {
    if ($Element.Current.ProcessId -ne $process.Id -or -not $Element.Current.IsEnabled -or $Element.Current.IsOffscreen) {
        throw 'Refusing foreign or unreachable numeric input.'
    }
    $pattern=$null
    if ($Element.TryGetCurrentPattern([System.Windows.Automation.RangeValuePattern]::Pattern,[ref]$pattern) -and -not $pattern.Current.IsReadOnly) {
        $pattern.SetValue([double]$Value); return
    }
    $pattern=$null
    if ($Element.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern,[ref]$pattern) -and -not $pattern.Current.IsReadOnly) {
        $pattern.SetValue([string]$Value); return
    }
    throw 'Numeric control has no writable UIA value pattern; no global-keyboard fallback.'
}
function Close-BrightnessPopup {
    $resetCondition=[System.Windows.Automation.AndCondition]::new([System.Windows.Automation.Condition[]]@(
        [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ProcessIdProperty,[int]$process.Id),
        [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::NameProperty,'恢复原始显示'),
        [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ControlTypeProperty,[System.Windows.Automation.ControlType]::Button)
    ))
    $owned=@()
    $diagnostics=@()
    foreach ($root in @(Get-OwnedRoots)) {
        if ($root.window.Handle -eq $mainHandle) { continue }
        $record=[ordered]@{handle=$root.window.Handle; owner=$root.window.Owner; process_id=$process.Id;
            class=$null; name=$null; automation_id=$null; has_reset=$false; popup_identity=$false; error=$null}
        try {
            $current=$root.element.Current
            $record.class=$current.ClassName
            $record.name=$current.Name
            $record.automation_id=$current.AutomationId
            $reset=$root.element.FindFirst([System.Windows.Automation.TreeScope]::Subtree,$resetCondition)
            $record.has_reset=($null -ne $reset -and $reset.Current.IsEnabled -and -not $reset.Current.IsOffscreen)
            # The actual Qt UIA numeric control is rooted at
            # QApplication.slideBrightnessPopup.slideBrightnessValue. Qt Popup
            # HWNDs can have native Owner=0; ownership is established by PID,
            # popup class/object identity and the exact reset button instead.
            $record.popup_identity=($current.ClassName -eq 'SlideBrightnessPopup' -or
                $current.AutomationId -match '(^|\.)slideBrightnessPopup$')
            if ($current.ProcessId -eq $process.Id -and $record.has_reset -and $record.popup_identity) {
                $owned += $root
            }
        } catch { $record.error=$_.Exception.Message }
        $diagnostics += $record
    }
    $result.popup_candidates += [ordered]@{utc=[DateTime]::UtcNow.ToString('o'); candidates=$diagnostics}
    Save-Result
    if ($owned.Count -ne 1) { throw 'Cannot identify exactly one owned brightness popup.' }
    if (-not [BrightnessProbeNative]::Close([uint32]$process.Id,$owned[0].window.Handle)) { throw 'Could not request popup close.' }
    $deadline=[DateTime]::UtcNow.AddSeconds(5)
    while (@(Find-Owned '恢复原始显示').Count -gt 0) {
        if ([DateTime]::UtcNow -gt $deadline) { throw 'Brightness popup did not close.' }
        Start-Sleep -Milliseconds 100
    }
}
function Capture([string]$Phase) {
    Start-Sleep -Milliseconds 500
    $path=Join-Path $runDirectory ($Phase+'.png')
    $metrics=[BrightnessProbeNative]::Capture([uint32]$process.Id,$mainHandle,$path)
    $entry=[ordered]@{phase=$Phase; file=[IO.Path]::GetFileName($path); width=$metrics.Width; height=$metrics.Height;
        sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash;
        center_sha256=(Get-FileHash -LiteralPath ([IO.Path]::ChangeExtension($path,'.center.png')) -Algorithm SHA256).Hash;
        center_mean=$metrics.CenterMean; center_colors=$metrics.CenterColors; source='PID-owned PrintWindow; fixed center ROI'}
    $result.snapshots += $entry
    Save-Result
    if ($metrics.CenterColors -lt 8) { throw 'Center screenshot is blank/uninformative; inspect capture and fixture.' }
    return $entry
}
function Read-Numeric($Element) {
    $pattern=$null
    if($Element.TryGetCurrentPattern([System.Windows.Automation.RangeValuePattern]::Pattern,[ref]$pattern)){return [double]$pattern.Current.Value}
    $pattern=$null
    if($Element.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern,[ref]$pattern)){
        $match=[regex]::Match($pattern.Current.Value,'[-+]?\d+(\.\d+)?')
        if($match.Success){return [double]::Parse($match.Value,[Globalization.CultureInfo]::InvariantCulture)}
    }
    throw 'Numeric value is not readable.'
}
function Assert-Pair([int]$Brightness,[double]$Exposure) {
    $b=Read-Numeric (Wait-Control '切片查看亮度数值')
    $e=Read-Numeric (Wait-Control '切片查看曝光补偿数值')
    if($b -ne $Brightness -or [Math]::Abs($e-$Exposure) -gt .00001){throw "Unexpected pair: $b / $e"}
    $result.pairs += [ordered]@{brightness=$b;exposure=$e;utc=[DateTime]::UtcNow.ToString('o')}
}
function Set-Pair([int]$Brightness,[double]$Exposure) {
    Set-BrightnessValue (Wait-Control '切片查看亮度数值') $Brightness
    Set-BrightnessValue (Wait-Control '切片查看曝光补偿数值') $Exposure
    Start-Sleep -Milliseconds 150
    Assert-Pair $Brightness $Exposure
}
function Remember-State {
    $control=Wait-Control '记住当前切片查看设置'
    $pattern=$control.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
    return ($pattern.Current.ToggleState -eq [System.Windows.Automation.ToggleState]::On)
}
function Set-Remember([bool]$Enabled) {
    $control=Wait-Control '记住当前切片查看设置'
    $pattern=$control.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
    if(($pattern.Current.ToggleState -eq [System.Windows.Automation.ToggleState]::On) -ne $Enabled){$pattern.Toggle()}
    if((Remember-State) -ne $Enabled){throw 'Remember checkbox did not change.'}
}
function Assert-Saved([bool]$Remember,[int]$Brightness,[int]$ExposureTenths) {
    $saved=Get-Content -LiteralPath (Join-Path $appConfig 'settings.json') -Raw -Encoding UTF8|ConvertFrom-Json
    if($saved.digital_slide_remember_view_adjustments -ne $Remember -or $saved.digital_slide_view_brightness -ne $Brightness -or $saved.digital_slide_view_exposure_tenths -ne $ExposureTenths){throw 'Persisted defaults mismatch.'}
    $result.settings_checks += [ordered]@{remember=$Remember;brightness=$Brightness;exposure_tenths=$ExposureTenths}
}
function Open-Adjustments {
    Invoke-Owned (Wait-Control '切片亮度与曝光')
    [void](Wait-Control '切片查看曝光补偿数值')
}
function Start-OwnedApp {
    $script:process=[Diagnostics.Process]::new()
    $process.StartInfo=[Diagnostics.ProcessStartInfo]::new()
    $process.StartInfo.FileName=$Executable
    $process.StartInfo.Arguments='"'+$Slide+'"'
    $process.StartInfo.WorkingDirectory=$exe.DirectoryName
    $process.StartInfo.UseShellExecute=$false
    foreach($name in @('PSExecutionPolicyPreference','QT_SCREEN_SCALE_FACTORS','QT_FONT_DPI','PYTHONPATH','PYTHONHOME','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH')){$process.StartInfo.EnvironmentVariables.Remove($name)}
    $process.StartInfo.EnvironmentVariables['APPDATA']=$config
    $process.StartInfo.EnvironmentVariables['LOCALAPPDATA']=$config
    $process.StartInfo.EnvironmentVariables['TEMP']=$temporary
    $process.StartInfo.EnvironmentVariables['TMP']=$temporary
    $process.StartInfo.EnvironmentVariables['QT_ACCESSIBILITY']='1'
    $process.StartInfo.EnvironmentVariables['QT_QPA_PLATFORM']='windows'
    $process.StartInfo.EnvironmentVariables['QT_SCALE_FACTOR']=(96.0/[BrightnessProbeNative]::GetDpiForSystem()).ToString([Globalization.CultureInfo]::InvariantCulture)
    [void]$process.Start()
    $result.process_id=$process.Id
    $result.process_ids+=@($process.Id)
    Save-Result
    [void](Wait-Control '切片亮度与曝光' 90)
    $main=@(Get-OwnedRoots|Where-Object{$_.window.Owner -eq 0 -and $_.element.Current.ControlType -eq [System.Windows.Automation.ControlType]::Window})
    if($main.Count -ne 1){throw 'Could not identify one owned main window.'}
    $script:mainHandle=$main[0].window.Handle
    $area=[Windows.Forms.Screen]::PrimaryScreen.WorkingArea
    [BrightnessProbeNative]::Place([uint32]$process.Id,$mainHandle,[Math]::Min(1366,$area.Width-30),[Math]::Min(850,$area.Height-30))
    Start-Sleep -Milliseconds 1500
}
function Stop-OwnedApp {
    if($null -ne $process -and -not $process.HasExited){
        [void][BrightnessProbeNative]::Close([uint32]$process.Id,$mainHandle)
        [void]$process.WaitForExit(15000)
    }
    if($null -ne $process){
        $result.process_still_running=-not $process.HasExited
        if(-not $process.HasExited){throw 'Owned EXE did not stop; no forced termination.'}
        $result.exit_code=$process.ExitCode
        $result.exits+=@{pid=$process.Id;code=$process.ExitCode}
        if($process.ExitCode -ne 0){throw 'Owned EXE returned a nonzero exit code.'}
    }
}
$result.schema_version=2
$result.pairs=@();$result.settings_checks=@();$result.process_ids=@();$result.exits=@()
$result.exposure_verified=$false;$result.remember_restart_verified=$false;$result.uncheck_restart_verified=$false
$result.reset_remember_verified=$false
try {
    Start-OwnedApp
    Record-Control 'entry' (Wait-Control '切片亮度与曝光')
    $initial=Capture '01-original'
    Open-Adjustments
    Assert-Pair 0 0
    if(Remember-State){throw 'Fresh profile must not remember defaults.'}
    Set-Pair 0 1.0
    Close-BrightnessPopup
    $exposed=Capture '02-exposure-plus-1'
    $result.exposure_verified=($exposed.center_mean -gt $initial.center_mean+1 -and $exposed.center_sha256 -ne $initial.center_sha256)
    if(-not $result.exposure_verified){throw 'Exposure did not change the displayed pixels.'}
    Open-Adjustments
    Set-Pair 40 1.3
    Set-Remember $true
    Assert-Saved $true 40 13
    Close-BrightnessPopup
    $combined=Capture '03-combined-remembered'
    Stop-OwnedApp
    Start-OwnedApp
    Open-Adjustments
    Assert-Pair 40 1.3
    if(-not (Remember-State)){throw 'Remember state did not survive restart.'}
    Close-BrightnessPopup
    $restarted=Capture '04-restarted-remembered'
    $result.remember_restart_verified=($restarted.center_sha256 -eq $combined.center_sha256)
    if(-not $result.remember_restart_verified){throw 'Remembered display pixels differ after restart.'}
    Open-Adjustments
    Invoke-Owned (Wait-Control '恢复原始显示')
    Assert-Pair 0 0
    if(-not (Remember-State)){throw 'Reset must keep remember checked.'}
    Assert-Saved $true 0 0
    Close-BrightnessPopup
    $reset=Capture '05-reset-original'
    $result.reset_remember_verified=($reset.center_sha256 -eq $initial.center_sha256)
    if(-not $result.reset_remember_verified){throw 'Reset did not restore exact original pixels.'}
    Open-Adjustments
    Set-Pair 20 -0.7
    Close-BrightnessPopup
    Assert-Saved $true 20 -7
    $beforeUncheck=Capture '06-before-uncheck'
    Open-Adjustments
    Set-Remember $false
    Assert-Pair 20 -0.7
    Assert-Saved $false 0 0
    Close-BrightnessPopup
    $afterUncheck=Capture '07-unchecked-current-kept'
    if($afterUncheck.center_sha256 -ne $beforeUncheck.center_sha256){throw 'Uncheck changed the current slide view.'}
    Stop-OwnedApp
    Start-OwnedApp
    Open-Adjustments
    Assert-Pair 0 0
    if(Remember-State){throw 'Uncheck did not survive restart.'}
    Close-BrightnessPopup
    $neutral=Capture '08-restarted-neutral'
    $result.uncheck_restart_verified=($neutral.center_sha256 -eq $initial.center_sha256)
    if(-not $result.uncheck_restart_verified){throw 'Unremembered restart did not use neutral display.'}
} catch {$result.errors+=$_.Exception.Message} finally {
    try {Stop-OwnedApp;$result.graceful_exit=$true}catch{$result.errors+=('Close: '+$_.Exception.Message)}
    $result.source_sha256_after=(Get-FileHash -LiteralPath $Slide -Algorithm SHA256).Hash
    $result.source_unchanged=($result.source_sha256_after -eq $result.source_sha256_before)
    $result.passed=($result.errors.Count -eq 0 -and $result.exposure_verified -and $result.remember_restart_verified -and $result.reset_remember_verified -and $result.uncheck_restart_verified -and $result.source_unchanged -and $result.graceful_exit)
    $result.finished_at=[DateTime]::UtcNow.ToString('o')
    Save-Result
}
$result | ConvertTo-Json -Depth 14
if($result.passed){exit 0}else{exit 1}
