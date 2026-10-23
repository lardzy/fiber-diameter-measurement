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
$allowedRoot = 'C:\tmp\fdm-brightness-release-20261023\'
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
    brightness_60_verified=$false; reset_0_verified=$false; center_brighter=$false; center_restored=$false
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
function Set-BrightnessValue($Element,[int]$Value) {
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
        [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::NameProperty,'恢复原始亮度'),
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
    while (@(Find-Owned '恢复原始亮度').Count -gt 0) {
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
try {
    Save-Result
    $process=[Diagnostics.Process]::new()
    $process.StartInfo=[Diagnostics.ProcessStartInfo]::new()
    $process.StartInfo.FileName=$Executable
    $process.StartInfo.Arguments='"'+$Slide+'"'
    $process.StartInfo.WorkingDirectory=$exe.DirectoryName
    $process.StartInfo.UseShellExecute=$false
    foreach ($name in @('PSExecutionPolicyPreference','QT_SCREEN_SCALE_FACTORS','QT_FONT_DPI')) { $process.StartInfo.EnvironmentVariables.Remove($name) }
    $process.StartInfo.EnvironmentVariables['APPDATA']=$config
    $process.StartInfo.EnvironmentVariables['LOCALAPPDATA']=$config
    $process.StartInfo.EnvironmentVariables['TEMP']=$temporary
    $process.StartInfo.EnvironmentVariables['TMP']=$temporary
    $process.StartInfo.EnvironmentVariables['QT_ACCESSIBILITY']='1'
    $process.StartInfo.EnvironmentVariables['QT_QPA_PLATFORM']='windows'
    $process.StartInfo.EnvironmentVariables['QT_SCALE_FACTOR']=(96.0/[BrightnessProbeNative]::GetDpiForSystem()).ToString([Globalization.CultureInfo]::InvariantCulture)
    [void]$process.Start()
    $result.process_id=$process.Id
    Save-Result
    $button=Wait-Control '亮度：0' 90
    $main=@(Get-OwnedRoots | Where-Object {$_.window.Owner -eq 0 -and $_.element.Current.ControlType -eq [System.Windows.Automation.ControlType]::Window})
    if ($main.Count -ne 1) { throw 'Could not identify one PID-owned main window.' }
    $mainHandle=$main[0].window.Handle
    $area=[Windows.Forms.Screen]::PrimaryScreen.WorkingArea
    [BrightnessProbeNative]::Place([uint32]$process.Id,$mainHandle,[Math]::Min(1366,$area.Width-30),[Math]::Min(850,$area.Height-30))
    Start-Sleep -Milliseconds 1200
    Record-Control 'initial-0' (Wait-Control '亮度：0')
    $initial=Capture '01-original'
    Invoke-Owned (Wait-Control '亮度：0')
    $numeric=Wait-Control '切片查看亮度数值'
    Record-Control 'numeric' $numeric
    Set-BrightnessValue $numeric 60
    Record-Control 'bright-60' (Wait-Control '亮度：60')
    $result.brightness_60_verified=$true
    Close-BrightnessPopup
    $bright=Capture '02-bright-60'
    $result.center_brighter=($bright.center_mean -gt $initial.center_mean+1 -and $bright.center_sha256 -ne $initial.center_sha256)
    Invoke-Owned (Wait-Control '亮度：60')
    $reset=Wait-Control '恢复原始亮度'
    Record-Control 'reset-action' $reset
    Invoke-Owned $reset
    Record-Control 'restored-0' (Wait-Control '亮度：0')
    $result.reset_0_verified=$true
    Close-BrightnessPopup
    $restored=Capture '03-restored-0'
    $result.center_restored=($restored.center_sha256 -eq $initial.center_sha256)
    if (-not $result.center_brighter -or -not $result.center_restored) { throw 'Center pixels did not brighten and restore exactly; inspect screenshots.' }
} catch {
    $result.errors += $_.Exception.Message
} finally {
    if ($null -ne $process -and $null -ne $result.process_id) {
        try {
            if (-not $process.HasExited) {
                if ($mainHandle -eq 0) {
                    $roots=@(Get-OwnedRoots | Where-Object {$_.window.Owner -eq 0})
                    if ($roots.Count -eq 1) { $mainHandle=$roots[0].window.Handle }
                }
                if ($mainHandle -ne 0) { [void][BrightnessProbeNative]::Close([uint32]$process.Id,$mainHandle) }
                [void]$process.WaitForExit(15000)
            }
            $result.process_still_running=-not $process.HasExited
            if ($process.HasExited) {
                $result.exit_code=$process.ExitCode
                $result.graceful_exit=($mainHandle -ne 0 -and $process.ExitCode -eq 0)
            } else { $result.errors += 'Owned EXE did not exit normally; left running for inspection. No force termination.' }
        } catch { $result.errors += ('Close: '+$_.Exception.Message) }
    }
    try {
        $result.source_sha256_after=(Get-FileHash -LiteralPath $Slide -Algorithm SHA256).Hash
        $result.source_unchanged=($result.source_sha256_after -eq $result.source_sha256_before)
    } catch { $result.errors += ('Source hash: '+$_.Exception.Message) }
    $result.passed=($result.errors.Count -eq 0 -and $result.brightness_60_verified -and $result.reset_0_verified -and
        $result.center_brighter -and $result.center_restored -and $result.source_unchanged -and $result.graceful_exit)
    $result.finished_at=(Get-Date).ToUniversalTime().ToString('o')
    Save-Result
}
$result | ConvertTo-Json -Depth 14
if ($result.passed) { exit 0 } else { exit 1 }
