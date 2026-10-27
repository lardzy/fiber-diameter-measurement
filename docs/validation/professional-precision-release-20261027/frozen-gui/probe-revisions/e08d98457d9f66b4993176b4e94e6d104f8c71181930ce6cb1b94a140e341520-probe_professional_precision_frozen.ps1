# Windows PowerShell 5.1; UTF-8 BOM required for the Chinese UIA names.
# Run as the ordinary interactive user in Session 1. This tests only the
# extracted EXE and a synthetic project; it never enters capture/device controls.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][string]$Project,
    [Parameter(Mandatory=$true)][string]$Image,
    [Parameter(Mandatory=$true)][string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
$allowedRoot = 'C:\tmp\fdm-professional-precision-release-20261027\'
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
$Project = Assert-TaskPath $Project
$Image = Assert-TaskPath $Image
$OutputRoot = Assert-TaskPath $OutputRoot
$exe = Get-Item -LiteralPath $Executable
$projectFile = Get-Item -LiteralPath $Project
if ($exe.PSIsContainer -or $exe.Name -ne 'FiberDiameterMeasurement.exe' -or
    $projectFile.PSIsContainer -or $projectFile.Extension -ne '.fdmproj') {
    throw 'An extracted FiberDiameterMeasurement.exe and a synthetic .fdmproj are required.'
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
public sealed class MetrologyWindow { public long Handle; public long Owner; }
public sealed class MetrologyCapture {
    public int Width; public int Height; public double CenterMean; public int CenterColors;
}
public static class MetrologyProbeNative {
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
    public static MetrologyWindow[] Windows(uint pid) {
        var result = new List<MetrologyWindow>();
        EnumWindows(delegate(IntPtr h, IntPtr unused) {
            uint actual; GetWindowThreadProcessId(h, out actual);
            if (actual == pid && IsWindowVisible(h))
                result.Add(new MetrologyWindow {Handle=h.ToInt64(),Owner=GetWindow(h,4).ToInt64()});
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
    public static MetrologyCapture Capture(uint pid, long handle, string path) {
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
                return new MetrologyCapture {Width=bitmap.Width,Height=bitmap.Height,
                    CenterMean=sum/count,CenterColors=colors.Count};
            }
        }
    }
}
'@
[void][MetrologyProbeNative]::SetProcessDPIAware()
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
    theme_mode='light'; selected_capture_device_id='professional-probe-unavailable'
    digital_slide_motor_output_enabled=$false; digital_slide_render_cache_gib=0
})
$result = [ordered]@{
    schema_version=1; passed=$false; started_at=(Get-Date).ToUniversalTime().ToString('o')
    scope='Frozen EXE precision UIA smoke; independent synthetic project and actual PNG/CSV exports; no installation or device accuracy claim'
    executable=$Executable; executable_sha256=(Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
    project=$Project; image=$Image; source_sha256_before=(Get-FileHash -LiteralPath $Image -Algorithm SHA256).Hash
    source_sha256_after=$null; source_unchanged=$false; run_directory=$runDirectory
    session_id=1; process_id=$null; controls=@(); snapshots=@(); popup_candidates=@(); errors=@()
    process_ids=@(); exits=@(); project_snapshots=@(); settings=@(); images=@(); csv=@(); steps=@()
    graceful_exit=$false; exit_code=$null; process_still_running=$false; forced_termination=$false
}
$process = $null
$mainHandle = 0L
function Save-Result { Write-Json (Join-Path $runDirectory 'result.json') $result }
function Get-OwnedRoots {
    foreach ($window in [MetrologyProbeNative]::Windows([uint32]$process.Id)) {
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
        try {$elements=$root.element.FindAll([System.Windows.Automation.TreeScope]::Subtree,$condition)} catch {
            $cause=$_.Exception
            while($cause.InnerException){$cause=$cause.InnerException}
            if($cause -is [Runtime.InteropServices.COMException] -or $cause -is [System.Windows.Automation.ElementNotAvailableException]){
                # Qt can invalidate its provider while a popup closes. The
                # caller's bounded wait takes a fresh PID-owned tree next time.
                $result.steps += [ordered]@{action='transient-uia-read';control=$Name;error=$cause.Message}
                continue
            }
            throw
        }
        foreach ($element in $elements) {
            $key=[string]::Join(',',$element.GetRuntimeId())
            if (-not $seen.ContainsKey($key) -and $element.Current.IsEnabled -and -not $element.Current.IsOffscreen) {
                $seen[$key]=$true; $element
            }
        }
    }
}
function Wait-Control([string]$Name,[int]$Seconds=20,[System.Windows.Automation.ControlType]$ControlType=$null) {
    $deadline=[DateTime]::UtcNow.AddSeconds($Seconds)
    do {
        if ($process.HasExited) { throw "EXE exited before control appeared: $Name" }
        $found=@(Find-Owned $Name)
        if($null -ne $ControlType){$found=@($found|Where-Object{$_.Current.ControlType -eq $ControlType})}
        if($Name -eq '专业测量'){$found=@($found|Where-Object{$_.Current.ControlType -in @([System.Windows.Automation.ControlType]::CheckBox,[System.Windows.Automation.ControlType]::Button)})}
        if($Name -eq '搜索半宽'){$found=@($found|Where-Object{$_.Current.ControlType -eq [System.Windows.Automation.ControlType]::Spinner})}
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
function Capture([string]$Phase) {
    Start-Sleep -Milliseconds 500
    $path=Join-Path $runDirectory ($Phase+'.png')
    $metrics=[MetrologyProbeNative]::Capture([uint32]$process.Id,$mainHandle,$path)
    $entry=[ordered]@{phase=$Phase; file=[IO.Path]::GetFileName($path); width=$metrics.Width; height=$metrics.Height;
        sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash;
        center_sha256=(Get-FileHash -LiteralPath ([IO.Path]::ChangeExtension($path,'.center.png')) -Algorithm SHA256).Hash;
        center_mean=$metrics.CenterMean; center_colors=$metrics.CenterColors; source='PID-owned PrintWindow; fixed center ROI'}
    $result.snapshots += $entry
    Save-Result
    if ($metrics.Width -lt 100 -or $metrics.Height -lt 100) { throw 'Owned window capture is too small' }
    return $entry
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class MetrologyInput {
    [StructLayout(LayoutKind.Sequential)] struct Point { public int X; public int Y; }
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hwnd,out uint pid);
    [DllImport("user32.dll")] static extern bool ScreenToClient(IntPtr hwnd,ref Point p);
    [DllImport("user32.dll")] static extern bool PostMessage(IntPtr hwnd,uint msg,IntPtr wp,IntPtr lp);
    [DllImport("user32.dll")] static extern bool IsChild(IntPtr parent,IntPtr child);
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern int GetClassName(IntPtr hwnd,StringBuilder name,int n);
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern IntPtr SendMessage(IntPtr hwnd,uint msg,IntPtr wp,string text);
    static void Check(uint pid,IntPtr hwnd) {uint actual;GetWindowThreadProcessId(hwnd,out actual);if(actual!=pid)throw new InvalidOperationException("Foreign window");}
    public static void Mouse(uint pid,long window,int x,int y,uint message) {
        var h=new IntPtr(window);Check(pid,h);var p=new Point{X=x,Y=y};
        if(!ScreenToClient(h,ref p))throw new InvalidOperationException("ScreenToClient failed");
        if(!PostMessage(h,message,new IntPtr(message==0x0202?0:1),new IntPtr((p.Y<<16)|(p.X&65535))))throw new InvalidOperationException("Mouse message failed");
    }
    public static void Enter(uint pid,long window) {
        var h=new IntPtr(window);Check(pid,h);
        PostMessage(h,0x0100,new IntPtr(0x0D),new IntPtr(1));
        PostMessage(h,0x0101,new IntPtr(0x0D),new IntPtr(unchecked((int)0xC0000001)));
    }
    public static void Filename(uint pid,long dialog,long edit,string text) {
        var h=new IntPtr(edit); Check(pid,h);Check(pid,new IntPtr(dialog));
        var n=new StringBuilder(100);GetClassName(h,n,100);
        if(!IsChild(new IntPtr(dialog),h)||n.ToString()!="Edit")throw new InvalidOperationException("Not owned filename edit");
        SendMessage(h,0x000C,IntPtr.Zero,text);
    }
    public static void DialogSave(uint pid,long dialog,long button) {
        var h=new IntPtr(button);Check(pid,h);Check(pid,new IntPtr(dialog));
        var n=new StringBuilder(100);GetClassName(h,n,100);
        if(!IsChild(new IntPtr(dialog),h)||n.ToString()!="Button")throw new InvalidOperationException("Not owned save button");
        PostMessage(h,0x00F5,IntPtr.Zero,IntPtr.Zero);
    }
}
'@
$result.schema_version=1
$result.steps=@();$result.project_snapshots=@();$result.process_ids=@();$result.exits=@();$result.tables=@()
function All-Owned {
    foreach($root in @(Get-OwnedRoots)){
        try {
            $items=@($root.element.FindAll([System.Windows.Automation.TreeScope]::Subtree,[System.Windows.Automation.Condition]::TrueCondition)|Where-Object{$_.Current.ProcessId -eq $process.Id -and -not $_.Current.IsOffscreen})
        } catch {
            $cause=$_.Exception;while($cause.InnerException){$cause=$cause.InnerException}
            if($cause -is [Runtime.InteropServices.COMException] -or $cause -is [System.Windows.Automation.ElementNotAvailableException]){
                $result.steps += [ordered]@{action='transient-uia-tree';error=$cause.Message}
                continue
            }
            throw
        }
        foreach($e in $items){$e}
    }
}
function Owned-Class([string]$Class) {
    $deadline=[DateTime]::UtcNow.AddSeconds(10)
    do {
        $found=@(All-Owned | Where-Object {$_.Current.ClassName -eq $Class})
        if($found.Count -eq 1){return $found[0]}
        if($found.Count -gt 1){throw "Ambiguous visible owned $Class"}
        Start-Sleep -Milliseconds 100
    }while([DateTime]::UtcNow -lt $deadline)
    throw "Expected one visible owned $Class, got $($found.Count)"
}
function Tree-Record([string]$Name) {
    $rows=@(All-Owned | ForEach-Object {[ordered]@{name=$_.Current.Name;class=$_.Current.ClassName;type=$_.Current.ControlType.ProgrammaticName;id=$_.Current.AutomationId;enabled=$_.Current.IsEnabled;bounds=$_.Current.BoundingRectangle.ToString()}})
    Write-Json (Join-Path $runDirectory ($Name+'.json')) $rows
}
function Click([double]$X,[double]$Y) {
    $r=(Owned-Class 'DocumentCanvas').Current.BoundingRectangle
    if($X -lt $r.Left -or $X -ge $r.Right -or $Y -lt $r.Top -or $Y -ge $r.Bottom){throw 'Point outside owned canvas'}
    [MetrologyInput]::Mouse([uint32]$process.Id,$mainHandle,[int]$X,[int]$Y,0x0201)
    Start-Sleep -Milliseconds 80
    [MetrologyInput]::Mouse([uint32]$process.Id,$mainHandle,[int]$X,[int]$Y,0x0202)
    Start-Sleep -Milliseconds 180
}
function Drag([double]$X,[double]$Y,[double]$Dx,[double]$Dy) {
    $r=(Owned-Class 'DocumentCanvas').Current.BoundingRectangle
    if($X -lt $r.Left -or $X+$Dx -ge $r.Right -or $Y -lt $r.Top -or $Y+$Dy -ge $r.Bottom){throw 'Drag outside owned canvas'}
    [MetrologyInput]::Mouse([uint32]$process.Id,$mainHandle,[int]$X,[int]$Y,0x0201)
    Start-Sleep -Milliseconds 120
    for($i=1;$i -le 8;$i++){
        [MetrologyInput]::Mouse([uint32]$process.Id,$mainHandle,[int]($X+$Dx*$i/8),[int]($Y+$Dy*$i/8),0x0200)
        Start-Sleep -Milliseconds 40
    }
    [MetrologyInput]::Mouse([uint32]$process.Id,$mainHandle,[int]($X+$Dx),[int]($Y+$Dy),0x0202)
    Start-Sleep -Milliseconds 250
}
function Wait-Root([string]$Name,[int]$Seconds=20) {
    $deadline=[DateTime]::UtcNow.AddSeconds($Seconds)
    do {
        $roots=@(Get-OwnedRoots|Where-Object{$_.element.Current.Name -eq $Name -and $_.element.Current.ControlType -eq [System.Windows.Automation.ControlType]::Window})
        if($roots.Count -eq 1){return $roots[0]}
        if($roots.Count -gt 1){throw "Ambiguous owned dialog: $Name"}
        if($process.HasExited){throw "EXE exited before $Name"}
        Start-Sleep -Milliseconds 100
    }while([DateTime]::UtcNow -lt $deadline)
    throw "Timed out waiting for owned dialog: $Name"
}
function Find-In($Root,[string]$Name,[System.Windows.Automation.ControlType]$Type=$null,[string]$IdSuffix='') {
    try {$items=$Root.element.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition)}catch{
        $cause=$_.Exception;while($cause.InnerException){$cause=$cause.InnerException}
        if($cause -is [Runtime.InteropServices.COMException] -or $cause -is [System.Windows.Automation.ElementNotAvailableException]){
            $result.steps+=@{action='transient-scoped-uia-read';control=$Name;id=$IdSuffix;error=$cause.Message};return
        }
        throw
    }
    foreach($e in $items){
        if($e.Current.ProcessId -ne $process.Id -or -not $e.Current.IsEnabled -or $e.Current.IsOffscreen){continue}
        if($Name -and $e.Current.Name -ne $Name){continue}
        if($null -ne $Type -and $e.Current.ControlType -ne $Type){continue}
        if($IdSuffix -and -not $e.Current.AutomationId.EndsWith($IdSuffix,[StringComparison]::Ordinal)){continue}
        $e
    }
}
function Wait-In($Root,[string]$Name,[System.Windows.Automation.ControlType]$Type=$null,[string]$IdSuffix='') {
    $deadline=[DateTime]::UtcNow.AddSeconds(20)
    do{
        $items=@(Find-In $Root $Name $Type $IdSuffix)
        if($items.Count -eq 1){return $items[0]}
        if($items.Count -gt 1){throw "Ambiguous scoped control: $Name/$IdSuffix"}
        Start-Sleep -Milliseconds 100
    }while([DateTime]::UtcNow -lt $deadline)
    throw "Scoped control not found: $Name/$IdSuffix"
}
function Main-Root {
    $roots=@(Get-OwnedRoots|Where-Object{$_.window.Handle -eq $mainHandle})
    if($roots.Count -ne 1){throw 'Expected current PID-owned main root'}
    return $roots[0]
}
function Click-In($Root,$Element) {
    if($Element.Current.ProcessId -ne $process.Id -or $Root.element.Current.ProcessId -ne $process.Id -or -not $Element.Current.IsEnabled -or $Element.Current.IsOffscreen){throw 'Foreign, disabled or offscreen UIA element'}
    $name=$Element.Current.Name;$id=$Element.Current.AutomationId
    $r=$Element.Current.BoundingRectangle;$box=$Root.element.Current.BoundingRectangle
    $x=$r.Left+$r.Width/2;$y=$r.Top+$r.Height/2
    if($r.IsEmpty -or -not $box.Contains($x,$y)){throw 'Control not inside specified owned window'}
    $result.steps+=@{action='before-owned-post-click';name=$name;id=$id;root_name=$Root.element.Current.Name;
        root_handle=$Root.window.Handle;pid=$process.Id;utc=[DateTime]::UtcNow.ToString('o')};Save-Result
    [MetrologyInput]::Mouse([uint32]$process.Id,$Root.window.Handle,[int]$x,[int]$y,0x0201)
    Start-Sleep -Milliseconds 80
    [MetrologyInput]::Mouse([uint32]$process.Id,$Root.window.Handle,[int]$x,[int]$y,0x0202)
    $result.steps+=@{action='owned-post-click-queued';name=$name;root_handle=$Root.window.Handle;pid=$process.Id;utc=[DateTime]::UtcNow.ToString('o')};Save-Result
}
function Click-Owned($Element) {
    # Only popup menu items use inferred roots. Dialog and main-window controls
    # always pass their known HWND through Click-In. Qt exposes owned dialogs
    # in the main accessibility subtree, so neither subtree nor area is enough.
    $roots=@(Get-OwnedRoots)
    $walker=[System.Windows.Automation.TreeWalker]::RawViewWalker
    $ancestor=$Element
    for($depth=0;$null -ne $ancestor -and $depth -lt 30;$depth++){
        $handle=[long]$ancestor.Current.NativeWindowHandle
        $match=@($roots|Where-Object{$_.window.Handle -eq $handle -and $_.window.Handle -ne $mainHandle})
        if($match.Count -eq 1){Click-In $match[0] $Element;return}
        $ancestor=$walker.GetParent($ancestor)
    }
    # Some Qt menus expose a zero native handle through UIA. A unique non-main
    # popup containing this runtime ID is sufficient; never fall back to main.
    $key=[string]::Join(',',$Element.GetRuntimeId());$matches=@()
    foreach($root in @($roots|Where-Object{$_.window.Handle -ne $mainHandle})){
        $items=$root.element.FindAll([System.Windows.Automation.TreeScope]::Subtree,[System.Windows.Automation.Condition]::TrueCondition)
        foreach($item in $items){if([string]::Join(',',$item.GetRuntimeId()) -eq $key){$matches+=@($root);break}}
    }
    if($matches.Count -ne 1){throw 'Cannot identify unique non-main popup root for menu item'}
    Click-In $matches[0] $Element
}
function Snapshot-Dialog([string]$Phase,$Root) {
    Start-Sleep -Milliseconds 400
    $path=Join-Path $runDirectory ($Phase+'.png')
    $metrics=[MetrologyProbeNative]::Capture([uint32]$process.Id,$Root.window.Handle,$path)
    $result.snapshots+=@{phase=$Phase;file=[IO.Path]::GetFileName($path);sha256=(Get-FileHash $path).Hash;width=$metrics.Width;height=$metrics.Height;source='PID-owned dialog PrintWindow'}
    Save-Result
}
function Open-Preferences {
    Click-In (Main-Root) (Wait-Control '设置' -ControlType ([System.Windows.Automation.ControlType]::Button))
    $dialog=Wait-Root '首选项'
    $result.steps+=@{action='preferences-dialog-observed';pid=$process.Id;handle=$dialog.window.Handle};Save-Result
    $item=Wait-In $dialog '专业测量' ([System.Windows.Automation.ControlType]::ListItem)
    Click-In $dialog $item
    $spin=Wait-In $dialog '' ([System.Windows.Automation.ControlType]::Spinner) 'professionalMeasurementLabelDecimals'
    Record-Control 'professional-precision-spin' $spin
    return $dialog
}
function Precision-Spin($Dialog) {
    return Wait-In $Dialog '' ([System.Windows.Automation.ControlType]::Spinner) 'professionalMeasurementLabelDecimals'
}
function Precision-Value($Dialog) {
    $spin=Precision-Spin $Dialog;$pattern=$null
    if($spin.TryGetCurrentPattern([System.Windows.Automation.RangeValuePattern]::Pattern,[ref]$pattern)){return [int]$pattern.Current.Value}
    if($spin.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern,[ref]$pattern)){return [int]$pattern.Current.Value}
    throw 'Precision spinner has no readable numeric UIA pattern'
}
function Wait-Precision($Dialog,[int]$Value) {
    $deadline=[DateTime]::UtcNow.AddSeconds(10)
    do{if((Precision-Value $Dialog) -eq $Value){return};Start-Sleep -Milliseconds 100}while([DateTime]::UtcNow -lt $deadline)
    throw "Precision spinner did not become $Value"
}
function Set-Precision($Dialog,[int]$Value) {
    if($Value -lt 0 -or $Value -gt 8){throw 'Out-of-range test precision'}
    $spin=Precision-Spin $Dialog;$pattern=$null
    if($spin.TryGetCurrentPattern([System.Windows.Automation.RangeValuePattern]::Pattern,[ref]$pattern)){$pattern.SetValue([double]$Value)}
    elseif($spin.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern,[ref]$pattern)){$pattern.SetValue([string]$Value)}
    else{throw 'Precision spinner has no writable UIA pattern'}
    Wait-Precision $Dialog $Value
    $result.steps+=@{action='set-precision';value=$Value};Save-Result
}
function Disk-Precision {
    $data=Get-Content (Join-Path $appConfig 'settings.json') -Raw -Encoding UTF8|ConvertFrom-Json
    if($null -eq $data.professional_measurement_label_decimals){return 2}
    return [int]$data.professional_measurement_label_decimals
}
function Apply-Precision($Dialog,[int]$Value,[string]$Phase) {
    Click-In $Dialog (Wait-In $Dialog '应用' ([System.Windows.Automation.ControlType]::Button))
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    do{
        $data=Get-Content (Join-Path $appConfig 'settings.json') -Raw -Encoding UTF8|ConvertFrom-Json
        if($null -ne $data.professional_measurement_label_decimals -and [int]$data.professional_measurement_label_decimals -eq $Value){break}
        Start-Sleep -Milliseconds 100
    }while([DateTime]::UtcNow -lt $deadline)
    if($null -eq $data.professional_measurement_label_decimals -or [int]$data.professional_measurement_label_decimals -ne $Value){throw 'Applied precision was not persisted'}
    $file='settings-'+$Phase+'.json';Copy-Item (Join-Path $appConfig 'settings.json') (Join-Path $runDirectory $file)
    $result.settings+=@{phase=$Phase;file=$file;decimals=$Value;sha256=(Get-FileHash (Join-Path $runDirectory $file)).Hash}
    Save-Result
}
function Close-Preferences($Dialog) {
    Click-In $Dialog (Wait-In $Dialog '取消' ([System.Windows.Automation.ControlType]::Button))
    $deadline=[DateTime]::UtcNow.AddSeconds(10)
    do{if(@(Get-OwnedRoots|Where-Object{$_.element.Current.Name -eq '首选项'}).Count -eq 0){Start-Sleep -Milliseconds 200;return};Start-Sleep -Milliseconds 100}while([DateTime]::UtcNow -lt $deadline)
    throw 'Preferences did not close'
}
function Native-Save([string]$Title,[string]$Path) {
    $result.steps+=@{action='waiting-native-save';title=$Title};Save-Result
    $dialog=Wait-Root $Title
    $children=@($dialog.element.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition))
    $edits=@($children|Where-Object{$_.Current.ClassName -eq 'Edit' -and $_.Current.AutomationId -eq '1001' -and $_.Current.ProcessId -eq $process.Id})
    $buttons=@($children|Where-Object{$_.Current.ClassName -eq 'Button' -and $_.Current.AutomationId -eq '1' -and $_.Current.Name -match '^(保存|Save)' -and $_.Current.ProcessId -eq $process.Id})
    if($edits.Count -ne 1 -or $buttons.Count -ne 1){throw 'Native save dialog controls are ambiguous'}
    [MetrologyInput]::Filename([uint32]$process.Id,$dialog.window.Handle,$edits[0].Current.NativeWindowHandle,$Path)
    [MetrologyInput]::DialogSave([uint32]$process.Id,$dialog.window.Handle,$buttons[0].Current.NativeWindowHandle)
    $result.steps+=@{action='native-save-queued';title=$Title;path=$Path;pid=$process.Id;handle=$dialog.window.Handle};Save-Result
}
function Export-Overlay([int]$Decimals) {
    Click-In (Main-Root) (Wait-Control '导出' -ControlType ([System.Windows.Automation.ControlType]::Button))
    Click-Owned (Wait-Control '导出测量叠加图' -ControlType ([System.Windows.Automation.ControlType]::MenuItem))
    $options=Wait-Root '导出选项'
    Snapshot-Dialog ('export-options-'+$Decimals) $options
    Click-In $options (Wait-In $options '' ([System.Windows.Automation.ControlType]::Button) 'exportOptionsSaveButton')
    $file='overlay-'+$Decimals+'.png';$path=Join-Path $runDirectory $file
    Native-Save '选择导出文件' $path
    $done=Wait-Root '导出完成' 45
    $buttons=@(Find-In $done '' ([System.Windows.Automation.ControlType]::Button)|Where-Object{$_.Current.Name -in @('确定','OK','&OK')})
    if($buttons.Count -ne 1){throw 'Export completion button ambiguous'}
    Click-In $done $buttons[0]
    if(-not(Test-Path $path)){throw 'Export completion did not create PNG'}
    $result.images+=@{file=$file;decimals=$Decimals;sha256=(Get-FileHash $path).Hash}
    Save-Result
}
function Export-Csv([string]$Phase) {
    Click-In (Main-Root) (Wait-Control '导出明细')
    $file='professional-'+$Phase+'.csv';$path=Join-Path $runDirectory $file
    Native-Save '导出专业测量明细' $path
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    while(-not(Test-Path $path)){if([DateTime]::UtcNow -gt $deadline){throw 'CSV not created'};Start-Sleep -Milliseconds 100}
    Start-Sleep -Milliseconds 150
    $result.csv+=@{phase=$Phase;file=$file;sha256=(Get-FileHash $path).Hash;rows=@(Import-Csv $path -Encoding UTF8).Count}
    Save-Result
}
function Save-ProjectSnapshot([string]$Phase) {
    # request_save always publishes through atomic_write_json, even if clean.
    # The old "已保存" UI text alone cannot acknowledge this invocation.
    $before=Get-Item -LiteralPath $Project
    $beforeTicks=$before.LastWriteTimeUtc.Ticks
    $beforeHash=(Get-FileHash -LiteralPath $Project -Algorithm SHA256).Hash
    Click-In (Main-Root) (Wait-Control '保存' -ControlType ([System.Windows.Automation.ControlType]::Button))
    $deadline=[DateTime]::UtcNow.AddSeconds(30)
    do {
        $after=Get-Item -LiteralPath $Project
        if($after.LastWriteTimeUtc.Ticks -gt $beforeTicks){break}
        if($process.HasExited){throw 'EXE exited before project publication'}
        Start-Sleep -Milliseconds 100
    }while([DateTime]::UtcNow -lt $deadline)
    if($after.LastWriteTimeUtc.Ticks -le $beforeTicks){throw 'No new project publication for this Save invocation'}
    [void](Wait-Control '已保存' 30)
    $publishedTicks=$after.LastWriteTimeUtc.Ticks
    $file=$Phase+'.fdmproj';$target=Join-Path $runDirectory $file
    # Disallow concurrent writes/replacement while taking the evidence copy.
    $reader=[IO.File]::Open($Project,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
    try {
        $writer=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try {$reader.CopyTo($writer)} finally {$writer.Dispose()}
    } finally {$reader.Dispose()}
    $sourceHash=(Get-FileHash -LiteralPath $Project -Algorithm SHA256).Hash
    $copyHash=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
    if($sourceHash -ne $copyHash -or (Get-Item -LiteralPath $Project).LastWriteTimeUtc.Ticks -ne $publishedTicks){throw 'Project changed during evidence snapshot'}
    $result.project_snapshots+=@{phase=$Phase;file=$file;sha256=$copyHash;source_sha256=$sourceHash;
        before_sha256=$beforeHash;before_last_write_utc_ticks=[string]$beforeTicks;
        published_last_write_utc_ticks=[string]$publishedTicks;publication_barrier='strictly newer LastWriteTimeUtc, then saved UI, exclusive-read copy'}
    Save-Result
}
function Start-App {
    $script:process=[Diagnostics.Process]::new();$process.StartInfo=[Diagnostics.ProcessStartInfo]::new()
    $process.StartInfo.FileName=$Executable;$process.StartInfo.Arguments='"'+$Project+'"';$process.StartInfo.WorkingDirectory=$exe.DirectoryName;$process.StartInfo.UseShellExecute=$false
    foreach($name in @('PSExecutionPolicyPreference','QT_SCREEN_SCALE_FACTORS','QT_FONT_DPI','PYTHONPATH','PYTHONHOME','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH')){$process.StartInfo.EnvironmentVariables.Remove($name)}
    $process.StartInfo.EnvironmentVariables['APPDATA']=$config;$process.StartInfo.EnvironmentVariables['LOCALAPPDATA']=$config
    $process.StartInfo.EnvironmentVariables['TEMP']=$temporary;$process.StartInfo.EnvironmentVariables['TMP']=$temporary
    $process.StartInfo.EnvironmentVariables['QT_ACCESSIBILITY']='1';$process.StartInfo.EnvironmentVariables['QT_QPA_PLATFORM']='windows'
    $process.StartInfo.EnvironmentVariables['QT_SCALE_FACTOR']=(96.0/[MetrologyProbeNative]::GetDpiForSystem()).ToString([Globalization.CultureInfo]::InvariantCulture)
    [void]$process.Start();$result.process_id=$process.Id;$result.process_ids+=@($process.Id);Save-Result
    [void](Wait-Control '专业测量' 90)
    $main=@(Get-OwnedRoots|Where-Object{$_.window.Owner -eq 0 -and $_.element.Current.ControlType -eq [System.Windows.Automation.ControlType]::Window})
    if($main.Count -ne 1){throw 'One owned main window required'}
    $script:mainHandle=$main[0].window.Handle
    [MetrologyProbeNative]::Place([uint32]$process.Id,$mainHandle,1600,1000)
    Click-In (Main-Root) (Wait-Control '专业测量');[void](Wait-Control '开始取点')
    Start-Sleep -Milliseconds 500
}
function Stop-App {
    if($null -eq $process){return}
    # Failure may leave a modal open, in which case main WM_CLOSE is ignored.
    # Cancel only this PID's non-main dialogs normally before closing main.
    for($attempt=0;$attempt -lt 3 -and -not $process.HasExited;$attempt++){
        $modals=@(Get-OwnedRoots|Where-Object{$_.window.Handle -ne $mainHandle -and $_.element.Current.ControlType -eq [System.Windows.Automation.ControlType]::Window -and $_.element.Current.Name -ne '未保存的改动'})
        if($modals.Count -eq 0){break}
        foreach($modal in $modals){
            $result.steps+=@{action='normal-close-owned-modal';name=$modal.element.Current.Name;handle=$modal.window.Handle;pid=$process.Id};Save-Result
            [void][MetrologyProbeNative]::Close([uint32]$process.Id,$modal.window.Handle)
        }
        Start-Sleep -Milliseconds 400
    }
    if(-not $process.HasExited){[void][MetrologyProbeNative]::Close([uint32]$process.Id,$mainHandle);[void]$process.WaitForExit(3000)}
    if(-not $process.HasExited){
        $prompt=@(Get-OwnedRoots|Where-Object{$_.element.Current.Name -eq '未保存的改动'})
        if($prompt.Count -eq 1){Click-In $prompt[0] (Wait-In $prompt[0] '保存' ([System.Windows.Automation.ControlType]::Button));$result.steps+=@{action='save-owned-project-on-close'};[void]$process.WaitForExit(15000)}
    }
    if(-not $process.HasExited){$result.process_still_running=$true;throw 'Owned EXE still running; no forced termination'}
    $result.exits+=@{pid=$process.Id;code=$process.ExitCode};$result.process_still_running=$false
    if($process.ExitCode -ne 0){throw 'Nonzero frozen EXE exit'}
    $script:process=$null
}
try {
    Start-App;Tree-Record '01-controls';Save-ProjectSnapshot 'baseline'
    $dialog=Open-Preferences;Wait-Precision $dialog 2;Snapshot-Dialog '02-default-two' $dialog;Close-Preferences $dialog
    [void](Capture '03-canvas-two');Export-Overlay 2;Export-Csv 'two'
    $dialog=Open-Preferences;Set-Precision $dialog 0;Apply-Precision $dialog 0 'zero';Snapshot-Dialog '04-zero-applied' $dialog;Close-Preferences $dialog
    [void](Capture '05-canvas-zero');Export-Overlay 0
    $dialog=Open-Preferences;Set-Precision $dialog 8;Snapshot-Dialog '06-eight-draft' $dialog;Close-Preferences $dialog
    if((Disk-Precision) -ne 0){throw 'Cancelled draft changed persisted precision'}
    $result.cancel_preserved_zero=$true
    $dialog=Open-Preferences;Wait-Precision $dialog 0
    Click-In $dialog (Wait-In $dialog '恢复本页默认值' ([System.Windows.Automation.ControlType]::Button));Wait-Precision $dialog 2
    if((Disk-Precision) -ne 0){throw 'Restore defaults applied without confirmation'}
    $result.restore_default_draft=$true;Snapshot-Dialog '07-restored-default-draft' $dialog
    Apply-Precision $dialog 2 'restored';Set-Precision $dialog 4;Apply-Precision $dialog 4 'four';Snapshot-Dialog '08-four-applied' $dialog;Close-Preferences $dialog
    [void](Capture '09-canvas-four');Export-Overlay 4;Export-Csv 'four';Save-ProjectSnapshot 'after-settings'
    Stop-App;Start-App
    $dialog=Open-Preferences;Wait-Precision $dialog 4;$result.reopened_decimals=Precision-Value $dialog
    Snapshot-Dialog '10-reopened-four' $dialog;Close-Preferences $dialog
    [void](Capture '11-reopened-canvas');Tree-Record '11-controls';Save-ProjectSnapshot 'reopened'
} catch {
    $result.errors+=$_.Exception.Message;$result.failure_stack=$_.ScriptStackTrace
    try{Tree-Record '99-failure-controls';[void](Capture '99-failure')}catch{}
} finally {
    try{Stop-App;$result.graceful_exit=$true}catch{$result.errors+=('Close: '+$_.Exception.Message)}
    $result.source_sha256_after=(Get-FileHash $Image).Hash
    $result.source_unchanged=($result.source_sha256_before -eq $result.source_sha256_after)
    $result.passed=($result.errors.Count -eq 0 -and $result.reopened_decimals -eq 4 -and $result.cancel_preserved_zero -and $result.restore_default_draft -and $result.graceful_exit -and $result.source_unchanged -and $result.exits.Count -eq 2 -and $result.images.Count -eq 3 -and $result.csv.Count -eq 2)
    $result.finished_at=[DateTime]::UtcNow.ToString('o');Save-Result
}
$result|ConvertTo-Json -Depth 14
if($result.passed){exit 0}else{exit 1}
