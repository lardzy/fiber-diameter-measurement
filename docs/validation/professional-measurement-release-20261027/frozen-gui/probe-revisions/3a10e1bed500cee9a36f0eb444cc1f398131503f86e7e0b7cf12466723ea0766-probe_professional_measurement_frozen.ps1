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
$allowedRoot = 'C:\tmp\fdm-professional-measurement-release-20261027\'
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
    scope='Frozen EXE UIA smoke; synthetic image/project; no hardware, installation or performance acceptance'
    executable=$Executable; executable_sha256=(Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
    project=$Project; image=$Image; source_sha256_before=(Get-FileHash -LiteralPath $Image -Algorithm SHA256).Hash
    source_sha256_after=$null; source_unchanged=$false; run_directory=$runDirectory
    session_id=1; process_id=$null; controls=@(); snapshots=@(); popup_candidates=@(); errors=@()
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
    if ($metrics.CenterColors -lt 8) { throw 'Center screenshot is blank/uninformative; inspect capture and fixture.' }
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
        foreach($e in $root.element.FindAll([System.Windows.Automation.TreeScope]::Subtree,[System.Windows.Automation.Condition]::TrueCondition)){
            if($e.Current.ProcessId -eq $process.Id -and -not $e.Current.IsOffscreen){$e}
        }
    }
}
function Owned-Class([string]$Class) {
    $found=@(All-Owned | Where-Object {$_.Current.ClassName -eq $Class})
    if($found.Count -ne 1){throw "Expected one visible owned $Class, got $($found.Count)"}
    return $found[0]
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
function Saved([string]$Name,[int]$Count) {
    $before=(Get-Item -LiteralPath $Project).LastWriteTimeUtc.Ticks
    Invoke-Owned (Wait-Control '保存')
    $deadline=[DateTime]::UtcNow.AddSeconds(30)
    do{
        Start-Sleep -Milliseconds 100
        $payload=Get-Content -LiteralPath $Project -Raw -Encoding UTF8|ConvertFrom-Json
        if((Get-Item -LiteralPath $Project).LastWriteTimeUtc.Ticks -gt $before -and @($payload.documents[0].professional_measurements).Count -eq $Count){break}
    }while([DateTime]::UtcNow -lt $deadline)
    if((Get-Item -LiteralPath $Project).LastWriteTimeUtc.Ticks -le $before -or @($payload.documents[0].professional_measurements).Count -ne $Count){throw "Project did not finish saving $Name"}
    $file=$Name+'.fdmproj';Copy-Item -LiteralPath $Project -Destination (Join-Path $runDirectory $file)
    $result.project_snapshots += [ordered]@{phase=$Name;file=$file;count=$Count;sha256=(Get-FileHash -LiteralPath (Join-Path $runDirectory $file)).Hash}
    Save-Result
    return $payload.documents[0]
}
function Read-Table([string]$Phase) {
    $rows=@(All-Owned | Where-Object {$_.Current.ControlType -eq [System.Windows.Automation.ControlType]::DataItem -or $_.Current.ControlType -eq [System.Windows.Automation.ControlType]::Text} | ForEach-Object {$_.Current.Name})
    $result.tables += [ordered]@{phase=$Phase;names=$rows}
    Save-Result
}
function Combo-Value($Box) {
    $pattern=$null
    if($Box.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern,[ref]$pattern)){return [string]$pattern.Current.Value}
    $pattern=$null
    if($Box.TryGetCurrentPattern([System.Windows.Automation.SelectionPattern]::Pattern,[ref]$pattern)){
        $selected=@($pattern.GetCurrentSelection())
        if($selected.Count -eq 1){return [string]$selected[0].Current.Name}
    }
    foreach($supported in $Box.GetSupportedPatterns()){
        if($supported.ProgrammaticName -eq 'LegacyIAccessiblePatternIdentifiers.Pattern'){
            return [string]$Box.GetCurrentPattern($supported).Current.Value
        }
    }
    throw 'Professional tool combo value is not readable'
}
function Select-Tool([string]$Name) {
    $dock=@(All-Owned|Where-Object{$_.Current.AutomationId -match 'professionalMeasurementDock$'})
    if($dock.Count -ne 1){throw 'Professional tool dock unavailable'}
    $boxes=@($dock[0].FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition)|Where-Object {$_.Current.ControlType -eq [System.Windows.Automation.ControlType]::ComboBox -and -not $_.Current.IsOffscreen}|Sort-Object {$_.Current.BoundingRectangle.Top})
    if($boxes.Count -lt 1){throw 'No professional tool combobox'}
    $box=$boxes[0];$before=Combo-Value $box
    if($before -eq $Name){return}
    $pattern=$null
    if($box.TryGetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern,[ref]$pattern)){$pattern.Expand()}else{Invoke-Owned $box}
    $item=Wait-Control $Name
    # SelectionItem.Select only highlights a Qt popup item. A real owned-window
    # item click is required to activate the QComboBox and close its popup.
    $ancestor=$item
    while($null -ne $ancestor -and $ancestor.Current.NativeWindowHandle -eq 0){
        $ancestor=[System.Windows.Automation.TreeWalker]::ControlViewWalker.GetParent($ancestor)
    }
    if($null -eq $ancestor -or $ancestor.Current.ProcessId -ne $process.Id){throw 'No owned HWND for combo popup item'}
    $r=$item.Current.BoundingRectangle;$owner=$ancestor.Current.BoundingRectangle
    $x=$r.Left+$r.Width/2;$y=$r.Top+$r.Height/2
    if($r.IsEmpty -or -not $owner.Contains($x,$y)){throw 'Combo item is outside its owned popup'}
    [MetrologyInput]::Mouse([uint32]$process.Id,[long]$ancestor.Current.NativeWindowHandle,[int]$x,[int]$y,0x0201)
    Start-Sleep -Milliseconds 80
    [MetrologyInput]::Mouse([uint32]$process.Id,[long]$ancestor.Current.NativeWindowHandle,[int]$x,[int]$y,0x0202)
    $deadline=[DateTime]::UtcNow.AddSeconds(5)
    do{
        Start-Sleep -Milliseconds 100
        $value=Combo-Value $box
        $pattern=$null;$closed=$true
        if($box.TryGetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern,[ref]$pattern)){$closed=$pattern.Current.ExpandCollapseState -eq [System.Windows.Automation.ExpandCollapseState]::Collapsed}
        if($value -eq $Name -and $closed){break}
    }while([DateTime]::UtcNow -lt $deadline)
    if($value -ne $Name -or -not $closed){throw "Tool selection was not activated: $value; expected $Name"}
    $result.steps += [ordered]@{action='select-professional-tool';before=$before;after=$value;popup_closed=$closed}
    Save-Result
}
function Select-Caliper {
    Select-Tool '直线卡尺：带状找边'
    [void](Wait-Control '搜索半宽' -ControlType ([System.Windows.Automation.ControlType]::Spinner))
}
function Caliper-Preview($Document) {
    Invoke-Owned (Wait-Control '开始取点')
    $r=(Owned-Class 'DocumentCanvas').Current.BoundingRectangle
    $view=$Document.view_state
    $y=$r.Top+$view.pan.y+400*$view.zoom
    Click ($r.Left+$view.pan.x+180*$view.zoom) $y
    Click ($r.Left+$view.pan.x+680*$view.zoom) $y
    [void](Wait-Control '确认拟合并保存测量' 30)
}
function Csv-Export {
    Invoke-Owned (Wait-Control '导出明细')
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    do{
        $dialogs=@(Get-OwnedRoots|Where-Object {$_.element.Current.Name -eq '导出专业测量明细'})
        if($dialogs.Count -eq 1){break};Start-Sleep -Milliseconds 100
    }while([DateTime]::UtcNow -lt $deadline)
    if($dialogs.Count -ne 1){throw 'CSV save dialog unavailable'}
    $dialog=$dialogs[0];$children=@($dialog.element.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition))
    $edits=@($children|Where-Object{$_.Current.ClassName -eq 'Edit' -and $_.Current.AutomationId -eq '1001' -and $_.Current.ProcessId -eq $process.Id})
    $buttons=@($children|Where-Object{$_.Current.ClassName -eq 'Button' -and $_.Current.AutomationId -eq '1' -and $_.Current.Name -match '^(保存|Save)' -and $_.Current.ProcessId -eq $process.Id})
    if($edits.Count -ne 1 -or $buttons.Count -ne 1){throw 'CSV dialog controls ambiguous'}
    $path=Join-Path $runDirectory 'professional.csv'
    [MetrologyInput]::Filename([uint32]$process.Id,$dialog.window.Handle,$edits[0].Current.NativeWindowHandle,$path)
    [MetrologyInput]::DialogSave([uint32]$process.Id,$dialog.window.Handle,$buttons[0].Current.NativeWindowHandle)
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    while(-not (Test-Path -LiteralPath $path)){if([DateTime]::UtcNow -gt $deadline){throw 'CSV export timeout'};Start-Sleep -Milliseconds 100}
    $result.csv=[ordered]@{file='professional.csv';sha256=(Get-FileHash -LiteralPath $path).Hash;rows=@(Import-Csv -LiteralPath $path -Encoding UTF8).Count}
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
    Invoke-Owned (Wait-Control '专业测量');[void](Wait-Control '开始取点')
    Start-Sleep -Milliseconds 750
    Record-Control 'canvas' (Owned-Class 'DocumentCanvas')
}
function Stop-App {
    if($null -ne $process -and -not $process.HasExited){[void][MetrologyProbeNative]::Close([uint32]$process.Id,$mainHandle);[void]$process.WaitForExit(15000)}
    if($null -ne $process){$result.process_still_running=-not $process.HasExited;if(-not $process.HasExited){throw 'Owned EXE still running; no forced termination'};$result.exits+=@{pid=$process.Id;code=$process.ExitCode};$result.exit_code=$process.ExitCode;if($process.ExitCode -ne 0){throw 'Nonzero frozen EXE exit'}}
}
try {
    Start-App;Tree-Record '01-controls';[void](Capture '01-professional-entry')
    Invoke-Owned (Wait-Control '开始取点')
    $r=(Owned-Class 'DocumentCanvas').Current.BoundingRectangle
    Click ($r.Left+$r.Width*.35) ($r.Top+$r.Height*.3)
    Click ($r.Left+$r.Width*.55) ($r.Top+$r.Height*.3)
    $created=Saved '02-created-distance' 1
    $d=$created.professional_measurements[0];$f=@($created.professional_features|Where-Object{$_.id -eq $d.feature_ids[0]})[0]
    if($d.kind -ne 'distance' -or $d.annotation_layout.leader_style -ne 'dual'){throw 'New distance is not dual'}
    [void](Capture '02-created-distance');Read-Table 'distance'
    $centerX=($f.points_px[0].x+$f.points_px[1].x)/2+$d.annotation_layout.offset_px.x
    $centerY=($f.points_px[0].y+$f.points_px[1].y)/2+$d.annotation_layout.offset_px.y
    $r=(Owned-Class 'DocumentCanvas').Current.BoundingRectangle;$view=$created.view_state
    Drag ($r.Left+$view.pan.x+$centerX*$view.zoom) ($r.Top+$view.pan.y+$centerY*$view.zoom) 50 30
    $moved=Saved '03-dragged-label' 1;[void](Capture '03-dragged-label')
    Invoke-Owned (Wait-Control '撤销');$restored=Saved '04-undo-label' 1
    $a=$created.professional_measurements[0].annotation_layout|ConvertTo-Json -Compress
    $b=$restored.professional_measurements[0].annotation_layout|ConvertTo-Json -Compress
    if($a -ne $b){throw 'Undo did not restore original label layout'}
    Select-Caliper;Caliper-Preview $restored
    Invoke-Owned (Wait-Control '取消 / Esc');Start-Sleep -Milliseconds 200
    $unchanged=Get-Content -LiteralPath $Project -Raw -Encoding UTF8|ConvertFrom-Json
    if(@($unchanged.documents[0].professional_measurements).Count -ne 1){throw 'Cancelled caliper changed saved definitions'}
    $result.caliper_cancelled_without_commit=$true
    Caliper-Preview $restored;[void](Capture '05-caliper-preview')
    Invoke-Owned (Wait-Control '确认拟合并保存测量')
    # UIA Invoke returns before Qt dispatches the button click. Wait for the
    # committed result and dirty indicator before asking the project to save.
    [void](Wait-Control 'P2' 20)
    [void](Wait-Control '未保存' 20)
    $fitted=Saved '06-confirmed-caliper' 2
    [void](Capture '06-confirmed-caliper');Select-Tool '两点距离 / X、Y 分量';Read-Table 'fitted';Csv-Export
    Stop-App;Start-App
    Read-Table 'reopened';Tree-Record '07-reopened-controls';[void](Capture '07-reopened')
    if(@(Find-Owned 'P1').Count -lt 1 -or @(Find-Owned 'P2').Count -lt 1){throw 'Reopened table does not expose both professional results'}
    $result.reopened=$true
} catch {$result.errors+=$_.Exception.Message;$result.failure_stack=$_.ScriptStackTrace;try{Tree-Record '99-failure-controls';[void](Capture '99-failure')}catch{}} finally {
    try{Stop-App;$result.graceful_exit=$true}catch{$result.errors+=('Close: '+$_.Exception.Message)}
    $result.source_sha256_after=(Get-FileHash -LiteralPath $Image -Algorithm SHA256).Hash
    $result.source_unchanged=($result.source_sha256_after -eq $result.source_sha256_before)
    $result.passed=($result.errors.Count -eq 0 -and $result.reopened -and $result.caliper_cancelled_without_commit -and $result.source_unchanged -and $result.graceful_exit)
    $result.finished_at=[DateTime]::UtcNow.ToString('o');Save-Result
}
$result|ConvertTo-Json -Depth 14
if($result.passed){exit 0}else{exit 1}
