# Windows PowerShell 5.1; saved with a UTF-8 BOM for Chinese UIA names.
# Run only in an exclusive GUI validation window. Uses an owned temporary
# project for save/reentry checks; not a camera, timing, or installer test.
[CmdletBinding()]
param(
    [string]$Executable = 'C:\tmp\fdm-cu-live-ui-redesign-20261016\dist\windows\FiberDiameterMeasurement\FiberDiameterMeasurement.exe',
    [string]$OutputRoot = 'C:\tmp\fdm-cu-live-ui-redesign-20261016\build\ui-redesign\frozen-gui',
    [ValidateSet('dark','light','navy','warm')][string]$Theme = 'dark',
    [int]$ClientWidth = 1366,
    [int]$ClientHeight = 690,
    [string]$LegacyProject = '',
    [string]$LegacySettings = '',
    [string]$UnavailableDeviceId = 'microview:0',
    [switch]$AcknowledgeIsolatedEmptyProject = $true,
    [switch]$ExerciseBackgroundSaving = $true
)

$ErrorActionPreference = 'Stop'
if (-not $AcknowledgeIsolatedEmptyProject) {
    throw 'This probe may discard only its own new isolated empty project; pass -AcknowledgeIsolatedEmptyProject.'
}
$exe = Get-Item -LiteralPath $Executable
$validationRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$allowedRoot = [IO.Path]::GetFullPath((Join-Path $validationRoot 'dist\windows\'))
if (-not $exe.FullName.StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $exe.Name -ne 'FiberDiameterMeasurement.exe') {
    throw 'Only the task-specific extracted application is permitted.'
}

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
public sealed class CuProbeWindow {
    public long Handle;
    public long Owner;
    public string Title;
    public bool Visible;
}
public static class CuProbeNative {
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr hwnd, out Rect rect);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hwnd, out Rect rect);
    [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr hwnd);
    [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr hwnd, int x, int y, int width, int height, bool repaint);
    public static int[] ClientSize(long handle) {
        Rect r; GetClientRect(new IntPtr(handle), out r); return new int[]{r.Right-r.Left,r.Bottom-r.Top};
    }
    public static bool ResizeClient(long handle, int width, int height) {
        var hwnd=new IntPtr(handle); Rect outer, client;
        GetWindowRect(hwnd,out outer); GetClientRect(hwnd,out client);
        return MoveWindow(hwnd,10,10,width+outer.Right-outer.Left-client.Right,height+outer.Bottom-outer.Top-client.Bottom,true);
    }
    private delegate bool EnumProc(IntPtr hwnd, IntPtr state);
    [DllImport("user32.dll")] private static extern bool EnumWindows(EnumProc proc, IntPtr state);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint processId);
    [DllImport("user32.dll")] private static extern IntPtr GetWindow(IntPtr hwnd, uint command);
    [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr hwnd);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] private static extern int GetWindowText(IntPtr hwnd, StringBuilder value, int count);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] private static extern bool PostMessage(IntPtr hwnd, uint message, IntPtr wparam, IntPtr lparam);
    public static CuProbeWindow[] Windows(uint processId) {
        var windows = new List<CuProbeWindow>();
        EnumWindows(delegate(IntPtr hwnd, IntPtr state) {
            uint ownerProcess;
            GetWindowThreadProcessId(hwnd, out ownerProcess);
            if (ownerProcess == processId) {
                var title = new StringBuilder(2048);
                GetWindowText(hwnd, title, title.Capacity);
                windows.Add(new CuProbeWindow {
                    Handle=hwnd.ToInt64(), Owner=GetWindow(hwnd, 4).ToInt64(),
                    Title=title.ToString(), Visible=IsWindowVisible(hwnd)
                });
            }
            return true;
        }, IntPtr.Zero);
        return windows.ToArray();
    }
    public static bool Close(uint processId, long handle) {
        var hwnd = new IntPtr(handle);
        uint actualProcess;
        GetWindowThreadProcessId(hwnd, out actualProcess);
        if (actualProcess != processId || !IsWindow(hwnd)) return false;
        // Asynchronous WM_CLOSE lets the probe inspect a resulting modal dialog.
        // A synchronous UIA Close/Invoke may otherwise block behind that dialog.
        return PostMessage(hwnd, 0x0010, IntPtr.Zero, IntPtr.Zero);
    }
}
'@
[void][CuProbeNative]::SetProcessDPIAware()

$runId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N')
$runDirectory = Join-Path $OutputRoot $runId
$config = Join-Path $runDirectory 'isolated-config'
[void](New-Item -ItemType Directory -Path $config)
$utf8 = [Text.UTF8Encoding]::new($false)
$appConfig = Join-Path $config 'FiberDiameterMeasurement'
[void](New-Item -ItemType Directory -Path $appConfig)
$isolatedSettings = [pscustomobject]@{theme_mode=$Theme; selected_capture_device_id=$UnavailableDeviceId}
if($LegacySettings) {
    $isolatedSettings = Get-Content -LiteralPath $LegacySettings -Raw -Encoding UTF8 | ConvertFrom-Json
    $isolatedSettings | Add-Member -NotePropertyName theme_mode -NotePropertyValue $Theme -Force
    $isolatedSettings | Add-Member -NotePropertyName selected_capture_device_id -NotePropertyValue $UnavailableDeviceId -Force
}
[IO.File]::WriteAllText((Join-Path $appConfig 'settings.json'),($isolatedSettings|ConvertTo-Json -Depth 20),$utf8)
$p = [Diagnostics.Process]::new()
$p.StartInfo = [Diagnostics.ProcessStartInfo]::new()
$p.StartInfo.FileName = $exe.FullName
$p.StartInfo.WorkingDirectory = $exe.DirectoryName
$p.StartInfo.UseShellExecute = $false
$p.StartInfo.EnvironmentVariables.Remove('PSExecutionPolicyPreference')
$p.StartInfo.EnvironmentVariables['LOCALAPPDATA'] = $config
$p.StartInfo.EnvironmentVariables['APPDATA'] = $config
$p.StartInfo.EnvironmentVariables['QT_ACCESSIBILITY'] = '1'
$p.StartInfo.EnvironmentVariables['QT_QPA_PLATFORM'] = 'windows'
$p.StartInfo.EnvironmentVariables['QT_SCALE_FACTOR'] = '0.5'
$probeProject = Join-Path $runDirectory 'save-lifecycle-probe.fdmproj'
if($ExerciseBackgroundSaving) {
    $python = 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
    $env:PYTHONPATH = Join-Path $validationRoot 'src'
    $env:FDM_SAVE_PROBE_PROJECT = $probeProject
    $createProject = @'
import os
from fdm.models import ProjectState
from fdm.project_io import ProjectIO
from fdm.live_experiment import ExperimentSession, default_materials
p=ProjectState.empty()
s=ExperimentSession(sample_name="save-lifecycle-probe",save_images=False)
s.add_operator("probe")
s.set_materials(default_materials())
p.live_experiments.append(s)
ProjectIO.save(p, os.environ["FDM_SAVE_PROBE_PROJECT"])
'@
    $fixtureScript = Join-Path $runDirectory 'create-save-fixture.py'
    [IO.File]::WriteAllText($fixtureScript, $createProject, $utf8)
    & $python $fixtureScript
    if($LASTEXITCODE -ne 0){throw 'Could not prepare isolated save fixture'}
    if($LegacyProject) {Copy-Item -LiteralPath $LegacyProject -Destination $probeProject -Force}
    $p.StartInfo.Arguments = '"' + $probeProject + '"'
}

$result = [ordered]@{
    schema = 2; started_at = (Get-Date).ToString('o'); exe = $exe.FullName
    theme = $Theme; real_hardware = $null; hardware_acceptance_tested = $false
    capture_note = 'GUI navigation may auto-start a VM-mapped camera; this probe does not certify camera hardware.'
    requested_capture_device_id = $UnavailableDeviceId
    legacy_project = $LegacyProject; legacy_settings = $LegacySettings
    run_directory = $runDirectory; config = $config; pid = $null
    isolated_empty_project_verified = $false; main_window = $false
    experiment_menu = $false; live_experiment_entry = $false; live_window = $false
    key_controls = @(); key_controls_verified = $false; workspace_closed = $false
    unsaved_dialog = $null; graceful_exit = $false; forced_cleanup = $false
    requested_client_size = @($ClientWidth, $ClientHeight)
    observed_client_sizes = @()
    scroll_actions = @()
    snapshots = @(); errors = @(); ok = $false
}

function Write-JsonFile($Path, $Value) {
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 14), $utf8)
}

function Get-ElementRecord($Element) {
    $current = $Element.Current
    $rect = $current.BoundingRectangle
    $bounds = @($rect.X, $rect.Y, $rect.Width, $rect.Height)
    if (@($bounds | Where-Object { [double]::IsNaN($_) -or [double]::IsInfinity($_) }).Count -gt 0) {
        $bounds = $null
    }
    $valuePattern = $null
    $value = $null
    if($Element.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern,[ref]$valuePattern)) {
        $value = $valuePattern.Current.Value
    }
    return [ordered]@{
        name = $current.Name; process_id = $current.ProcessId
        type = $current.ControlType.ProgrammaticName; class = $current.ClassName
        automation_id = $current.AutomationId; native_handle = $current.NativeWindowHandle
        enabled = $current.IsEnabled; offscreen = $current.IsOffscreen
        bounds = $bounds
        help_text = $current.HelpText
        value = $value
        runtime_id = @($Element.GetRuntimeId())
        patterns = @($Element.GetSupportedPatterns() | ForEach-Object { $_.ProgrammaticName })
    }
}

function Get-OwnedRoots {
    # EnumWindows includes owned top-level HWNDs, which may be nested beneath the
    # main window in the UIA desktop tree. Never assume Desktop.Children == windows.
    foreach ($window in [CuProbeNative]::Windows([uint32]$p.Id)) {
        try {
            $element = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$window.Handle)
            if ($element.Current.ProcessId -eq $p.Id) {
                [pscustomobject]@{ window = $window; element = $element }
            }
        } catch { }
    }
}

function Find-OwnNamed($Name, $ControlType = $null) {
    $conditions = [Collections.Generic.List[System.Windows.Automation.Condition]]::new()
    $conditions.Add([System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, [int]$p.Id))
    $conditions.Add([System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::NameProperty, [string]$Name))
    if ($null -ne $ControlType) {
        $conditions.Add([System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty, $ControlType))
    }
    $condition = [System.Windows.Automation.AndCondition]::new($conditions.ToArray())
    $seen = @{}
    foreach ($root in @(Get-OwnedRoots)) {
        # Each native root + all PID-filtered UIA descendants, not just UIA windows.
        foreach ($element in $root.element.FindAll([System.Windows.Automation.TreeScope]::Subtree, $condition)) {
            $key = [string]::Join(',', $element.GetRuntimeId())
            if (-not $seen.ContainsKey($key)) { $seen[$key] = $true; $element }
        }
    }
}

function Find-Within($Root, $Name, $ControlType = $null) {
    $conditions = [Collections.Generic.List[System.Windows.Automation.Condition]]::new()
    $conditions.Add([System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, [int]$p.Id))
    $conditions.Add([System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::NameProperty, [string]$Name))
    if ($null -ne $ControlType) {
        $conditions.Add([System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty, $ControlType))
    }
    $condition = [System.Windows.Automation.AndCondition]::new($conditions.ToArray())
    return $Root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $condition)
}

function Confirm-ClientSize($Phase) {
    if ($null -eq $main) { return }
    $actual = @([CuProbeNative]::ClientSize($mainHandle))
    $result.observed_client_sizes += [ordered]@{phase=$Phase; actual=$actual; requested=@($ClientWidth,$ClientHeight)}
    if($actual[0] -ne $ClientWidth -or $actual[1] -ne $ClientHeight) {
        throw "Client size changed after $Phase : $($actual[0]) x $($actual[1]); requested $ClientWidth x $ClientHeight"
    }
}

function Save-Snapshot($Phase) {
    $nodes = [Collections.Generic.List[object]]::new()
    $nativeWindows = @([CuProbeNative]::Windows([uint32]$p.Id))
    $queue = [Collections.Generic.Queue[object]]::new()
    foreach ($root in @(Get-OwnedRoots)) {
        $queue.Enqueue(@{ element = $root.element; depth = 0; parent = $null; hwnd = $root.window.Handle })
    }
    $walker = [System.Windows.Automation.TreeWalker]::RawViewWalker
    $truncated = $false
    while ($queue.Count -gt 0) {
        if ($nodes.Count -ge 1800) { $truncated = $true; break }
        $item = $queue.Dequeue()
        try {
            if ($item.element.Current.ProcessId -ne $p.Id) { continue }
            $record = Get-ElementRecord $item.element
            $record['id'] = $nodes.Count
            $record['parent'] = $item.parent
            $record['depth'] = $item.depth
            $record['root_hwnd'] = $item.hwnd
            $nodes.Add($record)
            if ($item.depth -ge 18) { $truncated = $true; continue }
            $child = $walker.GetFirstChild($item.element)
            while ($null -ne $child) {
                $queue.Enqueue(@{ element = $child; depth = $item.depth + 1; parent = $record.id; hwnd = $item.hwnd })
                $child = $walker.GetNextSibling($child)
            }
        } catch {
            $nodes.Add(@{ parent = $item.parent; depth = $item.depth; error = $_.Exception.Message })
        }
    }
    $path = Join-Path $runDirectory ($Phase + '-uia.json')
    Write-JsonFile $path ([ordered]@{
        phase = $Phase; timestamp = (Get-Date).ToString('o'); pid = $p.Id
        native_windows = $nativeWindows; nodes = @($nodes.ToArray()); truncated = $truncated
    })
    $result.snapshots += $path
    if ($null -ne $main) {
        $r = $main.Current.BoundingRectangle
        if ($r.Width -gt 0 -and $r.Height -gt 0) {
            $bitmap = [Drawing.Bitmap]::new([int]$r.Width,[int]$r.Height)
            $graphics = [Drawing.Graphics]::FromImage($bitmap)
            try {
                $graphics.CopyFromScreen([int]$r.X,[int]$r.Y,0,0,$bitmap.Size)
                $bitmap.Save((Join-Path $runDirectory ($Phase + '.png')),[Drawing.Imaging.ImageFormat]::Png)
            } finally { $graphics.Dispose(); $bitmap.Dispose() }
        }
    }
    if(-not $Phase.StartsWith('99-')) { Confirm-ClientSize $Phase }
}

function Get-LocalScrollContext($Element) {
    # Only walk ancestors of the exact PID-owned control. A hidden trial's
    # scroll pane is not an alternative target, and the desktop is never scrolled.
    $walker = [System.Windows.Automation.TreeWalker]::RawViewWalker
    $ancestor = $walker.GetParent($Element)
    for($depth=0; $null -ne $ancestor -and $depth -lt 24; $depth++) {
        if($ancestor.Current.ProcessId -ne $p.Id) { break }
        $pattern = $null
        if(-not $ancestor.Current.IsOffscreen -and
            $ancestor.TryGetCurrentPattern([System.Windows.Automation.ScrollPattern]::Pattern,[ref]$pattern) -and
            $pattern.Current.VerticallyScrollable) {
            return [pscustomobject]@{element=$ancestor; pattern=$pattern; kind='ScrollPattern'}
        }
        # Qt 6's Windows UIA bridge exposes QScrollArea without ScrollPattern,
        # but its own vertical QScrollBar provides RangeValuePattern. Captured
        # UIA trees also report clipped children as IsOffscreen=false, so the
        # local viewport bounds remain authoritative before invoking a control.
        if(-not $ancestor.Current.IsOffscreen -and $ancestor.Current.ClassName -eq 'QScrollArea') {
            $barCondition = [System.Windows.Automation.PropertyCondition]::new(
                [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
                [System.Windows.Automation.ControlType]::ScrollBar)
            $bars = @($ancestor.FindAll([System.Windows.Automation.TreeScope]::Descendants,$barCondition) | Where-Object {
                $_.Current.ProcessId -eq $p.Id -and -not $_.Current.IsOffscreen -and
                $_.Current.BoundingRectangle.Height -gt $_.Current.BoundingRectangle.Width
            })
            if($bars.Count -eq 1) {
                $range = $null
                if($bars[0].TryGetCurrentPattern([System.Windows.Automation.RangeValuePattern]::Pattern,[ref]$range) -and
                    -not $range.Current.IsReadOnly -and $range.Current.Maximum -gt $range.Current.Minimum) {
                    return [pscustomobject]@{element=$ancestor; pattern=$range; kind='RangeValuePattern'}
                }
            }
        }
        $ancestor = $walker.GetParent($ancestor)
    }
    return $null
}

function Test-OwnedReachable($Element) {
    if($null -eq $Element -or $Element.Current.ProcessId -ne $p.Id) { return $false }
    return (-not $Element.Current.IsOffscreen -or $null -ne (Get-LocalScrollContext $Element))
}

function Ensure-OwnedVisible($Element) {
    if($null -eq $Element -or $Element.Current.ProcessId -ne $p.Id) { throw 'Cannot scroll a foreign or missing control' }
    $context = Get-LocalScrollContext $Element
    if($null -eq $context) {
        if($Element.Current.IsOffscreen) { throw 'Offscreen control has no visible local ScrollPattern ancestor' }
        return
    }
    for($step=0; $step -lt 80; $step++) {
        $bounds = $Element.Current.BoundingRectangle
        $viewport = $context.element.Current.BoundingRectangle
        if(-not $Element.Current.IsOffscreen -and $bounds.Height -gt 0 -and
            $bounds.Top -ge $viewport.Top -and $bounds.Bottom -le $viewport.Bottom -and
            $bounds.Left -ge $viewport.Left -and $bounds.Right -le $viewport.Right) { return }
        if($context.kind -eq 'ScrollPattern') {
            $before = $context.pattern.Current.VerticalScrollPercent
            $amount = [System.Windows.Automation.ScrollAmount]::SmallIncrement
            if($bounds.Height -gt 0 -and $bounds.Top -lt $viewport.Top) {
                $amount = [System.Windows.Automation.ScrollAmount]::SmallDecrement
            }
            $context.pattern.Scroll([System.Windows.Automation.ScrollAmount]::NoAmount,$amount)
            Start-Sleep -Milliseconds 50
            $after = $context.pattern.Current.VerticalScrollPercent
        } else {
            $before = $context.pattern.Current.Value
            $delta = [Math]::Max(16,$bounds.Bottom-$viewport.Bottom+4)
            if($bounds.Height -gt 0 -and $bounds.Top -lt $viewport.Top) { $delta=$bounds.Top-$viewport.Top-4 }
            $amount = [Math]::Min($context.pattern.Current.Maximum,[Math]::Max($context.pattern.Current.Minimum,$before+$delta))
            $context.pattern.SetValue($amount)
            Start-Sleep -Milliseconds 50
            $after = $context.pattern.Current.Value
        }
        $result.scroll_actions += [ordered]@{
            target=$Element.Current.Name; container=$context.element.Current.Name
            pattern=$context.kind; before=$before; after=$after; amount=[string]$amount
        }
        if([Math]::Abs($before-$after) -lt 0.001) { throw 'Local viewport cannot scroll the requested control into view' }
    }
    throw 'Local viewport scroll limit exceeded'
}

function Invoke-Exact($Element) {
    Ensure-OwnedVisible $Element
    if ($null -eq $Element -or $Element.Current.ProcessId -ne $p.Id -or
        -not $Element.Current.IsEnabled -or $Element.Current.IsOffscreen) {
        throw 'Refusing to invoke absent, foreign, disabled, or offscreen UIA element.'
    }
    $pattern = $null
    if ($Element.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$pattern)) {
        $pattern.Invoke()
        return 'InvokePattern'
    }
    foreach ($supported in $Element.GetSupportedPatterns()) {
        if ($supported.ProgrammaticName -eq 'LegacyIAccessiblePatternIdentifiers.Pattern') {
            $Element.GetCurrentPattern($supported).DoDefaultAction()
            return 'LegacyIAccessiblePattern'
        }
    }
    throw 'Named UIA element has no Invoke or LegacyIAccessible action.'
}

function Visible-Unique($Elements, $Description) {
    $visible = @($Elements | Where-Object { $_.Current.ProcessId -eq $p.Id -and -not $_.Current.IsOffscreen })
    if ($visible.Count -ne 1) { throw "$Description expected one visible match, found $($visible.Count)." }
    return $visible[0]
}

try {
    [void]$p.Start()
    $result.pid = $p.Id
    $deadline = [DateTime]::UtcNow.AddSeconds(25)
    do {
        Start-Sleep -Milliseconds 150
        $p.Refresh()
        $mainWindows = @([CuProbeNative]::Windows([uint32]$p.Id) | Where-Object {
            $_.Visible -and $_.Title -eq '显微测量工作台'
        })
    } while (-not $p.HasExited -and $mainWindows.Count -ne 1 -and [DateTime]::UtcNow -lt $deadline)
    if ($p.HasExited -or $mainWindows.Count -ne 1) { throw 'Exactly one expected main window did not appear.' }
    $mainHandle = $mainWindows[0].Handle
    $main = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$mainHandle)
    $result.main_window = $true
    $result.main_title = $main.Current.Name
    if(-not [CuProbeNative]::ResizeClient($mainHandle,$ClientWidth,$ClientHeight)){throw 'Could not resize owned window'}
    Start-Sleep -Milliseconds 200
    $result['client_size'] = @([CuProbeNative]::ClientSize($mainHandle))
    $result['os_dpi'] = [CuProbeNative]::GetDpiForWindow([IntPtr]$mainHandle)
    $screen = [Windows.Forms.Screen]::FromHandle([IntPtr]$mainHandle)
    $result['screen_bounds'] = @($screen.Bounds.X,$screen.Bounds.Y,$screen.Bounds.Width,$screen.Bounds.Height)
    $result['available_geometry'] = @($screen.WorkingArea.X,$screen.WorkingArea.Y,$screen.WorkingArea.Width,$screen.WorkingArea.Height)
    if($result.client_size[0] -ne $ClientWidth -or $result.client_size[1] -ne $ClientHeight){throw 'Window exceeded requested client size'}
    Save-Snapshot '01-main-before-menu'
    $emptyLabels = @(Find-Within $main '0 张图片 · 无缺失 · 尚未保存')
    if($ExerciseBackgroundSaving) {
        if(-not (Test-Path $probeProject)){throw 'Missing owned save fixture'}
    } elseif ($emptyLabels.Count -eq 0) { throw 'Cannot verify a new empty unsaved project; refusing further interaction.' }
    $result.isolated_empty_project_verified = $true
    $entry = Visible-Unique @(Find-Within $main '采集与试验' ([System.Windows.Automation.ControlType]::CheckBox)) 'Acquisition toolbar entry'
    $result.entry_action = Invoke-Exact $entry
    $result.live_experiment_entry = $true
    Start-Sleep -Milliseconds 300
    Save-Snapshot '02-acquisition'
    $content = Visible-Unique @(Find-Within $main '含量试验' ([System.Windows.Automation.ControlType]::TabItem)) 'Content trial tab'
    $content.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
    Start-Sleep -Milliseconds 300
    for ($i=0; $i -lt 20; $i++) {
        [void](Invoke-Exact $entry)
        Start-Sleep -Milliseconds 20
        Confirm-ClientSize ('main-entry-' + $i)
    }
    $result.live_window = $true
    $requiredNames = @('样品编号','样品名称','操作者','测量标尺','新视场','操作','记录','报告','先选','后选','保存项目')
    foreach ($name in $requiredNames) {
        $matches = @(Find-Within $main $name)
        $reachable = @($matches | Where-Object { Test-OwnedReachable $_ })
        if($reachable.Count -gt 0) { Ensure-OwnedVisible $reachable[0] }
        $result.key_controls += [ordered]@{ name=$name; present=($matches.Count -gt 0); visible=(@($matches | Where-Object {-not $_.Current.IsOffscreen}).Count -gt 0); reached_with_local_scroll=($reachable.Count -gt 0) }
    }
    $result.key_controls_verified = (@($result.key_controls | Where-Object {-not $_.present -or -not $_.visible}).Count -eq 0)
    Save-Snapshot '03-content-reentry'
    if (-not $result.key_controls_verified) { throw 'Some required workbench controls are absent or offscreen.' }
    $number = Visible-Unique @(Find-Within $main '样品编号' ([System.Windows.Automation.ControlType]::Edit)) 'Persistent sample number'
    $numberBounds = $number.Current.BoundingRectangle
    foreach ($name in @('记录','报告','操作')) {
        $tab = Visible-Unique @(Find-Within $main $name ([System.Windows.Automation.ControlType]::TabItem)) 'Workflow tab'
        $tab.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
        Start-Sleep -Milliseconds 150
        if($number.Current.IsOffscreen -or $number.Current.BoundingRectangle -ne $numberBounds) {
            throw 'Common sample metadata moved or disappeared during sidebar navigation'
        }
        Save-Snapshot ('04-sidebar-' + $name)
    }
    foreach ($name in @('直径试验','实时采集','含量试验')) {
        $tab = Visible-Unique @(Find-Within $main $name ([System.Windows.Automation.ControlType]::TabItem)) 'Acquisition family tab'
        $tab.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
        Start-Sleep -Milliseconds 200
        Confirm-ClientSize ('acquisition-page-' + $name)
    }
    $result['stable_navigation_verified'] = $true

    if($ExerciseBackgroundSaving) {
        $sampleName = Visible-Unique @(Find-Within $main '样品名称' ([System.Windows.Automation.ControlType]::Edit)) 'Owned project sample name'
        $sampleValue = $null
        if(-not $sampleName.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern,[ref]$sampleValue) -or
            $sampleValue.Current.Value -ne 'save-lifecycle-probe') { throw 'Owned project sample name value does not match; refusing save' }
        $result['loaded_fixture_name'] = $sampleValue.Current.Value
        $save = Visible-Unique @(Find-Within $main '保存项目' ([System.Windows.Automation.ControlType]::Button)) 'Trial save button'
        $adds = @(Find-Within $main '+' ([System.Windows.Automation.ControlType]::Button) | Where-Object { Test-OwnedReachable $_ })
        if($adds.Count -ne 2){throw 'Expected the two default material count buttons'}
        Ensure-OwnedVisible $adds[0]
        $result['saved_count_checkpoints'] = @()
        $baseline = 0
        $initialProject = Get-Content -Raw -Encoding UTF8 $probeProject | ConvertFrom-Json
        foreach($event in $initialProject.live_experiments[0].count_adjustments) {$baseline += [int]$event.delta}
        $result['initial_count'] = $baseline
        for($round=1; $round -le 20; $round++) {
            $targetCount = $baseline + $round
            [void](Invoke-Exact $adds[0])
            # Qt UIA Invoke can return before animateClick emits clicked. Verify
            # the visible count before saving; invocation order is not event order.
            $countDeadline = [DateTime]::UtcNow.AddSeconds(5)
            $plusBounds = $adds[0].Current.BoundingRectangle
            do {
                Start-Sleep -Milliseconds 30
                $shown = @(Find-Within $main ([string]$targetCount) ([System.Windows.Automation.ControlType]::Text) | Where-Object {
                    -not $_.Current.IsOffscreen -and
                    [Math]::Abs($_.Current.BoundingRectangle.Y - $plusBounds.Y) -lt 3 -and
                    $_.Current.BoundingRectangle.X -gt $plusBounds.Right -and
                    $_.Current.BoundingRectangle.X -lt ($plusBounds.Right + 40)
                })
            } while ($shown.Count -eq 0 -and [DateTime]::UtcNow -lt $countDeadline)
            if($shown.Count -ne 1){throw "Visible count was not published at round $round"}
            [void](Invoke-Exact $save)
            Start-Sleep -Milliseconds 150
            [void](Invoke-Exact $save)
            $deadline = [DateTime]::UtcNow.AddSeconds(10)
            $count = -1
            do {
                Start-Sleep -Milliseconds 40
                try {
                    $doc = Get-Content -Raw -Encoding UTF8 $probeProject | ConvertFrom-Json
                    $count = 0
                    foreach($event in $doc.live_experiments[0].count_adjustments) {$count += [int]$event.delta}
                } catch { $count = -1 }
            } while ($count -ne $targetCount -and [DateTime]::UtcNow -lt $deadline)
            if($count -ne $targetCount){throw "Saved count mismatch at round $round : $count"}
            Confirm-ClientSize ('save-round-' + $round)
            $result.saved_count_checkpoints += $count
        }
        $result['background_save_verified'] = $true
        Save-Snapshot '05-save-rounds-complete'
    }
    $result.workspace_closed = $true
    if (-not [CuProbeNative]::Close([uint32]$p.Id, $mainHandle)) { throw 'Could not post close to the verified main HWND.' }
    $deadline = [DateTime]::UtcNow.AddSeconds(12)
    $discardClicked = $false
    do {
        Start-Sleep -Milliseconds 150
        $p.Refresh()
        if ($p.HasExited) { break }
        if (-not $discardClicked) {
            $dialogs = @([CuProbeNative]::Windows([uint32]$p.Id) | Where-Object {
                $_.Visible -and $_.Title -eq '未保存的改动'
            })
            if ($dialogs.Count -gt 1) { throw 'Multiple unsaved-change dialogs; refusing ambiguous discard.' }
            if ($dialogs.Count -eq 1) {
                $dialog = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$dialogs[0].Handle)
                if ($dialog.Current.ProcessId -ne $p.Id -or $dialog.Current.Name -ne '未保存的改动') {
                    throw 'Exact native unsaved-dialog title did not agree with its UIA element.'
                }
                Save-Snapshot '05-unsaved-dialog-before-decision'
                $buttonCondition = [System.Windows.Automation.PropertyCondition]::new(
                    [System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button)
                $buttons = @($dialog.FindAll([System.Windows.Automation.TreeScope]::Descendants, $buttonCondition))
                $buttonRecords = @($buttons | ForEach-Object { Get-ElementRecord $_ })
                $result.unsaved_dialog = [ordered]@{
                    title = $dialog.Current.Name; buttons = $buttonRecords; choice = $null
                }
                $message = @(Find-Within $dialog '当前项目的比例尺、图片、类别、实时试验或继承关系有未保存改动。')
                if (-not $result.isolated_empty_project_verified -or $message.Count -eq 0) {
                    throw 'Unsaved dialog is not the verified isolated empty-project extension change.'
                }
                $actualNames = @($buttons | Where-Object { $_.Current.ProcessId -eq $p.Id -and -not $_.Current.IsOffscreen } | ForEach-Object { $_.Current.Name })
                foreach ($expectedName in @('保存', '放弃', '取消')) {
                    if ($actualNames -notcontains $expectedName) { throw "Unexpected actual unsaved-dialog buttons: $($actualNames -join ', ')." }
                }
                $discard = Visible-Unique @(Find-Within $dialog '放弃' ([System.Windows.Automation.ControlType]::Button)) 'Isolated-project discard button'
                if(-not $ExerciseBackgroundSaving) { throw 'An untouched trial page incorrectly prompted to save.' }
                if($ExerciseBackgroundSaving) {
                    $saveClose = Visible-Unique @(Find-Within $dialog '保存' ([System.Windows.Automation.ControlType]::Button)) 'Save owned fixture before close'
                    $result.unsaved_dialog.action = Invoke-Exact $saveClose
                    $result.unsaved_dialog.choice = '保存'
                } else {
                    $result.unsaved_dialog.action = Invoke-Exact $discard
                    $result.unsaved_dialog.choice = '放弃'
                }
                $discardClicked = $true
            }
        }
    } while ([DateTime]::UtcNow -lt $deadline)
    $p.Refresh()
    $result.graceful_exit = $p.HasExited
    if ($p.HasExited) { $result.exit_code = $p.ExitCode }
    else {
        Save-Snapshot '06-still-open-after-close'
        throw 'Application did not exit naturally after the explicit isolated-project close sequence.'
    }
} catch {
    $result.errors += $_.Exception.Message
    if ($null -ne $result.pid) {
        try { if (-not $p.HasExited) { Save-Snapshot '99-failure-observed-tree' } } catch { $result.errors += $_.Exception.Message }
    }
} finally {
    if ($null -ne $result.pid) {
        $p.Refresh()
        if (-not $p.HasExited) {
            # Never sweep process names, close existing apps, or click unknown dialogs.
            # Kill only the Process instance created by this run if graceful close failed.
            $p.Kill()
            $result.forced_cleanup = $true
            [void]$p.WaitForExit(3000)
        }
        $p.Refresh()
        $result.process_exited_after_cleanup = $p.HasExited
    }
    $result.finished_at = (Get-Date).ToString('o')
    $result.ok = ($result.main_window -and $result.isolated_empty_project_verified -and
        $result.live_experiment_entry -and $result.live_window -and $result.key_controls_verified -and
        $result.workspace_closed -and $result.graceful_exit -and -not $result.forced_cleanup -and
        $result.exit_code -eq 0 -and $result.errors.Count -eq 0 -and
        (-not $ExerciseBackgroundSaving -or $result.background_save_verified))
    $resultPath = Join-Path $runDirectory 'result.json'
    Write-JsonFile $resultPath $result
    Write-Output $resultPath
    Write-Output ($result | ConvertTo-Json -Depth 14)
    $p.Dispose()
}
if (-not $result.ok) { exit 1 }
