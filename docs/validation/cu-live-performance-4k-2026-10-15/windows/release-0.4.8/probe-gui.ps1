# Windows PowerShell 5.1; saved with a UTF-8 BOM for Chinese UIA names.
# Run only in an exclusive GUI validation window. This is a bounded empty-project
# smoke test, not a camera/measurement/performance test and not an installer.
[CmdletBinding()]
param(
    [string]$Executable = 'C:\tmp\fdm-cu-live-unpacked-0.4.8-20261015\{app}\FiberDiameterMeasurement.exe',
    [string]$OutputRoot = 'C:\tmp\fdm-cu-live-opt4k-20261015\build\opt4k\release-0.4.8\gui',
    [switch]$AcknowledgeIsolatedEmptyProject
)

$ErrorActionPreference = 'Stop'
if (-not $AcknowledgeIsolatedEmptyProject) {
    throw 'This probe may discard only its own new isolated empty project; pass -AcknowledgeIsolatedEmptyProject.'
}
$exe = Get-Item -LiteralPath $Executable
$allowedRoot = [IO.Path]::GetFullPath('C:\tmp\fdm-cu-live-unpacked-0.4.8-20261015\')
if (-not $exe.FullName.StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $exe.Name -ne 'FiberDiameterMeasurement.exe') {
    throw 'Only the task-specific extracted application is permitted.'
}

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
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

$runId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N')
$runDirectory = Join-Path $OutputRoot $runId
$config = Join-Path $runDirectory 'isolated-config'
[void](New-Item -ItemType Directory -Path $config)
$utf8 = [Text.UTF8Encoding]::new($false)
$p = [Diagnostics.Process]::new()
$p.StartInfo = [Diagnostics.ProcessStartInfo]::new()
$p.StartInfo.FileName = $exe.FullName
$p.StartInfo.WorkingDirectory = $exe.DirectoryName
$p.StartInfo.UseShellExecute = $false
$p.StartInfo.EnvironmentVariables['LOCALAPPDATA'] = $config
$p.StartInfo.EnvironmentVariables['APPDATA'] = $config
$p.StartInfo.EnvironmentVariables['QT_ACCESSIBILITY'] = '1'
$result = [ordered]@{
    schema = 2; started_at = (Get-Date).ToString('o'); exe = $exe.FullName
    run_directory = $runDirectory; config = $config; pid = $null
    isolated_empty_project_verified = $false; main_window = $false
    experiment_menu = $false; live_experiment_entry = $false; live_window = $false
    key_controls = @(); key_controls_verified = $false; workspace_closed = $false
    unsaved_dialog = $null; graceful_exit = $false; forced_cleanup = $false
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
    return [ordered]@{
        name = $current.Name; process_id = $current.ProcessId
        type = $current.ControlType.ProgrammaticName; class = $current.ClassName
        automation_id = $current.AutomationId; native_handle = $current.NativeWindowHandle
        enabled = $current.IsEnabled; offscreen = $current.IsOffscreen
        bounds = $bounds
        help_text = $current.HelpText
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
}

function Invoke-Exact($Element) {
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
    Save-Snapshot '01-main-before-menu'
    $emptyLabels = @(Find-Within $main '0 张图片 · 无缺失 · 尚未保存')
    if ($emptyLabels.Count -eq 0) { throw 'Cannot verify a new empty unsaved project; refusing further interaction.' }
    $result.isolated_empty_project_verified = $true
    $menu = Visible-Unique @(Find-Within $main '试验' ([System.Windows.Automation.ControlType]::MenuItem)) 'Experiment menu'
    $result.experiment_menu = $true
    $menu.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern).Expand()
    Start-Sleep -Milliseconds 250
    $entry = Visible-Unique @(Find-OwnNamed '实时试验' ([System.Windows.Automation.ControlType]::MenuItem)) 'Live experiment menu item'
    $result.entry_element = Get-ElementRecord $entry
    $result.entry_action = Invoke-Exact $entry
    $result.live_experiment_entry = $true
    # Capture immediately after the actual click, including any unexpected dialog.
    Save-Snapshot '02-after-entry-invocation'
    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    do {
        $liveWindows = @([CuProbeNative]::Windows([uint32]$p.Id) | Where-Object {
            $_.Visible -and $_.Title -eq '实时试验 · 纤维直径与含量'
        })
        if ($liveWindows.Count -eq 1) { break }
        Start-Sleep -Milliseconds 150
        $p.Refresh()
    } while (-not $p.HasExited -and [DateTime]::UtcNow -lt $deadline)
    if ($liveWindows.Count -ne 1) {
        Save-Snapshot '03-live-window-not-confirmed'
        throw 'Live workspace exact title was not uniquely confirmed; inspect post-click UIA tree.'
    }
    $liveHandle = $liveWindows[0].Handle
    $live = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$liveHandle)
    if ($live.Current.ProcessId -ne $p.Id -or $live.Current.Name -ne '实时试验 · 纤维直径与含量') {
        throw 'Exact native workspace title did not agree with its UIA element.'
    }
    $result.live_window = $true
    $result.live_element = Get-ElementRecord $live
    $requiredNames = @('当前图片', '本机相机', 'CU 实时画面', '分析当前视场',
        '当前视场', '试验历史（全部操作者）', '全图候选', '改名', '撤销', '重做', '保存项目')
    foreach ($name in $requiredNames) {
        $matches = @(Find-Within $live $name)
        $details = @($matches | ForEach-Object { Get-ElementRecord $_ })
        $result.key_controls += [ordered]@{
            name = $name; present = ($matches.Count -gt 0)
            visible = (@($matches | Where-Object { -not $_.Current.IsOffscreen }).Count -gt 0)
            elements = $details
        }
    }
    $result.key_controls_verified = (@($result.key_controls | Where-Object { -not $_.present -or -not $_.visible }).Count -eq 0)
    Save-Snapshot '03-live-workspace-ready'
    if (-not $result.key_controls_verified) { $result.errors += 'Some required workspace controls were absent or offscreen; see snapshot.' }

    # Always close workspace first; no source, measurement, calibration, save,
    # template, or external-device action is performed by this probe.
    if (-not [CuProbeNative]::Close([uint32]$p.Id, $liveHandle)) { throw 'Could not post close to the verified workspace HWND.' }
    $deadline = [DateTime]::UtcNow.AddSeconds(8)
    do {
        Start-Sleep -Milliseconds 150
        $result.workspace_closed = -not [CuProbeNative]::IsWindow([IntPtr]$liveHandle)
    } while (-not $result.workspace_closed -and [DateTime]::UtcNow -lt $deadline)
    Save-Snapshot '04-after-workspace-close'
    if (-not $result.workspace_closed) { throw 'Workspace refused to close; do not continue closing other windows.' }
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
                $result.unsaved_dialog.action = Invoke-Exact $discard
                $result.unsaved_dialog.choice = '放弃'
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
        $result.exit_code -eq 0 -and $result.errors.Count -eq 0)
    $resultPath = Join-Path $runDirectory 'result.json'
    Write-JsonFile $resultPath $result
    Write-Output $resultPath
    Write-Output ($result | ConvertTo-Json -Depth 14)
    $p.Dispose()
}
if (-not $result.ok) { exit 1 }
