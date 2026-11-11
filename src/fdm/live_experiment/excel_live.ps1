# A persistent STA host for the workbooks owned by FDM live trials.
# Input and output are UTF-8 JSON lines. Excel files are saved by the user.
param()
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$script:excel = $null
$script:excelIdentity = $null
$script:sessions = @{}
$script:closed = @{}
$script:request = $null

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class FdmLiveExcelWindow {
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr window, out uint process);
}
'@

function Release-Com($value) {
    if ($null -ne $value -and [Runtime.InteropServices.Marshal]::IsComObject($value)) {
        try { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($value) } catch {}
    }
}

function Reply($body) {
    $result = [ordered]@{
        request_id = $script:request.request_id
        session_id = $script:request.session_id
        generation = $script:request.generation
        input_revision = $script:request.input_revision
        operation = $script:request.operation
    }
    foreach ($key in $body.Keys) { $result[$key] = $body[$key] }
    [Console]::Out.WriteLine(($result | ConvertTo-Json -Depth 30 -Compress))
    [Console]::Out.Flush()
}

function Failure([string]$code, [string]$message) {
    $error = [InvalidOperationException]::new($message)
    $error.Data['fdm_code'] = $code
    throw $error
}

function Error-Code($error) {
    $exception = $error.Exception
    while ($exception) {
        if ($exception.Data.Contains('fdm_code')) { return [string]$exception.Data['fdm_code'] }
        if ($exception.HResult -in @(-2147418111, -2147417846, -2146777998)) { return 'excel_busy' }
        $exception = $exception.InnerException
    }
    return 'excel_failed'
}

function Start-Excel {
    if ($script:excel) {
        $process = Get-Process -Id $script:excelIdentity.id -ErrorAction SilentlyContinue
        if ($process -and $process.StartTime.ToUniversalTime().Ticks.ToString() -eq $script:excelIdentity.created) {
            return
        }
        foreach ($key in @($script:sessions.Keys)) {
            $script:closed[$key] = $script:sessions[$key].generation
            Release-Com $script:sessions[$key].book
        }
        $script:sessions.Clear()
        Release-Com $script:excel
        $script:excel = $null
    }
    $existing = @(Get-Process EXCEL -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
    $candidate = New-Object -ComObject Excel.Application
    [uint32]$processId = 0
    [void][FdmLiveExcelWindow]::GetWindowThreadProcessId([IntPtr]$candidate.Hwnd, [ref]$processId)
    if (!$processId -or $existing -contains [int]$processId) {
        Release-Com $candidate
        Failure 'excel_not_independent' 'Excel did not create an independent live-report instance.'
    }
    $process = Get-Process -Id $processId
    $script:excel = $candidate
    $script:excelIdentity = @{id = $processId; created = $process.StartTime.ToUniversalTime().Ticks.ToString()}
    # Preserve Office policy. The user controls macro trust and file saving.
    $script:excel.AutomationSecurity = 2
    $script:excel.Visible = $true
}

function Book-Metadata($state) {
    $name = [string]$state.book.Name
    $directory = [string]$state.book.Path
    return @{
        name = $name
        path = $(if ($directory) { [string]$state.book.FullName } else { '' })
        saved = [bool]$state.book.Saved
        has_file = [bool]$directory
        excel_pid = $script:excelIdentity.id
        excel_version = [string]$script:excel.Version
    }
}

function Test-BookOpen($state) {
    if (![string]::IsNullOrEmpty([string]$state.book.Name)) { return $true }
    $process = Get-Process -Id $script:excelIdentity.id -ErrorAction SilentlyContinue
    if (!$process -or $process.StartTime.ToUniversalTime().Ticks.ToString() -ne $script:excelIdentity.created) {
        return $false
    }
    # Excel can return an empty COM property while a native modal dialog is
    # open, as well as after workbook closure. Only discard the association
    # when the application is responsive enough to distinguish the two.
    $ready = $script:excel.Ready
    $books = $script:excel.Workbooks
    try { $count = $books.Count } finally { Release-Com $books }
    if ($null -eq $ready -or !$ready -or $null -eq $count) {
        Failure 'excel_busy' 'Excel is busy in an interactive operation. The workbook remains connected.'
    }
    return $false
}

function Get-State {
    $key = [string]$script:request.session_id
    if (!$script:sessions.ContainsKey($key)) { Failure 'workbook_closed' 'The live workbook is no longer connected.' }
    $state = $script:sessions[$key]
    if ([int64]$state.generation -ne [int64]$script:request.generation) {
        Failure 'stale_generation' 'This request belongs to an earlier workbook connection.'
    }
    try {
        if (!(Test-BookOpen $state)) { throw 'Closed workbook has no name.' }
    } catch {
        if ((Error-Code $_) -eq 'excel_busy') { throw }
        $script:closed[$key] = $state.generation
        $script:sessions.Remove($key)
        Release-Com $state.book
        Failure 'workbook_closed' 'The live workbook was closed. Use reconnect to open a new workbook.'
    }
    if (!$state.initialized) { Failure 'initialization_failed' 'Input protection was not initialized. Reconnect to use a new workbook.' }
    return $state
}

function Get-SheetName($item, $profile) {
    if ($item.sheet) { return [string]$item.sheet }
    return [string]$profile.data_sheet
}

function Set-Cell($book, [string]$sheetName, [string]$address, $value) {
    $sheet = $null; $range = $null
    try {
        $sheet = $book.Worksheets.Item($sheetName)
        $range = $sheet.Range($address)
        if ($value -is [string]) { $value = "'" + $value }
        elseif ($null -ne $value) { $value = [double]$value }
        [void]$range.GetType().InvokeMember('Value2', [Reflection.BindingFlags]::SetProperty,
            $null, $range, @($value))
    } finally { Release-Com $range; Release-Com $sheet }
}

function Get-Cell($book, [string]$sheetName, [string]$address) {
    $sheet = $null; $range = $null
    try {
        $sheet = $book.Worksheets.Item($sheetName)
        $range = $sheet.Range($address)
        return $range.Value2
    } finally { Release-Com $range; Release-Com $sheet }
}

function Remember-Locks($range, $states) {
    $locked = $range.Locked
    if ($null -ne $locked) {
        $states.Add(@{range = [string]$range.Address(); locked = [bool]$locked})
        return
    }
    # Mixed formatting is split into homogeneous rectangles for exact restoration.
    $first = $null; $second = $null; $anchor = $null
    try {
        $rows = [int]$range.Rows.Count
        $columns = [int]$range.Columns.Count
        $anchor = $range.Cells.Item(1, 1)
        if ($rows -gt 1 -and $rows -ge $columns) {
            $half = [int][Math]::Floor($rows / 2)
            $first = $anchor.Resize($half, $columns)
            $next = $anchor.Offset($half, 0)
            try { $second = $next.Resize($rows - $half, $columns) } finally { Release-Com $next }
        } else {
            $half = [int][Math]::Floor($columns / 2)
            $first = $anchor.Resize(1, $half)
            $next = $anchor.Offset(0, $half)
            try { $second = $next.Resize(1, $columns - $half) } finally { Release-Com $next }
        }
        Remember-Locks $first $states
        Remember-Locks $second $states
    } finally { Release-Com $first; Release-Com $second; Release-Com $anchor }
}

function Protect-Inputs($state, $payload) {
    $groups = @{}
    foreach ($item in $payload.managed_ranges) {
        $name = Get-SheetName $item $payload.profile
        if (!$groups.ContainsKey($name)) { $groups[$name] = [Collections.Generic.List[string]]::new() }
        $groups[$name].Add([string]$item.range)
    }
    if (!$groups.Count) { Failure 'invalid_request' 'The live template must declare its managed input ranges.' }
    foreach ($name in $groups.Keys) {
        $sheet = $null
        try {
            $sheet = $state.book.Worksheets.Item($name)
            $prior = @{sheet = $name; protected = [bool]$sheet.ProtectContents;
                drawings = [bool]$sheet.ProtectDrawingObjects; scenarios = [bool]$sheet.ProtectScenarios;
                locks = [Collections.Generic.List[object]]::new()}
            if ($prior.protected) {
                try { $sheet.Unprotect() } catch { Failure 'template_protected' "Cannot update the protected template sheet: $name" }
            }
            $state.protection.Add($prior)
            if (!$prior.protected) {
                # Snapshot the original format before making all non-input
                # cells editable, including report cells on the same sheet.
                $all = $sheet.Cells
                try { Remember-Locks $all $prior.locks; $all.Locked = $false } finally { Release-Com $all }
            }
            foreach ($address in $groups[$name]) {
                $range = $sheet.Range($address)
                try {
                    if ($prior.protected) { Remember-Locks $range $prior.locks }
                    $range.Locked = $true
                } finally { Release-Com $range }
            }
            # Pre-existing protection retains its original locked-cell policy.
            $sheet.Protect('', $false, $true, $false, $true, $true, $true, $true)
        } finally { Release-Com $sheet }
    }
}

function Restore-Protection($state) {
    foreach ($prior in $state.protection) {
        $sheet = $null
        try {
            $sheet = $state.book.Worksheets.Item([string]$prior.sheet)
            $sheet.Unprotect()
            foreach ($saved in $prior.locks) {
                $range = $sheet.Range([string]$saved.range)
                try { $range.Locked = [bool]$saved.locked } finally { Release-Com $range }
            }
            if ($prior.protected) { $sheet.Protect('', $prior.drawings, $true, $prior.scenarios) }
        } finally { Release-Com $sheet }
    }
    $state.protection.Clear()
}

function Set-InputProtection($state, [bool]$enabled) {
    foreach ($prior in $state.protection) {
        $sheet = $null
        try {
            $sheet = $state.book.Worksheets.Item([string]$prior.sheet)
            if ($enabled) { $sheet.Protect('', $false, $true, $false, $true, $true, $true, $true) }
            else { $sheet.Unprotect() }
        } finally { Release-Com $sheet }
    }
}

function Read-Checks($state, $payload) {
    $warnings = [Collections.Generic.List[string]]::new()
    $comparisons = [Collections.Generic.List[object]]::new()
    foreach ($warning in $payload.warnings) { $warnings.Add([string]$warning) }
    foreach ($check in $payload.checks) {
        $sheet = Get-SheetName $check $payload.profile
        $actual = Get-Cell $state.book $sheet ([string]$check.count_cell)
        if ($null -eq $actual -or $actual -is [string] -or [double]$actual -ne [double]$check.expected) {
            $warnings.Add(('Measured count mismatch at {0}!{1}: expected {2}, template {3}' -f
                $sheet, $check.count_cell, $check.expected, $actual))
        }
        $comparison = @{operator = $check.operator; expected_count = $check.expected; template_count = $actual}
        if ($check.content_cell) {
            $comparison.direction = $check.direction
            $comparison.material = $check.material
            $comparison.local_content = $check.local_content
            $comparison.template_content = Get-Cell $state.book $sheet ([string]$check.content_cell)
        }
        if ($check.mean_cell) {
            $comparison.local_mean = $check.local_mean
            $comparison.template_mean = Get-Cell $state.book $sheet ([string]$check.mean_cell)
            $comparison.template_mean_semantics = 'grouped'
        }
        if ($check.raw_range) {
            $dataSheet = $null; $rawRange = $null; $functions = $null
            try {
                $dataSheet = $state.book.Worksheets.Item($sheet)
                $rawRange = $dataSheet.Range([string]$check.raw_range)
                $functions = $script:excel.WorksheetFunction
                $comparison.raw_mean = [double]$functions.Average($rawRange)
                $expectedMean = [double]$check.local_mean
                $tolerance = [Math]::Max(1.0, [Math]::Abs($expectedMean)) * 1.0e-9
                if ([Math]::Abs($comparison.raw_mean - $expectedMean) -gt $tolerance) {
                    $warnings.Add(('Raw mean mismatch at {0}!{1}: expected {2}, actual {3}' -f
                        $sheet, $check.raw_range, $expectedMean, $comparison.raw_mean))
                }
            } finally { Release-Com $functions; Release-Com $rawRange; Release-Com $dataSheet }
        }
        $comparisons.Add($comparison)
    }
    return @{warnings = @($warnings.ToArray()); comparisons = @($comparisons.ToArray())}
}

function Calculate-Book($state, $payload) {
    $profile = $payload.profile
    $book = $state.book
    if ($payload.trial_kind -eq 'diameter' -or $profile.trial_kind -eq 'diameter') {
        if ($profile.group_width_cell) {
            Set-Cell $book ([string]$profile.group_width_sheet) ([string]$profile.group_width_cell) $payload.group_width
        }
    } else {
        $methodSheet = $null; $methodRange = $null
        try {
            $methodSheet = $book.Worksheets.Item([string]$profile.method_sheet)
            $methodRange = $methodSheet.Range([string]$profile.method_range)
            $values = $methodRange.Value2
            $selected = @()
            for ($row = 1; $row -le $values.GetLength(0); $row++) {
                if (([string]$values[$row,1] -replace '\s','') -eq ([string]$payload.method_label -replace '\s','')) {
                    $selected += $row
                }
            }
            if ($selected.Count -ne 1) { Failure 'template_method' 'Template standard is missing or ambiguous.' }
            Set-Cell $book ([string]$profile.standard_sheet) ([string]$profile.method_index_cell) ([double]$selected[0])
            $width = $(if ($payload.grouped) { [double]$payload.group_width } else { '不分组' })
            Set-Cell $book ([string]$profile.standard_sheet) ([string]$profile.group_width_cell) $width
        } finally { Release-Com $methodRange; Release-Com $methodSheet }
    }
    # Re-enable protection before potentially long-running template code.
    Set-InputProtection $state $true
    $script:excel.Calculate()
    if ($profile.recalculate_macro) {
        # CU VBA uses ActiveWorkbook/Selection internally. A qualified macro
        # name does not change those globals; select its own workbook for the
        # call, then Sync-Book restores the user's prior window/sheet/scroll.
        [void]$book.Activate()
        $macro = "'" + ([string]$book.Name -replace "'", "''") + "'!" + [string]$profile.recalculate_macro
        [void]$script:excel.Run($macro)
    }
    $script:excel.Calculate()
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while ([int]$script:excel.CalculationState -ne 0) {
        if ($watch.ElapsedMilliseconds -ge 30000) { Failure 'calculation_timeout' 'Excel calculation did not finish within 30 seconds.' }
        Start-Sleep -Milliseconds 25
    }
}

function Sync-Book($state, $payload) {
    if ([int64]$script:request.input_revision -lt [int64]$state.revision) {
        Failure 'stale_revision' 'An older trial state cannot replace the displayed workbook.'
    }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $window = $null; $activeSheet = $null
    $oldEvents = $script:excel.EnableEvents
    $oldUpdating = $script:excel.ScreenUpdating
    $oldCalculation = $script:excel.Calculation
    $state.dirty = $true
    try {
        $window = $script:excel.ActiveWindow
        $activeSheet = $script:excel.ActiveSheet
        if ($window) { $scrollRow = $window.ScrollRow; $scrollColumn = $window.ScrollColumn }
        $script:excel.EnableEvents = $false
        $script:excel.ScreenUpdating = $false
        $script:excel.Calculation = -4135
        # Some Excel versions reject external COM ClearContents even with
        # UserInterfaceOnly enabled. Suspend only our protection for the STA
        # input batch, restoring it before template macros and repaint. Never
        # disable Application.Interactive: killing a blocked worker must not
        # leave the user's visible Excel instance permanently non-interactive.
        Set-InputProtection $state $false
        foreach ($item in $payload.clears) {
            $sheetName = $(if ($item -is [string]) { [string]$payload.profile.data_sheet } else { Get-SheetName $item $payload.profile })
            $address = $(if ($item -is [string]) { $item } else { [string]$item.range })
            $target = $null; $range = $null
            try {
                $target = $state.book.Worksheets.Item($sheetName)
                $range = $target.Range($address)
                $state.stage = "clear $sheetName!$address (UIOnly=$($target.ProtectionMode))"
                [void]$range.ClearContents()
            } finally { Release-Com $range; Release-Com $target }
        }
        foreach ($cell in $payload.cells) {
            $state.stage = "write $($cell.cell)"
            Set-Cell $state.book (Get-SheetName $cell $payload.profile) ([string]$cell.cell) $cell.value
        }
        foreach ($block in $payload.blocks) {
            $rows = @($block.values)
            if (!$rows.Count) { continue }
            $matrix = [Array]::CreateInstance([object], @([int]$rows.Count, [int]1))
            for ($row = 0; $row -lt $rows.Count; $row++) { $matrix.SetValue([double]$rows[$row][0], $row, 0) }
            $target = $null; $range = $null; $start = $null
            try {
                $target = $state.book.Worksheets.Item((Get-SheetName $block $payload.profile))
                $start = $target.Range([string]$block.cell)
                $range = $start.Resize($rows.Count, 1)
                $range.Value2 = $matrix
            } finally { Release-Com $range; Release-Com $start; Release-Com $target }
        }
        Reply @{event = 'inputs_written'; status = 'calculating'; elapsed_ms = $watch.ElapsedMilliseconds}
        Calculate-Book $state $payload
        $checks = Read-Checks $state $payload
        $state.revision = [int64]$script:request.input_revision
        $state.dirty = $false
        $state.warnings = $checks.warnings
        $state.status = $(if ($checks.warnings.Count) { 'review_required' } else { 'synced' })
        $result = Book-Metadata $state
        $result.ok = $true
        $result.status = $state.status
        $result.warnings = $checks.warnings
        $result.comparisons = $checks.comparisons
        $result.elapsed_ms = $watch.ElapsedMilliseconds
        return $result
    } finally {
        $restoreFailure = $null
        try { Set-InputProtection $state $true }
        catch { $restoreFailure = $_; $state.dirty = $true }
        try { $script:excel.Calculation = $oldCalculation } catch {}
        try { $script:excel.EnableEvents = $oldEvents } catch {}
        try { $script:excel.ScreenUpdating = $oldUpdating } catch {}
        # The template macro may select another sheet. Restore the user's view.
        if ($window -and $activeSheet) {
            try {
                $currentWindow = $script:excel.ActiveWindow
                $currentSheet = $script:excel.ActiveSheet
                try {
                    if (!$currentWindow -or $currentWindow.Hwnd -ne $window.Hwnd) { [void]$window.Activate() }
                    if (!$currentSheet -or $currentSheet.Name -ne $activeSheet.Name -or
                        $currentSheet.Parent.Name -ne $activeSheet.Parent.Name) { [void]$activeSheet.Activate() }
                } finally { Release-Com $currentSheet; Release-Com $currentWindow }
                $window.ScrollRow = $scrollRow
                $window.ScrollColumn = $scrollColumn
            } catch {}
        }
        Release-Com $activeSheet
        Release-Com $window
        if ($restoreFailure) { throw $restoreFailure }
    }
}

function Detach-Book([string]$key) {
    if (!$script:sessions.ContainsKey($key)) { return }
    $state = $script:sessions[$key]
    $open = $true
    try {
        if (!(Test-BookOpen $state)) { throw 'Closed workbook has no name.' }
    } catch {
        if ((Error-Code $_) -eq 'excel_busy') { throw }
        $open = $false
    }
    if ($open) { Restore-Protection $state }
    $script:sessions.Remove($key)
    $script:closed[$key] = $state.generation
    Release-Com $state.book
}

try {
    $stopWorker = $false
    while ($null -ne ($line = [Console]::In.ReadLine())) {
        $script:request = $null
        $state = $null
        try {
            $script:request = $line | ConvertFrom-Json
            if (!$script:request.request_id -or @('open','sync','status','show','detach','shutdown') -notcontains $script:request.operation) {
                Failure 'invalid_request' 'A supported operation and request_id are required.'
            }
            $key = [string]$script:request.session_id
            $payload = $script:request.payload
            switch ([string]$script:request.operation) {
                'open' {
                    if (!$key -or !$payload.template -or !(Test-Path -LiteralPath $payload.template -PathType Leaf)) {
                        Failure 'template_missing' 'The selected live-report template does not exist.'
                    }
                    Start-Excel
                    if ($script:sessions.ContainsKey($key)) {
                        $previous = $script:sessions[$key]
                        if ([int64]$previous.generation -eq [int64]$script:request.generation) {
                            $state = Get-State
                            Reply (Sync-Book $state $payload)
                            continue
                        }
                        if ([int64]$previous.generation -gt [int64]$script:request.generation) {
                            Failure 'stale_generation' 'An older connection cannot replace this workbook.'
                        }
                        Detach-Book $key
                    }
                    if ($script:closed.ContainsKey($key) -and [int64]$script:closed[$key] -ge [int64]$script:request.generation) {
                        Failure 'workbook_closed' 'The workbook was closed. A new connection generation is required.'
                    }
                    # Add(template) creates a new unsaved workbook even when
                    # the selected template is XLS/XLSM/XLSX, never edits it.
                    $book = $script:excel.Workbooks.Add([string]$payload.template)
                    $book.CheckCompatibility = $false
                    $state = @{book = $book; generation = [int64]$script:request.generation; revision = -1;
                        dirty = $true; initialized = $false; protection = [Collections.Generic.List[object]]::new(); warnings = @(); status = 'connecting'}
                    $script:sessions[$key] = $state
                    Protect-Inputs $state $payload
                    $state.initialized = $true
                    Reply (Sync-Book $state $payload)
                }
                'sync' { $state = Get-State; Reply (Sync-Book $state $payload) }
                'status' {
                    $state = Get-State
                    $result = Book-Metadata $state
                    $result.ok = $true; $result.status = $state.status
                    $result.warnings = $state.warnings; $result.synced_revision = $state.revision
                    $result.needs_full_sync = $state.dirty
                    Reply $result
                }
                'show' {
                    $state = Get-State
                    $script:excel.Visible = $true
                    [void]$state.book.Activate()
                    $result = Book-Metadata $state
                    $result.ok = $true; $result.status = $state.status
                    Reply $result
                }
                'detach' {
                    if ($script:sessions.ContainsKey($key)) {
                        if ([int64]$script:sessions[$key].generation -ne [int64]$script:request.generation) {
                            Failure 'stale_generation' 'An older connection cannot detach this workbook.'
                        }
                        Detach-Book $key
                    }
                    Reply @{ok = $true; status = 'detached'}
                }
                'shutdown' {
                    foreach ($entry in @($script:sessions.Keys)) { Detach-Book $entry }
                    Reply @{ok = $true; status = 'stopped'}
                    $stopWorker = $true
                    break
                }
            }
            if ($stopWorker) { break }
        } catch {
            $code = Error-Code $_
            $failure = @{ok = $false; status = 'interrupted'; code = $code; error_code = $code;
                error = $_.Exception.Message; message = $_.Exception.Message;
                error_stack = $_.ScriptStackTrace;
                stage = $(if($state){$state.stage}else{''});
                workbook_connected = [bool]($script:sessions.ContainsKey([string]$script:request.session_id));
                needs_full_sync = [bool]($state -and $state.dirty)}
            if ($failure.workbook_connected) {
                try {
                    $metadata = Book-Metadata $script:sessions[[string]$script:request.session_id]
                    foreach ($property in $metadata.Keys) { $failure[$property] = $metadata[$property] }
                } catch {}
            }
            Reply $failure
        }
    }
} finally {
    foreach ($entry in @($script:sessions.Keys)) {
        try { Detach-Book $entry } catch { Release-Com $script:sessions[$entry].book }
    }
    # Visible workbooks are handed to the user, including after input EOF.
    # Releasing COM ownership must never discard an unsaved report.
    Release-Com $script:excel
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
