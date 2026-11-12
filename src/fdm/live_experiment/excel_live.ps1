# Persistent STA host. Quantitative data belongs to FDM; reports belong to users.
# No input protection, Save interception, or per-measurement macro rebuilding.
param()
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
# Process.StandardInput in .NET Framework can emit a UTF-8 preamble before a
# caller writes directly to its BaseStream. Console.In does not consistently
# consume that preamble; use an explicit BOM-aware UTF-8 JSON-lines reader.
$script:inputReader = [IO.StreamReader]::new([Console]::OpenStandardInput(), [Text.UTF8Encoding]::new($false), $true)
$script:excel = $null
$script:excelIdentity = $null
$script:sessions = @{}
$script:candidates = @{}
$script:closed = @{}
$script:request = $null
$script:requestWatch = $null

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class FdmLiveExcelWindow {
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr window, out uint process);
    [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr window);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr window);
    [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr window, int command);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [StructLayout(LayoutKind.Sequential)] private struct FLASHWINFO {
        public uint cbSize;
        public IntPtr hwnd;
        public uint dwFlags;
        public uint uCount;
        public uint dwTimeout;
    }
    [DllImport("user32.dll")] private static extern bool FlashWindowEx(ref FLASHWINFO info);
    public static void FlashTaskbar(IntPtr window) {
        var info = new FLASHWINFO { cbSize=(uint)Marshal.SizeOf(typeof(FLASHWINFO)),
            hwnd=window, dwFlags=2, uCount=3, dwTimeout=0 };
        FlashWindowEx(ref info);
    }
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
    if ($script:requestWatch) { $result.request_elapsed_ms = $script:requestWatch.Elapsed.TotalMilliseconds }
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
        if ($exception.Data -and $exception.Data.Contains('fdm_code')) { return [string]$exception.Data['fdm_code'] }
        if ($exception.HResult -in @(-2147418111, -2147417846, -2146777998)) { return 'excel_busy' }
        $exception = $exception.InnerException
    }
    return 'excel_failed'
}

function Test-ExcelProcess {
    if (!$script:excelIdentity) { return $false }
    $process = Get-Process -Id $script:excelIdentity.id -ErrorAction SilentlyContinue
    return [bool]($process -and $process.StartTime.ToUniversalTime().Ticks.ToString() -eq $script:excelIdentity.created)
}

function Ensure-Ready {
    if (!(Test-ExcelProcess)) { Failure 'workbook_closed' 'The associated Excel process has exited.' }
    $ready = $script:excel.Ready
    if ($null -eq $ready -or !$ready) {
        Failure 'excel_busy' 'Excel is busy editing, saving, or running another operation. Pending input remains queued.'
    }
}

function Require-Object($value, [string]$description) {
    if ($null -eq $value) {
        # A null COM object may be temporary during editing; a responsive server
        # with a missing mapped object is instead a concrete template problem.
        Ensure-Ready
        Failure 'template_mapping' "The template object is unavailable: $description"
    }
}
function Start-Excel {
    if ($script:excel) {
        $process = Get-Process -Id $script:excelIdentity.id -ErrorAction SilentlyContinue
        if ($process -and $process.StartTime.ToUniversalTime().Ticks.ToString() -eq $script:excelIdentity.created) {
            return
        }
        foreach ($key in @($script:sessions.Keys)) { Forget-Book $key }
        foreach ($entry in @($script:candidates.Keys)) { Forget-Candidate $entry }
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

function Test-BookOpen($state) {
    if (!(Test-ExcelProcess)) { return $false }
    Ensure-Ready
    # A closed COM workbook may return null OR throw on Name. Confirm absence
    # from a responsive Workbooks collection rather than equating either with
    # closure; a modal dialog can produce the same symptoms temporarily.
    $nameError = $null
    try { $name = $state.book.Name }
    catch {
        if ((Error-Code $_) -eq 'excel_busy') { throw }
        $nameError = $_
        $name = $null
    }
    if (![string]::IsNullOrEmpty([string]$name)) { return $true }
    Ensure-Ready
    $books = $null; $identity = [IntPtr]::Zero
    $found = $false
    try {
        $books = $script:excel.Workbooks
        Require-Object $books 'Excel.Workbooks'
        $count = $books.Count
        if ($null -eq $count) { Failure 'excel_busy' 'Excel cannot enumerate workbooks during this operation.' }
        $identity = [Runtime.InteropServices.Marshal]::GetIUnknownForObject($state.book)
        for ($index=1; $index -le [int]$count; $index++) {
            $candidate = $null; $candidateIdentity = [IntPtr]::Zero
            try {
                $candidate = $books.Item($index)
                Require-Object $candidate 'enumerated workbook'
                $candidateIdentity = [Runtime.InteropServices.Marshal]::GetIUnknownForObject($candidate)
                if ($identity -eq $candidateIdentity) { $found=$true; break }
            } finally {
                if ($candidateIdentity -ne [IntPtr]::Zero) { [void][Runtime.InteropServices.Marshal]::Release($candidateIdentity) }
                Release-Com $candidate
            }
        }
    } finally {
        if ($identity -ne [IntPtr]::Zero) { [void][Runtime.InteropServices.Marshal]::Release($identity) }
        Release-Com $books
    }
    if ($found) {
        if ($nameError) { throw $nameError }
        Failure 'excel_busy' 'The workbook remains open but Excel temporarily cannot return its name.'
    }
    return $false
}

function Release-BookState($state) {
    # A released association never saves, closes or discards a user-visible book.
    foreach ($sheet in @($state.sheets.Values)) { Release-Com $sheet }
    $state.sheets.Clear()
    Release-Com $state.book
}

function Forget-Book([string]$key) {
    if (!$script:sessions.ContainsKey($key)) { return }
    $state = $script:sessions[$key]
    $script:sessions.Remove($key)
    $script:closed[$key] = $state.generation
    Release-BookState $state
}

function Candidate-Key([string]$session, [int64]$generation) { return "$session|$generation" }
function Forget-Candidate([string]$key) {
    if (!$script:candidates.ContainsKey($key)) { return }
    $state = $script:candidates[$key]
    $script:candidates.Remove($key)
    Release-BookState $state
}

function Template-Identity([string]$path, $expected) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
        Failure 'template_missing' 'The selected live-report template does not exist.'
    }
    $file = Get-Item -LiteralPath $path
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expected -and $expected.sha256 -and [string]$expected.sha256 -ne $hash) {
        Failure 'template_changed' 'The selected template changed before Excel opened it. Select or recreate it again.'
    }
    return @{path=$file.FullName;size=$file.Length;mtime_utc=$file.LastWriteTimeUtc.ToString('o');sha256=$hash}
}

function Capture-BookView($state) {
    $sheets=$null; $active=$null; $window=$null
    $items=[Collections.Generic.List[object]]::new()
    try {
        $sheets=$state.book.Sheets
        Require-Object $sheets 'Workbook.Sheets'
        for ($index=1; $index -le [int]$sheets.Count; $index++) {
            $sheet=$null
            try {
                $sheet=$sheets.Item($index)
                $items.Add(@{name=[string]$sheet.Name;visible=[int]$sheet.Visible})
            } finally { Release-Com $sheet }
        }
        $active=$state.book.ActiveSheet
        Require-Object $active 'Workbook.ActiveSheet'
        $view=@{sheets=@($items.ToArray());active_sheet=[string]$active.Name}
        # The workbook view can be captured without selecting it. Window scroll
        # positions are restored only for explicit macro actions, never per frame.
        $windows=$null
        try {
            $windows=$state.book.Windows
            if ([int]$windows.Count -gt 0) {
                $window=$windows.Item(1)
                $state.window_handle=[int64]$window.Hwnd
                $view.scroll_row=[int]$window.ScrollRow; $view.scroll_column=[int]$window.ScrollColumn
            }
        } finally { Release-Com $windows }
        return $view
    } finally { Release-Com $window; Release-Com $active; Release-Com $sheets }
}

function Restore-BookView($state, $view) {
    if (!$view) { return }
    $sheets=$null; $window=$null; $windows=$null
    try {
        $sheets=$state.book.Sheets
        # Restore visible pages before hidden pages so Excel always has one
        # visible sheet, even if a CU macro temporarily hid every custom report.
        foreach ($visible in @($true,$false)) {
            foreach ($item in $view.sheets) {
                if (([int]$item.visible -eq -1) -ne $visible) { continue }
                $sheet=$null
                try {
                    $sheet=$sheets.Item([string]$item.name)
                    if ([int]$sheet.Visible -ne [int]$item.visible) { $sheet.Visible=[int]$item.visible }
                } finally { Release-Com $sheet }
            }
        }
        $sheet=$null
        try {
            $sheet=$sheets.Item([string]$view.active_sheet)
            if ([int]$sheet.Visible -eq -1) { [void]$sheet.Activate() }
        } finally { Release-Com $sheet }
        $windows=$state.book.Windows
        if ([int]$windows.Count -gt 0 -and $view.ContainsKey('scroll_row')) {
            $window=$windows.Item(1)
            $window.ScrollRow=[int]$view.scroll_row; $window.ScrollColumn=[int]$view.scroll_column
        }
    } finally { Release-Com $window; Release-Com $windows; Release-Com $sheets }
}

function Open-TemplateBook([string]$path) {
    $books=$null
    try {
        $books=$script:excel.Workbooks
        Require-Object $books 'Excel.Workbooks'
        if ([IO.Path]::GetExtension($path).ToLowerInvariant() -in @('.xlt','.xltx','.xltm')) {
            $missing=[Type]::Missing
            # CU's template-open semantics: a new workbook, not template editing.
            return ,$books.Open($path,0,$false,$missing,$missing,$missing,$true,$missing,$missing,$false)
        }
        # A workbook chosen as a template must also produce a separate new book.
        return ,$books.Add($path)
    } finally { Release-Com $books }
}

function Get-State {
    $key = [string]$script:request.session_id
    if (!$script:sessions.ContainsKey($key)) { Failure 'workbook_closed' 'The live workbook is no longer connected.' }
    $state = $script:sessions[$key]
    if ([int64]$state.generation -ne [int64]$script:request.generation) {
        Failure 'stale_generation' 'This request belongs to an earlier workbook connection.'
    }
    $state.stage = 'check workbook'
    if (!(Test-BookOpen $state)) {
        Forget-Book $key
        Failure 'workbook_closed' 'The workbook was closed. Use reconnect to create a new workbook.'
    }
    return $state
}

function Present-NativeWindow($state, [bool]$foreground) {
    $window=[IntPtr]([int64]$state.window_handle)
    $result=@{window_handle=[int64]$window;window_visible=$false;window_minimized=$false;foreground=$false}
    # The cached HWND may have been destroyed or recycled. Never raise an
    # unrelated application's window, even if it now uses the same handle.
    if (!(Test-ExcelProcess) -or ![FdmLiveExcelWindow]::IsWindow($window)) { return $result }
    [uint32]$owner=0
    [void][FdmLiveExcelWindow]::GetWindowThreadProcessId($window,[ref]$owner)
    if ($owner -ne [uint32]$script:excelIdentity.id) { return $result }
    $command=$(if ([FdmLiveExcelWindow]::IsIconic($window)) {9} else {5})
    [void][FdmLiveExcelWindow]::ShowWindowAsync($window,$command)
    if ($foreground) { [void][FdmLiveExcelWindow]::SetForegroundWindow($window) }
    $result.window_visible=[FdmLiveExcelWindow]::IsWindowVisible($window)
    $result.window_minimized=[FdmLiveExcelWindow]::IsIconic($window)
    $result.foreground=[FdmLiveExcelWindow]::GetForegroundWindow() -eq $window
    return $result
}

function Show-Book($payload) {
    $key=[string]$script:request.session_id
    if (!$script:sessions.ContainsKey($key)) { Failure 'workbook_closed' 'The live workbook is no longer connected.' }
    $state=$script:sessions[$key]
    if ([int64]$state.generation -ne [int64]$script:request.generation) {
        Failure 'stale_generation' 'This request belongs to an earlier workbook connection.'
    }
    if (!(Test-ExcelProcess)) { Failure 'workbook_closed' 'The associated Excel process has exited.' }
    $foreground=($null -eq $payload.request_foreground -or [bool]$payload.request_foreground)
    $state.stage='restore workbook window'
    $window=Present-NativeWindow $state $foreground
    try {
        # Native restoration intentionally precedes COM readiness. A cell edit
        # or modal dialog must not keep the Excel window minimized/invisible.
        $state=Get-State
        $state.stage='activate workbook'
        $script:excel.Visible=$true
        [void]$state.book.Activate()
        $windows=$null; $bookWindow=$null
        try {
            $windows=$state.book.Windows
            Require-Object $windows 'Workbook.Windows'
            $bookWindow=$windows.Item(1)
            Require-Object $bookWindow 'Workbook.Window'
            $state.window_handle=[int64]$bookWindow.Hwnd
        } finally { Release-Com $bookWindow; Release-Com $windows }
        $window=Present-NativeWindow $state $foreground
        $result=Safe-Metadata $state
        $result.activation_deferred=$false
        $result.presentation_status=$(if ($foreground -and !$window.foreground) {'attention_required'} else {'shown'})
        $result.presentation_message=$(if ($result.presentation_status -eq 'attention_required') {
            'Excel 窗口已恢复；Windows 未允许切到前台，请点击任务栏中的 Excel。'
        } else {''})
    } catch {
        if ((Error-Code $_) -ne 'excel_busy') { throw }
        $result=@{activation_deferred=$true;presentation_status='deferred';
            presentation_message='等待 Excel 当前编辑或对话框结束后定位工作簿。'}
    }
    # A single finite taskbar notification follows a refused explicit request.
    # Re-check ownership here; no topmost/foreground-lock workaround is used.
    # Deferred activation retries carry request_foreground=false and never flash.
    $result.attention_requested=$false
    $handle=[IntPtr]([int64]$window.window_handle)
    if ($foreground -and $window.window_visible -and !$window.foreground -and
        (Test-ExcelProcess) -and [FdmLiveExcelWindow]::IsWindow($handle) -and
        [FdmLiveExcelWindow]::GetForegroundWindow() -ne $handle) {
        [uint32]$owner=0
        [void][FdmLiveExcelWindow]::GetWindowThreadProcessId($handle,[ref]$owner)
        if ($owner -eq [uint32]$script:excelIdentity.id) {
            [FdmLiveExcelWindow]::FlashTaskbar($handle)
            $result.attention_requested=$true
        }
    }
    foreach ($field in $window.Keys) { $result[$field]=$window[$field] }
    $result.ok=$true; $result.status='synced'
    return $result
}

function Get-SheetName($item, $profile) {
    if ($item.sheet) { return [string]$item.sheet }
    return [string]$profile.data_sheet
}

function Get-Sheet($state, [string]$name) {
    if ([string]::IsNullOrWhiteSpace($name)) { Failure 'template_mapping' 'A mapped worksheet name is missing.' }
    if ($state.sheets.ContainsKey($name)) { return ,$state.sheets[$name] }
    $sheets = $null; $sheet = $null
    try {
        $sheets = $state.book.Worksheets
        Require-Object $sheets 'Workbook.Worksheets'
        try { $sheet = $sheets.Item($name) }
        catch {
            if ((Error-Code $_) -eq 'excel_busy') { throw }
            Ensure-Ready
            Failure 'template_mapping' "The mapped worksheet is unavailable: $name"
        }
        Require-Object $sheet "worksheet $name"
        $state.sheets[$name] = $sheet
        return ,$sheet
    } finally { Release-Com $sheets }
}

function Get-Range($state, [string]$sheetName, [string]$address) {
    $sheet = Get-Sheet $state $sheetName
    $range = $sheet.Range($address)
    Require-Object $range "$sheetName!$address"
    return ,$range
}

function Set-Cell($state, [string]$sheetName, [string]$address, $value) {
    $range = $null
    try {
        $range = Get-Range $state $sheetName $address
        if ($null -eq $value) { [void]$range.ClearContents(); return }
        # A one-cell matrix follows the same COM Value2 route as bulk input and
        # avoids reflection dispatch. A leading apostrophe keeps strings literal.
        $matrix = [Array]::CreateInstance([object], @([int]1, [int]1))
        if ($value -is [string]) { $matrix.SetValue(("'" + $value), 0, 0) }
        else { $matrix.SetValue([double]$value, 0, 0) }
        $range.Value2 = $matrix
    } finally { Release-Com $range }
}

function Book-Metadata($state) {
    $state.stage = 'read Workbook.Name'
    $name = $state.book.Name
    Require-Object $name 'Workbook.Name'
    $state.stage = 'read Workbook.Path'
    $directory = $state.book.Path
    Require-Object $directory 'Workbook.Path'
    $state.stage = 'read Workbook.Saved'
    $saved = $state.book.Saved
    Require-Object $saved 'Workbook.Saved'
    $saveState = $(if (![string]$directory) { 'never_saved' } elseif ([bool]$saved) { 'saved' } else { 'unsaved_changes' })
    $state.stage = 'read Application.Calculation'
    $mode = $script:excel.Calculation
    $state.stage = 'read Application.CalculationState'
    $calculation = $script:excel.CalculationState
    $calculationState = $(if ($null -eq $mode -or $null -eq $calculation) { 'unknown' }
        elseif ([int]$calculation -eq 1) { 'calculating' }
        elseif ([int]$mode -eq -4135) { 'manual' }
        elseif ([int]$calculation -eq 2) { 'pending' }
        else { 'automatic' })
    $state.stage = 'read Workbook.FullName'
    $fullName = $(if ([string]$directory) { $state.book.FullName } else { '' })
    Require-Object $fullName 'Workbook.FullName'
    return @{
        name = [string]$name
        path = [string]$fullName
        saved = [bool]$saved
        save_state = $saveState
        has_file = [bool]([string]$directory)
        excel_pid = $script:excelIdentity.id
        excel_created = $script:excelIdentity.created
        excel_version = $state.excel_version
        calculation_state = $calculationState
        metadata_checked_at = [DateTime]::UtcNow.ToString('o')
        last_input_at = $state.last_input_at
        template_identity = $state.template_identity
        initial_view = $state.initial_view
        active_generation = $state.generation
        window_handle = $state.window_handle
    }
}

function Safe-Metadata($state) {
    try { return Book-Metadata $state }
    catch {
        # Input commits must survive a later status read failure. Do not turn
        # an unknown Saved value into false, or reuse an old false as current.
        return @{save_state = 'unknown'; saved = $null; calculation_state = 'unknown';
            metadata_error = $_.Exception.Message; metadata_error_code = (Error-Code $_);
            metadata_error_stack = $_.ScriptStackTrace; metadata_stage=$state.stage; last_input_at = $state.last_input_at;
            excel_pid = $script:excelIdentity.id; excel_created=$script:excelIdentity.created; excel_version = $state.excel_version;
            template_identity=$state.template_identity;initial_view=$state.initial_view;active_generation=$state.generation}
    }
}

function Apply-ReportSettings($state, $settings, $profile) {
    if ($profile.trial_kind -eq 'diameter' -or $state.trial_kind -eq 'diameter') {
        Set-Cell $state ([string]$profile.group_width_sheet) ([string]$profile.group_width_cell) $settings.group_width
        return
    }
    $range = $null
    try {
        $range = Get-Range $state ([string]$profile.method_sheet) ([string]$profile.method_range)
        $values = $range.Value2
        Require-Object $values 'template standard list'
        if ($values -isnot [Array] -or $values.Rank -ne 2) {
            Failure 'template_method' 'The template standard list must be a two-dimensional range.'
        }
        $selected = @()
        for ($row = $values.GetLowerBound(0); $row -le $values.GetUpperBound(0); $row++) {
            if (([string]$values[$row,$values.GetLowerBound(1)] -replace '\s','') -eq
                ([string]$settings.method_label -replace '\s','')) { $selected += ($row - $values.GetLowerBound(0) + 1) }
        }
        if ($selected.Count -ne 1) { Failure 'template_method' 'Template standard is missing or ambiguous.' }
        Set-Cell $state ([string]$profile.standard_sheet) ([string]$profile.method_index_cell) ([double]$selected[0])
        $width = $(if ($settings.grouped) { [double]$settings.group_width } else { '不分组' })
        Set-Cell $state ([string]$profile.standard_sheet) ([string]$profile.group_width_cell) $width
    } finally { Release-Com $range }
}

function Recalculate-Report($state, $profile) {
    # Only an explicit user action runs the template macro. CU's grouping macro
    # hides arbitrary custom reports, so restore the pre-action presentation even
    # though its intentional formula/statistics changes remain in the workbook.
    if (!$state.pending_view) { $state.pending_view=Capture-BookView $state }
    $view=$state.pending_view
    $original=$null
    try {
        if ($profile.recalculate_macro) {
            [void]$state.book.Activate()
            $macro = "'" + ([string]$state.book.Name -replace "'", "''") + "'!" + [string]$profile.recalculate_macro
            try { [void]$script:excel.Run($macro) }
            catch {
                if ((Error-Code $_) -eq 'excel_busy') { throw }
                Failure 'template_macro' ("Template macro failed: " + $_.Exception.Message)
            }
        }
        $script:excel.Calculate()
    } catch { $original=$_ }
    try { Restore-BookView $state $view; $state.pending_view=$null }
    catch {
        if (!$original) { throw }
        $original.Exception.Data['view_restore_error']=$_.Exception.Message
    }
    if ($original) { throw $original }
}

function New-Batch($state, $payload, [string]$operation) {
    $id = [string]$payload.batch_id
    if (!$id) { $id = [string]$script:request.request_id }
    $signature = $payload | ConvertTo-Json -Depth 30 -Compress
    if ($state.pending) {
        if ($state.pending.id -ne $id -or $state.pending.operation -ne $operation -or $state.pending.signature -ne $signature) {
            Failure 'batch_pending' 'An unfinished input batch must be resumed before another batch is applied.'
        }
        return $state.pending
    }
    if ($state.last_batch -and $state.last_batch.id -eq $id) {
        if ($state.last_batch.operation -ne $operation -or $state.last_batch.signature -ne $signature) {
            Failure 'batch_identity' 'The same batch identity cannot carry different input.'
        }
        return $state.last_batch
    }
    $ops = [Collections.Generic.List[object]]::new()
    if ($operation -in @('open','update_report')) {
        foreach ($cell in $payload.report_cells) {
            $ops.Add(@{kind='cell'; family='report'; sheet=(Get-SheetName $cell $payload.profile); address=[string]$cell.cell; value=$cell.value})
        }
        if ($operation -eq 'update_report' -and $null -ne $payload.report_settings) {
            $ops.Add(@{kind='settings'; family='report'; settings=$payload.report_settings})
            $ops.Add(@{kind='macro'; family='macro'})
        }
    }
    if ($operation -eq 'recalculate') { $ops.Add(@{kind='macro';family='macro'}) }
    $inputStart = $ops.Count
    if ($operation -in @('open','sync')) {
        foreach ($item in $payload.clears) {
            $sheet = $(if ($item -is [string]) { [string]$payload.profile.data_sheet } else { Get-SheetName $item $payload.profile })
            $address = $(if ($item -is [string]) { $item } else { [string]$item.range })
            $ops.Add(@{kind='clear';family='input';sheet=$sheet;address=$address})
        }
        foreach ($cell in $payload.cells) {
            $ops.Add(@{kind='cell';family='input';sheet=(Get-SheetName $cell $payload.profile);address=[string]$cell.cell;value=$cell.value})
        }
        foreach ($block in $payload.blocks) {
            if (@($block.values).Count) {
                $ops.Add(@{kind='block';family='input';sheet=(Get-SheetName $block $payload.profile);address=[string]$block.cell;values=@($block.values)})
            }
        }
    }
    $batch = @{id=$id;signature=$signature;operation=$operation;revision=[int64]$script:request.input_revision;
        profile=$payload.profile;ops=$ops;index=0;input_start=$inputStart;input_written=$false;done=$false;
        warnings=@($payload.warnings);timings=@{input_ms=0.0;report_ms=0.0;macro_ms=0.0;metadata_ms=0.0}}
    $state.pending = $batch
    return $batch
}

function Invoke-BatchOperation($state, $op, $profile) {
    $state.stage = "$($op.kind) $($op.sheet)!$($op.address)"
    $range = $null; $start = $null
    try {
        switch ($op.kind) {
            'clear' { $range = Get-Range $state $op.sheet $op.address; [void]$range.ClearContents() }
            'cell' { Set-Cell $state $op.sheet $op.address $op.value }
            'block' {
                $rows = @($op.values)
                $matrix = [Array]::CreateInstance([object], @([int]$rows.Count, [int]1))
                for ($row=0; $row -lt $rows.Count; $row++) { $matrix.SetValue([double]$rows[$row][0],$row,0) }
                $start = Get-Range $state $op.sheet $op.address
                $range = $start.Resize($rows.Count,1)
                Require-Object $range "$($op.sheet)!$($op.address) resized input"
                $range.Value2 = $matrix
            }
            'settings' { $state.stage='write report settings'; Apply-ReportSettings $state $op.settings $profile }
            'macro' { $state.stage='recalculate report'; Recalculate-Report $state $profile }
        }
    } catch {
        $original = $_
        if ($op.kind -in @('clear','cell','block') -and (Error-Code $original) -eq 'excel_failed') {
            try {
                $sheet = Get-Sheet $state $op.sheet
                if ($sheet.ProtectContents) {
                    Failure 'template_protected' ("The template protects this input region: $($op.sheet)!$($op.address). Its original protection was not changed.")
                }
            } catch {
                if ((Error-Code $_) -eq 'template_protected') { throw }
            }
        }
        throw $original
    } finally { Release-Com $range; Release-Com $start }
}

function Run-Batch($state, $payload, [string]$operation, [bool]$suppressProgress=$false) {
    if ($operation -in @('open','sync') -and [int64]$script:request.input_revision -lt [int64]$state.revision) {
        Failure 'stale_revision' 'An older trial state cannot replace the displayed workbook.'
    }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $batch = New-Batch $state $payload $operation
    while ($batch.index -lt $batch.ops.Count) {
        $op = $batch.ops[$batch.index]
        $opWatch = [Diagnostics.Stopwatch]::StartNew()
        try { Invoke-BatchOperation $state $op $batch.profile }
        finally { $batch.timings[($op.family + '_ms')] += $opWatch.Elapsed.TotalMilliseconds }
        # Advance only after the exact COM write returned. A rejected call resumes
        # at the same address; already acknowledged operations are not repeated.
        $batch.index++
    }
    if (!$batch.done) {
        $batch.done = $true
        if ($operation -in @('open','sync')) {
            $batch.input_written = $true
            $state.revision = $batch.revision
            $state.last_input_at = [DateTime]::UtcNow.ToString('o')
            $state.warnings = $batch.warnings
            $state.initialized = $true
        }
        $state.last_batch = $batch
        $state.pending = $null
    }
    $state.stage = 'input complete'
    if ($batch.input_written -and !$suppressProgress) {
        # ScreenUpdating was never disabled. This is an input ACK, not a promise
        # that every report formula or an independent user macro has completed.
        Reply @{event='inputs_written';input_written=$true;input_committed=$true;batch_id=$batch.id;
            completed_ops=$batch.index;status='synced';elapsed_ms=$watch.Elapsed.TotalMilliseconds}
    }
    $metaWatch = [Diagnostics.Stopwatch]::StartNew()
    $result = Safe-Metadata $state
    $batch.timings.metadata_ms = $metaWatch.Elapsed.TotalMilliseconds
    $result.ok = $true
    $result.status = 'synced'
    $result.input_written = $batch.input_written
    $result.input_committed = $batch.input_written
    $result.batch_id = $batch.id
    $result.completed_ops = $batch.index
    $result.warnings = $state.warnings
    $result.synced_revision = $state.revision
    $result.elapsed_ms = $watch.Elapsed.TotalMilliseconds
    $result.timings = $batch.timings
    return $result
}

try {
    $stopWorker = $false
    while ($null -ne ($line = $script:inputReader.ReadLine())) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $script:request = $null
        $script:requestWatch = [Diagnostics.Stopwatch]::StartNew()
        $state = $null
        $key = $null
        $payload = $null
        $candidateKey = $null
        try {
            $script:request = $line | ConvertFrom-Json
            if (!$script:request.request_id -or @('open','sync','status','show','update_report','recalculate','detach','discard_candidate','shutdown') -notcontains $script:request.operation) {
                Failure 'invalid_request' 'A supported operation and request_id are required.'
            }
            $key = [string]$script:request.session_id
            $payload = $script:request.payload
            switch ([string]$script:request.operation) {
                'open' {
                    if (!$key -or !$payload.template) { Failure 'template_missing' 'A selected template is required.' }
                    Start-Excel
                    Ensure-Ready
                    if ($script:sessions.ContainsKey($key)) {
                        $previous = $script:sessions[$key]
                        if ([int64]$previous.generation -eq [int64]$script:request.generation) {
                            $state = Get-State
                            Reply (Run-Batch $state $payload 'open')
                            continue
                        }
                        if ([int64]$previous.generation -gt [int64]$script:request.generation) {
                            Failure 'stale_generation' 'An older connection cannot replace this workbook.'
                        }
                    }
                    if ($script:closed.ContainsKey($key) -and [int64]$script:closed[$key] -ge [int64]$script:request.generation) {
                        Failure 'workbook_closed' 'The workbook was closed. A new connection generation is required.'
                    }
                    $candidateKey=Candidate-Key $key ([int64]$script:request.generation)
                    if ($script:candidates.ContainsKey($candidateKey)) {
                        $state=$script:candidates[$candidateKey]
                        if (!(Test-BookOpen $state)) { Failure 'workbook_closed' 'The pending template workbook was closed.' }
                    } else {
                        $identity=Template-Identity ([string]$payload.template) $payload.template_identity
                        $book=Open-TemplateBook ([string]$identity.path)
                        Require-Object $book 'new template workbook'
                        $state=@{book=$book;generation=[int64]$script:request.generation;revision=-1;
                            initialized=$false;sheets=@{};warnings=@();pending=$null;last_batch=$null;
                            last_input_at=$null;trial_kind=[string]$payload.trial_kind;excel_version=[string]$script:excel.Version;
                            template_identity=$identity;initial_view=$null;candidate_name=$null;source_verified=$false;stage='verify opened template'}
                        $script:candidates[$candidateKey]=$state
                    }
                    # Each step is resumable: an Excel busy response while reading
                    # Name/view must not skip source verification on the retry.
                    if (!$state.candidate_name) { $state.candidate_name=[string]$state.book.Name }
                    if (!$state.source_verified) {
                        [void](Template-Identity ([string]$state.template_identity.path) $state.template_identity)
                        $state.source_verified=$true
                    }
                    if (!$state.initial_view) { $state.initial_view=Capture-BookView $state }
                    $result=Run-Batch $state $payload 'open' $true
                    # Switch only after the candidate's inputs are complete. The
                    # previous workbook remains usable after any failed attempt.
                    $replaced=$script:sessions.ContainsKey($key)
                    if ($replaced) { Forget-Book $key }
                    $script:sessions[$key]=$state
                    $script:candidates.Remove($candidateKey)
                    $result.template_replaced=$replaced
                    $result.binding_committed=$true
                    Reply $result
                }
                'sync' { $state=Get-State; Reply (Run-Batch $state $payload 'sync') }
                'update_report' { $state=Get-State; Reply (Run-Batch $state $payload 'update_report') }
                'recalculate' { $state=Get-State; Reply (Run-Batch $state $payload 'recalculate') }
                'status' {
                    $state = Get-State
                    $state.stage = 'read workbook status'
                    $result = Book-Metadata $state
                    $result.ok=$true; $result.status='synced'; $result.warnings=$state.warnings
                    $result.synced_revision=$state.revision
                    Reply $result
                }
                'show' {
                    Reply (Show-Book $payload)
                }
                'detach' {
                    if ($script:sessions.ContainsKey($key) -and
                        [int64]$script:sessions[$key].generation -ne [int64]$script:request.generation) {
                        Failure 'stale_generation' 'An older connection cannot detach this workbook.'
                    }
                    # No protection or Excel state needs restoration. A user can
                    # stop synchronization while Excel is editing or saving.
                    Forget-Book $key
                    foreach ($entry in @($script:candidates.Keys)) {
                        if ($entry.StartsWith($key + '|', [StringComparison]::Ordinal)) { Forget-Candidate $entry }
                    }
                    Reply @{ok=$true;status='detached'}
                }
                'discard_candidate' {
                    Forget-Candidate (Candidate-Key $key ([int64]$script:request.generation))
                    Reply @{ok=$true;status='candidate_discarded'}
                }
                'shutdown' {
                    foreach ($entry in @($script:sessions.Keys)) { Forget-Book $entry }
                    foreach ($entry in @($script:candidates.Keys)) { Forget-Candidate $entry }
                    Reply @{ok=$true;status='stopped'}
                    $stopWorker=$true
                }
            }
            if ($stopWorker) { break }
        } catch {
            $errorRecord = $_
            $code = Error-Code $errorRecord
            if ($code -eq 'workbook_closed' -and !$candidateKey -and $key -and $script:sessions.ContainsKey($key)) { Forget-Book $key }
            $connected = [bool]($key -and $script:sessions.ContainsKey($key))
            if ($connected -and !$state) { $state=$script:sessions[$key] }
            $batch = $(if ($state) { $state.pending } else { $null })
            $requestBatchId = $(if($payload.batch_id){[string]$payload.batch_id}else{[string]$script:request.request_id})
            $ownsBatch = [bool]($batch -and $batch.id -eq $requestBatchId -and
                $batch.operation -eq [string]$script:request.operation)
            $batchConflict = $code -in @('batch_pending','batch_identity','stale_revision','stale_generation','invalid_request')
            $unknownInput = [bool]($ownsBatch -and !$batchConflict -and $batch.operation -in @('open','sync') -and
                $batch.index -ge $batch.input_start -and !$batch.input_written)
            if ($ownsBatch -and !$batchConflict -and $code -ne 'excel_busy') {
                # A genuine failure requires an explicit new command. Keeping
                # this failed cursor active would block the user's repair action.
                $state.failed_batch = $batch
                $state.pending = $null
            }
            $failure = @{ok=$false;status='interrupted';code=$code;error_code=$code;
                error=$errorRecord.Exception.Message;message=$errorRecord.Exception.Message;error_stack=$errorRecord.ScriptStackTrace;
                stage=$(if($state){$state.stage}else{'connect'});workbook_connected=$connected;
                input_written=$false;input_committed=$false;save_state='unknown';saved=$null;
                calculation_state='unknown';view_restore_pending=[bool]($state -and $state.pending_view);
                needs_full_sync=($unknownInput -and !$candidateKey -and $code -ne 'excel_busy');
                batch_id=$(if($batch){$batch.id}else{$payload.batch_id});completed_ops=$(if($batch){$batch.index}else{0})}
            if ($script:excelIdentity) {
                $failure.excel_pid=$script:excelIdentity.id
                $failure.excel_created=$script:excelIdentity.created
            }
            # Do not issue more COM calls while Excel is busy: they can block or
            # turn a recoverable error into a different diagnosis.
            if ($connected) {
                $failure.excel_pid=$script:excelIdentity.id
                $failure.excel_created=$script:excelIdentity.created
                $active=$script:sessions[$key]
                $failure.last_input_at=$active.last_input_at
                $failure.synced_revision=$active.revision
                $failure.active_generation=$active.generation
            }
            if ($candidateKey) {
                $pendingCandidate=($code -eq 'excel_busy' -and $script:candidates.ContainsKey($candidateKey))
                $failure.candidate_pending=$pendingCandidate
                $failure.candidate_generation=[int64]$script:request.generation
                if ($script:candidates.ContainsKey($candidateKey)) {
                    $failure.candidate_template_identity=$script:candidates[$candidateKey].template_identity
                    $failure.candidate_name=$script:candidates[$candidateKey].candidate_name
                }
                if (!$pendingCandidate) { Forget-Candidate $candidateKey }
            }
            Reply $failure
        }
    }
} finally {
    foreach ($entry in @($script:sessions.Keys)) { Forget-Book $entry }
    foreach ($entry in @($script:candidates.Keys)) { Forget-Candidate $entry }
    # Releasing only our references must never discard the user's unsaved book.
    Release-Com $script:excel
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
