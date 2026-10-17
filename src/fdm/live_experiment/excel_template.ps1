param([Parameter(Mandatory=$true)][string]$RequestPath,
      [Parameter(Mandatory=$true)][string]$ResultPath, [switch]$Cleanup)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$ownerPath = $ResultPath + '.owner.json'
if ($Cleanup) {
    try {
        $cleanupStatus = 'owner_absent'
        if (Test-Path -LiteralPath $ownerPath) {
            $owner = Get-Content -LiteralPath $ownerPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $ownedProcess = Get-Process -Id $owner.id -ErrorAction SilentlyContinue
            $cleanupStatus = 'not_owned_or_already_closed'
            if (!$owner.handed_off -and $ownedProcess -and $ownedProcess.ProcessName -eq 'EXCEL' -and
                $ownedProcess.StartTime.ToUniversalTime().Ticks.ToString() -eq $owner.created) {
                Stop-Process -Id $owner.id -Force
                if (!$ownedProcess.WaitForExit(5000)) { throw 'Owned Excel instance did not exit after cleanup.' }
                $cleanupStatus = 'terminated_owned_instance'
            }
        }
        @{ok=$true; status=$cleanupStatus} | ConvertTo-Json |
            Set-Content -LiteralPath ($ResultPath + '.cleanup.json') -Encoding UTF8
        exit 0
    } catch {
        @{ok=$false; error=$_.Exception.Message; error_type=$_.Exception.GetType().FullName} | ConvertTo-Json |
            Set-Content -LiteralPath ($ResultPath + '.cleanup.json') -Encoding UTF8
        exit 1
    }
}
$excel = $null; $book = $null; $owned = $false; $handedOff = $false; $excelVersion = $null
$phase = 'request'
$warnings = [System.Collections.Generic.List[string]]::new()
$comparisons = [System.Collections.Generic.List[object]]::new()
function Stage([string]$name) {
    $script:phase = $name
    @{stage=$name} | ConvertTo-Json | Set-Content -LiteralPath ($ResultPath + '.progress.json') -Encoding UTF8
}
function SetCell($sheet, [string]$address, $value) {
    $range = $sheet.Range($address)
    try {
        $range.GetType().InvokeMember('Value2', [Reflection.BindingFlags]::SetProperty,
            $null, $range, @($value)) | Out-Null
    } finally { [Runtime.InteropServices.Marshal]::FinalReleaseComObject($range) | Out-Null }
}
function LiteralValue($value) {
    if ($null -eq $value) { return $null }
    if ($value -is [string]) { return "'" + $value }
    return [double]$value
}
try {
    $request = Get-Content -LiteralPath $RequestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($null -eq $request -or @('edit', 'inspect', 'export') -notcontains $request.operation -or
        [string]::IsNullOrWhiteSpace([string]$request.template)) {
        throw 'Invalid template request: operation and template are required.'
    }
    Stage '启动独立 Excel'
    $existing = @(Get-Process EXCEL -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
    Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class FdmExcelWindow { [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint id); }'
    $excel = New-Object -ComObject Excel.Application
    [uint32]$excelId = 0
    [FdmExcelWindow]::GetWindowThreadProcessId([IntPtr]$excel.Hwnd, [ref]$excelId) | Out-Null
    if (!$excelId -or $existing -contains [int]$excelId) { throw 'Excel did not create an independent report instance.' }
    $owned = $true
    $ownedProcess = Get-Process -Id $excelId
    $owner = @{id=$excelId; created=$ownedProcess.StartTime.ToUniversalTime().Ticks.ToString(); handed_off=$false}
    $owner | ConvertTo-Json | Set-Content -LiteralPath $ownerPath -Encoding UTF8
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.EnableEvents = $false
    # Respect the user's Office macro policy. Never change Trust Center settings.
    $excel.AutomationSecurity = 2
    $excelVersion = [string]$excel.Version
    $missing = [Type]::Missing
    Stage '打开模板副本'
    $book = $excel.Workbooks.Open([string]$request.template, 0, $false, $missing, $missing,
        $missing, $true, $missing, $missing, ($request.operation -eq 'edit'))
    if ($request.operation -eq 'edit') {
        $excel.EnableEvents=$true; $excel.DisplayAlerts=$true; $excel.Visible=$true
        $handedOff=$true
        $owner.handed_off=$true
        $owner | ConvertTo-Json | Set-Content -LiteralPath $ownerPath -Encoding UTF8
        @{ok=$true; operation='edit'; template=[string]$request.template; handed_off=$true; excel_version=$excelVersion} |
            ConvertTo-Json | Set-Content -LiteralPath $ResultPath -Encoding UTF8
        return
    }
    $profile = $request.profile
    $data = $book.Worksheets.Item([string]$profile.data_sheet)
    $standard = $book.Worksheets.Item([string]$profile.standard_sheet)
    $methods = $book.Worksheets.Item([string]$profile.method_sheet).Range([string]$profile.method_range).Value2
    $labels = @()
    for ($r = 1; $r -le $methods.GetLength(0); $r++) {
        $label = [string]$methods[$r,1]
        if ($label) { $labels += @{index=$r; label=$label} }
    }
    if ($request.operation -eq 'inspect') {
        $result = @{ok=$true; methods=$labels; sheets=@($book.Worksheets | ForEach-Object {$_.Name});
            template=[string]$request.template}
    } else {
        if ([IO.Path]::GetFullPath($request.destination) -eq [IO.Path]::GetFullPath($request.template)) {
            throw 'The report must not overwrite its template.'
        }
        $selected = @($labels | Where-Object { ($_.label -replace '\s','') -eq ($request.method_label -replace '\s','') })
        if ($selected.Count -ne 1) { throw ('Template standard is missing or ambiguous: ' + $request.method_label) }
        $excel.Calculation = -4135 # manual while writing all input blocks
        Stage '清理原始输入区'
        foreach ($range in $request.clears) { $data.Range([string]$range).ClearContents() | Out-Null }
        Stage '写入试验记录'
        foreach ($cell in $request.cells) { SetCell $data ([string]$cell.cell) (LiteralValue $cell.value) }
        foreach ($block in $request.blocks) {
            $rows = @($block.values)
            $matrix = [Array]::CreateInstance([object], @([int]$rows.Count, [int]1))
            for ($r=0; $r -lt $rows.Count; $r++) { $matrix.SetValue([double]$rows[$r][0], $r, 0) }
            $data.Range([string]$block.cell).Resize($rows.Count,1).Value2 = $matrix
        }
        SetCell $standard ([string]$profile.method_index_cell) ([double]$selected[0].index)
        SetCell $standard ([string]$profile.group_width_cell) $(if ($request.grouped) { [double]$request.group_width } else { '不分组' })
        # The CU macro reads C10 and row-9 counts. Refresh those formulas before
        # invoking it; manual calculation must not expose stale template values.
        Stage '更新输入公式'
        $excel.CalculateFull()
        # Do not call Standard_Change: that resets the user's selected group width.
        if ($profile.recalculate_macro) {
            Stage ('运行模板宏：' + [string]$profile.recalculate_macro)
            try {
                $macro = "'" + ($book.Name -replace "'", "''") + "'!" + [string]$profile.recalculate_macro
                $excel.Run($macro) | Out-Null
            } catch { $warnings.Add('Template grouping macro did not complete: ' + $_.Exception.Message) }
        }
        Stage '重算模板结果'
        $excel.CalculateFull()
        foreach ($warning in $request.warnings) { $warnings.Add([string]$warning) }
        foreach ($check in $request.checks) {
            $actual = $data.Range([string]$check.count_cell).Value2
            if ($null -eq $actual -or $actual -is [string] -or [double]$actual -ne [double]$check.expected) {
                $warnings.Add(('Measured count mismatch at {0}!{1}: expected {2}, template {3}' -f
                    $profile.data_sheet, $check.count_cell, $check.expected, $actual))
            }
            $value = $data.Range([string]$check.content_cell).Value2
            $comparisons.Add(@{operator=$check.operator; direction=$check.direction; material=$check.material;
                local_content=$check.local_content; template_content=$value})
        }
        $data.Activate()
        $destination = [IO.Path]::GetFullPath($request.destination)
        $extension = [IO.Path]::GetExtension($destination).ToLowerInvariant()
        $format = switch ($extension) { '.xls' {56}; '.xlsm' {52}; '.xlsx' {51}; default {throw 'Unsupported report extension.'} }
        if ($extension -eq '.xlsx' -and $book.HasVBProject) { throw 'XLSX would discard the template VBA; select XLS or XLSM.' }
        Stage '保存报告副本'
        # An inherited XLT compatibility-check setting opens a modal dialog
        # even in this hidden automation instance. Only the report copy changes.
        $book.CheckCompatibility = $false
        $excel.DisplayAlerts = $false
        $excel.EnableEvents = $false
        $book.SaveAs($destination, $format)
        $excel.Calculation = -4105
        $excel.EnableEvents = $true
        $excel.DisplayAlerts = $true
        $result = @{ok=$true; status=$(if($warnings.Count){'review_required'}else{'ready'});
            destination=$destination; warnings=@($warnings.ToArray()); comparisons=@($comparisons.ToArray());
            method=[string]$selected[0].label; snapshot_id=$request.snapshot_id; snapshot_revision=$request.snapshot_revision;
            sheets=@($book.Worksheets | ForEach-Object {$_.Name})}
        if ($request.show) {
            $excel.Visible=$true; $handedOff=$true
            $owner.handed_off=$true
            $owner | ConvertTo-Json | Set-Content -LiteralPath $ownerPath -Encoding UTF8
        }
    }
    $result.excel_version = $excelVersion
    $result | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath $ResultPath -Encoding UTF8
} catch {
    @{ok=$false; error=$_.Exception.Message; stack=$_.ScriptStackTrace;
        error_type=$_.Exception.GetType().FullName; error_id=$_.FullyQualifiedErrorId; phase=$phase; excel_version=$excelVersion;
        code=$(if ($phase -eq 'request') {'invalid_request'} else {'excel_failed'})} | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $ResultPath -Encoding UTF8
    exit 1
} finally {
    $closeErrors = [System.Collections.Generic.List[string]]::new()
    if ($owned -and !$handedOff) {
        if ($book) { try { $book.Close($false) } catch { $closeErrors.Add('Workbook.Close: ' + $_.Exception.Message) } }
        if ($excel) { try { $excel.Quit() } catch { $closeErrors.Add('Excel.Quit: ' + $_.Exception.Message) } }
    }
    if ($book) {
        try { [Runtime.InteropServices.Marshal]::FinalReleaseComObject($book) | Out-Null }
        catch { $closeErrors.Add('Workbook.Release: ' + $_.Exception.Message) }
    }
    if ($excel) {
        try { [Runtime.InteropServices.Marshal]::FinalReleaseComObject($excel) | Out-Null }
        catch { $closeErrors.Add('Excel.Release: ' + $_.Exception.Message) }
    }
    try { [GC]::Collect(); [GC]::WaitForPendingFinalizers() }
    catch { $closeErrors.Add('COM finalization: ' + $_.Exception.Message) }
    # Closing is independent of the report result: a saved report must not be
    # rewritten as a failed calculation when COM shutdown needs recovery.
    try {
        @{ok=($closeErrors.Count -eq 0); owned=$owned; handed_off=$handedOff; errors=@($closeErrors.ToArray())} |
            ConvertTo-Json -Depth 4 | Set-Content -LiteralPath ($ResultPath + '.close.json') -Encoding UTF8
    } catch {}
}
