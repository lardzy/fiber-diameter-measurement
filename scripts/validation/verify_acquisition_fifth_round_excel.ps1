# Verify only the report just created by this probe; never attach to an
# unrelated active workbook or terminate an existing Excel process.
param([string]$Report,[string]$Template,[string]$Expected,[string]$Output)
$ErrorActionPreference='Stop'
$prefix='C:\tmp\fdm-cu-live-fifth-round-20261020\build\fifth-round\frozen-gui\'
foreach($path in @($Report,$Template,$Expected,$Output)) {
    if(-not [IO.Path]::GetFullPath($path).StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) {throw 'Excel validation path escaped owned GUI run'}
}
$request=[IO.File]::ReadAllText($Expected)|ConvertFrom-Json
$beforeHash=(Get-FileHash -LiteralPath $Template).Hash
$excel=$null;$book=$null;$original=$null;$owned=$false
$result=[ordered]@{passed=$false;report=$Report;template=$Template;expected=$Expected;inputs_verified=0;diameters_verified=0;counts_verified=0;sheets=@();errors=@()}
try {
    $excel=[Runtime.InteropServices.Marshal]::GetActiveObject('Excel.Application')
    $matches=@($excel.Workbooks|Where-Object {$_.FullName -eq [IO.Path]::GetFullPath($Report)})
    if($matches.Count -ne 1 -or $excel.Workbooks.Count -ne 1) {throw 'Active Excel is not exclusively the owned report'}
    $book=$matches[0];$owned=$true
    $result['excel_version']=[string]$excel.Version
    $sheet=$book.Worksheets.Item([string]$request.profile.data_sheet)
    foreach($item in $request.cells) {
        $actual=$sheet.Range([string]$item.cell).Value2
        if($null -eq $item.value) {if($null -ne $actual){throw "Expected blank $($item.cell)"}}
        elseif($item.value -is [string]) {if($actual -isnot [string] -or $actual -cne $item.value){throw "String/leading zero mismatch $($item.cell)"}}
        elseif([Math]::Abs([double]$actual-[double]$item.value) -gt 0.0000001){throw "Input mismatch $($item.cell)"}
        $result.inputs_verified++
    }
    foreach($block in $request.blocks) {
        $start=$sheet.Range([string]$block.cell)
        for($i=0;$i -lt $block.values.Count;$i++) {
            $actual=$start.Offset($i,0).Value2
            if([Math]::Abs([double]$actual-[double]$block.values[$i][0]) -gt 0.0000001){throw 'Raw diameter mismatch'}
            $result.diameters_verified++
        }
    }
    foreach($check in $request.checks) {
        if([double]$sheet.Range([string]$check.count_cell).Value2 -ne [double]$check.expected){throw "Calculated count mismatch $($check.count_cell)"}
        $result.counts_verified++
    }
    $excel.DisplayAlerts=$false
    $original=$excel.Workbooks.Open([IO.Path]::GetFullPath($Template),0,$true)
    if($book.HasVBProject -ne $original.HasVBProject){throw 'VBA project presence changed'}
    if($book.Worksheets.Count -ne $original.Worksheets.Count){throw 'Template sheet count changed'}
    $result['vba_preserved']=[bool]$book.HasVBProject
    foreach($sheet in $original.Worksheets) {
        $outputSheet=$book.Worksheets.Item($sheet.Name)
        $printArea=[string]$sheet.PageSetup.PrintArea
        if([string]$outputSheet.PageSetup.PrintArea -ne $printArea){throw "Print area changed: $($sheet.Name)"}
        $formulaCount=0
        try {$formulaCount=$outputSheet.UsedRange.SpecialCells(-4123).Count} catch { }
        $result.sheets+=@{name=[string]$sheet.Name;print_area=$printArea;formula_cells=[int]$formulaCount}
    }
    # The original formula strip is outside all raw-data inputs and must remain
    # intact; recalculate macro output below this area is intentionally allowed.
    $a=$original.Worksheets.Item([string]$request.profile.data_sheet)
    $b=$book.Worksheets.Item([string]$request.profile.data_sheet)
    $checked=0
    for($row=9;$row -le 24;$row++) {for($col=1;$col -le 57;$col++) {
        $cell=$a.Cells.Item($row,$col)
        if($cell.HasFormula) {
            if([string]$b.Cells.Item($row,$col).Formula -cne [string]$cell.Formula){throw "Formula changed at $row,$col"}
            $checked++
        }
    }}
    if($checked -eq 0){throw 'No template formulas were verified'}
    $result['formulas_verified']=$checked
    $result.passed=$true
} catch {$result.errors+=@($_.Exception.Message)} finally {
    if($original){$original.Close($false);[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($original)}
    if($owned -and $book){$book.Close($false);[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($book);$excel.Quit()}
    if($excel){[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($excel)}
    [GC]::Collect();[GC]::WaitForPendingFinalizers()
    if((Get-FileHash -LiteralPath $Template).Hash -ne $beforeHash){$result.passed=$false;$result.errors+=@('Template hash changed')}
    $result['report_sha256']=(Get-FileHash -LiteralPath $Report).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText($Output,($result|ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
}
if(-not $result.passed){throw ('Excel verification failed: '+($result.errors -join '; '))}
