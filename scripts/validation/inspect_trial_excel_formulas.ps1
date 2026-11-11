# Formula/capacity evidence from private template copies, independent Excel PID.
param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$excel=$null;$book=$null;$owned=$false
$result=[ordered]@{passed=$false;templates=@();errors=@()}
$existing=@(Get-Process EXCEL -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
Add-Type -TypeDefinition 'using System;using System.Runtime.InteropServices;public static class FdmFormulaWindow{[DllImport("user32.dll")]public static extern uint GetWindowThreadProcessId(IntPtr h,out uint p);}'
try {
    $excel=New-Object -ComObject Excel.Application
    [uint32]$processId=0;[void][FdmFormulaWindow]::GetWindowThreadProcessId([IntPtr]$excel.Hwnd,[ref]$processId)
    if(!$processId -or $existing -contains [int]$processId){throw 'Not an independent instance'}
    $owned=$true;$excel.Visible=$false;$excel.DisplayAlerts=$false;$excel.EnableEvents=$false;$excel.AutomationSecurity=2
    foreach($name in @('sheet.xlt','sheet11.xlt','sheet_dia.xlt')) {
        $path=Join-Path $Root $name;$hash=(Get-FileHash -LiteralPath $path).Hash
        $book=$excel.Workbooks.Add($path)
        $record=[ordered]@{name=$name;sha256=$hash;sheets=@();names=@();ranges=@();formulas=@();calculation_cells=@()}
        $dataSheetName=$(if($name -eq 'sheet_dia.xlt'){'直径实验原始数据'}else{'原始数据'})
        $column=$(if($name -eq 'sheet_dia.xlt'){'C'}else{'D'})
        $dataSheet=$book.Worksheets.Item($dataSheetName)
        for($row=8;$row -le 24;$row++) {
            $cell=$dataSheet.Range("$column$row")
            $cellFormula=$cell.GetType().InvokeMember('Formula',[Reflection.BindingFlags]::GetProperty,$null,$cell,@())
            $record.calculation_cells+=@{cell="$column$row";value=$cell.Value2;formula=$cellFormula}
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($cell)
        }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($dataSheet)
        foreach($named in $book.Names) {$record.names+=@{name=[string]$named.Name;refers_to=[string]$named.RefersTo}}
        foreach($sheet in $book.Worksheets) {
            $record.sheets+=@{name=[string]$sheet.Name;protected=[bool]$sheet.ProtectContents;rows=[int]$sheet.UsedRange.Rows.Count;columns=[int]$sheet.UsedRange.Columns.Count}
            $scan=$sheet.UsedRange
            $matrix=$scan.GetType().InvokeMember('Formula',[Reflection.BindingFlags]::GetProperty,$null,$scan,@())
            if($matrix -isnot [Array]){$matrix=@($matrix)}
            $rangeCounts=@{}
            foreach($formula in $matrix) {
                if($formula -isnot [string] -or !$formula.StartsWith('=')){continue}
                foreach($match in [regex]::Matches($formula,'\$?[A-Z]{1,3}\$?(?:25|9|13):\$?[A-Z]{1,3}\$?[0-9]+')) {
                    $range=$match.Value
                    if(!$rangeCounts.ContainsKey($range)){$rangeCounts[$range]=0}
                    $rangeCounts[$range]++
                }
                if($record.formulas.Count -lt 30 -and $formula -match '(?i)(COUNT|AVERAGE|STDEV|SUMPRODUCT).*25:') {
                    $record.formulas+=@{sheet=[string]$sheet.Name;formula=$formula}
                }
            }
            foreach($key in $rangeCounts.Keys){$record.ranges+=@{sheet=[string]$sheet.Name;range=$key;count=$rangeCounts[$key]}}
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($scan)
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($sheet)
        }
        $book.Close($false);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($book);$book=$null
        if((Get-FileHash -LiteralPath $path).Hash -ne $hash){throw 'Template changed'}
        $result.templates+=@($record)
    }
    $result.passed=$true
} catch {$result.errors+=@{message=$_.Exception.Message;stack=$_.ScriptStackTrace}} finally {
    if($owned){if($book){$book.Close($false)};$excel.Quit()}
    if($book){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($book)}
    if($excel){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel)}
    [GC]::Collect();[GC]::WaitForPendingFinalizers()
    [IO.File]::WriteAllText((Join-Path $Root 'template-formulas.json'),($result|ConvertTo-Json -Depth 20),[Text.UTF8Encoding]::new($false))
    @{passed=$result.passed;templates=$result.templates.Count;errors=$result.errors}|ConvertTo-Json -Depth 4
}
if(!$result.passed){exit 1}
