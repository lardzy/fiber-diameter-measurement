# Read-only COM verification of disposable reports produced by validate_cu_template_excel.py.
param([Parameter(Mandatory=$true)][string]$Directory)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($Directory)
$excel=New-Object -ComObject Excel.Application
$excel.DisplayAlerts=$false; $excel.AutomationSecurity=3
$out=@()
try {
foreach($name in @('cu-normal','cu-multiscale','user-custom')) {
$original=$excel.Workbooks.Open((Join-Path $root "$name.xlt"),0,$true,[Type]::Missing,[Type]::Missing,[Type]::Missing,$true,[Type]::Missing,[Type]::Missing,$true)
$formulas=$original.Worksheets.Item('原始数据').Range('D9:BE24').Formula
$originalNames=@($original.Worksheets | ForEach-Object {$_.Name})
foreach($count in @(0,1,1488,1500,2000)) {
$book=$excel.Workbooks.Open((Join-Path $root "$name-$count.xls"),0,$true)
try {
$data=$book.Worksheets.Item('原始数据');$f=$data.Range('D9:BE24').Formula
$changes=0
for($r=1;$r -le 16;$r++){for($c=1;$c -le 54;$c++){if([string]$f[$r,$c] -ne [string]$formulas[$r,$c]){$changes++}}}
$actual=$data.Range('AN25:AN2024').Value2
for($i=0;$i -lt $count;$i++) {
    $actualValue = $actual.GetValue([int]($i+1),[int]1)
    $expectedValue = 15 + ($i % 7) * 0.3
    $difference = [double]$actualValue - $expectedValue
    if ([math]::Abs($difference) -gt 1e-10) { throw "Raw record mismatch $name $count $i" }
}
if($count -lt 2000 -and $null -ne $actual[($count+1),1]){throw 'Unexpected leftover diameter'}
if([string]$data.Range('G1').Value2 -ne '00123'){throw 'Sample ID changed'}
if($changes -ne 0){throw "Input formulas changed $name $count"}
if(!$book.HasVBProject){throw 'VBA missing'}
$shapeChanges=0;$printChanges=0
foreach($sheet in $original.Worksheets){$peer=$book.Worksheets.Item($sheet.Name);if($peer.Shapes.Count -ne $sheet.Shapes.Count){$shapeChanges++};if($peer.PageSetup.PrintArea -ne $sheet.PageSetup.PrintArea){$printChanges++}}
if($shapeChanges -or $printChanges){throw 'Shapes or print areas changed'}
$out+=@{template=$name;count=$count;raw_records_exact=$true;formula_changes=$changes;vba=$book.HasVBProject;sheet_count=$book.Worksheets.Count;original_sheet_count=$original.Worksheets.Count;shape_changes=$shapeChanges;print_area_changes=$printChanges}
}finally{$book.Close($false)}
}
$original.Close($false)
}
$out | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'preservation-results.json') -Encoding UTF8
Write-Output ('verified='+$out.Count)
}finally{$excel.Quit()}
