# Run only in a disposable directory containing the current two worker scripts.
# Every Excel instance created or attached here is checked against its PID/birth.
param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$Root=[IO.Path]::GetFullPath($Root)
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class FdmMutableTemplateProbe {
 public delegate bool Visit(IntPtr h,IntPtr p);
 [DllImport("user32.dll")] static extern bool EnumWindows(Visit v,IntPtr p);
 [DllImport("user32.dll")] static extern bool EnumChildWindows(IntPtr h,Visit v,IntPtr p);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h,out uint p);
 [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern int GetClassName(IntPtr h,System.Text.StringBuilder s,int n);
 [DllImport("oleacc.dll")] static extern int AccessibleObjectFromWindow(IntPtr h,uint id,ref Guid g,[MarshalAs(UnmanagedType.Interface)] out object o);
 public static object Window(uint pid) {object found=null;
  Visit child=(h,p)=> {var s=new System.Text.StringBuilder(128);GetClassName(h,s,128);if(s.ToString()!="EXCEL7")return true;
   Guid g=new Guid("00020400-0000-0000-C000-000000000046");object o;
   if(AccessibleObjectFromWindow(h,0xfffffff0,ref g,out o)==0 && o!=null){found=o;return false;}return true;};
  EnumWindows((h,p)=> {uint actual;GetWindowThreadProcessId(h,out actual);if(actual==pid)EnumChildWindows(h,child,IntPtr.Zero);return found==null;},IntPtr.Zero);return found;
 }
}
'@
$evidence=[ordered]@{passed=$false;checks=@();replies=@();owners=@();boundary='Windows desktop Excel with disposable synthetic XLT source; no installer or physical camera'}
$existing=@(Get-Process EXCEL -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$owned=[Collections.Generic.List[object]]::new()
$process=$null
function Check([bool]$ok,[string]$name) {
    $script:evidence.checks+=@{name=$name;passed=$ok}
    if (!$ok) { throw $name }
}
function Own($application) {
    [uint32]$id=0
    [void][FdmMutableTemplateProbe]::GetWindowThreadProcessId([IntPtr]$application.Hwnd,[ref]$id)
    Check ($id -and $script:existing -notcontains [int]$id) 'Excel instance is independent of pre-existing user processes'
    $owner=@{id=[int]$id;created=(Get-Process -Id $id).StartTime.ToUniversalTime().Ticks.ToString();application=$application}
    $script:owned.Add($owner)
    $script:evidence.owners+=@{id=$owner.id;created=$owner.created}
    return $owner
}
function Attach($owner) {
    $p=Get-Process -Id ([int]$owner.id)
    Check ($p.StartTime.ToUniversalTime().Ticks.ToString() -eq [string]$owner.created -and $script:existing -notcontains [int]$owner.id) 'Attached Excel PID and creation time match owned workbook'
    $window=[FdmMutableTemplateProbe]::Window([uint32]$owner.id)
    Check ($null -ne $window) 'Owned Excel window is available'
    $owner.application=$window.Application
    $script:owned.Add($owner)
    $script:evidence.owners+=@{id=$owner.id;created=$owner.created}
    return $owner.application
}
function Clone($value) { return $value|ConvertTo-Json -Depth 30|ConvertFrom-Json }
function Request([string]$operation,[string]$session,[int]$generation,$payload) {
    $payload=Clone $payload
    $payload|Add-Member NoteProperty batch_id ([Guid]::NewGuid().ToString('N')) -Force
    $request=@{request_id=[Guid]::NewGuid().ToString('N');session_id=$session;generation=$generation;input_revision=1;operation=$operation;payload=$payload}
    $bytes=[Text.UTF8Encoding]::new($false).GetBytes(($request|ConvertTo-Json -Depth 30 -Compress)+"`n")
    $script:process.StandardInput.BaseStream.Write($bytes,0,$bytes.Length)
    $script:process.StandardInput.BaseStream.Flush()
    do {
        $read=$script:process.StandardOutput.ReadLineAsync()
        if (!$read.Wait(45000)) { throw "Live worker timed out during $operation" }
        if (!$read.Result) { throw ('Live worker ended: '+$script:process.StandardError.ReadToEnd()) }
        $reply=$read.Result|ConvertFrom-Json
        Check ($reply.request_id -eq $request.request_id) 'Live reply belongs to the requested operation'
        $script:evidence.replies+=@($reply)
    } while ($reply.event)
    return $reply
}
function OneOff($request,[string]$name) {
    $requestPath=Join-Path $Root ($name+'-request.json');$resultPath=Join-Path $Root ($name+'-result.json')
    [IO.File]::WriteAllText($requestPath,($request|ConvertTo-Json -Depth 30),[Text.UTF8Encoding]::new($true))
    & powershell.exe -NoLogo -NoProfile -NonInteractive -STA -ExecutionPolicy Bypass -File (Join-Path $Root 'excel_template.ps1') -RequestPath $requestPath -ResultPath $resultPath | Out-Null
    $result=Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8|ConvertFrom-Json
    $owner=$result.owner
    if (!$owner -and (Test-Path -LiteralPath ($resultPath+'.owner.json'))) {
        $owner=Get-Content -LiteralPath ($resultPath+'.owner.json') -Raw -Encoding UTF8|ConvertFrom-Json
    }
    if ($owner -and $owner.handed_off) { [void](Attach @{id=$owner.id;created=$owner.created}) }
    $script:evidence.replies+=@{operation=$name;result=$result}
    return $result
}
try {
    # Author a real XLT with an extra report page and required input sheets.
    $author=New-Object -ComObject Excel.Application
    $authorOwner=Own $author
    $author.DisplayAlerts=$false;$author.EnableEvents=$false
    $book=$author.Workbooks.Add()
    $book.Worksheets.Item(1).Name='原始数据'
    $sheet=$book.Worksheets.Add();$sheet.Name='标准选择'
    $sheet=$book.Worksheets.Add();$sheet.Name='比重';$sheet.Range('E3').Value2='自定义验证标准'
    $sheet=$book.Worksheets.Add();$sheet.Name='新增报表页';$sheet.Range('A1').Value2='Selected template report survives'
    $sheet.Activate()
    $source=Join-Path $Root 'selected.xlt'
    $book.CheckCompatibility=$false;$book.SaveAs($source,17);$book.Close($false)
    $sourceHash=(Get-FileHash -LiteralPath $source).Hash.ToLowerInvariant()
    $bad=Join-Path $Root 'missing-input.xlt'
    $book=$author.Workbooks.Add();$book.Worksheets.Item(1).Name='报表但无原始数据';$book.SaveAs($bad,17);$book.Close($false)
    $profile=@{data_sheet='原始数据';standard_sheet='标准选择';method_sheet='比重';method_range='E3:E16';recalculate_macro=''}
    $stale=@{path=$source;sha256=('0'*64);size=1}
    $payload=@{template=$source;template_identity=$stale;trial_kind='content';profile=$profile;
        cells=@();blocks=@(@{cell='D25';values=@(@(18.25))});clears=@();warnings=@();report_cells=@();report_settings=$null}
    $start=[Diagnostics.ProcessStartInfo]::new('powershell.exe',('-NoLogo -NoProfile -NonInteractive -STA -ExecutionPolicy Bypass -File "'+(Join-Path $Root 'excel_live.ps1')+'"'))
    $start.UseShellExecute=$false;$start.CreateNoWindow=$true
    $start.RedirectStandardInput=$true;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    $start.StandardOutputEncoding=[Text.UTF8Encoding]::new($false)
    $process=[Diagnostics.Process]::Start($start)
    $reply=Request 'open' 'content' 1 $payload
    Check ($reply.ok -and $reply.input_written) 'Live open accepts stale expected SHA and commits numeric input'
    $excel=Attach @{id=$reply.excel_pid;created=$reply.excel_created}
    $opened=$excel.Workbooks.Item([string]$reply.name)
    Check ($opened.Worksheets.Item('新增报表页').Range('A1').Value2 -eq 'Selected template report survives') 'Live workbook preserves the selected report page'
    Check ([double]$opened.Worksheets.Item('原始数据').Range('D25').Value2 -eq 18.25) 'Live input is actually visible in Excel'
    Check ([string]$opened.Path -eq '') 'Live opening creates a new workbook rather than editing source'
    # Change the source via an independent workbook while keeping the live book.
    $missing=[Type]::Missing
    $edited=$author.Workbooks.Open($source,0,$false,$missing,$missing,$missing,$true,$missing,$missing,$true)
    $edited.Worksheets.Item('新增报表页').Range('A1').Value2='Replacement at the same path'
    $edited.Save();$edited.Close($false)
    Check ((Get-FileHash -LiteralPath $source).Hash.ToLowerInvariant() -ne $sourceHash) 'Source template was changed at the same path'
    $payload.blocks=@(@{cell='D26';values=@(@(21.5))})
    $reply=Request 'sync' 'content' 1 $payload
    Check ($reply.ok -and [double]$opened.Worksheets.Item('原始数据').Range('D26').Value2 -eq 21.5) 'Changing source does not stop the already-open live workbook'
    Check ($opened.Worksheets.Item('新增报表页').Range('A1').Value2 -eq 'Selected template report survives') 'Existing workbook retains its own report edits and source snapshot'
    $payload.blocks=@(@{cell='D25';values=@(@(18.25))})
    $reply=Request 'open' 'content' 2 $payload
    Check $reply.ok 'Explicit recreation reads same-path replacement without stale SHA rejection'
    $recreated=$excel.Workbooks.Item([string]$reply.name)
    Check ($recreated.Worksheets.Item('新增报表页').Range('A1').Value2 -eq 'Replacement at the same path') 'Recreated workbook uses current selected template content'
    $badPayload=Clone $payload;$badPayload.template=$bad
    $reply=Request 'open' 'invalid' 1 $badPayload
    Check (!$reply.ok -and $reply.error_code -eq 'template_mapping' -and $reply.error -match '原始数据') 'Missing input worksheet remains a real live-opening failure'
    $badPayload.template=Join-Path $Root 'does-not-exist.xlt'
    $reply=Request 'open' 'missing' 1 $badPayload
    Check (!$reply.ok -and $reply.error_code -eq 'template_missing') 'Missing source still reports template_missing'
    $request=@{operation='inspect';template=$source;template_identity=$stale;profile=$profile}
    $result=OneOff $request 'inspect'
    Check ($result.ok -and $result.sheets -contains '新增报表页') 'One-off inspect accepts stale SHA and preserves report page'
    $request.operation='export';$request.destination=Join-Path $Root 'exported.xls';$request.show=$false
    $request.cells=@();$request.clears=@();$request.blocks=@(@{cell='D25';values=@(@(22.75))});$request.warnings=@();$request.checks=@()
    $result=OneOff $request 'export'
    Check ($result.ok -and (Test-Path -LiteralPath $request.destination)) 'One-off export accepts stale SHA and saves a report'
    $report=$author.Workbooks.Open($request.destination,0,$true)
    Check ([double]$report.Worksheets.Item('原始数据').Range('D25').Value2 -eq 22.75 -and $report.Worksheets.Item('新增报表页').Range('A1').Value2 -eq 'Replacement at the same path') 'Export contains the requested numeric data and current report page'
    $report.Close($false)
    $request.operation='edit'
    $result=OneOff $request 'edit'
    Check ($result.ok -and $result.handed_off) 'Explicit template editing accepts stale SHA and hands the source to the user'
    $null=Request 'shutdown' '' 0 @{}
    Check ($process.WaitForExit(10000)) 'Live worker shuts down after detaching owned workbooks'
    $evidence.passed=$true
} catch {
    $evidence.error=$_.Exception.Message;$evidence.stack=$_.ScriptStackTrace
} finally {
    if ($process -and !$process.HasExited) { try {$process.StandardInput.Close();[void]$process.WaitForExit(5000)} catch {} }
    foreach ($owner in $owned) {
        try {
            $p=Get-Process -Id ([int]$owner.id) -ErrorAction SilentlyContinue
            if ($p -and $p.StartTime.ToUniversalTime().Ticks.ToString() -eq $owner.created) {
                $owner.application.DisplayAlerts=$false
                $owner.application.EnableEvents=$false
                while ($owner.application.Workbooks.Count) { $owner.application.Workbooks.Item(1).Close($false) }
                $owner.application.Quit()
            }
        } catch { $evidence.cleanup_error=$_.Exception.Message }
    }
    $evidence.preexisting_excel_ids=$existing
    $evidence.worker_sha256=(Get-FileHash -LiteralPath (Join-Path $Root 'excel_live.ps1')).Hash.ToLowerInvariant()
    $evidence.oneoff_worker_sha256=(Get-FileHash -LiteralPath (Join-Path $Root 'excel_template.ps1')).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText((Join-Path $Root 'result.json'),($evidence|ConvertTo-Json -Depth 30),[Text.UTF8Encoding]::new($false))
}
$evidence|ConvertTo-Json -Depth 30 -Compress
if (!$evidence.passed) { exit 1 }
