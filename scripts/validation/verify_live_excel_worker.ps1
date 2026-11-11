# Windows desktop Excel acceptance probe. All templates and reports must be
# copies inside Root. The only instance inspected/closed is the worker's PID.
param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class FdmExcelProbe {
    public delegate bool Visit(IntPtr h, IntPtr p);
    [DllImport("user32.dll")] static extern bool EnumWindows(Visit v, IntPtr p);
    [DllImport("user32.dll")] static extern bool EnumChildWindows(IntPtr h, Visit v, IntPtr p);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint p);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] static extern int GetClassName(IntPtr h, System.Text.StringBuilder s, int n);
    [DllImport("oleacc.dll")] static extern int AccessibleObjectFromWindow(IntPtr h, uint id, ref Guid g, [MarshalAs(UnmanagedType.Interface)] out object obj);
    public static object Window(uint pid) {
        object found=null;
        Visit children=(h,p)=> { var cls=new System.Text.StringBuilder(128); GetClassName(h,cls,128);
            if(cls.ToString()!="EXCEL7")return true;
            Guid id=new Guid("00020400-0000-0000-C000-000000000046"); object obj;
            if(AccessibleObjectFromWindow(h,0xfffffff0,ref id,out obj)==0 && obj!=null) {found=obj; return false;} return true; };
        EnumWindows((h,p)=> { uint actual; GetWindowThreadProcessId(h,out actual);
            if(actual==pid) EnumChildWindows(h,children,IntPtr.Zero); return found==null; },IntPtr.Zero);
        return found;
    }
}
'@
$Root=[IO.Path]::GetFullPath($Root)
$evidence=[ordered]@{passed=$false;os=[Environment]::OSVersion.ToString();events=@();checks=@();templates=@();errors=@()}
$process=$null;$excel=$null;$window=$null
$before=@(Get-Process EXCEL -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
function Check($condition,[string]$description) {
    if(!$condition){throw $description}
    $script:evidence.checks+=@($description)
}
function Request([string]$operation,[string]$session='content',[int]$generation=1,[int]$revision=0,$payload=@{}) {
    $request=@{operation=$operation;session_id=$session;generation=$generation;input_revision=$revision;payload=$payload;request_id=[Guid]::NewGuid().ToString('N')}
    $bytes=[Text.UTF8Encoding]::new($false).GetBytes(($request|ConvertTo-Json -Depth 30 -Compress)+"`n")
    $script:process.StandardInput.BaseStream.Write($bytes,0,$bytes.Length)
    $script:process.StandardInput.BaseStream.Flush()
    do {
        $read=$script:process.StandardOutput.ReadLineAsync()
        if(!$read.Wait(60000)){throw "Worker timeout during $operation"}
        if($null -eq $read.Result){throw ('Worker EOF: '+$script:process.StandardError.ReadToEnd())}
        $reply=$read.Result|ConvertFrom-Json
        if($reply.request_id -ne $request.request_id){throw 'Mismatched request identity'}
        $script:evidence.events+=@($reply)
    } while($reply.event)
    return $reply
}
function Clone($object){return ($object|ConvertTo-Json -Depth 30|ConvertFrom-Json)}
try {
    $scriptPath=Join-Path $Root 'excel_live.ps1'
    $parseErrors=$null;$tokens=$null
    [void][Management.Automation.Language.Parser]::ParseFile($scriptPath,[ref]$tokens,[ref]$parseErrors)
    Check (!$parseErrors) 'PowerShell source parses'
    $start=[Diagnostics.ProcessStartInfo]::new('powershell.exe',('-NoLogo -NoProfile -NonInteractive -STA -ExecutionPolicy Bypass -File "'+$scriptPath+'"'))
    $start.UseShellExecute=$false;$start.CreateNoWindow=$true
    $start.RedirectStandardInput=$true;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    $start.StandardOutputEncoding=[Text.UTF8Encoding]::new($false);$start.StandardErrorEncoding=[Text.UTF8Encoding]::new($false)
    $process=[Diagnostics.Process]::Start($start)
    $input=Get-Content -LiteralPath (Join-Path $Root 'requests.json') -Raw -Encoding UTF8|ConvertFrom-Json
    foreach($item in $input.PSObject.Properties) {
        $payload=Clone $item.Value
        $payload.template=Join-Path $Root $payload.template
        $hash=(Get-FileHash -LiteralPath $payload.template).Hash
        $reply=Request 'open' $item.Name 1 1 $payload
        if(!$excel -and $reply.excel_pid) {
            Check ($before -notcontains [int]$reply.excel_pid) 'Worker Excel PID is independent'
            $window=[FdmExcelProbe]::Window([uint32]$reply.excel_pid)
            Check ($null -ne $window) 'Attached only to worker-owned Excel window'
            $excel=$window.Application
        }
        Check ($reply.ok) ("Opened and calculated $($item.Name): $($reply.error)")
        $book=$excel.Workbooks.Item([string]$reply.name)
        $originalClosed=$false
        $data=$book.Worksheets.Item([string]$payload.profile.data_sheet)
        $block=$payload.blocks[0];$cell=$block.cell
        Check ([double]$data.Range($cell).Value2 -eq [double]$block.values[0][0]) "$($item.Name) raw numeric input"
        Check ([bool]$data.ProtectContents) "$($item.Name) managed input protected"
        Check ([bool]$data.ProtectionMode) "$($item.Name) automation-only writes enabled"
        $controlName=$(if($payload.trial_kind -eq 'diameter'){$payload.profile.group_width_sheet}else{$payload.profile.standard_sheet})
        $controls=$book.Worksheets.Item([string]$controlName)
        Check (!$controls.Range('A1').Locked) "$($item.Name) unmanaged report/control cell remains editable"
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($controls)
        $formula=[string]$data.Range([string]$payload.checks[0].count_cell).Formula
        Check ([double]$data.Range([string]$payload.checks[0].count_cell).Value2 -eq 3) "$($item.Name) computed count verified"
        $script:evidence.templates+=@{name=$item.Name;excel_version=$reply.excel_version;sheets=@($book.Worksheets|ForEach-Object {$_.Name});count_formula=$formula;checks=$reply.comparisons;sha256=$hash;warnings=$reply.warnings}
        # View retention, append, delete and a save-as all occur on the same book.
        $data.Activate();$excel.ActiveWindow.ScrollRow=37;$excel.ActiveWindow.ScrollColumn=2
        $viewRow=$excel.ActiveWindow.ScrollRow;$viewColumn=$excel.ActiveWindow.ScrollColumn
        $update=Clone $payload;$update.cells=@();$update.clears=@();$update.blocks=@(@{cell=$data.Range($cell).Offset(3,0).Address($false,$false);values=@(@(22.75))})
        $update.checks[0].expected=4
        $reply=Request 'sync' $item.Name 1 2 $update
        Check ($reply.ok) "$($item.Name) append acknowledged"
        Check ([double]$data.Range($cell).Offset(3,0).Value2 -eq 22.75) "$($item.Name) appended cell verified"
        Check ($excel.ActiveSheet.Name -eq $data.Name -and $excel.ActiveWindow.ScrollRow -eq $viewRow -and $excel.ActiveWindow.ScrollColumn -eq $viewColumn) "$($item.Name) active sheet and scrolling retained ($viewRow/$viewColumn -> $($excel.ActiveWindow.ScrollRow)/$($excel.ActiveWindow.ScrollColumn))"
        $update.blocks=@();$update.clears=@(@{sheet=$data.Name;range=$data.Range($cell).Offset(2,0).Resize(2,1).Address($false,$false)})
        $update.checks[0].expected=2
        $reply=Request 'sync' $item.Name 1 3 $update
        Check ($reply.ok -and $null -eq $data.Range($cell).Offset(2,0).Value2 -and $null -eq $data.Range($cell).Offset(3,0).Value2) "$($item.Name) removed tail cleared"
        $saved=Join-Path $Root ($item.Name+'-saved.xls')
        $excel.DisplayAlerts=$false;$book.SaveAs($saved,56);$excel.DisplayAlerts=$true
        $reply=Request 'status' $item.Name 1 3
        Check ($reply.ok -and $reply.saved -and $reply.path -eq $saved) "$($item.Name) user SaveAs identity followed"
        Check ((Get-FileHash -LiteralPath $payload.template).Hash -eq $hash) "$($item.Name) original template hash unchanged"
        if($item.Name -eq 'custom') {
            $otherPayload=Clone $input.diameter;$otherPayload.template=Join-Path $Root 'sheet_dia.xlt'
            $other=Request 'open' 'parallel' 1 1 $otherPayload
            Check ($other.ok -and $excel.Workbooks.Count -eq 2) 'Independent trials have simultaneous workbooks'
            $otherBook=$excel.Workbooks.Item([string]$other.name)
            $otherBook.Activate();$viewSheet=$excel.ActiveSheet.Name
            $reply=Request 'sync' $item.Name 1 4 $payload
            Check ($reply.ok -and $excel.ActiveWorkbook.Name -eq $otherBook.Name -and $excel.ActiveSheet.Name -eq $viewSheet) 'Background trial sync preserves other workbook view'
            $failurePayload=Clone $payload;$failurePayload.profile.recalculate_macro='Missing_Fdm_Probe_Macro'
            $reply=Request 'sync' $item.Name 1 5 $failurePayload
            Check (!$reply.ok -and $reply.workbook_connected -and $reply.needs_full_sync) 'Macro failure has no success ACK and retains reconnectable workbook'
            $reply=Request 'sync' $item.Name 1 6 $payload
            Check ($reply.ok -and $reply.name -eq $book.Name) 'Full sync recovers failed calculation in same workbook'
            $reply=Request 'detach' 'parallel' 1 1
            Check ($reply.ok) 'Parallel workbook released independently'
            $otherBook.Close($false);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($otherBook)
        }
        $reply=Request 'sync' $item.Name 1 1 $update
        Check (!$reply.ok -and $reply.code -eq 'stale_revision') "$($item.Name) stale revision rejected"
        if($item.Name -eq 'content') {
            $book.Close($false)
            $reply=Request 'status' $item.Name 1 3
            Check (!$reply.ok -and $reply.code -eq 'workbook_closed') 'User-closed workbook reported'
            $reply=Request 'open' $item.Name 1 4 $payload
            Check (!$reply.ok -and $reply.code -eq 'workbook_closed') 'User closure does not cause automatic reopen'
            $reply=Request 'open' $item.Name 2 5 $payload
            Check ($reply.ok) 'Explicit reconnect with new generation rebuilds complete input'
            $book=$excel.Workbooks.Item([string]$reply.name);$data=$book.Worksheets.Item([string]$payload.profile.data_sheet)
            $generation=2
        } else {$generation=1}
        $count=$excel.Workbooks.Count
        $reply=Request 'detach' $item.Name $generation 5
        Check ($reply.ok -and $excel.Workbooks.Count -eq $count) "$($item.Name) detach leaves workbook open"
        Check (!$data.ProtectContents) "$($item.Name) detach removes FDM input protection"
        if($item.Name -eq 'diameter') {
            $book.Close($false);$originalClosed=$true
            $savedHash=(Get-FileHash -LiteralPath $saved).Hash
            $ordinary=Clone $payload;$ordinary.template=$saved
            $reply=Request 'open' 'normal-xls' 1 1 $ordinary
            Check ($reply.ok -and !$reply.has_file -and !$reply.path) 'Ordinary XLS template also produces new unsaved workbook'
            $newBook=$excel.Workbooks.Item([string]$reply.name)
            $reply=Request 'detach' 'normal-xls' 1 1
            Check ($reply.ok -and (Get-FileHash -LiteralPath $saved).Hash -eq $savedHash) 'Ordinary XLS template unchanged after detach'
            $newBook.Close($false);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($newBook)
        }
        # These are solely probe-created disposable reports, never user books.
        if(!$originalClosed){$book.Close($false)}
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($data)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($book)
    }
    $reply=Request 'shutdown' '' 0 0
    Check ($reply.ok) 'Worker shuts down without closing reports'
    $process.StandardInput.Close()
    Check ($process.WaitForExit(5000)) 'Worker process exits'
    Check (!$process.StandardError.ReadToEnd()) 'Worker stderr empty'
    $evidence.passed=$true
} catch {$evidence.errors+=@{message=$_.Exception.Message;stack=$_.ScriptStackTrace}} finally {
    if($process -and !$process.HasExited){$process.StandardInput.Close();if(!$process.WaitForExit(5000)){$process.Kill()}}
    if($excel) {
        $excel.DisplayAlerts=$false
        while($excel.Workbooks.Count){$excel.Workbooks.Item(1).Close($false)}
        $excel.Quit();[void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
    }
    if($window){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($window)}
    [GC]::Collect();[GC]::WaitForPendingFinalizers()
    [IO.File]::WriteAllText((Join-Path $Root 'worker-validation.json'),($evidence|ConvertTo-Json -Depth 30),[Text.UTF8Encoding]::new($false))
    $evidence|ConvertTo-Json -Depth 30
}
if(!$evidence.passed){exit 1}
