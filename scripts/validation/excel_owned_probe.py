"""Identity-checked probe for disposable Windows Excel validation workbooks."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess


def view_differences(before, after):
    """Compare literal template presentation, independently of worker metadata."""
    differences = []
    if before.get("active_sheet") != after.get("active_sheet"):
        differences.append("active worksheet changed")
    left, right = before.get("sheets", []), after.get("sheets", [])
    if [item["name"] for item in left] != [item["name"] for item in right]:
        differences.append("worksheet names/order changed")
    right_by_name = {item["name"]: item for item in right}
    for item in left:
        other = right_by_name.get(item["name"])
        if other is None:
            continue
        for field in ("visible", "shapes", "sample"):
            if item.get(field) != other.get(field):
                differences.append(f"{item['name']}: {field} changed")
    return differences


PROBE_PS = r'''param([string]$RequestPath)
$ErrorActionPreference='Stop'
Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class FdmTemplateFlowWindow {
 public delegate bool Visit(IntPtr h, IntPtr p);
 [DllImport("user32.dll")] static extern bool EnumWindows(Visit v, IntPtr p);
 [DllImport("user32.dll")] static extern bool EnumChildWindows(IntPtr h, Visit v, IntPtr p);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint p);
 [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern int GetClassName(IntPtr h,System.Text.StringBuilder s,int n);
 [DllImport("oleacc.dll")] static extern int AccessibleObjectFromWindow(IntPtr h,uint id,ref Guid g,[MarshalAs(UnmanagedType.Interface)] out object obj);
 public static object Window(uint pid) { object found=null;
  Visit children=(h,p)=> {var s=new System.Text.StringBuilder(128);GetClassName(h,s,128);if(s.ToString()!="EXCEL7")return true;
   Guid id=new Guid("00020400-0000-0000-C000-000000000046");object o;
   if(AccessibleObjectFromWindow(h,0xfffffff0,ref id,out o)==0 && o!=null){found=o;return false;}return true;};
  EnumWindows((h,p)=>{uint actual;GetWindowThreadProcessId(h,out actual);if(actual==pid)EnumChildWindows(h,children,IntPtr.Zero);return found==null;},IntPtr.Zero);return found;
 }
}
'@
function Snapshot($book) {
 $items=@(); $active=[string]$book.ActiveSheet.Name
 foreach($sheet in $book.Worksheets) {
  $sample=@()
  if([string]$sheet.Name -in @('特种毛(报出)','麻棉(报出)','分组加权统计','织物纤维含量报告')) {
   foreach($address in @('A1','B2','D5','F8','G10','K20')) {
    $cell=$sheet.Range($address)
    $sample+=@{address=$address;formula=$cell.Formula;number_format=$cell.NumberFormat;
      font_name=$cell.Font.Name;font_size=$cell.Font.Size;font_color=$cell.Font.Color;
      bold=$cell.Font.Bold;fill=$cell.Interior.Color;merge=$cell.MergeArea.Address()}
   }
  }
  $items+=@{name=[string]$sheet.Name;visible=[int]$sheet.Visible;shapes=[int]$sheet.Shapes.Count;sample=$sample}
 }
 $result=@{name=[string]$book.Name;active_sheet=$active;sheets=$items;count=[int]$book.Worksheets.Count}
 foreach($sheet in $book.Worksheets) {
  if([string]$sheet.Name -eq '原始数据') {
   $result.first=$sheet.Range('D25').Value2;$result.second=$sheet.Range('D26').Value2
  }
 }
 return $result
}
function SetValidationName($book,[string]$name,[string]$value) {
 # BuiltinDocumentProperties.Item('Comments') is locale/COM-version dependent.
 # These unique, workbook-scoped constants belong to the probe and neither
 # overwrite report cells nor add pages/controls or disturb the active sheet.
 $definition='="'+$value.Replace('"','""')+'"'
 $created=$book.Names.Add($name,$definition)
 if($null -eq $created){throw ('Could not create validation name: '+$name)}
}
function ReadValidationName($book,[string]$name) {
 foreach($item in $book.Names) {
  if([string]$item.Name -eq $name) {
   $definition=[string]$item.RefersTo
   if($definition.StartsWith('="') -and $definition.EndsWith('"')) {
    return $definition.Substring(2,$definition.Length-3).Replace('""','"')
   }
   throw ('Unexpected validation name value: '+$name)
  }
 }
 return $null
}
$r=Get-Content -LiteralPath $RequestPath -Raw -Encoding UTF8|ConvertFrom-Json
function ProbeStage([string]$name) {
 @{utc=[DateTime]::UtcNow.ToString('o');stage=$name;operation=[string]$r.operation}|
  ConvertTo-Json -Compress|Add-Content -LiteralPath ($RequestPath+'.stages.jsonl') -Encoding UTF8
}
ProbeStage 'request'
$excel=$null;$book=$null;$owned=$false
if($r.operation -in @('baseline','variant','invalid')) {
 $before=@(Get-Process EXCEL -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
 ProbeStage 'creating independent Excel'
 $excel=New-Object -ComObject Excel.Application
 [uint32]$pidValue=0;[void][FdmTemplateFlowWindow]::GetWindowThreadProcessId([IntPtr]$excel.Hwnd,[ref]$pidValue)
 if(!$pidValue -or $before -contains [int]$pidValue){throw 'Baseline Excel is not independent'}
 $owned=$true
 @{id=$pidValue;created=(Get-Process -Id $pidValue).StartTime.ToUniversalTime().Ticks.ToString();
   validation_probe=$true;operation=[string]$r.operation;preexisting_excel_ids=$before}|
   ConvertTo-Json|Set-Content -LiteralPath ($RequestPath+'.owner.json') -Encoding UTF8
 try {
  $excel.DisplayAlerts=$false
  # Use the same Office policy as production, including the selected template's
  # own Workbook_Open/Activate events (the supplied template creates buttons).
  # FDM must not introduce an extra grouping macro on top of those events.
  $excel.AutomationSecurity=2
  ProbeStage 'opening source with normal template events'
  $book=$excel.Workbooks.Open([string]$r.source,[Type]::Missing,[Type]::Missing,[Type]::Missing,[Type]::Missing,
    [Type]::Missing,[Type]::Missing,[Type]::Missing,[Type]::Missing,$false)
  ProbeStage 'reading template presentation'
  $result=Snapshot $book
  if($r.operation -ne 'baseline') {
   # This only authors disposable test inputs. Production opening, the native
   # baseline, live updates and actual report export retain their normal events.
   ProbeStage 'preparing disposable fixture without save events'
   $excel.EnableEvents=$false;$excel.DisplayAlerts=$false
   if($r.operation -eq 'variant') {SetValidationName $book '_FDM_Template_Flow_Version' ([string]$r.marker)}
   if($r.operation -eq 'invalid') {$book.Worksheets.Item('原始数据').Name='验证缺少映射输入页'}
   $book.CheckCompatibility=$false
   ProbeStage 'saving disposable template fixture'
   $book.SaveAs([string]$r.destination,17)
   ProbeStage 'fixture saved'
  }
 } finally {
  ProbeStage 'closing independent probe workbook'
  if($book){$book.Close($false)}
  ProbeStage 'quitting independent probe Excel'
  if($owned){$excel.Quit()}
 }
} else {
 ProbeStage 'checking exact existing Excel identity'
 $process=Get-Process -Id ([int]$r.excel_pid)
 if($process.ProcessName -ne 'EXCEL' -or $process.StartTime.ToUniversalTime().Ticks.ToString() -ne [string]$r.excel_created){throw 'Owned Excel identity mismatch'}
 $window=[FdmTemplateFlowWindow]::Window([uint32]$r.excel_pid)
 if(!$window){throw 'Owned Excel window unavailable'}
 $excel=$window.Application
 if($r.operation -eq 'cleanup') {
  foreach($name in $r.names){$excel.Workbooks.Item([string]$name).Close($false)}
  $remaining=[int]$excel.Workbooks.Count
  if($remaining -eq 0){$excel.Quit()}
  $result=@{remaining=$remaining}
 } else {
  ProbeStage 'reading exact existing workbook'
  $book=$excel.Workbooks.Item([string]$r.name)
  if($r.operation -eq 'note') {SetValidationName $book '_FDM_Template_Flow_Note' 'manual report note retained'}
  $result=Snapshot $book
  $result.note=ReadValidationName $book '_FDM_Template_Flow_Note'
 }
}
ProbeStage 'complete'
$result|ConvertTo-Json -Depth 15|Set-Content -LiteralPath $r.result -Encoding UTF8
'''


PROBE_TIMEOUT_CLEANUP_PS = r'''param([string]$OwnerPath)
$ErrorActionPreference='Stop'
$owner=Get-Content -LiteralPath $OwnerPath -Raw -Encoding UTF8|ConvertFrom-Json
if(!$owner.validation_probe -or $owner.operation -notin @('baseline','variant','invalid') -or
   !$owner.id -or !$owner.created -or @($owner.preexisting_excel_ids) -contains [int]$owner.id) {
 throw 'Refusing cleanup: no independent validation-probe ownership proof'
}
$process=Get-Process -Id ([int]$owner.id) -ErrorAction SilentlyContinue
if(!$process) {@{cleaned=$true;already_exited=$true}|ConvertTo-Json -Compress;exit 0}
if($process.ProcessName -ne 'EXCEL' -or $process.StartTime.ToUniversalTime().Ticks.ToString() -ne [string]$owner.created) {
 throw 'Refusing cleanup: Excel PID or creation-time identity does not match'
}
Stop-Process -InputObject $process -Force -ErrorAction Stop
@{cleaned=$true;forced=$true;id=$owner.id;created=$owner.created}|ConvertTo-Json -Compress
'''


class ExcelProbe:
    def __init__(self, output):
        self.output, self.sequence = Path(output), 0
        self.output.mkdir(parents=True, exist_ok=True)
        self.script = self.output / "inspect-owned-workbook.ps1"
        self.script.write_text(PROBE_PS, encoding="utf-8-sig")

    def _cleanup_owned_timeout(self, request):
        owner = Path(str(request) + ".owner.json")
        if not owner.is_file():
            return {"cleaned": False, "reason": "No independently created probe owner record; no Excel process touched"}
        cleanup_script = self.output / "cleanup-owned-probe-timeout.ps1"
        cleanup_script.write_text(PROBE_TIMEOUT_CLEANUP_PS, encoding="utf-8-sig")
        try:
            completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass", "-File", str(cleanup_script), "-OwnerPath", str(owner)],
                capture_output=True, timeout=20)
            if completed.returncode:
                return {"cleaned": False, "error": completed.stderr.decode(errors="replace")}
            return json.loads(completed.stdout.decode("utf-8-sig"))
        except Exception as exc:
            return {"cleaned": False, "error": str(exc)}

    def call(self, operation, **values):
        self.sequence += 1
        request = self.output / f"request-{self.sequence}.json"
        result = self.output / f"result-{self.sequence}.json"
        request.write_text(json.dumps(dict(operation=operation, result=str(result), **values), ensure_ascii=False), encoding="utf-8")
        try:
            completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-STA",
                "-ExecutionPolicy", "Bypass", "-File", str(self.script), "-RequestPath", str(request)],
                capture_output=True, timeout=90)
        except subprocess.TimeoutExpired as exc:
            cleanup = self._cleanup_owned_timeout(request)
            timeout = {"operation": operation, "request": str(request), "seconds": exc.timeout,
                       "stages": str(request) + ".stages.jsonl", "cleanup": cleanup}
            (self.output / f"timeout-{self.sequence}.json").write_text(
                json.dumps(timeout, ensure_ascii=False, indent=2), encoding="utf-8")
            for stream in ("stdout", "stderr"):
                value = getattr(exc, stream, None)
                if value:
                    (self.output / f"timeout-{self.sequence}.{stream}.txt").write_bytes(
                        value if isinstance(value, bytes) else value.encode("utf-8"))
            raise RuntimeError(f"Excel probe timed out ({operation}); diagnostics: {self.output / f'timeout-{self.sequence}.json'}") from exc
        if completed.returncode:
            (self.output / f"error-{self.sequence}.txt").write_bytes(completed.stderr)
            raise RuntimeError(f"Excel probe failed ({operation}): {completed.stderr.decode(errors='replace')}")
        return json.loads(result.read_text(encoding="utf-8-sig"))
