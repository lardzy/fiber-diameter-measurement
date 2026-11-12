# Production presentation functions with fake native/COM boundaries; no Excel is touched.
param([Parameter(Mandatory=$true)][string]$WorkerPath,
      [Parameter(Mandatory=$true)][string]$ResultPath)
$ErrorActionPreference='Stop'
$tokens=$null; $parseErrors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($WorkerPath,[ref]$tokens,[ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
$names=@('Failure','Error-Code','Present-NativeWindow','Show-Book')
foreach ($definition in $ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)) {
    if ($names -contains $definition.Name) { Invoke-Expression $definition.Extent.Text }
}
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
public static class FdmLiveExcelWindow {
    public static List<string> Calls = new List<string>();
    public static uint Owner = 42;
    public static bool Valid = true, Minimized = true, Visible = false, AllowForeground = true;
    public static IntPtr Foreground = IntPtr.Zero;
    public static uint GetWindowThreadProcessId(IntPtr w, out uint p) { p=Owner; return 1; }
    public static bool IsWindow(IntPtr w) { return Valid; }
    public static bool IsIconic(IntPtr w) { return Minimized; }
    public static bool IsWindowVisible(IntPtr w) { return Visible; }
    public static bool ShowWindowAsync(IntPtr w, int c) {
        Calls.Add("show:"+c); Visible=true; if(c==9) Minimized=false; return true;
    }
    public static bool SetForegroundWindow(IntPtr w) {
        Calls.Add("foreground"); if(AllowForeground) Foreground=w; return AllowForeground;
    }
    public static IntPtr GetForegroundWindow() { return Foreground; }
    public static void FlashTaskbar(IntPtr w) { Calls.Add("taskbar"); }
}
'@
$checks=[Collections.Generic.List[object]]::new()
function Check([bool]$ok,[string]$name) {
    $checks.Add(@{name=$name;passed=$ok})
    if (!$ok) { throw $name }
}
$script:alive=$true; $script:readyCode=''; $script:activated=0
$script:excelIdentity=@{id=42;created='verified-birth'}
$script:excel=[pscustomobject]@{Visible=$false}
$script:request=@{session_id='trial';generation=3}
$window=[pscustomobject]@{Hwnd=567}
$windows=[pscustomobject]@{Window=$window}
$windows | Add-Member ScriptMethod Item {param($index) return $this.Window}
$book=[pscustomobject]@{Windows=$windows}
$book | Add-Member ScriptMethod Activate {$script:activated++}
$script:state=@{generation=3;window_handle=567;book=$book}
$script:sessions=@{trial=$script:state}
function Test-ExcelProcess { return $script:alive }
function Release-Com($value) {}
function Require-Object($value,$description) { if ($null -eq $value) { throw $description } }
function Get-State {
    Check ([FdmLiveExcelWindow]::Calls.Count -gt 0) 'native restoration precedes COM readiness'
    if ($script:readyCode) { Failure $script:readyCode 'synthetic COM boundary' }
    return $script:state
}
function Safe-Metadata($state) { return @{save_state='never_saved';saved=$false} }

$script:readyCode='excel_busy'
$result=Show-Book @{request_foreground=$true}
Check ($result.ok -and $result.activation_deferred -and $result.presentation_status -eq 'deferred') 'busy display is deferred success, not an input failure'
Check ($result.window_visible -and !$result.window_minimized -and $result.window_handle -eq 567) 'minimized busy workbook is restored with top-level window metadata'
Check ($script:activated -eq 0) 'busy workbook COM activation waits'
Check ([FdmLiveExcelWindow]::Calls[0] -eq 'show:9') 'minimized window uses restore command'

$script:readyCode=''; [FdmLiveExcelWindow]::Calls.Clear()
$result=Show-Book @{request_foreground=$false}
Check ($result.ok -and !$result.activation_deferred -and $script:activated -eq 1) 'deferred activation resumes without rebuilding or writing inputs'
Check (![FdmLiveExcelWindow]::Calls.Contains('foreground')) 'activation retry does not repeatedly request foreground'
Check ([FdmLiveExcelWindow]::Calls[0] -eq 'show:5') 'normal window is shown without restoring away its maximized state'

[FdmLiveExcelWindow]::AllowForeground=$false; [FdmLiveExcelWindow]::Foreground=[IntPtr]::Zero
[FdmLiveExcelWindow]::Calls.Clear()
$result=Show-Book @{request_foreground=$true}
Check ($result.ok -and $result.presentation_status -eq 'attention_required') 'OS foreground refusal is a notice and keeps data success'
Check ($result.attention_requested -and @([FdmLiveExcelWindow]::Calls | Where-Object {$_ -eq 'taskbar'}).Count -eq 1) 'refused explicit request flashes the verified taskbar once'

[FdmLiveExcelWindow]::Calls.Clear(); $script:readyCode='excel_busy'
$result=Show-Book @{request_foreground=$true}
Check ($result.activation_deferred -and $result.attention_requested -and
    @([FdmLiveExcelWindow]::Calls | Where-Object {$_ -eq 'taskbar'}).Count -eq 1) 'busy refused display gives one finite taskbar notification'
[FdmLiveExcelWindow]::Calls.Clear()
$result=Show-Book @{request_foreground=$false}
Check (!$result.attention_requested -and ![FdmLiveExcelWindow]::Calls.Contains('taskbar')) 'automatic busy retry never repeats taskbar flashing'
$script:readyCode=''

[FdmLiveExcelWindow]::Calls.Clear(); [FdmLiveExcelWindow]::Owner=99
$result=Present-NativeWindow $script:state $true
Check ([FdmLiveExcelWindow]::Calls.Count -eq 0 -and !$result.window_visible) 'recycled HWND owned by another process is never shown'
[FdmLiveExcelWindow]::Owner=42; $script:alive=$false
$result=Present-NativeWindow $script:state $true
Check ([FdmLiveExcelWindow]::Calls.Count -eq 0) 'process birth identity failure prevents native calls'
$script:alive=$true; $script:readyCode='workbook_closed'
$closed=$false
try { $null=Show-Book @{} } catch { $closed=(Error-Code $_) -eq 'workbook_closed' }
Check $closed 'confirmed workbook closure remains a distinct recoverable error'

$result=@{passed=$true;checks=@($checks.ToArray());worker_sha256=(Get-FileHash -LiteralPath $WorkerPath).Hash.ToLowerInvariant()}
[IO.File]::WriteAllText($ResultPath,($result | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
$result | ConvertTo-Json -Depth 8 -Compress
