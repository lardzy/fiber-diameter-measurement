# Execute the production PowerShell function bodies with local in-memory objects.
# No Excel, device, Office settings, or user configuration is accessed.
param([Parameter(Mandatory=$true)][string]$WorkerPath,
      [Parameter(Mandatory=$true)][string]$ResultPath)
$ErrorActionPreference='Stop'
$tokens=$null; $parseErrors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($WorkerPath,[ref]$tokens,[ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
foreach ($definition in $ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)) {
    Invoke-Expression $definition.Extent.Text
}
$checks=[Collections.Generic.List[object]]::new()
function Check([bool]$ok,[string]$name) {
    $checks.Add(@{name=$name;passed=$ok})
    if (!$ok) { throw $name }
}
$script:request=@{request_id='probe-1';input_revision=3;session_id='trial';generation=2;operation='open'}
$payload=@{batch_id='b1';profile=@{data_sheet='原始数据'};report_cells=@(@{cell='A1';value='Sample'});
    report_settings=@{method_label='must not run';group_width=5};clears=@('D25:D26');cells=@();blocks=@();warnings=@()}
$state=@{pending=$null;last_batch=$null}
$batch=New-Batch $state $payload 'open'
Check (@($batch.ops | Where-Object {$_.kind -in @('settings','macro')}).Count -eq 0) 'open ignores even explicitly supplied settings and macro initialization'
Check (@($batch.ops | Where-Object {$_.family -eq 'report'}).Count -eq 1) 'open retains requested initial report information'
$state=@{pending=$null;last_batch=$null}
$batch=New-Batch $state $payload 'update_report'
Check (@($batch.ops | Where-Object {$_.kind -eq 'settings'}).Count -eq 1 -and @($batch.ops | Where-Object {$_.kind -eq 'macro'}).Count -eq 1) 'explicit settings update remains available'

$temp=Join-Path ([IO.Path]::GetTempPath()) ('fdm-template-identity-'+[Guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($temp)
try {
    $file=Join-Path $temp '模板.xlt'; [IO.File]::WriteAllText($file,'template fixture')
    $identity=Template-Identity $file
    Check ($identity.sha256 -eq (Get-FileHash -LiteralPath $file).Hash.ToLowerInvariant()) 'identity hashes the actual selected file'
    [IO.File]::WriteAllText($file,'changed fixture')
    $changed=Template-Identity $file
    Check ($changed.sha256 -ne $identity.sha256) 'same-path changed content is recorded without rejecting the source'
    function Get-FileHash { throw 'synthetic diagnostic hash failure' }
    $unavailable=Template-Identity $file
    Check ($unavailable.path -eq $file -and $unavailable.hash_status -eq 'unavailable') 'unavailable diagnostic hash does not reject a readable source'
    Remove-Item Function:\Get-FileHash
    $missingRejected=$false
    try { [void](Template-Identity (Join-Path $temp 'missing.xlt')) } catch { $missingRejected=(Error-Code $_) -eq 'template_missing' }
    Check $missingRejected 'missing source still fails before Excel opens'
} finally { Remove-Item -LiteralPath $temp -Recurse -Force }

# Exercise the real view helpers using worksheet-like objects, including a
# custom report, a hidden original sheet and a remembered scroll position.
$script:fakeSheets=@(
    [pscustomobject]@{Name='标准选择';Visible=-1},
    [pscustomobject]@{Name='特种毛(报出)';Visible=-1},
    [pscustomobject]@{Name='比重';Visible=0})
foreach ($sheet in $script:fakeSheets) { $sheet | Add-Member ScriptMethod Activate {$script:activatedSheet=$this.Name} }
$sheetCollection=[pscustomobject]@{Count=$script:fakeSheets.Count}
$sheetCollection | Add-Member ScriptMethod Item {param($selector)
    if ($selector -is [int]) { return $script:fakeSheets[$selector-1] }
    return ($script:fakeSheets | Where-Object {$_.Name -eq $selector})
}
$script:fakeWindow=[pscustomobject]@{ScrollRow=12;ScrollColumn=4}
$windowCollection=[pscustomobject]@{Count=1}
$windowCollection | Add-Member ScriptMethod Item {param($index) return $script:fakeWindow}
$fakeBook=[pscustomobject]@{Name='新报表';Sheets=$sheetCollection;ActiveSheet=$script:fakeSheets[1];Windows=$windowCollection}
$fakeBook | Add-Member ScriptMethod Activate {}
$state=@{book=$fakeBook;sheets=@{}}
$script:excel=[pscustomobject]@{}
$script:excel | Add-Member ScriptMethod Run {param($macro)
    $script:fakeSheets[1].Visible=0; $script:fakeSheets[2].Visible=-1
    $script:fakeWindow.ScrollRow=1; $script:fakeWindow.ScrollColumn=1
}
$script:excel | Add-Member ScriptMethod Calculate {}
Recalculate-Report $state @{recalculate_macro='分组'}
Check ($script:fakeSheets[1].Visible -eq -1 -and $script:fakeSheets[2].Visible -eq 0) 'explicit grouping restores added and hidden sheet visibility'
Check ($script:activatedSheet -eq '特种毛(报出)' -and $script:fakeWindow.ScrollRow -eq 12 -and $script:fakeWindow.ScrollColumn -eq 4) 'explicit grouping restores active report and scroll'

# Extract the production request switch's clauses rather than duplicating its
# commit logic. COM calls alone are replaced; session/candidate bookkeeping runs.
$requestSwitch=$ast.FindAll({param($n)
    $n -is [Management.Automation.Language.SwitchStatementAst] -and
    @($n.Clauses | Where-Object {$_.Item1.Value -eq 'discard_candidate'}).Count -gt 0
},$true)[0]
foreach ($clause in $requestSwitch.Clauses) {
    $name=$clause.Item1.Value
    Invoke-Expression ('function Invoke-Request-'+$name+' '+$clause.Item2.Extent.Text)
}
function Start-Excel {}
function Ensure-Ready {}
function Template-Identity([string]$path) { return @{path=$path;sha256='fixture-hash'} }
$script:created=0; $script:failBatch=''
$failureBody=$ast.FindAll({param($n)
    $n -is [Management.Automation.Language.CatchClauseAst] -and $n.Extent.Text.Contains('$failure.candidate_pending=')
},$true)[0].Body.Extent.Text
$failureHandler=[ScriptBlock]::Create($failureBody.Substring(1,$failureBody.Length-2))
function Open-TemplateBook([string]$path) {
    $script:created++
    return [pscustomobject]@{Name=('candidate'+$script:created)}
}
function Capture-BookView($state) { return @{sheets=@(@{name='特种毛(报出)';visible=-1});active_sheet='特种毛(报出)'} }
function Test-BookOpen($state) { return $true }
function Run-Batch($state,$payload,[string]$operation,[bool]$suppressProgress=$false) {
    Check ($script:sessions['trial'].generation -eq 1) 'old association remains current until candidate inputs finish'
    Check $suppressProgress 'candidate cannot emit an early input acknowledgement before association commit'
    if ($script:failBatch) {
        $state.pending=@{id=$payload.batch_id;operation='open';index=1;input_start=0;input_written=$false}
        if ($script:failBatch -eq 'busy') { Failure 'excel_busy' 'synthetic busy during partial write' }
        throw 'synthetic partial write failure'
    }
    $state.revision=3; $state.initialized=$true
    return @{ok=$true;input_written=$true;template_identity=$state.template_identity;initial_view=$state.initial_view}
}
function Reply($body) {$script:lastReply=$body}
$script:excel=[pscustomobject]@{Version='mock'}
$script:sessions=@{trial=@{generation=1;revision=2;book=[pscustomobject]@{Name='old'};sheets=@{}}}
$script:candidates=@{}; $script:closed=@{}
$key='trial'; $payload.template='selected.xlt'
$script:failBatch='fatal'
try { Invoke-Request-open } catch {
    $state=$script:candidates['trial|2'];$candidateKey='trial|2'
    & $failureHandler
}
Check (!$script:lastReply.ok -and $script:lastReply.workbook_connected -and !$script:lastReply.needs_full_sync) 'failed candidate reports old association intact without invalidating input'
Check ($script:sessions['trial'].generation -eq 1 -and $script:candidates.Count -eq 0) 'fatal candidate releases pending association and preserves old workbook'
$script:failBatch='busy'
try { Invoke-Request-open } catch {
    $state=$script:candidates['trial|2'];$candidateKey='trial|2'
    & $failureHandler
}
Check ($script:lastReply.candidate_pending -and $script:lastReply.candidate_generation -eq 2 -and $script:lastReply.active_generation -eq 1) 'busy candidate exposes pending and current generations separately'
Check ($script:sessions['trial'].generation -eq 1 -and $script:candidates.Count -eq 1) 'busy candidate cannot replace old workbook'
$script:failBatch=''
Invoke-Request-open
Check ($script:created -eq 2) 'pending candidate resumes without creating a duplicate workbook'
Check ($script:sessions['trial'].generation -eq 2 -and $script:candidates.Count -eq 0) 'completed candidate commits atomically'
Check ($script:lastReply.binding_committed -and $script:lastReply.template_replaced) 'commit acknowledgement identifies the actual binding change'

$script:candidates['trial|3']=@{generation=3;book=[pscustomobject]@{Name='pending'};sheets=@{}}
$script:candidates['other|1']=@{generation=1;book=[pscustomobject]@{Name='other'};sheets=@{}}
$script:request.generation=2
Invoke-Request-detach
Check (!$script:sessions.ContainsKey('trial') -and !$script:candidates.ContainsKey('trial|3') -and $script:candidates.ContainsKey('other|1')) 'stop releases current trial candidate without touching another trial'
@{passed=$true;checks=@($checks.ToArray());boundary='production PowerShell AST/function execution with in-memory COM substitutes; no Excel'} |
    ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ResultPath -Encoding UTF8
