# Exercise the production batch engine with deterministic in-memory writes.
# No Excel process is created. Real COM/desktop coverage is in the two probes.
param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$source=Get-Content -LiteralPath (Join-Path $Root 'excel_live.ps1') -Raw -Encoding UTF8
$main=[regex]::Match($source,'(?m)^try \{')
if(!$main.Success){throw 'Cannot locate production worker command loop'}
. ([ScriptBlock]::Create($source.Substring(0,$main.Index)))
$result=[ordered]@{passed=$false;checks=@();errors=@()}
function Check($condition,$message){if(!$condition){throw $message};$script:result.checks+=@($message)}
$script:writes=[Collections.Generic.List[string]]::new()
$script:values=@{}
$script:failAt='D26'
$script:failOnce=$true
function Reply($body) { }
function Safe-Metadata($state){return @{save_state='unknown';saved=$null;calculation_state='unknown';metadata_error='simulated read failure'}}
function Invoke-BatchOperation($state,$op,$profile){
    $state.stage=$op.address
    if($op.address -eq $script:failAt -and $script:failOnce){$script:failOnce=$false;Failure 'excel_busy' 'simulated rejected COM call'}
    $script:writes.Add($op.address)
    $script:values[$op.address]=$op.value
}
try{
    $state=@{pending=$null;last_batch=$null;revision=0;initialized=$true;last_input_at=$null;warnings=@()}
    $payload=@{batch_id='partial';profile=@{data_sheet='raw'};cells=@(@{cell='D25';value=20},@{cell='D26';value=30});clears=@();blocks=@();warnings=@()}
    $script:request=@{request_id='rpc-1';input_revision=1;operation='sync'}
    try{Run-Batch $state $payload 'sync'|Out-Null;throw 'Expected busy'}catch{Check ((Error-Code $_) -eq 'excel_busy') 'second operation rejects with busy'}
    Check ($state.pending.index -eq 1 -and $state.revision -eq 0 -and $script:values.D25 -eq 20) 'partial commit retains cursor and does not advance input ACK'
    $new=@{batch_id='newer';profile=@{data_sheet='raw'};cells=@(@{cell='D25';value=10});clears=@();blocks=@();warnings=@()}
    try{Run-Batch $state $new 'sync'|Out-Null;throw 'Expected pending rejection'}catch{Check ((Error-Code $_) -eq 'batch_pending') 'newest state cannot bypass an unfinished fixed batch'}
    $script:request.request_id='rpc-2'
    $reply=Run-Batch $state $payload 'sync'
    Check ($reply.ok -and $reply.input_written -and $state.revision -eq 1 -and $null -eq $state.pending) 'same batch resumes and confirms input despite metadata failure'
    Check ($script:writes.Count -eq 2 -and $script:writes[0] -eq 'D25' -and $script:writes[1] -eq 'D26') 'already completed operations are not repeated'
    $script:request.request_id='rpc-3'
    $reply=Run-Batch $state $payload 'sync'
    Check ($reply.input_written -and $script:writes.Count -eq 2) 'lost final acknowledgement can be retried without any extra writes'
    $script:request.input_revision=2
    $reply=Run-Batch $state $new 'sync'
    Check ($reply.input_written -and $state.revision -eq 2 -and $script:values.D25 -eq 10) 'a change undone while busy is written back after the first batch completes'
    $script:request.input_revision=1
    try{Run-Batch $state $payload 'sync'|Out-Null;throw 'Expected stale rejection'}catch{Check ((Error-Code $_) -eq 'stale_revision') 'older input revision cannot replace a newer committed state'}
    $aux=@{batch_id='info';profile=@{data_sheet='raw'};report_cells=@(@{cell='E1';value='manual update'});report_settings=$null}
    $reply=Run-Batch $state $aux 'update_report'
    Check ($reply.ok -and !$reply.input_written -and $state.revision -eq 2) 'older explicit report snapshot does not change quantitative revision'
    $poison=@{batch_id='quant-only';profile=@{data_sheet='raw'};cells=@(@{cell='D25';value=40});clears=@();blocks=@();warnings=@();report_cells=@(@{cell='E1';value='must not overwrite'});report_settings=@{group_width=999}}
    $script:request.input_revision=3
    $reply=Run-Batch $state $poison 'sync'
    Check ($reply.input_written -and $script:values.E1 -eq 'manual update') 'ordinary synchronization ignores report information and settings even if supplied'
    $result.passed=$true
}catch{$result.errors+=@{message=$_.Exception.Message;stack=$_.ScriptStackTrace}}
[IO.File]::WriteAllText((Join-Path $Root 'worker-batch-protocol.json'),($result|ConvertTo-Json -Depth 20),[Text.UTF8Encoding]::new($false))
$result|ConvertTo-Json -Depth 20
if(!$result.passed){exit 1}
