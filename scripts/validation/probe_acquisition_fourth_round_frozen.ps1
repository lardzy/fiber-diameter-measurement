# Windows PowerShell 5.1. Reuse the audited, PID-scoped UIA tools of the first
# redesign probe, but execute the fourth-round navigation contract below.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [ValidateSet('dark','light','navy','warm')][string]$Theme='light',
    [int]$ClientWidth=1600,
    [int]$ClientHeight=900,
    [switch]$Populated,
    [switch]$ExportReport
)
$ErrorActionPreference='Stop'
$helperPath=Join-Path $PSScriptRoot 'probe_workflow_redesign_frozen.ps1'
$helper=[IO.File]::ReadAllText($helperPath)
$marker="try {`n    [void]`$p.Start()"
$normalized=$helper.Replace("`r`n","`n")
$offset=$normalized.IndexOf($marker,[StringComparison]::Ordinal)
if($offset -lt 0 -or $normalized.LastIndexOf($marker,[StringComparison]::Ordinal) -ne $offset) {
    throw 'UIA helper boundary changed; review the shared probe before proceeding.'
}
# The prefix defines only parameters, isolated process/profile preparation and
# safety helpers. It does not execute the legacy navigation or launch a process.
$prefix=$normalized.Substring(0,$offset).Replace("(Join-Path `$PSScriptRoot '..\..')","'C:\tmp\fdm-cu-live-fourth-round-20261019'")
. ([scriptblock]::Create($prefix)) -Executable $Executable -OutputRoot $OutputRoot -Theme $Theme -ClientWidth $ClientWidth -ClientHeight $ClientHeight -ExerciseBackgroundSaving:$false
if($validationRoot -ne 'C:\tmp\fdm-cu-live-fourth-round-20261019') {throw 'Use the fourth-round isolated checkout'}
$result['schema']=4
$result['populated']=[bool]$Populated
$result['capture_note']='VM host-camera sharing is temporarily disabled and restored after scoped installation validation. No Microview hardware acceptance.'
$result['steps']=@()
$result['fixture_sha256']=$null
$result['excel_export_tested']=$false
if($ExportReport -and -not $Populated){throw 'Export requires the populated fixture'}
$main=$null
$mainHandle=0
if($Populated) {
    # Evidence asset hashes plus atomic suffixes exceed MAX_PATH under the
    # nested run/case/timestamp log tree. Keep only the fixture in a short,
    # uniquely owned path; preserve the full run directory for audit output.
    $fixtureRoot=Join-Path $validationRoot ('build\fourth-round\fixtures\'+[guid]::NewGuid().ToString('N').Substring(0,12))
    [void](New-Item -ItemType Directory -Path $fixtureRoot)
    $probeProject=Join-Path $fixtureRoot 'review.fdmproj'
    $result['fixture_path']=$probeProject
    $python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
    $env:PYTHONPATH=Join-Path $validationRoot 'src'
    $env:FDM_FOURTH_FIXTURE=$probeProject
    $env:FDM_FOURTH_EXPORT=[string][int][bool]$ExportReport
    $fixture=@'
import os,json
from pathlib import Path
from fdm.models import ProjectState
from fdm.project_io import ProjectIO
from fdm.live_experiment import ExperimentSession, MaterialSnapshot, default_materials
from fdm.live_experiment.excel_template import ExportSnapshot,template_request
from fdm.live_experiment.evidence import EvidenceStore
from fdm.live_experiment.frames import FrameHandle
import numpy as np
p=ProjectState.empty()
s=ExperimentSession(sample_name="第四轮视觉验证",sample_number="0000123",workspace_options={"mode":"content"},evidence_options={"version":1,"mode":"crop_jpeg"})
materials=default_materials() if os.environ["FDM_FOURTH_EXPORT"] == "1" else [MaterialSnapshot(f"fiber-{n}",f"验证纤维 {n}",1.3) for n in range(1,9)]
for name in ("验证甲","验证乙","验证丙"):
    op=s.add_operator(name)
    for direction in ("warp","weft"):
        s.set_materials(materials,operator_id=op,direction=direction)
        for i in range(24):
            s.add_measurement(18+i/10,materials[i%len(materials)].id,operator_id=op,direction=direction,field_id=f"{name}-{direction}")
        s.adjust_count(materials[0].id,3,operator_id=op,direction=direction,field_id=f"{name}-{direction}")
if os.environ["FDM_FOURTH_EXPORT"] != "1":
    # Owned synthetic pixels validate persisted-crop review, never algorithm accuracy.
    pixels=np.full((576,768,3),205,dtype=np.uint8)
    pixels[:,380:400]=70
    frame=FrameHandle.snapshot(pixels,source_id="installed-review-fixture",live=False)
    store=EvidenceStore(Path(os.environ["FDM_FOURTH_FIXTURE"]).parent/"evidence-source")
    store.reserve(frame,options=s.effective_evidence_options,points=[(380,288),(400,288)])
    evidence,=store.close()
    s.add_measurement(20,materials[0].id,operator_id=op,direction="weft",field_id="fixture-crop",
        endpoints=((380,288),(400,288)),evidence=evidence,
        provenance={"measurement_number":145,"original_diameter_um":20})
s.establish_history_baseline()
p.live_experiments.append(s)
ProjectIO.save(p,os.environ["FDM_FOURTH_FIXTURE"])
Path(os.environ["FDM_FOURTH_FIXTURE"]).with_suffix(".expected.json").write_text(json.dumps(template_request(ExportSnapshot.capture(s),"","",show=False),ensure_ascii=False),encoding="utf-8")
'@
    $fixturePath=Join-Path $runDirectory 'fixture.py'
    [IO.File]::WriteAllText($fixturePath,$fixture,$utf8)
    & $python $fixturePath
    if($LASTEXITCODE -ne 0) {throw 'Fixture creation failed'}
    $result.fixture_sha256=(Get-FileHash -LiteralPath $probeProject).Hash.ToLowerInvariant()
    $p.StartInfo.Arguments='"'+$probeProject+'"'
}
function Named($Name,$Type=$null) {
    return Visible-Unique @(Find-OwnNamed $Name $Type) $Name
}
function Click-Named($Name,$Type=$null) {
    [void](Invoke-Exact (Named $Name $Type))
    Start-Sleep -Milliseconds 150
    $result.steps+=@{action='invoke';name=$Name}
}
function Select-Trial($Name) {
    $tab=Visible-Unique @(Find-Within $main $Name ([System.Windows.Automation.ControlType]::TabItem)) $Name
    $tab.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
    Start-Sleep -Milliseconds 180
    $result.steps+=@{action='select';name=$Name}
}
function Confirm-ExcelEntry {
    $entry=Named '导出 Excel 报告' ([System.Windows.Automation.ControlType]::Button)
    $bounds=$entry.Current.BoundingRectangle
    $mainBounds=$main.Current.BoundingRectangle
    if($bounds.Top -lt $mainBounds.Top -or $bounds.Bottom -gt $mainBounds.Bottom) {throw 'Excel entry is clipped'}
    $result['excel_entry']=Get-ElementRecord $entry
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class FourthRoundDialogControls {
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hwnd,out uint pid);
    [DllImport("user32.dll")] static extern bool IsChild(IntPtr parent,IntPtr child);
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern int GetClassName(IntPtr hwnd,StringBuilder name,int count);
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern IntPtr SendMessage(IntPtr hwnd,uint msg,IntPtr wParam,string text);
    [DllImport("user32.dll")] static extern bool PostMessage(IntPtr hwnd,uint msg,IntPtr wParam,IntPtr lParam);
    static void Check(uint expected,long dialog,long handle,string wanted) {
        uint actual;GetWindowThreadProcessId(new IntPtr(handle),out actual);
        var name=new StringBuilder(128);GetClassName(new IntPtr(handle),name,name.Capacity);
        if(actual!=expected || !IsChild(new IntPtr(dialog),new IntPtr(handle)) || name.ToString()!=wanted)
            throw new InvalidOperationException("Refusing foreign or unexpected dialog control");
    }
    public static void SetFilename(uint pid,long dialog,long edit,string text) {
        Check(pid,dialog,edit,"Edit");SendMessage(new IntPtr(edit),0x000C,IntPtr.Zero,text);
    }
    public static void Save(uint pid,long dialog,long button) {
        Check(pid,dialog,button,"Button");
        if(!PostMessage(new IntPtr(button),0x00F5,IntPtr.Zero,IntPtr.Zero)) throw new InvalidOperationException("Save click failed");
    }
}
'@
function Exercise-ExcelExport {
    if(@(Get-Process EXCEL -ErrorAction SilentlyContinue).Count){throw 'Excel must be closed before this isolated export case'}
    $destination=Join-Path $runDirectory 'fixed-entry-report.xls'
    Click-Named '导出 Excel 报告' ([System.Windows.Automation.ControlType]::Button)
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    do {
        Start-Sleep -Milliseconds 100
        $dialogs=@(Get-OwnedRoots|Where-Object {$_.window.Visible -and $_.window.Title -eq '生成模板报告'})
    } while($dialogs.Count -ne 1 -and [DateTime]::UtcNow -lt $deadline)
    if($dialogs.Count -ne 1){throw 'Direct Excel entry did not open the save dialog'}
    $dialog=$dialogs[0].element
    Save-Snapshot '09-excel-save-dialog'
    $edits=@($dialog.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition)|Where-Object {
        $_.Current.ProcessId -eq $p.Id -and $_.Current.ClassName -eq 'Edit' -and $_.Current.AutomationId -eq '1001'
    })
    if($edits.Count -ne 1){throw 'Expected one owned filename edit'}
    [FourthRoundDialogControls]::SetFilename([uint32]$p.Id,$dialogs[0].window.Handle,$edits[0].Current.NativeWindowHandle,$destination)
    $save=@($dialog.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition)|Where-Object {
        $_.Current.ProcessId -eq $p.Id -and $_.Current.ClassName -eq 'Button' -and $_.Current.AutomationId -eq '1' -and $_.Current.Name -match '^(保存|Save)'
    })
    if($save.Count -ne 1){throw 'Expected one owned Save button'}
    [FourthRoundDialogControls]::Save([uint32]$p.Id,$dialogs[0].window.Handle,$save[0].Current.NativeWindowHandle)
    $deadline=[DateTime]::UtcNow.AddSeconds(180)
    do {Start-Sleep -Milliseconds 250} while(-not (Test-Path -LiteralPath $destination) -and [DateTime]::UtcNow -lt $deadline)
    if(-not (Test-Path -LiteralPath $destination)){throw 'Direct Excel report did not appear before deadline'}
    Start-Sleep -Milliseconds 1200
    $template=Join-Path $appConfig 'templates\sheet.xlt'
    $expected=Join-Path $runDirectory 'expected.json'
    Copy-Item -LiteralPath ([IO.Path]::ChangeExtension($probeProject,'expected.json')) -Destination $expected
    $audit=Join-Path $runDirectory 'excel-verification.json'
    & (Join-Path $validationRoot 'scripts\validation\verify_acquisition_fourth_round_excel.ps1') -Report $destination -Template $template -Expected $expected -Output $audit
    $verified=[IO.File]::ReadAllText($audit)|ConvertFrom-Json
    if(-not $verified.passed){throw 'Excel workbook audit failed'}
    $result['excel_export_tested']=$true
    $result['excel_verification']=$verified
    Save-Snapshot '10-excel-success'
}
function Confirm-Navigation {
    [void](Named '采集与试验' ([System.Windows.Automation.ControlType]::CheckBox))
    foreach($name in @('实时采集','含量试验','直径试验')) {
        [void](Visible-Unique @(Find-Within $main $name ([System.Windows.Automation.ControlType]::TabItem)) $name)
    }
}
function Confirm-FourthEntries {
    foreach($name in @('全图分析或刷新候选','候选显示与不透明度','查看测量证据及存档设置')) {
        $entry=Named $name
        if($entry.Current.IsOffscreen){throw "Fourth-round entry is offscreen: $name"}
        $result.steps+=@{action='fourth-entry-visible';name=$name;bounds=(Get-ElementRecord $entry)}
    }
    $auto=Named '稳定后自动分析' ([System.Windows.Automation.ControlType]::CheckBox)
    $toggle=$auto.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
    if($toggle.Current.ToggleState -ne [System.Windows.Automation.ToggleState]::Off){throw 'Automatic analysis must default off'}
    $toggle.Toggle();Start-Sleep -Milliseconds 120
    if($toggle.Current.ToggleState -ne [System.Windows.Automation.ToggleState]::On){throw 'Automatic control did not enable'}
    $toggle.Toggle();Start-Sleep -Milliseconds 120
    $result['automatic_toggle_verified']=$true
}
function Exercise-EvidenceReview {
    Click-Named '查看测量证据及存档设置'
    $deadline=[DateTime]::UtcNow.AddSeconds(10)
    do {Start-Sleep -Milliseconds 100;$pins=@(Find-OwnNamed '固定为小窗' ([System.Windows.Automation.ControlType]::CheckBox)|Where-Object {-not $_.Current.IsOffscreen})}
    while($pins.Count -ne 1 -and [DateTime]::UtcNow -lt $deadline)
    if($pins.Count -ne 1){throw 'Evidence review did not appear'}
    Save-Snapshot '04a-evidence-review'
    $pins[0].GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern).Toggle()
    Start-Sleep -Milliseconds 200
    [void](Named '显示测量线' ([System.Windows.Automation.ControlType]::CheckBox))
    Save-Snapshot '04b-evidence-pinned'
    Click-Named '收起' ([System.Windows.Automation.ControlType]::Button)
    $result['cropped_evidence_review_verified']=$true
}
try {
    [void]$p.Start()
    $result.pid=$p.Id
    $deadline=[DateTime]::UtcNow.AddSeconds(35)
    do {
        Start-Sleep -Milliseconds 200
        $p.Refresh()
        $windows=@([CuProbeNative]::Windows([uint32]$p.Id) | Where-Object {$_.Visible -and $_.Title -eq '显微测量工作台'})
    } while(-not $p.HasExited -and $windows.Count -ne 1 -and [DateTime]::UtcNow -lt $deadline)
    if($p.HasExited -or $windows.Count -ne 1) {throw 'Expected one main window'}
    $mainHandle=$windows[0].Handle
    $main=[System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$mainHandle)
    $result.main_window=$true
    [void][CuProbeNative]::ResizeClient($mainHandle,$ClientWidth,$ClientHeight)
    Start-Sleep -Milliseconds 250
    $result['os_dpi']=[CuProbeNative]::GetDpiForWindow([IntPtr]$mainHandle)
    Save-Snapshot '01-main'
    Click-Named '采集与试验' ([System.Windows.Automation.ControlType]::CheckBox)
    Save-Snapshot '02-realtime'
    Select-Trial '含量试验'
    Confirm-Navigation
    Confirm-ExcelEntry
    Confirm-FourthEntries
    Save-Snapshot '03-content'
    if(-not $Populated) {
        [void](Named '选择纤维类别' ([System.Windows.Automation.ControlType]::Button))
        $result['empty_category_entry_verified']=$true
        Click-Named '返回采集' ([System.Windows.Automation.ControlType]::Button)
        Select-Trial '含量试验'
        [void](Named '选择纤维类别' ([System.Windows.Automation.ControlType]::Button))
        $result['empty_return_verified']=$true
    } elseif(-not $ExportReport) {
        Exercise-EvidenceReview
        Click-Named '记录' ([System.Windows.Automation.ControlType]::CheckBox)
        $operator=Named '记录操作者筛选' ([System.Windows.Automation.ControlType]::ComboBox)
        $direction=Named '记录方向筛选' ([System.Windows.Automation.ControlType]::ComboBox)
        $result['record_operator_filter']=Get-ElementRecord $operator
        $result['record_direction_filter']=Get-ElementRecord $direction
        $tables=@(Find-OwnNamed '' ([System.Windows.Automation.ControlType]::Table) | Where-Object {-not $_.Current.IsOffscreen})
        $result['visible_record_tables']=@($tables | ForEach-Object {Get-ElementRecord $_})
        foreach($filter in @($operator,$direction)) {
            $selection=$null
            if($filter.TryGetCurrentPattern([System.Windows.Automation.SelectionPattern]::Pattern,[ref]$selection)) {
                $selected=@($selection.GetCurrentSelection() | ForEach-Object {$_.Current.Name})
                $result.steps+=@{action='filter-selection';name=$filter.Current.Name;selected=$selected}
                if($selected.Count -ne 1 -or $selected[0] -notmatch '^全部') {throw 'Records did not default to all scopes'}
            }
        }
        Save-Snapshot '04-records-default'
        foreach($placement in @('停靠底部','停靠左侧','浮动窗口')) {
            Click-Named '位置' ([System.Windows.Automation.ControlType]::Button)
            Click-Named $placement ([System.Windows.Automation.ControlType]::MenuItem)
            Confirm-Navigation
            Confirm-ExcelEntry
            Save-Snapshot ('05-'+$placement)
        }
        # Returning to realtime must hide the owned floating records window.
        Click-Named '返回采集' ([System.Windows.Automation.ControlType]::Button)
        if(@(Find-OwnNamed '记录操作者筛选' | Where-Object {-not $_.Current.IsOffscreen}).Count) {throw 'Records window persisted over realtime'}
        Save-Snapshot '06-realtime-return'
        Select-Trial '含量试验'
        [void](Named '记录操作者筛选' ([System.Windows.Automation.ControlType]::ComboBox))
        Click-Named '记录' ([System.Windows.Automation.ControlType]::CheckBox)
        Click-Named '展开或收起样品信息' ([System.Windows.Automation.ControlType]::CheckBox)
        Confirm-ExcelEntry
        Save-Snapshot '07-sample-expanded'
    }
    if($ExportReport){Exercise-ExcelExport}
    for($i=0;$i -lt 5 -and -not $ExportReport;$i++) {
        foreach($name in @('直径试验','实时采集','含量试验')) {Select-Trial $name; Confirm-Navigation}
    }
    Save-Snapshot '08-navigation-return'
    Click-Named '测量分析' ([System.Windows.Automation.ControlType]::CheckBox)
    [void][CuProbeNative]::Close([uint32]$p.Id,$mainHandle)
    $result.graceful_exit=$p.WaitForExit(12000)
    if(-not $result.graceful_exit) {Save-Snapshot '99-close-failure';throw 'Untouched/loaded trial did not close cleanly'}
    $result['exit_code']=$p.ExitCode
    if($p.ExitCode -ne 0) {throw 'Candidate exited with an error'}
    if($Populated -and (Get-FileHash -LiteralPath $probeProject).Hash.ToLowerInvariant() -ne $result.fixture_sha256) {throw 'Read-only navigation rewrote the owned fixture'}
    $result.ok=$true
} catch {
    $result.errors+=@($_.Exception.Message)
    try {Save-Snapshot '99-failure'} catch {$result.errors+=@($_.Exception.Message)}
} finally {
    if($result.pid) {
        $p.Refresh()
        if(-not $p.HasExited) {$p.Kill();$result.forced_cleanup=$true;[void]$p.WaitForExit(5000)}
        $result['process_exited_after_cleanup']=$p.HasExited
        $p.Dispose()
    }
    $result['finished_at']=(Get-Date).ToString('o')
    Write-JsonFile (Join-Path $runDirectory 'result.json') $result
    [Console]::WriteLine('FDM_FOURTH_GUI_RESULT='+($result|ConvertTo-Json -Depth 14 -Compress))
}
if(-not $result.ok) {exit 1}
