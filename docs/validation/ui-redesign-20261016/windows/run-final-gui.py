"""Run five isolated, strict client-size checks against the final extracted EXE."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts'))
from sync_cu_live_windows import powershell
remote=r'C:\tmp\fdm-cu-live-ui-redesign-20261016'
for theme,width,height,legacy in [('dark',1366,690,False),('light',1280,670,False),('navy',1366,690,False),('warm',1280,670,False),('dark',1366,690,True)]:
    if '--remaining' in sys.argv and theme == 'dark' and not legacy:
        continue
    name='frozen-legacy-upgrade' if legacy else 'frozen-clean-'+theme
    args=f'-Theme {theme} -ClientWidth {width} -ClientHeight {height}'
    if legacy:
        args+=r" -LegacyProject 'C:\tmp\fdm-cu-live-ui-redesign-20261016\docs\validation\cu-save-publish-20261016\windows-frozen-gui\20261016-013257-2d491e5edc1649fb961d01adea859e46\save-lifecycle-probe.fdmproj' -LegacySettings 'C:\tmp\fdm-cu-live-save-peaks-20261016\build\save-peaks\frozen-gui\20261016-013257-2d491e5edc1649fb961d01adea859e46\isolated-config\FiberDiameterMeasurement\settings.json'"
    else:
        args+=' -ExerciseBackgroundSaving:$false'
    code=f"& '{remote}\\scripts\\validation\\probe_workflow_redesign_frozen.ps1' -Executable '{remote}\\dist\\windows\\installer-payload\\{{app}}\\FiberDiameterMeasurement.exe' -OutputRoot '{remote}\\build\\ui-redesign-final\\{name}' {args}"
    print('START',name,flush=True)
    log=ROOT/f'docs/validation/ui-redesign-20261016/windows/final-{name}-launch.log'
    try:
        result=powershell('Windows 11',code,timeout=900)
        log.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        print('PASS',name,flush=True)
    except Exception as error:
        log.write_text(str(error),encoding='utf-8')
        print('FAIL',name,flush=True)
        raise
