"""Disposable real-COM failure/cancel probes; never change machine/user policies."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from unittest.mock import patch
sys.path.insert(0,str(Path.cwd()/'src'))
from fdm.live_experiment import ExperimentSession
from fdm.live_experiment import excel_template as module

root=Path.cwd()/'.tmp'/'windows-excel-20261017'
template=root/'matrix'/'user-custom.xlt'
source=root/'matrix'/'user-custom-1.xls'
results={}
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def policy():
    p=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command','Get-ExecutionPolicy'],capture_output=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    return p.stdout.decode('ascii').strip()
def record(): (root/'exception-probes.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
original=os.environ.pop('PSExecutionPolicyPreference',None)
try:
    normal_policy=policy()
    os.environ['PSExecutionPolicyPreference']='Restricted'
    inherited=policy()
    assert inherited=='Restricted', inherited
    r=module.run_excel_request({'operation':'inspect','template':str(template),'profile':asdict(module.TemplateProfile())})
    assert r['diagnostics']['environment']['effective_policy']=='Bypass'
    results['process_scope']={'parent_effective':inherited,'worker_environment':r['diagnostics']['environment'],'methods':len(r['methods'])}
    os.environ.pop('PSExecutionPolicyPreference',None)
    assert policy()==normal_policy
    results['process_scope']['restored_parent_effective']=normal_policy
    record()
    target=root/'failure-preserved.xls'; shutil.copy2(source,target); before=digest(target)
    session=ExperimentSession(sample_name='Failure probe')
    req=module.template_request(module.ExportSnapshot.capture(session),template,target,method_label='NO_SUCH_STANDARD_FOR_FAILURE_PROBE',show=False)
    try: module.run_excel_request(req)
    except module.ExcelTemplateError as e:
        assert e.code=='worker_failed',e.code
        assert digest(target)==before
        results['failed_export_preserves_report']={'ok':True,'before_sha256':before,'after_sha256':digest(target),'diagnostics':e.diagnostics}
    else: raise AssertionError('invalid standard unexpectedly succeeded')
    record()
    target=root/'cancel-preserved.xls'; shutil.copy2(source,target); before=digest(target)
    req=module.template_request(module.ExportSnapshot.capture(session),template,target,show=False)
    event=threading.Event(); holder={}; original_temporary=module.tempfile.TemporaryDirectory
    def tracked_temp(*args,**kwargs):
        temporary=original_temporary(*args,**kwargs);holder['root']=Path(temporary.name);return temporary
    with patch.object(module.tempfile,'TemporaryDirectory',tracked_temp), ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(module.run_excel_request,req,cancel_event=event)
        deadline=time.monotonic()+60
        while not future.done() and time.monotonic()<deadline:
            owner=holder.get('root',root/'absent')/'result.json.owner.json'
            if owner.is_file():
                try: owner_record=json.loads(owner.read_text(encoding='utf-8-sig'))
                except (ValueError,OSError):time.sleep(.01);continue
                event.set();break
            time.sleep(.01)
        else:
            event.set();raise AssertionError('No owned instance observed before cancellation')
        try:future.result(timeout=30)
        except module.ExcelTemplateCancelled as e:
            assert digest(target)==before
            assert e.diagnostics['cleanup']['ok'],e.diagnostics
            assert e.diagnostics['cleanup']['result']['status'] in ('terminated_owned_instance','not_owned_or_already_closed')
            results['cancel_owned_export']={'ok':True,'owner':owner_record,'before_sha256':before,'after_sha256':digest(target),'diagnostics':e.diagnostics}
        else:raise AssertionError('Cancellation was not observed')
    record()
finally:
    if original is None:os.environ.pop('PSExecutionPolicyPreference',None)
    else:os.environ['PSExecutionPolicyPreference']=original
print(json.dumps({'ok':True,'checks':list(results)},allow_nan=False))
