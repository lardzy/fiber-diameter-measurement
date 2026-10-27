"""Resolve missing full-profile test assets after the no-area installer was built.

Only copy pinned existing VM development assets into the owned test workspace;
never rebuild or modify the already verified installer. Preserve the first run.
"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tomllib,xml.etree.ElementTree as ET
ROOT=Path(r'C:\tmp\fdm-professional-precision-release-20261027')
OUT=ROOT/'build/source-validation'
OLD=Path(r'C:\Users\lishuyang\Downloads\fiber-diameter-measurement')
NAMES=['test_collect_runtime_datas_omits_area_infer_docker_tests_and_pycache','test_full_profile_required_assets_have_complete_authoritative_metadata','test_runtime_profiles_keep_core_small_and_full_complete']
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 initial=ET.parse(OUT/'windows-regression.xml')
 failures=[t for t in initial.iter('testcase') if t.find('failure') is not None or t.find('error') is not None]
 assert len(failures)==3 and sum(len(t.findall('failure'))+len(t.findall('error')) for t in failures)==8 and all(any(t.get('name','').startswith(n) for n in NAMES) for t in failures)
 identity=json.loads((ROOT/'build/package/installer-identity.json').read_text())
 installer=Path(identity['installer']);assert identity['passed'] and sha(installer)==identity['sha256']
 rows=[]
 for item in tomllib.loads((ROOT/'runtime_assets.toml').read_text())['assets']:
  if item.get('group')!='area-models':continue
  source=OLD/item['source'];target=ROOT/item['source']
  assert source.is_file() and source.stat().st_size==item['size'] and sha(source)==item['sha256']
  assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  assert target.stat().st_size==item['size'] and sha(target)==item['sha256']
  rows.append(dict(source=str(source),target=str(target),bytes=item['size'],sha256=item['sha256']))
 assert len(rows)==6
 write(OUT/'test-assets-restored.json',dict(purpose='Temporary fixtures for three full-profile tests; installer was already built and remains unchanged',files=rows,installer_sha256_before=identity['sha256']))
 shutil.copy2(OUT/'status.json',OUT/'initial-status.json')
 env=os.environ.copy();env.update(APPDATA=str(OUT/'isolated-config'),LOCALAPPDATA=str(OUT/'isolated-config'),TEMP=str(OUT/'temp'),TMP=str(OUT/'temp'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',QT_QPA_PLATFORM='windows',PYTHONPATH=str(ROOT/'src'))
 for key in ('QT_SCALE_FACTOR','QT_SCREEN_SCALE_FACTORS','QT_FONT_DPI'):env.pop(key,None)
 args=[sys.executable,'-m','pytest','-q','--junitxml='+str(OUT/'windows-runtime-fixture-rerun.xml')]+['tests/test_build_support.py::BuildSupportTests::'+n for n in NAMES]
 with (OUT/'windows-runtime-fixture-rerun.log').open('w',encoding='utf-8') as stdout,(OUT/'windows-runtime-fixture-rerun.stderr.log').open('w',encoding='utf-8') as stderr:
  code=subprocess.run(args,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=300).returncode
 status=dict(exit_code=code,ui=[],initial_failures=[t.get('name') for t in failures],targeted_rerun=NAMES,scope='All initial regression failures resolved by targeted rerun using pinned missing test assets; initial results retained')
 if code==0:
  tree=ET.parse(OUT/'windows-runtime-fixture-rerun.xml');assert not any(t.find('failure') is not None or t.find('error') is not None for t in tree.iter('testcase'))
  for scale in (1,1.5,2):
   tag=int(scale*100)
   with (OUT/f'ui-{tag}.log').open('w',encoding='utf-8') as stdout,(OUT/f'ui-{tag}.stderr.log').open('w',encoding='utf-8') as stderr:
    code=subprocess.run([sys.executable,str(ROOT/'scripts/validation/validate_professional_precision_ui.py'),'--platform','windows','--scale',str(scale),'--output',str(OUT/f'windows-{tag}')],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=180).returncode
   status['ui'].append(dict(scale=scale,exit_code=code));status['exit_code']=code
   if code:break
 assert sha(installer)==identity['sha256']
 status['installer_sha256_after']=identity['sha256']
 write(OUT/'status.json',status)
 print(json.dumps(status));return status['exit_code']
if __name__=='__main__':raise SystemExit(main())
