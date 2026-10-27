"""Measure the one authorized VM and invoke documented stopped-disk compaction."""
import argparse,json,os,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
VM=Path('/Users/lishuyang/Parallels/Windows 11.pvm')
BIN=Path('/Applications/Parallels Desktop.app/Contents/MacOS')
def status():
 p=subprocess.run([str(BIN/'prlctl'),'status','Windows 11'],capture_output=True,text=True,check=True)
 return p.stdout.strip()
def measure(label):
 state=status(); rows=[]
 for parent,dirs,files in os.walk(VM,followlinks=False):
  for n in files:
   p=Path(parent)/n;s=p.lstat()
   if stat.S_ISREG(s.st_mode):rows.append({'path':str(p.relative_to(VM)),'logical_bytes':s.st_size,'allocated_bytes':s.st_blocks*512})
 out={'utc':datetime.now(timezone.utc).isoformat(),'state':state,'method':'stat regular-file allocated bytes, no APFS shared extent ownership','logical_bytes':sum(x['logical_bytes'] for x in rows),'allocated_bytes':sum(x['allocated_bytes'] for x in rows),'files':rows}
 (ROOT/('host-'+label+'.json')).write_text(json.dumps(out,indent=2)+'\n')
 return {k:v for k,v in out.items() if k!='files'}
def compact(info=False):
 state=status()
 if 'stopped' not in state.lower():raise RuntimeError('VM must be stopped: '+state)
 args=[str(BIN/'prl_disk_tool'),'compact']
 if info:args+=['--info','--details']
 args+=['--hdd',str(VM/'harddisk.hdd')]
 label='compact-info' if info else 'compact'
 if not info:
  receipt=json.loads((ROOT/'compact-info-result.json').read_text())
  if receipt['exit_code']!=0:raise RuntimeError('Compaction information check did not pass')
 if not info and (ROOT/'compact-started.json').exists():raise RuntimeError('Compaction already dispatched; inspect receipts, never repeat automatically')
 if not info:(ROOT/'compact-started.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'state':state}))
 with (ROOT/(label+'.log')).open('w') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 result={'exit_code':p.returncode,'log':label+'.log','utc':datetime.now(timezone.utc).isoformat(),'state_after':status()}
 (ROOT/(label+'-result.json')).write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['measure','info','compact']);a.add_argument('label',nargs='?',default='measurement');x=a.parse_args()
 print(json.dumps(measure(x.label) if x.action=='measure' else compact(x.action=='info')))
