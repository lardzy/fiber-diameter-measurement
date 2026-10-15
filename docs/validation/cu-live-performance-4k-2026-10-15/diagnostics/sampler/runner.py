from pathlib import Path
import subprocess,sys,json
out=Path('build/opt4k/sampler-diagnostic');out.mkdir(exist_ok=True)
for i,mode in enumerate(['on','off','on','off']):
 target=out/(str(i)+'-'+mode);print('START',str(target),flush=True)
 with (out/(str(i)+'.log')).open('w',encoding='utf8') as f:
  p=subprocess.run([sys.executable,'-u','build/opt4k/sampler-diagnostic.py',mode,str(target)],stdout=f,stderr=subprocess.STDOUT)
 if not (target/'result.json').exists():print((out/(str(i)+'.log')).read_text()[-5000:]);break
 d=json.loads((target/'result.json').read_text());print('RESULT',mode,p.returncode,json.dumps({k:v for k,v in d.items() if k in ['accepted','distinct_painted_fps','gui_publish_over_4ms','active_seconds']}),flush=True)
 print('SAMPLER',json.loads((target/'sampler.json').read_text())['timings'],flush=True)
