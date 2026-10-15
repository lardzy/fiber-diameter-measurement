from pathlib import Path
import json,time,sys
import cv2,numpy as np
from fdm.live_experiment import motion
from scripts import benchmark_cu_live_ui as bench
from scripts.benchmark_cu_live_measurement import distribution
old=motion.reduced_gray

def flat(pixels,factor=3):
 h,w=pixels.shape[:2]
 if pixels.ndim==3 and pixels.flags.c_contiguous and h*w>2048**2:
  code=cv2.COLOR_BGRA2GRAY if pixels.shape[2]==4 else cv2.COLOR_BGR2GRAY
  pixels=cv2.cvtColor(pixels.reshape(1,h*w,pixels.shape[2]),code).reshape(h,w)
  return np.ascontiguousarray(cv2.resize(pixels,(max(1,w//factor),max(1,h//factor)),interpolation=cv2.INTER_LINEAR))
 return old(pixels,factor)

out=Path('build/opt4k/flat-probe');out.mkdir(exist_ok=True)
if __name__=='__main__':
 rng=np.random.default_rng(351);rows=[]
 for shape in [(2160,3840),(2730,4096),(2049,2049)]:
  for ch in (3,4):
   pixels=rng.integers(0,256,(*shape,ch),dtype=np.uint8);pixels.flags.writeable=False
   assert np.array_equal(flat(pixels),old(pixels)),(shape,ch)
   timings={n:[] for n in ('blocks','flat')}
   for i in range(35):
    order=[('blocks',old),('flat',flat)]
    if i%2:order.reverse()
    for n,fn in order:
     s=time.perf_counter();a=fn(pixels);elapsed=(time.perf_counter()-s)*1000
     if i>=5:timings[n].append(elapsed)
   row={'shape':list(pixels.shape),'exact':True,'metrics':{n:distribution(v) for n,v in timings.items()}};rows.append(row);print(json.dumps({'shape':row['shape'],**{n:d['p95_ms'] for n,d in row['metrics'].items()}}),flush=True)
 (out/'gray.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
 motion.reduced_gray=flat
 for i in range(1,4):
  print('START flat UI',i,flush=True)
  bench.main(['--sizes','4096','--evidence','on','--motion-input','lsb-noise','--output',str(out/f'ui-{i}.json')])
  c=json.loads((out/f'ui-{i}.json').read_text(encoding='utf8'))['cases'][0];m=c['metrics'];print(json.dumps(dict(i=i,count=m['second_click_to_record_paint_ms']['count'],p95=m['second_click_to_record_paint_ms']['p95_ms'],gui=m['gui_publish_ms']['max_ms'],fps=c['painted_frames_per_second'],failures=c['failures'])),flush=True)
