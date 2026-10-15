from pathlib import Path
import sys,types,time,tempfile,os,json,hashlib
import numpy as np,cv2
from fdm.live_experiment import png_stream
from scripts.benchmark_cu_live_measurement import fixture,distribution
source=Path(png_stream.__file__).read_text(encoding='utf8')
old='''                colors[:rows, :, 0] = pixels[y:y + rows, :, 2]
                colors[:rows, :, 1] = pixels[y:y + rows, :, 1]
                colors[:rows, :, 2] = pixels[y:y + rows, :, 0]
                if channels == 4:
                    colors[:rows, :, 3] = pixels[y:y + rows, :, 3]'''
assert old in source
source=source.replace(old,'''                cv2.cvtColor(pixels[y:y + rows], cv2.COLOR_BGRA2RGBA if channels == 4 else cv2.COLOR_BGR2RGB, dst=colors[:rows])''').replace('import numpy as np','import numpy as np\nimport cv2')
module=types.ModuleType('fdm_png_color_probe');sys.modules[module.__name__]=module;exec(compile(source,'<diagnostic-png-color-probe>','exec'),module.__dict__)
pixels,_,_=fixture(4096);rng=np.random.default_rng(20261015);pixels=np.bitwise_xor(pixels,rng.integers(0,2,pixels.shape,dtype=np.uint8));pixels.flags.writeable=False
report={'scope':'isolated PNG probe; not GUI acceptance','opencv_threads':cv2.getNumThreads(),'rows':[]}
with tempfile.TemporaryDirectory(prefix='fdm-png-pack-probe-') as temp:
 path=Path(temp)/'test.png'
 for step in range(35):
  row={'warmup':step<5}
  pairs=[('current',png_stream),('cv2_pack',module)]
  if step%2:pairs.reverse()
  for name,m in pairs:
   begin=time.perf_counter()
   with path.open('wb',buffering=0) as f:
    encoded=m.write_png_stream(pixels,f);os.fsync(f.fileno())
   row[name]=(time.perf_counter()-begin)*1000
   decoded=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_UNCHANGED);assert np.array_equal(decoded,pixels)
  report['rows'].append(row)
report['metrics']={n:distribution([r[n] for r in report['rows'] if not r['warmup']]) for n in ('current','cv2_pack')}
out=Path('build/opt4k/contention/png-pack.json');out.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report['metrics']),flush=True)
