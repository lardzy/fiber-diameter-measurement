from pathlib import Path
import sys,time,json,threading,subprocess,os
root=Path.cwd();sys.path[:0]=[str(root),str(root/'src')]
if __name__=='__main__':
 import multiprocessing
 multiprocessing.freeze_support()
 mode=sys.argv[1];target=Path(sys.argv[2]);target.mkdir(parents=True,exist_ok=True)
 import scripts.benchmark_cu_live_measurement as b
 timings=[];original=b.resource_snapshot
 def snapshot():
  t=time.perf_counter();v=original();timings.append((time.perf_counter()-t)*1000);return v
 b.resource_snapshot=snapshot
 if mode=='off':b.ResourceSampler._run=lambda self:self.stop.wait()
 import scripts.validation.soak_cu_workspace as s
 sys.argv=['soak','--size','4096','--duration-seconds','30','--minimum-accepted','30','--output',str(target/'result.json')]
 rc=s.main()
 (target/'sampler.json').write_text(json.dumps({'mode':mode,'timings':b.distribution(timings)},indent=2))
 print('DIAGNOSTIC_DONE',mode,rc,flush=True)
