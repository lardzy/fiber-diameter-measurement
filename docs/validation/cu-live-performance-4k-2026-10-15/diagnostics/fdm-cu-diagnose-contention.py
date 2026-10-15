from pathlib import Path
import json, time, sys, threading, subprocess
import psutil

OUT = Path('build/opt4k/contention'); OUT.mkdir(parents=True, exist_ok=True)
if len(sys.argv) > 1 and sys.argv[1] == 'child':
    from scripts import benchmark_cu_live_ui as bench
    from fdm.live_experiment import png_stream, evidence, motion
    from fdm.live_experiment.controller import LiveExperimentController
    rows=[]; lock=threading.Lock()
    def wrap(obj, name, label):
        original=getattr(obj,name)
        def measured(*args,**kwargs):
            start=time.perf_counter(); cpu=time.thread_time()
            try:return original(*args,**kwargs)
            finally:
                item={'label':label,'start':start,'wall_ms':(time.perf_counter()-start)*1000,'thread_cpu_ms':(time.thread_time()-cpu)*1000,'thread':threading.get_native_id()}
                with lock: rows.append(item)
        setattr(obj,name,measured)
    for obj,name,label in [(png_stream,'write_png_stream','png'),(evidence,'write_png_stream','png_encode'),(evidence.os,'fsync','fsync'),(evidence.EvidenceStore,'_write','evidence_total'),(motion.MotionPreparer,'prepare','prepare'),(motion.FieldMotionDetector,'compare_prepared','decision'),(LiveExperimentController,'_accept','publish')]:wrap(obj,name,label)
    i=sys.argv[2]
    bench.main(['--sizes','4096','--evidence','on','--motion-input','lsb-noise','--output',str(OUT/f'diagnostic-{i}.json')])
    (OUT/f'phases-{i}.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
elif __name__ == '__main__':
    for i in range(1,4):
        print('START diagnostic',i,flush=True)
        observations=[]
        with (OUT/f'diagnostic-{i}.log').open('w',encoding='utf8') as log:
            child=subprocess.Popen([sys.executable,'-u',__file__,'child',str(i)],stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                snapshot={'time':time.perf_counter(),'cpu':psutil.cpu_times()._asdict(),'memory':psutil.virtual_memory()._asdict(),'disk':psutil.disk_io_counters()._asdict(),'processes':[]}
                for p in psutil.process_iter(['pid','name','cpu_times','memory_info']):
                    try:
                        t=p.info['cpu_times']; info={'pid':p.pid,'name':p.info['name'],'cpu':t.user+t.system,'rss':p.info['memory_info'].rss}
                        if p.pid==child.pid:info.update(threads=[t._asdict() for t in p.threads()],io=p.io_counters()._asdict())
                        snapshot['processes'].append(info)
                    except (psutil.Error,AttributeError,TypeError):pass
                observations.append(snapshot);time.sleep(1)
        (OUT/f'system-{i}.json').write_text(json.dumps(observations,indent=2),encoding='utf8')
        if child.returncode:raise SystemExit(child.returncode)
        c=json.loads((OUT/f'diagnostic-{i}.json').read_text(encoding='utf8'))['cases'][0];m=c['metrics']
        print(json.dumps(dict(repeat=i,count=m['second_click_to_record_paint_ms']['count'],p95=m['second_click_to_record_paint_ms']['p95_ms'],gui=m['gui_publish_ms']['max_ms'],fps=c['painted_frames_per_second'],failures=c['failures'])),flush=True)
