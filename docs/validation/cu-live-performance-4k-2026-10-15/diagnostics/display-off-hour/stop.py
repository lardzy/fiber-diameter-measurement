from pathlib import Path
import psutil, json, time
matches=[]
for p in psutil.process_iter(['pid','cmdline']):
    cmd=p.info['cmdline'] or []
    if any(x.endswith('soak_cu_workspace.py') for x in cmd) and 'build\\opt4k\\hour\\workspace-4k-hour.json' in cmd:
        matches.append(p)
roots=[p for p in matches if p.ppid() not in {q.pid for q in matches}]
assert len(roots)==1, [(p.pid,p.ppid(),p.cmdline()) for p in matches]
p=roots[0]
assert any(Path(q.cwd()).resolve()==Path.cwd().resolve() for q in matches), [(q.pid,q.cwd()) for q in matches]
children=p.children(recursive=True)
report={'kind':'aborted diagnostic; not completed endurance acceptance','reason':'Repeated ~9.5 FPS for several minutes; preserve artifacts, investigate then rerun','at':time.time(),'process':{'pid':p.pid,'cmdline':p.cmdline(),'cwd':p.cwd()},'children':[{'pid':c.pid,'cmdline':c.cmdline()} for c in children],'interventions':['One 2-second read-only system CPU/RSS/disk sampler while running','Precise termination of this synthetic benchmark and its automatic worker; accepted synthetic in-memory ledger is not persisted; existing PNG files retained']}
Path('build/opt4k/hour/aborted.json').write_text(json.dumps(report,indent=2),encoding='utf8')
for c in children:c.terminate()
p.terminate()
_,alive=psutil.wait_procs([p,*children],timeout=10)
assert not alive, [(x.pid,x.cmdline()) for x in alive]
print(json.dumps(report))
