"""Exercise the real Excel exporter with disposable copies of CU/user templates."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from fdm.live_experiment.models import ExperimentSession, built_in_materials
from fdm.live_experiment.excel_template import ExportSnapshot, TemplateProfile, template_request, run_excel_request


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--counts',type=int,nargs='+',default=[0,1,1488,1500,2000])
    parser.add_argument('--templates',nargs='+',default=['cu-normal','cu-multiscale','user-custom'])
    args=parser.parse_args(); results=[]
    for name in args.templates:
        source=args.directory/(name+'.xlt')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        inspect=run_excel_request({'operation':'inspect','template':str(source),'profile':asdict(TemplateProfile())})
        for count in args.counts:
            session=ExperimentSession(sample_name='Template regression',sample_number='00123')
            for operator in ('A','B','C'):
                key=session.add_operator(operator)
                for direction in ('warp','weft'):
                    session.set_materials(built_in_materials()[:9],operator_id=key,direction=direction)
            material=session.operators[session.active_operator_id].materials['warp'][0].id
            for i in range(count): session.add_measurement(15+(i%7)*.3,material)
            session.adjust_count(material,3)
            destination=args.directory/f'{name}-{count}.xls'
            request=template_request(ExportSnapshot.capture(session),source,destination,show=False)
            started=time.perf_counter()
            try:
                result=run_excel_request(request)
                assert source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==digest
                assert len(result['sheets'])==len(inspect['sheets'])
                result['elapsed_seconds']=time.perf_counter()-started
            except Exception as exc:
                result={'ok':False,'error':str(exc),'elapsed_seconds':time.perf_counter()-started}
            (args.directory/f'{name}-{count}-request.json').write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8-sig')
            results.append({'template':name,'count':count,'result':result})
            (args.directory/'export-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'template':name,'count':count,'ok':result['ok'],'status':result.get('status'),
                'warning_count':len(result.get('warnings',[])),'error':result.get('error')},ensure_ascii=True),flush=True)
    assert all(r['result']['ok'] for r in results)

if __name__=='__main__':main()
