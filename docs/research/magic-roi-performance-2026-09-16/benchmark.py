from __future__ import annotations
import argparse
import collections
import gc
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PySide6.QtGui import QImage
from fdm.geometry import Point
import fdm.services.prompt_segmentation as mod

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'local-runs'
mod.append_runtime_log = lambda *args, **kwargs: None

class SessionProbe:
    def __init__(self, session, owner, label):
        self.session, self.owner, self.label = session, owner, label
    def __getattr__(self, name):
        return getattr(self.session, name)
    def run(self, *args, **kwargs):
        t = time.perf_counter()
        result = self.session.run(*args, **kwargs)
        elapsed = (time.perf_counter() - t) * 1000
        self.owner.events.append({'stage': self.label, 'ms': elapsed,
                                  'input_shapes': {k: list(v.shape) for k, v in args[1].items()}})
        return result

class Probe(mod.PromptSegmentationService):
    def __init__(self, variant):
        folder = ROOT / 'runtime/segment-anything' / variant
        super().__init__(model_variant=variant, encoder_path=folder / f'{variant}_encoder.onnx',
                         decoder_path=folder / f'{variant}_decoder.onnx')
        self.local_masks = True
        self.events = []
    def _ensure_sessions(self):
        cold = self._encoder_session is None
        t = time.perf_counter()
        super()._ensure_sessions()
        if cold:
            self.events.append({'stage': 'session_init', 'ms': (time.perf_counter()-t)*1000})
            self._encoder_session = SessionProbe(self._encoder_session, self, 'encoder_ort')
            self._decoder_session = SessionProbe(self._decoder_session, self, 'decoder_ort')
    def _image_to_rgb_array(self, image):
        t = time.perf_counter()
        result = super()._image_to_rgb_array(image)
        self.events.append({'stage': 'full_qimage_rgb', 'ms': (time.perf_counter()-t)*1000})
        return result
    def _embedding_for_rgb_array(self, cv_image, *, cache_key):
        cached = cache_key in self._embedding_cache
        t = time.perf_counter()
        result = super()._embedding_for_rgb_array(cv_image, cache_key=cache_key)
        self.events.append({'stage': 'embedding_inclusive', 'ms': (time.perf_counter()-t)*1000,
                            'hit': cached, 'key': cache_key, 'crop_hw': list(cv_image.shape[:2])})
        return result
    def _predict_mask_candidates_from_embedding(self, *args, **kwargs):
        t = time.perf_counter()
        result = super()._predict_mask_candidates_from_embedding(*args, **kwargs)
        self.events.append({'stage':'decoder_and_postprocess', 'ms': (time.perf_counter()-t)*1000,
                            'candidates': len(result)})
        return result

GEOMETRY_ORIGINAL = mod.magic_mask_to_geometry
GEOMETRY_EVENTS = []
GEOMETRY_DEPTH = 0

def geometry_probe(*args, **kwargs):
    global GEOMETRY_DEPTH
    outer = GEOMETRY_DEPTH == 0
    GEOMETRY_DEPTH += 1
    t = time.perf_counter()
    try:
        result = GEOMETRY_ORIGINAL(*args, **kwargs)
        return result
    finally:
        GEOMETRY_DEPTH -= 1
        if outer:
            GEOMETRY_EVENTS.append({'stage': 'geometry', 'ms': (time.perf_counter()-t)*1000})
mod.magic_mask_to_geometry = geometry_probe


def qimage(array):
    return QImage(array.data, array.shape[1], array.shape[0], array.strides[0],
                  QImage.Format.Format_RGB888).copy()

def picture():
    rng = np.random.default_rng(613)
    a = np.clip(rng.normal(200, 4, (1536, 2048, 3)), 0, 255).astype(np.uint8)
    cv2.circle(a, (512,512), 45, (60,70,90), -1)
    cv2.circle(a, (1280,1024), 45, (60,70,90), -1)
    return a

def run_case(service, name, image, pos, neg=(), **kw):
    service.events.clear()
    GEOMETRY_EVENTS.clear()
    t = time.perf_counter()
    cpu = time.process_time()
    result = service.predict_polygon(image=image, cache_key='synthetic', positive_points=pos,
        negative_points=list(neg), tool_mode='magic_segment', roi_enabled=True, **kw)
    ms = (time.perf_counter()-t)*1000
    cpu_ms = (time.process_time()-cpu)*1000
    events = list(service.events) + list(GEOMETRY_EVENTS)
    sums = collections.defaultdict(float)
    for e in events:
        sums[e['stage']] += e['ms']
    embeddings = [e for e in events if e['stage'] == 'embedding_inclusive']
    data = {'name': name, 'wall_ms': ms, 'process_cpu_ms': cpu_ms,
        'stage_ms': dict(sums), 'encoder_calls': sum(e['stage']=='encoder_ort' for e in events),
        'decoder_calls':sum(e['stage']=='decoder_ort' for e in events),
        'cache_hits': sum(e['hit'] for e in embeddings), 'cache_misses':sum(not e['hit'] for e in embeddings),
        'round':result.metadata.get('segmentation_roi_round'),
        'fallback':result.metadata.get('segmentation_fallback_from_roi',False),
        'crop':result.metadata.get('segmentation_crop_box'), 'area':result.area_px,
        'rings':len(result.area_rings_px), 'polygon_vertices':len(result.polygon_px),
        'events':events}
    print(json.dumps({k:v for k,v in data.items() if k!='events'}), flush=True)
    return data

def main():
    global ROOT, OUT
    parser=argparse.ArgumentParser(description='Read-only magic ROI profiling using repository ONNX models.')
    parser.add_argument('--mode', choices=['core','schedule','threads','source','elongated'], default='core')
    parser.add_argument('--variant',default='edge_sam_3x')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args=parser.parse_args()
    ROOT=args.root.resolve()
    OUT=args.output_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    cpu_name=platform.processor()
    if platform.system() == 'Darwin':
        cpu_name=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string']).decode().strip()
    try:
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,stderr=subprocess.DEVNULL).decode().strip()
    except (OSError,subprocess.CalledProcessError):
        git_head='unavailable'
    report={'env':{'platform':platform.platform(),'ort':ort.__version__,'providers_available':ort.get_available_providers(),
        'cv2':cv2.__version__,'cv2_threads':cv2.getNumThreads(),
        'cpu':cpu_name,'logical_cpus':os.cpu_count(),'head':git_head},
        'mode':args.mode,'variant':args.variant}
    if args.mode=='core':
        s=Probe(args.variant)
        img=qimage(picture())
        cases=[]
        cases.append(run_case(s,'new_roi_first_ever',img,[Point(512,512)]))
        cases.append(run_case(s,'repeat_same_roi',img,[Point(512,512)]))
        cases.append(run_case(s,'negative_point_same_roi',img,[Point(512,512)],[Point(557,542)]))
        cases.append(run_case(s,'new_roi_elsewhere',img,[Point(1280,1024)]))
        cases.append(run_case(s,'positive_point_shift_8',img,[Point(512,512),Point(520,512)]))
        report['cases']=cases
        report['encoder_input']=[{'name':x.name,'shape':x.shape,'type':x.type} for x in s._encoder_session.get_inputs()]
        report['decoder_input']=[{'name':x.name,'shape':x.shape,'type':x.type} for x in s._decoder_session.get_inputs()]
        report['providers_used']=s._encoder_session.get_providers()
        opts=s._encoder_session.get_session_options()
        report['session_options']={'intra_op_num_threads':opts.intra_op_num_threads,'inter_op_num_threads':opts.inter_op_num_threads,
            'execution_mode':str(opts.execution_mode)}
        sizes=[]
        for n in [128,256,512,1024]:
            a=cv2.resize(picture()[384:640,384:640],(n,n))
            trials=[]
            for r in range(3):
                s.events.clear()
                t=time.perf_counter()
                emb,_=s._run_encoder(a)
                elapsed=(time.perf_counter()-t)*1000
                trials.append({'total_ms':elapsed,'encoder_ms':next(e['ms'] for e in s.events if e['stage']=='encoder_ort'),
                    'tensor_shapes':next(e['input_shapes'] for e in s.events if e['stage']=='encoder_ort')})
            row={'roi_side':n,'median_ms':statistics.median(x['total_ms'] for x in trials), 'trials':trials}
            sizes.append(row)
            print(json.dumps(row),flush=True)
        report['encoder_size_sweep']=sizes
        del s
    elif args.mode=='schedule':
        class SchedulingProbe(Probe):
            def _run_encoder(self, a):
                self.events.append({'stage':'encoder_ort','ms':0.0,'mock':True})
                return np.zeros((1,1),np.float32),a.shape[:2]
            def _predict_mask_from_embedding(self, embedding,**kwargs):
                self.events.append({'stage':'decoder_ort','ms':0.0,'mock':True})
                # Deliberately fills crop: triggers expansion by area and border criteria.
                return np.ones(embedding.original_size,dtype=bool)
        s=SchedulingProbe(args.variant)
        img=qimage(np.full((1536,2048,3),128,np.uint8))
        report['synthetic_control_flow_only']=True
        report['cases']=[run_case(s,'all_edges_first',img,[Point(1024,768)]),
                         run_case(s,'all_edges_repeat',img,[Point(1024,768)])]
    elif args.mode=='threads':
        original=ort.InferenceSession
        rows=[]
        a=picture()[384:640,384:640].copy()
        for threads in [0,1,2,4,8]:
            def factory(*args,**kwargs):
                opts=ort.SessionOptions()
                opts.intra_op_num_threads=threads
                kwargs['sess_options']=opts
                return original(*args,**kwargs)
            ort.InferenceSession=factory
            s=Probe(args.variant)
            t=time.perf_counter()
            s._ensure_sessions()
            init=(time.perf_counter()-t)*1000
            trials=[]
            for r in range(4):
                s.events.clear()
                t=time.perf_counter()
                cpu=time.process_time()
                embedding=s._embedding_for_rgb_array(a,cache_key=f'thread-bench-{r}')
                s._predict_mask_candidates_from_embedding(embedding,positive_points=[Point(128,128)],negative_points=[])
                trials.append({'wall_ms':(time.perf_counter()-t)*1000,'cpu_ms':(time.process_time()-cpu)*1000,
                    'encoder_ms':next(e['ms'] for e in s.events if e['stage']=='encoder_ort'),
                    'decoder_ms':next(e['ms'] for e in s.events if e['stage']=='decoder_ort')})
            row={'threads':threads,'session_init_ms':init,
                 'warm_median_wall_ms':statistics.median(x['wall_ms'] for x in trials[1:]),
                 'warm_median_encoder_ms':statistics.median(x['encoder_ms'] for x in trials[1:]),
                 'warm_median_decoder_ms':statistics.median(x['decoder_ms'] for x in trials[1:]),
                 'trials':trials}
            rows.append(row)
            print(json.dumps(row),flush=True)
            del s
            gc.collect()
        ort.InferenceSession=original
        report['thread_sweep']=rows
    elif args.mode=='elongated':
        a=np.full((1536,2048,3),205,np.uint8)
        cv2.rectangle(a,(0,730),(2047,806),(55,65,85),-1)
        service=Probe(args.variant)
        img=qimage(a)
        report['fixture']='synthetic horizontal strip, actual ONNX inference'
        report['cases']=[run_case(service,'real_model_cross_frame_strip_first',img,[Point(1024,768)]),
                         run_case(service,'real_model_cross_frame_strip_repeat',img,[Point(1024,768)])]
    elif args.mode=='source':
        from fdm.services.segmentation_source import _qimage_content_version
        rows=[]
        for w,h in [(2048,1536),(6000,4000)]:
            img=qimage(np.full((h,w,3),128,np.uint8)).convertToFormat(QImage.Format.Format_ARGB32)
            trials=[]
            for r in range(4):
                t=time.perf_counter(); _qimage_content_version(img); hash_ms=(time.perf_counter()-t)*1000
                t=time.perf_counter(); rgb=mod.qimage_to_rgb_array(img); rgb_ms=(time.perf_counter()-t)*1000
                del rgb
                trials.append({'hash_ms':hash_ms,'rgb_ms':rgb_ms})
            row={'image_wh':[w,h],'hash_median_ms':statistics.median(x['hash_ms'] for x in trials[1:]),
                 'rgb_median_ms':statistics.median(x['rgb_ms'] for x in trials[1:]),'trials':trials}
            rows.append(row); print(json.dumps(row),flush=True)
        report['source_sweep']=rows
    path=OUT/f'{args.mode}-{args.variant}.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULT_FILE '+str(path),flush=True)

if __name__=='__main__': main()
