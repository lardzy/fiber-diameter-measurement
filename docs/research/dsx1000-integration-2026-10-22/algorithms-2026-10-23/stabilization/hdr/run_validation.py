"""Static-derived 8-bit Hamal synthetic checks, explicitly without vendor oracle."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np
import scipy

from hamal8_reference import (Settings, KERNEL, enhance, gain_map,
                              thumbnail_geometry, UnsupportedThumbnailGeometry)

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation'
SEED=2026102301


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    rng=np.random.default_rng(SEED)
    h,w=240,320
    y,x=np.mgrid[:h,:w]
    ramp=(x*255//(w-1)).astype(np.uint8)
    gradient=np.repeat(ramp[:,:,None],3,axis=2)
    dark=np.clip(8+x/18+5*np.sin(x*.7)+4*np.cos(y*.8)+rng.normal(0,1,(h,w)),0,255).astype(np.uint8)
    dark=np.repeat(dark[:,:,None],3,axis=2)
    dark[60:140,210:290]=255
    color=rng.integers(0,256,(h,w,3),dtype=np.uint8)
    cases={'black':np.zeros_like(gradient),'white':np.full_like(gradient,255),
           'gradient':gradient,'dark_texture_and_saturation':dark,'color_noise':color}
    fixtures={}
    records=[]
    checks=[]
    def check(name,condition,detail):
        checks.append({'name':name,'passed':bool(condition),'detail':detail})
        if not condition:raise AssertionError(name)
    check('kernel_identity',KERNEL.shape==(33,33) and KERNEL.sum()==2001 and np.array_equal(KERNEL,KERNEL[::-1,::-1]),
          'Extracted 1089 coefficients, symmetric, sum2001')
    # Native broad range is not a promise that fixed-point mapping covers all
    # thumbnail cells. Refuse uncovered cells instead of inventing their values.
    geometry_results=[]
    for name,shape,expected in [('missing_last_column',(240,1445,3),(79,60)),
                               ('missing_last_row',(1221,320,3),(80,59))]:
        src=np.zeros(shape,np.uint8)
        fixtures[name+'__source']=src
        error=None
        try: enhance(src)
        except UnsupportedThumbnailGeometry as e: error=e
        g=thumbnail_geometry(*shape[:2])
        geometry_results.append(g)
        check(name+'_explicit_rejection',error is not None and
              (g['visited_columns'],g['visited_rows'])==expected,
              {'geometry':g,'error':str(error)})
    widths=[v for v in range(121,10001) if not thumbnail_geometry(240,v)['supported']]
    heights=[v for v in range(91,10001) if not thumbnail_geometry(v,320)['supported']]
    check('dimension_domain_exhaustive_counts',
          len(widths)==3701 and widths[0]==1445 and len(heights)==3861 and heights[0]==1221,
          {'unsupported_widths':len(widths),'first_width':widths[0],
           'unsupported_heights':len(heights),'first_height':heights[0]})
    # Independent direct enumeration verifies the analytical classifier on
    # both sides of the first failures and normal/end-range dimensions.
    mapped_ok=True
    for hh,ww in [(91,121),(240,320),(240,1444),(240,1445),(1220,320),(1221,320),(10000,10000)]:
        g=thumbnail_geometry(hh,ww)
        cols=len(np.unique((np.arange(ww)*g['sx']+(g['sx']>>1))>>13))
        rows=len(np.unique((np.arange(hh)*g['sy']+(g['sy']>>1))>>13))
        mapped_ok &= (cols,rows)==(g['visited_columns'],g['visited_rows'])
    check('dimension_classifier_matches_direct_samples',mapped_ok,
          'Direct enumeration includes supported and unsupported row/column boundaries')
    for name,value in [('noise_slope1',(-1,)*8),('noise_slope1',(32768,)*8),
                       ('noise_slope2',(8192,)*8),('noise_bins',(1024,)*8),
                       ('noise_slope1',(25,)),('noise_bins',(1,)*9)]:
        refused=False
        try: enhance(gradient,replace(Settings(),**{name:value}))
        except ValueError: refused=True
        check('invalid_'+name+'_'+str(value[0])+'_length'+str(len(value)),refused,
              'Reject native-invalid range/length before broadcasting or image processing')
    for name,src in cases.items():
        before=hashlib.sha256(src.tobytes()).hexdigest()
        out,d=enhance(src,return_intermediates=True)
        fixtures[name+'__source']=src
        fixtures[name+'__output']=out
        fixtures[name+'__gain']=d['gain']
        records.append({'name':name,'input_sha256':before,'output_sha256':hashlib.sha256(out.tobytes()).hexdigest(),
                        'input_mean':src.mean(axis=(0,1)).tolist(),'output_mean':out.mean(axis=(0,1)).tolist(),
                        'gain_range':[int(d['gain'].min()),int(d['gain'].max())],
                        'tone_range':[int(d['tone'].min()),int(d['tone'].max())],
                        'adaptive_knots':d['adaptive_knots'].tolist(),'adaptive_slopes':d['adaptive_slopes'].tolist()})
        check(name+'_input_unchanged',before==hashlib.sha256(src.tobytes()).hexdigest(),'Source retained')
        check(name+'_shapes_ranges',out.shape==src.shape and out.dtype==np.uint8 and
              d['tone'].shape==(h,w) and d['gain'].min()>=0 and d['gain'].max()<=32767,'Native-style bounds')
        check(name+'_cumulative_monotonic',np.all(np.diff(d['cumulative_planes'],axis=0)>=0),'Each local curve is monotonic')
        check(name+'_repeatable',np.array_equal(enhance(src),out),'Same implementation/settings/bytes reproduce result')
        if name in ('gradient','dark_texture_and_saturation'):
            cv2.imwrite(str(OUT/(name+'-before.png')),src)
            cv2.imwrite(str(OUT/(name+'-after.png')),out)
    check('black_remains_black',np.max(fixtures['black__output'])==0,'No division-by-zero brightness path')
    check('white_remains_clipped',np.min(fixtures['white__output'])==255,'Does not imply recovery of hidden detail')
    identity=enhance(gradient,Settings(strength=0))
    check('zero_strength_gray_identity',np.array_equal(identity,gradient),'Gray matrix round trip and unity gain')
    unity=enhance(color,Settings(strength=0))
    color_error=np.abs(unity.astype(int)-color.astype(int))
    check('zero_strength_color_rounding',color_error.max()<=2,'Observed matrix rounding still changes some color values')
    low=np.ones((h,w,3),dtype=np.uint8)
    low_out=enhance(low)
    check('dim_constant_is_modified',not np.array_equal(low_out,low),'Enhancement may amplify low signal; not evidence of recovered structure')
    # Two different latent scenes collapse to exactly the same saved 8-bit input.
    latent1=np.full((h,w,3),300.0)
    latent2=300.0+20*np.sin(x*.4)[:,:,None]+np.zeros((h,w,3))
    clipped1=np.clip(latent1,0,255).astype(np.uint8)
    clipped2=np.clip(latent2,0,255).astype(np.uint8)
    check('saturation_information_loss',not np.array_equal(latent1,latent2) and np.array_equal(clipped1,clipped2)
          and np.array_equal(enhance(clipped1),enhance(clipped2)),
          'Identical clipped inputs have identical deterministic outputs; input-information limitation')
    fixtures['latent_scene1']=latent1
    fixtures['latent_scene2']=latent2
    fixtures['clipped_scene1']=clipped1
    fixtures['clipped_scene2']=clipped2
    np.savez_compressed(OUT/'fixtures.npz',**fixtures)
    result={'scope':'8bit AsymMode1 BlendMode1 whole-frame core; no vendor execution',
            'seed':SEED,'settings':asdict(Settings()),'versions':{'python':platform.python_version(),
              'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__},
            'source_sha256':{n:sha(ROOT/n) for n in ['hamal8_reference.py','run_validation.py','native-tables.json']},
            'fixtures_sha256':sha(OUT/'fixtures.npz'),'cases':records,'checks':checks,
            'rejected_geometry_cases':geometry_results,
            'zero_strength_color_max_abs_error':int(color_error.max()),
            'counts':{'cases':len(cases),'checks':len(checks),'passed':sum(c['passed'] for c in checks)},
            'limitations':['No vendor numeric oracle','Prototype invariants are not proof of vendor equivalence',
              'Color channel handoff must be verified','Native stripe memory scheduling omitted',
              'Only geometry covering all 80x60 thumbnail cells is supported; uncovered native cells unresolved',
              '14bit, external Spica denoising and multiexposure HDR excluded']}
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# Hamal 8 bit 合成验证','',f'Seed `{SEED}`，{len(cases)} 类图像，{len(checks)} 项检查通过。',
           '', '原始输入、重建输出与 gain 图保存在 fixtures.npz。示例 PNG 可直接并排查看。',
           '', '这些检查证明研究实现可执行、可重复、保留输入及满足所列数值约束；没有执行原厂 DLL，不能当作原厂等价验收。',
           '', '两个不同的过曝前场景都截断成全白 8 bit 输入，重建输出相同。这验证输入信息丢失，不是对原厂采集效果的实测评价。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(result['counts'])


if __name__=='__main__':main()
