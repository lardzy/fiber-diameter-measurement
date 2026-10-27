"""Deterministic semantic/counterexample validation; never loads vendor binaries."""
from __future__ import annotations
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import cv2
import scipy
from scipy.interpolate import CubicSpline
from reference_efi import (Profile, natural_spline, spline_values, estimate_height,
    texture_fill, GuidedFilter, reconstruct, sample_z, focus_bands, pyr_up, pyr_down,
    stabilize_given_vectors, normalize_stack, band_blend, input_order_argmax, axial_candidates)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reference-validation'
OUT.mkdir(exist_ok=True)
results = []
fixtures = {}


def check(name, assertion, **metrics):
    results.append({'name':name,'passed':bool(assertion), **metrics})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


start_time = time.monotonic()
rng = np.random.default_rng(20261023)

# Independent SciPy oracle checks coefficients/evaluation, including irregular Z.
x = np.array([.6,1.3,2.7,4.6,7.8,9.2,12.4])
y = rng.normal(size=(len(x),4))
query = np.linspace(x[0],x[-1],107)
reference = spline_values(x,natural_spline(x,y),query)
oracle = CubicSpline(x,y,bc_type='natural')(query)
error = float(np.max(np.abs(reference-oracle)))
check('natural_spline_independent_scipy_oracle', error < 1e-11, max_absolute_error=error)
fixtures.update(spline_x=x,spline_y=y,spline_query=query)

# A partition-of-unity invariant across all knots and transitions is independent
# of individual branch assertions; profiles here are static managed fallbacks.
for name, thresholds in [('3d',[70,35,20,15,10,0]),('efi',[35,20,15,10,5,3])]:
    weights=np.array([band_blend(v,thresholds) for v in np.linspace(-5,90,381)])
    check('band_blend_partition_'+name, np.all(weights>=0) and np.all(weights<=1)
          and np.max(np.abs(weights.sum(1)-1))<1e-12)
    fixtures['band_weights_'+name]=weights
tie=np.ones((3,2,2),np.float32)
check('response_tie_uses_original_input_order',
      np.all(input_order_argmax(tie,[2,1,0])==0))

z = np.array([.6,1.3,2.7,4.6,7.8,9.2,12.4],np.float32)
response = np.exp(-((z-4.4)/1.7)**2).astype(np.float32)[:,None,None]*20
height,peak,info = estimate_height(response,z)
estimate = float(height[0,0])
check('nonuniform_fractional_z', abs(estimate-4.4)<.7,
      truth=4.4, estimate=estimate, grid_step=info['grid_step'],
      note='Synthetic Gaussian focus curve; grid resolution is not optical accuracy.')
fixtures.update(nonuniform_z=z,nonuniform_response=response)

z2 = np.arange(9,dtype=np.float32)*1.3+.4
boundary = np.exp(-(z2-z2[0])/2).astype(np.float32)[:,None,None]
bh,bp,bi = estimate_height(boundary,z2)
check('boundary_peak_is_flagged', bool(bi['boundary_winner'][0,0]) and abs(float(bh[0,0])-float(z2[0]))<.2,
      estimate=float(bh[0,0]), boundary=True)
last_response=np.arange(9,dtype=np.float32)[:,None,None]
last_h,_,_=estimate_height(last_response,z2)
check('fractional_z_last_layer_float32_grid_endpoint',last_h[0,0]==z2[-1],
      estimated=float(last_h[0,0]),expected=float(z2[-1]),
      regression='Prior mixed float64 grid dropped the last candidate, yielding 10.6374998093.')
interior_grid,_=axial_candidates(z2,3,5)
check('fractional_z_interior_window_float32_endpoint',len(interior_grid)==33
      and interior_grid[0]==np.float32(4.299999713897705) and interior_grid[-1]==z2[7],
      count=len(interior_grid),first=float(interior_grid[0]),last=float(interior_grid[-1]),
      oracle='Native float32 ceil/floor yields global grid indices 24..56 inclusive.')
fixtures.update(last_layer_response=last_response,rounding_z=z2,interior_candidate_grid=interior_grid)
flat = np.ones((9,3,4),np.float32)*.001
fh,fp,fi = estimate_height(flat,z2)
filled,fm = texture_fill(fh,fp,1.8,cap=40)
check('textureless_reconstruction_has_no_measured_support',
      not fm['measured'].any() and not fm['supported'].any(),
      raw_height_range=[float(fh.min()),float(fh.max())],
      filled_numeric_range=[float(filled.min()),float(filled.max())],
      warning='Finite raw/filled numbers do not establish measured height; all pixels invalid.')
fixtures.update(boundary_response=boundary,flat_response=flat)

# Two distinct axial peaks demonstrate the single-surface information loss.
zm = np.linspace(0,12,17,dtype=np.float32)
multi = (np.exp(-((zm-3)/.7)**2)+.96*np.exp(-((zm-9)/.7)**2))[:,None,None]
mh,mp,mi = estimate_height(multi,zm)
check('two_surfaces_single_depth_counterexample', abs(float(mh[0,0])-3)<.4,
      reconstructed=float(mh[0,0]), other_surface=9.0,
      limitation_confirmed='Only one peak is retained; full Z stack remains necessary.')
fixtures.update(multipeak_z=zm,multipeak_response=multi)

# Texture fill must preserve reliable anchors, and expose both unsupported and filled.
d = np.zeros((31,35),np.float32)
d[10:20,:9] = 2.5
d[10:20,-9:] = 8.5
r = np.zeros_like(d)
r[10:20,:9]=10
r[10:20,-9:]=10
f,mask = texture_fill(d,r,2,scale=1.,kernel=5,sad_limit=.01,cap=300)
anchor_err = float(np.max(np.abs(f[r>=2]-d[r>=2])))
check('texture_fill_keeps_anchors_and_labels_synthesis',
      anchor_err==0 and mask['filled'].any() and np.isfinite(f).all(),
      anchor_max_error=anchor_err, synthesized_pixels=int(mask['filled'].sum()),
      iterations=mask['iterations'],converged=mask['converged'])
f0,m0 = texture_fill(d,r,2,scale=1.,kernel=5,sad_limit=0,cap=3)
check('zero_sad_threshold_bounded_reference_counterexample',
      m0['iterations']==3 and not m0['converged'],
      note='Native strict SAD<threshold loop has no observed count cap; reference stops at 3.')
fixtures.update(texture_depth=d,texture_quality=r)

# Gray and RGB guided filters preserve constants. Flat guidance permits independent
# double-box oracle, rather than checking an implementation against itself.
constant_guide=np.ones((32,40),np.float32)*100
p=rng.normal(size=(32,40)).astype(np.float32)
actual=GuidedFilter(constant_guide,2,.01,scale=1).apply(p)
expected=cv2.blur(cv2.blur(p,(5,5),borderType=cv2.BORDER_REFLECT_101),(5,5),borderType=cv2.BORDER_REFLECT_101)
gferr=float(np.max(np.abs(actual-expected)))
check('guided_filter_constant_guide_double_box_oracle',gferr<.005,max_absolute_error=gferr)
guide_rgb=rng.uniform(0,255,(32,40,3)).astype(np.float32)
constant=np.ones((32,40),np.float32)*4.75
gf=GuidedFilter(guide_rgb,3,.01,scale=.5).apply(constant)
check('rgb_guided_filter_preserves_constant',np.max(np.abs(gf-constant))<.002,
      max_absolute_error=float(np.max(np.abs(gf-constant))))
high_gray=np.random.default_rng(2026102301).uniform(0,65535,(32,40)).astype(np.float32)
high_rgb=np.repeat(high_gray[...,None],3,axis=2)
rgb_filtered=GuidedFilter(high_rgb,3,.01,scale=.5).apply(p)
gray_filtered=GuidedFilter(high_gray,3,.01/3,scale=.5).apply(p)
# For I=(g,g,g), sum(a_i) has denominator var(g)+eps/3: this scalar identity
# is independent of the 3x3 inverse and detects high-dynamic-range degeneracy.
collinear_error=float(np.max(np.abs(rgb_filtered-gray_filtered)))
check('uint16_range_collinear_rgb_scalar_oracle',np.isfinite(rgb_filtered).all()
      and collinear_error<2e-4,max_absolute_error=collinear_error,
      difference_from_native='Reference moments/solve use float64; original numeric parity unclaimed.')
fixtures.update(collinear_high_guide=high_rgb,guided_field=p)

# Constants check DC preservation only. Nonconstant hand-computed edges below
# discriminate the recovered parity-dependent halo rule.
for sh in [(33,47),(32,48),(33,48),(32,47)]:
    a=np.full(sh,157.,np.float32)
    q=pyr_up(pyr_down(a),sh)
    bands=focus_bands(a,4,3)
    check('pyramid_constant_'+str(sh), np.max(np.abs(q-a))<1e-5 and bands.max()<1e-5,
          reconstructed_shape=list(q.shape),maximum_focus_response=float(bands.max()))
edge=np.array([[0,8,0,16]],np.float32)
for size,expected in [(7,[2,4,6,4,3,8,12]),(8,[2,4,6,4,3,8,14,16])]:
    actual=pyr_up(edge,(1,size))[0]
    vertical=pyr_up(edge.T,(size,1))[:,0]
    check('pyramid_nonconstant_edge_halo_'+str(size),
          np.array_equal(actual,expected) and np.array_equal(vertical,expected),
          observed=actual.tolist(),expected=expected,
          oracle='Hand calculation: even output=(left+6*center+right)/8; odd=(center+right)/2.')
impulse=np.array([[0,0,0,8]],np.float32)
impulse_output=pyr_up(impulse,(1,7))[0]
check('pyramid_odd_terminal_impulse_halo',np.array_equal(impulse_output,[0,0,0,0,1,4,6]),
      observed=impulse_output.tolist(),regression='Always replicating right edge incorrectly returns 7 at last sample.')
fixtures.update(pyramid_edge_input=edge,pyramid_terminal_impulse=impulse)

# Physical Z sampling test intentionally uses unequal gaps and a minimum-boundary
# pixel after an upper pixel. Reference clamps the boundary instead of reproducing
# the native uninitialized/carried alpha path (see gap E07).
sampling=np.stack([np.full((2,3),v,np.float32) for v in [10,50,90]])
sz=np.array([.5,2.,7.],np.float32)
sd=np.array([[7.,.5,4.5],[1.25,2.,20.]],np.float32)
sampled=sample_z(sampling,sz,sd)
desired=np.array([[90,10,70],[30,50,90]],np.float32)
check('fine_efi_physical_z_and_boundaries',np.max(abs(sampled-desired))<1e-5,
      result=sampled.tolist(),note='Explicit boundary clamp; native alpha carry is not reproduced.')
fixtures.update(sample_images=sampling,sample_z=sz,sample_depth=sd)

# End-to-end microscopy-like scene with two plane patches and known PSF blur.
h,w=96,128
texture=(rng.uniform(0,1,(h,w))*.7 + .3*np.sin(np.arange(w)[None,:]*.8))
texture=cv2.GaussianBlur(texture.astype(np.float32),(3,3),.55)
texture=(30+190*(texture-texture.min())/(texture.max()-texture.min())).astype(np.float32)
true=np.where(np.arange(w)[None,:]<w//2,3.2,7.1).repeat(h,0).astype(np.float32)
zs=np.arange(9,dtype=np.float32)*1.3+.4
frames=[]
for zz in zs:
    left=cv2.GaussianBlur(texture,(0,0),.35+.9*abs(float(zz)-3.2))
    right=cv2.GaussianBlur(texture,(0,0),.35+.9*abs(float(zz)-7.1))
    frame=np.where(true<5,left,right)
    frames.append(np.clip(frame+rng.normal(0,.12,frame.shape),0,255).astype(np.float32))
stack=np.stack(frames)
profile=Profile()
t0=time.monotonic()
recon=reconstruct(stack,zs,profile)
runtime=time.monotonic()-t0
valid=np.zeros((h,w),bool)
valid[20:-20,16:48]=True
valid[20:-20,80:-16]=True
raw_mae=float(np.mean(np.abs(recon['raw_height'][valid]-true[valid])))
post_mae=float(np.mean(np.abs(recon['height'][valid]-true[valid])))
middle=stack[len(stack)//2]
efi_err=float(np.mean(np.abs(recon['efi'][valid]-texture[valid])))
single_err=float(np.mean(np.abs(middle[valid]-texture[valid])))
check('end_to_end_two_plane_synthetic_scene',raw_mae<.8 and post_mae<1.0 and efi_err<single_err,
      raw_height_mae=raw_mae,post_height_mae=post_mae,efi_pixel_mae=efi_err,
      center_slice_pixel_mae=single_err,elapsed_seconds=runtime,
      note='Synthetic scene only; thresholds are test acceptance, not DSX measurement specifications.')
check('postprocessing_is_not_an_accuracy_guarantee',post_mae>raw_mae,
      raw_height_mae=raw_mae,post_height_mae=post_mae,
      counterexample='With this finite noisy scene, smoothing worsens interior height error despite producing a smoother surface.')
fixtures.update(scene_stack=stack,scene_z=zs,scene_true_height=true,scene_texture=texture,
                scene_test_mask=valid)
np.savez_compressed(OUT/'scene-result.npz',efi=recon['efi'],height=recon['height'],raw_height=recon['raw_height'],
                    height_response=recon['height_response'],fast_response=recon['fast_response'])
for name,arr in [('texture',texture),('middle-slice',middle),('efi',recon['efi'])]:
    cv2.imwrite(str(OUT/(name+'.png')),np.clip(arr,0,255).astype(np.uint8))
for name,arr in [('true-height',true),('raw-height',recon['raw_height']),('filtered-height',recon['height'])]:
    palette=cv2.applyColorMap(np.clip((arr-zs[0])/(zs[-1]-zs[0])*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
    cv2.imwrite(str(OUT/(name+'.png')),palette)

# Direction normalization and arbitrary-order refusal are explicit policies.
reversed_result=reconstruct(stack[::-1],zs[::-1],profile)
diff=float(np.max(np.abs(reversed_result['height']-recon['height'])))
check('reverse_z_same_unique_data',diff<1e-5,max_height_difference=diff)
refused=False
try: normalize_stack(stack[[0,2,1]],zs[[0,2,1]])
except ValueError:refused=True
check('nonmonotonic_z_requires_opt_in',refused)
duplicate_refused=False
try: normalize_stack(stack[[0,0,1]],zs[[0,0,1]])
except ValueError:duplicate_refused=True
check('duplicate_z_requires_explicit_policy',duplicate_refused)
chosen,unique,ordinal=normalize_stack(stack[[0,0,1]],zs[[0,0,1]],duplicate_policy='first')
check('explicit_duplicate_first_policy_retains_identity',len(chosen)==2 and ordinal.tolist()==[0,2])

# Short stacks take the discrete path. Assembly shows Q is assigned per band,
# so it is intentionally not the normal full-stack sum of quality responses.
short_profile=dataclasses.replace(profile,wm_radius_3d=0,wm_radius_efi=0,
                                  self_radius_3d=0,self_radius_efi=0)
short_result=reconstruct(stack[:3],zs[:3],short_profile)
last_res=np.stack([cv2.GaussianBlur(focus_bands(im,4,3)[3],(11,11),0,
                         borderType=cv2.BORDER_REPLICATE) for im in stack[:3]])
check('short_stack_quality_overwrites_last_active_band',
      np.max(abs(short_result['height_response']-.25*last_res.max(0)))<1e-5,
      note='Static assembly contract; no vendor execution oracle.')
single=np.arange(32*40,dtype=np.uint16).reshape(1,32,40)*40
one=reconstruct(single,[2.375])
check('uint16_single_plane_exact_intensity_and_fractional_z',
      np.array_equal(one['efi'],single[0]) and np.all(one['height']==np.float32(2.375)))
flat_result=reconstruct(np.full((7,32,40),80,np.float32),np.arange(7,dtype=np.float32)+.25)
check('full_pipeline_flat_stack_remains_explicitly_unsupported',
      not flat_result['masks']['3d']['supported'].any()
      and np.isfinite(flat_result['efi']).all(),
      numeric_height=float(flat_result['height'][0,0]),
      warning='Output numbers exist even though no measured surface is supported.')
fixtures.update(single_uint16=single,single_z=np.array([2.375],np.float32),
                short_stack=stack[:3],short_z=zs[:3],
                flat_stack=np.full((7,32,40),80,np.float32),flat_z=np.arange(7,dtype=np.float32)+.25)
rgb_stack=np.stack([stack,np.roll(stack,3,axis=2),255-stack],axis=-1)
rgb_result=reconstruct(rgb_stack,zs,profile)
check('bgr_full_pipeline_dimensions_and_finite_result',
      rgb_result['efi'].shape==(h,w,3) and np.isfinite(rgb_result['efi']).all()
      and np.isfinite(rgb_result['height']).all(),
      note='Exercises color guidance and output channels; no vendor equivalence claim.')
fixtures['bgr_scene_stack']=rgb_stack
uint16_scene=np.rint(stack*257).astype(np.uint16)
for kind,multi16 in [('gray',uint16_scene),('collinear_bgr',np.repeat(uint16_scene[...,None],3,axis=-1))]:
    multi16_result=reconstruct(multi16,zs,profile)
    check('uint16_multilayer_full_pipeline_'+kind,
          multi16_result['efi'].shape==multi16.shape[1:] and np.isfinite(multi16_result['efi']).all()
          and np.isfinite(multi16_result['height']).all(),
          height_range=[float(multi16_result['height'].min()),float(multi16_result['height'].max())],
          note='Full pipeline robustness only; E11 now evidences native uint16-to-float32 raw intensity preservation; full pipeline numeric parity remains unverified.')
fixtures['scene_uint16']=uint16_scene

# SGmv's wrapper is tested separately from the uncalled proprietary MV estimators.
base=np.zeros((48,64),np.float32)
base[10:20,15:27]=200
shifted=cv2.warpAffine(base,np.float32([[1,0,2],[0,1,-1]]),(64,48))
stabilized,geometry=stabilize_given_vectors(np.stack([base,shifted,shifted]),
            [[0,0],[2,-1],[99,99]],[1,1,0],4,3,.5)
se=float(np.max(np.abs(stabilized[0]-stabilized[1])))
check('stack_motion_gate_accumulate_crop',se==0 and geometry['trajectory_xy'][2]==[2.,-1.],
      maximum_aligned_error=se,geometry=geometry)
_,geo=stabilize_given_vectors(np.stack([base]),[[0,0]],[1],4,3,.5,resize_back=True)
check('crop_resize_records_pixel_mapping',geo['output_scale_to_input_xy']==[56/64,42/48],geometry=geo)

fixture_path=OUT/'fixtures.npz'
np.savez_compressed(fixture_path,**fixtures)
report={'scope':'independent reconstructed reference; Olympus binaries never executed',
        'not_claimed':['bit-exact equivalence','real optical accuracy','DSX runtime/driver compatibility','CUDA equivalence'],
        'environment':{'python':platform.python_version(),'platform':platform.platform(),
                       'numpy':np.__version__,'opencv':cv2.__version__,'scipy':scipy.__version__},
        'seed':20261023,'profile':dataclasses.asdict(profile),
        'source_sha256':{n:sha(ROOT/n) for n in ['reference_efi.py','validate_reference.py']},
        'fixture_sha256':sha(fixture_path),'results':results,
        'passed':sum(x['passed'] for x in results),'total':len(results),
        'elapsed_seconds':time.monotonic()-start_time}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'passed':report['passed'],'total':report['total'],
                  'failed':[x for x in results if not x['passed']],
                  'scene':[x for x in results if x['name'].startswith('end_to_end')]},ensure_ascii=False,indent=2))
raise SystemExit(0 if report['passed']==report['total'] else 1)
