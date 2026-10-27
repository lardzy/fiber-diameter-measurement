"""Deterministic, independent geometry checks; not vendor-output validation."""
from pathlib import Path
import json
import hashlib
import platform
import numpy as np
from reference import (fit_profile, flatten_profile, sample_point_mean, plane_from_three,
    three_point_flatten, offset_three_point_encoded, encode_rescaled_height,
    sphere_from_four, subtract_sphere, width_profile, manual_quadratic,
    flatten_manual, encode_projected_height)

ROOT=Path(__file__).parent
checks=[]
def record(name, value, limit, description):
    ok=bool(np.isfinite(value) and value <= limit)
    checks.append(dict(name=name,passed=ok,value=float(value),limit=limit,description=description))

y,x=np.indices((43,61),dtype=float)
plane=100+2*x+3*y
h,co=flatten_profile(plane,plane[7],axis='x'); h,co=flatten_profile(h,h[:,9],axis='y')
record('two_profile_linear_flatten',np.ptp(h),1e-10,'Remove independently supplied x/y linear trends')
curve=500+.01*x*x-.3*x+.05*y*y
h,_=flatten_profile(curve,curve[0],degree=2);h,_=flatten_profile(h,h[:,0],axis='y',degree=2)
record('two_profile_quadratic_flatten',np.ptp(h),1e-8,'ALGLIB-equivalent least-squares mathematical core, different solver')
step=np.where((x>28)&(y>18),7.,0.)
h,_=flatten_profile(curve+step,curve[0],degree=2);h,_=flatten_profile(h,h[:,0],axis='y',degree=2)
record('local_step_retained',abs((h[30,40]-h[10,40])-7),1e-8,'Background-only reference profiles preserve local step')
pts=np.array([[3,4],[51,5],[7,35]])
h,p,deg=three_point_flatten(plane,pts)
record('three_point_plane_flatten',np.ptp(h),1e-9,'Cross-product slopes and centered corrections')
record('clamped_corner_mean',abs(sample_point_mean(plane,[0,0],3)-plane[np.ix_([0,0,1],[0,0,1])].mean()),1e-12,'Clamps out-of-range coordinates as observed')
try:three_point_flatten(plane,[[0,0],[1,1],[2,2]])
except ValueError:blocked=True
else:blocked=False
record('degenerate_pick_guard',0 if blocked else 1,0,'Reference guard, not claimed as vendor behavior')
_,_,deg=three_point_flatten(plane,[[0,0],[1,1],[2,2]],vendor_degenerate_fallback=True)
record('vendor_C_zero_branch',0 if deg else 1,0,'Observed denominator fallback selectable for research')
encoded=encode_rescaled_height(3000*x-700*y-50)
roundtrip=encoded.decode()
record('height_affine_roundtrip',float(np.max(np.abs(roundtrip-(3000*x-700*y-50)))),.500001/encoded.scale,'Bounded by half an output quantization step')
_,offset=offset_three_point_encoded(np.array([[-100,0],[70000,80000.]]))
record('wide_range_offset_branch',abs(offset-((80100-65535)*.5-100)),1e-12,'Retains the unusual original add-offset expression')
center=np.array([5.,8.,-30.]);radius=100.
xy=np.array([[-20.,-30.],[50.,-20.],[-30.,40.],[60.,35.]])
z=center[2]+np.sqrt(radius**2-((xy-center[:2])**2).sum(axis=1))
cc,rr,concave=sphere_from_four(np.c_[xy,z])
record('sphere_center',np.linalg.norm(cc-center),1e-9,'Non-coplanar four-point sphere solve')
record('sphere_radius',abs(rr-radius),1e-9,'Radius from point-to-center distance')
sphere=center[2]+np.sqrt(radius**2-(x-center[0])**2-(y-center[1])**2)
flat,valid=subtract_sphere(sphere+step,x,y,cc,rr,concave)
record('sphere_preserves_local_step',np.max(np.abs(flat-step)),1e-8,'Subtract fitted curved baseline; no GUI clipping or code conversion')
_,invalid=subtract_sphere(np.ones((2,2)),np.array([[0,2],[0,2]]),np.array([[0,0],[2,2]]),[0,0,0],1)
record('sphere_invalid_domain',abs(int(invalid.sum())-1),0,'Explicit validity result of the reference API')
band=np.arange(24,dtype=np.uint16).reshape(4,6)
profile,bounds=width_profile(band,0,3)
record('profile_edge_band_shift',np.max(np.abs(profile-band[:3].mean(axis=0))) if bounds==(0,2) else 1,0,'Top edge start clamps then end derives from start, no replicated pixels')
profile,bounds=width_profile(band,3,3)
record('profile_bottom_band_shrink',np.max(np.abs(profile-np.trunc(band[2:].mean(axis=0)))) if bounds==(2,3) else 1,0,'Bottom band shrinks at last row and u16 mean truncates')
profile,_=width_profile(band,2,0,axis='y')
record('single_column_profile',np.max(np.abs(profile-band[:,2])),0,'Width zero axis-aligned path')
cx=np.array([0.,20.,60.]); cy=5+.2*cx+.01*cx*cx
manual,adjusted=manual_quadratic(np.c_[cx,cy],61)
record('manual_curve_is_quadratic',np.max(np.abs(manual-(5+.2*np.arange(61)+.01*np.arange(61)**2))),1e-10,'Three point interpolation, not natural cubic spline')
_,adjusted=manual_quadratic([[0,1],[.2,2],[.3,4]],7)
record('manual_close_x_nudging',np.max(np.abs(adjusted-[0,1,2])),0,'Observed sequential x-coordinate adjustment')
flat_manual,_=flatten_manual(plane,[[0,0],[21,63],[42,126]],axis='y')
record('manual_vertical_reversal',abs((flat_manual[-1,0]-flat_manual[0,0])-252),1e-10,'UI vertical controls reverse before correction; direct image-row slope would have wrong sign')
crop_result,crop_meta=encode_projected_height(band,12,4)
record('top_left_projected_crop',abs(crop_meta['crop_xywh'][2]-3)+abs(crop_meta['crop_xywh'][3]-4),0,'Crop width=floor(W^2/projectedW), anchored at zero')
record('projected_encode_resize_shape',sum(abs(a-b) for a,b in zip(crop_result.values.shape,band.shape)),0,'Converted crop resized back to original matrix dimensions')
result=dict(scope='Independent synthetic mathematical tests; no DSX binaries or hardware executed',
    numpy=np.__version__,python=platform.python_version(),checks=checks,
    status='passed' if all(v['passed'] for v in checks) else 'failed',
    source_sha256=hashlib.sha256((ROOT/'reference.py').read_bytes()).hexdigest())
(ROOT/'reference-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
np.savez_compressed(ROOT/'reference-fixtures.npz',plane=plane,curve=curve,step=step,sphere=sphere,
                    corrected_sphere=flat,encoded=encoded.values,decoded=roundtrip,
                    band=band,manual_quadratic=manual,projected_codes=crop_result.values)
print(json.dumps(dict(status=result['status'],checks=len(checks),failures=[v for v in checks if not v['passed']]),ensure_ascii=False))
raise SystemExit(result['status']!='passed')
