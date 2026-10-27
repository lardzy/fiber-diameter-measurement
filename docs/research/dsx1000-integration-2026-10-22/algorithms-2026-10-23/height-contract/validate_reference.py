from pathlib import Path
import hashlib,json
import numpy as np
from reference import lens_cutoff,quantize_sg_depth,map_nominal_to_absolute

root=Path(__file__).parent
checks=[]
def check(name,ok,detail):checks.append(dict(name=name,passed=bool(ok),detail=detail))
check('optical_not_display_magnification',abs(lens_cutoff(.3,20,1)-54.54545454545455)<1e-12,'NA .3, objective20, optical1, wavelength550 gives cutoff54.54545')
check('optical_zoom_changes_weight_driver',abs(lens_cutoff(.3,20,2)*2-lens_cutoff(.3,20,1))<1e-12,'Cutoff varies inversely with optical zoom at fixed NA')
d=np.linspace(2000,10000,201,dtype=np.float32)
r=quantize_sg_depth(d,2000,5)
check('integer_bins_per_layer',r['bins_per_layer']==13107,'65535//5, not a separately rounded float step')
check('buffer_type_boundary',r['codes_int32'].dtype==np.int32 and r['codes_uint16'].dtype==np.uint16,'32-bit elements contain 16-bit bounded code values')
decoded=r['codes_int32']*r['depth_per_code']
err=float(np.max(np.abs(decoded-d)))
check('quantization_error',err<r['depth_per_code']*1.01,dict(max_error=err,step=r['depth_per_code']))
check('first_nominal_height_is_nonzero',abs(int(r['codes_int32'][0])-13107)<=1,'Fine acquisition starts nominal Z at pitch*1, not at zero')
check('wire_and_buffer_units_differ',abs(r['wire_resolution']/r['depth_per_code']-1000)<.001,'Outgoing common-info scalar approximately1000 times depth-unit/code')
cl=quantize_sg_depth(np.array([-1,1e8],np.float32),2000,5)
check('saturation_explicit',list(cl['codes_int32'])==[0,65535] and cl['clipped'].all(),'Clamping loses out-of-range height information')
check('absolute_origin_not_inferred',np.allclose(map_nominal_to_absolute([2000,4000],2000,120,-1,.001),[120,118]),'Explicit first absolute position, sign and nm-to-um scale supplied by caller')
for n in (0,65536):
 try:quantize_sg_depth(d,2000,n)
 except ValueError:ok=True
 else:ok=False
 check('guard_count_'+str(n),ok,'Independent guard, not claimed as original behavior')
for name,fn in [('nonfinite_optics',lambda:lens_cutoff(float('nan'),20,1)),('nonfinite_origin',lambda:map_nominal_to_absolute([1],1,float('nan'),1,1))]:
 try:fn()
 except ValueError:ok=True
 else:ok=False
 check(name,ok,'Independent input guard prevents plausible-looking NaN metadata')
result={'status':'passed' if all(c['passed'] for c in checks) else 'failed','checks':checks,'scope':'Scalar CIL formulas and proposed explicit coordinate contract; no device or original runtime','numpy':np.__version__,'source_sha256':hashlib.sha256((root/'reference.py').read_bytes()).hexdigest()}
(root/'reference-validation.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(root/'reference-fixtures.npz',nominal_depth=d,codes_int32=r['codes_int32'],decoded=decoded)
print(json.dumps({'status':result['status'],'checks':len(checks),'failures':[c for c in checks if not c['passed']]}))
raise SystemExit(result['status']!='passed')
