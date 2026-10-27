from pathlib import Path
import json,hashlib
import numpy as np
import cv2
from reference import DisplayParameters, display_dimensions, auto_depth_factor, normalize_dsx_height, grid_mesh, dsx_display_positions, prepare_display, write_ply
root=Path(__file__).parent
checks=[]
def check(name,error,limit,meaning):
 checks.append(dict(name=name,passed=bool(np.isfinite(error) and error<=limit),error=float(error),limit=limit,meaning=meaning))
check('landscape_1024_resampling',sum(abs(a-b) for a,b in zip(display_dimensions(1920,1200),(1024,640))),0,'Observed normalization, not original image size')
check('small_images_upsampled',sum(abs(a-b) for a,b in zip(display_dimensions(96,64),(1024,682))),0,'Code is fixed long edge, not a maximum cap')
check('wire_interval_multiple',sum(v%8 for v in display_dimensions(96,64,8,8)),0,'Rounded dimensions before mesh sampling')
y,x=np.indices((13,19),dtype=float)
height=2*x+3*y+50
v,t,n=grid_mesh(height,pitch_x=2,pitch_y=3)
expected=np.array([-1.,-1.,1.]);expected/=np.linalg.norm(expected)
check('analytic_plane_normals',np.max(np.abs(n-expected)),1e-12,'Independent normal direction from physical derivatives')
check('triangle_count',abs(len(t)-2*12*18),0,'Two triangles per grid quad')
check('triangle_index_range',max(0,int(t.max())-len(v)+1),0,'No out-of-range vertex indices')
valid=np.ones(height.shape,bool);valid[5,5]=False
_,ta,_=grid_mesh(height,valid=valid,drop_invalid=False)
_,tb,_=grid_mesh(height,valid=valid,drop_invalid=True)
check('mask_mode_distinction',0 if len(tb)<len(ta)==len(t) and np.all(valid.ravel()[tb]) else 1,0,'Display-like full grid and scientific hole-preserving option explicitly differ')
raw,rounded=auto_depth_factor(1920,1200,.5,0,20)
check('auto_exaggeration_formula',abs(raw-12),0,'Display height exaggeration is not measured geometry')
params=DisplayParameters(.5,.1,2,1,10,100,3)
codes=np.array([[0,50],[500,1000]],dtype=np.uint16)
z=normalize_dsx_height(codes,(1024,1024),params)
check('height_scale_and_clamp',np.max(np.abs(z-np.array([[0,0],[36,36]],dtype=np.float32))),1e-5,'Calibration and display clipping both included')
positions=dsx_display_positions(z,params)
check('display_axis_mapping',np.max(np.abs(positions[-1]-np.array([.2,.2,-108]))),1e-5,'Row->X, column->Y, negative displayed Z')
rev=dsx_display_positions(z,DisplayParameters(.5,.1,2,1,10,100,3,True))
check('reverse_only_display',np.max(np.abs(rev[:,2]+positions[:,2])),1e-5,'Reverse changes displayed elevation, not source heights')
_,_,dense=prepare_display(np.full((64,96),500,np.uint16),params,8)
check('mesh_sample_count',abs(len(dense)-(1024//8)*(680//8)),0,'Buffer dimensions and interval sampling')
sy,sx=np.indices((5,5),dtype=float);sr=1.5**2-(sx-2)**2-(sy-2)**2
sv=sr>=0;sh=np.where(sv,np.sqrt(np.maximum(sr,0)),np.nan)
vv,tt,nn=grid_mesh(sh,valid=sv,drop_invalid=True)
check('sphere_nan_mask_to_mesh',0 if len(tt)==8 and np.isfinite(vv).all() and np.all(sv.ravel()[tt]) else 1,0,'Surface sphere output can flow into scientific hole mesh; invalid vertices are unreferenced placeholders')
try:grid_mesh(sh,valid=np.ones_like(sv),drop_invalid=True)
except ValueError:blocked=True
else:blocked=False
check('nonfinite_valid_point_rejected',0 if blocked else 1,0,'Mask cannot declare a nonfinite height valid')
try:display_dimensions(96,64,2048,2048)
except ValueError:blocked=True
else:blocked=False
check('oversized_interval_rejected',0 if blocked else 1,0,'Reject zero-sized aligned image before OpenCV')
write_ply(root/'reference-plane.ply',v,t,n)
result=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',scope='Independent synthetic renderer math; no DSX or Direct3D runtime',numpy=np.__version__,opencv=cv2.__version__,checks=checks,source_sha256=hashlib.sha256((root/'reference.py').read_bytes()).hexdigest())
(root/'reference-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],checks=len(checks),failures=[c for c in checks if not c['passed']])))
raise SystemExit(result['status']!='passed')
