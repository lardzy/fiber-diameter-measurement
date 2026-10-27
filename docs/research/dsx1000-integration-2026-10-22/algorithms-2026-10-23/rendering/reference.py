"""Independent reference of observed height-field rendering data operations.

No Direct3D/SlimDX/vendor modules are loaded. Topology/normals use a portable
triangular-grid implementation; exact vendor boundary-face ordering remains a
separate parity item. All display normalization is distinct from physical mesh.
"""
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass(frozen=True)
class DisplayParameters:
    x_pitch: float
    z_pitch: float
    maker_z: float = 1.
    user_z: float = 1.
    minimum: float = 0.
    maximum: float = 65535.
    height_factor: float = 1.
    reverse: bool = False


def display_dimensions(width, height, interval_x=1, interval_y=1):
    """ImageControl 269/204: scale long edge to 1024 (including upsampling)."""
    if min(width,height,interval_x,interval_y)<=0:raise ValueError('positive dimensions and intervals required')
    w=h=1024
    if width>height:h=int(1024*height/width)
    elif height>width:w=int(1024*width/height)
    w=max(w,10);h=max(h,10)
    result=w-w%interval_x,h-h%interval_y
    if min(result)<=0:raise ValueError('grid interval exceeds normalized image dimension')
    return result


def auto_depth_factor(width,height,pitch,minimum,maximum,divisor=4.):
    """ImageControl 1f3; excludes the original Int16 overflow domain."""
    if not 0<max(width,height)<32768 or pitch<=0 or maximum<=minimum or divisor<=0:
        raise ValueError('unsupported dimensions or zero/negative physical range')
    raw=max(width,height)*pitch/(divisor*(maximum-minimum))
    return raw,float(np.float32(round(raw,1)))


def normalize_dsx_height(encoded, original_size, params:DisplayParameters):
    """ImageControl 1fb: input is already resized u16, dimensions (w,h)."""
    h=np.asarray(encoded)
    if h.ndim!=2 or h.dtype!=np.uint16:raise ValueError('2D uint16 height required')
    w0,h0=original_size
    if min(w0,h0,params.x_pitch,params.z_pitch)<=0 or params.maximum<=params.minimum:
        raise ValueError('positive geometry and increasing display limits required')
    ratio=1024/max(w0,h0)
    scale=.20000000298023224*ratio/params.x_pitch
    z= h.astype(np.float64)*params.z_pitch*params.maker_z*params.user_z
    z=np.clip((z-params.minimum)*scale,0,(params.maximum-params.minimum)*scale)
    return z.astype(np.float32)


def grid_mesh(height, *, pitch_x=1.,pitch_y=1.,valid=None,drop_invalid=False):
    """Portable physical height-field mesh; this API keeps x=column,y=row,z=height.

    Original DSX display positions use (row*.2,col*.2,-height*factor). This
    separate physical representation deliberately keeps units interpretable.
    Faces are equally weighted unit normals, then summed per vertex.
    """
    z=np.asarray(height,dtype=np.float64)
    if z.ndim!=2 or min(z.shape)<2 or min(pitch_x,pitch_y)<=0:
        raise ValueError('2D height grid and positive pitches required')
    if valid is not None:
        valid=np.asarray(valid,dtype=bool)
        if valid.shape!=z.shape:raise ValueError('valid mask shape mismatch')
    if drop_invalid and valid is not None:
        if not np.isfinite(z[valid]).all():raise ValueError('valid heights must be finite')
        # Placeholder only for unreferenced vertices; no invalid face survives.
        # Never mutate raw data or label these zeros as measured heights.
        z=np.where(valid,z,0.)
    elif not np.isfinite(z).all():
        raise ValueError('nonfinite height requires valid mask and drop_invalid=True')
    yy,xx=np.indices(z.shape,dtype=np.float64)
    vertices=np.stack((xx*pitch_x,yy*pitch_y,z),axis=-1).reshape(-1,3)
    ids=np.arange(z.size).reshape(z.shape)
    a=ids[:-1,:-1].ravel();b=ids[:-1,1:].ravel();c=ids[1:,:-1].ravel();d=ids[1:,1:].ravel()
    triangles=np.stack((np.stack((a,b,d),axis=1),np.stack((a,d,c),axis=1)),axis=1).reshape(-1,3)
    if valid is not None:
        if drop_invalid:triangles=triangles[np.all(valid.ravel()[triangles],axis=1)]
    edges1=vertices[triangles[:,1]]-vertices[triangles[:,0]]
    edges2=vertices[triangles[:,2]]-vertices[triangles[:,0]]
    normal=np.cross(edges1,edges2);length=np.linalg.norm(normal,axis=1)
    normal/=np.maximum(length[:,None],np.finfo(float).eps)
    normals=np.zeros_like(vertices)
    for j in range(3):np.add.at(normals,triangles[:,j],normal)
    norm=np.linalg.norm(normals,axis=1)
    normals/=np.maximum(norm[:,None],np.finfo(float).eps)
    return vertices,triangles,normals


def dsx_display_positions(normalized,params:DisplayParameters,stride=1):
    z=np.asarray(normalized,dtype=np.float32)
    yy,xx=np.indices(z.shape,dtype=np.float32)
    factor=params.height_factor*(-1 if params.reverse else 1)
    return np.stack((yy*.20000000298023224,xx*.20000000298023224,-z*factor),axis=-1)[::stride,::stride].reshape(-1,3)


def prepare_display(encoded,params,interval=1):
    """Color/mask resampling is a separate caller operation, as in the vendor."""
    h=np.asarray(encoded)
    if h.dtype!=np.uint16 or h.ndim!=2:raise ValueError('encoded uint16 required')
    size=display_dimensions(h.shape[1],h.shape[0],interval,interval)
    resized=cv2.resize(h,size,interpolation=cv2.INTER_LINEAR)
    normalized=normalize_dsx_height(resized,(h.shape[1],h.shape[0]),params)
    return resized,normalized,dsx_display_positions(normalized,params,interval)


def write_ply(path,vertices,triangles,normals):
    with open(path,'w',encoding='ascii',newline='\n') as stream:
        stream.write(f'ply\nformat ascii 1.0\ncomment independent DSX research reference physical mesh\nelement vertex {len(vertices)}\n')
        stream.write('property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n')
        stream.write(f'element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n')
        for v,n in zip(vertices,normals):stream.write(' '.join(f'{a:.9g}' for a in (*v,*n))+'\n')
        for a,b,c in triangles:stream.write(f'3 {a} {b} {c}\n')
