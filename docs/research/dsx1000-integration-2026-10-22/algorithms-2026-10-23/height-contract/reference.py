"""Independent numerical bridge contract; no vendor code execution.

Units are named at each boundary. pitch is the acquisition Zpitch numeric unit;
this API does not infer an absolute stage origin from a reconstruction height.
"""
import numpy as np


def lens_cutoff(na, objective_magnification, optical_zoom, wavelength=550.):
    values=np.asarray([na,objective_magnification,optical_zoom,wavelength],dtype=float)
    if not np.isfinite(values).all() or min(values)<=0:
        raise ValueError('positive optical parameters required')
    return 2*na/(objective_magnification*optical_zoom)/(wavelength*1e-6)


def quantize_sg_depth(depth, pitch, building_count):
    """GetExtendZResolution + _GetExtendHeightDataSG scalar operations.

    Preserves float32 arithmetic and UInt16 truncation stored in Int32 elements.
    Unsupported count/NaN inputs are rejected here; original does not prove this
    guard. 'wire_resolution' denotes observed outgoing factor, not micrometers.
    """
    if not isinstance(building_count,(int,np.integer)) or not 1<=building_count<=65535:
        raise ValueError('building_count must be within 1..65535')
    if not isinstance(pitch,(int,np.integer)) or not 1<=pitch<=2**32-1:
        raise ValueError('pitch must be a positive UInt32 acquisition value')
    depth=np.asarray(depth,dtype=np.float32)
    if not np.isfinite(depth).all():raise ValueError('finite depth required')
    bins=65535//building_count
    helper=np.float32(np.float32(np.float32(pitch)/np.float32(1000))/np.float32(bins))
    step=np.float32(helper*np.float32(1000))
    inv=np.float32(np.float32(1)/step)
    scaled=np.clip(depth*inv,0,65535)
    codes=scaled.astype(np.uint16).astype(np.int32)
    return {'codes_int32':codes,'codes_uint16':codes.astype(np.uint16),
            'depth_per_code':float(step),'wire_resolution':float(helper)*1000*1000,
            'bins_per_layer':bins,'helper_resolution':float(helper),
            'clipped':(depth<0)|(depth*inv>65535)}


def map_nominal_to_absolute(depth, nominal_first, absolute_first, direction, scale_to_absolute):
    """Our explicit origin mapping; requires independently known first position.

    This is a proposed reconstruction-data contract, not a found vendor method.
    """
    if not np.isfinite([nominal_first,absolute_first,scale_to_absolute]).all() or direction not in (-1,1) or scale_to_absolute<=0:
        raise ValueError('finite origins, explicit sign and positive scale required')
    a=np.asarray(depth,dtype=float)
    if not np.isfinite(a).all():raise ValueError('finite input required')
    return absolute_first+direction*(a-nominal_first)*scale_to_absolute
