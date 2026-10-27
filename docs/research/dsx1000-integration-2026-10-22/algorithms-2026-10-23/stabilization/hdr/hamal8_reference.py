"""Research reconstruction of Hamal's 8-bit, AsymMode=1, BlendMode=1 core.

No vendor library is loaded. Input is 3-channel native component order (C0,C1,C2).
The observed matrix weights C0 like B, C1 like G, C2 like R; callers must verify
their storage order. This omits external Spica denoising and 14-bit processing.
It is static-derived, not validated against vendor output.
The native range check is wider than the reconstructed thumbnail's fully
covered geometry. Uncovered last-row/column cases are explicitly unsupported;
no invented padding or native uninitialized-memory behavior is reproduced.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate2d

TABLES = json.loads((Path(__file__).parent/'native-tables.json').read_text())
def table(addr): return np.array(TABLES[addr]['values'], dtype=np.int64)
KERNEL = table('0x180010600').reshape(33,33)
ASYM_INPUT = table('0x180011bc0').reshape(11,15)
ASYM_SLOPE = table('0x180011680').reshape(11,15)
ASYM_OUTPUT = table('0x180011920').reshape(11,15)
RECIPROCAL = table('0x180014800')[:129]


@dataclass(frozen=True)
class Settings:
    asymmetry: int = 50
    saturation: int = 64
    strength: int = 20
    noise_slope1: tuple = (25,20,15,12,8,5,3,1)
    noise_slope2: tuple = (163,204,265,356,505,772,1326,2811)
    noise_bins: tuple = (26,33,42,57,81,123,212,450)


def validate_settings(settings):
    for name, maximum in [('noise_slope1',32767),('noise_slope2',8191),('noise_bins',1023)]:
        values=np.asarray(getattr(settings,name))
        if values.shape!=(8,) or values.dtype.kind not in 'iu':
            raise ValueError(f'{name} must contain exactly 8 integer values')
        if np.any(values<0) or np.any(values>maximum):
            raise ValueError(f'{name} values must be in 0..{maximum}')
    if settings.asymmetry not in range(0,101,10):
        raise ValueError('Asymmetry must be 0..100 step10')
    if not 0<=settings.strength<=255 or not 0<=settings.saturation<=255:
        raise ValueError('Native bounds')


def _round10(v):
    return np.where(v<0, -(((-v>>9)+1)>>1), ((v>>9)+1)>>1)


def _reciprocal_product(numerator, denominator, multiplier_shift=0):
    """Observed normalized 13-bit reciprocal table and 6-bit interpolation."""
    den=np.asarray(denominator,dtype=np.int64)
    num=np.asarray(numerator,dtype=np.int64)
    if np.any(den<=0): raise ValueError('Reciprocal denominator must be positive')
    norm=den.copy()
    shifts=np.zeros_like(norm)
    while np.any(norm<8192):
        mask=norm<8192
        norm=np.where(mask,norm*2,norm)
        shifts+=mask.astype(np.int64)
    idx=(norm-8192)>>6
    fraction=norm&63
    lut=RECIPROCAL[idx]*(64-fraction)+RECIPROCAL[idx+1]*fraction
    return (lut*num*(1<<multiplier_shift))>>(38-shifts)


class UnsupportedThumbnailGeometry(ValueError):
    """The recovered mapper does not cover all 80x60 thumbnail cells."""

    def __init__(self, geometry):
        self.geometry = geometry
        super().__init__(
            f"Unsupported research geometry {geometry['width']}x{geometry['height']}: "
            f"native fixed-point mapping visits {geometry['visited_columns']}x"
            f"{geometry['visited_rows']} of 80x60 thumbnail cells; "
            "native uncovered-cell initialization is not established")


def thumbnail_geometry(h, w):
    """Classify the narrower research domain without allocating an image.

    Within the native input bounds, steps are positive and <8192, so destination
    indices begin at zero and cannot skip a cell. Endpoint+1 is the visited count.
    """
    if not 121<=w<=10000 or not 91<=h<=10000:
        raise ValueError('Native image bounds: width 121..10000, height 91..10000')
    sx=int(79/(w-2)*8192+.5)
    sy=int(59/(h-2)*8192+.5)
    columns=(((w-1)*sx+(sx>>1))>>13)+1
    rows=(((h-1)*sy+(sy>>1))>>13)+1
    return {'width':w, 'height':h, 'sx':sx, 'sy':sy,
            'visited_columns':columns, 'visited_rows':rows,
            'supported':columns==80 and rows==60}


def thumbnail(src):
    h,w=src.shape[:2]
    geometry=thumbnail_geometry(h,w)
    if not geometry['supported']:
        raise UnsupportedThumbnailGeometry(geometry)
    sx,sy=geometry['sx'],geometry['sy']
    # First source sample that maps into each destination cell, not area resize.
    xi=((np.arange(w)*sx+(sx>>1))>>13)
    yi=((np.arange(h)*sy+(sy>>1))>>13)
    _,x=np.unique(xi,return_index=True)
    _,y=np.unique(yi,return_index=True)
    assert len(x)==80 and len(y)==60, 'Thumbnail classifier disagrees with sample mapping'
    return src[y[:,None],x[None,:]].max(axis=2).astype(np.int64),sx,sy


def asym_map(values, settings):
    i=settings.asymmetry//10
    knots=ASYM_INPUT[i]
    segment=np.clip(np.searchsorted(knots,values,side='right')-1,0,14)
    out=np.minimum(8191,((values-knots[segment])*ASYM_SLOPE[i,segment]+32)>>6)
    return np.minimum(8191,out+ASYM_OUTPUT[i,segment])


def adaptive_knots(mapped_thumb):
    hist=np.bincount((mapped_thumb.ravel()>>10),minlength=8)
    knots=np.arange(1,9,dtype=np.int64)*1024
    # Native top-bin imbalance changes the highest three knots.
    def reduction(a,b,scale):
        if a==0: return 0
        if a>=10*b: return 8*scale
        if a>=5*b: return 7*scale
        if 3*a>=10*b: return 5*scale
        if 2*a>=5*b: return 3*scale
        if a>=2*b: return scale
        return 0
    knots[4]-=reduction(hist[5],hist[4],8)
    knots[5]-=reduction(hist[6],hist[5],16)
    knots[6]-=reduction(hist[7],hist[6],32)
    slopes=np.full(8,512,dtype=np.int64)
    for i in range(4,8):
        slopes[i]=_reciprocal_product(1,knots[i]-knots[i-1],19)
    return hist,knots,slopes


def normalize_asym(mapped,knots,slopes):
    seg=np.clip(np.searchsorted(knots,mapped,side='right'),0,7)
    left=np.where(seg==0,0,knots[np.maximum(0,seg-1)])
    fractional=((mapped-left)*slopes[seg]+256)>>9
    out=seg*1024+np.clip(fractional,0,1023)
    # Separate native low branch caps at 8190, though usual range is below 1024.
    return np.where(seg==0,np.clip(fractional,0,8190),out)


def local_histogram(thumb_normalized):
    labels=thumb_normalized>>10
    padded=np.pad(labels,16,mode='symmetric')
    planes=[]
    for b in range(8):
        counts=correlate2d((padded==b).astype(np.int64),KERNEL,mode='valid')
        planes.append((np.minimum(counts,2001)*131)>>6)
    return np.array(planes,dtype=np.int64)


def condition_histogram(planes,settings):
    tin=np.array([0,1755,3510,5266,7021,8777,10532])
    tsl=np.array([60,43,24,12,5,2,1])
    tout=np.array([0,1655,2845,3514,3838,3984,4048])
    a=np.array(settings.noise_slope1)[:,None,None]
    b=np.array(settings.noise_slope2)[:,None,None]
    c=np.array(settings.noise_bins)[:,None,None]
    t=np.minimum(12288,(a*planes+8)>>4)
    segment=np.clip(np.searchsorted(tin,t,side='right')-1,0,6)
    f=np.minimum(4095,((t-tin[segment])*tsl[segment]+32)>>6)
    f=np.minimum(4095,f+tout[segment])
    f=np.minimum(4095,(b*f+128)>>8)
    remaining=np.maximum(0,4095-f.sum(axis=0))
    return np.minimum(4095,f+np.minimum(4095,(c*remaining+512)>>10))


def blend16_and_cumulative(planes):
    out=np.empty((16,60,80),dtype=np.int64)
    for i in range(8):
        out[2*i]=planes[i]>>1 if i==0 else (3*planes[i]+planes[i-1])>>3
        out[2*i+1]=planes[i]>>1 if i==7 else (3*planes[i]+planes[i+1])>>3
    return np.cumsum(out,axis=0)


def spatial_tone(normalized,planes,sx,sy):
    h,w=normalized.shape
    yf=np.arange(h,dtype=np.int64)*sy
    xf=np.arange(w,dtype=np.int64)*sx
    y0=np.clip(yf>>13,0,59); y1=np.clip((yf>>13)+1,0,59)
    x0=np.clip(xf>>13,0,79); x1=np.clip((xf>>13)+1,0,79)
    fy=(yf&8191)[:,None];fx=(xf&8191)[None,:]
    band=normalized>>9; rem=normalized&511
    def corner(y,x):
        high=planes[band,y[:,None],x[None,:]]
        low=np.where(band==0,0,planes[np.maximum(band-1,0),y[:,None],x[None,:]])
        return np.minimum(4095,low+(((high-low)*rem+256)>>9))
    q00,q10,q01,q11=corner(y0,x0),corner(y0,x1),corner(y1,x0),corner(y1,x1)
    top=q00+(((q10-q00)*fx+4096)>>13)
    bottom=q01+(((q11-q01)*fx+4096)>>13)
    return np.minimum(4095,top+(((bottom-top)*fy+4096)>>13))


def gain_map(brightness,tone,strength):
    ratio=_reciprocal_product(tone,np.maximum(brightness,1),4)
    mixed=((strength*np.minimum(ratio,32767)+32)>>6)+4*(64-strength)
    return np.clip(np.where(brightness==0,4*(64-strength),mixed),0,32767)


def apply_color_gain(src,gain,saturation):
    a=src.astype(np.int64)
    c0,c1,c2=np.moveaxis(a,-1,0)
    y=np.clip(_round10(74*c0+732*c1+218*c2),0,255)
    cb=np.clip(_round10(512*c0-395*c1-117*c2)+128,0,255)
    cr=np.clip(_round10(-47*c0-465*c1+512*c2)+128,0,255)
    yy=np.minimum(255,(y*gain+128)>>8)
    chroma_gain=np.clip(((saturation*gain+32)>>6)+4*(64-saturation),0,32767)
    cb=np.clip((chroma_gain*(cb-128)+128)>>8,-128,127)
    cr=np.clip((chroma_gain*(cr-128)+128)>>8,-128,127)
    c2=np.clip(_round10(1024*yy+1613*cr),0,255)
    c1=np.clip(_round10(1024*yy-191*cb-479*cr),0,255)
    c0=np.clip(_round10(1024*yy+1901*cb),0,255)
    return np.stack((c0,c1,c2),axis=-1).astype(np.uint8)


def enhance(src,settings=Settings(),return_intermediates=False):
    if src.dtype!=np.uint8 or src.ndim!=3 or src.shape[2]!=3:
        raise ValueError('8bit core prototype requires uint8 C0/C1/C2 image')
    h,w=src.shape[:2]
    if not 121<=w<=10000 or not 91<=h<=10000: raise ValueError('Native image bounds')
    validate_settings(settings)
    thumb,sx,sy=thumbnail(src)
    mapped=asym_map(thumb,settings)
    hist,knots,slopes=adaptive_knots(mapped)
    normal=normalize_asym(mapped,knots,slopes)
    raw_hist=local_histogram(normal)
    conditioned=condition_histogram(raw_hist,settings)
    planes=blend16_and_cumulative(conditioned)
    brightness=src.max(axis=2).astype(np.int64)
    full=normalize_asym(asym_map(brightness,settings),knots,slopes)
    tone=spatial_tone(full,planes,sx,sy)
    gain=gain_map(brightness,tone,settings.strength)
    output=apply_color_gain(src,gain,settings.saturation)
    if not return_intermediates:return output
    return output,{"thumbnail":thumb,"thumbnail_geometry":thumbnail_geometry(h,w),
                   "mapped_thumbnail":mapped,"global_histogram":hist,
                   "adaptive_knots":knots,"adaptive_slopes":slopes,"local_histogram":raw_hist,
                   "conditioned_histogram":conditioned,"cumulative_planes":planes,"tone":tone,
                   "gain":gain,"normalized_full":full}
