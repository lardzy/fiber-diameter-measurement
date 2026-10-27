"""Independent numerical reference for the statically observed DSX surface core.

No vendor DLL is loaded. Geometry uses explicit units; GUI, image-entity storage,
OpenCV 2.2 rounding, and native resize are not emulated here. See implementation-spec.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


def _height(value):
    h = np.asarray(value, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 2 or not np.isfinite(h).all():
        raise ValueError('finite 2D height with both dimensions >= 2 required')
    return h


def fit_profile(profile, degree=1):
    """S1 8b/ad/97: ordinary line/quadratic profile least squares.

    The quadratic solver is numpy.linalg.lstsq, not ALGLIB. Coefficients in
    increasing powers; returned fit is a mathematical reference, not bit-exact.
    """
    h = np.asarray(profile, dtype=np.float64)
    if h.ndim != 1 or len(h) <= degree or degree not in (1, 2) or not np.isfinite(h).all():
        raise ValueError('finite profile, degree 1 or 2, enough samples required')
    x = np.arange(len(h), dtype=np.float64)
    if degree == 1:
        n = len(x); sx = x.sum(); sxx = (x*x).sum(); sh = h.sum(); sxh = (x*h).sum()
        denom = n*sxx-sx*sx
        coefficients = np.array([(sh*sxx-sx*sxh)/denom, (n*sxh-sx*sh)/denom])
    else:
        coefficients = np.linalg.lstsq(np.vander(x, degree+1, increasing=True), h, rcond=None)[0]
    return np.polynomial.polynomial.polyval(x, coefficients), coefficients


def flatten_profile(height, profile, axis='x', degree=1):
    """S1 87/97 + S2 e4/e6: h + mean(fit) - fit, broadcast per row/column."""
    h = _height(height)
    if axis not in ('x', 'y'): raise ValueError('axis must be x or y')
    expected = h.shape[1 if axis == 'x' else 0]
    if len(profile) != expected: raise ValueError('profile length does not match image axis')
    fit, coefficients = fit_profile(profile, degree)
    delta = fit.mean()-fit
    return h+(delta[None, :] if axis == 'x' else delta[:, None]), coefficients


def width_profile(encoded_height, position, width=1, axis='x'):
    """S1 105/106 and S2 1b2/1b3: bounded band, mean then truncate u16.

    Supports the full-width/full-height calls actually used by this capsule.
    No validity mask enters the observed averaging interface. Width=0 selects
    a single integer row/column here (axis-aligned profile path).
    """
    h = _height(encoded_height)
    if axis not in ('x', 'y') or width < 0 or not np.isfinite(width):
        raise ValueError('axis x/y and nonnegative finite width required')
    if h.min() < 0 or h.max() > 65535: raise ValueError('u16 code domain required')
    span = h.shape[0 if axis == 'x' else 1]
    if not np.isfinite(position) or not 0 <= position < span: raise ValueError('position out of range')
    if width == 0: lo = hi = int(position)
    else:
        lo = max(0, int(position)-int(width/2))
        hi = max(lo, min(span-1, lo+int(width)-1))
    values = h[lo:hi+1].mean(axis=0) if axis == 'x' else h[:,lo:hi+1].mean(axis=1)
    return np.trunc(values).astype(np.uint16), (lo,hi)


def manual_quadratic(control_xy, length, *, vendor_nudge=True):
    """S1 9e: despite 'Spline' name this is three-point quadratic interpolation.

    Close x positions are adjusted in the observed order when vendor_nudge is
    true; the original input array is not mutated by this independent API.
    """
    p = np.asarray(control_xy, dtype=np.float64)
    if p.shape != (3,2) or not np.isfinite(p).all() or not 0 < length < 32768:
        raise ValueError('three finite control points and supported Int16 length required')
    x = p[:,0].copy(); y = p[:,1]
    if y[0] == y[1] == y[2]: return np.full(length,y[0]), x
    if vendor_nudge:
        if abs(x[0]-x[1]) < 1: x[1] = x[0]+1
        if abs(x[0]-x[2]) < 1: x[2] = x[0]+1
        if abs(x[1]-x[2]) < 1: x[2] = x[1]+1
    matrix = np.vander(x,3,increasing=True)
    if np.linalg.matrix_rank(matrix) < 3: raise ValueError('degenerate control coordinates')
    coeff = np.linalg.solve(matrix,y)
    return np.polynomial.polynomial.polyval(np.arange(length),coeff), x


def flatten_manual(height, controls, axis='x', *, vendor_nudge=True):
    """S1 9c/a0: evaluate trend, subtract trend, preserve its mean.

    Controls for y are expressed in UI bottom-to-top profile coordinates, as in
    the observed reverse before vertical correction. This differs from raw rows.
    """
    h=_height(height)
    if axis not in ('x','y'): raise ValueError('axis must be x or y')
    trend,adjusted=manual_quadratic(controls,h.shape[1 if axis=='x' else 0],vendor_nudge=vendor_nudge)
    if axis=='y': trend=trend[::-1]
    delta=trend.mean()-trend
    return h+(delta[None,:] if axis=='x' else delta[:,None]),adjusted


def sample_point_mean(height, xy, size=1, quantize_u16=False):
    """S1 a4 clamped-neighborhood mean; optional encoded-height truncation."""
    h = _height(height)
    if size < 1 or size % 2 != 1: raise ValueError('positive odd neighborhood size required')
    x, y = map(int, xy); radius = (size-1)//2
    xx = np.clip(np.arange(x-radius, x+radius+1), 0, h.shape[1]-1)
    yy = np.clip(np.arange(y-radius, y+radius+1), 0, h.shape[0]-1)
    value = float(h[np.ix_(yy, xx)].mean())
    if quantize_u16:
        if not 0 <= value <= 65535: raise ValueError('u16 conversion outside supported input domain')
        return float(int(value))
    return value


def plane_from_three(points):
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (3, 3) or not np.isfinite(p).all(): raise ValueError('3x3 finite xyz points required')
    normal = np.cross(p[1]-p[0], p[2]-p[0])
    return np.r_[normal, -normal @ p[0]]


def three_point_flatten(height, xy, size=1, quantize_u16=True, vendor_degenerate_fallback=False):
    """S1 a3/a5 core, before S2 OffsetHeightFor3Points and u16 rescaling.

    Default explicit rejection of degenerate picks is a documented validation
    guard. vendor_degenerate_fallback=True reproduces the observed C==0 -> 1.
    """
    h = _height(height); xy = np.asarray(xy, dtype=np.float64)
    if xy.shape != (3, 2): raise ValueError('three xy points required')
    p = np.c_[xy, [sample_point_mean(h, v, size, quantize_u16) for v in xy]]
    coefficients = plane_from_three(p)
    a, b, c, _ = coefficients
    degenerate = c == 0
    if degenerate and not vendor_degenerate_fallback: raise ValueError('degenerate plane')
    denom = 1 if degenerate else c
    yy, xx = np.indices(h.shape, dtype=np.float64)
    trend = (-a/denom)*(xx-h.shape[1]/2)+(-b/denom)*(yy-h.shape[0]/2)
    return h-trend, coefficients, degenerate


def offset_three_point_encoded(height):
    """S2 e8 exact observed scalar branches, including unusual wide-range branch.

    This is intentionally not 'corrected' to an assumed intended algorithm.
    Input/output are height codes, not micrometers.
    """
    h = _height(height); low = float(h.min()); high = float(h.max()); span = high-low
    if span > 65535: offset = (span-65535)*.5 + low
    elif low < 0: offset = -low
    elif high > 65535: offset = -(high-65535)
    else: offset = 0.
    return h+offset, offset


@dataclass(frozen=True)
class EncodedHeight:
    values: np.ndarray
    offset: float
    scale: float

    def decode(self):
        return self.values.astype(np.float64)/self.scale-self.offset


def encode_rescaled_height(height):
    """S2 e9 scalar remapping, without its optional spatial crop/resize.

    np.rint is the prototype conversion rule; old OpenCV tie/overflow rounding
    was not executed for comparison. Reversible affine metadata is explicit.
    """
    h = _height(height); offset = 1-float(h.min()) if h.min() <= 0 else 0.
    shifted = h+offset; maximum = float(shifted.max())
    scale = 65535/maximum if maximum > 65535 else 1.
    result = np.clip(np.rint(shifted*scale), 0, 65535).astype(np.uint16)
    return EncodedHeight(result, offset, scale)


def encode_projected_height(height, projected_width, projected_height):
    """S2 e9 full spatial boundary: top-left crop, code conversion, linear resize.

    Projected dimensions are supplied by the caller after its slope/pitch
    calculation. Returned metadata is ours; vendor returns only scale here.
    """
    h=_height(height); rows,cols=h.shape
    if min(projected_width,projected_height)<=0:raise ValueError('positive projected dimensions required')
    cw=max(1,int(cols*cols/projected_width)) if projected_width>cols else cols
    ch=max(1,int(rows*rows/projected_height)) if projected_height>rows else rows
    region=h[:ch,:cw]
    offset=1-float(region.min()) if region.min()<=0 else 0.
    peak=float(region.max())+offset
    scale=65535/peak if peak>65535 else 1.
    small=np.clip(np.rint((region+offset)*scale),0,65535).astype(np.uint16)
    values=cv2.resize(small,(cols,rows),interpolation=cv2.INTER_LINEAR) if (cw,ch)!=(cols,rows) else small
    return EncodedHeight(values,offset,scale), {'crop_xywh':[0,0,cw,ch],'output_wh':[cols,rows]}


def sphere_from_four(points):
    """S2 1f9/1fa geometric core; singular systems explicitly rejected."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (4, 3) or not np.isfinite(p).all(): raise ValueError('4x3 finite xyz points required')
    a = 2*(p[1:]-p[0]); b = (p[1:]**2).sum(axis=1)-(p[0]**2).sum()
    if np.linalg.matrix_rank(a) < 3: raise ValueError('degenerate sphere points')
    center = np.linalg.solve(a, b)
    radius = float(np.linalg.norm(p[0]-center))
    concave = bool(np.any(center[2] > p[:, 2]))
    return center, radius, concave


def subtract_sphere(height, x, y, center, radius, concave=False):
    """S2 1f8 interior sphere subtraction in shared physical coordinate units.

    Outside the radicand domain returns NaN/valid mask, a deliberate analysis
    interface rather than one of the vendor background-fill UI modes.
    """
    h = _height(height); x, y = np.broadcast_arrays(x, y)
    if x.shape != h.shape or radius <= 0: raise ValueError('coordinate shape/radius mismatch')
    cx, cy, cz = np.asarray(center, dtype=np.float64)
    radicand = radius*radius-(x-cx)**2-(y-cy)**2
    valid = radicand >= 0
    surface = cz+(-1 if concave else 1)*np.sqrt(np.maximum(radicand, 0))
    return np.where(valid, h-surface, np.nan), valid
