"""Independent CPU reference for the statically recovered DSX SG algorithm.

Does NOT load Olympus DLLs or communicate with equipment. See implementation-spec.md.
Floating point/OpenCV-version parity, allocator/cache protocol and legacy undefined
boundary behavior are deliberately not claimed. Arrays use BGR channel order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import cv2


@dataclass(frozen=True)
class Profile:
    # Band weights are an explicit research profile, NOT this instrument's settings.
    lap_levels: int = 4
    depth_level: int = 3
    fast_weights: tuple = (0.25, 0.25, 0.25, 0.25)
    height_weights: tuple = (0.25, 0.25, 0.25, 0.25)
    fast_gauss: int = 3
    height_gauss: int = 11
    near_samples: int = 5
    interp: int = 8
    texture_threshold_3d: float = 1.8
    texture_threshold_efi: float = 1.5
    texture_scale: float = 0.5
    texture_gauss: int = 9
    texture_sad: float = 10.0
    texture_iteration_cap: int = 500  # Independent guard: native cap is not enforced.
    guide_scale: float = 0.5
    wm_radius_3d: int = 5
    wm_radius_efi: int = 1
    wm_eps_3d: float = 0.01
    wm_eps_efi: float = 0.01
    median_3d: int = 5
    median_efi: int = 3
    self_radius_3d: int = 9
    self_radius_efi: int = 1
    self_eps_3d: float = 0.0001
    self_eps_efi: float = 0.01
    sweep_threshold: float = 5.0
    sweep_max: float = 1.0
    fine_threshold: float = 30.0
    fine_max: float = 1.0
    blend_adjacent_z: bool = True


def resize(a, shape, interpolation=cv2.INTER_LINEAR):
    a = np.asarray(a)
    if a.shape[:2] == tuple(shape):
        return a.copy()
    return cv2.resize(a, (int(shape[1]), int(shape[0])), interpolation=interpolation)


def scaled_shape(shape, factor):
    return tuple(max(1, int(np.rint(d * factor))) for d in shape)


def gray(a):
    a = np.asarray(a, np.float32)
    return a if a.ndim == 2 else (a[..., 0] * 0.114 + a[..., 1] * 0.587 + a[..., 2] * 0.299)


def gaussian(a, kernel):
    # Native SmoothLapPyr uses borderType=1 (replicate), sigma=0.
    return cv2.GaussianBlur(np.asarray(a, np.float32), (kernel, kernel), 0,
                            borderType=cv2.BORDER_REPLICATE)


def pyr_down(a):
    k = np.array([1, 4, 6, 4, 1], np.float32) / 16
    return cv2.sepFilter2D(a, -1, k, k, borderType=cv2.BORDER_REFLECT_101)[::2, ::2]


def _up_axis(a, length, axis):
    # Recovered parity-dependent halo: left excludes edge, right includes it
    # when the target dimension is even (2N), excludes it for 2N-1.
    n = a.shape[axis]
    j = np.arange(length) // 2
    previous = np.maximum(j - 1, 0)
    previous[j == 0] = min(1, n - 1)
    following = j + 1
    following[following >= n] = n - 1 if length % 2 == 0 else max(0, n - 2)
    center = np.take(a, j, axis=axis)
    left = np.take(a, previous, axis=axis)
    right = np.take(a, following, axis=axis)
    shape = [1] * a.ndim
    shape[axis] = length
    even = (np.arange(length) % 2 == 0).reshape(shape)
    return np.where(even, (left + 6 * center + right) / 8, (center + right) / 2)


def pyr_up(a, shape):
    return _up_axis(_up_axis(a, shape[1], 1), shape[0], 0).astype(np.float32)


def focus_bands(image, levels, depth_level):
    if not 1 <= depth_level <= levels <= 4:
        raise ValueError('require 1 <= depth_level <= levels <= 4')
    cur = gray(image)
    shapes, bands = [cur.shape], []
    for _ in range(levels):
        smaller = pyr_down(cur)
        bands.append(np.abs(cur - pyr_up(smaller, cur.shape)))
        shapes.append(smaller.shape)
        cur = smaller
    shape = shapes[depth_level - 1]
    return np.stack([resize(b, shape, cv2.INTER_AREA if i < depth_level - 1 else cv2.INTER_LINEAR)
                     for i, b in enumerate(bands)])


def last_argmax(a, axis=0):
    return a.shape[axis] - 1 - np.argmax(np.flip(a, axis), axis=axis)


def input_order_argmax(a, input_ordinals=None):
    """Tie chooses last original input even when the Z representation is sorted."""
    if input_ordinals is None:
        return last_argmax(a)
    priority = np.asarray(input_ordinals).reshape((-1,) + (1,)*(a.ndim-1))
    return np.argmax(np.where(a == a.max(0), priority, -1), axis=0)


def band_blend(x, thresholds):
    """Managed _CalcBandBlend; x meaning must be supplied by the caller."""
    t0,t1,t2,t3,t4,t5 = thresholds
    if not t0 > t1 >= t2 > t3 >= t4 > t5:
        raise ValueError('reference requires ordered non-degenerate thresholds')
    a = 1. if x >= t0 else (x-t1)/(t0-t1) if x >= t1 else 0.
    b = 1.-a if x >= t1 else 1. if x >= t2 else (x-t3)/(t2-t3) if x >= t3 else 0.
    c = 0. if x >= t2 else 1.-b if x >= t3 else 1. if x >= t4 else (x-t5)/(t4-t5) if x >= t5 else 0.
    d = 1. if x < t5 else 1.-c if x <= t4 else 0.
    return a,b,c,d


def normalize_stack(images, z, arbitrary_order=False, duplicate_policy='reject'):
    images = np.asarray(images)
    z = np.asarray(z, np.float32)
    if images.ndim not in (3, 4) or len(images) != len(z) or len(z) < 1:
        raise ValueError('expected N,H,W[,3] image stack with N finite Z values')
    if images.ndim == 4 and images.shape[3] != 3:
        raise ValueError('only 1 or 3 channels')
    if not np.isfinite(z).all() or not np.isfinite(images).all():
        raise ValueError('non-finite input')
    changes = np.diff(z)
    nz = changes[np.abs(changes) > np.float32(4.93e-32)]
    if not arbitrary_order and len(nz) and (np.any(nz > 0) and np.any(nz < 0)):
        raise ValueError('Z reverses direction without arbitrary_order')
    order = np.argsort(z, kind='stable')
    sorted_z = z[order]
    keep = np.r_[True, np.diff(sorted_z) > np.float32(4.93e-32)]
    if not keep.all() and duplicate_policy == 'reject':
        raise ValueError('duplicate Z has mode-specific native semantics; explicit first policy required')
    if duplicate_policy not in ('reject','first'):
        raise ValueError('duplicate_policy must be reject or first')
    # Optional first policy is independent: native descending and arbitrary
    # duplicates have different focus/cache/mean semantics, documented in E04.
    order = order[keep]
    return images[order].astype(np.float32), z[order], order


def natural_spline(x, y):
    """Natural cubic coefficients a,b,c,d; independent Thomas recurrence."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    h = np.diff(x)
    if len(x) < 2 or np.any(h <= 0):
        raise ValueError('spline nodes must strictly increase')
    n = len(x)
    rhs = np.zeros_like(y)
    hs = h.reshape((-1,) + (1,) * (y.ndim - 1))
    slopes = np.diff(y, axis=0) / hs
    rhs[1:-1] = 3 * np.diff(slopes, axis=0)
    diagonal = np.ones(n)
    mu = np.zeros(n)
    q = np.zeros_like(y)
    for i in range(1, n - 1):
        diagonal[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / diagonal[i]
        q[i] = (rhs[i] - h[i-1] * q[i-1]) / diagonal[i]
    c = np.zeros_like(y)
    for i in range(n - 2, -1, -1):
        c[i] = q[i] - mu[i] * c[i+1]
    b = slopes - hs * (c[1:] + 2*c[:-1]) / 3
    d = (c[1:] - c[:-1]) / (3 * hs)
    return y[:-1], b, c[:-1], d


def spline_values(x, coefficients, candidate):
    candidate = np.asarray(candidate, np.float64)
    indices = np.clip(np.searchsorted(x, candidate, side='right') - 1, 0, len(x)-2)
    trailing = coefficients[0].ndim - 1
    t = (candidate - np.asarray(x)[indices]).reshape((-1,) + (1,) * trailing)
    a, b, c, d = (q[indices] for q in coefficients)
    return ((d*t + c)*t + b)*t + a


def axial_candidates(z, start, count, interp=8):
    """0x180014890 candidate bounds; preserve native float32 rounding order."""
    z = np.asarray(z, np.float32)
    step = np.float32((z[-1]-z[0])/np.float32((len(z)-1)*interp))
    low, high = z[start], z[start+count-1]
    lower = int(np.ceil(np.float32((low-z[0])/step)))
    upper = int(np.floor(np.float32((high-z[0])/step)))
    while upper >= lower and high < np.float32(np.float32(upper)*step+z[0]):
        upper -= 1
    return np.float32(np.arange(lower,upper+1,dtype=np.float32)*step+z[0]), float(step)


def estimate_height(response, z, near_samples=5, interp=8, input_ordinals=None):
    """response N,H,W; return raw height, peak value, provenance diagnostics."""
    response = np.asarray(response, np.float32)
    z = np.asarray(z, np.float32)
    n, *shape = response.shape
    winner = input_order_argmax(response, input_ordinals)
    if n == 1:
        return np.full(shape, z[0], np.float32), response[0].copy(), {'path': 'single'}
    if n < near_samples or near_samples < 2:
        return z[winner], np.take_along_axis(response, winner[None], 0)[0], {'path': 'discrete-fallback'}
    # Native 0x180014890 uses float32 ceilf/floorf bounds and separately rounded
    # j*dz+z0. A float64 grid can wrongly drop the highest float32 Z by one step.
    starts = np.clip(winner - near_samples // 2, 0, n-near_samples)
    flat = response.reshape(n, -1)
    output, peak = np.empty(flat.shape[1]), np.empty(flat.shape[1])
    for start in np.unique(starts):
        pixels = np.flatnonzero(starts.ravel() == start)
        xx = z[start:start+near_samples]
        yy = flat[start:start+near_samples, pixels]
        candidates, step = axial_candidates(z, start, near_samples, interp)
        if len(candidates) == 0:
            raise ValueError('no candidate grid samples in near-peak window')
        curves = spline_values(xx, natural_spline(xx, yy), candidates)
        best = np.argmax(curves, axis=0)  # strict maximum retains earlier candidate.
        output[pixels] = candidates[best]
        peak[pixels] = curves[best, np.arange(len(pixels))]
    sorted_responses = np.sort(response, axis=0)
    margin = (sorted_responses[-1] - sorted_responses[-2]) / np.maximum(sorted_responses[-1], 1e-8)
    return output.reshape(shape).astype(np.float32), peak.reshape(shape).astype(np.float32), {
        'path': 'natural-cubic-grid', 'grid_step': float(step),
        'boundary_winner': (winner == 0) | (winner == n-1),
        'peak_margin': margin, 'winner': winner,
        'note': 'peak_margin/boundary mask are additional diagnostics, not vendor outputs'}


def texture_fill(depth, response, threshold, scale=.5, kernel=9, sad_limit=10., cap=500):
    depth = np.asarray(depth, np.float32)
    small = scaled_shape(depth.shape, scale)
    reliable = response >= threshold
    small_response = resize(response, small, cv2.INTER_AREA)
    valid = small_response >= threshold
    base = resize(depth, small, cv2.INTER_AREA) + 1
    value = np.where(valid, base, 0).astype(np.float32)
    support = valid.astype(np.float32)
    converged, sad, iterations = False, None, 0
    for iterations in range(1, cap+1):
        numerator = gaussian(value, kernel)
        denominator = gaussian(support, kernel)
        proposed = np.divide(numerator, denominator, out=np.zeros_like(value), where=denominator != 0)
        proposed[valid] = value[valid]
        sad = float(np.abs(proposed[~valid] - value[~valid]).sum(dtype=np.float64))
        value = proposed
        support = (denominator > 0).astype(np.float32)
        if sad < sad_limit:
            converged = True
            break
    filled = depth.copy()
    proposed = resize(value, depth.shape) - 1
    filled[~reliable] = proposed[~reliable]
    propagated_support = resize(support, depth.shape) > 0
    # Unsupported pixels remain explicit in a separate mask, even if numeric
    # legacy result is -1. Do not silently interpret -1 as measured surface.
    return filled, {'measured': reliable, 'supported': propagated_support,
                    'filled': ~reliable & propagated_support, 'iterations': iterations,
                    'converged': converged, 'sad': sad,
                    'native_loop_difference': 'independent hard cap is enforced here'}


def box(a, radius):
    a = np.asarray(a)
    dtype = np.float64 if a.dtype == np.float64 else np.float32
    return cv2.blur(np.asarray(a, dtype), (2*radius+1, 2*radius+1),
                    borderType=cv2.BORDER_REFLECT_101)


class GuidedFilter:
    """Recovered local linear equations, with independent float64 stability.

    Native float32 moment arithmetic is not reproduced: for 16-bit co-linear
    RGB, its epsilon can vanish when added to a large covariance. We compute
    all moments and the solve in float64, then return float32 filter results.
    This is an explicit numerical safeguard, not evidence of vendor behavior.
    """
    def __init__(self, guide, radius, eps, scale=.5):
        self.guide = np.asarray(guide, np.float64)
        self.radius = radius
        self.small = scaled_shape(self.guide.shape[:2], scale)
        self.low = resize(self.guide, self.small)
        self.mu = box(self.low, radius)
        if self.low.ndim == 2:
            self.inverse = 1 / (box(self.low*self.low, radius) - self.mu*self.mu + eps)
        else:
            cov = np.empty(self.small+(3,3), np.float64)
            for i in range(3):
                for j in range(3):
                    cov[..., i,j] = box(self.low[..., i]*self.low[..., j], radius) - self.mu[...,i]*self.mu[...,j]
            cov += np.eye(3, dtype=np.float64)*eps
            self.inverse = np.linalg.inv(cov)

    def apply(self, field):
        p = resize(np.asarray(field, np.float64), self.small)
        mean_p = box(p, self.radius)
        if self.low.ndim == 2:
            a = (box(self.low*p, self.radius) - self.mu*mean_p)*self.inverse
            b = mean_p - a*self.mu
            return (resize(box(a, self.radius), self.guide.shape[:2])*self.guide + resize(box(b, self.radius), self.guide.shape[:2])).astype(np.float32)
        cov = np.stack([box(self.low[...,i]*p, self.radius)-self.mu[...,i]*mean_p for i in range(3)], -1)
        a = np.einsum('...ij,...j->...i', self.inverse, cov)
        b = mean_p - np.sum(a*self.mu, -1)
        return (np.sum(resize(box(a, self.radius), self.guide.shape[:2])*self.guide, -1) + resize(box(b, self.radius), self.guide.shape[:2])).astype(np.float32)


def guided_weighted_median(depth, guide, z0, dz, label_count, radius, eps, median_size, scale=.5):
    if radius <= 0:
        return depth.copy(), np.zeros(depth.shape, bool)
    labels = np.rint((depth-z0)/dz)
    filt = GuidedFilter(guide, radius, eps, scale)
    total = np.zeros(depth.shape, np.float32)
    result = np.zeros(depth.shape, np.float32)
    assigned = np.zeros(depth.shape, bool)
    # Native processes four one-hot label planes per iteration. Scalar batching
    # here produces the same logical label order without its scratch allocator.
    for label in range(label_count):
        q = filt.apply((labels == label).astype(np.float32))
        total += np.where(assigned, 0, q)
        new = ~assigned & (total > .5)
        result[new] = z0+label*dz
        assigned |= new
    return cv2.medianBlur(result, median_size), ~assigned


def self_guided_depth(depth, z0, radius, eps, scale=.5):
    if radius <= 0:
        return depth.copy()
    extent = float(np.max(depth)-z0)  # native overwrites min with sorted unique Z[0].
    if extent == 0:
        return depth.copy()
    normalized = (depth-z0)/extent
    return GuidedFilter(normalized, radius, eps, scale).apply(normalized)*extent+z0


def sample_z(images, z, depth, blend=True):
    """FineEFI: physical-Z bracketing; exact midpoint tie selects upper layer."""
    if len(z) == 1:
        return images[0].copy()
    high = np.clip(np.searchsorted(z, depth, side='right'), 1, len(z)-1)
    low = high-1
    alpha = np.clip((depth-z[low])/(z[high]-z[low]), 0, 1)
    yy, xx = np.indices(depth.shape)
    if not blend:
        return images[np.where(alpha < .5, low, high), yy, xx].copy()
    if images.ndim == 4:
        alpha = alpha[...,None]
    return images[low, yy, xx]*(1-alpha)+images[high, yy, xx]*alpha


def reliability_blend(first, second, response, threshold, maximum):
    w = resize(np.minimum(response*maximum/threshold, maximum), first.shape[:2])
    if first.ndim == 3:
        w = w[...,None]
    return first*w+second*(1-w)


def reconstruct(images, z, profile=Profile(), arbitrary_order=False, duplicate_policy='reject'):
    stack, z, original_indices = normalize_stack(images, z, arbitrary_order, duplicate_policy)
    n, h, w = stack.shape[:3]
    if n == 1:
        return {'efi': stack[0], 'height': np.full((h,w), z[0], np.float32),
                'raw_height': np.full((h,w), z[0], np.float32), 'metadata': {'path': 'single'}}
    p = profile
    bands = np.stack([focus_bands(im, p.lap_levels, p.depth_level) for im in stack])
    fast_response = np.stack([[gaussian(b,p.fast_gauss) for b in row] for row in bands])
    height_response = np.stack([[gaussian(b,p.height_gauss) for b in row] for row in bands])
    shape = bands.shape[-2:]
    fast_max = np.zeros(shape, np.float32)
    fast = np.zeros_like(stack[0])
    raw_height = np.zeros(shape, np.float32)
    height_quality = np.zeros(shape, np.float32)
    diagnostics = []
    yy, xx = np.indices((h,w))
    sy = np.minimum((np.arange(h)*shape[0]/h).astype(int), shape[0]-1)
    sx = np.minimum((np.arange(w)*shape[1]/w).astype(int), shape[1]-1)
    for b in range(p.lap_levels):
        winners = input_order_argmax(fast_response[:,b], original_indices)
        selected = winners[sy[:,None], sx[None,:]]
        fast += p.fast_weights[b]*stack[selected, yy, xx]
        fast_max += p.fast_weights[b]*fast_response[:,b].max(0)
        height, peak, info = estimate_height(height_response[:,b], z, p.near_samples, p.interp,
                                            original_indices)
        raw_height += p.height_weights[b]*height
        if p.height_weights[b] > 0:
            if info['path'] == 'discrete-fallback':
                # Native 0x180022715 creates weight*Mat then assigns Q, rather
                # than scaleAdd: the last active band overwrites prior Q.
                height_quality = p.height_weights[b]*peak
            else:
                height_quality += p.height_weights[b]*peak
        diagnostics.append(info)
    mean = stack.mean(0)
    guide_full = reliability_blend(fast, mean, fast_max, p.sweep_threshold, p.sweep_max)
    guide = resize(guide_full, shape)
    dz = float((z[-1]-z[0])/((n-1)*p.interp))
    outputs = {}
    masks = {}
    for tag in ('3d','efi'):
        filled, fill_info = texture_fill(raw_height, height_quality,
              getattr(p, 'texture_threshold_'+tag), p.texture_scale, p.texture_gauss,
              p.texture_sad, p.texture_iteration_cap)
        median, unassigned = guided_weighted_median(filled, guide, float(z[0]), dz,
              (n-1)*p.interp+1, getattr(p,'wm_radius_'+tag), getattr(p,'wm_eps_'+tag),
              getattr(p,'median_'+tag), p.guide_scale)
        outputs[tag] = self_guided_depth(median, float(z[0]), getattr(p,'self_radius_'+tag),
                           getattr(p,'self_eps_'+tag), p.guide_scale)
        masks[tag] = fill_info | {'weighted_median_unassigned': unassigned}
    fine = sample_z(stack, z, resize(outputs['efi'], (h,w)), p.blend_adjacent_z)
    efi = reliability_blend(fast, fine, fast_max, p.fine_threshold, p.fine_max)
    return {'efi': efi, 'height': resize(outputs['3d'], (h,w)),
            'raw_height': resize(raw_height, (h,w)), 'height_small': outputs['3d'],
            'efi_height_small': outputs['efi'], 'raw_height_small': raw_height,
            'height_response': height_quality, 'fast_response': fast_max, 'fast_efi': fast,
            'guide': guide_full, 'masks': masks, 'band_diagnostics': diagnostics,
            'metadata': {'profile': asdict(p), 'z': z.tolist(),
                         'source_indices': original_indices.tolist(),
                         'height_shape': list(shape), 'path': 'SG-CPU-reference',
                         'comparison_status': 'not run against Olympus binary'}}


def stabilize_given_vectors(images, deltas, quality, limit_x, limit_y,
                            reliability_threshold, resize_back=False):
    """SGmv's trajectory/crop/resample wrapper; caller supplies MV estimates."""
    images = np.asarray(images)
    h,w = images.shape[1:3]
    if 2*limit_x >= w or 2*limit_y >= h or min(limit_x,limit_y) < 0:
        raise ValueError('invalid crop limits')
    if len(images) != len(deltas) or len(images) != len(quality):
        raise ValueError('one displacement and reliability per input')
    trajectory, output = [], []
    current = np.zeros(2, np.float64)
    for frame, delta, score in zip(images,deltas,quality):
        if score >= reliability_threshold:
            current = np.clip(current+delta, [-limit_x,-limit_y], [limit_x,limit_y])
        trajectory.append(current.copy())
        affine = np.array([[1,0,-current[0]-limit_x], [0,1,-current[1]-limit_y]], np.float32)
        warped = cv2.warpAffine(frame, affine, (w-2*limit_x,h-2*limit_y), flags=cv2.INTER_LINEAR)
        output.append(resize(warped,(h,w)) if resize_back else warped)
    crop_w, crop_h = w-2*limit_x, h-2*limit_y
    # x_source=(x_dest+.5)*cropW/W-.5 + marginX + accepted_shiftX.
    return np.stack(output), {'trajectory_xy': np.asarray(trajectory).tolist(),
            'crop_origin_xy': [limit_x,limit_y], 'crop_shape_hw': [crop_h,crop_w],
            'output_scale_to_input_xy': [crop_w/w,crop_h/h] if resize_back else [1.,1.],
            'resize_back': resize_back, 'mv_estimator': 'externally supplied; not simulated here'}


def main():
    """Replay an explicit offline NPZ; no device or proprietary code is loaded."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--images-key', default='images')
    parser.add_argument('--z-key', default='z')
    parser.add_argument('--z-unit', required=True, help='Caller-declared unit; never inferred from values')
    parser.add_argument('--profile-json', type=Path)
    parser.add_argument('--arbitrary-order', action='store_true')
    parser.add_argument('--duplicate-policy', choices=('reject','first'), default='reject')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.npz':
        parser.error('output must use the .npz extension')
    if args.input.resolve() == args.output.resolve():
        parser.error('output must differ from input')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('output already exists; choose a new name')
    values = json.loads(args.profile_json.read_text()) if args.profile_json else {}
    profile = Profile(**values)
    with np.load(args.input, allow_pickle=False) as data:
        result = reconstruct(data[args.images_key], data[args.z_key], profile,
                             args.arbitrary_order, args.duplicate_policy)
    arrays = {k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    for branch, mask in result.get('masks',{}).items():
        for name,value in mask.items():
            if isinstance(value,np.ndarray): arrays[branch+'_'+name] = value
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**arrays)
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    metadata = result['metadata'] | {'declared_z_unit':args.z_unit,
        'input_sha256':sha(args.input),'source_sha256':sha(Path(__file__)),
        'output_sha256':sha(args.output),'vendor_executed':False,
        'numpy':np.__version__,'opencv':cv2.__version__,
        'duplicate_policy':args.duplicate_policy,
        'masks':{branch:{k:v for k,v in mask.items() if not isinstance(v,np.ndarray)}
                 for branch,mask in result.get('masks',{}).items()}}
    args.output.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(json.dumps({'output':str(args.output),'metadata':str(args.output.with_suffix('.json'))}))


if __name__ == '__main__':
    main()
