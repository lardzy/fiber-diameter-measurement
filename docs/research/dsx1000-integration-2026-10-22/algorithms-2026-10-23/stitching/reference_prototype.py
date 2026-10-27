"""Independent algorithm reconstruction; never loads or executes Olympus code.

The formulas are linked in implementation-spec.md. This is NOT a replacement for
the vendor executable: ROI orientation, file codecs, complete traversal/tie rules,
historical OpenCV/ALGLIB numerics and the original random sampler are not emulated.
Numerical guards and graph gauge choices are analyst choices, explicitly returned.
"""
from __future__ import annotations

import math
import numpy as np
import cv2
from scipy.optimize import least_squares


def cosine_error(a, b):
    """CompositeDLL64 180014140: stride-2 sample, no mean subtraction."""
    a, b = np.asarray(a, np.float64)[::2, ::2], np.asarray(b, np.float64)[::2, ::2]
    if a.shape != b.shape or a.size == 0:
        raise ValueError("equal, nonempty 2-D ROI required")
    aa, bb = np.sum(a*a), np.sum(b*b)
    return 1.0 if aa*bb == 0 else abs(1.0-float(np.sum(a*b)/math.sqrt(aa*bb)))


def composite_search(search, template, x_range, y_range):
    """Explicit top-left ROI coordinates; integer 2-px search and texture score.

    Full-contained ROI variant only. Original GetMatching also intersects cropped
    ROIs at search borders; unsupported coordinates fail here rather than pretend.
    """
    search, template = np.asarray(search), np.asarray(template)
    h, w = template.shape
    best = None
    all_costs = []
    for y in range(*y_range, 2):
        for x in range(*x_range, 2):
            if x < 0 or y < 0 or y+h > search.shape[0] or x+w > search.shape[1]:
                raise ValueError("partial-border ROI branch is not implemented")
            roi = search[y:y+h, x:x+w]
            cost = cosine_error(roi, template)
            all_costs.append((cost, x, y))
            if best is None or cost < best["cost"]:
                best = {"x": x, "y": y, "cost": cost,
                        "min_vote_value": min(float(np.ptp(roi)), float(np.ptp(template)))/65535.0}
    if best is None:
        raise ValueError("empty search grid")
    # This uniqueness diagnostic is an improvement, not a vendor output.
    best["equal_minima"] = sum(abs(c-best["cost"]) < 1e-12 for c, _, _ in all_costs)
    return best


def texture_accept(min_vote_value, threshold=0.1):
    return min_vote_value > threshold  # strict comparison in OrderMatch


def global_normalize_gray(images):
    """Approximate Composite input grayscale + shared 16-bit range normalization.

    We omit vendor Gaussian noise preprocessing and uint rounding details.
    RGB array convention; cvCvtColor(code 7) is RGB2GRAY in the observed ABI.
    """
    gray = []
    for image in images:
        image = np.asarray(image)
        if image.dtype == np.uint8:
            image = image.astype(np.uint16)*256
        if image.dtype != np.uint16:
            raise ValueError("uint8 or uint16 required")
        gray.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image)
    lo, hi = min(float(g.min()) for g in gray), max(float(g.max()) for g in gray)
    if hi == lo:
        return [np.zeros_like(g) for g in gray], {"flat_input": True, "analyst_guard": True}
    return [np.rint((g.astype(float)-lo)*65535/(hi-lo)).astype(np.uint16) for g in gray], {"flat_input": False}


def border_distance(x, y, left, top, width, height):
    if not (left <= x < left+width and top <= y < top+height):
        return 0
    # right/bottom are exclusive but original uses right-x, not right-1-x.
    return max(1, min(x-left, left+width-x, y-top, top+height-y))


def blend_pixel(values, distances, power=1, enabled=True, count=100,
                threshold=math.inf, adjuster=1.0, previous_cap=math.inf):
    """APP 14001cf60/01d320 and weights 01ec10/01ed40, given covering tiles.

    Order is paint order; last covering tile owns the base weight. Config-derived
    area cap tables are supplied by caller, not reconstructed here. No overflow.
    """
    values = np.asarray(values)
    weights = [int(d)**power for d in distances]
    if len(values) == 0 or len(values) != len(weights) or any(w <= 0 for w in weights):
        raise ValueError("nonempty covering tiles with positive weights required")
    if count < 1 or power not in (1, 2, 3, 4) or adjuster < 0:
        raise ValueError("invalid configuration")
    if len(values) == 1 or count == 1 or not enabled or weights[-1] > threshold:
        return values[-1].copy()
    indices = list(range(len(values)-1, max(-1, len(values)-count-1), -1))
    w = [weights[-1]] + [int(min(weights[i], previous_cap)*adjuster) for i in indices[1:]]
    return np.floor(np.sum(values[indices].astype(np.float64)*np.array(w)[:, None], axis=0)/sum(w)).astype(values.dtype)


def align_height(parent, child, overlap_mask):
    parent, child, mask = np.asarray(parent, float), np.asarray(child, float), np.asarray(overlap_mask, bool)
    if parent.shape != child.shape or child.shape != mask.shape or not mask.any():
        raise ValueError("nonempty equal-size overlap required")
    if not np.isfinite(parent[mask]).all() or not np.isfinite(child[mask]).all():
        raise ValueError("nonfinite height requires an explicit invalid-data mask")
    offset = float(parent[mask].mean()-child[mask].mean())
    return child+offset, offset


def _components(n, edges):
    adjacency = [set() for _ in range(n)]
    for i, j, _, _ in edges:
        adjacency[i].add(j); adjacency[j].add(i)
    unseen, out = set(range(n)), []
    while unseen:
        component, todo = [], [min(unseen)]
        unseen.remove(todo[0])
        while todo:
            node = todo.pop(); component.append(node)
            for nxt in sorted(adjacency[node] & unseen):
                unseen.remove(nxt); todo.append(nxt)
        out.append(sorted(component))
    return out


def global_positions(initial, edges):
    """OLS quadratic objective. edge=(i,j,delta,score), delta = p_i - p_j.

    Vendor builds weight=sqrt(score if score>0 else .05), then objective w*r².
    Use linear least squares for the same quadratic, not original ALGLIB LM.
    Per-component centroid gauge is explicit analyst handling. Original objective
    has no anchor term; no assertion that its optimizer chooses this exact gauge.
    """
    initial = np.asarray(initial, float)
    if initial.ndim != 1 or not len(initial) or not np.isfinite(initial).all():
        raise ValueError("finite initial coordinate vector required")
    n = len(initial)
    for i, j, delta, score in edges:
        if i == j or not (0 <= i < n and 0 <= j < n) or not np.isfinite([delta, score]).all():
            raise ValueError("invalid graph edge")
    components = _components(n, edges)
    rows, rhs = [], []
    weights = []
    for i, j, delta, score in edges:
        weight = math.sqrt(score if score > 0 else .05)
        weights.append(weight)
        row = np.zeros(n); row[i] = math.sqrt(weight); row[j] = -math.sqrt(weight)
        rows.append(row); rhs.append(delta*math.sqrt(weight))
    for component in components:
        row = np.zeros(n); row[component] = 1/len(component)
        rows.append(row); rhs.append(float(initial[component].mean()))
    positions, _, rank, _ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    errors = [float(positions[i]-positions[j]-delta) for i, j, delta, score in edges]
    return {"positions": positions, "components": components, "rank": int(rank),
            "objective": float(sum(w*r*r for w, r in zip(weights, errors))),
            "edge_weights": weights, "residuals": errors,
            "gauge": "analyst component centroid; vendor objective is unanchored",
            "zero_score_edges_are_retained": True}


def _dirichlet(t, cutoff, size):
    # Removable-singularity guard differs from direct vendor sin()/sin().
    return cutoff*size*np.sinc(cutoff*t)/np.sinc(t/size)


def phase_ols(a, b, lpf=False, cutoff=.75, search_size=None, hann=True):
    """OLS phase variants: 5x5 centroid or 3x3 Dirichlet+amplitude fit.

    Returns displacement of B relative to A. Numerical solver and sine zero guard
    are analyst choices. Does not implement outer histogram rejection/CLAHE.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < 8:
        raise ValueError("equal 2-D images of at least 8x8 required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite images")
    h, w = a.shape
    if hann:
        win = cv2.createHanningWindow((w, h), cv2.CV_64F)
        a, b = a*win, b*win
    H, W = cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)
    cross = np.fft.fft2(a, (H, W))*np.conj(np.fft.fft2(b, (H, W)))
    cross /= abs(cross)+np.finfo(float).eps
    cutoff = min(1.0, max(.1, cutoff))
    if lpf:
        # Literal rectangular corner keep: floor(cutoff*N/2) bins at each end.
        mx, my = int(W*cutoff/2), int(H*cutoff/2)
        mask_x = (np.arange(W) < mx) | (np.arange(W) >= W-mx)
        mask_y = (np.arange(H) < my) | (np.arange(H) >= H-my)
        cross *= mask_y[:, None] & mask_x[None, :]
    corr = np.fft.fftshift(np.fft.ifft2(cross).real)*(W*H)
    sw, sh = search_size or (W, H)
    x0, y0 = max(0, round(W/2-sw/2)), max(0, round(H/2-sh/2))
    roi = corr[y0:min(H, y0+sh), x0:min(W, x0+sw)]
    if not roi.size:
        raise ValueError("empty peak search")
    yy, xx = np.unravel_index(np.argmax(roi), roi.shape); xx += x0; yy += y0
    if lpf:
        if xx < 1 or yy < 1 or xx+1 >= W or yy+1 >= H:
            return {"shift": [W/2-xx, H/2-yy], "score": None, "status": "fit_border"}
        observed = corr[yy-1:yy+2, xx-1:xx+2]
        ys, xs = np.mgrid[-1:2, -1:2]
        def residual(v):
            model = _dirichlet(ys+v[1], cutoff, H)*_dirichlet(xs+v[0], cutoff, W)*v[2]
            return (observed-model).ravel()/(H*W)
        def initial(l, c, r):
            den = 2*l-4*c+2*r
            return float(np.clip(-(l-r)/den, -1, 1)) if abs(den) > 1e-15 else 0.0
        start = [initial(observed[1, 0], observed[1, 1], observed[1, 2]),
                 initial(observed[0, 1], observed[1, 1], observed[2, 1]), 1.0]
        fit = least_squares(residual, start, max_nfev=400, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        dx, dy = np.clip(fit.x[:2], -1, 1)
        return {"shift": [float(W/2-xx+dx), float(H/2-yy+dy)], "score": float(fit.x[2]),
                "status": "fit", "fit_rms": float(np.sqrt(np.mean(residual(fit.x)**2))),
                "nfev": int(fit.nfev)}
    xlo, xhi, ylo, yhi = max(0, xx-2), min(W, xx+3), max(0, yy-2), min(H, yy+3)
    weights = corr[ylo:yhi, xlo:xhi]; sy, sx = np.mgrid[ylo:yhi, xlo:xhi]
    total = weights.sum(); denominator = total+np.finfo(float).eps
    return {"shift": [float(W/2-(sx*weights).sum()/denominator), float(H/2-(sy*weights).sum()/denominator)],
            "score": float(total/(W*H)), "status": "centroid"}


def ols_gate(score, threshold=.05):
    return score is not None and threshold < score < 1.2


def transform_five(points, parameters):
    """140125310: R(theta) @ (diag(sx, sy) @ p + [tx, ty])."""
    tx, ty, sx, sy, theta = parameters
    c, s = math.cos(theta), math.sin(theta)
    return (np.asarray(points)*[sx, sy]+[tx, ty]) @ np.array([[c, s], [-s, c]])


def fit_hardware_transform(source, target, seed, inlier_threshold, trials=100):
    """Five-parameter RANSAC-style study; not original rand()/LM implementation.

    Explicitly reject collinear geometry, unlike observed vendor wrapper where
    no such check has been identified. Test exposes this modeling boundary.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("at least 3 paired 2-D points")
    if np.linalg.matrix_rank(source-source.mean(0)) < 2:
        raise ValueError("degenerate geometry; analyst guard")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(min(trials, math.comb(len(source), 3))):
        idx = rng.choice(len(source), 3, replace=False)
        if np.linalg.matrix_rank(source[idx]-source[idx].mean(0)) < 2:
            continue
        fit = least_squares(lambda p: (transform_five(source[idx], p)-target[idx]).ravel(),
                            [0, 0, 1, 1, 0], max_nfev=200)
        err = np.linalg.norm(transform_five(source, fit.x)-target, axis=1)
        mask = err < inlier_threshold
        if best is None or mask.sum() > best[0].sum():
            best = (mask, fit.x)
        if mask.all():
            break
    if best is None or best[0].sum() < 3:
        raise ValueError("no nondegenerate consensus")
    mask, p = best
    fit = least_squares(lambda q: (transform_five(source[mask], q)-target[mask]).ravel(), p)
    return {"parameters": fit.x, "inliers": mask,
            "rms": float(np.sqrt(np.mean((transform_five(source[mask], fit.x)-target[mask])**2)))}


def pair_from_prior(image_i, image_j, prior_i, prior_j):
    """Explicit common-world ROI, returning d_ij = position_i - position_j.

    Our A/B order gives d = prior_i-prior_j + phase(A_i,A_j). The vendor wrapper
    uses the opposite pair orientation and subtraction; do not copy a sign without
    this coordinate convention. Integer priors only in this reference pipeline.
    """
    pi, pj = np.asarray(prior_i, int), np.asarray(prior_j, int)
    image_i, image_j = np.asarray(image_i), np.asarray(image_j)
    if image_i.shape != image_j.shape:
        raise ValueError("equal-sized grayscale tiles")
    h, w = image_i.shape
    lo, hi = np.maximum(pi, pj), np.minimum(pi+[w,h], pj+[w,h])
    shape = hi-lo
    if np.any(shape < 12):
        return {"accepted":False,"reason":"insufficient_prior_overlap","delta":(pi-pj).astype(float),"score":0.0}
    ai, aj = lo-pi, lo-pj
    a=image_i[ai[1]:ai[1]+shape[1], ai[0]:ai[0]+shape[0]]
    b=image_j[aj[1]:aj[1]+shape[1], aj[0]:aj[0]+shape[0]]
    m=phase_ols(a,b,lpf=True,cutoff=.75,search_size=(min(14,int(shape[0])),min(14,int(shape[1]))))
    accepted=ols_gate(m["score"], .1)
    return {"accepted":accepted,"reason":"score_gate" if not accepted else "estimated",
            "delta":pi-pj+np.asarray(m["shift"]) if accepted else (pi-pj).astype(float),
            "score":m["score"] if accepted else 0.0,"phase":m,"roi_size":shape}


def reconstruct_grid(tiles, priors, rows, cols, reject_bad_edges=False):
    """Functional 2-D research pipeline, not exact native traversal/IO emulation.

    Enumerate right/down grid adjacencies; estimate translation with hardware ROI;
    solve all XY edges; normalize minimum; round to integer pixel canvas; feather.
    reject_bad_edges=True is an explicit improvement relative to retaining .05.
    """
    tiles, priors=np.asarray(tiles),np.asarray(priors)
    if len(tiles)!=rows*cols or priors.shape!=(len(tiles),2):
        raise ValueError("grid shape mismatch")
    pairs=[]
    for y in range(rows):
        for x in range(cols):
            i=y*cols+x
            for j in ([i-1] if x else [])+([i-cols] if y else []):
                result=pair_from_prior(tiles[i],tiles[j],priors[i],priors[j])
                pairs.append({"i":i,"j":j,**result})
    edges=[p for p in pairs if p["accepted"] or not reject_bad_edges]
    solutions=[]
    for axis in range(2):
        solutions.append(global_positions(priors[:,axis],[(p["i"],p["j"],p["delta"][axis],p["score"]) for p in edges]))
    xy=np.stack([s["positions"] for s in solutions],axis=1)
    xy-=xy.min(axis=0)
    integer_xy=np.rint(xy).astype(int)
    h,w=tiles.shape[1:]
    W,H=integer_xy.max(axis=0)+[w,h]
    sum_image=np.zeros((H,W));sum_weight=np.zeros((H,W))
    yy,xx=np.mgrid[:h,:w]
    # No count/cap shortcut: all covering tiles, linear distance weights.
    weights=np.maximum(1,np.minimum.reduce([xx,w-xx,yy,h-yy]))
    for tile,(x,y) in zip(tiles,integer_xy):
        sum_image[y:y+h,x:x+w]+=tile*weights
        sum_weight[y:y+h,x:x+w]+=weights
    valid=sum_weight>0
    mosaic=np.divide(sum_image,sum_weight,out=np.zeros_like(sum_image),where=valid)
    return {"positions":xy,"integer_positions":integer_xy,"pairs":pairs,
            "graph_x":solutions[0],"graph_y":solutions[1],"mosaic":mosaic,"valid":valid,
            "reject_bad_edges":reject_bad_edges}
