"""Bounded same-plane translation registration and robust graph alignment.

Algorithm references: MIST (10.1038/s41598-017-04567-y), ASHLAR
(10.1093/bioinformatics/btac544), and BigStitcher global optimization.
No source image is modified and no unconstrained homography is estimated.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from fdm.services.slide_layout import (
    LayoutTile, LayerRegistrationEvidence, PairRegistrationResult, SlideLayoutSnapshot, canonical_bytes,
    read_connection, read_source_tiles, source_digest,
)
from fdm.atomic_io import atomic_write_json

# Bump when candidate generation/acceptance changes: older evidence must not
# bypass improved rejection checks simply because the pixels are unchanged.
REGISTRATION_VERSION = "same-plane-translation-1"

@dataclass(frozen=True, slots=True)
class RegistrationConfig:
    min_overlap: float = .10
    min_strip_pixels: int = 64
    uncertainty_fraction: float = .02
    max_shift_fraction: float = .04
    min_ncc: float = .70
    ambiguity_margin: float = .06
    max_disagreement: float = 1.5
    min_texture: float = 2.0
    max_graph_iterations: int = 8


@dataclass(frozen=True, slots=True)
class TranslationEvidence:
    accepted: bool
    reason: str
    dx: float = 0.0
    dy: float = 0.0
    score: float = 0.0


def _overlap(a, b, dx: float, dy: float):
    left, top = max(0, math.ceil(dx)), max(0, math.ceil(dy))
    right, bottom = min(a.shape[1], math.floor(dx + b.shape[1])), min(a.shape[0], math.floor(dy + b.shape[0]))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _crops(a, b, dx, dy):
    dx, dy = float(dx), float(dy)
    rect = _overlap(a, b, dx, dy)
    if rect is None:
        return None
    left, top, right, bottom = rect
    aa = a[top:bottom, left:right]
    # Remap from global sample centres; never warp the source stored image.
    xx, yy = np.meshgrid(np.arange(left, right, dtype=np.float32) - dx,
                         np.arange(top, bottom, dtype=np.float32) - dy)
    bb = cv2.remap(b, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aa, bb


def _ncc(a, b) -> float:
    a = a.astype(np.float32) - float(a.mean())
    b = b.astype(np.float32) - float(b.mean())
    norm = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    return float(np.sum(a * b)) / norm if norm > 1e-8 else 0.0


def _prepared(gray):
    f = gray.astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), .6)
    return f - cv2.GaussianBlur(f, (0, 0), 8.0)


def _periodic_texture(crop, bound):
    # A high matching score is insufficient for repeated fibres/cells. Search
    # self-similar translations within the admissible mechanical uncertainty.
    h, w = crop.shape
    y0, x0 = max(0, (h - 512) // 2), max(0, (w - 512) // 2)
    sample = np.ascontiguousarray(crop[y0:y0 + 512, x0:x0 + 512])
    h, w = sample.shape
    if min(h, w) < 32:
        return True
    spectrum = np.fft.rfft2(sample)
    correlation = np.fft.irfft2(spectrum * np.conj(spectrum), s=sample.shape).real
    yy = np.minimum(np.arange(h), h - np.arange(h))
    xx = np.minimum(np.arange(w), w - np.arange(w))
    valid = (np.maximum(yy[:, None], xx[None, :]) >= 4) & (np.maximum(yy[:, None], xx[None, :]) <= 2 * bound)
    valid &= (yy[:, None] < h // 3) & (xx[None, :] < w // 3)
    correlation[~valid] = -np.inf
    for _ in range(12):
        py, px = np.unravel_index(np.argmax(correlation), correlation.shape)
        if not np.isfinite(correlation[py, px]):
            break
        dy, dx = (py if py <= h // 2 else py - h), (px if px <= w // 2 else px - w)
        pair = _crops(sample, sample, dx, dy)
        if pair is not None and _ncc(*pair) > .97:
            return True
        correlation[max(0,py-1):py+2,max(0,px-1):px+2] = -np.inf
    return False


def _phase_candidates(a, b, dx, dy):
    crops = _crops(a, b, dx, dy)
    if crops is None:
        return []
    aa, bb = crops
    # Retain at least 32 pixels across the narrower overlap dimension.
    scale = min(1.0, 1024 / max(aa.shape), min(aa.shape) / 32)
    if scale < 1:
        aa = cv2.resize(aa, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        bb = cv2.resize(bb, (aa.shape[1], aa.shape[0]), interpolation=cv2.INTER_AREA)
    window = cv2.createHanningWindow((aa.shape[1], aa.shape[0]), cv2.CV_32F)
    fa, fb = np.fft.rfft2(aa * window), np.fft.rfft2(bb * window)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-12)
    response = np.fft.irfft2(cross, s=aa.shape).real
    proposals = []
    for _ in range(4):
        py, px = np.unravel_index(np.argmax(response), response.shape)
        sx = px if px <= aa.shape[1] // 2 else px - aa.shape[1]
        sy = py if py <= aa.shape[0] // 2 else py - aa.shape[0]
        proposals.append((dx + sx / scale, dy + sy / scale))
        for oy in range(-3, 4):
            for ox in range(-3, 4):
                response[(py + oy) % aa.shape[0], (px + ox) % aa.shape[1]] = -np.inf
    return proposals


def _feature_candidate(a, b, dx, dy):
    rect = _overlap(a, b, dx, dy)
    if rect is None:
        return None
    left, top, right, bottom = rect
    bx, by = int(round(left - dx)), int(round(top - dy))
    width, height = right - left, bottom - top
    aa = a[top:bottom, left:right]
    bb = b[by:by + height, bx:bx + width]
    if aa.shape != bb.shape or min(aa.shape) < 16:
        return None
    scale = min(1.0, 1200 / max(aa.shape))
    if scale < 1:
        aa = cv2.resize(aa, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        bb = cv2.resize(bb, (aa.shape[1], aa.shape[0]), interpolation=cv2.INTER_AREA)
    detector = cv2.SIFT_create(nfeatures=2000)
    ka, da = detector.detectAndCompute(aa, None)
    kb, db = detector.detectAndCompute(bb, None)
    if da is None or db is None or min(len(da), len(db)) < 8:
        return None
    matcher = cv2.BFMatcher()
    forward = {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(da, db, k=2)
               if len(pair) == 2 for m, n in [pair] if m.distance < .75 * n.distance}
    backward = {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(db, da, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < .75 * n.distance}
    matches = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    if len(matches) < 8:
        return None
    pa = np.array([ka[i].pt for i, _ in matches]) / scale + (left, top)
    pb = np.array([kb[j].pt for _, j in matches]) / scale + (bx, by)
    shifts = pa - pb
    # Translation-only consensus. No affine fit that could alter geometry.
    counts = [np.count_nonzero(np.linalg.norm(shifts - s, axis=1) <= 1.5) for s in shifts]
    inliers = np.linalg.norm(shifts - shifts[int(np.argmax(counts))], axis=1) <= 1.5
    if int(inliers.sum()) < 8 or float(inliers.mean()) < .6:
        return None
    span = np.ptp(pa[inliers], axis=0)
    if span[0] < width * .15 or span[1] < height * .15:
        return None
    return tuple(np.median(shifts[inliers], axis=0))


def _estimate_pair_window(a, b, *, dx: float, dy: float, axis: str,
                          config: RegistrationConfig, original_extent: int, fixed: bool = False) -> TranslationEvidence:
    if a is None or b is None or a.ndim != 2 or b.ndim != 2 or min(*a.shape, *b.shape) < 16:
        return TranslationEvidence(False, "empty")
    horizontal = axis == "x"
    extent = original_extent
    uncertainty = max(2.0, extent * config.uncertainty_fraction)
    required = max(config.min_strip_pixels, extent * config.min_overlap)
    rect = _overlap(a, b, dx, dy)
    if rect is None or (rect[2] - rect[0] if horizontal else rect[3] - rect[1]) - uncertainty < required:
        return TranslationEvidence(False, "overlap")
    aa, bb = _prepared(a), _prepared(b)
    crops = _crops(aa, bb, dx, dy)
    if crops is None or min(float(c.std()) for c in crops) < config.min_texture:
        return TranslationEvidence(False, "texture")
    # Aperture ambiguity: a straight fibre does not constrain both axes.
    for crop in crops:
        gx, gy = cv2.Sobel(crop, cv2.CV_32F, 1, 0), cv2.Sobel(crop, cv2.CV_32F, 0, 1)
        eigen = np.linalg.eigvalsh([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])
        if eigen[-1] <= 1e-8 or eigen[0] / eigen[-1] < .005:
            return TranslationEvidence(False, "ambiguous")
    bound = max(4.0, extent * config.max_shift_fraction)
    if any(_periodic_texture(crop, bound) for crop in crops):
        return TranslationEvidence(False, "ambiguous")

    def evaluate(proposal):
        px, py = (float(v) for v in proposal)
        if max(abs(px - dx), abs(py - dy)) > bound:
            return None
        pair = _crops(aa, bb, px, py)
        if pair is None or min(pair[0].shape) < 16:
            return None
        # A bounded full-resolution phase refinement follows the coarse peak.
        h, w = pair[0].shape
        residual, _response = cv2.phaseCorrelate(pair[0], pair[1], cv2.createHanningWindow((w, h), cv2.CV_32F))
        if not fixed and math.hypot(*residual) <= 3.0:
            px, py = px - residual[0], py - residual[1]
        if max(abs(px - dx), abs(py - dy)) > bound:
            return None
        pair = _crops(aa, bb, px, py)
        if pair is None:
            return None
        strip = pair[0].shape[1 if horizontal else 0]
        if strip - uncertainty < required:
            return None
        score = _ncc(*pair)
        # Demand agreement in two separated portions, not one tiny motif.
        direction = 0 if horizontal else 1
        chunks_a, chunks_b = np.array_split(pair[0], 3, axis=direction), np.array_split(pair[1], 3, axis=direction)
        supported = sum(_ncc(ca, cb) >= config.min_ncc for ca, cb in zip(chunks_a, chunks_b))
        if score < config.min_ncc or supported < 2:
            return None
        return px, py, score

    phase = [item for p in ([(dx, dy)] if fixed else _phase_candidates(aa, bb, dx, dy)) if (item := evaluate(p)) is not None]
    feature_seed = None if fixed else _feature_candidate(a, b, dx, dy)
    feature = evaluate(feature_seed) if feature_seed is not None else None
    phase.sort(key=lambda p: p[2], reverse=True)
    if phase:
        best = phase[0]
        if any(math.hypot(p[0] - best[0], p[1] - best[1]) > config.max_disagreement and
               p[2] >= best[2] - config.ambiguity_margin for p in phase[1:]):
            return TranslationEvidence(False, "ambiguous")
        if feature is not None and math.hypot(feature[0] - best[0], feature[1] - best[1]) > config.max_disagreement:
            return TranslationEvidence(False, "conflict")
    else:
        best = feature
    if best is None:
        return TranslationEvidence(False, "registration")
    if not fixed:
        # Phase-correlation centroids can be biased by a fraction of a pixel.
        # Translation-only ECC on an interior window refines the accepted peak;
        # it cannot introduce rotation, scale or leave the mechanical search box.
        px, py, score = best
        pair = _crops(aa, bb, px, py)
        if pair is not None:
            h, w = pair[0].shape
            y0, x0 = max(0, (h - 512) // 2), max(0, (w - 512) // 2)
            crop_a = np.ascontiguousarray(pair[0][y0:y0 + 512, x0:x0 + 512])
            crop_b = np.ascontiguousarray(pair[1][y0:y0 + 512, x0:x0 + 512])
            try:
                _, warp = cv2.findTransformECC(crop_a, crop_b, np.eye(2, 3, dtype=np.float32),
                    cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5), None, 3)
                rx, ry = px - float(warp[0, 2]), py - float(warp[1, 2])
                if (math.hypot(rx - px, ry - py) <= 2 and max(abs(rx - dx), abs(ry - dy)) <= bound):
                    refined = _crops(aa, bb, rx, ry)
                    refined_score = _ncc(*refined) if refined is not None else 0
                    if refined_score >= score:
                        best = (rx, ry, refined_score)
            except cv2.error:
                pass  # Preserve the independently validated, bounded candidate.
    return TranslationEvidence(True, "verified", best[0], best[1], best[2])


def estimate_pair(a, b, *, dx: float, dy: float, axis: str,
                  config: RegistrationConfig = RegistrationConfig(), fixed: bool = False) -> TranslationEvidence:
    if a is None or b is None or a.ndim != 2 or b.ndim != 2:
        return TranslationEvidence(False, "empty")
    extent = min(a.shape[1], b.shape[1]) if axis == "x" else min(a.shape[0], b.shape[0])
    rect = _overlap(a, b, dx, dy)
    uncertainty = max(2, extent * config.uncertainty_fraction)
    if rect is None or (rect[2] - rect[0] if axis == "x" else rect[3] - rect[1]) - uncertainty < max(config.min_strip_pixels, extent * config.min_overlap):
        return TranslationEvidence(False, "overlap")
    # Up to three separated full-resolution windows keep FFT/float scratch
    # bounded even for 8K fields. No downsampling of the final displacement.
    bound = math.ceil(max(4, extent * config.max_shift_fraction)) + 24
    left, top, right, bottom = rect
    long_size = bottom - top if axis == "x" else right - left
    starts = [0] if long_size <= 1024 else sorted(set((0, (long_size - 1024) // 2, long_size - 1024)))
    evidence = []
    reasons = []
    for start in starts:
        x0, y0, x1, y1 = left, top, right, bottom
        if axis == "x":
            y0, y1 = top + start, min(bottom, top + start + 1024)
        else:
            x0, x1 = left + start, min(right, left + start + 1024)
        ax, ay = max(0, x0 - bound), max(0, y0 - bound)
        bx, by = max(0, math.floor(x0 - dx) - bound), max(0, math.floor(y0 - dy) - bound)
        aa = a[ay:min(a.shape[0], y1 + bound), ax:min(a.shape[1], x1 + bound)]
        bb = b[by:min(b.shape[0], math.ceil(y1 - dy) + bound), bx:min(b.shape[1], math.ceil(x1 - dx) + bound)]
        result = _estimate_pair_window(aa, bb, dx=dx + bx - ax, dy=dy + by - ay,
                                       axis=axis, config=config, original_extent=extent, fixed=fixed)
        if result.accepted:
            evidence.append(replace(result, dx=result.dx + ax - bx, dy=result.dy + ay - by))
        else:
            reasons.append(result.reason)
    if len(evidence) < min(2, len(starts)):
        return TranslationEvidence(False, reasons[0] if reasons else "registration")
    vectors = np.array([(e.dx, e.dy) for e in evidence])
    median = np.median(vectors, axis=0)
    if np.max(np.linalg.norm(vectors - median, axis=1)) > config.max_disagreement:
        return TranslationEvidence(False, "conflict")
    return TranslationEvidence(True, "verified", float(median[0]), float(median[1]), min(e.score for e in evidence))


def adjacent_fovs(tiles):
    fovs = {}
    for tile in tiles:
        fovs.setdefault(tile.fov_id, tile)
    physical_grid = len({(t.stage_x, t.stage_y) for t in fovs.values()}) == len(fovs)
    for axis in ("x", "y"):
        groups = defaultdict(list)
        for tile in fovs.values():
            key = (tile.stage_y if axis == "x" else tile.stage_x) if physical_grid else (tile.nominal_y if axis == "x" else tile.nominal_x)
            groups[key].append(tile)
        for key in sorted(groups):
            ordered = sorted(groups[key], key=lambda t: t.nominal_x if axis == "x" else t.nominal_y)
            for first, second in zip(ordered, ordered[1:]):
                yield first, second, axis


def solve_layout(tiles, pairs, digest, config=RegistrationConfig()):
    representatives = {t.fov_id: t for t in tiles}
    ids = sorted(representatives)
    index = {fid: i for i, fid in enumerate(ids)}
    nominal = np.array([(representatives[f].nominal_x, representatives[f].nominal_y) for f in ids], dtype=float)
    nominal = nominal.reshape((-1, 2))
    prior_precision = np.array([1 / max(4.0, min(representatives[f].width,
        representatives[f].height) * config.max_shift_fraction) ** 2 for f in ids])
    pairs = list(pairs)
    positions = nominal.copy()
    # Sparse edge relaxation avoids an N x N dense matrix for large slides.
    for _round in range(config.max_graph_iterations):
        active = [p for p in pairs if p.accepted]
        positions[:] = nominal
        if not active:
            break
        ii = np.array([index[p.first] for p in active])
        jj = np.array([index[p.second] for p in active])
        delta = np.array([(p.dx, p.dy) for p in active])
        weights = np.array([max(.1, p.confidence) for p in active])
        # Components are anchored by their centroid in the stage layout;
        # individual edges remain free to correct their relative positions.
        parent = list(range(len(ids)))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, j in zip(ii, jj):
            parent[root(int(j))] = root(int(i))
        components = defaultdict(list)
        for i in range(len(ids)):
            components[root(i)].append(i)
        for _ in range(500):
            residual = positions[jj] - positions[ii] - delta
            robust = weights * np.minimum(1.0, config.max_disagreement / np.maximum(np.linalg.norm(residual, axis=1), 1e-6))
            # Weak physical-position priors bound drift along long chains;
            # isolated fields remain at their original positions.
            sums = nominal * prior_precision[:, None]
            totals = prior_precision.copy()
            np.add.at(sums, ii, (positions[jj] - delta) * robust[:, None])
            np.add.at(sums, jj, (positions[ii] + delta) * robust[:, None])
            np.add.at(totals, ii, robust)
            np.add.at(totals, jj, robust)
            updated = positions.copy()
            valid = totals > 0
            updated[valid] = .5 * positions[valid] + .5 * sums[valid] / totals[valid, None]
            for component in components.values():
                updated[component] += nominal[component].mean(axis=0) - updated[component].mean(axis=0)
            distance = float(np.max(np.abs(updated - positions)))
            positions = updated
            if distance < 1e-4:
                break
        bad = []
        for k, p in enumerate(pairs):
            if p.accepted:
                residual = float(np.linalg.norm(positions[index[p.second]] - positions[index[p.first]] - (p.dx, p.dy)))
                pairs[k] = replace(p, residual=residual)
                if residual > config.max_disagreement:
                    bad.append((residual, k))
        if not bad:
            break
        _, worst = max(bad)
        pairs[worst] = replace(pairs[worst], accepted=False, reason="graph_conflict", verified_focus=())
        if _round + 1 == config.max_graph_iterations:
            # Do not publish positions solved using an edge that was just removed.
            pairs = [replace(p, accepted=False, reason="graph_iteration_limit", verified_focus=()) if p.accepted else p for p in pairs]
            positions = nominal.copy()
    # Fail closed if bounded optimization has not resolved all conflicts.
    pairs = [replace(p, accepted=False, reason="graph_conflict", verified_focus=())
             if p.accepted and p.residual > config.max_disagreement else p for p in pairs]
    if not any(p.accepted for p in pairs):
        positions = nominal.copy()
    offset = np.minimum(0, np.floor(positions.min(axis=0))) if len(ids) else np.zeros(2)
    corrected = tuple(replace(t, x=float(positions[index[t.fov_id], 0] - offset[0]),
                              y=float(positions[index[t.fov_id], 1] - offset[1])) for t in tiles)
    return SlideLayoutSnapshot(digest, corrected, tuple(pairs),
                               max(1, math.ceil(max((t.x + t.width for t in corrected), default=1))),
                               max(1, math.ceil(max((t.y + t.height for t in corrected), default=1)))).sealed()


def register_slide(path: str | Path, *, config=RegistrationConfig(),
                   cancelled: Callable[[], bool] = lambda: False,
                   progress: Callable[[int, int], None] = lambda done, total: None,
                   checkpoint: str | Path | None = None,
                   final: bool = True) -> SlideLayoutSnapshot:
    connection = read_connection(path)
    saved = {}
    try:
        if final:
            connection.execute("BEGIN")
        # Capture only appends committed tiles; final publication uses a read snapshot.
        tiles = read_source_tiles(connection)
        manifest = json.loads(connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        focus_levels = manifest.get("focus_levels", [])
        by_fov = defaultdict(dict)
        for tile in tiles:
            if tile.z_index in by_fov[tile.fov_id]:
                raise ValueError("同一视场存在重复焦层，不能自动修复")
            by_fov[tile.fov_id][tile.z_index] = tile
        neighbors = list(adjacent_fovs(tiles))
        saved = {}
        if checkpoint is not None and Path(checkpoint).is_file():
            try:
                payload = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
                if payload.get("config") == asdict(config) and payload.get("algorithm") == REGISTRATION_VERSION:
                    saved = payload.get("evidence", {})
            except (ValueError, OSError):
                pass
        output = []
        used_evidence = set()
        from hashlib import sha256
        for n, (a, b, axis) in enumerate(neighbors):
            if cancelled():
                raise InterruptedError("拼接检查已暂停")
            evidence = []
            reasons = []
            layers = []
            common = sorted(set(by_fov[a.fov_id]) & set(by_fov[b.fov_id]))
            for z in common:
                ta, tb = by_fov[a.fov_id][z], by_fov[b.fov_id][z]
                if z < 0 or z >= len(focus_levels) or ta.focus_z != tb.focus_z or ta.focus_z != focus_levels[z]:
                    reasons.append("focus_mismatch")
                    layers.append(LayerRegistrationEvidence(z, ta.focus_z, False, "focus_mismatch"))
                    continue
                extent = min(ta.width, tb.width) if axis == "x" else min(ta.height, tb.height)
                prior = manifest.get("metadata", {}).get("xy_calibration", {}).get(axis, {})
                uncertainty = max(extent * config.uncertainty_fraction, float(prior.get("uncertainty_px", 0)), 2)
                pair_config = replace(config, uncertainty_fraction=uncertainty / max(1, extent))
                strip = (min(ta.nominal_x + ta.width, tb.nominal_x + tb.width) - max(ta.nominal_x, tb.nominal_x) if axis == "x" else
                         min(ta.nominal_y + ta.height, tb.nominal_y + tb.height) - max(ta.nominal_y, tb.nominal_y))
                if strip - uncertainty < max(config.min_strip_pixels, extent * config.min_overlap):
                    reasons.append("overlap")
                    layers.append(LayerRegistrationEvidence(z, ta.focus_z, False, "overlap"))
                    continue
                blobs = []
                for tile in (ta, tb):
                    row = connection.execute("SELECT image_png FROM tiles WHERE id=?", (tile.tile_id,)).fetchone()
                    blobs.append(bytes(row[0]))
                key = sha256(canonical_bytes([asdict(ta), asdict(tb), asdict(pair_config)]) + b"".join(sha256(v).digest() for v in blobs)).hexdigest()
                used_evidence.add(key)
                if key in saved:
                    result = TranslationEvidence(**saved[key])
                else:
                    images = [cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_GRAYSCALE) for v in blobs]
                    result = estimate_pair(*images, dx=b.nominal_x - a.nominal_x, dy=b.nominal_y - a.nominal_y, axis=axis, config=pair_config)
                    saved[key] = asdict(result)
                    del images
                layers.append(LayerRegistrationEvidence(z, ta.focus_z, result.accepted, result.reason,
                    result.dx, result.dy, result.score))
                if result.accepted:
                    evidence.append((z, result))
                else:
                    reasons.append(result.reason)
                if cancelled():
                    raise InterruptedError("拼接检查已暂停")
            pair = PairRegistrationResult(a.fov_id, b.fov_id, axis, False,
                "focus_mismatch" if "focus_mismatch" in reasons or not common else (reasons[0] if reasons else "registration"),
                layers=tuple(layers))
            if evidence and "focus_mismatch" not in reasons:
                vectors = np.array([(e.dx, e.dy) for _, e in evidence])
                median = np.median(vectors, axis=0)
                if np.max(np.linalg.norm(vectors - median, axis=1)) <= config.max_disagreement:
                    pair = replace(pair, accepted=True, reason="verified" if len(evidence) == len(focus_levels) else "partial_focus",
                        dx=float(median[0]), dy=float(median[1]), confidence=min(e.score for _, e in evidence),
                        verified_focus=tuple(z for z, _ in evidence), evidence_count=len(evidence))
                else:
                    pair = replace(pair, reason="focus_conflict")
            output.append(pair)
            if n + 1 == len(neighbors):
                saved = {key: saved[key] for key in used_evidence}
            if checkpoint is not None and ((n + 1) % 16 == 0 or n + 1 == len(neighbors)):
                atomic_write_json(Path(checkpoint), {"algorithm": REGISTRATION_VERSION, "config": asdict(config), "evidence": saved})
            progress(n + 1, len(neighbors))
        digest = source_digest(connection, cancelled) if final else "capture-in-progress"
        for final_round in range(3):
            layout = solve_layout(tiles, output, digest, config)
            # Verify the actual final seam; graph connectivity is not evidence.
            fovs = {t.fov_id: t for t in layout.tiles}
            verified = []
            rejected = False
            for pair in layout.pairs:
                if not pair.accepted:
                    verified.append(pair)
                    continue
                aa, bb = fovs[pair.first], fovs[pair.second]
                actual = (bb.x - aa.x, bb.y - aa.y)
                ok = []
                for z in pair.verified_focus:
                    images = []
                    for tile in (by_fov[pair.first][z], by_fov[pair.second][z]):
                        blob = connection.execute("SELECT image_png FROM tiles WHERE id=?", (tile.tile_id,)).fetchone()[0]
                        images.append(cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE))
                    extent = min(aa.width, bb.width) if pair.axis == "x" else min(aa.height, bb.height)
                    prior = manifest.get("metadata", {}).get("xy_calibration", {}).get(pair.axis, {})
                    uncertainty = max(extent * config.uncertainty_fraction, float(prior.get("uncertainty_px", 0)), 2)
                    pair_config = replace(config, uncertainty_fraction=uncertainty / max(1, extent))
                    evidence = estimate_pair(*images, dx=actual[0], dy=actual[1], axis=pair.axis, config=pair_config, fixed=True)
                    if evidence.accepted:
                        ok.append(z)
                    if cancelled():
                        raise InterruptedError("拼接检查已暂停")
                rejected |= not bool(ok)
                verified.append(replace(pair, accepted=bool(ok), verified_focus=tuple(ok),
                    evidence_count=len(ok),
                    reason=("verified" if len(ok) == len(focus_levels) else "partial_focus") if ok else "final_seam_failed",
                    layers=tuple(replace(layer, accepted=False, reason="final_seam_failed")
                        if layer.accepted and layer.z_index not in ok else layer for layer in pair.layers)))
            if not rejected:
                return replace(layout, pairs=tuple(verified)).sealed()
            output = verified
        # All iterations exhausted: preserve source positions rather than leave
        # a layout influenced by failed final constraints.
        return solve_layout(tiles, [replace(p, accepted=False, verified_focus=(), reason="final_seam_failed") for p in output], digest, config)
    except InterruptedError:
        if checkpoint is not None and saved:
            atomic_write_json(Path(checkpoint), {"algorithm": REGISTRATION_VERSION, "config": asdict(config), "evidence": saved})
        raise
    finally:
        connection.close()
