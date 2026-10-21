"""Optical proposals with shared, source-resolution edge metrology.

Detection copies are disposable. Widths and evidence always refer to the input
frame, never a threshold mask, resized preview, or an enhanced display image.
The two CU-compatible algorithms are deliberately not changed by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from fdm.services.cu_automatic_measurement import CuAutomaticCancelled

ALGORITHM_REVISION = 1
DEFAULT_MEMORY_BUDGET = 2 * 1024**3


@dataclass(frozen=True)
class OpticalParameters:
    """Revisioned implementation parameters, not material-specific limits."""

    revision: int = 1
    noise_floor: float = 0.5
    foreground_noise_multiple: float = 4.0
    foreground_floor: float = 2.0
    detection_sigma: float = 1.0
    ambiguous_core_noise: float = 6.0
    candidate_output_bytes: int = 16384
    min_component_area: int = 20
    min_axis_length: float = 12.0
    min_axis_ratio: float = 2.0
    max_width_fraction: float = 0.45
    pair_angle_degrees: float = 12.0
    pair_min_overlap: float = 0.65
    pair_rim_fraction: float = 0.34
    curved_axis_width_ratio: float = 1.5
    min_local_span: float = 24.0
    sections: int = 11
    section_step: float = 0.25
    edge_search_fraction: float = 0.24
    edge_search_minimum: float = 2.5
    section_span_widths: float = 2.5
    max_section_span: float = 80.0
    weak_edge_snr: float = 5.0
    variable_width_cv: float = 0.12
    duplicate_angle_degrees: float = 10.0
    duplicate_width_fraction: float = 0.20
    temporary_bytes_per_pixel: int = 112
    noise_blur_sigma: float = 0.8
    background_plane_samples: int = 12000
    background_plane_iterations: int = 3
    background_plane_percentile: float = 40.0
    component_width_percentiles: tuple = (2., 98.)
    core_width_percentile: float = 90.0
    junction_clearance_widths: float = 1.5
    local_axis_widths: float = 3.0
    pair_profile_samples: int = 101
    pair_outer_fraction: float = .15
    pair_core_fraction: float = .15
    profile_outer_margin: float = 3.0
    profile_tangent_offsets: tuple = (-.75, 0., .75)
    profile_smooth_kernel: int = 5
    profile_smooth_sigma: float = .8
    local_span_length_fraction: float = .45
    retained_min_deviation: float = 1.0
    retained_mad_multiple: float = 3.0
    duplicate_center_fraction: float = .15
    duplicate_length_fraction: float = .5
    contained_width_fraction: float = .5
    containment_margin: float = 2.0


PARAMETERS = OpticalParameters()


@dataclass
class _Hypothesis:
    center: np.ndarray
    tangent: np.ndarray
    width: float
    length: float
    structure: str
    identity: str
    parents: tuple[str, ...] = ()
    ambiguous: bool = False
    near_crossing: bool = False
    support_owner: str | None = None


def _check(checkpoint: Callable | None) -> None:
    if checkpoint is not None and checkpoint():
        raise CuAutomaticCancelled("全图分析已取消。")


def _gray(source: np.ndarray) -> np.ndarray:
    if source.ndim == 2:
        return source.astype(np.float32)
    code = cv2.COLOR_BGRA2GRAY if source.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(source, code).astype(np.float32)


def _noise(gray: np.ndarray, p: OpticalParameters) -> float:
    # Median high-pass residual estimates noise without treating a few dark
    # fibers as noise. It is a signal scale, not an accuracy probability.
    smooth = cv2.GaussianBlur(gray, (3, 3), p.noise_blur_sigma)
    residual = gray - smooth
    return max(p.noise_floor, float(np.median(np.abs(residual - np.median(residual)))) / 0.67448975)


def _prepare_local(gray: np.ndarray, max_width: float, p: OpticalParameters) -> tuple[np.ndarray, dict]:
    size = max(3, int(math.ceil(max_width * 2)) | 1)
    # Rectangular morphology is separable in OpenCV, unlike a very large disc.
    # Closing estimates slowly varying bright background without scaling data.
    smooth = cv2.GaussianBlur(gray, (0, 0), p.detection_sigma)
    # Remove a robust illumination plane before morphology. Closing a ramp
    # directly produces a false dark band at a replicated image border.
    stride = max(1, int(math.sqrt(gray.size / p.background_plane_samples)))
    yy, xx = np.mgrid[0:gray.shape[0]:stride, 0:gray.shape[1]:stride]
    design = np.column_stack((xx.ravel() / gray.shape[1], yy.ravel() / gray.shape[0], np.ones(xx.size)))
    values = smooth[::stride, ::stride].ravel()
    keep = np.ones(len(values), bool)
    for _ in range(p.background_plane_iterations):
        coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        residual = values - design @ coefficients
        keep = residual >= np.percentile(residual, p.background_plane_percentile)
    plane = (coefficients[0] * np.arange(gray.shape[1])[None, :] / gray.shape[1] +
             coefficients[1] * np.arange(gray.shape[0])[:, None] / gray.shape[0] + coefficients[2]).astype(np.float32)
    residual = smooth - plane
    background = plane + cv2.morphologyEx(residual, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8),
                                         borderType=cv2.BORDER_REPLICATE)
    contrast = np.maximum(background - smooth, 0)
    return contrast, {"background_method": "local_closing", "background_kernel": size,
                      "background_span": float(np.percentile(background, 95) - np.percentile(background, 5))}


def _components(mask: np.ndarray, prefix: str, max_width: float, p: OpticalParameters,
                checkpoint: Callable | None, maximum: int, owner_labels=None) -> list[_Hypothesis]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result: list[_Hypothesis] = []
    for label in range(1, count):
        _check(checkpoint)
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < p.min_component_area or max(w, h) < p.min_axis_length:
            continue
        roi = labels[y:y + h, x:x + w] == label
        support_owner = f"{prefix}{label}" if prefix == "body-" else None
        if owner_labels is not None:
            owners = np.unique(owner_labels[y:y+h, x:x+w][roi])
            if len(owners) == 1 and owners[0] > 0:
                support_owner = f"body-{int(owners[0])}"
        yy, xx = np.nonzero(roi)
        coordinates = np.column_stack((xx + x, yy + y)).astype(np.float64)
        center = coordinates.mean(axis=0)
        centered = coordinates - center
        eig, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
        if eig[-1] <= p.min_axis_ratio**2 * max(eig[0], 0.25):
            # Crossings/branches need separate local support. Do not fit one
            # diagonal across an entire intersection or swollen component.
            result.extend(_branched_components(roi, (x, y), prefix + str(label), max_width,
                                                p, checkpoint, support_owner))
            if len(result) > maximum:
                raise MemoryError("automatic_candidate_limit")
            continue
        tangent = vectors[:, -1]
        if tangent[0] < 0 or (abs(tangent[0]) < 1e-8 and tangent[1] < 0):
            tangent = -tangent
        normal = np.array([-tangent[1], tangent[0]])
        along, across = centered @ tangent, centered @ normal
        length = float(np.ptp(along)) + 1
        width = float(np.percentile(across, p.component_width_percentiles[1]) - np.percentile(across, p.component_width_percentiles[0])) + 1
        distance = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        core_width = 2 * float(np.percentile(distance[roi], p.core_width_percentile))
        if width > p.curved_axis_width_ratio * max(core_width, 1.):
            result.extend(_branched_components(roi, (x, y), prefix + str(label), max_width, p, checkpoint, support_owner))
            if len(result) > maximum:
                raise MemoryError("automatic_candidate_limit")
            continue
        if width <= max_width and length >= p.min_axis_length:
            result.append(_Hypothesis(center, tangent, width, length, "solid", f"{prefix}{label}", support_owner=support_owner))
        if len(result) > maximum:
            raise MemoryError("automatic_candidate_limit")
    return result


def _branched_components(mask, origin, prefix, max_width, p, checkpoint, support_owner=None):
    from skimage.morphology import skeletonize
    _check(checkpoint)
    skeleton = skeletonize(mask).astype(np.uint8)
    degree = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16)) - skeleton
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    junctions = np.argwhere((degree > 2) & (skeleton != 0))
    for y, x in junctions:
        cv2.circle(skeleton, (int(x), int(y)), max(2, int(distance[y, x] * p.junction_clearance_widths)), 0, -1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    result = []
    for label in range(1, count):
        _check(checkpoint)
        if stats[label, cv2.CC_STAT_AREA] < p.min_axis_length:
            continue
        yy, xx = np.nonzero(labels == label)
        remaining = set(zip(xx.tolist(), yy.tolist()))
        def neighbors(point):
            x, y = point
            return [(x + dx, y + dy) for dx, dy in ((-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1))
                    if (x + dx, y + dy) in remaining]
        ends = [point for point in remaining if len(neighbors(point)) == 1]
        current = min(ends or remaining)
        ordered = []
        while current in remaining:
            if len(ordered) % 128 == 0:
                _check(checkpoint)
            remaining.remove(current)
            ordered.append(current)
            choices = neighbors(current)
            if not choices:
                break
            current = min(choices)
        if len(ordered) < p.min_axis_length:
            continue
        index = len(ordered) // 2
        cx, cy = ordered[index]
        width = 2 * float(distance[cy, cx])
        # Fit one bounded local shaft, never average the whole curved fiber.
        half = int(min(p.max_section_span, max(p.min_local_span, width * p.local_axis_widths)) / 2)
        points = np.asarray(ordered[max(0, index-half):index+half+1], float)
        center = np.asarray(ordered[index], float)
        _, vectors = np.linalg.eigh(np.cov(points.T))
        tangent = vectors[:, -1]
        length = float(np.ptp((points - center) @ tangent))
        if width <= max_width and length >= p.min_axis_length:
            result.append(_Hypothesis(center + origin, tangent, width, length,
                                       "solid", f"{prefix}-branch{label}", near_crossing=bool(len(junctions)), support_owner=support_owner))
    return result


def _sample(gray: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x = np.asarray(centers[..., 0], np.float32)
    y = np.asarray(centers[..., 1], np.float32)
    return cv2.remap(gray, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class _RimIndex:
    """Direction bins and local spatial cells; never an all-rims pair scan."""

    def __init__(self, rims, max_width, p, checkpoint):
        self.cell_size = max(1., max_width)
        self.bin_count = max(1, int(math.ceil(180. / p.pair_angle_degrees)))
        self.bin_width = 180. / self.bin_count
        self.bin_reach = int(math.ceil(p.pair_angle_degrees / self.bin_width))
        self.cells, self.max_lengths = {}, {}
        for index, rim in enumerate(rims):
            _check(checkpoint)
            direction = self.direction(rim.tangent)
            cell = tuple(np.floor(rim.center / self.cell_size).astype(int))
            self.cells.setdefault(direction, {}).setdefault(cell, []).append(index)
            self.max_lengths[direction] = max(self.max_lengths.get(direction, 0.), rim.length)

    def direction(self, tangent):
        angle = math.degrees(math.atan2(tangent[1], tangent[0])) % 180.
        return min(self.bin_count - 1, int(angle / self.bin_width))

    def query(self, rim, max_width, checkpoint):
        direction = self.direction(rim.tangent)
        normal = np.array([-rim.tangent[1], rim.tangent[0]])
        bins = {(direction + d) % self.bin_count for d in range(-self.bin_reach, self.bin_reach + 1)}
        for bucket in sorted(bins):
            cells = self.cells.get(bucket, {})
            if not cells:
                continue
            # A true overlap needs centers within the sum of half lengths;
            # transverse separation is bounded by the configured search width.
            along = (rim.length + self.max_lengths[bucket]) / 2
            extent = np.abs(rim.tangent) * along + np.abs(normal) * max_width
            lo = np.floor((rim.center - extent) / self.cell_size).astype(int)
            hi = np.floor((rim.center + extent) / self.cell_size).astype(int)
            cell_count = int(np.prod(hi - lo + 1))
            if cell_count < len(cells):
                keys = ((x, y) for x in range(lo[0], hi[0] + 1) for y in range(lo[1], hi[1] + 1))
            else:
                keys = (key for key in cells if lo[0] <= key[0] <= hi[0] and lo[1] <= key[1] <= hi[1])
            for key in keys:
                _check(checkpoint)
                yield from cells.get(key, ())


def _pairs(rims: list[_Hypothesis], gray: np.ndarray, noise: float, max_width: float,
           p: OpticalParameters, checkpoint, maximum: int, owner_widths=None, diagnostics=None) -> list[_Hypothesis]:
    result = []
    index = _RimIndex(rims, max_width, p, checkpoint)
    owners = owner_widths or {}
    examined = 0
    for i, a in enumerate(rims):
        _check(checkpoint)
        normal = np.array([-a.tangent[1], a.tangent[0]])
        neighbors = []
        for j in index.query(a, max_width, checkpoint):
            b = rims[j]
            examined += 1
            if i == j:
                continue
            angle = math.degrees(math.acos(min(1., abs(float(a.tangent @ b.tangent)))))
            if angle > p.pair_angle_degrees:
                continue
            displacement = b.center - a.center
            separation = abs(float(displacement @ normal))
            width = separation + (a.width + b.width) / 2
            if width > max_width or separation <= max(a.width, b.width):
                continue
            if max(a.width, b.width) > width * p.pair_rim_fraction:
                continue
            overlap = (a.length + b.length) / 2 - abs(float(displacement @ a.tangent))
            if overlap < min(a.length, b.length) * p.pair_min_overlap:
                continue
            if a.support_owner != b.support_owner:
                # A rim already explained as one edge of a complete body must
                # not be paired across the background with another body's rim.
                # Whole narrow fibers remain ambiguous alternatives; no fixed
                # pixel gap or material-specific width is used to remove them.
                owner_a, owner_b = owners.get(a.support_owner), owners.get(b.support_owner)
                if ((owner_a is not None and a.width < owner_a * p.contained_width_fraction)
                        or (owner_b is not None and b.width < owner_b * p.contained_width_fraction)):
                    continue
            neighbors.append((separation, j, width, overlap))
        # Only the nearest compatible side on either side can be a partner.
        # Considering distant outer rims freely would merge adjacent fibers.
        nearest = {}
        for separation, j, width, overlap in sorted(neighbors):
            side = int(np.sign((rims[j].center - a.center) @ normal))
            nearest.setdefault(side, (separation, j, width, overlap))
        for separation, j, width, overlap in nearest.values():
            if j <= i:
                continue
            b = rims[j]
            center = (a.center + b.center) / 2
            positions = np.linspace(-width, width, p.pair_profile_samples)
            values = _sample(gray, (center + positions[:, None] * normal)[None, ...])[0]
            edge_count = max(1, int(len(values) * p.pair_outer_fraction))
            core_count = max(1, int(len(values) * p.pair_core_fraction))
            middle = len(values) // 2
            outer = np.r_[values[:edge_count], values[-edge_count:]]
            core = values[middle-core_count//2:middle+core_count//2+1]
            darkness = float(np.median(outer) - np.median(core))
            # A gap indistinguishable from outside background is ambiguous,
            # not affirmative evidence that two real dark fibers are one.
            ambiguous = darkness < max(noise * p.ambiguous_core_noise, p.foreground_floor)
            result.append(_Hypothesis(center, a.tangent, width, min(a.length, b.length, overlap),
                                       "double_edge", f"pair-{a.identity}-{b.identity}",
                                       (a.identity, b.identity), ambiguous))
            if len(result) > maximum:
                raise MemoryError("automatic_candidate_limit")
    if diagnostics is not None:
        diagnostics.update(index="direction_spatial_grid", direction_bins=index.bin_count,
                           rims=len(rims), examined=examined, all_pairs=len(rims)**2)
    return result


def _peak(positions, gradient, expected, sign, half_window):
    valid = np.flatnonzero((np.abs(positions - expected) <= half_window))
    if not len(valid):
        return None
    strength = gradient[valid] * sign
    index = int(valid[int(np.argmax(strength))])
    if index == 0 or index == len(gradient) - 1 or gradient[index] * sign <= 0:
        return None
    before, peak, after = gradient[index - 1:index + 2] * sign
    denominator = before - 2 * peak + after
    offset = 0. if abs(denominator) < 1e-12 else float(np.clip(.5 * (before - after) / denominator, -.5, .5))
    position = float(positions[index] + offset * (positions[index + 1] - positions[index]))
    return position, float(gradient[index] * sign)


def _refine(h: _Hypothesis, gray, noise, p: OpticalParameters, checkpoint):
    _check(checkpoint)
    tangent = h.tangent / np.linalg.norm(h.tangent)
    normal = np.array([-tangent[1], tangent[0]])
    half_window = max(p.edge_search_minimum, h.width * p.edge_search_fraction)
    extent = h.width / 2 + half_window + p.profile_outer_margin
    positions = np.arange(-extent, extent + p.section_step / 2, p.section_step)
    span = min(p.max_section_span, h.width * p.section_span_widths, h.length * p.local_span_length_fraction)
    sections = np.linspace(-span / 2, span / 2, p.sections)
    centers = h.center + sections[:, None] * tangent
    grid = centers[:, None, :] + positions[None, :, None] * normal
    # Average only a narrow strip along the tangent; no change in normal scale.
    profiles = sum(_sample(gray, grid + d * tangent) for d in p.profile_tangent_offsets) / len(p.profile_tangent_offsets)
    profiles = cv2.GaussianBlur(profiles, (p.profile_smooth_kernel, 1), p.profile_smooth_sigma)
    gradient = np.gradient(profiles, p.section_step, axis=1)
    samples, edge_strength = [], []
    height, width = gray.shape
    for idx, center in enumerate(centers):
        _check(checkpoint)
        left = _peak(positions, gradient[idx], -h.width / 2, -1, half_window)
        right = _peak(positions, gradient[idx], h.width / 2, 1, half_window)
        if left is None or right is None or right[0] <= left[0]:
            continue
        a, b = center + left[0] * normal, center + right[0] * normal
        if any(x < 1 or y < 1 or x >= width - 1 or y >= height - 1 for x, y in (a, b)):
            continue
        measured = float(right[0] - left[0])
        samples.append({"center": center.tolist(), "start": a.tolist(), "end": b.tolist(),
                        "raw_width": measured, "corrected_width": measured, "retained": True})
        edge_strength.append(min(left[1], right[1]))
    if not samples:
        return None
    widths = np.array([s["raw_width"] for s in samples])
    median = float(np.median(widths))
    mad = float(np.median(np.abs(widths - median)))
    retained = np.abs(widths - median) <= max(p.retained_min_deviation, p.retained_mad_multiple * 1.4826 * mad)
    for sample, keep in zip(samples, retained):
        sample["retained"] = bool(keep)
    chosen = [s for s in samples if s["retained"]]
    a = np.array([s["start"] for s in chosen], dtype=np.float32)
    b = np.array([s["end"] for s in chosen], dtype=np.float32)
    if len(chosen) >= 3:
        la = cv2.fitLine(a, cv2.DIST_HUBER, 0, .01, .01).ravel()
        lb = cv2.fitLine(b, cv2.DIST_HUBER, 0, .01, .01).ravel()
        angle = math.degrees(math.acos(np.clip(abs(float(la[:2] @ lb[:2])), 0, 1)))
        # Fitted edges describe parallelism. They must not silently replace a
        # sampled section by its projection while leaving a longer line drawn.
        measured = float(np.median(widths[retained]))
    else:
        angle, measured = 0., float(np.median(widths[retained]))
    if not math.isfinite(measured) or measured <= 0:
        return None
    representative = min(chosen, key=lambda s: abs(s["raw_width"] - measured) +
                         1e-4 * float(np.linalg.norm(np.asarray(s["center"]) - h.center)))
    snr = float(np.median(edge_strength)) / noise
    cv = float(np.std(widths) / max(np.mean(widths), 1e-6))
    warnings = []
    if snr < p.weak_edge_snr:
        warnings.append("low_edge_support")
    if cv > p.variable_width_cv:
        warnings.append("variable_width")
    if h.ambiguous:
        warnings.append("ambiguous_pair")
    if h.near_crossing:
        warnings.append("near_crossing")
    outer = np.median(np.r_[profiles[:, :4].ravel(), profiles[:, -4:].ravel()])
    core = np.median(profiles[:, np.abs(positions) < max(1., h.width * .15)])
    return {"hypothesis_id": h.identity, "structure": h.structure,
            "width_px": measured, "start": representative["start"], "end": representative["end"],
            "path": [s["center"] for s in samples], "samples": samples, "pass": 0,
            "quality_features": {"edge_snr": snr, "width_cv": cv,
                                 "edge_parallel_degrees": angle, "valid_sections": len(chosen),
                                 "total_sections": p.sections, "background_contrast": float(outer - core)},
            "warning_codes": warnings, "diagnostics": {"structure": h.structure,
                "warning_codes": list(warnings), "edge_snr": snr, "width_cv": cv,
                "edge_parallel_degrees": angle, "valid_sections": len(chosen),
                "total_sections": p.sections, "background_contrast": float(outer - core)},
            "conflict_group_id": None, "conflicts_with": [],
            "_center": h.center, "_tangent": tangent, "_parents": h.parents,
            "_length": h.length, "_owner": h.support_owner,
            "_ambiguous_hypothesis": h.ambiguous,
            "_complete_body_ids": [h.identity] if h.identity.startswith("body-") and "-branch" not in h.identity and not h.near_crossing else []}


def _merge(candidates, p: OpticalParameters, checkpoint):
    # Prefer a well-supported full section to a rim-only proposal at the same
    # location. Ambiguous pairs retain their independent alternatives.
    candidates.sort(key=lambda c: (-c["width_px"], -c["quality_features"]["edge_snr"]))
    output = []
    for c in candidates:
        _check(checkpoint)
        absorbed = False
        for prior in output:
            # A high-threshold rim belongs to one exact low-threshold region,
            # even when its local representative is at the remote end cap. This
            # is support ownership, never a larger spatial-neighbor tolerance.
            if (c["hypothesis_id"].startswith("rim-") and c.get("_owner") in prior["_complete_body_ids"]
                    and not prior["_ambiguous_hypothesis"]
                    and "near_crossing" not in prior["warning_codes"]):
                absorbed = True
                break
            angle = math.degrees(math.acos(np.clip(abs(float(c["_tangent"] @ prior["_tangent"])), 0, 1)))
            if angle > p.duplicate_angle_degrees:
                continue
            offset = c["_center"] - prior["_center"]
            along = abs(float(offset @ prior["_tangent"]))
            normal = np.array([-prior["_tangent"][1], prior["_tangent"][0]])
            across = abs(float(offset @ normal))
            if along > min(c["_length"], prior["_length"]) * p.duplicate_length_fraction:
                continue
            if across < max(1., prior["width_px"] * p.duplicate_center_fraction) and abs(c["width_px"] - prior["width_px"]) <= max(1., prior["width_px"] * p.duplicate_width_fraction):
                if c["structure"] == "double_edge":
                    prior["structure"] = "double_edge"
                prior["warning_codes"] = sorted(set(prior["warning_codes"] + c["warning_codes"]))
                prior["_ambiguous_hypothesis"] = prior["_ambiguous_hypothesis"] or c["_ambiguous_hypothesis"]
                prior["_complete_body_ids"] = sorted(set(prior["_complete_body_ids"] + c["_complete_body_ids"]))
                absorbed = True
                break
            contained = across + c["width_px"] / 2 <= prior["width_px"] / 2 + p.containment_margin
            if contained and c["width_px"] < prior["width_px"] * p.contained_width_fraction:
                if not prior["_ambiguous_hypothesis"]:
                    absorbed = True
                    break
                group = prior["conflict_group_id"] or "conflict-" + prior["hypothesis_id"]
                prior["conflict_group_id"] = c["conflict_group_id"] = group
                prior["conflicts_with"].append(c["hypothesis_id"])
                c["conflicts_with"].append(prior["hypothesis_id"])
                if "ambiguous_pair" not in c["warning_codes"]:
                    c["warning_codes"].append("ambiguous_pair")
            # Ambiguous whole-width alternatives are ordered after their
            # constituent fibers. Link both directions without deleting either.
            reverse_contained = across + prior["width_px"] / 2 <= c["width_px"] / 2 + p.containment_margin
            if reverse_contained and prior["width_px"] < c["width_px"] * p.contained_width_fraction and c["_ambiguous_hypothesis"]:
                group = c["conflict_group_id"] or "conflict-" + c["hypothesis_id"]
                c["conflict_group_id"] = prior["conflict_group_id"] = group
                c["conflicts_with"].append(prior["hypothesis_id"])
                prior["conflicts_with"].append(c["hypothesis_id"])
                if "ambiguous_pair" not in prior["warning_codes"]:
                    prior["warning_codes"].append("ambiguous_pair")
        if not absorbed:
            output.append(c)
    output.sort(key=lambda c: (round(float(c["_center"][1]), 3), round(float(c["_center"][0]), 3), c["width_px"]))
    surviving = {c["hypothesis_id"] for c in output}
    for c in output:
        c["conflicts_with"] = sorted(set(c["conflicts_with"]) & surviving)
        if not c["conflicts_with"]:
            c["conflict_group_id"] = None
        c["diagnostics"].update(structure=c["structure"], warning_codes=list(c["warning_codes"]),
            conflict_group_id=c["conflict_group_id"], conflicts_with=list(c["conflicts_with"]))
        for key in list(c):
            if key.startswith("_"):
                del c[key]
    return output


def analyze_optical_frame(image, *, config: dict | None = None, checkpoint=None,
                          max_candidates: int = 10000, memory_budget: int = DEFAULT_MEMORY_BUDGET) -> dict:
    """Return proposals, never measurements accepted into a trial ledger.

    Quality findings are explanatory; they do not introduce a second acceptance
    policy. Invalid/incomplete geometry is not replaced with a free-line width.
    """
    started = perf_counter()
    source = np.asarray(image)
    if source.dtype != np.uint8:
        raise TypeError("全图分析图像必须为 uint8。")
    if source.ndim not in (2, 3) or (source.ndim == 3 and source.shape[2] not in (3, 4)):
        raise ValueError("全图分析图像必须为灰度、BGR 或 BGRA。")
    if min(source.shape[:2]) < 16 or max(source.shape[:2]) > 32766:
        raise ValueError("全图分析图像尺寸必须在 16～32766 像素范围内。")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 1000000:
        raise ValueError("max_candidates must be in 1..1000000")
    if isinstance(memory_budget, bool) or not isinstance(memory_budget, int) or memory_budget <= 0:
        raise ValueError("memory_budget must be a positive integer")
    config = dict(config or {})
    mode = config.get("imaging_mode", "mixed")
    if mode not in {"mixed", "solid", "double_edge"}:
        raise ValueError("未知的光学成像模式。")
    background_mode = config.get("background_mode", "local")
    if background_mode != "local":
        raise ValueError("当前光学背景模式尚未配置。")
    p = PARAMETERS
    from fdm.services.optical_flat_field import prepare_detection_image, estimate_flat_field_scratch_bytes
    flat_bytes = estimate_flat_field_scratch_bytes(source.shape, operation="apply")
    minimum_bytes = source.nbytes + source.shape[0] * source.shape[1] * p.temporary_bytes_per_pixel + flat_bytes
    diagnostics = {"algorithm": "optical_adaptive", "revision": ALGORITHM_REVISION,
                   "parameters_revision": p.revision, "parameters": asdict(p),
                   "input_shape": tuple(source.shape), "imaging_mode": mode,
                   "backend": "optical_source_geometry", "memory_budget": memory_budget,
                   "minimum_reserved_bytes": minimum_bytes, "stage_ms": {}}

    def finish(status, reason, candidates):
        diagnostics["total_ms"] = (perf_counter() - started) * 1000
        return {"status": status, "reason": reason, "candidates": candidates, "diagnostics": diagnostics}

    _check(checkpoint)
    if minimum_bytes > memory_budget:
        return finish("resource_limit", "automatic_memory_budget", [])
    try:
        prepared = prepare_detection_image(source, config, memory_budget=flat_bytes,
                                           checkpoint=lambda: _check(checkpoint))
        diagnostics.update(prepared["diagnostics"])
        if prepared["status"] == "resource_limit":
            return finish("resource_limit", "automatic_memory_budget", [])
        if prepared["status"] == "cancelled":
            raise CuAutomaticCancelled("全图分析已取消。")
        gray = _gray(source)
        detection_gray = _gray(prepared["image"])
        del prepared
        noise = _noise(gray, p)
        max_width = float(config.get("max_width_px", min(gray.shape) * p.max_width_fraction))
        if not math.isfinite(max_width) or max_width < 2 or max_width > max(gray.shape):
            raise ValueError("光学搜索宽度无效。")
        contrast, lighting = _prepare_local(detection_gray, max_width, p)
        del detection_gray
        diagnostics["quality"] = {**lighting, "noise_gray": noise,
                                  "saturated_fraction": float(np.mean((gray <= 1) | (gray >= 254)))}
        _check(checkpoint)
        threshold = max(p.foreground_floor, p.foreground_noise_multiple * noise)
        low_mask = np.asarray(contrast >= threshold, np.uint8)
        scaled = np.clip(contrast, 0, 255).astype(np.uint8)
        otsu, high_mask = cv2.threshold(scaled, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        high_mask[contrast < threshold] = 0
        diagnostics["quality"].update(detection_floor=threshold, histogram_threshold=float(otsu))
        diagnostics["stage_ms"]["preparation"] = (perf_counter() - started) * 1000
        before = perf_counter()
        bodies = _components(low_mask, "body-", max_width, p, checkpoint, max_candidates)
        _, owner_labels = cv2.connectedComponents(low_mask, connectivity=8)
        rims = _components(high_mask, "rim-", max_width, p, checkpoint, max_candidates, owner_labels)
        del owner_labels
        hypotheses = bodies + rims if mode != "double_edge" else []
        # Detection smoothing expands a 3 px rim into a wider low-threshold
        # component. Only its original-image metrology can establish that it
        # belongs to a genuinely wider body, rather than another narrow fiber.
        body_cache, owner_widths = {}, {}
        for body in bodies:
            if body.near_crossing or "-branch" in body.identity:
                continue
            refined = _refine(body, gray, noise, p, checkpoint)
            body_cache[body.identity] = refined
            if refined is not None:
                owner_widths[body.identity] = refined["width_px"]
                if minimum_bytes + len(body_cache) * p.candidate_output_bytes > memory_budget:
                    raise MemoryError("automatic_memory_budget")
        diagnostics["pairing"] = {}
        pairs = _pairs(rims, gray, noise, max_width, p, checkpoint, max_candidates,
                       owner_widths, diagnostics["pairing"]) if mode != "solid" else []
        hypotheses += pairs
        if len(hypotheses) > max_candidates * 4:
            raise MemoryError("automatic_candidate_limit")
        diagnostics["counts"] = {"solid": len(bodies) + len(rims), "double_edge": len(pairs),
                                 "hypotheses": len(hypotheses)}
        diagnostics["stage_ms"]["hypotheses"] = (perf_counter() - before) * 1000
        before = perf_counter()
        candidates = []
        for hypothesis in hypotheses:
            candidate = body_cache.pop(hypothesis.identity) if hypothesis.identity in body_cache else _refine(hypothesis, gray, noise, p, checkpoint)
            if candidate:
                if minimum_bytes + (len(body_cache) + len(candidates) + 1) * p.candidate_output_bytes > memory_budget:
                    raise MemoryError("automatic_memory_budget")
                candidates.append(candidate)
        candidates = _merge(candidates, p, checkpoint)
        if len(candidates) > max_candidates:
            raise MemoryError("automatic_candidate_limit")
        diagnostics["counts"]["candidates"] = len(candidates)
        diagnostics["stage_ms"]["metrology"] = (perf_counter() - before) * 1000
        _check(checkpoint)
        return finish("ok" if candidates else "empty", "" if candidates else "no_valid_fibers", candidates)
    except MemoryError as exc:
        return finish("resource_limit", str(exc) or "automatic_memory_budget", [])
