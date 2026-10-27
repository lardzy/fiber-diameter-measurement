"""Calibrated, read-only evaluation of professional measurements.

The direction of every dependency is image/construction -> feature -> result.
No result is written back into a fibre measurement, a source image, or a fit ROI.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from fdm.construction_geometry import ConstructionIssue, ResolvedCircle, ResolvedLine, ResolvedPoint
from fdm.geometry import Point
from fdm.metrology import (
    MeasurementDefinition, MeasurementFeature, MeasurementMetric,
    MeasurementPrimitive, MeasurementResult, ResolvedMeasurementFeature,
)

_EPS = 1e-10
_DOCUMENT_RESULTS: OrderedDict[int, tuple[str, dict[str, MeasurementResult]]] = OrderedDict()


def calibration_scale(document: Any) -> tuple[float, float]:
    calibration = document.calibration
    return (calibration.pixel_size_x, calibration.pixel_size_y) if calibration else (1.0, 1.0)


def _p(value: Any) -> Point:
    if isinstance(value, Point):
        return value
    return Point(float(value[0]), float(value[1]))


def _array(point: Point) -> np.ndarray:
    return np.array((point.x, point.y), dtype=float)


def _distance(a: Point, b: Point) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def _mid(a: Point, b: Point) -> Point:
    return Point((a.x + b.x) / 2, (a.y + b.y) / 2)


def _scaled(point: Point, scale: tuple[float, float]) -> Point:
    return Point(point.x * scale[0], point.y * scale[1])


def _pixel(point: Point, scale: tuple[float, float]) -> Point:
    return Point(point.x / scale[0], point.y / scale[1])


def _circle_three(points: tuple[Point, ...]) -> tuple[Point, float]:
    if len(points) != 3:
        raise ValueError("三点圆/圆弧需要恰好三个点")
    a, b, c = (_array(point) for point in points)
    # Translate before solving to retain precision at large slide coordinates.
    u, v = b - a, c - a
    matrix = 2.0 * np.array((u, v))
    span = max(float(np.linalg.norm(u)), float(np.linalg.norm(v)))
    if span <= _EPS or abs(float(np.linalg.det(matrix))) <= _EPS * span * span:
        raise ValueError("三点重合或共线，不能确定唯一圆")
    offset = np.linalg.solve(matrix, np.array((u @ u, v @ v)))
    radius = float(np.linalg.norm(offset))
    if not math.isfinite(radius) or radius <= _EPS:
        raise ValueError("圆半径无效")
    return _p(a + offset), radius


def _source_revision(document: Any) -> str | None:
    runtime_revision = document.professional_source_revision
    if runtime_revision is not None:
        return runtime_revision
    slide = document.metadata.get("digital_slide", {})
    revision = slide.get("stitching_revision", slide.get("source_revision")) if isinstance(slide, dict) else None
    return str(revision) if revision is not None else None


def professional_source_unavailable_reason(document: Any) -> str | None:
    if not document.is_digital_slide():
        return None
    if document.professional_source_revision_pending:
        return "正在核对切片来源，请等待完成后测量"
    if document.professional_source_revision_error:
        return document.professional_source_revision_error
    if _source_revision(document) is None:
        return "尚未核对切片来源，请重新打开后测量"
    return None


def resolve_feature(document: Any, feature: MeasurementFeature | str) -> ResolvedMeasurementFeature:
    """Resolve live references, then fit/measure in the physical X/Y plane."""
    if isinstance(feature, str):
        candidate = document.get_professional_feature(feature)
        if candidate is None:
            return ResolvedMeasurementFeature(feature, "", (), (), error="来源特征已删除或不存在")
        feature = candidate
    points_px = tuple(Point(p.x, p.y) for p in feature.points_px)
    scale = calibration_scale(document)
    quality = copy.deepcopy(feature.quality)
    try:
        if feature.image_id != document.id:
            raise ValueError("不能引用其它图像的特征")
        unavailable = professional_source_unavailable_reason(document)
        if unavailable:
            raise ValueError(unavailable)
        if document.is_digital_slide() and not feature.source_revision:
            raise ValueError("该专业特征缺少切片来源版本，请复核并重新建立测量")
        current_revision = _source_revision(document)
        if document.is_digital_slide() and feature.source_revision and current_revision is None:
            raise ValueError("尚未核对切片来源，请重新打开后测量")
        if feature.source_revision and current_revision and feature.source_revision != current_revision:
            raise ValueError("切片拼接/来源版本已改变，请在当前来源重新拟合")
        if feature.source_ref is not None:
            from fdm.construction_document import make_construction_resolver

            shape = make_construction_resolver(document).resolve_feature(feature.source_ref)
            if isinstance(shape, ConstructionIssue):
                raise ValueError(shape.message)
            if isinstance(shape, ResolvedPoint):
                if feature.kind != "point":
                    raise ValueError("引用对象不是所需几何类型")
                points_px = (shape.point,)
            elif isinstance(shape, ResolvedLine):
                if feature.kind not in {"line", "fit_line", "caliper_line"}:
                    raise ValueError("引用对象不是直线")
                points_px = (shape.start, shape.end)
            elif isinstance(shape, ResolvedCircle):
                if feature.kind not in {"circle", "fit_circle", "caliper_circle"}:
                    raise ValueError("引用对象不是圆")
                if not math.isclose(scale[0], scale[1], rel_tol=1e-12):
                    raise ValueError("非等方标定下，辅助像素圆没有唯一物理半径；请使用原始边缘点重新拟合")
                center = _scaled(shape.center, scale)
                points_px = tuple(Point(shape.center.x + shape.radius * math.cos(t), shape.center.y + shape.radius * math.sin(t)) for t in (0, 2 * math.pi / 3, 4 * math.pi / 3))
                return ResolvedMeasurementFeature(feature.id, "circle", tuple(_scaled(p, scale) for p in points_px), points_px, center, shape.radius * scale[0], quality)
            else:
                raise ValueError("请选择单个辅助点、线或圆")
        if feature.parameters.get("legacy_pixel_circle") and not math.isclose(scale[0], scale[1], rel_tol=1e-12):
            raise ValueError("冻结的辅助像素圆在非等方标定下没有唯一物理半径")
        points = tuple(_scaled(p, scale) for p in points_px)
        kind = feature.kind
        if kind.startswith("caliper_") and quality.get("status", "ready") != "ready":
            raise ValueError("原始卡尺提取未成功：" + str(quality.get("reason") or quality["status"]))
        if kind == "point":
            if len(points) != 1:
                raise ValueError("点特征需要一个控制点")
        elif kind in {"line", "polyline", "polygon", "caliper_width"}:
            minimum = 3 if kind == "polygon" else 2
            if len(points) < minimum:
                raise ValueError(f"几何对象至少需要 {minimum} 个点")
            if kind == "line" and (len(points) != 2 or _distance(*points) <= _EPS):
                raise ValueError("直线需要两个不重合的点")
            if kind == "caliper_width" and len(points) % 2:
                raise ValueError("双边卡尺控制点必须成对保存")
        elif kind in {"circle", "arc"}:
            center, radius = _circle_three(points)
            return ResolvedMeasurementFeature(feature.id, kind, points, points_px, center, radius, quality)
        elif kind in {"fit_line", "fit_circle", "caliper_line", "caliper_circle"}:
            from fdm.services.metrology_fitting import CaliperOptions, fit_circle_points, fit_line_points

            circular = kind in {"fit_circle", "caliper_circle"}
            options = feature.parameters.get("caliper", {})
            if not isinstance(options, dict):
                raise ValueError("卡尺参数必须是对象")
            robust = feature.parameters.get("robust", options.get("robust", True))
            if type(robust) is not bool:
                raise ValueError("稳健拟合参数必须是布尔值")
            caliper_options = CaliperOptions.from_dict(options) if kind.startswith("caliper_") else None
            fit = (fit_circle_points if circular else fit_line_points)(points_px, scale=scale, robust=robust)
            if not fit.success:
                raise ValueError(fit.reason or "拟合失败")
            # Preserve acquisition evidence under its original calibration;
            # never overwrite it with a fit evaluated after calibration edits.
            quality = {"source_evidence": quality, "evaluated_fit": fit.to_dict()}
            if caliper_options is not None:
                profiles = quality["source_evidence"].get("profiles", [])
                total_profiles = len(profiles) if isinstance(profiles, (tuple, list)) else 0
                if total_profiles and len(fit.accepted_indices) / total_profiles < caliper_options.min_valid_fraction:
                    raise ValueError("当前物理标定下剔除异常点后的有效截面比例不足")
                if fit.max_residual is not None and fit.max_residual / min(scale) > caliper_options.max_residual_px:
                    raise ValueError("当前物理标定下拟合残差超过设置的像素阈值")
                if circular and fit.coverage_deg is not None and fit.coverage_deg < caliper_options.min_circle_coverage_deg:
                    raise ValueError("当前物理标定下圆周覆盖范围不足")
            if circular:
                return ResolvedMeasurementFeature(feature.id, "circle", points, points_px, _p(fit.geometry["center"]), float(fit.geometry["radius"]), quality)
            endpoints = tuple(_p(p) for p in fit.geometry["endpoints"])
            return ResolvedMeasurementFeature(feature.id, "line", endpoints, tuple(_pixel(p, scale) for p in endpoints), quality=quality)
        return ResolvedMeasurementFeature(feature.id, kind, points, points_px, quality=quality)
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        return ResolvedMeasurementFeature(feature.id, feature.kind, (), points_px, quality=quality, error=str(error))


def freeze_feature(document: Any, feature: MeasurementFeature | str) -> MeasurementFeature:
    """Explicitly detach a live construction reference; preserve its provenance."""
    if isinstance(feature, str):
        feature = document.get_professional_feature(feature)
    if feature is None:
        raise ValueError("来源特征不存在")
    resolved = resolve_feature(document, feature)
    if not resolved.valid:
        raise ValueError(resolved.error)
    if feature.source_ref is None:
        return feature.clone()
    parameters = dict(feature.parameters)
    parameters["frozen_source"] = feature.source_ref.to_dict()
    parameters["frozen_calibration"] = document.calibration.to_dict() if document.calibration else None
    if resolved.center is not None:
        parameters["legacy_pixel_circle"] = True
    return feature.clone(source_ref=None, kind=resolved.kind, points_px=resolved.points_px, parameters=parameters)


def freeze_measurement(document: Any, definition: MeasurementDefinition) -> tuple[MeasurementFeature, ...]:
    """Return detached replacement features for an explicit undoable UI action."""
    return tuple(freeze_feature(document, feature_id) for feature_id in definition.feature_ids)


def _line(feature: ResolvedMeasurementFeature) -> tuple[Point, Point]:
    if feature.kind not in {"line", "fit_line", "caliper_line"} or len(feature.points) != 2:
        raise ValueError("此测量需要直线特征")
    if _distance(*feature.points) <= _EPS:
        raise ValueError("直线两个端点重合")
    return feature.points[0], feature.points[1]


def _point(feature: ResolvedMeasurementFeature) -> Point:
    if feature.kind != "point" or len(feature.points) != 1:
        raise ValueError("此测量需要点特征")
    return feature.points[0]


def _circle(feature: ResolvedMeasurementFeature) -> tuple[Point, float]:
    if feature.center is None or feature.radius is None:
        raise ValueError("此测量需要圆特征")
    return feature.center, feature.radius


def _direction(a: Point, b: Point) -> np.ndarray:
    vector = _array(b) - _array(a)
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        raise ValueError("零长度边没有方向")
    return vector / norm


def _foot(point: Point, line: tuple[Point, Point], segment: bool = False) -> Point:
    a, b = line
    vector = _array(b) - _array(a)
    square = float(vector @ vector)
    if square <= _EPS * _EPS:
        raise ValueError("零长度直线不能作投影")
    t = float((_array(point) - _array(a)) @ vector / square)
    if segment:
        t = min(1.0, max(0.0, t))
    return _p(_array(a) + t * vector)


def _self_intersects(points: tuple[Point, ...]) -> bool:
    def orientation(a: Point, b: Point, c: Point) -> float:
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    def on_segment(a: Point, b: Point, p: Point) -> bool:
        return abs(orientation(a, b, p)) <= _EPS and min(a.x, b.x) - _EPS <= p.x <= max(a.x, b.x) + _EPS and min(a.y, b.y) - _EPS <= p.y <= max(a.y, b.y) + _EPS

    count = len(points)
    for i in range(count):
        a, b = points[i], points[(i + 1) % count]
        if _distance(a, b) <= _EPS:
            return True
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            c, d = points[j], points[(j + 1) % count]
            o1, o2, o3, o4 = orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b)
            if (o1 * o2 < 0 and o3 * o4 < 0) or any((on_segment(a, b, c), on_segment(a, b, d), on_segment(c, d, a), on_segment(c, d, b))):
                return True
    return False


def _curve(center: Point, radius: float, start: float = 0.0, sweep: float = 2 * math.pi) -> tuple[Point, ...]:
    samples = max(12, int(abs(sweep) / (2 * math.pi) * 96))
    return tuple(Point(center.x + radius * math.cos(t), center.y + radius * math.sin(t)) for t in np.linspace(start, start + sweep, samples + 1))


def feature_overlay(document: Any, feature: MeasurementFeature | str) -> tuple[MeasurementPrimitive, ...]:
    resolved = resolve_feature(document, feature)
    if not resolved.valid:
        return ()
    if resolved.center is not None:
        start, sweep = 0.0, 2 * math.pi
        if resolved.kind == "arc":
            angles = [math.atan2(p.y - resolved.center.y, p.x - resolved.center.x) for p in resolved.points]
            positive = (angles[2] - angles[0]) % (2 * math.pi)
            middle = (angles[1] - angles[0]) % (2 * math.pi)
            start, sweep = angles[0], positive if middle <= positive + _EPS else positive - 2 * math.pi
        return (MeasurementPrimitive("polyline", tuple(_pixel(p, calibration_scale(document)) for p in _curve(resolved.center, resolved.radius, start, sweep)), "source"),)
    return (MeasurementPrimitive("point" if resolved.kind == "point" else "polyline", resolved.points_px, "source"),)


def evaluate_measurement(document: Any, definition: MeasurementDefinition, *, resolved_features: dict[str, ResolvedMeasurementFeature] | None = None) -> MeasurementResult:
    scale = calibration_scale(document)
    unit = document.calibration.unit if document.calibration else "px"
    features = [resolved_features.get(feature_id) if resolved_features is not None else resolve_feature(document, feature_id) for feature_id in definition.feature_ids]
    geometry: list[MeasurementPrimitive] = []
    metrics: list[MeasurementMetric] = []
    anchor = Point(0, 0)
    quality: dict[str, Any] = {"calibration": document.calibration.to_dict() if document.calibration else None, "angle_basis": "physical" if document.calibration else "pixel"}

    def metric(key: str, label: str, value: float, value_unit: str = unit) -> None:
        if not math.isfinite(value):
            raise ValueError("计算结果不是有限数")
        metrics.append(MeasurementMetric(key, label, float(value), value_unit))

    def draw(points: Iterable[Point], role: str = "measurement", kind: str = "line") -> None:
        geometry.append(MeasurementPrimitive(kind, tuple(_pixel(p, scale) for p in points), role))

    try:
        if definition.image_id != document.id:
            raise ValueError("测量所属图像不匹配")
        if any(feature is None or not feature.valid for feature in features):
            errors = [feature.error if feature is not None else "来源特征不存在" for feature in features if feature is None or not feature.valid]
            return MeasurementResult(definition.id, "invalid", "；".join(errors))
        source_rows = [document.get_professional_feature(feature_id) for feature_id in definition.feature_ids]
        layers = {row.focus_index for row in source_rows if row is not None and row.focus_index is not None}
        if document.is_digital_slide() and (definition.focus_index is None or any(row is not None and row.focus_index is None for row in source_rows)):
            raise ValueError("数字切片测量缺少来源焦层，不能用于当前层的有效测量")
        if len(layers) > 1 or (definition.focus_index is not None and layers and layers != {definition.focus_index}):
            raise ValueError("同一测量不能混合不同焦层的来源")
        quality["features"] = {feature.feature_id: feature.quality for feature in features if feature.quality}
        kind = definition.kind
        params = definition.parameters
        if "supplement" in params and type(params["supplement"]) is not bool:
            raise ValueError("补角参数必须是布尔值")
        counts = {
            "distance": {1, 2}, "point_line_distance": {2}, "point_segment_distance": {2},
            "parallel_distance": {2}, "projection": {2, 3}, "angle": {1, 3}, "line_angle": {2},
            "circle": {1}, "radius": {1}, "diameter": {1}, "arc": {1},
            "circle_center_distance": {2}, "circle_gap": {2}, "point_circle_distance": {2},
            "polyline": {1}, "polygon": {1}, "coordinates": {3}, "width": {1},
            "fit_line": {1}, "fit_circle": {1},
        }
        if len(features) not in counts.get(kind, set()):
            raise ValueError("测量来源数量与所选工具不匹配")
        if kind == "distance":
            a, b = _line(features[0]) if len(features) == 1 else (_point(features[0]), _point(features[1]))
            if _distance(a, b) <= _EPS:
                raise ValueError("距离测量的两个点重合")
            metric("distance", "距离", _distance(a, b))
            metric("dx", "水平分量", b.x - a.x)
            metric("dy", "垂直分量", b.y - a.y)
            draw((a, b)); anchor = _mid(a, b)
        elif kind in {"point_line_distance", "point_segment_distance"}:
            point, line = _point(features[0]), _line(features[1])
            foot = _foot(point, line, kind == "point_segment_distance")
            metric("distance", "点到线段距离" if kind == "point_segment_distance" else "点到直线垂距", _distance(point, foot))
            draw(line, "source"); draw((point, foot)); anchor = _mid(point, foot)
        elif kind == "parallel_distance":
            line_a, line_b = _line(features[0]), _line(features[1])
            u, v = _direction(*line_a), _direction(*line_b)
            tolerance = float(params.get("parallel_tolerance_deg", 1e-6))
            if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
                raise ValueError("平行角容差必须在 0～1° 范围内")
            deviation = math.degrees(math.asin(float(np.clip(abs(u[0] * v[1] - u[1] * v[0]), 0, 1))))
            if deviation > tolerance + 1e-8:
                raise ValueError("两直线不平行，不能给出唯一平行间距；请测夹角或指定截面")
            a = _mid(*line_a); b = _foot(a, line_b)
            metric("distance", "平行间距", _distance(a, b))
            quality["parallel_deviation_deg"] = deviation
            draw(line_a, "source"); draw(line_b, "source"); draw((a, b)); anchor = _mid(a, b)
        elif kind == "projection":
            if len(features) == 2:
                a, b = _line(features[0]); baseline = _line(features[1])
            else:
                a, b, baseline = _point(features[0]), _point(features[1]), _line(features[2])
            direction = _direction(*baseline)
            projected = float((_array(b) - _array(a)) @ direction)
            end = _p(_array(a) + direction * projected)
            metric("projection", "沿基准投影", projected)
            draw((a, end)); draw((b, end), "witness"); draw(baseline, "source"); anchor = _mid(a, end)
        elif kind in {"angle", "line_angle"}:
            if kind == "angle":
                a, vertex, b = features[0].points if len(features) == 1 else tuple(_point(feature) for feature in features)
                u, v = _direction(vertex, a), _direction(vertex, b)
                draw((a, vertex, b), "source", "polyline")
                radius = min(_distance(vertex, a), _distance(vertex, b)) * 0.28
            else:
                line_a, line_b = _line(features[0]), _line(features[1])
                u, v = _direction(*line_a), _direction(*line_b)
                if u @ v < 0:
                    v = -v
                matrix = np.column_stack((u, -v))
                if abs(float(np.linalg.det(matrix))) > _EPS:
                    t = np.linalg.solve(matrix, _array(line_b[0]) - _array(line_a[0]))[0]
                    vertex = _p(_array(line_a[0]) + t * u)
                else:
                    vertex = _mid(*line_a)
                radius = min(_distance(*line_a), _distance(*line_b)) * 0.28
                draw(line_a, "source"); draw(line_b, "source")
            angle = math.acos(float(np.clip(u @ v, -1, 1)))
            cross = float(u[0] * v[1] - u[1] * v[0])
            sweep = math.copysign(angle, cross if abs(cross) > _EPS else 1)
            if bool(params.get("supplement", False)):
                angle = math.pi - angle
                sweep = -math.copysign(angle, sweep)
                draw((vertex, _p(_array(vertex) - v * radius * 1.5)), "witness")
            metric("angle", "夹角" if kind == "line_angle" else "三点角", math.degrees(angle), "°")
            start = math.atan2(u[1], u[0]); arc = _curve(vertex, radius, start, sweep)
            draw(arc, "measurement", "polyline"); anchor = arc[len(arc) // 2]
            quality["angle_domain"] = {"start_rad": start, "sweep_rad": sweep, "supplement": bool(params.get("supplement", False))}
        elif kind in {"circle", "radius", "diameter", "fit_circle", "arc"}:
            center, radius = _circle(features[0])
            if kind == "arc":
                if features[0].kind != "arc" or len(features[0].points) != 3:
                    raise ValueError("圆弧需要起点、经过点和终点")
                angles = [math.atan2(p.y - center.y, p.x - center.x) for p in features[0].points]
                positive = (angles[2] - angles[0]) % (2 * math.pi)
                middle = (angles[1] - angles[0]) % (2 * math.pi)
                sweep = positive if middle <= positive + _EPS else positive - 2 * math.pi
                metric("arc_length", "弧长", abs(sweep) * radius)
                metric("radius", "半径", radius)
                metric("angle", "圆心角", math.degrees(abs(sweep)), "°")
                arc = _curve(center, radius, angles[0], sweep)
                draw(arc, "measurement", "polyline"); draw((features[0].points[0], center, features[0].points[2]), "witness", "polyline")
                anchor = arc[len(arc) // 2]
                quality["arc_sweep_rad"] = sweep
            else:
                metric("radius" if kind == "radius" else "diameter", "半径" if kind == "radius" else "直径", radius if kind == "radius" else 2 * radius)
                metric("diameter" if kind == "radius" else "radius", "直径" if kind == "radius" else "半径", 2 * radius if kind == "radius" else radius)
                metric("center_x", "圆心 X", center.x); metric("center_y", "圆心 Y", center.y)
                draw(_curve(center, radius), "source", "polyline")
                right = Point(center.x + radius, center.y)
                draw((center if kind == "radius" else Point(center.x - radius, center.y), right))
                anchor = right
        elif kind in {"circle_center_distance", "circle_gap"}:
            c1, r1 = _circle(features[0]); c2, r2 = _circle(features[1])
            distance = _distance(c1, c2)
            u = (_array(c2) - _array(c1)) / distance if distance > _EPS else np.array((1., 0.))
            if kind == "circle_center_distance":
                a, b = c1, c2; value = distance
            elif distance > r1 + r2:
                a, b = _p(_array(c1) + u * r1), _p(_array(c2) - u * r2); value = distance - r1 - r2
            elif distance < abs(r1 - r2):
                sign = 1 if r1 >= r2 else -1
                a, b = _p(_array(c1) + sign * u * r1), _p(_array(c2) + sign * u * r2); value = abs(r1 - r2) - distance
            else:
                value = 0.0
                if distance <= _EPS:
                    a = b = _p(_array(c1) + u * r1)
                else:
                    t = (r1 * r1 - r2 * r2 + distance * distance) / (2 * distance)
                    h = math.sqrt(max(0., r1 * r1 - t * t))
                    a = b = _p(_array(c1) + u * t + np.array((-u[1], u[0])) * h)
            metric("distance", "圆心距" if kind == "circle_center_distance" else "圆边界最短距", max(0., value))
            draw(_curve(c1, r1), "source", "polyline"); draw(_curve(c2, r2), "source", "polyline")
            draw((a, b)); anchor = _mid(a, b)
        elif kind == "point_circle_distance":
            point = _point(features[0]); center, radius = _circle(features[1])
            delta = _array(point) - _array(center); length = float(np.linalg.norm(delta))
            direction = delta / length if length > _EPS else np.array((1., 0.))
            edge = _p(_array(center) + radius * direction)
            metric("distance", "点到圆边界距离", abs(length - radius))
            draw(_curve(center, radius), "source", "polyline"); draw((point, edge)); anchor = _mid(point, edge)
        elif kind in {"polyline", "polygon"}:
            points = features[0].points
            if len(points) < (3 if kind == "polygon" else 2):
                raise ValueError("控制点不足")
            if kind == "polygon":
                if _self_intersects(points):
                    raise ValueError("多边形自交、重叠或含重复相邻点，面积无效")
                shifted = [_array(point) - _array(points[0]) for point in points]
                twice = sum(float(a[0] * b[1] - b[0] * a[1]) for a, b in zip(shifted, shifted[1:] + shifted[:1]))
                if abs(twice) <= _EPS:
                    raise ValueError("退化多边形不能确定面积")
                metric("area", "面积", abs(twice) / 2, f"{unit}²")
                metric("perimeter", "周长", sum(_distance(a, b) for a, b in zip(points, points[1:] + points[:1])))
                draw((*points, points[0]), kind="polyline")
            else:
                length = sum(_distance(a, b) for a, b in zip(points, points[1:]))
                if length <= _EPS:
                    raise ValueError("折线没有有效长度")
                metric("length", "长度", length); draw(points, kind="polyline")
            anchor = Point(sum(p.x for p in points) / len(points), sum(p.y for p in points) / len(points))
        elif kind == "coordinates":
            point, origin, baseline = _point(features[0]), _point(features[1]), _line(features[2])
            u = _direction(*baseline); v = np.array((-u[1], u[0])); delta = _array(point) - _array(origin)
            x, y = float(delta @ u), float(delta @ v)
            metric("x", "相对 X", x); metric("y", "相对 Y", y)
            projected = _p(_array(origin) + x * u)
            draw((origin, projected, point), kind="polyline"); draw(baseline, "source"); anchor = point
            quality["coordinate_basis"] = "辅助几何相对坐标；不是样品台坐标"
        elif kind == "width":
            points = features[0].points
            if features[0].kind != "caliper_width" or len(points) < 2 or len(points) % 2:
                raise ValueError("宽度测量需要成对的截面边缘点")
            widths = [_distance(a, b) for a, b in zip(points[::2], points[1::2])]
            if any(width <= _EPS for width in widths):
                raise ValueError("截面边缘点重合，宽度无效")
            metric("mean", "平均宽度", float(np.mean(widths)))
            metric("min", "最小宽度", min(widths)); metric("max", "最大宽度", max(widths))
            metric("stddev", "宽度标准差（样本）", float(np.std(widths, ddof=1)) if len(widths) > 1 else 0.0)
            metric("count", "有效截面数", len(widths), "")
            for index, ((a, b), width) in enumerate(zip(zip(points[::2], points[1::2]), widths), 1):
                draw((a, b)); metric(f"section_{index}", f"截面 {index}", width)
            anchor = _mid(points[0], points[1]); quality["section_widths"] = widths
            quality["stddev_basis"] = "sample_ddof_1"
        elif kind == "fit_line":
            a, b = _line(features[0]); metric("length", "拟合线跨度", _distance(a, b)); draw((a, b)); anchor = _mid(a, b)
        else:
            raise ValueError(f"不支持的测量类型：{kind}")
        chosen = params.get("metric")
        if chosen:
            selected = [item for item in metrics if item.key == chosen]
            if not selected:
                raise ValueError("所选指标不适用于当前测量")
            metrics = selected + [item for item in metrics if item.key != chosen]
        return MeasurementResult(definition.id, "ready", metrics=tuple(metrics), anchor_px=_pixel(anchor, scale), geometry=tuple(geometry), quality=quality)
    except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError) as error:
        return MeasurementResult(definition.id, "invalid", str(error), anchor_px=_pixel(anchor, scale), quality=quality)


def evaluate_document(document: Any) -> dict[str, MeasurementResult]:
    """Bounded result cache excludes label layout; moving labels never refits."""
    definitions = []
    for definition in document.professional_measurements:
        data = definition.to_dict()
        data.pop("annotation_layout", None); data.pop("name", None)
        definitions.append(data)
    signature = json.dumps({
        "id": document.id, "calibration": document.calibration.to_dict() if document.calibration else None,
        "features": [item.to_dict() for item in document.professional_features], "definitions": definitions,
        "construction_revision": document.construction_geometry_revision,
        "measurement_revision": document.measurement_geometry_revision,
        "source_revision": _source_revision(document),
        "source_pending": document.professional_source_revision_pending,
        "source_error": document.professional_source_revision_error,
    }, ensure_ascii=False, sort_keys=True, allow_nan=False)
    key = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    cached = _DOCUMENT_RESULTS.get(id(document))
    if cached is not None and cached[0] == key:
        _DOCUMENT_RESULTS.move_to_end(id(document))
        return dict(cached[1])
    resolved = {feature.id: resolve_feature(document, feature) for feature in document.professional_features}
    results = {definition.id: evaluate_measurement(document, definition, resolved_features=resolved) for definition in document.professional_measurements}
    _DOCUMENT_RESULTS[id(document)] = (key, results)
    _DOCUMENT_RESULTS.move_to_end(id(document))
    while len(_DOCUMENT_RESULTS) > 16:
        _DOCUMENT_RESULTS.popitem(last=False)
    return dict(results)


def clear_document_cache(document: Any) -> None:
    _DOCUMENT_RESULTS.pop(id(document), None)
