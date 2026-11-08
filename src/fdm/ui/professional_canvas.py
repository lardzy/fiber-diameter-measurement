"""Image-coordinate professional command state, selection and label dragging."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QMenu

from fdm.construction_geometry import LiveFeatureRef, ResolvedCircle, ResolvedLine, ResolvedPoint, SourceObjectKind
from fdm.construction_document import clip_line_to_rect
from fdm.geometry import Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.metrology import ControlPointRef, MeasurementFeature
from fdm.services.metrology_service import (calibration_scale, control_points_px, freeze_feature,
                                            projection_foot_px, resolve_feature, source_name)
from fdm.services.object_snap_service import SnapCandidate, SnapKind, professional_snap_primitives
from fdm.models import new_id
from fdm.ui.professional_measurement import SOURCE_TITLES, TOOLS
from fdm.ui.professional_rendering import ProfessionalDisplayIndex, dimension_endpoints, draw_professional_measurements, presentation_result


@dataclass(frozen=True, slots=True)
class ObjectPick:
    source_type: str
    source_id: str
    feature_key: str
    outline: tuple
    distance: float
    line: tuple = ()

    @property
    def identity(self):
        return self.source_type, self.source_id, self.feature_key


def _ellipse_outline(center, rx, ry):
    return tuple(Point(center.x + rx * math.cos(t), center.y + ry * math.sin(t)) for t in np_angles())


def _segment_distance(point, a, b):
    dx, dy = b.x - a.x, b.y - a.y
    length = dx * dx + dy * dy
    t = max(0., min(1., ((point.x-a.x)*dx + (point.y-a.y)*dy) / length)) if length else 0.
    return math.hypot(point.x - a.x - t*dx, point.y - a.y - t*dy)


class ProfessionalCanvasController:
    def __init__(self, canvas, workspace):
        self.canvas = canvas
        self.workspace = workspace
        self.key = None
        self.points = []
        self.point_refs = []
        self.pointer = None
        self.choices = ()
        self.choice = 0
        self._snap_key = None
        self._snap_features = {}
        self._primitive_groups = ()
        self._outlines = {}
        self._choice_override = None
        self.features = []
        self.selections = []
        self.object_slots = set()
        self.slot = 0
        self.start_focus = None
        self.evidence = None
        self.label_hits = {}
        self.hovered_id = None
        self.drag = None
        self._drag_press_widget = None
        self._drag_moved = False
        self.preview_layouts = {}
        self.geometry_drag = None
        self.preview_features = {}
        self._results_key = None
        self._results = {}
        self._display_index_key = None
        self._display_index = None
        self._drag_layer = None
        self._drag_layer_key = None
        self._drag_layer_hits = {}

    def focus(self):
        method = getattr(self.canvas, "focus_index", None)
        return method() if method else None

    def results(self):
        document = self.canvas._document
        if document is None:
            return {}
        key = (id(document), document.id, document.professional_geometry_revision, document.construction_geometry_revision,
               document.measurement_geometry_revision, repr(document.calibration),
               document.professional_source_revision, document.professional_source_revision_pending, document.professional_source_revision_error,
               document.metadata.get("digital_slide", {}).get("stitching_revision"),
               document.metadata.get("digital_slide", {}).get("source_revision"))
        if key != self._results_key:
            from fdm.services.metrology_service import evaluate_document
            self._results = evaluate_document(document)
            self._results_key = key
        return self._results

    def cancel(self):
        self.key = None; self.points = []; self.features = []; self.selections = []
        self.slot = 0; self.evidence = None; self.drag = None; self.geometry_drag = None
        self.point_refs = []
        self.object_slots.clear()
        self._clear_pick()
        self._drag_press_widget = None; self._drag_moved = False
        self.preview_layouts.clear(); self.preview_features.clear()
        self._drag_layer = None; self._drag_layer_key = None; self._drag_layer_hits = {}
        self.canvas.update()

    def invalidate_display(self):
        """Discard derived label pixels/hits without touching geometry or drags."""
        self.label_hits.clear()
        self._drag_layer = None
        self._drag_layer_key = None
        self._drag_layer_hits = {}

    def begin(self, key, selections):
        self.cancel(); self.key = key; self.selections = list(selections); self.start_focus = self.focus()
        self._advance_sources()

    def _clear_pick(self):
        self.pointer = None
        self.choices = ()
        self.choice = 0
        self._choice_override = None
        self.canvas._set_active_snap_candidate(None)

    def _follow(self):
        return (not (self.key or "").startswith("caliper_")
                and self.workspace.panel.follow_sources.isChecked())

    def _feature(self, kind, **fields):
        document = self.canvas._document
        from fdm.ui.professional_measurement import slide_revision
        return MeasurementFeature(new_id("feature"), document.id, kind, focus_index=self.focus(),
            source_revision=slide_revision(document), parameters={"robust": True,
            "calibration_at_creation": document.calibration.to_dict() if document.calibration else None}, **fields)

    def _selected_feature(self, kind, source_kind, identity, key="geometry"):
        if source_kind in {"feature", "professional"}:
            feature = self._feature(kind, professional_source_id=identity)
        else:
            ref = LiveFeatureRef(self.canvas._document.id, identity, SourceObjectKind(source_kind), key)
            feature = self._feature(kind, source_ref=ref)
        return feature if self._follow() else freeze_feature(self.canvas._document, feature)

    def set_slot_source(self, index, selection):
        if not self.key or not 0 <= index < len(self.selections):
            return
        if index <= self.slot:
            self.features = self.features[:index]
            self.object_slots.intersection_update(range(index))
            self.slot = index
            self.points = []; self.point_refs = []
        self.selections[index] = selection
        self._clear_pick()
        self._advance_sources()
        self.canvas.update()

    def step_back(self):
        if not self.key:
            return
        if self.points:
            self.points.pop()
            self.point_refs.pop()
        elif self.features:
            previous = self.features.pop()
            self.slot -= 1
            was_object = self.slot in self.object_slots
            self.object_slots.discard(self.slot)
            self.selections[self.slot] = None
            self.workspace.panel.sources[self.slot].setCurrentIndex(0)
            if not was_object:
                self.points = list(previous.points_px[:-1])
                refs = previous.point_refs or (None,) * len(previous.points_px)
                self.point_refs = list(refs[:-1])
        self._clear_pick()
        self._advance_sources()
        self.canvas.update()

    def _step_prompt(self):
        kind, count = TOOLS[self.key][1][self.slot]
        if self.key in SOURCE_TITLES:
            role = SOURCE_TITLES[self.key][self.slot]
        elif self.key in {"distance", "arc", "angle", "caliper_circle"}:
            names = {"distance": ("起点", "终点"), "arc": ("起点", "经过点", "终点"),
                     "angle": ("第一点", "角顶点", "第三点"), "caliper_circle": ("圆心", "半径点")}
            role = names[self.key][len(self.points)]
        else:
            role = {"line": "线", "circle": "圆", "point": "点"}.get(kind, "控制点")
        if self._whole_kind():
            instruction = f"选{role}，或在空白处开始取点"
        elif count < 0:
            instruction = f"选控制点（已取 {len(self.points)}），Enter 完成"
        else:
            instruction = f"选{role}（第 {len(self.points) + 1}/{count} 点）"
        return f"{TOOLS[self.key][0]} · 第 {self.slot + 1}/{len(self.selections)} 步：{instruction}"

    def _whole_kind(self):
        if self.points or self.key not in {"point_line_distance", "point_segment_distance", "parallel_distance",
                "projection", "line_angle", "circle_center_distance", "circle_gap", "point_circle_distance", "coordinates"}:
            return None
        if self.key == "projection" and self.slot == 0:
            return None
        kind = TOOLS[self.key][1][self.slot][0]
        return kind if kind in {"circle", "line"} else None

    def _advance_sources(self):
        if not self.key:
            return
        slots = TOOLS[self.key][1]
        while self.slot < len(slots):
            selection = self.selections[self.slot]
            if selection is None:
                break
            source_kind, identity = selection
            try:
                feature = self._selected_feature(slots[self.slot][0], source_kind, identity)
            except ValueError as exc:
                self.selections[self.slot] = None
                self.workspace.message(str(exc))
                return
            self.object_slots.add(self.slot)
            self.features.append(feature); self.slot += 1
        if self.slot >= len(slots):
            self._complete()
        else:
            self.workspace.message(self._step_prompt())

    def _point_allowed(self, point):
        document = self.canvas._document
        width, height = document.image_size
        if not 0 <= point.x < width or not 0 <= point.y < height:
            return False
        gate = getattr(self.canvas, "vector_measurement_available", None)
        return gate(point) if gate else not self.canvas._read_only

    def finish(self):
        if self.key is None:
            return
        kind, count = TOOLS[self.key][1][self.slot]
        minimum = 3 if kind in {"polygon", "fit_circle"} else 2
        if count != -1 or len(self.points) < minimum:
            self.workspace.message(f"当前来源至少需要 {minimum if count < 0 else count} 个点。")
            return
        self._finish_slot(kind)

    def _finish_slot(self, kind):
        self.features.append(self._feature(kind, points_px=tuple(self.points), point_refs=tuple(self.point_refs)))
        self.points = []; self.point_refs = []; self._clear_pick()
        self.slot += 1; self._advance_sources()

    def _complete(self):
        key = self.key
        if key.startswith("caliper_"):
            points = self.features[0].points_px
            self.workspace.start_caliper(self, key, points)
        else:
            self.workspace.commit(self, key, self.features,
                parameters={"supplement": self.workspace.panel.supplement.isChecked()})

    def snap_features(self):
        document = self.canvas._document
        if not document or not document.professional_features:
            self._snap_features = {}; self._primitive_groups = (); self._outlines = {}; self._snap_key = None
            return self._snap_features
        results = self.results()
        excluded = self.geometry_drag[0].id if self.geometry_drag else None
        key = (self._results_key, self.focus(), excluded)
        if self._snap_key != key:
            blocked = {excluded} if excluded else set()
            if blocked:
                while True:
                    dependencies = {f.id for f in document.professional_features if f.professional_dependencies() & blocked}
                    if dependencies <= blocked:
                        break
                    blocked.update(dependencies)
            committed = {fid for d in document.professional_measurements
                         if d.focus_index == self.focus() and d.id in results and results[d.id].valid
                         for fid in d.feature_ids}
            resolved = {}
            eligible = {}
            for feature in document.professional_features:
                if feature.id in committed and feature.id not in blocked and feature.focus_index == self.focus():
                    value = resolve_feature(document, feature, _cache=resolved)
                    if value.valid:
                        eligible[feature.id] = value
            scale = calibration_scale(document)
            self._snap_features = eligible
            groups = []
            self._outlines = {}
            for fid, feature in eligible.items():
                bounds_points = feature.points_px
                if feature.kind == "line":
                    self._outlines[fid] = feature.points_px
                if feature.center:
                    center = Point(feature.center.x / scale[0], feature.center.y / scale[1])
                    rx, ry = feature.radius / scale[0], feature.radius / scale[1]
                    bounds_points = (Point(center.x-rx, center.y-ry), Point(center.x+rx, center.y+ry))
                    if feature.kind == "circle":
                        self._outlines[fid] = _ellipse_outline(center, rx, ry)
                primitives = tuple(professional_snap_primitives(feature, scale))
                if primitives:
                    xs, ys = zip(*((p.x, p.y) for p in bounds_points))
                    groups.append(((min(xs), min(ys), max(xs), max(ys)), fid, primitives))
            self._primitive_groups = tuple(groups)
            self._snap_key = key
        return self._snap_features

    def _nearby_groups(self, point, tolerance):
        self.snap_features()
        for (left, top, right, bottom), fid, primitives in self._primitive_groups:
            if (left-tolerance <= point.x <= right+tolerance and top-tolerance <= point.y <= bottom+tolerance):
                yield fid, primitives

    def snap_primitives(self, point, tolerance):
        return (p for _, group in self._nearby_groups(point, tolerance) for p in group)

    def _objects_at(self, point, kind):
        canvas = self.canvas
        settings = canvas._object_snap_engine.settings
        if not canvas._object_snap_pointer_enabled():
            return []
        tolerance = settings.aperture_px / max(canvas._zoom, .001)
        accepted = (ResolvedLine,) if kind == "line" else (ResolvedCircle,)
        candidates = canvas._source_candidates_at(point, tolerance, False, accepted)
        hits = []
        for candidate in candidates:
            if candidate.object_kind == "measurement" and not settings.include_measurements:
                continue
            shape = candidate.geometry
            if isinstance(shape, ResolvedLine):
                extent = (0., 0., *canvas._document.image_size)
                outline = clip_line_to_rect(shape, extent) or (shape.start, shape.end)
            else:
                outline = _ellipse_outline(shape.center, shape.radius, shape.radius)
            hits.append(ObjectPick(candidate.object_kind, candidate.object_id, candidate.feature, outline,
                                   candidate.distance_px, (shape.start, shape.end) if isinstance(shape, ResolvedLine) else ()))
        if settings.include_measurements:
            for fid, _ in self._nearby_groups(point, tolerance):
                feature = self._snap_features[fid]
                if feature.kind == kind:
                    outline = self._outlines[fid]
                    distance = min(_segment_distance(point, a, b) for a, b in zip(outline, outline[1:]))
                    if distance <= tolerance:
                        hits.append(ObjectPick("professional", fid, "geometry", outline, distance,
                                               outline if kind == "line" else ()))
        return sorted(hits, key=lambda h: (h.distance, h.source_type, h.source_id, h.feature_key))

    def _point_reference(self, candidate):
        if (candidate is None or not self._follow() or candidate.source_type not in
                {"construction", "measurement", "professional"} or candidate.kind in
                {SnapKind.NEAREST, SnapKind.TANGENT, SnapKind.PERPENDICULAR}):
            return None
        if candidate.source_type != "professional":
            ref = LiveFeatureRef(self.canvas._document.id, candidate.source_id,
                                 SourceObjectKind(candidate.source_type), candidate.feature_key)
            resolved = self.canvas._ensure_construction_preview_resolver().resolve_feature(ref)
            if not isinstance(resolved, ResolvedPoint) or math.hypot(
                    resolved.point.x-candidate.point_px.x, resolved.point.y-candidate.point_px.y) > 1e-6:
                return None
        return ControlPointRef(candidate.source_type, candidate.source_id, candidate.feature_key, candidate.kind.value)

    def _update_pick(self, point, *, message=True):
        self.pointer = point
        whole = self._whole_kind() if self.key else None
        choices = self._objects_at(point, whole) if whole else []
        if choices:
            self.canvas._set_active_snap_candidate(None)
        else:
            candidate = self.canvas._query_object_snap(point)
            if candidate is not None:
                choices = [candidate] + [c for c in self.canvas._object_snap_engine.last_candidates
                    if c.identity != candidate.identity and self._point_allowed(c.point_px)]
        self.choices = tuple(choices)
        self.choice = next((i for i, c in enumerate(choices) if c.identity == self._choice_override), 0)
        if self.choice == 0 and (not choices or choices[0].identity != self._choice_override):
            self._choice_override = None
        return self._display_choice(message)

    def _display_choice(self, message):
        candidate = self.choices[self.choice] if self.choices else None
        text = "固定坐标"
        if isinstance(candidate, ObjectPick):
            name = source_name(self.canvas._document, candidate.source_type, candidate.source_id)
            text = f"{'关联' if self._follow() else '固定几何'} · {name}"
        elif candidate is not None:
            self.canvas._set_active_snap_candidate(candidate)
            ref = self._point_reference(candidate)
            label = candidate.label
            if candidate.source_type in {"professional", "construction", "measurement"}:
                label = f"{source_name(self.canvas._document, candidate.source_type, candidate.source_id)} · {label}"
            text = f"{'关联' if ref else '固定坐标'} · {label}"
        if message and self.key:
            if len(self.choices) > 1:
                text += f"（Tab 切换 {self.choice+1}/{len(self.choices)}）"
            self.workspace.message(f"{self._step_prompt()} · {text}")
        return candidate

    def cycle_candidate(self, step=1):
        if not self.key or len(self.choices) < 2:
            return False
        self.choice = (self.choice + step) % len(self.choices)
        self._choice_override = self.choices[self.choice].identity
        self._display_choice(True)
        self.canvas.update()
        return True

    def _hit_label(self, pos):
        document = self.canvas._document
        if document is None:
            return None
        for identity, label in reversed(tuple(self.label_hits.items())):
            definition = document.get_professional_measurement(identity)
            if definition is None or (definition.focus_index is not None and definition.focus_index != self.focus()):
                continue
            if label.contains_text(pos):
                return identity
        return None

    def _hit_geometry(self, pos):
        best, distance = None, 7.
        for definition in self.canvas._document.professional_measurements:
            if definition.focus_index is not None and definition.focus_index != self.focus():
                continue
            result = self.results().get(definition.id)
            if result is None:
                continue
            result = presentation_result(self.canvas._document, definition, result)
            for primitive in result.geometry:
                points = [self.canvas.image_to_widget(p) for p in primitive.points_px]
                for first, second in zip(points, points[1:]):
                    dx, dy = second.x() - first.x(), second.y() - first.y()
                    t = max(0., min(1., ((pos.x()-first.x())*dx+(pos.y()-first.y())*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy > 0 else 0.
                    d = math.hypot(pos.x()-first.x()-t*dx, pos.y()-first.y()-t*dy)
                    if d < distance:
                        best, distance = definition.id, d
        return best

    def _hit_handle(self, pos):
        document = self.canvas._document
        definition = document.get_professional_measurement(document.selected_professional_id)
        if definition is None or definition.focus_index != self.focus():
            return None
        for fid in definition.feature_ids:
            feature = document.get_professional_feature(fid)
            if feature is None or feature.source_ref or feature.professional_source_id or feature.kind.startswith("caliper_"):
                continue
            try:
                points = control_points_px(document, feature)
            except ValueError:
                continue
            for index, point in enumerate(points):
                widget = self.canvas.image_to_widget(point)
                if math.hypot(pos.x()-widget.x(), pos.y()-widget.y()) <= 7:
                    return feature, index
        return None

    def _annotation_input_enabled(self):
        """Do not let an existing label steal another canvas command's input."""
        canvas = self.canvas
        if self.key or self.workspace.active is self or canvas._document is None or canvas._panning:
            return False
        if canvas._tool_mode != "professional":
            return canvas._measurement_annotation_input_enabled()
        # The professional mode is the one additional idle editing mode. Its
        # own multi-step acquisition is excluded above; all legacy gates stay.
        return not (canvas._read_only or canvas._temporary_grab_active
                    or canvas._roi_capture is not None or canvas._scale_anchor_pick_active
                    or canvas._drawing_polygon_points or canvas._drawing_freehand_active
                    or canvas._drawing_line is not None)

    def mouse_press(self, event):
        if self.canvas._document is None or self.canvas._read_only:
            return False
        if event.button() not in {Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton}:
            return False
        # Space/middle-button navigation retains the canvas's existing behavior.
        if getattr(self.canvas, "_space_pressed", False):
            return False
        if self.workspace.active is self and self.key is None:
            # A caliper owns left/right clicks until confirmed or cancelled.
            return True
        pos = event.position()
        editing = self._annotation_input_enabled()
        identity = self._hit_label(pos) if editing else None
        if identity and not self.key:
            if event.button() == Qt.MouseButton.RightButton:
                self.context_menu(identity, event.globalPosition().toPoint()); return True
            self.workspace.select(identity)
            definition = self.canvas._document.get_professional_measurement(identity)
            layout = definition.annotation_layout or MeasurementAnnotationLayout.legacy_professional()
            result = presentation_result(self.canvas._document, definition, self.results()[identity])
            center = self.canvas.widget_to_image(self.label_hits[identity].text_rect.center())
            initial = Point(center.x-result.anchor_px.x, center.y-result.anchor_px.y)
            self.drag = (identity, self.canvas.widget_to_image(pos), layout, initial)
            self._drag_press_widget = QPointF(pos)
            self._drag_moved = False
            self._drag_layer_key = None
            return True
        if event.button() != Qt.MouseButton.LeftButton:
            return bool(self.key)
        if not self.key:
            if not editing or self.canvas._tool_mode not in {"select", "professional"}:
                return self.canvas._tool_mode == "professional"
            handle = self._hit_handle(pos)
            if handle:
                self.geometry_drag = handle; return True
            identity = self._hit_geometry(pos)
            if identity:
                self.workspace.select(identity); return True
            return self.canvas._tool_mode == "professional"
        if self.focus() != self.start_focus:
            self.workspace.cancel(); self.workspace.message("焦层已改变，未完成测量已取消。"); return True
        point = self.canvas.widget_to_image(pos)
        candidate = self._update_pick(point, message=False)
        if isinstance(candidate, SnapCandidate):
            point = candidate.point_px
        if not self._point_allowed(point):
            self.workspace.message("此处原始图像尚不可测量，请等待加载或在有效图像范围内取点。"); return True
        kind, count = TOOLS[self.key][1][self.slot]
        if isinstance(candidate, ObjectPick):
            try:
                feature = self._selected_feature(kind, candidate.source_type, candidate.source_id, candidate.feature_key)
            except ValueError as exc:
                self.workspace.message(str(exc)); return True
            self.features.append(feature)
            self.object_slots.add(self.slot)
            self.slot += 1; self._clear_pick(); self._advance_sources()
            self.canvas.update(); return True
        self.points.append(point)
        self.point_refs.append(self._point_reference(candidate))
        self._clear_pick()
        if count > 0 and len(self.points) == count:
            self._finish_slot(kind)
        else:
            self._advance_sources()
        self.canvas.update(); return True

    def mouse_move(self, event):
        if self.key and not self.canvas._panning and not getattr(self.canvas, "_space_pressed", False):
            self._update_pick(self.canvas.widget_to_image(event.position()))
            self.canvas.update()
            return True
        if self.drag:
            identity, start, layout, offset = self.drag
            press = self._drag_press_widget or self.canvas.image_to_widget(start)
            delta = event.position() - press
            if not self._drag_moved and math.hypot(delta.x(), delta.y()) < 3:
                # Selection and pointer jitter do not create a layout change.
                return True
            self._drag_moved = True
            point = self.canvas.widget_to_image(event.position())
            self.preview_layouts[identity] = layout.with_manual_offset(Point(offset.x+point.x-start.x, offset.y+point.y-start.y))
            self.canvas.update(); return True
        if self.geometry_drag:
            feature, index = self.geometry_drag
            point = self.canvas.widget_to_image(event.position())
            candidate = self._update_pick(point, message=False)
            if candidate is not None:
                point = candidate.point_px
            if self._point_allowed(point):
                points = list(feature.points_px); points[index] = point
                refs = list(feature.point_refs or (None,) * len(points))
                refs[index] = self._point_reference(candidate)
                self.preview_features[feature.id] = replace(feature, points_px=tuple(points), point_refs=tuple(refs))
                self.canvas.update()
            return True
        hovered = (self._hit_label(event.position()) or self._hit_geometry(event.position())) if self._annotation_input_enabled() else None
        if hovered != self.hovered_id:
            self.hovered_id = hovered; self.workspace.hover(hovered); self.canvas.update()
        return False

    def mouse_release(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self.canvas._panning:
            return False
        if self.drag:
            self.mouse_move(event)
            identity, _start, _original, initial = self.drag
            layout = self.preview_layouts.get(identity)
            changed = layout is not None and layout.offset_px is not None and (
                abs(layout.offset_px.x - initial.x) > 1e-9 or abs(layout.offset_px.y - initial.y) > 1e-9)
            self.drag = None; self.preview_layouts.clear()
            self._drag_press_widget = None; self._drag_moved = False
            self._drag_layer = None; self._drag_layer_key = None; self._drag_layer_hits = {}
            if changed:
                self.workspace.edit_layout(self, identity, layout)
            self.canvas.update(); return True
        if self.geometry_drag:
            self.mouse_move(event)
            feature, _ = self.geometry_drag
            replacement = self.preview_features.get(feature.id)
            self.geometry_drag = None; self.preview_features.clear()
            self._clear_pick()
            if replacement and replacement != feature:
                document = self.canvas._document
                self.workspace.change(document, "修改专业测量控制点", lambda: document.apply_professional_state(
                    [replacement if f.id == feature.id else f for f in document.professional_features],
                    document.professional_measurements, selected_id=document.selected_professional_id, mark_dirty=False))
            self.canvas.update(); return True
        return bool(self.key or self.canvas._tool_mode == "professional" and not self.canvas._panning
                    and not getattr(self.canvas, "_space_pressed", False))

    def key_press(self, event):
        if self.canvas._document is None:
            return False
        if event.key() == Qt.Key.Key_Escape and (self.key or self.drag or self.geometry_drag or self.evidence or self.workspace.active is self):
            self.workspace.cancel_command(); self.canvas.update(); return True
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and self.key:
            self.finish(); return True
        if event.key() == Qt.Key.Key_Backspace and self.key:
            self.step_back(); return True
        if event.key() in {Qt.Key.Key_Tab, Qt.Key.Key_Backtab} and self.key:
            return self.cycle_candidate(-1 if event.key() == Qt.Key.Key_Backtab else 1)
        if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace} and self.canvas._document.selected_professional_id and self._annotation_input_enabled():
            self.workspace.delete_selected(); return True
        return False

    def context_menu(self, identity, position):
        definition = self.canvas._document.get_professional_measurement(identity)
        layout = definition.annotation_layout or MeasurementAnnotationLayout.legacy_professional()
        menu = QMenu(self.canvas)
        automatic = menu.addAction("恢复自动位置")
        leader = menu.addAction("显示 / 隐藏引出线")
        styles = menu.addMenu("标注样式")
        single = styles.addAction("单引出")
        dual = styles.addAction("双端引出（CAD 对齐尺寸）")
        single.setCheckable(True); dual.setCheckable(True)
        single.setChecked(layout.leader_style == "single"); dual.setChecked(layout.leader_style == "dual")
        dual.setEnabled(dimension_endpoints(presentation_result(self.canvas._document, definition, self.results()[identity])) is not None)
        properties = menu.addAction("标注属性…")
        number = menu.addAction("显示 / 隐藏编号")
        supplement = menu.addAction("切换补角") if definition.kind in {"angle", "line_angle"} else None
        metric_menu = menu.addMenu("标注属性：显示指标")
        metric_actions = {}
        result = self.results().get(identity)
        if result is not None:
            for metric in result.metrics:
                action = metric_menu.addAction(f"{metric.label} ({metric.unit})")
                metric_actions[action] = metric.key
        chosen = menu.exec(position)
        if chosen is automatic:
            self.workspace.edit_layout(self, identity, layout.with_automatic_position())
        elif chosen is leader:
            self.workspace.edit_layout(self, identity, replace(layout, show_leader=not layout.show_leader))
        elif chosen is single or chosen is dual:
            self.workspace.edit_layout(self, identity, replace(layout, leader_style="dual" if chosen is dual else "single"))
        elif chosen is properties:
            self.workspace.professional_properties(self, definition)
        elif chosen is number:
            self.workspace.edit_layout(self, identity, replace(layout, show_identifier=not layout.show_identifier))
        elif chosen in metric_actions:
            self.workspace.edit_layout(self, identity, replace(layout, metric_key=metric_actions[chosen]))
        elif supplement is not None and chosen is supplement:
            self.workspace.edit_parameters(self, identity, {**definition.parameters, "supplement": not definition.parameters.get("supplement", False)})

    def paint(self, painter: QPainter):
        if self.key and (self.focus() != self.start_focus or self.canvas._tool_mode != "professional"):
            self.workspace.cancel()
        document = self.canvas._document
        if document is None:
            return
        if self.key:
            from fdm.services.metrology_service import professional_source_unavailable_reason
            reason = professional_source_unavailable_reason(document)
            if reason:
                self.workspace.cancel()
                self.workspace.message(reason)
        presentation = getattr(self.canvas, "presentation_state", None)
        if presentation is not None:
            frame = presentation()
            if frame.focus_index != self.focus() or frame.quality == "placeholder" or frame.generation != self.canvas._view_generation:
                self.label_hits.clear()
                return
        results = self.results()
        if self.preview_features:
            import copy
            from fdm.services.metrology_service import evaluate_document
            scratch = copy.copy(document)
            scratch.professional_features = [self.preview_features.get(f.id, f) for f in document.professional_features]
            # An isolated revision prevents the shared service cache from
            # mistaking draft geometry for the persisted document state.
            scratch.mark_professional_geometry_changed()
            results = evaluate_document(scratch)
        mode = getattr(self.canvas, "_screen_label_mode", "all")
        label_decimals = self.canvas._settings.professional_measurement_label_decimals
        arguments = dict(results=results, focus_index=self.focus(), label_decimals=label_decimals, settings=self.canvas._settings,
                         selected_id=document.selected_professional_id, hovered_id=self.hovered_id,
                         show_labels=mode != "hidden", selected_labels_only=mode == "selected")
        if len(document.professional_measurements) >= 64 and not self.preview_features and not self.drag:
            index_key = (self._results_key, self.canvas._canvas_visual_settings_signature)
            if (self._display_index_key != index_key or self._display_index is None
                    or self._display_index.definitions is not document.professional_measurements):
                self._display_index = ProfessionalDisplayIndex(document, results, self.canvas._settings)
                self._display_index_key = index_key
            context = self.canvas._paint_context()
            # Users can drag a caption into the margin outside the source image.
            arguments["visible_definitions"] = self._display_index.visible(
                context.widget_to_image_transform.mapRect(context.widget_rect), context.zoom, self.canvas._settings,
            )
        elif len(document.professional_measurements) < 64:
            self._display_index = None
        dpr = self.canvas.devicePixelRatioF()
        width, height = math.ceil(self.canvas.width()*dpr), math.ceil(self.canvas.height()*dpr)
        if self.drag and width * height * 4 <= 64 * 1024 * 1024:
            identity = self.drag[0]
            origin = self.canvas.image_to_widget(Point(0, 0))
            cache_key = (self._results_key, identity, width, height, dpr, self.canvas._zoom,
                         origin.x(), origin.y(), self.focus(), painter.font().toString(),
                         document.selected_professional_id, self.hovered_id, mode, label_decimals, self.canvas._canvas_visual_settings_signature,
                         tuple((d.id, d.annotation_layout) for d in document.professional_measurements))
            if self._drag_layer is None or self._drag_layer_key != cache_key:
                layer = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
                layer.setDevicePixelRatio(dpr); layer.fill(Qt.GlobalColor.transparent)
                layer_painter = QPainter(layer); layer_painter.setFont(painter.font())
                self._drag_layer_hits = draw_professional_measurements(layer_painter, document, self.canvas.image_to_widget,
                    exclude_label_ids={identity}, **arguments)
                layer_painter.end()
                self._drag_layer, self._drag_layer_key = layer, cache_key
            painter.drawImage(QPointF(0, 0), self._drag_layer)
            self.label_hits = dict(self._drag_layer_hits)
            self.label_hits.update(draw_professional_measurements(painter, document, self.canvas.image_to_widget,
                only_measurement_id=identity, draw_geometry=False, layouts=self.preview_layouts, **arguments))
        else:
            self.label_hits = draw_professional_measurements(painter, document, self.canvas.image_to_widget,
                layouts=self.preview_layouts, **arguments)
        painter.save()
        painter.setPen(QPen(QColor("#ffce60"), 1.5))
        if self.key:
            transform = self.canvas.image_to_widget
            previous = [resolve_feature(document, f) for f in self.features]
            for feature in previous:
                if not feature.valid:
                    continue
                points = feature.points_px
                if feature.center and feature.radius:
                    sx, sy = calibration_scale(document)
                    points = _ellipse_outline(Point(feature.center.x/sx, feature.center.y/sy),
                                              feature.radius/sx, feature.radius/sy)
                if len(points) > 1:
                    painter.drawPolyline(QPolygonF([transform(p) for p in points]))
                else:
                    painter.drawEllipse(transform(points[0]), 4, 4)
            choice = self.choices[self.choice] if self.choices else None
            active_point = choice.point_px if isinstance(choice, SnapCandidate) else self.pointer
            source_line = ()
            if isinstance(choice, ObjectPick):
                painter.drawPolyline(QPolygonF([transform(p) for p in choice.outline]))
                source_line = choice.line
            elif self.points and active_point:
                source_line = (self.points[-1], active_point)
                painter.drawLine(*[transform(p) for p in source_line])
            if (len(previous) == 1 and previous[0].valid and source_line and
                    source_line[0] != source_line[-1] and self.key in {"point_line_distance", "point_segment_distance"}):
                point = previous[0].points_px[0]
                foot = projection_foot_px(document, point, source_line, segment=self.key == "point_segment_distance")
                painter.setPen(QPen(QColor("#48dace"), 1.5, Qt.PenStyle.DashLine))
                painter.drawLine(transform(point), transform(foot))
                painter.drawEllipse(transform(foot), 3, 3)
        if self.points:
            points = [self.canvas.image_to_widget(p) for p in self.points]
            if len(points) > 1:
                painter.drawPolyline(QPolygonF(points))
            for point in points:
                painter.drawEllipse(point, 3, 3)
        selected = document.get_professional_measurement(document.selected_professional_id)
        if selected and selected.focus_index == self.focus():
            for fid in selected.feature_ids:
                feature = self.preview_features.get(fid) or document.get_professional_feature(fid)
                if (feature is not None and not feature.source_ref and not feature.professional_source_id
                        and not feature.kind.startswith("caliper_")):
                    try:
                        editable_points = control_points_px(document, feature)
                    except ValueError:
                        continue
                    for point in editable_points:
                        p = self.canvas.image_to_widget(point)
                        painter.drawRect(QRectF(p.x()-3, p.y()-3, 6, 6))
                    if feature.kind in {"fit_line", "fit_circle"}:
                        fit = results.get(selected.id).quality.get("features", {}).get(fid, {}).get("evaluated_fit", {})
                        accepted = set(fit.get("accepted_indices", []))
                        for index, point in enumerate(editable_points):
                            painter.setPen(QPen(QColor("#42e392" if index in accepted else "#ff675f"), 2))
                            painter.drawEllipse(self.canvas.image_to_widget(point), 3, 3)
        evidence = self.evidence
        if evidence is None and selected:
            feature = document.get_professional_feature(selected.feature_ids[0])
            if feature and feature.kind.startswith("caliper_"):
                evidence = feature.quality
        if evidence:
            profiles = evidence.get("profiles", []) if isinstance(evidence, dict) else evidence.to_dict().get("profiles", [])
            fit = evidence.get("fit") if isinstance(evidence, dict) else evidence.to_dict().get("fit")
            rejected = {tuple(p) for i, p in enumerate((fit or {}).get("points_px", [])) if i in (fit or {}).get("rejected_indices", [])}
            for profile in profiles:
                polygon = [self.canvas.image_to_widget(Point(*p)) for p in profile["search_polygon_px"]]
                painter.setPen(QPen(QColor(30, 170, 220, 130), .7))
                if polygon:
                    painter.drawPolygon(QPolygonF(polygon))
                for index, candidate in enumerate(profile.get("candidates", [])):
                    point = tuple(candidate["point_px"])
                    used = profile.get("status") == "ready" and index in profile.get("selected_indices", []) and point not in rejected
                    painter.setPen(QPen(QColor("#42e392" if used else "#ff675f"), 2))
                    painter.drawEllipse(self.canvas.image_to_widget(Point(*point)), 2, 2)
            if fit and fit.get("status") == "ready":
                geometry = fit["geometry_px"]
                painter.setPen(QPen(QColor("#42e392"), 2))
                if "endpoints" in geometry:
                    painter.drawLine(*[self.canvas.image_to_widget(Point(*p)) for p in geometry["endpoints"]])
                elif "center" in geometry:
                    center = geometry["center"]; radii = geometry["radii"]
                    circle = [self.canvas.image_to_widget(Point(center[0]+radii[0]*math.cos(t), center[1]+radii[1]*math.sin(t))) for t in np_angles()]
                    painter.drawPolyline(QPolygonF(circle))
        painter.restore()


def np_angles():
    return (i * 2 * math.pi / 96 for i in range(97))
