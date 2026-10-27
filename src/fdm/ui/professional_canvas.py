"""Image-coordinate professional command state, selection and label dragging."""
from __future__ import annotations

from dataclasses import replace
import math
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QMenu

from fdm.construction_geometry import LiveFeatureRef, SourceObjectKind
from fdm.geometry import Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.metrology import MeasurementFeature
from fdm.models import new_id
from fdm.ui.professional_measurement import TOOLS
from fdm.ui.professional_rendering import dimension_endpoints, draw_professional_measurements, presentation_result


class ProfessionalCanvasController:
    def __init__(self, canvas, workspace):
        self.canvas = canvas
        self.workspace = workspace
        self.key = None
        self.points = []
        self.features = []
        self.selections = []
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
        self.cancel(); self.key = key; self.selections = selections; self.start_focus = self.focus()
        self._advance_sources()

    def _advance_sources(self):
        if not self.key:
            return
        slots = TOOLS[self.key][1]
        document = self.canvas._document
        while self.slot < len(slots):
            selection = self.selections[self.slot]
            if selection is None:
                break
            source_kind, identity = selection
            if source_kind == "feature":
                feature = document.get_professional_feature(identity)
                if feature is None or feature.focus_index != self.focus():
                    self.workspace.message("来源特征不存在或属于其他焦层。"); self.cancel(); return
            else:
                feature = MeasurementFeature(new_id("feature"), document.id, slots[self.slot][0],
                    source_ref=LiveFeatureRef(document.id, identity, SourceObjectKind(source_kind)), focus_index=self.focus(),
                    parameters={"calibration_at_creation": document.calibration.to_dict() if document.calibration else None})
            self.features.append(feature); self.slot += 1
        if self.slot >= len(slots):
            self._complete()
        else:
            count = slots[self.slot][1]
            self.workspace.message(f"{TOOLS[self.key][0]}：来源 {self.slot + 1}，" +
                ("连续点选，Enter 完成；Esc 取消。" if count < 0 else f"点选 {count} 个控制点（已取 {len(self.points)}）。"))

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
        document = self.canvas._document
        self.features.append(MeasurementFeature(new_id("feature"), document.id, kind,
            tuple(self.points), focus_index=self.focus(), parameters={"robust": True,
                "calibration_at_creation": document.calibration.to_dict() if document.calibration else None}))
        self.points = []; self.slot += 1; self._advance_sources()

    def _complete(self):
        key = self.key
        if key.startswith("caliper_"):
            points = self.features[0].points_px
            self.workspace.start_caliper(self, key, points)
        else:
            self.workspace.commit(self, key, self.features,
                parameters={"supplement": self.workspace.panel.supplement.isChecked()})

    def _hit_label(self, pos):
        document = self.canvas._document
        if document is None:
            return None
        for identity, label in reversed(tuple(self.label_hits.items())):
            definition = document.get_professional_measurement(identity)
            if definition is None or (definition.focus_index is not None and definition.focus_index != self.focus()):
                continue
            if label.text_rect.contains(pos):
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
            if feature is None or feature.source_ref or feature.kind.startswith("caliper_"):
                continue
            for index, point in enumerate(feature.points_px):
                widget = self.canvas.image_to_widget(point)
                if math.hypot(pos.x()-widget.x(), pos.y()-widget.y()) <= 7:
                    return feature, index
        return None

    def _annotation_input_enabled(self):
        """Do not let an existing label steal another canvas command's input."""
        canvas = self.canvas
        if self.key or canvas._document is None or canvas._panning:
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
        pos = event.position()
        editing = self._annotation_input_enabled()
        identity = self._hit_label(pos) if editing else None
        if identity and not self.key:
            if event.button() == Qt.MouseButton.RightButton:
                self.context_menu(identity, event.globalPosition().toPoint()); return True
            self.workspace.select(identity)
            definition = self.canvas._document.get_professional_measurement(identity)
            layout = definition.annotation_layout or MeasurementAnnotationLayout()
            result = presentation_result(self.canvas._document, definition, self.results()[identity])
            center = self.canvas.widget_to_image(self.label_hits[identity].text_rect.center())
            initial = Point(center.x-result.anchor_px.x, center.y-result.anchor_px.y)
            self.drag = (identity, self.canvas.widget_to_image(pos), layout, initial)
            self._drag_press_widget = QPointF(pos)
            self._drag_moved = False
            self._drag_layer_key = None
            return True
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if not self.key:
            if not editing or self.canvas._tool_mode not in {"select", "professional"}:
                return False
            handle = self._hit_handle(pos)
            if handle:
                self.geometry_drag = handle; return True
            identity = self._hit_geometry(pos)
            if identity:
                self.workspace.select(identity); return True
            return False
        if self.focus() != self.start_focus:
            self.workspace.cancel(); self.workspace.message("焦层已改变，未完成测量已取消。"); return True
        point = self.canvas.widget_to_image(pos)
        candidate = self.canvas._query_object_snap(point)
        if candidate is not None:
            point = candidate.point_px
        if not self._point_allowed(point):
            self.workspace.message("此处原始图像尚不可测量，请等待加载或在有效图像范围内取点。"); return True
        self.points.append(point)
        kind, count = TOOLS[self.key][1][self.slot]
        if count > 0 and len(self.points) == count:
            self._finish_slot(kind)
        else:
            self._advance_sources()
        self.canvas.update(); return True

    def mouse_move(self, event):
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
            if self._point_allowed(point):
                points = list(feature.points_px); points[index] = point
                self.preview_features[feature.id] = replace(feature, points_px=tuple(points))
                self.canvas.update()
            return True
        hovered = (self._hit_label(event.position()) or self._hit_geometry(event.position())) if self._annotation_input_enabled() else None
        if hovered != self.hovered_id:
            self.hovered_id = hovered; self.workspace.hover(hovered); self.canvas.update()
        return False

    def mouse_release(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
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
            if replacement and replacement != feature:
                document = self.canvas._document
                self.workspace.change(document, "修改专业测量控制点", lambda: document.apply_professional_state(
                    [replacement if f.id == feature.id else f for f in document.professional_features],
                    document.professional_measurements, selected_id=document.selected_professional_id, mark_dirty=False))
            self.canvas.update(); return True
        return bool(self.key)

    def key_press(self, event):
        if self.canvas._document is None:
            return False
        if event.key() == Qt.Key.Key_Escape and (self.key or self.drag or self.geometry_drag or self.evidence or self.workspace.active is self):
            self.workspace.cancel(); self.canvas.update(); return True
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and self.key:
            self.finish(); return True
        if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace} and self.canvas._document.selected_professional_id and self._annotation_input_enabled():
            self.workspace.delete_selected(); return True
        return False

    def context_menu(self, identity, position):
        definition = self.canvas._document.get_professional_measurement(identity)
        layout = definition.annotation_layout or MeasurementAnnotationLayout()
        menu = QMenu(self.canvas)
        automatic = menu.addAction("恢复自动位置")
        leader = menu.addAction("显示 / 隐藏引出线")
        styles = menu.addMenu("标注样式")
        single = styles.addAction("单引出")
        dual = styles.addAction("双端引出（CAD 对齐尺寸）")
        single.setCheckable(True); dual.setCheckable(True)
        single.setChecked(layout.leader_style == "single"); dual.setChecked(layout.leader_style == "dual")
        dual.setEnabled(dimension_endpoints(presentation_result(self.canvas._document, definition, self.results()[identity])) is not None)
        number = menu.addAction("标注属性：显示 / 隐藏编号")
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
        arguments = dict(results=results, focus_index=self.focus(), label_decimals=label_decimals,
                         selected_id=document.selected_professional_id, hovered_id=self.hovered_id,
                         show_labels=mode != "hidden", selected_labels_only=mode == "selected")
        dpr = self.canvas.devicePixelRatioF()
        width, height = math.ceil(self.canvas.width()*dpr), math.ceil(self.canvas.height()*dpr)
        if self.drag and width * height * 4 <= 64 * 1024 * 1024:
            identity = self.drag[0]
            origin = self.canvas.image_to_widget(Point(0, 0))
            cache_key = (self._results_key, identity, width, height, dpr, self.canvas._zoom,
                         origin.x(), origin.y(), self.focus(), painter.font().toString(),
                         document.selected_professional_id, self.hovered_id, mode, label_decimals,
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
                if feature is not None and not feature.source_ref and not feature.kind.startswith("caliper_"):
                    for point in feature.points_px:
                        p = self.canvas.image_to_widget(point)
                        painter.drawRect(QRectF(p.x()-3, p.y()-3, 6, 6))
                    if feature.kind in {"fit_line", "fit_circle"}:
                        fit = results.get(selected.id).quality.get("features", {}).get(fid, {}).get("evaluated_fit", {})
                        accepted = set(fit.get("accepted_indices", []))
                        for index, point in enumerate(feature.points_px):
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
