"""Live measurement viewport, with screen-space overlays and candidate picking."""
from __future__ import annotations

import math
import time
import uuid
from collections import OrderedDict

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QAbstractScrollArea, QApplication, QWidget

from fdm.live_experiment.frames import PresentedFrameToken
from fdm.live_experiment.presentation import category_color
from fdm.ui.live_measurement_annotations import Annotation, layout_annotations, segment_distance


class LiveMeasurementCanvas(QWidget):
    pointPressed = Signal(object)
    pointReleased = Signal(object)
    gestureCancelled = Signal()
    categoryRequested = Signal(object)
    candidateHovered = Signal(object)
    candidateClicked = Signal(str)
    candidateAmbiguous = Signal(object)
    manualGestureStarted = Signal(object, object)
    manualGestureFinished = Signal(object)
    gestureRejected = Signal(str)

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.frame = None
        self.image = None
        self.display_mode = "pixel"
        self.display_value = 100.0
        self.screen_pixel_mm = None
        self.display_revision = 0
        self.cursor_point = None
        self.active_material_id = None
        self.candidate_visible = True
        self.candidate_opacity = .6
        self.input_mode = "mixed"
        self.manual_gesture = "click"
        self.candidate_input_enabled = True
        self._pointer_gesture = None
        # Hit testing an input press must use what Qt actually painted.
        self._painted_candidates = ()
        self._painted_candidate_batch = None
        self._last_business_revision = controller.session.revision
        self._last_candidate_click = None
        self._suppress_candidate_release = False
        self.selected_candidate_id = None
        self.hovered_candidate_id = None
        self._hovered_record_id = None
        self._annotation_cache_key = None
        self._annotation_layout_cache = OrderedDict()
        self._label_placements = {}
        self._record_ordinals = {}
        self._next_record_ordinal = 1
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("")
        controller.fieldChanged.connect(self._field_changed)
        controller.changed.connect(self._controller_changed)
        controller.candidatesChanged.connect(self._candidates_changed)
        controller.candidateStateChanged.connect(self._candidate_state_changed)
        if hasattr(controller, "retainedSelectionsChanged"):
            controller.retainedSelectionsChanged.connect(self.update)

    def _controller_changed(self):
        revision = self.controller.session.revision
        business_changed = revision != self._last_business_revision
        self._last_business_revision = revision
        # The next already-scheduled preview (or its original deadline) will
        # repaint these records. Avoid drawing the old image a second time
        # while its successor is being verified; input/state changes stay live.
        if (business_changed and self._pointer_gesture is None
                and self.controller.should_defer_canvas_business_update()):
            return
        self.update()

    def set_candidate_presentation(self, visible, opacity=.6):
        opacity = float(opacity)
        if not math.isfinite(opacity):
            raise ValueError("候选不透明度必须是有限数值。")
        if opacity > 1:
            opacity /= 100.
        self.candidate_visible = bool(visible)
        self.candidate_opacity = max(.2, min(1., opacity))
        if not self.candidate_visible:
            self._set_hovered_candidate(None)
            if self._pointer_gesture and self._pointer_gesture["kind"] in {"candidate", "blocked"}:
                self.cancel_pointer_gesture()
        self.update()

    def set_candidate_input_enabled(self, enabled):
        """Compatibility for callers that explicitly request an exclusive tool."""
        self.set_input_mode("candidate" if enabled else "manual")

    def set_input_mode(self, mode):
        if mode not in {"mixed", "manual", "candidate"}:
            raise ValueError("未知测量交互方式")
        if mode != self.input_mode:
            self.cancel_pointer_gesture()
            self.gestureCancelled.emit()
            self._last_candidate_click = None
        self.input_mode = mode
        self.candidate_input_enabled = mode != "manual"
        self._update_pointer_cursor()
        self.update()

    def set_manual_gesture(self, gesture):
        if gesture not in {"click", "drag"}:
            raise ValueError("未知手工测量手势")
        self.manual_gesture = gesture

    def cancel_pointer_gesture(self, *, cancel_draft=True, reason="cancelled"):
        gesture, self._pointer_gesture = self._pointer_gesture, None
        if gesture is not None:
            self._trace_pointer("press_cancelled", gesture=gesture, cancel_draft=bool(cancel_draft), reason=reason)
            self.controller.release_pointer_frame(gesture["lease"])
            if cancel_draft and gesture.get("started"):
                self.gestureCancelled.emit()
        self._update_pointer_cursor()

    def _candidates_changed(self, *_):
        self._candidate_state_changed()

    def set_selected_candidate(self, candidate_id):
        self.selected_candidate_id = str(candidate_id) if candidate_id is not None else None
        self.update()

    def _candidate_state_changed(self):
        if not self._candidates_visible():
            self._set_hovered_candidate(None)
        gesture = self._pointer_gesture
        # Manual input owns its own source/field identity, never the candidate
        # batch. A refresh may cancel a pending pick, not a manual first point
        # or an already-promoted drag that began over a candidate.
        if gesture and gesture["kind"] in {"candidate", "blocked"} and not gesture.get("started"):
            current = {item.get("candidate_id"): item.get("state", "available")
                       for item in self.controller.candidates}
            if (gesture["batch"] != self._candidate_batch_identity()
                    or not self._candidates_visible()
                    or any(current.get(key) != state for key, state in gesture["candidates"].items())):
                self.cancel_pointer_gesture(reason="candidate_changed")
        self._update_pointer_cursor()
        self.update()

    def _update_pointer_cursor(self):
        pointing = (self.input_mode != "manual" and not self.controller.draft
                    and self.hovered_candidate_id is not None)
        self.setCursor(Qt.CursorShape.PointingHandCursor if pointing else Qt.CursorShape.CrossCursor)

    def _set_hovered_candidate(self, candidate_id):
        if candidate_id != self.hovered_candidate_id:
            self.hovered_candidate_id = candidate_id
            self.candidateHovered.emit(candidate_id)
            self._update_pointer_cursor()
            self.update()

    def _candidates_visible(self):
        # A repaint can display a newer frame than controller.presented until
        # the end of paintEvent. Gate the pixels being drawn, not the old token.
        return self.candidate_visible and self.controller.candidates_aligned_for(self.frame)

    def _visible_viewport(self):
        """Return the scroll viewport in canvas coordinates, not the full image."""
        rect = QRectF(self.rect())
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                viewport = parent.viewport()
                origin = self.mapFrom(viewport, viewport.rect().topLeft())
                return rect.intersected(QRectF(origin, viewport.size()))
            parent = parent.parentWidget()
        return rect

    def set_frame(self, frame):
        old_size = self.frame.size if self.frame is not None else None
        old_source = (self.frame.source_id, self.frame.generation) if self.frame is not None else None
        new_source = (frame.source_id, frame.generation) if frame is not None else None
        if old_source != new_source or old_size != (frame.size if frame is not None else None):
            self.cancel_pointer_gesture()
        self.frame = frame
        if frame is None:
            self.image = None
            self._update_pixel_extent()
            self.update()
            return
        pixels = frame.pixels
        fmt = {"GRAY": QImage.Format.Format_Grayscale8,
               "BGR": QImage.Format.Format_BGR888,
               "BGRA": QImage.Format.Format_ARGB32}[frame.pixel_format]
        self.image = QImage(pixels.data, frame.size[0], frame.size[1], pixels.strides[0], fmt)
        if old_size != frame.size:
            self._update_pixel_extent()
        self.update()

    def _update_pixel_extent(self):
        # QScrollArea supplies explicit scrollbars for images larger than the
        # viewport. Never silently shrink a 768/2K/4K frame to make it fit.
        if self.frame is None or self.display_mode == "fit":
            self.setMinimumSize(320, 240)
        else:
            scale, _ = self._transform()
            self.setMinimumSize(max(320, math.ceil(self.frame.size[0] * scale)),
                                max(240, math.ceil(self.frame.size[1] * scale)))

    def _field_changed(self, *_):
        self.cancel_pointer_gesture()
        self._last_candidate_click = None
        self.cursor_point = None
        self.selected_candidate_id = None
        self._hovered_record_id = None
        self._set_hovered_candidate(None)
        self._record_ordinals.clear()
        self._next_record_ordinal = 1
        self._annotation_cache_key = None
        self._annotation_layout_cache.clear()
        self._label_placements = {}
        self.update()

    def _transform(self):
        if self.frame is None:
            return 1.0, (0.0, 0.0)
        width, height = self.frame.size
        dpr = self.devicePixelRatioF()
        if self.display_mode == "fit":
            scale = min(self.width() / width, self.height() / height)
        elif self.display_mode == "percent":
            scale = self.display_value / 100 / dpr
        elif self.display_mode == "magnification" and self.screen_pixel_mm and self.controller.calibration:
            scale = self.controller.calibration.x_um * self.display_value / (self.screen_pixel_mm * 1000 * dpr)
        else:
            scale = 1 / dpr
        # Align image edges to device pixels. Actual-pixel mode never resamples
        # merely because the application is running at 125/150/200% DPI.
        return scale, (round((self.width() - width * scale) * dpr / 2) / dpr,
                       round((self.height() - height * scale) * dpr / 2) / dpr)

    def set_display_mode(self, mode, value=100.0, screen_pixel_mm=None):
        if mode not in {"pixel", "fit", "percent", "magnification"}:
            raise ValueError("未知显示模式")
        if self.controller.pending_classification or self.controller._confirmation:
            raise ValueError("请先核对待确认测量，再更换显示模式。")
        if self.controller._jobs:
            raise ValueError("请先完成或取消当前测量，再更换显示模式。")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("显示比例必须大于零。")
        if mode == "magnification" and (not self.controller.calibration or not self.controller.calibration.isotropic
                or screen_pixel_mm is None or not math.isfinite(screen_pixel_mm) or screen_pixel_mm <= 0):
            raise ValueError("指定倍率需要等比例显微镜标尺和显示器标线长度。")
        if self.frame is not None and mode in {"percent", "magnification"}:
            scale = (value / 100 if mode == "percent" else
                     self.controller.calibration.x_um * value / (screen_pixel_mm * 1000))
            extent = max(self.frame.size) * scale / self.devicePixelRatioF()
            if not math.isfinite(extent) or extent > 16777215:
                raise ValueError("显示范围过大，请降低比例／倍率，并核对屏幕标线的毫米长度。")
        self.cancel_pointer_gesture()
        self.gestureCancelled.emit()
        self.display_mode, self.display_value = mode, value
        self.screen_pixel_mm = screen_pixel_mm
        self.cursor_point = None
        self.display_revision += 1
        self._update_pixel_extent()
        self.update()

    def fit_view(self):
        self.set_display_mode("fit")

    def _overlay_items(self):
        """Snapshot overlay geometry in screen space; no source pixels are edited."""
        scale, offset = self._transform()
        def point(value):
            return (value[0] * scale + offset[0], value[1] * scale + offset[1])
        items = []
        retention = getattr(self.controller, "selection_retention", None)
        if retention is not None:
            for mark in retention.visible_marks():
                items.append(dict(key="retained:" + mark["record_id"], start=point(mark["start"]),
                                  end=point(mark["end"]), label=f"#{mark['number']}", kind="retained",
                                  color=mark["color"], priority=-100))
        session = self.controller.session
        for record_id in self.controller.current_records:
            record = session.measurements.get(record_id)
            endpoints = getattr(record, "endpoints", None) if record else None
            if not endpoints or len(endpoints) != 2:
                continue
            provenance = record.provenance or {}
            number = provenance.get("measurement_number")
            if number is None:
                if record_id not in self._record_ordinals:
                    self._record_ordinals[record_id] = self._next_record_ordinal
                    self._next_record_ordinal += 1
                number = self._record_ordinals[record_id]
            elif isinstance(number, int):
                self._next_record_ordinal = max(self._next_record_ordinal, number + 1)
            items.append(dict(key="record:" + record_id, start=point(endpoints[0]),
                              end=point(endpoints[1]), label=f"#{number}", kind="record",
                              color=category_color(session, record.operator_id, record.direction, record.material_id),
                              priority=180 if record_id == self._hovered_record_id else
                                  120 if record_id == self.controller.current_records[-1] else 0))
        if self._candidates_visible():
            for index, candidate in enumerate(self.controller.candidates):
                state = candidate.get("state", "available")
                if state not in {"available", "confirming"}:
                    continue
                candidate_id = str(candidate.get("candidate_id", f"legacy-{index}"))
                active = candidate_id in (self.selected_candidate_id, self.hovered_candidate_id)
                label = f"C{candidate.get('ordinal', index + 1)}"
                if state == "confirming":
                    label += " · 待计入"
                items.append(dict(key="candidate:" + candidate_id, candidate_id=candidate_id,
                                  start=point(candidate["start"]), end=point(candidate["end"]),
                                  label=label, kind="candidate", state=state, active=active,
                                  color="#006fb9" if active else "#dc9800",
                                  priority=200 if active else 60))
        pending = [(intent, result, f"待确认 {index}") for index, (intent, result)
                   in enumerate(self.controller.pending_classification, 1)]
        request = self.controller._confirmation
        if request and request["kind"] == "manual":
            pending.extend((intent, result, "确认核验中") for intent, result in request["intents"])
        for intent, result, prefix in pending:
            items.append(dict(key="pending:" + intent.id, start=point(result["start"]),
                              end=point(result["end"]), kind="pending", color="#a45300",
                              label=f"{prefix} · #{intent.measurement_number or intent.id[:6]}（未入账）",
                              priority=250))
        protected = []
        if self.controller.draft:
            start = self.controller.draft[1]
            protected.append((point(start), point(self.cursor_point or start)))
        return items, protected

    def _arrange_labels(self, items, protected):
        metrics = self.fontMetrics()
        annotations = [Annotation(item["key"], item["start"], item["end"],
                                  metrics.horizontalAdvance(item["label"]) + 10.,
                                  metrics.height() + 4., item["priority"]) for item in items]
        viewport = self._visible_viewport()
        key = (tuple((a.key, a.start, a.end, a.width, a.height, a.priority) for a in annotations),
               tuple(protected), (viewport.x(), viewport.y(), viewport.width(), viewport.height()))
        if key != self._annotation_cache_key:
            # Keep geometry layouts warm across candidate and retained-marker
            # updates; ordinary preview frames never move annotation labels.
            placements = self._annotation_layout_cache.get(key)
            if placements is None:
                # Prefer the most recent layout with the same candidate/draft
                # kind and viewport. Visible and temporarily gated candidates
                # each retain their own warm record geometry. The layout helper
                # still rejects unsafe rectangles and resets after deletions.
                auxiliary = {item.key for item in annotations if not item.key.startswith("record:")}
                seed = {}
                for old_key in reversed(self._annotation_layout_cache):
                    if old_key[2] != key[2] or auxiliary != {
                            item[0] for item in old_key[0] if not item[0].startswith("record:")}:
                        continue
                    seed = dict(previous_annotations=[Annotation(*item) for item in old_key[0]],
                                previous_placements=self._annotation_layout_cache[old_key],
                                previous_protected_segments=old_key[1])
                    break
                placements = layout_annotations(annotations, viewport, protected_segments=protected, **seed)
                self._annotation_layout_cache[key] = placements
                while len(self._annotation_layout_cache) > 4:
                    self._annotation_layout_cache.popitem(last=False)
            else:
                self._annotation_layout_cache.move_to_end(key)
            self._label_placements = placements
            self._annotation_cache_key = key
        return self._label_placements

    def paintEvent(self, event):
        painter = QPainter(self)
        from fdm.ui.theme import semantic_colors
        colors = semantic_colors(self.palette())
        painter.fillRect(self.rect(), QColor(colors.canvas))
        if self.frame is None:
            painter.setPen(QColor(colors.canvas_text))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待实时画面\n在采集栏连接设备，在右侧完成试验准备")
            painter.end()
            self._painted_candidates = ()
            self._painted_candidate_batch = None
            return
        frame = self.frame
        scale, offset = self._transform()
        painter.save()
        painter.translate(*offset)
        painter.scale(scale, scale)
        painter.drawImage(QPointF(0, 0), self.image)
        painter.restore()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        items, protected = self._overlay_items()
        placements = self._arrange_labels(items, protected)
        # Paint leaders first, geometry second, labels last. Label placement
        # protects every line and endpoint, including the unfinished gesture.
        for item in items:
            placement = placements.get(item["key"])
            if placement:
                painter.save()
                if item["kind"] == "retained":
                    painter.setOpacity(.7)
                leader_color = QColor(item["color"])
                leader_color.setAlpha(125)
                painter.setPen(QPen(leader_color, .8))
                painter.drawLine(placement.anchor, placement.leader_end)
                painter.restore()
        for item in items:
            start, end = QPointF(*item["start"]), QPointF(*item["end"])
            if item["kind"] == "retained":
                painter.save()
                painter.setOpacity(.7)
                painter.setPen(QPen(QColor(item["color"]), 1.5, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(start, end)
                for endpoint in (start, end):
                    painter.drawEllipse(endpoint, 3., 3.)
                painter.restore()
            elif item["kind"] == "candidate":
                active = item["active"]
                painter.save()
                painter.setOpacity(1. if active else self.candidate_opacity)
                painter.setPen(QPen(QColor(25, 30, 35, 190), 4. if active else 3.5))
                painter.drawLine(start, end)
                pen = QPen(QColor(item["color"]), 2. if active else 1.5)
                if item["state"] == "confirming":
                    pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(start, end)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for endpoint in (start, end):
                    painter.drawEllipse(endpoint, 3., 3.)
                painter.restore()
            elif item["kind"] == "pending":
                painter.setPen(QPen(QColor(item["color"]), 2., Qt.PenStyle.DashLine))
                painter.drawLine(start, end)
            else:
                painter.setPen(QPen(QColor(255, 255, 255, 180), 3.5))
                painter.drawLine(start, end)
                painter.setPen(QPen(QColor(item["color"]), 1.5))
                painter.drawLine(start, end)
        if protected:
            start, end = protected[0]
            session = self.controller.session
            color = category_color(session, session.active_operator_id,
                                   session.active_direction, self.active_material_id)
            painter.setPen(QPen(QColor(color), 2.))
            painter.drawLine(QPointF(*start), QPointF(*end))
            painter.drawEllipse(QPointF(*start), 3., 3.)
        for item in items:
            placement = placements.get(item["key"])
            if placement is None:
                continue
            rect = placement.rect
            painter.save()
            if item["kind"] == "retained":
                painter.setOpacity(.7)
            painter.fillRect(rect, QColor(255, 255, 255, 232))
            painter.fillRect(QRectF(rect.left(), rect.top(), 3., rect.height()), QColor(item["color"]))
            painter.setPen(QColor("#111820"))
            painter.drawText(rect.adjusted(5., 0., -2., 0.), Qt.AlignmentFlag.AlignVCenter, item["label"])
            painter.restore()
        painter.end()
        # This snapshot is updated only after a paint, never from asynchronous
        # candidate notifications. Stale visible geometry blocks an accidental
        # manual fallback until Qt has actually removed it from the screen.
        self._painted_candidates = tuple(dict(item,
            label_rect=QRectF(placements[item["key"]].rect) if item["key"] in placements else None)
            for item in items if item["kind"] == "candidate")
        self._painted_candidate_batch = self._candidate_batch_identity()
        # Publish only after painting; set_frame merely schedules a repaint.
        self.controller.mark_presented(PresentedFrameToken(
            frame, self.controller.field_epoch, self.display_revision, scale, offset))
        candidates_visible = any(item["kind"] == "candidate" for item in items)
        self.controller.record_candidate_visibility(frame, candidates_visible,
                                                    display_enabled=self.candidate_visible)
        if candidates_visible:
            self.controller.record_candidate_painted(frame)

    def candidate_hits(self, position, *, include_pending=False):
        """Return available IDs; near ties require the user's disambiguation."""
        if self.frame is None or not self._candidates_visible():
            return []
        items, protected = self._overlay_items()
        placements = self._arrange_labels(items, protected)
        point = (position.x(), position.y())
        hits = []
        for item in items:
            if (item["kind"] != "candidate"
                    or item["state"] not in ({"available", "confirming"} if include_pending else {"available"})):
                continue
            placement = placements.get(item["key"])
            label_hit = placement is not None and placement.rect.contains(position)
            distance = segment_distance(point, item["start"], item["end"])
            if label_hit or distance <= 8.:
                hits.append((0. if label_hit else distance, item["candidate_id"]))
        hits.sort(key=lambda hit: hit[0])
        if not hits:
            return []
        # The same pointer may hit two short adjacent cross-sections; never
        # silently choose between similarly close lines merely by list order.
        return [candidate_id for distance, candidate_id in hits if distance <= hits[0][0] + 2.]

    @staticmethod
    def _hits_in_shapes(position, shapes):
        point = (position.x(), position.y())
        hits = []
        for item in shapes:
            rect = item.get("label_rect")
            label_hit = rect is not None and rect.contains(position)
            distance = segment_distance(point, item["start"], item["end"])
            if label_hit or distance <= 8.:
                hits.append((0. if label_hit else distance, item["candidate_id"]))
        hits.sort(key=lambda item: item[0])
        return [] if not hits else [key for distance, key in hits if distance <= hits[0][0] + 2.]

    def _point(self, event):
        token = self.controller.presented
        if token is None:
            return None
        return token.image_point(event.position().x(), event.position().y())

    @staticmethod
    def _inside(point, token):
        return (point is not None and all(math.isfinite(value) for value in point)
                and 0 <= point[0] < token.frame.size[0]
                and 0 <= point[1] < token.frame.size[1])

    def _recent_candidate_hit(self, position):
        recent = self._last_candidate_click
        return (recent is not None and self.candidate_visible
                and recent[2] == self._candidate_batch_identity()
                and time.monotonic() - recent[0] <= QApplication.doubleClickInterval() / 1000.
                and (position - recent[1]).manhattanLength() <= QApplication.startDragDistance())

    def _candidate_batch_identity(self):
        context = getattr(self.controller, "_automatic_context", None)
        if context is not None:
            # Task identity exists before candidates arrive, so completion of
            # the same analysis does not look like a new batch to a drag.
            return (id(context), context.get("task_id"), context.get("batch_id"))
        return tuple(item.get("candidate_id", id(item)) for item in self.controller.candidates)

    def _trace_pointer(self, name, event=None, *, gesture=None, **details):
        trace = getattr(self.controller, "trace_input_event", None)
        if trace is None:
            return
        gesture = self._pointer_gesture if gesture is None else gesture
        if event is not None:
            details.update(button=event.button().value, buttons=event.buttons().value,
                           x=float(event.position().x()), y=float(event.position().y()))
        if gesture is not None:
            details.update(pointer_id=gesture["pointer_id"], kind=gesture["kind"],
                           dragging=bool(gesture["started"]),
                           candidate_ids=",".join(gesture["candidates"]))
        trace(name, **details)

    def _begin_pointer_gesture(self, event):
        self.cancel_pointer_gesture()
        token = self.controller.presented
        if token is None or token.frame is not self.frame:
            return
        position = QPointF(event.position())
        point = token.image_point(position.x(), position.y())
        force_manual = (self.input_mode == "mixed"
                        and bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
        draft = self.controller.draft
        shapes = self._painted_candidates if self.candidate_visible else ()
        ids = [] if draft or force_manual else self._hits_in_shapes(position, shapes)
        states = {item["candidate_id"]: item.get("state", "available")
                  for item in self.controller.candidates if item.get("candidate_id") in ids}
        if ids and (not self._candidates_visible()
                    or self._painted_candidate_batch != self._candidate_batch_identity()
                    or any(states.get(key) not in {"available", "confirming"} for key in ids)):
            self._trace_pointer("press_blocked", event, reason="painted_candidate_unavailable",
                                candidate_ids=",".join(ids))
            # A queued repaint can remove the stale geometry before the second
            # event of a double click. Keep that click sequence candidate-owned.
            self._last_candidate_click = (time.monotonic(), position, self._candidate_batch_identity())
            self.gestureRejected.emit("该候选已变化或暂不可用，本次点击未应用。")
            return
        if draft and self.input_mode != "candidate":
            kind = "finish"
        elif ids:
            kind = "candidate" if any(state == "available" for state in states.values()) else "blocked"
        elif not force_manual and self._recent_candidate_hit(position):
            kind = "blocked_recent"
        elif self.input_mode == "candidate" and not force_manual:
            return
        else:
            kind = "manual"
        if kind in {"manual", "finish"} and not self._inside(point, token):
            return
        try:
            lease = self.controller.retain_pointer_frame(token)
        except (ValueError, RuntimeError) as exc:
            self.gestureRejected.emit(str(exc))
            return
        self._pointer_gesture = {"token": token, "lease": lease, "position": position,
                                 "point": point, "kind": kind, "candidates": states,
                                 "candidate_shapes": tuple(shape for shape in shapes if shape["candidate_id"] in ids),
                                 "started": False, "draft": draft, "force_manual": force_manual,
                                 "batch": self._candidate_batch_identity(), "pointer_id": uuid.uuid4().hex}
        self._trace_pointer("press", event)

    def _pointer_current(self, gesture):
        return (self.controller.pointer_token_is_current(gesture["token"])
                and gesture["token"].display_revision == self.display_revision
                and (gesture["kind"] not in {"candidate", "blocked"} or gesture["started"]
                     or gesture["batch"] == self._candidate_batch_identity()))

    def _start_pointer_manual(self, gesture):
        if (not self._pointer_current(gesture) or not self._inside(gesture["point"], gesture["token"])):
            self.cancel_pointer_gesture()
            return False
        gesture["started"] = True
        self.manualGestureStarted.emit(gesture["point"], gesture["token"])
        self._trace_pointer("manual_start", gesture=gesture)
        # The workspace validates readiness synchronously and the controller
        # obtains its own draft lease. A rejected start must not become an end.
        if self.controller.draft is None:
            self.cancel_pointer_gesture(cancel_draft=False)
            return False
        return True

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            if self.input_mode != "manual":
                self._suppress_candidate_release = False
                self._begin_pointer_gesture(event)
                event.accept()
                return
            point = self._point(event)
            if point:
                self._trace_pointer("legacy_press", event)
                self.pointPressed.emit(point)
        self.update()

    def mouseReleaseEvent(self, event):
        self._trace_pointer("release", event)
        if event.button() == Qt.MouseButton.RightButton:
            self.cancel_pointer_gesture()
            self.gestureCancelled.emit()
            self.categoryRequested.emit(event.globalPosition().toPoint())
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.input_mode != "manual":
                self._finish_pointer_gesture(event)
                event.accept()
                return
            point = self._point(event)
            if point:
                self.pointReleased.emit(point)

    def _finish_pointer_gesture(self, event):
        if self._suppress_candidate_release:
            self._suppress_candidate_release = False
            self.cancel_pointer_gesture()
            return
        gesture = self._pointer_gesture
        if gesture is None:
            return
        try:
            if not self._pointer_current(gesture):
                return
            point = gesture["token"].image_point(event.position().x(), event.position().y())
            if gesture["kind"] == "finish":
                if (self.controller.draft is gesture["draft"] and self._inside(point, gesture["token"])
                        and tuple(point) != tuple(gesture["draft"][1])):
                    self._trace_pointer("manual_finish", gesture=gesture, submit_reason="second_point")
                    self.manualGestureFinished.emit(point)
                return
            # A release coordinate can differ even if no held movement was
            # delivered. Only mouseMoveEvent may promote this press to drag.
            if gesture["started"]:
                if self._inside(point, gesture["token"]) and tuple(point) != tuple(gesture["point"]):
                    self._trace_pointer("manual_finish", gesture=gesture, submit_reason="held_drag")
                    self.manualGestureFinished.emit(point)
                    gesture["started"] = False
                return
            if gesture["kind"] in {"candidate", "blocked"}:
                if not self._candidates_visible():
                    return
                hits = self._hits_in_shapes(event.position(), gesture["candidate_shapes"])
                hits = [key for key in hits if key in gesture["candidates"]]
                states = {item["candidate_id"]: item.get("state", "available")
                          for item in self.controller.candidates if item.get("candidate_id") in hits}
                available = [key for key in hits if states.get(key) == "available"]
                if not available:
                    if any(states.get(key) == "confirming" for key in hits):
                        self.gestureRejected.emit("该候选正在核验，请等待计入。")
                    return
                self._last_candidate_click = (time.monotonic(), QPointF(event.position()),
                                              self._candidate_batch_identity())
                self.controller.candidate_interaction_started()
                if len(available) == 1 and len(hits) == 1:
                    self.set_selected_candidate(available[0])
                    self.candidateClicked.emit(available[0])
                else:
                    self.candidateAmbiguous.emit(available)
                return
            if gesture["kind"] == "manual":
                if self._start_pointer_manual(gesture):
                    # A short empty-space click leaves the first point intact.
                    gesture["started"] = False
        finally:
            self.cancel_pointer_gesture()
            self.update()

    def mouseMoveEvent(self, event):
        self._trace_pointer("move", event)
        gesture = self._pointer_gesture
        if gesture is not None:
            if (not event.buttons() & Qt.MouseButton.LeftButton
                    or not self._pointer_current(gesture)):
                self.cancel_pointer_gesture()
            elif (gesture["kind"] != "finish" and not gesture["started"]
                    and (self.input_mode == "mixed" or gesture["force_manual"])
                    and (event.position() - gesture["position"]).manhattanLength() >= QApplication.startDragDistance()):
                self._start_pointer_manual(gesture)
        self.cursor_point = self._point(event)
        hits = self.candidate_hits(event.position())
        self._set_hovered_candidate(hits[0] if len(hits) == 1 else None)
        point = (event.position().x(), event.position().y())
        record_hits = []
        for item in self._overlay_items()[0]:
            if item["kind"] != "record":
                continue
            distance = segment_distance(point, item["start"], item["end"])
            placement = self._label_placements.get(item["key"])
            if distance <= 8. or placement is not None and placement.rect.contains(event.position()):
                record_hits.append((distance, item["key"].removeprefix("record:")))
        hovered_record = min(record_hits)[1] if record_hits else None
        if hovered_record != self._hovered_record_id:
            self._hovered_record_id = hovered_record
            self.update()
        if self.controller.draft:
            self.update()

    def leaveEvent(self, event):
        self._set_hovered_candidate(None)
        self._hovered_record_id = None
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            event.accept()
        else:
            if (event.button() == Qt.MouseButton.LeftButton and self.input_mode != "manual"
                    and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                    and self._recent_candidate_hit(event.position())):
                self.cancel_pointer_gesture()
                self._suppress_candidate_release = True
                event.accept()
                return
            # A fast second left click remains the second manual point.
            self.mousePressEvent(event)

    def resizeEvent(self, event):
        self.cancel_pointer_gesture()
        self.gestureCancelled.emit()
        self.display_revision += 1
        super().resizeEvent(event)

    def event(self, event):
        if event.type() == QEvent.Type.DevicePixelRatioChange:
            self.cancel_pointer_gesture()
            self.gestureCancelled.emit()
            self._update_pixel_extent()
            self.display_revision += 1
            self.update()
        elif event.type() in {QEvent.Type.Hide, QEvent.Type.WindowDeactivate, QEvent.Type.UngrabMouse}:
            self.cancel_pointer_gesture()
        elif event.type() == QEvent.Type.FocusOut:
            # Opening a menu must not leave a mouse-down intent alive. A
            # completed first manual point still permits choosing a category.
            self.cancel_pointer_gesture()
        return super().event(event)
