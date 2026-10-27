"""Non-mutating notice for manually placed labels outside an export region."""
from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QFontMetricsF

from fdm.geometry import Point
from fdm.ui.measurement_annotation_layout import resolve_linked_label
from fdm.ui.professional_rendering import presentation_result, result_caption


def clipped_manual_labels(canvas, frame: QRectF) -> tuple[str, ...]:
    document = canvas._document
    if document is None:
        return ()
    clipped = []
    from fdm.ui.rendering import resolve_measurement_annotation
    for index, measurement in enumerate(document.measurements, 1):
        layout = measurement.annotation_layout
        if layout is None or layout.placement_mode != "manual":
            continue
        resolved = resolve_measurement_annotation(measurement, document, canvas._settings, canvas.image_to_widget)
        if resolved is not None and not frame.contains(resolved.text_rect):
            clipped.append(f"M{index}")
    professional = getattr(canvas, "_professional_controller", None)
    if professional is not None:
        results = professional.results()
        focus = professional.focus()
        metrics = QFontMetricsF(canvas.font())
        for index, definition in enumerate(document.professional_measurements, 1):
            layout = definition.annotation_layout
            if layout is None or layout.placement_mode != "manual":
                continue
            if definition.focus_index is not None and definition.focus_index != focus:
                continue
            result = results.get(definition.id)
            if result is None:
                continue
            # Export includes labels independently of the screen's hidden /
            # selected-only setting. Compute the current persisted layout so a
            # warning also works before the next screen paint or after edits.
            result = presentation_result(document, definition, result)
            anchor = result.anchor_px
            offset = layout.offset_px or Point(65, -38)
            caption = result_caption(definition, result, index)
            resolved = resolve_linked_label(
                canvas.image_to_widget(anchor),
                canvas.image_to_widget(Point(anchor.x + offset.x, anchor.y + offset.y)),
                metrics.horizontalAdvance(caption) + 14, metrics.height() + 8,
                show_leader=layout.show_leader,
            )
            if not frame.contains(resolved.text_rect):
                clipped.append(f"P{index}")
    return tuple(clipped)
