"""Non-mutating notice for manually placed labels outside an export region."""
from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QFontMetricsF

from fdm.ui.rendering import measurement_label_font
from fdm.ui.professional_rendering import presentation_result, professional_label_layout, professional_text_scale


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
        metrics = QFontMetricsF(measurement_label_font(canvas._settings))
        scale = professional_text_scale(canvas._settings, canvas.image_to_widget)
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
            _, resolved = professional_label_layout(definition, result, index, canvas.image_to_widget, layout,
                decimals=canvas._settings.professional_measurement_label_decimals, metrics=metrics, scale=scale,
                parallel=canvas._settings.length_measurement_label_style.parallel_to_line)
            if not frame.contains(resolved.text_rect):
                clipped.append(f"P{index}")
    return tuple(clipped)
