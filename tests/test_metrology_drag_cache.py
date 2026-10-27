"""The accelerated label drag must paint exactly the uncached presentation."""
from dataclasses import replace
import math

from PySide6.QtGui import QImage, QPainter

from fdm.geometry import Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.metrology import MeasurementDefinition, MeasurementFeature
from test_metrology_workspace import mounted_window


def pixels(controller, *, cached):
    dpr = controller.canvas.devicePixelRatioF()
    image = QImage(math.ceil(controller.canvas.width()*dpr), math.ceil(controller.canvas.height()*dpr), QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(0)
    painter = QPainter(image)
    drag = controller.drag
    if not cached:
        controller.drag = None
    try:
        controller.paint(painter)
    finally:
        painter.end()
        controller.drag = drag
    return bytes(image.constBits())


def test_cached_drag_matches_direct_pixels_through_layout_view_and_geometry_changes(mounted_window):
    window, document, workspace, _ = mounted_window()
    canvas = window.current_canvas(); controller = workspace.controller()
    features = [MeasurementFeature(f"f{i}", document.id, "line", (Point(20, 20+i*30), Point(150, 30+i*30))) for i in range(3)]
    definitions = [MeasurementDefinition(f"m{i}", document.id, "distance", (f.id,), annotation_layout=MeasurementAnnotationLayout(offset_px=Point(20, -14))) for i, f in enumerate(features)]
    document.apply_professional_state(features, definitions, selected_id="m0")
    controller.drag = ("m0", Point(0, 0), definitions[0].annotation_layout, Point(20, -14))
    for offset in (Point(0, 30), Point(80, -40), Point(-60, 10)):
        controller.preview_layouts["m0"] = definitions[0].annotation_layout.with_manual_offset(offset)
        assert pixels(controller, cached=True) == pixels(controller, cached=False)
    canvas._zoom *= 1.3
    canvas._pan = Point(canvas._pan.x+15, canvas._pan.y-10)
    assert pixels(controller, cached=True) == pixels(controller, cached=False)
    features[1] = replace(features[1], points_px=(Point(10, 40), Point(180, 60)))
    document.apply_professional_state(features, definitions, selected_id="m0")
    assert pixels(controller, cached=True) == pixels(controller, cached=False)
    controller.cancel()
    assert controller._drag_layer is None
