from PySide6.QtCore import Qt

from fdm.geometry import Line, Point
from fdm.measurement_annotation import MeasurementAnnotationLayout
from fdm.models import Measurement
from test_metrology_workspace import mounted_window


def test_legacy_label_and_result_hover_share_feedback_without_selection_or_dirty(mounted_window):
    window, document, workspace, _ = mounted_window()
    item = Measurement("legacy", document.id, None, "manual", line_px=Line(Point(20, 30), Point(100, 30)),
                       annotation_layout=MeasurementAnnotationLayout())
    document.add_measurement(item)
    document.select_measurement(None)
    window._update_ui_for_current_document()
    stamp = document.state_stamp
    canvas = window.current_canvas()
    model = window._records_controller.source_model
    canvas.measurementAnnotationHoverChanged.emit(document.id, item.id)
    assert model.data(model.index(0, 0), Qt.ItemDataRole.BackgroundRole) is not None
    workspace.legacy_hover(document.id, None)
    assert model.data(model.index(0, 0), Qt.ItemDataRole.BackgroundRole) is None
    proxy_index = window._records_controller.proxy_model.index(0, 0)
    window._inspector_records_pane.table.entered.emit(proxy_index)
    assert canvas._hovered_measurement_annotation_id == item.id
    assert document.view_state.selected_measurement_id is None
    assert document.state_stamp == stamp
