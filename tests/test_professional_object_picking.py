"""Professional canvas selection and document-level associative control points."""
from dataclasses import replace
import math

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest

from fdm.construction_geometry import ConstructionEntity, LineDefinition
from fdm.geometry import Point
from fdm.metrology import ControlPointRef, MeasurementFeature
from fdm.models import ImageDocument
from fdm.services.metrology_service import evaluate_document, freeze_feature, resolve_feature
from test_metrology_workspace import mounted_window, _begin, _click, _distance, _mouse, _paint


def test_pick_move_fixed_roundtrip_freeze_delete_and_undo(mounted_window):
    window, doc, workspace, _ = mounted_window()
    guide = ConstructionEntity("guide", "测试基准", LineDefinition(Point(20, 60), Point(90, 60)))
    doc.add_construction_entity(guide)
    canvas = workspace.controller().canvas
    linked = _distance(workspace, (20.2, 60.1), (80, 110))
    feature = doc.get_professional_feature(linked.feature_ids[0])
    assert feature.point_refs[0].source_id == "guide" and feature.point_refs[1] is None
    workspace.panel.follow_sources.setChecked(False)
    fixed = _distance(workspace, (20.2, 60.1), (80, 110))
    assert not doc.get_professional_feature(fixed.feature_ids[0]).point_refs
    move = replace(guide, definition=LineDefinition(Point(40, 60), Point(90, 60)))
    workspace.change(doc, "移动基准", lambda: doc.replace_construction_entity(guide.id, move))
    results = workspace.controller().results()
    assert results[linked.id].primary.value == pytest.approx(math.hypot(40, 50))
    assert results[fixed.id].primary.value == pytest.approx(math.hypot(60, 50))
    restored = ImageDocument.from_dict(doc.to_dict())
    assert evaluate_document(restored)[linked.id].primary == results[linked.id].primary
    frozen = freeze_feature(doc, feature)
    assert not frozen.point_refs and frozen.points_px[0] == Point(40, 60)
    workspace.change(doc, "删除基准", lambda: doc.remove_construction_entity(guide.id))
    assert not workspace.controller().results()[linked.id].valid
    assert workspace.controller().results()[fixed.id].valid
    window.undo_current_document()
    assert workspace.controller().results()[linked.id].primary.value == pytest.approx(math.hypot(40, 50))
    assert not canvas._drawing_line


def test_professional_endpoint_and_whole_line_chain(mounted_window):
    _, doc, workspace, _ = mounted_window()
    canvas = workspace.controller().canvas
    original = _distance(workspace, (20, 40), (120, 40))
    source = doc.get_professional_feature(original.feature_ids[0])
    second = _distance(workspace, (120.1, 40), (160, 115))
    dependent = doc.get_professional_feature(second.feature_ids[0])
    assert dependent.point_refs[0].source_id == source.id
    _begin(workspace, "point_line_distance")
    _click(canvas, (60, 100))
    pos = canvas.image_to_widget(Point(72, 40))
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, pos))
    controller = workspace.controller()
    assert "关联" in workspace.panel.prompt.text() and controller.choices
    _paint(controller)
    _click(canvas, (72, 40))
    perpendicular = doc.professional_measurements[-1]
    assert perpendicular.kind == "point_line_distance"
    assert doc.get_professional_feature(perpendicular.feature_ids[1]).professional_source_id == source.id
    assert controller.results()[perpendicular.id].primary.value == pytest.approx(60)
    changed = replace(source, points_px=(Point(20, 50), Point(120, 50)))
    doc.apply_professional_state([changed if f.id == source.id else f for f in doc.professional_features],
                                doc.professional_measurements)
    assert controller.results()[perpendicular.id].primary.value == pytest.approx(50)
    assert controller.results()[second.id].primary.value == pytest.approx(math.hypot(40, 65))


def test_coincident_points_cycle_preserves_chosen_identity_and_backstep(mounted_window):
    _, doc, workspace, _ = mounted_window()
    guides = [ConstructionEntity(name, name, LineDefinition(Point(35, 60), finish))
              for name, finish in (("横线", Point(105, 60)), ("竖线", Point(35, 130)))]
    for guide in guides:
        doc.add_construction_entity(guide)
    controller = workspace.controller()
    canvas = controller.canvas
    _begin(workspace, "distance")
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, canvas.image_to_widget(Point(35, 60))))
    identities = {c.source_id for c in controller.choices}
    assert {g.id for g in guides} <= identities
    first = controller.choices[controller.choice]
    QTest.keyClick(canvas, Qt.Key.Key_Backtab)
    assert controller.choices[controller.choice].identity != first.identity
    QTest.keyClick(canvas, Qt.Key.Key_Tab)
    assert controller.choices[controller.choice].identity == first.identity
    for _ in controller.choices:
        previous = controller.choice
        QTest.keyClick(canvas, Qt.Key.Key_Tab)
        assert controller.choice != previous
        selected = controller.choices[controller.choice]
        if selected.source_id in identities - {first.source_id} and selected.source_type == "construction":
            break
    assert selected.source_id != first.source_id
    _click(canvas, (35, 60))
    assert controller.point_refs[0].source_id == selected.source_id
    canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier))
    assert not controller.points and not doc.professional_measurements
    workspace.cancel()
    pos = canvas.image_to_widget(Point(170, 140))
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, pos))
    assert canvas._drawing_line is None
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, pos))
    assert not doc.measurements and not doc.professional_measurements


def test_cycles_and_missing_professional_landmarks_stay_invalid():
    doc = ImageDocument("doc", "", (256, 192))
    a = MeasurementFeature("a", "doc", "line", (Point(5, 10), Point(25, 35)),
                           point_refs=(ControlPointRef("professional", "b", "vertex:0", "endpoint"), None))
    b = MeasurementFeature("b", "doc", "line", professional_source_id="a")
    doc.apply_professional_state([a, b], [])
    assert "循环" in resolve_feature(doc, a).error
    doc.apply_professional_state([a], [])
    assert "删除" in resolve_feature(doc, a).error


def test_choose_whole_source_back_from_next_step_returns_picker_to_canvas(mounted_window):
    _, doc, workspace, _ = mounted_window()
    guide = ConstructionEntity("guide", "一号线", LineDefinition(Point(20, 40), Point(100, 40)))
    doc.add_construction_entity(guide)
    workspace.refresh()
    _begin(workspace, "parallel_distance")
    controller = workspace.controller()
    box = workspace.panel.sources[0]
    index = box.findData(("construction", "guide"))
    # Qt's tuple user-data is a Python object on some supported versions.
    if index == -1:
        index = next(i for i in range(box.count()) if box.itemData(i) == ("construction", "guide"))
    box.setCurrentIndex(index)
    box.activated.emit(index)
    assert controller.slot == 1 and controller.features[0].source_ref.object_id == "guide"
    controller.step_back()
    assert controller.slot == 0 and controller.selections[0] is None and box.currentData() is None
    assert not controller.features and not controller.points


def test_circular_landmarks_use_calibrated_plane_and_snap_filter(mounted_window):
    from fdm.models import Calibration
    from fdm.services.object_snap_service import ObjectSnapSettings, SnapKind
    from fdm.services.metrology_service import professional_landmarks

    _, doc, workspace, _ = mounted_window()
    doc.calibration = Calibration("manual", .5, "µm", "test", 1)
    _begin(workspace, "circle")
    controller = workspace.controller()
    for xy in ((95, 90), (80, 120), (65, 90)):
        _click(controller.canvas, xy)
    feature = doc.professional_features[0]
    geometry = resolve_feature(doc, feature)
    landmarks = professional_landmarks(geometry, (doc.calibration.pixel_size_x, doc.calibration.pixel_size_y))
    assert landmarks["circle:quadrant:1"][1] == Point(80, 120)
    controller.canvas._object_snap_engine.update_settings(ObjectSnapSettings(enabled_kinds=frozenset({SnapKind.CENTER})))
    candidate = controller.canvas._query_object_snap(Point(80.1, 90))
    assert candidate.feature_key == "circle:center"
    assert controller.canvas._query_object_snap(Point(80, 120)) is None
