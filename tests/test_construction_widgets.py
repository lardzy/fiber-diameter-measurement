from __future__ import annotations

import os
from types import MethodType, SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QImage
from PySide6.QtWidgets import QApplication, QMessageBox

import fdm.ui.main_window as main_window_module
from fdm.construction_document import make_construction_resolver
from fdm.construction_geometry import (
    ArraySide,
    CircleCenterRadiusDefinition,
    CommonTangentDefinition,
    CommonTangentMode,
    ConstructionEntity,
    ConstructionValidationError,
    FreePointDefinition,
    FrozenFeatureSnapshot,
    LineAxisConstraint,
    LineDefinition,
    LineExtent,
    MidpointDefinition,
    OffsetParallelDefinition,
    ParallelArrayDefinition,
    ParallelThroughPointDefinition,
    PerpendicularBisectorDefinition,
    PerpendicularDefinition,
    LiveFeatureRef,
    ResolvedCircle,
    SourceObjectKind,
)
from fdm.geometry import Line, Point
from fdm.history import DocumentChangeImpact
from fdm.models import Calibration, ImageDocument, Measurement
from fdm.settings import AppSettings
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.construction_widgets import (
    ConstructionContextWidget,
    ConstructionManagerPanel,
    construction_kind_label,
)
from fdm.ui.main_window import MainWindow
from fdm.ui.object_inspector import CurrentObjectInspector
from fdm.ui.widgets import MeasurementToolStrip


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _document(document_id: str, entities=()) -> ImageDocument:
    document = ImageDocument(
        id=document_id,
        path=f"/tmp/{document_id}.png",
        image_size=(120, 80),
        construction_entities=list(entities),
    )
    document.initialize_runtime_state()
    return document


def _workspace_history_window(
    documents: list[ImageDocument],
    current: ImageDocument,
) -> MainWindow:
    window = MainWindow()
    window.project.documents = list(documents)
    window.current_document = MethodType(lambda self: current, window)
    window._update_ui_for_current_document = MethodType(lambda self: None, window)
    window._refresh_document_analysis_validity = MethodType(
        lambda self, _document: None,
        window,
    )
    window._discard_detached_area_geometry = MethodType(
        lambda self, _before, _document: None,
        window,
    )
    return window


def test_manager_and_inspector_expose_unresolved_reason(app: QApplication) -> None:
    valid = ConstructionEntity(
        id="valid",
        name="基准点",
        definition=FreePointDefinition(Point(10.0, 12.0)),
    )
    invalid = ConstructionEntity(
        id="invalid",
        name="退化圆",
        definition=CircleCenterRadiusDefinition(Point(20.0, 20.0), 0.0),
    )
    document = _document("status-document", (valid, invalid))
    resolution_by_id = make_construction_resolver(document).resolve_all()

    panel = ConstructionManagerPanel()
    panel.setEntities(
        document.construction_entities,
        selected_id=invalid.id,
        resolution_by_id=resolution_by_id,
    )
    assert panel.tree.headerItem().text(1) == "状态"
    assert panel.tree.topLevelItem(0).text(1) == "有效"
    invalid_item = panel.tree.topLevelItem(1)
    assert invalid_item.text(1) == "不可解"
    assert "半径" in invalid_item.toolTip(1)
    assert "不可解" in invalid_item.toolTip(0)

    changes: list[tuple[str, str, object]] = []
    panel.metadataChangeRequested.connect(
        lambda entity_id, field, value: changes.append((entity_id, field, value))
    )
    invalid_item.setCheckState(2, Qt.CheckState.Unchecked)
    assert changes == [(invalid.id, "visible", False)]

    inspector = CurrentObjectInspector()
    inspector.set_context(
        document,
        settings=AppSettings(),
        construction_id=invalid.id,
    )
    summary = inspector._summary_label.text()
    assert "解析状态：不可解" in summary
    assert "原因：" in summary
    assert "半径" in summary

    panel.close()
    inspector.close()


def test_manager_reuses_rows_for_selection_and_metadata_updates(
    app: QApplication,
) -> None:
    first = ConstructionEntity(
        id="manager-first",
        name="第一对象",
        definition=FreePointDefinition(Point(10.0, 12.0)),
    )
    second = ConstructionEntity(
        id="manager-second",
        name="第二对象",
        definition=FreePointDefinition(Point(20.0, 22.0)),
    )
    document = _document("manager-incremental", (first, second))
    resolutions = make_construction_resolver(document).resolve_all()
    panel = ConstructionManagerPanel()
    panel.setEntities(
        document.construction_entities,
        selected_id=first.id,
        resolution_by_id=resolutions,
        content_revision=(1, 0),
    )
    first_item = panel.tree.topLevelItem(0)
    second_item = panel.tree.topLevelItem(1)

    panel.setEntities(
        document.construction_entities,
        selected_id=second.id,
        resolution_by_id=resolutions,
        content_revision=(1, 0),
    )
    assert panel.tree.topLevelItem(0) is first_item
    assert panel.tree.topLevelItem(1) is second_item
    assert panel.selectedIds() == (second.id,)

    changed_resolution = dict(resolutions)
    changed_resolution[first.id] = SimpleNamespace(
        valid=False,
        error=SimpleNamespace(message="测量来源已退化"),
    )
    panel.setEntities(
        document.construction_entities,
        selected_id=second.id,
        resolution_by_id=changed_resolution,
        content_revision=(1, 0),
    )
    assert panel.tree.topLevelItem(0) is first_item
    assert first_item.text(1) == "不可解"
    assert "测量来源已退化" in first_item.toolTip(1)

    updated = ConstructionEntity(
        id=first.id,
        name=first.name,
        definition=first.definition,
        visible=False,
        revision=1,
    )
    panel.setEntities(
        (updated, second),
        selected_id=second.id,
        resolution_by_id=resolutions,
        content_revision=(1, 1),
    )
    assert panel.tree.topLevelItem(0) is first_item
    assert first_item.checkState(2) == Qt.CheckState.Unchecked
    panel.close()


def test_main_window_caches_construction_resolution_across_metadata_updates(
    monkeypatch,
) -> None:
    entity = ConstructionEntity(
        id="resolution-cache-point",
        name="缓存点",
        definition=FreePointDefinition(Point(10.0, 12.0)),
    )
    document = _document("resolution-cache", (entity,))
    owner = SimpleNamespace(_construction_resolution_cache={})
    original_factory = main_window_module.make_construction_resolver
    calls = 0

    def counted_factory(target: ImageDocument):
        nonlocal calls
        calls += 1
        return original_factory(target)

    monkeypatch.setattr(
        main_window_module,
        "make_construction_resolver",
        counted_factory,
    )
    first = MainWindow._construction_resolutions(owner, document)
    second = MainWindow._construction_resolutions(owner, document)
    assert second is first
    assert calls == 1

    document.replace_construction_entity(
        entity.id,
        ConstructionEntity(
            id=entity.id,
            name=entity.name,
            definition=entity.definition,
            visible=False,
        ),
        mark_dirty=False,
    )
    third = MainWindow._construction_resolutions(owner, document)
    assert third is first
    assert calls == 1


def test_definition_kind_labels_reflect_real_line_extent_and_tangent_mode() -> None:
    line = ConstructionEntity(
        id="ray",
        name="",
        definition=LineDefinition(
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            LineExtent.RAY,
        ),
    )
    circle = ResolvedCircle(Point(0.0, 0.0), 5.0)
    source = FrozenFeatureSnapshot(circle)
    tangent = ConstructionEntity(
        id="inner-tangent",
        name="",
        definition=CommonTangentDefinition(
            source,
            FrozenFeatureSnapshot(ResolvedCircle(Point(20.0, 0.0), 4.0)),
            CommonTangentMode.INTERNAL,
        ),
    )
    horizontal = ConstructionEntity(
        id="horizontal",
        name="",
        definition=LineDefinition(
            Point(0.0, 2.0),
            Point(5.0, 9.0),
            LineExtent.INFINITE,
            LineAxisConstraint.HORIZONTAL,
        ),
    )
    assert construction_kind_label(line) == "射线"
    assert construction_kind_label(horizontal) == "水平辅助线"
    assert construction_kind_label(tangent) == "两圆内公切线"


def test_context_snapshot_updates_every_parameter_without_feedback(
    app: QApplication,
) -> None:
    widget = ConstructionContextWidget()
    emitted: list[tuple[str, object]] = []
    widget.parameterChanged.connect(
        lambda name, value: emitted.append((name, value))
    )
    widget.setDistanceUnit("mm", 2.0)
    widget.setCommandState(
        distance_px=30.0,
        count=7,
        both_sides=True,
        extend=True,
    )
    assert widget.distanceSpin.value() == pytest.approx(15.0)
    assert widget.countSpin.value() == 7
    assert widget.bothSidesCheck.isChecked()
    assert widget.extendCheck.isChecked()
    assert emitted == []
    widget.close()


@pytest.mark.parametrize(
    ("tool", "expected_prompt"),
    (
        ("point", "单击放点"),
        ("midpoint", "选择有限线段"),
        ("intersection", "选两对象；再选交点"),
        ("segment", "两端点 · Shift 正交 · Ctrl 像素中心"),
        ("ray", "起点和方向 · Shift 正交 · Ctrl 像素中心"),
        ("infinite_line", "直线两点 · Shift 正交 · Ctrl 像素中心"),
        ("horizontal_line", "通过点 · Ctrl 像素中心"),
        ("vertical_line", "通过点 · Ctrl 像素中心"),
        ("circle_center_radius", "指定圆心和圆周点"),
        ("circle_center_diameter", "指定圆心和圆周点"),
        ("circle_diameter_2p", "指定直径两端点"),
        ("circle_3p", "指定圆上三点"),
        ("parallel_through", "选择源线和通过点"),
        ("parallel_offset", "选择源线和偏移侧"),
        ("parallel_array", "选择源线和阵列侧"),
        ("perpendicular", "选择源线和通过点"),
        ("perpendicular_bisector", "选择有限线段"),
        ("concentric_circle", "选择源圆和新圆周点"),
        ("offset_circle", "选择源圆和偏移侧"),
        ("tangent_point_circle", "选择点和圆；再选切线"),
        ("common_tangent_external", "选择两圆；再选外公切线"),
        ("common_tangent_internal", "选择两圆；再选内公切线"),
        ("tangent_circle_ttr", "选择两对象；再选定半径解"),
        ("tangent_circle_3", "选择三对象；再选相切圆"),
    ),
)
def test_construction_prompts_are_concise_and_keep_modifier_help(
    app: QApplication,
    tool: str,
    expected_prompt: str,
) -> None:
    context = ConstructionContextWidget()
    document = SimpleNamespace(id="concise-prompt-document")
    host = SimpleNamespace(
        _construction_context_widget=context,
        _construction_tool_kind=tool,
    )
    host.current_document = MethodType(lambda self: document, host)

    MainWindow._on_canvas_construction_command_changed(
        host,
        document.id,
        {
            "tool": tool,
            "point_count": 0,
            "source_count": 0,
        },
    )

    assert context.promptLabel.text() == expected_prompt
    assert len(expected_prompt) <= 32
    assert not context.promptLabel.wordWrap()
    context.close()


def test_unchanged_construction_prompt_does_not_relayout_on_mouse_updates(
    app: QApplication,
) -> None:
    context = ConstructionContextWidget()
    document = SimpleNamespace(id="stable-prompt-document")
    refresh_calls: list[bool] = []
    host = SimpleNamespace(
        _construction_context_widget=context,
        _construction_tool_kind="segment",
        _measurement_tool_strip=SimpleNamespace(
            refreshContextLayout=lambda: refresh_calls.append(True)
        ),
    )
    host.current_document = MethodType(lambda self: document, host)
    payload = {
        "tool": "segment",
        "point_count": 0,
        "source_count": 0,
    }

    MainWindow._on_canvas_construction_command_changed(host, document.id, payload)
    MainWindow._on_canvas_construction_command_changed(host, document.id, payload)

    assert refresh_calls == [True]
    context.close()


def test_completed_construction_prompt_stays_inline_after_layout_refresh(
    app: QApplication,
) -> None:
    """A delayed size-hint refresh must not push the prompt onto a second row."""

    strip = MeasurementToolStrip()
    for index in range(8):
        strip.addModeAction(f"mode-{index}", QAction(f"测量工具 {index + 1}", strip))
    context = ConstructionContextWidget(strip)
    strip.setConstructionContextWidget(context)
    strip.setConstructionContextVisible(True)
    document = SimpleNamespace(id="construction-context-layout-document")
    host = SimpleNamespace(
        _construction_context_widget=context,
        _construction_tool_kind="segment",
        _measurement_tool_strip=strip,
    )
    host.current_document = MethodType(lambda self: document, host)
    try:
        MainWindow._on_canvas_construction_command_changed(
            host,
            document.id,
            {
                "tool": "segment",
                "point_count": 0,
                "source_count": 0,
            },
        )
        available_width = (
            strip._expanded_primary_width()
            + strip._top_row_layout.spacing()
            + context.sizeHint().width()
            + 80
        )
        strip.resize(available_width, strip.sizeHint().height())
        strip.show()
        strip.refreshContextLayout()
        app.processEvents()

        assert strip.isContextInline()

        MainWindow._on_canvas_construction_command_changed(
            host,
            document.id,
            {
                "tool": "segment",
                "point_count": 1,
                "source_count": 0,
            },
        )
        app.processEvents()
        progress_prompt = context.promptLabel.text()
        assert "1点" in progress_prompt
        assert "0源" not in progress_prompt
        assert strip.isContextInline()

        MainWindow._on_canvas_construction_command_changed(
            host,
            document.id,
            {
                "tool": "segment",
                "point_count": 0,
                "source_count": 0,
            },
        )
        app.processEvents()

        assert "Shift 正交" in context.promptLabel.text()
        assert "Ctrl 像素中心" in context.promptLabel.text()
        assert strip.isContextInline()
        assert not strip.isContextStacked()

        # A longer validation message may legitimately need the second row,
        # but returning to the short command prompt must shrink immediately;
        # no window resize or other incidental layout event should be needed.
        MainWindow._on_canvas_construction_command_changed(
            host,
            document.id,
            {
                "tool": "segment",
                "point_count": 1,
                "source_count": 0,
                "invalid_reason": "当前对象暂时无法构造，请重新选择有效的几何来源" * 4,
            },
        )
        app.processEvents()
        assert strip.isContextStacked()

        MainWindow._on_canvas_construction_command_changed(
            host,
            document.id,
            {
                "tool": "segment",
                "point_count": 0,
                "source_count": 0,
            },
        )
        app.processEvents()

        assert strip.isContextInline()
        assert not strip.isContextStacked()
    finally:
        strip.close()


def test_switching_canvas_republishes_document_local_command_state(
    app: QApplication,
) -> None:
    first_document = _document("first-command-state")
    second_document = _document("second-command-state")
    image = QImage(120, 80, QImage.Format.Format_RGB32)
    image.fill(QColor("#FFFFFF"))
    first_canvas = DocumentCanvas()
    second_canvas = DocumentCanvas()
    first_canvas.set_document(first_document, image)
    second_canvas.set_document(second_document, image)

    context = ConstructionContextWidget()
    host = SimpleNamespace(
        _construction_context_widget=context,
        _construction_tool_kind="parallel_array",
        active_document=first_document,
    )
    host.current_document = MethodType(lambda self: self.active_document, host)
    receive = MethodType(MainWindow._on_canvas_construction_command_changed, host)
    first_canvas.constructionCommandChanged.connect(receive)
    second_canvas.constructionCommandChanged.connect(receive)

    first_canvas.set_construction_parameter("distance", 12.0)
    first_canvas.set_construction_parameter("count", 3)
    first_canvas.set_construction_parameter("both_sides", False)
    first_canvas.set_construction_parameter("extend", True)
    second_canvas.set_construction_parameter("distance", 45.0)
    second_canvas.set_construction_parameter("count", 8)
    second_canvas.set_construction_parameter("both_sides", True)
    second_canvas.set_construction_parameter("extend", False)

    assert context.distanceSpin.value() == pytest.approx(12.0)
    assert context.countSpin.value() == 3
    assert not context.bothSidesCheck.isChecked()
    assert context.extendCheck.isChecked()

    host.active_document = second_document
    second_canvas.publish_construction_command_state()
    assert context.distanceSpin.value() == pytest.approx(45.0)
    assert context.countSpin.value() == 8
    assert context.bothSidesCheck.isChecked()
    assert not context.extendCheck.isChecked()

    first_canvas.close()
    second_canvas.close()
    context.close()


def test_inspector_edits_array_parameters_in_calibrated_units(
    app: QApplication,
) -> None:
    source = ConstructionEntity(
        id="array-source",
        name="基准线",
        definition=LineDefinition(Point(0.0, 20.0), Point(100.0, 20.0)),
    )
    array = ConstructionEntity(
        id="editable-array",
        name="平行阵列",
        definition=ParallelArrayDefinition(
            LiveFeatureRef("array-parameters", source.id),
            spacing=20.0,
            count=3,
            side=ArraySide.POSITIVE,
        ),
    )
    document = _document("array-parameters", (source, array))
    document.calibration = Calibration(
        mode="manual",
        pixels_per_unit=2.0,
        unit="mm",
        source_label="test",
    )
    inspector = CurrentObjectInspector()
    emitted: list[tuple[str, object]] = []
    inspector.constructionDefinitionChangeRequested.connect(
        lambda entity_id, definition: emitted.append((entity_id, definition))
    )
    inspector.set_context(
        document,
        settings=AppSettings(),
        construction_id=array.id,
    )

    assert inspector._construction_distance_spin.suffix() == " mm"
    assert inspector._construction_distance_spin.value() == pytest.approx(10.0)
    assert inspector._construction_count_spin.value() == 3
    assert inspector._construction_side_combo.currentData() == "positive"

    inspector._construction_distance_spin.setValue(12.5)
    inspector._request_construction_distance_change()
    assert emitted and emitted[-1][0] == array.id
    changed = emitted[-1][1]
    assert isinstance(changed, ParallelArrayDefinition)
    assert changed.spacing == pytest.approx(25.0)

    inspector._construction_count_spin.setValue(7)
    inspector._request_construction_count_change()
    changed = emitted[-1][1]
    assert isinstance(changed, ParallelArrayDefinition)
    assert changed.count == 7

    side_index = inspector._construction_side_combo.findData("both")
    inspector._construction_side_combo.setCurrentIndex(side_index)
    inspector._request_construction_side_change()
    changed = emitted[-1][1]
    assert isinstance(changed, ParallelArrayDefinition)
    assert changed.side is ArraySide.BOTH
    inspector.close()


def test_inspector_keeps_axis_construction_lines_infinite(
    app: QApplication,
) -> None:
    horizontal = ConstructionEntity(
        id="fixed-horizontal-line",
        name="水平辅助线",
        definition=LineDefinition(
            Point(0.0, 20.0),
            Point(1.0, 20.0),
            LineExtent.INFINITE,
            LineAxisConstraint.HORIZONTAL,
        ),
    )
    document = _document("axis-line-properties", (horizontal,))
    inspector = CurrentObjectInspector()
    emitted: list[tuple[str, object]] = []
    inspector.constructionDefinitionChangeRequested.connect(
        lambda entity_id, definition: emitted.append((entity_id, definition))
    )

    inspector.set_context(
        document,
        settings=AppSettings(),
        construction_id=horizontal.id,
    )

    assert not inspector._construction_extent_combo.isVisibleTo(inspector)
    inspector._construction_extent_combo.setCurrentIndex(
        inspector._construction_extent_combo.findData(LineExtent.SEGMENT.value)
    )
    inspector._request_construction_extent_change()
    assert emitted == []
    assert horizontal.definition.extent is LineExtent.INFINITE
    inspector.close()


def test_inspector_keeps_unit_vector_derived_lines_infinite(
    app: QApplication,
) -> None:
    source = ConstructionEntity(
        id="derived-extent-source",
        name="源线",
        definition=LineDefinition(Point(0.0, 20.0), Point(80.0, 20.0)),
    )
    source_ref = LiveFeatureRef("derived-extent", source.id)
    definitions = (
        ParallelThroughPointDefinition(source_ref, Point(0.0, 30.0)),
        OffsetParallelDefinition(source_ref, 10.0),
        ParallelArrayDefinition(source_ref, 10.0, 2),
        PerpendicularDefinition(source_ref, Point(40.0, 40.0)),
        PerpendicularBisectorDefinition(source_ref),
    )
    entities = [
        ConstructionEntity(
            id=f"derived-extent-{index}",
            name="派生构造线",
            definition=definition,
        )
        for index, definition in enumerate(definitions)
    ]
    document = _document("derived-extent", (source, *entities))
    inspector = CurrentObjectInspector()
    emitted: list[tuple[str, object]] = []
    inspector.constructionDefinitionChangeRequested.connect(
        lambda entity_id, definition: emitted.append((entity_id, definition))
    )

    for entity in entities:
        inspector.set_context(
            document,
            settings=AppSettings(),
            construction_id=entity.id,
        )
        assert not inspector._construction_extent_combo.isVisibleTo(inspector)
        inspector._construction_extent_combo.setCurrentIndex(
            inspector._construction_extent_combo.findData(LineExtent.SEGMENT.value)
        )
        inspector._request_construction_extent_change()

    assert emitted == []
    assert all(entity.definition.extent is LineExtent.INFINITE for entity in entities)
    inspector.close()


def test_locked_construction_cannot_detach_live_sources(
    app: QApplication,
) -> None:
    source = ConstructionEntity(
        id="locked-detach-source",
        name="源线",
        definition=LineDefinition(Point(0.0, 10.0), Point(40.0, 10.0)),
    )
    dependent = ConstructionEntity(
        id="locked-detach-dependent",
        name="锁定阵列",
        definition=ParallelArrayDefinition(
            LiveFeatureRef("locked-detach", source.id),
            spacing=5.0,
            count=2,
        ),
        locked=True,
    )
    document = _document("locked-detach", (source, dependent))
    inspector = CurrentObjectInspector()
    inspector.set_context(
        document,
        settings=AppSettings(),
        construction_id=dependent.id,
    )
    assert not inspector._construction_detach_button.isEnabled()

    window = MainWindow()
    window.project.documents = [document]
    window.current_document = MethodType(lambda self: document, window)
    window._refresh_object_inspector = MethodType(lambda self: None, window)
    before = dependent.definition
    window._on_construction_detach_requested(dependent.id)
    current = document.get_construction_entity(dependent.id)
    assert current is not None and current.definition == before
    assert document.history is not None and not document.history.can_undo()
    window.project.documents = []
    inspector.close()
    window.close()


def test_locked_construction_disables_manager_edits_and_dependency_deletion(
    app: QApplication,
) -> None:
    source = ConstructionEntity(
        id="locked-delete-source",
        name="源线",
        definition=LineDefinition(Point(0.0, 10.0), Point(40.0, 10.0)),
    )
    locked_dependent = ConstructionEntity(
        id="locked-delete-dependent",
        name="锁定中点",
        definition=MidpointDefinition(
            LiveFeatureRef("locked-delete", source.id)
        ),
        locked=True,
    )
    document = _document("locked-delete", (source, locked_dependent))
    panel = ConstructionManagerPanel()
    panel.setEntities(document.construction_entities, selected_id=locked_dependent.id)
    assert not panel.colorButton.isEnabled()
    assert not panel.deleteButton.isEnabled()

    window = MainWindow()
    window.project.documents = [document]
    window.current_document = MethodType(lambda self: document, window)
    window._on_construction_delete_requested((source.id,))
    assert document.get_construction_entity(source.id) is not None
    assert document.get_construction_entity(locked_dependent.id) is not None
    assert document.history is not None and not document.history.can_undo()
    assert "锁定" in window.statusBar().currentMessage()
    window.project.documents = []
    panel.close()
    window.close()


def test_locate_source_respects_measurement_kind_when_ids_overlap(
    app: QApplication,
) -> None:
    shared_id = "same-locate-source-id"
    source_entity = ConstructionEntity(
        id=shared_id,
        name="同 ID 辅助对象",
        definition=LineDefinition(Point(0.0, 10.0), Point(40.0, 10.0)),
    )
    dependent = ConstructionEntity(
        id="measurement-source-dependent",
        name="测量来源下游",
        definition=OffsetParallelDefinition(
            LiveFeatureRef(
                "locate-source-kind",
                shared_id,
                SourceObjectKind.MEASUREMENT,
            ),
            4.0,
        ),
    )
    document = _document("locate-source-kind", (source_entity, dependent))
    document.add_measurement(
        Measurement(
            id=shared_id,
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(5.0, 5.0), Point(35.0, 5.0)),
        )
    )
    located_measurements: list[str] = []
    construction_selections: list[str] = []
    canvas = SimpleNamespace(
        set_selected_construction=construction_selections.append,
        center_on_construction=construction_selections.append,
    )
    window = MainWindow()
    window.project.documents = [document]
    window.current_document = MethodType(lambda self: document, window)
    window.current_canvas = MethodType(lambda self: canvas, window)
    window._activate_measurement_id = MethodType(
        lambda self, measurement_id: located_measurements.append(measurement_id),
        window,
    )
    window._refresh_object_inspector = MethodType(lambda self: None, window)

    window._on_construction_locate_sources_requested(dependent.id)

    assert located_measurements == [shared_id]
    assert construction_selections == []
    window.project.documents = []
    window.close()


def test_batch_construction_color_edit_is_one_undoable_command(
    app: QApplication,
) -> None:
    first = ConstructionEntity(
        id="batch-color-first",
        name="点一",
        definition=FreePointDefinition(Point(10.0, 10.0)),
    )
    second = ConstructionEntity(
        id="batch-color-second",
        name="点二",
        definition=FreePointDefinition(Point(20.0, 20.0)),
    )
    document = _document("batch-color", (first, second))
    window = MainWindow()
    window.project.documents = [document]
    window.current_document = MethodType(lambda self: document, window)
    window._update_ui_for_current_document = MethodType(lambda self: None, window)
    window._refresh_document_analysis_validity = MethodType(
        lambda self, _document: None,
        window,
    )
    window._discard_detached_area_geometry = MethodType(
        lambda self, _before, _document: None,
        window,
    )

    window._on_construction_batch_color_change_requested(
        (first.id, second.id),
        "#123456",
    )
    assert all(
        document.get_construction_entity(entity.id).style.stroke_color == "#123456"
        for entity in (first, second)
    )
    assert document.history is not None and document.history.can_undo()

    assert document.history.undo(document)
    assert all(
        document.get_construction_entity(entity.id).style.stroke_color
        == entity.style.stroke_color
        for entity in (first, second)
    )
    assert not document.history.can_undo()
    window.project.documents = []
    window.close()


@pytest.mark.parametrize("dependency_action", ["cascade", "freeze"])
def test_measurement_delete_keeps_same_id_construction_dependency_separate(
    dependency_action: str,
) -> None:
    shared_id = "same-kind-collision"
    base = ConstructionEntity(
        id=shared_id,
        name="同 ID 辅助线",
        definition=LineDefinition(Point(0.0, 10.0), Point(40.0, 10.0)),
    )
    construction_child = ConstructionEntity(
        id=f"construction-child-{dependency_action}",
        name="依赖辅助线",
        definition=OffsetParallelDefinition(
            LiveFeatureRef(
                f"same-id-{dependency_action}",
                shared_id,
                SourceObjectKind.CONSTRUCTION,
            ),
            4.0,
        ),
    )
    measurement_child = ConstructionEntity(
        id=f"measurement-child-{dependency_action}",
        name="依赖测量线",
        definition=OffsetParallelDefinition(
            LiveFeatureRef(
                f"same-id-{dependency_action}",
                shared_id,
                SourceObjectKind.MEASUREMENT,
            ),
            6.0,
        ),
    )
    document = _document(
        f"same-id-{dependency_action}",
        (base, construction_child, measurement_child),
    )
    document.add_measurement(
        Measurement(
            id=shared_id,
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(5.0, 5.0), Point(35.0, 5.0)),
        )
    )

    removed = MainWindow._remove_measurements_with_dependencies(
        document,
        (shared_id,),
        dependency_action,
    )

    assert removed == 1
    assert document.get_construction_entity(base.id) is base
    assert document.get_construction_entity(construction_child.id) is construction_child
    remaining_measurement_child = document.get_construction_entity(
        measurement_child.id
    )
    if dependency_action == "cascade":
        assert remaining_measurement_child is None
    else:
        assert remaining_measurement_child is not None
        assert isinstance(
            remaining_measurement_child.definition,
            OffsetParallelDefinition,
        )
        assert isinstance(
            remaining_measurement_child.definition.source,
            FrozenFeatureSnapshot,
        )
        assert isinstance(construction_child.definition.source, LiveFeatureRef)
        assert (
            construction_child.definition.source.object_kind
            is SourceObjectKind.CONSTRUCTION
        )


@pytest.mark.parametrize("dependency_action", ["cascade", "freeze"])
def test_locked_transitive_construction_blocks_measurement_source_deletion(
    dependency_action: str,
) -> None:
    document_id = f"locked-measurement-delete-{dependency_action}"
    measurement_id = "locked-source-measurement"
    direct = ConstructionEntity(
        id=f"unlocked-direct-{dependency_action}",
        name="直接测量下游",
        definition=OffsetParallelDefinition(
            LiveFeatureRef(
                document_id,
                measurement_id,
                SourceObjectKind.MEASUREMENT,
            ),
            4.0,
        ),
    )
    locked_transitive = ConstructionEntity(
        id=f"locked-transitive-{dependency_action}",
        name="锁定的间接下游",
        definition=PerpendicularDefinition(
            LiveFeatureRef(
                document_id,
                direct.id,
                SourceObjectKind.CONSTRUCTION,
            ),
            Point(8.0, 8.0),
        ),
        locked=True,
    )
    document = _document(document_id, (direct, locked_transitive))
    document.add_measurement(
        Measurement(
            id=measurement_id,
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(5.0, 5.0), Point(35.0, 5.0)),
        )
    )
    before = document.to_dict()
    before_measurement_revision = document.measurement_geometry_revision
    before_construction_revision = document.construction_geometry_revision

    with pytest.raises(ConstructionValidationError) as error:
        MainWindow._remove_measurements_with_dependencies(
            document,
            (measurement_id,),
            dependency_action,
        )

    assert error.value.code == "locked_dependent_objects"
    assert error.value.entity_ids == (locked_transitive.id,)
    assert document.to_dict() == before
    assert document.measurement_geometry_revision == before_measurement_revision
    assert document.construction_geometry_revision == before_construction_revision


def test_measurement_dependency_prompt_summarizes_locked_downstream_and_stops(
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = "locked-measurement-prompt"
    measurement_id = "prompt-source-measurement"
    locked = ConstructionEntity(
        id="prompt-locked-dependent",
        name="锁定辅助线 A",
        definition=OffsetParallelDefinition(
            LiveFeatureRef(
                document_id,
                measurement_id,
                SourceObjectKind.MEASUREMENT,
            ),
            3.0,
        ),
        locked=True,
    )
    document = _document(document_id, (locked,))
    document.add_measurement(
        Measurement(
            id=measurement_id,
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(5.0, 5.0), Point(35.0, 5.0)),
        )
    )
    window = _workspace_history_window([document], document)
    warnings: list[tuple[str, str]] = []

    class WarningOnlyMessageBox:
        @staticmethod
        def warning(_parent: object, title: str, message: str) -> None:
            warnings.append((title, message))

        def __init__(self, *_args: object) -> None:
            raise AssertionError("锁定下游存在时不应再打开级联/冻结选择对话框")

    monkeypatch.setattr(
        main_window_module,
        "QMessageBox",
        WarningOnlyMessageBox,
    )
    action = window._prompt_measurement_dependency_action(
        {document.id: (document, (measurement_id,))}
    )

    assert action is None
    assert len(warnings) == 1
    assert warnings[0][0] == "测量来源存在锁定下游"
    assert "锁定辅助线 A" in warnings[0][1]
    assert document.get_measurement(measurement_id) is not None
    assert document.get_construction_entity(locked.id) is locked
    window.project.documents = []
    window.close()


def test_construction_freeze_keeps_same_id_measurement_dependency_live(
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = "construction-delete-same-id"
    shared_id = "same-kind-collision"
    base = ConstructionEntity(
        id=shared_id,
        name="待删除辅助线",
        definition=LineDefinition(Point(0.0, 10.0), Point(40.0, 10.0)),
    )
    construction_child = ConstructionEntity(
        id="construction-freeze-child",
        name="辅助线下游",
        definition=OffsetParallelDefinition(
            LiveFeatureRef(
                document_id,
                shared_id,
                SourceObjectKind.CONSTRUCTION,
            ),
            4.0,
        ),
    )
    measurement_child = ConstructionEntity(
        id="measurement-stays-live",
        name="测量线下游",
        definition=OffsetParallelDefinition(
            LiveFeatureRef(
                document_id,
                shared_id,
                SourceObjectKind.MEASUREMENT,
            ),
            6.0,
        ),
    )
    document = _document(
        document_id,
        (base, construction_child, measurement_child),
    )
    document.add_measurement(
        Measurement(
            id=shared_id,
            image_id=document.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(5.0, 5.0), Point(35.0, 5.0)),
        )
    )
    window = _workspace_history_window([document], document)

    class FreezeChoiceBox:
        Icon = QMessageBox.Icon
        ButtonRole = QMessageBox.ButtonRole

        def __init__(self, *_args: object) -> None:
            self._freeze_button: object | None = None

        def setIcon(self, _icon: object) -> None:
            pass

        def setWindowTitle(self, _title: str) -> None:
            pass

        def setText(self, _text: str) -> None:
            pass

        def setInformativeText(self, _text: str) -> None:
            pass

        def addButton(self, text: str, _role: object) -> object:
            button = object()
            if text == "解除关联并保留下游":
                self._freeze_button = button
            return button

        def setDefaultButton(self, _button: object) -> None:
            pass

        def exec(self) -> int:
            return 0

        def clickedButton(self) -> object | None:
            return self._freeze_button

    monkeypatch.setattr(main_window_module, "QMessageBox", FreezeChoiceBox)
    window._on_construction_delete_requested((base.id,))

    assert document.get_construction_entity(base.id) is None
    frozen_child = document.get_construction_entity(construction_child.id)
    assert frozen_child is not None
    assert isinstance(frozen_child.definition, OffsetParallelDefinition)
    assert isinstance(frozen_child.definition.source, FrozenFeatureSnapshot)
    live_measurement_child = document.get_construction_entity(measurement_child.id)
    assert live_measurement_child is measurement_child
    assert isinstance(live_measurement_child.definition.source, LiveFeatureRef)
    assert (
        live_measurement_child.definition.source.object_kind
        is SourceObjectKind.MEASUREMENT
    )
    assert document.get_measurement(shared_id) is not None
    window.project.documents = []
    window.close()


def test_project_wide_change_undoes_and_redoes_all_documents_atomically(
    app: QApplication,
) -> None:
    first = _document("workspace-undo-first")
    second = _document("workspace-undo-second")
    for index, document in enumerate((first, second)):
        document.add_measurement(
            Measurement(
                id=f"workspace-measurement-{index}",
                image_id=document.id,
                fiber_group_id=None,
                mode="manual",
                measurement_kind="line",
                line_px=Line(Point(10.0, 10.0), Point(30.0, 10.0)),
            )
        )

    window = MainWindow()
    window.project.documents = [first, second]
    window.current_document = MethodType(lambda self: first, window)
    window._update_ui_for_current_document = MethodType(lambda self: None, window)
    window._refresh_document_analysis_validity = MethodType(
        lambda self, _document: None,
        window,
    )
    window._discard_detached_area_geometry = MethodType(
        lambda self, _before, _document: None,
        window,
    )

    removed = window._apply_documents_change(
        [first, second],
        "删除整个项目测量",
        lambda document: (
            len(document.measurements),
            document.measurements.clear(),
        )[0],
        workspace_scope=True,
    )
    assert removed == 2
    assert not first.measurements and not second.measurements
    assert len(window._workspace_composite_undo) == 1

    local_point = ConstructionEntity(
        id="post-workspace-local-point",
        name="后续局部修改",
        definition=FreePointDefinition(Point(5.0, 5.0)),
    )
    window._apply_document_change(
        second,
        "第二张图片后续修改",
        lambda: second.add_construction_entity(local_point, mark_dirty=False),
        impact=DocumentChangeImpact.CONSTRUCTION,
    )
    # The first document must not peel off only its half of the composite while
    # another participant still has a newer local command.
    window.undo_current_document()
    assert not first.measurements and not second.measurements

    window.current_document = MethodType(lambda self: second, window)
    window.undo_current_document()
    assert second.get_construction_entity(local_point.id) is None
    assert not first.measurements and not second.measurements

    window.current_document = MethodType(lambda self: first, window)
    window.undo_current_document()
    assert len(first.measurements) == 1
    assert len(second.measurements) == 1
    assert len(window._workspace_composite_redo) == 1

    window.redo_current_document()
    assert not first.measurements and not second.measurements
    assert len(window._workspace_composite_undo) == 1
    window.project.documents = []
    window.close()


def test_project_wide_single_noncurrent_change_is_undoable_from_current_page(
    app: QApplication,
) -> None:
    current = _document("workspace-single-current")
    changed = _document("workspace-single-changed")
    changed.add_measurement(
        Measurement(
            id="workspace-single-measurement",
            image_id=changed.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(10.0, 10.0), Point(30.0, 10.0)),
        )
    )
    window = _workspace_history_window([current, changed], current)

    removed = window._apply_documents_change(
        [current, changed],
        "删除整个项目测量",
        lambda document: (
            len(document.measurements),
            document.measurements.clear(),
        )[0],
        workspace_scope=True,
    )

    assert removed == 1
    assert not changed.measurements
    assert current.history is not None and not current.history.can_undo()
    assert len(window._workspace_composite_undo) == 1
    assert window._workspace_composite_undo[-1].document_sequences[0][0] == changed.id
    window._update_action_states()
    assert window.undo_action.isEnabled()

    window.undo_current_document()
    assert len(changed.measurements) == 1
    window._update_action_states()
    assert window.redo_action.isEnabled()

    window.redo_current_document()
    assert not changed.measurements
    window.project.documents = []
    window.close()


def test_current_scope_change_never_becomes_cross_document_workspace_history(
    app: QApplication,
) -> None:
    first = _document("current-scope-first")
    second = _document("current-scope-second")
    first.add_measurement(
        Measurement(
            id="current-scope-measurement",
            image_id=first.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(10.0, 10.0), Point(30.0, 10.0)),
        )
    )
    window = _workspace_history_window([first, second], first)
    window._apply_documents_change(
        [first],
        "删除当前图片测量",
        lambda document: (
            len(document.measurements),
            document.measurements.clear(),
        )[0],
        workspace_scope=False,
    )

    assert not first.measurements
    assert not window._workspace_composite_undo
    window.current_document = MethodType(lambda self: second, window)
    window.undo_current_document()
    assert not first.measurements

    window.current_document = MethodType(lambda self: first, window)
    window.undo_current_document()
    assert len(first.measurements) == 1
    window.project.documents = []
    window.close()


def test_nonparticipant_newer_local_command_preserves_global_undo_redo_order(
    app: QApplication,
) -> None:
    current = _document("workspace-order-current")
    changed = _document("workspace-order-changed")
    changed.add_measurement(
        Measurement(
            id="workspace-order-measurement",
            image_id=changed.id,
            fiber_group_id=None,
            mode="manual",
            measurement_kind="line",
            line_px=Line(Point(10.0, 10.0), Point(30.0, 10.0)),
        )
    )
    window = _workspace_history_window([current, changed], current)
    window._apply_documents_change(
        [current, changed],
        "删除整个项目测量",
        lambda document: (
            len(document.measurements),
            document.measurements.clear(),
        )[0],
        workspace_scope=True,
    )
    local_point = ConstructionEntity(
        id="workspace-order-local-point",
        name="项目命令后的本地修改",
        definition=FreePointDefinition(Point(6.0, 6.0)),
    )
    window._apply_document_change(
        current,
        "较新的本地修改",
        lambda: current.add_construction_entity(local_point, mark_dirty=False),
        impact=DocumentChangeImpact.CONSTRUCTION,
    )

    window.undo_current_document()
    assert current.get_construction_entity(local_point.id) is None
    assert not changed.measurements
    window.undo_current_document()
    assert len(changed.measurements) == 1

    window.redo_current_document()
    assert not changed.measurements
    assert current.get_construction_entity(local_point.id) is None
    window.redo_current_document()
    assert current.get_construction_entity(local_point.id) is not None
    window.project.documents = []
    window.close()


def test_local_branch_discards_every_participant_composite_redo(
    app: QApplication,
) -> None:
    first = _document("workspace-branch-first")
    second = _document("workspace-branch-second")
    for index, document in enumerate((first, second)):
        document.add_measurement(
            Measurement(
                id=f"workspace-branch-measurement-{index}",
                image_id=document.id,
                fiber_group_id=None,
                mode="manual",
                measurement_kind="line",
                line_px=Line(Point(10.0, 10.0), Point(30.0, 10.0)),
            )
        )
    window = _workspace_history_window([first, second], first)
    window._apply_documents_change(
        [first, second],
        "删除整个项目测量",
        lambda document: (
            len(document.measurements),
            document.measurements.clear(),
        )[0],
        workspace_scope=True,
    )
    composite_sequences = dict(
        window._workspace_composite_undo[-1].document_sequences
    )
    window.undo_current_document()
    assert len(first.measurements) == len(second.measurements) == 1
    assert len(window._workspace_composite_redo) == 1

    local_point = ConstructionEntity(
        id="workspace-branch-local-point",
        name="新分支",
        definition=FreePointDefinition(Point(5.0, 5.0)),
    )
    window._apply_document_change(
        first,
        "本地新分支",
        lambda: first.add_construction_entity(local_point, mark_dirty=False),
        impact=DocumentChangeImpact.CONSTRUCTION,
    )

    assert not window._workspace_composite_redo
    for document in (first, second):
        assert document.history is not None
        assert not document.history.contains_redo_sequence(
            composite_sequences[document.id]
        )
    window.current_document = MethodType(lambda self: second, window)
    window.redo_current_document()
    assert len(first.measurements) == len(second.measurements) == 1
    window.project.documents = []
    window.close()


@pytest.mark.parametrize("limit_kind", ["max_commands", "max_bytes"])
def test_composite_history_eviction_prunes_other_participant_command(
    app: QApplication,
    limit_kind: str,
) -> None:
    first = _document(f"workspace-eviction-first-{limit_kind}")
    second = _document(f"workspace-eviction-second-{limit_kind}")
    for index, document in enumerate((first, second)):
        document.add_measurement(
            Measurement(
                id=f"workspace-eviction-measurement-{limit_kind}-{index}",
                image_id=document.id,
                fiber_group_id=None,
                mode="manual",
                measurement_kind="line",
                line_px=Line(Point(10.0, 10.0), Point(30.0, 10.0)),
            )
        )
    window = _workspace_history_window([first, second], first)
    window._apply_documents_change(
        [first, second],
        "删除整个项目测量",
        lambda document: (
            len(document.measurements),
            document.measurements.clear(),
        )[0],
        workspace_scope=True,
    )
    sequences = dict(window._workspace_composite_undo[-1].document_sequences)
    assert first.history is not None and second.history is not None
    if limit_kind == "max_commands":
        first.history._max_commands = 1
    else:
        first.history._max_bytes = 1

    local_point = ConstructionEntity(
        id=f"workspace-eviction-local-{limit_kind}",
        name="预算淘汰后的局部命令",
        definition=FreePointDefinition(Point(8.0, 8.0)),
    )
    window._apply_document_change(
        first,
        "触发历史预算淘汰",
        lambda: first.add_construction_entity(local_point, mark_dirty=False),
        impact=DocumentChangeImpact.CONSTRUCTION,
    )
    assert not first.history.contains_undo_sequence(sequences[first.id])
    assert second.history.contains_undo_sequence(sequences[second.id])

    # Action-state refresh is one of the coordinator entry points.  It must
    # notice the missing participant and remove the other half immediately.
    window._update_action_states()
    assert not window._workspace_composite_undo
    assert not second.history.contains_undo_sequence(sequences[second.id])
    assert first.get_construction_entity(local_point.id) is not None

    window.undo_current_document()
    assert first.get_construction_entity(local_point.id) is None
    assert not first.measurements and not second.measurements
    window.project.documents = []
    window.close()


def test_main_window_parameter_edit_is_one_undoable_construction_command(
    app: QApplication,
) -> None:
    source = ConstructionEntity(
        id="history-array-source",
        name="基准线",
        definition=LineDefinition(Point(0.0, 20.0), Point(100.0, 20.0)),
    )
    array = ConstructionEntity(
        id="history-array",
        name="阵列",
        definition=ParallelArrayDefinition(
            LiveFeatureRef("parameter-history", source.id),
            spacing=12.0,
            count=2,
        ),
    )
    document = _document("parameter-history", (source, array))
    window = MainWindow()
    window.project.documents = [document]
    window.current_document = MethodType(lambda self: document, window)
    window._update_ui_for_current_document = MethodType(lambda self: None, window)
    window._refresh_document_analysis_validity = MethodType(
        lambda self, _document: None,
        window,
    )
    window._discard_detached_area_geometry = MethodType(
        lambda self, _before, _document: None,
        window,
    )
    window._focus_current_canvas = MethodType(lambda self: None, window)

    window._on_construction_definition_change_requested(
        array.id,
        ParallelArrayDefinition(
            array.definition.source,
            spacing=18.0,
            count=6,
            side=ArraySide.BOTH,
        ),
    )
    changed = document.get_construction_entity(array.id)
    assert changed is not None
    assert isinstance(changed.definition, ParallelArrayDefinition)
    assert changed.definition.spacing == pytest.approx(18.0)
    assert changed.definition.count == 6
    assert document.history is not None and document.history.can_undo()

    assert document.history.undo(document)
    restored = document.get_construction_entity(array.id)
    assert restored is not None
    assert isinstance(restored.definition, ParallelArrayDefinition)
    assert restored.definition.spacing == pytest.approx(12.0)
    assert restored.definition.count == 2
    window.project.documents = []
    window.close()
