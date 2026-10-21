"""Category-first layout acceptance through the real acquisition navigation."""
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from fdm.live_experiment import built_in_materials
from fdm.live_experiment.feedback import feedback_for_reason
from test_acquisition_second_round import workflow


def prepare_nine_categories(harness):
    harness.enter()
    page = harness.select(1)
    page.session.set_materials(built_in_materials("normal")[:9])
    page.session.rename_operator("测试员")
    page._refresh_operators()
    page.ruler_choice.setCurrentIndex(1)
    page.ruler_choice.activated.emit(1)
    page._refresh_summary()
    workflow.settle()
    return page


def visible_categories(page):
    viewport = page.category_scroll.viewport()
    visible = []
    for row in page.material_rows.values():
        selector = row["select"]
        point = selector.mapTo(viewport, QPoint())
        if viewport.rect().contains(point) and viewport.rect().contains(point + selector.rect().bottomRight()):
            visible.append(selector)
    return visible


def geometry(widget, window):
    point = widget.mapTo(window, QPoint())
    return point.x(), point.y(), widget.width(), widget.height()


@pytest.mark.parametrize("theme", ["light", "dark", "navy", "warm"])
@pytest.mark.parametrize("size,minimum", [((1600, 900), 6), ((1280, 720), 4)])
def test_categories_have_usable_space_and_export_is_always_reachable(tmp_path, theme, size, minimum):
    # 1280x720 is the logical client budget of a 1600x900 display at 125%.
    # Actual device-pixel scaling is additionally checked by the visual harness.
    with workflow.WorkflowHarness(tmp_path / "profile", theme=theme, size=size) as harness:
        page = prepare_nine_categories(harness)
        assert len(visible_categories(page)) >= minimum
        assert page.workflow_tabs.widget(0).verticalScrollBar().maximum() == 0
        for control in (page.export_button, page.records_toggle, page.save_button):
            assert control.isVisible()
            assert page.trial_sidebar.visibleRegion().contains(control.mapTo(page.trial_sidebar, control.rect().center()))
        assert not harness.slot_errors


def test_scrolling_categories_and_receiving_feedback_keeps_tools_in_place(tmp_path):
    with workflow.WorkflowHarness(tmp_path / "profile", size=(1280, 720)) as harness:
        page = prepare_nine_categories(harness)
        controls = (page.latest_value, page.measure_mode, page.weights_button,
                    page.operator, page.ruler_choice, page.export_button, page.records_toggle)
        before = [geometry(control, harness.window) for control in controls]
        scroll = page.category_scroll.verticalScrollBar()
        assert scroll.maximum() > 0
        scroll.setFocus()
        QTest.keyClick(scroll, Qt.Key.Key_End)
        workflow.settle()
        assert scroll.value() == scroll.maximum()
        assert list(page.material_rows.values())[-1]["select"] in visible_categories(page)
        page.controller.measurementFeedback.emit(feedback_for_reason(
            "no_complete_crossing", **page.controller.feedback_context()))
        workflow.settle()
        assert "本次未通过" in page.measurement_outcome.text()
        assert before == [geometry(control, harness.window) for control in controls]
        assert "未识别到完整的纤维两侧边缘" in page.measurement_outcome.toolTip()


def test_quantity_summary_does_not_repeat_category_rows(tmp_path):
    with workflow.WorkflowHarness(tmp_path / "profile", size=(1600, 900)) as harness:
        page = prepare_nine_categories(harness)
        assert "\n" not in page.quantity_targets.progress.text()
        assert "1500" in page.quantity_targets.progress.text()
        assert "绵羊毛" in page.quantity_targets.progress.toolTip()
        assert page.quantity_targets.height() <= 32
        assert all("/300" in row["measured_count"].text() for row in page.material_rows.values())


def test_metadata_expansion_with_bottom_records_does_not_enlarge_short_window(tmp_path):
    with workflow.WorkflowHarness(tmp_path / "profile", size=(1067, 600)) as harness:
        page = prepare_nine_categories(harness)
        QTest.mouseClick(page.records_toggle, Qt.MouseButton.LeftButton)
        workflow.settle()
        # Opening the default left dock is the tightest width, even when the
        # user intends to move it to the bottom immediately afterwards.
        harness.assert_size()
        controls = (page.analyze_button, page.auto_analysis,
                    page.candidate_display_button, page.display_choice)
        for control in controls:
            assert control.isVisible()
            assert control.geometry().left() >= 0
            assert control.geometry().right() < page.view_controls.width()
        assert all(left.geometry().right() < right.geometry().left()
                   for left, right in zip(controls, controls[1:]))
        assert page.display_choice.width() >= page.display_choice.minimumSizeHint().width()
        records = harness.hub.records
        next(action for action in records.panel.position_button.menu().actions()
             if action.text() == "停靠底部").trigger()
        QTest.qWait(250)
        navigation = geometry(harness.hub.navigation, harness.window)
        QTest.mouseClick(page.sample_details_toggle, Qt.MouseButton.LeftButton)
        QTest.qWait(250)
        harness.assert_size()
        assert page.sample_details.isVisible()
        assert page.metadata_scroll.verticalScrollBar().maximum() > 0
        assert navigation == geometry(harness.hub.navigation, harness.window)
        for control in (page.save_button, page.export_button):
            assert control.isVisible()
            assert page.trial_sidebar.rect().contains(control.mapTo(page.trial_sidebar, control.rect().topLeft()))
            assert page.trial_sidebar.rect().contains(control.mapTo(page.trial_sidebar, control.rect().bottomRight()))
