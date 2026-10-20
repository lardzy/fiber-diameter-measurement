"""Real MainWindow navigation: fourth-round controls fit without losing classes.

Domain category fixtures isolate layout; the standalone validation script also
exercises the real modal category editor and automatic worker end to end.
"""
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'validation'))
from validate_workflow_redesign import WorkflowHarness, settle
from fdm.live_experiment import built_in_materials


@pytest.mark.parametrize('theme', ['light', 'dark', 'navy', 'warm'])
@pytest.mark.parametrize('size,minimum', [((1600, 900), 6), ((1280, 720), 4), ((1067, 600), 1), ((1366, 768), 4)])
def test_fourth_round_toolbar_and_categories_fit_through_real_navigation(tmp_path, theme, size, minimum):
    with WorkflowHarness(tmp_path, theme=theme, size=size) as harness:
        harness.enter()
        page = harness.select(1)
        page.session.rename_operator('Operator A')
        page.session.set_materials(built_in_materials('normal')[:9])
        page._refresh_operators()
        page._refresh_materials()
        page._changed()
        page.ruler_choice.setCurrentIndex(1)
        page.ruler_choice.activated.emit(1)
        settle()
        navigation = harness.hub.navigation.geometry()
        vp = page.category_scroll.viewport()
        visible = [row for row in page.material_rows.values()
                   if vp.rect().contains(row['select'].mapTo(vp, QPoint()))
                   and vp.rect().contains(row['select'].mapTo(vp, row['select'].rect().bottomRight()))]
        assert len(visible) >= minimum
        controls = [page.analyze_button, page.auto_analysis, page.candidate_display_button, page.display_choice]
        for index, control in enumerate(controls):
            assert control.isVisible()
            assert page.view_controls.rect().contains(control.mapTo(page.view_controls, QPoint()))
            assert page.view_controls.rect().contains(control.mapTo(page.view_controls, control.rect().bottomRight()))
            assert all(not control.geometry().intersects(other.geometry()) for other in controls[index + 1:])
        for index in (0, 2, 1, 0, 1):
            harness.select(index)
            assert harness.hub.navigation.geometry() == navigation
        assert page.export_button.isVisible() and page.evidence_button.isVisible()
        assert not harness.slot_errors and not harness.modal_messages


def test_candidate_preferences_are_application_only_and_survive_trial_navigation(tmp_path):
    with WorkflowHarness(tmp_path, size=(1600, 900)) as harness:
        harness.enter()
        content = harness.select(1)
        original = content.session.revision
        # Missing calibration prevents work; changes here are UI preferences.
        QTest.mouseClick(content.auto_analysis, Qt.MouseButton.LeftButton, pos=QPoint(8, content.auto_analysis.height() // 2))
        assert content.auto_analysis.isChecked()
        content.measure_mode.setFocus()
        QTest.keyClick(content.measure_mode, Qt.Key.Key_End)
        content.show_candidates_action.trigger()
        content.candidate_opacity.setValue(35)
        settle()
        assert content.canvas.candidate_input_enabled
        assert not content.canvas.candidate_visible
        assert content.session.revision == original
        diameter = harness.select(2)
        assert diameter.auto_analysis.isChecked()
        assert diameter.measure_mode.currentData() == 'free'
        assert diameter.input_mode.currentData() == 'mixed'
        assert not diameter.canvas.candidate_visible
        assert diameter.canvas.candidate_opacity == .35
        restored = harness.select(1)
        assert restored is content and restored.canvas.candidate_input_enabled
        assert not harness.slot_errors and not harness.modal_messages
