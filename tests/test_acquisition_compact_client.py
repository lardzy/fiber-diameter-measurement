"""Trial chrome must fit the actual client area after every navigation step."""
import json

import pytest
from PySide6.QtCore import QPoint, QRect

from test_acquisition_workflow_redesign import workflow


@pytest.mark.parametrize("size", [(1280, 670), (1366, 690), (1366, 768)])
@pytest.mark.parametrize("restored_trial", [1, 2])
def test_compact_client_keeps_metadata_save_and_scroll_controls_reachable(tmp_path, size, restored_trial):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "acquisition-workbench.json").write_text(
        json.dumps({"version": 1, "page": restored_trial}), encoding="utf-8")
    with workflow.WorkflowHarness(profile, size=size) as h:
        h.enter()
        assert h.hub.navigation.currentIndex() == restored_trial
        # Exercise restored construction, direct navigation and cached pages.
        for route in (0, 1, 2, 0, restored_trial):
            h.hub.select_page(route)
            workflow.settle()
            h.backend.tick()
            h.assert_size()
            h.assert_navigation()
            if route == 0:
                continue
            page = h.hub.pages.currentWidget()
            image_rectangle = page.image_scroll.geometry()
            for tab in (0, 1, 2, 0):
                page.workflow_tabs.setCurrentIndex(tab)
                workflow.settle()
                h.assert_size()
                assert page.image_scroll.geometry() == image_rectangle
                assert page.image_scroll.viewport().height() >= 200
                for control in (page.sample_number, page.sample, page.operator,
                                page.ruler_choice, page.save_button):
                    rectangle = QRect(control.mapTo(h.window, QPoint()), control.size())
                    assert control.isVisible()
                    assert h.window.rect().contains(rectangle)
                    assert control.visibleRegion().contains(control.rect())
            operations = page.workflow_tabs.widget(0)
            if route == 1:
                assert operations.verticalScrollBar().maximum() > 0
            operations.ensureWidgetVisible(page.operation_help)
            workflow.settle()
            assert page.operation_help.visibleRegion().contains(page.operation_help.rect())
            h.assert_size()
        h.leave()
        h.enter()
        h.assert_clean()
