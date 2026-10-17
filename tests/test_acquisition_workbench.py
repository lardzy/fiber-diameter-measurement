from PySide6.QtWidgets import QApplication
from fdm.ui.main_window import MainWindow
from fdm.live_experiment.models import ExperimentSession, MaterialSnapshot
import pytest


@pytest.fixture(autouse=True)
def no_physical_camera(monkeypatch):
    monkeypatch.setattr("fdm.services.capture.available_capture_backends", lambda: [])


def test_real_workspace_action_reentry_and_trial_switch_preserve_data_and_ruler(monkeypatch):
    import numpy as np
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from fdm.live_experiment.frames import FrameHandle
    from fdm.live_experiment.controller import CalibrationSnapshot
    from fdm.models import CalibrationPreset
    from fdm.ui.live_experiment_workspace import LiveExperimentWorkspace
    from fdm.ui.acquisition_workbench import AcquisitionWorkbench
    window = MainWindow()
    try:
        window._app_settings.calibration_presets = [CalibrationPreset("CU 40x", 2, "um")]
        window.show()
        window.resize(1280, 720)
        window.acquisition_workspace_action.trigger()
        hub = window._acquisition_workbench
        native_handle = window._microview_preview_host.ensure_native_handle()
        assert native_handle
        assert not window._center_stack.testAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        assert not hub.testAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        QTest.mouseClick(hub.navigation, Qt.MouseButton.LeftButton, pos=hub.navigation.tabRect(1).center())
        QApplication.processEvents()
        page = window._live_experiment_workspace
        frame = FrameHandle.snapshot(np.zeros((576, 768, 3), np.uint8),
                                     source_id="camera:microview:1", generation=1, live=True)
        page._receive_frame(frame)
        page.ruler_choice.setCurrentIndex(1)
        page.ruler_choice.activated.emit(1)
        value = page.controller.calibration
        assert value.x_um == .5 and value.source == "CU 40x"
        record = page.session.add_measurement(20, "wool", field_id=page.controller.field_id)
        page.controller.current_records.append(record.id)
        session = page.session
        QApplication.processEvents()
        for _ in range(20):
            window.acquisition_workspace_action.trigger()
            QApplication.processEvents()
            assert hub.pages.currentWidget() is page
            assert page.controller.current_records == [record.id]
            assert page.controller.calibration == value
            assert not window.statusBar().isVisible()
            assert not window._microview_preview_host.isVisible()
            assert window._microview_preview_host.native_preview_handle() == native_handle
        assert len(window.findChildren(AcquisitionWorkbench)) == 1
        for index in (2, 1) * 20:
            QTest.mouseClick(hub.navigation, Qt.MouseButton.LeftButton, pos=hub.navigation.tabRect(index).center())
            QApplication.processEvents()
            current = window._live_experiment_workspace
            current._receive_frame(frame)
            assert current.controller.calibration == value
            assert hub.pages.count() == 3
            assert not window.statusBar().isVisible()
            from PySide6.QtCore import QCoreApplication, QEvent
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert current.session is session and record.id in current.session.measurements
        assert len(window.findChildren(LiveExperimentWorkspace)) == 2
        assert current.controller.current_records == []
        QTest.mouseClick(hub.navigation, Qt.MouseButton.LeftButton, pos=hub.navigation.tabRect(0).center())
        assert not window.statusBar().isVisible()
        assert window._preview_inline_status.isVisible()
    finally:
        monkeypatch.setattr(window, "_confirm_close_documents_for_transition", lambda *_: True)
        window.close()


def test_workspace_entry_and_trial_pages_keep_one_session(monkeypatch):
    window=MainWindow()
    try:
        trial=ExperimentSession(); trial.add_operator('甲')
        trial.set_materials([MaterialSnapshot(str(i),f'类别{i+1}',1.3) for i in range(9)])
        window.project.live_experiments.append(trial)
        window.show(); window.resize(1280,720)
        hub=window._open_acquisition_workbench(trial_index=1)
        QApplication.processEvents()
        page=window._live_experiment_workspace
        assert page.embedded and hub.pages.currentWidget() is page
        assert len(page.material_rows)==9 and not page.keep_images.isChecked()
        assert not window._inspector_dock.isVisible()
        manager=window._capture_manager
        hub.select_page(0); hub.select_page(2)
        assert window._capture_manager is manager
        assert window._live_experiment_workspace.session.workspace_options['mode']=='diameter'
        hub.select_page(1)
        assert window._live_experiment_workspace.session is trial
        window.resize(1280,720)
        QApplication.processEvents()
        from PySide6.QtTest import QTest
        QTest.qWait(50)
        window.resize(1280,720)
        QApplication.processEvents()
        import os
        if os.environ.get('FDM_LAYOUT_OUTPUT'):
            window.grab().save(os.environ['FDM_LAYOUT_OUTPUT'])
        assert window.width()<=1280 and window.height()<=720
        page=window._live_experiment_workspace
        page.pending_label.setText('直径超出类别提示范围，请保留、改类或放弃本次结果。')
        page.pending_panel.show()
        QApplication.processEvents()
        assert window.width()<=1280 and window.height()<=720
        assert page.material_panel.isVisible()
        # Nine categories use the operation console's own scroll viewport,
        # while the image retains its full rectangle.
        operation_scroll = page.workflow_tabs.widget(0)
        last_category = list(page.material_rows.values())[-1]["select"]
        before_image = page.image_scroll.geometry()
        operation_scroll.ensureWidgetVisible(last_category)
        QApplication.processEvents()
        assert last_category.mapTo(window, last_category.rect().bottomRight()).y() < window.height()
        assert page.image_scroll.geometry() == before_image
        window._activate_measure_workspace()
        assert not window._acquisition_workspace_selected
    finally:
        monkeypatch.setattr(window,"_confirm_close_documents_for_transition",lambda *_:True)
        window.close()
