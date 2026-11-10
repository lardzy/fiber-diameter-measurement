from pathlib import Path

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QDialog

from fdm.geometry import Line, Point
from fdm.image_processing_models import RasterSemantic
from fdm.models import CalibrationPreset, ProjectCompatibilityState
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.dialogs import CalibrationInputDialog, CalibrationScopeSelector
from fdm.ui.image_loader import ImageLoadRequest
from fdm.ui.main_window import MainWindow


@pytest.fixture
def window():
    widget = MainWindow()
    yield widget
    widget._reset_workspace()
    widget.close()


def open_image(window, path: Path):
    image = QImage(80, 60, QImage.Format.Format_RGB32)
    image.fill(0xffdddddd)
    assert image.save(str(path))
    window._add_loaded_document(ImageLoadRequest(path=str(path), document=None), image)
    return window.current_document()


def test_empty_presets_menu_exposes_create_and_import_without_applying(window, tmp_path):
    doc = open_image(window, tmp_path / "sample.png")
    window._app_settings.calibration_presets = []
    window._refresh_preset_combo()
    bar = window._calibration_bar
    menu = bar.choose.menu()
    menu.aboutToShow.emit()
    assert bar.choose.isEnabled() and bar.draw.isEnabled()
    assert {"新增标尺…", "导入 CU 标尺…", "标定详情与标尺管理…"} <= {a.text() for a in menu.actions()}
    assert doc.calibration is None and window.project.project_default_calibration is None
    bar.draw.click()
    assert window._tool_mode == "calibration"
    assert doc.calibration is None
    window._remove_document(doc.id)
    assert bar.notice.text() == "请先打开图像"
    assert bar.notice.property("statusTone") is None and bar.actions.isHidden()


@pytest.mark.parametrize("entry", ["bar", "toolbar", "draw"])
def test_single_image_project_scope_defaults_to_all_and_new_images_inherit(window, tmp_path, monkeypatch, entry):
    first = open_image(window, tmp_path / "first.png")
    window._app_settings.calibration_presets = [
        CalibrationPreset(name="校准镜头", pixels_per_unit=4., pixels_per_unit_y=2., unit="um"),
    ]
    window._refresh_preset_combo()
    dialogs = []

    def accept_scope(dialog):
        scope = dialog.findChild(CalibrationScopeSelector)
        assert scope is not None
        assert scope.project.isChecked() and "1 张" in scope.effect.text()
        assert first.calibration is None
        dialogs.append(dialog.windowTitle())
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", accept_scope)
    if entry == "draw":
        window._apply_calibration_line(first, Line(Point(0, 0), Point(40, 0)))
    else:
        menu = window._calibration_bar.choose.menu() if entry == "bar" else window._measurement_context_bar.calibrationMenu
        menu.aboutToShow.emit()
        next(action for action in menu.actions() if action.text().startswith("应用：")).trigger()
        assert "X 0.25 / Y 0.5 μm/px" in window._calibration_bar.status.accessibleName()
    assert len(dialogs) == 1
    assert first.calibration.mode == "project_default"
    second = open_image(window, tmp_path / "second.png")
    assert second.calibration.to_dict() == first.calibration.to_dict()
    assert "项目标尺" in window._calibration_bar.status.accessibleName()


@pytest.mark.parametrize("accept_current", [True, False])
def test_draw_calibration_respects_current_or_cancel(window, tmp_path, monkeypatch, accept_current):
    first = open_image(window, tmp_path / "first.png")
    second = open_image(window, tmp_path / "second.png")

    def respond(dialog):
        assert "2 张" in dialog.scope.effect.text()
        dialog.scope.current.click()
        assert "1 张" in dialog.scope.effect.text()
        return QDialog.DialogCode.Accepted if accept_current else QDialog.DialogCode.Rejected

    monkeypatch.setattr(CalibrationInputDialog, "exec", respond)
    window._apply_calibration_line(second, Line(Point(0, 0), Point(40, 0)))
    assert first.calibration is None and window.project.project_default_calibration is None
    assert (second.calibration is not None) == accept_current


@pytest.mark.parametrize("entry", ["draw", "preset"])
def test_calibration_rechecks_document_at_confirmation(window, tmp_path, monkeypatch, entry):
    first = open_image(window, tmp_path / "first.png")
    window._app_settings.calibration_presets = [CalibrationPreset(name="测试", pixels_per_unit=2., unit="um")]
    window._refresh_preset_combo()

    def switch_image(dialog):
        open_image(window, tmp_path / "second.png")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", switch_image)
    if entry == "draw":
        window._apply_calibration_line(first, Line(Point(0, 0), Point(40, 0)))
    else:
        window.apply_selected_preset()
    assert len(window.project.documents) == 2
    assert all(doc.calibration is None for doc in window.project.documents)
    assert window.project.project_default_calibration is None


def test_read_only_and_non_measurement_workspaces_cannot_apply_scale(window, tmp_path, monkeypatch):
    doc = open_image(window, tmp_path / "image.png")
    window._app_settings.calibration_presets = [CalibrationPreset(name="测试", pixels_per_unit=2., unit="um")]
    monkeypatch.setattr(QDialog, "exec", lambda *_: pytest.fail("Unexpected calibration dialog"))
    for read_only, acquisition in ((True, False), (False, True)):
        window.project.compatibility = ProjectCompatibilityState(unknown_required_features=("new/v1",) if read_only else ())
        window._acquisition_workspace_selected = acquisition
        window._update_action_states()
        window._refresh_preset_combo()
        assert not window._apply_preset_button.isEnabled()
        assert not window._mode_actions["calibration"].isEnabled()
        window.apply_selected_preset()
        window._apply_calibration_line(doc, Line(Point(0, 0), Point(40, 0)))
    window._acquisition_workspace_selected = False
    assert doc.calibration is None and window.project.project_default_calibration is None


def test_slide_requires_a_native_work_view_and_height_images_allow_xy_calibration(window, tmp_path, monkeypatch):
    doc = open_image(window, tmp_path / "image.png")
    window._app_settings.calibration_presets = [CalibrationPreset(name="测试", pixels_per_unit=2., unit="um")]
    window._refresh_preset_combo()
    slide = DigitalSlideCanvas(window)
    with monkeypatch.context() as scoped:
        scoped.setattr(window, "current_canvas", lambda: slide)
        scoped.setattr(slide, "pixel_work_enabled", lambda: False)
        scoped.setattr(QDialog, "exec", lambda *_: pytest.fail("Native slide viewport is unavailable"))
        window.apply_selected_preset()
        window._apply_calibration_line(doc, Line(Point(0, 0), Point(40, 0)))
    assert doc.calibration is None
    doc.raster_semantic = RasterSemantic.HEIGHT
    monkeypatch.setattr(QDialog, "exec", lambda *_: QDialog.DialogCode.Accepted)
    window.apply_selected_preset()
    assert doc.raster_semantic is RasterSemantic.HEIGHT
    assert doc.calibration.pixels_per_unit == 2.


def test_management_dialog_dismissed_when_measurement_workspace_changes(window, tmp_path):
    doc = open_image(window, tmp_path / "image.png")
    closed_on_switch = []

    def switch_workspace():
        window._acquisition_workspace_selected = True
        window._update_action_states()
        closed_on_switch.append(not window._calibration_manager.isVisible())
        window._calibration_manager.reject()

    QTimer.singleShot(0, switch_workspace)
    window._show_calibration_controls()
    window._acquisition_workspace_selected = False
    assert closed_on_switch == [True]
    assert doc.calibration is None
