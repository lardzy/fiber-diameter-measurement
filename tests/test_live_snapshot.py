from unittest.mock import patch
from time import monotonic

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from acquisition_test_support import wait_workbench_ready

from fdm.services.capture import CaptureBackend, CaptureDevice
from fdm.ui.main_window import MainWindow
from fdm.ui.workspace import WorkspacePage


class StreamCamera(CaptureBackend):
    def __init__(self, backend_key="opencv"):
        self.backend_key = backend_key
        self.starts = self.stops = 0

    def list_devices(self):
        return [CaptureDevice(self.backend_key + ":0", "测试相机", self.backend_key, 0)]

    def can_capture_still(self, device):
        return True

    def capture_still_frame(self, device):
        raise AssertionError("a still must never reopen the camera")

    def start_preview(self, device, *, preview_target, frame_callback, error_callback):
        self.starts += 1
        self.feed = frame_callback

    def stop_preview(self):
        self.stops += 1

    def publish(self, color):
        image = QImage(160, 120, QImage.Format.Format_RGB32)
        image.fill(QColor(color))
        self.feed(image)
        for _ in range(3):
            QApplication.processEvents()
        return image


def start_stream(window, backend="opencv"):
    camera = StreamCamera(backend)
    window._capture_manager._backends = [camera]
    hub = window._open_acquisition_workbench(trial_index=0, auto_start=False)
    hub.request_start()
    deadline = monotonic() + 3
    while not window._capture_manager.is_preview_active():
        assert monotonic() < deadline, "Asynchronous camera did not start"
        QTest.qWait(2)
    camera.publish("#fefefe")
    return hub, camera


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr("fdm.services.capture.available_capture_backends", lambda: [])
    w = MainWindow()
    w.show()
    yield w
    with patch.object(w, "_confirm_close_documents_for_transition", return_value=True):
        w.close()


@pytest.mark.parametrize("backend", ["opencv", "qt_video", "microview"])
def test_three_clicks_keep_streaming_and_view_only_on_request(window, backend):
    hub, camera = start_stream(window, backend)
    control = window._snapshot_controls
    for index, color in enumerate(("#ff0000", "#00ff00", "#0000ff"), 1):
        prior = window._capture_manager.preview_frame_cursor()
        control.grab_button.click()
        control.grab_button.click()
        assert len(window.project.documents) == index - 1
        assert not window.capture_frame_action.isEnabled()
        expected = camera.publish(color)
        assert len(window.project.documents) == index
        document = window.project.documents[-1]
        assert window._images[document.id] == expected
        assert document.metadata["capture"]["generation"] == prior[0]
        assert document.metadata["capture"]["sequence"] > prior[1]
        assert control.notice.text() == f"已抓拍第 {index} 张"
        assert control.view.isVisible()
        assert window._adaptive_layout.page is WorkspacePage.CAPTURE
        assert hub.pages.currentWidget() is hub.capture_page
        assert window._center_stack.currentWidget() is hub
        assert (camera.starts, camera.stops) == (1, 0)
    control.view.click()
    assert window._adaptive_layout.page is WorkspacePage.MEASURE
    assert window.current_document().id == document.id
    assert len(set(d.path for d in window.project.documents)) == 3


@pytest.mark.parametrize("interrupt", ["stop", "leave", "timeout", "failure"])
def test_no_false_receipt_for_cancelled_or_failed_snapshot(window, interrupt):
    hub, camera = start_stream(window)
    control = window._snapshot_controls
    window.capture_current_frame()
    assert control.request is not None
    if interrupt == "stop":
        hub.toggle_capture()
    elif interrupt == "leave":
        hub.select_page(1)
        wait_workbench_ready(hub)
    elif interrupt == "timeout":
        control.timeout.timeout.emit()
    else:
        with patch.object(window, "_add_project_asset_image", side_effect=OSError("写入失败")):
            camera.publish("#aa33ff")
    camera.publish("#4455ff")
    assert not window.project.documents
    assert control.request is None and control.count == 0
    assert control.view.isHidden() and "已抓拍" not in control.notice.text()


@pytest.mark.parametrize("stage", ["convert", "attach_navigator", "after_tab_insert"])
def test_failed_capture_is_atomic_and_previous_success_stays_viewable(window, stage):
    hub, camera = start_stream(window)
    control = window._snapshot_controls
    control.grab_button.click()
    camera.publish("green")
    first = window.project.documents[0]
    first_pixels = bytes(window._rasters[first.id].data)
    old_tab, old_item = window.tab_widget.currentWidget(), window.image_list.currentItem()
    before = window.project.to_dict()
    window._mark_project_saved()
    if stage == "convert":
        failure = patch("fdm.ui.main_window.qimage_to_raster_plane", side_effect=ValueError("无法转换栅格"))
    elif stage == "attach_navigator":
        failure = patch.object(window, "_attach_canvas_navigator", side_effect=OSError("无法创建导航"))
    else:
        failure = patch.object(window, "_apply_open_view_mode", side_effect=MemoryError())
    control.grab_button.click()
    with failure as failing:
        camera.publish("red")
        camera.publish("blue")  # Later frames do not retry even after MemoryError.
        assert failing.call_count == 1
    assert window.project.to_dict() == before and not window._project_dirty()
    assert window._document_order == [first.id] and bytes(window._rasters[first.id].data) == first_pixels
    assert all(set(cache) == {first.id} for cache in (window._images, window._rasters,
               window._canvases, window._canvas_navigators, window._display_cache_transforms))
    assert window.tab_widget.count() == window.image_list.count() == 1
    assert window.tab_widget.currentWidget() is old_tab and window.image_list.currentItem() is old_item
    assert control.count == 1 and control.latest_document_id == first.id and control.view.isVisible()
    assert "抓拍失败" in control.notice.text() and not control.timeout.isActive()
    assert control.request is None and window.capture_frame_action.isEnabled()
    assert hub.pages.currentWidget() is hub.capture_page and (camera.starts, camera.stops) == (1, 0)
    control.grab_button.click()
    camera.publish("blue")
    assert control.count == 2 and len(window.project.documents) == 2
