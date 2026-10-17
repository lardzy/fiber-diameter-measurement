"""Native display ownership must not change acquisition or SDK processing."""
from types import SimpleNamespace
import json
import pytest

from PySide6.QtCore import QCoreApplication, QProcess

from fdm.microview_helper import build_parser
from fdm.services.capture import (
    CaptureBackend, CaptureDevice, CaptureSessionManager, MicroviewCaptureBackend,
    MicroviewIsolatedBackend, _microview_helper_command,
)


def test_capture_api_version_is_independent_of_the_computation_kernel_api():
    from fdm.services.microview_transport import bridge_build_matches
    kernel = {"api_version": 1, "python_abi": "win_amd64", "source_sha256": "current"}
    capture = {**kernel, "api_version": 3, "image_info_size": 20}
    assert bridge_build_matches(capture, kernel)


@pytest.mark.parametrize("change", [
    {"api_version": 1}, {"api_version": 2}, {"image_info_size": 24}, {"python_abi": "other"},
    {"source_sha256": "stale"},
])
def test_capture_build_contract_rejects_stale_or_incompatible_bridge(change):
    from fdm.services.microview_transport import bridge_build_matches
    kernel = {"api_version": 1, "python_abi": "win_amd64", "source_sha256": "current"}
    capture = {**kernel, "api_version": 3, "image_info_size": 20, **change}
    assert not bridge_build_matches(capture, kernel)


def test_initial_display_owner_is_passed_to_helper_before_device_start():
    target = SimpleNamespace(native_preview_handle=lambda: 1234,
                             native_preview_size=lambda: (768, 576))
    for visible in (False, True):
        _, args = _microview_helper_command("preview", device_index=0,
            preview_target=target, native_preview_visible=visible)
        parsed = build_parser().parse_args(args[args.index("preview"):])
        assert parsed.native_preview_visible == int(visible)
        assert (parsed.preview_width, parsed.preview_height) == (768, 576)
        assert parsed.preview_hwnd == 1234


def test_manager_preserves_display_owner_across_restart_and_sets_it_before_open():
    app = QCoreApplication.instance() or QCoreApplication([])

    class Backend(CaptureBackend):
        backend_key = "display-test"
        def __init__(self):
            self.visible = True
            self.opens = []
            self.stops = 0
        def list_devices(self):
            return [CaptureDevice("display-test:0", "Test", self.backend_key, 0)]
        def set_native_preview_visible(self, visible):
            self.visible = bool(visible)
        def start_preview(self, device, **kwargs):
            self.opens.append(self.visible)
        def stop_preview(self):
            self.stops += 1

    backend = Backend()
    manager = CaptureSessionManager(backends=[backend], refresh_on_init=False)
    manager.refresh_devices()
    manager.set_native_preview_visible(False)
    assert manager.start_preview()
    assert backend.opens == [False]
    manager.set_native_preview_visible(True)
    assert backend.visible is True and backend.stops == 0
    manager.set_native_preview_visible(False)
    assert manager.stop_preview()
    assert manager.start_preview()
    assert backend.opens == [False, False]
    manager.stop_preview()
    app.processEvents()


def test_running_helper_visibility_command_does_not_rebind_or_restart_target():
    class Process:
        def __init__(self):
            self.writes = []
        def state(self):
            return QProcess.ProcessState.Running
        def write(self, value):
            self.writes.append(json.loads(value))
            return len(value)
    backend = MicroviewIsolatedBackend()
    backend.set_native_preview_visible(False)
    assert backend._native_preview_visible is False
    assert backend.preview_kind(None) == "native_embed"
    backend._preview_process = process = Process()
    backend.set_native_preview_visible(True)
    backend.set_native_preview_visible(False)
    assert process.writes == [
        {"command": "native_preview_visible", "visible": True},
        {"command": "native_preview_visible", "visible": False},
    ]


def test_backend_display_switch_only_touches_the_processed_callback():
    backend = MicroviewCaptureBackend(frame_tap=True)
    changes = []
    backend._frame_tap = SimpleNamespace(set_native_preview_visible=changes.append)
    class Dll:
        def MV_SetDeviceParameter(self, *_):
            raise AssertionError("Display routing must not disable SDK processing or change field mode")
    backend._dll = Dll()
    backend._device_handle = 1
    backend.set_native_preview_visible(False)
    backend.set_native_preview_visible(True)
    assert changes == [False, True]
