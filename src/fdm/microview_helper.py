from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import sys
import tempfile
from pathlib import Path

from fdm.services.capture import CaptureDevice, MicroviewCaptureBackend


class _PreviewTarget:
    def __init__(self, hwnd: int, width: int, height: int) -> None:
        self._hwnd = max(0, int(hwnd))
        self._width = max(1, int(width))
        self._height = max(1, int(height))

    def native_preview_handle(self) -> int:
        return self._hwnd

    def native_preview_size(self) -> tuple[int, int]:
        return self._width, self._height


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


def _device(index: int) -> CaptureDevice:
    return CaptureDevice(
        id=f"microview:{index}",
        name=f"Microview #{index + 1}",
        backend_key="microview",
        native_id=index,
    )


def _save_temp_png(image) -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    path = Path(handle.name)
    handle.close()
    if not image.save(str(path), "PNG"):
        path.unlink(missing_ok=True)
        raise RuntimeError("无法写出临时抓拍图像。")
    return path


def _run_list() -> int:
    backend = MicroviewCaptureBackend()
    try:
        devices = []
        for device in backend.list_devices():
            resolution = backend.preview_resolution(device)
            try:
                board_type = backend._resolve_board_type_for_device(device)
            except Exception:
                board_type = 0
            devices.append(
                {
                    "index": int(device.native_id),
                    "name": device.name,
                    "resolution": list(resolution) if resolution is not None else [],
                    "board_type": int(board_type),
                }
            )
        _emit({"status": "ok", "devices": devices})
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit({"status": "error", "message": str(exc)})
        return 1
    finally:
        try:
            backend.stop_preview()
        except Exception:
            pass


def _run_preview(args) -> int:
    from fdm.services.microview_transport import FramePublisher
    backend = MicroviewCaptureBackend(frame_tap=True)
    publisher = None
    commands = queue.Queue()
    control_closed = threading.Event()
    stage = "open_device"
    def enqueue_command(payload):
        if control_closed.is_set():
            return
        commands.put(payload)
        # Retain the tap across wake(). Even if stop_preview releases the
        # backend reference concurrently, its event cannot be destroyed here.
        tap = backend._frame_tap
        if tap is not None:
            tap.wake()
    def read_commands():
        try:
            for line in sys.stdin:
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        enqueue_command(payload)
                except json.JSONDecodeError:
                    continue
        finally:
            enqueue_command({"command": "stop"})
    threading.Thread(target=read_commands, name="microview-control", daemon=True).start()
    try:
        target = _PreviewTarget(args.preview_hwnd, args.preview_width, args.preview_height)
        backend.set_native_preview_visible(bool(args.native_preview_visible))
        backend.start_preview(_device(args.device_index), preview_target=target,
            frame_callback=lambda image: None,
            error_callback=lambda message: _emit({"type": "error", "stage": stage, "message": message}))
        stage = "await_complete_frame"
        publisher = FramePublisher(backend, _emit)
        announced = False
        last_frame = time.perf_counter()
        while True:
            try:
                command = commands.get_nowait()
            except queue.Empty:
                command = {}
            kind = command.get("command")
            if kind == "stop":
                break
            if kind == "ack":
                publisher.acknowledge(command["sequence"])
            elif kind == "update_target":
                backend.update_preview_target(_PreviewTarget(int(command["hwnd"]),
                    int(command["width"]), int(command["height"])))
            elif kind == "native_preview_visible":
                backend.set_native_preview_visible(bool(command["visible"]))
            if publisher.publish():
                last_frame = time.perf_counter()
                if not announced:
                    announced = True
                    stage = "streaming"
                    _emit({"type": "started", "resolution": list(backend._preview_resolution or ()),
                           "warning": backend.active_warning(), "board_type": int(backend._board_type or 0)})
            if time.perf_counter() - last_frame > (5 if not announced else 3):
                diagnostic = backend._frame_tap.diagnostics()
                raise TimeoutError("未收到有效完整帧或接收端未确认帧；" + json.dumps(diagnostic, allow_nan=False))
            if commands.empty():
                # Native callbacks and control input share a retained wake.
                # The timeout is only for the disconnect watchdog; frame
                # delivery never waits for a 3 ms Windows polling timer.
                backend._frame_tap.wait(100)
        return 0
    except Exception as exc:
        _emit({"type": "error", "stage": stage, "message": f"Microview [{stage}]: {exc}"})
        return 1
    finally:
        control_closed.set()
        # On failed SDK retirement the process owns all callback memory until
        # termination. Never unload DLLs or unmap buffers underneath a writer.
        try:
            backend.stop_preview()
        except Exception as exc:
            _emit({"type": "error", "stage": "stop_device", "message": str(exc)})
            os._exit(2)
        if publisher is not None:
            publisher.close()


def _run_capture(args) -> int:
    backend = MicroviewCaptureBackend()
    device = _device(args.device_index)
    try:
        image = backend.capture_still_frame(device)
        if image is None or image.isNull():
            raise RuntimeError("未抓拍到有效图像。")
        image_path = _save_temp_png(image)
        _emit(
            {
                "status": "ok",
                "image_path": str(image_path),
                "width": image.width(),
                "height": image.height(),
                "diagnostics": backend.last_capture_diagnostics(),
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit(
            {
                "status": "error",
                "message": str(exc),
                "diagnostics": backend.last_capture_diagnostics(),
            }
        )
        return 1


def _run_optimize(args) -> int:
    backend = MicroviewCaptureBackend()
    device = _device(args.device_index)
    try:
        message = backend.optimize_signal(device)
        _emit({"status": "ok", "message": message})
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit({"status": "error", "message": str(exc)})
        return 1
    finally:
        try:
            backend.stop_preview()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fdm.microview_helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    preview = subparsers.add_parser("preview")
    preview.add_argument("--device-index", type=int, required=True)
    preview.add_argument("--preview-hwnd", type=int, required=True)
    preview.add_argument("--preview-width", type=int, required=True)
    preview.add_argument("--preview-height", type=int, required=True)
    preview.add_argument("--native-preview-visible", type=int, choices=(0, 1), default=1)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--device-index", type=int, required=True)

    optimize = subparsers.add_parser("optimize")
    optimize.add_argument("--device-index", type=int, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _run_list()
    if args.command == "preview":
        return _run_preview(args)
    if args.command == "capture":
        return _run_capture(args)
    if args.command == "optimize":
        return _run_optimize(args)
    _emit({"status": "error", "message": f"未知命令: {args.command}"})
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
