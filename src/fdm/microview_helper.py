from __future__ import annotations

import argparse
import json
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
    print(json.dumps(payload, ensure_ascii=False), flush=True)


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
    backend = MicroviewCaptureBackend()
    target = _PreviewTarget(args.preview_hwnd, args.preview_width, args.preview_height)
    device = _device(args.device_index)
    try:
        backend.start_preview(
            device,
            preview_target=target,
            frame_callback=lambda image: None,
            error_callback=lambda message: _emit({"type": "error", "message": message}),
        )
        resolution = backend.preview_resolution(device)
        _emit(
            {
                "type": "started",
                "resolution": list(resolution) if resolution is not None else [],
                "warning": backend.active_warning(),
                "board_type": int(backend._board_type or 0),
            }
        )
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            command = str(payload.get("command", "")).strip()
            if command == "stop":
                break
            if command == "update_target":
                try:
                    target = _PreviewTarget(
                        int(payload.get("hwnd", 0)),
                        int(payload.get("width", 1)),
                        int(payload.get("height", 1)),
                    )
                    backend.update_preview_target(target)
                except Exception as exc:  # noqa: BLE001
                    _emit({"type": "error", "message": f"Microview 预览目标更新失败: {exc}"})
            if command == "snapshot":
                request_id = int(payload.get("request_id", 0))
                try:
                    frame = backend._capture_single_frame_image(handle=backend._device_handle, process=False)
                except Exception as exc:  # noqa: BLE001
                    _emit({"type": "snapshot_error", "request_id": request_id, "message": str(exc)})
                    continue
                if frame.isNull():
                    _emit(
                        {
                            "type": "snapshot_error",
                            "request_id": request_id,
                            "message": backend.last_capture_diagnostics() or "Microview 分析帧抓取失败。",
                        }
                    )
                    continue
                try:
                    image_path = _save_temp_png(frame)
                except Exception as exc:  # noqa: BLE001
                    _emit({"type": "snapshot_error", "request_id": request_id, "message": str(exc)})
                    continue
                _emit(
                    {
                        "type": "snapshot",
                        "request_id": request_id,
                        "image_path": str(image_path),
                        "width": frame.width(),
                        "height": frame.height(),
                    }
                )
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit({"type": "error", "message": str(exc)})
        return 1
    finally:
        try:
            backend.stop_preview()
        except Exception:
            pass


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
