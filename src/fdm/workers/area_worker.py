from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import base64
import importlib.util
import json
import os
import sys
import traceback
from typing import TextIO

from fdm.area_worker_protocol import AREA_WORKER_PROTOCOL, AREA_WORKER_PROTOCOL_VERSION


PROTOCOL_NAME = AREA_WORKER_PROTOCOL
PROTOCOL_VERSION = AREA_WORKER_PROTOCOL_VERSION
PERSISTENT_MAX_REQUESTS = 100
PERSISTENT_IDLE_TIMEOUT_S = 60.0
PERSISTENT_MAX_RSS_BYTES = 1536 * 1024 * 1024
MAX_DESKTOP_IMAGE_BYTES = 48 * 1024 * 1024
ALLOW_UNTRUSTED_AREA_MODELS_ENV = "FDM_ALLOW_UNTRUSTED_AREA_MODELS"
AREA_WORKER_DIAGNOSTICS_ENV = "FDM_AREA_WORKER_DIAGNOSTICS"
AREA_WORKER_LOG_PATH_ENV = "FDM_AREA_WORKER_LOG_PATH"
AREA_WORKER_REQUEST_ID_ENV = "FDM_AREA_WORKER_REQUEST_ID"


def _configure_protocol_streams() -> tuple[TextIO, TextIO, TextIO]:
    """Force the worker pipe protocol to UTF-8 on every Windows code page.

    A frozen console executable can recreate ``sys.std*`` using the active
    Windows ANSI code page even when the parent process requested UTF-8.  The
    parent always sends and receives UTF-8 JSON, so leaving the worker streams
    on GBK corrupts paths containing Chinese characters in both directions.
    """

    stream_specs = (
        (sys.stdin, "strict"),
        (sys.stdout, "strict"),
        (sys.stderr, "backslashreplace"),
    )
    for stream, errors in stream_specs:
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors=errors)
    return sys.stdin, sys.stdout, sys.stderr


def _trace_worker_stage(
    stage: str,
    *,
    request_id: str = "",
    diagnostic_stream: TextIO | None = None,
    **details: object,
) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "pid": os.getpid(),
        "request_id": str(request_id or ""),
        "stage": str(stage),
        "details": {key: value for key, value in details.items()},
    }
    try:
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        line = json.dumps(
            {
                "timestamp": payload["timestamp"],
                "pid": payload["pid"],
                "request_id": payload["request_id"],
                "stage": payload["stage"],
                "details": {"serialization_error": True},
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    stream = diagnostic_stream or sys.stderr
    try:
        print(f"area_worker_stage {line}", file=stream, flush=True)
    except (OSError, ValueError):
        pass
    enabled = str(os.environ.get(AREA_WORKER_DIAGNOSTICS_ENV, "")).strip().lower()
    log_token = str(os.environ.get(AREA_WORKER_LOG_PATH_ENV, "")).strip()
    if enabled not in {"1", "true", "yes", "on"} or not log_token:
        return
    try:
        log_path = Path(log_token)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        return


class _RequestFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


class _AreaWorkerRuntime:
    def __init__(self) -> None:
        self._engine = None
        self._engine_key: tuple[str, str, str, bool, bool] | None = None

    def engine_for(
        self,
        *,
        vendor_root: Path,
        weights_dir: Path,
        infer_device: str,
        require_trusted_weights: bool,
        verify_trusted_weights: bool,
    ):
        key = (
            str(vendor_root),
            str(weights_dir),
            infer_device,
            require_trusted_weights,
            verify_trusted_weights,
        )
        if self._engine is not None and self._engine_key == key:
            return self._engine
        engine_module = _load_engine_module(vendor_root)
        self._engine = engine_module.AreaNativeEngine(
            weights_dir=str(weights_dir),
            vendor_root=str(vendor_root),
            infer_device=infer_device,
            max_cached_models=2,
            require_trusted_weights=require_trusted_weights,
            verify_trusted_weights=verify_trusted_weights,
            max_image_bytes=MAX_DESKTOP_IMAGE_BYTES,
        )
        self._engine_key = key
        return self._engine


def _load_engine_module(vendor_root: Path):
    area_infer_root = vendor_root.parent.parent
    engine_path = area_infer_root / "app" / "engine.py"
    if not engine_path.exists():
        raise RuntimeError(f"未找到参考 area engine: {engine_path}")
    root_token = str(area_infer_root)
    if root_token not in sys.path:
        sys.path.insert(0, root_token)
    module_name = "fdm_area_ref_engine"
    existing = sys.modules.get(module_name)
    if existing is not None and Path(str(getattr(existing, "__file__", ""))).resolve() == engine_path.resolve():
        return existing
    spec = importlib.util.spec_from_file_location(module_name, engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 area engine: {engine_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _parse_request(raw: str) -> tuple[str, dict[str, object]]:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise _RequestFailure("invalid_request", f"输入 JSON 无法解析: {exc}") from exc
    if not isinstance(payload, dict):
        raise _RequestFailure("invalid_request", "请求必须是 JSON 对象。")

    request_id = str(payload.get("request_id") or "").strip()
    if not request_id or len(request_id) > 128:
        raise _RequestFailure("invalid_request", "request_id 为空或过长。")
    if payload.get("protocol") != PROTOCOL_NAME:
        raise _RequestFailure("invalid_request", "协议标识无效。")
    version = payload.get("version")
    if isinstance(version, bool) or version != PROTOCOL_VERSION:
        raise _RequestFailure("invalid_request", f"不支持的协议版本: {version!r}。")
    if payload.get("op") not in {"hello", "infer"}:
        raise _RequestFailure("invalid_request", "仅支持 hello、infer 操作。")
    return request_id, payload


def _mapping(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise _RequestFailure("invalid_request", f"{name} 必须是对象。")
    return value


def _runtime_bool(payload: dict[str, object], name: str, *, default: bool) -> bool:
    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise _RequestFailure("invalid_request", f"runtime.{name} 必须是布尔值。")
    return value


def _allow_untrusted_development_weights(runtime: dict[str, object]) -> bool:
    requested = _runtime_bool(runtime, "allow_untrusted_weights", default=False)
    # Validate legacy/requested policy fields, but never let a request weaken
    # the worker-owned policy. Frozen workers always require the allowlist and
    # SHA256 verification, even if the parent process environment was changed.
    _runtime_bool(runtime, "require_trusted_weights", default=True)
    _runtime_bool(runtime, "verify_trusted_weights", default=True)
    if getattr(sys, "frozen", False) or not requested:
        return False
    token = str(os.environ.get(ALLOW_UNTRUSTED_AREA_MODELS_ENV, "")).strip().lower()
    return token in {"1", "true", "yes", "on"}


def _normalize_device(value: object) -> str:
    token = str(value or "cpu").strip().lower()
    if token == "cuda":
        token = "cuda:0"
    if token not in {"cpu", "auto", "cuda:0"}:
        raise _RequestFailure("invalid_request", "runtime.device 仅支持 cpu、auto、cuda:0。")
    return token


def _execute_request(
    payload: dict[str, object],
    *,
    worker_runtime: _AreaWorkerRuntime | None = None,
    diagnostic_stream: TextIO | None = None,
) -> dict[str, object]:
    image = _mapping(payload, "image")
    model = _mapping(payload, "model")
    runtime = _mapping(payload, "runtime")
    options = _mapping(payload, "options")

    image_path = Path(str(image.get("path") or "").strip()).expanduser().resolve()
    model_name = str(model.get("name") or "").strip()
    model_file = Path(str(model.get("file") or "").strip()).name
    vendor_root = Path(str(runtime.get("vendor_root") or "").strip()).expanduser().resolve()
    weights_dir = Path(str(runtime.get("weights_dir") or "").strip()).expanduser().resolve()
    infer_device = _normalize_device(runtime.get("device", "cpu"))
    allow_untrusted_weights = _allow_untrusted_development_weights(runtime)
    require_trusted_weights = not allow_untrusted_weights
    verify_trusted_weights = True
    include_overlay = options.get("include_overlay", False)
    inference_options = options.get("inference", {})
    request_id = str(payload.get("request_id") or "")

    if not image_path.is_file():
        raise _RequestFailure("invalid_request", f"未找到图片: {image_path}")
    if image_path.stat().st_size > MAX_DESKTOP_IMAGE_BYTES:
        raise _RequestFailure("invalid_request", "图片文件超过 48 MiB 上限。")
    if not model_name or not model_file:
        raise _RequestFailure("invalid_request", "模型名称或权重文件名为空。")
    if not vendor_root.is_dir():
        raise _RequestFailure("runtime_unavailable", f"未找到 YOLACT vendor 目录: {vendor_root}", exit_code=3)
    if not weights_dir.is_dir():
        raise _RequestFailure("model_not_found", f"未找到权重目录: {weights_dir}", exit_code=3)
    weight_path = weights_dir / model_file
    if not weight_path.is_file():
        raise _RequestFailure("model_not_found", f"未找到权重文件: {weight_path}", exit_code=3)
    if not isinstance(include_overlay, bool):
        raise _RequestFailure("invalid_request", "options.include_overlay 必须是布尔值。")
    if not isinstance(inference_options, dict):
        raise _RequestFailure("invalid_request", "options.inference 必须是对象。")

    _trace_worker_stage(
        "image_read_started",
        request_id=request_id,
        diagnostic_stream=diagnostic_stream,
        image_path=str(image_path),
    )
    raw = image_path.read_bytes()
    _trace_worker_stage(
        "image_read_completed",
        request_id=request_id,
        diagnostic_stream=diagnostic_stream,
        image_bytes=len(raw),
    )
    runtime_state = worker_runtime or _AreaWorkerRuntime()
    _trace_worker_stage(
        "engine_prepare_started",
        request_id=request_id,
        diagnostic_stream=diagnostic_stream,
        model_file=model_file,
        device=infer_device,
    )
    engine = runtime_state.engine_for(
        vendor_root=vendor_root,
        weights_dir=weights_dir,
        infer_device=infer_device,
        require_trusted_weights=require_trusted_weights,
        verify_trusted_weights=verify_trusted_weights,
    )
    _trace_worker_stage(
        "engine_prepare_completed",
        request_id=request_id,
        diagnostic_stream=diagnostic_stream,
    )
    engine_options = dict(inference_options)
    engine_options["include_overlay"] = include_overlay
    os.environ[AREA_WORKER_REQUEST_ID_ENV] = request_id
    _trace_worker_stage(
        "inference_started",
        request_id=request_id,
        diagnostic_stream=diagnostic_stream,
        model_file=model_file,
        device=infer_device,
    )
    result = engine.infer(
        model_name=model_name,
        model_file=model_file,
        image_bytes_b64=base64.b64encode(raw).decode("ascii"),
        inference_options=engine_options,
    )
    _trace_worker_stage(
        "inference_completed",
        request_id=request_id,
        diagnostic_stream=diagnostic_stream,
        instance_count=(len(result.get("instances", [])) if isinstance(result, dict) else -1),
    )
    if not isinstance(result, dict):
        raise RuntimeError("area engine 返回值不是对象。")
    result = dict(result)
    if not include_overlay:
        result.pop("overlay_png_b64", None)
    return result


def _response(
    *,
    request_id: str,
    ok: bool,
    result: dict[str, object] | None = None,
    error_code: str = "",
    error_message: str = "",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": ok,
    }
    if ok:
        payload["result"] = dict(result or {})
    else:
        payload["error"] = {
            "code": error_code or "internal_error",
            "message": error_message or "面积识别失败。",
        }
    return payload


def _write_response(stream: TextIO, response: dict[str, object]) -> None:
    try:
        # Keep the wire envelope ASCII-only.  It remains valid UTF-8 JSON and
        # also survives a frozen Windows runtime recreating a pipe with the
        # active legacy code page before stream reconfiguration takes effect.
        response_text = json.dumps(response, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        traceback.print_exc(file=sys.stderr)
        response_text = json.dumps(
            _response(
                request_id=str(response.get("request_id") or ""),
                ok=False,
                error_code="internal_error",
                error_message=f"响应无法序列化: {exc}",
            ),
            ensure_ascii=True,
            allow_nan=False,
        )
    stream.write(response_text)
    stream.write("\n")
    stream.flush()


def _process_request(
    raw_request: str,
    *,
    worker_runtime: _AreaWorkerRuntime,
    diagnostic_stream: TextIO,
) -> tuple[dict[str, object], int]:
    request_id = _request_id_from_raw(raw_request)
    _trace_worker_stage(
        "request_received",
        request_id=request_id,
        diagnostic_stream=diagnostic_stream,
        request_bytes=len(raw_request.encode("utf-8", errors="replace")),
    )
    try:
        request_id, payload = _parse_request(raw_request)
        if payload.get("op") == "hello":
            result = {
                "status": "ready",
                "mode": "persistent",
                "max_requests": PERSISTENT_MAX_REQUESTS,
                "idle_timeout_s": PERSISTENT_IDLE_TIMEOUT_S,
                "max_rss_bytes": PERSISTENT_MAX_RSS_BYTES,
            }
        else:
            result = _execute_request(
                payload,
                worker_runtime=worker_runtime,
                diagnostic_stream=diagnostic_stream,
            )
        _trace_worker_stage(
            "response_ready",
            request_id=request_id,
            diagnostic_stream=diagnostic_stream,
            ok=True,
        )
        return _response(request_id=request_id, ok=True, result=result), 0
    except _RequestFailure as exc:
        _trace_worker_stage(
            "request_failed",
            request_id=request_id,
            diagnostic_stream=diagnostic_stream,
            error_code=exc.code,
            error=str(exc),
        )
        print(str(exc), file=diagnostic_stream)
        return (
            _response(
                request_id=request_id,
                ok=False,
                error_code=exc.code,
                error_message=str(exc),
            ),
            exc.exit_code,
        )
    except ModuleNotFoundError as exc:
        _trace_worker_stage(
            "request_failed",
            request_id=request_id,
            diagnostic_stream=diagnostic_stream,
            error_code="runtime_unavailable",
            error=str(exc),
        )
        traceback.print_exc(file=diagnostic_stream)
        return (
            _response(
                request_id=request_id,
                ok=False,
                error_code="runtime_unavailable",
                error_message=str(exc),
            ),
            3,
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=diagnostic_stream)
        engine_code = str(getattr(exc, "code", "") or "")
        if engine_code == "infer_model_load_failed":
            error_code, exit_code = "model_not_found", 3
        elif engine_code == "infer_service_unavailable":
            error_code, exit_code = "runtime_unavailable", 3
        elif engine_code == "infer_request_too_large":
            error_code, exit_code = "invalid_request", 2
        else:
            error_code, exit_code = "internal_error", 1
        _trace_worker_stage(
            "request_failed",
            request_id=request_id,
            diagnostic_stream=diagnostic_stream,
            error_code=error_code,
            error=str(exc),
        )
        return (
            _response(
                request_id=request_id,
                ok=False,
                error_code=error_code,
                error_message=str(exc),
            ),
            exit_code,
        )


def _stdin_lines(stream: TextIO, output: Queue[str | None]) -> None:
    try:
        for line in stream:
            output.put(line)
    finally:
        output.put(None)


def serve_persistent(
    *,
    protocol_stdout: TextIO,
    diagnostic_stream: TextIO,
    input_stream: TextIO,
    idle_timeout_s: float = PERSISTENT_IDLE_TIMEOUT_S,
    max_requests: int = PERSISTENT_MAX_REQUESTS,
    max_rss_bytes: int = PERSISTENT_MAX_RSS_BYTES,
) -> int:
    requests: Queue[str | None] = Queue()
    Thread(target=_stdin_lines, args=(input_stream, requests), name="area-worker-stdin", daemon=True).start()
    worker_runtime = _AreaWorkerRuntime()
    processed = 0
    while processed < max(1, int(max_requests)):
        try:
            raw_request = requests.get(timeout=max(0.01, float(idle_timeout_s)))
        except Empty:
            print("persistent_worker_idle_timeout", file=diagnostic_stream)
            return 0
        if raw_request is None:
            return 0
        if not raw_request.strip():
            continue
        with redirect_stdout(diagnostic_stream):
            response, exit_code = _process_request(
                raw_request,
                worker_runtime=worker_runtime,
                diagnostic_stream=diagnostic_stream,
            )
        _write_response(protocol_stdout, response)
        if _request_op(raw_request) == "hello" and exit_code == 0 and response.get("ok") is True:
            continue
        processed += 1
        if exit_code == 1:
            print("persistent_worker_recycle_after_internal_error", file=diagnostic_stream)
            return 0
        rss_bytes = _current_rss_bytes()
        if rss_bytes is not None and rss_bytes > max_rss_bytes:
            print(f"persistent_worker_rss_limit:{rss_bytes}>{max_rss_bytes}", file=diagnostic_stream)
            return 0
    print(f"persistent_worker_request_limit:{processed}", file=diagnostic_stream)
    return 0


def _current_rss_bytes() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except Exception:
            return None
        return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def main() -> int:
    input_stream, protocol_stdout, diagnostic_stream = _configure_protocol_streams()
    if "--persistent" in sys.argv[1:]:
        return serve_persistent(
            protocol_stdout=protocol_stdout,
            diagnostic_stream=diagnostic_stream,
            input_stream=input_stream,
        )

    raw_request = input_stream.read()
    worker_runtime = _AreaWorkerRuntime()
    with redirect_stdout(diagnostic_stream):
        response, exit_code = _process_request(
            raw_request,
            worker_runtime=worker_runtime,
            diagnostic_stream=diagnostic_stream,
        )
    _write_response(protocol_stdout, response)
    return exit_code


def _request_id_from_raw(raw: str) -> str:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return ""
    if isinstance(payload, dict):
        return str(payload.get("request_id") or "").strip()[:128]
    return ""


def _request_op(raw: str) -> str:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return ""
    if isinstance(payload, dict):
        return str(payload.get("op") or "").strip()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
