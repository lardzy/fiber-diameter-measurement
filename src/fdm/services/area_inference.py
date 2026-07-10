from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
import json
import math
import os
import subprocess
import sys
import time
import uuid

from fdm.cancellation import CancellationToken
from fdm.geometry import Point
from fdm.settings import AppSettings


AREA_WORKER_PROTOCOL = "fdm.area-worker"
AREA_WORKER_PROTOCOL_VERSION = 1
DEFAULT_AREA_INFERENCE_TIMEOUT_S = 180.0
MIN_AREA_INFERENCE_TIMEOUT_S = 30.0
MAX_AREA_INFERENCE_TIMEOUT_S = 600.0
AREA_INFERENCE_POLL_INTERVAL_S = 0.1
AREA_INFERENCE_TERMINATE_GRACE_S = 2.0
MAX_AREA_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_AREA_INSTANCES = 10_000
MAX_AREA_STDERR_CHARS = 1024 * 1024
PERSISTENT_HANDSHAKE_TIMEOUT_S = 5.0
ALLOW_UNTRUSTED_AREA_MODELS_ENV = "FDM_ALLOW_UNTRUSTED_AREA_MODELS"

LABEL_ALIAS: dict[str, str] = {
    "粘": "粘纤",
    "莱": "莱赛尔",
    "莫": "莫代尔",
}

LABEL_SWAP_BY_MODEL: dict[str, dict[str, str]] = {
    "棉-莱赛尔": {
        "棉": "莱赛尔",
        "莱赛尔": "棉",
    },
    "粘纤-莱赛尔": {
        "粘纤": "莱赛尔",
        "莱赛尔": "粘纤",
    },
}


def normalize_area_label(label: str) -> str:
    token = str(label or "").strip()
    if not token:
        return "未分类"
    return LABEL_ALIAS.get(token, token)


def parse_area_model_labels(model_name: str) -> list[str]:
    labels: list[str] = []
    for item in str(model_name or "").split("-"):
        normalized = normalize_area_label(item)
        if normalized not in labels:
            labels.append(normalized)
    return labels or ["未分类"]


def normalize_area_model_name(model_name: str) -> str:
    return str(model_name or "").replace(" ", "").strip()


def normalize_area_result_label(model_name: str, label: str) -> str:
    normalized_label = normalize_area_label(label)
    swap_mapping = LABEL_SWAP_BY_MODEL.get(normalize_area_model_name(model_name), {})
    return swap_mapping.get(normalized_label, normalized_label)


def _allow_untrusted_development_area_models() -> bool:
    """Return the explicit source-development escape hatch state.

    Frozen releases deliberately ignore this environment variable: their
    runtime models must remain pinned to the hashes in model_metadata.py and
    runtime_assets.toml.
    """

    if getattr(sys, "frozen", False):
        return False
    token = str(os.environ.get(ALLOW_UNTRUSTED_AREA_MODELS_ENV, "")).strip().lower()
    return token in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AreaInstanceResult:
    class_name: str
    score: float
    bbox: list[int]
    polygon_px: list[Point]
    area_px: float


@dataclass(slots=True)
class AreaInferenceResult:
    instances: list[AreaInstanceResult]
    engine_meta: dict[str, object]


class AreaInferenceError(RuntimeError):
    pass


class AreaInferenceCancelledError(AreaInferenceError):
    pass


def _raise_if_area_cancelled(token: CancellationToken | None) -> None:
    if token is not None and token.is_cancelled:
        raise AreaInferenceCancelledError("面积识别已取消。")


class AreaInferenceTimeoutError(AreaInferenceError):
    pass


class AreaInferenceProtocolError(AreaInferenceError):
    pass


class AreaInferenceTransportError(AreaInferenceError):
    pass


class AreaWorkerSession:
    """Batch-scoped JSONL worker process with bounded recycling."""

    def __init__(
        self,
        *,
        worker_command: list[str],
        subprocess_kwargs: dict[str, object] | None = None,
        max_requests: int = 100,
    ) -> None:
        self._worker_command = list(worker_command)
        self._subprocess_kwargs = dict(subprocess_kwargs or {})
        self._max_requests = max(1, min(int(max_requests), 100))
        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: Queue[str | None] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_lock = Lock()
        self._request_count = 0
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    def __enter__(self) -> "AreaWorkerSession":
        return self

    def __exit__(self, exc_type, exc, traceback_object) -> None:
        self.close()

    def request(
        self,
        *,
        payload: dict[str, object],
        timeout_s: float,
        cancellation_token: CancellationToken | None,
    ) -> tuple[str, str]:
        if self._disabled:
            raise AreaInferenceTransportError("持久面积识别 worker 已禁用。")
        if self._request_count >= self._max_requests:
            self.close()
        self._ensure_started(cancellation_token=cancellation_token)
        process = self._process
        output_queue = self._stdout_queue
        if process is None or process.stdin is None or output_queue is None:
            self.disable()
            raise AreaInferenceTransportError("持久面积识别 worker 未正确启动。")
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            self.disable()
            raise AreaInferenceTransportError(f"无法写入持久面积识别 worker: {exc}") from exc

        started_at = time.monotonic()
        while True:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                self.close()
                raise AreaInferenceCancelledError("面积识别已取消。")
            elapsed_s = time.monotonic() - started_at
            if elapsed_s >= timeout_s:
                self.close()
                raise AreaInferenceTimeoutError(f"面积识别超过 {timeout_s:g} 秒，已终止 worker。")
            try:
                line = output_queue.get(
                    timeout=min(AREA_INFERENCE_POLL_INTERVAL_S, timeout_s - elapsed_s)
                )
            except Empty:
                if process.poll() is not None:
                    self.disable()
                    raise AreaInferenceTransportError(
                        "持久面积识别 worker 提前退出，"
                        f"退出码 {process.returncode}。{self.stderr_tail()}"
                    )
                continue
            if line is None:
                self.disable()
                raise AreaInferenceTransportError(
                    f"持久面积识别 worker 未返回完整响应。{self.stderr_tail()}"
                )
            if len(line.encode("utf-8", errors="replace")) > MAX_AREA_RESPONSE_BYTES:
                self.disable()
                raise AreaInferenceTransportError("持久面积识别响应超过上限。")
            self._request_count += 1
            return line, self.stderr_tail()

    def disable(self) -> None:
        self._disabled = True
        self.close()

    def close(self) -> None:
        process = self._process
        self._process = None
        self._stdout_queue = None
        self._request_count = 0
        if process is None:
            return
        _terminate_then_wait(process)

    def stderr_tail(self) -> str:
        with self._stderr_lock:
            return "".join(self._stderr_lines)[-MAX_AREA_STDERR_CHARS:]

    def _ensure_started(self, *, cancellation_token: CancellationToken | None) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None:
            self.close()
        environment = dict(os.environ)
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            process = subprocess.Popen(
                [*self._worker_command, "--persistent"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                **self._subprocess_kwargs,
            )
        except OSError as exc:
            self._disabled = True
            raise AreaInferenceTransportError(f"无法启动持久面积识别 worker: {exc}") from exc
        if process.stdout is None or process.stderr is None:
            _terminate_then_wait(process)
            self._disabled = True
            raise AreaInferenceTransportError("持久面积识别 worker 管道创建失败。")
        output_queue: Queue[str | None] = Queue()
        self._process = process
        self._stdout_queue = output_queue
        self._request_count = 0
        with self._stderr_lock:
            self._stderr_lines.clear()
        Thread(
            target=_read_worker_stdout,
            args=(process.stdout, output_queue),
            name="area-worker-stdout",
            daemon=True,
        ).start()
        Thread(
            target=self._read_stderr,
            args=(process.stderr,),
            name="area-worker-stderr",
            daemon=True,
        ).start()
        self._perform_handshake(cancellation_token=cancellation_token)

    def _perform_handshake(self, *, cancellation_token: CancellationToken | None) -> None:
        process = self._process
        output_queue = self._stdout_queue
        if process is None or process.stdin is None or output_queue is None:
            self.disable()
            raise AreaInferenceTransportError("持久面积识别 worker 握手管道无效。")
        request_id = f"hello-{uuid.uuid4().hex}"
        payload = {
            "protocol": AREA_WORKER_PROTOCOL,
            "version": AREA_WORKER_PROTOCOL_VERSION,
            "request_id": request_id,
            "op": "hello",
        }
        try:
            process.stdin.write(
                json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            self.disable()
            raise AreaInferenceTransportError(f"无法写入持久 worker 握手: {exc}") from exc

        started_at = time.monotonic()
        while True:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                self.close()
                raise AreaInferenceCancelledError("面积识别已取消。")
            elapsed_s = time.monotonic() - started_at
            if elapsed_s >= PERSISTENT_HANDSHAKE_TIMEOUT_S:
                self.disable()
                raise AreaInferenceTransportError("持久面积识别 worker 不支持 v1 JSONL 握手。")
            try:
                line = output_queue.get(timeout=AREA_INFERENCE_POLL_INTERVAL_S)
            except Empty:
                if process.poll() is not None:
                    self.disable()
                    raise AreaInferenceTransportError("持久面积识别 worker 在握手期间退出。")
                continue
            if line is None:
                self.disable()
                raise AreaInferenceTransportError("持久面积识别 worker 未返回握手响应。")
            try:
                response = _parse_worker_response(line.strip(), expected_request_id=request_id)
            except AreaInferenceProtocolError as exc:
                self.disable()
                raise AreaInferenceTransportError("持久面积识别 worker 握手协议无效。") from exc
            result = response.get("result")
            if response.get("ok") is not True or not isinstance(result, dict) or result.get("mode") != "persistent":
                self.disable()
                raise AreaInferenceTransportError("持久面积识别 worker 拒绝 JSONL 握手。")
            return

    def _read_stderr(self, stream) -> None:
        try:
            for line in stream:
                with self._stderr_lock:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > 2048:
                        del self._stderr_lines[:1024]
        except (OSError, ValueError):
            return


class AreaInferenceService:
    def __init__(self) -> None:
        self._worker_path = Path(__file__).resolve().parents[1] / "workers" / "area_worker.py"

    @staticmethod
    def _subprocess_kwargs() -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return kwargs

    def _worker_command(self, settings: AppSettings) -> list[str]:
        if getattr(sys, "frozen", False):
            executable = Path(sys.executable).resolve()
            sibling_worker = executable.with_name("FiberAreaWorker.exe")
            if sibling_worker.exists():
                return [str(sibling_worker)]
        configured = settings.resolved_area_worker_program()
        if configured:
            configured_path = Path(configured)
            if configured_path.exists() and configured_path.name.lower().startswith("fiberareaworker"):
                return [str(configured_path)]
            return [configured, str(self._worker_path)]
        return [sys.executable, str(self._worker_path)]

    def create_batch_session(self, settings: AppSettings) -> AreaWorkerSession:
        return AreaWorkerSession(
            worker_command=self._worker_command(settings),
            subprocess_kwargs=self._subprocess_kwargs(),
            max_requests=100,
        )

    def _friendly_failure_message(self, message: str, *, worker_command: list[str]) -> str:
        token = str(message or "").strip()
        missing_module = None
        if "No module named 'PIL'" in token or 'No module named "PIL"' in token:
            missing_module = "Pillow(PIL)"
        elif "No module named 'torchvision'" in token or 'No module named "torchvision"' in token:
            missing_module = "torchvision"
        elif "No module named 'torch'" in token or 'No module named "torch"' in token:
            missing_module = "torch"
        if missing_module is not None:
            command_hint = worker_command[0] if worker_command else "当前 Worker"
            return (
                f"面积识别运行环境缺少 {missing_module}。"
                f"\n当前使用的 Worker: {command_hint}"
                "\n如果你在源码环境运行，请安装面积识别依赖；"
                "如果你在打包后的程序中运行，请检查设置里的 Worker 是否仍指向外部 Python，建议留空使用自动模式。"
            )
        if "untrusted_model_file:" in token:
            model_file = token.split("untrusted_model_file:", 1)[1].split(":", 1)[0]
            if getattr(sys, "frozen", False):
                return (
                    f"面积模型 {model_file} 不在正式包的可信模型清单中，已拒绝加载。"
                    "请恢复 full 发布包内的原始模型文件，或重新安装通过哈希校验的正式版本。"
                )
            return (
                f"面积模型 {model_file} 不在可信模型清单中，默认拒绝加载。"
                f"仅在确认该本地 checkpoint 来源可信且为纯 state_dict 时，开发者可设置 "
                f"{ALLOW_UNTRUSTED_AREA_MODELS_ENV}=1 后重新启动；正式打包版本不接受此开关。"
            )
        if "safe_weights_only_" in token:
            return (
                "面积模型无法通过 PyTorch 安全权重加载。仅支持兼容的纯 tensor state_dict，"
                "不会回退到可执行任意 pickle 代码的不安全反序列化。\n"
                f"详细信息: {token}"
            )
        return token or "面积识别失败"

    def infer_image(
        self,
        *,
        image_path: str,
        model_name: str,
        model_file: str,
        settings: AppSettings,
        inference_options: dict[str, object] | None = None,
        timeout_s: float = DEFAULT_AREA_INFERENCE_TIMEOUT_S,
        cancellation_token: CancellationToken | None = None,
        worker_session: AreaWorkerSession | None = None,
    ) -> AreaInferenceResult:
        if not self._worker_path.exists() and not getattr(sys, "frozen", False):
            raise RuntimeError(f"未找到面积识别 worker: {self._worker_path}")

        validated_timeout_s = _validate_timeout_s(timeout_s)
        _raise_if_area_cancelled(cancellation_token)

        resolved_weights_dir = settings.resolved_area_weights_dir()
        resolved_vendor_root = settings.resolved_area_vendor_root()
        allow_untrusted_weights = _allow_untrusted_development_area_models()
        request_id = uuid.uuid4().hex
        payload = {
            "protocol": AREA_WORKER_PROTOCOL,
            "version": AREA_WORKER_PROTOCOL_VERSION,
            "request_id": request_id,
            "op": "infer",
            "image": {
                "path": str(Path(image_path).expanduser().resolve()),
            },
            "model": {
                "name": str(model_name),
                "file": str(model_file),
            },
            "runtime": {
                "weights_dir": str(resolved_weights_dir),
                "vendor_root": str(resolved_vendor_root),
                "device": str(getattr(settings, "area_infer_device", "cpu") or "cpu"),
                "allow_untrusted_weights": allow_untrusted_weights,
                "require_trusted_weights": not allow_untrusted_weights,
                "verify_trusted_weights": True,
            },
            "options": {
                "include_overlay": False,
                "inference": dict(inference_options or {}),
            },
        }
        worker_command = self._worker_command(settings)
        used_persistent_session = worker_session is not None and not worker_session.disabled
        if used_persistent_session:
            try:
                stdout, stderr = worker_session.request(
                    payload=payload,
                    timeout_s=validated_timeout_s,
                    cancellation_token=cancellation_token,
                )
                returncode = 0
            except (AreaInferenceCancelledError, AreaInferenceTimeoutError):
                raise
            except AreaInferenceTransportError:
                worker_session.disable()
                used_persistent_session = False
                returncode, stdout, stderr = self._run_worker_process(
                    worker_command=worker_command,
                    payload=payload,
                    timeout_s=validated_timeout_s,
                    cancellation_token=cancellation_token,
                )
        else:
            returncode, stdout, stderr = self._run_worker_process(
                worker_command=worker_command,
                payload=payload,
                timeout_s=validated_timeout_s,
                cancellation_token=cancellation_token,
            )
        _raise_if_area_cancelled(cancellation_token)
        stdout = stdout.strip()
        stderr = stderr.strip()
        if not stdout:
            raise AreaInferenceError(
                self._friendly_failure_message(
                    stderr[-MAX_AREA_STDERR_CHARS:] or "面积识别 worker 没有返回结果。",
                    worker_command=worker_command,
                )
            )

        try:
            response = _parse_worker_response(stdout, expected_request_id=request_id)
        except AreaInferenceProtocolError:
            if not used_persistent_session or worker_session is None:
                raise
            worker_session.disable()
            returncode, stdout, stderr = self._run_worker_process(
                worker_command=worker_command,
                payload=payload,
                timeout_s=validated_timeout_s,
                cancellation_token=cancellation_token,
            )
            stdout = stdout.strip()
            stderr = stderr.strip()
            response = _parse_worker_response(stdout, expected_request_id=request_id)
            _raise_if_area_cancelled(cancellation_token)
        _raise_if_area_cancelled(cancellation_token)
        if (
            response["ok"] is False
            and used_persistent_session
            and worker_session is not None
            and _worker_error_code(response) == "internal_error"
        ):
            worker_session.disable()
            returncode, stdout, stderr = self._run_worker_process(
                worker_command=worker_command,
                payload=payload,
                timeout_s=validated_timeout_s,
                cancellation_token=cancellation_token,
            )
            stdout = stdout.strip()
            stderr = stderr.strip()
            response = _parse_worker_response(stdout, expected_request_id=request_id)
            _raise_if_area_cancelled(cancellation_token)
        if response["ok"] is False:
            message = self._friendly_failure_message(
                _worker_error_message(response, stderr=stderr),
                worker_command=worker_command,
            )
            raise AreaInferenceError(message)
        if returncode != 0:
            raise AreaInferenceProtocolError(
                f"面积识别 worker 返回成功数据但退出码为 {returncode}。"
            )

        result_payload = response.get("result")
        if not isinstance(result_payload, dict):
            raise AreaInferenceProtocolError("面积识别响应缺少 result 对象。")
        _ensure_finite_numbers(result_payload, path="result")
        engine_meta = result_payload.get("engine_meta", {})
        if not isinstance(engine_meta, dict):
            raise AreaInferenceProtocolError("面积识别 result.engine_meta 必须是对象。")
        uses_trusted_class_mapping = engine_meta.get("class_mapping") == "trusted_metadata_v1"

        instances: list[AreaInstanceResult] = []
        raw_instances = result_payload.get("instances", [])
        if not isinstance(raw_instances, list):
            raise AreaInferenceProtocolError("面积识别 result.instances 必须是数组。")
        if len(raw_instances) > MAX_AREA_INSTANCES:
            raise AreaInferenceProtocolError(
                f"面积识别实例数量超过上限: {len(raw_instances)} > {MAX_AREA_INSTANCES}。"
            )
        for index, item in enumerate(raw_instances):
            _raise_if_area_cancelled(cancellation_token)
            if not isinstance(item, dict):
                raise AreaInferenceProtocolError(f"面积识别实例 {index} 不是对象。")
            polygon_px: list[Point] = []
            raw_polygon = item.get("polygon", [])
            if not isinstance(raw_polygon, list):
                raise AreaInferenceProtocolError(f"面积识别实例 {index} 的 polygon 不是数组。")
            for point_index, point in enumerate(raw_polygon):
                if isinstance(point, dict):
                    x = _finite_number(point.get("x"), path=f"instances[{index}].polygon[{point_index}].x")
                    y = _finite_number(point.get("y"), path=f"instances[{index}].polygon[{point_index}].y")
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    x = _finite_number(point[0], path=f"instances[{index}].polygon[{point_index}][0]")
                    y = _finite_number(point[1], path=f"instances[{index}].polygon[{point_index}][1]")
                else:
                    raise AreaInferenceProtocolError(
                        f"面积识别实例 {index} 的 polygon 点 {point_index} 格式无效。"
                    )
                polygon_px.append(Point(x=x, y=y))
            if len(polygon_px) < 3:
                continue
            score = _finite_number(item.get("score", 0.0), path=f"instances[{index}].score")
            area_px = _finite_number(item.get("area_px", 0.0), path=f"instances[{index}].area_px")
            if area_px < 0.0:
                raise AreaInferenceProtocolError(f"面积识别实例 {index} 的 area_px 不能为负数。")
            raw_bbox = item.get("bbox", [0, 0, 0, 0])
            if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
                raise AreaInferenceProtocolError(f"面积识别实例 {index} 的 bbox 格式无效。")
            bbox = [
                int(_finite_number(value, path=f"instances[{index}].bbox[{bbox_index}]"))
                for bbox_index, value in enumerate(raw_bbox[:4])
            ]
            instances.append(
                AreaInstanceResult(
                    class_name=(
                        normalize_area_label(str(item.get("class_name", "")))
                        if uses_trusted_class_mapping
                        else normalize_area_result_label(model_name, str(item.get("class_name", "")))
                    ),
                    score=score,
                    bbox=bbox,
                    polygon_px=polygon_px,
                    area_px=area_px,
                )
            )
        _raise_if_area_cancelled(cancellation_token)
        return AreaInferenceResult(
            instances=instances,
            engine_meta=dict(engine_meta),
        )

    def _run_worker_process(
        self,
        *,
        worker_command: list[str],
        payload: dict[str, object],
        timeout_s: float,
        cancellation_token: CancellationToken | None,
    ) -> tuple[int, str, str]:
        environment = dict(os.environ)
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            process = subprocess.Popen(
                worker_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                **self._subprocess_kwargs(),
            )
        except OSError as exc:
            raise AreaInferenceError(f"无法启动面积识别 worker: {exc}") from exc

        request_text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        started_at = time.monotonic()
        first_communicate = True
        while True:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                _terminate_then_kill(process)
                raise AreaInferenceCancelledError("面积识别已取消。")
            elapsed_s = time.monotonic() - started_at
            if elapsed_s >= timeout_s:
                _terminate_then_kill(process)
                raise AreaInferenceTimeoutError(f"面积识别超过 {timeout_s:g} 秒，已终止 worker。")
            wait_s = min(AREA_INFERENCE_POLL_INTERVAL_S, timeout_s - elapsed_s)
            try:
                stdout, stderr = process.communicate(
                    input=request_text if first_communicate else None,
                    timeout=wait_s,
                )
                break
            except subprocess.TimeoutExpired:
                first_communicate = False

        stdout = stdout or ""
        stderr = stderr or ""
        if len(stdout.encode("utf-8", errors="replace")) > MAX_AREA_RESPONSE_BYTES:
            raise AreaInferenceProtocolError(
                f"面积识别响应超过 {MAX_AREA_RESPONSE_BYTES // (1024 * 1024)} MiB 上限。"
            )
        return int(process.returncode or 0), stdout, stderr[-MAX_AREA_STDERR_CHARS:]


def _validate_timeout_s(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("面积识别超时必须是秒数。")
    try:
        timeout_s = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("面积识别超时必须是秒数。") from exc
    if not math.isfinite(timeout_s) or not MIN_AREA_INFERENCE_TIMEOUT_S <= timeout_s <= MAX_AREA_INFERENCE_TIMEOUT_S:
        raise ValueError(
            "面积识别超时必须位于 "
            f"{MIN_AREA_INFERENCE_TIMEOUT_S:g}-{MAX_AREA_INFERENCE_TIMEOUT_S:g} 秒。"
        )
    return timeout_s


def _terminate_then_kill(process: subprocess.Popen[str]) -> None:
    try:
        if process.poll() is None:
            process.terminate()
    except OSError:
        pass
    try:
        process.communicate(timeout=AREA_INFERENCE_TERMINATE_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.communicate()
    except OSError:
        pass


def _terminate_then_wait(process: subprocess.Popen[str]) -> None:
    try:
        if process.poll() is not None:
            return
        process.terminate()
    except OSError:
        return
    try:
        process.wait(timeout=AREA_INFERENCE_TERMINATE_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
        process.wait(timeout=AREA_INFERENCE_TERMINATE_GRACE_S)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _read_worker_stdout(stream, output: Queue[str | None]) -> None:
    try:
        for line in stream:
            output.put(line)
    except (OSError, ValueError):
        pass
    finally:
        output.put(None)


def _parse_worker_response(stdout: str, *, expected_request_id: str) -> dict[str, object]:
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AreaInferenceProtocolError(f"面积识别返回了无法解析的数据: {stdout[:300]}") from exc
    if not isinstance(response, dict):
        raise AreaInferenceProtocolError("面积识别响应必须是 JSON 对象。")
    if response.get("protocol") != AREA_WORKER_PROTOCOL:
        raise AreaInferenceProtocolError("面积识别响应协议标识无效。")
    version = response.get("version")
    if isinstance(version, bool) or version != AREA_WORKER_PROTOCOL_VERSION:
        raise AreaInferenceProtocolError(f"不支持的面积识别响应协议版本: {version!r}。")
    if response.get("request_id") != expected_request_id:
        raise AreaInferenceProtocolError("面积识别响应 request_id 与请求不匹配。")
    if not isinstance(response.get("ok"), bool):
        raise AreaInferenceProtocolError("面积识别响应缺少布尔类型的 ok 字段。")
    if response["ok"] is True and not isinstance(response.get("result"), dict):
        raise AreaInferenceProtocolError("面积识别成功响应缺少 result 对象。")
    if response["ok"] is False and not isinstance(response.get("error"), dict):
        raise AreaInferenceProtocolError("面积识别失败响应缺少 error 对象。")
    return response


def _worker_error_message(response: dict[str, object], *, stderr: str) -> str:
    error = response.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        code = str(error.get("code") or "").strip()
        if message:
            return message
        if code:
            return code
    return stderr[-MAX_AREA_STDERR_CHARS:] or "面积识别失败"


def _worker_error_code(response: dict[str, object]) -> str:
    error = response.get("error")
    if not isinstance(error, dict):
        return ""
    return str(error.get("code") or "").strip()


def _ensure_finite_numbers(value: object, *, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise AreaInferenceProtocolError(f"面积识别响应 {path} 包含非有限数值。")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_finite_numbers(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _ensure_finite_numbers(item, path=f"{path}.{key}")


def _finite_number(value: object, *, path: str) -> float:
    if isinstance(value, bool):
        raise AreaInferenceProtocolError(f"面积识别响应 {path} 必须是有限数值。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AreaInferenceProtocolError(f"面积识别响应 {path} 必须是有限数值。") from exc
    if not math.isfinite(number):
        raise AreaInferenceProtocolError(f"面积识别响应 {path} 必须是有限数值。")
    return number
