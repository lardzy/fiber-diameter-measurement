from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fdm.cancellation import CancellationError, CancellationToken, CancellationTokenSource
from fdm.image_processing_models import ImageOperationSpec, ImageProcessingRecipe
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.image_processing import (
    ImageOperation,
    execute_image_operation_tiled,
    resolve_image_operation_capability,
)
from fdm.ui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox


class WorkbenchTaskKind(str, Enum):
    PREVIEW = "preview"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class WorkbenchTaskRequest:
    kind: WorkbenchTaskKind
    request_id: str
    generation: int
    source_document_id: str
    source: RasterPlane
    operations: tuple[ImageOperationSpec, ...]
    roi_mask: NDArray[np.bool_] | None = field(default=None, compare=False, repr=False)
    secondary_images: tuple[tuple[str, RasterPlane], ...] = field(
        default=(),
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.operations:
            raise ValueError("图像处理任务至少需要一个步骤")
        mask = self.roi_mask
        if mask is not None:
            normalized = np.ascontiguousarray(mask, dtype=np.bool_)
            if normalized.shape != (self.source.height, self.source.width):
                raise ValueError("ROI 掩膜尺寸必须与源图片一致")
            normalized.setflags(write=False)
            object.__setattr__(self, "roi_mask", normalized)
        secondary_images: list[tuple[str, RasterPlane]] = []
        seen_secondary_ids: set[str] = set()
        for raw_id, plane in self.secondary_images:
            document_id = str(raw_id).strip()
            if not document_id or document_id in seen_secondary_ids:
                raise ValueError("第二幅图像的文档 ID 不能为空或重复")
            if not isinstance(plane, RasterPlane):
                raise TypeError("第二幅图像必须是 RasterPlane")
            seen_secondary_ids.add(document_id)
            secondary_images.append((document_id, plane))
        object.__setattr__(self, "secondary_images", tuple(secondary_images))

    @property
    def recipe(self) -> ImageProcessingRecipe:
        return ImageProcessingRecipe.from_operations(self.operations)


@dataclass(frozen=True, slots=True)
class WorkbenchTaskResult:
    kind: WorkbenchTaskKind
    request_id: str
    generation: int
    source_document_id: str
    raster: RasterPlane
    recipe: ImageProcessingRecipe


@dataclass(frozen=True, slots=True)
class _TaskCompletion:
    request: WorkbenchTaskRequest
    raster: RasterPlane | None = None
    recipe: ImageProcessingRecipe | None = None
    error: str | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class _WorkbenchExecutionOutput:
    raster: RasterPlane
    recipe: ImageProcessingRecipe


ImageTaskExecutor = Callable[
    [WorkbenchTaskRequest, CancellationToken],
    RasterPlane | _WorkbenchExecutionOutput,
]


class _TaskSignals(QObject):
    completed = Signal(object)


class _ImageTaskRunnable(QRunnable):
    def __init__(
        self,
        *,
        request: WorkbenchTaskRequest,
        token: CancellationToken,
        executor: ImageTaskExecutor,
        signals: _TaskSignals,
    ) -> None:
        super().__init__()
        self._request = request
        self._token = token
        self._executor = executor
        self._signals = signals

    @Slot()
    def run(self) -> None:
        try:
            self._token.raise_if_cancelled()
            execution = self._executor(self._request, self._token)
            self._token.raise_if_cancelled()
            if isinstance(execution, _WorkbenchExecutionOutput):
                completion = _TaskCompletion(
                    request=self._request,
                    raster=execution.raster,
                    recipe=execution.recipe,
                )
            elif isinstance(execution, RasterPlane):
                completion = _TaskCompletion(
                    request=self._request,
                    raster=execution,
                )
            else:
                raise TypeError("图像处理执行器必须返回 RasterPlane。")
        except CancellationError:
            completion = _TaskCompletion(request=self._request, cancelled=True)
        except Exception as exc:
            if self._token.is_cancelled:
                completion = _TaskCompletion(request=self._request, cancelled=True)
            else:
                message = str(exc).strip() or type(exc).__name__
                completion = _TaskCompletion(request=self._request, error=message)
        self._signals.completed.emit(completion)


@dataclass(slots=True)
class _TaskLane:
    pool: QThreadPool
    generation: int = 0
    active: WorkbenchTaskRequest | None = None
    active_cancellation: CancellationTokenSource | None = None
    pending: WorkbenchTaskRequest | None = None
    busy: bool = False


class ImageProcessingTaskController(QObject):
    """Two bounded task lanes with cooperative cancellation and stale-result rejection."""

    previewReady = Signal(object)
    finalReady = Signal(object)
    taskFailed = Signal(str, str)
    busyChanged = Signal(str, bool)
    staleResultDiscarded = Signal(str, int)

    def __init__(
        self,
        *,
        executor: ImageTaskExecutor | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._executor = executor or _execute_workbench_request_with_metadata
        self._signals = _TaskSignals(self)
        self._signals.completed.connect(self._on_completed)
        self._lanes = {
            WorkbenchTaskKind.PREVIEW: _TaskLane(_single_thread_pool(self)),
            WorkbenchTaskKind.FINAL: _TaskLane(_single_thread_pool(self)),
        }
        self._closed = False

    def start_preview(
        self,
        *,
        source_document_id: str,
        source: RasterPlane,
        operations: tuple[ImageOperationSpec, ...],
        roi_mask: NDArray[np.bool_] | None = None,
        secondary_images: Mapping[str, RasterPlane] | None = None,
    ) -> WorkbenchTaskRequest:
        return self._start(
            WorkbenchTaskKind.PREVIEW,
            source_document_id=source_document_id,
            source=source,
            operations=operations,
            roi_mask=roi_mask,
            secondary_images=secondary_images,
        )

    def start_final(
        self,
        *,
        source_document_id: str,
        source: RasterPlane,
        operations: tuple[ImageOperationSpec, ...],
        roi_mask: NDArray[np.bool_] | None = None,
        secondary_images: Mapping[str, RasterPlane] | None = None,
    ) -> WorkbenchTaskRequest:
        return self._start(
            WorkbenchTaskKind.FINAL,
            source_document_id=source_document_id,
            source=source,
            operations=operations,
            roi_mask=roi_mask,
            secondary_images=secondary_images,
        )

    def cancel_preview(self) -> None:
        self._cancel_lane(WorkbenchTaskKind.PREVIEW)

    def cancel_final(self) -> None:
        self._cancel_lane(WorkbenchTaskKind.FINAL)

    def cancel_all(self) -> None:
        self.cancel_preview()
        self.cancel_final()

    def close(self) -> None:
        self._closed = True
        self.cancel_all()

    def is_busy(self, kind: WorkbenchTaskKind | str) -> bool:
        lane = self._lanes[WorkbenchTaskKind(kind)]
        return lane.active is not None or lane.pending is not None

    def generation(self, kind: WorkbenchTaskKind | str) -> int:
        return self._lanes[WorkbenchTaskKind(kind)].generation

    def wait_for_done(self, timeout_ms: int = 5_000) -> bool:
        return all(lane.pool.waitForDone(timeout_ms) for lane in self._lanes.values())

    def _start(
        self,
        kind: WorkbenchTaskKind,
        *,
        source_document_id: str,
        source: RasterPlane,
        operations: tuple[ImageOperationSpec, ...],
        roi_mask: NDArray[np.bool_] | None,
        secondary_images: Mapping[str, RasterPlane] | None,
    ) -> WorkbenchTaskRequest:
        if self._closed:
            raise RuntimeError("图像处理任务控制器已经关闭")
        validate_workbench_operation_sequence(source, operations)
        lane = self._lanes[kind]
        lane.generation += 1
        request = WorkbenchTaskRequest(
            kind=kind,
            request_id=uuid4().hex,
            generation=lane.generation,
            source_document_id=str(source_document_id),
            source=source,
            operations=tuple(operations),
            roi_mask=roi_mask,
            secondary_images=tuple((secondary_images or {}).items()),
        )
        if lane.active is not None:
            if lane.active_cancellation is not None:
                lane.active_cancellation.cancel()
            lane.pending = request
        else:
            self._launch(lane, request)
        return request

    def _launch(self, lane: _TaskLane, request: WorkbenchTaskRequest) -> None:
        cancellation = CancellationTokenSource()
        lane.active = request
        lane.active_cancellation = cancellation
        self._set_lane_busy(request.kind, lane, True)
        lane.pool.start(
            _ImageTaskRunnable(
                request=request,
                token=cancellation.token,
                executor=self._executor,
                signals=self._signals,
            )
        )

    def _cancel_lane(self, kind: WorkbenchTaskKind) -> None:
        lane = self._lanes[kind]
        lane.generation += 1
        lane.pending = None
        if lane.active_cancellation is not None:
            lane.active_cancellation.cancel()
        if lane.active is None:
            self._set_lane_busy(kind, lane, False)

    def _set_lane_busy(
        self,
        kind: WorkbenchTaskKind,
        lane: _TaskLane,
        busy: bool,
    ) -> None:
        """Emit busy state only when the externally visible state changes."""

        resolved = bool(busy)
        if lane.busy == resolved:
            return
        lane.busy = resolved
        self.busyChanged.emit(kind.value, resolved)

    @Slot(object)
    def _on_completed(self, completion: object) -> None:
        if not isinstance(completion, _TaskCompletion):
            return
        request = completion.request
        lane = self._lanes[request.kind]
        if lane.active is None or lane.active.request_id != request.request_id:
            self.staleResultDiscarded.emit(request.request_id, request.generation)
            return

        lane.active = None
        lane.active_cancellation = None
        is_current = (
            not self._closed
            and request.generation == lane.generation
            and lane.pending is None
        )
        if is_current and not completion.cancelled:
            if completion.error is not None:
                self.taskFailed.emit(request.kind.value, completion.error)
            elif completion.raster is not None:
                result = WorkbenchTaskResult(
                    kind=request.kind,
                    request_id=request.request_id,
                    generation=request.generation,
                    source_document_id=request.source_document_id,
                    raster=completion.raster,
                    recipe=completion.recipe or request.recipe,
                )
                if request.kind is WorkbenchTaskKind.PREVIEW:
                    self.previewReady.emit(result)
                else:
                    self.finalReady.emit(result)
        elif not completion.cancelled:
            self.staleResultDiscarded.emit(request.request_id, request.generation)

        pending = lane.pending
        lane.pending = None
        if pending is not None and not self._closed:
            self._launch(lane, pending)
        else:
            self._set_lane_busy(request.kind, lane, False)


def _single_thread_pool(parent: QObject) -> QThreadPool:
    pool = QThreadPool(parent)
    pool.setMaxThreadCount(1)
    pool.setExpiryTimeout(5_000)
    return pool


def execute_workbench_request(
    request: WorkbenchTaskRequest,
    token: CancellationToken,
) -> RasterPlane:
    """Execute a request and return pixels for legacy direct callers."""

    return _execute_workbench_request_with_metadata(request, token).raster


def _execute_workbench_request_with_metadata(
    request: WorkbenchTaskRequest,
    token: CancellationToken,
) -> _WorkbenchExecutionOutput:
    validate_workbench_operation_sequence(request.source, request.operations)
    image = raster_plane_to_array(request.source)
    secondary_images = dict(request.secondary_images)
    executed_operations: list[ImageOperationSpec] = []
    dynamic_metadata_keys = {
        "nonfinite_replacement_count",
        "repaired_count",
        "computed_threshold",
        "cropped_right",
        "cropped_bottom",
    }
    for operation_spec in request.operations:
        token.raise_if_cancelled()
        parameters = operation_spec.parameters
        secondary_image = None
        if operation_spec.operation_id == ImageOperation.IMAGE_CALCULATOR.value:
            secondary_document_id = str(
                parameters.pop("secondary_document_id", "")
            )
            try:
                secondary_plane = secondary_images[secondary_document_id]
            except KeyError as exc:
                raise ValueError("图像计算器选择的第二幅图像已不可用") from exc
            secondary_image = raster_plane_to_array(secondary_plane)
        result = execute_image_operation_tiled(
            ImageOperation(operation_spec.operation_id),
            image,
            parameters=parameters,
            secondary_image=secondary_image,
            roi_mask=request.roi_mask,
            request_id=request.request_id,
            generation=request.generation,
            tile_size=PROCESSING_TILE_EDGE,
            cancellation_check=token.raise_if_cancelled,
        )
        token.raise_if_cancelled()
        image = np.asarray(result.image)
        dynamic_metadata = {
            key: value
            for key, value in result.metadata_map.items()
            if key in dynamic_metadata_keys
        }
        merged_metadata = operation_spec.result_metadata
        merged_metadata.update(dynamic_metadata)
        executed_operations.append(
            ImageOperationSpec(
                operation_spec.operation_id,
                parameters,
                implementation=operation_spec.implementation,
                implementation_version=operation_spec.implementation_version,
                result_metadata=merged_metadata,
            )
        )
    token.raise_if_cancelled()
    return _WorkbenchExecutionOutput(
        raster=array_to_raster_plane(image),
        recipe=ImageProcessingRecipe.from_operations(executed_operations),
    )


def validate_workbench_operation_sequence(
    source: RasterPlane,
    operations: tuple[ImageOperationSpec, ...],
) -> None:
    """Reject pixel-layout combinations that the authoritative model cannot store."""

    channels = int(source.pixel_type.channel_count)
    scalar_outputs = {
        ImageOperation.THRESHOLD.value,
        ImageOperation.SOBEL_EDGES.value,
        ImageOperation.LAPLACIAN_EDGES.value,
        ImageOperation.CANNY_EDGES.value,
        ImageOperation.AUTO_THRESHOLD.value,
        ImageOperation.BINARIZE.value,
        ImageOperation.ERODE.value,
        ImageOperation.DILATE.value,
        ImageOperation.MORPHOLOGY_OPEN.value,
        ImageOperation.MORPHOLOGY_CLOSE.value,
        ImageOperation.FILL_HOLES.value,
        ImageOperation.CONTOUR_EXTRACT.value,
        ImageOperation.REMOVE_SMALL_OBJECTS.value,
        ImageOperation.FILL_SMALL_HOLES.value,
        ImageOperation.DISTANCE_TRANSFORM.value,
        ImageOperation.SKELETONIZE.value,
        ImageOperation.WATERSHED.value,
        ImageOperation.TOP_HAT.value,
        ImageOperation.BLACK_HAT.value,
    }
    for operation in operations:
        parameters = operation.parameters
        if operation.operation_id == ImageOperation.CONVERT_TYPE.value:
            target = str(parameters.get("target_type", "uint8"))
            if channels > 1 and target != "uint8":
                raise ValueError(
                    "RGB/RGBA 图像不能直接转换为 16 位或 32 位浮点；"
                    "请先添加“转换颜色模型 → 灰度”步骤。"
                )
        elif operation.operation_id == ImageOperation.CONVERT_COLOR.value:
            channels = (
                1
                if str(parameters.get("target_model", "grayscale"))
                == "grayscale"
                else 3
            )
        elif operation.operation_id in scalar_outputs:
            channels = 1
        elif (
            operation.operation_id == ImageOperation.FFT_FILTER.value
            and str(parameters.get("channel", "per_channel"))
            != "per_channel"
        ):
            channels = 1


def raster_plane_to_array(plane: RasterPlane) -> NDArray[np.generic]:
    dtype: np.dtype[object]
    channels = 1
    if plane.pixel_type is RasterPixelType.GRAY8:
        dtype = np.dtype(np.uint8)
    elif plane.pixel_type is RasterPixelType.GRAY16:
        dtype = np.dtype("<u2")
    elif plane.pixel_type is RasterPixelType.GRAY32_FLOAT:
        dtype = np.dtype("<f4")
    elif plane.pixel_type is RasterPixelType.RGB8:
        dtype = np.dtype(np.uint8)
        channels = 3
    elif plane.pixel_type is RasterPixelType.RGBA8:
        dtype = np.dtype(np.uint8)
        channels = 4
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"不支持的栅格类型: {plane.pixel_type}")
    shape = (
        (plane.height, plane.width)
        if channels == 1
        else (plane.height, plane.width, channels)
    )
    array = np.frombuffer(plane.data, dtype=dtype).reshape(shape)
    array.setflags(write=False)
    return array


def array_to_raster_plane(image: NDArray[np.generic]) -> RasterPlane:
    array = np.ascontiguousarray(image)
    if array.dtype == np.dtype(np.uint8):
        if array.ndim == 2:
            pixel_type = RasterPixelType.GRAY8
        elif array.ndim == 3 and array.shape[2] == 3:
            pixel_type = RasterPixelType.RGB8
        elif array.ndim == 3 and array.shape[2] == 4:
            pixel_type = RasterPixelType.RGBA8
        else:
            raise ValueError("8 位处理结果必须是灰度、RGB 或 RGBA 图像")
    elif array.dtype == np.dtype(np.uint16):
        if array.ndim != 2:
            raise ValueError("16 位处理结果仅支持单通道灰度")
        pixel_type = RasterPixelType.GRAY16
        array = np.ascontiguousarray(array.astype("<u2", copy=False))
    elif array.dtype == np.dtype(np.float32):
        if array.ndim != 2:
            raise ValueError("32 位浮点处理结果仅支持单通道灰度")
        pixel_type = RasterPixelType.GRAY32_FLOAT
        array = np.ascontiguousarray(array.astype("<f4", copy=False))
    else:
        raise ValueError(f"不支持的处理结果 dtype: {array.dtype}")
    return RasterPlane(
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        pixel_type=pixel_type,
        data=array.tobytes(order="C"),
    )


PROCESSING_TILE_EDGE = 1024
MAX_FINAL_WORKING_SET_BYTES = 1 << 30
MIN_FREE_DISK_RESERVE_BYTES = 2 << 30


@dataclass(frozen=True, slots=True)
class FinalResourceEstimate:
    peak_working_set_bytes: int
    output_bytes: int
    output_width: int
    output_height: int


class FinalResourcePreflightError(RuntimeError):
    """Raised before a final task when its declared resource budget is unsafe."""


def estimate_final_resources(
    source: RasterPlane,
    operations: tuple[ImageOperationSpec, ...],
) -> FinalResourceEstimate:
    """Conservatively estimate peak RAM and uncompressed result bytes.

    The estimator intentionally uses operation families instead of compressed
    file-size guesses.  This keeps the preflight deterministic and prevents a
    large FFT, watershed, rotation, or type conversion from first discovering
    its memory requirement after allocating the full-size destination.
    """

    width = max(0, int(source.width))
    height = max(0, int(source.height))
    channels = int(source.pixel_type.channel_count)
    bytes_per_channel = int(source.pixel_type.bytes_per_channel)
    current_bytes = width * height * channels * bytes_per_channel
    peak_bytes = max(source.byte_count + current_bytes, 1)

    scalar_outputs = {
        "threshold",
        "sobel_edges",
        "laplacian_edges",
        "canny_edges",
        "auto_threshold",
        "binarize",
        "erode",
        "dilate",
        "morphology_open",
        "morphology_close",
        "fill_holes",
        "contour_extract",
        "remove_small_objects",
        "fill_small_holes",
        "distance_transform",
        "skeletonize",
        "watershed",
        "top_hat",
        "black_hat",
    }
    float_outputs = {"distance_transform"}
    optionally_float_outputs = {"sobel_edges", "laplacian_edges", "fft_filter"}
    frequency_operations = {"fft_filter", "stripe_suppression"}
    global_label_operations = {
        "remove_small_objects",
        "fill_small_holes",
        "watershed",
    }

    for spec in operations:
        operation_id = spec.operation_id
        parameters = spec.parameters
        previous_bytes = current_bytes

        if operation_id == "convert_type":
            target = str(parameters.get("target_type", "uint8"))
            bytes_per_channel = {
                "uint8": 1,
                "uint16": 2,
                "float32": 4,
            }.get(target, bytes_per_channel)
        elif operation_id == "convert_color":
            target_model = str(parameters.get("target_model", "grayscale"))
            channels = 1 if target_model == "grayscale" else 3
        elif operation_id == "image_calculator":
            peak_bytes = max(
                peak_bytes,
                previous_bytes * 4 + width * height * max(channels, 1) * 16,
            )
        elif operation_id == "crop":
            x = max(0, int(parameters.get("x", 0)))
            y = max(0, int(parameters.get("y", 0)))
            width = max(1, min(int(parameters.get("width", width)), max(1, width - x)))
            height = max(
                1,
                min(int(parameters.get("height", height)), max(1, height - y)),
            )
        elif operation_id in {"resize", "resize_canvas"}:
            width = max(1, int(parameters.get("width", width)))
            height = max(1, int(parameters.get("height", height)))
        elif operation_id == "pixel_bin":
            factor = max(1, int(parameters.get("factor", 1)))
            width = max(1, width // factor)
            height = max(1, height // factor)
            if str(parameters.get("method", "mean")) == "sum" and channels == 1:
                bytes_per_channel = 4
        elif operation_id == "rotate" and bool(parameters.get("expand", True)):
            radians = math.radians(float(parameters.get("angle_degrees", 0.0)))
            cosine = abs(math.cos(radians))
            sine = abs(math.sin(radians))
            width, height = (
                max(1, int(math.ceil(width * cosine + height * sine))),
                max(1, int(math.ceil(width * sine + height * cosine))),
            )
        elif operation_id in {
            "rotate_90_clockwise",
            "rotate_90_counterclockwise",
        }:
            width, height = height, width

        if operation_id in scalar_outputs:
            channels = 1
        if operation_id in float_outputs or (
            operation_id in optionally_float_outputs
            and bool(parameters.get("output_float", True))
        ):
            bytes_per_channel = 4
        elif operation_id == "canny_edges":
            bytes_per_channel = 1

        current_bytes = width * height * channels * bytes_per_channel
        pixel_samples = width * height * max(channels, 1)
        if operation_id in frequency_operations:
            # Source float, complex spectrum, filter response, product, inverse,
            # destination, plus the immutable input/output snapshots.
            operation_peak = previous_bytes + current_bytes + pixel_samples * 64
        elif operation_id in global_label_operations:
            # Binary masks, int32 labels/markers, statistics and destinations.
            operation_peak = previous_bytes + current_bytes + width * height * 40
        elif operation_id in {"distance_transform", "skeletonize"}:
            operation_peak = previous_bytes + current_bytes + width * height * 24
        elif operation_id in {
            "rotate",
            "resize",
            "resize_canvas",
            "pixel_bin",
            "background_subtract",
            "custom_convolution",
        }:
            operation_peak = previous_bytes + current_bytes + pixel_samples * 24
        else:
            operation_peak = previous_bytes + current_bytes + pixel_samples * 16
        peak_bytes = max(peak_bytes, operation_peak)

    return FinalResourceEstimate(
        peak_working_set_bytes=int(peak_bytes),
        output_bytes=int(current_bytes),
        output_width=int(width),
        output_height=int(height),
    )


def validate_final_resources(
    source: RasterPlane,
    operations: tuple[ImageOperationSpec, ...],
    *,
    storage_directory: Path | str | None = None,
) -> FinalResourceEstimate:
    estimate = estimate_final_resources(source, operations)
    if estimate.peak_working_set_bytes > MAX_FINAL_WORKING_SET_BYTES:
        gib = estimate.peak_working_set_bytes / float(1 << 30)
        raise FinalResourcePreflightError(
            f"预计处理工作集约 {gib:.2f} GiB，超过 1 GiB 安全上限。"
            "请缩小图片或拆分处理范围。"
        )
    check_path = Path(storage_directory or tempfile.gettempdir()).expanduser()
    try:
        check_path = check_path.resolve()
        while not check_path.exists() and check_path.parent != check_path:
            check_path = check_path.parent
        free_bytes = int(shutil.disk_usage(check_path).free)
    except OSError as exc:
        raise FinalResourcePreflightError(
            f"无法检查派生图片存储空间：{exc}"
        ) from exc
    required = estimate.output_bytes + MIN_FREE_DISK_RESERVE_BYTES
    if free_bytes < required:
        free_gib = free_bytes / float(1 << 30)
        output_gib = estimate.output_bytes / float(1 << 30)
        raise FinalResourcePreflightError(
            f"可用磁盘空间约 {free_gib:.2f} GiB，预计输出需 "
            f"{output_gib:.2f} GiB，无法在完成后保留至少 2 GiB 空间。"
        )
    return estimate


@dataclass(frozen=True, slots=True)
class ParameterField:
    key: str
    label: str
    kind: str
    default: object
    minimum: float | None = None
    maximum: float | None = None
    decimals: int = 2
    choices: tuple[tuple[str, object], ...] = ()
    suffix: str = ""
    help_text: str = ""


@dataclass(frozen=True, slots=True)
class WorkbenchOperationDefinition:
    operation: ImageOperation
    category: str
    label: str
    parameters: tuple[ParameterField, ...] = ()
    purpose: str = ""
    pixel_effect: str = "生成新的像素结果，不覆盖源图片。"
    calibration_effect: str = "保持现有标定。"
    supported_types: str = "GRAY8、GRAY16、GRAY32_FLOAT、RGB8、RGBA8"
    roi_behavior: str = "支持 ROI；ROI 外像素保持不变。"


def _operation_catalog() -> tuple[WorkbenchOperationDefinition, ...]:
    """Return every executable single-source Image/Process operation."""

    integer_types = (
        ("8 位灰度", "uint8"),
        ("16 位灰度", "uint16"),
        ("32 位浮点", "float32"),
    )
    scale_modes = (
        ("保留数值", "preserve_values"),
        ("映射完整类型范围", "full_type_range"),
        ("映射当前数据范围", "data_range"),
    )
    border_modes = (
        ("镜像 Reflect101", "reflect"),
        ("复制边缘", "replicate"),
        ("常量", "constant"),
        ("循环", "wrap"),
    )
    interpolation_modes = (
        ("最近邻", "nearest"),
        ("双线性", "linear"),
        ("双三次", "cubic"),
        ("区域平均", "area"),
        ("Lanczos", "lanczos"),
    )
    scalar_channels = (
        ("加权亮度", "luminance"),
        ("红色通道", "red"),
        ("绿色通道", "green"),
        ("蓝色通道", "blue"),
    )
    fft_channels = (("逐颜色通道", "per_channel"),) + scalar_channels
    connectivity_modes = (("4 邻域", 4), ("8 邻域", 8))
    kernel_modes = (
        ("椭圆", "ellipse"),
        ("矩形", "rectangle"),
        ("十字", "cross"),
    )
    foreground = ParameterField(
        "foreground_is_high",
        "亮色为前景",
        "bool",
        True,
        help_text="关闭后按暗色前景解释二值图。",
    )
    scalar_channel = ParameterField(
        "channel",
        "彩色图通道",
        "choice",
        "luminance",
        choices=scalar_channels,
        help_text="灰度图忽略此项；彩色图必须明确选取一个标量通道。",
    )
    border = ParameterField(
        "border_mode",
        "边界",
        "choice",
        "reflect",
        choices=border_modes,
        help_text="邻域越过图片边缘时的取样规则；默认 Reflect101。",
    )
    local_border = ParameterField(
        "border_mode",
        "边界",
        "choice",
        "reflect",
        choices=border_modes[:-1],
        help_text=(
            "邻域越过图片边缘时的取样规则；默认 Reflect101。"
            "该操作的 OpenCV 后端不支持循环边界。"
        ),
    )
    radius = ParameterField(
        "radius",
        "半径",
        "int",
        1,
        1,
        99,
        help_text="以原始图片像素为单位。",
    )
    iterations = ParameterField("iterations", "迭代次数", "int", 1, 1, 100)
    kernel = ParameterField(
        "kernel",
        "结构元素",
        "choice",
        "ellipse",
        choices=kernel_modes,
    )
    scalar_types = "GRAY8、GRAY16、GRAY32_FLOAT；彩色图需选择一个通道"
    no_roi = "几何变换不接受 ROI 掩膜，必须作用于完整图像。"

    def define(
        value: ImageOperation,
        category: str,
        label: str,
        parameters: tuple[ParameterField, ...] = (),
        *,
        purpose: str,
        pixel_effect: str = "生成新的像素结果，不覆盖源图片。",
        calibration_effect: str = "保持现有标定。",
        supported_types: str = (
            "GRAY8、GRAY16、GRAY32_FLOAT、RGB8、RGBA8"
        ),
        roi_behavior: str = "支持 ROI；ROI 外像素保持不变。",
    ) -> WorkbenchOperationDefinition:
        return WorkbenchOperationDefinition(
            operation=value,
            category=category,
            label=label,
            parameters=parameters,
            purpose=purpose,
            pixel_effect=pixel_effect,
            calibration_effect=calibration_effect,
            supported_types=supported_types,
            roi_behavior=roi_behavior,
        )

    morphology = tuple(
        define(
            value,
            "处理",
            label,
            (radius, iterations, kernel, local_border, scalar_channel),
            purpose=purpose,
            supported_types=scalar_types,
        )
        for value, label, purpose in (
            (ImageOperation.ERODE, "腐蚀", "缩小二值或灰度前景。"),
            (ImageOperation.DILATE, "膨胀", "扩大二值或灰度前景。"),
            (ImageOperation.MORPHOLOGY_OPEN, "开运算", "去除小亮结构并平滑边界。"),
            (ImageOperation.MORPHOLOGY_CLOSE, "闭运算", "闭合小间隙并填补窄裂缝。"),
            (ImageOperation.TOP_HAT, "顶帽", "提取比结构元素更小的亮结构。"),
            (ImageOperation.BLACK_HAT, "黑帽", "提取比结构元素更小的暗结构。"),
        )
    )

    return (
        define(
            ImageOperation.CONVERT_TYPE,
            "类型",
            "转换像素类型",
            (
                ParameterField(
                    "target_type",
                    "目标类型",
                    "choice",
                    "uint8",
                    choices=integer_types,
                    help_text="16 位和 32 位结果当前仅支持单通道灰度。",
                ),
                ParameterField(
                    "scale_mode",
                    "数值映射",
                    "choice",
                    "full_type_range",
                    choices=scale_modes,
                ),
                ParameterField(
                    "nonfinite_policy",
                    "NaN/Inf 转整数",
                    "choice",
                    "reject",
                    choices=(
                        ("拒绝并提示", "reject"),
                        ("替代为零", "zero"),
                        ("按输出范围边界替代", "range_bounds"),
                    ),
                    help_text=(
                        "仅在 32 位浮点转 8/16 位整数时使用；"
                        "默认拒绝，替代数量会写入派生记录。"
                    ),
                ),
            ),
            purpose="显式转换位深，并记录数值映射规则。",
            pixel_effect="改变像素位深，可能改变样本数值；不覆盖源图片。",
            supported_types="GRAY8、GRAY16、GRAY32_FLOAT；彩色图仅可输出 8 位",
        ),
        define(
            ImageOperation.CONVERT_COLOR,
            "类型",
            "转换颜色模型",
            (
                ParameterField(
                    "target_model",
                    "目标模型",
                    "choice",
                    "grayscale",
                    choices=(("灰度", "grayscale"), ("RGB 彩色", "rgb")),
                ),
                ParameterField(
                    "grayscale_method",
                    "灰度换算",
                    "choice",
                    "rec601",
                    choices=(("Rec.601 加权", "rec601"), ("通道平均", "average")),
                ),
                ParameterField(
                    "drop_alpha",
                    "移除 Alpha",
                    "bool",
                    False,
                    help_text="仅在明确允许丢弃透明度时开启。",
                ),
            ),
            purpose="在灰度、RGB 和 RGBA 表示之间显式转换。",
            pixel_effect="改变像素通道模型；不覆盖源图片。",
            supported_types="GRAY8、RGB8、RGBA8",
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.COLOR_BALANCE,
            "调整",
            "色彩平衡",
            (
                ParameterField("red_gain", "红色增益", "float", 1.0, 0, 100, 4),
                ParameterField("green_gain", "绿色增益", "float", 1.0, 0, 100, 4),
                ParameterField("blue_gain", "蓝色增益", "float", 1.0, 0, 100, 4),
                ParameterField("red_offset", "红色偏移", "float", 0.0, -65535, 65535, 3),
                ParameterField("green_offset", "绿色偏移", "float", 0.0, -65535, 65535, 3),
                ParameterField("blue_offset", "蓝色偏移", "float", 0.0, -65535, 65535, 3),
            ),
            purpose="分别调整 RGB 三个颜色通道，Alpha 保持不变。",
            supported_types="仅 RGB8、RGBA8",
        ),
        define(
            ImageOperation.BRIGHTNESS_CONTRAST,
            "调整",
            "亮度 / 对比度",
            (
                ParameterField("brightness", "亮度偏移", "float", 0.0, -65535, 65535),
                ParameterField("contrast", "对比度", "float", 1.0, 0, 20, 3),
                ParameterField("gamma", "Gamma", "float", 1.0, 0.01, 20, 3),
            ),
            purpose="将亮度、对比度和 Gamma 烘焙到派生像素。",
        ),
        define(
            ImageOperation.ADJUST_LEVELS,
            "调整",
            "窗口 / 色阶",
            (
                ParameterField("black_point", "黑场", "float", 0.0, -1e12, 1e12, 4),
                ParameterField("white_point", "白场", "float", "$working_max", -1e12, 1e12, 4),
                ParameterField("gamma", "Gamma", "float", 1.0, 0.01, 20, 3),
            ),
            purpose="按黑场、白场和 Gamma 映射输入动态范围。",
        ),
        define(
            ImageOperation.THRESHOLD,
            "调整",
            "阈值",
            (
                ParameterField("lower", "下限", "float", 0.0, -1e12, 1e12, 4),
                ParameterField("upper", "上限", "float", "$working_max", -1e12, 1e12, 4),
                ParameterField("invert", "反相", "bool", False),
                scalar_channel,
            ),
            purpose="将指定强度区间转换为二值结果。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.FLIP_HORIZONTAL,
            "变换",
            "水平翻转",
            purpose="沿垂直轴镜像完整图像。",
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.FLIP_VERTICAL,
            "变换",
            "垂直翻转",
            purpose="沿水平轴镜像完整图像。",
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.ROTATE_90_COUNTERCLOCKWISE,
            "变换",
            "左转 90°",
            purpose="将完整图像逆时针旋转 90°。",
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.ROTATE_90_CLOCKWISE,
            "变换",
            "右转 90°",
            purpose="将完整图像顺时针旋转 90°。",
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.ROTATE_180,
            "变换",
            "旋转 180°",
            purpose="将完整图像旋转 180°。",
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.ROTATE,
            "变换",
            "任意角度旋转",
            (
                ParameterField("angle_degrees", "角度", "float", 0.0, -360, 360, 2, suffix="°"),
                ParameterField("expand", "扩展画布", "bool", True),
                ParameterField("interpolation", "插值", "choice", "linear", choices=interpolation_modes),
                ParameterField("border_mode", "边界", "choice", "constant", choices=border_modes),
                ParameterField("border_value", "常量边界值", "float", 0.0, -1e12, 1e12, 3),
            ),
            purpose="按指定角度旋转完整图像。",
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.CROP,
            "变换",
            "裁剪",
            (
                ParameterField("x", "左", "int", 0, 0, 1_000_000),
                ParameterField("y", "上", "int", 0, 0, 1_000_000),
                ParameterField("width", "宽度", "int", -1, 1, 1_000_000),
                ParameterField("height", "高度", "int", -1, 1, 1_000_000),
            ),
            purpose="截取原始像素坐标中的矩形区域。",
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.RESIZE,
            "变换",
            "调整图像大小",
            (
                ParameterField("width", "宽度", "int", -1, 1, 1_000_000),
                ParameterField("height", "高度", "int", -1, 1, 1_000_000),
                ParameterField("interpolation", "插值", "choice", "area", choices=interpolation_modes),
            ),
            purpose="重采样为指定宽高。",
            calibration_effect=(
                "等比例缩放时同步调整 pixels_per_unit；"
                "已标定图像的非等比例缩放必须先明确清除标定。"
            ),
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.TRANSLATE,
            "变换",
            "平移",
            (
                ParameterField("offset_x", "水平偏移", "float", 0.0, -1_000_000, 1_000_000, 2, suffix=" px"),
                ParameterField("offset_y", "垂直偏移", "float", 0.0, -1_000_000, 1_000_000, 2, suffix=" px"),
                ParameterField("interpolation", "插值", "choice", "linear", choices=interpolation_modes),
                ParameterField("border_mode", "边界", "choice", "constant", choices=border_modes),
                ParameterField("border_value", "常量填充值", "float", 0.0, -1e12, 1e12, 3),
            ),
            purpose="在固定画布内按原始像素坐标平移图像。",
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.RESIZE_CANVAS,
            "变换",
            "调整画布大小",
            (
                ParameterField("width", "画布宽度", "int", -1, 1, 1_000_000),
                ParameterField("height", "画布高度", "int", -1, 1, 1_000_000),
                ParameterField(
                    "anchor",
                    "原图锚点",
                    "choice",
                    "center",
                    choices=(
                        ("左上", "top_left"),
                        ("上中", "top_center"),
                        ("右上", "top_right"),
                        ("左中", "center_left"),
                        ("居中", "center"),
                        ("右中", "center_right"),
                        ("左下", "bottom_left"),
                        ("下中", "bottom_center"),
                        ("右下", "bottom_right"),
                    ),
                ),
                ParameterField("fill_value", "新增区域填充值", "float", 0.0, -1e12, 1e12, 3),
            ),
            purpose="不重采样原像素，只扩展或裁去画布边缘。",
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.PIXEL_BIN,
            "变换",
            "像素合并",
            (
                ParameterField("factor", "合并系数", "int", 2, 1, 4096),
                ParameterField(
                    "method",
                    "聚合方式",
                    "choice",
                    "mean",
                    choices=(
                        ("均值", "mean"),
                        ("最小值", "minimum"),
                        ("最大值", "maximum"),
                        ("求和", "sum"),
                    ),
                ),
                ParameterField(
                    "remainder_policy",
                    "不能整除时",
                    "choice",
                    "reject",
                    choices=(("拒绝处理", "reject"), ("裁去右侧和底部余数", "crop")),
                ),
            ),
            purpose="将 k×k 原始像素块聚合为一个输出像素。",
            calibration_effect="pixels_per_unit 除以像素合并系数。",
            roi_behavior=no_roi,
        ),
        define(
            ImageOperation.GAUSSIAN_BLUR,
            "处理",
            "高斯滤波",
            (
                ParameterField("sigma_x", "横向 Sigma", "float", 1.0, 0.01, 100, 2),
                ParameterField("sigma_y", "纵向 Sigma", "float", 1.0, 0.01, 100, 2),
                border,
            ),
            purpose="使用高斯核平滑噪声和细节。",
        ),
        define(
            ImageOperation.MEDIAN_FILTER,
            "处理",
            "中值滤波 / 去斑",
            (radius,),
            purpose="抑制脉冲噪声并尽量保留边缘。",
        ),
        define(
            ImageOperation.MEAN_FILTER,
            "处理",
            "均值 / 方框滤波",
            (radius, local_border),
            purpose="使用方框邻域求均值以平滑图像。",
        ),
        define(
            ImageOperation.BILATERAL_FILTER,
            "处理",
            "双边滤波",
            (
                ParameterField("diameter", "邻域直径", "int", 5, 1, 99),
                ParameterField("sigma_color", "颜色域 Sigma", "float", 25.0, 0.01, 1e6, 2),
                ParameterField("sigma_space", "空间域 Sigma", "float", 2.0, 0.01, 1e6, 2),
                border,
            ),
            purpose="在平滑噪声时尽量保留强边缘。",
        ),
        define(
            ImageOperation.UNSHARP_MASK,
            "处理",
            "反锐化遮罩",
            (
                ParameterField("sigma", "模糊 Sigma", "float", 1.0, 0.01, 100, 2),
                ParameterField("amount", "增强量", "float", 1.0, 0, 20, 2),
                ParameterField("threshold", "阈值", "float", 0.0, 0, 65535, 2),
            ),
            purpose="通过高频差分增强局部边缘。",
        ),
        define(
            ImageOperation.SOBEL_EDGES,
            "处理",
            "Sobel 边缘",
            (
                ParameterField("kernel_size", "核尺寸", "choice", 3, choices=(("3", 3), ("5", 5), ("7", 7))),
                scalar_channel,
                ParameterField("output_float", "输出 32 位浮点", "bool", True),
            ),
            purpose="计算一阶梯度幅值。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.LAPLACIAN_EDGES,
            "处理",
            "Laplacian 边缘",
            (
                ParameterField("kernel_size", "核尺寸", "choice", 3, choices=(("1", 1), ("3", 3), ("5", 5), ("7", 7))),
                scalar_channel,
                ParameterField("output_float", "输出 32 位浮点", "bool", True),
            ),
            purpose="计算二阶导数以突出快速强度变化。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.CANNY_EDGES,
            "处理",
            "Canny 边缘",
            (
                ParameterField("threshold_low", "低阈值", "float", 50.0, 0, 65535, 2),
                ParameterField("threshold_high", "高阈值", "float", 150.0, 0, 65535, 2),
                ParameterField("aperture_size", "Sobel 孔径", "choice", 3, choices=(("3", 3), ("5", 5), ("7", 7))),
                ParameterField("l2_gradient", "使用 L2 梯度", "bool", True),
                scalar_channel,
            ),
            purpose="使用双阈值和边缘连接提取单像素边缘。",
            supported_types="GRAY8；RGB8/RGBA8 需选择一个通道",
        ),
        define(
            ImageOperation.NORMALIZE,
            "处理",
            "归一化",
            (
                ParameterField("output_min", "输出下限", "float", "$working_min", -1e12, 1e12, 4),
                ParameterField("output_max", "输出上限", "float", "$working_max", -1e12, 1e12, 4),
                ParameterField("per_channel", "彩色图逐通道", "bool", True),
            ),
            purpose="将有限像素的数据范围线性映射到显式输出范围。",
        ),
        define(
            ImageOperation.HISTOGRAM_EQUALIZATION,
            "处理",
            "直方图均衡",
            purpose="重新分配强度直方图以增强全局对比度。",
        ),
        define(
            ImageOperation.CLAHE,
            "处理",
            "CLAHE 局部对比度",
            (
                ParameterField("clip_limit", "对比度限制", "float", 2.0, 0.01, 1000, 2),
                ParameterField("tile_grid_size", "网格大小", "int", 8, 2, 64),
            ),
            purpose="对局部网格执行受限直方图均衡。",
            supported_types="GRAY8、GRAY16、RGB8、RGBA8",
        ),
        define(
            ImageOperation.REMOVE_OUTLIERS,
            "处理",
            "热点 / 坏点剔除",
            (
                radius,
                ParameterField("threshold", "偏差阈值", "float", 25.0, 0, 1e12, 3),
                ParameterField("polarity", "极性", "choice", "both", choices=(("亮点和暗点", "both"), ("仅亮点", "bright"), ("仅暗点", "dark"))),
            ),
            purpose="用局部中值替换孤立的亮点或暗点。",
        ),
        define(
            ImageOperation.REPAIR_NONFINITE,
            "处理",
            "修复 NaN / Inf",
            (
                ParameterField("radius", "邻域半径", "int", 1, 1, 32),
                ParameterField("fallback_value", "无有效邻域回退值", "float", 0.0, -1e12, 1e12, 4),
            ),
            purpose="用有限邻域均值替换非有限样本。",
            supported_types="仅 GRAY32_FLOAT",
        ),
        define(
            ImageOperation.AUTO_THRESHOLD,
            "处理",
            "自动阈值",
            (
                ParameterField("method", "方法", "choice", "otsu", choices=(("Otsu", "otsu"), ("IsoData", "isodata"), ("Triangle", "triangle"))),
                scalar_channel,
                ParameterField("invert", "反相", "bool", False),
            ),
            purpose="根据直方图自动计算阈值并输出二值图。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.BINARIZE,
            "处理",
            "二值化",
            (
                ParameterField("threshold", "阈值", "float", 127.0, -1e12, 1e12, 4),
                scalar_channel,
                ParameterField("invert", "反相", "bool", False),
            ),
            purpose="使用显式阈值生成二值图。",
            supported_types=scalar_types,
        ),
        *morphology,
        define(
            ImageOperation.FILL_HOLES,
            "处理",
            "填充孔洞",
            (foreground, scalar_channel),
            purpose="填充与图像边界不连通的背景孔洞。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.CONTOUR_EXTRACT,
            "处理",
            "轮廓提取",
            (foreground, scalar_channel),
            purpose="从二值前景中提取单像素内边界。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.REMOVE_SMALL_OBJECTS,
            "处理",
            "删除小对象",
            (
                ParameterField("minimum_area", "最小面积", "int", 10, 1, 2_147_483_647, suffix=" px²"),
                ParameterField("connectivity", "连通性", "choice", 8, choices=connectivity_modes),
                foreground,
                scalar_channel,
            ),
            purpose="删除面积小于阈值的连通前景对象。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.FILL_SMALL_HOLES,
            "处理",
            "填充小孔洞",
            (
                ParameterField("maximum_area", "最大孔洞面积", "int", 10, 1, 2_147_483_647, suffix=" px²"),
                ParameterField("connectivity", "连通性", "choice", 8, choices=connectivity_modes),
                foreground,
                scalar_channel,
            ),
            purpose="只填充面积不超过阈值的内部孔洞。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.DISTANCE_TRANSFORM,
            "处理",
            "距离变换",
            (
                foreground,
                ParameterField("distance_type", "距离类型", "choice", "l2", choices=(("欧氏 L2", "l2"), ("曼哈顿 L1", "l1"), ("棋盘距离", "chessboard"))),
                scalar_channel,
            ),
            purpose="计算每个前景像素到最近背景的距离。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.SKELETONIZE,
            "处理",
            "骨架化",
            (foreground, scalar_channel),
            purpose="使用 Zhang-Suen 细化生成单像素骨架。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.WATERSHED,
            "处理",
            "分水岭分割",
            (
                foreground,
                ParameterField("seed_threshold", "种子阈值比例", "float", 0.45, 0.001, 0.999, 3),
                scalar_channel,
            ),
            purpose="按距离峰值拆分相互接触的二值对象。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.BACKGROUND_SUBTRACT,
            "处理",
            "背景扣除",
            (
                ParameterField("radius", "背景半径", "int", 25, 1, 2048, suffix=" px"),
                ParameterField("light_background", "亮背景", "bool", False),
                ParameterField("preserve_offset", "保留背景中位偏移", "bool", False),
                local_border,
            ),
            purpose="用滑动形态学背景估计扣除缓慢变化背景。",
        ),
        define(
            ImageOperation.CUSTOM_CONVOLUTION,
            "处理",
            "自定义卷积核",
            (
                ParameterField("kernel_width", "核宽度", "int", 3, 1, 99),
                ParameterField("kernel_height", "核高度", "int", 3, 1, 99),
                ParameterField(
                    "kernel",
                    "核元素",
                    "number_list",
                    (-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0),
                    help_text="按行输入，以逗号或空格分隔；元素数必须等于核宽×核高。",
                ),
                ParameterField("normalize_kernel", "归一化核", "bool", False),
                ParameterField("offset", "结果偏移", "float", 0.0, -1e12, 1e12, 4),
                local_border,
            ),
            purpose="使用用户提供的奇数尺寸卷积核处理图像。",
        ),
        define(ImageOperation.INVERT, "处理", "反相", purpose="在当前类型的工作范围内反转强度。"),
        *tuple(
            define(
                value,
                "处理",
                label,
                (ParameterField("value", "常数", "float", default, -1e12, 1e12, 6),),
                purpose=purpose,
            )
            for value, label, default, purpose in (
                (ImageOperation.ADD, "加常数", 0.0, "为每个颜色通道加上常数。"),
                (ImageOperation.SUBTRACT, "减常数", 0.0, "从每个颜色通道减去常数。"),
                (ImageOperation.MULTIPLY, "乘常数", 1.0, "将每个颜色通道乘以常数。"),
                (ImageOperation.DIVIDE, "除常数", 1.0, "将每个颜色通道除以非零常数。"),
            )
        ),
        define(
            ImageOperation.GAMMA,
            "处理",
            "Gamma 运算",
            (ParameterField("gamma", "Gamma", "float", 1.0, 0.001, 100, 4),),
            purpose="在当前类型的工作范围内应用幂律变换。",
        ),
        define(ImageOperation.LOG, "处理", "Log 运算", purpose="对非负像素计算 log(1+x)。"),
        define(ImageOperation.EXP, "处理", "Exp 运算", purpose="对像素计算指数；溢出时明确拒绝。"),
        define(ImageOperation.SQRT, "处理", "Sqrt 运算", purpose="对非负像素计算平方根。"),
        define(ImageOperation.ABS, "处理", "绝对值", purpose="对每个像素取绝对值。"),
        define(
            ImageOperation.CLAMP,
            "处理",
            "范围截断",
            (
                ParameterField("minimum", "下限", "float", "$working_min", -1e12, 1e12, 4),
                ParameterField("maximum", "上限", "float", "$working_max", -1e12, 1e12, 4),
            ),
            purpose="将像素限制在显式上下限范围内。",
        ),
        define(
            ImageOperation.IMAGE_CALCULATOR,
            "处理",
            "图像计算器",
            (
                ParameterField(
                    "secondary_document_id",
                    "第二幅图像",
                    "secondary_image",
                    "",
                    help_text="只列出由工作区提供的候选图像；执行时仍会校验宽高、通道和位深。",
                ),
                ParameterField(
                    "calculator_operation",
                    "运算",
                    "choice",
                    "add",
                    choices=(
                        ("相加", "add"),
                        ("相减", "subtract"),
                        ("相乘", "multiply"),
                        ("相除", "divide"),
                        ("绝对差", "difference"),
                        ("最小值", "minimum"),
                        ("最大值", "maximum"),
                        ("均值", "mean"),
                        ("AND", "and"),
                        ("OR", "or"),
                        ("XOR", "xor"),
                    ),
                ),
            ),
            purpose="对两幅同宽高、同通道、同位深图像逐像素运算。",
            calibration_effect="要求调用方同时确认两幅图像的标定和空间对齐。",
        ),
        define(
            ImageOperation.FFT_FILTER,
            "处理",
            "FFT 频域滤波",
            (
                ParameterField("mode", "模式", "choice", "lowpass", choices=(("低通", "lowpass"), ("高通", "highpass"), ("带通", "bandpass"), ("带阻", "bandstop"))),
                ParameterField("low_cutoff", "低截止频率", "float", 0.0, 0, 0.5, 4),
                ParameterField("high_cutoff", "高截止频率", "float", 0.15, 0, 0.5, 4),
                ParameterField("order", "阶数", "int", 2, 1, 16),
                ParameterField("channel", "彩色图通道", "choice", "per_channel", choices=fft_channels),
                ParameterField("output_float", "输出 32 位浮点", "bool", False),
            ),
            purpose="使用 Butterworth 响应进行低通、高通、带通或带阻滤波。",
            roi_behavior="支持 ROI 写回；FFT 仍读取完整图像，ROI 外像素保持不变。",
        ),
        define(
            ImageOperation.STRIPE_SUPPRESSION,
            "处理",
            "条纹抑制",
            (
                ParameterField("direction", "条纹方向", "choice", "horizontal", choices=(("水平条纹", "horizontal"), ("垂直条纹", "vertical"))),
                ParameterField("notch_width", "陷波宽度", "float", 0.02, 0.0001, 0.25, 4),
                ParameterField("protect_radius", "低频保护半径", "float", 0.02, 0.0, 0.25, 4),
                ParameterField("strength", "抑制强度", "float", 1.0, 0.0, 1.0, 3),
            ),
            purpose="在频域衰减与周期性水平或垂直条纹对应的频率轴。",
            roi_behavior="支持 ROI 写回；频域计算仍读取完整图像，ROI 外像素保持不变。",
        ),
    )


_OPERATION_CATALOG = _operation_catalog()
_DEFINITION_BY_ID = {
    definition.operation.value: definition for definition in _OPERATION_CATALOG
}


def image_operation_display_name(operation_id: ImageOperation | str) -> str:
    """Return the stable Chinese operation label shared by processing UIs."""

    resolved_id = (
        operation_id.value
        if isinstance(operation_id, ImageOperation)
        else str(operation_id).strip()
    )
    definition = _DEFINITION_BY_ID.get(resolved_id)
    return definition.label if definition is not None else resolved_id


def _resolved_parameter_default(
    parameter: ParameterField,
    *,
    source_width: int,
    source_height: int,
    source_pixel_type: RasterPixelType,
    secondary_document_id: str | None = None,
) -> object:
    if parameter.default == "$working_min":
        return 0.0
    if parameter.default == "$working_max":
        maximum = source_pixel_type.sample_maximum
        return 1.0 if maximum is None else float(maximum)
    if parameter.kind == "secondary_image":
        return str(secondary_document_id or "")
    if parameter.default != -1:
        return parameter.default
    if parameter.key == "width":
        return int(source_width)
    if parameter.key == "height":
        return int(source_height)
    return parameter.default


def default_operation_spec(
    operation_id: ImageOperation | str,
    source_width: int,
    source_height: int,
    *,
    source_pixel_type: RasterPixelType | str = RasterPixelType.GRAY8,
    secondary_document_id: str | None = None,
) -> ImageOperationSpec:
    """Build an explicit, service-compatible operation preset for the UI."""

    resolved_id = (
        operation_id.value
        if isinstance(operation_id, ImageOperation)
        else str(operation_id).strip()
    )
    try:
        definition = _DEFINITION_BY_ID[resolved_id]
    except KeyError as exc:
        raise ValueError(f"工作台不支持操作: {resolved_id}") from exc
    width = int(source_width)
    height = int(source_height)
    if width <= 0 or height <= 0:
        raise ValueError("默认操作参数需要正数源图片宽高")
    pixel_type = RasterPixelType.parse(source_pixel_type)
    parameters = {
        field.key: _resolved_parameter_default(
            field,
            source_width=width,
            source_height=height,
            source_pixel_type=pixel_type,
            secondary_document_id=secondary_document_id,
        )
        for field in definition.parameters
    }
    if (
        definition.operation is ImageOperation.IMAGE_CALCULATOR
        and not parameters.get("secondary_document_id")
    ):
        raise ValueError("图像计算器默认参数需要第二幅图像文档 ID")
    return ImageOperationSpec(definition.operation.value, parameters)


class ImageProcessingWorkbench(QDialog):
    """Non-modal Chinese workbench for an ordered, non-destructive recipe."""

    derivedImageReady = Signal(object)
    cancelled = Signal()
    previewChanged = Signal(object)
    recipeSaveRequested = Signal(object)
    recipeLoadRequested = Signal()
    batchApplyRequested = Signal(object)

    PREVIEW_DEBOUNCE_MS = 150

    def __init__(
        self,
        source: RasterPlane,
        *,
        source_document_id: str,
        source_name: str = "",
        roi_summary: str = "整张图片",
        roi_mask: NDArray[np.bool_] | None = None,
        secondary_images: Mapping[str, RasterPlane] | None = None,
        secondary_image_names: Mapping[str, str] | None = None,
        executor: ImageTaskExecutor | None = None,
        resource_check_directory: Path | str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("图像处理工作台")
        self.setModal(False)
        self.setMinimumSize(780, 520)
        self.resize(1080, 720)

        self._source = source
        self._source_document_id = str(source_document_id)
        self._source_name = source_name.strip() or "未命名图片"
        self._roi_summary = roi_summary.strip() or "整张图片"
        self._roi_mask = roi_mask
        self._secondary_images = dict(secondary_images or {})
        self._secondary_image_names = {
            document_id: str((secondary_image_names or {}).get(document_id) or document_id)
            for document_id in self._secondary_images
        }
        self._resource_check_directory = (
            None
            if resource_check_directory is None
            else Path(resource_check_directory).expanduser()
        )
        self._steps: tuple[ImageOperationSpec, ...] = ()
        self._undo_stack: list[tuple[ImageOperationSpec, ...]] = []
        self._redo_stack: list[tuple[ImageOperationSpec, ...]] = []
        self._parameter_widgets: dict[str, QWidget] = {}
        self._updating_parameter_form = False
        self._final_in_progress = False

        self._controller = ImageProcessingTaskController(
            executor=executor,
            parent=self,
        )
        self._controller.previewReady.connect(self._on_preview_ready)
        self._controller.finalReady.connect(self._on_final_ready)
        self._controller.taskFailed.connect(self._on_task_failed)
        self._controller.busyChanged.connect(self._on_busy_changed)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(self.PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self.request_preview)

        self._build_ui()
        self._populate_operations()
        self._refresh_steps()
        self._show_preview_raster(self._source)
        self._update_actions()

    @property
    def task_controller(self) -> ImageProcessingTaskController:
        return self._controller

    def operation_steps(self) -> tuple[ImageOperationSpec, ...]:
        return self._steps

    def current_recipe(self) -> ImageProcessingRecipe:
        return ImageProcessingRecipe.from_operations(self._steps)

    def apply_loaded_recipe(self, recipe: ImageProcessingRecipe) -> None:
        """Apply a host-selected preset without giving the workbench file access."""

        if not isinstance(recipe, ImageProcessingRecipe):
            raise TypeError("recipe 必须是 ImageProcessingRecipe")
        self.set_operation_steps(recipe.operations)

    def generate_derived_image(self) -> None:
        """Start the same validated final task as the visible footer button."""

        self._generate_derived_image()

    def set_operation_steps(self, operations: tuple[ImageOperationSpec, ...]) -> None:
        normalized_operations: list[ImageOperationSpec] = []
        for operation in operations:
            if operation.operation_id not in _DEFINITION_BY_ID:
                raise ValueError(f"工作台不支持操作: {operation.operation_id}")
            secondary_document_id = (
                str(operation.parameters.get("secondary_document_id", ""))
                or next(iter(self._secondary_images), "")
            )
            defaults = default_operation_spec(
                operation.operation_id,
                self._source.width,
                self._source.height,
                source_pixel_type=self._source.pixel_type,
                secondary_document_id=(
                    secondary_document_id
                    if operation.operation_id
                    == ImageOperation.IMAGE_CALCULATOR.value
                    else None
                ),
            )
            parameters = defaults.parameters
            parameters.update(operation.parameters)
            normalized_operation = ImageOperationSpec(
                operation.operation_id,
                parameters,
                implementation=operation.implementation,
                implementation_version=operation.implementation_version,
            )
            if operation.operation_id == ImageOperation.IMAGE_CALCULATOR.value:
                secondary_document_id = str(
                    normalized_operation.parameters.get("secondary_document_id", "")
                )
                if secondary_document_id not in self._secondary_images:
                    raise ValueError("图像计算器选择的第二幅图像不在当前候选列表中")
            normalized_operations.append(normalized_operation)
        normalized = tuple(normalized_operations)
        self._commit_steps(normalized, selected_index=0 if normalized else -1)

    def request_preview(self) -> None:
        self._preview_timer.stop()
        if not self._steps or self._final_in_progress:
            self._controller.cancel_preview()
            self._show_preview_raster(self._source)
            self._status_label.setText(
                "尚未添加处理步骤。" if not self._steps else "正在生成派生图片…"
            )
            return
        self._status_label.setText("正在计算 1:1 预览…")
        try:
            self._controller.start_preview(
                source_document_id=self._source_document_id,
                source=self._source,
                operations=self._steps,
                roi_mask=self._roi_mask,
                secondary_images=self._secondary_images,
            )
        except ValueError as exc:
            self._status_label.setText(f"无法开始预览：{exc}")

    def cancel_tasks(self) -> None:
        self._preview_timer.stop()
        self._controller.cancel_all()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.cancel_tasks()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        source_bar = QWidget(self)
        source_layout = QHBoxLayout(source_bar)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(18)
        self._source_label = QLabel(
            f"<b>源图片：</b>{self._source_name}　"
            f"{self._source.width} × {self._source.height}　"
            f"{self._source.pixel_type.value}",
            source_bar,
        )
        self._source_label.setTextFormat(Qt.TextFormat.RichText)
        self._roi_label = QLabel(f"<b>处理范围：</b>{self._roi_summary}", source_bar)
        self._roi_label.setTextFormat(Qt.TextFormat.RichText)
        source_layout.addWidget(self._source_label, 1)
        source_layout.addWidget(self._roi_label, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_steps_panel(splitter))
        splitter.addWidget(self._build_preview_panel(splitter))
        splitter.addWidget(self._build_parameters_panel(splitter))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([270, 520, 280])
        self._main_splitter = splitter

        footer = QWidget(self)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        self._status_label = QLabel("尚未添加处理步骤。", footer)
        self._status_label.setObjectName("imageProcessingStatus")
        footer_layout.addWidget(self._status_label, 1)
        self._save_recipe_button = QPushButton("保存配方…", footer)
        self._save_recipe_button.setToolTip(
            "请求工作区保存当前有序步骤；工作台本身不直接写设置文件。"
        )
        self._save_recipe_button.clicked.connect(self._request_recipe_save)
        self._load_recipe_button = QPushButton("载入配方…", footer)
        self._load_recipe_button.setToolTip(
            "请求工作区选择并校验一个已保存的处理配方。"
        )
        self._load_recipe_button.clicked.connect(self.recipeLoadRequested.emit)
        self._batch_apply_button = QPushButton("批量应用…", footer)
        self._batch_apply_button.setToolTip(
            "把当前配方交给批处理窗口；成功结果仍需由项目工作区统一提交。"
        )
        self._batch_apply_button.clicked.connect(self._request_batch_apply)
        self._generate_button = QPushButton("生成派生图片", footer)
        self._generate_button.clicked.connect(self._generate_derived_image)
        self._cancel_button = QPushButton("取消", footer)
        self._cancel_button.clicked.connect(self._cancel_and_close)
        footer_layout.addWidget(self._save_recipe_button)
        footer_layout.addWidget(self._load_recipe_button)
        footer_layout.addWidget(self._batch_apply_button)
        footer_layout.addWidget(self._generate_button)
        footer_layout.addWidget(self._cancel_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        root.addWidget(source_bar)
        root.addWidget(splitter, 1)
        root.addWidget(footer)

    def _build_steps_panel(self, parent: QWidget) -> QWidget:
        panel = QGroupBox("处理步骤", parent)
        layout = QVBoxLayout(panel)

        add_form = QFormLayout()
        self._category_combo = NoWheelComboBox(panel)
        for category in ("类型", "调整", "变换", "处理"):
            self._category_combo.addItem(category, category)
        self._operation_combo = NoWheelComboBox(panel)
        self._category_combo.currentIndexChanged.connect(self._populate_operations)
        add_form.addRow("分类", self._category_combo)
        add_form.addRow("操作", self._operation_combo)
        layout.addLayout(add_form)

        self._add_step_button = QPushButton("添加步骤", panel)
        self._add_step_button.clicked.connect(self._add_selected_operation)
        layout.addWidget(self._add_step_button)

        self._steps_list = QListWidget(panel)
        self._steps_list.setAlternatingRowColors(True)
        self._steps_list.currentRowChanged.connect(self._on_step_selected)
        layout.addWidget(self._steps_list, 1)

        row = QHBoxLayout()
        self._move_up_button = QPushButton("上移", panel)
        self._move_down_button = QPushButton("下移", panel)
        self._remove_step_button = QPushButton("删除", panel)
        self._move_up_button.clicked.connect(lambda: self._move_step(-1))
        self._move_down_button.clicked.connect(lambda: self._move_step(1))
        self._remove_step_button.clicked.connect(self._remove_current_step)
        row.addWidget(self._move_up_button)
        row.addWidget(self._move_down_button)
        row.addWidget(self._remove_step_button)
        layout.addLayout(row)

        history_row = QHBoxLayout()
        self._reset_steps_button = QPushButton("重置", panel)
        self._undo_button = QPushButton("撤销", panel)
        self._redo_button = QPushButton("重做", panel)
        self._reset_steps_button.clicked.connect(self._reset_steps)
        self._undo_button.clicked.connect(self._undo_steps)
        self._redo_button.clicked.connect(self._redo_steps)
        history_row.addWidget(self._reset_steps_button)
        history_row.addWidget(self._undo_button)
        history_row.addWidget(self._redo_button)
        layout.addLayout(history_row)
        return panel

    def _build_preview_panel(self, parent: QWidget) -> QWidget:
        panel = QGroupBox("预览", parent)
        layout = QVBoxLayout(panel)

        hint = QLabel(
            "1:1 预览使用原始像素语义；最终处理始终以原始分辨率执行。",
            panel,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._preview_stack = QStackedWidget(panel)
        self._preview_scroll = QScrollArea(self._preview_stack)
        self._preview_scroll.setWidgetResizable(False)
        self._preview_scroll.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._preview_image_label = QLabel(self._preview_scroll)
        self._preview_image_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._preview_scroll.setWidget(self._preview_image_label)
        self._preview_stack.addWidget(self._preview_scroll)

        self._overview_page = QWidget(self._preview_stack)
        overview_layout = QVBoxLayout(self._overview_page)
        self._overview_image_label = QLabel(self._overview_page)
        self._overview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overview_image_label.setMinimumSize(240, 180)
        self._overview_image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        overview_note = QLabel(
            "预览已缩放，仅用于观察全图趋势；像素级判断请切回 1:1。",
            self._overview_page,
        )
        overview_note.setObjectName("scaledOverviewNotice")
        overview_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overview_note.setWordWrap(True)
        overview_layout.addWidget(self._overview_image_label, 1)
        overview_layout.addWidget(overview_note)
        self._preview_stack.addWidget(self._overview_page)
        layout.addWidget(self._preview_stack, 1)

        preview_mode_row = QHBoxLayout()
        self._overview_checkbox = QCheckBox("显示近似全图概览", panel)
        self._overview_checkbox.toggled.connect(self._toggle_overview)
        preview_mode_row.addWidget(self._overview_checkbox)
        preview_mode_row.addStretch(1)
        layout.addLayout(preview_mode_row)
        return panel

    def _build_parameters_panel(self, parent: QWidget) -> QWidget:
        panel = QGroupBox("当前步骤参数", parent)
        layout = QVBoxLayout(panel)
        self._parameter_scroll = QScrollArea(panel)
        self._parameter_scroll.setWidgetResizable(True)
        self._parameter_content = QWidget(self._parameter_scroll)
        self._parameter_form = QFormLayout(self._parameter_content)
        self._parameter_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._parameter_scroll.setWidget(self._parameter_content)
        layout.addWidget(self._parameter_scroll)
        return panel

    def _populate_operations(self) -> None:
        current = self._operation_combo.currentData() if self._operation_combo.count() else None
        category = str(self._category_combo.currentData() or "类型")
        self._operation_combo.clear()
        for definition in _OPERATION_CATALOG:
            if definition.category == category:
                if (
                    definition.operation is ImageOperation.IMAGE_CALCULATOR
                    and not self._secondary_images
                ):
                    continue
                self._operation_combo.addItem(definition.label, definition.operation.value)
                index = self._operation_combo.count() - 1
                self._operation_combo.setItemData(
                    index,
                    (
                        f"{definition.purpose}\n"
                        f"适用类型：{definition.supported_types}"
                    ),
                    Qt.ItemDataRole.ToolTipRole,
                )
        if current is not None:
            index = self._operation_combo.findData(current)
            if index >= 0:
                self._operation_combo.setCurrentIndex(index)

    def _add_selected_operation(self) -> None:
        operation_id = str(self._operation_combo.currentData() or "")
        definition = _DEFINITION_BY_ID.get(operation_id)
        if definition is None:
            return
        step = default_operation_spec(
            operation_id,
            self._source.width,
            self._source.height,
            source_pixel_type=self._source.pixel_type,
            secondary_document_id=next(iter(self._secondary_images), None),
        )
        self._commit_steps(self._steps + (step,), selected_index=len(self._steps))

    def _resolved_default(self, parameter: ParameterField) -> object:
        return _resolved_parameter_default(
            parameter,
            source_width=self._source.width,
            source_height=self._source.height,
            source_pixel_type=self._source.pixel_type,
            secondary_document_id=next(iter(self._secondary_images), None),
        )

    def _remove_current_step(self) -> None:
        row = self._steps_list.currentRow()
        if not 0 <= row < len(self._steps):
            return
        updated = self._steps[:row] + self._steps[row + 1 :]
        self._commit_steps(updated, selected_index=min(row, len(updated) - 1))

    def _move_step(self, offset: int) -> None:
        row = self._steps_list.currentRow()
        target = row + offset
        if not 0 <= row < len(self._steps) or not 0 <= target < len(self._steps):
            return
        values = list(self._steps)
        values[row], values[target] = values[target], values[row]
        self._commit_steps(tuple(values), selected_index=target)

    def _reset_steps(self) -> None:
        if self._steps:
            self._commit_steps((), selected_index=-1)

    def _undo_steps(self) -> None:
        if not self._undo_stack:
            return
        previous = self._undo_stack.pop()
        self._redo_stack.append(self._steps)
        self._steps = previous
        self._refresh_steps()
        self._schedule_preview()

    def _redo_steps(self) -> None:
        if not self._redo_stack:
            return
        following = self._redo_stack.pop()
        self._undo_stack.append(self._steps)
        self._steps = following
        self._refresh_steps()
        self._schedule_preview()

    def _commit_steps(
        self,
        steps: tuple[ImageOperationSpec, ...],
        *,
        selected_index: int,
    ) -> None:
        if steps == self._steps:
            return
        self._undo_stack.append(self._steps)
        if len(self._undo_stack) > 100:
            del self._undo_stack[0]
        self._redo_stack.clear()
        self._steps = tuple(steps)
        self._refresh_steps(selected_index=selected_index)
        self._schedule_preview()

    def _refresh_steps(self, *, selected_index: int | None = None) -> None:
        if selected_index is None:
            selected_index = self._steps_list.currentRow()
        self._steps_list.blockSignals(True)
        self._steps_list.clear()
        for index, step in enumerate(self._steps, start=1):
            definition = _DEFINITION_BY_ID[step.operation_id]
            item = QListWidgetItem(f"{index}. {definition.label}")
            item.setToolTip(f"{definition.category} · {definition.label}")
            self._steps_list.addItem(item)
        self._steps_list.blockSignals(False)
        if self._steps:
            selected_index = max(0, min(int(selected_index), len(self._steps) - 1))
            self._steps_list.setCurrentRow(selected_index)
        else:
            self._steps_list.setCurrentRow(-1)
            self._rebuild_parameter_form(-1)
        self._update_actions()

    def _on_step_selected(self, row: int) -> None:
        self._rebuild_parameter_form(row)
        self._update_actions()

    def _clear_parameter_form(self) -> None:
        while self._parameter_form.rowCount():
            self._parameter_form.removeRow(0)
        self._parameter_widgets.clear()

    def _rebuild_parameter_form(self, row: int) -> None:
        self._updating_parameter_form = True
        self._clear_parameter_form()
        if not 0 <= row < len(self._steps):
            self._parameter_form.addRow(QLabel("请选择一个处理步骤。", self._parameter_content))
            self._updating_parameter_form = False
            return
        step = self._steps[row]
        definition = _DEFINITION_BY_ID[step.operation_id]
        capability = resolve_image_operation_capability(
            definition.operation,
            step.parameters,
        )
        if capability.tileable:
            execution_help = (
                "可分块精确执行"
                if capability.halo_x == capability.halo_y == 0
                else (
                    "可分块精确执行；从原图读取横向 ±"
                    f"{capability.halo_x} px、纵向 ±"
                    f"{capability.halo_y} px 邻域"
                )
            )
        else:
            execution_help = "整图执行"
            if capability.reason:
                execution_help += f"；{capability.reason}"
            execution_help += (
                "；取消请求会在当前整图算法返回后立即确认，"
                "取消后不会提交派生图片。"
            )
        title = QLabel(f"<b>{definition.category} · {definition.label}</b>", self._parameter_content)
        title.setTextFormat(Qt.TextFormat.RichText)
        self._parameter_form.addRow(title)
        help_label = QLabel(
            "\n".join(
                (
                    f"用途：{definition.purpose or definition.label}",
                    f"像素：{definition.pixel_effect}",
                    f"标定：{definition.calibration_effect}",
                    f"适用类型：{definition.supported_types}",
                    f"ROI：{definition.roi_behavior}",
                    f"执行：{execution_help}",
                )
            ),
            self._parameter_content,
        )
        help_label.setObjectName("imageOperationHelp")
        help_label.setWordWrap(True)
        help_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._parameter_form.addRow(help_label)
        if not definition.parameters:
            self._parameter_form.addRow(QLabel("此操作没有可调参数。", self._parameter_content))
        values = step.parameters
        for parameter in definition.parameters:
            value = values.get(parameter.key, self._resolved_default(parameter))
            widget = self._create_parameter_widget(parameter, value)
            self._parameter_widgets[parameter.key] = widget
            parameter_label = QLabel(parameter.label, self._parameter_content)
            tooltip = parameter.help_text or f"{parameter.label}会写入可追溯的处理配方。"
            parameter_label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            self._parameter_form.addRow(parameter_label, widget)
        self._updating_parameter_form = False

    def _create_parameter_widget(self, parameter: ParameterField, value: object) -> QWidget:
        if parameter.kind == "bool":
            widget = QCheckBox(self._parameter_content)
            widget.setChecked(bool(value))
            widget.toggled.connect(self._parameter_value_changed)
            return widget
        if parameter.kind == "int":
            widget = NoWheelSpinBox(self._parameter_content)
            minimum = (
                -2_147_483_647
                if parameter.minimum is None
                else int(parameter.minimum)
            )
            maximum = (
                2_147_483_647
                if parameter.maximum is None
                else int(parameter.maximum)
            )
            widget.setRange(minimum, maximum)
            widget.setValue(int(value))
            widget.setSuffix(parameter.suffix)
            widget.editingFinished.connect(self._parameter_value_changed)
            return widget
        if parameter.kind == "float":
            widget = NoWheelDoubleSpinBox(self._parameter_content)
            widget.setDecimals(parameter.decimals)
            widget.setRange(
                float(parameter.minimum if parameter.minimum is not None else -1e12),
                float(parameter.maximum if parameter.maximum is not None else 1e12),
            )
            widget.setValue(float(value))
            widget.setSuffix(parameter.suffix)
            widget.editingFinished.connect(self._parameter_value_changed)
            return widget
        if parameter.kind == "choice":
            widget = NoWheelComboBox(self._parameter_content)
            for label, data in parameter.choices:
                widget.addItem(label, data)
            selected = widget.findData(value)
            widget.setCurrentIndex(max(0, selected))
            widget.currentIndexChanged.connect(self._parameter_value_changed)
            return widget
        if parameter.kind == "number_list":
            widget = QLineEdit(self._parameter_content)
            values = value if isinstance(value, (list, tuple)) else (value,)
            widget.setText(", ".join(f"{float(item):g}" for item in values))
            widget.editingFinished.connect(self._parameter_value_changed)
            return widget
        if parameter.kind == "secondary_image":
            widget = NoWheelComboBox(self._parameter_content)
            for document_id, plane in self._secondary_images.items():
                name = self._secondary_image_names.get(document_id, document_id)
                widget.addItem(
                    f"{name} · {plane.width}×{plane.height} · {plane.pixel_type.value}",
                    document_id,
                )
            selected = widget.findData(value)
            widget.setCurrentIndex(max(0, selected))
            widget.currentIndexChanged.connect(self._parameter_value_changed)
            return widget
        raise ValueError(f"未知参数控件类型: {parameter.kind}")

    def _parameter_value_changed(self) -> None:
        if self._updating_parameter_form:
            return
        row = self._steps_list.currentRow()
        if not 0 <= row < len(self._steps):
            return
        definition = _DEFINITION_BY_ID[self._steps[row].operation_id]
        parameters: dict[str, object] = {}
        for field in definition.parameters:
            widget = self._parameter_widgets[field.key]
            if isinstance(widget, QCheckBox):
                value: object = widget.isChecked()
            elif isinstance(widget, NoWheelSpinBox):
                value = widget.value()
            elif isinstance(widget, NoWheelDoubleSpinBox):
                value = widget.value()
            elif isinstance(widget, NoWheelComboBox):
                value = widget.currentData()
            elif isinstance(widget, QLineEdit):
                tokens = (
                    widget.text()
                    .replace(";", " ")
                    .replace(",", " ")
                    .split()
                )
                try:
                    numbers = tuple(float(token) for token in tokens)
                except ValueError:
                    self._status_label.setText(
                        f"{field.label}只能包含以逗号或空格分隔的数值。"
                    )
                    widget.setFocus()
                    return
                if not numbers or not all(math.isfinite(item) for item in numbers):
                    self._status_label.setText(
                        f"{field.label}必须包含至少一个有限数值。"
                    )
                    widget.setFocus()
                    return
                value = numbers
            else:  # pragma: no cover - construction is exhaustive
                continue
            parameters[field.key] = value
        replacement = ImageOperationSpec(definition.operation.value, parameters)
        updated = list(self._steps)
        updated[row] = replacement
        self._commit_steps(tuple(updated), selected_index=row)

    def _schedule_preview(self) -> None:
        self._controller.cancel_preview()
        self._status_label.setText(
            "等待更新预览…" if self._steps else "尚未添加处理步骤。"
        )
        self._preview_timer.start()
        self._update_actions()

    def _on_preview_ready(self, result: object) -> None:
        if not isinstance(result, WorkbenchTaskResult):
            return
        self._show_preview_raster(result.raster)
        self._status_label.setText(
            f"预览已更新 · {result.raster.width} × {result.raster.height}"
        )
        self.previewChanged.emit(result)

    def _generate_derived_image(self) -> None:
        if not self._steps or self._final_in_progress:
            return
        try:
            validate_workbench_operation_sequence(
                self._source,
                self._steps,
            )
            estimate = validate_final_resources(
                self._source,
                self._steps,
                storage_directory=self._resource_check_directory,
            )
        except (FinalResourcePreflightError, ValueError) as exc:
            message = str(exc)
            self._status_label.setText(f"无法开始最终处理：{message}")
            QMessageBox.warning(self, "无法开始处理", message)
            return
        self._preview_timer.stop()
        self._controller.cancel_preview()
        self._final_in_progress = True
        output_mib = estimate.output_bytes / float(1 << 20)
        self._status_label.setText(
            "正在以原始分辨率生成派生图片…"
            f"（预计未压缩输出 {output_mib:.1f} MiB）"
        )
        self._update_actions()
        self._controller.start_final(
            source_document_id=self._source_document_id,
            source=self._source,
            operations=self._steps,
            roi_mask=self._roi_mask,
            secondary_images=self._secondary_images,
        )

    def _on_final_ready(self, result: object) -> None:
        if not isinstance(result, WorkbenchTaskResult):
            return
        self._final_in_progress = False
        self._status_label.setText("处理完成，等待加入项目。")
        self._update_actions()
        self.derivedImageReady.emit(result)

    def _on_task_failed(self, kind: str, message: str) -> None:
        if kind == WorkbenchTaskKind.FINAL.value:
            self._final_in_progress = False
        title = "生成派生图片失败" if kind == WorkbenchTaskKind.FINAL.value else "预览失败"
        self._status_label.setText(f"{title}：{message}")
        self._update_actions()

    def _on_busy_changed(self, kind: str, busy: bool) -> None:
        if kind == WorkbenchTaskKind.FINAL.value and not busy and self._final_in_progress:
            self._final_in_progress = False
        self._update_actions()

    def _cancel_and_close(self) -> None:
        self.cancel_tasks()
        self.cancelled.emit()
        self.reject()

    def _request_recipe_save(self) -> None:
        if self._steps:
            self.recipeSaveRequested.emit(self.current_recipe())

    def _request_batch_apply(self) -> None:
        if self._steps:
            self.batchApplyRequested.emit(self.current_recipe())

    def _update_actions(self) -> None:
        row = self._steps_list.currentRow()
        count = len(self._steps)
        final_busy = self._controller.is_busy(WorkbenchTaskKind.FINAL)
        self._move_up_button.setEnabled(0 < row < count)
        self._move_down_button.setEnabled(0 <= row < count - 1)
        self._remove_step_button.setEnabled(0 <= row < count)
        self._reset_steps_button.setEnabled(bool(self._steps))
        self._undo_button.setEnabled(bool(self._undo_stack))
        self._redo_button.setEnabled(bool(self._redo_stack))
        self._generate_button.setEnabled(bool(self._steps) and not final_busy)
        self._save_recipe_button.setEnabled(bool(self._steps) and not final_busy)
        self._load_recipe_button.setEnabled(not final_busy)
        self._batch_apply_button.setEnabled(bool(self._steps) and not final_busy)
        self._add_step_button.setEnabled(not final_busy)

    def _show_preview_raster(self, raster: RasterPlane) -> None:
        image = raster_plane_to_display_image(raster)
        pixmap = QPixmap.fromImage(image)
        self._preview_image_label.setPixmap(pixmap)
        self._preview_image_label.resize(pixmap.size())
        overview_size = self._overview_image_label.size()
        if overview_size.width() < 2 or overview_size.height() < 2:
            overview_size.setWidth(320)
            overview_size.setHeight(240)
        self._overview_image_label.setPixmap(
            pixmap.scaled(
                overview_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _toggle_overview(self, checked: bool) -> None:
        self._preview_stack.setCurrentWidget(
            self._overview_page if checked else self._preview_scroll
        )


def raster_plane_to_display_image(plane: RasterPlane) -> QImage:
    array = raster_plane_to_array(plane)
    if plane.pixel_type in {RasterPixelType.GRAY16, RasterPixelType.GRAY32_FLOAT}:
        work = np.asarray(array, dtype=np.float64)
        finite = np.isfinite(work)
        if not np.any(finite):
            display = np.zeros(work.shape, dtype=np.uint8)
        else:
            minimum = float(np.min(work[finite]))
            maximum = float(np.max(work[finite]))
            if math.isclose(minimum, maximum):
                display = np.zeros(work.shape, dtype=np.uint8)
            else:
                normalized = (work - minimum) / (maximum - minimum)
                normalized[~finite] = 0.0
                display = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
    else:
        display = np.ascontiguousarray(array, dtype=np.uint8)

    if display.ndim == 2:
        height, width = display.shape
        return QImage(
            display.data,
            width,
            height,
            int(display.strides[0]),
            QImage.Format.Format_Grayscale8,
        ).copy()
    height, width, channels = display.shape
    if channels == 3:
        image_format = QImage.Format.Format_RGB888
    elif channels == 4:
        image_format = QImage.Format.Format_RGBA8888
    else:  # pragma: no cover - RasterPlane validation guards this
        raise ValueError("显示图像必须为灰度、RGB 或 RGBA")
    return QImage(
        display.data,
        width,
        height,
        int(display.strides[0]),
        image_format,
    ).copy()


__all__ = [
    "FinalResourceEstimate",
    "FinalResourcePreflightError",
    "ImageProcessingTaskController",
    "ImageProcessingWorkbench",
    "WorkbenchTaskKind",
    "WorkbenchTaskRequest",
    "WorkbenchTaskResult",
    "array_to_raster_plane",
    "default_operation_spec",
    "estimate_final_resources",
    "execute_workbench_request",
    "raster_plane_to_array",
    "raster_plane_to_display_image",
    "validate_final_resources",
    "validate_workbench_operation_sequence",
]
