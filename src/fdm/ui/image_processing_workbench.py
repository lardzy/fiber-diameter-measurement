from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QRunnable, QRectF, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QCloseEvent, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
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
    QVBoxLayout,
    QWidget,
)

from fdm.cancellation import CancellationError, CancellationToken, CancellationTokenSource
from fdm.image_processing_models import (
    ImageOperationSpec,
    ImageProcessingRecipe,
    RasterSemantic,
    RasterTypeState,
)
from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.image_processing import (
    ImageOperation,
    InterpolationMode,
    RecipeValidationResult,
    auto_threshold,
    canny_gradient_magnitude,
    execute_image_operation_tiled,
    flat_field_reference_levels,
    get_image_operation_descriptor,
    resolve_resize_interpolation,
    resolve_image_operation_capability,
    validate_image_processing_recipe,
)
from fdm.ui.image_parameter_data import (
    count_parameter_range,
    parameter_histogram_snapshot,
    scalar_parameter_samples,
)
from fdm.ui.image_parameter_widgets import (
    AnchorGridEditor,
    CropBoundsEditor,
    FrequencyResponseEditor,
    HistogramRangeEditor,
    KernelMatrixEditor,
    LinkedDimensionsEditor,
    PercentileRangeEditor,
    SliderNumberEditor,
    StripeSuppressionEditor,
    StructuringElementEditor,
)
from fdm.ui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox


class WorkbenchTaskKind(str, Enum):
    PREVIEW = "preview"
    FINAL = "final"


PREVIEW_MAX_EDGE = 2_048
PREVIEW_MAX_PIXELS = 4_000_000
PREVIEW_MAX_WORKING_SET_BYTES = 256 << 20
OVERVIEW_MAX_EDGE = 2_048
OVERVIEW_MAX_PIXELS = 2_000_000


class ProcessingPreviewView(QGraphicsView):
    """Zoomable raster preview without resampling the authoritative pixels."""

    zoomChanged = Signal(float, bool)

    MIN_ZOOM = 0.05
    MAX_ZOOM = 8.0
    ZOOM_STEP = 1.25

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = self._scene.addPixmap(QPixmap())
        self._fit_mode = True
        self.setScene(self._scene)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter
        )
        self.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            True,
        )
        self.setMinimumSize(240, 180)

    @property
    def fit_mode(self) -> bool:
        return self._fit_mode

    def image_size(self) -> tuple[int, int]:
        pixmap = self._pixmap_item.pixmap()
        return pixmap.width(), pixmap.height()

    def zoom_factor(self) -> float:
        return float(self.transform().m11())

    def set_image(
        self,
        image: QImage,
        *,
        force_fit: bool = False,
    ) -> None:
        pixmap = QPixmap.fromImage(image)
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        if force_fit:
            self.fit_image()
        elif self._fit_mode:
            self.fit_image()
        else:
            self._emit_zoom_changed()

    def fit_image(self) -> None:
        self._fit_mode = True
        self.resetTransform()
        scene_rect = self._scene.sceneRect()
        if not scene_rect.isEmpty():
            self.fitInView(
                scene_rect,
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        self._emit_zoom_changed()

    def actual_size(self) -> None:
        self.set_zoom_factor(1.0)

    def set_zoom_factor(self, factor: float) -> None:
        resolved = max(
            self.MIN_ZOOM,
            min(self.MAX_ZOOM, float(factor)),
        )
        self._fit_mode = False
        self.resetTransform()
        self.scale(resolved, resolved)
        self._emit_zoom_changed()

    def zoom_by(self, multiplier: float) -> None:
        self.set_zoom_factor(self.zoom_factor() * float(multiplier))

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_image()

    def wheelEvent(self, event: object) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = int(event.angleDelta().y())
            if delta:
                self.zoom_by(
                    self.ZOOM_STEP
                    if delta > 0
                    else 1.0 / self.ZOOM_STEP
                )
            event.accept()
            return
        super().wheelEvent(event)

    def _emit_zoom_changed(self) -> None:
        self.zoomChanged.emit(self.zoom_factor(), self._fit_mode)


@dataclass(frozen=True, slots=True)
class ProcessingPreviewSnapshot:
    """Bounded, immutable source material for an exact 1:1 preview.

    ``origin`` is expressed in full-source image pixels.  The preview raster is
    never resampled: one sample pixel always represents one authoritative
    source pixel.  Geometry operations may adapt their full-image coordinates
    to this local origin, but the persisted recipe remains unchanged.
    """

    source: RasterPlane
    origin: tuple[int, int]
    full_source_size: tuple[int, int]
    roi_mask: NDArray[np.bool_] | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    secondary_images: tuple[tuple[str, RasterPlane], ...] = field(
        default=(),
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        origin_x, origin_y = (int(self.origin[0]), int(self.origin[1]))
        full_width, full_height = (
            int(self.full_source_size[0]),
            int(self.full_source_size[1]),
        )
        if origin_x < 0 or origin_y < 0:
            raise ValueError("预览原点不能为负数")
        if full_width <= 0 or full_height <= 0:
            raise ValueError("预览完整源尺寸必须为正数")
        if (
            origin_x + self.source.width > full_width
            or origin_y + self.source.height > full_height
        ):
            raise ValueError("预览样本超出完整源图片范围")
        mask = self.roi_mask
        if mask is not None:
            normalized = np.ascontiguousarray(mask, dtype=np.bool_)
            if normalized.shape != (self.source.height, self.source.width):
                raise ValueError("预览 ROI 掩膜尺寸必须与预览样本一致")
            normalized.setflags(write=False)
            object.__setattr__(self, "roi_mask", normalized)
        object.__setattr__(self, "origin", (origin_x, origin_y))
        object.__setattr__(self, "full_source_size", (full_width, full_height))
        object.__setattr__(self, "secondary_images", tuple(self.secondary_images))

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        x, y = self.origin
        return x, y, self.source.width, self.source.height

    @property
    def is_full_source(self) -> bool:
        return (
            self.origin == (0, 0)
            and (self.source.width, self.source.height) == self.full_source_size
        )


def build_processing_preview_snapshot(
    source: RasterPlane,
    *,
    visible_rect: tuple[float, float, float, float] | None = None,
    roi_mask: NDArray[np.bool_] | None = None,
    secondary_images: Mapping[str, RasterPlane] | None = None,
) -> ProcessingPreviewSnapshot:
    """Freeze a bounded, non-resampled sample for workbench preview.

    ``visible_rect`` uses ``(x, y, width, height)`` in source-pixel
    coordinates.  If the visible field is larger than the safety budget, a
    centred subset is selected; the overview explicitly shows that sample.
    """

    full_width = int(source.width)
    full_height = int(source.height)
    if visible_rect is None:
        left = 0
        top = 0
        requested_width = full_width
        requested_height = full_height
    else:
        raw_x, raw_y, raw_width, raw_height = (
            float(visible_rect[0]),
            float(visible_rect[1]),
            float(visible_rect[2]),
            float(visible_rect[3]),
        )
        if not all(
            math.isfinite(value)
            for value in (raw_x, raw_y, raw_width, raw_height)
        ):
            raise ValueError("预览视场必须使用有限像素坐标")
        if raw_width <= 0.0 or raw_height <= 0.0:
            raise ValueError("预览视场宽高必须为正数")
        left = max(0, min(full_width - 1, int(math.floor(raw_x))))
        top = max(0, min(full_height - 1, int(math.floor(raw_y))))
        right = max(
            left + 1,
            min(full_width, int(math.ceil(raw_x + raw_width))),
        )
        bottom = max(
            top + 1,
            min(full_height, int(math.ceil(raw_y + raw_height))),
        )
        requested_width = right - left
        requested_height = bottom - top

    sample_width = min(requested_width, PREVIEW_MAX_EDGE)
    sample_height = min(requested_height, PREVIEW_MAX_EDGE)
    if sample_width * sample_height > PREVIEW_MAX_PIXELS:
        scale = math.sqrt(
            PREVIEW_MAX_PIXELS / float(sample_width * sample_height)
        )
        sample_width = max(1, min(sample_width, int(math.floor(sample_width * scale))))
        sample_height = max(
            1,
            min(sample_height, int(math.floor(sample_height * scale))),
        )
    left += max(0, (requested_width - sample_width) // 2)
    top += max(0, (requested_height - sample_height) // 2)
    left = min(left, full_width - sample_width)
    top = min(top, full_height - sample_height)

    source_array = raster_plane_to_array(source)
    sample_array = np.ascontiguousarray(
        source_array[
            top : top + sample_height,
            left : left + sample_width,
            ...,
        ]
    )
    sample_plane = array_to_raster_plane(sample_array)

    sample_mask: NDArray[np.bool_] | None = None
    if roi_mask is not None:
        normalized_mask = np.asarray(roi_mask, dtype=np.bool_)
        if normalized_mask.shape != (full_height, full_width):
            raise ValueError("ROI 掩膜尺寸必须与完整源图片一致")
        sample_mask = np.ascontiguousarray(
            normalized_mask[
                top : top + sample_height,
                left : left + sample_width,
            ],
            dtype=np.bool_,
        )

    sampled_secondary: list[tuple[str, RasterPlane]] = []
    for document_id, plane in (secondary_images or {}).items():
        if (plane.width, plane.height) != (full_width, full_height):
            raise ValueError("第二幅图像必须与源图片尺寸一致")
        array = raster_plane_to_array(plane)
        sampled_secondary.append(
            (
                str(document_id),
                array_to_raster_plane(
                    np.ascontiguousarray(
                        array[
                            top : top + sample_height,
                            left : left + sample_width,
                            ...,
                        ]
                    )
                ),
            )
        )
    return ProcessingPreviewSnapshot(
        source=sample_plane,
        origin=(left, top),
        full_source_size=(full_width, full_height),
        roi_mask=sample_mask,
        secondary_images=tuple(sampled_secondary),
    )


def expand_processing_preview_snapshot_for_halo(
    snapshot: ProcessingPreviewSnapshot,
    *,
    full_source: RasterPlane,
    full_roi_mask: NDArray[np.bool_] | None,
    full_secondary_images: Mapping[str, RasterPlane],
    halo_x: int,
    halo_y: int,
) -> tuple[ProcessingPreviewSnapshot, tuple[int, int, int, int] | None]:
    """Read real neighbour pixels around a bounded 1:1 preview sample.

    The returned crop is expressed in the expanded result.  Callers must crop
    the processed output back to that rectangle before displaying it.  This is
    intentionally refused when the exact expanded sample would exceed the
    preview safety budget; silently clipping the halo would make filter edges
    disagree with final full-resolution processing.
    """

    hx = max(0, int(halo_x))
    hy = max(0, int(halo_y))
    if (hx == 0 and hy == 0) or snapshot.is_full_source:
        return snapshot, None
    base_x, base_y, base_width, base_height = snapshot.bounds
    left = max(0, base_x - hx)
    top = max(0, base_y - hy)
    right = min(full_source.width, base_x + base_width + hx)
    bottom = min(full_source.height, base_y + base_height + hy)
    expanded_width = right - left
    expanded_height = bottom - top
    if (
        expanded_width > PREVIEW_MAX_EDGE
        or expanded_height > PREVIEW_MAX_EDGE
        or expanded_width * expanded_height > PREVIEW_MAX_PIXELS
    ):
        raise ValueError(
            "当前视场加上处理邻域后超过 2048×2048 / 4MP 的精确预览上限；"
            "请缩小画布视场后重新打开工作台。"
        )
    source_array = raster_plane_to_array(full_source)
    expanded_source = array_to_raster_plane(
        np.ascontiguousarray(
            source_array[top:bottom, left:right, ...]
        )
    )
    expanded_mask: NDArray[np.bool_] | None = None
    if full_roi_mask is not None:
        normalized_mask = np.asarray(full_roi_mask, dtype=np.bool_)
        if normalized_mask.shape != (full_source.height, full_source.width):
            raise ValueError("ROI 掩膜尺寸必须与完整源图片一致")
        expanded_mask = np.ascontiguousarray(
            normalized_mask[top:bottom, left:right],
            dtype=np.bool_,
        )
    expanded_secondary: list[tuple[str, RasterPlane]] = []
    for document_id, plane in full_secondary_images.items():
        if (plane.width, plane.height) != (
            full_source.width,
            full_source.height,
        ):
            raise ValueError("第二幅图像必须与源图片尺寸一致")
        array = raster_plane_to_array(plane)
        expanded_secondary.append(
            (
                str(document_id),
                array_to_raster_plane(
                    np.ascontiguousarray(
                        array[top:bottom, left:right, ...]
                    )
                ),
            )
        )
    expanded = ProcessingPreviewSnapshot(
        source=expanded_source,
        origin=(left, top),
        full_source_size=(full_source.width, full_source.height),
        roi_mask=expanded_mask,
        secondary_images=tuple(expanded_secondary),
    )
    return expanded, (
        base_x - left,
        base_y - top,
        base_width,
        base_height,
    )


def adapt_operations_for_preview(
    snapshot: ProcessingPreviewSnapshot,
    operations: tuple[ImageOperationSpec, ...],
) -> tuple[ImageOperationSpec, ...]:
    """Translate full-image geometry parameters to a bounded preview sample.

    The returned operations are ephemeral and are never persisted.  Pixel
    values for point and neighbourhood operations are unchanged.  Coordinates
    are adjusted only where the workbench exposes full-image positions.
    """

    origin_x, origin_y = snapshot.origin
    full_width, full_height = snapshot.full_source_size
    sample_width = snapshot.source.width
    sample_height = snapshot.source.height
    transformed: list[ImageOperationSpec] = []
    coordinate_space_is_source = True
    for operation in operations:
        parameters = operation.parameters
        operation_id = operation.operation_id
        if operation_id == ImageOperation.CROP.value:
            crop_x = int(parameters.get("x", 0))
            crop_y = int(parameters.get("y", 0))
            crop_width = int(parameters.get("width", full_width))
            crop_height = int(parameters.get("height", full_height))
            if coordinate_space_is_source:
                intersection_left = max(origin_x, crop_x)
                intersection_top = max(origin_y, crop_y)
                intersection_right = min(
                    origin_x + sample_width,
                    crop_x + crop_width,
                )
                intersection_bottom = min(
                    origin_y + sample_height,
                    crop_y + crop_height,
                )
                if (
                    intersection_right <= intersection_left
                    or intersection_bottom <= intersection_top
                ):
                    raise ValueError(
                        "当前 1:1 预览样本与裁剪区域不相交；"
                        "请在画布中移动到裁剪区域后重开工作台。"
                    )
                parameters.update(
                    {
                        "x": intersection_left - origin_x,
                        "y": intersection_top - origin_y,
                        "width": intersection_right - intersection_left,
                        "height": intersection_bottom - intersection_top,
                    }
                )
                sample_width = int(parameters["width"])
                sample_height = int(parameters["height"])
                full_width = crop_width
                full_height = crop_height
                origin_x = 0
                origin_y = 0
            coordinate_space_is_source = False
        elif operation_id == ImageOperation.RESIZE.value:
            target_width = max(1, int(parameters.get("width", full_width)))
            target_height = max(1, int(parameters.get("height", full_height)))
            parameters["width"] = max(
                1,
                int(round(sample_width * target_width / max(1, full_width))),
            )
            parameters["height"] = max(
                1,
                int(round(sample_height * target_height / max(1, full_height))),
            )
            sample_width = int(parameters["width"])
            sample_height = int(parameters["height"])
            full_width = target_width
            full_height = target_height
            coordinate_space_is_source = False
        elif operation_id == ImageOperation.PIXEL_BIN.value:
            factor = max(1, int(parameters.get("factor", 1)))
            if str(parameters.get("remainder_policy", "reject")) == "reject":
                # A full image can be divisible while a centred preview is not.
                # Cropping the sample edge affects preview only and is recorded
                # nowhere in the authoritative recipe.
                parameters["remainder_policy"] = "crop"
            sample_width = max(1, sample_width // factor)
            sample_height = max(1, sample_height // factor)
            full_width = max(1, full_width // factor)
            full_height = max(1, full_height // factor)
            coordinate_space_is_source = False
        elif operation_id in {
            ImageOperation.ROTATE_90_CLOCKWISE.value,
            ImageOperation.ROTATE_90_COUNTERCLOCKWISE.value,
        }:
            sample_width, sample_height = sample_height, sample_width
            full_width, full_height = full_height, full_width
            coordinate_space_is_source = False
        elif operation_id in {
            ImageOperation.ROTATE.value,
            ImageOperation.TRANSLATE.value,
            ImageOperation.RESIZE_CANVAS.value,
            ImageOperation.FLIP_HORIZONTAL.value,
            ImageOperation.FLIP_VERTICAL.value,
            ImageOperation.ROTATE_180.value,
        }:
            coordinate_space_is_source = False
        transformed.append(
            ImageOperationSpec(
                operation_id,
                parameters,
                implementation=operation.implementation,
                implementation_version=operation.implementation_version,
                result_metadata=operation.result_metadata,
            )
        )
    return tuple(transformed)


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
    capture_step_input_index: int | None = None
    source_semantic: RasterSemantic | None = None
    secondary_semantics: tuple[tuple[str, RasterSemantic], ...] = field(
        default=(),
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.operations:
            raise ValueError("图像处理任务至少需要一个步骤")
        source_semantic = self.source_semantic
        if source_semantic is None:
            source_semantic = (
                RasterSemantic.INTENSITY
                if self.source.pixel_type.is_grayscale
                else RasterSemantic.COLOR
            )
        elif not isinstance(source_semantic, RasterSemantic):
            source_semantic = RasterSemantic(str(source_semantic))
        RasterTypeState(
            pixel_type=self.source.pixel_type,
            semantic=source_semantic,
        )
        object.__setattr__(self, "source_semantic", source_semantic)
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
        provided_secondary_semantics = {
            str(raw_id).strip(): semantic
            for raw_id, semantic in self.secondary_semantics
        }
        unknown_semantic_ids = (
            set(provided_secondary_semantics) - seen_secondary_ids
        )
        if unknown_semantic_ids:
            raise ValueError(
                "第二幅图像语义引用了不存在的文档 ID："
                + "、".join(sorted(unknown_semantic_ids))
            )
        normalized_secondary_semantics: list[
            tuple[str, RasterSemantic]
        ] = []
        for document_id, plane in secondary_images:
            semantic = RasterTypeState(
                pixel_type=plane.pixel_type,
                semantic=provided_secondary_semantics.get(document_id),
            ).semantic
            normalized_secondary_semantics.append(
                (document_id, semantic)
            )
        object.__setattr__(
            self,
            "secondary_semantics",
            tuple(normalized_secondary_semantics),
        )
        capture_index = self.capture_step_input_index
        if capture_index is not None:
            capture_index = int(capture_index)
            if not 0 <= capture_index < len(self.operations):
                raise ValueError("参数输入快照步骤超出处理配方范围")
            object.__setattr__(
                self,
                "capture_step_input_index",
                capture_index,
            )

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
    output_semantic: RasterSemantic | None = None
    parameter_input_raster: RasterPlane | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    parameter_input_roi_mask: NDArray[np.bool_] | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    parameter_input_step_index: int | None = None


@dataclass(frozen=True, slots=True)
class _TaskCompletion:
    request: WorkbenchTaskRequest
    raster: RasterPlane | None = None
    recipe: ImageProcessingRecipe | None = None
    output_semantic: RasterSemantic | None = None
    parameter_input_raster: RasterPlane | None = None
    parameter_input_roi_mask: NDArray[np.bool_] | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    error: str | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class _WorkbenchExecutionOutput:
    raster: RasterPlane
    recipe: ImageProcessingRecipe
    output_semantic: RasterSemantic
    parameter_input_raster: RasterPlane | None = None
    parameter_input_roi_mask: NDArray[np.bool_] | None = field(
        default=None,
        compare=False,
        repr=False,
    )


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
                    output_semantic=execution.output_semantic,
                    parameter_input_raster=(
                        execution.parameter_input_raster
                    ),
                    parameter_input_roi_mask=(
                        execution.parameter_input_roi_mask
                    ),
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
        source_semantic: RasterSemantic | None = None,
        roi_mask: NDArray[np.bool_] | None = None,
        secondary_images: Mapping[str, RasterPlane] | None = None,
        capture_step_input_index: int | None = None,
        secondary_semantics: Mapping[str, RasterSemantic] | None = None,
    ) -> WorkbenchTaskRequest:
        return self._start(
            WorkbenchTaskKind.PREVIEW,
            source_document_id=source_document_id,
            source=source,
            operations=operations,
            source_semantic=source_semantic,
            roi_mask=roi_mask,
            secondary_images=secondary_images,
            capture_step_input_index=capture_step_input_index,
            secondary_semantics=secondary_semantics,
        )

    def start_final(
        self,
        *,
        source_document_id: str,
        source: RasterPlane,
        operations: tuple[ImageOperationSpec, ...],
        source_semantic: RasterSemantic | None = None,
        roi_mask: NDArray[np.bool_] | None = None,
        secondary_images: Mapping[str, RasterPlane] | None = None,
        secondary_semantics: Mapping[str, RasterSemantic] | None = None,
    ) -> WorkbenchTaskRequest:
        return self._start(
            WorkbenchTaskKind.FINAL,
            source_document_id=source_document_id,
            source=source,
            operations=operations,
            source_semantic=source_semantic,
            roi_mask=roi_mask,
            secondary_images=secondary_images,
            capture_step_input_index=None,
            secondary_semantics=secondary_semantics,
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
        source_semantic: RasterSemantic | None,
        roi_mask: NDArray[np.bool_] | None,
        secondary_images: Mapping[str, RasterPlane] | None,
        capture_step_input_index: int | None,
        secondary_semantics: Mapping[str, RasterSemantic] | None,
    ) -> WorkbenchTaskRequest:
        if self._closed:
            raise RuntimeError("图像处理任务控制器已经关闭")
        validate_workbench_operation_sequence(
            source,
            operations,
            source_semantic=source_semantic,
            roi_requested=roi_mask is not None,
            secondary_images=secondary_images,
            secondary_semantics=secondary_semantics,
        )
        lane = self._lanes[kind]
        lane.generation += 1
        request = WorkbenchTaskRequest(
            kind=kind,
            request_id=uuid4().hex,
            generation=lane.generation,
            source_document_id=str(source_document_id),
            source=source,
            operations=tuple(operations),
            source_semantic=source_semantic,
            roi_mask=roi_mask,
            secondary_images=tuple((secondary_images or {}).items()),
            capture_step_input_index=capture_step_input_index,
            secondary_semantics=tuple(
                (secondary_semantics or {}).items()
            ),
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
                    output_semantic=completion.output_semantic,
                    parameter_input_raster=(
                        completion.parameter_input_raster
                    ),
                    parameter_input_roi_mask=(
                        completion.parameter_input_roi_mask
                    ),
                    parameter_input_step_index=(
                        request.capture_step_input_index
                    ),
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
    validation = validate_workbench_operation_sequence(
        request.source,
        request.operations,
        source_semantic=request.source_semantic,
        roi_requested=request.roi_mask is not None,
        secondary_images=dict(request.secondary_images),
        secondary_semantics=dict(request.secondary_semantics),
    )
    image = raster_plane_to_array(request.source)
    working_roi_mask = request.roi_mask
    secondary_images = dict(request.secondary_images)
    executed_operations: list[ImageOperationSpec] = []
    parameter_input_raster: RasterPlane | None = None
    parameter_input_roi_mask: NDArray[np.bool_] | None = None
    dynamic_metadata_keys = {
        "nonfinite_replacement_count",
        "repaired_count",
        "computed_threshold",
        "cropped_right",
        "cropped_bottom",
        "roi_bounds",
    }
    for operation_index, operation_spec in enumerate(request.operations):
        token.raise_if_cancelled()
        if request.capture_step_input_index == operation_index:
            parameter_input_raster = array_to_raster_plane(
                np.ascontiguousarray(image)
            )
            if working_roi_mask is not None:
                parameter_input_roi_mask = np.ascontiguousarray(
                    working_roi_mask,
                    dtype=np.bool_,
                )
                parameter_input_roi_mask.setflags(write=False)
        parameters = operation_spec.parameters
        if (
            operation_spec.operation_id == ImageOperation.RESIZE.value
            and str(parameters.get("interpolation", "auto")).strip().lower()
            == InterpolationMode.AUTO.value
        ):
            validation_step = validation.steps[operation_index]
            parameters["interpolation"] = resolve_resize_interpolation(
                source_width=int(image.shape[1]),
                source_height=int(image.shape[0]),
                width=int(parameters.get("width", image.shape[1])),
                height=int(parameters.get("height", image.shape[0])),
                requested=InterpolationMode.AUTO,
                semantic=validation_step.input_state.semantic,
            ).value
        secondary_image = None
        flat_field_source = str(
            parameters.get("flat_field_source", "estimated")
        ).strip().lower()
        uses_secondary_image = (
            operation_spec.operation_id
            == ImageOperation.IMAGE_CALCULATOR.value
            or (
                operation_spec.operation_id
                == ImageOperation.FLAT_FIELD_CORRECTION.value
                and flat_field_source == "reference"
            )
        )
        if uses_secondary_image:
            secondary_document_id = str(
                parameters.get("secondary_document_id", "")
            ).strip()
            try:
                secondary_plane = secondary_images[secondary_document_id]
            except KeyError as exc:
                message = (
                    "参考图平场校正选择的第二幅参考图像已不可用"
                    if operation_spec.operation_id
                    == ImageOperation.FLAT_FIELD_CORRECTION.value
                    else "图像计算器选择的第二幅图像已不可用"
                )
                raise ValueError(message) from exc
            if (
                operation_spec.operation_id
                == ImageOperation.FLAT_FIELD_CORRECTION.value
            ):
                actual_sha256 = secondary_plane.sha256()
                expected_sha256 = str(
                    parameters.get("secondary_sha256", "")
                ).strip()
                if (
                    request.kind is WorkbenchTaskKind.FINAL
                    and expected_sha256
                    and expected_sha256 != actual_sha256
                ):
                    raise ValueError(
                        "参考平场像素已变化，已拒绝使用过期的参考图执行最终处理"
                    )
                if request.kind is WorkbenchTaskKind.FINAL:
                    parameters["secondary_sha256"] = actual_sha256
                if not parameters.get("reference_levels"):
                    parameters["reference_levels"] = (
                        flat_field_reference_levels(
                            raster_plane_to_array(secondary_plane),
                            preserve_mean=bool(
                                parameters.get("preserve_mean", True)
                            ),
                        )
                    )
            secondary_image = raster_plane_to_array(secondary_plane)
        result = execute_image_operation_tiled(
            ImageOperation(operation_spec.operation_id),
            image,
            parameters=parameters,
            secondary_image=secondary_image,
            roi_mask=working_roi_mask,
            request_id=request.request_id,
            generation=request.generation,
            tile_size=PROCESSING_TILE_EDGE,
            cancellation_check=token.raise_if_cancelled,
        )
        token.raise_if_cancelled()
        image = np.asarray(result.image)
        if result.roi_mask is not None:
            # Geometry-changing operations own the transformed ROI mask.  Do
            # not repeat their clipping arithmetic here: the service result is
            # the authoritative mask for the next step in the recipe.
            working_roi_mask = result.roi_mask
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
        output_semantic=validation.output_state.semantic,
        parameter_input_raster=parameter_input_raster,
        parameter_input_roi_mask=parameter_input_roi_mask,
    )


def validate_workbench_operation_sequence(
    source: RasterPlane,
    operations: tuple[ImageOperationSpec, ...],
    *,
    source_semantic: RasterSemantic | None = None,
    roi_requested: bool = False,
    secondary_images: Mapping[str, RasterPlane] | None = None,
    secondary_semantics: Mapping[str, RasterSemantic] | None = None,
) -> RecipeValidationResult:
    """Validate the complete pixel/semantic chain before allocating output."""

    source_state = RasterTypeState(
        pixel_type=source.pixel_type,
        semantic=source_semantic,
        width=source.width,
        height=source.height,
    )
    secondary_states = {
        document_id: RasterTypeState(
            pixel_type=plane.pixel_type,
            semantic=(secondary_semantics or {}).get(document_id),
            width=plane.width,
            height=plane.height,
        )
        for document_id, plane in (secondary_images or {}).items()
    }
    return validate_image_processing_recipe(
        ImageProcessingRecipe.from_operations(operations),
        source_state,
        roi_requested=roi_requested,
        secondary_states=secondary_states,
    )


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
        "watershed_v2",
        "adaptive_threshold",
        "morphological_reconstruction",
        "regional_extrema",
        "clear_border",
        "fft_power_spectrum",
        "top_hat",
        "black_hat",
    }
    float_outputs = {"distance_transform", "fft_power_spectrum"}
    optionally_float_outputs = {"sobel_edges", "laplacian_edges", "fft_filter"}
    frequency_operations = {
        "fft_filter",
        "fft_power_spectrum",
        "stripe_suppression",
    }
    global_label_operations = {
        "remove_small_objects",
        "fill_small_holes",
        "watershed",
        "watershed_v2",
        "morphological_reconstruction",
        "regional_extrema",
        "clear_border",
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
            if str(parameters.get("result_mode", "preserve")) == "float32":
                bytes_per_channel = 4
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
        elif operation_id in {"log_v2", "exp_v2", "sqrt_v2"} and (
            str(parameters.get("result_mode", "float32")) == "float32"
        ):
            bytes_per_channel = 4
        elif operation_id == "rank_filter" and (
            str(parameters.get("method", "minimum")) == "variance"
        ):
            bytes_per_channel = 4
        elif operation_id == "morphology_derivative" and (
            str(parameters.get("method", "gradient")) == "laplacian"
        ):
            channels = 1
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
    available_for_new_recipe: bool = True


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
        ("自动（缩小 Area／放大 Bilinear）", "auto"),
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
        available_for_new_recipe: bool = True,
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
            available_for_new_recipe=available_for_new_recipe,
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
            ImageOperation.COPY,
            "类型",
            "复制像素",
            (
                ParameterField(
                    "roi_mode",
                    "ROI 复制方式",
                    "choice",
                    "bounds",
                    choices=(
                        ("仅使用 ROI 包围框", "bounds"),
                        ("按 ROI 掩膜", "mask"),
                    ),
                ),
                ParameterField(
                    "outside_value",
                    "ROI 外填充值",
                    "float",
                    0.0,
                    -1e12,
                    1e12,
                    4,
                ),
                ParameterField(
                    "transparent_outside",
                    "ROI 外透明",
                    "bool",
                    False,
                ),
            ),
            purpose="冻结当前像素；有 ROI 时可复制包围框或精确掩膜。",
            roi_behavior="支持 ROI bounds/mask；结果携带冻结后的 ROI 掩膜。",
        ),
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
                    "preserve_values",
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
                ParameterField(
                    "roi_mode",
                    "ROI 裁剪方式",
                    "choice",
                    "bounds",
                    choices=(
                        ("仅使用 ROI 包围框", "bounds"),
                        ("按 ROI 掩膜", "mask"),
                    ),
                    help_text=(
                        "仅在存在当前 ROI 时生效；按掩膜会把包围框内、"
                        "ROI 外的像素设为指定填充值或透明。"
                    ),
                ),
                ParameterField(
                    "fill_value",
                    "ROI 外填充值",
                    "float",
                    0.0,
                    -1e12,
                    1e12,
                    4,
                ),
                ParameterField(
                    "transparent_outside",
                    "ROI 外透明",
                    "bool",
                    False,
                    help_text="开启后输出 RGBA8；只适用于 8 位源图。",
                ),
            ),
            purpose="截取原始像素坐标中的矩形或精确 ROI 包围区域。",
            roi_behavior=(
                "有 ROI 时默认按包围框裁剪；可显式选择按精确掩膜填充"
                " ROI 外区域。"
            ),
        ),
        define(
            ImageOperation.RESIZE,
            "变换",
            "调整图像大小",
            (
                ParameterField("width", "宽度", "int", -1, 1, 1_000_000),
                ParameterField("height", "高度", "int", -1, 1, 1_000_000),
                ParameterField("interpolation", "插值", "choice", "auto", choices=interpolation_modes),
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
            "分水岭分割 v1（兼容）",
            (
                foreground,
                ParameterField("seed_threshold", "种子阈值比例", "float", 0.45, 0.001, 0.999, 3),
                scalar_channel,
            ),
            purpose="按距离峰值拆分相互接触的二值对象。",
            supported_types=scalar_types,
            available_for_new_recipe=False,
        ),
        define(
            ImageOperation.WATERSHED_V2,
            "处理",
            "标记控制分水岭 v2",
            (
                foreground,
                ParameterField(
                    "seed_threshold",
                    "种子阈值比例",
                    "float",
                    0.35,
                    0.001,
                    0.999,
                    3,
                ),
                ParameterField(
                    "minimum_seed_area",
                    "最小种子面积",
                    "int",
                    1,
                    1,
                    2_147_483_647,
                    suffix=" px²",
                ),
                scalar_channel,
            ),
            purpose="使用平台安全区域极大值标记拆分接触对象；不改变旧版分水岭。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.BACKGROUND_SUBTRACT,
            "处理",
            "形态学背景扣除",
            (
                ParameterField("radius", "背景半径", "int", 25, 1, 2048, suffix=" px"),
                ParameterField("light_background", "亮背景", "bool", False),
                ParameterField("preserve_offset", "保留背景中位偏移", "bool", False),
                local_border,
            ),
            purpose="用滑动形态学背景估计扣除缓慢变化背景。",
        ),
        define(
            ImageOperation.ROLLING_BALL_BACKGROUND_SUBTRACT,
            "处理",
            "滑动抛物面背景扣除",
            (
                ParameterField(
                    "radius",
                    "抛物面半径",
                    "float",
                    25.0,
                    0.1,
                    2048,
                    2,
                    suffix=" px",
                ),
                ParameterField(
                    "ball_height",
                    "抛物面高度",
                    "float",
                    255.0,
                    0.001,
                    1e12,
                    3,
                ),
                ParameterField("light_background", "亮背景", "bool", False),
                ParameterField(
                    "preserve_offset",
                    "保留背景中位偏移",
                    "bool",
                    False,
                ),
            ),
            purpose="使用真正的灰度抛物面开/闭重建估计缓慢变化背景。",
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
        define(
            ImageOperation.LOG,
            "处理",
            "Log 运算 v1（兼容）",
            purpose="按旧版规则对非负像素计算 log(1+x)。",
            available_for_new_recipe=False,
        ),
        define(
            ImageOperation.EXP,
            "处理",
            "Exp 运算 v1（兼容）",
            purpose="按旧版规则计算指数；溢出时明确拒绝。",
            available_for_new_recipe=False,
        ),
        define(
            ImageOperation.SQRT,
            "处理",
            "Sqrt 运算 v1（兼容）",
            purpose="按旧版规则对非负像素计算平方根。",
            available_for_new_recipe=False,
        ),
        *tuple(
            define(
                operation,
                "处理",
                label,
                (
                    ParameterField(
                        "result_mode",
                        "结果类型",
                        "choice",
                        "float32",
                        choices=(
                            ("32 位浮点", "float32"),
                            ("保持输入类型", "preserve"),
                            ("重映射到指定范围", "remap"),
                        ),
                    ),
                    ParameterField(
                        "output_min",
                        "重映射下限",
                        "float",
                        0.0,
                        -1e12,
                        1e12,
                        4,
                    ),
                    ParameterField(
                        "output_max",
                        "重映射上限",
                        "float",
                        1.0,
                        -1e12,
                        1e12,
                        4,
                    ),
                ),
                purpose=purpose,
                supported_types="GRAY8、GRAY16、GRAY32_FLOAT；float32 输出仅单通道",
            )
            for operation, label, purpose in (
                (ImageOperation.LOG_V2, "Log 变换 v2", "科学型 log(1+x)，默认输出 float32。"),
                (ImageOperation.EXP_V2, "Exp 变换 v2", "科学型指数变换，默认输出 float32。"),
                (ImageOperation.SQRT_V2, "Sqrt 变换 v2", "科学型平方根变换，默认输出 float32。"),
            )
        ),
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
                        ("复制第二幅图", "copy"),
                    ),
                ),
                ParameterField(
                    "result_mode",
                    "结果类型",
                    "choice",
                    "preserve",
                    choices=(
                        ("保持输入类型", "preserve"),
                        ("32 位浮点", "float32"),
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
                ParameterField(
                    "boundary",
                    "边界策略",
                    "choice",
                    "mirror_pad",
                    choices=(
                        ("周期边界（旧版）", "periodic"),
                        ("镜像扩展", "mirror_pad"),
                        ("Tukey 窗", "tukey"),
                    ),
                ),
                ParameterField(
                    "tukey_alpha",
                    "Tukey alpha",
                    "float",
                    0.25,
                    0.0,
                    1.0,
                    3,
                ),
                ParameterField(
                    "frequency_unit",
                    "频率单位",
                    "choice",
                    "cycles_per_pixel",
                    choices=(
                        ("周期/像素", "cycles_per_pixel"),
                        ("周期/物理单位", "cycles_per_unit"),
                    ),
                ),
                ParameterField(
                    "pixel_size",
                    "像素物理尺寸",
                    "float",
                    1.0,
                    0.000000001,
                    1e12,
                    8,
                    help_text="选择周期/物理单位时必须显式提供。",
                ),
            ),
            purpose="使用 Butterworth 响应进行低通、高通、带通或带阻滤波。",
            roi_behavior="支持 ROI 写回；FFT 仍读取完整图像，ROI 外像素保持不变。",
        ),
        define(
            ImageOperation.FFT_POWER_SPECTRUM,
            "处理",
            "FFT 功率谱",
            (
                scalar_channel,
                ParameterField("logarithmic", "对数功率", "bool", True),
                ParameterField("centered", "零频居中", "bool", True),
                ParameterField(
                    "window",
                    "窗函数",
                    "choice",
                    "none",
                    choices=(("无", "none"), ("Tukey", "tukey")),
                ),
                ParameterField(
                    "tukey_alpha",
                    "Tukey alpha",
                    "float",
                    0.25,
                    0.0,
                    1.0,
                    3,
                ),
            ),
            purpose="生成未归一化的 float32 频域功率谱。",
            supported_types=scalar_types,
            available_for_new_recipe=False,
        ),
        define(
            ImageOperation.ADAPTIVE_THRESHOLD,
            "处理",
            "局部自适应阈值",
            (
                ParameterField(
                    "method",
                    "方法",
                    "choice",
                    "gaussian",
                    choices=(
                        ("Mean", "mean"),
                        ("Gaussian", "gaussian"),
                        ("Sauvola", "sauvola"),
                        ("Phansalkar", "phansalkar"),
                    ),
                ),
                ParameterField("radius", "半径", "int", 7, 1, 255),
                ParameterField("offset", "阈值偏移", "float", 0.0, -1e12, 1e12, 4),
                ParameterField("k", "k", "float", 0.2, -10.0, 10.0, 4),
                ParameterField("r", "R", "float", 128.0, 0.000001, 1e12, 4),
                ParameterField("p", "p", "float", 2.0, -100.0, 100.0, 4),
                ParameterField("q", "q", "float", 10.0, -100.0, 100.0, 4),
                foreground,
                scalar_channel,
            ),
            purpose="执行 Mean/Gaussian/Sauvola/Phansalkar 局部阈值。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.PERCENTILE_SATURATION,
            "处理",
            "百分位饱和增强",
            (
                ParameterField(
                    "lower_percentile",
                    "下百分位",
                    "float",
                    0.5,
                    0.0,
                    99.999,
                    3,
                ),
                ParameterField(
                    "upper_percentile",
                    "上百分位",
                    "float",
                    99.5,
                    0.001,
                    100.0,
                    3,
                ),
                ParameterField("per_channel", "逐通道", "bool", True),
            ),
            purpose="饱和直方图两端并映射到原生数值范围。",
        ),
        define(
            ImageOperation.RANK_FILTER,
            "处理",
            "Rank 滤波",
            (
                ParameterField(
                    "method",
                    "方法",
                    "choice",
                    "minimum",
                    choices=(
                        ("Minimum", "minimum"),
                        ("Maximum", "maximum"),
                        ("Variance", "variance"),
                    ),
                ),
                ParameterField("radius", "半径", "int", 1, 1, 255),
            ),
            purpose="执行最小值、最大值或局部总体方差 Rank 滤波。",
        ),
        define(
            ImageOperation.MORPHOLOGY_DERIVATIVE,
            "处理",
            "形态学微分",
            (
                ParameterField(
                    "method",
                    "方法",
                    "choice",
                    "gradient",
                    choices=(
                        ("Gradient", "gradient"),
                        ("Laplacian", "laplacian"),
                    ),
                ),
                ParameterField("radius", "半径", "int", 1, 1, 255),
                scalar_channel,
            ),
            purpose="计算形态学梯度或有符号 float32 Laplacian。",
        ),
        define(
            ImageOperation.MORPHOLOGICAL_RECONSTRUCTION,
            "处理",
            "形态学重建",
            (
                ParameterField(
                    "method",
                    "方法",
                    "choice",
                    "opening",
                    choices=(
                        ("开重建", "opening"),
                        ("闭重建", "closing"),
                    ),
                ),
                ParameterField("radius", "种子半径", "int", 1, 1, 255),
                ParameterField(
                    "connectivity",
                    "连通性",
                    "choice",
                    8,
                    choices=connectivity_modes,
                ),
                scalar_channel,
            ),
            purpose="执行受掩膜约束的测地膨胀/腐蚀重建。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.REGIONAL_EXTREMA,
            "处理",
            "区域/扩展极值",
            (
                ParameterField(
                    "kind",
                    "类型",
                    "choice",
                    "maxima",
                    choices=(("极大值", "maxima"), ("极小值", "minima")),
                ),
                ParameterField("h", "扩展高度 h", "float", 0.0, 0.0, 1e12, 4),
                ParameterField(
                    "connectivity",
                    "连通性",
                    "choice",
                    8,
                    choices=connectivity_modes,
                ),
                scalar_channel,
            ),
            purpose="提取区域极值；h>0 时提取扩展极值。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.CLEAR_BORDER,
            "处理",
            "清除边界对象",
            (
                foreground,
                ParameterField(
                    "connectivity",
                    "连通性",
                    "choice",
                    8,
                    choices=connectivity_modes,
                ),
                scalar_channel,
            ),
            purpose="删除与任一图像边界相连的前景对象。",
            supported_types=scalar_types,
        ),
        define(
            ImageOperation.FLAT_FIELD_CORRECTION,
            "处理",
            "平场校正",
            (
                ParameterField(
                    "flat_field_source",
                    "平场来源",
                    "choice",
                    "estimated",
                    choices=(
                        ("估算照明场", "estimated"),
                        ("选择参考图像", "reference"),
                    ),
                    help_text=(
                        "参考图像必须与源图像的尺寸、通道、像素类型和标定一致。"
                    ),
                ),
                ParameterField(
                    "secondary_document_id",
                    "参考图像",
                    "secondary_image",
                    "",
                    help_text=(
                        "候选列表仅显示尺寸、通道、像素类型和标定完全兼容的图片。"
                    ),
                ),
                ParameterField(
                    "radius",
                    "平场半径",
                    "float",
                    25.0,
                    0.1,
                    2048,
                    2,
                    suffix=" px",
                ),
                ParameterField(
                    "method",
                    "估计方法",
                    "choice",
                    "gaussian",
                    choices=(
                        ("Gaussian", "gaussian"),
                        ("形态学开运算", "morphology"),
                    ),
                ),
                ParameterField("preserve_mean", "保持平均亮度", "bool", True),
            ),
            purpose=(
                "用估算照明场或用户选择的参考图像执行乘性平场校正；"
                "参考图像内容摘要会写入派生配方。"
            ),
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


def _default_parameter_suffix(
    operation: ImageOperation,
    parameter_key: str,
) -> str:
    """Return presentation-only units where the service contract is unambiguous."""

    if parameter_key == "radius":
        return " px"
    if parameter_key in {
        "sigma",
        "sigma_x",
        "sigma_y",
        "sigma_space",
        "diameter",
    }:
        return " px"
    if parameter_key in {
        "lower_percentile",
        "upper_percentile",
    }:
        return " %"
    if parameter_key == "pixel_size":
        return " 单位/px"
    if operation in {
        ImageOperation.CROP,
        ImageOperation.RESIZE,
        ImageOperation.RESIZE_CANVAS,
    } and parameter_key in {"x", "y", "width", "height"}:
        return " px"
    if (
        operation is ImageOperation.CUSTOM_CONVOLUTION
        and parameter_key in {"kernel_width", "kernel_height"}
    ):
        return " px"
    return ""


def _bind_descriptor_parameter_schemas(
    catalog: tuple[WorkbenchOperationDefinition, ...],
) -> tuple[WorkbenchOperationDefinition, ...]:
    """Attach service-owned parameter constraints to UI presentation data.

    Labels, suffixes and help text remain presentation-only.  Kind, defaults,
    ranges and allowed values are taken from ``ImageOperationDescriptor`` so a
    widget can never advertise a value that recipe validation rejects.
    """

    bound: list[WorkbenchOperationDefinition] = []
    for definition in catalog:
        descriptor = get_image_operation_descriptor(definition.operation)
        schemas = {item.key: item for item in descriptor.parameter_schema}
        parameters: list[ParameterField] = []
        for presentation in definition.parameters:
            try:
                schema = schemas[presentation.key]
            except KeyError as exc:
                raise RuntimeError(
                    f"工作台参数 {definition.operation.value}."
                    f"{presentation.key} 未在服务注册表声明"
                ) from exc
            choice_labels = {
                value: label for label, value in presentation.choices
            }
            choices = tuple(
                (choice_labels.get(value, str(value)), value)
                for value in schema.choices
            )
            parameters.append(
                replace(
                    presentation,
                    kind=schema.kind,
                    default=schema.default,
                    minimum=schema.minimum,
                    maximum=schema.maximum,
                    choices=choices,
                    suffix=(
                        presentation.suffix
                        or _default_parameter_suffix(
                            definition.operation,
                            presentation.key,
                        )
                    ),
                )
            )
        bound.append(
            replace(definition, parameters=tuple(parameters))
        )
    return tuple(bound)


_OPERATION_CATALOG = _bind_descriptor_parameter_schemas(
    _operation_catalog()
)
_DEFINITION_BY_ID = {
    definition.operation.value: definition for definition in _OPERATION_CATALOG
}

_EXPLICIT_BINARY_RECOMMENDED_OPERATIONS = {
    ImageOperation.FILL_HOLES.value,
    ImageOperation.CONTOUR_EXTRACT.value,
    ImageOperation.REMOVE_SMALL_OBJECTS.value,
    ImageOperation.FILL_SMALL_HOLES.value,
    ImageOperation.DISTANCE_TRANSFORM.value,
    ImageOperation.SKELETONIZE.value,
    ImageOperation.WATERSHED.value,
    ImageOperation.WATERSHED_V2.value,
    ImageOperation.CLEAR_BORDER.value,
}
_VERSIONED_REPLAY_OPERATIONS = (
    _EXPLICIT_BINARY_RECOMMENDED_OPERATIONS
    | {
        ImageOperation.BRIGHTNESS_CONTRAST.value,
        ImageOperation.HISTOGRAM_EQUALIZATION.value,
        ImageOperation.IMAGE_CALCULATOR.value,
        ImageOperation.FLAT_FIELD_CORRECTION.value,
        ImageOperation.ROTATE.value,
        ImageOperation.TRANSLATE.value,
        ImageOperation.RESIZE.value,
    }
) - {ImageOperation.WATERSHED.value}


def _operation_step_is_replay_only(step: ImageOperationSpec) -> bool:
    definition = _DEFINITION_BY_ID[step.operation_id]
    if step.implementation != "fdm":
        return True
    if not definition.available_for_new_recipe:
        return True
    if step.operation_id not in _VERSIONED_REPLAY_OPERATIONS:
        return False
    descriptor = get_image_operation_descriptor(step.operation_id)
    return step.implementation_version != descriptor.version

_HISTOGRAM_EDITOR_PARAMETERS: dict[str, tuple[str, ...]] = {
    ImageOperation.ADJUST_LEVELS.value: ("black_point", "white_point"),
    ImageOperation.THRESHOLD.value: ("lower", "upper"),
    ImageOperation.BINARIZE.value: ("threshold",),
    ImageOperation.CANNY_EDGES.value: (
        "threshold_low",
        "threshold_high",
    ),
}

_PERCENTILE_RANGE_PARAMETERS: dict[str, tuple[str, ...]] = {
    ImageOperation.PERCENTILE_SATURATION.value: (
        "lower_percentile",
        "upper_percentile",
    ),
}

_CROP_BOUNDS_PARAMETERS: dict[str, tuple[str, ...]] = {
    ImageOperation.CROP.value: ("x", "y", "width", "height"),
}

_LINKED_DIMENSION_OPERATIONS = {
    ImageOperation.RESIZE,
    ImageOperation.RESIZE_CANVAS,
}

_STRUCTURING_ELEMENT_OPERATIONS = {
    ImageOperation.ERODE,
    ImageOperation.DILATE,
    ImageOperation.MORPHOLOGY_OPEN,
    ImageOperation.MORPHOLOGY_CLOSE,
    ImageOperation.TOP_HAT,
    ImageOperation.BLACK_HAT,
}

_FREQUENCY_RESPONSE_PARAMETERS: dict[str, tuple[str, ...]] = {
    ImageOperation.FFT_FILTER.value: (
        "mode",
        "low_cutoff",
        "high_cutoff",
        "order",
    ),
}

_STRIPE_FREQUENCY_PARAMETERS: dict[str, tuple[str, ...]] = {
    ImageOperation.STRIPE_SUPPRESSION.value: (
        "direction",
        "notch_width",
        "protect_radius",
    ),
}

_CONDITIONAL_PARAMETER_FIELDS: dict[str, tuple[str, ...]] = {
    ImageOperation.CONVERT_TYPE.value: (
        "target_type",
        "scale_mode",
        "nonfinite_policy",
    ),
    ImageOperation.CONVERT_COLOR.value: (
        "target_model",
        "grayscale_method",
        "drop_alpha",
    ),
    ImageOperation.ROTATE.value: ("border_mode", "border_value"),
    ImageOperation.TRANSLATE.value: ("border_mode", "border_value"),
    ImageOperation.ADAPTIVE_THRESHOLD.value: (
        "method",
        "k",
        "r",
        "p",
        "q",
    ),
    ImageOperation.LOG_V2.value: (
        "result_mode",
        "output_min",
        "output_max",
    ),
    ImageOperation.EXP_V2.value: (
        "result_mode",
        "output_min",
        "output_max",
    ),
    ImageOperation.SQRT_V2.value: (
        "result_mode",
        "output_min",
        "output_max",
    ),
    ImageOperation.FFT_FILTER.value: (
        "mode",
        "low_cutoff",
        "high_cutoff",
        "boundary",
        "tukey_alpha",
        "frequency_unit",
        "pixel_size",
    ),
    ImageOperation.FLAT_FIELD_CORRECTION.value: (
        "flat_field_source",
        "secondary_document_id",
        "radius",
        "method",
    ),
}

_COMMON_PARAMETER_HELP: dict[str, str] = {
    "channel": "彩色图必须明确选择参与计算的标量通道；灰度图忽略此项。",
    "border_mode": "定义邻域越过图片边缘时的取样方式，并写入处理配方。",
    "border_value": "仅在常量边界模式下使用，数值单位与当前像素强度一致。",
    "interpolation": "定义重采样算法；自动模式会在执行前解析为确定算法。",
    "radius": "以当前步骤输入的原始像素为单位；实际邻域直径通常为 2×半径+1。",
    "iterations": "重复执行相同操作的次数；多次小核不等同于任意形状的大核。",
    "kernel": "选择结构元素形状；形状会影响边缘、细线和角点的处理结果。",
    "gamma": "无量纲 Gamma；1 保持线性，小于 1 与大于 1 会产生相反的非线性映射。",
    "foreground_is_high": "明确二值前景位于高值端还是低值端，不依赖主题或 LUT 推断。",
    "output_float": "开启后保留浮点计算结果；关闭时按输入类型的明确规则恢复。",
    "tukey_alpha": "Tukey 窗过渡比例，仅在选择 Tukey 边界策略时参与计算。",
    "pixel_size": "每个像素对应的物理长度；用于把物理频率换算为周期/像素。",
    "nonfinite_policy": "从浮点转为整数前，明确指定 NaN/Inf 的替代规则。",
    "secondary_document_id": "选择参与双图计算的第二幅图像；尺寸、类型和通道必须兼容。",
}

_OPERATION_PARAMETER_HELP: dict[tuple[str, str], str] = {
    (
        ImageOperation.ADAPTIVE_THRESHOLD.value,
        "method",
    ): "Mean、Gaussian、Sauvola 和 Phansalkar 使用不同的局部阈值公式。",
    (
        ImageOperation.ADAPTIVE_THRESHOLD.value,
        "offset",
    ): "从局部阈值中减去的原始强度值；不是百分比。",
    (
        ImageOperation.ADAPTIVE_THRESHOLD.value,
        "k",
    ): "Sauvola/Phansalkar 的无量纲对比度系数。",
    (
        ImageOperation.ADAPTIVE_THRESHOLD.value,
        "r",
    ): "Sauvola/Phansalkar 的动态范围尺度 R，单位与当前原始像素强度一致。",
    (
        ImageOperation.ADAPTIVE_THRESHOLD.value,
        "p",
    ): "Phansalkar 低亮度修正系数 p，仅该方法使用。",
    (
        ImageOperation.ADAPTIVE_THRESHOLD.value,
        "q",
    ): "Phansalkar 指数衰减系数 q，仅该方法使用。",
    (
        ImageOperation.FFT_FILTER.value,
        "low_cutoff",
    ): "高通、带通和带阻的低截止频率；响应曲线仅作参数说明。",
    (
        ImageOperation.FFT_FILTER.value,
        "high_cutoff",
    ): "低通、带通和带阻的高截止频率，不得超过 Nyquist 频率。",
    (
        ImageOperation.FFT_FILTER.value,
        "order",
    ): "Butterworth 阶数；阶数越高，截止附近过渡越陡。",
    (
        ImageOperation.PERCENTILE_SATURATION.value,
        "lower_percentile",
    ): "低端累计百分位；实际解析出的强度值会由当前输入计算。",
    (
        ImageOperation.PERCENTILE_SATURATION.value,
        "upper_percentile",
    ): "高端累计百分位，必须大于低端百分位。",
}


def _parameter_help_text(
    definition: WorkbenchOperationDefinition,
    parameter: ParameterField,
) -> str:
    """Return useful Chinese guidance for every visible parameter."""

    if parameter.help_text:
        return parameter.help_text
    specific = _OPERATION_PARAMETER_HELP.get(
        (definition.operation.value, parameter.key)
    )
    if specific:
        return specific
    common = _COMMON_PARAMETER_HELP.get(parameter.key)
    if common:
        return common
    if parameter.kind == "bool":
        return f"切换“{parameter.label}”；最终状态会明确写入可追溯配方。"
    if parameter.kind == "choice":
        options = "、".join(
            label for label, _value in parameter.choices
        )
        return (
            f"选择{parameter.label}"
            + (f"：{options}。" if options else "。")
            + "所选枚举值会写入配方。"
        )
    if parameter.kind in {"int", "float"}:
        range_text = ""
        if (
            parameter.minimum is not None
            and parameter.maximum is not None
        ):
            range_text = (
                f"允许范围 {parameter.minimum:g}–"
                f"{parameter.maximum:g}{parameter.suffix}；"
            )
        return (
            f"{range_text}{parameter.label}使用精确数值保存，"
            "普通鼠标滚轮不会修改。"
        )
    return f"{parameter.label}会以显式值写入可追溯处理配方。"


def _operation_parameter_context_message(
    definition: WorkbenchOperationDefinition,
    input_state: RasterTypeState,
) -> str:
    """Return operation-level scientific context that should remain visible."""

    operation = definition.operation
    if operation is ImageOperation.ADAPTIVE_THRESHOLD:
        return (
            f"局部阈值按当前 {input_state.pixel_type.value} 原始强度计算；"
            "窗口尺寸为 2×半径+1。R=128 只天然对应常见 8 位尺度，"
            "16 位或 float32 输入应显式核对 R，本版本不会静默归一化。"
        )
    if operation in _STRUCTURING_ELEMENT_OPERATIONS:
        if input_state.semantic is RasterSemantic.BINARY_MASK:
            return (
                "当前输入语义为二值掩膜：结构元素作用于显式前景；"
                "半径和迭代次数分别参与计算。"
            )
        return (
            "当前输入语义为连续灰度/颜色：腐蚀、膨胀及顶帽按局部"
            "最小/最大强度运算，不等同于二值形态学。"
        )
    if operation is ImageOperation.FFT_FILTER:
        return (
            "频率上限 0.5 周期/像素即 Nyquist；响应图只解释参数，"
            "不会自动拉伸输出强度。边界策略、通道和输出类型仍显式保存。"
        )
    if operation is ImageOperation.CUSTOM_CONVOLUTION:
        return (
            "卷积核按二维矩阵逐元素计算；启用归一化时使用核元素和，"
            "零和核会明确阻止归一化。宽和高必须为正奇数，"
            "边界策略保持显式。"
        )
    return ""


def _parameter_is_relevant(
    operation_id: str,
    parameter_key: str,
    parameters: Mapping[str, object],
    *,
    input_state: RasterTypeState,
    roi_available: bool,
) -> bool:
    """Return whether a parameter participates in the current algorithm branch.

    Hidden values remain in the immutable recipe so switching a method back
    restores the user's previous value.  Visibility therefore changes only the
    editor, never persisted or execution semantics.
    """

    if parameter_key == "channel" and input_state.pixel_type.is_grayscale:
        return False
    if parameter_key == "per_channel" and input_state.pixel_type.is_grayscale:
        return False
    if parameter_key == "roi_mode":
        return roi_available
    if operation_id in {
        ImageOperation.COPY.value,
        ImageOperation.CROP.value,
    } and parameter_key in {
        "outside_value",
        "fill_value",
        "transparent_outside",
    }:
        return roi_available and parameters.get("roi_mode", "bounds") == "mask"
    if (
        operation_id == ImageOperation.CONVERT_TYPE.value
        and parameter_key == "nonfinite_policy"
    ):
        return (
            input_state.pixel_type is RasterPixelType.GRAY32_FLOAT
            and parameters.get("target_type", "uint8") != "float32"
        )
    if operation_id == ImageOperation.CONVERT_COLOR.value:
        if parameter_key == "grayscale_method":
            return (
                not input_state.pixel_type.is_grayscale
                and parameters.get("target_model", "grayscale") == "grayscale"
            )
        if parameter_key == "drop_alpha":
            return input_state.pixel_type is RasterPixelType.RGBA8
    if (
        operation_id
        in {
            ImageOperation.THRESHOLD.value,
            ImageOperation.BINARIZE.value,
        }
        and parameter_key == "invert"
    ):
        return False
    if (
        operation_id
        in {
            ImageOperation.ROTATE.value,
            ImageOperation.TRANSLATE.value,
        }
        and parameter_key == "border_value"
    ):
        return parameters.get("border_mode", "constant") == "constant"
    if operation_id == ImageOperation.ADAPTIVE_THRESHOLD.value:
        method = str(parameters.get("method", "gaussian"))
        if parameter_key in {"k", "r"}:
            return method in {"sauvola", "phansalkar"}
        if parameter_key in {"p", "q"}:
            return method == "phansalkar"
    if (
        operation_id
        in {
            ImageOperation.LOG_V2.value,
            ImageOperation.EXP_V2.value,
            ImageOperation.SQRT_V2.value,
        }
        and parameter_key in {"output_min", "output_max"}
    ):
        return parameters.get("result_mode", "float32") == "remap"
    if operation_id == ImageOperation.FFT_FILTER.value:
        mode = str(parameters.get("mode", "lowpass"))
        if parameter_key == "low_cutoff":
            return mode in {"highpass", "bandpass", "bandstop"}
        if parameter_key == "high_cutoff":
            return mode in {"lowpass", "bandpass", "bandstop"}
        if parameter_key == "tukey_alpha":
            return parameters.get("boundary", "periodic") == "tukey"
        if parameter_key == "pixel_size":
            return (
                parameters.get("frequency_unit", "cycles_per_pixel")
                == "cycles_per_unit"
            )
    if operation_id == ImageOperation.FLAT_FIELD_CORRECTION.value:
        reference = parameters.get("flat_field_source", "estimated") == "reference"
        if parameter_key == "secondary_document_id":
            return reference
        if parameter_key in {"radius", "method"}:
            return not reference
    return True


def _parameter_relationship_error(
    operation_id: str,
    parameters: Mapping[str, object],
    *,
    input_state: RasterTypeState,
) -> str:
    """Validate relationships that cannot be expressed by scalar ranges."""

    def number(key: str) -> float:
        value = float(parameters[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} 必须是有限数")
        return value

    try:
        if operation_id == ImageOperation.ADJUST_LEVELS.value:
            if number("black_point") >= number("white_point"):
                return "黑场必须小于白场。"
        elif operation_id == ImageOperation.THRESHOLD.value:
            if number("lower") > number("upper"):
                return "阈值下限不能大于上限。"
        elif operation_id == ImageOperation.CANNY_EDGES.value:
            if number("threshold_low") >= number("threshold_high"):
                return "Canny 低阈值必须小于高阈值。"
        elif operation_id in {
            ImageOperation.NORMALIZE.value,
            ImageOperation.CLAMP.value,
        }:
            low_key, high_key = (
                ("output_min", "output_max")
                if operation_id == ImageOperation.NORMALIZE.value
                else ("minimum", "maximum")
            )
            if number(low_key) > number(high_key):
                return "范围下限不能大于上限。"
        elif operation_id == ImageOperation.PERCENTILE_SATURATION.value:
            if number("lower_percentile") >= number("upper_percentile"):
                return "下百分位必须小于上百分位。"
        elif operation_id in {
            ImageOperation.LOG_V2.value,
            ImageOperation.EXP_V2.value,
            ImageOperation.SQRT_V2.value,
        }:
            if (
                parameters.get("result_mode") == "remap"
                and number("output_min") > number("output_max")
            ):
                return "重映射输出下限不能大于上限。"
        elif operation_id == ImageOperation.FFT_FILTER.value:
            mode = str(parameters.get("mode", "lowpass"))
            low = number("low_cutoff")
            high = number("high_cutoff")
            if mode in {"bandpass", "bandstop"} and high <= low:
                return "带通/带阻的高截止频率必须大于低截止频率。"
            if mode == "lowpass" and high <= 0:
                return "低通滤波的高截止频率必须大于零。"
            if mode == "highpass" and low <= 0:
                return "高通滤波的低截止频率必须大于零。"
        elif operation_id == ImageOperation.CUSTOM_CONVOLUTION.value:
            width = int(parameters["kernel_width"])
            height = int(parameters["kernel_height"])
            if width % 2 == 0 or height % 2 == 0:
                return "卷积核宽度和高度必须为正奇数。"
            expected = width * height
            values = tuple(parameters["kernel"])  # type: ignore[arg-type]
            if len(values) != expected:
                return f"卷积核需要 {expected} 个数值，当前为 {len(values)} 个。"
            if bool(parameters.get("normalize_kernel", False)):
                total = math.fsum(float(value) for value in values)
                if math.isclose(total, 0.0, abs_tol=1e-15):
                    return "卷积核系数和为零，不能启用归一化。"
        elif operation_id == ImageOperation.CROP.value:
            x = int(parameters["x"])
            y = int(parameters["y"])
            width = int(parameters["width"])
            height = int(parameters["height"])
            if (
                input_state.width is not None
                and input_state.height is not None
                and (
                    x + width > input_state.width
                    or y + height > input_state.height
                )
            ):
                return (
                    "裁剪范围超出当前步骤输入尺寸 "
                    f"{input_state.width}×{input_state.height}。"
                )
    except (KeyError, TypeError, ValueError) as exc:
        return str(exc)
    return ""


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
    descriptor = get_image_operation_descriptor(definition.operation)
    return ImageOperationSpec(
        definition.operation.value,
        parameters,
        implementation_version=descriptor.version,
    )


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
        source_semantic: RasterSemantic | None = None,
        roi_summary: str = "整张图片",
        roi_mask: NDArray[np.bool_] | None = None,
        preview_rect: tuple[float, float, float, float] | None = None,
        secondary_images: Mapping[str, RasterPlane] | None = None,
        secondary_image_names: Mapping[str, str] | None = None,
        secondary_image_semantics: Mapping[str, RasterSemantic] | None = None,
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
        self._source_semantic = RasterTypeState(
            pixel_type=source.pixel_type,
            semantic=source_semantic,
        ).semantic
        self._source_document_id = str(source_document_id)
        self._source_name = source_name.strip() or "未命名图片"
        self._roi_summary = roi_summary.strip() or "整张图片"
        self._roi_mask = roi_mask
        self._secondary_images = dict(secondary_images or {})
        provided_secondary_semantics = dict(
            secondary_image_semantics or {}
        )
        unknown_secondary_semantics = (
            set(provided_secondary_semantics) - set(self._secondary_images)
        )
        if unknown_secondary_semantics:
            raise ValueError(
                "第二幅图像语义引用了不存在的文档 ID："
                + "、".join(sorted(unknown_secondary_semantics))
            )
        self._secondary_image_semantics = {
            document_id: RasterTypeState(
                pixel_type=plane.pixel_type,
                semantic=provided_secondary_semantics.get(document_id),
            ).semantic
            for document_id, plane in self._secondary_images.items()
        }
        self._secondary_sha256_cache: dict[str, str] = {}
        self._flat_field_level_cache: dict[
            tuple[str, bool],
            tuple[float, ...],
        ] = {}
        self._preview_snapshot = build_processing_preview_snapshot(
            source,
            visible_rect=preview_rect,
            roi_mask=roi_mask,
            secondary_images=self._secondary_images,
        )
        self._overview_image = raster_plane_to_bounded_overview_image(source)
        self._latest_preview_image = raster_plane_to_display_image(
            self._preview_snapshot.source
        )
        self._latest_preview_raster = self._preview_snapshot.source
        self._processed_overview_image = self._overview_image.copy()
        self._overview_note_text = ""
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
        self._preview_crop_by_request_id: dict[
            str,
            tuple[int, int, int, int],
        ] = {}
        self._undo_stack: list[tuple[ImageOperationSpec, ...]] = []
        self._redo_stack: list[tuple[ImageOperationSpec, ...]] = []
        self._parameter_widgets: dict[str, QWidget] = {}
        self._parameter_row_widgets: dict[str, QWidget] = {}
        self._structured_parameter_editors: dict[str, QWidget] = {}
        self._structured_parameter_error_message = ""
        self._histogram_parameter_editor: HistogramRangeEditor | None = None
        self._histogram_parameter_keys: tuple[str, ...] = ()
        self._percentile_parameter_editor: (
            PercentileRangeEditor | None
        ) = None
        self._parameter_input_raster = self._preview_snapshot.source
        self._parameter_input_roi_mask = (
            self._preview_snapshot.roi_mask
            if roi_mask is not None
            else None
        )
        self._parameter_input_step_index: int | None = 0
        self._parameter_validation_label: QLabel | None = None
        self._parameter_error_message = ""
        self._pending_parameter_result_metadata: dict[str, object] = {}
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
        self._show_preview_raster(self._preview_snapshot.source)
        self._update_actions()

    @property
    def task_controller(self) -> ImageProcessingTaskController:
        return self._controller

    def operation_steps(self) -> tuple[ImageOperationSpec, ...]:
        return self._steps

    def current_recipe(self) -> ImageProcessingRecipe:
        return ImageProcessingRecipe.from_operations(self._steps)

    def make_default_operation_spec(
        self,
        operation_id: ImageOperation | str,
    ) -> ImageOperationSpec:
        """Build a prefix-aware default, including frozen ROI crop bounds.

        A step consumes the output of the preceding recipe, not the original
        source raster.  In particular, dimensions and integer sample ranges
        may already have changed after crop, resize or type conversion.
        """

        operation_value = (
            operation_id.value
            if isinstance(operation_id, ImageOperation)
            else str(operation_id)
        )
        input_state = self._resolve_prefix_output_state(self._steps)
        step = default_operation_spec(
            operation_value,
            int(input_state.width or self._source.width),
            int(input_state.height or self._source.height),
            source_pixel_type=input_state.pixel_type,
            secondary_document_id=next(iter(self._secondary_images), None),
        )
        if operation_value == ImageOperation.CROP.value and self._roi_mask is not None:
            rows, columns = np.nonzero(self._roi_mask)
            if rows.size and columns.size:
                parameters = step.parameters
                left = int(columns.min())
                top = int(rows.min())
                parameters.update(
                    {
                        "x": left,
                        "y": top,
                        "width": int(columns.max()) - left + 1,
                        "height": int(rows.max()) - top + 1,
                        "roi_mode": "bounds",
                    }
                )
                step = ImageOperationSpec(
                    step.operation_id,
                    parameters,
                    implementation=step.implementation,
                    implementation_version=step.implementation_version,
                )
        return step

    def _source_type_state(self) -> RasterTypeState:
        return RasterTypeState(
            pixel_type=self._source.pixel_type,
            semantic=self._source_semantic,
            width=self._source.width,
            height=self._source.height,
        )

    def _resolve_prefix_output_state(
        self,
        operations: tuple[ImageOperationSpec, ...],
    ) -> RasterTypeState:
        """Resolve a pixel-free prefix without executing or copying pixels."""

        if not operations:
            return self._source_type_state()
        return validate_workbench_operation_sequence(
            self._source,
            operations,
            source_semantic=self._source_semantic,
            roi_requested=self._roi_is_active(),
            secondary_images=self._secondary_images,
            secondary_semantics=self._secondary_image_semantics,
        ).output_state

    def _input_state_for_step(self, row: int) -> RasterTypeState:
        if row <= 0:
            return self._source_type_state()
        return self._resolve_prefix_output_state(self._steps[:row])

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
            descriptor = get_image_operation_descriptor(operation.operation_id)
            if (
                operation.operation_id
                == ImageOperation.FFT_POWER_SPECTRUM.value
                and (
                    operation.implementation != "fdm"
                    or operation.implementation_version != "1"
                )
            ):
                raise ValueError("旧版 FFT 功率谱只允许按 fdm v1 配方重放")
            if operation.implementation != "fdm":
                raise ValueError(
                    "不支持的图像处理实现："
                    f"{operation.implementation}；当前工作台只允许 fdm 实现"
                )
            supported_versions = {"1", descriptor.version}
            if operation.implementation_version not in supported_versions:
                raise ValueError(
                    "不支持的算法版本："
                    f"{operation.operation_id} "
                    f"v{operation.implementation_version}；当前版本仅支持 "
                    + "、".join(
                        f"v{version}"
                        for version in sorted(supported_versions, key=int)
                    )
                )
            secondary_document_id = (
                str(operation.parameters.get("secondary_document_id", ""))
                or next(iter(self._secondary_images), "")
            )
            input_state = self._resolve_prefix_output_state(
                tuple(normalized_operations)
            )
            defaults = default_operation_spec(
                operation.operation_id,
                int(input_state.width or self._source.width),
                int(input_state.height or self._source.height),
                source_pixel_type=input_state.pixel_type,
                secondary_document_id=(
                    secondary_document_id
                    if operation.operation_id
                    in {
                        ImageOperation.IMAGE_CALCULATOR.value,
                        ImageOperation.FLAT_FIELD_CORRECTION.value,
                    }
                    else None
                ),
            )
            parameters = defaults.parameters
            persisted_parameters = operation.parameters
            # A loaded recipe must keep the service semantics that were in
            # effect when it was saved.  Newly exposed UI fields may have safer
            # defaults, but injecting those defaults over a legacy alias would
            # silently change a replay result.
            if (
                operation.operation_id
                == ImageOperation.GAUSSIAN_BLUR.value
                and "sigma" in persisted_parameters
            ):
                persisted_sigma = persisted_parameters["sigma"]
                persisted_parameters.setdefault("sigma_x", persisted_sigma)
                persisted_parameters.setdefault("sigma_y", persisted_sigma)
            if operation.operation_id == ImageOperation.COPY.value:
                if (
                    "fill_value" in persisted_parameters
                    and "outside_value" not in persisted_parameters
                ):
                    persisted_parameters["outside_value"] = (
                        persisted_parameters["fill_value"]
                    )
            elif operation.operation_id == ImageOperation.CROP.value:
                if (
                    "outside_value" in persisted_parameters
                    and "fill_value" not in persisted_parameters
                ):
                    persisted_parameters["fill_value"] = (
                        persisted_parameters["outside_value"]
                    )
            elif operation.operation_id == ImageOperation.FFT_FILTER.value:
                # fdm v1 used periodic boundaries when the field was absent.
                persisted_parameters.setdefault("boundary", "periodic")
            elif operation.operation_id == ImageOperation.CONVERT_TYPE.value:
                # The service's historical omission default was value
                # preservation.  Keep it even though future UI defaults may
                # become source-aware.
                persisted_parameters.setdefault(
                    "scale_mode",
                    "preserve_values",
                )
            parameters.update(persisted_parameters)
            normalized_operation = ImageOperationSpec(
                operation.operation_id,
                parameters,
                implementation=operation.implementation,
                implementation_version=operation.implementation_version,
                result_metadata=operation.result_metadata,
            )
            flat_field_source = str(
                normalized_operation.parameters.get(
                    "flat_field_source",
                    "estimated",
                )
            ).strip().lower()
            if (
                operation.operation_id
                == ImageOperation.IMAGE_CALCULATOR.value
                or (
                    operation.operation_id
                    == ImageOperation.FLAT_FIELD_CORRECTION.value
                    and flat_field_source == "reference"
                )
            ):
                secondary_document_id = str(
                    normalized_operation.parameters.get("secondary_document_id", "")
                )
                if secondary_document_id not in self._secondary_images:
                    message = (
                        "参考图平场校正选择的参考图像不在当前兼容候选列表中"
                        if operation.operation_id
                        == ImageOperation.FLAT_FIELD_CORRECTION.value
                        else "图像计算器选择的第二幅图像不在当前候选列表中"
                    )
                    raise ValueError(message)
                expected_sha256 = str(
                    normalized_operation.parameters.get(
                        "secondary_sha256",
                        "",
                    )
                ).strip()
                if (
                    operation.operation_id
                    == ImageOperation.FLAT_FIELD_CORRECTION.value
                    and expected_sha256
                    and (
                        self._secondary_image_sha256(
                            secondary_document_id
                        )
                        != expected_sha256
                    )
                ):
                    raise ValueError(
                        "参考平场像素摘要与当前候选图片不一致，"
                        "请重新选择参考图像"
                    )
            normalized_operations.append(normalized_operation)
        normalized = tuple(normalized_operations)
        self._commit_steps(normalized, selected_index=0 if normalized else -1)

    def request_preview(self) -> None:
        self._preview_timer.stop()
        if not self._steps or self._final_in_progress:
            self._controller.cancel_preview()
            self._show_preview_raster(self._preview_snapshot.source)
            self._status_label.setText(
                "尚未添加处理步骤。" if not self._steps else "正在生成派生图片…"
            )
            return
        try:
            prepared_steps = self._prepare_reference_flat_field_steps(
                self._steps
            )
            self._validate_active_roi_operations(prepared_steps)
            preview_snapshot = self._preview_snapshot
            halo_x = 0
            halo_y = 0
            geometry_operations = {
                ImageOperation.CROP.value,
                ImageOperation.RESIZE.value,
                ImageOperation.TRANSLATE.value,
                ImageOperation.RESIZE_CANVAS.value,
                ImageOperation.PIXEL_BIN.value,
                ImageOperation.ROTATE.value,
                ImageOperation.ROTATE_90_CLOCKWISE.value,
                ImageOperation.ROTATE_90_COUNTERCLOCKWISE.value,
                ImageOperation.ROTATE_180.value,
                ImageOperation.FLIP_HORIZONTAL.value,
                ImageOperation.FLIP_VERTICAL.value,
            }
            for operation in prepared_steps:
                capability = resolve_image_operation_capability(
                    operation.operation_id,
                    operation.parameters,
                )
                halo_x += int(capability.halo_x)
                halo_y += int(capability.halo_y)
            if (halo_x or halo_y) and any(
                operation.operation_id in geometry_operations
                for operation in prepared_steps
            ):
                raise ValueError(
                    "当前配方同时包含邻域处理和几何变换，无法在有界样本中"
                    "保证边缘完全等价；请拆分为两个派生步骤后预览。"
                )
            preview_snapshot, preview_crop = (
                expand_processing_preview_snapshot_for_halo(
                    self._preview_snapshot,
                    full_source=self._source,
                    full_roi_mask=(
                        self._roi_mask if self._roi_is_active() else None
                    ),
                    full_secondary_images=self._secondary_images,
                    halo_x=halo_x,
                    halo_y=halo_y,
                )
            )
            preview_operations = adapt_operations_for_preview(
                preview_snapshot,
                prepared_steps,
            )
            estimate = estimate_final_resources(
                preview_snapshot.source,
                preview_operations,
            )
            if estimate.peak_working_set_bytes > PREVIEW_MAX_WORKING_SET_BYTES:
                raise ValueError(
                    "当前预览步骤预计需要 "
                    f"{estimate.peak_working_set_bytes / float(1 << 20):.1f} MiB，"
                    "超过 256 MiB 预览上限；请缩小画布视场后重新打开工作台。"
                )
        except (TypeError, ValueError) as exc:
            self._controller.cancel_preview()
            self._status_label.setText(f"自动预览已暂停：{exc}")
            return
        self._status_label.setText("正在计算 1:1 预览…")
        try:
            request = self._controller.start_preview(
                source_document_id=self._source_document_id,
                source=preview_snapshot.source,
                operations=preview_operations,
                source_semantic=self._source_semantic,
                roi_mask=(
                    preview_snapshot.roi_mask
                    if self._roi_is_active()
                    else None
                ),
                secondary_images=dict(preview_snapshot.secondary_images),
                secondary_semantics={
                    document_id: self._secondary_image_semantics[
                        document_id
                    ]
                    for document_id, _plane in (
                        preview_snapshot.secondary_images
                    )
                },
                capture_step_input_index=max(
                    0,
                    min(
                        self._steps_list.currentRow(),
                        len(preview_operations) - 1,
                    ),
                ),
            )
            self._preview_crop_by_request_id.clear()
            if preview_crop is not None:
                self._preview_crop_by_request_id[request.request_id] = (
                    preview_crop
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
        self._use_roi_checkbox = QCheckBox("限制在当前 ROI", source_bar)
        self._use_roi_checkbox.setVisible(self._roi_mask is not None)
        self._use_roi_checkbox.setChecked(self._roi_mask is not None)
        self._use_roi_checkbox.setToolTip(
            "关闭后按整张图片处理。改变坐标或通道结构的操作"
            "不允许隐式套用 ROI。"
        )
        self._use_roi_checkbox.toggled.connect(
            lambda _checked: self._schedule_preview()
        )
        source_layout.addWidget(self._use_roi_checkbox)

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

        sample_x, sample_y, sample_width, sample_height = (
            self._preview_snapshot.bounds
        )
        sample_description = (
            "完整图片"
            if self._preview_snapshot.is_full_source
            else (
                f"原始像素样本 x={sample_x}, y={sample_y}, "
                f"{sample_width} × {sample_height}"
            )
        )
        hint = QLabel(
            "预览计算始终使用未缩放的原始像素；显示缩放不会改变处理数据。"
            f"当前使用{sample_description}。"
            "最终处理始终以完整原始分辨率执行。",
            panel,
        )
        self._preview_hint = hint
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._preview_view = ProcessingPreviewView(panel)
        self._preview_view.zoomChanged.connect(
            self._on_preview_zoom_changed
        )
        layout.addWidget(self._preview_view, 1)

        self._overview_note = QLabel(panel)
        self._overview_note.setObjectName("scaledOverviewNotice")
        self._overview_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overview_note.setWordWrap(True)
        self._overview_note.setVisible(False)
        layout.addWidget(self._overview_note)

        preview_mode_row = QHBoxLayout()
        self._overview_checkbox = QCheckBox("显示近似全图概览", panel)
        self._overview_checkbox.toggled.connect(self._toggle_overview)
        preview_mode_row.addWidget(self._overview_checkbox)
        preview_mode_row.addStretch(1)
        self._fit_preview_button = QPushButton("适合", panel)
        self._fit_preview_button.setToolTip("完整显示当前预览")
        self._fit_preview_button.clicked.connect(
            self._preview_view.fit_image
        )
        preview_mode_row.addWidget(self._fit_preview_button)
        self._actual_preview_button = QPushButton("1:1", panel)
        self._actual_preview_button.setToolTip(
            "一个原始图像像素对应一个界面逻辑像素"
        )
        self._actual_preview_button.clicked.connect(
            self._preview_view.actual_size
        )
        preview_mode_row.addWidget(self._actual_preview_button)
        self._zoom_out_button = QPushButton("−", panel)
        self._zoom_out_button.setToolTip("缩小预览（Ctrl+滚轮向下）")
        self._zoom_out_button.clicked.connect(
            lambda: self._preview_view.zoom_by(
                1.0 / ProcessingPreviewView.ZOOM_STEP
            )
        )
        preview_mode_row.addWidget(self._zoom_out_button)
        self._preview_zoom_label = QLabel("100%", panel)
        self._preview_zoom_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self._preview_zoom_label.setMinimumWidth(54)
        preview_mode_row.addWidget(self._preview_zoom_label)
        self._zoom_in_button = QPushButton("+", panel)
        self._zoom_in_button.setToolTip("放大预览（Ctrl+滚轮向上）")
        self._zoom_in_button.clicked.connect(
            lambda: self._preview_view.zoom_by(
                ProcessingPreviewView.ZOOM_STEP
            )
        )
        preview_mode_row.addWidget(self._zoom_in_button)
        layout.addLayout(preview_mode_row)
        return panel

    def _build_parameters_panel(self, parent: QWidget) -> QWidget:
        panel = QGroupBox("当前步骤参数", parent)
        layout = QVBoxLayout(panel)
        self._parameter_scroll = QScrollArea(panel)
        self._parameter_scroll.setWidgetResizable(True)
        self._parameter_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._parameter_content = QWidget(self._parameter_scroll)
        self._parameter_content.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self._parameter_form = QFormLayout(self._parameter_content)
        self._parameter_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._parameter_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self._parameter_form.setFormAlignment(
            Qt.AlignmentFlag.AlignTop
        )
        self._parameter_scroll.setWidget(self._parameter_content)
        layout.addWidget(self._parameter_scroll)
        return panel

    def _populate_operations(self) -> None:
        current = self._operation_combo.currentData() if self._operation_combo.count() else None
        category = str(self._category_combo.currentData() or "类型")
        self._operation_combo.clear()
        for definition in _OPERATION_CATALOG:
            if (
                definition.category == category
                and definition.available_for_new_recipe
            ):
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
        step = self.make_default_operation_spec(operation_id)
        self._commit_steps(self._steps + (step,), selected_index=len(self._steps))

    def _resolved_default(self, parameter: ParameterField) -> object:
        row = self._steps_list.currentRow()
        input_state = self._input_state_for_step(row)
        return _resolved_parameter_default(
            parameter,
            source_width=int(input_state.width or self._source.width),
            source_height=int(input_state.height or self._source.height),
            source_pixel_type=input_state.pixel_type,
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
        if (
            0 <= row < len(self._steps)
            and row != self._parameter_input_step_index
        ):
            self._schedule_preview()

    def _clear_parameter_form(self) -> None:
        while self._parameter_form.rowCount():
            self._parameter_form.removeRow(0)
        self._parameter_widgets.clear()
        self._parameter_row_widgets.clear()
        self._structured_parameter_editors.clear()
        self._structured_parameter_error_message = ""
        self._histogram_parameter_editor = None
        self._histogram_parameter_keys = ()
        self._percentile_parameter_editor = None
        self._parameter_validation_label = None
        self._parameter_error_message = ""

    def _rebuild_parameter_form(self, row: int) -> None:
        self._updating_parameter_form = True
        self._clear_parameter_form()
        if not 0 <= row < len(self._steps):
            self._parameter_form.addRow(QLabel("请选择一个处理步骤。", self._parameter_content))
            self._updating_parameter_form = False
            return
        step = self._steps[row]
        definition = _DEFINITION_BY_ID[step.operation_id]
        replay_only = _operation_step_is_replay_only(step)
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
                    *(
                        (
                            "兼容：该步骤仅用于完整重放旧版 fdm v1 配方；"
                            "参数和步骤顺序已锁定。",
                        )
                        if replay_only
                        else ()
                    ),
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
        input_state = self._input_state_for_step(row)
        if (
            definition.operation
            in {
                ImageOperation.BRIGHTNESS_CONTRAST,
                ImageOperation.HISTOGRAM_EQUALIZATION,
            }
            and input_state.pixel_type
            is RasterPixelType.GRAY32_FLOAT
        ):
            float_range_text = (
                "兼容回放提醒：这是旧版 v1 步骤，将继续按历史"
                "0–1 工作范围只读重放。若要修改，请先用“色阶”"
                "或“归一化”显式限定范围并转换为 8 位或 16 位。"
                if replay_only
                else (
                    "数据范围阻断：32 位浮点图像没有可安全推断的"
                    "0–1 工作范围。请先用“色阶”或“归一化”显式"
                    "限定输入/输出范围，并转换为 8 位或 16 位。"
                )
            )
            float_range_warning = QLabel(
                float_range_text,
                self._parameter_content,
            )
            float_range_warning.setObjectName(
                "imageParameterScientificWarning"
            )
            float_range_warning.setWordWrap(True)
            float_range_warning.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._parameter_form.addRow(float_range_warning)
        if (
            definition.operation.value
            in _EXPLICIT_BINARY_RECOMMENDED_OPERATIONS
            and input_state.semantic is not RasterSemantic.BINARY_MASK
        ):
            if (
                step.implementation_version
                == get_image_operation_descriptor(
                    step.operation_id
                ).version
            ):
                semantic_text = (
                    "数据语义阻断：当前步骤输入不是显式二值掩膜。"
                    "请先添加“二值化”“自动阈值”或"
                    "“局部自适应阈值”；本步骤不会再用中间灰度"
                    "自动猜测前景。"
                )
            else:
                semantic_text = (
                    "兼容回放提醒：这是旧版 v1 步骤，输入不是显式"
                    "二值掩膜，将按旧版中间灰度规则只读重放。"
                    "若要修改，请用显式阈值和当前版本操作重建配方。"
                )
            semantic_warning = QLabel(
                semantic_text,
                self._parameter_content,
            )
            semantic_warning.setObjectName(
                "imageParameterSemanticWarning"
            )
            semantic_warning.setWordWrap(True)
            semantic_warning.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._parameter_form.addRow(semantic_warning)
        context_message = _operation_parameter_context_message(
            definition,
            input_state,
        )
        if context_message:
            context_label = QLabel(
                context_message,
                self._parameter_content,
            )
            context_label.setObjectName(
                "imageParameterScientificContext"
            )
            context_label.setWordWrap(True)
            context_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._parameter_form.addRow(context_label)
        if not definition.parameters:
            self._parameter_form.addRow(QLabel("此操作没有可调参数。", self._parameter_content))
        values = step.parameters
        structured_keys: set[str] = set()
        histogram_keys = _HISTOGRAM_EDITOR_PARAMETERS.get(
            definition.operation.value,
            (),
        )
        if histogram_keys:
            self._add_histogram_range_editor(
                definition,
                values,
                histogram_keys,
            )
            structured_keys.update(histogram_keys)
        percentile_keys = _PERCENTILE_RANGE_PARAMETERS.get(
            definition.operation.value,
            (),
        )
        if percentile_keys:
            self._add_percentile_range_editor(
                definition,
                values,
                percentile_keys,
            )
            structured_keys.update(percentile_keys)
        crop_keys = _CROP_BOUNDS_PARAMETERS.get(
            definition.operation.value,
            (),
        )
        if crop_keys:
            self._add_crop_bounds_editor(
                definition,
                values,
                crop_keys,
            )
            structured_keys.update(crop_keys)
        if definition.operation in _LINKED_DIMENSION_OPERATIONS:
            self._add_linked_dimensions_editor(definition, values)
            structured_keys.update({"width", "height"})
        if definition.operation in _STRUCTURING_ELEMENT_OPERATIONS:
            self._add_structuring_element_editor(definition, values)
            structured_keys.update({"radius", "iterations", "kernel"})
        frequency_keys = _FREQUENCY_RESPONSE_PARAMETERS.get(
            definition.operation.value,
            (),
        )
        if frequency_keys:
            self._add_frequency_response_editor(
                definition,
                values,
                frequency_keys,
            )
            structured_keys.update(frequency_keys)
        stripe_frequency_keys = _STRIPE_FREQUENCY_PARAMETERS.get(
            definition.operation.value,
            (),
        )
        if stripe_frequency_keys:
            self._add_stripe_frequency_editor(
                definition,
                values,
                stripe_frequency_keys,
            )
            structured_keys.update(stripe_frequency_keys)
        if definition.operation is ImageOperation.CUSTOM_CONVOLUTION:
            self._add_kernel_matrix_editor(definition, values)
            structured_keys.update(
                {"kernel_width", "kernel_height", "kernel"}
            )
        for parameter in definition.parameters:
            if parameter.key in structured_keys:
                continue
            value = values.get(parameter.key, self._resolved_default(parameter))
            if (
                definition.operation is ImageOperation.RESIZE_CANVAS
                and parameter.key == "anchor"
            ):
                widget = self._add_anchor_grid_editor(
                    parameter,
                    value,
                    enabled=definition.available_for_new_recipe,
                )
                continue
            if parameter.kind == "secondary_image":
                self._add_compatible_image_picker(
                    definition,
                    parameter,
                    value,
                    enabled=definition.available_for_new_recipe,
                )
                continue
            if self._parameter_prefers_slider(
                parameter,
                definition.operation.value,
            ):
                self._add_slider_number_editor(
                    parameter,
                    value,
                    enabled=definition.available_for_new_recipe,
                )
                continue
            widget = self._create_parameter_widget(parameter, value)
            widget.setEnabled(definition.available_for_new_recipe)
            self._parameter_widgets[parameter.key] = widget
            self._parameter_row_widgets[parameter.key] = widget
            parameter_label = QLabel(parameter.label, self._parameter_content)
            tooltip = _parameter_help_text(definition, parameter)
            parameter_label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            self._parameter_form.addRow(parameter_label, widget)
        self._parameter_validation_label = QLabel(self._parameter_content)
        self._parameter_validation_label.setObjectName(
            "imageParameterValidation"
        )
        self._parameter_validation_label.setWordWrap(True)
        self._parameter_validation_label.setVisible(False)
        self._parameter_form.addRow(self._parameter_validation_label)
        if replay_only:
            for widget in {
                *self._parameter_widgets.values(),
                *self._parameter_row_widgets.values(),
                *self._structured_parameter_editors.values(),
            }:
                widget.setEnabled(False)
        self._updating_parameter_form = False
        self._refresh_parameter_conditions()
        self._refresh_parameter_validation()
        self._update_specialized_parameter_data()

    def _add_histogram_range_editor(
        self,
        definition: WorkbenchOperationDefinition,
        values: Mapping[str, object],
        parameter_keys: tuple[str, ...],
    ) -> HistogramRangeEditor:
        """Add a range-aware editor while keeping native spin-box proxies.

        The editor is purely a parameter/preview surface. Final recipes retain
        the existing exact floating-point values and operation IDs.
        """

        fields = {field.key: field for field in definition.parameters}
        first = fields[parameter_keys[0]]
        lower = float(
            values.get(
                first.key,
                self._resolved_default(first),
            )
        )
        single_threshold = len(parameter_keys) == 1
        if single_threshold:
            upper = lower
        else:
            second = fields[parameter_keys[1]]
            upper = float(
                values.get(
                    second.key,
                    self._resolved_default(second),
                )
            )
        input_state = self._input_state_for_step(
            self._steps_list.currentRow()
        )
        native_maximum = input_state.pixel_type.sample_maximum
        minimum = min(0.0, lower, upper)
        maximum = max(
            1.0 if native_maximum is None else float(native_maximum),
            lower,
            upper,
        )
        if math.isclose(minimum, maximum):
            maximum = minimum + 1.0
        editor = HistogramRangeEditor(
            self._parameter_content,
            single_threshold=single_threshold,
            minimum=minimum,
            maximum=maximum,
            lower=lower,
            upper=None if single_threshold else upper,
            decimals=max(
                fields[key].decimals for key in parameter_keys
            ),
            suffix=first.suffix,
        )
        editor.setEnabled(definition.available_for_new_recipe)
        if definition.operation is ImageOperation.CANNY_EDGES:
            editor.autoButton.setVisible(False)
            editor.resetButton.setToolTip("恢复默认 Canny 双阈值")
        elif definition.operation is ImageOperation.ADJUST_LEVELS:
            editor.autoButton.setToolTip(
                "按当前冻结的 1:1 输入样本及 ROI 统计有限像素的 "
                "0.35% 和 99.65% 分位数；若样本覆盖完整源图则等同整图自动。"
                "样本范围会写入配方来源信息"
            )
        else:
            editor.autoButton.setToolTip(
                "使用当前冻结的 1:1 输入样本及 ROI 计算 Otsu 阈值；"
                "若样本覆盖完整源图则等同整图自动。"
                "精确阈值和样本范围都会写入配方"
            )

        if definition.operation is ImageOperation.THRESHOLD:
            editor.polarityCombo.setItemText(0, "区间内为前景")
            editor.polarityCombo.setItemText(1, "区间外为前景")
        elif definition.operation is ImageOperation.BINARIZE:
            editor.polarityCombo.setItemText(0, "亮像素为前景")
            editor.polarityCombo.setItemText(1, "暗像素为前景")
        else:
            editor.displayModeCombo.setVisible(False)
            editor.displayModeLabel.setVisible(False)
            editor.polarityCombo.setVisible(False)
            editor.polarityLabel.setVisible(False)

        for index, key in enumerate(parameter_keys):
            field = fields[key]
            proxy = (
                editor.lowerSpin
                if index == 0
                else editor.upperSpin
            )
            proxy.setDecimals(field.decimals)
            proxy.setSuffix(field.suffix)
            self._parameter_widgets[key] = proxy
            self._parameter_row_widgets[key] = editor
        self._histogram_parameter_editor = editor
        self._histogram_parameter_keys = parameter_keys
        self._structured_parameter_editors["histogram_range"] = editor

        title = {
            ImageOperation.ADJUST_LEVELS: "输入色阶",
            ImageOperation.THRESHOLD: "阈值范围",
            ImageOperation.BINARIZE: "二值阈值",
            ImageOperation.CANNY_EDGES: "Canny 梯度双阈值",
        }.get(definition.operation, "强度范围")
        label = QLabel(title, self._parameter_content)
        if definition.operation is ImageOperation.CANNY_EDGES:
            label.setToolTip(
                "直方图显示当前 1:1 样本的 Sobel 梯度幅值，"
                "与 Canny 低/高阈值使用同一数值域；"
                "它不是原始像素强度直方图。"
            )
        else:
            label.setToolTip(
                "直方图来自当前步骤输入的冻结 1:1 样本；"
                "最终处理仍使用完整原始分辨率。"
            )
        editor.setToolTip(label.toolTip())
        self._parameter_form.addRow(label)
        self._parameter_form.addRow(editor)

        editor.thresholdsChanged.connect(
            lambda _low, _high: self._histogram_preview_value_changed()
        )
        editor.editFinished.connect(self._parameter_value_changed)
        editor.autoRequested.connect(self._auto_histogram_parameters)
        editor.resetRequested.connect(self._reset_histogram_parameters)
        editor.foregroundPolarityChanged.connect(
            self._histogram_polarity_changed
        )
        editor.displayModeChanged.connect(
            lambda _mode: self._update_preview_display()
        )
        return editor

    def _histogram_preview_value_changed(self) -> None:
        self._refresh_histogram_selection_statistics()
        self._update_preview_display()

    def _add_percentile_range_editor(
        self,
        definition: WorkbenchOperationDefinition,
        values: Mapping[str, object],
        parameter_keys: tuple[str, ...],
    ) -> PercentileRangeEditor:
        fields = {field.key: field for field in definition.parameters}
        lower_key, upper_key = parameter_keys
        editor = PercentileRangeEditor(
            self._parameter_content,
            lower=float(
                values.get(
                    lower_key,
                    self._resolved_default(fields[lower_key]),
                )
            ),
            upper=float(
                values.get(
                    upper_key,
                    self._resolved_default(fields[upper_key]),
                )
            ),
            decimals=max(
                fields[lower_key].decimals,
                fields[upper_key].decimals,
            ),
        )
        editor.setEnabled(definition.available_for_new_recipe)
        self._parameter_widgets[lower_key] = editor.lowerSpin
        self._parameter_widgets[upper_key] = editor.upperSpin
        self._parameter_row_widgets[lower_key] = editor
        self._parameter_row_widgets[upper_key] = editor
        self._structured_parameter_editors[
            "percentile_range"
        ] = editor
        self._percentile_parameter_editor = editor
        tooltip = (
            "两个百分位以 0–100% 精确保存；下方显示它们在当前步骤"
            "冻结输入和当前 ROI 中解析出的实际强度值。"
        )
        label = QLabel("饱和百分位范围", self._parameter_content)
        label.setToolTip(tooltip)
        editor.setToolTip(tooltip)
        self._parameter_form.addRow(label)
        self._parameter_form.addRow(editor)
        editor.validationChanged.connect(
            self._percentile_editor_validation_changed
        )
        editor.valueChanged.connect(
            lambda _lower, _upper: (
                self._update_percentile_editor_data()
            )
        )
        editor.editFinished.connect(self._parameter_value_changed)
        return editor

    def _percentile_editor_validation_changed(
        self,
        valid: bool,
        message: str,
    ) -> None:
        self._structured_parameter_error_message = (
            "" if valid else message
        )
        if not self._updating_parameter_form:
            self._refresh_parameter_validation()

    @staticmethod
    def _parameter_prefers_slider(
        parameter: ParameterField,
        operation_id: str = "",
    ) -> bool:
        if (
            parameter.kind != "float"
            or parameter.minimum is None
            or parameter.maximum is None
        ):
            return False
        minimum = float(parameter.minimum)
        maximum = float(parameter.maximum)
        if (
            (
                str(operation_id)
                == ImageOperation.UNSHARP_MASK.value
                and parameter.key == "threshold"
            )
            or (
                str(operation_id)
                == ImageOperation.BRIGHTNESS_CONTRAST.value
                and parameter.key == "brightness"
            )
        ):
            return True
        return (
            math.isfinite(minimum)
            and math.isfinite(maximum)
            and 0.0 < maximum - minimum <= 1_000.0
        )

    def _add_slider_number_editor(
        self,
        parameter: ParameterField,
        value: object,
        *,
        enabled: bool,
    ) -> SliderNumberEditor:
        """Add a slider for bounded continuous parameters.

        The paired spin box remains the authoritative exact value and private
        compatibility proxy; the slider is only a linear interaction surface.
        """

        minimum = float(parameter.minimum)
        maximum = float(parameter.maximum)
        editor = SliderNumberEditor(
            self._parameter_content,
            minimum=minimum,
            maximum=maximum,
            value=float(value),
            decimals=parameter.decimals,
            suffix=parameter.suffix,
            resolution=10_000,
        )
        exact_step = 10.0 ** (-max(0, parameter.decimals))
        editor.setSingleStep(
            max(
                exact_step,
                round(
                    (maximum - minimum) / 1_000.0,
                    max(0, parameter.decimals),
                ),
            )
        )
        editor.setEnabled(enabled)
        proxy = editor.spinBox
        self._parameter_widgets[parameter.key] = proxy
        self._parameter_row_widgets[parameter.key] = editor
        self._structured_parameter_editors[
            f"slider:{parameter.key}"
        ] = editor
        tooltip = (
            parameter.help_text
            or (
                f"拖动滑块快速调整{parameter.label}；"
                "右侧数值框保存精确值，普通滚轮不会修改参数。"
            )
        )
        label = QLabel(parameter.label, self._parameter_content)
        label.setToolTip(tooltip)
        editor.setToolTip(tooltip)
        self._parameter_form.addRow(label, editor)
        editor.editFinished.connect(self._parameter_value_changed)
        return editor

    def _add_linked_dimensions_editor(
        self,
        definition: WorkbenchOperationDefinition,
        values: Mapping[str, object],
    ) -> LinkedDimensionsEditor:
        """Add exact dimensions with source context and optional aspect lock."""

        fields = {field.key: field for field in definition.parameters}
        input_state = self._input_state_for_step(
            self._steps_list.currentRow()
        )
        source_width = int(input_state.width or self._source.width)
        source_height = int(input_state.height or self._source.height)
        width = int(
            values.get(
                "width",
                self._resolved_default(fields["width"]),
            )
        )
        height = int(
            values.get(
                "height",
                self._resolved_default(fields["height"]),
            )
        )
        maximum = max(
            int(fields["width"].maximum or 1_000_000),
            int(fields["height"].maximum or 1_000_000),
        )
        resize_pixels = definition.operation is ImageOperation.RESIZE
        editor = LinkedDimensionsEditor(
            self._parameter_content,
            source_width=source_width,
            source_height=source_height,
            width=width,
            height=height,
            lock_aspect=resize_pixels,
            aspect_lock_available=resize_pixels,
            maximum_dimension=maximum,
        )
        editor.setEnabled(definition.available_for_new_recipe)
        self._parameter_widgets["width"] = editor.widthSpin
        self._parameter_widgets["height"] = editor.heightSpin
        self._parameter_row_widgets["width"] = editor
        self._parameter_row_widgets["height"] = editor
        self._structured_parameter_editors[
            "linked_dimensions"
        ] = editor

        title = (
            "输出像素尺寸"
            if resize_pixels
            else "输出画布尺寸"
        )
        tooltip = (
            "宽度和高度始终以最终精确像素值写入配方。"
            + (
                "比例锁定和百分比只是便捷输入，不会改变插值或标定规则。"
                if resize_pixels
                else (
                    "画布扩展不会重采样原像素；百分比仅用于同时输入"
                    "宽度和高度。"
                )
            )
        )
        label = QLabel(title, self._parameter_content)
        label.setToolTip(tooltip)
        editor.setToolTip(tooltip)
        self._parameter_form.addRow(label)
        self._parameter_form.addRow(editor)
        editor.editFinished.connect(self._parameter_value_changed)
        return editor

    def _add_crop_bounds_editor(
        self,
        definition: WorkbenchOperationDefinition,
        values: Mapping[str, object],
        parameter_keys: tuple[str, ...],
    ) -> CropBoundsEditor:
        fields = {field.key: field for field in definition.parameters}
        input_state = self._input_state_for_step(
            self._steps_list.currentRow()
        )
        editor = CropBoundsEditor(
            self._parameter_content,
            source_width=int(input_state.width or self._source.width),
            source_height=int(
                input_state.height or self._source.height
            ),
            x=int(
                values.get(
                    "x",
                    self._resolved_default(fields["x"]),
                )
            ),
            y=int(
                values.get(
                    "y",
                    self._resolved_default(fields["y"]),
                )
            ),
            width=int(
                values.get(
                    "width",
                    self._resolved_default(fields["width"]),
                )
            ),
            height=int(
                values.get(
                    "height",
                    self._resolved_default(fields["height"]),
                )
            ),
        )
        editor.setEnabled(definition.available_for_new_recipe)
        proxies: dict[str, QWidget] = {
            "x": editor.xSpin,
            "y": editor.ySpin,
            "width": editor.widthSpin,
            "height": editor.heightSpin,
        }
        for key in parameter_keys:
            self._parameter_widgets[key] = proxies[key]
            self._parameter_row_widgets[key] = editor
        self._structured_parameter_editors["crop_bounds"] = editor
        tooltip = (
            "坐标始终使用当前步骤输入的原始像素；右边界和下边界"
            "会被限制在图片范围内。ROI 裁剪仍按下方模式决定"
            "使用包围框还是掩膜。"
        )
        label = QLabel("裁剪像素范围", self._parameter_content)
        label.setToolTip(tooltip)
        editor.setToolTip(tooltip)
        self._parameter_form.addRow(label)
        self._parameter_form.addRow(editor)
        editor.editFinished.connect(self._parameter_value_changed)
        return editor

    def _add_structuring_element_editor(
        self,
        definition: WorkbenchOperationDefinition,
        values: Mapping[str, object],
    ) -> StructuringElementEditor:
        """Add one visual editor for morphology radius, passes and shape."""

        fields = {field.key: field for field in definition.parameters}
        editor = StructuringElementEditor(
            self._parameter_content,
            radius=int(
                values.get(
                    "radius",
                    self._resolved_default(fields["radius"]),
                )
            ),
            iterations=int(
                values.get(
                    "iterations",
                    self._resolved_default(fields["iterations"]),
                )
            ),
            shape=str(
                values.get(
                    "kernel",
                    self._resolved_default(fields["kernel"]),
                )
            ),
            maximum_radius=int(fields["radius"].maximum or 255),
            maximum_iterations=int(
                fields["iterations"].maximum or 100
            ),
        )
        editor.setEnabled(definition.available_for_new_recipe)
        self._parameter_widgets["radius"] = editor.radiusSpin
        self._parameter_widgets["iterations"] = editor.iterationsSpin
        self._parameter_widgets["kernel"] = editor.shapeCombo
        for key in ("radius", "iterations", "kernel"):
            self._parameter_row_widgets[key] = editor
        self._structured_parameter_editors[
            "structuring_element"
        ] = editor
        tooltip = (
            "预览显示结构元素形状及实际核尺寸 2×半径+1；"
            "迭代次数单独记录，最终处理仍使用原始分辨率。"
        )
        label = QLabel("结构元素", self._parameter_content)
        label.setToolTip(tooltip)
        editor.setToolTip(tooltip)
        self._parameter_form.addRow(label)
        self._parameter_form.addRow(editor)
        editor.editFinished.connect(self._parameter_value_changed)
        return editor

    def _add_frequency_response_editor(
        self,
        definition: WorkbenchOperationDefinition,
        values: Mapping[str, object],
        parameter_keys: tuple[str, ...],
    ) -> FrequencyResponseEditor:
        """Add a Butterworth response editor without changing FFT semantics."""

        fields = {field.key: field for field in definition.parameters}
        frequency_unit = str(
            values.get("frequency_unit", "cycles_per_pixel")
        )
        pixel_size = float(values.get("pixel_size", 1.0))
        nyquist = (
            0.5 / pixel_size
            if frequency_unit == "cycles_per_unit"
            else 0.5
        )
        editor = FrequencyResponseEditor(
            self._parameter_content,
            mode=str(
                values.get(
                    "mode",
                    self._resolved_default(fields["mode"]),
                )
            ),
            low_cutoff=float(
                values.get(
                    "low_cutoff",
                    self._resolved_default(fields["low_cutoff"]),
                )
            ),
            high_cutoff=float(
                values.get(
                    "high_cutoff",
                    self._resolved_default(fields["high_cutoff"]),
                )
            ),
            order=int(
                values.get(
                    "order",
                    self._resolved_default(fields["order"]),
                )
            ),
            minimum=float(fields["low_cutoff"].minimum or 0.0),
            maximum=nyquist,
            decimals=max(
                fields["low_cutoff"].decimals,
                fields["high_cutoff"].decimals,
            ),
            suffix=(
                " 周期/物理单位"
                if frequency_unit == "cycles_per_unit"
                else " 周期/像素"
            ),
        )
        editor.setEnabled(definition.available_for_new_recipe)
        proxies: dict[str, QWidget] = {
            "mode": editor.modeCombo,
            "low_cutoff": editor.lowCutoffSpin,
            "high_cutoff": editor.highCutoffSpin,
            "order": editor.orderSpin,
        }
        for key in parameter_keys:
            self._parameter_widgets[key] = proxies[key]
            self._parameter_row_widgets[key] = editor
        self._structured_parameter_editors[
            "frequency_response"
        ] = editor
        tooltip = (
            "曲线显示当前 Butterworth 幅频响应；截止频率和阶数"
            "以精确数值写入配方。边界策略与频率单位在下方单独设置。"
        )
        label = QLabel("频率响应", self._parameter_content)
        label.setToolTip(tooltip)
        editor.setToolTip(tooltip)
        self._parameter_form.addRow(label)
        self._parameter_form.addRow(editor)
        editor.validationChanged.connect(
            self._frequency_editor_validation_changed
        )
        editor.editFinished.connect(self._parameter_value_changed)
        return editor

    def _frequency_editor_validation_changed(
        self,
        valid: bool,
        message: str,
    ) -> None:
        self._structured_parameter_error_message = (
            "" if valid else message
        )
        if not self._updating_parameter_form:
            self._refresh_parameter_validation()

    def _add_stripe_frequency_editor(
        self,
        definition: WorkbenchOperationDefinition,
        values: Mapping[str, object],
        parameter_keys: tuple[str, ...],
    ) -> StripeSuppressionEditor:
        fields = {field.key: field for field in definition.parameters}
        editor = StripeSuppressionEditor(
            self._parameter_content,
            direction=str(
                values.get(
                    "direction",
                    self._resolved_default(fields["direction"]),
                )
            ),
            notch_width=float(
                values.get(
                    "notch_width",
                    self._resolved_default(fields["notch_width"]),
                )
            ),
            protect_radius=float(
                values.get(
                    "protect_radius",
                    self._resolved_default(fields["protect_radius"]),
                )
            ),
            decimals=max(
                fields["notch_width"].decimals,
                fields["protect_radius"].decimals,
            ),
        )
        editor.setEnabled(definition.available_for_new_recipe)
        proxies: dict[str, QWidget] = {
            "direction": editor.directionCombo,
            "notch_width": editor.notchWidthSpin,
            "protect_radius": editor.protectRadiusSpin,
        }
        for key in parameter_keys:
            self._parameter_widgets[key] = proxies[key]
            self._parameter_row_widgets[key] = editor
        self._structured_parameter_editors[
            "stripe_frequency"
        ] = editor
        tooltip = (
            "频谱示意明确空间条纹方向、被抑制的频率轴、陷波宽度"
            "与中心低频保护区；它不改变算法，只帮助核对参数。"
        )
        label = QLabel("方向频谱", self._parameter_content)
        label.setToolTip(tooltip)
        editor.setToolTip(tooltip)
        self._parameter_form.addRow(label)
        self._parameter_form.addRow(editor)
        editor.editFinished.connect(self._parameter_value_changed)
        return editor

    def _add_compatible_image_picker(
        self,
        definition: WorkbenchOperationDefinition,
        parameter: ParameterField,
        value: object,
        *,
        enabled: bool,
    ) -> QWidget:
        """Show a second-image selector with explicit compatibility evidence."""

        container = QWidget(self._parameter_content)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        combo = NoWheelComboBox(container)
        input_state = self._input_state_for_step(
            self._steps_list.currentRow()
        )
        for document_id, plane in self._secondary_images.items():
            name = self._secondary_image_names.get(
                document_id,
                document_id,
            )
            compatible = (
                plane.width == input_state.width
                and plane.height == input_state.height
                and plane.pixel_type is input_state.pixel_type
            )
            prefix = "✓" if compatible else "⚠"
            combo.addItem(
                f"{prefix} {name} · {plane.width}×{plane.height} · "
                f"{plane.pixel_type.value}",
                document_id,
            )
            combo.setItemData(
                combo.count() - 1,
                (
                    "尺寸、通道和像素类型与当前步骤输入一致。"
                    if compatible
                    else (
                        "与当前步骤输入 "
                        f"{input_state.width}×{input_state.height} · "
                        f"{input_state.pixel_type.value} 不兼容。"
                    )
                ),
                Qt.ItemDataRole.ToolTipRole,
            )
        selected = combo.findData(value)
        combo.setCurrentIndex(max(0, selected))
        layout.addWidget(combo)
        status = QLabel(container)
        status.setObjectName("compatibleImageStatus")
        status.setWordWrap(True)
        layout.addWidget(status)
        container.setEnabled(enabled)

        self._parameter_widgets[parameter.key] = combo
        self._parameter_row_widgets[parameter.key] = container
        self._structured_parameter_editors[
            f"compatible-image:{parameter.key}"
        ] = container
        label = QLabel(parameter.label, self._parameter_content)
        tooltip = parameter.help_text or (
            "仅尺寸、像素类型和通道与当前步骤输入一致的图片可用于计算；"
            "项目入口还会校验标定。"
        )
        label.setToolTip(tooltip)
        container.setToolTip(tooltip)
        self._parameter_form.addRow(label)
        self._parameter_form.addRow(container)

        def refresh_status() -> None:
            document_id = str(combo.currentData() or "")
            plane = self._secondary_images.get(document_id)
            if plane is None:
                status.setText("未选择仍然可用的第二幅图像。")
                status.setProperty("compatible", False)
            else:
                compatible = (
                    plane.width == input_state.width
                    and plane.height == input_state.height
                    and plane.pixel_type is input_state.pixel_type
                )
                status.setText(
                    (
                        "兼容：尺寸、通道和像素类型一致；"
                        "像素摘要会在最终处理时冻结。"
                    )
                    if compatible
                    else (
                        "不兼容：当前步骤输入为 "
                        f"{input_state.width}×{input_state.height} · "
                        f"{input_state.pixel_type.value}。"
                    )
                )
                status.setProperty("compatible", compatible)
            style = status.style()
            style.unpolish(status)
            style.polish(status)

        def selection_changed(_index: int) -> None:
            refresh_status()
            self._parameter_value_changed()

        combo.currentIndexChanged.connect(selection_changed)
        refresh_status()
        return container

    def _histogram_context(
        self,
    ) -> tuple[RasterPlane, NDArray[np.bool_] | None, str] | None:
        row = self._steps_list.currentRow()
        if (
            self._histogram_parameter_editor is None
            or row != self._parameter_input_step_index
        ):
            return None
        channel = "luminance"
        channel_widget = self._parameter_widgets.get("channel")
        if isinstance(channel_widget, NoWheelComboBox):
            channel = str(channel_widget.currentData() or "luminance")
        elif (
            0 <= row < len(self._steps)
            and self._steps[row].operation_id
            == ImageOperation.ADJUST_LEVELS.value
            and not self._parameter_input_raster.pixel_type.is_grayscale
        ):
            channel = "all_channels"
        mask = (
            self._parameter_input_roi_mask
            if self._roi_is_active()
            else None
        )
        return self._parameter_input_raster, mask, channel

    def _update_specialized_parameter_data(self) -> None:
        self._update_histogram_editor_data()
        self._update_percentile_editor_data()

    def _update_histogram_editor_data(self) -> None:
        editor = self._histogram_parameter_editor
        if editor is None:
            return
        context = self._histogram_context()
        if context is None:
            editor.clearSelectionStatistics()
            editor.selectionStatisticsLabel.setText(
                "正在读取当前步骤输入的 1:1 样本…"
            )
            return
        raster, roi_mask, channel = context
        row = self._steps_list.currentRow()
        is_canny = (
            0 <= row < len(self._steps)
            and self._steps[row].operation_id
            == ImageOperation.CANNY_EDGES.value
        )
        lower, upper = editor.thresholds()
        try:
            histogram_raster = (
                self._canny_gradient_raster(raster, channel=channel)
                if is_canny
                else raster
            )
            snapshot = parameter_histogram_snapshot(
                histogram_raster,
                channel=(
                    "luminance"
                    if is_canny
                    else channel
                ),
                roi_mask=roi_mask,
                range_hint=(lower, upper),
            )
        except (TypeError, ValueError) as exc:
            editor.clearSelectionStatistics()
            editor.selectionStatisticsLabel.setText(
                f"直方图不可用：{exc}"
            )
            return
        editor.setHistogram(
            snapshot.counts,
            value_range=(snapshot.minimum, snapshot.maximum),
        )
        invert_widget = self._parameter_widgets.get("invert")
        if isinstance(invert_widget, QCheckBox):
            editor.setForegroundPolarity(
                "dark" if invert_widget.isChecked() else "bright",
                emit_signal=False,
            )
        self._refresh_histogram_selection_statistics()
        if snapshot.nonfinite_count:
            editor.selectionStatisticsLabel.setText(
                editor.selectionStatisticsLabel.text()
                + f"；另有 {snapshot.nonfinite_count:,} 个 NaN/Inf 未参与统计"
            )

    def _update_percentile_editor_data(self) -> None:
        editor = self._percentile_parameter_editor
        row = self._steps_list.currentRow()
        if editor is None:
            return
        if (
            row != self._parameter_input_step_index
            or not 0 <= row < len(self._steps)
        ):
            editor.setResolvedText(
                "正在读取当前步骤输入的实际强度分位值…"
            )
            return
        raster = self._parameter_input_raster
        array = np.asarray(raster_plane_to_array(raster))
        roi_mask = (
            self._parameter_input_roi_mask
            if self._roi_is_active()
            else None
        )
        if roi_mask is not None and roi_mask.shape != array.shape[:2]:
            editor.setResolvedText(
                "当前 ROI 与步骤输入尺寸不一致，无法解析强度分位值。"
            )
            return
        lower, upper = editor.value()

        def finite_values(
            values: NDArray[np.generic],
        ) -> NDArray[np.float64]:
            selected = (
                values
                if roi_mask is None
                else values[np.asarray(roi_mask, dtype=np.bool_)]
            )
            normalized = np.asarray(selected, dtype=np.float64)
            return normalized[np.isfinite(normalized)]

        try:
            per_channel_widget = self._parameter_widgets.get(
                "per_channel"
            )
            per_channel = (
                per_channel_widget.isChecked()
                if isinstance(per_channel_widget, QCheckBox)
                else True
            )
            if array.ndim == 3 and per_channel:
                channel_names = ("R", "G", "B")
                parts: list[str] = []
                sample_count = 0
                for channel_index, channel_name in enumerate(
                    channel_names
                ):
                    values = finite_values(
                        array[..., channel_index]
                    )
                    if not values.size:
                        continue
                    low_value, high_value = np.percentile(
                        values,
                        (lower, upper),
                    )
                    sample_count += int(values.size)
                    parts.append(
                        f"{channel_name} "
                        f"{float(low_value):.6g}–"
                        f"{float(high_value):.6g}"
                    )
                if not parts:
                    raise ValueError("当前范围不含有限像素")
                editor.setResolvedText(
                    "按通道解析："
                    + "；".join(parts)
                    + f" · 共 {sample_count:,} 个通道样本"
                )
            else:
                values = finite_values(
                    array[..., :3]
                    if array.ndim == 3
                    else array
                )
                if not values.size:
                    raise ValueError("当前范围不含有限像素")
                low_value, high_value = np.percentile(
                    values,
                    (lower, upper),
                )
                label = (
                    "合并 RGB 通道"
                    if array.ndim == 3
                    else "灰度"
                )
                editor.setResolvedText(
                    f"{label}解析强度："
                    f"{float(low_value):.6g}–"
                    f"{float(high_value):.6g}"
                    f" · {values.size:,} 个有限样本"
                )
        except (TypeError, ValueError) as exc:
            editor.setResolvedText(f"无法解析强度分位值：{exc}")

    def _refresh_histogram_selection_statistics(self) -> None:
        editor = self._histogram_parameter_editor
        context = self._histogram_context()
        if editor is None or context is None:
            return
        raster, roi_mask, channel = context
        row = self._steps_list.currentRow()
        is_canny = (
            0 <= row < len(self._steps)
            and self._steps[row].operation_id
            == ImageOperation.CANNY_EDGES.value
        )
        lower, upper = editor.thresholds()
        invert_widget = self._parameter_widgets.get("invert")
        invert = (
            invert_widget.isChecked()
            if isinstance(invert_widget, QCheckBox)
            else False
        )
        try:
            statistics_raster = (
                self._canny_gradient_raster(raster, channel=channel)
                if is_canny
                else raster
            )
            statistics_channel = "luminance" if is_canny else channel
            selected, total = count_parameter_range(
                statistics_raster,
                lower=lower,
                upper=upper,
                channel=statistics_channel,
                roi_mask=roi_mask,
                single_threshold=editor.isSingleThreshold(),
                invert=invert,
            )
            scalar = np.asarray(
                scalar_parameter_samples(
                    statistics_raster,
                    channel=statistics_channel,
                ),
                dtype=np.float64,
            )
            active = np.isfinite(scalar)
            if roi_mask is not None:
                if roi_mask.shape != scalar.shape[:2]:
                    raise ValueError(
                        "当前 ROI 与阈值统计输入尺寸不一致"
                    )
                normalized_roi = np.asarray(
                    roi_mask,
                    dtype=np.bool_,
                )
                active &= (
                    normalized_roi
                    if scalar.ndim == 2
                    else normalized_roi[..., np.newaxis]
                )
            below = int(np.count_nonzero(active & (scalar < lower)))
            if editor.isSingleThreshold():
                within = int(
                    np.count_nonzero(active & (scalar == lower))
                )
                above = int(
                    np.count_nonzero(active & (scalar > lower))
                )
            else:
                within = int(
                    np.count_nonzero(
                        active
                        & (scalar >= lower)
                        & (scalar <= upper)
                    )
                )
                above = int(
                    np.count_nonzero(active & (scalar > upper))
                )
        except (TypeError, ValueError):
            editor.clearSelectionStatistics()
            return
        editor.setBandStatistics(
            below_count=below,
            within_count=within,
            above_count=above,
            total_count=total,
            foreground_count=selected,
        )
        if is_canny:
            editor.selectionStatisticsLabel.setText(
                f"低于低阈值 {below:,} · "
                f"弱梯度候选 {within:,} · "
                f"强梯度候选 {above:,} · "
                f"样本总数 {total:,}"
            )
            editor.selectionStatisticsLabel.setToolTip(
                "这里统计的是阈值前的 Sobel 梯度候选；"
                "Canny 还会执行非极大值抑制和滞后连接，"
                "因此这些数量不等于最终边缘像素数。"
            )

    def _canny_gradient_raster(
        self,
        raster: RasterPlane,
        *,
        channel: str,
    ) -> RasterPlane:
        row = self._steps_list.currentRow()
        if not 0 <= row < len(self._steps):
            raise ValueError("当前没有可用的 Canny 步骤")
        definition = _DEFINITION_BY_ID[self._steps[row].operation_id]
        parameters = self._steps[row].parameters
        try:
            parameters.update(
                self._parameter_values_from_form(definition)
            )
        except (KeyError, TypeError, ValueError):
            # During the first form population some dependent widgets may not
            # exist yet.  Persisted values are still scientifically valid for
            # constructing the first histogram.
            pass
        magnitude = canny_gradient_magnitude(
            raster_plane_to_array(raster),
            aperture_size=int(parameters.get("aperture_size", 3)),
            l2_gradient=bool(parameters.get("l2_gradient", True)),
            channel=channel,
        )
        return array_to_raster_plane(magnitude)

    def _auto_histogram_parameters(self) -> None:
        editor = self._histogram_parameter_editor
        context = self._histogram_context()
        row = self._steps_list.currentRow()
        if editor is None or context is None or not 0 <= row < len(self._steps):
            self._status_label.setText(
                "当前步骤输入尚未准备好，暂时不能自动计算参数。"
            )
            return
        raster, roi_mask, channel = context
        operation_id = self._steps[row].operation_id
        try:
            if operation_id == ImageOperation.ADJUST_LEVELS.value:
                samples = scalar_parameter_samples(
                    raster,
                    channel=channel,
                )
                active = np.isfinite(samples)
                if roi_mask is not None:
                    if roi_mask.shape != samples.shape[:2]:
                        raise ValueError("当前 ROI 与色阶输入尺寸不一致")
                    active &= (
                        roi_mask
                        if samples.ndim == 2
                        else roi_mask[..., np.newaxis]
                    )
                finite = np.asarray(samples[active], dtype=np.float64)
                if finite.size == 0:
                    raise ValueError("当前输入不含可用于自动色阶的有限像素")
                lower, upper = (
                    float(value)
                    for value in np.percentile(
                        finite,
                        (0.35, 99.65),
                    )
                )
                if upper <= lower:
                    lower = float(np.min(finite))
                    upper = float(np.max(finite))
                if upper <= lower:
                    raise ValueError("当前输入是常量图像，无法自动设置黑白场")
                editor.setThresholds(lower, upper, emit_signal=False)
            elif operation_id in {
                ImageOperation.THRESHOLD.value,
                ImageOperation.BINARIZE.value,
            }:
                _binary, threshold = auto_threshold(
                    raster_plane_to_array(raster),
                    method="otsu",
                    channel=channel,
                    statistics_mask=roi_mask,
                )
                if editor.isSingleThreshold():
                    editor.setThreshold(threshold, emit_signal=False)
                else:
                    _minimum, maximum = editor.range()
                    editor.setThresholds(
                        threshold,
                        maximum,
                        emit_signal=False,
                    )
            else:
                return
        except (TypeError, ValueError) as exc:
            self._status_label.setText(f"自动参数计算失败：{exc}")
            return
        self._refresh_histogram_selection_statistics()
        self._update_preview_display()
        x, y, width, height = self._preview_snapshot.bounds
        full_width, full_height = self._preview_snapshot.full_source_size
        self._pending_parameter_result_metadata = {
            "auto_parameter_source": {
                "scope": (
                    "full_source"
                    if self._preview_snapshot.is_full_source
                    else "preview_sample"
                ),
                "sample_bounds": [x, y, width, height],
                "full_source_size": [full_width, full_height],
                "roi_statistics": bool(roi_mask is not None),
                "method": (
                    "percentile_0.35_99.65"
                    if operation_id
                    == ImageOperation.ADJUST_LEVELS.value
                    else "otsu"
                ),
            }
        }
        self._parameter_value_changed()
        if self._preview_snapshot.is_full_source:
            self._status_label.setText(
                "自动参数已按完整源图统计并写入配方。"
            )
        else:
            self._status_label.setText(
                "自动参数已按当前 1:1 预览样本统计并写入配方；"
                f"样本范围 x={x}、y={y}、{width}×{height}，"
                "它不是整图自动统计。"
            )

    def _reset_histogram_parameters(self) -> None:
        editor = self._histogram_parameter_editor
        row = self._steps_list.currentRow()
        if editor is None or not 0 <= row < len(self._steps):
            return
        definition = _DEFINITION_BY_ID[self._steps[row].operation_id]
        fields = {field.key: field for field in definition.parameters}
        defaults = tuple(
            float(self._resolved_default(fields[key]))
            for key in self._histogram_parameter_keys
        )
        minimum, maximum = editor.range()
        lower = min(maximum, max(minimum, defaults[0]))
        upper = (
            maximum
            if editor.isSingleThreshold()
            else min(maximum, max(lower, defaults[1]))
        )
        editor.setThresholds(lower, upper, emit_signal=False)
        self._refresh_histogram_selection_statistics()
        self._update_preview_display()
        self._parameter_value_changed()

    def _histogram_polarity_changed(self, polarity: str) -> None:
        invert_widget = self._parameter_widgets.get("invert")
        if not isinstance(invert_widget, QCheckBox):
            return
        inverted = str(polarity) == "dark"
        blocked = invert_widget.blockSignals(True)
        try:
            invert_widget.setChecked(inverted)
        finally:
            invert_widget.blockSignals(blocked)
        self._refresh_histogram_selection_statistics()
        self._update_preview_display()
        self._parameter_value_changed()

    def _add_anchor_grid_editor(
        self,
        parameter: ParameterField,
        value: object,
        *,
        enabled: bool,
    ) -> AnchorGridEditor:
        """Add a visual 3×3 anchor while preserving the choice-widget contract."""

        editor = AnchorGridEditor(
            self._parameter_content,
            value=str(value),
        )
        editor.setEnabled(enabled)
        proxy = NoWheelComboBox(editor)
        for label, data in parameter.choices:
            proxy.addItem(label, data)
        selected = proxy.findData(value)
        proxy.setCurrentIndex(max(0, selected))
        proxy.hide()

        self._parameter_widgets[parameter.key] = proxy
        self._parameter_row_widgets[parameter.key] = editor
        self._structured_parameter_editors[parameter.key] = editor

        tooltip = (
            parameter.help_text
            or "选择原图在调整后画布中的固定锚点；该值会写入可追溯配方。"
        )
        label = QLabel(parameter.label, self._parameter_content)
        label.setToolTip(tooltip)
        editor.setToolTip(tooltip)
        self._parameter_form.addRow(label)
        self._parameter_form.addRow(editor)

        editor.valueChanged.connect(
            lambda anchor, choice=proxy: self._anchor_editor_changed(
                choice,
                anchor,
            )
        )
        proxy.currentIndexChanged.connect(
            lambda _index, choice=proxy, grid=editor: (
                self._anchor_proxy_changed(choice, grid)
            )
        )
        return editor

    def _anchor_editor_changed(
        self,
        proxy: NoWheelComboBox,
        anchor: str,
    ) -> None:
        selected = proxy.findData(anchor)
        if selected < 0:
            return
        blocked = proxy.blockSignals(True)
        try:
            proxy.setCurrentIndex(selected)
        finally:
            proxy.blockSignals(blocked)
        self._parameter_value_changed()

    def _anchor_proxy_changed(
        self,
        proxy: NoWheelComboBox,
        editor: AnchorGridEditor,
    ) -> None:
        value = proxy.currentData()
        if value is None:
            return
        editor.setValue(str(value), emit_signal=False)
        self._parameter_value_changed()

    def _add_kernel_matrix_editor(
        self,
        definition: WorkbenchOperationDefinition,
        values: Mapping[str, object],
    ) -> KernelMatrixEditor:
        """Add one matrix editor backed by the three legacy parameter widgets."""

        fields = {field.key: field for field in definition.parameters}
        width = int(
            values.get(
                "kernel_width",
                self._resolved_default(fields["kernel_width"]),
            )
        )
        height = int(
            values.get(
                "kernel_height",
                self._resolved_default(fields["kernel_height"]),
            )
        )
        raw_values = values.get(
            "kernel",
            self._resolved_default(fields["kernel"]),
        )
        flat_values = tuple(
            float(item)
            for item in (
                raw_values
                if isinstance(raw_values, (tuple, list))
                else (raw_values,)
            )
        )
        padded_values = list(flat_values[: width * height])
        padded_values.extend(
            0.0 for _ in range(width * height - len(padded_values))
        )
        matrix = tuple(
            tuple(padded_values[row * width : (row + 1) * width])
            for row in range(height)
        )
        editor = KernelMatrixEditor(
            self._parameter_content,
            kernel=matrix,
            maximum_dimension=99,
        )
        editor.setEnabled(definition.available_for_new_recipe)

        # Keep the existing private mapping contract: external automation and
        # the generic serializer still see two spin boxes and one line edit.
        width_proxy = editor.widthSpin
        height_proxy = editor.heightSpin
        kernel_proxy = QLineEdit(editor)
        kernel_proxy.setText(
            ", ".join(f"{value:g}" for value in flat_values)
        )
        kernel_proxy.hide()
        self._parameter_widgets["kernel_width"] = width_proxy
        self._parameter_widgets["kernel_height"] = height_proxy
        self._parameter_widgets["kernel"] = kernel_proxy
        for key in ("kernel_width", "kernel_height", "kernel"):
            self._parameter_row_widgets[key] = editor
        self._structured_parameter_editors["kernel"] = editor

        tooltip = (
            "按二维矩阵编辑卷积核；宽高与元素数量会同步写入可追溯配方。"
            "可使用预设后继续逐格调整。"
        )
        label = QLabel("卷积核", self._parameter_content)
        label.setToolTip(tooltip)
        editor.setToolTip(tooltip)
        self._parameter_form.addRow(label)
        self._parameter_form.addRow(editor)

        editor.kernelChanged.connect(
            lambda matrix_value, line=kernel_proxy: (
                self._kernel_editor_changed(line, matrix_value)
            )
        )
        editor.validationChanged.connect(
            self._kernel_editor_validation_changed
        )
        kernel_proxy.editingFinished.connect(
            lambda line=kernel_proxy, matrix_editor=editor: (
                self._kernel_proxy_edited(line, matrix_editor)
            )
        )
        return editor

    def _kernel_editor_changed(
        self,
        proxy: QLineEdit,
        matrix: object,
    ) -> None:
        rows = tuple(tuple(row) for row in matrix)  # type: ignore[arg-type]
        flat_values = tuple(value for row in rows for value in row)
        blocked = proxy.blockSignals(True)
        try:
            proxy.setText(", ".join(f"{float(value):g}" for value in flat_values))
        finally:
            proxy.blockSignals(blocked)
        self._structured_parameter_error_message = ""
        self._parameter_value_changed()

    def _kernel_proxy_edited(
        self,
        proxy: QLineEdit,
        editor: KernelMatrixEditor,
    ) -> None:
        field = next(
            field
            for field in _DEFINITION_BY_ID[
                ImageOperation.CUSTOM_CONVOLUTION.value
            ].parameters
            if field.key == "kernel"
        )
        try:
            values = tuple(
                self._parameter_widget_value(field, proxy)  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            self._parameter_value_changed()
            return
        width, height = editor.dimensions()
        if len(values) != width * height:
            self._parameter_value_changed()
            return
        matrix = tuple(
            tuple(values[row * width : (row + 1) * width])
            for row in range(height)
        )
        try:
            editor.setKernel(matrix)
        except ValueError as exc:
            self._structured_parameter_error_message = str(exc)
            self._refresh_parameter_validation()

    def _kernel_editor_validation_changed(
        self,
        valid: bool,
        message: str,
    ) -> None:
        self._structured_parameter_error_message = (
            "" if valid else message
        )
        if not self._updating_parameter_form:
            self._refresh_parameter_validation()

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

    def _parameter_widget_value(
        self,
        field: ParameterField,
        widget: QWidget,
    ) -> object:
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, NoWheelSpinBox):
            return widget.value()
        if isinstance(widget, NoWheelDoubleSpinBox):
            return widget.value()
        if isinstance(widget, NoWheelComboBox):
            return widget.currentData()
        if isinstance(widget, QLineEdit):
            tokens = (
                widget.text()
                .replace(";", " ")
                .replace(",", " ")
                .split()
            )
            try:
                numbers = tuple(float(token) for token in tokens)
            except ValueError as exc:
                raise ValueError(
                    f"{field.label}只能包含以逗号或空格分隔的数值。"
                ) from exc
            if not numbers or not all(math.isfinite(item) for item in numbers):
                raise ValueError(
                    f"{field.label}必须包含至少一个有限数值。"
                )
            return numbers
        raise TypeError(f"未知参数控件: {type(widget).__name__}")

    def _parameter_values_from_form(
        self,
        definition: WorkbenchOperationDefinition,
    ) -> dict[str, object]:
        return {
            field.key: self._parameter_widget_value(
                field,
                self._parameter_widgets[field.key],
            )
            for field in definition.parameters
        }

    def _refresh_parameter_conditions(self) -> None:
        if self._updating_parameter_form:
            return
        row = self._steps_list.currentRow()
        if not 0 <= row < len(self._steps):
            return
        definition = _DEFINITION_BY_ID[self._steps[row].operation_id]
        try:
            parameters = self._parameter_values_from_form(definition)
            input_state = self._input_state_for_step(row)
        except (TypeError, ValueError):
            return
        for field in definition.parameters:
            widget = self._parameter_widgets[field.key]
            row_widget = self._parameter_row_widgets.get(
                field.key,
                widget,
            )
            visible = _parameter_is_relevant(
                definition.operation.value,
                field.key,
                parameters,
                input_state=input_state,
                roi_available=self._roi_mask is not None,
            )
            self._parameter_form.setRowVisible(row_widget, visible)
        frequency_editor = self._structured_parameter_editors.get(
            "frequency_response"
        )
        frequency_unit = self._parameter_widgets.get("frequency_unit")
        if (
            isinstance(frequency_editor, FrequencyResponseEditor)
            and isinstance(frequency_unit, NoWheelComboBox)
        ):
            unit_value = str(
                frequency_unit.currentData()
                or "cycles_per_pixel"
            )
            suffix = (
                " 周期/物理单位"
                if unit_value == "cycles_per_unit"
                else " 周期/像素"
            )
            frequency_editor.lowCutoffEditor.setSuffix(suffix)
            frequency_editor.highCutoffEditor.setSuffix(suffix)
            pixel_size_widget = self._parameter_widgets.get(
                "pixel_size"
            )
            pixel_size = (
                float(pixel_size_widget.value())
                if isinstance(
                    pixel_size_widget,
                    NoWheelDoubleSpinBox,
                )
                else 1.0
            )
            nyquist = (
                0.5 / pixel_size
                if unit_value == "cycles_per_unit"
                else 0.5
            )
            frequency_editor.setFrequencyRange(0.0, nyquist)

    def _refresh_parameter_validation(self) -> str:
        row = self._steps_list.currentRow()
        if not 0 <= row < len(self._steps):
            self._parameter_error_message = ""
            return ""
        definition = _DEFINITION_BY_ID[self._steps[row].operation_id]
        message = self._structured_parameter_error_message
        if not message:
            try:
                parameters = self._steps[row].parameters
                parameters.update(
                    self._parameter_values_from_form(definition)
                )
                input_state = self._input_state_for_step(row)
                get_image_operation_descriptor(
                    definition.operation
                ).validate_parameters(parameters)
                message = _parameter_relationship_error(
                    definition.operation.value,
                    parameters,
                    input_state=input_state,
                )
                if not message:
                    message = self._secondary_image_parameter_error(
                        definition.operation,
                        parameters,
                        input_state=input_state,
                    )
                if not message:
                    current = self._steps[row]
                    replacement = ImageOperationSpec(
                        current.operation_id,
                        parameters,
                        implementation=current.implementation,
                        implementation_version=(
                            current.implementation_version
                        ),
                        result_metadata=current.result_metadata,
                    )
                    candidate_steps = list(self._steps)
                    candidate_steps[row] = replacement
                    validate_workbench_operation_sequence(
                        self._source,
                        tuple(candidate_steps),
                        source_semantic=self._source_semantic,
                        roi_requested=self._roi_is_active(),
                        secondary_images=self._secondary_images,
                        secondary_semantics=(
                            self._secondary_image_semantics
                        ),
                    )
            except (TypeError, ValueError) as exc:
                message = str(exc)
        self._parameter_error_message = message
        label = self._parameter_validation_label
        if label is not None:
            label.setText(f"参数需要调整：{message}" if message else "")
            label.setVisible(bool(message))
        self._update_actions()
        return message

    def _secondary_image_parameter_error(
        self,
        operation: ImageOperation,
        parameters: Mapping[str, object],
        *,
        input_state: RasterTypeState,
    ) -> str:
        needs_secondary = operation is ImageOperation.IMAGE_CALCULATOR
        if operation is ImageOperation.FLAT_FIELD_CORRECTION:
            needs_secondary = (
                str(
                    parameters.get(
                        "flat_field_source",
                        "estimated",
                    )
                )
                == "reference"
            )
        if not needs_secondary:
            return ""
        document_id = str(
            parameters.get("secondary_document_id", "")
        ).strip()
        plane = self._secondary_images.get(document_id)
        if plane is None:
            return "请选择一幅仍然可用的兼容第二图像。"
        if (
            plane.pixel_type is not input_state.pixel_type
            or plane.width != input_state.width
            or plane.height != input_state.height
        ):
            return (
                "第二图像必须与当前步骤输入的尺寸、通道和像素类型"
                "完全一致。"
            )
        return ""

    def _parameter_value_changed(self, *_signal_values: object) -> None:
        if self._updating_parameter_form:
            return
        row = self._steps_list.currentRow()
        if not 0 <= row < len(self._steps):
            return
        definition = _DEFINITION_BY_ID[self._steps[row].operation_id]
        if _operation_step_is_replay_only(self._steps[row]):
            self._pending_parameter_result_metadata.clear()
            return
        pending_metadata = dict(
            self._pending_parameter_result_metadata
        )
        self._pending_parameter_result_metadata.clear()
        try:
            parameters = self._steps[row].parameters
            parameters.update(
                self._parameter_values_from_form(definition)
            )
        except (TypeError, ValueError) as exc:
            self._status_label.setText(str(exc))
            self._refresh_parameter_validation()
            return
        QTimer.singleShot(0, self._refresh_parameter_conditions)
        QTimer.singleShot(
            0,
            self._update_specialized_parameter_data,
        )
        if self._refresh_parameter_validation():
            return
        current = self._steps[row]
        try:
            result_metadata = current.result_metadata
            result_metadata.update(pending_metadata)
            replacement = ImageOperationSpec(
                current.operation_id,
                parameters,
                implementation=current.implementation,
                implementation_version=current.implementation_version,
                result_metadata=result_metadata,
            )
        except (TypeError, ValueError) as exc:
            self._status_label.setText(f"参数无效：{exc}")
            return
        if replacement == current:
            return
        updated = list(self._steps)
        updated[row] = replacement
        self._commit_parameter_steps(tuple(updated))

    def _commit_parameter_steps(
        self,
        steps: tuple[ImageOperationSpec, ...],
    ) -> None:
        """Commit parameter values without destroying the signal sender.

        Rebuilding the dynamic form synchronously from a checkbox, combo box
        or editor signal deletes that emitting QWidget before Qt has returned
        from the native signal.  Parameter edits do not change the step label
        or form structure, so only the immutable recipe model and history need
        updating here.  Switching steps, undo and redo continue to rebuild the
        form through their normal paths.
        """

        if steps == self._steps:
            return
        self._undo_stack.append(self._steps)
        if len(self._undo_stack) > 100:
            del self._undo_stack[0]
        self._redo_stack.clear()
        self._steps = tuple(steps)
        self._schedule_preview()

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
        if (
            result.parameter_input_raster is not None
            and result.parameter_input_step_index
            == self._steps_list.currentRow()
        ):
            self._parameter_input_raster = result.parameter_input_raster
            self._parameter_input_roi_mask = (
                result.parameter_input_roi_mask
            )
            self._parameter_input_step_index = (
                result.parameter_input_step_index
            )
        crop = self._preview_crop_by_request_id.pop(
            result.request_id,
            None,
        )
        if crop is not None:
            x, y, width, height = crop
            array = raster_plane_to_array(result.raster)
            if (
                x < 0
                or y < 0
                or x + width > result.raster.width
                or y + height > result.raster.height
            ):
                self._status_label.setText(
                    "预览结果尺寸与冻结邻域不一致，已拒绝显示。"
                )
                return
            result = WorkbenchTaskResult(
                kind=result.kind,
                request_id=result.request_id,
                generation=result.generation,
                source_document_id=result.source_document_id,
                raster=array_to_raster_plane(
                    np.ascontiguousarray(
                        array[y : y + height, x : x + width, ...]
                    )
                ),
                recipe=result.recipe,
                output_semantic=result.output_semantic,
                parameter_input_raster=result.parameter_input_raster,
                parameter_input_roi_mask=(
                    result.parameter_input_roi_mask
                ),
                parameter_input_step_index=(
                    result.parameter_input_step_index
                ),
            )
        self._show_preview_raster(result.raster)
        self._update_specialized_parameter_data()
        self._status_label.setText(
            f"预览已更新 · {result.raster.width} × {result.raster.height}"
        )
        self.previewChanged.emit(result)

    def _generate_derived_image(self) -> None:
        if not self._steps or self._final_in_progress:
            return
        try:
            prepared_steps = self._prepare_reference_flat_field_steps(
                self._steps
            )
            self._validate_active_roi_operations(prepared_steps)
            validate_workbench_operation_sequence(
                self._source,
                prepared_steps,
                source_semantic=self._source_semantic,
                roi_requested=self._roi_is_active(),
                secondary_images=self._secondary_images,
                secondary_semantics=self._secondary_image_semantics,
            )
            estimate = validate_final_resources(
                self._source,
                prepared_steps,
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
            operations=prepared_steps,
            source_semantic=self._source_semantic,
            roi_mask=self._roi_mask if self._roi_is_active() else None,
            secondary_images=self._secondary_images,
            secondary_semantics=self._secondary_image_semantics,
        )

    def _prepare_reference_flat_field_steps(
        self,
        operations: tuple[ImageOperationSpec, ...],
    ) -> tuple[ImageOperationSpec, ...]:
        """Freeze the full reference identity and normalization levels.

        A bounded 1:1 preview receives a cropped reference raster.  Freezing
        levels from the complete compatible reference here keeps that preview
        numerically consistent with final full-resolution processing.
        """

        prepared: list[ImageOperationSpec] = []
        for operation in operations:
            parameters = operation.parameters
            is_reference_flat_field = (
                operation.operation_id
                == ImageOperation.FLAT_FIELD_CORRECTION.value
                and str(
                    parameters.get("flat_field_source", "estimated")
                ).strip().lower()
                == "reference"
            )
            if not is_reference_flat_field:
                prepared.append(operation)
                continue
            secondary_document_id = str(
                parameters.get("secondary_document_id", "")
            ).strip()
            if secondary_document_id not in self._secondary_images:
                raise ValueError(
                    "参考图平场校正需要选择一幅仍然可用的兼容参考图像"
                )
            actual_sha256 = self._secondary_image_sha256(
                secondary_document_id
            )
            expected_sha256 = str(
                parameters.get("secondary_sha256", "")
            ).strip()
            if expected_sha256 and expected_sha256 != actual_sha256:
                raise ValueError(
                    "参考平场像素已变化，当前配方的来源摘要已经过期"
                )
            parameters["secondary_sha256"] = actual_sha256
            preserve_mean = bool(parameters.get("preserve_mean", True))
            parameters["reference_levels"] = (
                self._flat_field_reference_level_cache(
                    secondary_document_id,
                    preserve_mean=preserve_mean,
                )
            )
            prepared.append(
                ImageOperationSpec(
                    operation.operation_id,
                    parameters,
                    implementation=operation.implementation,
                    implementation_version=operation.implementation_version,
                    result_metadata=operation.result_metadata,
                )
            )
        return tuple(prepared)

    def _secondary_image_sha256(self, document_id: str) -> str:
        cached = self._secondary_sha256_cache.get(document_id)
        if cached is not None:
            return cached
        try:
            plane = self._secondary_images[document_id]
        except KeyError as exc:
            raise ValueError("第二幅图像已不在当前兼容候选列表中") from exc
        digest = plane.sha256()
        self._secondary_sha256_cache[document_id] = digest
        return digest

    def _flat_field_reference_level_cache(
        self,
        document_id: str,
        *,
        preserve_mean: bool,
    ) -> tuple[float, ...]:
        key = (document_id, bool(preserve_mean))
        cached = self._flat_field_level_cache.get(key)
        if cached is not None:
            return cached
        try:
            plane = self._secondary_images[document_id]
        except KeyError as exc:
            raise ValueError("参考平场图像已不在当前兼容候选列表中") from exc
        levels = flat_field_reference_levels(
            raster_plane_to_array(plane),
            preserve_mean=preserve_mean,
        )
        self._flat_field_level_cache[key] = levels
        return levels

    def _roi_is_active(self) -> bool:
        return bool(
            self._roi_mask is not None
            and getattr(self, "_use_roi_checkbox", None) is not None
            and self._use_roi_checkbox.isChecked()
        )

    def _validate_active_roi_operations(
        self,
        operations: tuple[ImageOperationSpec, ...],
    ) -> None:
        if not self._roi_is_active():
            return
        incompatible = [
            image_operation_display_name(operation.operation_id)
            for operation in operations
            if not resolve_image_operation_capability(
                operation.operation_id,
                operation.parameters,
            ).supports_roi
        ]
        if incompatible:
            raise ValueError(
                "以下操作不能隐式套用当前 ROI："
                + "、".join(incompatible)
                + "。请关闭“限制在当前 ROI”后按整图处理，或移除这些步骤。"
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
        if self._steps and not self._contains_replay_only_steps():
            self.recipeSaveRequested.emit(self.current_recipe())

    def _request_batch_apply(self) -> None:
        if self._steps and not self._contains_replay_only_steps():
            self.batchApplyRequested.emit(self.current_recipe())

    def _contains_replay_only_steps(self) -> bool:
        return any(_operation_step_is_replay_only(step) for step in self._steps)

    def _update_actions(self) -> None:
        row = self._steps_list.currentRow()
        count = len(self._steps)
        final_busy = self._controller.is_busy(WorkbenchTaskKind.FINAL)
        replay_only = self._contains_replay_only_steps()
        self._move_up_button.setEnabled(0 < row < count and not replay_only)
        self._move_down_button.setEnabled(
            0 <= row < count - 1 and not replay_only
        )
        self._remove_step_button.setEnabled(
            0 <= row < count and not replay_only
        )
        self._reset_steps_button.setEnabled(bool(self._steps) and not replay_only)
        self._undo_button.setEnabled(bool(self._undo_stack) and not replay_only)
        self._redo_button.setEnabled(bool(self._redo_stack) and not replay_only)
        parameters_valid = not bool(self._parameter_error_message)
        self._generate_button.setEnabled(
            bool(self._steps) and not final_busy and parameters_valid
        )
        self._save_recipe_button.setEnabled(
            bool(self._steps)
            and not final_busy
            and not replay_only
            and parameters_valid
        )
        self._load_recipe_button.setEnabled(not final_busy)
        self._batch_apply_button.setEnabled(
            bool(self._steps)
            and not final_busy
            and not replay_only
            and parameters_valid
        )
        replay_tooltip = (
            "此配方包含仅供旧项目重放的兼容步骤；"
            "旧版参数和步骤顺序已锁定，不能另存或批量应用。"
        )
        self._save_recipe_button.setToolTip(
            replay_tooltip
            if replay_only
            else "请求工作区保存当前有序步骤；工作台本身不直接写设置文件。"
        )
        self._batch_apply_button.setToolTip(
            replay_tooltip
            if replay_only
            else (
                "把当前配方交给批处理窗口；"
                "成功结果仍需由项目工作区统一提交。"
            )
        )
        self._add_step_button.setEnabled(not final_busy and not replay_only)

    def _show_preview_raster(self, raster: RasterPlane) -> None:
        image = raster_plane_to_display_image(raster)
        self._latest_preview_raster = raster
        self._latest_preview_image = image
        self._processed_overview_image = (
            self._overview_image_for_processed_preview(raster, image)
        )
        self._update_preview_display()

    def _threshold_parameter_preview_image(self) -> QImage | None:
        """Build a display-only threshold overlay over the frozen step input."""

        editor = self._histogram_parameter_editor
        row = self._steps_list.currentRow()
        if (
            editor is None
            or editor.displayMode() == "bw"
            or not 0 <= row < len(self._steps)
            or row != len(self._steps) - 1
            or row != self._parameter_input_step_index
        ):
            return None
        operation_id = self._steps[row].operation_id
        if operation_id not in {
            ImageOperation.THRESHOLD.value,
            ImageOperation.BINARIZE.value,
        }:
            return None
        raster = self._parameter_input_raster
        if (
            raster.width != self._latest_preview_raster.width
            or raster.height != self._latest_preview_raster.height
        ):
            return None
        context = self._histogram_context()
        if context is None:
            return None
        _raster, roi_mask, channel = context
        try:
            scalar = np.asarray(
                scalar_parameter_samples(raster, channel=channel),
                dtype=np.float64,
            )
        except ValueError:
            return None
        active = np.isfinite(scalar)
        if roi_mask is not None:
            if roi_mask.shape != scalar.shape:
                return None
            active &= np.asarray(roi_mask, dtype=np.bool_)
        lower, upper = editor.thresholds()
        selected = (
            scalar > lower
            if editor.isSingleThreshold()
            else (scalar >= lower) & (scalar <= upper)
        )
        selected &= active
        invert_widget = self._parameter_widgets.get("invert")
        if (
            isinstance(invert_widget, QCheckBox)
            and invert_widget.isChecked()
        ):
            selected = active & ~selected

        overlay = np.zeros(
            (raster.height, raster.width, 4),
            dtype=np.uint8,
        )
        if editor.displayMode() == "red_overlay":
            overlay[selected] = (239, 68, 68, 112)
        elif editor.displayMode() == "over_under":
            below = active & (scalar < lower)
            over_limit = lower if editor.isSingleThreshold() else upper
            above = active & (scalar > over_limit)
            overlay[below] = (37, 99, 235, 116)
            overlay[above] = (239, 68, 68, 116)
        else:
            return None
        overlay_image = QImage(
            overlay.data,
            raster.width,
            raster.height,
            int(overlay.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()
        image = raster_plane_to_display_image(raster).convertToFormat(
            QImage.Format.Format_RGBA8888
        )
        painter = QPainter(image)
        try:
            painter.drawImage(0, 0, overlay_image)
        finally:
            painter.end()
        return image

    def _overview_image_for_processed_preview(
        self,
        raster: RasterPlane,
        preview_image: QImage,
    ) -> QImage:
        if not self._steps:
            self._overview_note_text = (
                "当前显示原图概览；添加处理步骤后，这里会同步显示"
                "最新处理结果。"
            )
            return raster_plane_to_bounded_overview_image(raster)
        if self._preview_snapshot.is_full_source:
            self._overview_note_text = (
                "当前显示处理后的完整图片概览；显示已按比例缩放，"
                "像素级判断请切回 1:1。"
            )
            return raster_plane_to_bounded_overview_image(raster)

        sample_width = self._preview_snapshot.source.width
        sample_height = self._preview_snapshot.source.height
        if (raster.width, raster.height) != (
            sample_width,
            sample_height,
        ):
            self._overview_note_text = (
                "当前配方改变了样本几何尺寸，因此这里只显示处理后的"
                "样本概览，不将其伪装成完整图片结果。"
            )
            return raster_plane_to_bounded_overview_image(raster)

        overview = self._overview_image.copy()
        full_width, full_height = self._preview_snapshot.full_source_size
        x, y, width, height = self._preview_snapshot.bounds
        scale_x = overview.width() / float(full_width)
        scale_y = overview.height() / float(full_height)
        sample_rect = QRectF(
            x * scale_x,
            y * scale_y,
            max(1.0, width * scale_x),
            max(1.0, height * scale_y),
        )
        painter = QPainter(overview)
        try:
            painter.drawImage(
                sample_rect,
                preview_image,
                QRectF(preview_image.rect()),
            )
            painter.setPen(QPen(QColor("#2A9D8F"), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(sample_rect)
        finally:
            painter.end()
        self._overview_note_text = (
            "绿色框内为处理后的原始像素样本，框外保留原图用于定位；"
            "最终处理仍会作用于完整原始图片。"
        )
        return overview

    def _update_preview_display(self, *, force_fit: bool = False) -> None:
        overview = self._overview_checkbox.isChecked()
        if overview:
            image = self._processed_overview_image
        else:
            threshold_image = self._threshold_parameter_preview_image()
            image = (
                threshold_image
                if threshold_image is not None
                else self._latest_preview_image
            )
        self._overview_note.setText(self._overview_note_text)
        self._overview_note.setVisible(overview)
        self._preview_view.set_image(image, force_fit=force_fit)

    def _toggle_overview(self, checked: bool) -> None:
        self._update_preview_display(force_fit=True)

    def _on_preview_zoom_changed(
        self,
        factor: float,
        fit_mode: bool,
    ) -> None:
        percentage = max(1, int(round(float(factor) * 100.0)))
        self._preview_zoom_label.setText(
            f"适合 · {percentage}%" if fit_mode else f"{percentage}%"
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


def raster_plane_to_bounded_overview_image(plane: RasterPlane) -> QImage:
    """Create an approximate overview without constructing a full-size QPixmap."""

    width = int(plane.width)
    height = int(plane.height)
    edge_scale = max(
        width / float(OVERVIEW_MAX_EDGE),
        height / float(OVERVIEW_MAX_EDGE),
        1.0,
    )
    pixel_scale = math.sqrt(
        max(1.0, (width * height) / float(OVERVIEW_MAX_PIXELS))
    )
    stride = max(1, int(math.ceil(max(edge_scale, pixel_scale))))
    array = raster_plane_to_array(plane)
    sampled = np.ascontiguousarray(array[::stride, ::stride, ...])
    return raster_plane_to_display_image(array_to_raster_plane(sampled))


__all__ = [
    "FinalResourceEstimate",
    "FinalResourcePreflightError",
    "ImageProcessingTaskController",
    "ImageProcessingWorkbench",
    "ProcessingPreviewSnapshot",
    "WorkbenchTaskKind",
    "WorkbenchTaskRequest",
    "WorkbenchTaskResult",
    "array_to_raster_plane",
    "adapt_operations_for_preview",
    "build_processing_preview_snapshot",
    "expand_processing_preview_snapshot_for_halo",
    "default_operation_spec",
    "estimate_final_resources",
    "execute_workbench_request",
    "raster_plane_to_array",
    "raster_plane_to_display_image",
    "raster_plane_to_bounded_overview_image",
    "validate_final_resources",
    "validate_workbench_operation_sequence",
]
