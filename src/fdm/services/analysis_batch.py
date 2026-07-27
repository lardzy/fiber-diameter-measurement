"""Deterministic recipe/batch execution for advanced analyses.

An :class:`AnalysisRecipe` may contain one or more independent analysis tool
invocations.  Every source item runs the recipe steps sequentially and is
published only after all of its steps succeed.  The legacy single-tool
constructor and result ``execution`` attribute remain available to callers
that have not yet adopted multi-tool recipes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
import json
from types import MappingProxyType

from fdm.cancellation import CancellationError, CancellationToken
from fdm.services.advanced_analysis_registry import (
    AdvancedAnalysisExecution,
    AdvancedAnalysisInvocation,
    AdvancedAnalysisRegistry,
)
from fdm.services.advanced_image_analysis import (
    AdvancedAnalysisKind,
    AdvancedAnalysisLimits,
    DEFAULT_ADVANCED_ANALYSIS_LIMITS,
)


class AnalysisSourceKind(StrEnum):
    IMAGE = "image"
    DIGITAL_SLIDE = "digital_slide"


@dataclass(frozen=True, slots=True)
class AnalysisViewport:
    x: int
    y: int
    width: int
    height: int
    level: int = 0

    def __post_init__(self) -> None:
        for name in ("x", "y", "level"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"viewport.{name} 不能为负数")
            object.__setattr__(self, name, value)
        for name in ("width", "height"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"viewport.{name} 必须为正整数")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, init=False)
class AnalysisToolInvocation:
    """One immutable tool step inside an :class:`AnalysisRecipe`."""

    kind: AdvancedAnalysisKind
    required_inputs: tuple[str, ...]
    _parameters_json: str = field(repr=False)

    def __init__(
        self,
        kind: AdvancedAnalysisKind | str,
        *,
        parameters: Mapping[str, object] | None = None,
        required_inputs: Iterable[str] = (),
    ) -> None:
        normalized_inputs = tuple(
            dict.fromkeys(str(item).strip() for item in required_inputs)
        )
        if any(not item for item in normalized_inputs):
            raise ValueError("required_inputs 不能包含空值")
        parameters_json = json.dumps(
            dict(parameters or {}),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        object.__setattr__(self, "kind", AdvancedAnalysisKind(kind))
        object.__setattr__(self, "required_inputs", normalized_inputs)
        object.__setattr__(self, "_parameters_json", parameters_json)

    @property
    def parameters(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self._parameters_json))


@dataclass(frozen=True, slots=True, init=False)
class AnalysisRecipe:
    """An ordered, immutable collection of advanced-analysis tool steps.

    ``kind``/``parameters``/``required_inputs`` construct a legacy single-step
    recipe.  New callers can instead pass ``invocations``.  Compatibility
    properties with the legacy names expose the first step because existing UI
    code uses it to freeze the source request and to choose the result adapter.
    """

    recipe_id: str
    name: str
    invocations: tuple[AnalysisToolInvocation, ...]

    def __init__(
        self,
        recipe_id: str,
        name: str,
        kind: AdvancedAnalysisKind | str | None = None,
        *,
        parameters: Mapping[str, object] | None = None,
        required_inputs: Iterable[str] = (),
        invocations: Iterable[AnalysisToolInvocation] | None = None,
    ) -> None:
        recipe_token = str(recipe_id or "").strip()
        name_token = str(name or "").strip()
        if not recipe_token or not name_token:
            raise ValueError("recipe_id 和 name 不能为空")
        if invocations is None:
            if kind is None:
                raise ValueError("单工具配方必须指定 kind")
            normalized_invocations = (
                AnalysisToolInvocation(
                    kind,
                    parameters=parameters,
                    required_inputs=required_inputs,
                ),
            )
        else:
            normalized_invocations = tuple(invocations)
            if not normalized_invocations:
                raise ValueError("分析配方至少需要一个工具步骤")
            if any(
                not isinstance(item, AnalysisToolInvocation)
                for item in normalized_invocations
            ):
                raise TypeError(
                    "invocations 必须全部是 AnalysisToolInvocation"
                )
            if kind is not None and (
                AdvancedAnalysisKind(kind)
                is not normalized_invocations[0].kind
            ):
                raise ValueError("kind 必须与配方首个工具步骤一致")
            if parameters:
                raise ValueError(
                    "多工具配方请在 AnalysisToolInvocation 中设置 parameters"
                )
            if tuple(required_inputs):
                raise ValueError(
                    "多工具配方请在 AnalysisToolInvocation 中设置 required_inputs"
                )
        object.__setattr__(self, "recipe_id", recipe_token)
        object.__setattr__(self, "name", name_token)
        object.__setattr__(self, "invocations", normalized_invocations)

    @property
    def tool_invocations(self) -> tuple[AnalysisToolInvocation, ...]:
        """Explicit alias for callers that prefer the longer domain name."""

        return self.invocations

    @property
    def kind(self) -> AdvancedAnalysisKind:
        """Legacy first-step analysis kind."""

        return self.invocations[0].kind

    @property
    def parameters(self) -> Mapping[str, object]:
        """Legacy first-step parameters."""

        return self.invocations[0].parameters

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Legacy first-step required inputs."""

        return self.invocations[0].required_inputs

    @property
    def all_required_inputs(self) -> tuple[str, ...]:
        """Union of dependencies required by every recipe step."""

        return tuple(
            dict.fromkeys(
                required_input
                for invocation in self.invocations
                for required_input in invocation.required_inputs
            )
        )


@dataclass(frozen=True, slots=True)
class AnalysisInvocation:
    item_id: str
    display_name: str
    analysis: AdvancedAnalysisInvocation
    source_kind: AnalysisSourceKind = AnalysisSourceKind.IMAGE
    viewport: AnalysisViewport | None = None

    def __post_init__(self) -> None:
        item_id = str(self.item_id or "").strip()
        display_name = str(self.display_name or "").strip()
        if not item_id or not display_name:
            raise ValueError("item_id 和 display_name 不能为空")
        if not isinstance(self.analysis, AdvancedAnalysisInvocation):
            raise TypeError("analysis 必须是 AdvancedAnalysisInvocation")
        source_kind = AnalysisSourceKind(self.source_kind)
        if source_kind is AnalysisSourceKind.DIGITAL_SLIDE and self.viewport is None:
            raise ValueError("数字切片批量分析必须显式指定 viewport")
        if self.viewport is not None and not isinstance(
            self.viewport,
            AnalysisViewport,
        ):
            raise TypeError("viewport 必须是 AnalysisViewport")
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "source_kind", source_kind)


@dataclass(frozen=True, slots=True)
class AnalysisBatchRequest:
    request_id: str
    generation: int
    recipe: AnalysisRecipe
    invocations: tuple[AnalysisInvocation, ...]
    continue_on_error: bool = True

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        if not request_id:
            raise ValueError("request_id 不能为空")
        generation = int(self.generation)
        if generation < 0:
            raise ValueError("generation 不能为负数")
        if not isinstance(self.recipe, AnalysisRecipe):
            raise TypeError("recipe 必须是 AnalysisRecipe")
        invocations = tuple(self.invocations)
        if not invocations:
            raise ValueError("批量分析至少需要一个项目")
        if any(not isinstance(item, AnalysisInvocation) for item in invocations):
            raise TypeError("invocations 必须全部是 AnalysisInvocation")
        item_ids = tuple(item.item_id for item in invocations)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("批量分析 item_id 不能重复")
        for item in invocations:
            if item.analysis.kind is not self.recipe.kind:
                raise ValueError("批量项目的分析类型必须与配方首个步骤一致")
            if item.analysis.generation != generation:
                raise ValueError("批量项目 generation 必须与批次一致")
            available_inputs = {
                "pixel_size",
                "unit",
            }
            if item.analysis.plane is not None:
                available_inputs.add("plane")
            if item.analysis.roi_mask is not None:
                available_inputs.add("roi_mask")
            if item.analysis.binary_mask is not None:
                available_inputs.add("binary_mask")
            if item.analysis.points:
                available_inputs.add("points")
            if item.viewport is not None:
                available_inputs.add("viewport")
            for step_index, invocation in enumerate(
                self.recipe.invocations,
                start=1,
            ):
                missing_inputs = (
                    set(invocation.required_inputs) - available_inputs
                )
                if missing_inputs:
                    raise ValueError(
                        f"批量项目 {item.item_id} 的配方步骤 "
                        f"{step_index}（{invocation.kind.value}）"
                        "缺少配方依赖输入: "
                        + "、".join(sorted(missing_inputs))
                    )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "invocations", invocations)
        object.__setattr__(self, "continue_on_error", bool(self.continue_on_error))


@dataclass(frozen=True, slots=True)
class AnalysisBatchItemResult:
    item_id: str
    display_name: str
    success: bool
    execution: AdvancedAnalysisExecution | None = None
    error_type: str | None = None
    error_message: str | None = None
    executions: tuple[AdvancedAnalysisExecution, ...] = ()

    def __post_init__(self) -> None:
        normalized_executions = tuple(self.executions)
        if self.execution is not None and not normalized_executions:
            normalized_executions = (self.execution,)
        elif self.execution is None and normalized_executions:
            object.__setattr__(self, "execution", normalized_executions[0])
        elif (
            self.execution is not None
            and normalized_executions
            and self.execution is not normalized_executions[0]
        ):
            raise ValueError("execution 必须是 executions 中的首个结果")
        object.__setattr__(self, "executions", normalized_executions)


@dataclass(frozen=True, slots=True)
class AnalysisBatchResult:
    request_id: str
    generation: int
    recipe_id: str
    item_results: tuple[AnalysisBatchItemResult, ...]
    cancelled: bool = False

    @property
    def success_count(self) -> int:
        return sum(item.success for item in self.item_results)

    @property
    def failure_count(self) -> int:
        return sum(not item.success for item in self.item_results)


@dataclass(frozen=True, slots=True)
class AnalysisBatchProgress:
    request_id: str
    generation: int
    completed: int
    total: int
    item_id: str


def builtin_plane_analysis_recipes() -> tuple[AnalysisRecipe, ...]:
    """Return safe batch recipes that need only one immutable raster plane."""

    return (
        AnalysisRecipe(
            "directionality-v2",
            "纤维方向性 v2",
            AdvancedAnalysisKind.DIRECTIONALITY,
            parameters={"algorithm_version": 2},
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "tubeness-v1",
            "Tubeness",
            AdvancedAnalysisKind.TUBENESS,
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "glcm-haralick-v1",
            "Haralick GLCM 纹理",
            AdvancedAnalysisKind.GLCM_HARALICK,
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "intensity-surface-v1",
            "二维强度表面",
            AdvancedAnalysisKind.INTENSITY_SURFACE,
            required_inputs=("plane",),
        ),
        AnalysisRecipe(
            "directionality-and-glcm-v2",
            "方向性 v2 + Haralick GLCM",
            invocations=(
                AnalysisToolInvocation(
                    AdvancedAnalysisKind.DIRECTIONALITY,
                    parameters={"algorithm_version": 2},
                    required_inputs=("plane",),
                ),
                AnalysisToolInvocation(
                    AdvancedAnalysisKind.GLCM_HARALICK,
                    required_inputs=("plane",),
                ),
            ),
        ),
    )


def analysis_step_request_id(
    source_request_id: str,
    step_index: int,
    *,
    step_count: int,
) -> str:
    """Return the stable request id used by one recipe execution.

    A one-step recipe keeps the historical source request id.  Multi-step
    recipes suffix every step, including the first, so callbacks can never be
    accidentally paired with a sibling result from the same frozen source.
    """

    token = str(source_request_id or "").strip()
    if not token:
        raise ValueError("source_request_id 不能为空")
    index = int(step_index)
    count = int(step_count)
    if count < 1 or index < 0 or index >= count:
        raise ValueError("分析配方步骤索引超出范围")
    if count == 1:
        return token
    return f"{token}:step-{index + 1}"


def execute_analysis_batch(
    request: AnalysisBatchRequest,
    *,
    registry: AdvancedAnalysisRegistry | None = None,
    cancellation_token: CancellationToken | None = None,
    limits: AdvancedAnalysisLimits = DEFAULT_ADVANCED_ANALYSIS_LIMITS,
    progress: Callable[[AnalysisBatchProgress], None] | None = None,
) -> AnalysisBatchResult:
    """Execute sources and recipe steps sequentially.

    A source item is appended to the staged result only after every recipe step
    succeeds.  An exception in a later step therefore cannot leak partial
    executions for that source.
    """

    if not isinstance(request, AnalysisBatchRequest):
        raise TypeError("request 必须是 AnalysisBatchRequest")
    active_registry = registry or AdvancedAnalysisRegistry()
    staged: list[AnalysisBatchItemResult] = []
    cancelled = False
    for index, item in enumerate(request.invocations, start=1):
        try:
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            executions: list[AdvancedAnalysisExecution] = []
            for step_index, tool_invocation in enumerate(
                request.recipe.invocations,
            ):
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                merged_parameters = dict(tool_invocation.parameters)
                # Preserve the old per-source override contract for the first
                # step.  Later steps are fully described by their immutable
                # recipe invocation and share only the frozen source inputs.
                if step_index == 0:
                    merged_parameters.update(item.analysis.parameters)
                invocation = AdvancedAnalysisInvocation(
                    tool_invocation.kind,
                    request_id=analysis_step_request_id(
                        item.analysis.request_id,
                        step_index,
                        step_count=len(request.recipe.invocations),
                    ),
                    generation=item.analysis.generation,
                    plane=item.analysis.plane,
                    roi_mask=item.analysis.roi_mask,
                    binary_mask=item.analysis.binary_mask,
                    points=item.analysis.points,
                    pixel_size_x=item.analysis.pixel_size_x,
                    pixel_size_y=item.analysis.pixel_size_y,
                    unit=item.analysis.unit,
                    parameters=merged_parameters,
                )
                executions.append(
                    active_registry.execute(
                        invocation,
                        cancellation_token=cancellation_token,
                        limits=limits,
                    )
                )
            staged.append(
                AnalysisBatchItemResult(
                    item_id=item.item_id,
                    display_name=item.display_name,
                    success=True,
                    execution=executions[0],
                    executions=tuple(executions),
                )
            )
        except CancellationError:
            cancelled = True
            break
        except Exception as exc:
            staged.append(
                AnalysisBatchItemResult(
                    item_id=item.item_id,
                    display_name=item.display_name,
                    success=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            if not request.continue_on_error:
                break
        if progress is not None:
            progress(
                AnalysisBatchProgress(
                    request_id=request.request_id,
                    generation=request.generation,
                    completed=index,
                    total=len(request.invocations),
                    item_id=item.item_id,
                )
            )
    return AnalysisBatchResult(
        request_id=request.request_id,
        generation=request.generation,
        recipe_id=request.recipe.recipe_id,
        item_results=tuple(staged),
        cancelled=cancelled,
    )


# Concise public names used by recipe/batch callers.  The prefixed names remain
# available for clarity at mixed image-processing/analysis call sites.
Invocation = AnalysisInvocation
BatchRequest = AnalysisBatchRequest
ItemResult = AnalysisBatchItemResult
Result = AnalysisBatchResult


__all__ = [
    "AnalysisBatchItemResult",
    "AnalysisBatchProgress",
    "AnalysisBatchRequest",
    "AnalysisBatchResult",
    "AnalysisInvocation",
    "AnalysisRecipe",
    "AnalysisSourceKind",
    "AnalysisToolInvocation",
    "AnalysisViewport",
    "BatchRequest",
    "Invocation",
    "ItemResult",
    "Result",
    "analysis_step_request_id",
    "builtin_plane_analysis_recipes",
    "execute_analysis_batch",
]
