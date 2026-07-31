from __future__ import annotations

import numpy as np
import pytest

from fdm.cancellation import CancellationTokenSource
from fdm.services.advanced_analysis_registry import AdvancedAnalysisInvocation
from fdm.services.advanced_image_analysis import AdvancedAnalysisKind
from fdm.services.analysis_batch import (
    AnalysisBatchItemResult,
    AnalysisBatchRequest,
    AnalysisInvocation,
    AnalysisRecipe,
    AnalysisSourceKind,
    AnalysisToolInvocation,
    AnalysisViewport,
    builtin_plane_analysis_recipes,
    execute_analysis_batch,
)
from fdm.services.raster_io import numpy_to_raster_plane


def _invocation(
    item_id: str,
    *,
    generation: int = 3,
    valid: bool = True,
) -> AnalysisInvocation:
    image = np.zeros((24, 24), dtype=np.uint8)
    image[10:14, 2:22] = 255
    return AnalysisInvocation(
        item_id=item_id,
        display_name=item_id,
        analysis=AdvancedAnalysisInvocation(
            AdvancedAnalysisKind.DIRECTIONALITY,
            request_id=f"request-{item_id}",
            generation=generation,
            plane=numpy_to_raster_plane(image) if valid else None,
            parameters={"bins": 18},
        ),
    )


def test_batch_stages_successes_and_per_item_failures_until_final_result() -> None:
    recipe = AnalysisRecipe(
        "direction-v2",
        "方向性 v2",
        AdvancedAnalysisKind.DIRECTIONALITY,
        parameters={"algorithm_version": 2},
    )
    updates = []
    request = AnalysisBatchRequest(
        request_id="batch-1",
        generation=3,
        recipe=recipe,
        invocations=(_invocation("good"), _invocation("bad", valid=False)),
    )

    result = execute_analysis_batch(request, progress=updates.append)

    assert result.success_count == 1
    assert result.failure_count == 1
    assert not result.cancelled
    assert result.item_results[0].execution is not None
    assert result.item_results[1].error_message
    assert [update.completed for update in updates] == [1, 2]


def test_batch_cancellation_returns_only_completed_staged_items() -> None:
    source = CancellationTokenSource()
    source.cancel()
    request = AnalysisBatchRequest(
        request_id="batch-cancel",
        generation=3,
        recipe=AnalysisRecipe(
            "direction",
            "方向",
            AdvancedAnalysisKind.DIRECTIONALITY,
        ),
        invocations=(_invocation("one"), _invocation("two")),
    )

    result = execute_analysis_batch(
        request,
        cancellation_token=source.token,
    )

    assert result.cancelled
    assert result.item_results == ()


def test_digital_slide_invocation_requires_explicit_viewport() -> None:
    with pytest.raises(ValueError, match="viewport"):
        AnalysisInvocation(
            item_id="slide",
            display_name="切片",
            analysis=_invocation("inner").analysis,
            source_kind=AnalysisSourceKind.DIGITAL_SLIDE,
        )

    invocation = AnalysisInvocation(
        item_id="slide",
        display_name="切片",
        analysis=_invocation("inner").analysis,
        source_kind=AnalysisSourceKind.DIGITAL_SLIDE,
        viewport=AnalysisViewport(0, 0, 512, 512, level=2),
    )
    assert invocation.viewport.level == 2


def test_batch_rejects_mismatched_generation_before_execution() -> None:
    with pytest.raises(ValueError, match="generation"):
        AnalysisBatchRequest(
            request_id="batch",
            generation=4,
            recipe=AnalysisRecipe(
                "direction",
                "方向",
                AdvancedAnalysisKind.DIRECTIONALITY,
            ),
            invocations=(_invocation("old", generation=3),),
        )


def test_recipe_rejects_incomplete_dependency_inputs() -> None:
    with pytest.raises(ValueError, match="依赖输入"):
        AnalysisBatchRequest(
            request_id="batch",
            generation=3,
            recipe=AnalysisRecipe(
                "masked-direction",
                "掩膜方向",
                AdvancedAnalysisKind.DIRECTIONALITY,
                required_inputs=("plane", "roi_mask"),
            ),
            invocations=(_invocation("no-roi"),),
        )


def test_builtin_plane_recipes_include_a_safe_multi_tool_recipe() -> None:
    recipes = builtin_plane_analysis_recipes()

    assert tuple(recipe.kind for recipe in recipes) == (
        AdvancedAnalysisKind.DIRECTIONALITY,
        AdvancedAnalysisKind.TUBENESS,
        AdvancedAnalysisKind.GLCM_HARALICK,
        AdvancedAnalysisKind.INTENSITY_SURFACE,
        AdvancedAnalysisKind.DIRECTIONALITY,
    )
    assert all(recipe.all_required_inputs == ("plane",) for recipe in recipes)
    assert tuple(step.kind for step in recipes[-1].invocations) == (
        AdvancedAnalysisKind.DIRECTIONALITY,
        AdvancedAnalysisKind.GLCM_HARALICK,
    )
    assert recipes[-1].invocations[0].parameters["algorithm_version"] == 2


@pytest.mark.parametrize("recipe", builtin_plane_analysis_recipes())
def test_each_builtin_plane_recipe_executes_without_auxiliary_inputs(
    recipe: AnalysisRecipe,
) -> None:
    rows, columns = np.indices((32, 32))
    image = ((rows * 13 + columns * 7) % 256).astype(np.uint8)
    invocation = AnalysisInvocation(
        item_id="plane-only",
        display_name="普通图片",
        analysis=AdvancedAnalysisInvocation(
            recipe.kind,
            request_id=f"request-{recipe.recipe_id}",
            generation=1,
            plane=numpy_to_raster_plane(image),
        ),
    )

    result = execute_analysis_batch(
        AnalysisBatchRequest(
            request_id=f"batch-{recipe.recipe_id}",
            generation=1,
            recipe=recipe,
            invocations=(invocation,),
        )
    )

    assert result.success_count == 1
    assert result.failure_count == 0
    assert result.item_results[0].execution is not None
    assert result.item_results[0].execution.kind is recipe.kind
    assert len(result.item_results[0].executions) == len(recipe.invocations)


def test_multi_tool_recipe_executes_every_step_in_recipe_order() -> None:
    recipe = AnalysisRecipe(
        "direction-and-texture",
        "方向性与纹理",
        invocations=(
            AnalysisToolInvocation(
                AdvancedAnalysisKind.DIRECTIONALITY,
                parameters={"algorithm_version": 2, "bins": 18},
                required_inputs=("plane",),
            ),
            AnalysisToolInvocation(
                AdvancedAnalysisKind.GLCM_HARALICK,
                parameters={"levels": 16},
                required_inputs=("plane",),
            ),
        ),
    )
    rows, columns = np.indices((32, 32))
    image = ((rows * 13 + columns * 7) % 256).astype(np.uint8)
    invocation = AnalysisInvocation(
        item_id="multi",
        display_name="多工具来源",
        analysis=AdvancedAnalysisInvocation(
            recipe.kind,
            request_id="multi-source",
            generation=2,
            plane=numpy_to_raster_plane(image),
        ),
    )

    result = execute_analysis_batch(
        AnalysisBatchRequest(
            request_id="multi-batch",
            generation=2,
            recipe=recipe,
            invocations=(invocation,),
        )
    )

    assert result.success_count == 1
    item_result = result.item_results[0]
    assert tuple(execution.kind for execution in item_result.executions) == (
        AdvancedAnalysisKind.DIRECTIONALITY,
        AdvancedAnalysisKind.GLCM_HARALICK,
    )
    # The old API remains deterministic: it exposes the first recipe result.
    assert item_result.execution is item_result.executions[0]
    assert recipe.kind is AdvancedAnalysisKind.DIRECTIONALITY
    assert recipe.parameters["algorithm_version"] == 2
    assert recipe.required_inputs == ("plane",)
    assert recipe.all_required_inputs == ("plane",)


def test_multi_tool_recipe_does_not_publish_partial_item_execution() -> None:
    class FailingSecondRegistry:
        def __init__(self) -> None:
            self.calls = []

        def execute(self, invocation, **_kwargs):
            self.calls.append(invocation)
            if len(self.calls) == 2:
                raise RuntimeError("second step failed")
            return invocation

    recipe = AnalysisRecipe(
        "partial-must-not-leak",
        "不泄漏部分结果",
        invocations=(
            AnalysisToolInvocation(AdvancedAnalysisKind.DIRECTIONALITY),
            AnalysisToolInvocation(AdvancedAnalysisKind.GLCM_HARALICK),
        ),
    )
    registry = FailingSecondRegistry()

    result = execute_analysis_batch(
        AnalysisBatchRequest(
            request_id="partial-batch",
            generation=3,
            recipe=recipe,
            invocations=(_invocation("source"),),
        ),
        registry=registry,
    )

    assert len(registry.calls) == 2
    assert result.success_count == 0
    assert result.failure_count == 1
    assert result.item_results[0].execution is None
    assert result.item_results[0].executions == ()
    assert result.item_results[0].error_type == "RuntimeError"


def test_multi_tool_recipe_checks_every_steps_dependencies_up_front() -> None:
    recipe = AnalysisRecipe(
        "late-roi-dependency",
        "后续步骤需要 ROI",
        invocations=(
            AnalysisToolInvocation(
                AdvancedAnalysisKind.DIRECTIONALITY,
                required_inputs=("plane",),
            ),
            AnalysisToolInvocation(
                AdvancedAnalysisKind.GLCM_HARALICK,
                required_inputs=("plane", "roi_mask"),
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"步骤 2.*roi_mask"):
        AnalysisBatchRequest(
            request_id="missing-late-input",
            generation=3,
            recipe=recipe,
            invocations=(_invocation("no-roi"),),
        )


def test_source_parameter_overrides_apply_only_to_legacy_first_step() -> None:
    class RecordingRegistry:
        def __init__(self) -> None:
            self.calls = []

        def execute(self, invocation, **_kwargs):
            self.calls.append(invocation)
            return invocation

    recipe = AnalysisRecipe(
        "step-parameters",
        "步骤参数隔离",
        invocations=(
            AnalysisToolInvocation(
                AdvancedAnalysisKind.DIRECTIONALITY,
                parameters={"bins": 12, "algorithm_version": 2},
            ),
            AnalysisToolInvocation(
                AdvancedAnalysisKind.GLCM_HARALICK,
                parameters={"levels": 16},
            ),
        ),
    )
    registry = RecordingRegistry()

    result = execute_analysis_batch(
        AnalysisBatchRequest(
            request_id="parameters-batch",
            generation=3,
            recipe=recipe,
            invocations=(_invocation("parameters"),),
        ),
        registry=registry,
    )

    assert result.success_count == 1
    assert registry.calls[0].parameters["bins"] == 18
    assert registry.calls[0].parameters["algorithm_version"] == 2
    assert registry.calls[1].parameters == {"levels": 16}


def test_cancellation_between_recipe_steps_discards_the_whole_source_item() -> None:
    source = CancellationTokenSource()

    class CancellingRegistry:
        def __init__(self) -> None:
            self.calls = []

        def execute(self, invocation, **_kwargs):
            self.calls.append(invocation)
            source.cancel()
            return invocation

    recipe = AnalysisRecipe(
        "cancel-between-steps",
        "步骤间取消",
        invocations=(
            AnalysisToolInvocation(AdvancedAnalysisKind.DIRECTIONALITY),
            AnalysisToolInvocation(AdvancedAnalysisKind.GLCM_HARALICK),
        ),
    )
    registry = CancellingRegistry()

    result = execute_analysis_batch(
        AnalysisBatchRequest(
            request_id="cancel-between",
            generation=3,
            recipe=recipe,
            invocations=(_invocation("source"),),
        ),
        registry=registry,
        cancellation_token=source.token,
    )

    assert result.cancelled
    assert len(registry.calls) == 1
    assert result.item_results == ()


def test_batch_item_result_normalizes_legacy_and_multi_execution_views() -> None:
    marker = object()
    legacy = AnalysisBatchItemResult(
        item_id="legacy",
        display_name="旧结果",
        success=True,
        execution=marker,  # type: ignore[arg-type]
    )
    assert legacy.executions == (marker,)

    multi = AnalysisBatchItemResult(
        item_id="multi",
        display_name="多结果",
        success=True,
        executions=(marker,),  # type: ignore[arg-type]
    )
    assert multi.execution is marker
