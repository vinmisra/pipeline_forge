import pytest
import pandas as pd
import asyncio
from pipeline_forge.pipeline import Pipeline
from pipeline_forge.stages.functional_stage import FunctionalStage
from pipeline_forge.stages.llm_stage import LLMStage
from pipeline_forge.cache import InMemoryCache
from pipeline_forge.llm.provider import MockProvider
from pipeline_forge.stages.pipeline_stage import PipelineStage


class MockProvider:
    pass


@pytest.mark.asyncio
async def test_dependency_resolution():
    """Test that the pipeline correctly resolves and executes dependencies."""
    # Create test data
    data = pd.DataFrame(
        {
            "input_value": [1, 2, 3, 4],
        }
    )

    # Create stages with dependencies:
    # input_value -> A -> B -> C
    #             -> D
    # input_value -> filter_flag 
    # filter_flag + B -> E

    stage_a = FunctionalStage(
        input_columns=["input_value"],
        function=lambda x: x * 2,
        output_columns=["A"],
    )

    stage_b = FunctionalStage(
        input_columns=["A"], function=lambda x: x + 5, output_columns=["B"]
    )

    stage_c = FunctionalStage(
        input_columns=["B"], function=lambda x: x * 10, output_columns=["C"]
    )

    stage_d = FunctionalStage(
        input_columns=["input_value"],
        function=lambda x: x * 100,
        output_columns=["D"],
    )

    # Stage that computes a filter flag based on input_value
    filter_stage = FunctionalStage(
        input_columns=["input_value"],
        function=lambda x: x % 2 == 1,  # True for odd numbers
        output_columns=["filter_flag"],
    )

    # Stage with computed filter - only processes when filter_flag is True
    stage_e = FunctionalStage(
        input_columns=["B"],
        function=lambda x: x**2,
        output_columns=["E"],
        filter_colname="filter_flag",
    )

    # Create pipeline with stages in arbitrary order
    pipeline = Pipeline(
        stages=[stage_a, stage_b, stage_c, stage_d, filter_stage, stage_e]
    )

    # Run just stage E - should compute A, B, and filter_flag as dependencies
    result = await pipeline.run_stage(
        stage=stage_e, llm_provider=MockProvider(), data=data, cache=InMemoryCache()
    )

    assert "A" in result.columns
    assert "B" in result.columns
    assert "filter_flag" in result.columns
    assert "E" in result.columns
    assert "C" not in result.columns
    assert "D" not in result.columns

    # Verify filtering worked correctly (only rows where filter_flag is True - odd input values)
    assert result["filter_flag"].tolist() == [True, False, True, False]
    assert result["E"].tolist() == [
        49,
        None,
        121,
        None,
    ]  # B^2 where filter_flag is True

    # Run full pipeline
    result = await pipeline.run(
        data, llm_provider=MockProvider(), cache=InMemoryCache()
    )

    assert "A" in result.columns
    assert "B" in result.columns
    assert "C" in result.columns
    assert "D" in result.columns
    assert "E" in result.columns
    assert "filter_flag" in result.columns

    assert result["A"].tolist() == [2, 4, 6, 8]  # input_value * 2
    assert result["B"].tolist() == [7, 9, 11, 13]  # A + 5
    assert result["C"].tolist() == [70, 90, 110, 130]  # B * 10
    assert result["D"].tolist() == [100, 200, 300, 400]  # input_value * 100
    assert result["filter_flag"].tolist() == [
        True,
        False,
        True,
        False,
    ]  # odd numbers -> True
    assert result["E"].tolist() == [
        49,
        None,
        121,
        None,
    ]  # B^2 where filter_flag is True
