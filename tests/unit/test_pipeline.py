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

    # Create pipeline with stages in arbitrary order
    pipeline = Pipeline(stages=[stage_c, stage_a, stage_d, stage_b])

    # Run just stage C - should compute A and B as dependencies
    result = await pipeline.run_stage(
        stage=stage_c, llm_provider=MockProvider(), data=data, cache=InMemoryCache()
    )

    assert "A" in result.columns
    assert "B" in result.columns
    assert "C" in result.columns
    assert "D" not in result.columns

    # Verify values are correct
    assert result["A"].tolist() == [2, 4, 6, 8]  # input_value * 2
    assert result["B"].tolist() == [7, 9, 11, 13]  # A + 5
    assert result["C"].tolist() == [70, 90, 110, 130]  # B * 10

    # Run full pipeline
    result = await pipeline.run(data, llm_provider=MockProvider(), cache=InMemoryCache())

    assert "A" in result.columns
    assert "B" in result.columns
    assert "C" in result.columns
    assert "D" in result.columns

    assert result["A"].tolist() == [2, 4, 6, 8]  # input_value * 2
    assert result["B"].tolist() == [7, 9, 11, 13]  # A + 5
    assert result["C"].tolist() == [70, 90, 110, 130]  # B * 10
    assert result["D"].tolist() == [100, 200, 300, 400]  # input_value * 100


