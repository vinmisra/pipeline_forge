import pytest
import pandas as pd
from pipeline_forge.stages.pipeline_stage import PipelineStage
from pipeline_forge.stages.functional_stage import FunctionalStage
from pipeline_forge.stages.llm_stage import LLMStage
from pipeline_forge.pipeline import Pipeline
from pipeline_forge.llm.provider import MockProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_name, input_data, pipeline_config, expected_output, expected_columns, filter_config",
    [
        # Basic pipeline stage with a single functional stage inside
        (
            "basic_pipeline_stage",
            {"input_value": [1, 2, 3, 4]},
            {
                "input_columns": ["input_value"],
                "output_columns": ["result"],
                "inner_stages": [
                    {
                        "type": "functional",
                        "input_columns": ["input_value"],
                        "output_columns": ["result"],
                        "function": lambda x: x * 10,
                    }
                ],
            },
            {"result": [10, 20, 30, 40]},
            ["input_value", "result"],  # All expected columns in output
            None,
        ),
        # Pipeline stage with multiple inner stages and dependencies
        # Check both final and intermediate columns
        (
            "multi_stage_pipeline",
            {"input_value": [5, 10, 15]},
            {
                "input_columns": ["input_value"],
                "output_columns": ["final_result", "doubled"],  # Request both outputs
                "inner_stages": [
                    {
                        "type": "functional",
                        "input_columns": ["input_value"],
                        "output_columns": ["doubled"],
                        "function": lambda x: x * 2,
                    },
                    {
                        "type": "functional",
                        "input_columns": ["doubled"],
                        "output_columns": ["final_result"],
                        "function": lambda x: x + 5,
                    },
                ],
            },
            {"doubled": [10, 20, 30], "final_result": [15, 25, 35]},
            ["input_value", "doubled", "final_result"],  # All expected columns
            None,
        ),
        # Pipeline stage with functional stages
        # Request only final output
        (
            "final_output_only",
            {"length": [10, 20]},
            {
                "input_columns": ["length"],
                "output_columns": ["processed_length"],  # Only request final output
                "inner_stages": [
                    {
                        "type": "functional",
                        "input_columns": ["length"],
                        "output_columns": ["intermediate_response"],
                        "function": lambda x: "-" * x,
                    },
                    {
                        "type": "functional",
                        "input_columns": ["intermediate_response"],
                        "output_columns": ["processed_length"],
                        "function": lambda x: len(x),
                    },
                ],
            },
            {"processed_length": [10, 20]},
            ["length", "processed_length"],
            None,
        ),
        # Pipeline stage with filtering - check that all columns exist
        (
            "with_filter",
            {
                "input_value": [1, 2, 3, 4],
                "process_row": [True, False, True, False],
            },
            {
                "input_columns": ["input_value"],
                "output_columns": ["squared", "doubled"],
                "inner_stages": [
                    {
                        "type": "functional",
                        "input_columns": ["input_value"],
                        "output_columns": ["doubled"],
                        "function": lambda x: x * 2,
                    },
                    {
                        "type": "functional",
                        "input_columns": ["doubled"],
                        "output_columns": ["squared"],
                        "function": lambda x: x**2,
                    },
                ],
            },
            {"doubled": [2, None, 6, None], "squared": [4, None, 36, None]},
            ["input_value", "process_row", "doubled", "squared"],
            "process_row",
        ),
        # Empty dataframe
        (
            "empty_data",
            {"input_value": []},
            {
                "input_columns": ["input_value"],
                "output_columns": ["result"],
                "inner_stages": [
                    {
                        "type": "functional",
                        "input_columns": ["input_value"],
                        "output_columns": ["result"],
                        "function": lambda x: x * 2,
                    }
                ],
            },
            {"result": []},
            ["input_value", "result"],
            None,
        ),
    ],
)
async def test_pipeline_stage_parametrized(
    test_name,
    input_data,
    pipeline_config,
    expected_output,
    expected_columns,
    filter_config,
):
    """Parametrized test for PipelineStage with different configurations."""
    # Create test data
    data = pd.DataFrame(input_data)

    # Create inner pipeline stages
    inner_stages = []
    for stage_config in pipeline_config["inner_stages"]:
        if stage_config["type"] == "functional":
            inner_stages.append(
                FunctionalStage(
                    input_columns=stage_config["input_columns"],
                    output_columns=stage_config["output_columns"],
                    function=stage_config["function"],
                )
            )
        elif stage_config["type"] == "llm":
            inner_stages.append(
                LLMStage(
                    input_columns=stage_config["input_columns"],
                    output_columns=stage_config["output_columns"],
                    conversation_template=stage_config["conversation_template"],
                )
            )

    # Create inner pipeline
    inner_pipeline = Pipeline(stages=inner_stages)

    # Create pipeline stage
    pipeline_stage = PipelineStage(
        input_columns=pipeline_config["input_columns"],
        output_columns=pipeline_config["output_columns"],
        pipeline=inner_pipeline,
        filter_colname=filter_config,
    )

    # Create mock provider for LLM stages
    mock_provider = MockProvider(
        default_response="This is a mock LLM response" * 2
    )  # 40 chars

    # Process data
    result = await pipeline_stage.process(data, llm_provider=mock_provider)

    # Check that all expected columns are present
    for col in expected_columns:
        assert col in result.columns, f"Expected column '{col}' missing from result"

    # Check that only expected columns are present from the pipeline
    pipeline_columns = set(result.columns) - set(input_data.keys())
    expected_pipeline_columns = set(expected_columns) - set(input_data.keys())
    assert (
        pipeline_columns == expected_pipeline_columns
    ), f"Unexpected columns in result: {pipeline_columns - expected_pipeline_columns}"

    # For empty dataframe case, just check the structure
    if len(data) == 0:
        assert len(result) == 0
        return

    # Check values of expected outputs
    for col, expected_values in expected_output.items():
        for i, expected in enumerate(expected_values):
            if expected is None:
                assert pd.isna(result[col][i]), f"Expected None at {col}[{i}]"
            else:
                assert result[col][i] == expected, f"Mismatch at {col}[{i}]"
