import pytest
import pandas as pd
from pipeline_forge.stages.functional_stage import FunctionalStage
from pipeline_forge.llm.provider import MockProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_name, input_data, stage_config, expected_output, cache_hits_expected",
    [
        # Basic single input to single output
        (
            "single_input_output",
            {"value": [1, 2, 3, 4]},
            {
                "input_columns": ["value"],
                "output_columns": ["result"],
                "function": lambda x: x * 2,
                "filter_colname": None,
            },
            {"result": [2, 4, 6, 8]},
            0,
        ),
        # Multiple inputs to single output
        (
            "multiple_inputs",
            {"a": [1, 2, 3], "b": [10, 20, 30]},
            {
                "input_columns": ["a", "b"],
                "output_columns": ["sum"],
                "function": lambda a, b: a + b,
                "filter_colname": None,
            },
            {"sum": [11, 22, 33]},
            0,
        ),
        # Single input to multiple outputs
        (
            "multiple_outputs",
            {"value": [5, 10, 15]},
            {
                "input_columns": ["value"],
                "output_columns": ["doubled", "squared"],
                "function": lambda x: (x * 2, x**2),
                "filter_colname": None,
            },
            {"doubled": [10, 20, 30], "squared": [25, 100, 225]},
            0,
        ),
        # Using filter_colname
        (
            "with_filter",
            {"value": [1, 2, 3, 4], "process": [True, False, True, False]},
            {
                "input_columns": ["value"],
                "output_columns": ["result"],
                "function": lambda x: x * 10,
                "filter_colname": "process",
            },
            {"result": [10, None, 30, None]},
            0,
        ),
        # Empty dataframe
        (
            "empty_data",
            {"value": []},
            {
                "input_columns": ["value"],
                "output_columns": ["result"],
                "function": lambda x: x * 2,
                "filter_colname": None,
            },
            {"result": []},
            0,
        ),
    ],
)
async def test_functional_stage_parametrized(
    test_name, input_data, stage_config, expected_output, cache_hits_expected
):
    """Parametrized test for FunctionalStage with different configurations and scenarios."""
    # Create test data
    data = pd.DataFrame(input_data)

    # Create functional stage
    stage = FunctionalStage(
        input_columns=stage_config["input_columns"],
        output_columns=stage_config["output_columns"],
        function=stage_config["function"],
        filter_colname=stage_config["filter_colname"],
    )

    # Process data
    result = await stage.process(data, llm_provider=MockProvider())

    # Check that all expected output columns exist
    for col in stage_config["output_columns"]:
        assert col in result.columns

    # For empty dataframe case, just check the structure
    if len(data) == 0:
        assert len(result) == 0
        return

    # Check results match expected values
    for col, expected_values in expected_output.items():
        for i, expected in enumerate(expected_values):
            if expected is None:
                assert pd.isna(result[col][i]), f"Expected None at {col}[{i}]"
            else:
                assert result[col][i] == expected, f"Mismatch at {col}[{i}]"
