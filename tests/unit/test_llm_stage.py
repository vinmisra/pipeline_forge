import pandas as pd
from pipeline_forge.stages.llm_stage import LLMStage
from pipeline_forge.llm.provider import MockProvider
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_name, input_data, stage_config, expected_responses, provider_config",
    [
        # Basic operation (current test case)
        (
            "basic_operation",
            {"input_text": ["Hello, world!", "Test message"]},
            {
                "input_columns": ["input_text"],
                "conversation_template": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "{input_text}"},
                ],
                "output_columns": ["response"],
                "filter_colname": None,
            },
            ["Response for Hello, world!", "Response for Test message"],
            {
                "Hello, world!": "Response for Hello, world!",
                "Test message": "Response for Test message",
            },
        ),
        # Using filter_colname to process only certain rows
        (
            "with_filter",
            {
                "input_text": ["Hello, world!", "Test message"],
                "process_row": [True, False],
            },
            {
                "input_columns": ["input_text"],
                "conversation_template": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "{input_text}"},
                ],
                "output_columns": ["response"],
                "filter_colname": "process_row",
            },
            ["Response for Hello, world!", None],
            {"Hello, world!": "Response for Hello, world!"},
        ),
        # Empty dataframe handling
        (
            "empty_data",
            {"input_text": []},
            {
                "input_columns": ["input_text"],
                "conversation_template": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "{input_text}"},
                ],
                "output_columns": ["response"],
                "filter_colname": None,
            },
            [],
            {},
        ),
    ],
)
async def test_llm_stage_parametrized(
    test_name, input_data, stage_config, expected_responses, provider_config
):
    """Parametrized test for LLM stage with different configurations and scenarios."""
    # Create test data
    data = pd.DataFrame(input_data)

    # Create LLM stage
    llm_stage = LLMStage(
        input_columns=stage_config["input_columns"],
        conversation_template=stage_config["conversation_template"],
        output_columns=stage_config["output_columns"],
        filter_colname=stage_config["filter_colname"],
    )

    # Create mock provider with appropriate response mapping
    mock_provider = MockProvider(map_responses=provider_config)

    # Process data
    result = await llm_stage.process(data, llm_provider=mock_provider)

    # Check results
    assert stage_config["output_columns"][0] in result.columns

    # For empty dataframe case, just check the structure
    if len(expected_responses) == 0:
        assert len(result) == 0
        return

    # Check each expected response
    for i, expected in enumerate(expected_responses):
        if expected is None:
            assert pd.isna(result[stage_config["output_columns"][0]][i])
        else:
            assert result[stage_config["output_columns"][0]][i] == expected
