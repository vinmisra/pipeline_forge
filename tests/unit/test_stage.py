import pandas as pd
import pytest
import asyncio
from typing import Set

from pipeline_forge.stage import Stage
from pipeline_forge.llm.provider import MockProvider
from pipeline_forge.cache import InMemoryCache
from pipeline_forge.llm.provider import LLMProvider
from pipeline_forge.cache import Cache


class SimpleStage(Stage):
    """Simple implementation of Stage for testing."""

    async def _process_post_filter(
        self, data: pd.DataFrame, llm_provider: LLMProvider, cache: Cache | None = None
    ) -> pd.DataFrame:
        """Process each row by setting output columns to 'processed'."""
        result = data.copy()
        for out_col in self.output_columns:
            result[out_col] = "processed"
        return result


@pytest.mark.asyncio
async def test_stage_filter_operation():
    """Test that stage filtering works correctly."""
    # Create test data with a filter column
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "input_col": ["a", "b", "c", "d"],
            "should_process": [True, False, True, False],
        }
    )

    # Create a stage with filter
    stage = SimpleStage(
        input_columns=["input_col"],
        output_columns=["result"],
        filter_colname="should_process",
        filter_fallback_value="not_processed",
    )

    # Mock LLM provider
    provider = MockProvider()

    # Process the data
    result = await stage.process(data, provider)

    # Assertions
    assert len(result) == 4, "All rows should be present in the result"

    # Check that filtered rows have fallback value
    filtered_rows = result[~result["should_process"]]
    assert all(
        filtered_rows["result"] == "not_processed"
    ), "Filtered rows should have fallback value"

    # Check that processed rows have the processed value
    processed_rows = result[result["should_process"]]
    assert all(
        processed_rows["result"] == "processed"
    ), "Processed rows should have been processed"


@pytest.mark.asyncio
async def test_stage_without_filter():
    """Test stage operation without a filter."""
    # Create test data without a filter column
    data = pd.DataFrame({"id": [1, 2, 3, 4], "input_col": ["a", "b", "c", "d"]})

    # Create a stage without filter
    stage = SimpleStage(
        input_columns=["input_col"],
        output_columns=["result"],
        filter_colname=None,
        filter_fallback_value=None,
    )

    # Mock LLM provider
    provider = MockProvider()

    # Process the data
    result = await stage.process(data, provider)

    # Assertions
    assert len(result) == 4, "All rows should be present in the result"
    assert all(result["result"] == "processed"), "All rows should have been processed"
