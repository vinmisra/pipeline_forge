import inspect
import pandas as pd
import hashlib
import json
from typing import Hashable, List, Any, Callable, Optional, Dict

from pipeline_forge.cache import Cache
from pipeline_forge.llm.provider import LLMProvider
from pipeline_forge.stage import Stage


class FunctionalStage(Stage):
    """Stage for applying custom functions to data."""

    def __init__(
        self,
        input_columns: list[str],
        output_columns: list[str],
        function: Callable[[Any], Any],
        filter_colname: str | None = None,
    ):
        self.function = function
        super().__init__(input_columns, output_columns, filter_colname)

    def _get_cache_key(self, row: pd.Series) -> Hashable:
        """Generate a unique key for the cache based on the details of this stage and the input row"""
        function_code = inspect.getsource(self.function)

        config = {
            "input_columns": self.input_columns,
            "values": [row[col] for col in self.input_columns],
            "function_code": function_code,
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    async def process(
        self,
        data: pd.DataFrame,
        llm_provider: LLMProvider,
        cache: Cache | None = None,
        **kwargs
    ) -> pd.DataFrame:
        """Apply the lambda function to each row."""
        result = data.copy()

        # Initialize output columns with None
        for col in self.output_columns:
            if col not in result.columns:
                result[col] = None

        # Process each row
        for idx, row in result.iterrows():
            if not self._should_process_row(row):
                continue

            if cache:
                cache_key = self._get_cache_key(row)
                cached_value = cache.get(cache_key)
                if cached_value is not None:
                    assert len(cached_value) == len(self.output_columns)
                    for cache_value, col in zip(cached_value, self.output_columns):
                        result.at[idx, col] = cache_value
                    continue

            # Extract input values and apply function
            input_values = [row[col] for col in self.input_columns]
            output = self.function(*input_values)

            # Convert to list if not already
            if not isinstance(output, (list, tuple)):
                output = [output]

            # Ensure outputs and expected columns match
            if len(output) != len(self.output_columns):
                output = list(output) + [None] * (
                    len(self.output_columns) - len(output)
                )

            # Store in cache
            if cache:
                cache_key = self._get_cache_key(row)
                cache.set(cache_key, output)

            # Update dataframe
            for col, value in zip(self.output_columns, output):
                result.at[idx, col] = value

        return result


class FilterStage(FunctionalStage):
    """Special case of FunctionalStage that produces boolean filters."""

    def __init__(
        self,
        input_columns: list[str],
        function: Callable[[Any], bool],
        output_columns: list[str],
        filter_colname: str | None = None,
    ):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            function=function,
            filter_colname=filter_colname,
        )
