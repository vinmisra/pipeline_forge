from abc import ABC, abstractmethod
import pandas as pd
import hashlib
import json
from typing import List, Any, Dict, Optional, Callable, Set, Tuple, Hashable

from pipeline_forge.cache import Cache
from pipeline_forge.llm.provider import LLMProvider


class Stage(ABC):
    """Base class for all pipeline stages."""

    def __init__(
        self,
        input_columns: list[str],
        output_columns: list[str],
        filter_colname: str | None = None,
        filter_fallback_value: Any = None,
    ):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.filter_colname = filter_colname
        self.filter_fallback_value = filter_fallback_value

    async def process(
        self, data: pd.DataFrame, llm_provider: LLMProvider, cache: Cache | None = None
    ) -> pd.DataFrame:
        """Process the input data and return a DataFrame with new columns."""
        assert (
            self.filter_colname is None or self.filter_colname in data.columns
        ), f"Filter column {self.filter_colname} not found in dataset {data.head()}"

        if self.filter_colname is None:
            return await self._process_post_filter(data, llm_provider, cache)
        else:
            # initialize output columns with filter fallback value
            result = data.copy()
            for col in self.output_columns:
                result[col] = self.filter_fallback_value

            # filter data and process only those rows
            filtered_data = data[data[self.filter_colname]].copy()
            if not filtered_data.empty:
                processed = await self._process_post_filter(
                    filtered_data, llm_provider, cache
                )

                # Update only the filtered rows in the result
                for idx in processed.index:
                    for col in self.output_columns:
                        result.loc[idx, col] = processed.loc[idx, col]

            return result

    @abstractmethod
    async def _process_post_filter(
        self, data: pd.DataFrame, llm_provider: LLMProvider, cache: Cache | None = None
    ) -> pd.DataFrame:
        """Process the data after filtering."""
        pass

    def get_dependencies(self) -> Set[str]:
        """Return the columns this stage depends on."""
        if self.filter_colname is None:
            return set(self.input_columns)
        else:
            return set(self.input_columns + [self.filter_colname])

    def get_outputs(self) -> Set[str]:
        """Return the columns this stage produces."""
        return set(self.output_columns)

    def _should_process_row(self, row: pd.Series) -> bool:
        """Determine if this row should be processed based on filter column."""
        if self.filter_colname is None:
            return True
        if isinstance(row[self.filter_colname], bool):
            return row[self.filter_colname]
        else:
            raise ValueError(
                f"Filter column {self.filter_colname} must be a boolean, got {type(row[self.filter_colname])}"
            )
