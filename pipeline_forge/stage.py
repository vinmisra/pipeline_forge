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
    ):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.filter_colname = filter_colname

    @abstractmethod
    async def process(
        self, data: pd.DataFrame, llm_provider: LLMProvider, cache: Cache | None = None
    ) -> pd.DataFrame:
        """Process the input data and return a DataFrame with new columns."""
        pass

    def get_dependencies(self) -> Set[str]:
        """Return the columns this stage depends on."""
        return set(self.input_columns)

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
