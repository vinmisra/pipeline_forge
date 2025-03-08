import pandas as pd
from typing import List, Any, Optional
from pipeline_forge.stage import Stage
from pipeline_forge.pipeline import Pipeline
from pipeline_forge.llm.provider import LLMProvider
from pipeline_forge.cache import Cache


class PipelineStage(Stage):
    """Stage that encapsulates a nested pipeline."""

    def __init__(
        self,
        input_columns: List[str],
        pipeline: Pipeline,
        output_columns: List[str],
        filter_colname: Optional[str] = None,
    ):
        super().__init__(input_columns, output_columns, filter_colname)
        self.pipeline = pipeline

    async def process(
        self,
        data: pd.DataFrame,
        llm_provider: LLMProvider | None = None,
        cache: Cache | None = None,
        **kwargs
    ) -> pd.DataFrame:
        """Process through the nested pipeline."""
        result = data.copy()

        # Filter rows to process if needed
        if self.filter_colname is not None:
            rows_to_process = result[result[self.filter_colname]].index
            if len(rows_to_process) == 0:
                # Initialize output columns with None
                for col in self.output_columns:
                    if col not in result.columns:
                        result[col] = None
                return result

            # Process only the filtered subset
            subset = result.loc[rows_to_process].copy()
            processed = await self.pipeline.run(
                subset, llm_provider=llm_provider, cache=cache, **kwargs
            )

            # Merge results back
            for col in self.output_columns:
                if col in processed.columns:
                    result.loc[rows_to_process, col] = processed[col]
        else:
            # Process all rows
            processed = await self.pipeline.run(
                result[self.input_columns],
                llm_provider=llm_provider,
                cache=cache,
                **kwargs,
            )

            # Add new columns to result
            for col in self.output_columns:
                if col in processed.columns:
                    result[col] = processed[col]

        return result
