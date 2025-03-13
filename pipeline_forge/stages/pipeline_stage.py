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

    async def _process_post_filter(
        self,
        data: pd.DataFrame,
        llm_provider: LLMProvider | None = None,
        cache: Cache | None = None,
        **kwargs
    ) -> pd.DataFrame:
        """Process through the nested pipeline."""
        data = await self.pipeline.run(
            data, llm_provider=llm_provider, cache=cache, **kwargs
        )
        return data[self.input_columns + self.output_columns]
