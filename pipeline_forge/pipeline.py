import pandas as pd
from typing import List, Any, Dict, Set, Optional, Tuple, TypeVar, DefaultDict
from collections import defaultdict
import networkx as nx
import asyncio
import hashlib
import uuid

from pipeline_forge.cache import Cache
from pipeline_forge.llm.provider import LLMProvider
from pipeline_forge.stage import Stage

T = TypeVar("T")


class Pipeline:
    """Orchestrates the execution of multiple stages in sequence."""

    def __init__(self, stages: List["Stage"]):
        from pipeline_forge.stage import Stage

        self.stages = stages
        self._stage_by_output = self._index_stages_by_output()

    def _index_stages_by_output(self) -> Dict[str, "Stage"]:
        """Create a mapping from output column to the stage that produces it."""
        index = {}
        for stage in self.stages:
            for col in stage.get_outputs():
                index[col] = stage
        return index

    async def run(
        self,
        data: pd.DataFrame,
        llm_provider: Optional["LLMProvider"] = None,
        cache: Optional["Cache"] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Run the entire pipeline on the provided data."""
        result = data.copy()

        for stage in self.stages:
            result = await self.run_stage(
                stage, result, llm_provider=llm_provider, cache=cache, **kwargs
            )

        return result

    async def run_stage(
        self,
        stage: Stage,
        data: pd.DataFrame,
        llm_provider: LLMProvider | None = None,
        cache: Cache | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Run a specific stage and all its dependencies on the provided data."""
        result = data.copy()

        # Check if all dependencies are available
        missing_deps = set(stage.get_dependencies()) - set(result.columns)

        output_to_stage = self._index_stages_by_output()
        # If we have missing dependencies, compute them first
        if missing_deps:
            # Find which stages produce our missing dependencies
            for column in missing_deps:
                if column in output_to_stage:
                    # Recursively run the dependency stage
                    dep_stage = output_to_stage[column]
                    result = await self.run_stage(
                        dep_stage,
                        result,
                        llm_provider=llm_provider,
                        cache=cache,
                        **kwargs,
                    )

        # Verify dependencies are now available
        still_missing = set(stage.input_columns) - set(result.columns)
        if still_missing:
            raise ValueError(
                f"Unable to compute required dependencies: {still_missing}"
            )

        # Finally, process the stage itself
        return await stage.process(
            result, llm_provider=llm_provider, cache=cache, **kwargs
        )

    def clear_caches(self):
        """Clear caches for all stages."""
        for stage in self.stages:
            stage.clear_cache()
