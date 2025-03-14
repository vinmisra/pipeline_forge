import json
import pandas as pd
from typing import List, Dict, Any, Optional, Set, Hashable
import asyncio
from pipeline_forge.cache import Cache
from pipeline_forge.stage import Stage
from pipeline_forge.llm.provider import LLMProvider


class LLMStage(Stage):
    """Stage for processing data through an LLM."""

    def __init__(
        self,
        input_columns: list[str],
        conversation_template: list[dict[str, str]],
        output_columns: list[str],
        filter_colname: str | None = None,
        filter_fallback_value: Any = None,
    ):
        assert len(output_columns) == 1, "LLMStage must have exactly one output column"
        self.conversation_template = conversation_template
        super().__init__(
            input_columns, output_columns, filter_colname, filter_fallback_value
        )

    async def _process_post_filter(
        self,
        data: pd.DataFrame,
        llm_provider: LLMProvider,
        cache: Cache | None = None,
    ) -> pd.DataFrame:
        """Process the data using the provided LLM provider."""
        assert llm_provider is not None, "An LLM provider must be provided for LLMStage"

        result = data.copy()

        # Initialize output columns with None
        for col in self.output_columns:
            if col not in result.columns:
                result[col] = None

        # Process each row concurrently
        tasks = [
            self._process_row(row, llm_provider, cache) for _, row in result.iterrows()
        ]
        outputs = await asyncio.gather(*tasks)

        # Update dataframe with results
        for idx, output in zip(result.index, outputs):
            result.at[idx, self.output_columns[0]] = output[0]

        return result

    def _get_llm_cache_key(self, row: pd.Series, llm_provider: LLMProvider) -> Hashable:
        """Generate a cache key for LLM processing based on stage ID, inputs, and provider."""
        input_values = tuple(row[col] for col in self.input_columns)
        return (
            json.dumps(self.conversation_template, sort_keys=True),
            json.dumps(input_values),
            llm_provider.get_provider_id(),
        )

    async def _process_row(
        self, row: pd.Series, llm_provider: LLMProvider, cache: Cache | None = None
    ) -> list[str]:
        """Process a single row through the LLM."""
        # Check cache first if available
        if cache:
            cache_key = self._get_llm_cache_key(row, llm_provider)
            cached_value = cache.get(cache_key)

            if cached_value is not None:
                return cached_value

        messages = self._format_conversation(row)

        # No options needed - provider has all configuration
        response = await llm_provider.generate(messages)

        # Store result in cache if available
        if cache:
            cache_key = self._get_llm_cache_key(row, llm_provider)
            output = [response] * len(self.output_columns)
            cache.set(cache_key, output)
            return output

        # For simplicity, use the same content for all output columns
        return [response] * len(self.output_columns)

    def _format_conversation(self, row: pd.Series) -> List[Dict[str, str]]:
        """Format the conversation template with row values."""
        messages = []

        for message in self.conversation_template:
            content = message["content"]

            # Format content with row values
            for col in self.input_columns:
                if f"{{{col}}}" in content:
                    content = content.replace(f"{{{col}}}", str(row[col]))

            messages.append({"role": message["role"], "content": content})

        return messages
