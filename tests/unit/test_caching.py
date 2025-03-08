import pytest

from pipeline_forge.stages.llm_stage import LLMStage
from pipeline_forge.pipeline import Pipeline
from pipeline_forge.cache import InMemoryCache
from pipeline_forge.llm.provider import MockProvider
import pandas as pd


@pytest.mark.asyncio
async def test_caching():
    """Test that caching works correctly for LLM stages."""
    data = pd.DataFrame(
        {
            "question": ["What is 1+1?", "Who are you?", "What is 1+1?"],
        }
    )

    stage = LLMStage(
        input_columns=["question"],
        conversation_template=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{question}"},
        ],
        output_columns=["answer"],
    )

    pipeline = Pipeline(stages=[stage])

    # Create a cache
    cache = InMemoryCache()

    # First run - should process all rows
    provider = MockProvider(default_response="Mock response")
    result = await pipeline.run(data, llm_provider=provider, cache=cache)

    # Check cache stats --- should hit the cache for the duplicated question
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2

    # Second run - should use cache for all the questions.
    result2 = await pipeline.run(data, llm_provider=provider, cache=cache)

    # Check cache stats again
    stats = cache.get_stats()
    assert (
        stats["hits"] == 4
    )  # All three should hit the cache, in addition to the first question

    # Results should be identical
    assert result.equals(result2)


@pytest.mark.asyncio
async def test_compute_options_affect_cache():
    """Test that changing compute options invalidates cache."""
    data = pd.DataFrame(
        {
            "question": ["What is your name?"],
        }
    )

    stage = LLMStage(
        input_columns=["question"],
        conversation_template=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{question}"},
        ],
        output_columns=["answer"],
    )

    pipeline = Pipeline(stages=[stage])
    cache = InMemoryCache()

    # First run with one default response
    provider1 = MockProvider(default_response="Mock response")
    _ = await pipeline.run(data, llm_provider=provider1, cache=cache)

    # Check cache
    stats1 = cache.get_stats()
    assert stats1["hits"] == 0
    assert stats1["misses"] == 1

    # Second run with different default response
    provider2 = MockProvider(default_response="Different mock response")
    _ = await pipeline.run(data, llm_provider=provider2, cache=cache)

    # Check cache again
    stats2 = cache.get_stats()
    assert stats2["hits"] == 0
    assert stats2["misses"] == 2  # Miss count increased

    # Third run with original response again - should hit cache
    provider3 = MockProvider(default_response="Mock response")
    _ = await pipeline.run(data, llm_provider=provider3, cache=cache)

    # Check cache again
    stats3 = cache.get_stats()
    assert stats3["hits"] == 1  # Hit count increased
    assert stats3["misses"] == 2  # Miss count unchanged
