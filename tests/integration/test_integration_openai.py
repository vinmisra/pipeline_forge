import pytest
from pipeline_forge.llm.openai_provider import OpenAIProvider
from pipeline_forge.pipeline import Pipeline
from pipeline_forge.stages.llm_stage import LLMStage
from pipeline_forge.stages.functional_stage import FilterStage
from pipeline_forge.cache import InMemoryCache
from openai import AsyncOpenAI
from pandas import DataFrame
import numpy as np
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()


@pytest.mark.asyncio
async def test_pipeline_integration():
    data = DataFrame(
        {
            "name": ["John", "Jane", "Jim", "Jill"],
            "age": [28, 34, 29, 42],
            "city": ["New York metro area", "london", "sf bay area", "sugarland tx"],
            "submitted_description": [
                "I make software for IBM",
                "studying particle physics @ UCSB",
                "pharma sales rep",
                "in between jobs right now",
            ],
        }
    )

    pipeline = Pipeline(
        stages=[
            LLMStage(
                input_columns=["submitted_description"],
                conversation_template=[
                    {
                        "role": "system",
                        "content": """Your job is to extract the job category from a user's description of their job.
Valid categories are: "SWE", "scientist", "salesperson", "student", "doctor", "lawyer", "educator", "other".
Your output should end with a single word, the category.

Additional constraint: generate a random string of 10 characters at the start of your response.

# Examples
User: i write and illustrate kids books
Assistant: 23jflsiw3k This does not fit any of the categories. other

User: software engineering manager
Assistant: SWE""",
                    },
                    {"role": "user", "content": "{submitted_description}"},
                ],
                output_columns=["job_category"],
            ),
            FilterStage(
                input_columns=["job_category"],
                function=lambda job_category: job_category.lower().endswith("swe"),
                output_columns=["is_swe"],
            ),
            LLMStage(
                input_columns=[
                    "name",
                    "age",
                    "city",
                    "submitted_description",
                    "is_swe",
                ],
                filter_colname="is_swe",
                conversation_template=[
                    {
                        "role": "system",
                        "content": """Your job is to estimate the potential annual income of a software engineer in the year 1985, given their age, location, and job description.
Your output should be a single number, the annual income in USD.

# Examples
User: Name - Alice. Age - 25. Location - San Francisco. Job Description - Systems Engineer at Apple.
Assistant: 30000""",
                    },
                    {
                        "role": "user",
                        "content": "Name - {name}. Age - {age}. Location - {city}. Job Description - {submitted_description}.",
                    },
                ],
                output_columns=["estimated_income"],
            ),
            LLMStage(
                input_columns=["city"],
                conversation_template=[
                    {
                        "role": "system",
                        "content": """Your job is to categorize the location of a user in our system, into one of three categories: US Hub, International Hub, Other
Only respond with one of those three values --- do not include any explanation or any other text in your response.

#Examples
User: upper east side, nyc
Assistant: US Hub
                     
User: near asheville
Assistant: Other""",
                    },
                    {"role": "user", "content": "{city}"},
                ],
                output_columns=["location_category"],
            ),
        ]
    )

    provider = OpenAIProvider(model="gpt-4o-mini", temperature=0.0, seed=1)

    # Create a cache
    cache = InMemoryCache()

    # Run the first two stages on a couple of rows
    test_data = await pipeline.run_stage(
        pipeline.stages[1],
        data.iloc[[0, 1]],
        llm_provider=provider,
        cache=cache,
    )

    assert test_data.shape == (2, 6)
    assert test_data.columns.tolist() == [
        "name",
        "age",
        "city",
        "submitted_description",
        "job_category",
        "is_swe",
    ]
    assert test_data["is_swe"].tolist() == [True, False]

    # Run the third stage for a different set of rows
    test_data_B = await pipeline.run_stage(
        pipeline.stages[2],
        data.iloc[[1, 2]],
        llm_provider=provider,
        cache=cache,
    )
    assert test_data_B.shape == (2, 7)
    assert test_data_B.columns.tolist() == [
        "name",
        "age",
        "city",
        "submitted_description",
        "job_category",
        "is_swe",
        "estimated_income",
    ]
    assert test_data_B["job_category"].iloc[0] == test_data["job_category"].iloc[1]
    assert all(test_data_B["is_swe"] == False)
    assert test_data_B["estimated_income"].iloc[0] is None
    assert test_data_B["estimated_income"].iloc[1] is None

    original_result = await pipeline.run(data, llm_provider=provider, cache=cache)

    # Create a different provider with extreme randomness settings
    different_provider = OpenAIProvider(
        model="gpt-4o-mini",
        temperature=2.0,  # Maximum randomness
        top_p=0.5,  # More restrictive token selection
        frequency_penalty=2.0,  # Strongly discourage repetition
        presence_penalty=2.0,  # Strongly discourage repeating topics
        seed=42,  # Fixed but different sampling path
    )

    # Run the pipeline on the entire dataset with different cache and options
    result = await pipeline.run(data, llm_provider=different_provider, cache=cache)

    # The results should be different due to different temperature,
    assert not result.equals(original_result)
    # but structure should be the same
    assert result.shape == (4, 8)
    assert "location_category" in result.columns

    # Verify cache was used - run again with same cache and options
    cache_stats_before = cache.get_stats()
    result_cached = await pipeline.run(
        data, llm_provider=different_provider, cache=cache
    )
    cache_stats_after = cache.get_stats()

    assert cache_stats_after["hits"] > cache_stats_before["hits"]
    assert result_cached.equals(result)
