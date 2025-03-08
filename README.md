# Pipeline Forge

A Python package for data processing with expensive and precious LLM operations.

Goals:

- iterating on individual stages and pipelines at limited volume to test out / tune expensive data processing operations (e.g. LLM calls).
- "baking" a pipeline that can then be run at scale.
- (aspirational) supporting integrations into arbitrary data sources, caching mechanisms, destinations, data processing stages, and compute infrastructure.

Non-goal:

- A general purpose data processing framework / library.
- constructing pipelines that do not involve LLMs as a core processing component.
- Realtime / tight SLAs.

## Installation

`pip install pipeline_forge`

## Usage

See [tests/test_integration_openai.py](tests/test_integration_openai.py) for an example of how to use the package.

## Testing

To run unit tests, run `pytest tests/unit`.

To run integration tests, run `pytest tests/integration`. You will first need to create a `.env` file in the root directory with your OpenAI API key.
