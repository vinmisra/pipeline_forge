# Pipeline Forge

A Python package for building and managing data processing pipelines.

Goals:

- iterating on individual stages at very limited volume to test out / tune expensive data processing operations (e.g. LLM calls).
- "baking" a portable pipeline that can then be run at scale.
- supporting integrations into arbitrary data sources, destinations, and data processing tools.

Non-goal:

- A general purpose data processing framework / library.
- handling pipelines that do not involve LLMs as a core processing component.

## Installation

`pip install pipeline_forge`

## Usage

See [tests/test_integration_openai.py](tests/test_integration_openai.py) for an example of how to use the package.

## Testing

To run unit tests, run `pytest tests/unit`.

To run integration tests, run `pytest tests/integration`. You will first need to create a `.env` file in the root directory with your OpenAI API key.
