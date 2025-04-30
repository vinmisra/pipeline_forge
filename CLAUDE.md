# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands
- Run all tests: `PYTHONPATH=/Users/vinithmisra/sw/pipeline_forge pytest`
- Run unit tests: `PYTHONPATH=/Users/vinithmisra/sw/pipeline_forge pytest tests/unit`
- Run integration tests: `PYTHONPATH=/Users/vinithmisra/sw/pipeline_forge pytest tests/integration`
- Run a specific test: `PYTHONPATH=/Users/vinithmisra/sw/pipeline_forge pytest tests/path/to/test_file.py::test_function_name -v`
- Install package in development mode: `pip install -e .`

## Code Style Guidelines
- **Imports**: Standard Python imports organized with `from ... import` statements. Core dependencies: pandas, numpy, asyncio, typing
- **Type Annotations**: Use Python type hints consistently. Union types use pipe notation (`X | None`)
- **Naming**: CamelCase for classes, snake_case for functions/variables
- **Error Handling**: Use assertions for invariants, explicit ValueError for user errors
- **Formatting**: 4-space indentation, 88-character line length
- **Documentation**: Classes/methods documented with docstrings using triple quotes
- **Async**: Most processing functions are async and should use `await` appropriately
- **Testing**: Use pytest with pytest-asyncio for testing async functions
- **Framework**: Based on pandas DataFrames for data processing