from setuptools import setup, find_packages

setup(
    name="pipeline_forge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2.3",
        "pandas>=2.2.3",
        "openai>=1.65.4",
        "pyyaml>=6.0.0",
        "pytest>=8.3.5",
        "pytest-asyncio>=0.25.3",
        "networkx>=3.4.2",
        "python-dotenv>=1.0.1",
    ],
    description="A Python package for building and managing data processing pipelines",
)
