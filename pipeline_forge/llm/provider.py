from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response from the LLM based on the provided messages.
        All configuration should be set during provider initialization.
        """
        pass

    def get_provider_id(self) -> str:
        """Return a unique identifier for this provider configuration."""
        # Default implementation - subclasses should override with config details
        return self.__class__.__name__


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    def __init__(
        self,
        default_response: str = "Mock response",
        map_responses: Dict[str, str] | None = None,
    ):
        self.default_response = default_response
        self.map_responses = map_responses

    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """Mock implementation of generate."""
        if self.map_responses and len(messages) > 0:
            key = str(messages[-1]["content"])
            if key in self.map_responses:
                return self.map_responses[key]
        return self.default_response

    def get_provider_id(self) -> str:
        """Return a unique identifier for this provider configuration."""
        return f"MockProvider({self.default_response}, {self.map_responses})"
