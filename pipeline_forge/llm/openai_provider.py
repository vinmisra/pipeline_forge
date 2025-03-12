import hashlib
import json
from typing import List, Dict, Any, Optional
from .provider import LLMProvider
from openai import AsyncOpenAI


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI provider with all configuration parameters.

        Args:
            api_key: OpenAI API key (defaults to environment variable)
            organization: OpenAI organization ID (defaults to environment variable)
            **kwargs: All parameters for the OpenAI API (model, temperature, etc.)
        """
        # Client initialization params
        client_kwargs = {
            "api_key": api_key,
            "organization": organization,
            "base_url": base_url,
        }
        # Filter out None values
        client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}

        # Initialize the client
        self.client = AsyncOpenAI(**client_kwargs)

        # Store model parameters provided by the user
        self.params = kwargs

        # Make sure we have a model specified
        if "model" not in self.params:
            self.params["model"] = "gpt-4o-mini"

    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using the provider's configuration.
        All configuration is set during initialization.
        """
        response = await self.client.chat.completions.create(
            messages=messages, **self.params
        )

        return response.choices[0].message.content

    def get_provider_id(self) -> str:
        """Return a unique identifier that includes complete configuration."""
        # Create a copy without API credentials
        config_for_id = {
            "type": self.__class__.__name__,
            "params": dict(
                sorted(
                    (k, v)
                    for k, v in self.params.items()
                    if k not in ["api_key", "organization"]
                )
            ),
        }

        config_str = json.dumps(config_for_id, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
