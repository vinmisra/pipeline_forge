from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Hashable


class Cache(ABC):
    """Abstract base class for different cache implementations."""

    @abstractmethod
    def get(self, key: Hashable) -> Optional[Any]:
        """Retrieve a value from the cache."""
        pass

    @abstractmethod
    def set(self, key: Hashable, value: Any) -> None:
        """Store a value in the cache."""
        pass

    @abstractmethod
    def contains(self, key: Hashable) -> bool:
        """Check if a key exists in the cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        pass


class InMemoryCache(Cache):
    """Simple in-memory cache implementation."""

    def __init__(self):
        self._cache = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: Hashable) -> Optional[Any]:
        """Retrieve a value from the cache."""
        value = self._cache.get(key)
        if value is not None:
            self._hits += 1
        else:
            self._misses += 1
        return value

    def set(self, key: Hashable, value: Any) -> None:
        """Store a value in the cache."""
        self._cache[key] = value

    def contains(self, key: Hashable) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache = {}

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}
