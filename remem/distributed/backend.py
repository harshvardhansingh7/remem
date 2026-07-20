from __future__ import annotations

from abc import abstractmethod

from remem.storage.storage import StorageInterface


class DistributedBackend(StorageInterface):
    """Storage backend with the primitives needed for cross-node coordination."""

    @abstractmethod
    def acquire_lock(self, resource: str, token: str, ttl_ms: int) -> bool:
        """Acquire an expiring lock if it is currently free."""

    @abstractmethod
    def release_lock(self, resource: str, token: str) -> bool:
        """Release a lock only when ``token`` still owns it."""

    @abstractmethod
    def ping(self) -> bool:
        """Return whether the backend is currently reachable."""
