from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import List, Optional
from uuid import UUID

from remem.models.execution_record import ExecutionRecord


class StorageInterface(ABC):
    """Abstract interface for all storage backends."""

    @abstractmethod
    def put(self, record: ExecutionRecord) -> None:
        pass

    @abstractmethod
    def get(self, entry_id: UUID) -> Optional[ExecutionRecord]:
        pass

    def get_many(self, entry_ids: Sequence[UUID]) -> List[ExecutionRecord]:
        """Return available records in requested order using direct ID lookups.

        Existing custom backends inherit this compatibility implementation,
        which delegates to their mandatory ``get`` method without calling
        ``all``. Backends can override it with a native batch query.
        """

        records = []
        for entry_id in entry_ids:
            record = self.get(entry_id)
            if record is not None:
                records.append(record)
        return records

    @abstractmethod
    def delete(self, entry_id: UUID) -> bool:
        pass

    @abstractmethod
    def update(self, record: ExecutionRecord) -> None:
        pass

    @abstractmethod
    def all(self) -> List[ExecutionRecord]:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def increment_hit(self, entry_id: UUID) -> None:
        """Increments the reuse hit counter for a stored record."""
        pass
