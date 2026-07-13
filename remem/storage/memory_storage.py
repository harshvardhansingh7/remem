from collections.abc import Sequence
from typing import Dict, List, Optional
from uuid import UUID

from remem.models.execution_record import ExecutionRecord
from remem.storage.storage import StorageInterface


class InMemoryStorage(StorageInterface):
    """Volatile, in-process storage backend.

    Holds execution records in a plain dictionary with no disk I/O. Ideal for
    tests, notebooks, and ephemeral workloads where durability is not required.

    It implements the same :class:`StorageInterface` as the durable backends,
    so applications can swap between in-memory and persistent storage without
    changing any other code::

        from remem import Client, InMemoryStorage

        client = Client(storage_backend=InMemoryStorage())
    """

    def __init__(self) -> None:
        self._memory_store: Dict[UUID, ExecutionRecord] = {}

    def put(self, record: ExecutionRecord) -> None:
        self._memory_store[record.id] = record

    def get(self, entry_id: UUID) -> Optional[ExecutionRecord]:
        return self._memory_store.get(entry_id)

    def get_many(self, entry_ids: Sequence[UUID]) -> List[ExecutionRecord]:
        return [
            record
            for entry_id in entry_ids
            if (record := self._memory_store.get(entry_id)) is not None
        ]

    def delete(self, entry_id: UUID) -> bool:
        if entry_id in self._memory_store:
            del self._memory_store[entry_id]
            return True
        return False

    def update(self, record: ExecutionRecord) -> None:
        if record.id in self._memory_store:
            self._memory_store[record.id] = record

    def all(self) -> List[ExecutionRecord]:
        return list(self._memory_store.values())

    def flush(self) -> None:
        self._memory_store.clear()

    def load(self) -> None:
        """No-op. Kept for API symmetry with durable backends."""
        return None

    def save(self) -> None:
        """No-op. Kept for API symmetry with durable backends."""
        return None

    def increment_hit(self, entry_id: UUID) -> None:
        if entry_id in self._memory_store:
            self._memory_store[entry_id].increment_hit()
