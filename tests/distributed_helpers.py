from __future__ import annotations

import time
from collections.abc import Sequence
from threading import RLock
from uuid import UUID

from remem.distributed.backend import DistributedBackend
from remem.models.execution_record import ExecutionRecord
from remem.storage.serializer import Serializer


def _copy(record: ExecutionRecord) -> ExecutionRecord:
    return Serializer.deserialize(Serializer.serialize(record))


class SharedDistributedBackend(DistributedBackend):
    """Concrete thread-safe backend used for deterministic distributed tests."""

    def __init__(self) -> None:
        self.available = True
        self._records: dict[UUID, ExecutionRecord] = {}
        self._hits: dict[UUID, int] = {}
        self._locks: dict[str, tuple[str, float]] = {}
        self._mutex = RLock()

    def _guard(self) -> None:
        if not self.available:
            raise ConnectionError("distributed backend unavailable")

    def put(self, record: ExecutionRecord) -> None:
        self._guard()
        with self._mutex:
            self._records[record.id] = _copy(record)

    def get(self, entry_id: UUID) -> ExecutionRecord | None:
        self._guard()
        with self._mutex:
            record = self._records.get(entry_id)
            if record is None:
                return None
            result = _copy(record)
            result.hit_count += self._hits.get(entry_id, 0)
            return result

    def get_many(self, entry_ids: Sequence[UUID]) -> list[ExecutionRecord]:
        return [record for entry_id in entry_ids if (record := self.get(entry_id))]

    def delete(self, entry_id: UUID) -> bool:
        self._guard()
        with self._mutex:
            self._hits.pop(entry_id, None)
            return self._records.pop(entry_id, None) is not None

    def update(self, record: ExecutionRecord) -> None:
        self._guard()
        with self._mutex:
            if record.id in self._records:
                self._records[record.id] = _copy(record)

    def all(self) -> list[ExecutionRecord]:
        self._guard()
        with self._mutex:
            results = []
            for entry_id, record in sorted(
                self._records.items(), key=lambda item: str(item[0])
            ):
                result = _copy(record)
                result.hit_count += self._hits.get(entry_id, 0)
                results.append(result)
            return results

    def flush(self) -> None:
        self._guard()
        with self._mutex:
            self._records.clear()
            self._hits.clear()

    def load(self) -> None:
        self._guard()

    def increment_hit(self, entry_id: UUID) -> None:
        self._guard()
        with self._mutex:
            self._hits[entry_id] = self._hits.get(entry_id, 0) + 1

    def acquire_lock(self, resource: str, token: str, ttl_ms: int) -> bool:
        self._guard()
        with self._mutex:
            now = time.monotonic()
            existing = self._locks.get(resource)
            if existing is not None and existing[1] > now:
                return False
            self._locks[resource] = (token, now + ttl_ms / 1000)
            return True

    def release_lock(self, resource: str, token: str) -> bool:
        self._guard()
        with self._mutex:
            existing = self._locks.get(resource)
            if existing is None or existing[0] != token:
                return False
            del self._locks[resource]
            return True

    def ping(self) -> bool:
        self._guard()
        return True
