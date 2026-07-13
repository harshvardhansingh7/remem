import json
import os
import time
from collections.abc import Sequence
from typing import List, Optional
from uuid import UUID

from remem.models.execution_record import ExecutionRecord
from remem.storage.exceptions import CorruptedSnapshotException, PersistenceException
from remem.storage.serializer import Serializer
from remem.storage.snapshot import StorageSnapshot
from remem.storage.storage import StorageInterface


class JsonStorage(StorageInterface):
    """Durable file-backed persistence layer featuring atomic writes."""

    def __init__(self, filepath: str = "remem_store.json"):
        self.filepath = filepath
        self._memory_store = {}
        self.load()

    def put(self, record: ExecutionRecord) -> None:
        self._memory_store[record.id] = record
        self.save()

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
            self.save()
            return True
        return False

    def update(self, record: ExecutionRecord) -> None:
        if record.id in self._memory_store:
            self._memory_store[record.id] = record
            self.save()

    def all(self) -> List[ExecutionRecord]:
        return list(self._memory_store.values())

    def flush(self) -> None:
        self._memory_store.clear()
        self.save()

    def load(self) -> None:
        """Loads entries from disk, automatically called upon instantiation."""
        if not os.path.exists(self.filepath):
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)

            raw_records = payload.get("records", [])
            records = Serializer.deserialize_many(raw_records)

            self._memory_store = {r.id: r for r in records}
        except json.JSONDecodeError as e:
            raise CorruptedSnapshotException(
                f"Failed to parse corrupt snapshot JSON: {e}"
            ) from e
        except Exception as e:
            raise PersistenceException(
                f"Critical error loading storage snapshot: {e}"
            ) from e

    def save(self) -> None:
        """Flushes in-memory data to disk safely via an atomic rename cycle."""
        temp_filepath = f"{self.filepath}.tmp"
        try:
            serialized_records = Serializer.serialize_many(self.all())
            snapshot = StorageSnapshot.create(serialized_records)

            with open(temp_filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot.__dict__, f, indent=2)

            # Atomic swap rename operation.
            # On Windows os.replace can transiently raise PermissionError when
            # the destination is briefly held open by another process (e.g. an
            # antivirus or indexer), so retry a few times before giving up.
            self._atomic_replace(temp_filepath, self.filepath)

        except Exception as e:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            raise PersistenceException(
                f"Failed to perform atomic storage save: {e}"
            ) from e

    @staticmethod
    def _atomic_replace(
        src: str, dst: str, retries: int = 5, delay: float = 0.05
    ) -> None:
        """Replaces ``dst`` with ``src`` atomically, retrying on Windows lock errors."""
        for attempt in range(retries):
            try:
                os.replace(src, dst)
                return
            except PermissionError:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)

    def increment_hit(self, entry_id: UUID) -> None:
        """Atomically increments hit metrics and persists records to disk."""
        if entry_id in self._memory_store:
            self._memory_store[entry_id].increment_hit()
            self.save()
