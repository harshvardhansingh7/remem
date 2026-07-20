from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from enum import Enum
from threading import RLock
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from remem.distributed.backend import DistributedBackend
from remem.distributed.config import DistributedConfig
from remem.metrics.collector import MetricsCollector
from remem.metrics.events import MetricEvent
from remem.models.execution_context import ExecutionContext
from remem.models.execution_record import ExecutionRecord
from remem.storage.storage import StorageInterface


class LockStatus(str, Enum):
    ACQUIRED = "acquired"
    CONTENDED = "contended"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class DistributedStorage(StorageInterface):
    """Write-through local/remote adapter with reconnect replay."""

    is_distributed = True

    def __init__(
        self,
        local: StorageInterface,
        remote: DistributedBackend,
        config: DistributedConfig,
        metrics: MetricsCollector,
    ) -> None:
        self.local = local
        self.remote = remote
        self.config = config
        self.metrics = metrics
        self.last_backend_error: str | None = None
        self._lock = RLock()
        self._pending: list[tuple[str, Any]] = [
            ("put", record) for record in self.local.all()
        ]
        self._sources: dict[UUID, str] = {}

    def _failure(self, exc: Exception) -> None:
        self.last_backend_error = f"{type(exc).__name__}: {exc}"
        self.metrics.record(MetricEvent.DISTRIBUTED_BACKEND_FAILURE)
        if self.config.fallback_to_local:
            self.metrics.record(MetricEvent.FALLBACK_TO_LOCAL)
        else:
            raise exc

    def _replay_pending(self) -> bool:
        if not self._pending:
            return True
        remaining = list(self._pending)
        try:
            for operation, value in remaining:
                if operation == "put":
                    self.remote.put(value)
                elif operation == "delete":
                    self.remote.delete(value)
                elif operation == "increment_hit":
                    self.remote.increment_hit(value)
            del self._pending[: len(remaining)]
            self.metrics.record(MetricEvent.DISTRIBUTED_SYNC)
            return True
        except Exception as exc:
            self.metrics.record(MetricEvent.DISTRIBUTED_SYNC_FAILURE)
            self._failure(exc)
            return False

    def _replace_local(self, records: list[ExecutionRecord]) -> None:
        if not self.config.local_cache:
            return
        self.local.flush()
        for record in records:
            self.local.put(record)

    def synchronize(self) -> list[ExecutionRecord]:
        with self._lock:
            if not self._replay_pending():
                return self.local.all()
            try:
                invalid_before = (
                    self.remote.invalid_record_count
                    if hasattr(self.remote, "invalid_record_count")
                    else 0
                )
                records = self.remote.all()
                invalid_after = (
                    self.remote.invalid_record_count
                    if hasattr(self.remote, "invalid_record_count")
                    else invalid_before
                )
                for _ in range(invalid_after - invalid_before):
                    self.metrics.record(MetricEvent.DISTRIBUTED_INVALID_RECORD)
                self.last_backend_error = None
                self._sources = {record.id: "remote" for record in records}
                if self.config.sync_on_read:
                    self._replace_local(records)
                self.metrics.record(MetricEvent.DISTRIBUTED_SYNC)
                return records
            except Exception as exc:
                self.metrics.record(MetricEvent.DISTRIBUTED_SYNC_FAILURE)
                self._failure(exc)
                self._sources = {record.id: "local" for record in self.local.all()}
                return self.local.all()

    def put(self, record: ExecutionRecord) -> None:
        with self._lock:
            try:
                self.remote.put(record)
                self.metrics.record(MetricEvent.DISTRIBUTED_SYNC)
                self.last_backend_error = None
            except Exception as exc:
                self._pending.append(("put", record))
                self.metrics.record(MetricEvent.DISTRIBUTED_SYNC_FAILURE)
                self._failure(exc)
            self.local.put(record)
            self._sources[record.id] = "local"

    def get(self, entry_id: UUID) -> ExecutionRecord | None:
        with self._lock:
            try:
                record = self.remote.get(entry_id)
                self.last_backend_error = None
                if record is not None:
                    if self.config.local_cache:
                        self.local.put(record)
                    self._sources[record.id] = "remote"
                    return record
            except Exception as exc:
                self._failure(exc)
            record = self.local.get(entry_id)
            if record is not None:
                self._sources[record.id] = "local"
            return record

    def get_many(self, entry_ids: Sequence[UUID]) -> list[ExecutionRecord]:
        return [record for entry_id in entry_ids if (record := self.get(entry_id))]

    def delete(self, entry_id: UUID) -> bool:
        with self._lock:
            remote_deleted = False
            try:
                remote_deleted = self.remote.delete(entry_id)
                self.last_backend_error = None
            except Exception as exc:
                self._pending.append(("delete", entry_id))
                self._failure(exc)
            local_deleted = self.local.delete(entry_id)
            self._sources.pop(entry_id, None)
            return remote_deleted or local_deleted

    def update(self, record: ExecutionRecord) -> None:
        if self.get(record.id) is not None:
            self.put(record)

    def all(self) -> list[ExecutionRecord]:
        return self.synchronize()

    def flush(self) -> None:
        with self._lock:
            try:
                self.remote.flush()
                self._pending.clear()
                self.last_backend_error = None
            except Exception as exc:
                for record in self.local.all():
                    self._pending.append(("delete", record.id))
                self._failure(exc)
            self.local.flush()
            self._sources.clear()

    def load(self) -> None:
        self.synchronize()

    def increment_hit(self, entry_id: UUID) -> None:
        with self._lock:
            try:
                self.remote.increment_hit(entry_id)
            except Exception as exc:
                self._pending.append(("increment_hit", entry_id))
                self._failure(exc)
            self.local.increment_hit(entry_id)

    def record_source(self, entry_id: UUID) -> str:
        return self._sources.get(entry_id, "local")

    def computation_key(
        self, query_embedding: list[float], context: ExecutionContext
    ) -> str:
        payload = {
            "embedding": [round(float(value), 12) for value in query_embedding],
            "namespace": context.namespace,
            "kb_version": context.kb_version,
            "prompt_version": context.prompt_version,
            "model": context.model,
            "metadata": context.metadata,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def deterministic_record_id(
        self, query_embedding: list[float], context: ExecutionContext
    ) -> UUID:
        return uuid5(
            NAMESPACE_URL, f"remem:{self.computation_key(query_embedding, context)}"
        )

    def acquire_computation_lock(
        self, query_embedding: list[float], context: ExecutionContext
    ) -> tuple[LockStatus, str | None]:
        if not self.config.duplicate_work_prevention:
            return LockStatus.DISABLED, None
        token = uuid4().hex
        resource = self.computation_key(query_embedding, context)
        try:
            acquired = self.remote.acquire_lock(
                resource, token, int(self.config.lock_ttl_seconds * 1000)
            )
        except Exception as exc:
            self._failure(exc)
            return LockStatus.UNAVAILABLE, None
        if acquired:
            self.metrics.record(MetricEvent.DISTRIBUTED_LOCK_ACQUIRED)
            return LockStatus.ACQUIRED, token
        self.metrics.record(MetricEvent.DISTRIBUTED_LOCK_CONTENTION)
        return LockStatus.CONTENDED, None

    def release_computation_lock(
        self,
        query_embedding: list[float],
        context: ExecutionContext,
        token: str,
    ) -> None:
        try:
            self.remote.release_lock(
                self.computation_key(query_embedding, context), token
            )
        except Exception as exc:
            self._failure(exc)

    def ping(self) -> bool:
        try:
            healthy = self.remote.ping()
            self.last_backend_error = None
            return healthy
        except Exception as exc:
            self._failure(exc)
            return False

    @property
    def pending_operation_count(self) -> int:
        return len(self._pending)
