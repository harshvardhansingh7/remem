from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import UUID

from remem.distributed.backend import DistributedBackend
from remem.distributed.config import DistributedConfig
from remem.models.execution_record import ExecutionRecord
from remem.storage.serializer import Serializer

T = TypeVar("T")

_RECORD_SCHEMA_VERSION = 1
_RELEASE_LOCK_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


class RedisStorage(DistributedBackend):
    """Redis-backed shared storage using one namespaced record hash.

    The optional ``redis`` dependency is imported only when this backend is
    constructed, so local Remem installations retain their dependency surface.
    """

    def __init__(
        self,
        config: DistributedConfig | None = None,
        *,
        redis_client: Any | None = None,
    ) -> None:
        self.config = config or DistributedConfig()
        self._records_key = f"{self.config.key_prefix}:records"
        self._hits_key = f"{self.config.key_prefix}:hits"
        self.invalid_record_count = 0
        if redis_client is not None:
            self._client = redis_client
            return
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "Redis distributed mode requires the optional dependency. "
                "Install it with: pip install remem-ai[redis]"
            ) from exc
        self._client = redis.Redis.from_url(
            self.config.redis_url,
            decode_responses=True,
            socket_timeout=self.config.socket_timeout_seconds,
            socket_connect_timeout=self.config.socket_timeout_seconds,
            socket_keepalive=True,
        )

    def _execute(self, operation: Callable[[], T]) -> T:
        for attempt in range(self.config.retry_attempts + 1):
            try:
                return operation()
            except Exception:
                if attempt >= self.config.retry_attempts:
                    raise
                time.sleep(self.config.retry_delay_seconds)
        raise RuntimeError("unreachable retry state")

    def _encode(self, record: ExecutionRecord) -> str:
        return json.dumps(
            {
                "schema_version": _RECORD_SCHEMA_VERSION,
                "source_node": self.config.node_id,
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "record": Serializer.serialize(record),
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    def _decode(self, raw: Any) -> ExecutionRecord | None:
        try:
            payload = json.loads(raw)
            if payload.get("schema_version") != _RECORD_SCHEMA_VERSION:
                raise ValueError("unsupported distributed record schema")
            stored_at = datetime.fromisoformat(payload["stored_at"])
            if stored_at.tzinfo is None:
                stored_at = stored_at.replace(tzinfo=timezone.utc)
            ttl = self.config.record_ttl_seconds
            if ttl is not None:
                age = (datetime.now(timezone.utc) - stored_at).total_seconds()
                if age > ttl:
                    return None
            return Serializer.deserialize(payload["record"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.invalid_record_count += 1
            return None

    def put(self, record: ExecutionRecord) -> None:
        self._execute(
            lambda: self._client.hset(
                self._records_key, str(record.id), self._encode(record)
            )
        )

    def get(self, entry_id: UUID) -> ExecutionRecord | None:
        raw = self._execute(lambda: self._client.hget(self._records_key, str(entry_id)))
        if raw is None:
            return None
        record = self._decode(raw)
        if record is None:
            self.delete(entry_id)
            return None
        hits = self._execute(lambda: self._client.hget(self._hits_key, str(entry_id)))
        record.hit_count += int(hits or 0)
        return record

    def get_many(self, entry_ids: Sequence[UUID]) -> list[ExecutionRecord]:
        return [record for entry_id in entry_ids if (record := self.get(entry_id))]

    def delete(self, entry_id: UUID) -> bool:
        def delete_record() -> list[Any]:
            pipe = self._client.pipeline(transaction=True)
            pipe.hdel(self._records_key, str(entry_id))
            pipe.hdel(self._hits_key, str(entry_id))
            return pipe.execute()

        deleted, _ = self._execute(delete_record)
        return bool(deleted)

    def update(self, record: ExecutionRecord) -> None:
        exists = self._execute(
            lambda: self._client.hexists(self._records_key, str(record.id))
        )
        if exists:
            self.put(record)

    def all(self) -> list[ExecutionRecord]:
        raw_records = self._execute(lambda: self._client.hgetall(self._records_key))
        records = []
        stale_ids = []
        for raw_id, raw_record in sorted(raw_records.items()):
            record = self._decode(raw_record)
            if record is None:
                stale_ids.append(raw_id)
            else:
                records.append(record)
        if stale_ids:

            def delete_stale() -> list[Any]:
                pipe = self._client.pipeline(transaction=True)
                pipe.hdel(self._records_key, *stale_ids)
                pipe.hdel(self._hits_key, *stale_ids)
                return pipe.execute()

            self._execute(delete_stale)
        if records:
            hit_values = self._execute(
                lambda: self._client.hmget(
                    self._hits_key, [str(record.id) for record in records]
                )
            )
            for record, hits in zip(records, hit_values):
                record.hit_count += int(hits or 0)
        return records

    def flush(self) -> None:
        self._execute(lambda: self._client.delete(self._records_key, self._hits_key))

    def load(self) -> None:
        self.ping()

    def increment_hit(self, entry_id: UUID) -> None:
        self._execute(lambda: self._client.hincrby(self._hits_key, str(entry_id), 1))

    def acquire_lock(self, resource: str, token: str, ttl_ms: int) -> bool:
        lock_key = f"{self.config.key_prefix}:lock:{resource}"
        acquired = self._execute(
            lambda: self._client.set(lock_key, token, nx=True, px=ttl_ms)
        )
        return bool(acquired)

    def release_lock(self, resource: str, token: str) -> bool:
        lock_key = f"{self.config.key_prefix}:lock:{resource}"
        released = self._execute(
            lambda: self._client.eval(_RELEASE_LOCK_SCRIPT, 1, lock_key, token)
        )
        return bool(released)

    def ping(self) -> bool:
        return bool(self._execute(self._client.ping))
