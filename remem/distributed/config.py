from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from uuid import uuid4


def _default_node_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"


@dataclass(frozen=True)
class DistributedConfig:
    """Configuration for the optional distributed semantic cache."""

    enabled: bool = True
    backend: str = "redis"
    redis_url: str = field(
        default_factory=lambda: os.getenv("REMEM_REDIS_URL", "redis://localhost:6379/0")
    )
    key_prefix: str = "remem:v1.2"
    node_id: str = field(default_factory=_default_node_id)
    local_cache: bool = True
    sync_on_read: bool = True
    fallback_to_local: bool = True
    socket_timeout_seconds: float = 1.0
    retry_attempts: int = 1
    retry_delay_seconds: float = 0.05
    record_ttl_seconds: float | None = None
    duplicate_work_prevention: bool = True
    lock_ttl_seconds: float = 60.0
    lock_wait_timeout_seconds: float = 10.0
    lock_poll_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.backend != "redis":
            raise ValueError("distributed backend must currently be 'redis'")
        if not self.key_prefix.strip():
            raise ValueError("key_prefix must not be empty")
        if not self.node_id.strip():
            raise ValueError("node_id must not be empty")
        if self.socket_timeout_seconds <= 0:
            raise ValueError("socket_timeout_seconds must be positive")
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        if self.record_ttl_seconds is not None and self.record_ttl_seconds <= 0:
            raise ValueError("record_ttl_seconds must be positive when configured")
        if self.lock_ttl_seconds <= 0:
            raise ValueError("lock_ttl_seconds must be positive")
        if self.lock_wait_timeout_seconds < 0:
            raise ValueError("lock_wait_timeout_seconds must be non-negative")
        if self.lock_poll_interval_seconds <= 0:
            raise ValueError("lock_poll_interval_seconds must be positive")
