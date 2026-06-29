from abc import ABC, abstractmethod
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