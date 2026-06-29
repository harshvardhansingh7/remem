from dataclasses import dataclass
from typing import List, Dict, Any
from time import time


@dataclass(frozen=True)
class StorageSnapshot:
    """Read-only snapshot representation of serialized execution state."""
    timestamp: float
    records: List[Dict[str, Any]]

    @staticmethod
    def create(records_data: List[Dict[str, Any]]) -> 'StorageSnapshot':
        return StorageSnapshot(timestamp=time(), records=records_data)