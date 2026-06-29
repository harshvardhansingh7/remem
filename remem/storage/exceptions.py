class StorageException(Exception):
    """Base exception for all storage and persistence operations."""
    pass


class PersistenceException(StorageException):
    """Raised when underlying persistence mediums fail or throw IO errors."""
    pass


class CorruptedSnapshotException(StorageException):
    """Raised when data loaded from disk contains invalid or malformed JSON payloads."""
    pass