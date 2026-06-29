from uuid import UUID
from typing import Any, Dict, List
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_context import ExecutionContext


class Serializer:
    """Centralizes serialization boundaries for ExecutionRecords."""

    @staticmethod
    def serialize(record: ExecutionRecord) -> Dict[str, Any]:
        """Converts an ExecutionRecord instance into a JSON-serializable dictionary."""
        return {
            "id": str(record.id),
            "embedding": record.embedding,
            "references": record.references,
            "response": record.response,
            "hit_count": record.hit_count,
            "context": {
                "namespace": record.context.namespace,
                "kb_version": record.context.kb_version,
                "prompt_version": record.context.prompt_version,
            } if record.context else None
        }

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> ExecutionRecord:
        """Reconstructs an ExecutionRecord instance from a raw dictionary."""
        ctx_data = data.get("context") or {}
        context = ExecutionContext(
            namespace=ctx_data.get("namespace", ""),
            kb_version=ctx_data.get("kb_version", ""),
            prompt_version=ctx_data.get("prompt_version", ""),
        )

        return ExecutionRecord(
            id=UUID(data["id"]),
            embedding=data["embedding"],
            references=data["references"],
            response=data.get("response"),
            context=context,
        )

    @staticmethod
    def serialize_many(records: List[ExecutionRecord]) -> List[Dict[str, Any]]:
        """Serializes an iterable collection of records."""
        return [Serializer.serialize(r) for r in records]

    @staticmethod
    def deserialize_many(data_list: List[Dict[str, Any]]) -> List[ExecutionRecord]:
        """Deserializes an iterable collection of dictionary payloads."""
        return [Serializer.deserialize(item) for item in data_list]