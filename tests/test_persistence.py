import os
import unittest
from uuid import uuid4

from remem import Client
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_context import ExecutionContext
from remem.storage.json_storage import JsonStorage
from remem.storage.serializer import Serializer
from remem.storage.exceptions import CorruptedSnapshotException


class TestPersistenceEngine(unittest.TestCase):

    def setUp(self):
        self.test_file = "test_store.json"

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(f"{self.test_file}.tmp"):
            os.remove(f"{self.test_file}.tmp")

    def test_serializer_round_trip(self):
        record = ExecutionRecord(
            id=uuid4(),
            embedding=[0.12, 0.34],
            references=["src.txt"],
            response="Tested response payload.",
            context=ExecutionContext(namespace="test_ns", kb_version="1.0", prompt_version="2.0")
        )
        
        dict_data = Serializer.serialize(record)
        recovered = Serializer.deserialize(dict_data)

        self.assertEqual(record.id, recovered.id)
        self.assertEqual(record.response, recovered.response)
        self.assertEqual(record.context.namespace, recovered.context.namespace)

    def test_atomic_persistence_cycles(self):
        storage = JsonStorage(filepath=self.test_file)
        client = Client(storage_backend=storage)

        # Generate and store 100 entries
        for i in range(100):
            client.store(ExecutionRecord(
                embedding=[float(i)/100, 0.0],
                references=[f"doc_{i}"],
                response=f"Resp {i}",
                context=ExecutionContext(namespace="test")
            ))

        self.assertEqual(len(client.all()), 100)

        # Reload storage to simulate system restart
        reloaded_storage = JsonStorage(filepath=self.test_file)
        reloaded_client = Client(storage_backend=reloaded_storage)
        self.assertEqual(len(reloaded_records := reloaded_client.all()), 100)

        # Delete entries and verify persistence
        target_id = reloaded_records[0].id
        reloaded_client.delete(target_id)
        
        post_delete_storage = JsonStorage(filepath=self.test_file)
        post_delete_client = Client(storage_backend=post_delete_storage)
        self.assertEqual(len(post_delete_client.all()), 99)

    def test_malformed_json_exception(self):
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("{ invalid json structure")

        with self.assertRaises(CorruptedSnapshotException):
            JsonStorage(filepath=self.test_file)


if __name__ == "__main__":
    unittest.main()