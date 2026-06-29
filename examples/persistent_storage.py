import os
from remem import Client
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_context import ExecutionContext
from remem.models.execution_result import ExecutionResult
from remem.storage.json_storage import JsonStorage


def main():
    store_file = "durable_remem.json"
    
    # Clean up any leftover file from previous runs
    if os.path.exists(store_file):
        os.remove(store_file)

    print("🚀 Initializing File-Backed Persistence Engine...")
    storage = JsonStorage(filepath=store_file)
    client = Client(storage_backend=storage)

    # 1. Store records natively
    print("\n➕ Seeding 3 execution records...")
    client.store(ExecutionRecord(embedding=[0.1, 0.2], references=["doc_a"], response="Response A", context=ExecutionContext(namespace="ns1")))
    client.store(ExecutionRecord(embedding=[0.3, 0.4], references=["doc_b"], response="Response B", context=ExecutionContext(namespace="ns1")))
    client.store(ExecutionRecord(embedding=[0.5, 0.6], references=["doc_c"], response="Response C", context=ExecutionContext(namespace="ns1")))

    print(f"Total persisted records (Active Session): {len(client.all())}")
    
    # Explicitly persist data to disk file
    client.save_snapshot()
    print(f"✅ Snapshot written to disk at '{store_file}'. Size: {os.path.getsize(store_file)} bytes")

    # 2. Simulate Program Exit and Restart (Re-reading JSON snapshot without in-memory state)
    print("\n🔄 Simulating System Restart (Reinitializing Client)...")
    fresh_storage = JsonStorage(filepath=store_file)
    fresh_client = Client(storage_backend=fresh_storage)
    
    records_restored = fresh_client.all()
    print(f"Total records restored from disk: {len(records_restored)}")
    for i, rec in enumerate(records_restored, 1):
        print(f"  {i}. ID: {rec.id} | Context: {rec.context.namespace} | Response: {rec.response}")

    # Clean up local execution artifact
    if os.path.exists(store_file):
        os.remove(store_file)
    print("\n🧹 Temporary demo storage files cleaned up.")


if __name__ == "__main__":
    main()