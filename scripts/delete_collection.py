from qdrant_client import QdrantClient

client = QdrantClient(path="db.qdrant")
client.delete_collection("kb_policy_policy_chunks")
print("Deleted collection: kb_policy_policy_chunks")
