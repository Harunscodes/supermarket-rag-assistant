from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding

def main():
    client = QdrantClient(path="db.qdrant")
    collection = "kb_policy_policy_chunks"
    print("Using collection:", collection)

    # Same model family as ingestion
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    query_text = "What to do if the customer is not at home during delivery?"
    vec = list(embedder.embed([query_text]))[0]

    flt = Filter(must=[FieldCondition(key="domain", match=MatchValue(value="policy"))])

    hits = client.search(
        collection_name=collection,
        query_vector=vec,
        limit=3,
        query_filter=flt,
        with_payload=True,
    )

    hits = client.search(
    collection_name=collection,
    query_vector=vec,
    limit=3,
    query_filter=flt,
    with_payload=True,
)

    print("\nRaw hits:", hits)   # DEBUG

    print("\nTop matches:")
    for h in hits:
        payload = h.payload or {}
        title = payload.get("policy_title")
        snippet = (payload.get("text") or "")[:220].replace("\n", " ")
        print(f"- score={h.score:.3f} | title={title}")
        print("  ", snippet, "...\n")

if __name__ == "__main__":
    main()
