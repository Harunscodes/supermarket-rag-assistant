import json
from pathlib import Path
from typing import Iterator, Dict, Any
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ---- LangChain chunker (state-of-practice) ----
# You can tweak these two numbers and re-run evaluation (Hit@k/MRR)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # The default separator hierarchy already prefers paragraph/sentence breaks.
    # You can pass `separators=[...]` if your data has special structure.
)

def chunk_text(text: str) -> list[str]:
    """Return sentence/paragraph-aware chunks with overlap."""
    # Normalize whitespace lightly; Recursive splitter is tolerant anyway.
    text = " ".join(text.split())
    return splitter.split_text(text)


# ---- data iterator: yields chunked policy records ----
def iter_policy_chunks() -> Iterator[Dict[str, Any]]:
    src = Path("data/policies.jsonl")
    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines(), start=1):
        rec = json.loads(line)
        title = rec["policy_title"]
        full = rec["policy_text"]
        brand = rec["brand"]
        url = rec["source_url"]

        for j, ch in enumerate(chunk_text(full), start=1):
            yield {
                "id": f"{i}-{j}",            # local composite id (we'll convert to UUID)
                "brand": brand,
                "source_url": url,
                "policy_title": title,
                "text": ch,                  # <--- TEXT FIELD TO EMBED
                "domain": "policy",
            }


def main():
    collection = "kb_policy_policy_chunks"
    client = QdrantClient(path="db.qdrant")

    # Clean & recreate collection with UNNAMED/default vector slot (homework-style)
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # One fixed model (same as retrieval)
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    points = []
    for rec in iter_policy_chunks():
        vec = list(embedder.embed([rec["text"]]))[0].tolist()
        payload = {
            "brand": rec["brand"],
            "source_url": rec["source_url"],
            "policy_title": rec["policy_title"],
            "text": rec["text"],
            "domain": rec["domain"],
        }
        # Deterministic UUID from composite id (stable across runs)
        pid = uuid5(NAMESPACE_URL, f"{collection}:{rec['id']}")
        points.append(PointStruct(id=str(pid), vector=vec, payload=payload))

    client.upsert(collection_name=collection, points=points)
    print(
        f"Upserted {len(points)} vectors into '{collection}' "
        f"(default unnamed vector; chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )


if __name__ == "__main__":
    main()
