import json
import uuid
from pathlib import Path
from typing import Iterator, Dict, Any, List

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


# -----------------------------
# Stable IDs & simple chunking
# -----------------------------
NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000000")  # UUIDv5 namespace (constant)

def stable_uuid(name: str) -> str:
    """Deterministic UUIDv5 from a stable string (e.g., brand + title)."""
    return str(uuid.uuid5(NAMESPACE, name))

def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> List[str]:
    """Lightweight char-based chunker with small overlap."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # extend to end of sentence if possible (rough)
        dot = text.rfind(". ", start, end)
        if dot != -1 and dot + 2 - start > max_chars * 0.6:
            end = dot + 2
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    # drop empties
    return [c for c in chunks if c]

# -----------------------------
# Load raw policies & yield chunks with IDs
# -----------------------------
def iter_policy_chunks(src_path: Path) -> Iterator[Dict[str, Any]]:
    """
    Yields chunk records with:
      - parent_id: stable per-policy UUID (document-level id)
      - chunk_id:  parent_id + "-<n>" (string, human-friendly)
      - point_id:  UUID for Qdrant point id (required: UUID/int)
    """
    raw = [json.loads(l) for l in src_path.read_text(encoding="utf-8").splitlines()]
    for rec in raw:
        brand = rec["brand"]
        title = rec["policy_title"]
        url = rec.get("source_url")
        full = rec["policy_text"]
        domain = rec.get("domain", "policy")

        # stable document-level id (like 'id' in the homework)
        parent_id = stable_uuid(f"{brand}::{title}")

        chunks = chunk_text(full, max_chars=500, overlap=50)
        for idx, ch in enumerate(chunks, start=1):
            chunk_id = f"{parent_id}-{idx}"               # human-friendly
            point_id = uuid.uuid5(uuid.UUID(parent_id), str(idx))  # UUID required by Qdrant local

            yield {
                "point_id": str(point_id),     # used as Qdrant point id
                "parent_id": parent_id,        # document-level id (homework-style)
                "chunk_id": chunk_id,          # chunk-level id (string)
                "brand": brand,
                "policy_title": title,
                "source_url": url,
                "text": ch,
                "domain": domain,
            }

# -----------------------------
# Ingest to Qdrant (local path)
# -----------------------------
def main():
    DATA = Path("data/policies.jsonl")
    DB_PATH = "db.qdrant"
    COLLECTION = "kb_policy_policy_chunks"

    client = QdrantClient(path=DB_PATH)

    # (Re)create collection with UNNAMED dense vector (size 384, cosine)
    # This mirrors your homework style and avoids named-vector complications.
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=rest.VectorParams(size=384, distance=rest.Distance.COSINE),
    )

    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Batch upsert for efficiency
    BATCH = 64
    buffer = []

    def flush(batch: List[Dict[str, Any]]):
        if not batch:
            return
        texts = [r["text"] for r in batch]
        vecs = list(embedder.embed(texts))  # produces List[List[float]]
        points = []
        for r, v in zip(batch, vecs):
            points.append(
                rest.PointStruct(
                    id=r["point_id"],          # UUID string accepted by local Qdrant
                    vector=v,                  # UNNAMED dense vector
                    payload={
                        "parent_id": r["parent_id"],
                        "chunk_id": r["chunk_id"],
                        "brand": r["brand"],
                        "policy_title": r["policy_title"],
                        "source_url": r["source_url"],
                        "text": r["text"],
                        "domain": r["domain"],
                    },
                )
            )
        client.upsert(collection_name=COLLECTION, points=points)

    for rec in iter_policy_chunks(DATA):
        buffer.append(rec)
        if len(buffer) >= BATCH:
            flush(buffer)
            buffer = []
    flush(buffer)

    # Quick report
    cnt = client.count(COLLECTION, exact=True).count
    print(f"Upserted {cnt} chunks into '{COLLECTION}' (unnamed vector, 384-d, cosine).")
    # Show one sample with IDs
    hits = client.scroll(COLLECTION, limit=1, with_payload=True)[0]
    if hits:
        p = hits[0]
        print("Sample payload:", {
            "parent_id": p.payload.get("parent_id"),
            "chunk_id": p.payload.get("chunk_id"),
            "policy_title": p.payload.get("policy_title"),
            "brand": p.payload.get("brand"),
        })

if __name__ == "__main__":
    main()
