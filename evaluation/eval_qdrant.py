import json
import argparse
from pathlib import Path
from statistics import mean

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding

# -------- config --------
QDRANT_PATH = "db.qdrant"
COLLECTION = "kb_policy_policy_chunks"
K = 5
USE_HYBRID_RERANK = True  # set False to disable lexical tie-breaker

def lexical_overlap_score(query: str, text: str) -> int:
    qs = set(query.lower().split())
    ts = set((text or "").lower().split())
    return len(qs & ts)

def main():
    eval_file = Path("evaluation/eval_queries.jsonl")  # each line has query + gold_parent_id or gold_title
    rows = [json.loads(l) for l in eval_file.read_text(encoding="utf-8").splitlines()]

    client = QdrantClient(path=QDRANT_PATH)
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    hits_all, rr_all = [], []

    for rec in rows:
        query = rec["query"]
        gold_pid = rec.get("gold_parent_id")
        gold_title = rec.get("gold_title")  # fallback if no parent id provided

        vec = list(embedder.embed([query]))[0]
        flt = Filter(must=[FieldCondition(key="domain", match=MatchValue(value="policy"))])

        # ANN step
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=max(K, 20),     # fetch a bit more then prune/dedup
            query_filter=flt,
            with_payload=True,
        )

        # Optional: hybrid rerank by lexical overlap (cheap tie-breaker)
        if USE_HYBRID_RERANK:
            rescored = []
            for h in hits:
                txt = (h.payload or {}).get("text", "")
                rescored.append((lexical_overlap_score(query, txt), h))
            rescored.sort(key=lambda x: (x[0], x[1].score), reverse=True)  # lex score then ANN score
            hits = [h for _, h in rescored]

        # De-duplicate by parent_id (document-level)
        seen = set()
        dedup = []
        for h in hits:
            pid = (h.payload or {}).get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                dedup.append(h)

        # Keep top-K after dedup/rerank
        hits = dedup[:K]

        titles = [ (h.payload or {}).get("policy_title") for h in hits ]
        pids   = [ (h.payload or {}).get("parent_id")    for h in hits ]

        print(f"\nQuery: {query}")
        print("Retrieved (titles):", titles)

        # Decide match key
        if gold_pid:
            key_list = pids
            gold_key = gold_pid
        else:
            key_list = titles
            gold_key = gold_title

        # Hit@K and MRR
        hit = int(gold_key in key_list)
        hits_all.append(hit)
        try:
            rank = key_list.index(gold_key) + 1
            rr_all.append(1.0 / rank)
            print(f"✅ Found gold at rank {rank}")
        except ValueError:
            rr_all.append(0.0)
            print("❌ Gold not found")

    print("\n--- Retrieval Evaluation ---")
    print(f"Hit@{K}: {mean(hits_all):.2f}")
    print(f"MRR:   {mean(rr_all):.2f}")

if __name__ == "__main__":
    main()
