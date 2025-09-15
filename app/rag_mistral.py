import os
import requests
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding


# ---- CONFIG ----
QDRANT_PATH = os.getenv("QDRANT_PATH", "db.qdrant")
COLLECTION = os.getenv("QDRANT_COLLECTION", "kb_policy_policy_chunks")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # change if tunneling
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "phi3:mini")           # Ollama name for mistral-7b-instruct
TOP_K = int(os.getenv("TOP_K", "3"))


def retrieve(query: str, k: int = TOP_K) -> List[Dict]:
    """Vector search in Qdrant; returns payloads + scores for prompting and citations."""
    client = QdrantClient(path=QDRANT_PATH)
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vec = list(embedder.embed([query]))[0]

    # Domain filter keeps it to policy KB (adjust/add filters later for products/promos)
    flt = Filter(must=[FieldCondition(key="domain", match=MatchValue(value="policy"))])

    hits = client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=k,
        query_filter=flt,
        with_payload=True,
    )

    out = []
    for h in hits:
        p = h.payload or {}
        out.append({
            "title": p.get("policy_title"),
            "url": p.get("source_url"),
            "text": p.get("text"),
            "score": h.score
        })
    return out


def build_prompt(user_q: str, ctx: List[Dict]) -> str:
    """Homework-style prompt: strict, minimal, enforce citations and fallback."""

    blocks = []
    for i, c in enumerate(ctx, 1):
        blocks.append(
            f"[{i}] TITLE: {c['title']}\nTEXT: {c['text']}\n"
        )
    context_block = "\n".join(blocks)

    return (
        "You are a helpful assistant. Use ONLY the context below to answer.\n"
        "If the answer is not in the context, say: \"I don't know\".\n"
        "Answer in one short sentence. Cite the source like [1], [2], etc.\n\n"
        f"<context>\n{context_block}\n</context>\n\n"
        f"Question: {user_q}\n\n"
        "Answer:"
    )



def answer_with_ollama(prompt: str, model: str = MISTRAL_MODEL) -> str:
    """Call LLM via Ollama HTTP API."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2,
            "options": {"num_predict": 80}  # try 64–128 depending on speed
        },
        timeout=600,  # give it more headroom on first token (cold load)
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()



def rag_answer(user_query: str, k: int = TOP_K) -> Dict:
    ctx = retrieve(user_query, k=k)
    prompt = build_prompt(user_query, ctx)
    answer = answer_with_ollama(prompt)
    return {"answer": answer, "context": ctx, "prompt": prompt}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What happens if I am not at home during delivery?"
    resp = rag_answer(q)
    print("\n--- Answer ---\n")
    print(resp["answer"])

    print("\n--- Citations ---\n")
    for i, c in enumerate(resp["context"], 1):
        print(f"[{i}] {c['title']} — {c['url']} (score={c['score']:.3f})")

    # Uncomment to see the full prompt (useful for debugging)
    # print("\n--- Prompt ---\n")
    # print(resp["prompt"])
