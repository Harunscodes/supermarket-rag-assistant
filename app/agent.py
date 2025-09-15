# app/agent.py
import os, re, json
from typing import List, Dict
from neo4j import GraphDatabase

from app.rag_mistral import retrieve, build_prompt, answer_with_ollama
from kg.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# ---- Config (env overrides) ----
QDRANT_PATH = os.getenv("QDRANT_PATH", "db.qdrant")
POLICY_COLL = os.getenv("POLICY_COLL", "kb_policy_faqs")
PRODUCT_COLL = os.getenv("PRODUCT_COLL", "kb_product_faqs")
OLLAMA_MODEL = os.getenv("MISTRAL_MODEL", "phi3:mini")  # same as your rag_mistral default

AGENT_PROMPT = """
You're a course teaching assistant.

You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
At the beginning the context is EMPTY.

<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>

If CONTEXT is EMPTY, you can use our FAQ database.
In this case, use the following output template:

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>"
}}

If you can answer the QUESTION using CONTEXT, Use this template:
{{
  "action": "ANSWER",
  "answer": "<your answer>",
  "source": "CONTEXT"
}}

If the context doesn’t contain the answer, Use your own knowledge to answer the question.
{{
  "action": "ANSWER",
  "answer": "<your answer>",
  "source": "OWN_KNOWLEDGE"
}}
""".strip()


def _safe_json_from_text(raw: str) -> Dict:
    """
    Extract FIRST JSON object/array from model output; fallback to SEARCH if parsing fails.
    """
    try:
        m = re.search(r'(\{.*\}|\[.*\])', raw, flags=re.DOTALL)
        blob = m.group(1) if m else raw
        return json.loads(blob)
    except Exception:
        return {"action": "SEARCH", "reasoning": "Parse failure or ambiguous output."}


def agent_decide(question: str, context_text: str = "") -> Dict:
    """
    Ask the model to choose SEARCH vs ANSWER using the homework template.
    """
    prompt = AGENT_PROMPT.format(question=question, context=context_text)
    raw = answer_with_ollama(prompt, model=OLLAMA_MODEL)
    return _safe_json_from_text(raw)


def _kg_facts(query: str, limit: int = 2) -> List[Dict]:
    """
    Very small KG fetch (uses same matching idea as kg/query.py) but returns compact text snippets.
    """
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    CQL = """
    MATCH (n)
    WHERE any(w IN split(toLower($q), " ")
      WHERE (toLower(coalesce(n.question,"")) CONTAINS w OR toLower(coalesce(n.answer,"")) CONTAINS w))
    RETURN labels(n) AS labels, n.question AS q, n.answer AS a
    LIMIT $lim
    """
    out = []
    with drv.session() as s:
        for r in s.run(CQL, q=query, lim=limit):
            q = (r["q"] or "").strip()
            a = (r["a"] or "").strip()
            if not (q or a):
                continue
            text = (q + " — " + a).strip(" —")
            out.append({"title": "KG", "url": None, "text": text, "score": 1.0})
    return out


def _retrieve_parallel(user_q: str, k_policy=2, k_product=1) -> List[Dict]:
    """
    Parallel retrieval using your existing retriever; collects policy, product, and KG facts.
    """
    ctx: List[Dict] = []

    # Policy
    os.environ["QDRANT_COLLECTION"] = POLICY_COLL
    try:
        ctx += retrieve(user_q, k=k_policy)
    except Exception:
        pass  # keep going even if one side fails

    # Product
    os.environ["QDRANT_COLLECTION"] = PRODUCT_COLL
    try:
        ctx += retrieve(user_q, k=k_product)
    except Exception:
        pass

    # KG
    ctx += _kg_facts(user_q, limit=2)

    return ctx


def agent_answer(user_q: str) -> Dict:
    """
    The one-call agent entrypoint:
    - Decide SEARCH vs ANSWER with empty context.
    - If SEARCH: gather parallel context (policy+product+KG) and do RAG answer.
    - If ANSWER: return the direct answer.
    Returns a dict with keys: mode, answer, context (list), decision (raw).
    """
    decision = agent_decide(user_q, context_text="")
    action = (decision.get("action") or "").upper()

    if action == "SEARCH":
        ctx = _retrieve_parallel(user_q)
        prompt = build_prompt(user_q, ctx)
        ans = answer_with_ollama(prompt, model=OLLAMA_MODEL)
        return {"mode": "RAG_SEARCH", "answer": ans, "context": ctx, "decision": decision, "prompt": prompt}

    # Direct answer (no retrieval)
    if action == "ANSWER":
        return {"mode": "DIRECT", "answer": decision.get("answer", ""), "context": [], "decision": decision, "prompt": None}

    # Fallback safety
    ctx = _retrieve_parallel(user_q)
    prompt = build_prompt(user_q, ctx)
    ans = answer_with_ollama(prompt, model=OLLAMA_MODEL)
    return {"mode": "RAG_SEARCH", "answer": ans, "context": ctx, "decision": decision, "prompt": prompt}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "refund for late bakery delivery on Sunday"
    resp = agent_answer(q)

    print("\n=== MODE ===\n", resp["mode"])
    print("\n=== ANSWER ===\n", resp["answer"])
    if resp["mode"] == "RAG_SEARCH":
        print("\n=== CONTEXT TITLES ===")
        for i, c in enumerate(resp["context"], 1):
            print(f"[{i}] {c.get('title')}")
    print("\n=== DECISION ===\n", json.dumps(resp["decision"], indent=2))
