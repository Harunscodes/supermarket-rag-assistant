import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge import Rouge

# Import your RAG pipeline (retrieval + prompt + LLM)
# Run from repo root:  python -m evaluation.eval_rag_outputs
from app.rag_mistral import rag_answer

EVAL_FILE = Path("evaluation/eval_qna.jsonl")
OUT_CSV   = Path("evaluation/rag_outputs.csv")

# Optionally control retrieved k during eval (smaller k often helps precision)
import os
TOP_K_EVAL = int(os.getenv("TOP_K_EVAL", "3"))

# ---------- Normalization / post-processing ----------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_CITATION   = re.compile(r'\s*\[\d+\]')                   # [1], [2], ...
_WS         = re.compile(r'\s+')

# Remove common citation/artifact phrases left by model prompts
_ARTIFACT_PATTERNS = [
    r'\(Excerpt from Context\)', r'\(excerpt from context\)',
    r'\(as per context.*?\)', r'\(according to context.*?\)',
    r'\(see context.*?\)', r'\(from context.*?\)',
    r'\(\s*\)',                  # empty parentheses
]

def clean_artifacts(text: str) -> str:
    t = text
    t = _CITATION.sub("", t)
    for pat in _ARTIFACT_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    # remove stray spaces before punctuation and collapse whitespace
    t = re.sub(r'\s+([.,;:!?])', r'\1', t)
    t = _WS.sub(" ", t).strip()
    return t

def first_sentence(text: str) -> str:
    if not text:
        return text
    parts = _SENT_SPLIT.split(text.strip())
    return parts[0].strip() if parts else text.strip()

def normalize_for_eval(raw: str) -> str:
    # order matters: clean artifacts, then take the first sentence, then final tidy
    t = clean_artifacts(raw or "")
    t = first_sentence(t)
    t = _WS.sub(" ", t).strip()
    return t

def tfidf_cosine(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    vec = TfidfVectorizer().fit([a, b])
    X = vec.transform([a, b])
    num = X[0].multiply(X[1]).sum()
    den = (X[0].power(2).sum() ** 0.5) * (X[1].power(2).sum() ** 0.5)
    return float(num / den) if den != 0 else 0.0

def compute_rouge(hyp: str, ref: str) -> Dict[str, float]:
    if not hyp or not ref:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    r = Rouge()
    scores = r.get_scores(hyp, ref, avg=True)
    return {
        "rouge1_f": scores["rouge-1"]["f"],
        "rouge2_f": scores["rouge-2"]["f"],
        "rougeL_f": scores["rouge-l"]["f"],
    }

# ---------- Main ----------

def main():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"Missing {EVAL_FILE}. Each line must be JSON with keys like:\n"
            '{"query":"...","gold_answer":"..."}'
        )

    rows = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines()]
    if not rows:
        print("No eval rows found.")
        return

    results: List[Dict[str, Any]] = []

    for i, rec in enumerate(rows, start=1):
        q   = (rec.get("query") or "").strip()
        ref = (rec.get("gold_answer") or "").strip()
        raw = ""
        hyp = ""

        try:
            resp = rag_answer(q, k=TOP_K_EVAL)
            raw  = (resp.get("answer") or "").strip()
            hyp  = normalize_for_eval(raw)
        except Exception as e:
            print(f"\n[WARN] RAG failed on idx={i}: {e}")

        cos = tfidf_cosine(hyp, ref)
        rgs = compute_rouge(hyp, ref)

        results.append({
            "idx": i,
            "query": q,
            "gold_answer": ref,
            "model_answer_raw": raw,
            "model_answer_eval": hyp,  # normalized: first sentence, no citations/artifacts
            "cosine_tfidf": round(cos, 3),
            "rouge1_f": round(rgs["rouge1_f"], 3),
            "rouge2_f": round(rgs["rouge2_f"], 3),
            "rougeL_f": round(rgs["rougeL_f"], 3),
        })

    # Summary
    cos_mean = float(np.mean([r["cosine_tfidf"] for r in results])) if results else 0.0
    r1_mean  = float(np.mean([r["rouge1_f"]     for r in results])) if results else 0.0
    r2_mean  = float(np.mean([r["rouge2_f"]     for r in results])) if results else 0.0
    rl_mean  = float(np.mean([r["rougeL_f"]     for r in results])) if results else 0.0

    print("\n--- RAG Output Evaluation ---")
    print(f"Avg cosine (TF-IDF): {cos_mean:.3f}")
    print(f"Avg ROUGE-1 F1:      {r1_mean:.3f}")
    print(f"Avg ROUGE-2 F1:      {r2_mean:.3f}")
    print(f"Avg ROUGE-L F1:      {rl_mean:.3f}")

    # Save detailed CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved detailed results to {OUT_CSV}")

    # Show a couple of samples
    print("\nSamples:")
    for r in results[:2]:
        print(
            f"\nQ: {r['query']}\n"
            f"Ref: {r['gold_answer']}\n"
            f"Ans (eval): {r['model_answer_eval']}\n"
            f"cos={r['cosine_tfidf']}  R1={r['rouge1_f']}  R2={r['rouge2_f']}  RL={r['rougeL_f']}"
        )

if __name__ == "__main__":
    main()
