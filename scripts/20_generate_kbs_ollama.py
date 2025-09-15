# scripts/20_generate_kbs_ollama.py
import json
import hashlib
import time
import re
from pathlib import Path
from typing import List, Dict, Any
import requests

# ---------- CONFIG ----------
OLLAMA_URL = "http://127.0.0.1:11434"
GEN_MODEL  = "phi3:mini"         # smaller/faster for CPU Codespaces
TEMPERATURE = 0.2
MAX_RETRIES = 3
REQ_TIMEOUT = 300                # seconds

OUT_DIR = Path("data")
POLICY_OUT   = OUT_DIR / "policy_faqs.jsonl"    # 100 records (5 sections × 20)
PRODUCT_OUT  = OUT_DIR / "product_faqs.jsonl"   # 100 records (5 cats × 20)

# Single brand (unified KB)
BRANDS = ["SupermarketCo"]

# Domains
POLICY_SECTIONS = ["Delivery", "Refunds and Returns", "Substitutions", "Promotions", "Store Hours"]
PRODUCT_CATEGORIES = ["Bakery", "Dairy", "Fresh Produce", "Household", "Beverages"]

# Generation strategy
BATCH_ITEMS = 2         # generate 2 QAs per request (faster/safer on CPU)
TARGET_PER_GROUP = 20   # 5 groups * 20 = 100 per KB


def call_ollama(prompt: str, model: str = GEN_MODEL, timeout: int = REQ_TIMEOUT) -> str:
    """Call Ollama /api/generate and return raw response text."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": TEMPERATURE,
                    "options": {"num_predict": 128}  # keep outputs short and snappy
                },
                timeout=timeout,
            )
            r.raise_for_status()
            resp = r.json().get("response", "")
            if resp and resp.strip():
                return resp
        except requests.exceptions.ReadTimeout:
            pass
        time.sleep(0.6 * attempt)
    raise RuntimeError(f"Ollama returned empty/timeout after {MAX_RETRIES} attempts")


def extract_json_block(text: str) -> str:
    # Try to extract a JSON array first
    m = re.search(r'(\[\s*{.*}\s*\])', text, flags=re.DOTALL)
    if m:
        return m.group(1)
    # Try a single JSON object as fallback
    m = re.search(r'({.*})', text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return text


def parse_qa_list(raw: str) -> List[Dict[str, str]]:
    """Parse model output into list of {'question','answer'}."""
    raw = extract_json_block(raw).strip()
    items: List[Dict[str, str]] = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
            items = obj["items"]
        elif isinstance(obj, list):
            items = obj
    except Exception:
        # Fallback bullet/regex parser
        for m in re.finditer(r'Q:\s*(.*?)\nA:\s*(.*?)(?:\n\n|$)', raw, flags=re.DOTALL):
            q = m.group(1).strip()
            a = re.sub(r'\s+', ' ', m.group(2)).strip()
            if q and a:
                items.append({"question": q, "answer": a})

    cleaned = []
    for it in items:
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        if q and a:
            cleaned.append({"question": q, "answer": a})
    return cleaned


import hashlib

def make_id(*parts: str, take: int = 16) -> str:
    base = "||".join(p.strip().lower() for p in parts if p)
    h = hashlib.md5(base.encode()).hexdigest()
    return h[:take]


POLICY_PROMPT_TMPL = """Generate EXACTLY {n} realistic customer FAQs with concise, policy-accurate answers
for a supermarket brand.

BRAND: {brand}
SECTION: {section}

Rules:
- Short answers (<= 2 sentences), generic (no dates/urls/phone numbers).
- Stay on the section topic.
- Output JSON array ONLY: [{{"question":"...","answer":"..."}}, ...]
"""

PRODUCT_PROMPT_TMPL = """Generate EXACTLY {n} realistic customer FAQs with concise answers about PRODUCTS and PROMOTIONS.

BRAND: {brand}
CATEGORY: {category}

Rules:
- Include product attributes (size/dietary/allergens/availability) or promotions (eligibility/min spend/stacking).
- Short answers (<= 2 sentences), generic (no dates/urls/phone numbers).
- Output JSON array ONLY: [{{"question":"...","answer":"..."}}, ...]
"""


def gen_batch_policy(brand: str, section: str, n: int) -> List[Dict[str, str]]:
    prompt = POLICY_PROMPT_TMPL.format(brand=brand, section=section, n=n)
    raw = call_ollama(prompt)
    qa = parse_qa_list(raw)
    return qa[:n]


def gen_batch_product(brand: str, category: str, n: int) -> List[Dict[str, str]]:
    prompt = PRODUCT_PROMPT_TMPL.format(brand=brand, category=category, n=n)
    raw = call_ollama(prompt)
    qa = parse_qa_list(raw)
    return qa[:n]


def generate_policy_records() -> List[Dict[str, Any]]:
    records = []
    for brand in BRANDS:
        for section in POLICY_SECTIONS:
            print(f"[policy] {brand} / {section} ...", flush=True)
            items: List[Dict[str, str]] = []
            while len(items) < TARGET_PER_GROUP:
                need = min(BATCH_ITEMS, TARGET_PER_GROUP - len(items))
                batch = gen_batch_policy(brand, section, need)
                if not batch:
                    # minimal safe fallback if model fails
                    batch = [{"question": f"Question about {section.lower()}?",
                              "answer": f"Please refer to standard {section.lower()} policy."}]
                items.extend(batch)
            items = items[:TARGET_PER_GROUP]

            for it in items:
                q, a = it["question"], it["answer"]
                rec = {
                    "brand": brand,
                    "section": section,
                    "question": q,
                    "answer": a,
                    "domain": "policy",
                }
                rec["id"] = make_id(brand, section, q, a[:24])
                records.append(rec)

    expected = len(BRANDS) * len(POLICY_SECTIONS) * TARGET_PER_GROUP
    assert len(records) == expected, f"Expected {expected} policy records, got {len(records)}"
    return records


def generate_product_records() -> List[Dict[str, Any]]:
    records = []
    for brand in BRANDS:
        for cat in PRODUCT_CATEGORIES:
            print(f"[product] {brand} / {cat} ...", flush=True)
            items: List[Dict[str, str]] = []
            while len(items) < TARGET_PER_GROUP:
                need = min(BATCH_ITEMS, TARGET_PER_GROUP - len(items))
                batch = gen_batch_product(brand, cat, need)
                if not batch:
                    batch = [{"question": f"Question about {cat.lower()} products?",
                              "answer": "This product category follows standard availability and promotion rules."}]
                items.extend(batch)
            items = items[:TARGET_PER_GROUP]

            for it in items:
                q, a = it["question"], it["answer"]
                rec = {
                    "brand": brand,
                    "category": cat,
                    "question": q,
                    "answer": a,
                    "domain": "product",
                }
                rec["id"] = make_id(brand, cat, q, a[:24])
                records.append(rec)

    expected = len(BRANDS) * len(PRODUCT_CATEGORIES) * TARGET_PER_GROUP
    assert len(records) == expected, f"Expected {expected} product records, got {len(records)}"
    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Policy KB (100)...")
    policy = generate_policy_records()
    with POLICY_OUT.open("w", encoding="utf-8") as f:
        for r in policy:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(policy)} → {POLICY_OUT}")

    print("Generating Product KB (100)...")
    product = generate_product_records()
    with PRODUCT_OUT.open("w", encoding="utf-8") as f:
        for r in product:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(product)} → {PRODUCT_OUT}")

    print("Done.")


if __name__ == "__main__":
    main()
