import json, textwrap
from pathlib import Path
from tqdm import tqdm

INP = Path("data/faqs.jsonl")
OUT = Path("data/policies.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

TEMPLATE = """Policy: {title}

Scope:
  This policy applies to delivery drivers and customer service agents.

Procedure:
  1) Verify the issue and check the order details in the system.
  2) Follow the steps below based on the situation.
  3) Record the outcome in the appropriate system category.

Steps (derived from customer FAQ):
{steps}

Notes:
  - Communicate clearly and politely with the customer.
  - Do not leave age-restricted items with neighbors.
  - Escalate unusual cases to a supervisor.
  - Standard refund processing window: 3–5 working days unless otherwise stated.

Reference:
  - Source brand: {brand}
  - URL: {url}
"""

def faq_to_steps(question:str, answer:str)->str:
    parts = [p.strip() for p in answer.replace("•","-").split(".") if p.strip()]
    numbered = []
    for i, p in enumerate(parts, 1):
        numbered.append(f"  {i}) {p}.")
    if not numbered:
        numbered = ["  1) Follow standard customer service procedure for this case."]
    return "\n".join(numbered)

def main():
    out = OUT.open("w", encoding="utf-8")
    with INP.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Generating synthetic policies"):
            rec = json.loads(line)
            steps = faq_to_steps(rec["question"], rec["answer"])
            policy = TEMPLATE.format(
                title=rec["question"],
                steps=steps,
                brand=rec["brand"],
                url=rec["url"],
            )
            out_rec = {
                "brand": rec["brand"],
                "source_url": rec["url"],
                "policy_title": rec["question"],
                "policy_text": policy,
                "domain": "policy",
                "seed_type": "faq_to_policy_template"
            }
            out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    out.close()

if __name__ == "__main__":
    main()