import json
from pathlib import Path

faqs = Path("data/faqs.jsonl").read_text(encoding="utf-8").splitlines()
print("FAQ records:", len(faqs))

pol_path = Path("data/policies.jsonl")
if pol_path.exists():
    pols = pol_path.read_text(encoding="utf-8").splitlines()
    print("Policy records:", len(pols))
    if pols:
        ex = json.loads(pols[0])
        print("\nSample policy title:", ex.get("policy_title"))
        print(ex.get("policy_text", "")[:600], "...")
else:
    print("Policy records: 0 (run the generator)")
