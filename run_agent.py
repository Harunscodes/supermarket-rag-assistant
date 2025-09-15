from app.agent import agent_answer
import sys, json

q = " ".join(sys.argv[1:]) or "refund for late bakery delivery on Sunday"
resp = agent_answer(q)

print("\n=== MODE ===\n", resp["mode"])
print("\n=== ANSWER ===\n", resp["answer"])
if resp["mode"] == "RAG_SEARCH":
    print("\n=== CONTEXT TITLES ===")
    for i,c in enumerate(resp["context"], 1):
        print(f"[{i}] {c.get('title')}")
print("\n=== DECISION ===\n", json.dumps(resp["decision"], indent=2))
