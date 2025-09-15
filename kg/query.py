# kg/query.py
import sys
from neo4j import GraphDatabase
from kg.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def run_query(user_query: str):
    """Very simple KG search: looks for Policy + Product nodes related to query keywords."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE any(word IN split(toLower($q), " ") WHERE toLower(n.question) CONTAINS word OR toLower(n.answer) CONTAINS word)
            RETURN labels(n) AS labels, n.id AS id, n.question AS question, n.answer AS answer
            LIMIT 10
            """,
            q=user_query,
        )
        rows = list(result)

    if not rows:
        print("No matches found.")
    else:
        for r in rows:
            labels = ", ".join(r["labels"])
            print(f"[{labels}] {r['question']}\n→ {r['answer']}\n")

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "refund policy"
    run_query(q)
