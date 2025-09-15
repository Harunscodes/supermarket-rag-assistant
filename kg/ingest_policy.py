# kg/ingest_policy.py
import json
from pathlib import Path
from neo4j import GraphDatabase
from kg.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

DATA_PATH = Path("data/policy_faqs.jsonl")

def ingest():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        with DATA_PATH.open() as f:
            for line in f:
                doc = json.loads(line)
                session.run(
                    """
                    MERGE (p:Policy {id: $id})
                    SET p.section = $section,
                        p.question = $question,
                        p.answer = $answer
                    """,
                    id=doc["id"],
                    section=doc["section"],
                    question=doc["question"],
                    answer=doc["answer"],
                )
    print("✅ Ingested Policy KB into Neo4j")

if __name__ == "__main__":
    ingest()
