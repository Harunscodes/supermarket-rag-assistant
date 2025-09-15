# kg/ingest_product.py
import json
from pathlib import Path
from neo4j import GraphDatabase
from kg.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

DATA_PATH = Path("data/product_faqs.jsonl")

def ingest():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        with DATA_PATH.open() as f:
            for line in f:
                doc = json.loads(line)
                session.run(
                    """
                    MERGE (p:Product {id: $id})
                    SET p.category = $category,
                        p.question = $question,
                        p.answer = $answer
                    """,
                    id=doc["id"],
                    category=doc.get("category", ""),  # <-- FIX: use 'category'
                    question=doc["question"],
                    answer=doc["answer"],
                )
    print("✅ Ingested Product KB into Neo4j")

if __name__ == "__main__":
    ingest()
