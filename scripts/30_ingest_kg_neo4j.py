import json, re
from pathlib import Path
from neo4j import GraphDatabase

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "testpassword"

POLICY_FILE  = Path("data/policy_faqs.jsonl")
PRODUCT_FILE = Path("data/product_faqs.jsonl")

def extract_props(ans: str):
    props = {}
    m = re.search(r'(\d+)\s*[–-]\s*(\d+)\s*(?:working\s*)?days', ans, flags=re.I)
    if m:
      props["refund_days_min"] = int(m.group(1))
      props["refund_days_max"] = int(m.group(2))
    if re.search(r'\bsubstitut(e|ion)\b', ans, flags=re.I):
      props["allows_substitution"] = True
    return props

def run():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as s:
        # constraints + fulltext index
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:FAQ) REQUIRE f.id IS UNIQUE;")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE;")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE;")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;")
        s.run("""
        CREATE FULLTEXT INDEX faqText IF NOT EXISTS
        FOR (f:FAQ) ON EACH [f.question, f.answer];
        """)

        # ingest policy
        count = 0
        for line in POLICY_FILE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            props = extract_props(r["answer"])
            s.run("""
            MERGE (b:Brand {name: $brand})
            MERGE (t:Topic {name: $topic})
            MERGE (f:FAQ {id: $id})
              ON CREATE SET f.question=$q, f.answer=$a, f.domain='policy', f.brand=$brand
              ON MATCH  SET f.question=$q, f.answer=$a, f.domain='policy', f.brand=$brand
            MERGE (f)-[:OF_BRAND]->(b)
            MERGE (f)-[:IN_TOPIC]->(t)
            SET f += $props
            """, brand=r["brand"], topic=r["section"], id=r["id"], q=r["question"], a=r["answer"], props=props)
            count += 1
        print(f"Ingested policy FAQs: {count}")

        # ingest product
        count = 0
        for line in PRODUCT_FILE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            props = extract_props(r["answer"])
            s.run("""
            MERGE (b:Brand {name: $brand})
            MERGE (c:Category {name: $cat})
            MERGE (f:FAQ {id: $id})
              ON CREATE SET f.question=$q, f.answer=$a, f.domain='product', f.brand=$brand
              ON MATCH  SET f.question=$q, f.answer=$a, f.domain='product', f.brand=$brand
            MERGE (f)-[:OF_BRAND]->(b)
            MERGE (f)-[:IN_CATEGORY]->(c)
            SET f += $props
            """, brand=r["brand"], cat=r["category"], id=r["id"], q=r["question"], a=r["answer"], props=props)
            count += 1
        print(f"Ingested product FAQs: {count}")

    driver.close()

if __name__ == "__main__":
    run()
