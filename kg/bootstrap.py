# kg/bootstrap.py
from neo4j import GraphDatabase
from kg.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

DOMAINS = ["Delivery", "Refunds and Returns", "Substitutions", "Promotions", "Store Hours"]
CATEGORIES = ["Bakery", "Dairy", "Fresh Produce", "Household", "Beverages"]

def run():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        # constraints (id uniqueness where used)
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Policy) REQUIRE p.id IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Product) REQUIRE pr.id IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")

        for d in DOMAINS:
            s.run("MERGE (:Domain {name: $n})", n=d)
        for c in CATEGORIES:
            s.run("MERGE (:Category {name: $n})", n=c)

    driver.close()
    print("Bootstrap completed.")

if __name__ == "__main__":
    run()
