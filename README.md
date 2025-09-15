* Supermarket Agentic RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) system with Qdrant (vector DB), Neo4j (knowledge graph), and an agent layer on top of an LLM (Ollama + Mistral/Phi3).
It is designed to answer supermarket FAQs that span policies and products, sometimes both at once.

* Problem Description (2/2)

Supermarkets receive frequent customer questions, for example:

“What happens if my bakery delivery is late on Sunday?”

“If gluten-free bread is out of stock during promotion, can I get a substitution?”

“Do you provide free delivery?”

Answering these requires knowledge from two different domains:

Policy KB → rules about delivery, refunds, substitutions, promotions, and store hours.

Example: “Do you provide free delivery services? → Yes, orders over a certain amount qualify for complimentary next-day delivery.”

Product KB → FAQs about product categories (Bakery, Dairy, Fresh Produce, Beverages, Household).

Example: “Are there gluten-free bakery items available? → Yes, we carry several gluten-free bread and pastry options.”

The challenge: cross-domain queries often require combining both.
For instance, “refund for late bakery delivery on Sunday” touches:

Policy KB (refunds, delivery hours)

Product KB (bakery category)

This project solves that by combining:

Qdrant for semantic FAQ retrieval (Policy + Product KBs separately ingested)

Neo4j for structured policy–product relationships

An agent layer that decides whether to search or answer directly, then fuses both KBs.

* Retrieval Flow (2/2)

The flow is:

Agent decision (SEARCH vs ANSWER) based on a prompt.

If SEARCH → retrieve from both KBs in Qdrant (Policy + Product) and from Neo4j KG (structural facts).

Fuse contexts into a single prompt.

LLM answer generation via Ollama.

Both knowledge bases and the LLM are used in the flow.

* Retrieval Evaluation (2/2)

We evaluated retrieval using:

Qdrant-only retrieval (policy-only or product-only).

Hybrid retrieval (vector + lexical overlap rerank).

Parallel fusion (Policy + Product + KG) → improved coverage for cross-domain queries.

The agent now uses the fused Policy KB + Product KB + KG setup.

* LLM Evaluation (2/2)

We tested:

Different prompts (strict “only context” vs relaxed “partial facts allowed”).

Different models (Mistral, Phi3-mini) via Ollama.

Evaluation metrics: TF-IDF cosine similarity and ROUGE F1 against gold answers.
The strict prompt gave safest behavior (no hallucinations).

* Interface (1/2)

CLI interface via run_agent.py.

Example:

python run_agent.py "refund for late bakery delivery on Sunday"


Outputs:

Mode (DIRECT vs RAG_SEARCH)

Final Answer

Context titles (Policy KB, Product KB, KG)

Raw agent decision JSON

* For demo, CLI is enough. (UI would be next step for 2/2 points.)

* Ingestion Pipeline (2/2)

The pipeline is fully automated:

scripts/20_generate_kbs_ollama.py → Generates synthetic FAQs:

100 Policy KB records (Delivery, Refunds, Substitutions, Promotions, Store Hours).

100 Product KB records (Bakery, Dairy, Fresh Produce, Household, Beverages).

ingestion/policy_kb_to_qdrant.py / ingestion/product_kb_to_qdrant.py → Push Policy KB and Product KB into Qdrant (separate collections).

kg/ingest_policy.py / kg/ingest_product.py → Load both KBs into Neo4j as nodes.

* Monitoring (0–1/2)

Monitoring scripts:

evaluation/eval_qdrant.py → retrieval Hit@K / MRR.

evaluation/eval_rag_outputs.py → RAG answer quality (TF-IDF, ROUGE).

No live user feedback loop or dashboard yet.

* Containerization (1/2)

docker-compose.yml spins up Neo4j + Qdrant.

Application scripts run in Python (no Dockerfile for app yet).

* Reproducibility (2/2)

KBs are generated from scratch (via Ollama).

Ingestion is scripted.

Dependencies listed in requirements.txt.

Step-by-step run instructions (see below).

* Best Practices

✅ Hybrid search (vector + lexical tie-breaker).

✅ Document re-ranking.

❌ Query rewriting.

* How to Use
1. Start dependencies
docker-compose up -d

2. Generate KBs
python scripts/20_generate_kbs_ollama.py

3. Ingest into Qdrant + Neo4j
python ingestion/policy_kb_to_qdrant.py
python ingestion/product_kb_to_qdrant.py
python kg/bootstrap.py
python kg/ingest_policy.py
python kg/ingest_product.py

4. Run the Agent
python run_agent.py "refund for late bakery delivery on Sunday"


Example output:

=== MODE ===
 RAG_SEARCH

=== ANSWER ===
 I don't know [2] (contact customer service for weekend deliveries).

=== CONTEXT TITLES ===
[1] Policy KB: How do refunds work for missing or damaged items?
[2] Policy KB: What happens if I am not at home during delivery?
[3] Product KB: Bakery FAQ
[4] KG: Weekend delivery fact

=== DECISION ===
{
  "action": "SEARCH",
  "reasoning": "Need to search FAQ database for refund and delivery policies."
}

* Example Questions to Try

Policy-only → “Do you provide free delivery?”

Product-only → “Are there lactose-free dairy options?”

Cross-domain → “If gluten-free bread is out of stock during promotion, can I get a substitution?”

Cross-domain (time-sensitive) → “Refund for late bakery delivery on Sunday”

Policy-only (time) → “What are your store hours on weekends?”

* Summary

This project demonstrates a full Agentic RAG pipeline with:

Policy KB (policies and rules)

Product KB (category-specific FAQs)

Qdrant for semantic retrieval

Neo4j KG for cross-domain structure

Agent deciding when to search and fusing Policy + Product + KG
