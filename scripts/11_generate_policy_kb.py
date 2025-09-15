import hashlib, json
from pathlib import Path

OUT = Path("data/policy_faqs.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

brands   = ["Tesco", "Walmart", "Aldi", "Carrefour"]
sections = {
    "Delivery": [
        ("What happens if I am not at home during delivery?",
         "If you’re not at home, your order will be returned to the store and a refund will be processed. You can contact customer service to rearrange delivery."),
        ("Can I change my delivery slot?",
         "You can change your delivery slot up to 24 hours before the scheduled time in your account."),
        ("Will the driver call me before arrival?",
         "Drivers may contact you shortly before arrival if necessary, but this cannot be guaranteed."),
        ("Can I leave delivery instructions?",
         "Yes, you can add delivery instructions at checkout for safe drop-off where available."),
        ("Do you deliver to upper floors?",
         "Deliveries are made to the main entrance; assistance to upper floors is not guaranteed.")
    ],
    "Refunds and Returns": [
        ("How do I get a refund for missing groceries?",
         "If groceries are missing or damaged, request a refund within a few days of delivery. Refunds are usually processed in 3–5 working days."),
        ("Can I return perishable items?",
         "Perishable items may be refunded if damaged on delivery; returns are generally not accepted once delivered."),
        ("What proof do I need for a refund?",
         "Provide your order number and a brief description of the issue. Photos may be requested for damaged items."),
        ("How long do refunds take?",
         "Refunds typically take 3–5 working days to appear, depending on your payment provider."),
        ("Can I exchange an item?",
         "Exchanges are not supported; we process refunds and you can reorder the desired item.")
    ],
    "Substitutions": [
        ("Will substitutions be made for out-of-stock items?",
         "When items are unavailable, similar products may be substituted. You can accept or decline substitutions in your account settings."),
        ("How are substitutions chosen?",
         "Substitutions are selected based on closest match in brand, size, and price where possible."),
        ("Can I turn off substitutions?",
         "Yes, you can disable substitutions for individual items or your entire order in settings."),
        ("Do promotions apply to substituted items?",
         "Promotions generally apply to substituted items if the replacement qualifies under the same offer."),
        ("What if I don’t like the substitution?",
         "You can reject a substitution at the door and it will be removed from your order.")
    ],
    "Promotions": [
        ("How do I apply a promo code?",
         "Enter the promo code at checkout before payment. Only one code may be applied per order."),
        ("Can I combine promotions?",
         "Most promotions cannot be combined; the best eligible discount will apply."),
        ("Do promotions apply to delivery fees?",
         "Promotions usually exclude delivery fees unless explicitly stated."),
        ("Why didn’t my promo apply?",
         "The promo may have expired or your basket may not meet eligibility conditions like minimum spend."),
        ("Do promotions apply to age-restricted items?",
         "Age-restricted items are commonly excluded from promotions unless stated otherwise.")
    ],
    "Store Hours": [
        ("What are Sunday opening hours?",
         "Most stores open 8 AM–4 PM on Sundays; hours may vary by location."),
        ("Are holiday hours different?",
         "Holiday hours may differ; check the store finder for specific dates."),
        ("When is the bakery open?",
         "In-store bakeries typically operate during store opening hours, with fresh bakes in the morning."),
        ("Do you open earlier for seniors?",
         "Some locations offer early hours for vulnerable customers; availability varies by store."),
        ("Is customer service desk always staffed?",
         "Customer service desks are staffed during core hours; times vary by store.")
    ],
}

def make_id(brand: str, q: str, a: str) -> str:
    s = f"{brand}|{q}|{a[:24]}"
    return hashlib.md5(s.encode()).hexdigest()[:8]

records = []
for brand in brands:
    for section, qa_list in sections.items():
        for q, a in qa_list:
            rec = {
                "brand": brand,
                "section": section,
                "question": q,
                "answer": a,
                "domain": "policy",
            }
            rec["id"] = make_id(brand, q, a)
            records.append(rec)

with OUT.open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(records)} records to {OUT}")
