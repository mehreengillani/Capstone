# build_compositions_short.py (manual tiers)
# Builds short-item compositions with HAND-ASSIGNED product tiers
# based on CRM market positioning (Gartner/Forrester-aligned).
#
# The pipeline's max_citation per product is reported for validation
# but does NOT determine tier assignment.

import json
import random
from pathlib import Path

# ---------- CONFIG ----------
OUT_PATH = Path("test2_compositions_short.json")
RANDOM_SEED = 42
N_COMPOSITIONS = 3
K = 6
DOCS_PER_TIER = 2

# ---------- Products with manual tiers ----------
# Tier assignment based on Gartner Magic Quadrant for CRM (2024-2025)
# and Forrester Wave positioning. Adjust if a newer report is preferred.
# ---------- Products with manual tiers AND descriptions ----------
# Tier assignment: Gartner MQ / Forrester Wave positioning (2024–2025).
# Descriptions: 1–2 factual sentences per product to give Haiku enough
# content to evaluate without refusing the ranking task.
PRODUCTS = [
    # --- HIGH tier (leaders) ---
    {
        "product": "Salesforce Sales Cloud",
        "description": "Enterprise-grade CRM with extensive customization, "
                       "third-party integrations, and a mature ecosystem. "
                       "Market leader for large organizations.",
        "tier": "high",
    },
    {
        "product": "HubSpot CRM",
        "description": "Free-tier CRM with strong marketing and sales integration. "
                       "Popular with SMBs for its ease of use and clean interface.",
        "tier": "high",
    },
    {
        "product": "Microsoft Dynamics 365 Sales",
        "description": "Enterprise CRM tightly integrated with Microsoft 365 and Azure. "
                       "Strong for organizations already using Microsoft's ecosystem.",
        "tier": "high",
    },

    # --- LOW tier (challengers / strong performers) ---
    {
        "product": "Zoho CRM",
        "description": "Affordable CRM with broad feature coverage and integration "
                       "with Zoho's wider suite. Strong value for mid-market businesses.",
        "tier": "low",
    },
    {
        "product": "Pipedrive",
        "description": "Sales-pipeline-focused CRM with a visual deal tracker. "
                       "Best for sales teams wanting simplicity over feature depth.",
        "tier": "low",
    },
    {
        "product": "Freshworks CRM",
        "description": "Modern CRM with AI-assisted features and integrations with "
                       "the Freshworks suite. Popular for mid-market sales teams.",
        "tier": "low",
    },

    # --- NEVER_CITED tier (niche / emerging) ---
    {
        "product": "Monday CRM",
        "description": "Newer CRM built on Monday.com's work-management platform. "
                       "Suited for teams already using Monday for project management.",
        "tier": "never_cited",
    },
    {
        "product": "Zendesk Sell",
        "description": "CRM integrated with Zendesk's customer support tools. "
                       "Best for teams that already use Zendesk for help desk.",
        "tier": "never_cited",
    },
    {
        "product": "Nutshell CRM",
        "description": "SMB-focused CRM with pipeline management and email automation. "
                       "Simpler and more affordable than enterprise alternatives.",
        "tier": "never_cited",
    },
]
# ---------- Distribute products across compositions ----------
random.seed(RANDOM_SEED)

high = [p for p in PRODUCTS if p["tier"] == "high"]
low = [p for p in PRODUCTS if p["tier"] == "low"]
never = [p for p in PRODUCTS if p["tier"] == "never_cited"]

print(f"High-tier products:    {[p['product'] for p in high]}")
print(f"Low-tier products:     {[p['product'] for p in low]}")
print(f"Never-tier products:   {[p['product'] for p in never]}")

# Distribute so each composition gets 2 of each tier.
# High tier has 3 products → each product appears in 2 of 3 compositions.
high_assignments = [
    [high[0], high[1]],
    [high[1], high[2]],
    [high[0], high[2]],
]

# For low and never, rotate so different products appear in different compositions
random.shuffle(low)
random.shuffle(never)
low_assignments = [
    [low[0], low[1]],
    [low[1], low[2]],
    [low[0], low[2]],
]
never_assignments = [
    [never[0], never[1]],
    [never[1], never[2]],
    [never[0], never[2]],
]

# ---------- Build compositions ----------
compositions = []

for i in range(N_COMPOSITIONS):
    docs = high_assignments[i] + low_assignments[i] + never_assignments[i]
    assert len(docs) == K, f"S{i+1} has {len(docs)} products, expected {K}"
    base_id = f"S{i+1}"

    print(f"\n=== Base composition {base_id} ===")
    for d in docs:
        print(f"  [{d['tier']:12}] {d['product']}")

    product_names = [d["product"] for d in docs]
    product_tiers = [d["tier"] for d in docs]
    product_descriptions = [d["description"] for d in docs]

    # Base row
    compositions.append({
        "composition_id": f"{base_id}_base",
        "base_composition": base_id,
        "variant_type": "base",
        "removed_doc": None,
        "doc_ids": product_names,
        "doc_descriptions": product_descriptions,          # ← NEW
        "doc_tiers": product_tiers,
    })

    # LOO rows
    for removed in product_names:
        remaining = [x for x in product_names if x != removed]
        remaining_tiers = [t for p, t in zip(product_names, product_tiers) if p != removed]
        remaining_descriptions = [d for p, d in zip(product_names, product_descriptions) if p != removed]  # ← NEW
        safe_removed = removed.replace(" ", "_")
        compositions.append({
            "composition_id": f"{base_id}_minus_{safe_removed}",
            "base_composition": base_id,
            "variant_type": "leave_one_out",
            "removed_doc": removed,
            "doc_ids": remaining,
            "doc_descriptions": remaining_descriptions,    # ← NEW
            "doc_tiers": remaining_tiers,
        })

    

# ---------- Save ----------
OUT_PATH.write_text(json.dumps(compositions, indent=2))
print(f"\nWrote {len(compositions)} rows to {OUT_PATH}")
print(f"  {N_COMPOSITIONS} base × (1 + {K} LOO) = {N_COMPOSITIONS * (1 + K)} total")