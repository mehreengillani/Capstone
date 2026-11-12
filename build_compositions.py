# build_compositions.py
# Generates Test 2 composition JSON from docs_trimmed/.
# Run once after trimming. Output: test2_compositions.json
#
# Structure:
#   - 3 base compositions, each with 6 documents
#   - Each base is balanced: 2 high-tier + 2 low-tier + 2 never-cited
#   - For each base: 1 base row + 6 leave-one-out rows
#   - Total: 3 × 7 = 21 composition rows
#
# High-tier docs (only 3 exist) are reused across compositions:
#   C1 uses H1, H2
#   C2 uses H2, H3
#   C3 uses H1, H3
# This creates cross-composition overlap for IIA comparison.

import json
import os
import random
from pathlib import Path

import pandas as pd

DOCS_DIR = Path("docs_trimmed")
OUT_PATH = Path("test2_compositions.json")
RANDOM_SEED = 42
N_COMPOSITIONS = 3
K = 6  # docs per composition
DOCS_PER_TIER = 2  # 2 high + 2 low + 2 never_cited = 6

# ---------- Load and tier the docs ----------
metas = [json.loads((DOCS_DIR / f).read_text())
         for f in os.listdir(DOCS_DIR) if f.endswith(".meta.json")]
mdf = pd.DataFrame(metas)
mdf["tier"] = pd.cut(mdf["n_citation"], bins=[-1, 0, 3, 100],
                     labels=["never_cited", "low", "high"])

high = mdf[mdf.tier == "high"].to_dict("records")
low = mdf[mdf.tier == "low"].to_dict("records")
never = mdf[mdf.tier == "never_cited"].to_dict("records")

print(f"Pool: {len(high)} high, {len(low)} low, {len(never)} never_cited")

assert len(high) >= DOCS_PER_TIER, f"Need >= {DOCS_PER_TIER} high docs, have {len(high)}"
assert len(low) >= DOCS_PER_TIER,  f"Need >= {DOCS_PER_TIER} low docs, have {len(low)}"
assert len(never) >= DOCS_PER_TIER, f"Need >= {DOCS_PER_TIER} never docs, have {len(never)}"

random.seed(RANDOM_SEED)

# ---------- Assign high-tier docs to compositions with overlap ----------
# Each high-tier doc appears in exactly 2 of 3 compositions.
# Pattern for 3 high docs (H0, H1, H2):
#   C0: H0, H1
#   C1: H1, H2
#   C2: H0, H2
high_assignment = [
    [high[0], high[1]],
    [high[1], high[2]],
    [high[0], high[2]],
]

# ---------- Sample non-overlapping lows and nevers for each composition ----------
low_pool = low.copy()
never_pool = never.copy()
random.shuffle(low_pool)
random.shuffle(never_pool)

low_assignment = [low_pool[i*DOCS_PER_TIER:(i+1)*DOCS_PER_TIER]
                  for i in range(N_COMPOSITIONS)]
never_assignment = [never_pool[i*DOCS_PER_TIER:(i+1)*DOCS_PER_TIER]
                    for i in range(N_COMPOSITIONS)]

# ---------- Build base compositions ----------
compositions = []
for i in range(N_COMPOSITIONS):
    docs = high_assignment[i] + low_assignment[i] + never_assignment[i]
    assert len(docs) == K, f"C{i} has {len(docs)} docs, expected {K}"
    base_id = f"C{i+1}"
    base_doc_ids = [d["doc_id"] for d in docs]

    print(f"\n=== Base composition {base_id} ===")
    for d in docs:
        print(f"  [{d['tier']:12}] {d['doc_id']}  {d['url'][:70]}")

    # --- Base row (no removal) ---
    compositions.append({
        "composition_id": f"{base_id}_base",
        "base_composition": base_id,
        "variant_type": "base",
        "removed_doc": None,
        "doc_ids": base_doc_ids,
        "doc_urls": [d["url"] for d in docs],
        "doc_tiers": [d["tier"] for d in docs],
    })

    # --- Leave-one-out rows ---
    for removed in base_doc_ids:
        remaining = [x for x in base_doc_ids if x != removed]
        remaining_urls = [d["url"] for d in docs if d["doc_id"] != removed]
        remaining_tiers = [d["tier"] for d in docs if d["doc_id"] != removed]
        compositions.append({
            "composition_id": f"{base_id}_minus_{removed}",
            "base_composition": base_id,
            "variant_type": "leave_one_out",
            "removed_doc": removed,
            "doc_ids": remaining,
            "doc_urls": remaining_urls,
            "doc_tiers": remaining_tiers,
        })

# ---------- Save ----------
OUT_PATH.write_text(json.dumps(compositions, indent=2))
print(f"\nWrote {len(compositions)} composition rows to {OUT_PATH}")
print(f"  {N_COMPOSITIONS} base × (1 base + {K} LOO) = "
      f"{N_COMPOSITIONS * (1 + K)} total")