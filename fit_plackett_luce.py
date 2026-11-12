# fit_plackett_luce.py
# Fits Plackett-Luce per (composition_id, engine) from test2_runs.csv.
# Saves per-document worths to test2_pl_worths.csv.

import json
import pandas as pd
import numpy as np
import choix
from collections import defaultdict

# ---------- Load data ----------
df = pd.read_csv("test2_runs.csv")
comps = {c["composition_id"]: c for c in json.loads(open("test2_compositions.json").read())}

# Verify parse success
df = df[df["parse_ok"].astype(str).str.lower() == "true"]
print(f"Usable rows: {len(df)}")

# ---------- Group by (composition, engine) ----------
# Each group has 5 rankings of the same document set.
groups = defaultdict(list)
for _, row in df.iterrows():
    key = (row["composition_id"], row["engine"])
    ranked_doc_ids = [row[f"rank_{i}_doc_id"] for i in range(1, row["n_docs"] + 1)]
    ranked_doc_ids = [x for x in ranked_doc_ids if isinstance(x, str) and x]
    groups[key].append(ranked_doc_ids)

# ---------- Fit PL per group ----------
results = []
for (comp_id, engine), rankings in groups.items():
    # Map doc_ids to local indices 0..K-1
    all_docs = sorted({doc for ranking in rankings for doc in ranking})
    doc_to_idx = {doc: i for i, doc in enumerate(all_docs)}
    idx_rankings = [[doc_to_idx[d] for d in ranking] for ranking in rankings]

    n_items = len(all_docs)
    try:
        # MM algorithm, no regularization
        params = choix.ilsr_pairwise(
            n_items=n_items,
            data=[(ranking[:-1], ranking[1:]) for ranking in idx_rankings],
            alpha=0.01,
        )
        # Convert log-worths to worths
        worths = np.exp(params - params.max())  # softmax-normalize to avoid overflow
        worths = worths / worths.sum()

        for doc_id, idx in doc_to_idx.items():
            results.append({
                "composition_id": comp_id,
                "base_composition": comps[comp_id]["base_composition"],
                "variant_type": comps[comp_id]["variant_type"],
                "removed_doc": comps[comp_id]["removed_doc"] or "",
                "engine": engine,
                "doc_id": doc_id,
                "doc_tier": comps[comp_id]["doc_tiers"][comps[comp_id]["doc_ids"].index(doc_id)]
                            if doc_id in comps[comp_id]["doc_ids"] else "unknown",
                "pl_worth": float(worths[idx]),
                "log_worth": float(params[idx]),
                "n_rankings": len(rankings),
            })
    except Exception as e:
        print(f"FIT FAILED {comp_id} {engine}: {type(e).__name__}: {e}")

# ---------- Save ----------
out = pd.DataFrame(results)
out.to_csv("test2_pl_worths.csv", index=False)
print(f"\nWrote {len(out)} rows to test2_pl_worths.csv")

# ---------- Sanity check: per-engine worth distribution ----------
print(f"\n=== PL worth distribution by engine ===")
print(out.groupby("engine")["pl_worth"].describe().round(3))

print(f"\n=== Rank-1 worth vs rank-2 worth (should be higher) ===")
# For base compositions only
base = out[out["variant_type"] == "base"]
for engine in ["claude", "openai", "perplexity"]:
    subset = base[base["engine"] == engine]
    print(f"  {engine}: {len(subset)} docs across {subset['composition_id'].nunique()} compositions")