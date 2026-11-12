# test2_iia_permutation.py
# Permutation test for IIA.
#
# For each (composition, engine), the base and LOO rankings are pooled
# and re-split into two halves. PL is re-fit on each half. The mean δ²
# of the permuted split is compared to the observed mean δ² (actual
# base/LOO split).
#
# Under IIA, the observed split should be indistinguishable from the
# permuted splits — same statistic distribution. Under violation, the
# observed δ² should be larger than the permuted null.
#
# Output: test2_iia_permutation_results.csv

import json
import numpy as np
import pandas as pd
from choix import ilsr_pairwise

# ---------- Configuration ----------
N_PERMUTATIONS = 2000   # 2000 for stable p-values
RANDOM_SEED = 42
TASKS = {
    "long":  {"runs_csv": "test2_runs.csv",       "comps_json": "test2_compositions.json"},
    "short": {"runs_csv": "test2_short_runs.csv", "comps_json": "test2_compositions_short.json"},
}

# ---------- PL fitting helper ----------
def fit_pl(rankings):
    """Fit Plackett-Luce to a list of ordered rankings. Returns log_worth dict."""
    if len(rankings) < 2:
        return None
    all_items = sorted({item for r in rankings for item in r})
    if len(all_items) < 2:
        return None
    idx = {item: i for i, item in enumerate(all_items)}
    try:
        params = ilsr_pairwise(
            n_items=len(all_items),
            data=[(idx[r[i]], idx[r[i+1]]) for r in rankings for i in range(len(r)-1)],
            alpha=0.01,
        )
        return dict(zip(all_items, params))
    except Exception:
        return None

# ---------- Extract rankings from runs CSV ----------
def get_rankings(runs_df, composition_id, engine):
    """Return list of ordered item lists for a given composition + engine."""
    sub = runs_df[(runs_df["composition_id"] == composition_id)
                  & (runs_df["engine"] == engine)
                  & (runs_df["parse_ok"].astype(str).str.lower() == "true")]
    rankings = []
    for _, row in sub.iterrows():
        r = [row[f"rank_{i}_doc_id"] for i in range(1, int(row["n_docs"]) + 1)]
        r = [x for x in r if isinstance(x, str) and x]
        if r:
            rankings.append(r)
    return rankings

# ---------- Compute mean δ² for a pair of ranking sets ----------
def mean_delta_squared(rankings_a, rankings_b, common_items):
    """Fit PL on both sets, compute mean δ² over pairs of common items."""
    if not rankings_a or not rankings_b or len(common_items) < 2:
        return None
    w_a = fit_pl(rankings_a)
    w_b = fit_pl(rankings_b)
    if w_a is None or w_b is None:
        return None
    common = [i for i in common_items if i in w_a and i in w_b]
    if len(common) < 2:
        return None
    deltas = []
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            di, dj = common[i], common[j]
            ratio_a = w_a[di] - w_a[dj]
            ratio_b = w_b[di] - w_b[dj]
            deltas.append((ratio_a - ratio_b) ** 2)
    return float(np.mean(deltas))

# ---------- Main permutation loop for one task ----------
def run_permutation_test(task_name, runs_csv, comps_json, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    print(f"\n{'='*70}\nPERMUTATION TEST — TASK: {task_name.upper()}  (n_perm={n_perm})\n{'='*70}\n")
    rng = np.random.default_rng(seed)

    runs = pd.read_csv(runs_csv)
    comps = {c["composition_id"]: c for c in json.loads(open(comps_json).read())}

    base_ids = [cid for cid, c in comps.items() if c["variant_type"] == "base"]
    loo_by_base = {bid: [] for bid in base_ids}
    for cid, c in comps.items():
        if c["variant_type"] == "leave_one_out":
            base_label = c["base_composition"]
            for bid in base_ids:
                if comps[bid]["base_composition"] == base_label:
                    loo_by_base[bid].append(cid)
                    break

    engines = sorted(runs["engine"].unique())
    results = []

    for engine in engines:                     # FIX: was indented 8, now 4
        print(f"  Engine: {engine}")
        observed_deltas = []
        null_by_comparison = []

        for base_id in base_ids:
            base_rankings = get_rankings(runs, base_id, engine)
            if len(base_rankings) < 2:
                continue

            for loo_id in loo_by_base[base_id]:
                loo_rankings = get_rankings(runs, loo_id, engine)
                if len(loo_rankings) < 2:
                    continue

                base_items = set.intersection(*[set(r) for r in base_rankings])
                loo_items  = set.intersection(*[set(r) for r in loo_rankings])
                common = sorted(base_items & loo_items)
                if len(common) < 2:
                    continue

                # induce base to common
                base_ind = [[d for d in r if d in common] for r in base_rankings]

                # observed
                obs = mean_delta_squared(base_ind, loo_rankings, common)
                if obs is None:
                    continue

                # pool + permute
                pooled = base_ind + loo_rankings
                n_a, n_total = len(base_ind), len(pooled)
                null_for_this_comp = []
                for _ in range(n_perm):
                    perm_idx = rng.permutation(n_total)
                    perm_a = [pooled[k] for k in perm_idx[:n_a]]
                    perm_b = [pooled[k] for k in perm_idx[n_a:]]
                    if len(perm_a) < 2 or len(perm_b) < 2:
                        continue
                    perm_stat = mean_delta_squared(perm_a, perm_b, common)
                    if perm_stat is not None:
                        null_for_this_comp.append(perm_stat)

                # append both, together
                if len(null_for_this_comp) > 0:
                    observed_deltas.append(obs)
                    null_by_comparison.append(np.array(null_for_this_comp[:n_perm]))

        # assert once, after all comparisons
        assert len(observed_deltas) == len(null_by_comparison), \
            f"comparison sets misaligned: {len(observed_deltas)} vs {len(null_by_comparison)}"

        if not observed_deltas or not null_by_comparison:
            print(f"    Insufficient data — skipping")
            continue

        # Aggregate nulls across comparisons
        null_stack = np.vstack(null_by_comparison)
        null_mean = null_stack.mean(axis=0)

        obs_mean = float(np.mean(observed_deltas))
        p_value = (np.sum(null_mean >= obs_mean) + 1) / (len(null_mean) + 1)

        print(f"    Observed mean δ² : {obs_mean:.3f}")
        print(f"    Null mean δ²     : {np.mean(null_mean):.3f}")
        print(f"    Null 95th pct    : {np.percentile(null_mean, 95):.3f}")
        print(f"    Permutation p    : {p_value:.4f}")
        print(f"    Verdict (α=0.05)      : {'Violation detected' if p_value < 0.05 else 'Violation NOT detected'}")
        print(f"    Verdict (Bonferroni)  : {'Violation detected' if p_value < (0.05/6) else 'Violation NOT detected'}")
        print()

        results.append({
            "task": task_name,
            "engine": engine,
            "n_observed_comparisons": len(observed_deltas),
            "observed_mean_delta_sq": round(obs_mean, 4),
            "null_mean_delta_sq": round(float(np.mean(null_mean)), 4),
            "null_95pct": round(float(np.percentile(null_mean, 95)), 4),
            "null_max": round(float(np.max(null_mean)), 4),
            "p_value": round(p_value, 4),
            "verdict_alpha05": "Violation detected" if p_value < 0.05 else "Violation NOT detected",
            "verdict_bonferroni": "Violation detected" if p_value < (0.05/6) else "Violation NOT detected",
        })

    return results

# ---------- Run both tasks ----------
all_results = []
for task_name, cfg in TASKS.items():
    results = run_permutation_test(task_name, cfg["runs_csv"], cfg["comps_json"])
    all_results.extend(results)

results_df = pd.DataFrame(all_results)
results_df.to_csv("test2_iia_permutation_results.csv", index=False)

print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(results_df.to_string(index=False))
print("\nSaved to test2_iia_permutation_results.csv")