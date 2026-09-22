# analyze_variability.py
import pandas as pd
from urllib.parse import urlparse

df = pd.read_csv("perplexity_reasoning_results.csv")

print("=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Prompts: {df['prompt'].nunique()}")
print(f"Runs per prompt: {df['run'].nunique()}")
print(f"Unique URLs retrieved: {df['retrieved_url'].nunique()}")


# ============ STABILITY PER PROMPT ============
print("\n" + "=" * 60)
print("STABILITY PER PROMPT")
print("=" * 60)

results = []

for prompt in df["prompt"].unique():
    subset = df[df["prompt"] == prompt]
    runs = subset["run"].nunique()
    
    # URLs in each run
    url_sets = []
    for run in sorted(subset["run"].unique()):
        run_urls = set(subset[subset["run"] == run]["retrieved_url"].dropna())
        url_sets.append(run_urls)
    
    # URLs in ALL runs (stable)
    stable = set.intersection(*url_sets) if url_sets else set()
    
    # URLs in ANY run (union)
    all_urls = set.union(*url_sets) if url_sets else set()
    
    # URLs in only 1 run (volatile)
    url_run_counts = subset.groupby("retrieved_url")["run"].nunique()
    volatile = set(url_run_counts[url_run_counts == 1].index)
    
    results.append({
        "prompt": prompt[:60],
        "total_unique_urls": len(all_urls),
        "stable_urls": len(stable),
        "volatile_urls": len(volatile),
        "stability_pct": len(stable) / len(all_urls) * 100 if all_urls else 0,
        "overlap_pct": len(stable) / len(all_urls) * 100 if all_urls else 0,
    })

stability_df = pd.DataFrame(results)
print(stability_df.to_string(index=False))

# Overall averages
print(f"\n--- Overall Averages ---")
print(f"Avg unique URLs per prompt: {stability_df['total_unique_urls'].mean():.1f}")
print(f"Avg stable URLs per prompt: {stability_df['stable_urls'].mean():.1f}")
print(f"Avg volatile URLs per prompt: {stability_df['volatile_urls'].mean():.1f}")
print(f"Avg stability %: {stability_df['stability_pct'].mean():.1f}%")


# ============ WHICH URLS ARE STABLE VS VOLATILE ============
print("\n" + "=" * 60)
print("STABLE URLS (appear in ALL runs for their prompt)")
print("=" * 60)

for prompt in df["prompt"].unique()[:3]:
    subset = df[df["prompt"] == prompt]
    runs = subset["run"].nunique()
    
    url_runs = subset.groupby("retrieved_url")["run"].nunique().sort_values(ascending=False)
    
    print(f"\n--- {prompt[:60]} ---")
    stable = url_runs[url_runs == runs]
    print(f"Stable URLs ({len(stable)}):")
    for url, count in stable.head(10).items():
        print(f"  [{count}/{runs}] {url[:80]}")


# ============ RANK VARIATION ============
print("\n" + "=" * 60)
print("RANK VARIATION FOR STABLE URLS")
print("=" * 60)

# For URLs that appear in all runs, how much does their rank vary?
for prompt in df["prompt"].unique()[:3]:
    subset = df[df["prompt"] == prompt]
    runs = subset["run"].nunique()
    
    url_runs = subset.groupby("retrieved_url")["run"].nunique()
    stable_urls = url_runs[url_runs == runs].index.tolist()
    
    print(f"\n--- {prompt[:60]} ---")
    for url in stable_urls[:5]:
        ranks = subset[subset["retrieved_url"] == url]["rank"].tolist()
        print(f"  Ranks: {ranks} | URL: {url[:70]}")


# ============ RUN-TO-RUN OVERLAP MATRIX ============
print("\n" + "=" * 60)
print("RUN-TO-RUN OVERLAP (Jaccard similarity)")
print("=" * 60)

from itertools import combinations

for prompt in df["prompt"].unique()[:2]:
    subset = df[df["prompt"] == prompt]
    
    print(f"\n--- {prompt[:60]} ---")
    
    run_urls = {}
    for run in sorted(subset["run"].unique()):
        run_urls[run] = set(subset[subset["run"] == run]["retrieved_url"].dropna())
    
    for r1, r2 in combinations(sorted(run_urls.keys()), 2):
        set1, set2 = run_urls[r1], run_urls[r2]
        if set1 or set2:
            jaccard = len(set1 & set2) / len(set1 | set2)
            print(f"  Run {r1} vs Run {r2}: {jaccard:.2%} overlap ({len(set1 & set2)} shared URLs)")


# ============ REASONING STEP VARIATION ============
print("\n" + "=" * 60)
print("REASONING STEP VARIATION ACROSS RUNS")
print("=" * 60)

for prompt in df["prompt"].unique():
    subset = df[df["prompt"] == prompt]
    steps_per_run = subset.groupby("run")["num_reasoning_steps"].first()
    print(f"  {prompt[:50]}: {steps_per_run.tolist()}")


# ============ SAVE SUMMARY ============
stability_df.to_csv("variability_summary.csv", index=False)
print(f"\n[Saved] variability_summary.csv")