# analyze_citation_source.py
import pandas as pd
from urllib.parse import urlparse

# ============ LOAD DATA ============
reasoning = pd.read_csv("perplexity_reasoning_results.csv")
citations = pd.read_csv("all_citations.csv")

# Filter to Perplexity only
perplexity_citations = citations[citations["engine"] == "perplexity"].copy()

print("=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)
print(f"Prompts with reasoning data: {reasoning['prompt'].nunique()}")
print(f"Prompts with Perplexity citations: {perplexity_citations['prompt'].nunique()}")

# ============ FOR EACH PROMPT ============
results = []

for prompt in reasoning["prompt"].unique():
    # Get reasoning subset
    r_subset = reasoning[reasoning["prompt"] == prompt]
    runs = sorted(r_subset["run"].unique())
    num_runs = len(runs)
    
    # Get URLs from each run
    run_urls = {}
    for run in runs:
        run_urls[run] = set(r_subset[r_subset["run"] == run]["retrieved_url"].dropna())
    
    # Stable URLs: in all runs
    if run_urls:
        stable = set.intersection(*run_urls.values())
        all_retrieved = set.union(*run_urls.values())
    else:
        stable = set()
        all_retrieved = set()
    
    # Volatile URLs: in only 1 run
    url_run_counts = r_subset.groupby("retrieved_url")["run"].nunique()
    volatile = set(url_run_counts[url_run_counts == 1].index)
    
    # Cited URLs for this prompt (Perplexity)
    cited = set(perplexity_citations[
        perplexity_citations["prompt"] == prompt
    ]["cited_url"].dropna())
    
    # Cited FROM stable retrieval
    cited_from_stable = cited & stable
    # Cited FROM volatile retrieval
    cited_from_volatile = cited & volatile
    # Cited from retrieval but neither stable nor volatile (in 2 of 3 runs)
    cited_from_middle = cited & (all_retrieved - stable - volatile)
    # Cited but NOT retrieved at all
    cited_not_retrieved = cited - all_retrieved
    
    results.append({
        "prompt": prompt[:60],
        "total_cited": len(cited),
        "cited_from_stable": len(cited_from_stable),
        "cited_from_volatile": len(cited_from_volatile),
        "cited_from_middle": len(cited_from_middle),
        "cited_not_retrieved": len(cited_not_retrieved),
        "pct_cited_stable": len(cited_from_stable) / len(cited) * 100 if cited else 0,
        "pct_cited_volatile": len(cited_from_volatile) / len(cited) * 100 if cited else 0,
    })

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("CITED URLS: STABLE vs VOLATILE RETRIEVAL")
print("=" * 60)
print(df.to_string(index=False))

# ============ OVERALL AGGREGATE ============
print("\n" + "=" * 60)
print("AGGREGATE ACROSS ALL PROMPTS")
print("=" * 60)

total_cited = df["total_cited"].sum()
total_stable = df["cited_from_stable"].sum()
total_volatile = df["cited_from_volatile"].sum()
total_middle = df["cited_from_middle"].sum()
total_not_retrieved = df["cited_not_retrieved"].sum()

print(f"Total cited URLs analyzed: {total_cited}")
print(f"  From STABLE retrieval (in all runs): {total_stable} ({total_stable/total_cited*100:.1f}%)")
print(f"  From VOLATILE retrieval (in 1 run):  {total_volatile} ({total_volatile/total_cited*100:.1f}%)")
print(f"  From MIDDLE retrieval (in 2 runs):   {total_middle} ({total_middle/total_cited*100:.1f}%)")
print(f"  NOT retrieved at all:                {total_not_retrieved} ({total_not_retrieved/total_cited*100:.1f}%)")

# ============ KEY INTERPRETATION ============
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

if total_stable / total_cited > 0.5:
    print("MAJORITY of cited URLs come from STABLE retrievals.")
    print("→ Citations depend on being retrieved consistently.")
    print("→ AEO strategy = become part of the stable retrieval set.")
elif total_volatile / total_cited > 0.5:
    print("MAJORITY of cited URLs come from VOLATILE retrievals.")
    print("→ Citation is largely RANDOM.")
    print("→ AEO strategy = maximize retrieval chances broadly.")
else:
    print("Citations are SPLIT between stable and volatile retrievals.")
    print("→ Both retrieval position and chance play a role.")

# Save
df.to_csv("citation_source_analysis.csv", index=False)
print(f"\n[Saved] citation_source_analysis.csv")