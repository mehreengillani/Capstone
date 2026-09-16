# serpapi_collect.py
import pandas as pd
import requests
import time
from urllib.parse import urlparse
from prompts import PROMPTS

serp_api_key = ""  #SECRET
SERPAPI_KEY = serp_api_key


def get_google_results(query, num=10):
    """Get top N Google organic results for a query."""
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num,
        "engine": "google",
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("organic_results", []):
            results.append({
                "url": item.get("link"),
                "title": item.get("title"),
                "position": item.get("position"),
            })
        return results
    except Exception as e:
        print(f"  Error: {e}")
        return []


def run_serpapi_collection(prompts, output_file="google_results.csv"):
    """Collect Google top-10 for each prompt."""
    import os
    
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        done = set(existing["prompt"].unique())
        print(f"Resuming — {len(done)} prompts done")
        all_rows = existing.to_dict("records")
        prompts = [p for p in prompts if p not in done]
    else:
        all_rows = []
    
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt}")
        
        results = get_google_results(prompt, num=10)
        
        for r in results:
            if r["url"]:
                all_rows.append({
                    "prompt": prompt,
                    "url": r["url"],
                    "title": r["title"],
                    "google_position": r["position"],
                })
        
        print(f"  Got {len(results)} results")
        pd.DataFrame(all_rows).to_csv(output_file, index=False)
        time.sleep(1)
    
    df = pd.DataFrame(all_rows)
    df.to_csv(output_file, index=False)
    print(f"\n[Done] {len(df)} Google results saved")
    return df


if __name__ == "__main__":
    df = run_serpapi_collection(PROMPTS)

    