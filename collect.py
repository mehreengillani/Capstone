# collect.py
import pandas as pd
import time
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

# Local imports
from engines import query_openai, query_perplexity
from prompts import PROMPTS


def clean_url(url):
    """Remove tracking parameters from URLs."""
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query, keep_blank_values=True)
        for p in ['utm_source', 'utm_medium', 'utm_campaign',
                  'utm_term', 'utm_content', 'utm_id']:
            params.pop(p, None)
        return urlunparse(parsed._replace(query=urlencode(params, doseq=True)))
    except Exception:
        return url


# Only OpenAI and Perplexity (Gemini disabled)
ENGINES = {
    "openai": query_openai,
    "perplexity": query_perplexity,
}


def run_collection(prompts, engines, output_file="all_citations.csv"):
    """Collect citations from all engines for all prompts."""
    import os
    
    # Resume support
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        done_prompts = set(existing["prompt"].unique())
        print(f"Resuming — {len(done_prompts)} prompts already done")
        all_rows = existing.to_dict("records")
        prompts = [p for p in prompts if p not in done_prompts]
    else:
        all_rows = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt}")
        
        for engine_name, query_fn in engines.items():
            try:
                print(f"  -> {engine_name}...", end=" ", flush=True)
                answer, citations = query_fn(prompt)
                
                cleaned = [clean_url(u) for u in citations]
                
                for url in cleaned:
                    all_rows.append({
                        "prompt": prompt,
                        "engine": engine_name,
                        "cited_url": url,
                        "domain": urlparse(url).netloc,
                    })
                
                print(f"{len(cleaned)} citations")
                
                # Longer delay between engines to respect rate limits
                time.sleep(3)
            
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        # Save every prompt (safer)
        pd.DataFrame(all_rows).to_csv(output_file, index=False)
        
        # Small delay between prompts
        time.sleep(2)
    
    df = pd.DataFrame(all_rows)
    df.to_csv(output_file, index=False)
    print(f"\n[Done] {len(df)} total citations saved to {output_file}")
    return df


if __name__ == "__main__":
    df = run_collection(PROMPTS, ENGINES)