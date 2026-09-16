
# claude_collect.py
import anthropic
import pandas as pd
import time
import os
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from prompts import PROMPTS

# ============ CONFIG ============
CLAUDE_API_KEY = "" # HIDDEN
OUTPUT_FILE = "claude_citations.csv"

client = anthropic.Anthropic(api_key=CLAUDE_API_KEY.strip())


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


def query_claude(prompt):
    """Query Claude with web search enabled and extract citations."""
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3
        }]
    )
    
    answer = ""
    citations = []
    
    for block in response.content:
        if block.type == "text":
            answer += block.text
            # Guard against None citations
            if hasattr(block, 'citations') and block.citations is not None:
                for cite in block.citations:
                    citations.append({
                        "url": cite.url,
                        "title": getattr(cite, 'title', '')
                    })
    
    return answer, citations


def run_claude_collection(prompts, output_file=OUTPUT_FILE):
    """Collect Claude citations for all prompts."""
    
    # Resume support
    all_rows = []
    if os.path.exists(output_file):
        try:
            existing = pd.read_csv(output_file)
            if len(existing) > 0:
                done_prompts = set(existing["prompt"].unique())
                print(f"Resuming — {len(done_prompts)} prompts already done")
                all_rows = existing.to_dict("records")
                prompts = [p for p in prompts if p not in done_prompts]
            else:
                print("Existing file empty — starting fresh")
        except pd.errors.EmptyDataError:
            print("Existing file corrupt — starting fresh")
    
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt}")
        
        try:
            answer, citations = query_claude(prompt)
            cleaned = [clean_url(c["url"]) for c in citations]
            
            for url in cleaned:
                all_rows.append({
                    "prompt": prompt,
                    "engine": "claude",
                    "cited_url": url,
                    "domain": urlparse(url).netloc,
                })
            
            print(f"  -> {len(cleaned)} citations")
        
        except Exception as e:
            print(f"  ERROR: {e}")
        
        # Save after every prompt
        if all_rows:
            pd.DataFrame(all_rows).to_csv(output_file, index=False)
        
        # Claude rate limit — be polite
        time.sleep(3)
    
    # Final save
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_file, index=False)
        print(f"\n[Done] {len(df)} Claude citations saved to {output_file}")
        return df
    else:
        print("\n[Done] No citations collected")
        return pd.DataFrame()


if __name__ == "__main__":
    run_claude_collection(PROMPTS)