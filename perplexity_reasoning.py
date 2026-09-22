# perplexity_reasoning.py
import json
import time
import os
import pandas as pd
from perplexity import Perplexity
from prompts import PROMPTS

# Set your API key
Perplexity_key=""
client = Perplexity(api_key=Perplexity_key)

OUTPUT_FILE = "perplexity_reasoning_results.csv"
RUNS_PER_PROMPT = 3  # How many times to repeat each prompt


def query_perplexity_with_reasoning(prompt):
    """Query Perplexity with reasoning steps and search results."""
    response = client.chat.completions.create(
        model="sonar-pro",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_mode="concise",
        web_search_options={"search_type": "pro"}
    )
    
    full_content = ""
    reasoning_steps = []
    search_results = []
    fetched_urls = []  # ← NEW: store fetch_url_content URLs
    
    for chunk in response:
        chunk_type = getattr(chunk, "object", None)
        
        if chunk_type == "chat.reasoning":
            step = chunk.choices[0].delta.reasoning_steps
            if step:
                reasoning_steps.append(step)
                
                # Check the step type
                step_type = step.get("type") if isinstance(step, dict) else getattr(step, "type", None)
                
                # Log fetch operations
                if step_type == "fetch_url_content":
                    fetch_data = step.get("fetch_url_content", {}) if isinstance(step, dict) else getattr(step, "fetch_url_content", None)
                    if fetch_data:
                        url = fetch_data.get("url") if isinstance(fetch_data, dict) else getattr(fetch_data, "url", None)
                        if url:
                            fetched_urls.append(url)
                            print(f"  [FETCH] {url}")
                
                # Log search operations
                elif step_type == "web_search":
                    ws = step.get("web_search", {}) if isinstance(step, dict) else getattr(step, "web_search", None)
                    if ws:
                        kws = ws.get("search_keywords", []) if isinstance(ws, dict) else getattr(ws, "search_keywords", [])
                        print(f"  [SEARCH] {kws}")
        
        elif chunk_type == "chat.reasoning.done":
            pass
        
        elif chunk_type == "chat.completion.chunk":
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
        
        elif chunk_type == "chat.completion.done":
            if hasattr(chunk, 'search_results') and chunk.search_results:
                search_results.extend(chunk.search_results)
    
    return full_content, reasoning_steps, search_results, fetched_urls  # ← Return 4 items

def run_collection(prompts, runs_per_prompt=3, output_file="perplexity_reasoning_results_v2.csv"):
    """Run each prompt and save BOTH search results and fetch operations."""
    
    all_rows = []
    if os.path.exists(output_file):
        try:
            existing = pd.read_csv(output_file)
            if len(existing) > 0:
                done = set(zip(existing["prompt"], existing["run"]))
                print(f"Resuming — {len(done)} (prompt, run) pairs already done")
                all_rows = existing.to_dict("records")
            else:
                done = set()
        except pd.errors.EmptyDataError:
            done = set()
    else:
        done = set()
    
    for i, prompt in enumerate(prompts):
        for run in range(1, runs_per_prompt + 1):
            if (prompt, run) in done:
                continue
            
            print(f"[{i+1}/{len(prompts)}] Run {run}/{runs_per_prompt}: {prompt[:60]}...")
            
            try:
                answer, steps, results, fetched = query_perplexity_with_reasoning(prompt)
                
                print(f"  → {len(steps)} reasoning steps, {len(results)} search results, {len(fetched)} fetches")
                
                # Save each retrieved URL (search results)
                if results:
                    for rank, result in enumerate(results, start=1):
                        all_rows.append({
                            "prompt": prompt,
                            "run": run,
                            "source_type": "search_result",
                            "rank": rank,
                            "retrieved_url": getattr(result, "url", None),
                            "retrieved_title": getattr(result, "title", ""),
                            "retrieved_snippet": (getattr(result, "snippet", "") or "")[:200],
                            "answer_preview": answer[:300],
                            "num_reasoning_steps": len(steps),
                        })
                
                # Save each fetched URL
                for fetch_url in fetched:
                    all_rows.append({
                        "prompt": prompt,
                        "run": run,
                        "source_type": "fetch_url_content",
                        "rank": None,
                        "retrieved_url": fetch_url,
                        "retrieved_title": None,
                        "retrieved_snippet": None,
                        "answer_preview": answer[:300],
                        "num_reasoning_steps": len(steps),
                    })
                
                # If nothing at all, still save a row
                if not results and not fetched:
                    all_rows.append({
                        "prompt": prompt,
                        "run": run,
                        "source_type": "none",
                        "rank": None,
                        "retrieved_url": None,
                        "retrieved_title": None,
                        "retrieved_snippet": None,
                        "answer_preview": answer[:300],
                        "num_reasoning_steps": len(steps),
                    })
                
                pd.DataFrame(all_rows).to_csv(output_file, index=False)
                time.sleep(3)
            
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
    
    df = pd.DataFrame(all_rows)
    df.to_csv(output_file, index=False)
    print(f"\n[Done] {len(df)} rows saved to {output_file}")
    return df


if __name__ == "__main__":
    prompts_subset = PROMPTS[:5]
    df = run_collection(prompts_subset, runs_per_prompt=3)