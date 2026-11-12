# sanity_check.py
# Full collection run: all 150 prompts x 3 engines x RUNS repeats.
# Results saved to CSV with resume support — since Claude/OpenAI calls cost
# money, every (prompt, engine, run) combo already saved is skipped on rerun.
#
# COST NOTE: 150 prompts x 3 engines x RUNS repeats = total API calls below.
# RUNS=5  -> 2,250 calls   RUNS=8  -> 3,600 calls
# Check your Claude/OpenAI budget before launching at this scale.


import os
import re
import time
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import pandas as pd
from perplexity import Perplexity
from openai import OpenAI
import anthropic
from prompts import PROMPTS


#insert your API keys here
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

perplexity_client = Perplexity(api_key=PERPLEXITY_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

N_PROMPTS = len(PROMPTS) 
RUNS = 5         # repeats per prompt per engine — bump to up to 8 if you want more stability data
OUTPUT = "sanity_check.csv"


def clean_url(u):
    try:
        p = urlparse(u)
        q = parse_qs(p.query, keep_blank_values=True)
        for k in ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'utm_id']:
            q.pop(k, None)
        return urlunparse(p._replace(query=urlencode(q, doseq=True)))
    except Exception:
        return u


def dedup(urls):
    return list(dict.fromkeys(urls))


# ─── PERPLEXITY: retrieval (reasoning-phase search) vs citation (final answer) ───
def run_perplexity(prompt):
    resp = perplexity_client.chat.completions.create(
        model="sonar-pro",
        messages=[{"role": "user", "content": prompt}],
        stream=True, stream_mode="concise",
        web_search_options={"search_type": "pro"},
    )
    retrieval, citations = [], []
    full_content = ""
    for chunk in resp:
        t = getattr(chunk, "object", None)
        if t == "chat.reasoning":
            step = chunk.choices[0].delta.reasoning_steps
            if step and (step.get("type") if isinstance(step, dict) else getattr(step, "type", None)) == "web_search":
                ws = step.get("web_search", {}) if isinstance(step, dict) else getattr(step, "web_search", None)
                for r in (ws.get("search_results", []) if isinstance(ws, dict) else getattr(ws, "search_results", [])) or []:
                    u = r.get("url") if isinstance(r, dict) else getattr(r, "url", None)
                    if u:
                        retrieval.append(u)
        elif t == "chat.reasoning.done" and getattr(chunk, "search_results", None):
            retrieval += [getattr(r, "url", None) for r in chunk.search_results if getattr(r, "url", None)]
        elif t == "chat.completion.chunk":
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
        elif t == "chat.completion.done" and getattr(chunk, "search_results", None):
            citations += [getattr(r, "url", None) for r in chunk.search_results if getattr(r, "url", None)]

    # FIX: Perplexity's chat.completion.done search_results is the full
    # reference pool it had access to, not what it actually cited — the
    # real citations are only the ones referenced by [n] markers in the
    # text. Filter the pool down to those indices (1-indexed into the
    # pool, in the order they first appear in the text).
    deduped_citations = dedup(citations)
    cited_indices = [int(n) for n in re.findall(r'\[(\d+)\]', full_content)]
    seen, ordered_indices = set(), []
    for n in cited_indices:
        if n not in seen:
            seen.add(n)
            ordered_indices.append(n)
    true_citations = [deduped_citations[n - 1] for n in ordered_indices
                       if 1 <= n <= len(deduped_citations)]

    print(f"    [diagnostic] inline citation markers in text: {len(ordered_indices)}  "
          f"vs. full reference pool: {len(deduped_citations)}  "
          f"-> true citations: {len(true_citations)}")

    return dedup(retrieval), true_citations, full_content


# ─── CLAUDE: retrieval (web_search_tool_result) vs citation (text block citations) ───
def run_claude(prompt):
    resp = claude_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
    )
    retrieval, citations = [], []
    answer_text = ""
    for block in resp.content:
        if block.type == "web_search_tool_result":
            for r in (getattr(block, "content", None) or []):
                u = getattr(r, "url", None)
                if u:
                    retrieval.append(u)
        if block.type == "text":
            answer_text += block.text or ""
            if getattr(block, "citations", None):
                for c in block.citations:
                    u = getattr(c, "url", None)
                    if u:
                        citations.append(u)
    return dedup(retrieval), dedup(citations), answer_text


# ─── OPENAI: citation only (web_search_call has no URL list — see note below) ───
def run_openai(prompt):
    resp = openai_client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
        tools=[{"type": "web_search"}],
    )
    citations = []
    answer_text = ""
    for item in resp.output:
        if getattr(item, "type", None) == "message":
            for block in getattr(item, "content", []) or []:
                answer_text += getattr(block, "text", "") or ""
                for a in (getattr(block, "annotations", None) or []):
                    if getattr(a, "type", None) == "url_citation":
                        u = getattr(a, "url", None)
                        if u:
                            citations.append(u)
    return [], dedup(citations), answer_text  # retrieval always empty for OpenAI — API limitation


ENGINES = {
    "perplexity": run_perplexity,
    "claude": run_claude,
    "openai": run_openai,
}


def preview(urls, n=3):
    shown = [clean_url(u) for u in urls[:n]]
    more = f" (+{len(urls) - n} more)" if len(urls) > n else ""
    return ", ".join(shown) + more if shown else "(none)"


def load_done(output):
    """Resume support: (prompt, engine, run) triples already saved, so a
    rerun doesn't re-spend money on a call that already succeeded.

    Handles the older CSV format (from before repeats were added) which has
    no 'run' column — those rows are treated as run=1 so they still count
    as done and aren't re-fetched."""
    if not os.path.exists(output):
        return set(), []
    try:
        existing = pd.read_csv(output)
        if len(existing) == 0:
            return set(), []
        if "run" not in existing.columns:
            existing["run"] = 1
        existing["run"] = existing["run"].fillna(1).astype(int)
        done = set(zip(existing["prompt"], existing["engine"], existing["run"]))
        return done, existing.to_dict("records")
    except pd.errors.EmptyDataError:
        return set(), []


def main():
    prompts = PROMPTS[:N_PROMPTS]
    total_calls = len(prompts) * len(ENGINES) * RUNS
    print(f"Full run: {len(prompts)} prompts x {len(ENGINES)} engines x {RUNS} runs "
          f"= {total_calls} total API calls.")
    print(f"Saving results to {OUTPUT}\n")

    done, rows = load_done(OUTPUT)
    if done:
        print(f"Resuming — {len(done)} (prompt, engine, run) combos already saved, will skip those.\n")
    if rows:
        csv_prompts = list(dict.fromkeys(r["prompt"] for r in rows))
        prefix = prompts[:len(csv_prompts)]
        if prefix != csv_prompts:
            raise RuntimeError(
                f"PROMPTS has drifted from {OUTPUT} "
                f"({len(csv_prompts)} prompts in CSV, first mismatch at "
                f"{next(i for i, (a, b) in enumerate(zip(prefix, csv_prompts)) if a != b)}). "
                f"Fix prompts.py or move/rename the CSV before rerunning."
            )
        print(f"Prompt-order check passed: first {len(csv_prompts)} prompts match the CSV.\n")

    for i, prompt in enumerate(prompts, 1):
        print("=" * 90)
        print(f"[{i}/{len(prompts)}] {prompt}")
        print("=" * 90)

        for engine, fn in ENGINES.items():
            for run in range(1, RUNS + 1):
                if (prompt, engine, run) in done:
                    print(f"  {engine.upper()} run {run}/{RUNS}  (skipped — already in {OUTPUT})")
                    continue
                try:
                    ret, cit, answer_text = fn(prompt)
                    overlap = len(set(clean_url(u) for u in ret) & set(clean_url(u) for u in cit))
                    print(f"\n  {engine.upper()} run {run}/{RUNS}")
                    print(f"    retrieval : {len(ret):>3}  {preview(ret)}")
                    print(f"    citation  : {len(cit):>3}  {preview(cit)}")
                    if ret:
                        print(f"    overlap   : {overlap}/{len(cit)} cited URLs were also retrieved")

                    # One row holds the full generated answer text, separate
                    # from the URL rows below, so it isn't repeated per-URL.
                    rows.append({"prompt": prompt, "engine": engine, "run": run,
                                 "source_type": "answer", "rank": None, "url": None,
                                 "text": answer_text})

                    for rank, u in enumerate(ret, 1):
                        rows.append({"prompt": prompt, "engine": engine, "run": run,
                                     "source_type": "retrieval", "rank": rank, "url": clean_url(u),
                                     "text": None})
                    for rank, u in enumerate(cit, 1):
                        rows.append({"prompt": prompt, "engine": engine, "run": run,
                                     "source_type": "citation", "rank": rank, "url": clean_url(u),
                                     "text": None})
                    if not ret and not cit:
                        rows.append({"prompt": prompt, "engine": engine, "run": run,
                                     "source_type": "none", "rank": None, "url": None, "text": None})

                    # Save after every single call — so a crash, rate limit,
                    # or budget cutoff mid-run doesn't lose calls you already paid for.
                    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
                    time.sleep(1)
                except Exception as e:
                    print(f"\n  {engine.upper()} run {run}/{RUNS}  ERROR: {e}")

        print()

    print("=" * 90)
    print(f"Done — {len(rows)} rows saved to {OUTPUT}")


if __name__ == "__main__":
    main()
