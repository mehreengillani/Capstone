# collect_test2_short.py
# Short-item version of Test 2 collection.
# Reads test2_compositions_short.json and sends product names (not documents).
#
# Usage: python collect_test2_short.py
# Runtime: ~15–25 min for 315 calls.
# Rerun safely — skips completed (composition, run, engine) triples.

import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ---------- CONFIG ----------
RUNS_PER_COMPOSITION = 5
OUTPUT_CSV = "test2_short_runs.csv"
FAILURE_CSV = "test2_short_parse_failures.csv"
COMPOSITIONS_FILE = "test2_compositions_short.json"
RANDOM_BASE_SEED = 2000   # different from document test
QUERY_TOPIC = "What are the best CRM software in 2026?"

MODELS = {
    "claude": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    "perplexity": "sonar",
}

# ---------- CLIENTS (lazy-loaded) ----------
_clients = {}

def get_client(engine):
    if engine in _clients:
        return _clients[engine]
    if engine == "claude":
        import anthropic
        _clients[engine] = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
        
    elif engine == "openai":
        import openai
        _clients[engine] = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif engine == "perplexity":
        try:
            from perplexity import Perplexity
        except (ImportError, AttributeError):
            from perplexity import Client as Perplexity
        _clients[engine] = Perplexity(api_key=os.environ["PERPLEXITY_API_KEY"])
    return _clients[engine]

# ---------- PROMPT BUILDER (short-item version) ----------
def build_prompt(product_names, product_descriptions, seed):
    """
    Shuffles products and builds a ranking prompt that includes descriptions.
    Returns (prompt_text, label_to_product_map).
    """
    rng = random.Random(seed)
    indices = list(range(len(product_names)))
    rng.shuffle(indices)

    labels = ["A", "B", "C", "D", "E", "F", "G", "H"][:len(indices)]
    label_to_product = {}
    blocks = []
    for label, idx in zip(labels, indices):
        label_to_product[label] = product_names[idx]
        # ← THE KEY CHANGE: include the description on each line
        blocks.append(f"[Doc {label}] {product_names[idx]} — {product_descriptions[idx]}")

    doc_block = "\n".join(blocks)
    K = len(indices)

    task = (
        f"Below are {K} CRM software products, each labeled [Doc A] through [Doc {labels[-1]}], "
        f"with a brief description of each.\n"
        f"Rank them from most to least suitable for the question: '{QUERY_TOPIC}'.\n"
        f"Reply ONLY in this format, one line per product:\n"
        + "\n".join(f"{i+1}. [Doc {labels[i]}]: one-sentence reason"
                    for i in range(K))
        + "\nRank based only on the descriptions provided."
    )

    prompt = f"{doc_block}\n\n{task}"
    return prompt, label_to_product
    
# ---------- RESPONSE PARSER (identical to doc version) ----------
RANK_LINE_RE = re.compile(
    r"^\s*(\d+)[\.\)]\s*\[?Doc\s+([A-H])\]?",
    re.MULTILINE | re.IGNORECASE,
)

def parse_ranking(text, expected_k):
    if not text:
        return [], False, "empty response"
    matches = RANK_LINE_RE.findall(text)
    if not matches:
        return [], False, "no rank lines matched"
    try:
        matches_sorted = sorted(matches, key=lambda m: int(m[0]))
    except (ValueError, IndexError):
        return [], False, "rank number parsing failed"
    ordered = [label.upper() for _, label in matches_sorted]
    if len(ordered) != expected_k:
        return ordered, False, f"expected {expected_k} ranks, got {len(ordered)}"
    if len(set(ordered)) != len(ordered):
        return ordered, False, "duplicate labels in ranking"
    valid_labels = {chr(ord("A") + i) for i in range(expected_k)}
    if set(ordered) != valid_labels:
        return ordered, False, f"labels don't match expected set: {ordered}"
    return ordered, True, None

# ---------- ENGINE CALLS ----------
def call_claude(prompt):
    c = get_client("claude")
    resp = c.messages.create(
        model=MODELS["claude"], max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens

def call_openai(prompt):
    c = get_client("openai")
    resp = c.chat.completions.create(
        model=MODELS["openai"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    text = resp.choices[0].message.content or ""
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens

def call_perplexity(prompt):
    c = get_client("perplexity")
    resp = c.chat.completions.create(
        model=MODELS["perplexity"],
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return text, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)

ENGINE_FUNCS = {"claude": call_claude, "openai": call_openai, "perplexity": call_perplexity}

# ---------- RESUME ----------
def load_done(output_csv):
    if not os.path.exists(output_csv):
        return set()
    done = set()
    with open(output_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("parse_ok", "").lower() == "true":
                done.add((row["composition_id"], int(row["run"]), row["engine"]))
    return done

def append_row(output_csv, row, fieldnames):
    exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# ---------- MAIN ----------
def main():
    for k in ["CLAUDE_API_KEY", "OPENAI_API_KEY", "PERPLEXITY_API_KEY"]:
        if not os.environ.get(k):
            sys.exit(f"{k} not set")

    comps = json.loads(Path(COMPOSITIONS_FILE).read_text())
    print(f"Loaded {len(comps)} compositions from {COMPOSITIONS_FILE}")
    print(f"Runs per composition: {RUNS_PER_COMPOSITION}")
    print(f"Engines: {list(MODELS)}")
    print(f"Total calls: {len(comps) * RUNS_PER_COMPOSITION * len(MODELS)}\n")

    done = load_done(OUTPUT_CSV)
    if done:
        print(f"Resuming — {len(done)} triples already saved.\n")

    fieldnames = [
        "composition_id", "base_composition", "variant_type", "removed_doc",
        "run", "engine", "doc_order_seed", "n_docs",
        "rank_1_doc_id", "rank_2_doc_id", "rank_3_doc_id",
        "rank_4_doc_id", "rank_5_doc_id", "rank_6_doc_id",
        "parse_ok", "parse_error",
        "prompt_tokens", "completion_tokens",
        "raw_response", "timestamp",
    ]

    total = len(comps) * RUNS_PER_COMPOSITION * len(MODELS)
    counter = 0

    for comp in comps:
        comp_id = comp["composition_id"]
        product_names = comp["doc_ids"]
        product_descriptions = comp["doc_descriptions"]   # ← NEW
        K = len(product_names)

        for run in range(1, RUNS_PER_COMPOSITION + 1):
            seed = RANDOM_BASE_SEED + hash((comp_id, run)) % 10000
            prompt, label_map = build_prompt(product_names, product_descriptions, seed)  # ← 3 args now

            for engine in MODELS:
                counter += 1
                if (comp_id, run, engine) in done:
                    print(f"[{counter}/{total}] {comp_id} run {run} {engine} — SKIP")
                    continue

                print(f"[{counter}/{total}] {comp_id} run {run} {engine}...", end=" ", flush=True)
                try:
                    text, pt, ct = ENGINE_FUNCS[engine](prompt)
                    ordered_labels, ok, err = parse_ranking(text, K)
                    ranked_products = [label_map.get(lab, "") for lab in ordered_labels]
                    ranked_products += [""] * (6 - len(ranked_products))

                    row = {
                        "composition_id": comp_id,
                        "base_composition": comp["base_composition"],
                        "variant_type": comp["variant_type"],
                        "removed_doc": comp["removed_doc"] or "",
                        "run": run,
                        "engine": engine,
                        "doc_order_seed": seed,
                        "n_docs": K,
                        "rank_1_doc_id": ranked_products[0],
                        "rank_2_doc_id": ranked_products[1],
                        "rank_3_doc_id": ranked_products[2],
                        "rank_4_doc_id": ranked_products[3],
                        "rank_5_doc_id": ranked_products[4],
                        "rank_6_doc_id": ranked_products[5],
                        "parse_ok": str(ok),
                        "parse_error": err or "",
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "raw_response": text,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    append_row(OUTPUT_CSV, row, fieldnames)
                    print("OK" if ok else f"PARSE FAIL: {err}")
                    if not ok:
                        fexists = os.path.exists(FAILURE_CSV)
                        with open(FAILURE_CSV, "a", newline="") as ff:
                            fw = csv.DictWriter(ff, fieldnames=fieldnames)
                            if not fexists:
                                fw.writeheader()
                            fw.writerow(row)
                except Exception as e:
                    print(f"API FAIL: {type(e).__name__}: {e}")
                time.sleep(1.0)

    print(f"\n=== Done ===")
    print(f"Rows in {OUTPUT_CSV}: {sum(1 for _ in open(OUTPUT_CSV)) - 1}")

if __name__ == "__main__":
    main()