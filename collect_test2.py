# collect_test2.py
# Full Test 2 data collection: 3 engines × 21 compositions × 5 runs.
# Reads test2_compositions.json + docs_trimmed/, calls each engine, parses
# rankings, saves to test2_runs.csv with resume support.
#
# Usage:  python collect_test2.py
# Runtime: ~15-25 minutes for full 315 calls (saves after every call).
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

load_dotenv(dotenv_path=Path(".") / ".env")

# ---------- CONFIG ----------
RUNS_PER_COMPOSITION = 5   # change to 3 for a cheaper pilot
OUTPUT_CSV = "test2_runs.csv"
FAILURE_CSV = "test2_parse_failures.csv"
COMPOSITIONS_FILE = "test2_compositions.json"
DOCS_DIR = Path("docs_trimmed")
RANDOM_BASE_SEED = 1000    # base seed; per-run seed = RANDOM_BASE_SEED + hash

QUERY_TOPIC = "What are the best CRM software in 2026?"

# Model identifiers per engine
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
        CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
        _clients[engine] = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
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


# ---------- PROMPT BUILDER ----------
def build_prompt(doc_texts, doc_ids, seed):
    """
    Shuffles doc order deterministically, then builds the ranking prompt.
    Returns (prompt_text, label_to_doc_id_map, seed_used).
    """
    rng = random.Random(seed)
    indices = list(range(len(doc_texts)))
    rng.shuffle(indices)

    labels = ["A", "B", "C", "D", "E", "F", "G", "H"][:len(indices)]
    label_to_doc = {}
    blocks = []
    for label, idx in zip(labels, indices):
        label_to_doc[label] = doc_ids[idx]
        blocks.append(f"[Doc {label}]\n{doc_texts[idx]}")

    doc_block = "\n\n".join(blocks)
    K = len(indices)

    task = (
        f"Below are {K} documents, each labeled [Doc A] through [Doc {labels[-1]}].\n"
        f"Rank them from most to least relevant to the question: '{QUERY_TOPIC}'.\n"
        f"Reply ONLY in this format, one line per document:\n"
        + "\n".join(f"{i+1}. [Doc {labels[i]}]: one-sentence reason"
                    for i in range(K))
        + "\nDo not use any outside information; rank based only on the documents provided."
    )

    prompt = f"{doc_block}\n\n{task}"
    return prompt, label_to_doc, seed


# ---------- RESPONSE PARSER ----------
# Extracts "1. [Doc X]: ..." lines from a response, returns ordered list of labels.
# Handles flexible formatting — accepts "1. [Doc X]", "1.[Doc X]", "1) [Doc X]", etc.
RANK_LINE_RE = re.compile(
    r"^\s*(\d+)[\.\)]\s*\[?Doc\s+([A-H])\]?",
    re.MULTILINE | re.IGNORECASE,
)

def parse_ranking(text, expected_k):
    """
    Returns (ordered_labels_list, parse_ok, error_msg).
    ordered_labels_list is [label_rank1, label_rank2, ...] where each label is 'A'..'H'.
    """
    if not text:
        return [], False, "empty response"

    matches = RANK_LINE_RE.findall(text)
    if not matches:
        return [], False, "no rank lines matched"

    # Sort by rank number
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
        model=MODELS["claude"],
        max_tokens=1024,
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
    pt = usage.prompt_tokens if usage else 0
    ct = usage.completion_tokens if usage else 0
    return text, pt, ct


ENGINE_FUNCS = {
    "claude": call_claude,
    "openai": call_openai,
    "perplexity": call_perplexity,
}


# ---------- RESUME LOGIC ----------
def load_done(output_csv):
    """Return set of (composition_id, run, engine) already saved."""
    if not os.path.exists(output_csv):
        return set()
    done = set()
    with open(output_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("parse_ok", "").lower() == "true":
                done.add((row["composition_id"], int(row["run"]), row["engine"]))
    return done


def append_row(output_csv, row, fieldnames):
    """Append one row to CSV, creating with headers if needed."""
    exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------- MAIN ----------
def main():
    if not os.environ.get("CLAUDE_API_KEY"):
        sys.exit("CLAUDE_API_KEY not set")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    if not os.environ.get("PERPLEXITY_API_KEY"):
        sys.exit("PERPLEXITY_API_KEY not set")

    comps = json.loads(Path(COMPOSITIONS_FILE).read_text())
    print(f"Loaded {len(comps)} compositions from {COMPOSITIONS_FILE}")
    print(f"Runs per composition: {RUNS_PER_COMPOSITION}")
    print(f"Engines: {list(MODELS)}")
    print(f"Total calls: {len(comps) * RUNS_PER_COMPOSITION * len(MODELS)}")
    print()

    done = load_done(OUTPUT_CSV)
    if done:
        print(f"Resuming — {len(done)} (composition, run, engine) triples "
              f"already saved, will skip.\n")

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
        doc_ids = comp["doc_ids"]
        doc_texts = [(DOCS_DIR / f"{did}.txt").read_text() for did in doc_ids]
        K = len(doc_ids)

        for run in range(1, RUNS_PER_COMPOSITION + 1):
            # Deterministic seed per (composition, run) so reruns use same order
            seed = RANDOM_BASE_SEED + hash((comp_id, run)) % 10000

            prompt, label_map, _ = build_prompt(doc_texts, doc_ids, seed)

            for engine in MODELS:
                counter += 1
                if (comp_id, run, engine) in done:
                    print(f"[{counter}/{total}] {comp_id} run {run} {engine} — SKIP")
                    continue

                print(f"[{counter}/{total}] {comp_id} run {run} {engine}...",
                      end=" ", flush=True)
                try:
                    text, pt, ct = ENGINE_FUNCS[engine](prompt)
                    ordered_labels, ok, err = parse_ranking(text, K)

                    # Map label -> doc_id
                    ranked_doc_ids = [label_map.get(lab, "") for lab in ordered_labels]
                    ranked_doc_ids += [""] * (6 - len(ranked_doc_ids))

                    row = {
                        "composition_id": comp_id,
                        "base_composition": comp["base_composition"],
                        "variant_type": comp["variant_type"],
                        "removed_doc": comp["removed_doc"] or "",
                        "run": run,
                        "engine": engine,
                        "doc_order_seed": seed,
                        "n_docs": K,
                        "rank_1_doc_id": ranked_doc_ids[0],
                        "rank_2_doc_id": ranked_doc_ids[1],
                        "rank_3_doc_id": ranked_doc_ids[2],
                        "rank_4_doc_id": ranked_doc_ids[3],
                        "rank_5_doc_id": ranked_doc_ids[4],
                        "rank_6_doc_id": ranked_doc_ids[5],
                        "parse_ok": str(ok),
                        "parse_error": err or "",
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "raw_response": text,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    append_row(OUTPUT_CSV, row, fieldnames)
                    if ok:
                        print(f"OK")
                    else:
                        print(f"PARSE FAIL: {err}")
                        # Also write to failure file for later inspection
                        fexists = os.path.exists(FAILURE_CSV)
                        with open(FAILURE_CSV, "a", newline="") as ff:
                            fw = csv.DictWriter(ff, fieldnames=fieldnames)
                            if not fexists:
                                fw.writeheader()
                            fw.writerow(row)
                except Exception as e:
                    print(f"API FAIL: {type(e).__name__}: {e}")

                time.sleep(1.0)  # rate-limit safety

    print(f"\n=== Done ===")
    print(f"Rows in {OUTPUT_CSV}: {sum(1 for _ in open(OUTPUT_CSV)) - 1}")
    if os.path.exists(FAILURE_CSV):
        print(f"Parse failures in {FAILURE_CSV}: "
              f"{sum(1 for _ in open(FAILURE_CSV)) - 1}")


if __name__ == "__main__":
    main()