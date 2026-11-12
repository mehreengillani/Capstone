# collect_transitivity.py
# Direct pairwise transitivity test.
# For each composition, asks the engine to compare every pair of items directly.
# Output: test2_transitivity_runs.csv
#
# Usage: python collect_transitivity.py [TASK]
#   TASK = "long" or "short" (default: run both)
#
# Runtime: ~30 min per task × 3 engines × 675 prompts
# Cost: ~$5 per task

import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ---------- CONFIG ----------
OUTPUT_CSV = "test2_transitivity_runs_short.csv"
FAILURE_CSV = "test2_transitivity_parse_failures.csv"
DOCS_DIR = Path("docs_trimmed")

QUERY_TOPIC = "What are the best CRM software in 2026?"

MODELS = {
    "claude": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    "perplexity": "sonar",
}

# ---------- CLIENTS ----------
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


# ---------- LOAD CONTENT PER TASK ----------
def load_item_content(task, doc_id):
    """Return the "item content" to show in the prompt."""
    if task == "long":
        # 400-word trimmed document
        return (DOCS_DIR / f"{doc_id}.txt").read_text()
    elif task == "short":
        # product name + description, from compositions JSON
        comps = json.loads(Path("test2_compositions_short.json").read_text())
        for c in comps:
            if doc_id in c["doc_ids"]:
                idx = c["doc_ids"].index(doc_id)
                return f"{c['doc_ids'][idx]} — {c['doc_descriptions'][idx]}"
        return doc_id  # fallback
    else:
        raise ValueError(f"Unknown task: {task}")


# ---------- PROMPT BUILDER ----------
def build_pair_prompt(item_a_id, item_a_content, item_b_id, item_b_content,
                      task, flip):
    """
    Build a pairwise comparison prompt.
    flip: if True, swap A and B order (for position randomization).
    Returns (prompt_text, expected_correct_answer_if_winner_is_a)
    """
    if flip:
        item_a_id, item_b_id = item_b_id, item_a_id
        item_a_content, item_b_content = item_b_content, item_a_content

    prompt = (
        f"You are comparing two items.\n\n"
        f"[Doc A]\n{item_a_content}\n\n"
        f"[Doc B]\n{item_b_content}\n\n"
        f"Question: Which item is MORE suitable for the question "
        f"'{QUERY_TOPIC}'?\n\n"
        f"Reply with exactly one word:\n"
        f"- A (if [Doc A] is more suitable)\n"
        f"- B (if [Doc B] is more suitable)\n"
        f"- TIE (if equally suitable)\n\n"
        f"Your answer (single letter):"
    )
    return prompt, item_a_id, item_b_id


# ---------- PARSER ----------
ANSWER_RE = re.compile(r"\b(A|B|TIE)\b", re.IGNORECASE)

def parse_answer(text):
    """Extract A/B/TIE from the response. Returns (answer, ok, err)."""
    if not text:
        return None, False, "empty response"
    # Look for explicit A/B/TIE in the response
    m = ANSWER_RE.search(text)
    if not m:
        return None, False, f"no A/B/TIE found in: {text[:100]}"
    return m.group(1).upper(), True, None


# ---------- ENGINE CALLS ----------
def call_engine(engine, prompt):
    if engine == "claude":
        c = get_client("claude")
        resp = c.messages.create(
            model=MODELS["claude"], max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens
    elif engine == "openai":
        c = get_client("openai")
        resp = c.chat.completions.create(
            model=MODELS["openai"], max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or "", resp.usage.prompt_tokens, resp.usage.completion_tokens
    elif engine == "perplexity":
        c = get_client("perplexity")
        resp = c.chat.completions.create(
            model=MODELS["perplexity"],
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(resp, "usage", None)
        return (resp.choices[0].message.content or "",
                (usage.prompt_tokens if usage else 0),
                (usage.completion_tokens if usage else 0))


# ---------- RESUME ----------
def load_done(output_csv):
    if not os.path.exists(output_csv):
        return set()
    done = set()
    with open(output_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("parse_ok", "").lower() == "true":
                done.add((row["task"], row["composition_id"], row["engine"],
                          row["doc_a_id"], row["doc_b_id"]))
    return done


# ---------- MAIN ----------
def main(task):
    """Run transitivity collection for one task ("long" or "short")."""
    print(f"\n{'='*60}\nTASK: {task.upper()}\n{'='*60}\n")

    if task == "long":
        comps_file = "test2_compositions.json"
    else:
        comps_file = "test2_compositions_short.json"
    comps = json.loads(Path(comps_file).read_text())

    done = load_done(OUTPUT_CSV)
    fieldnames = ["task", "composition_id", "engine", "doc_a_id", "doc_b_id",
                  "flip", "raw_answer", "answer", "parse_ok", "parse_error",
                  "prompt_tokens", "completion_tokens", "raw_response", "timestamp"]

    total = 0
    for comp in comps:
        doc_ids = comp["doc_ids"]
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                for engine in MODELS:
                    total += 1
    print(f"Total pair prompts to run: {total}")

    counter = 0
    for comp in comps:
        comp_id = comp["composition_id"]
        doc_ids = comp["doc_ids"]

        # Pre-load content for each doc in this composition
        content = {did: load_item_content(task, did) for did in doc_ids}

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                doc_a_id = doc_ids[i]
                doc_b_id = doc_ids[j]
                rng = random.Random(hash((comp_id, doc_a_id, doc_b_id)))
                flip = rng.random() < 0.5

                prompt, actual_a_id, actual_b_id = build_pair_prompt(
                    doc_a_id, content[doc_a_id],
                    doc_b_id, content[doc_b_id],
                    task, flip
                )

                for engine in MODELS:
                    counter += 1
                    key = (task, comp_id, engine, doc_a_id, doc_b_id)
                    if key in done:
                        print(f"[{counter}/{total}] {comp_id} {doc_a_id} vs {doc_b_id} {engine} — SKIP")
                        continue

                    print(f"[{counter}/{total}] {comp_id} {engine} pair {i+1}/{j+1}...", end=" ", flush=True)
                    try:
                        text, pt, ct = call_engine(engine, prompt)
                        answer, ok, err = parse_answer(text)

                        # Reconstruct semantic answer (undo flip)
                        if answer == "TIE":
                            semantic = "TIE"
                        elif flip:
                            # If flipped, "A" means original B won
                            semantic = doc_b_id if answer == "A" else doc_a_id if answer == "B" else None
                        else:
                            semantic = doc_a_id if answer == "A" else doc_b_id if answer == "B" else None

                        row = {
                            "task": task,
                            "composition_id": comp_id,
                            "engine": engine,
                            "doc_a_id": doc_a_id,
                            "doc_b_id": doc_b_id,
                            "flip": flip,
                            "raw_answer": answer or "",
                            "answer": semantic or "",
                            "parse_ok": str(ok),
                            "parse_error": err or "",
                            "prompt_tokens": pt,
                            "completion_tokens": ct,
                            "raw_response": text,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        exists = os.path.exists(OUTPUT_CSV)
                        with open(OUTPUT_CSV, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            if not exists:
                                writer.writeheader()
                            writer.writerow(row)
                        print("OK" if ok else f"FAIL: {err}")
                    except Exception as e:
                        print(f"API FAIL: {type(e).__name__}: {e}")
                    time.sleep(0.5)

    print(f"\n=== Done ({task}) ===")


if __name__ == "__main__":
    tasks = sys.argv[1:] if len(sys.argv) > 1 else ["long", "short"]
    for task in tasks:
        main(task)