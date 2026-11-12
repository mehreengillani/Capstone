# fetch_urls.py
# End-to-end document fetcher for Test 2.
# Reads sanity_check.csv → builds URL pool for one query → fetches all URLs
# via Playwright → deduplicates → writes docs/{id}.txt + docs/{id}.meta.json
# → prints a summary table at the end.
#
# Usage:  python fetch_urls.py
#
# Runtime: ~5-8 min for ~30 URLs on a fresh run.

import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import trafilatura
from playwright.sync_api import sync_playwright

# ---------- CONFIG ----------
QUERY = "What are the best CRM software in 2026?"
CSV_PATH = "sanity_check.csv"
OUT_DIR = Path("docs")

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ---------- FETCH ----------
def fetch_html(url, timeout_ms=25000):
    """Launch a fresh headless Chromium per URL. Slow but reliable against WAFs."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=BROWSER_UA)
        try:
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            html = page.content()
        finally:
            browser.close()
    return html


def fetch_and_extract(url):
    """Return (text, error). text is None on failure."""
    try:
        html = fetch_html(url)
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if not text or len(text.split()) < 100:
            return None, f"extraction yielded {(len(text.split()) if text else 0)} words"
        return text, None
    except Exception as e:
        return None, str(e)


# ---------- POOL ----------
def build_pool(query, csv_path):
    """Union of every URL that any engine surfaced OR cited for this query."""
    df = pd.read_csv(csv_path)
    pool = (
        df[(df.prompt == query) & (df.source_type.isin(["retrieval", "citation"]))]
        .groupby("url")
        .agg(
            n_retrieval=("source_type", lambda s: (s == "retrieval").sum()),
            n_citation=("source_type", lambda s: (s == "citation").sum()),
            engines=("engine", lambda s: sorted(set(s))),
        )
        .reset_index()
    )
    return pool


# ---------- MAIN ----------
def main():
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Query: {QUERY!r}")
    pool = build_pool(QUERY, CSV_PATH)
    print(f"URL pool: {len(pool)} unique URLs "
          f"({(pool.n_citation > 0).sum()} cited, "
          f"{(pool.n_citation == 0).sum()} never cited)")
    print()

    succeeded, failed = [], []
    for i, row in pool.iterrows():
        url = row["url"]
        print(f"[{i+1}/{len(pool)}] {url[:80]}")
        text, err = fetch_and_extract(url)
        if text:
            doc_id = hashlib.md5(url.encode()).hexdigest()[:10]
            (OUT_DIR / f"{doc_id}.txt").write_text(text)
            (OUT_DIR / f"{doc_id}.meta.json").write_text(json.dumps({
                "doc_id": doc_id,
                "url": url,
                "char_count": len(text),
                "word_count": len(text.split()),
                "n_retrieval": int(row["n_retrieval"]),
                "n_citation": int(row["n_citation"]),
                "engines": list(row["engines"]),
            }))
            succeeded.append(url)
            print(f"    OK  ({len(text.split())} words)")
        else:
            failed.append((url, err))
            print(f"    FAIL ({err})")

    # ---------- DEDUP ----------
    print(f"\n=== Raw: {len(succeeded)} succeeded, {len(failed)} failed ===")
    print("\nDeduplicating by content hash...")
    seen_hashes = {}
    removed = 0
    for f in sorted(os.listdir(OUT_DIR)):
        if not f.endswith(".txt"):
            continue
        content = (OUT_DIR / f).read_text()
        h = hashlib.md5(content.encode()).hexdigest()
        if h in seen_hashes:
            print(f"  DUPLICATE: {f} == {seen_hashes[h]} — deleting")
            (OUT_DIR / f).unlink()
            meta_f = f.replace(".txt", ".meta.json")
            if (OUT_DIR / meta_f).exists():
                (OUT_DIR / meta_f).unlink()
            removed += 1
        else:
            seen_hashes[h] = f
    print(f"Dedup: {len(seen_hashes)} unique documents ({removed} removed)")

    # ---------- SUMMARY ----------
    metas = [json.loads((OUT_DIR / f).read_text())
             for f in os.listdir(OUT_DIR) if f.endswith(".meta.json")]
    if not metas:
        print("\nNo documents retrieved. Pool is empty.")
        return

    mdf = pd.DataFrame(metas)
    mdf["tier"] = pd.cut(
        mdf["n_citation"], bins=[-1, 0, 3, 100],
        labels=["never_cited", "low", "high"],
    )

    print("\n=== FINAL POOL ===")
    print(f"Total unique docs: {len(mdf)}")
    print(f"\nWord counts:\n{mdf['word_count'].describe().round(0)}")
    print(f"\nQuality tier distribution:\n{mdf['tier'].value_counts()}")
    print(f"\nFull list:")
    print(mdf[["url", "word_count", "n_citation", "tier"]]
          .sort_values(["tier", "n_citation"], ascending=[True, False])
          .to_string(index=False))


if __name__ == "__main__":
    main()