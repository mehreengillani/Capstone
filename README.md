# Measuring What Drives Citation in AI Search Engines

**Author:** Mehreen Ali Gillani  
**Supervisor:** Professor George Hagstrom  

## Overview
This repository contains the data collection, extraction, and statistical analysis pipelines for the Capstone project. The project investigates the stability of AI search engines (Perplexity, Claude, and OpenAI) and tests whether their document rankings obey choice axioms (Independence of Irrelevant Alternatives and Transitivity).

---

## 📂 Workflow & Scripts

The project is divided into two main tests. The scripts should be run in the order listed below.

### Phase 1: Test 1 — Live-Search Stability
*These scripts collect and clean the live-web retrieval and citation data.*

- **`data_extract.py`**: The main extraction script. Queries all three engines with 115 prompts, runs each 5 times, and extracts the answer text, retrieval URLs, and citation URLs. Outputs `sanity_check.csv`.
- **`test1_updated.ipynb`**: The Jupyter notebook for Test 1. Computes Jaccard similarity and Rank-Biased Overlap (RBO) to measure cross-run stability.

### Phase 2: Test 2 — Controlled Document Queues
*These scripts build the controlled document environments to test IIA and Transitivity.*

- **`fetch_urls.py`**: Fetches the 48 web pages used for the long-item task via Playwright and extracts the text using `trafilatura`.
- **`build_compositions.py`**: Builds the long-document compositions (3 bases, 18 leave-one-out variants) and saves them to `test2_compositions.json`.
- **`build_compositions_short.py`**: Builds the short-item (product description) compositions and saves them to `test2_compositions_short.json`.
- **`collect_test2.py`**: Collects the 315 rankings for the long-document task.
- **`collect_test2_short.py`**: Collects the 315 rankings for the short-item task.

### Phase 3: IIA Analysis
*These scripts test Independence of Irrelevant Alternatives.*

- **`fit_plackett_luce.py`**: Fits the Plackett-Luce model (Bradley-Terry approximation via `choix`) to estimate the "worth" of every document.
- **`fit_plackett_luce_short.py`**: Same as above, for the short-item data.
- **`test2_iia_permutation.py`**: The primary IIA test. Implements an orientation-invariant permutation null (2,000 shuffles) to compare base vs. leave-one-out compositions. Outputs `test2_iia_permutation_results.csv`.

### Phase 4: Transitivity Analysis
*These scripts test whether pairwise preferences are logically consistent.*

- **`collect_transitivity.py`**: Collects single-shot pairwise comparisons for all 225 item pairs per engine.
- **`transitivity_option2.py`**: Re-analyzes the transitivity data to report tie rates and conditional cycle rates, controlling for the fact that different engines abstain (say "TIE") at different rates.

---

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/capstone-ai-search.git
   cd capstone-ai-search