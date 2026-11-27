# ComplaintLens: Routing and Theme Discovery in Consumer Finance Complaints

Text Mining & Search project, University of Milano-Bicocca, A.Y. 2026/27
Team: Abdul Sami (data, pre-processing, Task 1), Fatemeh Baghchei (Task 2) · Instructors: Prof. Marco Viviani, Stefano Molari

Using real complaints from the US Consumer Financial Protection Bureau (CFPB), we address two text-mining tasks:

1. **Complaint routing (text classification):** assign each complaint to its financial product.
2. **Complaint themes (topic modeling):** discover what customers complain about.

For each task we compare a **sparse baseline** with a **dense, embedding-based alternative**, using separate train / validation / test data.

## Files

| File | Content |
|---|---|
| `ComplaintLens.ipynb` | The whole project: data, pre-processing, exploration, both tasks, results, discussion |
| `requirements.txt` | Pinned library versions |
| `data/raw/` | Put the archive CSV files here (not included: download them, see below) |
| `results/` | Created by the notebook: metrics (CSV/JSON) and figures |

## Dataset

- **Source:** CFPB Consumer Complaint Database, <https://www.consumerfinance.gov/data-research/consumer-complaints/>
- **Important:** the CFPB stopped publishing complaint narratives in August 2026, so the current export (`complaints.csv.zip`) and the API no longer contain the text. We use the official **narratives archive** (complaints received up to 14 August 2026): <https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/>
- **Files used (2023–2024):** `CCDB_Export_4` (Nov 2022–Aug 2023) to `CCDB_Export_8` (Nov–Dec 2024). Download the five ZIPs, extract them and put the CSV files in `data/raw/`. Do not open them in Excel (it truncates at 1,048,576 rows).
- **Use:** public data from a US government agency. Narratives are published with consumer consent, and personal information is masked (`XXXX`).
- **Subset used:** complaints with a narrative, 2023–2024, at least 20 words, from the 5 most frequent products; 4,000 per product; exact duplicates removed; stratified 70/15/15 split (seed 42); then near-duplicates of training complaints (TF-IDF cosine ≥ 0.90) removed from validation and test. Everything is defined in the notebook's configuration cell and documented in `results/data/data_card.json`.

## How to run (local machine)

Requires Python 3.10 or newer (we used 3.13 with Anaconda), about 8 GB RAM and about 10 GB free disk for the raw CSV files. A GPU is not needed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab                      # open ComplaintLens.ipynb
```

1. **Installation check:** set `SMOKE_TEST = True` in the configuration cell and *Run All*. It runs on a small **fake** dataset in seconds; never report those numbers.
2. **Real run:** set `SMOKE_TEST = False`, put the archive CSV files in `data/raw/`, and *Run All*.
    - The data step reads the big CSV in two passes: 5–15 minutes the first time, then the splits are reused.
    - Sentence embeddings are computed once (several minutes on a CPU) and cached.
3. After the data step, check the product-names table. If one product appears under two names, add both to `product_merge` and run again.
4. Optional zero-shot test: set `CONFIG["classification"]["zeroshot"]["enabled"] = True` (downloads ~1.6 GB; slow on a CPU).

## Method in brief

- **Pre-processing per model:**
    - all models: remove CFPB redaction masks;
    - TF-IDF and LDA: lowercase, stopwords (negations kept), lemmatization;
    - embeddings: raw cleaned text.
- **Task 1:** TF-IDF + Logistic Regression (baseline) vs sentence embeddings (`all-MiniLM-L6-v2`) + Logistic Regression. C is tuned on validation; the test set is used once. Metrics: macro-F1 (main) with a 95% bootstrap interval, accuracy, confusion matrices, error samples.
- **Task 2:** LDA on word counts (baseline) vs embedding topics: K-Means on sentence embeddings + class-based TF-IDF keywords (the BERTopic idea). The number of topics is chosen by coherence (C_v, or NPMI without gensim) on validation texts; we also report topic diversity and NMI/ARI against products on test.
- **No leakage:** every vocabulary, IDF and model is fitted on train only; exact duplicates are removed before splitting and near-duplicates of training complaints are removed from validation and test (section 1.1); section 4.5 measures their effect on the test score.

## How it meets the course requirements

| Requirement (TM&S Project Instructions 2026/27) | Where in the notebook |
|---|---|
| Two core tasks | Sections 4 (classification) and 5 (topic modeling) |
| Pre-processing only when appropriate, justified per model | Section 2 |
| Several representations, justified | TF-IDF / word counts vs sentence embeddings |
| Baseline vs meaningful alternative | 4.1 vs 4.2; LDA vs embedding topics |
| Appropriate metrics | Macro-F1; coherence, diversity, NMI/ARI |
| Separate training / model selection / evaluation | 70/15/15; C and k chosen on validation; test once |
| Documented subset | Section 1 + `data_card.json` |
| Critical discussion | Section 7 + error files + near-duplicate analysis (1.1, 4.5) |
| Reproducibility, versions | Fixed seed; `requirements.txt`; Section 8 |

## External resources

- **Models:** `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0); optional `facebook/bart-large-mnli` (MIT).
- **Libraries:** scikit-learn, gensim, NLTK, sentence-transformers, transformers, pandas, matplotlib.
- **Methods:** LDA (Blei et al., 2003); Sentence-BERT (Reimers & Gurevych, 2019); c-TF-IDF / BERTopic (Grootendorst, 2022); C_v coherence (Röder et al., 2015).
