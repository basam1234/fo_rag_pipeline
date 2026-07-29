# PolarityIQ Micro-RAG Pipeline

A local-first, cache-backed intelligence pipeline that discovers, enriches, and indexes validated Single Family Offices (SFOs) for retrieval-augmented generation (RAG) with grounded LLM responses.

## Project Overview

PolarityIQ builds a curated dataset of **50 validated Single Family Offices** and exposes it through a semantic chat interface. Every LLM-generated claim is verified against the structured source data before being presented to the user.

**Core loop:** Discovery → Enrichment → ONNX Indexing → Grounded RAG

## Architecture

```
discovery/          # Web scrapers (SEC EDGAR, IAPD, ProPublica)
    └── run_discovery.py, *_fetcher.py, normalize.py

enrichment/         # Enrichment pipeline with LLM extraction
    ├── run_enrichment.py      # Orchestrator (checkpoint, filter, fetch, score, persist)
    ├── llm_extractor.py       # Groq LLM extraction with file-based caching
    ├── ddg_scraper.py         # DuckDuckGo web search
    ├── structured_fetcher.py  # SEC EDGAR / IAPD XML data fetcher
    └── generate_validation_md.py

rag/                # Retrieval-Augmented Generation layer
    ├── build_index.py   # ONNX embedding with fastembed (BGE-small-en-v1.5)
    ├── retrieval.py     # Hybrid search: NLP filter parsing + semantic vector search
    └── generation.py    # Groq LLM grounded generation + RapidFuzz claim verification

streamlit_app.py    # Elite finance terminal UI with pipeline state machine

data/               # Pipeline artifacts
    ├── candidates.csv          # Raw discovery output
    ├── family_offices.csv      # Final validated 50-SFO dataset
    ├── provenance_log.csv      # Field-level provenance tracking
    └── rag_index/              # Persisted embeddings and corpus
        ├── embeddings.npy
        └── rag_corpus.csv
```

### Pipeline Stages

1. **Discovery** — Scrapes SEC EDGAR, IAPD, and ProPublica for potential SFO candidates. Outputs `data/candidates.csv`.

2. **Enrichment** — For each candidate, fetches structured XML data (SEC primary_doc.xml, IAPD Form ADV), runs DuckDuckGo web searches, extracts fields via Groq LLM (`llama-3.1-8b-instant`), merges structured data into LLM results, applies RapidFuzz relevance pre-filtering, and ranks candidates into 3 tiers. The top 50 are written to `data/family_offices.csv`.

3. **ONNX Indexing** — Generates chunk-level embeddings using `fastembed` (BAAI/bge-small-en-v1.5 ONNX runtime). Each record produces a `profile` chunk and optionally a `signal` chunk. Embeddings are L2-normalized and persisted to `data/rag_index/`.

4. **Grounded RAG** — User queries are parsed for structured filters (geography, AUM threshold, entity type), then semantically searched against the vector index. Retrieved records are sent to Groq's `llama-3.3-70b-versatile` (with `llama-3.1-8b-instant` fallback). Every claim the LLM makes is **post-verified** against the source CSV using RapidFuzz fuzzy matching on field values.

## Local Testing Guide

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier works)

### Setup

```bash
# Clone and enter the project
cd fo_rag_pipeline

# Create and activate a virtual environment
python -m venv venv
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your API key
# GROQ_API_KEY=gsk_your_key_here
# SEC_USER_AGENT="Your Name yourname@example.com"
```

### Run the Pipeline & Chat UI

```bash
streamlit run streamlit_app.py
```

1. The UI starts in **INIT** state. Click **"Initialize Pipeline"** to run enrichment and indexing.
2. Once the pipeline completes, the UI transitions to **READY** — you can query the dataset through the chat interface.
3. Pipeline artifacts (`family_offices.csv`, `embeddings.npy`, `rag_corpus.csv`) persist between runs. If they already exist, the UI starts in the READY state immediately.

### Development Notes

- The enrichment pipeline uses a **checkpoint CSV** to skip already-processed candidates across restarts.
- The LLM extractor uses a **file-based JSON cache** (`data/cache/llm_cache.json`) to avoid re-calling the API for identical inputs.
- Run individual scripts directly for debugging:
  ```bash
  python -m enrichment.run_enrichment
  python -m rag.build_index
  python -m rag.retrieval
  ```

## Deployment Guide (Streamlit Cloud)

1. Push the repository to GitHub.
2. In the [Streamlit Cloud dashboard](https://share.streamlit.io), deploy the repo with `streamlit_app.py` as the entry point.
3. Under **App Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   SEC_USER_AGENT = "Your Name yourname@example.com"
   ```
4. The pipeline reads from `st.secrets` automatically (see the `_get_client()` functions in both `llm_extractor.py` and `generation.py`).

**Important:** Streamlit Cloud runs are ephemeral — the pipeline must re-execute if the artifacts aren't present. For production, consider pre-building `family_offices.csv` and the RAG index and committing them (the `.npy` file must be tracked with Git LFS for files >100 MB).

## Trade-offs & Blind Spots

### fastembed vs PyTorch / sentence-transformers

**Choice:** `fastembed` with ONNX runtime — no PyTorch dependency, near-instant load times, and minimal memory footprint.

**Trade-off:** BGE-small-en-v1.5 (384-dim) is a smaller model than BGE-large. Semantic search precision is acceptable for a 50-record dataset, but recall on nuanced financial queries may be lower than with a larger transformer model. ONNX models also lack GPU acceleration without additional setup.

### Python-side verification vs prompt-only grounding

**Choice:** The LLM generates claims with field-level citations, then `rapidfuzz` fuzzy-matches each claim's extracted value against the actual CSV cell on the Python side.

**Trade-off:** This means the user always sees a verified/unverified split based on actual data parity, not just the LLM's self-reported confidence. But fuzzy matching with `partial_ratio > 85` is a heuristic — semantically equivalent values (e.g., "$500M" vs "500 million") may be incorrectly flagged as unverified, while near-miss strings could be incorrectly verified.

### Cache-backed pipeline vs live scraping

**Choice:** The enrichment pipeline caches LLM responses to disk (`data/cache/llm_cache.json`) and uses a checkpoint CSV to skip processed candidates.

**Trade-off:** This dramatically reduces API costs and speeds up re-runs, but the cache key is a hash of the input snippets — if the scraper returns different results or ordering for the same entity, the cache misses and you pay for a new API call. The dataset may also become stale over time as real-world data changes.

### Soft grounding in enrichment

**Choice:** The enrichment pipeline uses "soft grounding" — if the LLM extracts a value but its cited snippet ID cannot be mapped to a real URL, the value is **kept** (not discarded). Only the provenance URL is missing.

**Trade-off:** This prioritizes data completeness over evidence traceability. It means some fields in `family_offices.csv` may be accurate but lack a verifiable web source URL, reducing auditability.
