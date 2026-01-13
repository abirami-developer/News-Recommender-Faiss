# News Recommender (FAISS + SentenceTransformers)

A complete end-to-end, local, production-like project that ingests news, generates embeddings with SentenceTransformers, builds a FAISS index, and serves recommendations through a FastAPI backend.

## Features

- Data ingestion:
  - NewsAPI (when `NEWSAPI_KEY` provided)
  - Web-scraping fallback with `newspaper3k` + `beautifulsoup4`
- Data cleaning and preprocessing
- Saves processed articles as Parquet
- Embeddings via `sentence-transformers` (configurable model)
- FAISS `IndexFlatIP` (cosine similarity via normalized vectors)
- FastAPI endpoint: `/recommend?q=TEXT&k=10`
- Env-based configuration (.env)
- Logs and progress messages

---

## Project structure

