#!/usr/bin/env python3
"""
ingestion/process_articles.py

- Reads data/raw/news_raw.jsonl
- Cleans text, composes a 'text' field (title + description + content)
- Saves processed DataFrame to data/processed/articles.parquet
- Generates embeddings using SentenceTransformers (model name from env)
- Saves embeddings as .npy and ids mapping
- Builds a FAISS IndexFlatIP from normalized embeddings and saves index + ids

Outputs:
- data/processed/articles.parquet
- data/faiss_index/embeddings.npy
- data/faiss_index/ids.npy
- data/faiss_index/news.index
"""

import os
import json
import logging
from pathlib import Path
from typing import List
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import re

# load env
load_dotenv()

LOG = logging.getLogger("process_articles")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FAISS_DIR = BASE_DIR / "data" / "faiss_index"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "news_raw.jsonl"
PARQUET_FILE = PROCESSED_DIR / "articles.parquet"
EMB_FILE = FAISS_DIR / "embeddings.npy"
IDS_FILE = FAISS_DIR / "ids.npy"
INDEX_FILE = FAISS_DIR / "news.index"

MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH", 64))
EMBED_DTYPE = np.float32

def read_raw_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run ingestion/fetch_news.py first.")
    rows = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def clean_text(s: str) -> str:
    if not s:
        return ""
    # basic cleaning
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_text_field(row: dict) -> str:
    parts = []
    for k in ("title", "description", "content"):
        v = row.get(k)
        if v:
            parts.append(v)
    text = " \n ".join(parts)
    return clean_text(text)

def to_dataframe(rows: List[dict]) -> pd.DataFrame:
    data = []
    for i, r in enumerate(rows):
        text = build_text_field(r)
        data.append({
            "id": i,
            "title": r.get("title"),
            "content": r.get("content"),
            "description": r.get("description"),
            "text": text,
            "url": r.get("url"),
            "publishedAt": r.get("publishedAt"),
            "source": r.get("source"),
            "origin": r.get("origin"),
            "fetched_at": r.get("fetched_at"),
        })
    df = pd.DataFrame(data)
    return df

def compute_embeddings(model_name: str, texts: List[str], batch_size: int = 64):
    LOG.info("Loading SentenceTransformer model=%s", model_name)
    model = SentenceTransformer(model_name)
    model.max_seq_length = 512
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype(EMBED_DTYPE)
    return embeddings

def normalize_embeddings(emb: np.ndarray):
    # normalize rows to unit length for cosine similarity using inner product
    LOG.info("Normalizing embeddings for cosine similarity (IndexFlatIP)...")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    LOG.info("Building FAISS IndexFlatIP with dim=%d", dim)
    index = faiss.IndexFlatIP(dim)
    # ensure index is on CPU
    if not index.is_trained and not index.ntotal:
        LOG.info("Index created. ntotal=%d", index.ntotal)
    index.add(embeddings)
    LOG.info("Index ntotal=%d", index.ntotal)
    return index

def save_artifacts(df, embeddings, ids, index):
    LOG.info("Saving parquet to %s", PARQUET_FILE)
    df.to_parquet(PARQUET_FILE, index=False)
    LOG.info("Saving embeddings to %s", EMB_FILE)
    np.save(EMB_FILE, embeddings)
    LOG.info("Saving ids to %s", IDS_FILE)
    np.save(IDS_FILE, ids)
    LOG.info("Writing faiss index to %s", INDEX_FILE)
    faiss.write_index(index, str(INDEX_FILE))

def main():
    LOG.info("Starting processing pipeline...")
    rows = read_raw_jsonl(RAW_FILE)
    df = to_dataframe(rows)
    if df.empty:
        LOG.error("No data to process. Exiting.")
        return
    # assign stable ids (use existing df['id'] already)
    ids = df["id"].to_numpy(dtype=np.int64)
    LOG.info("Articles count: %d", len(df))

    # Compose text for embedding: prefer 'text' field
    texts = df["text"].fillna("").astype(str).tolist()

    # compute embeddings
    embeddings = compute_embeddings(MODEL_NAME, texts, batch_size=BATCH_SIZE)
    # normalize for cosine
    embeddings = normalize_embeddings(embeddings)

    # build index
    index = build_faiss_index(embeddings)

    # save everything
    save_artifacts(df, embeddings, ids, index)
    LOG.info("Processing pipeline finished successfully.")

if __name__ == "__main__":
    main()
