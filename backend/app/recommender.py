"""
backend/app/recommender.py

- Loads FAISS index + ids + processed parquet on startup
- Exposes Recommender class with method recommend(query: str, k: int, user_id: str | None)
- Uses SentenceTransformer to embed query, normalizes, performs FAISS IndexFlatIP search
- Adds user-based personalization: past queries influence current results
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# load env
load_dotenv()

LOG = logging.getLogger("recommender")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE = Path(__file__).resolve().parents[2]
FAISS_DIR = BASE / "data" / "faiss_index"
PROCESSED_FILE = BASE / "data" / "processed" / "articles.parquet"

INDEX_FILE = FAISS_DIR / "news.index"
IDS_FILE = FAISS_DIR / "ids.npy"

MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")


class UserProfileStore:
    """
    Simple in-memory store for user embeddings.
    Can be replaced with database for persistence later.
    """
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.user_embeddings = {}  # user_id -> list of embeddings
        LOG.info("UserProfileStore initialized")

    def add_query_embedding(self, user_id: str, embedding: np.ndarray):
        if user_id not in self.user_embeddings:
            self.user_embeddings[user_id] = []
        self.user_embeddings[user_id].append(embedding)

    def get_user_profile(self, user_id: str) -> np.ndarray | None:
        if user_id not in self.user_embeddings or not self.user_embeddings[user_id]:
            return None
        # Average all past embeddings for the user
        return np.mean(self.user_embeddings[user_id], axis=0)


class Recommender:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.ids: Optional[np.ndarray] = None
        self.df: Optional[pd.DataFrame] = None
        self.dim: Optional[int] = None
        self.user_store: Optional[UserProfileStore] = None
        self._load_resources()

    def _load_resources(self):
        LOG.info("Loading resources for recommender...")

        # load df
        if not PROCESSED_FILE.exists():
            LOG.error("Processed articles file missing: %s. Run ingestion pipeline first.", PROCESSED_FILE)
            raise FileNotFoundError(PROCESSED_FILE)
        self.df = pd.read_parquet(PROCESSED_FILE)
        LOG.info("Loaded %d articles", len(self.df))

        # load ids and index
        if not INDEX_FILE.exists() or not IDS_FILE.exists():
            LOG.error("FAISS artifacts missing in %s. Run ingestion pipeline first.", FAISS_DIR)
            raise FileNotFoundError("FAISS artifacts missing")
        self.ids = np.load(str(IDS_FILE))
        self.index = faiss.read_index(str(INDEX_FILE))
        self.dim = self.index.d
        LOG.info("Loaded FAISS index (dim=%d, ntotal=%d)", self.dim, self.index.ntotal)

        # load sentence-transformers model
        self.model = SentenceTransformer(self.model_name)
        self.model.max_seq_length = 512
        LOG.info("Loaded SentenceTransformer model=%s", self.model_name)

        # initialize user profile store
        self.user_store = UserProfileStore(self.dim)

    def _embed_query(self, query: str) -> np.ndarray:
        emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        # normalize
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        emb = emb / norm
        return emb.astype(np.float32)[0]  # return 1D vector

    def recommend(self, query: str, k: int = 10, user_id: str | None = None):
        if not query or not query.strip():
            return []

        k = max(1, int(k))
        query_emb = self._embed_query(query)

        # Personalization: combine with user profile
        if user_id:
            # store current query embedding
            self.user_store.add_query_embedding(user_id, query_emb)
            user_profile = self.user_store.get_user_profile(user_id)
            if user_profile is not None:
                # combine embeddings
                final_emb = 0.7 * query_emb + 0.3 * user_profile
                final_emb = final_emb / np.linalg.norm(final_emb)
            else:
                final_emb = query_emb
        else:
            final_emb = query_emb

        # search FAISS
        D, I = self.index.search(np.array([final_emb]).astype("float32"), k)
        scores = D[0].tolist()
        idxs = I[0].tolist()

        results = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            try:
                mapped_id = int(self.ids[idx]) if self.ids is not None else int(idx)
            except Exception:
                mapped_id = int(idx)

            row = self.df[self.df["id"] == mapped_id]
            if row.empty:
                try:
                    row = self.df.iloc[idx:idx+1]
                except Exception:
                    continue
            row = row.iloc[0].to_dict()
            item = {
                "score": float(score),
                "id": int(mapped_id),
                "title": row.get("title"),
                "description": row.get("description"),
                "url": row.get("url"),
                "source": row.get("source"),
                "publishedAt": row.get("publishedAt"),
            }
            results.append(item)

        return results
