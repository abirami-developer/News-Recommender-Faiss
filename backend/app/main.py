#!/usr/bin/env python3
"""
backend/app/main.py

- FastAPI server exposing /recommend?q=TEXT&k=10&user_id=USER
- Loads Recommender and answers requests quickly
- Includes healthcheck endpoint
"""

import os
import logging
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from .recommender import Recommender

load_dotenv()
LOG = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="News Recommender (FAISS + SentenceTransformers)")

# instantiate recommender at startup
recommender: Recommender | None = None

@app.on_event("startup")
def startup_event():
    global recommender
    try:
        recommender = Recommender()
        LOG.info("Recommender ready.")
    except Exception as e:
        LOG.exception("Failed to load recommender: %s", e)
        # do not raise here; we want app to start to show error endpoints
        recommender = None

class RecommendResponseItem(BaseModel):
    id: int
    score: float
    title: str | None = None
    description: str | None = None
    url: str | None = None
    source: str | None = None
    publishedAt: str | None = None

@app.get("/health")
def health():
    ok = recommender is not None
    return JSONResponse({"ok": ok})

@app.get("/recommend", response_model=List[RecommendResponseItem])
def recommend(
    q: str = Query(..., min_length=2),
    k: int = Query(10, ge=1, le=50),
    user_id: str | None = Query(None)
):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded. Run ingestion & processing first.")
    results = recommender.recommend(query=q, k=k, user_id=user_id)
    return results

# simple root
@app.get("/")
def root():
    return {"service": "news-recommender-faiss", "status": "ok"}
