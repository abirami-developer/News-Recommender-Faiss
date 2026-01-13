"""
Optional helper utilities for storing article metadata or metrics.
This module is intentionally minimal — in production you may swap this for a proper DB layer.
"""

import sqlite3
from pathlib import Path
from typing import Optional

BASE = Path(__file__).resolve().parents[2]
DB_PATH = BASE / "data" / "recommender_meta.sqlite3"

def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        k INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def log_request(query: str, k: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO requests(query, k) VALUES (?, ?)", (query, k))
    conn.commit()
    conn.close()
