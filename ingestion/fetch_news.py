#!/usr/bin/env python3
"""
ingestion/fetch_news.py

- Tries NewsAPI if NEWSAPI_KEY is present in env.
- Falls back to scraping a few seed news sites using newspaper3k.
- Writes line-delimited JSON to data/raw/news_raw.jsonl
- Produces human-readable progress logs.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone

import requests
from newspaper import Article
from newspaper import news_pool
from dotenv import load_dotenv
from tqdm import tqdm

# load environment variables
load_dotenv()

LOG = logging.getLogger("fetch_news")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RAW_DIR / "news_raw.jsonl"

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "").strip()
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/top-headlines"

# fallback seed sites (for newspaper3k)
SEED_SITES = [
    "https://www.bbc.com",
    "https://www.cnn.com",
    "https://www.reuters.com",
    "https://www.theguardian.com/international",
]

MAX_SCRAPE_ARTICLES_PER_SITE = int(os.environ.get("MAX_SCRAPE_ARTICLES_PER_SITE", 20))


def fetch_from_newsapi(page_size: int = 100, pages: int = 1, country: Optional[str] = None) -> List[Dict]:
    LOG.info("Fetching from NewsAPI...")
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY not set")

    articles = []
    headers = {"Authorization": NEWSAPI_KEY}
    for page in range(1, pages + 1):
        params = {"pageSize": page_size, "page": page}
        if country:
            params["country"] = country
        resp = requests.get(NEWSAPI_ENDPOINT, params=params, headers=headers, timeout=20)
        if resp.status_code != 200:
            LOG.warning("NewsAPI returned status %s: %s", resp.status_code, resp.text[:200])
            break
        data = resp.json()
        batch = data.get("articles", [])
        for a in batch:
            article = {
                "title": a.get("title"),
                "description": a.get("description"),
                "content": a.get("content"),
                "url": a.get("url"),
                "publishedAt": a.get("publishedAt"),
                "source": a.get("source", {}).get("name"),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "origin": "newsapi",
            }
            articles.append(article)
        total_results = data.get("totalResults")
        LOG.info("Fetched %d articles (page %d). totalResults=%s", len(batch), page, total_results)
        time.sleep(1.0)  # polite pause
    return articles


def scrape_with_newspaper(seed_sites: List[str] = SEED_SITES, max_per_site: int = MAX_SCRAPE_ARTICLES_PER_SITE) -> List[Dict]:
    LOG.info("Scraping with newspaper3k fallback...")
    sites_articles = []
    articles_objs = []

    # build newspaper Source objects (Article objects will be created)
    for site in seed_sites:
        try:
            site_paper = Article(site)
            # we don't need to download the site front page via Article (we'll crawl using a basic approach)
        except Exception:
            pass

    # Use simple approach: for each site, try to parse the site homepage, collect hrefs from a shallow crawl
    # newspaper3k has a build-in source builder, but it's sometimes slow or blocked; we implement a light scraping.
    for site in tqdm(seed_sites, desc="sites"):
        try:
            site_page = requests.get(site, timeout=15)
            site_page.raise_for_status()
            # naive: parse hrefs using newspaper or split — newspaper has build functionality: use Article to parse
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(site_page.text, "html.parser")
            anchors = soup.find_all("a", href=True)
            urls = []
            for a in anchors:
                href = a["href"]
                if href.startswith("http"):
                    urls.append(href)
                elif href.startswith("/"):
                    urls.append(site.rstrip("/") + href)
            # dedupe and limit
            seen = set()
            cleaned_urls = []
            for u in urls:
                if u in seen:
                    continue
                seen.add(u)
                cleaned_urls.append(u)
                if len(cleaned_urls) >= max_per_site:
                    break

            # create Article objects for each url
            site_articles = []
            for u in cleaned_urls:
                try:
                    art = Article(u)
                    art.download()
                    art.parse()
                    if not art.text or len(art.text) < 200:
                        continue
                    obj = {
                        "title": art.title,
                        "description": getattr(art, "meta_description", None),
                        "content": art.text,
                        "url": u,
                        "publishedAt": None,
                        "source": site,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "origin": "scrape",
                    }
                    site_articles.append(obj)
                except Exception:
                    continue
            sites_articles.extend(site_articles)
        except Exception as e:
            LOG.warning("Failed to scrape %s: %s", site, str(e))
            continue

    LOG.info("Scraped %d articles from %d sites", len(sites_articles), len(seed_sites))
    return sites_articles


def save_jsonl(records: List[Dict], path: Path):
    LOG.info("Saving %d raw articles to %s", len(records), path)
    with path.open("w", encoding="utf8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    LOG.info("Saved raw file.")


def main():
    LOG.info("Starting fetch_news pipeline...")
    all_articles = []
    # First, try NewsAPI
    if NEWSAPI_KEY:
        try:
            articles = fetch_from_newsapi(page_size=100, pages=1)
            if articles:
                all_articles.extend(articles)
        except Exception as e:
            LOG.exception("NewsAPI fetch failed, will fallback to scraping. %s", e)

    if not all_articles:
        LOG.info("No articles from NewsAPI, using scraping fallback.")
        scraped = scrape_with_newspaper()
        all_articles.extend(scraped)

    # Minimal dedupe by URL or title
    uniq = {}
    for a in all_articles:
        key = (a.get("url") or a.get("title") or "")[:500]
        if not key:
            continue
        if key in uniq:
            continue
        uniq[key] = a

    final = list(uniq.values())
    if not final:
        LOG.error("No articles fetched. Exiting.")
        return

    save_jsonl(final, OUTPUT_FILE)
    LOG.info("fetch_news completed. total=%d", len(final))


if __name__ == "__main__":
    main()
