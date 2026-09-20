# 📰 AI-Powered News Recommendation System

An end-to-end **AI-based news recommendation system** that understands the *meaning* of user queries and recommends relevant news articles using **semantic search**, not just keywords.

This project demonstrates **Data Engineering + NLP + ML system design**, with a modular pipeline for news ingestion, processing, semantic embedding, similarity search, and API-based recommendation.

---

## 🚀 Key Features

* 🔎 **Semantic Search (Not Keyword Search)**
  Uses SentenceTransformers to understand the *context* and *meaning* of user queries.

* ⚡ **FAISS Vector Similarity Search**
  Uses FAISS for fast similarity search over article embeddings.

* 🧠 **Pretrained NLP Model**
  Uses `all-MiniLM-L6-v2` for generating dense text embeddings.

* 👤 **Basic User Personalization**
  Uses previous user query embeddings to influence future recommendations. User profiles are maintained in memory during the application session.

* 🌐 **FastAPI Backend**
  Provides REST API endpoints for health checks and news recommendations.

* 🧱 **Clean Modular Architecture**
  Ingestion → Processing → Embedding → Indexing → Serving

---

## 🏗️ System Architecture

```text
News Sources
     ↓
NewsAPI / Web Scraping Fallback
     ↓
Article Ingestion
     ↓
Data Processing
     ↓
SentenceTransformer
     ↓
Article Embeddings
     ↓
FAISS Vector Index
     ↓
Semantic Similarity Search
     ↓
Top-K Recommendations
     ↓
FastAPI Response (JSON)
```

---

## 📂 Project Structure

```text
News Recommender Files/
│
├── ingestion/
│   ├── fetch_news.py
│   └── process_articles.py
│
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── recommender.py
│       └── db_helpers.py
│
├── data/
│   ├── raw/
│   │   └── news_raw.jsonl
│   │
│   ├── processed/
│   │   └── articles.parquet
│   │
│   └── faiss_index/
│       ├── embeddings.npy
│       ├── ids.npy
│       └── news.index
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🧠 Concepts Used

| Concept               | Simple Explanation                                            |
| --------------------- | ------------------------------------------------------------- |
| NLP                   | Techniques used to process and understand text                |
| Embeddings            | Converting text into numerical vectors                        |
| Sentence Transformers | Pretrained models used to generate meaningful text embeddings |
| FAISS                 | Library for efficient vector similarity search                |
| Semantic Search       | Searching based on meaning rather than exact keywords         |
| FastAPI               | Python framework used to build the backend API                |
| Vector Similarity     | Measuring how closely two embeddings are related              |

---

## 🔄 Data Pipeline

### 1️⃣ News Ingestion

The system attempts to collect news articles using **NewsAPI**.

If NewsAPI is unavailable or does not return suitable articles, a web-scraping fallback is used to collect articles from selected news sources.

Raw articles are stored in:

```text
data/raw/news_raw.jsonl
```

---

### 2️⃣ Article Processing

The raw news data is cleaned and processed.

Article information such as:

* Title
* Description
* Content
* URL
* Source
* Published date

is processed into a structured dataset.

The processed data is stored as:

```text
data/processed/articles.parquet
```

---

### 3️⃣ Generate Embeddings

The project uses the pretrained:

```text
all-MiniLM-L6-v2
```

SentenceTransformer model to convert article text into numerical embeddings.

These embeddings represent the semantic meaning of the articles.

---

### 4️⃣ Build FAISS Index

The generated embeddings are normalized and added to a FAISS:

```text
IndexFlatIP
```

index.

The index is used to efficiently search for articles that are semantically similar to a user's query.

The generated files are stored in:

```text
data/faiss_index/
```

---

### 5️⃣ Generate Recommendations

When a user submits a query:

```text
User Query
    ↓
SentenceTransformer
    ↓
Query Embedding
    ↓
FAISS Similarity Search
    ↓
Top-K Similar Articles
    ↓
FastAPI JSON Response
```

---

## 👤 User Personalization

The system includes a basic personalization mechanism.

When a `user_id` is provided, previous query embeddings are stored in an **in-memory user profile**.

The current query and the user's previous query profile are combined to influence the recommendation results.

This means recommendations can consider both:

* The user's current query
* The user's previous queries

Since the profile is stored in memory, the personalization history is reset when the application restarts.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/abirami-developer/News-Recommender-Faiss.git
cd News-Recommender-Faiss
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file based on `.env.example`.

If using NewsAPI, add:

```text
NEWSAPI_KEY=your_api_key
```

The system also includes a scraping fallback if NewsAPI is unavailable.

---

## ▶️ Running the Project

### Step 1: Start the API Server

```bash
uvicorn backend.app.main:app --reload
```

### Step 2: Test Health Check

Open in your browser:

```text
http://127.0.0.1:8000/health
```

### Step 3: Get Recommendations

Example:

```text
http://127.0.0.1:8000/recommend?q=Machine%20Learning&k=5&user_id=abi
```

Or using cURL:

```bash
curl "http://127.0.0.1:8000/recommend?q=Machine%20Learning&k=5&user_id=abi"
```

---

## 📌 Example API Response

```json
[
  {
    "id": 14,
    "score": 0.18,
    "title": "Business News",
    "description": "Example article description",
    "url": "https://example.com/article",
    "source": "CNN",
    "publishedAt": "2026-01-01T10:00:00"
  }
]
```

---

## 📊 Current Implementation

The current project implements:

✔ News ingestion using NewsAPI
✔ Web-scraping fallback
✔ Article preprocessing
✔ Parquet-based processed data
✔ SentenceTransformer embeddings
✔ `all-MiniLM-L6-v2` model
✔ Normalized embeddings
✔ FAISS `IndexFlatIP` similarity search
✔ FastAPI backend
✔ `/health` endpoint
✔ `/recommend` endpoint
✔ Basic in-memory user personalization

---

## 🌟 Why This Project Is Valuable

✔ Demonstrates semantic search and recommendation concepts
✔ Combines NLP, embeddings, vector similarity search, and backend development
✔ Implements an end-to-end data processing and recommendation pipeline
✔ Provides practical experience with pretrained ML models and APIs
✔ Suitable for demonstrating ML/NLP system design in interviews

---

## 🔮 Future Improvements

* Frontend using React or Streamlit
* Persistent user profiles and recommendation history
* Real-time news ingestion
* Hybrid ranking using semantic relevance and popularity
* Docker containerization
* Cloud deployment
* Automated scheduled news ingestion

---

## 👩‍💻 Author

**Abirami Seenivasan**
Aspiring Data Engineer & AI Engineer

GitHub: https://github.com/abirami-developer

---

⭐ If you like this project, give it a star!
