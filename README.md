# 📰 AI‑Powered News Recommender System (FAISS + NLP)

An end‑to‑end **AI-based news recommendation system** that understands the *meaning* of user queries and recommends relevant news articles using **semantic search**, not just keywords.

This project demonstrates real‑world **Data Engineering + NLP + ML system design**, suitable for hackathons, portfolios, and interviews.

---

## 🚀 Key Features

* 🔎 **Semantic Search (Not Keyword Search)**
  Uses SentenceTransformers to understand *context* and *meaning* of user queries.

* ⚡ **FAISS Vector Similarity Search**
  Ultra‑fast similarity search over article embeddings.

* 🧠 **Pretrained NLP Model**
  Uses `all-MiniLM-L6-v2` for generating dense embeddings.

* 👤 **User Personalization (Advanced Feature)**
  Learns from user queries and improves future recommendations.

* 🌐 **FastAPI Backend**
  Production‑ready API with health checks and recommendation endpoints.

* 🧱 **Clean Modular Architecture**
  Ingestion → Processing → Indexing → Serving

---

## 🏗️ System Architecture

```
User Query
   ↓
SentenceTransformer (Embedding)
   ↓
FAISS Vector Index
   ↓
Top‑K Similar Articles
   ↓
FastAPI Response (JSON)
```

---

## 📂 Project Structure

```
news-recommender-faiss/
│
├── ingestion/              # Data ingestion scripts
│   ├── fetch_news.py
│   └── process_articles.py
│
├── backend/
│   └── app/
│       ├── main.py         # FastAPI server
│       ├── recommender.py  # Core recommendation logic
│       ├── db_helpers.py   # User profile handling
│
├── data/
│   ├── raw/                # Raw news data
│   ├── processed/          # Cleaned & structured data
│   └── faiss_index/        # FAISS index + embeddings
│
├── requirements.txt
├── .env.example
├── README.md
```

---

## 🧠 Concepts Used (Beginner‑Friendly)

| Concept         | Simple Explanation                    |
| --------------- | ------------------------------------- |
| NLP             | Teaching computers to understand text |
| Embeddings      | Converting text into numbers          |
| FAISS           | Fast search for similar vectors       |
| Semantic Search | Search by meaning, not words          |
| FastAPI         | Backend API framework                 |
| Vector Database | Stores embeddings for similarity      |

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
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1: Start the API Server

```bash
uvicorn backend.app.main:app --reload
```

### Step 2: Test Health Check

Open browser:

```
http://127.0.0.1:8000/health
```

### Step 3: Get Recommendations

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
    "source": "CNN"
  }
]
```

---

## 🌟 Why This Project Is Valuable

✔ Real‑world recommender system design
✔ Used in industry‑level AI products
✔ Demonstrates ML + backend integration
✔ Strong portfolio project for interviews

---

## 🔮 Future Improvements

* Frontend (React / Streamlit)
* Real‑time news ingestion
* Hybrid ranking (popularity + relevance)
* Deployment (Docker + Cloud)

---

## 👩‍💻 Author

**Abirami**
Aspiring Data Engineer & AI Engineer
GitHub: [https://github.com/abirami-developer](https://github.com/abirami-developer)

---

⭐ *If you like this project, give it a star!*
