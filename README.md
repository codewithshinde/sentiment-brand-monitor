# Sentiment Brand Monitor (Live, End-to-End)

A **ready-to-run** project that:
- Scrapes social posts (Tweets via `snscrape`) for a brand query
- Runs **live sentiment** using a Hugging Face pipeline
- Shows a **Streamlit dashboard** (charts + tables)
- Exposes a **FastAPI** for programmatic access
- Is deployable to **Hugging Face Spaces** (Streamlit) and **Render/Railway/Fly.io** (FastAPI) or Docker anywhere

> No Twitter API keys required. `snscrape` pulls public results by query.

---

## 📦 Quickstart (Local)

**Requirements:** Python 3.10+ recommended

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Run the Streamlit Dashboard (Live UI)
```bash
streamlit run app_streamlit.py
```
- Enter a **Brand/Query** (e.g., `nike`, `"american express"`, `starbucks`)
- Choose how many posts to fetch (e.g., 200), set a **since** date (optional), hit **Run**
- You'll see **live sentiment** results, time trends, and a downloadable CSV

### 2) Run the FastAPI Service (Live API)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Endpoints:
- `GET /health` — health check
- `POST /predict` — body: `{ "text": "I love this shoe!" }`
- `GET /search?query=nike&limit=50` — scrape + analyze live

### 3) (Optional) Train a Lightweight Model
```bash
python train_baseline.py
```
- Trains a Logistic Regression on a small sample dataset (Twitter US Airline Sentiment via `datasets`)
- Saves `models/baseline.joblib` for demo; Streamlit can use either **Hugging Face** or **baseline**

> For robust results at scale, fine-tune a transformer (e.g., DistilBERT) — see `train_transformer.py` (skeleton).

---

## 🚀 Deployment

### Hugging Face Spaces (Streamlit)
1. Create a new **Space** → **Streamlit**
2. Push the repo contents
3. In **Space Settings → Hardware**, choose at least the default CPU. Make sure `requirements.txt` is included.
4. Set **App file** to `app_streamlit.py`

### Render / Railway / Fly.io (FastAPI)
- Use this **Dockerfile** (or a native Python buildpack on Render).
- Expose port **8000**.
- Start command (if not Docker): `uvicorn api:app --host 0.0.0.0 --port 8000`

### Docker (Anywhere)
```bash
docker build -t sentiment-brand-monitor .
docker run -p 8000:8000 sentiment-brand-monitor
```
- Then open `http://localhost:8000/docs` for the API
- For Streamlit in Docker, override CMD to:
  `streamlit run app_streamlit.py --server.port 7860 --server.address 0.0.0.0`
  and map `-p 7860:7860`

---

## 🧠 Notes & Tradeoffs

- `snscrape` is resilient but **rate-limited** by site behavior; for very large volumes, consider official APIs + queues.
- The default live predictor uses **`distilbert-base-uncased-finetuned-sst-2-english`** (binary: POSITIVE/NEGATIVE).
  We map confidence and show counts; a **Neutral** band is estimated via score thresholds (see `analyze.py`).
- For tri-class ground truth and robust benchmarking, use a labeled dataset (e.g., `tweet_eval`) and train/fine-tune.
- Dashboard intentionally simple; extend with auth, alerts, and storage (e.g., Postgres) for production.

---

## 📁 Project Structure

```
.
├── app_streamlit.py        # Streamlit dashboard (live)
├── api.py                  # FastAPI app (live)
├── scrape.py               # Social scraping utils (snscrape)
├── analyze.py              # Sentiment inference + analytics
├── train_baseline.py       # Tiny baseline (LogReg) on sample dataset
├── train_transformer.py    # Skeleton for HF fine-tuning (optional)
├── requirements.txt
├── Dockerfile
├── .streamlit/config.toml  # Streamlit tweaks
├── models/                 # Saved models (created at runtime)
└── data/                   # Cached CSVs (created at runtime)
```

---

## 🔐 Environment Variables (Optional)

Create a `.env` file (optional):
```
# Not required for snscrape, but placeholder if you add APIs later
TWITTER_BEARER_TOKEN=
```

---

## 📚 License
MIT
