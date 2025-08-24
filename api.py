from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from scrape import scrape_tweets
from analyze import infer_sentiment, aggregate_daily

app = FastAPI(title="Sentiment Brand Monitor API", version="1.0.0")

class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([{"content": req.text}])
    df = infer_sentiment(df)
    if df.empty:
        return {"label_bin": None, "label3": None, "score": None}
    row = df.iloc[0]
    return {"label_bin": row["label_bin"], "label3": row["label3"], "score": float(row["score"])}

@app.get("/search")
def search(query: str = Query(..., min_length=1), limit: int = 100, since: Optional[str] = None):
    df = scrape_tweets(query, limit=limit, since=since)
    if df.empty:
        return {"items": [], "summary": {}, "trend": []}
    df = infer_sentiment(df)
    trend = aggregate_daily(df).to_dict(orient="records")
    summary = df["label3"].value_counts().to_dict()
    items = df[["date","user","content","likeCount","retweetCount","replyCount","url","label3","score"]]\
              .sort_values("date", ascending=False)\
              .head(200)\
              .to_dict(orient="records")
    return {"items": items, "summary": summary, "trend": trend}
