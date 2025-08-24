from typing import Tuple
import pandas as pd
from transformers import pipeline

# Lazy global
_snt = None

def get_pipeline():
    global _snt
    if _snt is None:
        _snt = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _snt

def infer_sentiment(df: pd.DataFrame, text_col: str = "content", neutral_band: Tuple[float,float]=(0.45,0.55)) -> pd.DataFrame:
    """
    Run HF sentiment pipeline (binary) and approximate a Neutral class by confidence band.
    Returns df with columns: label_bin, score, label3 (POS/NEG/NEU)
    """
    if df.empty:
        return df.assign(label_bin=[], score=[], label3=[])
    clf = get_pipeline()
    preds = clf(df[text_col].tolist(), truncation=True)
    df = df.copy()
    df["label_bin"] = [p["label"] for p in preds]
    df["score"] = [p["score"] for p in preds]
    lo, hi = neutral_band
    def to_tri(label, score):
        if lo <= score <= hi:
            return "NEU"
        return "POS" if label.upper().startswith("POS") else "NEG"
    df["label3"] = [to_tri(l, s) for l, s in zip(df["label_bin"], df["score"])]
    return df

def aggregate_daily(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = (df
         .assign(day=pd.to_datetime(df[date_col]).dt.date)
         .groupby(["day","label3"], as_index=False)
         .size()
        )
    pivot = g.pivot(index="day", columns="label3", values="size").fillna(0).reset_index()
    return pivot
