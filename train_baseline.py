"""
Train a tiny baseline (Logistic Regression) on a sample sentiment dataset,
so you have a local model without Internet once cached.
"""
import os
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

os.makedirs("models", exist_ok=True)

def main():
    ds = load_dataset("tweet_eval", "sentiment")  # labels: 0=negative, 1=neutral, 2=positive
    df = pd.DataFrame(ds["train"])
    df = df.sample(n=20000, random_state=42)  # keep small for speed
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=3, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds, digits=4))
    joblib.dump(pipe, "models/baseline.joblib")
    print("Saved models/baseline.joblib")

if __name__ == "__main__":
    main()
