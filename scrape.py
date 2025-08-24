# scrape.py
import os, subprocess, json, datetime as dt
from typing import List, Optional
import pandas as pd
import requests

SNSCRAPE_CMD = ["snscrape", "--jsonl", "twitter-search"]

def _run(cmd: List[str]) -> str:
    """Run a shell command and return stdout (raise on non‑zero)."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout

def _scrape_with_snscrape(query: str, limit: int, since: Optional[str]) -> pd.DataFrame:
    q = f"{query} since:{since}" if since else query
    cmd = ["snscrape", "--jsonl", f"--max-results={limit}", "twitter-search", q]
    out = _run(cmd)
    records = [json.loads(line) for line in out.splitlines() if line.strip()]
    if not records:
        return pd.DataFrame(columns=["date","user","content","likeCount","retweetCount","replyCount","url"])
    df = pd.json_normalize(records)
    keep = ["date","user.username","content","likeCount","retweetCount","replyCount","url"]
    df = df[keep].rename(columns={"user.username":"user"})
    df["date"] = pd.to_datetime(df["date"])
    return df

def _scrape_with_twitter_api(query: str, limit: int, since: Optional[str]) -> pd.DataFrame:
    """Twitter API v2 Recent Search. Requires env TWITTER_BEARER_TOKEN."""
    bearer = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer:
        raise RuntimeError("TWITTER_BEARER_TOKEN not set")

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer}"}
    params = {
        "query": query,
        "max_results": min(max(limit, 10), 100),  # API cap 100 per request
        "tweet.fields": "created_at,public_metrics,author_id",
        "expansions": "author_id",
        "user.fields": "username",
    }
    if since:
        # Twitter API expects RFC3339 time; use midnight UTC of the date
        params["start_time"] = f"{since}T00:00:00Z"

    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data", [])
    users = {u["id"]: u.get("username","") for u in payload.get("includes", {}).get("users", [])}

    rows = []
    for t in data:
        metrics = t.get("public_metrics", {})
        user = users.get(t.get("author_id",""), "")
        tid = t["id"]
        rows.append({
            "date": t.get("created_at"),
            "user": user,
            "content": t.get("text",""),
            "likeCount": metrics.get("like_count", 0),
            "retweetCount": metrics.get("retweet_count", 0),
            "replyCount": metrics.get("reply_count", 0),
            "url": f"https://twitter.com/{user}/status/{tid}" if user else f"https://twitter.com/i/web/status/{tid}",
        })

    df = pd.DataFrame(rows, columns=["date","user","content","likeCount","retweetCount","replyCount","url"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

def _fallback_sample(query: str, limit: int) -> pd.DataFrame:
    """Offline/demo fallback so the UI still works when live sources fail."""
    now = pd.Timestamp.utcnow()
    sample = [
        {"date": now - pd.Timedelta(hours=1), "user": "demo_user1", "content": f"I love {query}!", "likeCount": 15, "retweetCount": 2, "replyCount": 1, "url": ""},
        {"date": now - pd.Timedelta(hours=2), "user": "demo_user2", "content": f"{query} quality is not great lately.", "likeCount": 3, "retweetCount": 0, "replyCount": 0, "url": ""},
        {"date": now - pd.Timedelta(days=1), "user": "demo_user3", "content": f"{query} is okay—mixed feelings.", "likeCount": 5, "retweetCount": 1, "replyCount": 0, "url": ""},
    ]
    df = pd.DataFrame(sample * max(1, min(limit // 3, 50)))
    df = df.head(limit).copy()
    return df

def scrape_tweets(query: str, limit: int = 200, since: Optional[str] = None) -> pd.DataFrame:
    """
    Try snscrape -> Twitter API v2 (if token present) -> offline fallback.
    Returns columns: date, user, content, likeCount, retweetCount, replyCount, url
    """
    # 1) snscrape first (no credentials)
    try:
        return _scrape_with_snscrape(query, limit, since)
    except Exception as e:
        # print or log if you want
        # print(f"snscrape failed: {e}")
        pass

    # 2) Twitter API v2 if bearer is configured
    try:
        if os.environ.get("TWITTER_BEARER_TOKEN"):
            return _scrape_with_twitter_api(query, limit, since)
    except Exception as e:
        # print(f"twitter api failed: {e}")
        pass

    # 3) Last resort: demo data so UI works
    return _fallback_sample(query, limit)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--since", type=str, default=None)
    args = ap.parse_args()
    df = scrape_tweets(args.query, args.limit, args.since)
    print(df.head().to_string(index=False))
