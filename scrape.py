import subprocess, json
from typing import List, Optional
import pandas as pd

def _run(cmd: List[str]) -> str:
    """Run a shell command and return stdout."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout

def scrape_tweets(query: str, limit: int = 200, since: Optional[str] = None) -> pd.DataFrame:
    """
    Scrape public tweets for a query using snscrape.
    since: 'YYYY-MM-DD' string to restrict results.
    """
    q = query
    if since:
        q = f"{query} since:{since}"
    cmd = ["snscrape", "--jsonl", f"--max-results={limit}", "twitter-search", q]
    out = _run(cmd)
    records = [json.loads(line) for line in out.splitlines() if line.strip()]
    if not records:
        return pd.DataFrame(columns=["date", "user", "content", "likeCount", "retweetCount", "replyCount", "url"])
    df = pd.json_normalize(records)
    keep = ["date", "user.username", "content", "likeCount", "retweetCount", "replyCount", "url"]
    df = df[keep].rename(columns={
        "user.username": "user",
    })
    df["date"] = pd.to_datetime(df["date"])
    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--since", type=str, default=None)
    args = ap.parse_args()
    df = scrape_tweets(args.query, args.limit, args.since)
    print(df.head().to_string(index=False))
