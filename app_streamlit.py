import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from scrape import scrape_tweets
from analyze import infer_sentiment, aggregate_daily

st.set_page_config(page_title="Sentiment Brand Monitor", layout="wide")

st.title("📊 Sentiment Brand Monitor (Live)")
st.write("Scrape recent Tweets for your brand query and analyze sentiment in real time.")

with st.sidebar:
    st.header("Settings")
    query = st.text_input("Brand/Query", value="nike")
    limit = st.slider("Max posts", min_value=50, max_value=1000, step=50, value=300)
    since_enabled = st.checkbox("Filter by date (since)", value=False)
    since = st.date_input("Since date", value=date.today(), disabled=not since_enabled)
    run = st.button("Run")

@st.cache_data(show_spinner=False)
def run_pipeline(q: str, n: int, since_str: str|None):
    df = scrape_tweets(q, n, since_str)
    if df.empty:
        return df, pd.DataFrame()
    df = infer_sentiment(df)
    trend = aggregate_daily(df)
    return df, trend

if run:
    since_str = since.isoformat() if since_enabled else None
    with st.spinner("Scraping + analyzing..."):
        df, trend = run_pipeline(query, limit, since_str)

    if df.empty:
        st.warning("No results found. Try a different query or increase the limit.")
    else:
        col1, col2 = st.columns([1,2], gap="large")
        with col1:
            st.subheader("Summary")
            counts = df["label3"].value_counts()
            st.write(counts)

            st.subheader("Download")
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="results.csv")

        with col2:
            st.subheader("Recent Posts (sample)")
            st.dataframe(df[["date","user","content","label3","score","likeCount","retweetCount","replyCount","url"]]\
                         .sort_values("date", ascending=False).head(200), use_container_width=True)

        st.subheader("Trend (Daily Counts)")
        if not trend.empty:
            fig = plt.figure()
            for label in ["NEG","NEU","POS"]:
                if label in trend.columns:
                    plt.plot(trend["day"], trend[label], label=label)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Not enough data for a daily trend.")

else:
    st.info("Enter a query on the left and click **Run** to begin.")
