# link_collector.py
import os, uuid, json, requests, pandas as pd, streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timezone
import praw
from newspaper import Article
from langdetect import detect
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# setup
load_dotenv()
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Reddit setup
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
reddit = None
if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

# --------- Helpers ----------
def fetch_reddit_post(url):
    sub = reddit.submission(url=url)
    return {
        "id": str(uuid.uuid4()),
        "source": "reddit",
        "title": sub.title,
        "author": sub.author.name if sub.author else "unknown",
        "timestamp": datetime.fromtimestamp(sub.created_utc, tz=timezone.utc).isoformat(),
        "text": sub.title + "\n" + sub.selftext,
        "url": url
    }

def fetch_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&pageSize=1"
    res = requests.get(url).json()
    if not res.get("articles"): return None
    a = res["articles"][0]
    return {
        "id": str(uuid.uuid4()),
        "source": "news",
        "title": a.get("title"),
        "author": a.get("author") or "unknown",
        "timestamp": a.get("publishedAt"),
        "text": (a.get("title") or "") + "\n" + (a.get("description") or ""),
        "url": a.get("url")
    }

def fetch_generic_url(url):
    art = Article(url)
    art.download(); art.parse()
    return {
        "id": str(uuid.uuid4()),
        "source": "web",
        "title": art.title,
        "author": art.authors[0] if art.authors else "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": art.text,
        "url": url
    }

def enrich(record):
    text = record["text"]
    try: record["language"] = detect(text)
    except: record["language"] = "unknown"
    record["sentiment"] = sia.polarity_scores(text)
    record["summary"] = " ".join(text.split()[:40])  # simple short summary
    return record

def save(records, fmt):
    df = pd.json_normalize(records)
    if fmt=="CSV":
        df.to_csv("link_output.csv", index=False, encoding="utf-8")
        return "Saved to link_output.csv"
    else:
        with open("link_output.json","w",encoding="utf-8") as f: json.dump(records,f,indent=4)
        return "Saved to link_output.json"

# --------- UI ----------
st.title("🔗 Link Collector")
opt = st.radio("Source type", ["Reddit Post", "News Query", "Generic URL"])
user_input = st.text_input("Enter link or query:")
fmt = st.selectbox("Save as", ["CSV","JSON"])

if st.button("Fetch & Save"):
    records=[]
    try:
        if opt=="Reddit Post": records.append(enrich(fetch_reddit_post(user_input)))
        elif opt=="News Query":
            r=fetch_news(user_input); 
            if r: records.append(enrich(r))
            else: st.error("No news found.")
        else: records.append(enrich(fetch_generic_url(user_input)))
        if records:
            msg=save(records,fmt)
            st.success(msg); st.json(records[0])
    except Exception as e:
        st.error(f"Error: {e}")
