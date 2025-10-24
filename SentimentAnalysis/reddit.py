import streamlit as st
import json
import praw
import pandas as pd
import uuid
from datetime import datetime, timezone
import os

# ----------------------------
# ✅ Reddit API authentication
# ----------------------------
def init_reddit():
    return praw.Reddit(
        client_id="iG9C35PhDmFPaObXbS2cvcRhnETVUA",
        client_secret="tt78RM7C0r9aknY1ocVVMA",
        user_agent="RedditTextExtractor by u/EastSlide9495"
    )

# ----------------------------
# ✅ Extract text data from a Reddit URL
# ----------------------------
def fetch_post_data(url):
    reddit = init_reddit()
    submission = reddit.submission(url=url)

    post_data = {
        "id": str(uuid.uuid4()),
        "source": "reddit",
        "author": str(submission.author) if submission.author else "unknown",
        "timestamp": datetime.fromtimestamp(
            submission.created_utc, tz=timezone.utc
        ).isoformat(),
        "text": submission.selftext if submission.selftext else submission.title,
        "metadata": {
            "language": "en",
            "likes": submission.score,
            "rating": None,
            "url": url,
        },
    }
    return post_data

# ----------------------------
# ✅ Save data to JSON/CSV
# ----------------------------
def save_data(data, format="json", filename="output_data"):
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", f"{filename}.{format}")

    if format == "json":
        pd.DataFrame([data]).to_json(filepath, orient="records", indent=4)
    elif format == "csv":
        pd.DataFrame([data]).to_csv(filepath, index=False, encoding="utf-8")

    return filepath

# ----------------------------
# ✅ Streamlit UI
# ----------------------------
st.set_page_config(page_title="Reddit Text Extractor", layout="wide")
st.title("📥 Reddit Text Data Extractor")

url = st.text_input("Enter Reddit link:")
format_option = st.selectbox("Save as:", ["JSON", "CSV"])

if st.button("Fetch & Save"):
    if url:
        with st.spinner("Fetching data..."):
            data = fetch_post_data(url)
            file_path = save_data(data, format=format_option.lower())

        st.success(f"✅ Data saved to {file_path}")

        st.subheader("Preview")
        st.json(data)
    else:
        st.error("Please enter a valid Reddit URL")
