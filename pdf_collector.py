# file_collector.py
import os, uuid, json, pandas as pd, streamlit as st, pdfplumber, docx
from datetime import datetime, timezone
from langdetect import detect
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Setup
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# --------- Helpers ----------
def read_pdf(file):
    """Extract text from PDF"""
    text = ""
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                text += t + "\n"
    return text

def read_docx(file):
    """Extract text from Word DOCX"""
    d = docx.Document(file)
    return "\n".join([para.text for para in d.paragraphs])

def create_record(name, content, ftype):
    """Create enriched record from file text"""
    rec = {
        "id": str(uuid.uuid4()),
        "source": ftype,
        "title": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": content
    }
    try:
        rec["language"] = detect(content)
    except:
        rec["language"] = "unknown"
    rec["sentiment"] = sia.polarity_scores(content)
    rec["summary"] = " ".join(content.split()[:40])  # first ~40 words
    return rec

def save(records, fmt):
    """Save records as CSV or JSON"""
    df = pd.json_normalize(records)
    if fmt == "CSV":
        df.to_csv("file_output.csv", index=False, encoding="utf-8")
        return "Saved to file_output.csv"
    else:
        with open("file_output.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        return "Saved to file_output.json"

# --------- Streamlit UI ----------
st.title("📂 File Collector (PDF + DOCX)")
uploaded = st.file_uploader(
    "Upload PDF or Word files", 
    type=["pdf", "docx"], 
    accept_multiple_files=True
)
fmt = st.selectbox("Save as", ["CSV", "JSON"])

if st.button("Process Files"):
    if uploaded:
        recs = []
        for f in uploaded:
            try:
                name = f.name
                ext = os.path.splitext(name)[1].lower()
                if ext == ".pdf":
                    content = read_pdf(f)
                elif ext == ".docx":
                    content = read_docx(f)
                else:
                    st.warning(f"Unsupported file type: {name}")
                    continue
                recs.append(create_record(name, content, ext))
                st.success(f"Processed {name}")
            except Exception as e:
                st.error(f"Error {f.name}: {e}")
        if recs:
            msg = save(recs, fmt)
            st.success(msg)
            st.subheader("Preview")
            st.json(recs[0])
    else:
        st.warning("Upload at least one file.")
