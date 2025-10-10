"""
news_gdelt.py (fixed version)
------------------------------------------------
Collects financial news from GDELT (free & global),
performs sentiment analysis with FinBERT,
and outputs a cleaned CSV file: news_with_sentiment.csv
"""

import os
import pandas as pd
import time
import io
from datetime import datetime, timedelta, timezone
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ===============================
# 1️⃣ CONFIGURATION
# ===============================
KEYWORDS = ["gold", "economy", "inflation", "usd", "fed"]
DAYS_BACK = 14
OUTPUT_FILE = "news_with_sentiment.csv"
MODEL_NAME = "yiyanghkust/finbert-tone"

# ===============================
# 2️⃣ FETCH NEWS FROM GDELT
# ===============================
def fetch_gdelt_news(keywords, days_back=14):
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    all_news = []
    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=days_back)

    print(f"\n📰 Fetching news from {from_date.date()} to {to_date.date()} ...\n")

    for kw in keywords:
        print(f"🔍 Searching keyword: {kw}")
        params = {
            "query": kw,
            "mode": "artlist",
            "maxrecords": 250,
            "sort": "DateDesc"
        }
        try:
            resp = requests.get(base_url, params=params)
            if resp.status_code != 200:
                print(f"⚠️ Request failed ({resp.status_code}) for {kw}")
                continue

            # ✅ use io.StringIO instead of pandas.compat
            df = pd.read_csv(io.StringIO(resp.text), sep="\t")

            if "DATE" not in df.columns:
                print(f"⚠️ No valid results for {kw}")
                continue

            cols = [c for c in df.columns if c in ["DATE", "DocumentIdentifier", "SourceCommonName", "V2ENHANCEDLOCATIONNAME", "Themes", "Tone", "URL"]]
            df = df[cols]
            df["keyword"] = kw

            df.rename(columns={
                "DATE": "date",
                "DocumentIdentifier": "title",
                "SourceCommonName": "source",
                "URL": "url",
                "Tone": "tone"
            }, inplace=True)

            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            df.dropna(subset=["title"], inplace=True)

            all_news.append(df)
            time.sleep(1)

        except Exception as e:
            print(f"⚠️ Error fetching {kw}: {e}")

    if not all_news:
        print("❌ No news collected.")
        return pd.DataFrame()

    result = pd.concat(all_news, ignore_index=True)
    result.drop_duplicates(subset=["title"], inplace=True)
    print(f"\n✅ Collected {len(result)} total articles.\n")
    return result

# ===============================
# 3️⃣ LOAD FINBERT MODEL
# ===============================
def load_finbert():
    print("🚀 Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("✅ FinBERT loaded.\n")
    return nlp

# ===============================
# 4️⃣ SENTIMENT ANALYSIS
# ===============================
def analyze_sentiment(news_df, nlp_pipeline):
    print("🔍 Analyzing sentiment for each headline...\n")
    results = []
    for text in tqdm(news_df["title"].astype(str).tolist()):
        try:
            pred = nlp_pipeline(text[:512])[0]
            results.append(pred)
        except Exception as e:
            print("Error:", e)
            results.append({"label": "neutral", "score": 0.0})

    news_df["label"] = [r["label"] for r in results]
    news_df["score"] = [r["score"] for r in results]
    mapping = {"positive": 1, "neutral": 0, "negative": -1}
    news_df["sent_value"] = news_df["label"].map(mapping) * news_df["score"]

    return news_df

# ===============================
# 5️⃣ MAIN PIPELINE
# ===============================
def main():
    try:
        df_raw = fetch_gdelt_news(KEYWORDS, DAYS_BACK)
        if len(df_raw) == 0:
            print("⚠️ No news found.")
            return

        finbert_pipeline = load_finbert()
        df_sent = analyze_sentiment(df_raw, finbert_pipeline)

        df_sent.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Sentiment analysis complete! Saved to {OUTPUT_FILE}\n")
        print(df_sent.head(10))

    except Exception as e:
        print("❌ Error:", e)

# ===============================
# RUN SCRIPT
# ===============================
if __name__ == "__main__":
    main()
