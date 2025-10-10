"""
news.py
------------------------------------------------
Collects financial news using NewsAPI, performs sentiment analysis with FinBERT,
filters for gold-related and financial-market-relevant articles,
and outputs a cleaned CSV file: news_with_sentiment.csv

Author: Lim Jia Xuan
"""

import os
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import requests
import html
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ===============================
# 1️⃣ CONFIGURATION
# ===============================

# NEWS_API_KEY = "e450698ba6784d9f983422b99b756214"  # <-- paste your own key here

NEWS_API_KEY = "6d0099a1a01b4eed9cb1787189807f3a"  # <-- paste your own key here

KEYWORDS = ["gold", "economy", "inflation", "usd", "fed"]
DAYS_BACK = 14
MODEL_NAME = "yiyanghkust/finbert-tone"
OUTPUT_FILE = "news_with_sentiment.csv"

# Financial sources whitelist (you can extend it)
FINANCIAL_SOURCES = [
    "Reuters", "Bloomberg", "CNBC", "MarketWatch", "Financial Times", "Livemint",
    "Business Insider", "Investing.com", "Yahoo Finance", "ETF Daily News", "Forbes"
]

# Gold/Finance-related keyword filters
RELEVANT_KEYWORDS = [
    "gold price", "gold prices", "gold market", "gold futures", "gold demand", "gold rate",
    "gold investment", "gold bullion", "gold trade", "goldman sachs", "central bank",
    "inflation", "interest rate", "usd", "federal reserve", "monetary policy"
]


# ===============================
# 2️⃣ FETCH NEWS DATA
# ===============================
def fetch_news(keywords, days_back=14, api_key=None, page_size=100):
    """
    Fetch financial news headlines from NewsAPI.
    Returns a DataFrame with columns: date, source, title, description, url
    """
    if api_key is None:
        raise ValueError("❌ Missing NewsAPI key. Get one at https://newsapi.org/")

    all_articles = []
    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=days_back)

    print(f"\n📰 Fetching news from {from_date.date()} to {to_date.date()} ...\n")

    for keyword in keywords:
        print(f"🔍 Searching keyword: {keyword}")
        for page in range(1, 6):  # 5 pages × 100 results = 500 per keyword
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={keyword}&language=en&from={from_date.date()}&to={to_date.date()}&"
                f"pageSize={page_size}&page={page}&sortBy=publishedAt&apiKey={api_key}"
            )
            resp = requests.get(url)
            if resp.status_code != 200:
                print(f"⚠️ Request failed (page {page}): {resp.status_code}")
                break
            data = resp.json().get("articles", [])
            if not data:
                break
            for article in data:
                all_articles.append({
                    "date": article.get("publishedAt", "")[:10],
                    "source": article.get("source", {}).get("name", ""),
                    "title": html.unescape(article.get("title", "")),
                    "description": html.unescape(article.get("description", "")),
                    "url": article.get("url", "")
                })
            time.sleep(1)  # avoid hitting API rate limit

    df = pd.DataFrame(all_articles)
    df.drop_duplicates(subset=["title"], inplace=True)
    df.dropna(subset=["title"], inplace=True)
    print(f"\n✅ Collected {len(df)} total articles before filtering.\n")
    return df


# ===============================
# 3️⃣ FILTER NEWS (Relevance Filter)
# ===============================
def filter_relevant_news(df):
    """Keep only financial or gold-related news based on keywords & sources."""
    df["title_lower"] = df["title"].str.lower()

    # Filter by keywords in title
    mask_keywords = df["title_lower"].apply(
        lambda t: any(k in t for k in RELEVANT_KEYWORDS)
    )

    # Filter by trusted financial sources
    mask_sources = df["source"].apply(
        lambda s: any(fin.lower() in str(s).lower() for fin in FINANCIAL_SOURCES)
    )

    filtered_df = df[mask_keywords | mask_sources].copy()
    filtered_df.drop(columns=["title_lower"], inplace=True)

    print(f"✅ Filtered down to {len(filtered_df)} relevant financial/gold articles.\n")
    return filtered_df


# ===============================
# 4️⃣ LOAD FinBERT MODEL
# ===============================
def load_finbert():
    print("🚀 Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("✅ FinBERT loaded.\n")
    return nlp


# ===============================
# 5️⃣ RUN SENTIMENT ANALYSIS
# ===============================
def analyze_sentiment(news_df, nlp_pipeline):
    print("🔍 Analyzing sentiment for each headline...\n")
    results = []
    for text in tqdm(news_df["title"].tolist()):
        try:
            pred = nlp_pipeline(text[:512])[0]  # truncate long titles
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
# 6️⃣ MAIN PIPELINE
# ===============================
def main():
    try:
        # Step 1: Fetch news
        raw_df = fetch_news(KEYWORDS, DAYS_BACK, NEWS_API_KEY)

        # Step 2: Filter relevant ones
        filtered_df = filter_relevant_news(raw_df)

        if len(filtered_df) == 0:
            print("⚠️ No relevant news found after filtering.")
            return

        # Step 3: Load FinBERT
        finbert_pipeline = load_finbert()

        # Step 4: Analyze sentiment
        analyzed_df = analyze_sentiment(filtered_df, finbert_pipeline)

        # Step 5: Save output
        analyzed_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Sentiment analysis complete! Saved to {OUTPUT_FILE}\n")
        print(analyzed_df.head(10))

    except Exception as e:
        print("❌ Error:", e)


if __name__ == "__main__":
    main()
