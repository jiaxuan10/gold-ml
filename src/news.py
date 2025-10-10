"""
generate_finbert_sentiment.py
--------------------------------------
Collects financial news using NewsAPI, performs sentiment analysis with FinBERT,
and outputs a CSV file: news_with_sentiment.csv

Author: Lim Jia Xuan
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import time
import requests

# ===============================
# 1️⃣  CONFIGURATION
# ===============================

# 👉 You need a free API key from https://newsapi.org/
NEWS_API_KEY = "e450698ba6784d9f983422b99b756214"

# Search keywords
KEYWORDS = [
    '"gold price"', '"gold futures"', '"gold market"', '"gold investment"',
    '"gold bullion"', '"central bank"', '"interest rate"', '"usd index"',
    '"inflation rate"', '"federal reserve"', '"monetary policy"'
]
# Date range (past N days)
DAYS_BACK = 14

# FinBERT model
MODEL_NAME = "yiyanghkust/finbert-tone"

# Output file
OUTPUT_FILE = "news_with_sentiment.csv"


# ===============================
# 2️⃣  FETCH NEWS DATA
# ===============================
def fetch_news(keywords, days_back=14, api_key=None, page_size=100):
    """
    Fetch financial news headlines from NewsAPI.
    Returns a DataFrame with columns: date, source, title, description, url
    """
    if api_key is None:
        raise ValueError("❌ Missing NewsAPI key. Get one at https://newsapi.org/")

    all_articles = []
    to_date = datetime.utcnow()
    from_date = to_date - timedelta(days=days_back)

    print(f"\n📰 Fetching news from {from_date.date()} to {to_date.date()} ...\n")

    for keyword in keywords:
        print(f"🔍 Searching keyword: {keyword}")
        for page in range(1, 6):  # up to 5 pages × 100 results = 500 per keyword
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
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", "")
                })
            time.sleep(1)  # avoid rate limits

    df = pd.DataFrame(all_articles)
    df.drop_duplicates(subset=["title"], inplace=True)
    df.dropna(subset=["title"], inplace=True)
    print(f"\n✅ Collected {len(df)} news articles.\n")
    return df


# ===============================
# 3️⃣  LOAD FinBERT MODEL
# ===============================
def load_finbert():
    print("🚀 Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("✅ FinBERT loaded.\n")
    return nlp


# ===============================
# 4️⃣  RUN SENTIMENT ANALYSIS
# ===============================
def analyze_sentiment(news_df, nlp_pipeline):
    print("🔍 Analyzing sentiment for each headline...\n")
    results = []
    for text in tqdm(news_df["title"].tolist()):
        try:
            pred = nlp_pipeline(text[:512])[0]  # truncate if too long
            results.append(pred)
        except Exception as e:
            print("Error:", e)
            results.append({"label": "neutral", "score": 0.0})

    news_df["label"] = [r["label"] for r in results]
    news_df["score"] = [r["score"] for r in results]

    # Map to numeric value
    mapping = {"positive": 1, "neutral": 0, "negative": -1}
    news_df["sent_value"] = news_df["label"].map(mapping) * news_df["score"]
    return news_df

# Add this after fetch_news()
def filter_financial_relevance(df):
    irrelevant_words = [
        "fashion", "music", "art", "gaming", "movie", "design",
        "phone", "wearable", "case", "apparel", "celebrity",
        "casetify", "murakami", "garmin", "android", "smartwatch"
    ]

    df["title_lower"] = df["title"].str.lower()
    mask_irrelevant = df["title_lower"].apply(
        lambda t: any(word in t for word in irrelevant_words)
    )
    df = df[~mask_irrelevant].drop(columns=["title_lower"])
    print(f"✅ Filtered out {mask_irrelevant.sum()} irrelevant news.\n")
    return df

# ===============================
# 5️⃣  MAIN PIPELINE
# ===============================
def main():
    try:
        news_df = fetch_news(KEYWORDS, DAYS_BACK, NEWS_API_KEY)
        news_df = filter_financial_relevance(news_df)
        finbert_pipeline = load_finbert()
        analyzed_df = analyze_sentiment(news_df, finbert_pipeline)

        analyzed_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Sentiment analysis complete! Saved to {OUTPUT_FILE}\n")

        print(analyzed_df.head(10))
    except Exception as e:
        print("❌ Error:", e)


if __name__ == "__main__":
    main()
