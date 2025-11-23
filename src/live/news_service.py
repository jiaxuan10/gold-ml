#!/usr/bin/env python3
# src/live/news_service.py

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import warnings

warnings.filterwarnings("ignore")

# ====== CONFIG ======
NEWS_API_KEY = "e450698ba6784d9f983422b99b756214" 
KEYWORDS = ['"gold price"', '"federal reserve"', '"inflation"', '"usd index"', '"geopolitical"']
IRRELEVANT_WORDS = ["fashion", "jewelry", "sport", "design", "music", "deal"]
MODEL_NAME = "yiyanghkust/finbert-tone"
REFRESH_INTERVAL = 1800  # 30分钟刷新一次

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "final")
SENTIMENT_STATE_FILE = os.path.join(DATA_DIR, "current_sentiment.json")
NEWS_LIST_FILE = os.path.join(DATA_DIR, "latest_news_headlines.csv") # 🆕 新闻列表文件

def load_finbert():
    print("🚀 Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def fetch_and_analyze(nlp):
    print(f"📡 Fetching news at {datetime.now().strftime('%H:%M')}...")
    to_date = datetime.now()
    from_date = to_date - timedelta(days=2) 
    
    articles_data = [] # 🆕 用于存储原始新闻
    all_scores = []
    
    for keyword in KEYWORDS:
        try:
            # Fetch
            url = (f"https://newsapi.org/v2/everything?q={keyword}&language=en&"
                   f"from={from_date.date()}&sortBy=publishedAt&pageSize=100&apiKey={NEWS_API_KEY}")
            resp = requests.get(url, timeout=10)
            articles = resp.json().get("articles", [])
            
            for art in articles:
                title = art.get("title", "")
                source = art.get("source", {}).get("name", "Unknown")
                url_link = art.get("url", "")
                date_str = art.get("publishedAt", "")
                
                # Filter
                if any(w in title.lower() for w in IRRELEVANT_WORDS): continue
                
                # Deduplicate (simple check in current batch)
                if any(d['title'] == title for d in articles_data): continue
                
                # Analyze
                res = nlp(title[:512])[0]
                label = res['label']
                prob = res['score']
                
                # Mapping
                score = prob if label == 'Positive' else -prob if label == 'Negative' else 0
                
                all_scores.append(score)
                articles_data.append({
                    "Date": date_str,
                    "Source": source,
                    "Title": title,
                    "Label": label,
                    "Score": round(score, 4),
                    "URL": url_link
                })
                
        except Exception as e:
            print(f"⚠️ Error {keyword}: {e}")

    # 1. Save List to CSV 🆕
    if articles_data:
        df = pd.DataFrame(articles_data)
        # 按时间倒序
        df = df.sort_values("Date", ascending=False)
        df.to_csv(NEWS_LIST_FILE, index=False)
        print(f"✅ Saved {len(df)} articles to CSV.")
    else:
        print("⚠️ No relevant articles found.")

    # 2. Calculate Average for Inference
    if all_scores:
        avg_sentiment = sum(all_scores) / len(all_scores)
    else:
        avg_sentiment = 0.0
        
    return avg_sentiment

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    nlp = load_finbert()
    
    while True:
        try:
            score = fetch_and_analyze(nlp)
            
            # 保存状态供 inference_service 读取
            state = {
                "last_updated": str(datetime.now()),
                "sentiment_score": round(score, 4), 
                "status": "Bullish" if score > 0.1 else "Bearish" if score < -0.1 else "Neutral"
            }
            
            with open(SENTIMENT_STATE_FILE, "w") as f:
                json.dump(state, f)
                
            print(f"📊 Sentiment Updated: {score:.4f} ({state['status']})")
            
            # 倒计时显示
            print(f"💤 Sleeping for 30 mins...")
            time.sleep(REFRESH_INTERVAL)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(60)