#!/usr/bin/env python3
# src/live/news_service.py

import os
import time
import json
import pandas as pd
import feedparser 
from datetime import datetime, timedelta, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import warnings

warnings.filterwarnings("ignore")


RSS_URLS = [
    "https://news.google.com/rss/search?q=gold%20price&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=federal%20reserve&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=inflation&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=usd%20index&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=geopolitical&hl=en-US&gl=US&ceid=US:en"
]
IRRELEVANT_WORDS = ["fashion", "jewelry", "sport", "design", "music", "deal", "cricket", "football"]

MODEL_NAME = "yiyanghkust/finbert-tone"
REFRESH_INTERVAL = 1800 


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "final")
SENTIMENT_STATE_FILE = os.path.join(DATA_DIR, "current_sentiment.json")
NEWS_LIST_FILE = os.path.join(DATA_DIR, "latest_news_headlines.csv")

def load_finbert():
    print(" Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def parse_pubdate(date_str):
    try:
        dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.replace(tzinfo=timezone.utc)
    except:
        return datetime.now(timezone.utc) 

def fetch_and_analyze(nlp):
    print(f" Fetching REAL-TIME Google News at {datetime.now().strftime('%H:%M')}...")
    
    articles_data = [] 
    all_scores = []
    
    for url in RSS_URLS:
        try:
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:20]: 
                title = entry.title
                link = entry.link
                pub_date_str = entry.published
                source = entry.source.title if 'source' in entry else "GoogleNews"
                
                if any(w in title.lower() for w in IRRELEVANT_WORDS): continue
                
                if any(d['Title'] == title for d in articles_data): continue
                

                res = nlp(title[:512])[0]
                label = res['label']
                prob = res['score']
                
                score = prob if label == 'Positive' else -prob if label == 'Negative' else 0
                

                if entry.published_parsed:
                    dt = datetime.fromtimestamp(time.mktime(entry.published_parsed), timezone.utc)
                else:
                    dt = datetime.now(timezone.utc)
                
                date_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                all_scores.append(score)
                articles_data.append({
                    "Date": date_iso,
                    "Source": source,
                    "Title": title,
                    "Label": label,
                    "Score": round(score, 4),
                    "URL": link
                })
                
        except Exception as e:
            print(f" Error processing RSS: {e}")

    if articles_data:
        new_df = pd.DataFrame(articles_data)
        
        if os.path.exists(NEWS_LIST_FILE):
            try:
                old_df = pd.read_csv(NEWS_LIST_FILE)
                combined_df = pd.concat([new_df, old_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["Title"], keep='first')
            except:
                combined_df = new_df
        else:
            combined_df = new_df
            
        combined_df = combined_df.sort_values("Date", ascending=False)
        combined_df = combined_df.head(100) 
        
        temp_file = NEWS_LIST_FILE + ".tmp"
        try:
            combined_df.to_csv(temp_file, index=False)
            if os.path.exists(NEWS_LIST_FILE):
                try:
                    os.replace(temp_file, NEWS_LIST_FILE)
                except PermissionError:
                    print(f" File Locked by Excel! Skipping update this time.")
            else:
                os.rename(temp_file, NEWS_LIST_FILE)
            print(f" Updated news list. Total count: {len(combined_df)}")
        except Exception as e:
            print(f" Save Failed: {e}")
    else:
        print(" No news found (Check internet connection).")

    if all_scores:
        active_scores = [s for s in all_scores if abs(s) > 0.2]
        
        if active_scores:
            avg_sentiment = sum(active_scores) / len(active_scores)
        else:
            avg_sentiment = 0.0
    else:
        avg_sentiment = 0.0
        
    return avg_sentiment

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    nlp = load_finbert()
    
    while True:
        try:
            score = fetch_and_analyze(nlp)
            
            state = {
                "last_updated": str(datetime.now()),
                "sentiment_score": round(score, 4), 
                "status": "Bullish" if score > 0.1 else "Bearish" if score < -0.1 else "Neutral"
            }
            
            with open(SENTIMENT_STATE_FILE, "w") as f:
                json.dump(state, f)
                
            print(f" Sentiment Updated: {score:.4f} ({state['status']})")
            print(f" Sleeping for 30 mins...")
            time.sleep(REFRESH_INTERVAL)
            
        except Exception as e:
            print(f" Error: {e}")
            time.sleep(60)