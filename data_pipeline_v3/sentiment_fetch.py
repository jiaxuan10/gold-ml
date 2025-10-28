
import yaml
import os
import logging
from newsapi import NewsApiClient
from transformers import pipeline
from datetime import datetime, timedelta
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

if not cfg['sentiment'].get('enabled', False):
    logging.info('Sentiment module disabled in config')
    exit(0)

api_key = cfg['sentiment']['newsapi_key']
newsapi = NewsApiClient(api_key=api_key)

nlp = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

end = datetime.today()
start = end - timedelta(days=cfg['sentiment'].get('lookback_days', 7))

all_articles = []
for q in ['gold OR XAU OR gold price', 'inflation OR CPI', 'FED OR Fed funds OR interest rate']:
    res = newsapi.get_everything(q=q, from_param=start.strftime('%Y-%m-%d'), to=end.strftime('%Y-%m-%d'), language='en', sort_by='relevancy', page_size=100)
    articles = res.get('articles', [])
    for a in articles:
        all_articles.append({'publishedAt': a['publishedAt'], 'title': a['title'], 'description': a['description'], 'source': a['source']['name']})

if not all_articles:
    logging.info('No articles fetched')
    exit(0)

df = pd.DataFrame(all_articles)

def sentiment_score(text):
    try:
        out = nlp(text[:512])
        lab = out[0]['label']
        score = out[0]['score']
        return score if lab == 'POSITIVE' else -score
    except Exception as e:
        return 0.0

df['text'] = df['title'].fillna('') + '. ' + df['description'].fillna('')
df['sentiment'] = df['text'].apply(sentiment_score)
df['publishedAt'] = pd.to_datetime(df['publishedAt'])

# Aggregate to daily sentiment
df['Date'] = df['publishedAt'].dt.date
agg = df.groupby('Date')['sentiment'].mean().reset_index()
agg.to_parquet(os.path.join(cfg['paths']['raw_dir'],'sentiment_daily.parquet'), index=False)
logging.info('Sentiment saved')