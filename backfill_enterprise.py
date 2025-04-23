import os
import pandas as pd
from newspaper import Article, Config
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime as dt
import nltk
import random

# check if NLTK dependencies are present
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        print(f"Downloading missing NLTK resource: {resource}")
        nltk.download(resource)

# config file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'online_sentiment/output')
main_csv_path = os.path.join(output_dir, 'enterprise_risks_online_sentiment.csv')
temp_csv_path = os.path.join(output_dir, 'enterprise_risks_online_sentiment_temp.csv')

# load dataset
if not os.path.exists(main_csv_path):
    print("Main CSV does not exist, exiting.")
    exit(0)
df = pd.read_csv(main_csv_path, encoding='utf-8')
df['PUBLISHED_DATE'] = pd.to_datetime(df['PUBLISHED_DATE'], errors='coerce')
print(df.dtypes)

# BACKFILLING REQUIREMENTS
start_of_window = pd.to_datetime("2025-03-17")
end_of_window = pd.to_datetime("2025-04-20")

# Include logic for blank or short summary
missing_df = df[
    (df['PUBLISHED_DATE'] >= start_of_window) &
    (df['PUBLISHED_DATE'] <= end_of_window) &
    ((df['SUMMARY'].isna()) | (df['SUMMARY'].str.strip() == '') | (df['SUMMARY'].str.len() < 40))
]

if missing_df.empty:
    print("No missing sentiment to backfill.")
    exit(0)

# group by date and find latest group with missing values
dates_missing = missing_df['PUBLISHED_DATE'].dropna().dt.date.unique()
dates_missing = sorted(dates_missing, reverse=True)

# attempt 3-day batch starting from latest unfilled
latest = dates_missing[0]
three_day_range = pd.date_range(end=latest, periods=3).date
print(f"Selected 3-day batch: {three_day_range[0]} to {three_day_range[-1]}")

# filter target rows
target_mask = df['PUBLISHED_DATE'].dt.date.isin(three_day_range) & (
    (df['SUMMARY'].isna()) | (df['SUMMARY'].str.strip() == '') | (df['SUMMARY'].str.len() < 40)
)
target_df = df[target_mask].copy()

# parse articles using links
user_agent_list = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0)',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5)',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0)'
]
user_agent = random.choice(user_agent_list)
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 20
analyzer = SentimentIntensityAnalyzer()

updated = 0


for idx, row in target_df.iterrows():
    try:
        url = row['LINK']
        article = Article(url, config=config)
        # start with newspaper's article
        try:
            article.download()
            article.parse()
            article.nlp()
        #falls back to requests.get() if we get a 403 error/ Cloudflare block
        except Exception as e:
            print(f"Direct download failed for {url}, trying fallback: {e}")
            response = requests.get(url, headers={'User-Agent': user_agent}, timeout=20)
            article.set_html(response.text)
            article.parse()
            article.nlp()
        summary = article.summary.strip() if article.summary else article.text.strip()
        if not summary or len(summary) < 40:
            continue
        score = analyzer.polarity_scores(summary)['compound']
        sentiment = (
            'positive' if score >= 0.05 else
            'negative' if score <= -0.05 else
            'neutral'
        )

        df.at[idx, 'SUMMARY'] = summary
        df.at[idx, 'SENTIMENT'] = sentiment
        df.at[idx, 'POLARITY'] = score
        df.at[idx, 'LAST_RUN_TIMESTAMP'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        updated += 1
    except Exception as e:
        print(f"Unable to process {row['LINK']}: {e}")

if updated == 0:
    print("No rows updated — possibly all articles failed or too short.")
else:
    print(f"Updated {updated} rows. Writing to temp CSV...")
    df.to_csv(temp_csv_path, index=False, encoding='utf-8')
    os.replace(temp_csv_path, main_csv_path)
    print("Main CSV is overwritten with updated data.")
