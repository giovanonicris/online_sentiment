# 7/25/25 - CG adds Debug mode for easier debugging and testing
# 9/1/25 - CG optimizes GitHub Actions to do the ff: parallel processing, reduct CSV size, limit rates
# 9/9/25 - CG removes file splitting, optimizes for power bi, reduces csv size, keeps full summaries

import requests
import random
import re
import time
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser
from newspaper import Article, Config
import datetime as dt
import nltk
from googlenewsdecoder import new_decoderv1
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import chardet
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import cProfile
import csv

# IMPORTANT!!!
# DEBUG MODE SETTINGS - CHANGE THIS TO False WHEN RUNNING IN PROD
DEBUG_MODE = False
MAX_SEARCH_TERMS = 2 if DEBUG_MODE else None
MAX_ARTICLES_PER_TERM = 3 if DEBUG_MODE else 20
SKIP_ARTICLE_PROCESSING = True if DEBUG_MODE else False

# DEBUG META INFO
print("*" * 50)
print(f"DEBUG_MODE: {DEBUG_MODE}")
if DEBUG_MODE:
    print(f"   - Limited to {MAX_SEARCH_TERMS} search terms")
    print(f"   - Max {MAX_ARTICLES_PER_TERM} articles per term")
    print(f"   - Skip article processing: {SKIP_ARTICLE_PROCESSING}")
print(f"Script started at: {dt.datetime.now()}")
print(f"Working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")
print(f"Script file location: {os.path.abspath(__file__)}")
print("*" * 50)

# Set dates for today and yesterday
now = dt.date.today()
yesterday = now - dt.timedelta(days=1)

# Check and download nltk resources
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        print(f"Downloading missing NLTK resource: {resource}")
        nltk.download(resource)

# Create a list of random user agents
user_agent_list = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'
]

config = Config()
user_agent = random.choice(user_agent_list)
config.browser_user_agent = user_agent
config.enable_image_fetching = False  # disable image fetching for speed
config.request_timeout = 10 if DEBUG_MODE else 20
header = {'User-Agent': user_agent}

# set up requests session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# load existing dataset to avoid duplicate fetching
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
main_csv_path = os.path.join(output_dir, 'enterprise_risks_online_sentiment.csv')

print("*" * 50)
print(f"Script directory: {script_dir}")
print(f"Output directory: {output_dir}")
print(f"Main CSV path: {main_csv_path}")
print(f"Output directory exists: {os.path.exists(output_dir)}")
print("*" * 50)

# skip existing links check in debug mode for speed
if DEBUG_MODE:
    existing_links = set()
    print("DEBUG: Skipping existing links check for faster testing")
else:
    if os.path.exists(main_csv_path):
        existing_df = pd.read_csv(main_csv_path, usecols=lambda x: 'LINK' in x, encoding="utf-8")
        existing_links = set(existing_df["LINK"].dropna().str.lower().str.strip())
    else:
        existing_links = set()
        print("No existing CSV found - starting fresh")

# Load and decode search terms
try:
    read_file = pd.read_csv('EnterpriseRisksListEncoded.csv', encoding='utf-8', usecols=['ENTERPRISE_RISK_ID', 'SEARCH_TERM_ID', 'ENCODED_TERMS'])
    read_file['ENTERPRISE_RISK_ID'] = pd.to_numeric(read_file['ENTERPRISE_RISK_ID'], downcast='integer', errors='coerce')
    print(f"Successfully loaded EnterpriseRisksListEncoded.csv with {len(read_file)} rows")
except FileNotFoundError:
    print("ERROR!!! EnterpriseRisksListEncoded.csv not found!")
    exit(1)
except Exception as e:
    print(f"ERROR loading EnterpriseRisksListEncoded.csv: {e}")
    exit(1)

def process_encoded_search_terms(term):
    try:
        encoded_number = int(term)
        byte_length = (encoded_number.bit_length() + 7) // 8
        byte_rep = encoded_number.to_bytes(byte_length, byteorder='little')
        decoded_text = byte_rep.decode('utf-8')
        return decoded_text
    except (ValueError, UnicodeDecodeError, OverflowError):
        return None

read_file['SEARCH_TERMS'] = read_file['ENCODED_TERMS'].apply(process_encoded_search_terms)

# DEBUG FOR SEARCH TERMS
print("*" * 50)
print(f"Loaded {len(read_file)} total rows from file")
valid_terms = read_file['SEARCH_TERMS'].dropna()
print(f"Valid search terms ({len(valid_terms)}): {valid_terms.tolist()}")

# limit search terms for testing
if DEBUG_MODE and MAX_SEARCH_TERMS:
    valid_terms = valid_terms.head(MAX_SEARCH_TERMS)
    print(f"DEBUGGING - Limited to {len(valid_terms)} terms: {valid_terms.tolist()}")
elif len(valid_terms) > 100:
    valid_terms = valid_terms.head(100)  # limit terms in prod if too many
    print(f"Limited to {len(valid_terms)} terms to reduce load")
print("*" * 50)

# load filter_out_sources.csv
filter_out_path = 'filter_out_sources.csv'
if os.path.exists(filter_out_path):
    filter_out_df = pd.read_csv(filter_out_path, encoding='utf-8')
    filtered_sources = set(filter_out_df.iloc[:, 0].dropna().str.lower().str.strip())
    print(f"Loaded {len(filtered_sources)} filtered sources")
else:
    filtered_sources = set()
    print("filter_out_sources.csv not found - no source filtering")

# fetch articles concurrently
def fetch_articles(term):
    print(f"Processing search term: '{term}'")
    url_start = 'https://news.google.com/rss/search?q={'
    url_end = '}%20when%3A1d'
    try:
        time.sleep(0.5)  # rate limit to avoid 429 errors
        req = session.get(url_start + term + url_end, headers=header)
        soup = BeautifulSoup(req.text, 'xml')
        articles = []
        article_count = 0
        for item in soup.find_all("item")[:10]:  # limit to 10 items per term
            if article_count >= MAX_ARTICLES_PER_TERM:
                print(f"DEBUGGING - stopping at {MAX_ARTICLES_PER_TERM} articles for term '{term}'")
                break
            title_text = item.title.text.strip()
            encoded_url = item.link.text.strip()
            source_text = item.source.text.strip().lower()
            interval_time = 5
            decoded_url = new_decoderv1(encoded_url, interval=interval_time)
            if decoded_url.get("status"):
                decoded_url = decoded_url['decoded_url'].strip().lower()
                parsed_url = urlparse(decoded_url)
                domain_name = parsed_url.netloc.lower()
                if not any(domain_name.endswith(ext) for ext in ('.com', '.edu', '.org', '.net')):
                    if DEBUG_MODE:
                        print(f"Skipping {decoded_url} (Invalid domain extension)")
                    continue
                if source_text in filtered_sources:
                    if DEBUG_MODE:
                        print(f"Skipping article from {source_text} (Filtered source)")
                    continue
                if "/en/" in decoded_url:
                    if DEBUG_MODE:
                        print(f"Skipping {decoded_url} (Detected translated article)")
                    continue
                if decoded_url in existing_links:
                    if DEBUG_MODE:
                        print(f"Skipping {decoded_url} (Already exists)")
                    continue
                try:
                    published_date = parser.parse(item.pubDate.text).date()
                except (ValueError, TypeError):
                    published_date = None
                    print(f"WARNING! Date Error: {item.pubDate.text}")
                regex_pattern = re.compile('(https?):((|(\\\\))+[\w\d:#@%;$()~_?\+-=\\\.&]*)')
                domain_search = regex_pattern.search(str(item.source))
                articles.append({
                    'term': term,
                    'title': title_text,
                    'source': source_text,
                    'link': decoded_url,
                    'published': published_date,
                    'domain': domain_search.group(0) if domain_search else None
                })
                article_count += 1
            else:
                print("Error:", decoded_url['message'])
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Request error for term {term}: {e}")
        return []

# Fetch articles
print("Fetching articles concurrently...")
with ThreadPoolExecutor(max_workers=20) as executor:
    results = executor.map(fetch_articles, valid_terms)
    article_data = [item for sublist in results for item in sublist]

# Extract data into lists
search_terms = [item['term'] for item in article_data]
title = [item['title'] for item in article_data]
source = [item['source'] for item in article_data]
link = [item['link'] for item in article_data]
published = [item['published'] for item in article_data]
domain = [item['domain'] for item in article_data]
summary = []
keywords = []
sentiments = []
polarity = []

print('Created lists')

# DEBUG AFTER COLLECTING ARTICLES
print("*" * 50)
print(f"Found {len(link)} articles before processing")
print(f"Existing links count: {len(existing_links)}")
print("*" * 50)

# process articles concurrently
def process_article(article_link):
    print(f"Processing article: {article_link}")
    try:
        time.sleep(0.5)  # rate limit to avoid 429 errors
        req = session.get(article_link, headers=header)
        soup = BeautifulSoup(req.text, 'html.parser')
        article_text = ' '.join([p.text.strip() for p in soup.find_all('p')]) or ""
        if len(article_text) < 100:
            print(f"Skip article - short or missing text: {article_link}")
            article_text = ''
        analyzer = SentimentIntensityAnalyzer().polarity_scores(article_text)
        comp = analyzer['compound']
        sentiment = 'negative' if comp <= -0.05 else 'positive' if comp >= 0.05 else 'neutral'
        return {
            'summary': article_text,
            'keywords': [],  # empty list, will convert to string later
            'sentiment': sentiment,
            'polarity': f'{comp}'
        }
    except Exception as e:
        print(f"Processing failed for {article_link}: {e}")
        return {'summary': '', 'keywords': [], 'sentiment': 'neutral', 'polarity': '0.0'}

if SKIP_ARTICLE_PROCESSING:
    print("DEBUGGING - skipping slow article processing, using dummy data")
    for i in range(len(link)):
        summary.append("DEBUG: Article processing skipped for faster testing")
        keywords.append(["debug", "test"])
        sentiments.append('neutral')
        polarity.append('0.0')
    print(f"DEBUGGING - added dummy data for {len(link)} articles")
else:
    print("Processing articles concurrently with BeautifulSoup...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(process_article, link)
        for result in results:
            summary.append(result['summary'])
            keywords.append(result['keywords'])
            sentiments.append(result['sentiment'])
            polarity.append(result['polarity'])

# filter articles to last 24 hours
valid_indices = [i for i, date in enumerate(published) if date and date >= yesterday]
search_terms = [search_terms[i] for i in valid_indices]
title = [title[i] for i in valid_indices]
published = [published[i] for i in valid_indices]
link = [link[i] for i in valid_indices]
source = [source[i] for i in valid_indices]
summary = [summary[i] for i in valid_indices]
keywords = [','.join(k) for k in [keywords[i] for i in valid_indices]]  # convert to string
sentiments = [sentiments[i] for i in valid_indices]
polarity = [polarity[i] for i in valid_indices]

# Create DataFrame
alerts = pd.DataFrame({
    'SEARCH_TERMS': search_terms,
    'TITLE': title,
    'SUMMARY': summary,
    'KEYWORDS': keywords,
    'PUBLISHED_DATE': published,
    'LINK': link,
    'SOURCE': source,
    'SENTIMENT': sentiments,
    'POLARITY': polarity
})

print('Created sentiments')

# Merge with search terms data
joined_df = pd.merge(alerts, read_file, on='SEARCH_TERMS', how='left')
final_df = joined_df[['ENTERPRISE_RISK_ID', 'SEARCH_TERM_ID', 'TITLE', 'SUMMARY', 'KEYWORDS', 'PUBLISHED_DATE', 'LINK', 'SOURCE', 'SENTIMENT', 'POLARITY']]
final_df['LAST_RUN_TIMESTAMP'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print("*" * 50)
print(f"Processed {len(summary)} articles")
print(f"Final DataFrame shape: {final_df.shape}")
print(f"Final DataFrame columns: {final_df.columns.tolist()}")
if len(final_df) > 0:
    print("Sample of final data:")
    print(final_df.head(2))
else:
    print("WARNING!!! Final DataFrame is empty!")
print("*" * 50)

# Load existing data and combine
if os.path.exists(main_csv_path):
    existing_main_df = pd.read_csv(main_csv_path, parse_dates=['PUBLISHED_DATE'], encoding='utf-8')
    print(f"loaded existing CSV with {len(existing_main_df)} records")
else:
    existing_main_df = pd.DataFrame()
    print("No existing CSV found - starting fresh")

# DEBUG BEFORE SAVING
print("*" * 50)
if not final_df.empty:
    print(f"Saving {len(final_df)} new records")
else:
    print("WARNING!!! No new records to save!")
print("*" * 50)

combined_df = pd.concat([existing_main_df, final_df], ignore_index=True).drop_duplicates(subset=['TITLE', 'LINK', 'PUBLISHED_DATE'])

# rolling 4-month window
cutoff_date = dt.datetime.now() - dt.timedelta(days=4 * 30)
combined_df['PUBLISHED_DATE'] = pd.to_datetime(combined_df['PUBLISHED_DATE'], errors='coerce')

if combined_df['PUBLISHED_DATE'].isna().any():
    print("Warning-some rows have invalid PUBLISHED_DATE values.")

# separate current and old data
current_df = combined_df[combined_df['PUBLISHED_DATE'] >= cutoff_date].copy()
old_df = combined_df[combined_df['PUBLISHED_DATE'] < cutoff_date].copy()

# DEBUG AFTER COMBINING DATA
print("*" * 50)
print(f"Combined DataFrame shape: {combined_df.shape}")
print(f"Current DataFrame shape (after filtering): {current_df.shape}")
print("*" * 50)

# Save current data
current_df.sort_values(by='PUBLISHED_DATE', ascending=False).to_csv(main_csv_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
file_size = os.path.getsize(main_csv_path)
print(f"Updated main CSV with {len(current_df)} records.")
print(f"File size: {file_size} bytes")

# DEBUG VERIFY FILE
print("*" * 50)
if os.path.exists(main_csv_path):
    file_size = os.path.getsize(main_csv_path)
    print(f"Output file exists at: {main_csv_path}")
    print(f"File size: {file_size} bytes")
    if file_size > 0:
        print("File preview:")
        try:
            preview_df = pd.read_csv(main_csv_path).head(2)
            print(preview_df)
        except Exception as e:
            print(f"Could not preview file: {e}")
    else:
        print("File is empty!")
else:
    print("ERROR!!! Output file not created!")
print(f"Script completed at: {dt.datetime.now()}")
print("*" * 50)

# archive old data
if not DEBUG_MODE and not old_df.empty:
    old_df = old_df.sort_values(by='PUBLISHED_DATE')
    archive_path = os.path.join(output_dir, 'enterprise_risks_sentiment_archive.csv')
    old_df.to_csv(archive_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
    print(f"Archived {len(old_df)} records to {archive_path}.")
elif DEBUG_MODE:
    print("DEBUGGING- Skipping archival process")
