# 7/25/25 - CG adds a Debug mode for easier debugging and testing

import requests
import random
import re
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser
from newspaper import Article
from newspaper import Config
import datetime as dt
import nltk
from googlenewsdecoder import new_decoderv1
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import chardet
from urllib.parse import urlparse

# IMPORTANT!!!
# DEBUG MODE SETTINGS - CHANGE THIS TO False WHEN RUNNING IN PROD
DEBUG_MODE = False
MAX_SEARCH_TERMS = 2 if DEBUG_MODE else None
MAX_ARTICLES_PER_TERM = 3 if DEBUG_MODE else None
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

# check and download both punkt and punkt_tab
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
# DEBUG: set faster request timeout in debug mode
config.request_timeout = 10 if DEBUG_MODE else 20
header = {'User-Agent': user_agent}

# load existing dataset to avoid duplicate fetching
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
main_csv_path = os.path.join(output_dir, 'emerging_risks_online_sentiment.csv')

print("*" * 50)
print("PATH INFORMATION:")
print(f"Script directory: {script_dir}")
print(f"Output directory: {output_dir}")
print(f"Main CSV path: {main_csv_path}")
print(f"Output directory exists: {os.path.exists(output_dir)}")
print("*" * 50)

# Skip existing links check in debug mode for speed
if DEBUG_MODE:
    existing_links = set()
    print("DEBUGGING - skipping existing links check for faster testing")
else:
    if os.path.exists(main_csv_path):
        existing_df = pd.read_csv(main_csv_path, usecols=lambda x: 'LINK' in x, encoding="utf-8")
        existing_links = set(existing_df["LINK"].dropna().str.lower().str.strip())
    else:
        existing_links = set()

# encode-decode search terms
try:
    read_file = pd.read_csv('EmergingRisksListEncoded.csv', encoding='utf-8', usecols=['EMERGING_RISK_ID', 'SEARCH_TERM_ID', 'ENCODED_TERMS'])
    read_file['EMERGING_RISK_ID'] = pd.to_numeric(read_file['EMERGING_RISK_ID'], downcast='integer', errors='coerce')
    print(f"Successfully loaded: EmergingRisksListEncoded.csv with {len(read_file)} rows")
except FileNotFoundError:
    print("ERROR!!! EmergingRisksListEncoded.csv not found!")
    exit(1)
except Exception as e:
    print(f"ERROR!!! loading EmergingRisksListEncoded.csv: {e}")
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
print("SEARCH TERMS INFORMATION:")
print(f"Loaded {len(read_file)} total rows from file")
valid_terms = read_file['SEARCH_TERMS'].dropna()
print(f"Valid search terms ({len(valid_terms)}): {valid_terms.tolist()}")

# DEBUG MODE: Limit search terms for testing
if DEBUG_MODE and MAX_SEARCH_TERMS:
    valid_terms = valid_terms.head(MAX_SEARCH_TERMS)
    print(f"DEBUG: Limited to {len(valid_terms)} terms: {valid_terms.tolist()}")
print("*" * 50)

# prep lists to store new entries
search_terms = []
title = []
published = []
link = []
domain = []
source = []
summary = []
keywords = []
sentiments = []
polarity = []

# load filter_out_sources.csv file
filter_out_path = 'filter_out_sources.csv'
if os.path.exists(filter_out_path):
    filter_out_df = pd.read_csv(filter_out_path, encoding='utf-8')
    filtered_sources = set(filter_out_df.iloc[:, 0].dropna().str.lower().str.strip())
    print(f"Loaded {len(filtered_sources)} filtered sources")
else:
    filtered_sources = set()
    print("filter_out_sources.csv not found - no source filtering")
    
# Grab Google links
url_start = 'https://news.google.com/rss/search?q={'
url_end = '}%20when%3A1d'  # fetch only recent articles

for term in valid_terms:
    print(f"Processing search term: '{term}'")
    try:
        req = requests.get(url=url_start + term + url_end, headers=header)
        soup = BeautifulSoup(req.text, 'xml')
        
        article_count = 0
        for item in soup.find_all("item"):
            # Debug mode: limit articles per term
            if DEBUG_MODE and MAX_ARTICLES_PER_TERM and article_count >= MAX_ARTICLES_PER_TERM:
                print(f"DEBUG: Stopping at {MAX_ARTICLES_PER_TERM} articles for term '{term}'")
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

                # FILTER LOGIC SEQUENCE
                # 1. Valid domain extension only
                valid_extensions = ('.com', '.edu', '.org', '.net')
                if not any(domain_name.endswith(ext) for ext in valid_extensions):
                    if DEBUG_MODE:
                        print(f"Skipping {decoded_url} (Invalid domain extension)")
                    continue

                # 2. Check if the source name is in filter-out list
                if source_text in filtered_sources:
                    if DEBUG_MODE:
                        print(f"Skipping article from {source_text} (Filtered source)")
                    continue

                # 3. Skip articles if the URL contains '/en/' (translated articles)
                if "/en/" in decoded_url:
                    if DEBUG_MODE:
                        print(f"Skipping {decoded_url} (Detected translated article)")
                    continue

                if decoded_url in existing_links:
                    if DEBUG_MODE:
                        print(f"Skipping {decoded_url} (Already exists)")
                    continue
                
                title.append(title_text)
                search_terms.append(term)
                source.append(source_text)
                link.append(decoded_url)
                
                # date parsing
                try:
                    published.append(parser.parse(item.pubDate.text).date())
                except (ValueError, TypeError):
                    published.append(None)
                    print(f"WARNING! Date Error: {item.pubDate.text}")

                regex_pattern = re.compile('(https?):((|(\\\\))+[\w\d:#@%;$()~_?\+-=\\\.&]*)')
                domain_search = regex_pattern.search(str(item.source))
                domain.append(domain_search.group(0) if domain_search else None)
                
                article_count += 1
                
            else:
                print("Error:", decoded_url['message'])
    except requests.exceptions.RequestException as e:
        print(f"Request error for term {term}: {e}")

print('Created lists')

# DEBUG AFTER COLLECTING ARTICLES
print("*" * 50)
print(f"Found {len(link)} articles before processing")
print(f"Existing links count: {len(existing_links)}")
print("*" * 50)

# Article processing - WITH DEBUG MODE OPTION
if SKIP_ARTICLE_PROCESSING:
    print("DEBUGGING: Skipping slow article processing, using dummy data")
    for i in range(len(link)):
        summary.append("DEBUG: Article processing skipped for faster testing")
        keywords.append(["debug", "test"])
        sentiments.append('neutral')
        polarity.append('0.0')
    print(f"DEBUGGING: Added dummy data for {len(link)} articles")
else:
    print("Processing articles with newspaper library...")
    for i, article_link in enumerate(link):
        print(f"Processing article {i+1}/{len(link)}: {article_link}")
        article = Article(article_link, config=config)
        try:
            article.download()
            article.parse()
            article.nlp()
        except Exception as e:
            print(f"nlp failed for {article_link}: {e}")
        
        article_text = (article.summary or article.text or "").strip()
        if len(article_text) < 100:
            print(f"Skip article - short or missing text: {article_link}")
            article_text = ''
        
        summary.append(article_text)
        keywords.append(article.keywords)
        
        # Sentiment analysis
        analyzer = SentimentIntensityAnalyzer().polarity_scores(article.summary)
        comp = analyzer['compound']
        if comp <= -0.05:
            sentiments.append('negative')
            polarity.append(f'{comp}')
        elif -0.05 < comp < 0.05:
            sentiments.append('neutral')
            polarity.append(f'{comp}')
        elif comp >= 0.05:
            sentiments.append('positive')
            polarity.append(f'{comp}')

# Create DataFrame
alerts = pd.DataFrame({
    'SEARCH_TERMS': search_terms,
    'TITLE': title,
    'SUMMARY': summary,
    'KEYWORDS': keywords,
    'PUBLISHED_DATE': published,
    'LINK': link,
    'SOURCE': source,
    'SOURCE_URL': domain,
    'SENTIMENT': sentiments,
    'POLARITY': polarity
})

print('Created sentiments')

# merge new alerts with search terms data
joined_df = pd.merge(alerts, read_file, on='SEARCH_TERMS', how='left')
final_df = joined_df[['EMERGING_RISK_ID', 'SEARCH_TERM_ID', 'TITLE', 'SUMMARY', 'KEYWORDS', 'PUBLISHED_DATE', 'LINK', 'SOURCE', 'SOURCE_URL', 'SENTIMENT', 'POLARITY']]
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

# load existing data and combine with new entries
if os.path.exists(main_csv_path):
    existing_main_df = pd.read_csv(main_csv_path, parse_dates=['PUBLISHED_DATE'], encoding='utf-8')
    print(f"✓ Loaded existing CSV with {len(existing_main_df)} records")
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
    print("Warning: Some rows have invalid PUBLISHED_DATE values.")

# separate current and old data
current_df = combined_df[combined_df['PUBLISHED_DATE'] >= cutoff_date].copy()
old_df = combined_df[combined_df['PUBLISHED_DATE'] < cutoff_date].copy()

# DEBUG AFTER COMBINING DATA
print("*" * 50)
print(f"Combined DataFrame shape: {combined_df.shape}")
print(f"Current DataFrame shape (after filtering): {current_df.shape}")
print("*" * 50)

# save current data
current_df.sort_values(by='PUBLISHED_DATE', ascending=False).to_csv(main_csv_path, index=False, encoding='utf-8')
print(f"Updated main CSV with {len(current_df)} records.")

# DEBUG VERIFY FILE
print("*" * 50)
if os.path.exists(main_csv_path):
    file_size = os.path.getsize(main_csv_path)
    print(f"✓ Output file exists at: {main_csv_path}")
    print(f"✓ File size: {file_size} bytes")
    
    # show first few lines of the file, for quick check
    if file_size > 0:
        print("Preview file:")
        try:
            preview_df = pd.read_csv(main_csv_path).head(2)
            print(preview_df)
        except Exception as e:
            print(f"Could not preview file: {e}")
    else:
        print("File is empty!!!")
else:
    print("ERROR - output file was not created")
print(f"Script completed at: {dt.datetime.now()}")
print("*" * 50)

# archive logic - split old data into 4-month chunks (skipping this in debug mode)
if not DEBUG_MODE and not old_df.empty:
    old_df = old_df.sort_values(by='PUBLISHED_DATE')
    start_date = old_df['PUBLISHED_DATE'].min()
    end_date = old_df['PUBLISHED_DATE'].max()

    archive_num = 1
    window_start = start_date
    while window_start < end_date:
        window_end = window_start + pd.DateOffset(months=4)
        mask = (old_df['PUBLISHED_DATE'] >= window_start) & (old_df['PUBLISHED_DATE'] < window_end)
        archive_chunk = old_df.loc[mask]
        if not archive_chunk.empty:
            archive_path = os.path.join(output_dir, f'emerging_risks_sentiment_archive_{archive_num}.csv')
            archive_chunk.to_csv(archive_path, index=False, encoding='utf-8')
            print(f"Archived {len(archive_chunk)} records to {archive_path}.")
            archive_num += 1
        window_start = window_end
elif DEBUG_MODE:
    print("DEBUGGING - skipping archival process")
