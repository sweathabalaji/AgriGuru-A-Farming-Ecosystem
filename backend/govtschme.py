import feedparser
import requests
import time
from translate import Translator
import ssl
import json
from bs4 import BeautifulSoup
from datetime import datetime
import urllib3
import certifi

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize session with proper SSL context
session = requests.Session()
session.verify = certifi.where()

# RSS Feed sources
SCHEME_FEEDS = [
    ("https://www.thehindu.com/sci-tech/agriculture/feeder/default.rss", "The Hindu Agriculture"),
    ("https://www.agrifarming.in/rss", "AgriFarming"),
    ("https://www.downtoearth.org.in/rss/agriculture", "Down To Earth Agriculture"),
]

def make_request(url, verify=True):
    """Make HTTP request with proper error handling and SSL verification"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # First try with SSL verification
        if verify:
            response = session.get(url, headers=headers, timeout=15)
        else:
            # If SSL verification fails, try without verification
            response = session.get(url, headers=headers, timeout=15, verify=False)
            
        response.raise_for_status()
        return response.text
    except requests.exceptions.SSLError:
        # If SSL verification fails, retry without verification
        if verify:
            return make_request(url, verify=False)
        raise
    except Exception as e:
        print(f"Error making request to {url}: {str(e)}")
        return None

# Tamil translations for common terms
TAMIL_TRANSLATIONS = {
    'January': 'ஜனவரி',
    'February': 'பிப்ரவரி',
    'March': 'மார்ச்',
    'April': 'ஏப்ரல்',
    'May': 'மே',
    'June': 'ஜூன்',
    'July': 'ஜூலை',
    'August': 'ஆகஸ்ட்',
    'September': 'செப்டம்பர்',
    'October': 'அக்டோபர்',
    'November': 'நவம்பர்',
    'December': 'டிசம்பர்',
    'The Hindu': 'த ஹிந்து',
    'Agriculture': 'வேளாண்மை',
    'AgriFarming': 'அக்ரிஃபார்மிங்',
    'Down To Earth': 'டவுன் டு எர்த்',
    'Scheme': 'திட்டம்',
    'Schemes': 'திட்டங்கள்',
    'Farming': 'விவசாயம்',
    'Farmer': 'விவசாயி',
    'Farmers': 'விவசாயிகள்'
}

def translate_date(date_str):
    """Translate date to Tamil format"""
    try:
        # Parse the date string
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        # Format it in Tamil
        tamil_month = TAMIL_TRANSLATIONS.get(date_obj.strftime('%B'), date_obj.strftime('%B'))
        return f"{date_obj.day} {tamil_month} {date_obj.year}"
    except:
        return date_str

def translate_source(source):
    """Translate source name to Tamil"""
    translated_source = source
    for eng, tam in TAMIL_TRANSLATIONS.items():
        if eng in source:
            translated_source = translated_source.replace(eng, tam)
    return translated_source

def translate_to_tamil(text):
    """Translate text to Tamil with fallback mechanisms"""
    try:
        if not text or not text.strip():
            return text
            
        translator = Translator(from_lang="en", to_lang="ta")
        translation = translator.translate(text)
        
        # Check if we hit the translation limit
        if "MYMEMORY WARNING" in translation:
            print("Translation limit reached, using basic translation")
            # Try to translate using our dictionary first
            for eng, tam in TAMIL_TRANSLATIONS.items():
                if eng.lower() in text.lower():
                    text = text.replace(eng, tam)
            return text
            
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        # Fallback to basic word replacement
        for eng, tam in TAMIL_TRANSLATIONS.items():
            if eng.lower() in text.lower():
                text = text.replace(eng, tam)
        return text

def fetch_from_rss(feed_url, source_name):
    """Fetch and parse RSS feed data"""
    try:
        print(f"Attempting to fetch RSS feed from {feed_url}")
        
        # Fetch content using requests with proper SSL handling
        response = session.get(feed_url, timeout=15)
        response.raise_for_status()
        
        # Parse the feed content
        feed = feedparser.parse(response.text)
        
        if hasattr(feed, 'bozo_exception'):
            print(f"Warning: Feed parsing error for {feed_url}: {feed.bozo_exception}")
        
        print(f"Feed version: {feed.version if hasattr(feed, 'version') else 'unknown'}")
        print(f"Number of entries: {len(feed.entries)}")
        
        schemes = []
        
        for entry in feed.entries:
            title = entry.get('title', '')
            description = entry.get('description', '')
            link = entry.get('link', '')
            date = entry.get('published', datetime.now().strftime("%Y-%m-%d"))
            
            print(f"Processing entry: {title}")
            
            # Try to convert the date to our format
            try:
                parsed_date = datetime.strptime(date, '%a, %d %b %Y %H:%M:%S %z')
                date = parsed_date.strftime("%Y-%m-%d")
            except:
                try:
                    parsed_date = datetime.strptime(date, '%Y-%m-%d')
                    date = parsed_date.strftime("%Y-%m-%d")
                except:
                    date = datetime.now().strftime("%Y-%m-%d")
                    print(f"Using current date for entry: {title}")
            
            # Clean up description (remove HTML tags)
            if description:
                soup = BeautifulSoup(description, 'html.parser')
                description = soup.get_text(strip=True)[:500]
            
            scheme = {
                'title': title,
                'description': description,
                'link': link,
                'date': date,
                'source': source_name,
                'tamil_title': translate_to_tamil(title),
                'tamil_description': translate_to_tamil(description),
                'tamil_date': translate_date(date),
                'tamil_source': translate_source(source_name)
            }
            schemes.append(scheme)
            print(f"Added scheme: {title}")
        
        return schemes
    except Exception as e:
        print(f"Error fetching RSS feed {feed_url}: {str(e)}")
        # Try without SSL verification as fallback
        try:
            print("Retrying without SSL verification...")
            response = session.get(feed_url, verify=False, timeout=15)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            
            schemes = []
            for entry in feed.entries:
                title = entry.get('title', '')
                description = entry.get('description', '')
                link = entry.get('link', '')
                date = entry.get('published', datetime.now().strftime("%Y-%m-%d"))
                
                # Clean up description (remove HTML tags)
                if description:
                    soup = BeautifulSoup(description, 'html.parser')
                    description = soup.get_text(strip=True)[:500]
                
                scheme = {
                    'title': title,
                    'description': description,
                    'link': link,
                    'date': date,
                    'source': source_name,
                    'tamil_title': translate_to_tamil(title),
                    'tamil_description': translate_to_tamil(description),
                    'tamil_date': translate_date(date),
                    'tamil_source': translate_source(source_name)
                }
                schemes.append(scheme)
            return schemes
        except Exception as e2:
            print(f"Error in fallback method for {feed_url}: {str(e2)}")
            return []

def get_agricultural_schemes():
    """Main function to retrieve agricultural schemes from RSS feeds"""
    print("🔍 வேளாண் திட்டங்களைத் தேடுகிறது...")
    start_time = time.time()
    all_schemes = []
    
    try:
        # Fetch from RSS feeds
        for feed_url, source_name in SCHEME_FEEDS:
            print(f"Fetching from {source_name}...")
            schemes = fetch_from_rss(feed_url, source_name)
            all_schemes.extend(schemes)
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_schemes = []
        for scheme in all_schemes:
            if scheme['title'] not in seen_titles:
                seen_titles.add(scheme['title'])
                unique_schemes.append(scheme)
        
        print(f"\nகிடைத்த மொத்த வேளாண் திட்டங்கள்: {len(unique_schemes)}")
        print(f"எடுத்துக்கொண்ட நேரம்: {time.time() - start_time:.2f} விநாடிகள்")
        
        return unique_schemes
    except Exception as e:
        print(f"Error fetching schemes: {e}")
        return []

if __name__ == "__main__":
    schemes = get_agricultural_schemes()
    
    if schemes:
        print("\n=== சமீபத்திய வேளாண் திட்டங்கள் ===")
        for i, scheme in enumerate(schemes[:20], 1):
            print(f"\n{i}. {scheme['tamil_title']}")
            print(f"   மூலம்: {scheme['tamil_source']}")
            print(f"   தேதி: {scheme['tamil_date']}")
            print(f"   இணைப்பு: {scheme['link']}")
            print(f"   விளக்கம்: {scheme['tamil_description']}")
    else:
        print("\nஎந்த திட்டங்களும் கிடைக்கவில்லை. இந்த தீர்வுகளை முயற்சிக்கவும்:")
        print("- RSS ஊட்டங்கள் அணுகக்கூடியதா என்பதைச் சரிபார்க்கவும்")
        print("- உங்கள் இணைய இணைப்பைச் சரிபார்க்கவும்")