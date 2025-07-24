import requests
from bs4 import BeautifulSoup
import redis
import json
import time
import logging
from datetime import datetime
import os
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsProducer:
    def __init__(self, redis_host='localhost', redis_port=6379):
        """Initialize the news producer with Redis connection"""
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.queue_name = 'sentiment_requests'
        
        # API Keys - you can set these as environment variables
        self.newsapi_key = os.getenv('NEWSAPI_KEY', None)  # Get free key from https://newsapi.org/
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY', None)  # Get free key from https://www.alphavantage.co/
        
    def fetch_newsapi_data(self, symbols: List[str]) -> List[Dict]:
        """Fetch real news data from NewsAPI"""
        news_items = []
        
        if not self.newsapi_key:
            logger.warning("NewsAPI key not found. Set NEWSAPI_KEY environment variable.")
            return []
        
        for symbol in symbols:
            try:
                # Search for news articles mentioning the stock symbol
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f'{symbol} stock OR {symbol} earnings OR {symbol} company',
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10,
                    'apiKey': self.newsapi_key,
                    'from': datetime.now().strftime('%Y-%m-%d')  # Today's news
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data.get('articles', []):
                        if article.get('title') and article.get('description'):
                            # Combine title and description for better sentiment analysis
                            full_text = f"{article['title']}. {article['description']}"
                            
                            news_item = {
                                'text': full_text,
                                'symbol': symbol,
                                'timestamp': datetime.now().isoformat(),
                                'source': 'newsapi',
                                'url': article.get('url', ''),
                                'published_at': article.get('publishedAt', ''),
                                'source_name': article.get('source', {}).get('name', 'Unknown')
                            }
                            news_items.append(news_item)
                            
                    logger.info(f"Fetched {len(data.get('articles', []))} articles for {symbol} from NewsAPI")
                else:
                    logger.error(f"NewsAPI error for {symbol}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error fetching NewsAPI data for {symbol}: {e}")
                
        return news_items
    
    def fetch_alpha_vantage_news(self, symbols: List[str]) -> List[Dict]:
        """Fetch news from Alpha Vantage News & Sentiment API"""
        news_items = []
        
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage key not found. Set ALPHA_VANTAGE_KEY environment variable.")
            return []
        
        for symbol in symbols:
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': self.alpha_vantage_key,
                    'limit': 10
                }
                
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data.get('feed', []):
                        if article.get('title') and article.get('summary'):
                            # Use title and summary
                            full_text = f"{article['title']}. {article['summary']}"
                            
                            # Extract sentiment score if available
                            sentiment_score = None
                            ticker_sentiment = article.get('ticker_sentiment', [])
                            for ticker in ticker_sentiment:
                                if ticker.get('ticker') == symbol:
                                    sentiment_score = ticker.get('ticker_sentiment_score')
                                    break
                            
                            news_item = {
                                'text': full_text,
                                'symbol': symbol,
                                'timestamp': datetime.now().isoformat(),
                                'source': 'alpha_vantage',
                                'url': article.get('url', ''),
                                'published_at': article.get('time_published', ''),
                                'source_name': article.get('source', 'Unknown'),
                                'relevance_score': ticker.get('relevance_score') if ticker_sentiment else None,
                                'av_sentiment_score': sentiment_score
                            }
                            news_items.append(news_item)
                            
                    logger.info(f"Fetched {len(data.get('feed', []))} articles for {symbol} from Alpha Vantage")
                else:
                    logger.error(f"Alpha Vantage error for {symbol}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
                
        return news_items
    
    def fetch_yahoo_finance_news(self, symbols: List[str]) -> List[Dict]:
        """Fetch news from Yahoo Finance (web scraping approach)"""
        news_items = []
        
        for symbol in symbols:
            try:
                # Use a simpler RSS feed approach
                url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'xml')  # Use XML parser for RSS
                    
                    # Find RSS items
                    items = soup.find_all('item')[:5]  # Get top 5 articles
                    
                    for item in items:
                        title_tag = item.find('title')
                        description_tag = item.find('description')
                        pubdate_tag = item.find('pubDate')
                        
                        if title_tag:
                            title = title_tag.get_text().strip()
                            description = description_tag.get_text().strip() if description_tag else ""
                            pubdate = pubdate_tag.get_text().strip() if pubdate_tag else ""
                            
                            # Combine title and description
                            full_text = f"{title}. {description}" if description else title
                            
                            if len(full_text) > 10:  # Basic validation
                                news_item = {
                                    'text': full_text,
                                    'symbol': symbol,
                                    'timestamp': datetime.now().isoformat(),
                                    'source': 'yahoo_finance_rss',
                                    'url': f"https://finance.yahoo.com/quote/{symbol}/news",
                                    'published_at': pubdate,
                                    'source_name': 'Yahoo Finance'
                                }
                                news_items.append(news_item)
                    
                    logger.info(f"Scraped {len(items)} articles for {symbol} from Yahoo Finance RSS")
                else:
                    # Fallback to web scraping if RSS fails
                    html_items = self.scrape_yahoo_html(symbol)
                    news_items.extend(html_items)
                    
            except Exception as e:
                logger.error(f"Error fetching Yahoo Finance RSS for {symbol}: {e}")
                # Try HTML scraping as fallback
                try:
                    html_items = self.scrape_yahoo_html(symbol)
                    news_items.extend(html_items)
                except:
                    pass
                
        return news_items
    
    def scrape_yahoo_html(self, symbol: str) -> List[Dict]:
        """Fallback HTML scraping for Yahoo Finance"""
        news_items = []
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for various possible selectors for news headlines
                selectors = [
                    'h3[data-test-id="clusterTitle"]',  # New Yahoo structure
                    'h3.Mb\\(5px\\)',  # Old structure
                    '[data-test-id="headline"]',  # Alternative
                    '.js-content-viewer h3',  # Another alternative
                ]
                
                articles_found = []
                for selector in selectors:
                    articles = soup.select(selector)
                    if articles:
                        articles_found = articles[:3]  # Get top 3
                        break
                
                for article in articles_found:
                    title = article.get_text().strip()
                    if title and len(title) > 15:  # Better validation
                        news_item = {
                            'text': title,
                            'symbol': symbol,
                            'timestamp': datetime.now().isoformat(),
                            'source': 'yahoo_finance_html',
                            'url': f"https://finance.yahoo.com/quote/{symbol}/news",
                            'published_at': datetime.now().isoformat(),
                            'source_name': 'Yahoo Finance'
                        }
                        news_items.append(news_item)
                        
                logger.info(f"HTML scraped {len(articles_found)} headlines for {symbol}")
                
        except Exception as e:
            logger.error(f"Error in HTML scraping for {symbol}: {e}")
            
        return news_items
    
    def fetch_free_financial_news(self, symbols: List[str]) -> List[Dict]:
        """Fetch news from free financial data sources"""
        news_items = []
        
        for symbol in symbols:
            try:
                # Use FinnHub free tier (limited requests)
                url = "https://finnhub.io/api/v1/company-news"
                params = {
                    'symbol': symbol,
                    'from': datetime.now().strftime('%Y-%m-%d'),
                    'to': datetime.now().strftime('%Y-%m-%d'),
                    'token': 'demo'  # Demo token - limited but free
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, list):
                        for article in data[:3]:  # Limit to 3 articles
                            if article.get('headline') and article.get('summary'):
                                full_text = f"{article['headline']}. {article['summary']}"
                                
                                news_item = {
                                    'text': full_text,
                                    'symbol': symbol,
                                    'timestamp': datetime.now().isoformat(),
                                    'source': 'finnhub_free',
                                    'url': article.get('url', ''),
                                    'published_at': datetime.fromtimestamp(article.get('datetime', 0)).isoformat() if article.get('datetime') else '',
                                    'source_name': 'FinnHub'
                                }
                                news_items.append(news_item)
                                
                        logger.info(f"Fetched {len(data)} articles for {symbol} from FinnHub free")
                    
            except Exception as e:
                logger.error(f"Error fetching FinnHub free data for {symbol}: {e}")
        
        return news_items
        
    def fetch_stock_news(self, symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']):
        """Fetch stock-related news from multiple real sources"""
        all_news_items = []
        
        logger.info(f"Fetching real news data for symbols: {symbols}")
        
        # Try different news sources in order of preference
        try:
            # 1. Try free FinnHub (no API key needed)
            free_items = self.fetch_free_financial_news(symbols)
            all_news_items.extend(free_items)
            
            # 2. Try NewsAPI (best for real-time news)
            newsapi_items = self.fetch_newsapi_data(symbols)
            all_news_items.extend(newsapi_items)
            
            # 3. Try Alpha Vantage (includes sentiment scores)
            av_items = self.fetch_alpha_vantage_news(symbols)
            all_news_items.extend(av_items)
            
            # 4. Try Yahoo Finance scraping
            yahoo_items = self.fetch_yahoo_finance_news(symbols)
            all_news_items.extend(yahoo_items)
            
        except Exception as e:
            logger.error(f"Error in primary news fetching: {e}")
        
        # Only use real news - no fallback generation for production system
        logger.info(f"Collected {len(all_news_items)} real news items from various sources")
        
        # Remove duplicates based on text similarity
        unique_items = self.remove_duplicates(all_news_items)
        
        logger.info(f"Total unique news items collected: {len(unique_items)}")
        return unique_items
    
    def remove_duplicates(self, news_items: List[Dict]) -> List[Dict]:
        """Remove duplicate news items based on text similarity"""
        unique_items = []
        seen_texts = set()
        
        for item in news_items:
            # Create a simple hash of the first 50 characters
            text_hash = item['text'][:50].lower().replace(' ', '')
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_items.append(item)
                
        return unique_items
    
    def send_to_queue(self, news_items):
        """Send news items to Redis queue for processing"""
        for item in news_items:
            try:
                self.redis_client.lpush(self.queue_name, json.dumps(item))
                logger.info(f"Sent news item to queue: {item['text'][:50]}...")
            except Exception as e:
                logger.error(f"Error sending to queue: {e}")
    
    def run_producer(self, interval_minutes=30):
        """Run the producer continuously"""
        logger.info("Starting news producer...")
        
        while True:
            try:
                logger.info("Fetching latest stock news...")
                news_items = self.fetch_stock_news()
                
                if news_items:
                    self.send_to_queue(news_items)
                    logger.info(f"Processed {len(news_items)} news items")
                else:
                    logger.warning("No news items found")
                
                logger.info(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Producer stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in producer loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

def main():
    producer = NewsProducer()
    producer.run_producer(interval_minutes=15)  # Run every 15 minutes for real-time updates

if __name__ == "__main__":
    main()
