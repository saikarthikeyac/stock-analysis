import redis
import json
import sqlite3
import logging
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentConsumer:
    def __init__(self, redis_host='localhost', redis_port=6379):
        """Initialize sentiment consumer with Redis and database connections"""
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.queue_name = 'sentiment_requests'
        self.analyzer = SentimentIntensityAnalyzer()
        self.db_path = os.path.join('data', 'stock_sentiment.db')
        
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
        try:
            scores = self.analyzer.polarity_scores(text)
            return scores
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None
    
    def save_to_database(self, news_item, sentiment_scores):
        """Save sentiment analysis results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sentiment_data 
                (text, compound_score, positive, negative, neutral, source)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                news_item['text'],
                sentiment_scores['compound'],
                sentiment_scores['pos'],
                sentiment_scores['neg'],
                sentiment_scores['neu'],
                news_item.get('source', 'unknown')
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved sentiment data for: {news_item['text'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def process_message(self, message_data):
        """Process a single message from the queue"""
        try:
            news_item = json.loads(message_data)
            text = news_item['text']
            
            # Analyze sentiment
            sentiment_scores = self.analyze_sentiment(text)
            
            if sentiment_scores:
                # Save to database
                self.save_to_database(news_item, sentiment_scores)
                
                # Log the result
                compound_score = sentiment_scores['compound']
                sentiment_label = "POSITIVE" if compound_score > 0.05 else "NEGATIVE" if compound_score < -0.05 else "NEUTRAL"
                
                logger.info(f"Processed: {text[:50]}... | Sentiment: {sentiment_label} ({compound_score:.3f})")
                
                return True
            else:
                logger.warning(f"Failed to analyze sentiment for: {text[:50]}...")
                return False
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return False
    
    def run_consumer(self):
        """Run the sentiment consumer continuously"""
        logger.info("Starting sentiment consumer...")
        
        while True:
            try:
                # Blocking pop from Redis queue (waits for messages)
                message = self.redis_client.brpop(self.queue_name, timeout=10)
                
                if message:
                    queue_name, message_data = message
                    self.process_message(message_data)
                else:
                    logger.info("No messages in queue, waiting...")
                    
            except KeyboardInterrupt:
                logger.info("Consumer stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                time.sleep(5)  # Wait before retrying

def main():
    consumer = SentimentConsumer()
    consumer.run_consumer()

if __name__ == "__main__":
    main()
