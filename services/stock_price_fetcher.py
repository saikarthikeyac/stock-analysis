import yfinance as yf
import sqlite3
import logging
import time
import os
from datetime import datetime, timedelta
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPriceFetcher:
    def __init__(self):
        """Initialize stock price fetcher"""
        self.db_path = os.path.join('data', 'stock_sentiment.db')
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']  # Popular stocks
        
    def fetch_stock_data(self, symbol, period="1d", interval="1h"):
        """Fetch stock data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                logger.info(f"Fetched {len(data)} data points for {symbol}")
                return data
            else:
                logger.warning(f"No data found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def save_stock_data(self, symbol, data):
        """Save stock data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for timestamp, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO stock_prices 
                    (symbol, price, volume, datetime)
                    VALUES (?, ?, ?, ?)
                ''', (
                    symbol,
                    float(row['Close']),
                    int(row['Volume']),
                    timestamp.strftime('%Y-%m-%d %H:%M:%S')
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(data)} records for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
    
    def get_latest_price(self, symbol):
        """Get the most recent price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if current_price:
                return current_price
            else:
                # Fallback to latest close price
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    return data['Close'].iloc[-1]
                    
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            
        return None
    
    def fetch_all_stocks(self):
        """Fetch data for all tracked symbols"""
        logger.info("Fetching data for all stocks...")
        
        for symbol in self.symbols:
            try:
                # Fetch recent data
                data = self.fetch_stock_data(symbol, period="5d", interval="1h")
                
                if data is not None:
                    self.save_stock_data(symbol, data)
                    
                    # Also get and save current price
                    current_price = self.get_latest_price(symbol)
                    if current_price:
                        self.save_current_price(symbol, current_price)
                        
                # Small delay between requests
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    def save_current_price(self, symbol, price):
        """Save current/latest price with current timestamp"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO stock_prices 
                (symbol, price, volume, datetime)
                VALUES (?, ?, ?, ?)
            ''', (
                symbol,
                float(price),
                0,  # Volume not available for current price
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved current price for {symbol}: ${price:.2f}")
            
        except Exception as e:
            logger.error(f"Error saving current price for {symbol}: {e}")
    
    def run_fetcher(self, interval_minutes=60):
        """Run the stock price fetcher continuously"""
        logger.info("Starting stock price fetcher...")
        
        # Initial fetch
        self.fetch_all_stocks()
        
        while True:
            try:
                logger.info(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
                self.fetch_all_stocks()
                
            except KeyboardInterrupt:
                logger.info("Stock fetcher stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in fetcher loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

def main():
    fetcher = StockPriceFetcher()
    fetcher.run_fetcher(interval_minutes=15)  # Fetch every 15 minutes for real-time data

if __name__ == "__main__":
    main()
