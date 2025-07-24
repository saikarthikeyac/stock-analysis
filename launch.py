#!/usr/bin/env python3
"""
Simple launcher for the Stock Sentiment Analysis System
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def main():
    print("🚀 Stock Sentiment Analysis System Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Error: Please run this script from the project root directory")
        return
    
    try:
        # Install dependencies
        print("📦 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        
        # Setup database
        print("🗃️ Setting up database...")
        subprocess.run([sys.executable, "-c", """
import sys, os
sys.path.append('services')
from database_setup import setup_database
setup_database()
print('Database structure created!')
        """], check=True, capture_output=True)
        print("✅ Database setup completed")
        
        # Fetch real stock data
        print("📈 Fetching real stock prices...")
        subprocess.run([sys.executable, "-c", """
import sys, os
sys.path.append('services')
from stock_price_fetcher import StockPriceFetcher
fetcher = StockPriceFetcher()
fetcher.fetch_all_stocks()
print('Real stock data fetched!')
        """], check=True, capture_output=True)
        print("✅ Real stock prices loaded")
        
        # Fetch real news data
        print("📰 Fetching real financial news...")
        subprocess.run([sys.executable, "-c", """
import sys, os
sys.path.append('producer')
from news_producer import NewsProducer
producer = NewsProducer()
news_items = producer.fetch_stock_news()
if news_items:
    producer.send_to_queue(news_items)
print(f'Fetched {len(news_items)} real news articles!')
        """], check=True, capture_output=True)
        print("✅ Real financial news loaded")
        
        # Start API server
        print("🌐 Starting API server...")
        api_process = subprocess.Popen([
            sys.executable, "api/main.py"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for API to start
        time.sleep(3)
        
        print("📊 Starting dashboard...")
        print("\n" + "=" * 50)
        print("🎉 System started successfully!")
        print("=" * 50)
        print("📈 Dashboard:    http://localhost:8501")
        print("🔌 API:          http://localhost:8000")
        print("📚 API Docs:     http://localhost:8000/docs")
        print("=" * 50)
        print("\nThe dashboard will open in your browser...")
        print("Press Ctrl+C to stop the system")
        print("=" * 50)
        
        # Open browser
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:8501')
        except:
            pass
        
        # Start dashboard (blocking)
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print("Please check that you have Python and pip installed correctly")
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        try:
            api_process.terminate()
        except:
            pass
        print("✅ System stopped")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
