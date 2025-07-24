from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))
from model_training import StockPredictionModel

app = FastAPI(title="Stock Sentiment Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = StockPredictionModel()

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_sentiment.db')

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Stock Sentiment Analysis API is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check database tables
        cursor.execute("SELECT COUNT(*) FROM sentiment_data")
        sentiment_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM stock_prices")
        price_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "sentiment_records": sentiment_count,
            "price_records": price_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/stocks/{symbol}/latest")
async def get_latest_stock_data(symbol: str):
    """Get latest stock price and sentiment data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get latest stock price
        price_query = '''
            SELECT price, volume, datetime 
            FROM stock_prices 
            WHERE symbol = ? 
            ORDER BY datetime DESC 
            LIMIT 1
        '''
        price_data = pd.read_sql_query(price_query, conn, params=(symbol.upper(),))
        
        # Get recent sentiment data
        sentiment_query = '''
            SELECT AVG(compound_score) as avg_sentiment, COUNT(*) as count
            FROM sentiment_data 
            WHERE timestamp >= datetime('now', '-1 day')
        '''
        sentiment_data = pd.read_sql_query(sentiment_query, conn)
        
        conn.close()
        
        if price_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        return {
            "symbol": symbol.upper(),
            "latest_price": float(price_data['price'].iloc[0]),
            "volume": int(price_data['volume'].iloc[0]) if price_data['volume'].iloc[0] else 0,
            "price_datetime": price_data['datetime'].iloc[0],
            "avg_sentiment_24h": float(sentiment_data['avg_sentiment'].iloc[0]) if pd.notna(sentiment_data['avg_sentiment'].iloc[0]) else 0,
            "sentiment_count_24h": int(sentiment_data['count'].iloc[0])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stocks/{symbol}/history")
async def get_stock_history(symbol: str, days: int = 7):
    """Get historical stock price and sentiment data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = '''
            SELECT 
                sp.datetime,
                sp.price,
                sp.volume,
                AVG(sd.compound_score) as avg_sentiment
            FROM stock_prices sp
            LEFT JOIN sentiment_data sd ON DATE(sp.datetime) = DATE(sd.timestamp)
            WHERE sp.symbol = ? AND sp.datetime >= datetime('now', '-{} days')
            GROUP BY sp.datetime, sp.price, sp.volume
            ORDER BY sp.datetime
        '''.format(days)
        
        data = pd.read_sql_query(query, conn, params=(symbol.upper(),))
        conn.close()
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for symbol {symbol}")
        
        # Convert to list of dictionaries
        history = []
        for _, row in data.iterrows():
            history.append({
                "datetime": row['datetime'],
                "price": float(row['price']),
                "volume": int(row['volume']) if pd.notna(row['volume']) else 0,
                "sentiment": float(row['avg_sentiment']) if pd.notna(row['avg_sentiment']) else 0
            })
        
        return {
            "symbol": symbol.upper(),
            "days": days,
            "data": history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{symbol}")
async def predict_stock_movement(symbol: str):
    """Predict next day stock movement"""
    try:
        prediction = model.predict(symbol.upper())
        
        if prediction is None:
            raise HTTPException(status_code=404, detail=f"Unable to make prediction for {symbol}")
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment/recent")
async def get_recent_sentiment(hours: int = 24):
    """Get recent sentiment analysis results"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = '''
            SELECT text, compound_score, positive, negative, neutral, timestamp, source
            FROM sentiment_data 
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
            LIMIT 50
        '''.format(hours)
        
        data = pd.read_sql_query(query, conn)
        conn.close()
        
        sentiment_data = []
        for _, row in data.iterrows():
            sentiment_data.append({
                "text": row['text'],
                "compound_score": float(row['compound_score']),
                "positive": float(row['positive']),
                "negative": float(row['negative']),
                "neutral": float(row['neutral']),
                "timestamp": row['timestamp'],
                "source": row['source']
            })
        
        return {
            "hours": hours,
            "count": len(sentiment_data),
            "data": sentiment_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get various statistics
        stats = {}
        
        # Total records
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sentiment_data")
        stats['total_sentiment_records'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM stock_prices")
        stats['total_price_records'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        stats['total_predictions'] = cursor.fetchone()[0]
        
        # Recent activity
        cursor.execute("SELECT COUNT(*) FROM sentiment_data WHERE timestamp >= datetime('now', '-24 hours')")
        stats['sentiment_last_24h'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM stock_prices WHERE timestamp >= datetime('now', '-24 hours')")
        stats['prices_last_24h'] = cursor.fetchone()[0]
        
        # Average sentiment
        cursor.execute("SELECT AVG(compound_score) FROM sentiment_data WHERE timestamp >= datetime('now', '-24 hours')")
        avg_sentiment = cursor.fetchone()[0]
        stats['avg_sentiment_24h'] = float(avg_sentiment) if avg_sentiment else 0
        
        # Available symbols
        cursor.execute("SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        stats['available_symbols'] = symbols
        
        conn.close()
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
