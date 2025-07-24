import sqlite3
import os

def setup_database():
    """Initialize SQLite database with required tables"""
    db_path = os.path.join('data', 'stock_sentiment.db')
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create sentiment_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            compound_score REAL,
            positive REAL,
            negative REAL,
            neutral REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source TEXT
        )
    ''')
    
    # Create stock_prices table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL,
            volume INTEGER,
            datetime DATETIME,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            predicted_direction TEXT,
            confidence REAL,
            features TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create merged_data view
    cursor.execute('''
        CREATE VIEW IF NOT EXISTS merged_sentiment_price AS
        SELECT 
            sp.symbol,
            sp.price,
            sp.volume,
            sp.datetime as price_datetime,
            AVG(sd.compound_score) as avg_sentiment,
            COUNT(sd.id) as sentiment_count,
            sp.timestamp
        FROM stock_prices sp
        LEFT JOIN sentiment_data sd ON 
            DATE(sp.datetime) = DATE(sd.timestamp)
        GROUP BY sp.symbol, sp.datetime, sp.price, sp.volume, sp.timestamp
    ''')
    
    conn.commit()
    conn.close()
    print("Database setup completed successfully!")

if __name__ == "__main__":
    setup_database()
