# 📈 Stock Sentiment Analysis System

A real-time stock market analysis system that predicts stock price movements using financial news sentiment and machine learning.

This project covers the entire data workflow: automated news collection and sentiment scoring feed into a Redis-based producer/consumer queue, which drives a predictive model trained with feature engineering and Random Forest. Stock price data is fetched in real time via yfinance and stored alongside sentiment and predictions in SQLite. Results are served through a FastAPI backend and visualized in an interactive Streamlit dashboard. Together, these components illustrate end-to-end data analysis, machine learning, and distributed system design in a single cohesive application.

## 🏗️ System Architecture

```
Yahoo Finance API
      │
      ▼
Stock Price Fetcher ───┐
                       ▼
Financial News → News Producer → Redis Queue → Sentiment Consumer
                                                     │
                                                     ▼
                                            SQLite Database
                                                     │
                                                     ▼
                                            FastAPI Server
                                                     │
                                                     ▼
                                            Streamlit Dashboard
```


## 🚀 How to Run

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the System**
   ```bash
   python launch.py
   ```

3. **Open Your Browser**
   - Dashboard: http://localhost:8501
   - API: http://localhost:8000

That's it! The system will automatically:
- Set up the database
- Fetch real stock data
- Collect financial news
- Start the web interface

## 💻 Technology Used

- **Python** - Main programming language  
- **Redis** - Message queue for producer/consumer pattern
- **Streamlit** - Web dashboard
- **FastAPI** - REST API
- **Machine Learning** - Random Forest for predictions
- **VADER** - Sentiment analysis of news
- **SQLite** - Database for storing data
- **yfinance** - Real stock market data

## 📊 What You'll See

- **Real-time stock prices** and charts
- **Sentiment analysis** of financial news
- **AI predictions** (UP/DOWN) with confidence scores
- **System statistics** and performance metrics



---
*A complete stock market analysis system built with modern Python technologies*
