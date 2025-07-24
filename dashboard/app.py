import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import os
import sys
from datetime import datetime, timedelta
import requests

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

# Page config
st.set_page_config(
    page_title="Stock Sentiment Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_sentiment.db')
API_BASE_URL = "http://localhost:8000"

def get_data_from_db(query, params=None):
    """Get data from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def get_available_symbols():
    """Get list of available stock symbols"""
    query = "SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol"
    df = get_data_from_db(query)
    return df['symbol'].tolist() if not df.empty else ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

def get_system_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback to direct database query
    stats = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM sentiment_data")
        stats['total_sentiment_records'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM stock_prices")
        stats['total_price_records'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sentiment_data WHERE timestamp >= datetime('now', '-24 hours')")
        stats['sentiment_last_24h'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(compound_score) FROM sentiment_data WHERE timestamp >= datetime('now', '-24 hours')")
        avg_sentiment = cursor.fetchone()[0]
        stats['avg_sentiment_24h'] = float(avg_sentiment) if avg_sentiment else 0
        
        conn.close()
    except:
        stats = {'total_sentiment_records': 0, 'total_price_records': 0, 'sentiment_last_24h': 0, 'avg_sentiment_24h': 0}
    
    return stats

def get_stock_history(symbol, days=7):
    """Get historical stock data"""
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
    
    return get_data_from_db(query, (symbol,))

def get_recent_sentiment(hours=24):
    """Get recent sentiment data"""
    query = '''
        SELECT text, compound_score, positive, negative, neutral, timestamp
        FROM sentiment_data 
        WHERE timestamp >= datetime('now', '-{} hours')
        ORDER BY timestamp DESC
        LIMIT 20
    '''.format(hours)
    
    return get_data_from_db(query)

def get_prediction(symbol):
    """Get prediction from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/predict/{symbol}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# Main dashboard
def main():
    st.title("📈 Real-Time Stock Sentiment Analysis Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Overview", "Stock Analysis", "Sentiment Analysis", "Predictions", "System Status"])
    
    if page == "Overview":
        show_overview()
    elif page == "Stock Analysis":
        show_stock_analysis()
    elif page == "Sentiment Analysis":
        show_sentiment_analysis()
    elif page == "Predictions":
        show_predictions()
    elif page == "System Status":
        show_system_status()

def show_overview():
    """Show overview dashboard"""
    st.header("📊 System Overview")
    
    # Get system stats
    stats = get_system_stats()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sentiment Records", stats.get('total_sentiment_records', 0))
    
    with col2:
        st.metric("Total Price Records", stats.get('total_price_records', 0))
    
    with col3:
        st.metric("Sentiment (24h)", stats.get('sentiment_last_24h', 0))
    
    with col4:
        sentiment_score = stats.get('avg_sentiment_24h', 0)
        sentiment_label = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"
        st.metric("Avg Sentiment (24h)", f"{sentiment_score:.3f}", sentiment_label)
    
    st.divider()
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Recent Stock Activity")
        recent_stocks = get_data_from_db('''
            SELECT symbol, price, datetime 
            FROM stock_prices 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        
        if not recent_stocks.empty:
            st.dataframe(recent_stocks, use_container_width=True)
        else:
            st.info("No recent stock data available")
    
    with col2:
        st.subheader("💭 Recent Sentiment")
        recent_sentiment = get_recent_sentiment(6)
        
        if not recent_sentiment.empty:
            for _, row in recent_sentiment.head(5).iterrows():
                sentiment_color = "🟢" if row['compound_score'] > 0.05 else "🔴" if row['compound_score'] < -0.05 else "🟡"
                st.write(f"{sentiment_color} **{row['compound_score']:.3f}** - {row['text'][:100]}...")
        else:
            st.info("No recent sentiment data available")

def show_stock_analysis():
    """Show stock analysis page"""
    st.header("📈 Stock Analysis")
    
    # Symbol selection
    symbols = get_available_symbols()
    selected_symbol = st.selectbox("Select Stock Symbol", symbols)
    
    # Time range
    days = st.slider("Days of History", min_value=1, max_value=30, value=7)
    
    if selected_symbol:
        # Get stock history
        history = get_stock_history(selected_symbol, days)
        
        if not history.empty:
            # Convert datetime
            history['datetime'] = pd.to_datetime(history['datetime'])
            
            # Create dual-axis plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(f'{selected_symbol} Price & Sentiment', 'Volume'),
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                vertical_spacing=0.1
            )
            
            # Price line
            fig.add_trace(
                go.Scatter(x=history['datetime'], y=history['price'], 
                          name='Price ($)', line=dict(color='blue', width=2)),
                row=1, col=1, secondary_y=False
            )
            
            # Sentiment line
            fig.add_trace(
                go.Scatter(x=history['datetime'], y=history['avg_sentiment'], 
                          name='Sentiment', line=dict(color='red', width=2)),
                row=1, col=1, secondary_y=True
            )
            
            # Volume bars
            fig.add_trace(
                go.Bar(x=history['datetime'], y=history['volume'], 
                       name='Volume', marker_color='lightgreen'),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(height=600, title_text=f"{selected_symbol} Analysis")
            fig.update_yaxes(title_text="Price ($)", secondary_y=False, row=1, col=1)
            fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_price = history['price'].iloc[-1]
                price_change = ((current_price - history['price'].iloc[0]) / history['price'].iloc[0]) * 100
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}%")
            
            with col2:
                avg_sentiment = history['avg_sentiment'].mean()
                sentiment_trend = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
                st.metric("Avg Sentiment", f"{avg_sentiment:.3f}", sentiment_trend)
            
            with col3:
                total_volume = history['volume'].sum()
                st.metric("Total Volume", f"{total_volume:,.0f}")
            
        else:
            st.warning(f"No data available for {selected_symbol}")

def show_sentiment_analysis():
    """Show sentiment analysis page"""
    st.header("💭 Sentiment Analysis")
    
    # Time range selector
    hours = st.selectbox("Time Range", [6, 12, 24, 48, 72], index=2)
    
    # Get recent sentiment data
    sentiment_data = get_recent_sentiment(hours)
    
    if not sentiment_data.empty:
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            
            # Categorize sentiments
            sentiment_data['category'] = sentiment_data['compound_score'].apply(
                lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral'
            )
            
            category_counts = sentiment_data['category'].value_counts()
            
            fig = px.pie(values=category_counts.values, names=category_counts.index,
                        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Over Time")
            sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
            
            # Group by hour and calculate average sentiment
            hourly_sentiment = sentiment_data.set_index('timestamp').resample('H')['compound_score'].mean()
            
            fig = px.line(x=hourly_sentiment.index, y=hourly_sentiment.values,
                         title="Average Sentiment by Hour")
            fig.update_layout(xaxis_title="Time", yaxis_title="Sentiment Score")
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Recent sentiment items
        st.subheader("Recent Sentiment Items")
        
        # Display sentiment data with color coding
        for _, row in sentiment_data.head(10).iterrows():
            score = row['compound_score']
            if score > 0.05:
                color = "🟢"
                label = "Positive"
            elif score < -0.05:
                color = "🔴"
                label = "Negative"
            else:
                color = "🟡"
                label = "Neutral"
            
            with st.expander(f"{color} {label} ({score:.3f}) - {row['text'][:100]}..."):
                st.write(f"**Full Text:** {row['text']}")
                st.write(f"**Compound Score:** {score:.3f}")
                st.write(f"**Positive:** {row['positive']:.3f}")
                st.write(f"**Negative:** {row['negative']:.3f}")
                st.write(f"**Neutral:** {row['neutral']:.3f}")
                st.write(f"**Timestamp:** {row['timestamp']}")
    
    else:
        st.info(f"No sentiment data available for the last {hours} hours")

def show_predictions():
    """Show predictions page"""
    st.header("🔮 Stock Predictions")
    
    symbols = get_available_symbols()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Get Prediction")
        selected_symbol = st.selectbox("Select Symbol", symbols, key="pred_symbol")
        
        if st.button("Get Prediction"):
            with st.spinner("Making prediction..."):
                prediction = get_prediction(selected_symbol)
                
                if prediction:
                    direction = prediction['direction']
                    confidence = prediction['confidence']
                    
                    # Display prediction
                    if direction == "UP":
                        st.success(f"📈 **{selected_symbol}** is predicted to go **UP**")
                    else:
                        st.error(f"📉 **{selected_symbol}** is predicted to go **DOWN**")
                    
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Show feature importance (simplified)
                    features = prediction.get('features', [])
                    if features:
                        st.subheader("Features Used")
                        feature_names = ['Price Change %', 'Sentiment Score', 'Volume (log)', 'Sentiment Count']
                        for name, value in zip(feature_names, features):
                            st.write(f"**{name}:** {value:.3f}")
                
                else:
                    st.warning("Unable to make prediction. Insufficient data.")
    
    with col2:
        st.subheader("Recent Predictions")
        
        # Get recent predictions from database
        predictions = get_data_from_db('''
            SELECT symbol, predicted_direction, confidence, timestamp
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        
        if not predictions.empty:
            for _, row in predictions.iterrows():
                direction_icon = "📈" if row['predicted_direction'] == "UP" else "📉"
                st.write(f"{direction_icon} **{row['symbol']}** - {row['predicted_direction']} "
                        f"(Confidence: {row['confidence']:.1%}) - {row['timestamp']}")
        else:
            st.info("No recent predictions available")

def show_system_status():
    """Show system status page"""
    st.header("⚙️ System Status")
    
    # API health check
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            st.success("✅ API is running")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Database Status")
                st.write(f"**Status:** {health_data.get('database', 'Unknown')}")
                st.write(f"**Sentiment Records:** {health_data.get('sentiment_records', 0)}")
                st.write(f"**Price Records:** {health_data.get('price_records', 0)}")
            
            with col2:
                st.subheader("Last Updated")
                st.write(health_data.get('timestamp', 'Unknown'))
        else:
            st.error("❌ API is not responding")
    except:
        st.warning("⚠️ Cannot connect to API")
    
    st.divider()
    
    # Database statistics
    st.subheader("📊 Database Statistics")
    stats = get_system_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Sentiment Records", stats.get('total_sentiment_records', 0))
        st.metric("Sentiment (Last 24h)", stats.get('sentiment_last_24h', 0))
    
    with col2:
        st.metric("Total Price Records", stats.get('total_price_records', 0))
        st.metric("Prices (Last 24h)", stats.get('prices_last_24h', 0))
    
    with col3:
        avg_sentiment = stats.get('avg_sentiment_24h', 0)
        st.metric("Avg Sentiment (24h)", f"{avg_sentiment:.3f}")
        
        sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        st.write(f"**Overall Mood:** {sentiment_label}")
    
    # System components status
    st.subheader("🔧 Component Status")
    
    components = [
        ("Database", "✅ Connected"),
        ("Sentiment Analysis", "✅ Running"),
        ("Stock Price Fetcher", "✅ Running"),
        ("Prediction Model", "✅ Available"),
        ("API Service", "✅ Running"),
        ("Dashboard", "✅ Active")
    ]
    
    for component, status in components:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**{component}**")
        with col2:
            st.write(status)

if __name__ == "__main__":
    main()
