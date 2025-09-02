import sqlite3
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import json

class DataManager:
    def __init__(self):
        self.conn = sqlite3.connect('fintech_ml.db')
        self.create_tables()
    
    def create_tables(self):
        """Initialize all database tables"""
        # Stock market data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                date DATE,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                sma_10 REAL,
                sma_30 REAL,
                volatility REAL,
                price_change REAL,
                created_at TIMESTAMP
            )
        """)
        
        # News sentiment analysis table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                date DATE,
                headline TEXT,
                sentiment_score REAL,
                source TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # Model predictions audit trail
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                prediction_date DATE,
                confidence REAL,
                signal TEXT,
                var_risk REAL,
                sharpe_ratio REAL,
                model_version TEXT,
                features_used TEXT,
                compliance_status TEXT,
                created_at TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def get_stock_data(self, symbol, period="1y"):
        """Retrieve stock data with intelligent caching"""
        # Check for existing recent data
        query = """
            SELECT * FROM stock_data 
            WHERE symbol = ? AND date >= date('now', '-7 days')
            ORDER BY date DESC
        """
        existing_data = pd.read_sql(query, self.conn, params=[symbol])
        
        if len(existing_data) > 5:
            # Return cached data
            all_query = """
                SELECT * FROM stock_data 
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 365
            """
            return pd.read_sql(all_query, self.conn, params=[symbol])
        else:
            # Fetch fresh data from Yahoo Finance
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            # Calculate technical indicators
            data['SMA_10'] = data['Close'].rolling(10).mean()
            data['SMA_30'] = data['Close'].rolling(30).mean()
            data['Price_Change'] = data['Close'].pct_change()
            data['Volatility'] = data['Price_Change'].rolling(10).std()
            
            # Store in database
            self.store_stock_data(data, symbol)
            return data
    
    def store_stock_data(self, data, symbol):
        """Store stock data in database"""
        for date, row in data.iterrows():
            self.conn.execute("""
                INSERT OR REPLACE INTO stock_data 
                (symbol, date, open_price, high_price, low_price, close_price, 
                 volume, sma_10, sma_30, volatility, price_change, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [symbol, date.date(), row['Open'], row['High'], row['Low'], 
                  row['Close'], row['Volume'], row.get('SMA_10'), 
                  row.get('SMA_30'), row.get('Volatility'), 
                  row.get('Price_Change'), datetime.now()])
        self.conn.commit()
    
    def get_news_data(self, company, api_key, days=7):
        """Fetch news data with caching mechanism"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Check for recent news in database
        query = """
            SELECT * FROM news_sentiment 
            WHERE symbol = ? AND date >= ?
            ORDER BY date DESC
        """
        existing_news = pd.read_sql(query, self.conn, 
                                   params=[company, cutoff_date.date()])
        
        if len(existing_news) > 5:
            return existing_news.to_dict('records')
        else:
            # Fetch fresh news from NewsAPI
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': company,
                'apiKey': api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'from': cutoff_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params)
            articles = response.json().get('articles', [])
            
            # Store news in database
            for article in articles:
                self.conn.execute("""
                    INSERT INTO news_sentiment 
                    (symbol, date, headline, source, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, [company, datetime.now().date(), article['title'], 
                      article.get('source', {}).get('name', 'Unknown'), 
                      datetime.now()])
            
            self.conn.commit()
            return articles
