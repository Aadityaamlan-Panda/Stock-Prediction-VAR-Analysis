import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLEngine:
    def __init__(self):
        # Initialize FinBERT for financial sentiment analysis
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert"
        )
        self.model = None
        self.feature_names = ['SMA_10', 'SMA_30', 'Price_Change', 'Volatility', 'Sentiment']
        
    def analyze_sentiment(self, text):
        """Analyze sentiment using FinBERT model"""
        try:
            result = self.sentiment_pipeline(text[:512])  # FinBERT character limit
            
            # Convert to numerical score (-1 to +1)
            if result[0]['label'] == 'positive':
                return result[0]['score']
            elif result[0]['label'] == 'negative':
                return -result[0]['score']
            else:
                return 0
        except:
            return 0  # Return neutral if analysis fails
    
    def process_news_sentiment(self, news_articles):
        """Process multiple news articles for average sentiment"""
        if not news_articles:
            return 0
        
        sentiments = []
        for article in news_articles:
            if isinstance(article, dict):
                text = article.get('title', '') or article.get('headline', '')
            else:
                text = str(article)
            
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
        
        return np.mean(sentiments) if sentiments else 0
    
    def create_features(self, stock_data, sentiment_score):
        """Engineer features for machine learning model"""
        try:
            df = stock_data.copy()
            
            if 'Close' not in df.columns:
                raise ValueError("Stock data missing 'Close' column")
            # Calculate technical indicators if not present
            if 'SMA_10' not in df.columns:
                df['SMA_10'] = df['Close'].rolling(10).mean()
            if 'SMA_30' not in df.columns:
                df['SMA_30'] = df['Close'].rolling(30).mean()
            if 'Price_Change' not in df.columns:
                df['Price_Change'] = df['Close'].pct_change()
            if 'Volatility' not in df.columns:
                df['Volatility'] = df['Price_Change'].rolling(10).std()
            
            # Add sentiment feature
            df['Sentiment'] = sentiment_score
            
            # Create prediction target: next day price direction
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            # Remove NaN values
            df = df.dropna()
            if len(df) < 30:
                raise ValueError(f"Insufficient data after cleaning: {len(df)} rows")
            return df
        
        except Exception as e:
            print(f"❌ Feature creation failed: {e}")
            raise e
    
    def train_model(self, features_df):
        """Train Random Forest classification model"""
        if len(features_df) < 50:
            raise ValueError("Insufficient training data - need at least 50 samples")
        
        # Prepare features and target variable
        X = features_df[self.feature_names]
        y = features_df['Target']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model performance
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate feature importance for explainability
        feature_importance = dict(zip(self.feature_names, 
                                    self.model.feature_importances_))
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def predict(self, current_features):
        """Generate prediction for current market conditions"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare feature array
        features_array = np.array([[
            current_features['SMA_10'],
            current_features['SMA_30'], 
            current_features['Price_Change'],
            current_features['Volatility'],
            current_features['Sentiment']
        ]])
        
        # Generate prediction and confidence score
        prediction = self.model.predict(features_array)[0]
        confidence = self.model.predict_proba(features_array)[0].max()
        
        signal = "BUY" if prediction == 1 else "SELL"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'prediction_value': prediction
        }
    
    def save_model(self, filepath="trained_model.pkl"):
        """Save trained model to disk"""
        if self.model:
            joblib.dump(self.model, filepath)
    
    def load_model(self, filepath="trained_model.pkl"):
        """Load trained model from disk"""
        try:
            self.model = joblib.load(filepath)
            return True
        except:
            return False
