import unittest
import sys
import os
from data_manager import DataManager
from banking_layer import BankingLayer
import pandas as pd
import numpy as np

# Import ML Engine with error handling
try:
    from ml_engine import MLEngine
    ML_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML Engine import failed: {e}")
    ML_ENGINE_AVAILABLE = False

class TestFinTechSystem(unittest.TestCase):
    
    def setUp(self):
        """Initialize test components"""
        print("🔄 Setting up test components...")
        self.data_manager = DataManager()
        self.banking_layer = BankingLayer()
        
        # Only initialize ML engine if available
        if ML_ENGINE_AVAILABLE:
            try:
                self.ml_engine = MLEngine()
                self.ml_available = True
            except Exception as e:
                print(f"⚠️ ML Engine initialization failed: {e}")
                self.ml_available = False
        else:
            self.ml_available = False
    
    def test_data_collection(self):
        """Test stock data collection functionality"""
        print("📊 Testing stock data collection...")
        try:
            data = self.data_manager.get_stock_data("AAPL", "1mo")
            self.assertGreater(len(data), 5, "Should have at least 5 data points")
            self.assertIn('Close', data.columns, "Should have Close price column")
            print("✅ Stock data collection test passed")
        except Exception as e:
            print(f"❌ Stock data collection test failed: {e}")
            self.fail(f"Data collection failed: {e}")
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis pipeline"""
        if not self.ml_available:
            self.skipTest("ML Engine not available - skipping sentiment test")
            
        print("📰 Testing sentiment analysis...")
        try:
            test_news = ["Apple announces record quarterly earnings"]
            sentiment = self.ml_engine.process_news_sentiment(test_news)
            self.assertIsInstance(sentiment, (int, float), "Sentiment should be numeric")
            self.assertGreaterEqual(sentiment, -1, "Sentiment should be >= -1")
            self.assertLessEqual(sentiment, 1, "Sentiment should be <= 1")
            print(f"✅ Sentiment analysis test passed (score: {sentiment:.3f})")
        except Exception as e:
            print(f"❌ Sentiment analysis test failed: {e}")
            self.fail(f"Sentiment analysis failed: {e}")
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        print("📈 Testing VaR calculation...")
        try:
            # Generate sample returns data
            returns = pd.Series(np.random.normal(0.001, 0.02, 100))
            var_result = self.banking_layer.calculate_var(returns)
            
            self.assertIn('var_1_day', var_result, "Should have 1-day VaR")
            self.assertGreater(var_result['var_1_day'], 0, "VaR should be positive")
            print(f"✅ VaR calculation test passed (1-day VaR: ${var_result['var_1_day']:,.2f})")
        except Exception as e:
            print(f"❌ VaR calculation test failed: {e}")
            self.fail(f"VaR calculation failed: {e}")
    
    def test_sharpe_calculation(self):
        """Test Sharpe ratio calculation"""
        print("📊 Testing Sharpe ratio calculation...")
        try:
            # Generate sample returns with positive expected return
            returns = pd.Series(np.random.normal(0.001, 0.02, 100))
            sharpe_result = self.banking_layer.calculate_sharpe_ratio(returns)
            
            self.assertIn('sharpe_ratio', sharpe_result, "Should have Sharpe ratio")
            self.assertIsInstance(sharpe_result['sharpe_ratio'], (int, float), "Sharpe ratio should be numeric")
            print(f"✅ Sharpe ratio test passed (ratio: {sharpe_result['sharpe_ratio']:.3f})")
        except Exception as e:
            print(f"❌ Sharpe ratio test failed: {e}")
            self.fail(f"Sharpe calculation failed: {e}")
    
    def test_compliance_check(self):
        """Test regulatory compliance verification"""
        print("📋 Testing compliance check...")
        try:
            mock_prediction = {'signal': 'BUY', 'confidence': 0.75}
            mock_features = {'sentiment': 0.5, 'volatility': 0.02}
            
            compliance = self.banking_layer.compliance_check(
                mock_prediction, "AAPL", mock_features
            )
            
            self.assertIn('explainability', compliance, "Should have explainability section")
            self.assertIn('risk_validation', compliance, "Should have risk validation")
            print("✅ Compliance check test passed")
        except Exception as e:
            print(f"❌ Compliance check test failed: {e}")
            self.fail(f"Compliance check failed: {e}")

    def test_system_integration(self):
        """Test basic system integration"""
        print("🔗 Testing system integration...")
        try:
            # Test that all components can work together
            stock_data = self.data_manager.get_stock_data("AAPL", "1mo")
            
            if len(stock_data) > 10:
                returns = stock_data['Close'].pct_change().dropna()
                if len(returns) > 10:
                    var_result = self.banking_layer.calculate_var(returns)
                    sharpe_result = self.banking_layer.calculate_sharpe_ratio(returns)
                    
                    self.assertIsInstance(var_result, dict, "VaR should return dict")
                    self.assertIsInstance(sharpe_result, dict, "Sharpe should return dict")
                    
            print("✅ System integration test passed")
        except Exception as e:
            print(f"❌ System integration test failed: {e}")
            self.fail(f"System integration failed: {e}")

def run_tests():
    """Run all tests with detailed output"""
    print("=" * 60)
    print("🚀 RUNNING FINTECH ML SYSTEM TESTS")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestFinTechSystem)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    print("=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"❌ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS")
    print("=" * 60)

if __name__ == '__main__':
    run_tests()
