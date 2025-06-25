from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['JSON_SORT_KEYS'] = False

class StockPredictor:
    def __init__(self):
        self.model_confidence = 0.75  # Simulated model confidence
    
    def get_stock_data(self, symbol, period="1y"):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            info = stock.info
            return data, info
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None, None
    
    def calculate_technical_indicators(self, data):
        """Calculate basic technical indicators"""
        if data is None or len(data) < 20:
            return {}
        
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        latest = data.iloc[-1]
        return {
            'current_price': float(latest['Close']),
            'sma_20': float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
            'sma_50': float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
            'rsi': float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
            'volatility': float(latest['Volatility']) if not pd.isna(latest['Volatility']) else None,
            'volume': int(latest['Volume'])
        }
    
    def predict_stock_movement(self, symbol):
        """Main prediction function"""
        try:
            # Get stock data
            data, info = self.get_stock_data(symbol)
            if data is None:
                return {"error": "Unable to fetch stock data"}
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(data)
            if not indicators:
                return {"error": "Insufficient data for analysis"}
            
            # Simple prediction logic (replace with your LLM model)
            current_price = indicators['current_price']
            sma_20 = indicators.get('sma_20', current_price)
            rsi = indicators.get('rsi', 50)
            
            # Basic trend analysis
            price_trend = (current_price - sma_20) / sma_20 * 100 if sma_20 else 0
            
            # Prediction logic
            if rsi < 30 and price_trend < -5:  # Oversold and downtrend
                prediction = "BUY"
                expected_change = np.random.uniform(3, 8)
                confidence = 0.8
            elif rsi > 70 and price_trend > 5:  # Overbought and uptrend
                prediction = "SELL"
                expected_change = np.random.uniform(-8, -3)
                confidence = 0.75
            elif price_trend > 2:
                prediction = "HOLD/BUY"
                expected_change = np.random.uniform(1, 5)
                confidence = 0.65
            else:
                prediction = "HOLD"
                expected_change = np.random.uniform(-2, 2)
                confidence = 0.6
            
            # Calculate target price
            target_price = current_price * (1 + expected_change / 100)
            
            return {
                "symbol": symbol.upper(),
                "company_name": info.get('longName', symbol.upper()) if info else symbol.upper(),
                "current_price": round(current_price, 2),
                "prediction": prediction,
                "expected_change_percent": round(expected_change, 2),
                "target_price": round(target_price, 2),
                "confidence": round(confidence * 100, 1),
                "timeframe": "1-3 months",
                "technical_indicators": indicators,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error in prediction for {symbol}: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

# Initialize predictor
predictor = StockPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        
        if not symbol:
            return jsonify({"error": "Stock symbol is required"}), 400
        
        prediction = predictor.predict_stock_movement(symbol)
        return jsonify(prediction)
    
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    import os
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Set debug based on environment
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
