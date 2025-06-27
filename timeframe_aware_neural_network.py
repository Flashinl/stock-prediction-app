#!/usr/bin/env python3
"""
Timeframe-Aware Neural Network for Stock Prediction
Addresses the straight-line prediction issue by incorporating timeframe as a feature
and predicting both direction AND realistic price movements
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf
from datetime import datetime, timedelta
import logging
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeframeAwareStockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        
        # Timeframe encoding
        self.timeframes = {
            '1_month': 30,
            '3_months': 90,
            '6_months': 180,
            '12_months': 365
        }
    
    def extract_features_with_timeframe(self, symbol, timeframe_days):
        """Extract features including timeframe information"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            info = ticker.info
            
            if hist.empty:
                return None
            
            # Basic price and volume data
            current_price = float(hist['Close'].iloc[-1])
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
            
            # Technical indicators
            close = hist['Close']
            
            # Moving averages
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            ema_12 = close.ewm(span=12).mean().iloc[-1]
            ema_26 = close.ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            
            # Volatility
            volatility = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Price momentum
            price_momentum = ((current_price - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) > 21 else 0
            
            # Market cap and sector info
            market_cap = info.get('marketCap', 0)
            sector = info.get('sector', 'Unknown')
            
            # Timeframe features - THIS IS THE KEY ADDITION
            timeframe_normalized = timeframe_days / 365.0  # Normalize to 0-1 scale
            timeframe_short = 1 if timeframe_days <= 60 else 0
            timeframe_medium = 1 if 60 < timeframe_days <= 180 else 0
            timeframe_long = 1 if timeframe_days > 180 else 0
            
            # Timeframe-adjusted volatility (longer timeframes should expect more movement)
            timeframe_volatility_factor = np.sqrt(timeframe_days / 30.0)
            
            features = {
                # Basic features
                'current_price': current_price,
                'volume': volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume / avg_volume if avg_volume > 0 else 1,
                
                # Technical indicators
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi if not pd.isna(rsi) else 50,
                'macd': macd if not pd.isna(macd) else 0,
                'volatility': volatility if not pd.isna(volatility) else 0.02,
                'price_momentum': price_momentum,
                
                # Market context
                'market_cap_log': np.log(market_cap) if market_cap > 0 else 15,
                
                # TIMEFRAME FEATURES - The missing piece!
                'timeframe_days': timeframe_days,
                'timeframe_normalized': timeframe_normalized,
                'timeframe_short': timeframe_short,
                'timeframe_medium': timeframe_medium,
                'timeframe_long': timeframe_long,
                'timeframe_volatility_factor': timeframe_volatility_factor,
                
                # Interaction features (timeframe * technical indicators)
                'timeframe_x_volatility': timeframe_normalized * (volatility if not pd.isna(volatility) else 0.02),
                'timeframe_x_momentum': timeframe_normalized * price_momentum,
                'timeframe_x_rsi': timeframe_normalized * (rsi if not pd.isna(rsi) else 50),
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            return None
    
    def build_multi_output_model(self, input_dim):
        """Build a multi-output model that predicts both direction and price targets"""
        
        # Input layer
        inputs = keras.Input(shape=(input_dim,), name='features')
        
        # Shared layers
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Branch 1: Direction prediction (BUY/HOLD/SELL)
        direction_branch = layers.Dense(32, activation='relu', name='direction_dense')(x)
        direction_output = layers.Dense(3, activation='softmax', name='direction')(direction_branch)
        
        # Branch 2: Price change magnitude prediction (regression)
        magnitude_branch = layers.Dense(32, activation='relu', name='magnitude_dense')(x)
        magnitude_output = layers.Dense(1, activation='linear', name='magnitude')(magnitude_branch)
        
        # Branch 3: Confidence/volatility prediction
        confidence_branch = layers.Dense(16, activation='relu', name='confidence_dense')(x)
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(confidence_branch)
        
        # Create model
        model = keras.Model(
            inputs=inputs,
            outputs=[direction_output, magnitude_output, confidence_output],
            name='timeframe_aware_stock_predictor'
        )
        
        # Compile with multiple loss functions
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'direction': 'sparse_categorical_crossentropy',
                'magnitude': 'mse',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={
                'direction': 1.0,
                'magnitude': 0.5,
                'confidence': 0.3
            },
            metrics={
                'direction': ['accuracy'],
                'magnitude': ['mae'],
                'confidence': ['accuracy']
            }
        )
        
        return model
    
    def predict_with_timeframe(self, symbol, timeframe_days):
        """Make timeframe-aware prediction"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Extract features with timeframe
        features = self.extract_features_with_timeframe(symbol, timeframe_days)
        if not features:
            return None
        
        # Convert to DataFrame and scale
        df = pd.DataFrame([features])
        
        # Ensure all feature columns exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder and scale
        df = df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(df.values)
        
        # Make prediction
        direction_probs, magnitude_pred, confidence_pred = self.model.predict(X_scaled, verbose=0)
        
        # Process outputs
        direction_idx = np.argmax(direction_probs[0])
        direction_labels = ['SELL', 'HOLD', 'BUY']
        direction = direction_labels[direction_idx]
        direction_confidence = direction_probs[0][direction_idx] * 100
        
        predicted_change = magnitude_pred[0][0]
        model_confidence = confidence_pred[0][0] * 100
        
        # Generate realistic price movement data
        current_price = features['current_price']
        prediction_data = self._generate_realistic_price_movement(
            current_price, predicted_change, timeframe_days, features['volatility']
        )
        
        return {
            'direction': direction,
            'direction_confidence': direction_confidence,
            'predicted_change_percent': predicted_change,
            'model_confidence': model_confidence,
            'current_price': current_price,
            'target_price': current_price * (1 + predicted_change / 100),
            'timeframe_days': timeframe_days,
            'prediction_data': prediction_data,
            'features_used': features
        }
    
    def _generate_realistic_price_movement(self, current_price, total_change_percent, timeframe_days, volatility):
        """Generate realistic price movement with market-like volatility"""
        
        # Create daily price movements
        num_points = min(timeframe_days, 100)  # Limit for performance
        
        # Calculate daily drift and volatility
        daily_drift = (total_change_percent / 100) / timeframe_days
        daily_vol = volatility / np.sqrt(252)  # Convert annual to daily
        
        # Generate random walk with drift
        np.random.seed(hash(f"{current_price}_{total_change_percent}_{timeframe_days}") % 2**32)
        
        prices = [current_price]
        for i in range(num_points - 1):
            # Random component
            random_change = np.random.normal(0, daily_vol)
            
            # Trend component (stronger at the beginning, weaker at the end)
            trend_strength = 1 - (i / num_points) * 0.3
            trend_change = daily_drift * trend_strength
            
            # Mean reversion component (prevents extreme movements)
            current_total_change = (prices[-1] - current_price) / current_price
            target_change_so_far = (total_change_percent / 100) * (i / num_points)
            reversion = (target_change_so_far - current_total_change) * 0.1
            
            # Combine all components
            total_daily_change = trend_change + random_change + reversion
            
            new_price = prices[-1] * (1 + total_daily_change)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        # Create data points for chart
        prediction_data = []
        for i, price in enumerate(prices):
            days_from_now = (i / len(prices)) * timeframe_days
            prediction_data.append({
                'date': (datetime.now() + timedelta(days=days_from_now)).strftime('%Y-%m-%d'),
                'price': round(price, 2)
            })
        
        return prediction_data

if __name__ == "__main__":
    # Test the new timeframe-aware predictor
    predictor = TimeframeAwareStockPredictor()
    
    # Test feature extraction
    features = predictor.extract_features_with_timeframe('AAPL', 90)
    if features:
        print("Timeframe-aware features extracted successfully!")
        print(f"Timeframe features: {[k for k in features.keys() if 'timeframe' in k]}")
    else:
        print("Failed to extract features")
