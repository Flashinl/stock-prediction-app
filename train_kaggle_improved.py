#!/usr/bin/env python3
"""
Improved Kaggle Stock Market Training
Uses historical backtesting for proper target generation
"""

import logging
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedKaggleTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.valid_symbols = set()
        self.invalid_symbols = set()
        
        # Common patterns for invalid symbols
        self.invalid_patterns = [
            r'^\$',  # Symbols starting with $
            r'_[A-Z]$',  # Symbols ending with _A, _B, etc.
            r'-[A-Z]+$',  # Symbols ending with -CL, -A, etc.
            r'[^A-Z0-9]',  # Symbols with special characters
            r'^[0-9]',  # Symbols starting with numbers
        ]
        
    def is_valid_symbol(self, symbol):
        """Check if a stock symbol is valid and not delisted"""
        if not symbol or len(symbol) < 1 or len(symbol) > 5:
            return False
            
        # Check against invalid patterns
        for pattern in self.invalid_patterns:
            if re.search(pattern, symbol):
                return False
                
        return True
    
    def load_kaggle_stock_data(self):
        """Load and process Kaggle stock data with historical backtesting"""
        logger.info("Loading Kaggle stock data...")
        
        all_stock_data = []
        processed_symbols = set()
        
        # Process main comprehensive dataset
        stocks_dir = 'kaggle_data/borismarjanovic_price-volume-data-for-all-us-stocks-etfs/Stocks'
        if os.path.exists(stocks_dir):
            stock_files = [f for f in os.listdir(stocks_dir) if f.endswith('.txt')]
            logger.info(f"Processing {len(stock_files)} stock files...")
            
            for i, stock_file in enumerate(stock_files):
                if i % 1000 == 0:
                    logger.info(f"Processed {i}/{len(stock_files)} stocks...")
                
                symbol = stock_file.replace('.us.txt', '').upper()
                
                if symbol in processed_symbols or not self.is_valid_symbol(symbol):
                    continue
                
                try:
                    stock_samples = self._process_kaggle_stock_file_with_backtest(
                        os.path.join(stocks_dir, stock_file), symbol
                    )
                    if stock_samples:
                        all_stock_data.extend(stock_samples)
                        processed_symbols.add(symbol)
                        self.valid_symbols.add(symbol)
                except Exception as e:
                    logger.debug(f"Error processing {symbol}: {e}")
                    self.invalid_symbols.add(symbol)
                    continue
        
        logger.info(f"Successfully processed {len(all_stock_data)} training samples from {len(self.valid_symbols)} stocks")
        return all_stock_data
    
    def _process_kaggle_stock_file_with_backtest(self, file_path, symbol):
        """Process stock file with historical backtesting approach"""
        try:
            df = pd.read_csv(file_path)
            
            if len(df) < 200:  # Need enough data for backtesting
                return None
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            samples = []
            
            # Create multiple training samples from historical data
            # Use sliding window approach
            window_size = 60  # 60 days of data to predict next 30 days
            prediction_horizon = 30  # Predict 30 days ahead
            
            for i in range(window_size, len(df) - prediction_horizon):
                # Get current window of data
                current_window = df.iloc[i-window_size:i]
                
                # Get future data for target generation
                future_window = df.iloc[i:i+prediction_horizon]
                
                if len(current_window) < window_size or len(future_window) < prediction_horizon:
                    continue
                
                # Calculate features from current window
                features = self._calculate_features(current_window)
                
                # Calculate target from future performance
                current_price = current_window['Close'].iloc[-1]
                future_price = future_window['Close'].iloc[-1]
                future_return = (future_price - current_price) / current_price
                
                # Generate target based on future return
                if future_return > 0.10:  # 10% gain
                    target = 'BUY'
                elif future_return < -0.10:  # 10% loss
                    target = 'SELL'
                else:
                    target = 'HOLD'
                
                if features is not None:
                    samples.append({
                        'symbol': symbol,
                        'features': features,
                        'target': target,
                        'future_return': future_return
                    })
            
            return samples if len(samples) > 10 else None  # Need at least 10 samples per stock
            
        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            return None
    
    def _calculate_features(self, window_data):
        """Calculate features from a window of historical data"""
        try:
            prices = window_data['Close'].values
            volumes = window_data['Volume'].values
            
            if len(prices) < 20:  # Need minimum data
                return None
            
            # Basic price features
            current_price = prices[-1]
            price_change_1d = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            price_change_5d = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0
            price_change_20d = (prices[-1] - prices[-21]) / prices[-21] if len(prices) > 20 else 0
            
            # Moving averages
            ma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else current_price
            ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
            ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
            
            # Volume features
            avg_volume = np.mean(volumes)
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            # Volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # RSI
            rsi = self._calculate_rsi(prices)
            
            # Trend features
            ma_trend = 1 if ma_5 > ma_20 > ma_50 else 0
            price_above_ma = 1 if current_price > ma_20 else 0
            
            # Momentum features
            momentum_5 = np.mean(prices[-5:]) / np.mean(prices[-10:-5]) - 1 if len(prices) >= 10 else 0
            momentum_20 = np.mean(prices[-20:]) / np.mean(prices[-40:-20]) - 1 if len(prices) >= 40 else 0
            
            return [
                current_price,
                price_change_1d,
                price_change_5d,
                price_change_20d,
                ma_5,
                ma_20,
                ma_50,
                volumes[-1],
                avg_volume,
                volume_ratio,
                volatility,
                rsi,
                ma_trend,
                price_above_ma,
                momentum_5,
                momentum_20
            ]
            
        except Exception as e:
            logger.debug(f"Error calculating features: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50
                
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except:
            return 50
    
    def train_model(self, stock_data_list):
        """Train Random Forest model with improved features"""
        logger.info("Training Random Forest model...")
        
        try:
            # Prepare features and targets
            X = []
            y = []
            
            for data in stock_data_list:
                X.append(data['features'])
                y.append(data['target'])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Training with {len(X)} samples")
            
            # Count target distribution
            unique, counts = np.unique(y, return_counts=True)
            target_dist = dict(zip(unique, counts))
            logger.info(f"Target distribution: {target_dist}")
            
            # Check if we have enough diversity
            if len(unique) < 2:
                logger.warning("Not enough target diversity for meaningful training")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42, 
                n_jobs=-1,
                class_weight='balanced'
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model accuracy: {accuracy:.4f}")
            logger.info("Classification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")
            
            # Feature importance
            feature_names = [
                'current_price', 'price_change_1d', 'price_change_5d', 'price_change_20d',
                'ma_5', 'ma_20', 'ma_50', 'volume', 'avg_volume', 'volume_ratio',
                'volatility', 'rsi', 'ma_trend', 'price_above_ma', 'momentum_5', 'momentum_20'
            ]
            
            importances = self.model.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Top 10 Feature Importances:")
            for name, importance in feature_importance[:10]:
                logger.info(f"  {name}: {importance:.4f}")
            
            # Save model
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, 'models/kaggle_improved_model.joblib')
            joblib.dump(self.scaler, 'models/kaggle_improved_scaler.joblib')
            
            # Save feature names
            with open('models/kaggle_improved_features.json', 'w') as f:
                json.dump(feature_names, f)
            
            logger.info("Model saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False
    
    def save_training_report(self, stock_data_list):
        """Save comprehensive training report"""
        try:
            target_counts = {}
            return_stats = {'BUY': [], 'HOLD': [], 'SELL': []}
            
            for data in stock_data_list:
                target = data['target']
                target_counts[target] = target_counts.get(target, 0) + 1
                if 'future_return' in data:
                    return_stats[target].append(data['future_return'])
            
            # Calculate return statistics
            return_summary = {}
            for target, returns in return_stats.items():
                if returns:
                    return_summary[target] = {
                        'count': len(returns),
                        'mean_return': np.mean(returns),
                        'std_return': np.std(returns),
                        'min_return': np.min(returns),
                        'max_return': np.max(returns)
                    }
            
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_training_samples': len(stock_data_list),
                'unique_stocks': len(self.valid_symbols),
                'target_distribution': target_counts,
                'return_statistics': return_summary,
                'feature_names': [
                    'current_price', 'price_change_1d', 'price_change_5d', 'price_change_20d',
                    'ma_5', 'ma_20', 'ma_50', 'volume', 'avg_volume', 'volume_ratio',
                    'volatility', 'rsi', 'ma_trend', 'price_above_ma', 'momentum_5', 'momentum_20'
                ]
            }
            
            os.makedirs('reports', exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_file = f'reports/kaggle_improved_training_{timestamp}.json'
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Training report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Error saving training report: {e}")
    
    def run_training(self):
        """Run the complete training process"""
        logger.info("Starting improved Kaggle training...")
        
        try:
            # Load Kaggle data
            stock_data_list = self.load_kaggle_stock_data()
            
            if not stock_data_list:
                logger.error("No valid stock data found")
                return False
            
            # Train model
            success = self.train_model(stock_data_list)
            
            # Save report
            self.save_training_report(stock_data_list)
            
            if success:
                logger.info("Improved Kaggle training completed successfully!")
                return True
            else:
                logger.error("Training failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during training process: {e}")
            return False

def main():
    """Main training function"""
    trainer = ImprovedKaggleTrainer()
    success = trainer.run_training()
    
    if success:
        logger.info("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
