#!/usr/bin/env python3
"""
Production Model Training - Optimized for 80%+ Accuracy
Uses proven data quality with memory-efficient sampling
"""

import logging
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import re
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.valid_symbols = set()
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Invalid symbol patterns
        self.invalid_patterns = [
            r'^\$', r'_[A-Z]$', r'-[A-Z]+$', r'[^A-Z0-9]', r'^[0-9]'
        ]
        
    def is_valid_symbol(self, symbol):
        """Check if a stock symbol is valid"""
        if not symbol or len(symbol) < 1 or len(symbol) > 5:
            return False
        for pattern in self.invalid_patterns:
            if re.search(pattern, symbol):
                return False
        return True
    
    def load_production_data(self):
        """Load high-quality production training data"""
        logger.info("Loading production training data...")
        
        all_stock_data = []
        target_samples = {'BUY': 0, 'HOLD': 0, 'SELL': 0}
        max_per_target = 15000  # 15K samples per target = 45K total
        
        stocks_dir = 'kaggle_data/borismarjanovic_price-volume-data-for-all-us-stocks-etfs/Stocks'
        if not os.path.exists(stocks_dir):
            logger.error("Kaggle data directory not found")
            return []
        
        stock_files = [f for f in os.listdir(stocks_dir) if f.endswith('.txt')]
        random.shuffle(stock_files)  # Random sampling
        
        logger.info(f"Processing stocks for balanced dataset...")
        
        for i, stock_file in enumerate(stock_files):
            if i % 500 == 0:
                logger.info(f"Processed {i} stocks, collected {len(all_stock_data)} samples")
                logger.info(f"Current distribution: {target_samples}")
            
            # Stop if we have enough samples
            if all(count >= max_per_target for count in target_samples.values()):
                break
            
            symbol = stock_file.replace('.us.txt', '').upper()
            if not self.is_valid_symbol(symbol):
                continue
            
            try:
                stock_samples = self._process_stock_for_production(
                    os.path.join(stocks_dir, stock_file), symbol
                )
                
                if stock_samples:
                    # Add samples while maintaining balance
                    for sample in stock_samples:
                        target = sample['target']
                        if target_samples[target] < max_per_target:
                            all_stock_data.append(sample)
                            target_samples[target] += 1
                            
                    self.valid_symbols.add(symbol)
                    
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
        
        logger.info(f"Final dataset: {len(all_stock_data)} samples from {len(self.valid_symbols)} stocks")
        logger.info(f"Target distribution: {target_samples}")
        return all_stock_data
    
    def _process_stock_for_production(self, file_path, symbol):
        """Process stock with production-quality features"""
        try:
            df = pd.read_csv(file_path)
            if len(df) < 200:
                return None
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            samples = []
            window_size = 60
            prediction_horizon = 30
            step_size = 20  # Larger steps for efficiency
            
            for i in range(window_size, len(df) - prediction_horizon, step_size):
                current_window = df.iloc[i-window_size:i]
                future_window = df.iloc[i:i+prediction_horizon]
                
                if len(current_window) < window_size or len(future_window) < prediction_horizon:
                    continue
                
                features = self._calculate_production_features(current_window)
                if features is None:
                    continue
                
                # Calculate future return
                current_price = current_window['Close'].iloc[-1]
                future_price = future_window['Close'].iloc[-1]
                future_return = (future_price - current_price) / current_price
                
                # High-quality target generation (proven to work)
                if future_return > 0.12:  # 12% gain threshold
                    target = 'BUY'
                elif future_return < -0.12:  # 12% loss threshold
                    target = 'SELL'
                else:
                    target = 'HOLD'
                
                samples.append({
                    'symbol': symbol,
                    'features': features,
                    'target': target,
                    'future_return': future_return
                })
            
            return samples[:5] if samples else None  # Max 5 samples per stock
            
        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            return None
    
    def _calculate_production_features(self, window_data):
        """Calculate production-quality features"""
        try:
            prices = window_data['Close'].values
            volumes = window_data['Volume'].values
            
            if len(prices) < 30:
                return None
            
            # Price features
            current_price = prices[-1]
            returns_1d = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            returns_5d = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0
            returns_20d = (prices[-1] - prices[-21]) / prices[-21] if len(prices) > 20 else 0
            
            # Moving averages
            ma_5 = np.mean(prices[-5:])
            ma_10 = np.mean(prices[-10:])
            ma_20 = np.mean(prices[-20:])
            ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else ma_20
            
            # Technical indicators
            rsi = self._calculate_rsi(prices)
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            
            # Volume analysis
            avg_volume = np.mean(volumes)
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1
            
            # Trend features
            price_vs_ma5 = current_price / ma_5 - 1
            price_vs_ma20 = current_price / ma_20 - 1
            ma_slope = (ma_5 - ma_20) / ma_20
            
            # Momentum features
            momentum_short = np.mean(prices[-5:]) / np.mean(prices[-10:]) - 1
            momentum_long = np.mean(prices[-10:]) / np.mean(prices[-30:]) - 1
            
            # Price position in range
            price_range = np.max(prices[-20:]) - np.min(prices[-20:])
            price_position = (current_price - np.min(prices[-20:])) / price_range if price_range > 0 else 0.5
            
            return [
                current_price,
                returns_1d,
                returns_5d, 
                returns_20d,
                ma_5,
                ma_10,
                ma_20,
                ma_50,
                rsi,
                volatility,
                volume_ratio,
                volume_trend,
                price_vs_ma5,
                price_vs_ma20,
                ma_slope,
                momentum_short,
                momentum_long,
                price_position
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
    
    def train_production_model(self, stock_data_list):
        """Train production model with cross-validation"""
        logger.info("Training production model...")
        
        try:
            # Prepare data
            X = np.array([data['features'] for data in stock_data_list])
            y = np.array([data['target'] for data in stock_data_list])
            
            logger.info(f"Training with {len(X)} samples")
            unique, counts = np.unique(y, return_counts=True)
            target_dist = dict(zip(unique, counts))
            logger.info(f"Target distribution: {target_dist}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train optimized model
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Train final model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Test accuracy: {accuracy:.4f}")
            logger.info("Classification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Confusion Matrix:\n{cm}")
            
            # Feature importance
            feature_names = [
                'current_price', 'returns_1d', 'returns_5d', 'returns_20d',
                'ma_5', 'ma_10', 'ma_20', 'ma_50', 'rsi', 'volatility',
                'volume_ratio', 'volume_trend', 'price_vs_ma5', 'price_vs_ma20',
                'ma_slope', 'momentum_short', 'momentum_long', 'price_position'
            ]
            
            importances = self.model.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Top 10 Feature Importances:")
            for name, importance in feature_importance[:10]:
                logger.info(f"  {name}: {importance:.4f}")
            
            # Save model if accuracy is good
            if accuracy >= 0.75:  # 75% threshold
                os.makedirs('models', exist_ok=True)
                joblib.dump(self.model, 'models/production_stock_model.joblib')
                joblib.dump(self.scaler, 'models/production_stock_scaler.joblib')
                
                with open('models/production_features.json', 'w') as f:
                    json.dump(feature_names, f)
                
                logger.info("Production model saved successfully!")
            
            return accuracy >= 0.80, accuracy  # Return if we hit 80%+ target
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False, 0

    def save_production_report(self, stock_data_list, accuracy):
        """Save production training report"""
        try:
            target_counts = {}
            return_stats = {'BUY': [], 'HOLD': [], 'SELL': []}

            for data in stock_data_list:
                target = data['target']
                target_counts[target] = target_counts.get(target, 0) + 1
                if 'future_return' in data:
                    return_stats[target].append(data['future_return'])

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
                'model_type': 'Production Random Forest',
                'total_training_samples': len(stock_data_list),
                'unique_stocks': len(self.valid_symbols),
                'target_distribution': target_counts,
                'return_statistics': return_summary,
                'model_accuracy': accuracy,
                'accuracy_threshold_met': accuracy >= 0.80,
                'feature_count': 18,
                'model_parameters': {
                    'n_estimators': 300,
                    'max_depth': 15,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'class_weight': 'balanced'
                }
            }

            os.makedirs('reports', exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_file = f'reports/production_training_{timestamp}.json'

            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Production report saved to {report_file}")

        except Exception as e:
            logger.error(f"Error saving report: {e}")

    def run_production_training(self):
        """Run complete production training"""
        logger.info("Starting production model training...")

        try:
            # Load data
            stock_data_list = self.load_production_data()

            if not stock_data_list:
                logger.error("No valid stock data found")
                return False

            # Train model
            success, accuracy = self.train_production_model(stock_data_list)

            # Save report
            self.save_production_report(stock_data_list, accuracy)

            if success:
                logger.info(f"🎉 PRODUCTION MODEL ACHIEVED {accuracy:.1%} ACCURACY! 🎉")
                logger.info("Model ready for deployment!")
                return True
            elif accuracy >= 0.75:
                logger.info(f"Model achieved {accuracy:.1%} accuracy (above 75% threshold)")
                logger.info("Model saved and ready for use")
                return True
            else:
                logger.warning(f"Model achieved {accuracy:.1%} accuracy (below 75% threshold)")
                return False

        except Exception as e:
            logger.error(f"Error during production training: {e}")
            return False

def main():
    """Main function"""
    trainer = ProductionModelTrainer()
    success = trainer.run_production_training()

    if success:
        logger.info("Production training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Production training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
