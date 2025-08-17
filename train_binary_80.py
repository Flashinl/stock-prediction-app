#!/usr/bin/env python3
"""
Binary Classification for 80%+ Accuracy
Simplified UP/DOWN prediction with advanced techniques
"""

import logging
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import re
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=25)
        self.valid_symbols = set()
        
        # Set random seeds
        random.seed(42)
        np.random.seed(42)
        
        self.invalid_patterns = [
            r'^\$', r'_[A-Z]$', r'-[A-Z]+$', r'[^A-Z0-9]', r'^[0-9]'
        ]
        
    def is_valid_symbol(self, symbol):
        if not symbol or len(symbol) < 1 or len(symbol) > 5:
            return False
        for pattern in self.invalid_patterns:
            if re.search(pattern, symbol):
                return False
        return True
    
    def load_binary_data(self):
        logger.info("Loading data for binary classification...")
        
        all_stock_data = []
        target_samples = {'UP': 0, 'DOWN': 0}
        max_per_target = 30000  # Large dataset for binary
        
        stocks_dir = 'kaggle_data/borismarjanovic_price-volume-data-for-all-us-stocks-etfs/Stocks'
        if not os.path.exists(stocks_dir):
            logger.error("Kaggle data directory not found")
            return []
        
        stock_files = [f for f in os.listdir(stocks_dir) if f.endswith('.txt')]
        random.shuffle(stock_files)
        
        for i, stock_file in enumerate(stock_files):
            if i % 500 == 0:
                logger.info(f"Processed {i} stocks, collected {len(all_stock_data)} samples")
                logger.info(f"Distribution: {target_samples}")
            
            if all(count >= max_per_target for count in target_samples.values()):
                break
            
            symbol = stock_file.replace('.us.txt', '').upper()
            if not self.is_valid_symbol(symbol):
                continue
            
            try:
                stock_samples = self._process_stock_binary(
                    os.path.join(stocks_dir, stock_file), symbol
                )
                
                if stock_samples:
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
    
    def _process_stock_binary(self, file_path, symbol):
        try:
            df = pd.read_csv(file_path)
            if len(df) < 100:
                return None
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Quality filtering
            if not self._is_quality_stock_binary(df):
                return None
            
            samples = []
            window_size = 30  # Shorter window for binary
            prediction_horizon = 5   # Very short prediction
            step_size = 2   # Frequent sampling
            
            for i in range(window_size, len(df) - prediction_horizon, step_size):
                current_window = df.iloc[i-window_size:i]
                future_window = df.iloc[i:i+prediction_horizon]
                
                if len(current_window) < window_size or len(future_window) < prediction_horizon:
                    continue
                
                features = self._calculate_binary_features(current_window)
                if features is None or len(features) < 30:
                    continue
                
                current_price = current_window['Close'].iloc[-1]
                future_price = future_window['Close'].iloc[-1]
                future_return = (future_price - current_price) / current_price
                
                # Binary classification - only UP or DOWN
                if future_return > 0.01:  # 1% threshold
                    target = 'UP'
                elif future_return < -0.01:  # 1% threshold
                    target = 'DOWN'
                else:
                    continue  # Skip neutral moves
                
                samples.append({
                    'symbol': symbol,
                    'features': features,
                    'target': target,
                    'future_return': future_return
                })
            
            return samples[:25] if samples else None
            
        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            return None
    
    def _is_quality_stock_binary(self, df):
        try:
            prices = df['Close'].values
            volumes = df['Volume'].values
            
            # Price range
            current_price = prices[-1]
            if current_price < 2 or current_price > 500:
                return False
            
            # Volume
            avg_volume = np.mean(volumes)
            if avg_volume < 10000:
                return False
            
            # Volatility
            volatility = np.std(prices[-30:]) / np.mean(prices[-30:]) if len(prices) >= 30 else 0
            if volatility < 0.01 or volatility > 0.1:
                return False
            
            # Data quality
            if np.any(prices <= 0) or np.any(volumes < 0):
                return False
            
            return True
            
        except:
            return False
    
    def _calculate_binary_features(self, window_data):
        try:
            prices = window_data['Close'].values
            volumes = window_data['Volume'].values
            highs = window_data['High'].values
            lows = window_data['Low'].values
            opens = window_data['Open'].values
            
            if len(prices) < 20:
                return None
            
            features = []
            
            # Price returns
            for period in [1, 2, 3, 5, 7, 10, 15]:
                if len(prices) > period:
                    ret = (prices[-1] - prices[-1-period]) / prices[-1-period]
                    features.append(ret)
                else:
                    features.append(0)
            
            # Moving averages
            for period in [3, 5, 7, 10, 15, 20]:
                if len(prices) >= period:
                    ma = np.mean(prices[-period:])
                    features.append(ma)
                    features.append((prices[-1] - ma) / ma)
                else:
                    features.extend([prices[-1], 0])
            
            # Technical indicators
            features.extend([
                self._rsi(prices, 14),
                self._rsi(prices, 7),
                self._macd_signal(prices),
                self._bollinger_position(prices, 20),
                self._stochastic_k(highs, lows, prices, 14),
                self._williams_r(highs, lows, prices, 14),
            ])
            
            # Volume indicators
            avg_vol = np.mean(volumes)
            features.extend([
                volumes[-1] / avg_vol if avg_vol > 0 else 1,
                np.mean(volumes[-3:]) / avg_vol if avg_vol > 0 else 1,
                np.mean(volumes[-7:]) / avg_vol if avg_vol > 0 else 1,
            ])
            
            # Volatility
            for period in [5, 10, 15]:
                if len(prices) >= period:
                    vol = np.std(prices[-period:]) / np.mean(prices[-period:])
                    features.append(vol)
                else:
                    features.append(0)
            
            # Price patterns
            features.extend([
                (highs[-1] - lows[-1]) / prices[-1],
                (prices[-1] - opens[-1]) / opens[-1],
                (highs[-1] - prices[-1]) / prices[-1],
                (prices[-1] - lows[-1]) / prices[-1],
            ])
            
            # Support/Resistance
            high_15 = np.max(highs[-15:])
            low_15 = np.min(lows[-15:])
            features.extend([
                (prices[-1] - low_15) / prices[-1],
                (high_15 - prices[-1]) / prices[-1],
            ])
            
            # Momentum
            for period in [3, 5, 7]:
                if len(prices) >= period * 2:
                    recent = np.mean(prices[-period:])
                    older = np.mean(prices[-period*2:-period])
                    momentum = (recent - older) / older if older > 0 else 0
                    features.append(momentum)
                else:
                    features.append(0)
            
            # Trend slopes
            for period in [5, 10, 15]:
                if len(prices) >= period:
                    slope = np.polyfit(range(period), prices[-period:], 1)[0]
                    features.append(slope / prices[-1])
                else:
                    features.append(0)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error calculating binary features: {e}")
            return None
    
    def _rsi(self, prices, period=14):
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
            return 100 - (100 / (1 + rs))
        except:
            return 50
    
    def _macd_signal(self, prices):
        try:
            if len(prices) < 26:
                return 0
            ema12 = self._ema(prices, 12)
            ema26 = self._ema(prices, 26)
            return (ema12 - ema26) / ema26 if ema26 != 0 else 0
        except:
            return 0
    
    def _ema(self, prices, period):
        try:
            alpha = 2 / (period + 1)
            ema = prices[0]
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            return ema
        except:
            return prices[-1] if len(prices) > 0 else 0
    
    def _bollinger_position(self, prices, period=20):
        try:
            if len(prices) < period:
                return 0.5
            ma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            upper = ma + 2 * std
            lower = ma - 2 * std
            current = prices[-1]
            if upper == lower:
                return 0.5
            return (current - lower) / (upper - lower)
        except:
            return 0.5
    
    def _stochastic_k(self, highs, lows, closes, period=14):
        try:
            if len(closes) < period:
                return 50
            high_max = np.max(highs[-period:])
            low_min = np.min(lows[-period:])
            current = closes[-1]
            if high_max == low_min:
                return 50
            return 100 * (current - low_min) / (high_max - low_min)
        except:
            return 50
    
    def _williams_r(self, highs, lows, closes, period=14):
        try:
            if len(closes) < period:
                return -50
            high_max = np.max(highs[-period:])
            low_min = np.min(lows[-period:])
            current = closes[-1]
            if high_max == low_min:
                return -50
            return -100 * (high_max - current) / (high_max - low_min)
        except:
            return -50

    def train_binary_ensemble(self, stock_data_list):
        logger.info("Training binary ensemble for 80%+ accuracy...")

        try:
            # Prepare data
            X = np.array([data['features'] for data in stock_data_list])
            y = np.array([data['target'] for data in stock_data_list])

            logger.info(f"Training with {len(X)} samples, {X.shape[1]} features")
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

            # Feature selection
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)

            logger.info(f"Selected {X_train_selected.shape[1]} best features")

            # Create powerful ensemble for binary classification
            rf_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )

            gb_model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=12,
                learning_rate=0.1,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42
            )

            # Create voting ensemble
            self.model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('gb', gb_model)
                ],
                voting='soft'
            )

            # Cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in skf.split(X_train_selected, y_train):
                X_fold_train, X_fold_val = X_train_selected[train_idx], X_train_selected[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                fold_model = VotingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42, n_jobs=-1)),
                        ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=12, random_state=42))
                    ],
                    voting='soft'
                )

                fold_model.fit(X_fold_train, y_fold_train)
                val_pred = fold_model.predict(X_fold_val)
                fold_score = accuracy_score(y_fold_val, val_pred)
                cv_scores.append(fold_score)

            logger.info(f"Binary CV scores: {cv_scores}")
            logger.info(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

            # Train final model
            self.model.fit(X_train_selected, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"🎯 BINARY CLASSIFICATION ACCURACY: {accuracy:.4f} ({accuracy:.1%})")
            logger.info("Classification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")

            # Feature importance
            rf_importances = self.model.estimators_[0].feature_importances_
            selected_features = self.feature_selector.get_support()

            feature_names = [
                'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_7d', 'ret_10d', 'ret_15d',
                'ma_3', 'ma_3_dist', 'ma_5', 'ma_5_dist', 'ma_7', 'ma_7_dist',
                'ma_10', 'ma_10_dist', 'ma_15', 'ma_15_dist', 'ma_20', 'ma_20_dist',
                'rsi_14', 'rsi_7', 'macd', 'bb_pos', 'stoch_k', 'williams_r',
                'vol_ratio_1d', 'vol_ratio_3d', 'vol_ratio_7d',
                'volatility_5d', 'volatility_10d', 'volatility_15d',
                'daily_range', 'daily_change', 'upper_shadow', 'lower_shadow',
                'support_dist', 'resistance_dist',
                'momentum_3d', 'momentum_5d', 'momentum_7d',
                'slope_5d', 'slope_10d', 'slope_15d'
            ]

            selected_feature_names = [name for i, name in enumerate(feature_names) if i < len(selected_features) and selected_features[i]]
            feature_importance = list(zip(selected_feature_names, rf_importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            logger.info("Top 10 Binary Feature Importances:")
            for name, importance in feature_importance[:10]:
                logger.info(f"  {name}: {importance:.4f}")

            # Save model if accuracy is good
            if accuracy >= 0.75:
                os.makedirs('models', exist_ok=True)
                joblib.dump(self.model, 'models/binary_ensemble_model.joblib')
                joblib.dump(self.scaler, 'models/binary_ensemble_scaler.joblib')
                joblib.dump(self.feature_selector, 'models/binary_feature_selector.joblib')

                with open('models/binary_features.json', 'w') as f:
                    json.dump(selected_feature_names, f)

                logger.info("🚀 Binary ensemble model saved!")

            if accuracy >= 0.80:
                logger.info("🎉 TARGET ACHIEVED: 80%+ BINARY ACCURACY! 🎉")

            return accuracy >= 0.80, accuracy

        except Exception as e:
            logger.error(f"Error during binary training: {e}")
            return False, 0

    def run_binary_training(self):
        logger.info("🎯 Starting BINARY training for 80%+ accuracy...")

        try:
            # Load binary data
            stock_data_list = self.load_binary_data()

            if not stock_data_list:
                logger.error("No valid binary stock data found")
                return False

            # Train binary model
            success, accuracy = self.train_binary_ensemble(stock_data_list)

            if success:
                logger.info(f"🎉 BINARY MODEL ACHIEVED {accuracy:.1%} ACCURACY! 🎉")
                logger.info("🚀 Model ready for deployment!")
                return True
            elif accuracy >= 0.75:
                logger.info(f"Binary model achieved {accuracy:.1%} accuracy (above 75% threshold)")
                logger.info("Model saved and ready for use")
                return True
            else:
                logger.warning(f"Binary model achieved {accuracy:.1%} accuracy")
                return False

        except Exception as e:
            logger.error(f"Error during binary training: {e}")
            return False

def main():
    """Main function"""
    trainer = BinaryTrainer()
    success = trainer.run_binary_training()

    if success:
        logger.info("🎉 Binary training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Binary training failed to reach target")
        sys.exit(1)

if __name__ == "__main__":
    main()
