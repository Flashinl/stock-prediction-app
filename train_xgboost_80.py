#!/usr/bin/env python3
"""
XGBoost Ensemble for 80%+ Accuracy
Advanced gradient boosting with sophisticated feature engineering
"""

import logging
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import re
import random

# Try XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XGBoostTrainer:
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(f_classif, k=30)
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
    
    def load_xgboost_data(self):
        logger.info("Loading data for XGBoost training...")
        
        all_stock_data = []
        target_samples = {'BUY': 0, 'HOLD': 0, 'SELL': 0}
        max_per_target = 25000  # Large dataset for XGBoost
        
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
                stock_samples = self._process_stock_xgboost(
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
    
    def _process_stock_xgboost(self, file_path, symbol):
        try:
            df = pd.read_csv(file_path)
            if len(df) < 150:
                return None
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Quality filtering
            if not self._is_quality_stock_xgb(df):
                return None
            
            samples = []
            window_size = 60  # Longer window for more features
            prediction_horizon = 7   # Shorter prediction for higher accuracy
            step_size = 3   # More frequent sampling
            
            for i in range(window_size, len(df) - prediction_horizon, step_size):
                current_window = df.iloc[i-window_size:i]
                future_window = df.iloc[i:i+prediction_horizon]
                
                if len(current_window) < window_size or len(future_window) < prediction_horizon:
                    continue
                
                features = self._calculate_xgb_features(current_window)
                if features is None or len(features) < 60:
                    continue
                
                current_price = current_window['Close'].iloc[-1]
                future_price = future_window['Close'].iloc[-1]
                future_return = (future_price - current_price) / current_price
                
                # More conservative thresholds for XGBoost
                if future_return > 0.03:  # 3% gain
                    target = 'BUY'
                elif future_return < -0.03:  # 3% loss
                    target = 'SELL'
                else:
                    target = 'HOLD'
                
                samples.append({
                    'symbol': symbol,
                    'features': features,
                    'target': target,
                    'future_return': future_return
                })
            
            return samples[:20] if samples else None
            
        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            return None
    
    def _is_quality_stock_xgb(self, df):
        try:
            prices = df['Close'].values
            volumes = df['Volume'].values
            
            # Price range
            current_price = prices[-1]
            if current_price < 3 or current_price > 1000:
                return False
            
            # Volume
            avg_volume = np.mean(volumes)
            if avg_volume < 25000:
                return False
            
            # Volatility
            volatility = np.std(prices[-60:]) / np.mean(prices[-60:]) if len(prices) >= 60 else 0
            if volatility < 0.005 or volatility > 0.15:
                return False
            
            # Data quality
            if np.any(prices <= 0) or np.any(volumes < 0):
                return False
            
            # Trend requirement
            if len(prices) >= 30:
                recent_trend = np.polyfit(range(30), prices[-30:], 1)[0]
                if abs(recent_trend) < 0.001:  # Need some trend
                    return False
            
            return True
            
        except:
            return False
    
    def _calculate_xgb_features(self, window_data):
        try:
            prices = window_data['Close'].values
            volumes = window_data['Volume'].values
            highs = window_data['High'].values
            lows = window_data['Low'].values
            opens = window_data['Open'].values
            
            if len(prices) < 50:
                return None
            
            features = []
            
            # Price returns (multiple timeframes)
            for period in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
                if len(prices) > period:
                    ret = (prices[-1] - prices[-1-period]) / prices[-1-period]
                    features.append(ret)
                else:
                    features.append(0)
            
            # Moving averages and crossovers
            ma_periods = [3, 5, 7, 10, 14, 21, 30, 50]
            mas = []
            for period in ma_periods:
                if len(prices) >= period:
                    ma = np.mean(prices[-period:])
                    mas.append(ma)
                    features.append(ma)
                    features.append((prices[-1] - ma) / ma)  # Distance from MA
                else:
                    mas.append(prices[-1])
                    features.extend([prices[-1], 0])
            
            # MA crossovers
            for i in range(len(mas)-1):
                for j in range(i+1, len(mas)):
                    if mas[j] != 0:
                        features.append((mas[i] - mas[j]) / mas[j])
                    else:
                        features.append(0)
            
            # Technical indicators
            features.extend([
                self._rsi(prices, 14),
                self._rsi(prices, 7),
                self._rsi(prices, 21),
                self._macd_signal(prices),
                self._macd_histogram(prices),
                self._bollinger_position(prices, 20),
                self._bollinger_width(prices, 20),
                self._stochastic_k(highs, lows, prices, 14),
                self._stochastic_d(highs, lows, prices, 14),
                self._williams_r(highs, lows, prices, 14),
                self._cci(highs, lows, prices, 20),
                self._atr(highs, lows, prices, 14),
            ])
            
            # Volume indicators
            avg_vol = np.mean(volumes)
            features.extend([
                volumes[-1] / avg_vol if avg_vol > 0 else 1,
                np.mean(volumes[-3:]) / avg_vol if avg_vol > 0 else 1,
                np.mean(volumes[-7:]) / avg_vol if avg_vol > 0 else 1,
                np.mean(volumes[-14:]) / avg_vol if avg_vol > 0 else 1,
                self._obv_trend(prices, volumes),
                self._volume_price_trend(prices, volumes),
            ])
            
            # Volatility measures
            for period in [5, 10, 14, 21, 30]:
                if len(prices) >= period:
                    vol = np.std(prices[-period:]) / np.mean(prices[-period:])
                    features.append(vol)
                else:
                    features.append(0)
            
            # Price patterns
            features.extend([
                (highs[-1] - lows[-1]) / prices[-1],  # Daily range
                (prices[-1] - opens[-1]) / opens[-1],  # Daily change
                (highs[-1] - prices[-1]) / prices[-1],  # Upper shadow
                (prices[-1] - lows[-1]) / prices[-1],  # Lower shadow
                (highs[-1] - opens[-1]) / opens[-1],   # High vs open
                (lows[-1] - opens[-1]) / opens[-1],    # Low vs open
            ])
            
            # Support/Resistance
            for period in [10, 20, 30]:
                if len(highs) >= period and len(lows) >= period:
                    high_level = np.max(highs[-period:])
                    low_level = np.min(lows[-period:])
                    features.extend([
                        (prices[-1] - low_level) / prices[-1],
                        (high_level - prices[-1]) / prices[-1],
                        (high_level - low_level) / prices[-1],
                    ])
                else:
                    features.extend([0, 0, 0])
            
            # Momentum and trends
            for period in [3, 5, 7, 10, 14]:
                if len(prices) >= period * 2:
                    recent = np.mean(prices[-period:])
                    older = np.mean(prices[-period*2:-period])
                    momentum = (recent - older) / older if older > 0 else 0
                    features.append(momentum)
                else:
                    features.append(0)
            
            # Trend slopes
            for period in [5, 10, 20, 30]:
                if len(prices) >= period:
                    slope = np.polyfit(range(period), prices[-period:], 1)[0]
                    features.append(slope / prices[-1])
                else:
                    features.append(0)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error calculating XGB features: {e}")
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

    def _macd_histogram(self, prices):
        try:
            if len(prices) < 35:
                return 0
            ema12 = self._ema(prices, 12)
            ema26 = self._ema(prices, 26)
            macd_line = ema12 - ema26
            signal_line = self._ema([macd_line], 9)
            return (macd_line - signal_line) / abs(signal_line) if signal_line != 0 else 0
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

    def _bollinger_width(self, prices, period=20):
        try:
            if len(prices) < period:
                return 0
            ma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            return (4 * std) / ma if ma != 0 else 0
        except:
            return 0

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

    def _stochastic_d(self, highs, lows, closes, period=14):
        try:
            if len(closes) < period + 2:
                return 50
            k_values = []
            for i in range(3):
                if len(closes) >= period + i:
                    high_max = np.max(highs[-period-i:-i if i > 0 else None])
                    low_min = np.min(lows[-period-i:-i if i > 0 else None])
                    current = closes[-1-i]
                    if high_max != low_min:
                        k = 100 * (current - low_min) / (high_max - low_min)
                        k_values.append(k)
            return np.mean(k_values) if k_values else 50
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

    def _cci(self, highs, lows, closes, period=20):
        try:
            if len(closes) < period:
                return 0
            typical_prices = (highs + lows + closes) / 3
            sma = np.mean(typical_prices[-period:])
            mean_deviation = np.mean(np.abs(typical_prices[-period:] - sma))
            if mean_deviation == 0:
                return 0
            return (typical_prices[-1] - sma) / (0.015 * mean_deviation)
        except:
            return 0

    def _atr(self, highs, lows, closes, period=14):
        try:
            if len(closes) < period + 1:
                return 0
            true_ranges = []
            for i in range(1, len(closes)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                true_ranges.append(tr)
            return np.mean(true_ranges[-period:]) / closes[-1] if closes[-1] != 0 else 0
        except:
            return 0

    def _obv_trend(self, prices, volumes):
        try:
            if len(prices) < 10:
                return 0
            obv = 0
            obv_values = [0]
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    obv += volumes[i]
                elif prices[i] < prices[i-1]:
                    obv -= volumes[i]
                obv_values.append(obv)

            recent_obv = np.mean(obv_values[-5:])
            older_obv = np.mean(obv_values[-10:-5])
            return (recent_obv - older_obv) / abs(older_obv) if older_obv != 0 else 0
        except:
            return 0

    def _volume_price_trend(self, prices, volumes):
        try:
            if len(prices) < 5:
                return 0
            vpt = 0
            for i in range(1, len(prices)):
                vpt += volumes[i] * ((prices[i] - prices[i-1]) / prices[i-1])
            return vpt / np.sum(volumes) if np.sum(volumes) != 0 else 0
        except:
            return 0

    def train_xgboost_ensemble(self, stock_data_list):
        logger.info("Training XGBoost ensemble for 80%+ accuracy...")

        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not available, falling back to sklearn")
            return self._train_sklearn_fallback(stock_data_list)

        try:
            # Prepare data
            X = np.array([data['features'] for data in stock_data_list])
            y = np.array([data['target'] for data in stock_data_list])

            logger.info(f"Training with {len(X)} samples, {X.shape[1]} features")

            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            num_classes = len(self.label_encoder.classes_)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Feature selection
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)

            logger.info(f"Selected {X_train_selected.shape[1]} best features")

            # Create ensemble of XGBoost models with different parameters
            models = []
            predictions = []

            # Model 1: Conservative
            model1 = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )

            # Model 2: Aggressive
            model2 = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=43,
                n_jobs=-1
            )

            # Model 3: Balanced
            model3 = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=44,
                n_jobs=-1
            )

            models = [model1, model2, model3]

            # Train models with cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []

            for i, model in enumerate(models):
                logger.info(f"Training XGBoost model {i+1}/3...")

                # Cross-validation
                fold_scores = []
                for train_idx, val_idx in skf.split(X_train_selected, y_train):
                    X_fold_train, X_fold_val = X_train_selected[train_idx], X_train_selected[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                    model.fit(X_fold_train, y_fold_train)
                    val_pred = model.predict(X_fold_val)
                    fold_score = accuracy_score(y_fold_val, val_pred)
                    fold_scores.append(fold_score)

                cv_score = np.mean(fold_scores)
                cv_scores.append(cv_score)
                logger.info(f"Model {i+1} CV accuracy: {cv_score:.4f}")

                # Train on full training set
                model.fit(X_train_selected, y_train)

                # Get predictions
                test_pred = model.predict(X_test_selected)
                predictions.append(test_pred)

                self.models.append(model)

            # Ensemble prediction (majority voting)
            ensemble_pred = []
            for i in range(len(y_test)):
                votes = [pred[i] for pred in predictions]
                ensemble_pred.append(max(set(votes), key=votes.count))

            ensemble_pred = np.array(ensemble_pred)
            accuracy = accuracy_score(y_test, ensemble_pred)

            # Convert back to original labels
            y_test_labels = self.label_encoder.inverse_transform(y_test)
            y_pred_labels = self.label_encoder.inverse_transform(ensemble_pred)

            logger.info(f"XGBoost Ensemble Accuracy: {accuracy:.4f} ({accuracy:.1%})")
            logger.info(f"Individual model CV scores: {cv_scores}")
            logger.info("Classification Report:")
            logger.info(f"\n{classification_report(y_test_labels, y_pred_labels)}")

            # Save models
            if accuracy >= 0.70:
                os.makedirs('models', exist_ok=True)
                for i, model in enumerate(self.models):
                    model.save_model(f'models/xgboost_model_{i+1}.json')
                joblib.dump(self.scaler, 'models/xgboost_scaler.joblib')
                joblib.dump(self.label_encoder, 'models/xgboost_label_encoder.joblib')
                joblib.dump(self.feature_selector, 'models/xgboost_feature_selector.joblib')
                logger.info("XGBoost ensemble saved!")

            return accuracy >= 0.80, accuracy

        except Exception as e:
            logger.error(f"Error during XGBoost training: {e}")
            return False, 0

    def _train_sklearn_fallback(self, stock_data_list):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.ensemble import VotingClassifier

        try:
            X = np.array([data['features'] for data in stock_data_list])
            y = np.array([data['target'] for data in stock_data_list])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Feature selection
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)

            # Create powerful ensemble
            rf = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42, n_jobs=-1)
            gb = GradientBoostingClassifier(n_estimators=300, max_depth=12, random_state=42)

            ensemble = VotingClassifier([('rf', rf), ('gb', gb)], voting='soft')
            ensemble.fit(X_train_selected, y_train)

            y_pred = ensemble.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"Sklearn Ensemble Accuracy: {accuracy:.4f} ({accuracy:.1%})")
            logger.info("Classification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")

            return accuracy >= 0.80, accuracy

        except Exception as e:
            logger.error(f"Error in sklearn fallback: {e}")
            return False, 0

    def run_xgboost_training(self):
        logger.info("Starting XGBoost training for 80%+ accuracy...")

        try:
            stock_data_list = self.load_xgboost_data()

            if not stock_data_list:
                logger.error("No valid stock data found")
                return False

            success, accuracy = self.train_xgboost_ensemble(stock_data_list)

            if success:
                logger.info(f"🎉 XGBOOST MODEL ACHIEVED {accuracy:.1%} ACCURACY! 🎉")
                return True
            else:
                logger.info(f"XGBoost model achieved {accuracy:.1%} accuracy")
                return False

        except Exception as e:
            logger.error(f"Error during XGBoost training: {e}")
            return False

def main():
    trainer = XGBoostTrainer()
    success = trainer.run_xgboost_training()

    if success:
        logger.info("XGBoost training completed successfully!")
        sys.exit(0)
    else:
        logger.error("XGBoost training failed to reach 80%")
        sys.exit(1)

if __name__ == "__main__":
    main()
