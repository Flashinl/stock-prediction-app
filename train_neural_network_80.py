#!/usr/bin/env python3
"""
Neural Network Approach for 80%+ Accuracy
Deep learning with advanced preprocessing
"""

import logging
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import random

# Try to import neural network libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralNetworkTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.valid_symbols = set()
        
        # Set random seeds
        random.seed(42)
        np.random.seed(42)
        if NEURAL_AVAILABLE:
            tf.random.set_seed(42)
        
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
    
    def load_neural_data(self):
        logger.info("Loading data for neural network training...")
        
        all_stock_data = []
        target_samples = {'BUY': 0, 'HOLD': 0, 'SELL': 0}
        max_per_target = 15000
        
        stocks_dir = 'kaggle_data/borismarjanovic_price-volume-data-for-all-us-stocks-etfs/Stocks'
        if not os.path.exists(stocks_dir):
            logger.error("Kaggle data directory not found")
            return []
        
        stock_files = [f for f in os.listdir(stocks_dir) if f.endswith('.txt')]
        random.shuffle(stock_files)
        
        for i, stock_file in enumerate(stock_files):
            if i % 500 == 0:
                logger.info(f"Processed {i} stocks, collected {len(all_stock_data)} samples")
            
            if all(count >= max_per_target for count in target_samples.values()):
                break
            
            symbol = stock_file.replace('.us.txt', '').upper()
            if not self.is_valid_symbol(symbol):
                continue
            
            try:
                stock_samples = self._process_stock_neural(
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
    
    def _process_stock_neural(self, file_path, symbol):
        try:
            df = pd.read_csv(file_path)
            if len(df) < 100:
                return None
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            samples = []
            window_size = 40
            prediction_horizon = 15
            step_size = 10
            
            for i in range(window_size, len(df) - prediction_horizon, step_size):
                current_window = df.iloc[i-window_size:i]
                future_window = df.iloc[i:i+prediction_horizon]
                
                if len(current_window) < window_size or len(future_window) < prediction_horizon:
                    continue
                
                features = self._calculate_neural_features(current_window)
                if features is None:
                    continue
                
                current_price = current_window['Close'].iloc[-1]
                future_price = future_window['Close'].iloc[-1]
                future_return = (future_price - current_price) / current_price
                
                # More aggressive thresholds for neural network
                if future_return > 0.08:
                    target = 'BUY'
                elif future_return < -0.08:
                    target = 'SELL'
                else:
                    target = 'HOLD'
                
                samples.append({
                    'symbol': symbol,
                    'features': features,
                    'target': target,
                    'future_return': future_return
                })
            
            return samples[:10] if samples else None
            
        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            return None
    
    def _calculate_neural_features(self, window_data):
        try:
            prices = window_data['Close'].values
            volumes = window_data['Volume'].values
            highs = window_data['High'].values
            lows = window_data['Low'].values
            opens = window_data['Open'].values
            
            if len(prices) < 30:
                return None
            
            features = []
            
            # Price features
            current_price = prices[-1]
            features.extend([
                current_price,
                (prices[-1] - prices[-2]) / prices[-2],
                (prices[-1] - prices[-3]) / prices[-3],
                (prices[-1] - prices[-5]) / prices[-5],
                (prices[-1] - prices[-10]) / prices[-10],
                (prices[-1] - prices[-20]) / prices[-20],
            ])
            
            # Moving averages
            for period in [3, 5, 10, 15, 20, 30]:
                if len(prices) >= period:
                    ma = np.mean(prices[-period:])
                    features.append(ma)
                    features.append((current_price - ma) / ma)
                else:
                    features.extend([current_price, 0])
            
            # Technical indicators
            features.append(self._calculate_rsi(prices))
            features.append(self._calculate_macd(prices))
            features.append(self._calculate_bollinger_position(prices))
            features.append(self._calculate_stochastic(highs, lows, prices))
            
            # Volume features
            avg_volume = np.mean(volumes)
            features.extend([
                volumes[-1] / avg_volume if avg_volume > 0 else 1,
                np.mean(volumes[-5:]) / avg_volume if avg_volume > 0 else 1,
                np.mean(volumes[-10:]) / avg_volume if avg_volume > 0 else 1,
            ])
            
            # Volatility features
            for period in [5, 10, 20]:
                if len(prices) >= period:
                    vol = np.std(prices[-period:]) / np.mean(prices[-period:])
                    features.append(vol)
                else:
                    features.append(0)
            
            # Price patterns
            features.extend([
                (highs[-1] - lows[-1]) / current_price,
                (current_price - opens[-1]) / opens[-1],
                (highs[-1] - current_price) / current_price,
                (current_price - lows[-1]) / current_price,
            ])
            
            # Momentum features
            for period in [3, 5, 10]:
                if len(prices) >= period * 2:
                    recent = np.mean(prices[-period:])
                    older = np.mean(prices[-period*2:-period])
                    momentum = (recent - older) / older if older > 0 else 0
                    features.append(momentum)
                else:
                    features.append(0)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error calculating features: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
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
    
    def _calculate_macd(self, prices):
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
    
    def _calculate_bollinger_position(self, prices, period=20):
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
    
    def _calculate_stochastic(self, highs, lows, closes, period=14):
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

    def train_neural_model(self, stock_data_list):
        logger.info("Training neural network model...")

        if not NEURAL_AVAILABLE:
            logger.error("TensorFlow not available, falling back to sklearn")
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

            # Convert to categorical
            y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
            y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

            # Build neural network
            self.model = Sequential([
                Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                BatchNormalization(),
                Dropout(0.3),

                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),

                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),

                Dense(32, activation='relu'),
                Dropout(0.2),

                Dense(num_classes, activation='softmax')
            ])

            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True
            )

            reduce_lr = ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=10,
                min_lr=0.00001
            )

            # Train model
            history = self.model.fit(
                X_train_scaled, y_train_cat,
                validation_data=(X_test_scaled, y_test_cat),
                epochs=200,
                batch_size=64,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

            # Evaluate
            y_pred_proba = self.model.predict(X_test_scaled)
            y_pred = np.argmax(y_pred_proba, axis=1)

            accuracy = accuracy_score(y_test, y_pred)

            # Convert back to original labels
            y_test_labels = self.label_encoder.inverse_transform(y_test)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)

            logger.info(f"Neural Network Accuracy: {accuracy:.4f} ({accuracy:.1%})")
            logger.info("Classification Report:")
            logger.info(f"\n{classification_report(y_test_labels, y_pred_labels)}")

            # Save model
            if accuracy >= 0.70:
                os.makedirs('models', exist_ok=True)
                self.model.save('models/neural_stock_model.h5')
                joblib.dump(self.scaler, 'models/neural_stock_scaler.joblib')
                joblib.dump(self.label_encoder, 'models/neural_label_encoder.joblib')
                logger.info("Neural network model saved!")

            return accuracy >= 0.80, accuracy

        except Exception as e:
            logger.error(f"Error during neural training: {e}")
            return False, 0

    def _train_sklearn_fallback(self, stock_data_list):
        from sklearn.ensemble import RandomForestClassifier

        try:
            X = np.array([data['features'] for data in stock_data_list])
            y = np.array([data['target'] for data in stock_data_list])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model = RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )

            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"Sklearn Fallback Accuracy: {accuracy:.4f} ({accuracy:.1%})")

            return accuracy >= 0.80, accuracy

        except Exception as e:
            logger.error(f"Error in sklearn fallback: {e}")
            return False, 0

    def run_neural_training(self):
        logger.info("Starting neural network training for 80%+ accuracy...")

        try:
            stock_data_list = self.load_neural_data()

            if not stock_data_list:
                logger.error("No valid stock data found")
                return False

            success, accuracy = self.train_neural_model(stock_data_list)

            if success:
                logger.info(f"🎉 NEURAL MODEL ACHIEVED {accuracy:.1%} ACCURACY! 🎉")
                return True
            else:
                logger.info(f"Neural model achieved {accuracy:.1%} accuracy")
                return False

        except Exception as e:
            logger.error(f"Error during neural training: {e}")
            return False

def main():
    trainer = NeuralNetworkTrainer()
    success = trainer.run_neural_training()

    if success:
        logger.info("Neural network training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Neural network training failed to reach 80%")
        sys.exit(1)

if __name__ == "__main__":
    main()
