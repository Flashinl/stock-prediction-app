#!/usr/bin/env python3
"""
Optimized Kaggle Stock Market Training
Fixes all issues with delisted stocks, rate limiting, and deprecated functions
"""

import logging
import sys
import os
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from neural_network_predictor import StockNeuralNetworkPredictor
from app import app, db, Stock
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedKaggleTrainer:
    def __init__(self):
        self.predictor = StockNeuralNetworkPredictor()
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
                
        # Check if already known to be invalid
        if symbol in self.invalid_symbols:
            return False
            
        return True
    
    def load_kaggle_stock_data(self):
        """Load and process Kaggle stock data without Yahoo Finance API calls"""
        logger.info("Loading Kaggle stock data...")
        
        all_stock_data = []
        processed_symbols = set()
        
        # Process main comprehensive dataset
        stocks_dir = 'kaggle_data/borismarjanovic_price-volume-data-for-all-us-stocks-etfs/Stocks'
        if os.path.exists(stocks_dir):
            stock_files = [f for f in os.listdir(stocks_dir) if f.endswith('.txt')]
            logger.info(f"Processing {len(stock_files)} stock files from main dataset...")
            
            for i, stock_file in enumerate(stock_files):
                if i % 1000 == 0:
                    logger.info(f"Processed {i}/{len(stock_files)} stocks...")
                
                symbol = stock_file.replace('.us.txt', '').upper()
                
                # Skip if already processed or invalid
                if symbol in processed_symbols or not self.is_valid_symbol(symbol):
                    continue
                
                try:
                    stock_data = self._process_kaggle_stock_file(
                        os.path.join(stocks_dir, stock_file), symbol
                    )
                    if stock_data:
                        all_stock_data.append(stock_data)
                        processed_symbols.add(symbol)
                        self.valid_symbols.add(symbol)
                except Exception as e:
                    logger.debug(f"Error processing {symbol}: {e}")
                    self.invalid_symbols.add(symbol)
                    continue
        
        # Process additional datasets
        self._process_additional_datasets(all_stock_data, processed_symbols)
        
        logger.info(f"Successfully processed {len(all_stock_data)} valid securities")
        logger.info(f"Valid symbols: {len(self.valid_symbols)}, Invalid symbols: {len(self.invalid_symbols)}")
        
        return all_stock_data
    
    def _process_additional_datasets(self, all_stock_data, processed_symbols):
        """Process additional Kaggle datasets"""
        
        # Process NYSE dataset
        nyse_dir = 'kaggle_data/dgawlik_nyse'
        if os.path.exists(nyse_dir):
            prices_file = os.path.join(nyse_dir, 'prices.csv')
            if os.path.exists(prices_file):
                logger.info("Processing NYSE dataset...")
                try:
                    df = pd.read_csv(prices_file)
                    symbols = df['symbol'].unique()
                    
                    for symbol in symbols:
                        if symbol in processed_symbols or not self.is_valid_symbol(symbol):
                            continue
                            
                        symbol_data = df[df['symbol'] == symbol].copy()
                        if len(symbol_data) >= 100:  # Minimum data requirement
                            stock_data = self._process_dataframe_to_features(symbol_data, symbol)
                            if stock_data:
                                all_stock_data.append(stock_data)
                                processed_symbols.add(symbol)
                                self.valid_symbols.add(symbol)
                                
                except Exception as e:
                    logger.warning(f"Error processing NYSE dataset: {e}")
        
        # Process Jackson Crow dataset
        jackson_dir = 'kaggle_data/jacksoncrow_stock-market-dataset/stocks'
        if os.path.exists(jackson_dir):
            logger.info("Processing Jackson Crow dataset...")
            stock_files = [f for f in os.listdir(jackson_dir) if f.endswith('.csv')]
            
            for stock_file in stock_files[:1000]:  # Limit to prevent overload
                symbol = stock_file.replace('.csv', '').upper()
                
                if symbol in processed_symbols or not self.is_valid_symbol(symbol):
                    continue
                    
                try:
                    df = pd.read_csv(os.path.join(jackson_dir, stock_file))
                    if len(df) >= 100:
                        stock_data = self._process_dataframe_to_features(df, symbol)
                        if stock_data:
                            all_stock_data.append(stock_data)
                            processed_symbols.add(symbol)
                            self.valid_symbols.add(symbol)
                except Exception as e:
                    logger.debug(f"Error processing {symbol}: {e}")
                    self.invalid_symbols.add(symbol)
    
    def _process_kaggle_stock_file(self, file_path, symbol):
        """Process individual Kaggle stock file"""
        try:
            # Read the CSV data
            df = pd.read_csv(file_path)
            
            # Ensure we have enough data (at least 100 days)
            if len(df) < 100:
                return None
            
            return self._process_dataframe_to_features(df, symbol)
            
        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            return None
    
    def _process_dataframe_to_features(self, df, symbol):
        """Convert dataframe to feature vector"""
        try:
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Handle different date column names
            date_cols = ['date', 'timestamp', 'time']
            date_col = None
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
            
            # Standardize price/volume column names
            price_cols = ['close', 'price', 'adj close', 'adjusted_close']
            volume_cols = ['volume', 'vol']
            
            close_col = None
            volume_col = None
            
            for col in price_cols:
                if col in df.columns:
                    close_col = col
                    break
                    
            for col in volume_cols:
                if col in df.columns:
                    volume_col = col
                    break
            
            if not close_col:
                return None
                
            # Get the most recent data
            recent_data = df.tail(252)  # Last year of trading data
            
            if len(recent_data) < 50:  # Need at least 50 days
                return None
            
            # Calculate features
            prices = recent_data[close_col].values
            volumes = recent_data[volume_col].values if volume_col else np.ones(len(prices))
            
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
            
            # Volatility (standard deviation of returns)
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # RSI calculation
            rsi = self._calculate_rsi(prices)
            
            # Market cap category (estimated from price)
            if current_price < 5:
                market_cap_category = 'penny'
            elif current_price < 50:
                market_cap_category = 'small'
            elif current_price < 200:
                market_cap_category = 'mid'
            else:
                market_cap_category = 'large'
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_1d': price_change_1d,
                'price_change_5d': price_change_5d,
                'price_change_20d': price_change_20d,
                'ma_5': ma_5,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'volume': volumes[-1],
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'rsi': rsi,
                'market_cap_category': market_cap_category,
                'data_points': len(df),
                'sector': 'Unknown',
                'industry': 'Unknown'
            }
            
        except Exception as e:
            logger.debug(f"Error processing features for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50  # Neutral RSI
                
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
            return 50  # Neutral RSI on error

    def prepare_training_data(self, stock_data_list):
        """Prepare training data for neural network"""
        logger.info("Preparing training data...")

        training_data = []
        for stock_data in stock_data_list:
            try:
                # Create feature vector
                features = [
                    stock_data['current_price'],
                    stock_data['price_change_1d'],
                    stock_data['price_change_5d'],
                    stock_data['price_change_20d'],
                    stock_data['ma_5'],
                    stock_data['ma_20'],
                    stock_data['ma_50'],
                    stock_data['volume'],
                    stock_data['avg_volume'],
                    stock_data['volume_ratio'],
                    stock_data['volatility'],
                    stock_data['rsi']
                ]

                # Generate target based on technical analysis
                target = self._generate_target_from_features(stock_data)

                training_data.append({
                    'symbol': stock_data['symbol'],
                    'features': features,
                    'target': target,
                    'market_cap': stock_data['market_cap_category']
                })

            except Exception as e:
                logger.debug(f"Error preparing training data for {stock_data['symbol']}: {e}")
                continue

        logger.info(f"Prepared {len(training_data)} training samples")
        return training_data

    def _generate_target_from_features(self, stock_data):
        """Generate target classification based on technical analysis"""
        try:
            # Technical analysis scoring
            score = 0

            # Price momentum
            if stock_data['price_change_5d'] > 0.02:  # 2% gain in 5 days
                score += 15
            elif stock_data['price_change_5d'] < -0.02:  # 2% loss in 5 days
                score -= 15

            # Moving average trends
            if stock_data['current_price'] > stock_data['ma_5']:
                score += 10
            if stock_data['ma_5'] > stock_data['ma_20']:
                score += 10
            if stock_data['ma_20'] > stock_data['ma_50']:
                score += 10

            # RSI analysis
            if 30 <= stock_data['rsi'] <= 70:  # Healthy range
                score += 5
            elif stock_data['rsi'] < 30:  # Oversold
                score += 10
            elif stock_data['rsi'] > 70:  # Overbought
                score -= 10

            # Volume analysis
            if stock_data['volume_ratio'] > 1.5:  # High volume
                score += 5

            # Volatility check
            if stock_data['volatility'] > 0.05:  # High volatility
                score -= 5

            # Market cap adjustments
            if stock_data['market_cap_category'] == 'penny':
                score -= 10  # Penny stocks are riskier
            elif stock_data['market_cap_category'] == 'large':
                score += 5   # Large caps are more stable

            # Convert score to classification
            if score >= 25:
                return 'BUY'
            elif score <= -15:
                return 'SELL'
            else:
                return 'HOLD'

        except Exception as e:
            logger.debug(f"Error generating target: {e}")
            return 'HOLD'

    def train_model(self, training_data):
        """Train the neural network model"""
        logger.info("Training neural network model...")

        try:
            # Prepare features and targets
            X = []
            y = []

            for data in training_data:
                X.append(data['features'])
                y.append(data['target'])

            X = np.array(X)
            y = np.array(y)

            logger.info(f"Training with {len(X)} samples")
            logger.info(f"Target distribution: BUY: {np.sum(y == 'BUY')}, HOLD: {np.sum(y == 'HOLD')}, SELL: {np.sum(y == 'SELL')}")

            # Train the model using the correct method
            # First we need to create a symbols list for the train method
            symbols_list = [data['symbol'] for data in training_data]
            success = self.predictor.train(symbols_list, epochs=50)

            if success:
                logger.info("Model training completed successfully")
                return True
            else:
                logger.error("Model training failed")
                return False

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False

    def save_training_report(self, stock_data_list, training_data):
        """Save comprehensive training report"""
        try:
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_stocks_processed': len(stock_data_list),
                'valid_training_samples': len(training_data),
                'valid_symbols_count': len(self.valid_symbols),
                'invalid_symbols_count': len(self.invalid_symbols),
                'target_distribution': {
                    'BUY': sum(1 for d in training_data if d['target'] == 'BUY'),
                    'HOLD': sum(1 for d in training_data if d['target'] == 'HOLD'),
                    'SELL': sum(1 for d in training_data if d['target'] == 'SELL')
                },
                'market_cap_distribution': {
                    'penny': sum(1 for d in training_data if d['market_cap'] == 'penny'),
                    'small': sum(1 for d in training_data if d['market_cap'] == 'small'),
                    'mid': sum(1 for d in training_data if d['market_cap'] == 'mid'),
                    'large': sum(1 for d in training_data if d['market_cap'] == 'large')
                },
                'sample_valid_symbols': list(self.valid_symbols)[:100],
                'sample_invalid_symbols': list(self.invalid_symbols)[:50]
            }

            # Save report
            os.makedirs('reports', exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_file = f'reports/kaggle_optimized_training_{timestamp}.json'

            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Training report saved to {report_file}")

        except Exception as e:
            logger.error(f"Error saving training report: {e}")

    def run_training(self):
        """Run the complete training process"""
        logger.info("Starting optimized Kaggle training...")

        try:
            # Load Kaggle data
            stock_data_list = self.load_kaggle_stock_data()

            if not stock_data_list:
                logger.error("No valid stock data found")
                return False

            # Prepare training data
            training_data = self.prepare_training_data(stock_data_list)

            if not training_data:
                logger.error("No valid training data prepared")
                return False

            # Train model
            success = self.train_model(training_data)

            # Save report
            self.save_training_report(stock_data_list, training_data)

            if success:
                logger.info("Optimized Kaggle training completed successfully!")
                return True
            else:
                logger.error("Training failed")
                return False

        except Exception as e:
            logger.error(f"Error during training process: {e}")
            return False

def main():
    """Main training function"""
    trainer = OptimizedKaggleTrainer()
    success = trainer.run_training()

    if success:
        logger.info("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
