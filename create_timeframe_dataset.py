#!/usr/bin/env python3
"""
Create timeframe-aware training dataset for neural network
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeframeAwareDatasetCreator:
    def __init__(self):
        self.timeframes = {
            '1_month': 30,
            '2_months': 60, 
            '3_months': 90,
            '6_months': 180,
            '12_months': 365
        }
        
    def get_stock_list(self):
        """Get a comprehensive list of stocks to train on"""
        # Major stocks across different sectors
        stocks = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 'ORCL',
            'ADBE', 'INTC', 'AMD', 'PYPL', 'UBER', 'ZOOM', 'SNOW', 'PLTR', 'RBLX', 'COIN',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD',
            
            # Consumer
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'DIS', 'CMCSA',
            
            # Industrial
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'EMR',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'OXY', 'HAL',
            
            # Small/Mid caps for variety
            'LUMN', 'QMCO', 'SIRI', 'F', 'GM', 'AAL', 'DAL', 'CCL', 'NCLH', 'MGM'
        ]
        return stocks
    
    def extract_features(self, symbol, date):
        """Extract features for a stock at a specific date"""
        try:
            # Get historical data up to the date
            end_date = date
            start_date = date - timedelta(days=365)  # 1 year of history
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if len(hist) < 50:  # Need sufficient history
                return None
                
            # Calculate technical indicators (same as current model)
            close = hist['Close']
            volume = hist['Volume']
            
            current_price = float(close.iloc[-1])
            
            # Moving averages
            sma_20 = close.rolling(window=20).mean().iloc[-1]
            sma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else sma_20
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            ema_12 = close.ewm(span=12).mean().iloc[-1]
            ema_26 = close.ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            
            # Volume indicators
            avg_volume = volume.rolling(window=20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1
            
            # Price momentum
            price_momentum = ((current_price - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) > 21 else 0
            
            # Volatility
            volatility = close.pct_change().rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100
            
            # Get company info
            info = ticker.info
            market_cap = info.get('marketCap', 0)
            sector = info.get('sector', 'Unknown')
            
            features = {
                'symbol': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi if not pd.isna(rsi) else 50,
                'macd': macd if not pd.isna(macd) else 0,
                'volume_ratio': volume_ratio,
                'price_momentum': price_momentum,
                'volatility': volatility if not pd.isna(volatility) else 0.02,
                'market_cap': market_cap,
                'sector': sector
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol} at {date}: {e}")
            return None
    
    def get_future_price(self, symbol, start_date, days_forward):
        """Get the actual price after a certain number of days"""
        try:
            end_date = start_date + timedelta(days=days_forward + 10)  # Buffer for weekends
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if len(hist) < days_forward // 7:  # Need reasonable amount of data
                return None
                
            # Get price closest to target date
            target_date = start_date + timedelta(days=days_forward)
            
            # Find closest trading day
            hist_dates = hist.index.date
            target_date_obj = target_date.date()
            
            closest_date = min(hist_dates, key=lambda x: abs((x - target_date_obj).days))
            closest_price = hist.loc[hist.index.date == closest_date]['Close'].iloc[0]
            
            return float(closest_price)
            
        except Exception as e:
            logger.error(f"Error getting future price for {symbol}: {e}")
            return None
    
    def create_timeframe_dataset(self, stocks, start_date='2020-01-01', end_date='2023-12-31'):
        """Create dataset with timeframe-aware labels"""
        logger.info("Creating timeframe-aware dataset...")
        
        dataset = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Sample dates every 2 weeks to get good coverage
        current_date = start_dt
        sample_dates = []
        while current_date <= end_dt:
            sample_dates.append(current_date)
            current_date += timedelta(days=14)
        
        logger.info(f"Processing {len(stocks)} stocks across {len(sample_dates)} dates...")
        
        for stock in tqdm(stocks, desc="Processing stocks"):
            for sample_date in tqdm(sample_dates, desc=f"Processing {stock}", leave=False):
                # Extract features at this date
                features = self.extract_features(stock, sample_date)
                if not features:
                    continue
                
                # For each timeframe, get the actual future performance
                for timeframe_name, days in self.timeframes.items():
                    # Skip if we don't have enough future data
                    if sample_date + timedelta(days=days) > end_dt:
                        continue
                    
                    future_price = self.get_future_price(stock, sample_date, days)
                    if future_price is None:
                        continue
                    
                    current_price = features['current_price']
                    actual_change = ((future_price - current_price) / current_price) * 100
                    
                    # Create timeframe-specific label
                    if actual_change > 10:
                        label = 'STRONG_BUY'
                        label_numeric = 4
                    elif actual_change > 3:
                        label = 'BUY'
                        label_numeric = 3
                    elif actual_change > -3:
                        label = 'HOLD'
                        label_numeric = 2
                    elif actual_change > -10:
                        label = 'SELL'
                        label_numeric = 1
                    else:
                        label = 'STRONG_SELL'
                        label_numeric = 0
                    
                    # Create row with timeframe as feature
                    row = features.copy()
                    row.update({
                        'timeframe': timeframe_name,
                        'timeframe_days': days,
                        'future_price': future_price,
                        'actual_change_percent': actual_change,
                        'label': label,
                        'label_numeric': label_numeric
                    })
                    
                    dataset.append(row)
        
        df = pd.DataFrame(dataset)
        logger.info(f"Created dataset with {len(df)} samples")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def save_dataset(self, df, filename='timeframe_aware_dataset.csv'):
        """Save the dataset"""
        df.to_csv(filename, index=False)
        logger.info(f"Dataset saved to {filename}")

def main():
    """Create the timeframe-aware dataset"""
    creator = TimeframeAwareDatasetCreator()
    
    # Get stock list
    stocks = creator.get_stock_list()
    logger.info(f"Creating dataset for {len(stocks)} stocks")
    
    # Create dataset (using smaller date range for faster processing)
    df = creator.create_timeframe_dataset(
        stocks, 
        start_date='2022-01-01', 
        end_date='2023-06-30'  # Leave recent data for testing
    )
    
    # Save dataset
    creator.save_dataset(df)
    
    print("\nDataset creation completed!")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len([col for col in df.columns if col not in ['label', 'label_numeric', 'actual_change_percent', 'future_price']])}")
    print(f"Timeframes: {df['timeframe'].unique()}")

if __name__ == "__main__":
    main()
