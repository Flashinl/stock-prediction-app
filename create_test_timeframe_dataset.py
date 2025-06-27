#!/usr/bin/env python3
"""
Create a small test timeframe-aware dataset for quick validation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_dataset():
    """Create a small test dataset with timeframe features"""
    
    # Use just a few stocks for quick testing
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'QMCO']
    
    timeframes = {
        '1_month': 30,
        '3_months': 90,
        '6_months': 180
    }
    
    # Use fewer sample dates for speed
    start_date = datetime(2022, 6, 1)
    end_date = datetime(2023, 6, 1)
    
    dataset = []
    
    # Sample every month instead of every 2 weeks
    current_date = start_date
    sample_dates = []
    while current_date <= end_date:
        sample_dates.append(current_date)
        current_date += timedelta(days=30)
    
    logger.info(f"Creating test dataset with {len(stocks)} stocks and {len(sample_dates)} dates")
    
    for stock in stocks:
        logger.info(f"Processing {stock}...")
        
        for sample_date in sample_dates:
            # Get basic features
            try:
                ticker = yf.Ticker(stock)
                hist = ticker.history(start=sample_date - timedelta(days=100), end=sample_date)
                
                if len(hist) < 20:
                    continue
                
                current_price = float(hist['Close'].iloc[-1])
                volume = hist['Volume'].iloc[-1]
                
                # Simple technical indicators
                sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                rsi = 50  # Simplified for test
                
                # For each timeframe, get actual future performance
                for timeframe_name, days in timeframes.items():
                    future_date = sample_date + timedelta(days=days)
                    if future_date > end_date:
                        continue
                    
                    # Get future price
                    future_hist = ticker.history(start=future_date, end=future_date + timedelta(days=5))
                    if len(future_hist) == 0:
                        continue
                    
                    future_price = float(future_hist['Close'].iloc[0])
                    actual_change = ((future_price - current_price) / current_price) * 100
                    
                    # Create label
                    if actual_change > 5:
                        label = 'BUY'
                        label_numeric = 2
                    elif actual_change < -5:
                        label = 'SELL'
                        label_numeric = 0
                    else:
                        label = 'HOLD'
                        label_numeric = 1
                    
                    # Create row with timeframe as feature
                    row = {
                        'symbol': stock,
                        'date': sample_date.strftime('%Y-%m-%d'),
                        'current_price': current_price,
                        'volume': volume,
                        'sma_20': sma_20,
                        'rsi': rsi,
                        'timeframe': timeframe_name,
                        'timeframe_days': days,
                        'timeframe_1_month': 1 if timeframe_name == '1_month' else 0,
                        'timeframe_3_months': 1 if timeframe_name == '3_months' else 0,
                        'timeframe_6_months': 1 if timeframe_name == '6_months' else 0,
                        'future_price': future_price,
                        'actual_change_percent': actual_change,
                        'label': label,
                        'label_numeric': label_numeric
                    }
                    
                    dataset.append(row)
                    
            except Exception as e:
                logger.error(f"Error processing {stock} at {sample_date}: {e}")
                continue
    
    df = pd.DataFrame(dataset)
    logger.info(f"Created test dataset with {len(df)} samples")
    
    if len(df) > 0:
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        logger.info(f"Timeframe distribution:\n{df['timeframe'].value_counts()}")
        
        # Save dataset
        df.to_csv('test_timeframe_dataset.csv', index=False)
        logger.info("Test dataset saved to test_timeframe_dataset.csv")
        
        return df
    else:
        logger.error("No data collected!")
        return None

if __name__ == "__main__":
    create_test_dataset()
