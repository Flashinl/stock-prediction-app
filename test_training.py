#!/usr/bin/env python3
"""
Test training script to verify everything works
"""

import logging
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_training():
    """Test training with a small sample"""
    logger.info("Starting test training...")
    
    try:
        # Check if Kaggle data exists
        stocks_dir = 'kaggle_data/borismarjanovic_price-volume-data-for-all-us-stocks-etfs/Stocks'
        if not os.path.exists(stocks_dir):
            logger.error("Kaggle data directory not found")
            return False
        
        stock_files = [f for f in os.listdir(stocks_dir) if f.endswith('.txt')]
        logger.info(f"Found {len(stock_files)} stock files")
        
        # Process just 10 stocks for testing
        all_data = []
        for i, stock_file in enumerate(stock_files[:10]):
            logger.info(f"Processing {stock_file}...")
            
            try:
                df = pd.read_csv(os.path.join(stocks_dir, stock_file))
                
                if len(df) < 50:
                    continue
                
                # Simple feature extraction
                prices = df['Close'].values[-30:]  # Last 30 days
                
                if len(prices) < 30:
                    continue
                
                # Calculate simple features
                current_price = prices[-1]
                price_change = (prices[-1] - prices[-5]) / prices[-5] if len(prices) > 5 else 0
                ma_10 = np.mean(prices[-10:])
                volatility = np.std(prices) / np.mean(prices)
                
                # Simple target: if price went up more than 2% in last 5 days
                target = 1 if price_change > 0.02 else 0
                
                all_data.append([current_price, price_change, ma_10, volatility, target])
                
            except Exception as e:
                logger.debug(f"Error processing {stock_file}: {e}")
                continue
        
        if len(all_data) < 5:
            logger.error("Not enough data collected")
            return False
        
        logger.info(f"Collected {len(all_data)} samples")
        
        # Convert to arrays
        data_array = np.array(all_data)
        X = data_array[:, :-1]  # Features
        y = data_array[:, -1]   # Target
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        logger.info(f"Target distribution: {np.bincount(y.astype(int))}")
        
        # Train simple model
        if len(np.unique(y)) < 2:
            logger.warning("Not enough target diversity")
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Save test model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/test_model.joblib')
        
        logger.info("Test training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during test training: {e}")
        return False

if __name__ == "__main__":
    success = test_training()
    if success:
        print("Test training successful!")
    else:
        print("Test training failed!")
