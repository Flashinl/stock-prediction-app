#!/usr/bin/env python3
"""
Debug single stock processing
"""

import logging
import yfinance as yf
import time
from app import app, db, Stock

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_yfinance():
    """Test yfinance directly"""
    logger.info("Testing yfinance directly...")
    
    try:
        # Test with a simple stock
        ticker = yf.Ticker("AAPL")
        
        # Add delay
        time.sleep(1)
        
        hist = ticker.history(period="1y")
        logger.info(f"AAPL historical data: {len(hist)} days")
        
        if not hist.empty:
            logger.info(f"Sample data: {hist.tail(3)}")
            return True
        else:
            logger.error("No historical data returned")
            return False
            
    except Exception as e:
        logger.error(f"Error with yfinance: {e}")
        return False

def test_basic_features():
    """Test basic feature extraction"""
    logger.info("Testing basic feature extraction...")
    
    try:
        # Simple feature calculation
        ticker = yf.Ticker("AAPL")
        time.sleep(1)
        
        hist = ticker.history(period="6mo")  # Shorter period
        
        if hist.empty:
            logger.error("No data from yfinance")
            return False
        
        close = hist['Close'].values
        logger.info(f"Close prices: {len(close)} values")
        
        # Simple features
        current_price = close[-1]
        price_change_1d = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
        ma_5 = close[-5:].mean() if len(close) >= 5 else current_price
        
        features = {
            'current_price': current_price,
            'price_change_1d': price_change_1d,
            'ma_5': ma_5
        }
        
        logger.info(f"Basic features: {features}")
        return True
        
    except Exception as e:
        logger.error(f"Error in basic features: {e}")
        return False

def test_target_generation():
    """Test target generation"""
    logger.info("Testing target generation...")
    
    try:
        ticker = yf.Ticker("AAPL")
        time.sleep(1)
        
        # Get extended history
        hist = ticker.history(period="1y")
        
        if len(hist) < 50:
            logger.error("Insufficient data for target generation")
            return False
        
        # Use data from 15 days ago to calculate what actually happened
        current_idx = len(hist) - 15 - 1
        if current_idx < 0:
            logger.error("Not enough data for future calculation")
            return False
        
        current_price = hist['Close'].iloc[current_idx]
        future_price = hist['Close'].iloc[current_idx + 15]
        
        return_pct = (future_price - current_price) / current_price
        
        logger.info(f"Return calculation: {current_price:.2f} -> {future_price:.2f} = {return_pct:.4f}")
        
        # Generate target
        if return_pct >= 0.08:
            target = 'STRONG_BUY'
        elif return_pct >= 0.04:
            target = 'BUY'
        elif return_pct <= -0.08:
            target = 'STRONG_SELL'
        elif return_pct <= -0.04:
            target = 'SELL'
        else:
            target = 'HOLD'
        
        logger.info(f"Target: {target}")
        return True
        
    except Exception as e:
        logger.error(f"Error in target generation: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== Debugging Single Stock Processing ===")
    
    if test_yfinance():
        logger.info("✅ yfinance working")
        
        if test_basic_features():
            logger.info("✅ Basic features working")
            
            if test_target_generation():
                logger.info("✅ Target generation working")
                logger.info("🎉 All tests passed!")
            else:
                logger.error("❌ Target generation failed")
        else:
            logger.error("❌ Basic features failed")
    else:
        logger.error("❌ yfinance failed")
