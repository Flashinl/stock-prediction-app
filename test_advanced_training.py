#!/usr/bin/env python3
"""
Test script for advanced training to debug issues
"""

import logging
import yfinance as yf
from app import app, db, Stock
from train_advanced_80plus_accuracy import AdvancedStockPredictor

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_single_stock():
    """Test processing a single stock"""
    predictor = AdvancedStockPredictor()
    
    with app.app_context():
        # Get a single stock
        stock = Stock.query.filter(
            Stock.is_active == True,
            Stock.current_price.isnot(None),
            Stock.current_price > 1.0,
            Stock.volume.isnot(None),
            Stock.volume > 10000
        ).first()
        
        if not stock:
            logger.error("No suitable stock found")
            return
        
        logger.info(f"Testing with stock: {stock.symbol}")
        
        # Test basic features
        basic_features = predictor.extract_advanced_features(stock.symbol)
        logger.info(f"Basic features: {len(basic_features) if basic_features else 0}")
        
        if basic_features:
            logger.info(f"Sample basic features: {list(basic_features.keys())[:5]}")
        
        # Test historical data
        ticker = yf.Ticker(stock.symbol)
        hist = ticker.history(period="2y")
        logger.info(f"Historical data: {len(hist)} days")
        
        if not hist.empty and len(hist) >= 100:
            close = hist['Close'].values
            high = hist['High'].values
            low = hist['Low'].values
            volume = hist['Volume'].values
            open_price = hist['Open'].values
            
            # Test technical indicators
            tech_features = predictor.extract_technical_indicators(close, high, low, volume)
            logger.info(f"Technical features: {len(tech_features)}")
            
            # Test volume features
            volume_features = predictor.extract_volume_features(volume, close)
            logger.info(f"Volume features: {len(volume_features)}")
            
            # Test pattern features
            pattern_features = predictor.extract_pattern_features(close, high, low, open_price)
            logger.info(f"Pattern features: {len(pattern_features)}")
            
            # Test target generation
            all_features = {**basic_features, **tech_features, **volume_features, **pattern_features}
            target = predictor.generate_target_labels(stock.symbol, all_features)
            logger.info(f"Target label: {target}")
            
            if target:
                logger.info("✅ Single stock processing successful!")
                return True
            else:
                logger.error("❌ Target generation failed")
        else:
            logger.error("❌ Insufficient historical data")
    
    return False

def test_multiple_stocks():
    """Test processing multiple stocks"""
    predictor = AdvancedStockPredictor()
    
    with app.app_context():
        stocks = Stock.query.filter(
            Stock.is_active == True,
            Stock.current_price.isnot(None),
            Stock.current_price > 1.0,
            Stock.volume.isnot(None),
            Stock.volume > 10000
        ).limit(10).all()
        
        logger.info(f"Testing with {len(stocks)} stocks")
        
        successful = 0
        for stock in stocks:
            try:
                basic_features = predictor.extract_advanced_features(stock.symbol)
                if basic_features:
                    ticker = yf.Ticker(stock.symbol)
                    hist = ticker.history(period="1y")
                    
                    if not hist.empty and len(hist) >= 50:
                        target = predictor.generate_target_labels(stock.symbol, basic_features)
                        if target:
                            successful += 1
                            logger.info(f"✅ {stock.symbol}: {target}")
                        else:
                            logger.warning(f"⚠️ {stock.symbol}: No target")
                    else:
                        logger.warning(f"⚠️ {stock.symbol}: Insufficient data")
                else:
                    logger.warning(f"⚠️ {stock.symbol}: No features")
            except Exception as e:
                logger.error(f"❌ {stock.symbol}: {e}")
        
        logger.info(f"Successfully processed {successful}/{len(stocks)} stocks")
        return successful > 0

if __name__ == "__main__":
    logger.info("Testing single stock processing...")
    if test_single_stock():
        logger.info("\nTesting multiple stocks...")
        test_multiple_stocks()
    else:
        logger.error("Single stock test failed, skipping multiple stock test")
