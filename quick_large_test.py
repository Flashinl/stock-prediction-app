#!/usr/bin/env python3
"""
Quick test on a larger dataset by generating predictions for more stocks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_network_predictor_production import neural_predictor
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_larger_dataset():
    """Test neural network on a larger set of stocks"""
    
    # Larger test set of diverse stocks
    test_stocks = [
        # Large cap tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM',
        # Large cap non-tech
        'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'V', 'MA', 'WMT', 'DIS', 'KO',
        # Mid cap growth
        'AVGO', 'PYPL', 'SHOP', 'SQ', 'ROKU', 'ZOOM', 'DOCU', 'OKTA', 'TWLO', 'SNOW',
        # Small cap / speculative
        'IONQ', 'RGTI', 'SOUN', 'QTUM', 'MVIS', 'LAZR', 'PLTR', 'SOFI', 'COIN', 'RBLX',
        # ETFs and others
        'SPY', 'QQQ', 'IWM', 'VTI', 'ARKK', 'TQQQ', 'SQQQ', 'GLD', 'TLT', 'XLK'
    ]
    
    logger.info(f"🧪 Testing neural network on {len(test_stocks)} stocks")
    logger.info("=" * 60)
    
    results = []
    successful_predictions = 0
    failed_predictions = 0
    
    # Track prediction distribution
    prediction_counts = {'BUY': 0, 'HOLD': 0, 'SELL': 0, 'STRONG BUY': 0, 'SPECULATIVE BUY': 0}
    confidence_levels = []
    expected_changes = []
    
    for i, symbol in enumerate(test_stocks, 1):
        try:
            logger.info(f"[{i}/{len(test_stocks)}] Testing {symbol}...")
            
            # Get prediction
            result = neural_predictor.predict_stock_movement(symbol)
            
            if 'error' in result:
                logger.warning(f"  ❌ {symbol}: {result['error']}")
                failed_predictions += 1
                continue
            
            # Extract key metrics
            prediction = result.get('prediction', 'UNKNOWN')
            confidence = result.get('confidence', 0)
            expected_change = result.get('expected_change_percent', 0)
            current_price = result.get('current_price', 0)
            
            # Track statistics
            if prediction in prediction_counts:
                prediction_counts[prediction] += 1
            confidence_levels.append(confidence)
            expected_changes.append(expected_change)
            
            # Store result
            results.append({
                'symbol': symbol,
                'prediction': prediction,
                'confidence': confidence,
                'expected_change': expected_change,
                'current_price': current_price,
                'company_name': result.get('company_name', symbol),
                'sector': result.get('sector', 'Unknown')
            })
            
            logger.info(f"  ✅ {symbol}: {prediction} ({confidence}% conf, {expected_change:+.1f}% change)")
            successful_predictions += 1
            
        except Exception as e:
            logger.error(f"  ❌ {symbol}: Exception - {e}")
            failed_predictions += 1
    
    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("📊 LARGE DATASET TEST RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"Total stocks tested: {len(test_stocks)}")
    logger.info(f"Successful predictions: {successful_predictions}")
    logger.info(f"Failed predictions: {failed_predictions}")
    logger.info(f"Success rate: {successful_predictions/len(test_stocks)*100:.1f}%")
    
    if successful_predictions > 0:
        logger.info(f"\n📈 PREDICTION DISTRIBUTION:")
        for pred_type, count in prediction_counts.items():
            if count > 0:
                percentage = count / successful_predictions * 100
                logger.info(f"  {pred_type}: {count} ({percentage:.1f}%)")
        
        logger.info(f"\n📊 CONFIDENCE STATISTICS:")
        logger.info(f"  Average confidence: {np.mean(confidence_levels):.1f}%")
        logger.info(f"  Median confidence: {np.median(confidence_levels):.1f}%")
        logger.info(f"  Min confidence: {np.min(confidence_levels):.1f}%")
        logger.info(f"  Max confidence: {np.max(confidence_levels):.1f}%")
        
        logger.info(f"\n📈 EXPECTED CHANGE STATISTICS:")
        logger.info(f"  Average expected change: {np.mean(expected_changes):+.2f}%")
        logger.info(f"  Median expected change: {np.median(expected_changes):+.2f}%")
        logger.info(f"  Min expected change: {np.min(expected_changes):+.2f}%")
        logger.info(f"  Max expected change: {np.max(expected_changes):+.2f}%")
        
        # Show top opportunities
        buy_stocks = [r for r in results if 'BUY' in r['prediction']]
        if buy_stocks:
            buy_stocks.sort(key=lambda x: x['expected_change'], reverse=True)
            logger.info(f"\n🚀 TOP 5 BUY OPPORTUNITIES:")
            for stock in buy_stocks[:5]:
                logger.info(f"  {stock['symbol']}: {stock['prediction']} "
                           f"({stock['confidence']}% conf, {stock['expected_change']:+.1f}% change)")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv('large_test_results.csv', index=False)
        logger.info(f"\n💾 Results saved to large_test_results.csv")
    
    return successful_predictions, failed_predictions, results

def analyze_features():
    """Quick analysis of what features the neural network uses"""
    logger.info("\n" + "=" * 60)
    logger.info("🔍 NEURAL NETWORK FEATURE ANALYSIS")
    logger.info("=" * 60)
    
    # Test feature extraction on one stock
    try:
        features = neural_predictor.extract_comprehensive_features('AAPL')
        if features:
            logger.info(f"Total features extracted: {len(features)}")
            
            # Categorize features
            technical_features = [k for k in features.keys() if any(term in k.lower() for term in 
                                ['rsi', 'sma', 'ema', 'macd', 'bb_', 'momentum', 'volume', 'volatility', 'price'])]
            
            fundamental_features = [k for k in features.keys() if any(term in k.lower() for term in 
                                  ['pe', 'pb', 'ps', 'market_cap', 'enterprise', 'margin', 'ratio', 'debt', 'revenue'])]
            
            other_features = [k for k in features.keys() if k not in technical_features and k not in fundamental_features]
            
            logger.info(f"Technical indicators: {len(technical_features)}")
            logger.info(f"Fundamental metrics: {len(fundamental_features)}")
            logger.info(f"Other features: {len(other_features)}")
            
            # Check for sentiment data
            sentiment_features = [k for k in features.keys() if any(term in k.lower() for term in 
                                ['sentiment', 'news', 'social', 'buzz', 'analyst'])]
            
            if sentiment_features:
                logger.info(f"✅ Sentiment features found: {sentiment_features}")
            else:
                logger.info("❌ No sentiment data detected in neural network features")
                logger.info("   Neural network uses only technical + fundamental analysis")
        
    except Exception as e:
        logger.error(f"Error analyzing features: {e}")

if __name__ == "__main__":
    logger.info("🚀 QUICK LARGE DATASET TEST")
    logger.info("Testing neural network performance on diverse stock portfolio")
    
    # Analyze what features are used
    analyze_features()
    
    # Run large test
    success, failed, results = test_larger_dataset()
    
    if success > 0:
        logger.info(f"\n🎉 Test completed successfully!")
        logger.info(f"Neural network processed {success} stocks with diverse predictions")
    else:
        logger.error(f"\n❌ Test failed - no successful predictions")
