#!/usr/bin/env python3
"""
Test script to verify prediction data structure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_network_predictor_production import neural_predictor
import json

def test_prediction_data():
    """Test that prediction returns all required fields"""
    print("🧪 Testing prediction data structure...")
    
    test_symbol = 'GOOGL'
    result = neural_predictor.predict_stock_movement(test_symbol)
    
    if 'error' in result:
        print(f"❌ Prediction failed: {result['error']}")
        return False
    
    # Required fields for frontend
    required_fields = [
        'symbol', 'company_name', 'exchange', 'industry', 'sector',
        'market_cap', 'is_penny_stock', 'stock_category', 'prediction',
        'confidence', 'expected_change_percent', 'target_price', 'current_price',
        'timeframe', 'model_type', 'technical_indicators', 'historical_data',
        'prediction_data', 'volume_data', 'analysis_date'
    ]
    
    required_technical_indicators = [
        'rsi', 'sma_20', 'sma_50', 'macd', 'bollinger_upper', 'bollinger_lower',
        'volume', 'volatility', 'current_price'
    ]
    
    print(f"\n📊 Testing prediction for {test_symbol}:")
    print(f"Company: {result.get('company_name', 'N/A')}")
    print(f"Prediction: {result.get('prediction', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}%")
    print(f"Expected Change: {result.get('expected_change_percent', 'N/A')}%")
    
    # Check required fields
    missing_fields = []
    for field in required_fields:
        if field not in result:
            missing_fields.append(field)
        else:
            print(f"✅ {field}: {type(result[field]).__name__}")
    
    # Check technical indicators
    if 'technical_indicators' in result:
        missing_indicators = []
        for indicator in required_technical_indicators:
            if indicator not in result['technical_indicators']:
                missing_indicators.append(indicator)
            else:
                print(f"✅ technical_indicators.{indicator}: {result['technical_indicators'][indicator]}")
        
        if missing_indicators:
            print(f"❌ Missing technical indicators: {missing_indicators}")
            return False
    else:
        print("❌ Missing technical_indicators object")
        return False
    
    # Check chart data
    if result.get('historical_data'):
        print(f"✅ historical_data: {len(result['historical_data'])} data points")
    else:
        print("❌ Missing historical_data")
    
    if result.get('prediction_data'):
        print(f"✅ prediction_data: {len(result['prediction_data'])} data points")
    else:
        print("❌ Missing prediction_data")
    
    if result.get('volume_data'):
        print(f"✅ volume_data: {len(result['volume_data'])} data points")
    else:
        print("❌ Missing volume_data")
    
    if missing_fields:
        print(f"\n❌ Missing required fields: {missing_fields}")
        return False
    
    print(f"\n🎉 All required fields present!")
    
    # Save sample data for inspection
    with open('sample_prediction_data.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"📄 Sample data saved to sample_prediction_data.json")
    
    return True

if __name__ == "__main__":
    success = test_prediction_data()
    if success:
        print("\n✅ Prediction data structure test PASSED!")
    else:
        print("\n❌ Prediction data structure test FAILED!")
    sys.exit(0 if success else 1)
