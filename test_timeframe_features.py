#!/usr/bin/env python3
"""
Test the updated neural network with timeframe features
"""

from neural_network_predictor_production import NeuralNetworkStockPredictor
import json

def test_timeframe_features():
    """Test the new timeframe-aware features"""
    print("Testing Timeframe-Aware Neural Network:")
    print("=" * 50)
    
    predictor = NeuralNetworkStockPredictor()
    
    # Test feature extraction with different timeframes
    symbol = 'QMCO'
    timeframes = ['1-2 months', '3-6 months', '6-12 months']
    
    print(f"\nTesting feature extraction for {symbol}:")
    print("-" * 40)
    
    for timeframe in timeframes:
        print(f"\nTimeframe: {timeframe}")
        
        # Parse timeframe to days
        timeframe_days = predictor._parse_timeframe_to_days(timeframe)
        print(f"  Parsed to {timeframe_days} days")
        
        # Extract features
        features = predictor.extract_comprehensive_features(symbol, timeframe_days)
        
        if features:
            # Show timeframe-specific features
            timeframe_features = {k: v for k, v in features.items() if 'timeframe' in k}
            print(f"  Timeframe features extracted: {len(timeframe_features)}")
            
            for feature_name, value in timeframe_features.items():
                print(f"    {feature_name}: {value:.4f}")
        else:
            print(f"  ❌ Failed to extract features")
    
    # Test predictions with different timeframes
    print(f"\n\nTesting predictions for {symbol}:")
    print("-" * 40)
    
    results = {}
    for timeframe in timeframes:
        try:
            result = predictor.predict_stock_movement(symbol, timeframe)
            
            if 'error' not in result:
                results[timeframe] = result
                print(f"\nTimeframe: {timeframe}")
                print(f"  Prediction: {result.get('prediction', 'N/A')}")
                print(f"  Expected Change: {result.get('expected_change_percent', 'N/A')}%")
                print(f"  Target Price: ${result.get('target_price', 'N/A')}")
                print(f"  Current Price: ${result.get('current_price', 'N/A')}")
                print(f"  Confidence: {result.get('confidence', 'N/A')}%")
                
                # Check prediction data realism
                pred_data = result.get('prediction_data', [])
                if len(pred_data) > 5:
                    first_price = pred_data[0]['price']
                    last_price = pred_data[-1]['price']
                    mid_price = pred_data[len(pred_data)//2]['price']
                    print(f"  Graph: Start=${first_price}, Mid=${mid_price}, End=${last_price}")
                    
                    # Check if it's more realistic now
                    prices = [p['price'] for p in pred_data]
                    price_range = max(prices) - min(prices)
                    current_price = result.get('current_price', 0)
                    variation_percent = (price_range / current_price) * 100 if current_price > 0 else 0
                    
                    if variation_percent < 2.0:
                        print(f"  ⚠️  Still too linear (only {variation_percent:.2f}% variation)")
                    else:
                        print(f"  ✅ Good variation ({variation_percent:.2f}%)")
            else:
                print(f"\n{timeframe}: Error - {result['error']}")
                
        except Exception as e:
            print(f"\n{timeframe}: Exception - {e}")
    
    # Analyze improvements
    print("\n" + "=" * 50)
    print("ANALYSIS OF IMPROVEMENTS:")
    print("=" * 50)
    
    if len(results) >= 2:
        timeframe_keys = list(results.keys())
        
        # Check if predictions are now different
        predictions = [results[tf]['prediction'] for tf in timeframe_keys]
        changes = [results[tf]['expected_change_percent'] for tf in timeframe_keys]
        
        if len(set(predictions)) > 1:
            print("✅ Different timeframes now produce different predictions!")
        else:
            print("❌ Predictions still identical across timeframes")
            
        if len(set(changes)) > 1:
            print("✅ Expected changes vary by timeframe")
        else:
            print("❌ Expected changes still identical")
            
        # Check graph realism
        all_realistic = True
        for tf, result in results.items():
            pred_data = result.get('prediction_data', [])
            if pred_data:
                prices = [p['price'] for p in pred_data]
                price_range = max(prices) - min(prices)
                current_price = result.get('current_price', 0)
                variation_percent = (price_range / current_price) * 100 if current_price > 0 else 0
                
                if variation_percent < 3.0:
                    all_realistic = False
                    break
        
        if all_realistic:
            print("✅ All prediction graphs show realistic market movement")
        else:
            print("❌ Some prediction graphs still too linear")
    
    return results

if __name__ == "__main__":
    test_timeframe_features()
