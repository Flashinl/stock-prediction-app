#!/usr/bin/env python3
"""
Investigate current neural network model issues
"""

from neural_network_predictor_production import NeuralNetworkStockPredictor
import json

def investigate_model_predictions():
    """Test the current model to understand its limitations"""
    print("Investigating Neural Network Model Issues:")
    print("=" * 50)
    
    predictor = NeuralNetworkStockPredictor()
    
    # Test the same stock with different timeframes
    symbol = 'QMCO'  # The stock from the user's example
    timeframes = ['1-2 months', '3-6 months', '6-12 months']
    
    print(f"\nTesting {symbol} with different timeframes:")
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
                
                # Check prediction data points
                pred_data = result.get('prediction_data', [])
                if len(pred_data) > 5:
                    first_price = pred_data[0]['price']
                    last_price = pred_data[-1]['price']
                    mid_price = pred_data[len(pred_data)//2]['price']
                    print(f"  Graph: Start=${first_price}, Mid=${mid_price}, End=${last_price}")
                    
                    # Check if it's a straight line (minimal variation)
                    prices = [p['price'] for p in pred_data]
                    price_range = max(prices) - min(prices)
                    current_price = result.get('current_price', 0)
                    variation_percent = (price_range / current_price) * 100 if current_price > 0 else 0
                    
                    if variation_percent < 2.0:
                        print(f"  ⚠️  ISSUE: Graph is too linear (only {variation_percent:.2f}% variation)")
                    else:
                        print(f"  ✓ Graph has {variation_percent:.2f}% variation")
            else:
                print(f"\n{timeframe}: Error - {result['error']}")
                
        except Exception as e:
            print(f"\n{timeframe}: Exception - {e}")
    
    # Analyze if timeframes produce different results
    print("\n" + "=" * 50)
    print("ANALYSIS:")
    print("=" * 50)
    
    if len(results) >= 2:
        timeframe_keys = list(results.keys())
        
        # Check if expected changes are different
        changes = [results[tf]['expected_change_percent'] for tf in timeframe_keys]
        predictions = [results[tf]['prediction'] for tf in timeframe_keys]
        
        if len(set(changes)) == 1:
            print("❌ ISSUE: All timeframes produce identical expected changes")
        else:
            print("✅ Timeframes produce different expected changes")
            
        if len(set(predictions)) == 1:
            print("❌ ISSUE: All timeframes produce identical predictions")
        else:
            print("✅ Timeframes produce different predictions")
            
        # Check model architecture limitations
        print("\nModel Architecture Analysis:")
        print("- Current model: Classification only (BUY/SELL/HOLD)")
        print("- Missing: Timeframe as input feature")
        print("- Missing: Price target prediction")
        print("- Missing: Timeframe-specific training data")
        
        print("\nGraph Generation Analysis:")
        print("- Current: Simple interpolation between current and target price")
        print("- Issue: No real market dynamics modeling")
        print("- Issue: Deterministic but unrealistic progression")
        
    return results

def analyze_training_data_structure():
    """Analyze what features the current model was trained on"""
    print("\n" + "=" * 50)
    print("TRAINING DATA ANALYSIS:")
    print("=" * 50)
    
    try:
        predictor = NeuralNetworkStockPredictor()
        
        # Check if model has feature names
        if hasattr(predictor, 'feature_names'):
            print(f"Model trained on {len(predictor.feature_names)} features:")
            for i, feature in enumerate(predictor.feature_names[:10]):  # Show first 10
                print(f"  {i+1}. {feature}")
            if len(predictor.feature_names) > 10:
                print(f"  ... and {len(predictor.feature_names) - 10} more")
                
            # Check if timeframe-related features exist
            timeframe_features = [f for f in predictor.feature_names if 'timeframe' in f.lower() or 'time' in f.lower()]
            if timeframe_features:
                print(f"\nTimeframe-related features found: {timeframe_features}")
            else:
                print("\n❌ NO timeframe-related features found in training data")
        else:
            print("❌ Cannot access model feature names")
            
    except Exception as e:
        print(f"Error analyzing training data: {e}")

if __name__ == "__main__":
    investigate_model_predictions()
    analyze_training_data_structure()
