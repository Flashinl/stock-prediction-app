#!/usr/bin/env python3
"""
Test the timeframe dropdown functionality
"""

from neural_network_predictor_production import NeuralNetworkStockPredictor

def test_timeframe_parsing():
    """Test that frontend timeframe values are parsed correctly"""
    print("Testing Timeframe Dropdown Parsing:")
    print("=" * 50)
    
    predictor = NeuralNetworkStockPredictor()
    
    # Test frontend dropdown values
    frontend_values = [
        'auto',
        '1month',
        '3months', 
        '6months',
        '1year',
        '2years'
    ]
    
    expected_days = [90, 30, 90, 180, 365, 730]
    
    print("Frontend Value -> Expected Days -> Actual Days")
    print("-" * 50)
    
    for i, timeframe in enumerate(frontend_values):
        actual_days = predictor._parse_timeframe_to_days(timeframe)
        expected = expected_days[i]
        status = "✅" if actual_days == expected else "❌"
        print(f"{timeframe:12} -> {expected:3} days -> {actual_days:3} days {status}")

def test_full_prediction_with_timeframes():
    """Test full prediction with different timeframes"""
    print("\n\nTesting Full Predictions with Different Timeframes:")
    print("=" * 60)
    
    predictor = NeuralNetworkStockPredictor()
    symbol = 'AAPL'  # Use a reliable stock
    
    timeframes = ['1month', '3months', '6months', '1year']
    
    for timeframe in timeframes:
        print(f"\nTesting {symbol} with timeframe: {timeframe}")
        print("-" * 40)
        
        try:
            result = predictor.predict_stock_movement(symbol, timeframe)
            
            if 'error' not in result:
                print(f"  ✅ Prediction: {result.get('prediction', 'N/A')}")
                print(f"  ✅ Expected Change: {result.get('expected_change_percent', 'N/A')}%")
                print(f"  ✅ Timeframe Used: {result.get('timeframe', 'N/A')}")
                print(f"  ✅ Current Price: ${result.get('current_price', 'N/A')}")
                print(f"  ✅ Target Price: ${result.get('target_price', 'N/A')}")
                
                # Check if prediction data exists and has reasonable length
                pred_data = result.get('prediction_data', [])
                print(f"  ✅ Graph Points: {len(pred_data)} points")
                
                if len(pred_data) > 0:
                    first_price = pred_data[0]['price']
                    last_price = pred_data[-1]['price']
                    price_change = ((last_price - first_price) / first_price) * 100
                    print(f"  ✅ Graph Change: {price_change:.2f}%")
                
            else:
                print(f"  ❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")

if __name__ == "__main__":
    test_timeframe_parsing()
    test_full_prediction_with_timeframes()
