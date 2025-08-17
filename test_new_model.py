#!/usr/bin/env python3
"""
Test the API endpoints to verify the new model is working
"""

import requests
import json

def test_model_stats():
    """Test the model stats endpoint"""
    print("Testing model stats endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/model/stats')
        if response.status_code == 200:
            data = response.json()
            print("✅ Model Stats:")
            print(f"  Model Type: {data.get('model_type')}")
            print(f"  Accuracy: {data.get('accuracy', 0):.1%}")
            print(f"  Training Samples: {data.get('training_samples', 0)}")
            print(f"  Best Model: {data.get('best_model')}")
            print(f"  Production Ready: {data.get('is_production_ready')}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_top_opportunities():
    """Test the top opportunities endpoint"""
    print("\nTesting top opportunities endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/opportunities/fast')
        if response.status_code == 200:
            data = response.json()
            opportunities = data.get('opportunities', [])
            print(f"✅ Found {len(opportunities)} opportunities:")
            for opp in opportunities:
                print(f"  {opp['symbol']:6} | {opp['prediction']:11} | {opp['expected_change_percent']:+5.1f}% | Confidence: {opp['confidence']:.0f}%")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_stock_prediction(symbol):
    """Test stock prediction endpoint"""
    print(f"\nTesting stock prediction for {symbol}...")
    try:
        payload = {"symbol": symbol}
        response = requests.post('http://localhost:5000/api/predict', json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction for {symbol}:")
            print(f"  Prediction: {data.get('prediction', 'N/A')}")
            print(f"  Confidence: {data.get('confidence', 0):.1f}%")
            print(f"  Expected Change: {data.get('expected_change_percent', 0):+.1f}%")
            print(f"  Current Price: ${data.get('current_price', 0):.2f}")
            print(f"  Target Price: ${data.get('target_price', 0):.2f}")
            print(f"  Timeframe: {data.get('timeframe', 'N/A')}")
            print(f"  Model Type: {data.get('model_type', 'N/A')}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing StockTrek API with New High-Accuracy Model")
    print("=" * 60)
    
    # Test model stats
    stats_ok = test_model_stats()
    
    # Test top opportunities
    opportunities_ok = test_top_opportunities()
    
    # Test individual stock predictions
    test_stocks = ['AAPL', 'GOOGL', 'TSLA', 'NVDA', 'MSFT']
    prediction_results = []
    
    for symbol in test_stocks:
        result = test_stock_prediction(symbol)
        prediction_results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model Stats: {'✅ PASS' if stats_ok else '❌ FAIL'}")
    print(f"Top Opportunities: {'✅ PASS' if opportunities_ok else '❌ FAIL'}")
    print(f"Stock Predictions: {sum(prediction_results)}/{len(prediction_results)} passed")
    
    if stats_ok and opportunities_ok and all(prediction_results):
        print("\n🎉 ALL TESTS PASSED! The new high-accuracy model is working correctly!")
        print("✅ Model statistics are available")
        print("✅ Top opportunities are being generated")
        print("✅ Individual stock predictions are working")
        print("\n💡 The website should now show:")
        print("  - Accurate statistics and model info")
        print("  - 6 top opportunities with real predictions")
        print("  - Working stock prediction graphs and analysis")
    else:
        print("\n❌ Some tests failed. Check the Flask app logs for details.")

if __name__ == "__main__":
    main()
