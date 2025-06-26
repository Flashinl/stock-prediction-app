#!/usr/bin/env python3
"""
Test script to validate the improved stock prediction API
"""
import requests
import json

def test_prediction(symbol):
    """Test prediction for a specific symbol"""
    url = "http://127.0.0.1:5000/api/predict"
    data = {
        "symbol": symbol,
        "timeframe": "auto"
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"\n=== {symbol} Prediction ===")
            print(f"Prediction: {result.get('prediction', 'N/A')}")
            print(f"Expected Change: {result.get('expected_change_percent', 'N/A')}%")
            print(f"Confidence: {result.get('confidence', 'N/A')}%")
            print(f"Timeframe: {result.get('timeframe', 'N/A')}")
            print(f"Current Price: ${result.get('current_price', 'N/A')}")
            print(f"Company: {result.get('company_name', 'N/A')}")
            print(f"Sector: {result.get('sector', 'N/A')}")
            print(f"Reasoning: {result.get('reasoning', 'N/A')}")
            return result
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Error testing {symbol}: {e}")
        return None

if __name__ == "__main__":
    # Test a few key stocks to see improved reasoning
    test_stocks = [
        "NVDA",  # AI leader
        "PLTR",  # AI data analytics
        "TSLA",  # EV leader
        "AAPL"   # Large cap tech
    ]
    
    print("Testing Enhanced Stock Prediction API")
    print("=" * 50)
    
    results = []
    for stock in test_stocks:
        result = test_prediction(stock)
        if result:
            results.append(result)
    
    # Summary of results
    print("\n" + "=" * 50)
    print("SUMMARY OF GROWTH OPPORTUNITIES")
    print("=" * 50)
    
    buy_predictions = [r for r in results if 'BUY' in r.get('prediction', '')]
    
    if buy_predictions:
        # Sort by expected change percentage
        buy_predictions.sort(key=lambda x: x.get('expected_change_percent', 0), reverse=True)
        
        print(f"\nFound {len(buy_predictions)} BUY recommendations:")
        for i, pred in enumerate(buy_predictions[:6], 1):
            print(f"{i}. {pred.get('symbol')} - {pred.get('prediction')} "
                  f"({pred.get('expected_change_percent', 0):.1f}% upside, "
                  f"{pred.get('confidence', 0)}% confidence)")
    else:
        print("\nNo BUY recommendations found in test stocks")
    
    print("\nTest completed!")
