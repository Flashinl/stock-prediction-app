#!/usr/bin/env python3
"""
Test script to check the actual AI reasoning output
"""
import requests
import json

def test_reasoning(symbol):
    """Test the reasoning for a specific symbol"""
    url = "http://127.0.0.1:5000/api/predict"
    data = {
        "symbol": symbol,
        "timeframe": "auto"
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"\n=== {symbol} REASONING TEST ===")
            print(f"Prediction: {result.get('prediction', 'N/A')}")
            print(f"Reasoning: {result.get('reasoning', 'N/A')}")
            print(f"Enhanced Reasoning: {result.get('enhanced_reasoning', 'N/A')}")
            
            # Check if reasoning mentions technical indicators
            reasoning_text = result.get('reasoning', '') + ' ' + str(result.get('enhanced_reasoning', ''))
            technical_terms = ['RSI', 'moving average', 'technical', 'MACD', 'Bollinger', 'momentum']
            
            found_technical = []
            for term in technical_terms:
                if term.lower() in reasoning_text.lower():
                    found_technical.append(term)
            
            if found_technical:
                print(f"⚠️  STILL MENTIONS TECHNICAL TERMS: {found_technical}")
            else:
                print("✅ NO TECHNICAL TERMS FOUND")
                
            return result
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Error testing {symbol}: {e}")
        return None

if __name__ == "__main__":
    # Test a few stocks to see actual reasoning
    test_stocks = ["NVDA", "PLTR", "AAPL"]
    
    for stock in test_stocks:
        test_reasoning(stock)
