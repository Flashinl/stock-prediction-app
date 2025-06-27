#!/usr/bin/env python3
"""
Test script to verify technical indicators are properly calculated and returned
"""
import requests
import json

def test_technical_indicators():
    """Test that all technical indicators are returned without N/A values"""
    url = "http://127.0.0.1:5000/api/predict"
    
    # Test with a common stock
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\n=== Testing {symbol} ===")
        
        data = {
            "symbol": symbol,
            "timeframe": "auto"
        }
        
        try:
            response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if technical_indicators exist
                if 'technical_indicators' in result:
                    indicators = result['technical_indicators']
                    
                    # List of required indicators
                    required_indicators = [
                        'current_price', 'rsi', 'sma_20', 'sma_50', 'macd',
                        'price_momentum', 'trend_strength', 'volume_trend',
                        'bollinger_upper', 'bollinger_lower', 'volume', 'volatility'
                    ]
                    
                    print("Technical Indicators:")
                    missing_indicators = []
                    
                    for indicator in required_indicators:
                        if indicator in indicators and indicators[indicator] is not None:
                            value = indicators[indicator]
                            print(f"  {indicator}: {value}")
                        else:
                            missing_indicators.append(indicator)
                            print(f"  {indicator}: MISSING or NULL")
                    
                    if missing_indicators:
                        print(f"❌ Missing indicators: {missing_indicators}")
                    else:
                        print("✅ All technical indicators present!")
                        
                else:
                    print("❌ No technical_indicators in response")
                    
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_technical_indicators()
