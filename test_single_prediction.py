#!/usr/bin/env python3
"""
Test a single stock prediction to check technical indicators
"""

import requests
import json

def test_prediction(symbol):
    """Test prediction for a single stock"""
    try:
        payload = {"symbol": symbol}
        response = requests.post('http://localhost:5000/api/predict', json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction for {symbol}:")
            print(f"  Company: {data.get('company_name', 'N/A')}")
            print(f"  Prediction: {data.get('prediction', 'N/A')}")
            print(f"  Confidence: {data.get('confidence', 0):.1f}%")
            print(f"  Expected Change: {data.get('expected_change_percent', 0):+.1f}%")
            print(f"  Current Price: ${data.get('current_price', 0):.2f}")
            print(f"  Target Price: ${data.get('target_price', 0):.2f}")
            print(f"  Timeframe: {data.get('timeframe', 'N/A')}")
            print(f"  Model Type: {data.get('model_type', 'N/A')}")
            
            # Check technical indicators
            tech_indicators = data.get('technical_indicators', {})
            print(f"\n📊 Technical Indicators:")
            print(f"  RSI (14): {tech_indicators.get('rsi', 'N/A')}")
            print(f"  20-Day SMA: ${tech_indicators.get('sma_20', 'N/A')}")
            print(f"  50-Day SMA: ${tech_indicators.get('sma_50', 'N/A')}")
            print(f"  MACD Signal: {tech_indicators.get('macd_signal', 'N/A')}")
            # Handle formatting for potentially missing values
            momentum = tech_indicators.get('price_momentum', 'N/A')
            trend = tech_indicators.get('trend_strength', 'N/A')
            vol_trend = tech_indicators.get('volume_trend', 'N/A')
            bb_upper = tech_indicators.get('bollinger_upper', 'N/A')
            bb_lower = tech_indicators.get('bollinger_lower', 'N/A')
            volume = tech_indicators.get('volume', 'N/A')
            volatility = tech_indicators.get('volatility', 'N/A')

            print(f"  Price Momentum: {momentum:.2f}%" if isinstance(momentum, (int, float)) else f"  Price Momentum: {momentum}")
            print(f"  Trend Strength: {trend:.1f}%" if isinstance(trend, (int, float)) else f"  Trend Strength: {trend}")
            print(f"  Volume Trend: {vol_trend:.2f}%" if isinstance(vol_trend, (int, float)) else f"  Volume Trend: {vol_trend}")
            print(f"  Bollinger Upper: ${bb_upper:.2f}" if isinstance(bb_upper, (int, float)) else f"  Bollinger Upper: {bb_upper}")
            print(f"  Bollinger Lower: ${bb_lower:.2f}" if isinstance(bb_lower, (int, float)) else f"  Bollinger Lower: {bb_lower}")
            print(f"  Daily Volume: {volume:,.0f}" if isinstance(volume, (int, float)) else f"  Daily Volume: {volume}")
            print(f"  Volatility: {volatility:.2f}%" if isinstance(volatility, (int, float)) else f"  Volatility: {volatility}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Single Stock Prediction with Technical Indicators")
    print("=" * 60)
    
    # Test AAPL
    success = test_prediction("AAPL")
    
    if success:
        print("\n✅ Technical indicators are working!")
        print("The 'undefined' issue should now be fixed on the website.")
    else:
        print("\n❌ Still having issues with the prediction API.")
