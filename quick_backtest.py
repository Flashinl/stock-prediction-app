#!/usr/bin/env python3
"""
Quick backtest to compare Simple vs Complex models
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def quick_test():
    # Test just a few stocks for speed
    symbols = ['AAPL', 'TSLA', 'PLTR', 'SPY']
    
    simple_correct = 0
    complex_correct = 0
    total_tests = 0
    
    print("🧪 Quick Backtest: Simple vs Complex Models")
    print("=" * 50)
    
    for symbol in symbols:
        print(f"Testing {symbol}...")
        
        try:
            # Get 60 days of data
            stock = yf.Ticker(symbol)
            data = stock.history(period="60d")
            
            if len(data) < 40:
                continue
            
            # Test last 10 data points
            for i in range(30, len(data) - 10):
                current_data = data.iloc[:i+1]
                
                # Calculate indicators
                current_price = current_data['Close'].iloc[-1]
                volume = current_data['Volume'].iloc[-1]
                avg_volume = current_data['Volume'].rolling(20).mean().iloc[-1]
                
                # Price momentum (5-day)
                if len(current_data) >= 5:
                    price_momentum = (current_price - current_data['Close'].iloc[-5]) / current_data['Close'].iloc[-5] * 100
                else:
                    price_momentum = 0
                
                # RSI
                delta = current_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
                
                # SMA
                sma_20 = current_data['Close'].rolling(20).mean().iloc[-1]
                
                # Simple Model Prediction
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1
                volume_signal = 1 if volume_ratio > 1.5 else (-1 if volume_ratio < 0.7 else 0)
                momentum_signal = 1 if price_momentum > 3 else (-1 if price_momentum < -3 else 0)
                simple_combined = (volume_signal * 0.6) + (momentum_signal * 0.4)
                
                if simple_combined > 0.3:
                    simple_pred = 'BUY'
                elif simple_combined < -0.3:
                    simple_pred = 'SELL'
                else:
                    simple_pred = 'HOLD'
                
                # Complex Model Prediction
                score = 50
                
                # RSI component
                if rsi < 30:
                    score += 15
                elif rsi > 70:
                    score -= 15
                
                # MA component
                if current_price > sma_20:
                    score += 10
                else:
                    score -= 10
                
                # Volume component
                if volume_ratio > 1.5:
                    score += 10
                elif volume_ratio < 0.7:
                    score -= 10
                
                # Momentum component
                if price_momentum > 3:
                    score += 10
                elif price_momentum < -3:
                    score -= 10
                
                if score > 65:
                    complex_pred = 'BUY'
                elif score < 35:
                    complex_pred = 'SELL'
                else:
                    complex_pred = 'HOLD'
                
                # Actual result (10 days later)
                if i + 10 < len(data):
                    future_price = data['Close'].iloc[i + 10]
                    actual_change = (future_price - current_price) / current_price * 100
                    
                    if actual_change > 2:
                        actual = 'BUY'
                    elif actual_change < -2:
                        actual = 'SELL'
                    else:
                        actual = 'HOLD'
                    
                    # Check accuracy
                    total_tests += 1
                    if simple_pred == actual:
                        simple_correct += 1
                    if complex_pred == actual:
                        complex_correct += 1
                    
                    print(f"  {symbol} Test {total_tests}: Simple={simple_pred}, Complex={complex_pred}, Actual={actual} ({actual_change:.1f}%)")
        
        except Exception as e:
            print(f"Error with {symbol}: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("🎯 RESULTS:")
    
    if total_tests > 0:
        simple_accuracy = (simple_correct / total_tests) * 100
        complex_accuracy = (complex_correct / total_tests) * 100
        
        print(f"Simple Model:  {simple_accuracy:.1f}% ({simple_correct}/{total_tests})")
        print(f"Complex Model: {complex_accuracy:.1f}% ({complex_correct}/{total_tests})")
        
        difference = simple_accuracy - complex_accuracy
        if difference > 5:
            print(f"\n✅ WINNER: Simple Model (+{difference:.1f}%)")
        elif difference < -5:
            print(f"\n✅ WINNER: Complex Model (+{abs(difference):.1f}%)")
        else:
            print(f"\n🤝 TIE: Similar performance (±{abs(difference):.1f}%)")
    else:
        print("❌ No valid tests completed")

if __name__ == "__main__":
    quick_test()
