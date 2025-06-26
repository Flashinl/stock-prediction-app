#!/usr/bin/env python3
"""
Analyze what the model is getting wrong with HOLD and SELL predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class FailureAnalyzer:
    def __init__(self):
        self.test_symbols = ['AAPL', 'TSLA', 'PLTR', 'SPY', 'NVDA']
        self.failures = {
            'hold_failures': [],
            'sell_failures': [],
            'patterns': {}
        }
    
    def get_historical_data(self, symbol, days_back=90):
        """Get historical data for analysis"""
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if not data.empty else None
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        if len(data) < 50:
            return None
        
        current_price = data['Close'].iloc[-1]
        
        # Moving averages
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        macd = (exp1 - exp2).iloc[-1]
        
        # Volume analysis
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
        
        # Price momentum (10-day rate of change)
        if len(data) >= 10:
            price_momentum = (current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10] * 100
        else:
            price_momentum = 0
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma = data['Close'].rolling(window=bb_period).mean()
        std = data['Close'].rolling(window=bb_period).std()
        bollinger_upper = (sma + (std * bb_std)).iloc[-1]
        bollinger_lower = (sma - (std * bb_std)).iloc[-1]
        
        return {
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'macd': macd,
            'volume': volume,
            'avg_volume': avg_volume,
            'price_momentum': price_momentum,
            'bollinger_upper': bollinger_upper,
            'bollinger_lower': bollinger_lower
        }
    
    def complex_model_prediction(self, indicators):
        """Current complex model prediction logic"""
        current_price = indicators['current_price']
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        rsi = indicators['rsi']
        macd = indicators['macd']
        volume = indicators['volume']
        avg_volume = indicators['avg_volume']
        price_momentum = indicators['price_momentum']
        bollinger_upper = indicators['bollinger_upper']
        bollinger_lower = indicators['bollinger_lower']
        
        score = 50  # Start neutral
        
        # EXACT replica of current scoring
        if rsi < 30:
            score += 15
        elif rsi > 70:
            score -= 15
        
        if current_price > sma_20 > sma_50:
            score += 15
        elif current_price < sma_20 < sma_50:
            score -= 15
        
        bb_position = (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower)
        if bb_position < 0.2:
            score += 10
        elif bb_position > 0.8:
            score -= 10
        
        if macd > 0:
            score += min(10, abs(macd) * 2)
        else:
            score -= min(10, abs(macd) * 2)
        
        if price_momentum > 5:
            score += 10
        elif price_momentum < -5:
            score -= 10
        
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 2:
            score += 15
        elif volume_ratio > 1.5:
            score += 8
        elif volume_ratio < 0.5:
            score -= 10
        
        # UPDATED prediction logic with new thresholds
        if score >= 60:  # LOWERED from 70
            return 'BUY', score
        elif score <= 20:  # LOWERED from 30
            return 'SELL', score
        elif 40 <= score <= 59:  # TIGHTER HOLD range
            return 'HOLD', score
        elif score > 59:  # Safety net
            return 'BUY', score
        else:  # 21-39 range
            return 'HOLD', score
    
    def get_actual_result(self, data, prediction_index):
        """Get actual result 20 days later"""
        if prediction_index + 20 >= len(data):
            return None
        
        current_price = data['Close'].iloc[prediction_index]
        future_price = data['Close'].iloc[prediction_index + 20]
        
        actual_change = (future_price - current_price) / current_price * 100
        
        if actual_change > 2:
            return 'BUY', actual_change
        elif actual_change < -2:
            return 'SELL', actual_change
        else:
            return 'HOLD', actual_change
    
    def analyze_failures(self):
        """Analyze what patterns lead to prediction failures"""
        print("🔍 Analyzing HOLD and SELL Prediction Failures")
        print("=" * 60)
        
        total_hold_wrong = 0
        total_sell_wrong = 0
        total_hold_predictions = 0
        total_sell_predictions = 0
        
        for symbol in self.test_symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            data = self.get_historical_data(symbol, days_back=120)
            if data is None or len(data) < 70:
                continue
            
            # Test multiple prediction points
            for i in range(50, len(data) - 20, 5):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                predicted, score = self.complex_model_prediction(indicators)
                actual_result = self.get_actual_result(data, i)
                
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                
                # Analyze HOLD failures
                if predicted == 'HOLD':
                    total_hold_predictions += 1
                    if actual_direction != 'HOLD':
                        total_hold_wrong += 1
                        self.failures['hold_failures'].append({
                            'symbol': symbol,
                            'score': score,
                            'predicted': predicted,
                            'actual': actual_direction,
                            'actual_change': actual_change,
                            'indicators': indicators.copy()
                        })
                        print(f"  HOLD FAIL: Score={score:.1f}, Actual={actual_direction} ({actual_change:.1f}%)")
                
                # Analyze SELL failures
                elif predicted == 'SELL':
                    total_sell_predictions += 1
                    if actual_direction != 'SELL':
                        total_sell_wrong += 1
                        self.failures['sell_failures'].append({
                            'symbol': symbol,
                            'score': score,
                            'predicted': predicted,
                            'actual': actual_direction,
                            'actual_change': actual_change,
                            'indicators': indicators.copy()
                        })
                        print(f"  SELL FAIL: Score={score:.1f}, Actual={actual_direction} ({actual_change:.1f}%)")
        
        self.analyze_failure_patterns()
        
        print(f"\n" + "=" * 60)
        print(f"📊 FAILURE SUMMARY:")
        print(f"HOLD Predictions: {total_hold_predictions}, Wrong: {total_hold_wrong}")
        if total_hold_predictions > 0:
            hold_accuracy = (total_hold_predictions - total_hold_wrong) / total_hold_predictions * 100
            print(f"HOLD Accuracy: {hold_accuracy:.1f}%")
        
        print(f"SELL Predictions: {total_sell_predictions}, Wrong: {total_sell_wrong}")
        if total_sell_predictions > 0:
            sell_accuracy = (total_sell_predictions - total_sell_wrong) / total_sell_predictions * 100
            print(f"SELL Accuracy: {sell_accuracy:.1f}%")
    
    def analyze_failure_patterns(self):
        """Find common patterns in failures"""
        print(f"\n🔍 FAILURE PATTERN ANALYSIS:")
        
        # Analyze HOLD failures
        if self.failures['hold_failures']:
            print(f"\n❌ HOLD FAILURES ({len(self.failures['hold_failures'])} cases):")
            
            # What did HOLD predictions actually become?
            actual_outcomes = {}
            score_ranges = {'30-40': 0, '40-50': 0, '50-60': 0, '60-70': 0}
            
            for failure in self.failures['hold_failures']:
                actual = failure['actual']
                score = failure['score']
                
                actual_outcomes[actual] = actual_outcomes.get(actual, 0) + 1
                
                if 30 <= score < 40:
                    score_ranges['30-40'] += 1
                elif 40 <= score < 50:
                    score_ranges['40-50'] += 1
                elif 50 <= score < 60:
                    score_ranges['50-60'] += 1
                elif 60 <= score < 70:
                    score_ranges['60-70'] += 1
            
            print(f"  HOLD predictions actually became:")
            for outcome, count in actual_outcomes.items():
                pct = count / len(self.failures['hold_failures']) * 100
                print(f"    {outcome}: {count} cases ({pct:.1f}%)")
            
            print(f"  Score distribution of failed HOLDs:")
            for range_name, count in score_ranges.items():
                if count > 0:
                    pct = count / len(self.failures['hold_failures']) * 100
                    print(f"    Score {range_name}: {count} cases ({pct:.1f}%)")
        
        # Analyze SELL failures
        if self.failures['sell_failures']:
            print(f"\n❌ SELL FAILURES ({len(self.failures['sell_failures'])} cases):")
            
            actual_outcomes = {}
            for failure in self.failures['sell_failures']:
                actual = failure['actual']
                actual_outcomes[actual] = actual_outcomes.get(actual, 0) + 1
            
            print(f"  SELL predictions actually became:")
            for outcome, count in actual_outcomes.items():
                pct = count / len(self.failures['sell_failures']) * 100
                print(f"    {outcome}: {count} cases ({pct:.1f}%)")

if __name__ == "__main__":
    analyzer = FailureAnalyzer()
    analyzer.analyze_failures()
