#!/usr/bin/env python3
"""
Analyze prediction patterns to understand what makes accurate predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PatternAnalyzer:
    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ', 'NVDA', 'TSLA', 'META']
        self.successful_patterns = {'BUY': [], 'SELL': [], 'HOLD': []}
        self.failed_patterns = {'BUY': [], 'SELL': [], 'HOLD': []}
    
    def get_historical_data(self, symbol, days_back=120):
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 80 else None
        except Exception as e:
            return None
    
    def calculate_indicators(self, data):
        if len(data) < 50:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            
            # Price momentum
            price_momentum = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100 if len(data) >= 5 else 0
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
            
            # Moving averages
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
            
            # MACD
            ema_12 = data['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['Close'].ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            
            # Bollinger Bands
            bb_std = data['Close'].rolling(20).std().iloc[-1]
            bollinger_upper = sma_20 + (bb_std * 2)
            bollinger_lower = sma_20 - (bb_std * 2)
            
            # Volatility
            volatility = data['Close'].pct_change().rolling(20).std().iloc[-1] * 100
            
            return {
                'current_price': current_price,
                'volume': volume,
                'avg_volume': avg_volume,
                'price_momentum': price_momentum,
                'rsi': rsi,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'macd': macd,
                'bollinger_upper': bollinger_upper,
                'bollinger_lower': bollinger_lower,
                'volatility': volatility,
                'volume_ratio': volume / avg_volume if avg_volume > 0 else 1,
                'price_vs_sma20': (current_price - sma_20) / sma_20 * 100,
                'price_vs_sma50': (current_price - sma_50) / sma_50 * 100,
                'bb_position': (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower) if bollinger_upper != bollinger_lower else 0.5
            }
        except:
            return None
    
    def get_actual_outcome(self, data, index, timeframe=15):
        if index + timeframe >= len(data):
            return None
        
        current_price = data['Close'].iloc[index]
        future_price = data['Close'].iloc[index + timeframe]
        change = (future_price - current_price) / current_price * 100
        
        if change > 3:
            return 'BUY', change
        elif change < -3:
            return 'SELL', change
        else:
            return 'HOLD', change
    
    def analyze_successful_patterns(self):
        print("🔍 Analyzing Successful Prediction Patterns")
        print("=" * 60)
        
        for symbol in self.test_symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            data = self.get_historical_data(symbol)
            if data is None:
                continue
            
            # Test multiple points
            for i in range(60, len(data) - 20, 3):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                # Get actual outcome
                actual_result = self.get_actual_outcome(data, i)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                
                # Store pattern data
                pattern_data = {
                    'symbol': symbol,
                    'indicators': indicators.copy(),
                    'actual_change': actual_change
                }
                
                # Categorize by actual outcome
                if actual_direction == 'BUY':
                    self.successful_patterns['BUY'].append(pattern_data)
                elif actual_direction == 'SELL':
                    self.successful_patterns['SELL'].append(pattern_data)
                else:
                    self.successful_patterns['HOLD'].append(pattern_data)
        
        self.print_pattern_analysis()
    
    def print_pattern_analysis(self):
        print("\n" + "=" * 60)
        print("📈 SUCCESSFUL PATTERN ANALYSIS")
        print("=" * 60)
        
        for outcome_type in ['BUY', 'SELL', 'HOLD']:
            patterns = self.successful_patterns[outcome_type]
            if not patterns:
                continue
            
            print(f"\n🎯 {outcome_type} Patterns ({len(patterns)} samples):")
            
            # Calculate averages for key indicators
            avg_rsi = np.mean([p['indicators']['rsi'] for p in patterns])
            avg_momentum = np.mean([p['indicators']['price_momentum'] for p in patterns])
            avg_volume_ratio = np.mean([p['indicators']['volume_ratio'] for p in patterns])
            avg_price_vs_sma20 = np.mean([p['indicators']['price_vs_sma20'] for p in patterns])
            avg_bb_position = np.mean([p['indicators']['bb_position'] for p in patterns])
            avg_volatility = np.mean([p['indicators']['volatility'] for p in patterns])
            avg_change = np.mean([p['actual_change'] for p in patterns])
            
            print(f"  Average RSI: {avg_rsi:.1f}")
            print(f"  Average Momentum: {avg_momentum:.1f}%")
            print(f"  Average Volume Ratio: {avg_volume_ratio:.2f}")
            print(f"  Average Price vs SMA20: {avg_price_vs_sma20:.1f}%")
            print(f"  Average BB Position: {avg_bb_position:.2f}")
            print(f"  Average Volatility: {avg_volatility:.1f}%")
            print(f"  Average Actual Change: {avg_change:.1f}%")
            
            # Find ranges for key indicators
            rsi_range = (min([p['indicators']['rsi'] for p in patterns]), 
                        max([p['indicators']['rsi'] for p in patterns]))
            momentum_range = (min([p['indicators']['price_momentum'] for p in patterns]), 
                            max([p['indicators']['price_momentum'] for p in patterns]))
            
            print(f"  RSI Range: {rsi_range[0]:.1f} - {rsi_range[1]:.1f}")
            print(f"  Momentum Range: {momentum_range[0]:.1f}% - {momentum_range[1]:.1f}%")
    
    def find_optimal_thresholds(self):
        print("\n" + "=" * 60)
        print("🎯 OPTIMAL THRESHOLD ANALYSIS")
        print("=" * 60)
        
        all_patterns = []
        for outcome_type in ['BUY', 'SELL', 'HOLD']:
            for pattern in self.successful_patterns[outcome_type]:
                pattern['actual_outcome'] = outcome_type
                all_patterns.append(pattern)
        
        if not all_patterns:
            print("No patterns to analyze")
            return
        
        print(f"Total patterns analyzed: {len(all_patterns)}")
        
        # Analyze RSI thresholds
        buy_patterns = [p for p in all_patterns if p['actual_outcome'] == 'BUY']
        sell_patterns = [p for p in all_patterns if p['actual_outcome'] == 'SELL']
        hold_patterns = [p for p in all_patterns if p['actual_outcome'] == 'HOLD']
        
        if buy_patterns:
            buy_rsi_avg = np.mean([p['indicators']['rsi'] for p in buy_patterns])
            buy_momentum_avg = np.mean([p['indicators']['price_momentum'] for p in buy_patterns])
            print(f"\n✅ BUY Patterns ({len(buy_patterns)}):")
            print(f"  Optimal RSI threshold: > {buy_rsi_avg - 10:.0f}")
            print(f"  Optimal Momentum threshold: > {buy_momentum_avg - 2:.1f}%")
        
        if sell_patterns:
            sell_rsi_avg = np.mean([p['indicators']['rsi'] for p in sell_patterns])
            sell_momentum_avg = np.mean([p['indicators']['price_momentum'] for p in sell_patterns])
            print(f"\n❌ SELL Patterns ({len(sell_patterns)}):")
            print(f"  Optimal RSI threshold: < {sell_rsi_avg + 10:.0f}")
            print(f"  Optimal Momentum threshold: < {sell_momentum_avg + 2:.1f}%")
        
        if hold_patterns:
            hold_rsi_avg = np.mean([p['indicators']['rsi'] for p in hold_patterns])
            hold_momentum_avg = np.mean([p['indicators']['price_momentum'] for p in hold_patterns])
            print(f"\n⏸️ HOLD Patterns ({len(hold_patterns)}):")
            print(f"  Optimal RSI range: {hold_rsi_avg - 5:.0f} - {hold_rsi_avg + 5:.0f}")
            print(f"  Optimal Momentum range: {hold_momentum_avg - 1:.1f}% - {hold_momentum_avg + 1:.1f}%")
    
    def generate_improved_rules(self):
        print("\n" + "=" * 60)
        print("🚀 IMPROVED PREDICTION RULES")
        print("=" * 60)
        
        buy_patterns = self.successful_patterns['BUY']
        sell_patterns = self.successful_patterns['SELL']
        hold_patterns = self.successful_patterns['HOLD']
        
        if buy_patterns:
            # Analyze what makes successful BUY predictions
            strong_buy_indicators = []
            for pattern in buy_patterns:
                ind = pattern['indicators']
                if (ind['rsi'] > 45 and ind['price_momentum'] > 2 and 
                    ind['volume_ratio'] > 1.2 and ind['price_vs_sma20'] > 0):
                    strong_buy_indicators.append(pattern)
            
            print(f"\n✅ STRONG BUY Rules ({len(strong_buy_indicators)} patterns):")
            if strong_buy_indicators:
                avg_rsi = np.mean([p['indicators']['rsi'] for p in strong_buy_indicators])
                avg_momentum = np.mean([p['indicators']['price_momentum'] for p in strong_buy_indicators])
                avg_volume = np.mean([p['indicators']['volume_ratio'] for p in strong_buy_indicators])
                print(f"  RSI > {avg_rsi - 5:.0f}")
                print(f"  Momentum > {avg_momentum - 1:.1f}%")
                print(f"  Volume Ratio > {avg_volume - 0.2:.1f}")
                print(f"  Price above SMA20")
        
        if sell_patterns:
            # Analyze what makes successful SELL predictions
            strong_sell_indicators = []
            for pattern in sell_patterns:
                ind = pattern['indicators']
                if (ind['rsi'] < 55 and ind['price_momentum'] < -2 and 
                    ind['volume_ratio'] > 1.0 and ind['price_vs_sma20'] < 0):
                    strong_sell_indicators.append(pattern)
            
            print(f"\n❌ STRONG SELL Rules ({len(strong_sell_indicators)} patterns):")
            if strong_sell_indicators:
                avg_rsi = np.mean([p['indicators']['rsi'] for p in strong_sell_indicators])
                avg_momentum = np.mean([p['indicators']['price_momentum'] for p in strong_sell_indicators])
                avg_volume = np.mean([p['indicators']['volume_ratio'] for p in strong_sell_indicators])
                print(f"  RSI < {avg_rsi + 5:.0f}")
                print(f"  Momentum < {avg_momentum + 1:.1f}%")
                print(f"  Volume Ratio > {avg_volume - 0.2:.1f}")
                print(f"  Price below SMA20")
        
        if hold_patterns:
            # Analyze what makes successful HOLD predictions
            true_hold_indicators = []
            for pattern in hold_patterns:
                ind = pattern['indicators']
                if (40 <= ind['rsi'] <= 60 and abs(ind['price_momentum']) < 2 and 
                    0.8 <= ind['volume_ratio'] <= 1.3 and abs(ind['price_vs_sma20']) < 2):
                    true_hold_indicators.append(pattern)
            
            print(f"\n⏸️ TRUE HOLD Rules ({len(true_hold_indicators)} patterns):")
            if true_hold_indicators:
                avg_rsi = np.mean([p['indicators']['rsi'] for p in true_hold_indicators])
                avg_momentum_abs = np.mean([abs(p['indicators']['price_momentum']) for p in true_hold_indicators])
                avg_volume = np.mean([p['indicators']['volume_ratio'] for p in true_hold_indicators])
                print(f"  RSI between {avg_rsi - 8:.0f} - {avg_rsi + 8:.0f}")
                print(f"  |Momentum| < {avg_momentum_abs + 0.5:.1f}%")
                print(f"  Volume Ratio between {avg_volume - 0.2:.1f} - {avg_volume + 0.2:.1f}")
                print(f"  Price within 2% of SMA20")

if __name__ == "__main__":
    analyzer = PatternAnalyzer()
    analyzer.analyze_successful_patterns()
    analyzer.find_optimal_thresholds()
    analyzer.generate_improved_rules()
