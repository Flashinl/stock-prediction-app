#!/usr/bin/env python3
"""
Direct test of enhanced HOLD and BUY logic without Flask context
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DirectLogicTester:
    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ', 'NVDA']
        
        self.results = {
            'old_logic': {'correct': 0, 'total': 0, 'predictions': []},
            'enhanced_logic': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        self.detailed_results = {
            'old_logic': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}},
            'enhanced_logic': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}}
        }
    
    def get_data(self, symbol, days_back=35):
        """Get stock data"""
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 20 else None
        except Exception as e:
            return None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        if len(data) < 15:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Moving averages
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            
            # Volume
            avg_volume = data['Volume'].rolling(10).mean().iloc[-1]
            
            # Momentum
            momentum_2 = (current_price - data['Close'].iloc[-3]) / data['Close'].iloc[-3] * 100 if len(data) >= 3 else 0
            momentum_5 = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100 if len(data) >= 6 else 0
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
            
            # Volatility
            volatility = data['Close'].pct_change().rolling(5).std().iloc[-1] * 100
            
            # Market regime (simple)
            price_10_ago = data['Close'].iloc[-11] if len(data) >= 11 else data['Close'].iloc[0]
            trend_strength = (current_price - price_10_ago) / price_10_ago * 100
            
            if trend_strength > 2:
                regime = 'BULL'
            elif trend_strength < -2:
                regime = 'BEAR'
            else:
                regime = 'SIDEWAYS'
            
            return {
                'current_price': current_price,
                'volume': volume,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'avg_volume': avg_volume,
                'momentum_2': momentum_2,
                'momentum_5': momentum_5,
                'rsi': rsi,
                'volatility': volatility,
                'regime': regime,
                'volume_ratio': volume / avg_volume if avg_volume > 0 else 1,
                'price_momentum': momentum_5,  # Alias for compatibility
                'momentum_strength': momentum_5 / 10,  # Normalized
                'price_vs_sma20': (current_price - sma_20) / sma_20 * 100
            }
        except Exception as e:
            return None
    
    def old_logic(self, indicators):
        """Old simple logic"""
        if indicators['momentum_5'] > 2 and indicators['rsi'] < 70:
            return 'BUY'
        elif indicators['momentum_5'] < -2 and indicators['rsi'] > 30:
            return 'SELL'
        else:
            return 'HOLD'
    
    def enhanced_buy_logic(self, indicators):
        """Enhanced BUY logic from our implementation"""
        price_momentum = indicators['price_momentum']
        momentum_strength = indicators['momentum_strength']
        rsi = indicators['rsi']
        current_price = indicators['current_price']
        sma_20 = indicators['sma_20']
        volume_ratio = indicators['volume_ratio']
        volatility = indicators['volatility']
        
        # High confidence BUY patterns
        high_confidence_buy = (
            price_momentum > 1.5 and 
            momentum_strength > 0.3 and
            rsi < 70 and
            current_price > sma_20 and
            volume_ratio > 1.2 and
            volatility < 3.5
        )
        
        # Strong momentum BUY
        strong_momentum_buy = (
            price_momentum > 2.0 and
            rsi < 75 and
            current_price > sma_20 and
            momentum_strength > 0.2 and
            volume_ratio > 1.1
        )
        
        # Oversold bounce BUY
        oversold_bounce_buy = (
            rsi < 35 and
            price_momentum > 1.0 and
            volume_ratio > 1.3 and
            current_price > sma_20 and
            momentum_strength > 0.1
        )
        
        # Breakout BUY
        breakout_buy = (
            price_momentum > 2.5 and
            volume_ratio > 1.5 and
            rsi < 75 and
            momentum_strength > 0.4
        )
        
        return (high_confidence_buy or strong_momentum_buy or 
                oversold_bounce_buy or breakout_buy)
    
    def enhanced_hold_logic(self, indicators):
        """Enhanced HOLD logic from our implementation"""
        price_momentum = indicators['price_momentum']
        momentum_strength = indicators['momentum_strength']
        rsi = indicators['rsi']
        current_price = indicators['current_price']
        sma_20 = indicators['sma_20']
        volume_ratio = indicators['volume_ratio']
        volatility = indicators['volatility']
        regime = indicators['regime']
        
        # Perfect consolidation pattern
        perfect_consolidation = (
            abs(price_momentum) < 0.5 and
            abs(momentum_strength) < 0.1 and
            volatility < 1.5 and
            48 <= rsi <= 52 and
            abs((current_price - sma_20) / sma_20 * 100) < 1 and
            0.9 <= volume_ratio <= 1.1
        )
        
        # Tight range consolidation
        tight_range_consolidation = (
            volatility < 1.0 and
            abs(price_momentum) < 0.3 and
            45 <= rsi <= 55 and
            abs((current_price - sma_20) / sma_20 * 100) < 0.5 and
            0.95 <= volume_ratio <= 1.05
        )
        
        # Neutral momentum zone
        neutral_momentum_zone = (
            abs(price_momentum) < 1.0 and
            45 <= rsi <= 55 and
            volatility < 2.0 and
            abs(momentum_strength) < 0.15 and
            0.8 <= volume_ratio <= 1.2
        )
        
        # Sideways pattern
        sideways_pattern = (
            regime == 'SIDEWAYS' and
            abs(price_momentum) < 1.5 and
            40 <= rsi <= 60 and
            volatility < 2.5 and
            0.85 <= volume_ratio <= 1.15
        )
        
        return (perfect_consolidation or tight_range_consolidation or 
                neutral_momentum_zone or sideways_pattern)
    
    def enhanced_logic(self, indicators):
        """Enhanced logic combining best BUY and HOLD patterns"""
        
        # Check for enhanced BUY patterns
        if self.enhanced_buy_logic(indicators):
            return 'BUY'
        
        # Check for enhanced HOLD patterns
        elif self.enhanced_hold_logic(indicators):
            return 'HOLD'
        
        # Enhanced SELL logic (simple for now)
        elif (indicators['momentum_5'] < -2.0 and 
              indicators['rsi'] > 30 and
              indicators['current_price'] < indicators['sma_20'] and
              indicators['volume_ratio'] > 1.1):
            return 'SELL'
        
        # Default to old logic
        else:
            return self.old_logic(indicators)
    
    def get_actual_outcome(self, data, index, timeframe=8):
        """Get actual outcome"""
        if index + timeframe >= len(data):
            return None
        
        current_price = data['Close'].iloc[index]
        future_price = data['Close'].iloc[index + timeframe]
        change = (future_price - current_price) / current_price * 100
        
        if change > 1.2:
            return 'BUY', change
        elif change < -1.2:
            return 'SELL', change
        else:
            return 'HOLD', change
    
    def run_direct_test(self):
        print("🧪 Direct Test of Enhanced HOLD and BUY Logic")
        print("=" * 55)
        print("Comparing old vs enhanced logic patterns")
        print("=" * 55)
        
        symbols_tested = 0
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            data = self.get_data(symbol)
            if data is None:
                print(f"  ❌ Could not get data for {symbol}")
                continue
            
            symbols_tested += 1
            tests_for_symbol = 0
            
            # Test multiple points
            for i in range(15, len(data) - 10, 2):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                old_pred = self.old_logic(indicators)
                enhanced_pred = self.enhanced_logic(indicators)
                
                actual_result = self.get_actual_outcome(data, i)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                tests_for_symbol += 1
                
                # Record results
                for model_name, prediction in [('old_logic', old_pred), ('enhanced_logic', enhanced_pred)]:
                    self.results[model_name]['total'] += 1
                    self.detailed_results[model_name][prediction]['total'] += 1
                    
                    if prediction == actual_direction:
                        self.results[model_name]['correct'] += 1
                        self.detailed_results[model_name][prediction]['correct'] += 1
                    
                    self.results[model_name]['predictions'].append({
                        'symbol': symbol,
                        'predicted': prediction,
                        'actual': actual_direction,
                        'actual_change': actual_change,
                        'correct': prediction == actual_direction
                    })
            
            print(f"  Completed {tests_for_symbol} tests for {symbol}")
        
        print(f"\n✅ Tested {symbols_tested} symbols")
        self.print_direct_results()
    
    def print_direct_results(self):
        print("\n" + "=" * 55)
        print("🎯 ENHANCED LOGIC TEST RESULTS")
        print("=" * 55)
        
        old_accuracy = (self.results['old_logic']['correct'] / 
                       self.results['old_logic']['total'] * 100) if self.results['old_logic']['total'] > 0 else 0
        
        enhanced_accuracy = (self.results['enhanced_logic']['correct'] / 
                           self.results['enhanced_logic']['total'] * 100) if self.results['enhanced_logic']['total'] > 0 else 0
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"Old Logic:      {old_accuracy:.1f}% ({self.results['old_logic']['correct']}/{self.results['old_logic']['total']})")
        print(f"Enhanced Logic: {enhanced_accuracy:.1f}% ({self.results['enhanced_logic']['correct']}/{self.results['enhanced_logic']['total']})")
        
        improvement = enhanced_accuracy - old_accuracy
        print(f"Improvement:    {improvement:+.1f}%")
        
        # Detailed accuracy by prediction type
        print(f"\n📈 DETAILED ACCURACY BY PREDICTION TYPE:")
        
        for model_name in ['old_logic', 'enhanced_logic']:
            print(f"\n{model_name.replace('_', ' ').title()}:")
            
            for pred_type in ['BUY', 'SELL', 'HOLD']:
                stats = self.detailed_results[model_name][pred_type]
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    print(f"  {pred_type}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
                else:
                    print(f"  {pred_type}: No predictions made")
        
        # Enhanced system assessment
        print(f"\n🎯 ENHANCED SYSTEM ASSESSMENT:")
        enhanced_results = self.detailed_results['enhanced_logic']
        
        targets_achieved = 0
        for pred_type in ['BUY', 'SELL', 'HOLD']:
            stats = enhanced_results[pred_type]
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                if accuracy >= 70:  # Lower target for this test
                    targets_achieved += 1
                    print(f"✅ {pred_type}: {accuracy:.1f}% >= 70% (Good performance)")
                elif accuracy >= 60:
                    print(f"⚠️ {pred_type}: {accuracy:.1f}% (Moderate performance)")
                else:
                    print(f"❌ {pred_type}: {accuracy:.1f}% (Needs improvement)")
            else:
                print(f"⚪ {pred_type}: No predictions made")
        
        print(f"\nTargets achieved (70%+): {targets_achieved}/3")
        
        if improvement > 5 and enhanced_accuracy > 60:
            print(f"\n🎊 ENHANCED LOGIC WORKING!")
            print(f"Significant improvement achieved: +{improvement:.1f}%")
            print(f"Enhanced HOLD and BUY systems showing promise")
        elif improvement > 0:
            print(f"\n⚠️ MODEST IMPROVEMENT")
            print(f"Enhanced logic shows +{improvement:.1f}% improvement")
            print(f"Continue refinement for better results")
        else:
            print(f"\n❌ NEEDS MORE WORK")
            print(f"Enhanced logic not outperforming old logic")

if __name__ == "__main__":
    tester = DirectLogicTester()
    tester.run_direct_test()
