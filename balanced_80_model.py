#!/usr/bin/env python3
"""
Balanced 80%+ accuracy model - makes predictions across all types
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Balanced80Model:
    def __init__(self):
        # Reliable stocks for testing
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ', 'NVDA', 'TSLA', 'META']
        
        self.results = {
            'simple_model': {'correct': 0, 'total': 0, 'predictions': []},
            'balanced_model': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        self.detailed_results = {
            'simple_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}},
            'balanced_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}}
        }
    
    def get_data(self, symbol, days_back=40):
        """Get stock data"""
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 20 else None
        except:
            return None
    
    def calculate_indicators(self, data):
        """Calculate essential indicators"""
        if len(data) < 10:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Simple moving averages
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            
            # Volume
            avg_volume = data['Volume'].rolling(5).mean().iloc[-1]
            
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
            
            return {
                'current_price': current_price,
                'volume': volume,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'avg_volume': avg_volume,
                'momentum_2': momentum_2,
                'momentum_5': momentum_5,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': volume / avg_volume if avg_volume > 0 else 1,
                'price_vs_sma5': (current_price - sma_5) / sma_5 * 100,
                'price_vs_sma10': (current_price - sma_10) / sma_10 * 100
            }
        except:
            return None
    
    def simple_model(self, indicators):
        """Simple baseline model"""
        if indicators['momentum_5'] > 1.5 and indicators['rsi'] < 70:
            return 'BUY'
        elif indicators['momentum_5'] < -1.5 and indicators['rsi'] > 30:
            return 'SELL'
        else:
            return 'HOLD'
    
    def balanced_80_model(self, indicators):
        """Balanced model targeting 80%+ across all types"""
        
        # Calculate confidence scores for each prediction type
        buy_score = 0
        sell_score = 0
        hold_score = 0
        
        # === BUY SCORING ===
        
        # Strong momentum
        if indicators['momentum_2'] > 1.5:
            buy_score += 25
        elif indicators['momentum_2'] > 1.0:
            buy_score += 15
        elif indicators['momentum_2'] > 0.5:
            buy_score += 5
        
        if indicators['momentum_5'] > 2.0:
            buy_score += 25
        elif indicators['momentum_5'] > 1.0:
            buy_score += 15
        
        # RSI favorable for buying
        if 40 <= indicators['rsi'] <= 60:
            buy_score += 20
        elif 35 <= indicators['rsi'] <= 65:
            buy_score += 10
        elif indicators['rsi'] > 75:
            buy_score -= 20  # Overbought penalty
        
        # Volume confirmation
        if indicators['volume_ratio'] > 1.5:
            buy_score += 20
        elif indicators['volume_ratio'] > 1.2:
            buy_score += 10
        
        # Price above moving averages
        if indicators['current_price'] > indicators['sma_5'] > indicators['sma_10']:
            buy_score += 20
        elif indicators['current_price'] > indicators['sma_10']:
            buy_score += 10
        
        # Low volatility (more predictable)
        if indicators['volatility'] < 2:
            buy_score += 10
        elif indicators['volatility'] > 4:
            buy_score -= 10
        
        # === SELL SCORING ===
        
        # Strong negative momentum
        if indicators['momentum_2'] < -1.5:
            sell_score += 25
        elif indicators['momentum_2'] < -1.0:
            sell_score += 15
        elif indicators['momentum_2'] < -0.5:
            sell_score += 5
        
        if indicators['momentum_5'] < -2.0:
            sell_score += 25
        elif indicators['momentum_5'] < -1.0:
            sell_score += 15
        
        # RSI favorable for selling
        if indicators['rsi'] > 70:
            sell_score += 20
        elif indicators['rsi'] > 60:
            sell_score += 10
        elif indicators['rsi'] < 25:
            sell_score -= 20  # Oversold penalty
        
        # Volume on decline
        if indicators['volume_ratio'] > 1.3 and indicators['momentum_2'] < 0:
            sell_score += 20
        elif indicators['volume_ratio'] > 1.1 and indicators['momentum_2'] < 0:
            sell_score += 10
        
        # Price below moving averages
        if indicators['current_price'] < indicators['sma_5'] < indicators['sma_10']:
            sell_score += 20
        elif indicators['current_price'] < indicators['sma_10']:
            sell_score += 10
        
        # === HOLD SCORING ===
        
        # Low momentum
        if abs(indicators['momentum_5']) < 0.5:
            hold_score += 25
        elif abs(indicators['momentum_5']) < 1.0:
            hold_score += 15
        
        if abs(indicators['momentum_2']) < 0.3:
            hold_score += 20
        elif abs(indicators['momentum_2']) < 0.7:
            hold_score += 10
        
        # Neutral RSI
        if 45 <= indicators['rsi'] <= 55:
            hold_score += 25
        elif 40 <= indicators['rsi'] <= 60:
            hold_score += 15
        
        # Normal volume
        if 0.8 <= indicators['volume_ratio'] <= 1.2:
            hold_score += 15
        elif 0.7 <= indicators['volume_ratio'] <= 1.3:
            hold_score += 10
        
        # Price near moving averages
        if abs(indicators['price_vs_sma10']) < 1:
            hold_score += 20
        elif abs(indicators['price_vs_sma10']) < 2:
            hold_score += 10
        
        # Low volatility
        if indicators['volatility'] < 1.5:
            hold_score += 15
        elif indicators['volatility'] < 2.5:
            hold_score += 10
        
        # === DECISION LOGIC ===
        
        # Require minimum confidence for any prediction
        min_confidence = 60
        
        # Find the highest scoring prediction
        max_score = max(buy_score, sell_score, hold_score)
        
        if max_score < min_confidence:
            # If no high confidence, default to HOLD
            return 'HOLD'
        
        # Return the highest confidence prediction
        if buy_score == max_score and buy_score >= min_confidence:
            return 'BUY'
        elif sell_score == max_score and sell_score >= min_confidence:
            return 'SELL'
        else:
            return 'HOLD'
    
    def get_actual_outcome(self, data, index, timeframe=8):
        """Get actual outcome with optimized timeframe"""
        if index + timeframe >= len(data):
            return None
        
        current_price = data['Close'].iloc[index]
        future_price = data['Close'].iloc[index + timeframe]
        change = (future_price - current_price) / current_price * 100
        
        # Balanced thresholds
        if change > 1.2:
            return 'BUY', change
        elif change < -1.2:
            return 'SELL', change
        else:
            return 'HOLD', change
    
    def run_balanced_test(self):
        print("🎯 Balanced 80%+ Accuracy Model Test")
        print("=" * 50)
        print("Targeting 80%+ accuracy across BUY, SELL, and HOLD")
        print("=" * 50)
        
        symbols_tested = 0
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            data = self.get_data(symbol)
            if data is None:
                print(f"  ❌ Could not get data for {symbol}")
                continue
            
            symbols_tested += 1
            tests_for_symbol = 0
            
            # Test every day
            for i in range(10, len(data) - 10, 1):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                simple_pred = self.simple_model(indicators)
                balanced_pred = self.balanced_80_model(indicators)
                
                actual_result = self.get_actual_outcome(data, i)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                tests_for_symbol += 1
                
                # Record results
                for model_name, prediction in [('simple_model', simple_pred), ('balanced_model', balanced_pred)]:
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
        self.print_balanced_results()
    
    def print_balanced_results(self):
        print("\n" + "=" * 50)
        print("🏆 BALANCED 80%+ MODEL RESULTS")
        print("=" * 50)
        
        simple_accuracy = (self.results['simple_model']['correct'] / 
                         self.results['simple_model']['total'] * 100) if self.results['simple_model']['total'] > 0 else 0
        
        balanced_accuracy = (self.results['balanced_model']['correct'] / 
                           self.results['balanced_model']['total'] * 100) if self.results['balanced_model']['total'] > 0 else 0
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"Simple Model:   {simple_accuracy:.1f}% ({self.results['simple_model']['correct']}/{self.results['simple_model']['total']})")
        print(f"Balanced Model: {balanced_accuracy:.1f}% ({self.results['balanced_model']['correct']}/{self.results['balanced_model']['total']})")
        
        improvement = balanced_accuracy - simple_accuracy
        print(f"Improvement:    {improvement:+.1f}%")
        
        # Detailed accuracy by prediction type
        print(f"\n📈 DETAILED ACCURACY BY PREDICTION TYPE:")
        
        for model_name in ['simple_model', 'balanced_model']:
            print(f"\n{model_name.replace('_', ' ').title()}:")
            
            for pred_type in ['BUY', 'SELL', 'HOLD']:
                stats = self.detailed_results[model_name][pred_type]
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    target = 80
                    status = '✅' if accuracy >= target else '❌'
                    print(f"  {pred_type}: {status} {accuracy:.1f}% ({stats['correct']}/{stats['total']}) [Target: {target}%]")
                else:
                    print(f"  {pred_type}: No predictions made")
        
        # 80% target assessment
        print(f"\n🎯 80% ACCURACY TARGET ASSESSMENT:")
        balanced_results = self.detailed_results['balanced_model']
        
        targets_achieved = 0
        all_types_predicted = True
        
        for pred_type in ['BUY', 'SELL', 'HOLD']:
            stats = balanced_results[pred_type]
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                if accuracy >= 80:
                    targets_achieved += 1
                    print(f"✅ {pred_type}: {accuracy:.1f}% >= 80% TARGET ACHIEVED!")
                else:
                    gap = 80 - accuracy
                    print(f"❌ {pred_type}: {accuracy:.1f}% < 80% (need {gap:.1f}% more)")
            else:
                all_types_predicted = False
                print(f"⚪ {pred_type}: No predictions made")
        
        print(f"\nTargets achieved: {targets_achieved}/3")
        print(f"Overall accuracy: {balanced_accuracy:.1f}%")
        print(f"All types predicted: {'Yes' if all_types_predicted else 'No'}")
        
        # Final assessment
        if targets_achieved == 3 and balanced_accuracy >= 80:
            print(f"\n🎉🎉🎉 MISSION ACCOMPLISHED! 🎉🎉🎉")
            print(f"🚀 ALL PREDICTION TYPES ACHIEVED 80%+ ACCURACY!")
            print(f"🏆 READY FOR PRODUCTION DEPLOYMENT!")
        elif targets_achieved >= 2 and balanced_accuracy >= 70:
            print(f"\n🎊 EXCELLENT PROGRESS! 🎊")
            print(f"✅ {targets_achieved}/3 targets achieved")
            print(f"📈 {balanced_accuracy:.1f}% overall accuracy")
            print(f"🚀 Very close to full 80%+ achievement!")
        elif targets_achieved >= 1 or balanced_accuracy >= 60:
            print(f"\n⚠️ GOOD PROGRESS")
            print(f"📊 {targets_achieved}/3 targets achieved")
            print(f"📈 {balanced_accuracy:.1f}% overall accuracy")
            print(f"🔧 Continue refinement for 80%+ targets")
        else:
            print(f"\n❌ MORE WORK NEEDED")
            print(f"📊 Current accuracy: {balanced_accuracy:.1f}%")
            print(f"🎯 Need significant improvement to reach 80%+")

if __name__ == "__main__":
    model = Balanced80Model()
    model.run_balanced_test()
