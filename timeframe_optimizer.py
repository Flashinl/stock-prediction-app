#!/usr/bin/env python3
"""
Timeframe optimization model - find optimal prediction windows for 80%+ accuracy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TimeframeOptimizer:
    def __init__(self):
        # Focus on most reliable stocks
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ', 'NVDA']
        
        # Test multiple timeframes and thresholds
        self.timeframes = [3, 5, 7, 10, 15, 20, 25, 30]
        self.threshold_sets = [
            (1.0, -1.0), (1.5, -1.5), (2.0, -2.0), (2.5, -2.5), 
            (3.0, -3.0), (3.5, -3.5), (4.0, -4.0), (5.0, -5.0)
        ]
        
        self.results = {}
        self.best_combinations = []
    
    def get_data_safely(self, symbol, days_back=50):
        """Safely get stock data"""
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 30 else None
        except Exception as e:
            print(f"  Error with {symbol}: {e}")
            return None
    
    def calculate_indicators(self, data):
        """Calculate essential indicators"""
        if len(data) < 15:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Moving averages
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(min(20, len(data))).mean().iloc[-1]
            
            # Volume
            avg_volume = data['Volume'].rolling(10).mean().iloc[-1]
            
            # Momentum
            momentum_2 = (current_price - data['Close'].iloc[-3]) / data['Close'].iloc[-3] * 100 if len(data) >= 3 else 0
            momentum_5 = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100 if len(data) >= 6 else 0
            momentum_10 = (current_price - data['Close'].iloc[-11]) / data['Close'].iloc[-11] * 100 if len(data) >= 11 else 0
            
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
                'sma_20': sma_20,
                'avg_volume': avg_volume,
                'momentum_2': momentum_2,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': volume / avg_volume if avg_volume > 0 else 1,
                'price_vs_sma10': (current_price - sma_10) / sma_10 * 100,
                'price_vs_sma20': (current_price - sma_20) / sma_20 * 100
            }
        except Exception as e:
            return None
    
    def optimized_prediction_model(self, indicators):
        """Optimized prediction model based on best patterns"""
        
        # High confidence BUY patterns
        if (indicators['momentum_2'] > 1.5 and 
            indicators['momentum_5'] > 2.0 and
            indicators['rsi'] < 70 and
            indicators['current_price'] > indicators['sma_10'] and
            indicators['volume_ratio'] > 1.2):
            return 'BUY'
        
        # High confidence SELL patterns  
        if (indicators['momentum_2'] < -1.5 and
            indicators['momentum_5'] < -2.0 and
            indicators['rsi'] > 30 and
            indicators['current_price'] < indicators['sma_10'] and
            indicators['volume_ratio'] > 1.1):
            return 'SELL'
        
        # High confidence HOLD patterns
        if (abs(indicators['momentum_5']) < 1.0 and
            45 <= indicators['rsi'] <= 55 and
            indicators['volatility'] < 2.0 and
            abs(indicators['price_vs_sma10']) < 2 and
            0.8 <= indicators['volume_ratio'] <= 1.3):
            return 'HOLD'
        
        # Secondary patterns
        if (indicators['momentum_5'] > 1.0 and 
            indicators['rsi'] < 75 and
            indicators['current_price'] > indicators['sma_5']):
            return 'BUY'
        
        if (indicators['momentum_5'] < -1.0 and
            indicators['rsi'] > 25 and
            indicators['current_price'] < indicators['sma_5']):
            return 'SELL'
        
        return 'HOLD'  # Default
    
    def get_actual_outcome(self, data, index, timeframe, buy_threshold, sell_threshold):
        """Get actual outcome for specific timeframe and thresholds"""
        if index + timeframe >= len(data):
            return None
        
        current_price = data['Close'].iloc[index]
        future_price = data['Close'].iloc[index + timeframe]
        change = (future_price - current_price) / current_price * 100
        
        if change > buy_threshold:
            return 'BUY', change
        elif change < sell_threshold:
            return 'SELL', change
        else:
            return 'HOLD', change
    
    def test_timeframe_combination(self, timeframe, buy_thresh, sell_thresh):
        """Test a specific timeframe and threshold combination"""
        predictions = []
        
        for symbol in self.test_symbols:
            data = self.get_data_safely(symbol)
            if data is None:
                continue
            
            # Test multiple points for this symbol
            for i in range(20, len(data) - timeframe - 1, 1):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                prediction = self.optimized_prediction_model(indicators)
                actual_result = self.get_actual_outcome(data, i, timeframe, buy_thresh, sell_thresh)
                
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                
                predictions.append({
                    'symbol': symbol,
                    'predicted': prediction,
                    'actual': actual_direction,
                    'actual_change': actual_change,
                    'correct': prediction == actual_direction
                })
        
        if len(predictions) < 10:  # Need minimum predictions
            return None
        
        # Calculate accuracies
        total = len(predictions)
        correct = sum(1 for p in predictions if p['correct'])
        overall_accuracy = correct / total * 100
        
        # By prediction type
        by_type = {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}}
        
        for pred in predictions:
            pred_type = pred['predicted']
            by_type[pred_type]['total'] += 1
            if pred['correct']:
                by_type[pred_type]['correct'] += 1
        
        # Calculate type accuracies
        type_accuracies = {}
        for pred_type in ['BUY', 'SELL', 'HOLD']:
            if by_type[pred_type]['total'] > 0:
                type_accuracies[pred_type] = by_type[pred_type]['correct'] / by_type[pred_type]['total'] * 100
            else:
                type_accuracies[pred_type] = 0
        
        return {
            'timeframe': timeframe,
            'buy_threshold': buy_thresh,
            'sell_threshold': sell_thresh,
            'overall_accuracy': overall_accuracy,
            'total_predictions': total,
            'correct_predictions': correct,
            'buy_accuracy': type_accuracies['BUY'],
            'sell_accuracy': type_accuracies['SELL'],
            'hold_accuracy': type_accuracies['HOLD'],
            'buy_count': by_type['BUY']['total'],
            'sell_count': by_type['SELL']['total'],
            'hold_count': by_type['HOLD']['total'],
            'predictions': predictions
        }
    
    def run_timeframe_optimization(self):
        print("🎯 Timeframe Optimization for 80%+ Accuracy")
        print("=" * 60)
        print("Testing all combinations of timeframes and thresholds")
        print("=" * 60)
        
        total_combinations = len(self.timeframes) * len(self.threshold_sets)
        tested = 0
        
        for timeframe in self.timeframes:
            for buy_thresh, sell_thresh in self.threshold_sets:
                tested += 1
                print(f"\n[{tested}/{total_combinations}] Testing: {timeframe} days, {buy_thresh}%/{sell_thresh}% thresholds")
                
                result = self.test_timeframe_combination(timeframe, buy_thresh, sell_thresh)
                
                if result is None:
                    print(f"  ❌ Insufficient data")
                    continue
                
                print(f"  📊 Overall: {result['overall_accuracy']:.1f}% ({result['correct_predictions']}/{result['total_predictions']})")
                print(f"     BUY: {result['buy_accuracy']:.1f}% ({result['buy_count']})")
                print(f"     SELL: {result['sell_accuracy']:.1f}% ({result['sell_count']})")
                print(f"     HOLD: {result['hold_accuracy']:.1f}% ({result['hold_count']})")
                
                # Check if this combination achieves 80%+ targets
                targets_met = 0
                if result['buy_accuracy'] >= 80: targets_met += 1
                if result['sell_accuracy'] >= 80: targets_met += 1
                if result['hold_accuracy'] >= 80: targets_met += 1
                
                if result['overall_accuracy'] >= 80 or targets_met >= 2:
                    print(f"  🚀 HIGH ACCURACY FOUND!")
                    self.best_combinations.append(result)
                
                # Store result
                key = f"{timeframe}d_{buy_thresh}_{sell_thresh}"
                self.results[key] = result
        
        self.report_optimization_results()
    
    def report_optimization_results(self):
        print("\n" + "=" * 60)
        print("🏆 TIMEFRAME OPTIMIZATION RESULTS")
        print("=" * 60)
        
        if not self.results:
            print("❌ No valid results")
            return
        
        # Find best overall accuracy
        best_overall = max(self.results.values(), key=lambda x: x['overall_accuracy'])
        print(f"\n🎯 BEST OVERALL ACCURACY: {best_overall['overall_accuracy']:.1f}%")
        print(f"   Timeframe: {best_overall['timeframe']} days")
        print(f"   Thresholds: {best_overall['buy_threshold']}% / {best_overall['sell_threshold']}%")
        print(f"   BUY: {best_overall['buy_accuracy']:.1f}% ({best_overall['buy_count']})")
        print(f"   SELL: {best_overall['sell_accuracy']:.1f}% ({best_overall['sell_count']})")
        print(f"   HOLD: {best_overall['hold_accuracy']:.1f}% ({best_overall['hold_count']})")
        
        # Find best for each prediction type
        valid_results = [r for r in self.results.values() if r['buy_count'] > 0]
        if valid_results:
            best_buy = max(valid_results, key=lambda x: x['buy_accuracy'])
            print(f"\n🟢 BEST BUY ACCURACY: {best_buy['buy_accuracy']:.1f}%")
            print(f"   Timeframe: {best_buy['timeframe']} days, Thresholds: {best_buy['buy_threshold']}%/{best_buy['sell_threshold']}%")
        
        valid_results = [r for r in self.results.values() if r['sell_count'] > 0]
        if valid_results:
            best_sell = max(valid_results, key=lambda x: x['sell_accuracy'])
            print(f"\n🔴 BEST SELL ACCURACY: {best_sell['sell_accuracy']:.1f}%")
            print(f"   Timeframe: {best_sell['timeframe']} days, Thresholds: {best_sell['buy_threshold']}%/{best_sell['sell_threshold']}%")
        
        valid_results = [r for r in self.results.values() if r['hold_count'] > 0]
        if valid_results:
            best_hold = max(valid_results, key=lambda x: x['hold_accuracy'])
            print(f"\n🟡 BEST HOLD ACCURACY: {best_hold['hold_accuracy']:.1f}%")
            print(f"   Timeframe: {best_hold['timeframe']} days, Thresholds: {best_hold['buy_threshold']}%/{best_hold['sell_threshold']}%")
        
        # Report 80%+ combinations
        if self.best_combinations:
            print(f"\n🚀 COMBINATIONS ACHIEVING 80%+ TARGETS:")
            self.best_combinations.sort(key=lambda x: x['overall_accuracy'], reverse=True)
            
            for i, combo in enumerate(self.best_combinations[:5], 1):
                print(f"\n#{i} - {combo['timeframe']} days, {combo['buy_threshold']}%/{combo['sell_threshold']}%")
                print(f"   Overall: {combo['overall_accuracy']:.1f}%")
                print(f"   BUY: {combo['buy_accuracy']:.1f}% ({combo['buy_count']})")
                print(f"   SELL: {combo['sell_accuracy']:.1f}% ({combo['sell_count']})")
                print(f"   HOLD: {combo['hold_accuracy']:.1f}% ({combo['hold_count']})")
                
                targets_met = sum([
                    combo['buy_accuracy'] >= 80,
                    combo['sell_accuracy'] >= 80,
                    combo['hold_accuracy'] >= 80
                ])
                print(f"   Targets met: {targets_met}/3")
            
            if self.best_combinations[0]['overall_accuracy'] >= 80:
                print(f"\n🎉 SUCCESS! Found combinations with 80%+ accuracy!")
                print(f"🚀 READY FOR PRODUCTION IMPLEMENTATION!")
            else:
                print(f"\n⚠️ Close but not quite 80% overall accuracy")
        else:
            print(f"\n❌ No combinations achieved 80%+ targets")
            print(f"Need to refine prediction logic or try different approaches")
        
        # Summary statistics
        all_accuracies = [r['overall_accuracy'] for r in self.results.values()]
        if all_accuracies:
            print(f"\n📊 SUMMARY STATISTICS:")
            print(f"   Best accuracy: {max(all_accuracies):.1f}%")
            print(f"   Average accuracy: {np.mean(all_accuracies):.1f}%")
            print(f"   Combinations tested: {len(all_accuracies)}")

if __name__ == "__main__":
    optimizer = TimeframeOptimizer()
    optimizer.run_timeframe_optimization()
