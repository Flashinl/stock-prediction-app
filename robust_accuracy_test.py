#!/usr/bin/env python3
"""
Robust accuracy test with simplified logic and better error handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class RobustAccuracyTest:
    def __init__(self):
        # Diverse set of reliable stocks
        self.test_stocks = [
            # Large Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            # Large Cap Traditional
            'JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA', 'HD',
            # ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'XLK', 'XLF',
            # Mid Cap
            'PLTR', 'SNOW', 'CRWD', 'ZM', 'ROKU',
            # Financial
            'BAC', 'GS', 'MS', 'C', 'WFC',
            # Healthcare
            'PFE', 'ABBV', 'TMO', 'ABT', 'MRNA',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG',
            # Consumer
            'NKE', 'SBUX', 'MCD', 'DIS', 'COST'
        ]
        
        self.results = {
            'total_tests': 0,
            'correct_predictions': 0,
            'predictions': [],
            'by_type': {
                'BUY': {'correct': 0, 'total': 0},
                'SELL': {'correct': 0, 'total': 0},
                'HOLD': {'correct': 0, 'total': 0}
            }
        }
    
    def get_stock_data_safe(self, symbol, days_back=30):
        """Safely get stock data with multiple attempts"""
        for attempt in range(2):
            try:
                stock = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                data = stock.history(start=start_date, end=end_date)
                
                if len(data) > 15:
                    return data
                    
            except Exception as e:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                else:
                    print(f"    Failed to get data for {symbol}: {e}")
        
        return None
    
    def calculate_simple_indicators(self, data):
        """Calculate simple but robust indicators"""
        try:
            if len(data) < 10:
                return None
            
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Simple moving averages
            sma_5 = data['Close'].rolling(5).mean().iloc[-1] if len(data) >= 5 else current_price
            sma_10 = data['Close'].rolling(10).mean().iloc[-1] if len(data) >= 10 else current_price
            sma_20 = data['Close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else current_price
            
            # Volume average
            avg_volume = data['Volume'].rolling(10).mean().iloc[-1] if len(data) >= 10 else volume
            
            # Simple momentum
            momentum_3 = 0
            momentum_5 = 0
            
            if len(data) >= 4:
                momentum_3 = (current_price - data['Close'].iloc[-4]) / data['Close'].iloc[-4] * 100
            
            if len(data) >= 6:
                momentum_5 = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100
            
            # Simple RSI
            rsi = 50  # Default neutral
            if len(data) >= 15:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi_calc = 100 - (100 / (1 + rs)).iloc[-1]
                if not pd.isna(rsi_calc):
                    rsi = rsi_calc
            
            # Simple volatility
            volatility = 2.0  # Default moderate
            if len(data) >= 5:
                vol_calc = data['Close'].pct_change().rolling(5).std().iloc[-1] * 100
                if not pd.isna(vol_calc):
                    volatility = vol_calc
            
            return {
                'current_price': current_price,
                'volume': volume,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'avg_volume': avg_volume,
                'momentum_3': momentum_3,
                'momentum_5': momentum_5,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': volume / avg_volume if avg_volume > 0 else 1.0
            }
            
        except Exception as e:
            print(f"    Error calculating indicators: {e}")
            return None
    
    def enhanced_prediction_simple(self, indicators):
        """Simplified enhanced prediction logic"""
        try:
            current_price = indicators['current_price']
            sma_20 = indicators['sma_20']
            momentum_5 = indicators['momentum_5']
            rsi = indicators['rsi']
            volume_ratio = indicators['volume_ratio']
            volatility = indicators['volatility']
            
            # Enhanced BUY patterns (simplified)
            strong_buy = (
                momentum_5 > 2.0 and
                rsi < 70 and
                current_price > sma_20 and
                volume_ratio > 1.2
            )
            
            moderate_buy = (
                momentum_5 > 1.5 and
                rsi < 75 and
                volume_ratio > 1.1
            )
            
            oversold_buy = (
                rsi < 35 and
                momentum_5 > 0.5 and
                volume_ratio > 1.2
            )
            
            # Enhanced HOLD patterns (simplified)
            consolidation = (
                abs(momentum_5) < 0.5 and
                48 <= rsi <= 52 and
                volatility < 1.5 and
                0.9 <= volume_ratio <= 1.1
            )
            
            neutral_zone = (
                abs(momentum_5) < 1.0 and
                45 <= rsi <= 55 and
                volatility < 2.0 and
                0.8 <= volume_ratio <= 1.2
            )
            
            # Enhanced SELL patterns (simplified)
            strong_sell = (
                momentum_5 < -2.0 and
                rsi > 30 and
                current_price < sma_20 and
                volume_ratio > 1.1
            )
            
            # Decision logic
            if strong_buy or moderate_buy or oversold_buy:
                return 'BUY'
            elif consolidation or neutral_zone:
                return 'HOLD'
            elif strong_sell:
                return 'SELL'
            else:
                # Default simple logic
                if momentum_5 > 1.5 and rsi < 70:
                    return 'BUY'
                elif momentum_5 < -1.5 and rsi > 30:
                    return 'SELL'
                else:
                    return 'HOLD'
                    
        except Exception as e:
            print(f"    Error in prediction logic: {e}")
            return 'HOLD'  # Safe default
    
    def get_actual_outcome_simple(self, data, index, timeframe=7):
        """Get actual outcome with error handling"""
        try:
            if index + timeframe >= len(data):
                return None
            
            current_price = data['Close'].iloc[index]
            future_price = data['Close'].iloc[index + timeframe]
            change = (future_price - current_price) / current_price * 100
            
            # Simple thresholds
            if change > 1.0:
                return 'BUY', change
            elif change < -1.0:
                return 'SELL', change
            else:
                return 'HOLD', change
                
        except Exception as e:
            return None
    
    def run_robust_test(self):
        print("🧪 Robust Accuracy Test - Wide Variety of Stocks")
        print("=" * 55)
        print(f"Testing enhanced systems across {len(self.test_stocks)} diverse stocks")
        print("=" * 55)
        
        stocks_tested = 0
        stocks_failed = 0
        
        for i, symbol in enumerate(self.test_stocks, 1):
            print(f"\n[{i}/{len(self.test_stocks)}] 📊 Testing {symbol}...")
            
            data = self.get_stock_data_safe(symbol)
            if data is None:
                print(f"  ❌ Could not get data for {symbol}")
                stocks_failed += 1
                continue
            
            stocks_tested += 1
            tests_for_symbol = 0
            correct_for_symbol = 0
            
            # Test multiple points (every 2 days)
            for j in range(10, len(data) - 8, 2):
                historical_data = data.iloc[:j+1]
                indicators = self.calculate_simple_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                prediction = self.enhanced_prediction_simple(indicators)
                actual_result = self.get_actual_outcome_simple(data, j)
                
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                tests_for_symbol += 1
                
                is_correct = prediction == actual_direction
                if is_correct:
                    correct_for_symbol += 1
                
                # Record results
                self.results['total_tests'] += 1
                self.results['by_type'][prediction]['total'] += 1
                
                if is_correct:
                    self.results['correct_predictions'] += 1
                    self.results['by_type'][prediction]['correct'] += 1
                
                self.results['predictions'].append({
                    'symbol': symbol,
                    'predicted': prediction,
                    'actual': actual_direction,
                    'actual_change': actual_change,
                    'correct': is_correct
                })
            
            if tests_for_symbol > 0:
                symbol_accuracy = (correct_for_symbol / tests_for_symbol) * 100
                print(f"  ✅ {tests_for_symbol} tests, {symbol_accuracy:.1f}% accuracy")
            else:
                print(f"  ⚠️ No valid tests for {symbol}")
        
        print(f"\n✅ Testing completed!")
        print(f"   Stocks tested: {stocks_tested}")
        print(f"   Stocks failed: {stocks_failed}")
        print(f"   Total tests: {self.results['total_tests']}")
        
        self.print_robust_results()
    
    def print_robust_results(self):
        print("\n" + "=" * 55)
        print("🎯 ROBUST ACCURACY TEST RESULTS")
        print("=" * 55)
        
        if self.results['total_tests'] == 0:
            print("❌ No valid tests completed")
            return
        
        overall_accuracy = (self.results['correct_predictions'] / self.results['total_tests']) * 100
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Correct Predictions: {self.results['correct_predictions']}")
        print(f"Overall Accuracy: {overall_accuracy:.1f}%")
        
        # Accuracy by prediction type
        print(f"\n📈 ACCURACY BY PREDICTION TYPE:")
        for pred_type in ['BUY', 'SELL', 'HOLD']:
            stats = self.results['by_type'][pred_type]
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                target = 80
                status = '✅' if accuracy >= target else '❌'
                print(f"  {pred_type}: {status} {accuracy:.1f}% ({stats['correct']}/{stats['total']}) [Target: {target}%]")
            else:
                print(f"  {pred_type}: No predictions made")
        
        # Distribution analysis
        print(f"\n📊 PREDICTION DISTRIBUTION:")
        total = self.results['total_tests']
        for pred_type in ['BUY', 'SELL', 'HOLD']:
            count = self.results['by_type'][pred_type]['total']
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {pred_type}: {count} predictions ({percentage:.1f}%)")
        
        # Enhanced system assessment
        print(f"\n🎯 ENHANCED SYSTEM ASSESSMENT:")
        targets_achieved = 0
        
        for pred_type in ['BUY', 'SELL', 'HOLD']:
            stats = self.results['by_type'][pred_type]
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                if accuracy >= 80:
                    targets_achieved += 1
                    print(f"✅ {pred_type}: {accuracy:.1f}% >= 80% TARGET ACHIEVED!")
                elif accuracy >= 70:
                    print(f"⚠️ {pred_type}: {accuracy:.1f}% (Close to 80% target)")
                elif accuracy >= 60:
                    print(f"🔶 {pred_type}: {accuracy:.1f}% (Moderate performance)")
                else:
                    print(f"❌ {pred_type}: {accuracy:.1f}% (Needs improvement)")
            else:
                print(f"⚪ {pred_type}: No predictions made")
        
        print(f"\nTargets achieved (80%+): {targets_achieved}/3")
        
        # Final assessment
        if targets_achieved >= 2 and overall_accuracy >= 75:
            print(f"\n🎉 EXCELLENT PERFORMANCE!")
            print(f"Enhanced systems achieving high accuracy across diverse stocks")
            print(f"Ready for production use")
        elif targets_achieved >= 1 and overall_accuracy >= 65:
            print(f"\n🎊 GOOD PERFORMANCE!")
            print(f"Enhanced systems showing strong improvement")
            print(f"Continue optimization for better results")
        elif overall_accuracy >= 55:
            print(f"\n⚠️ MODERATE PERFORMANCE")
            print(f"Enhanced systems working but need refinement")
        else:
            print(f"\n❌ NEEDS IMPROVEMENT")
            print(f"Enhanced systems require further development")
        
        # Sample results
        print(f"\n📋 SAMPLE RESULTS:")
        sample_size = min(15, len(self.results['predictions']))
        for i, result in enumerate(self.results['predictions'][:sample_size]):
            status = "✅" if result['correct'] else "❌"
            print(f"  {status} {result['symbol']}: {result['predicted']} → {result['actual']} ({result['actual_change']:+.1f}%)")

if __name__ == "__main__":
    tester = RobustAccuracyTest()
    tester.run_robust_test()
