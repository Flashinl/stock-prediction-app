#!/usr/bin/env python3
"""
Test the enhanced HOLD and BUY systems implemented in app.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path to import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class EnhancedSystemTester:
    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ', 'NVDA']
        
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
    
    def get_data(self, symbol, days_back=40):
        """Get stock data safely"""
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 25 else None
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def get_prediction_from_app(self, symbol):
        """Get prediction from the enhanced app.py"""
        try:
            from app import StockPredictor
            
            predictor = StockPredictor()
            result = predictor.predict_stock_movement(symbol)
            
            if result and 'prediction' in result:
                return result['prediction'], result.get('confidence', 50), result.get('reasoning', '')
            else:
                return None, None, None
        except Exception as e:
            print(f"Error getting prediction for {symbol}: {e}")
            return None, None, None
    
    def get_actual_outcome(self, symbol, days_forward=10):
        """Get actual price movement over the next 10 days"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get data from 20 days ago to now
            end_date = datetime.now()
            start_date = end_date - timedelta(days=20)
            data = stock.history(start=start_date, end=end_date)
            
            if len(data) < days_forward + 5:
                return None, None
            
            # Use price from 10 days ago vs current price
            past_price = data['Close'].iloc[-(days_forward+1)]
            current_price = data['Close'].iloc[-1]
            
            change = (current_price - past_price) / past_price * 100
            
            # Classification thresholds (matching our testing)
            if change > 1.2:
                return 'BUY', change
            elif change < -1.2:
                return 'SELL', change
            else:
                return 'HOLD', change
        except Exception as e:
            print(f"Error getting actual outcome for {symbol}: {e}")
            return None, None
    
    def test_enhanced_systems(self):
        print("🧪 Testing Enhanced HOLD and BUY Systems")
        print("=" * 50)
        print("Testing the implemented enhanced patterns from comprehensive testing")
        print("=" * 50)
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            # Get prediction from enhanced app
            predicted, confidence, reasoning = self.get_prediction_from_app(symbol)
            if predicted is None:
                print(f"  ❌ Could not get prediction for {symbol}")
                continue
            
            # Get actual outcome
            actual, actual_change = self.get_actual_outcome(symbol)
            if actual is None:
                print(f"  ❌ Could not get actual outcome for {symbol}")
                continue
            
            # Record results
            self.results['total_tests'] += 1
            self.results['by_type'][predicted]['total'] += 1
            
            is_correct = predicted == actual
            if is_correct:
                self.results['correct_predictions'] += 1
                self.results['by_type'][predicted]['correct'] += 1
            
            self.results['predictions'].append({
                'symbol': symbol,
                'predicted': predicted,
                'actual': actual,
                'actual_change': actual_change,
                'confidence': confidence,
                'reasoning': reasoning,
                'correct': is_correct
            })
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} Predicted: {predicted}, Actual: {actual} ({actual_change:.1f}%)")
            print(f"     Confidence: {confidence}%, Reasoning: {reasoning[:60]}...")
        
        self.print_test_results()
    
    def print_test_results(self):
        print("\n" + "=" * 50)
        print("🎯 ENHANCED SYSTEM TEST RESULTS")
        print("=" * 50)
        
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
        
        # Enhanced system analysis
        print(f"\n🔍 ENHANCED SYSTEM ANALYSIS:")
        
        # Analyze BUY predictions
        buy_preds = [p for p in self.results['predictions'] if p['predicted'] == 'BUY']
        if buy_preds:
            buy_correct = [p for p in buy_preds if p['correct']]
            buy_accuracy = len(buy_correct) / len(buy_preds) * 100
            avg_confidence = np.mean([p['confidence'] for p in buy_preds])
            print(f"  🟢 BUY System: {buy_accuracy:.1f}% accuracy, {avg_confidence:.1f}% avg confidence")
            
            # Show reasoning patterns for correct BUY predictions
            if buy_correct:
                print(f"     Successful BUY patterns:")
                for pred in buy_correct[:3]:  # Show first 3
                    print(f"       • {pred['symbol']}: {pred['reasoning'][:50]}...")
        
        # Analyze HOLD predictions
        hold_preds = [p for p in self.results['predictions'] if p['predicted'] == 'HOLD']
        if hold_preds:
            hold_correct = [p for p in hold_preds if p['correct']]
            hold_accuracy = len(hold_correct) / len(hold_preds) * 100
            avg_confidence = np.mean([p['confidence'] for p in hold_preds])
            print(f"  🟡 HOLD System: {hold_accuracy:.1f}% accuracy, {avg_confidence:.1f}% avg confidence")
            
            # Show reasoning patterns for correct HOLD predictions
            if hold_correct:
                print(f"     Successful HOLD patterns:")
                for pred in hold_correct[:3]:  # Show first 3
                    print(f"       • {pred['symbol']}: {pred['reasoning'][:50]}...")
        
        # Analyze SELL predictions
        sell_preds = [p for p in self.results['predictions'] if p['predicted'] == 'SELL']
        if sell_preds:
            sell_correct = [p for p in sell_preds if p['correct']]
            sell_accuracy = len(sell_correct) / len(sell_preds) * 100
            avg_confidence = np.mean([p['confidence'] for p in sell_preds])
            print(f"  🔴 SELL System: {sell_accuracy:.1f}% accuracy, {avg_confidence:.1f}% avg confidence")
        
        # Success assessment
        print(f"\n🎯 ENHANCED SYSTEM ASSESSMENT:")
        targets_met = 0
        
        for pred_type in ['BUY', 'SELL', 'HOLD']:
            stats = self.results['by_type'][pred_type]
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                if accuracy >= 80:
                    targets_met += 1
                    print(f"✅ {pred_type} system achieved 80%+ target: {accuracy:.1f}%")
                elif accuracy >= 70:
                    print(f"⚠️ {pred_type} system close to target: {accuracy:.1f}% (need {80-accuracy:.1f}% more)")
                else:
                    print(f"❌ {pred_type} system needs improvement: {accuracy:.1f}% (need {80-accuracy:.1f}% more)")
        
        print(f"\nTargets achieved: {targets_met}/3")
        
        if targets_met >= 2 and overall_accuracy >= 70:
            print(f"\n🎊 EXCELLENT PROGRESS!")
            print(f"Enhanced systems showing strong improvement")
            print(f"Ready for further optimization")
        elif overall_accuracy >= 60:
            print(f"\n⚠️ GOOD PROGRESS")
            print(f"Enhanced systems working, need fine-tuning")
        else:
            print(f"\n❌ NEEDS MORE WORK")
            print(f"Enhanced systems require further development")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for result in self.results['predictions']:
            status = "✅" if result['correct'] else "❌"
            print(f"  {status} {result['symbol']}: {result['predicted']} → {result['actual']} ({result['actual_change']:+.1f}%) [{result['confidence']}%]")

if __name__ == "__main__":
    tester = EnhancedSystemTester()
    tester.test_enhanced_systems()
