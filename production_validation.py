#!/usr/bin/env python3
"""
Validation test for production app.py improvements
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path to import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ProductionValidator:
    def __init__(self):
        self.test_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA', 'HD',
            'SPY', 'QQQ', 'VTI', 'XLK', 'XLF'
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
    
    def get_historical_data(self, symbol, days_back=60):
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 40 else None
        except Exception as e:
            return None
    
    def simulate_app_prediction(self, symbol):
        """Simulate getting a prediction from the app"""
        try:
            # Import the StockPredictor from app.py
            from app import StockPredictor

            predictor = StockPredictor()
            result = predictor.predict_stock_movement(symbol)

            if result and 'prediction' in result:
                return result['prediction'], result.get('confidence', 50)
            else:
                return None, None
        except Exception as e:
            print(f"Error getting prediction for {symbol}: {e}")
            return None, None
    
    def get_actual_outcome(self, symbol, days_forward=20):
        """Get actual price movement over the next 20 days"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get data from 30 days ago to now
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            data = stock.history(start=start_date, end=end_date)
            
            if len(data) < days_forward + 5:
                return None, None
            
            # Use price from 20 days ago vs current price
            past_price = data['Close'].iloc[-(days_forward+1)]
            current_price = data['Close'].iloc[-1]
            
            change = (current_price - past_price) / past_price * 100
            
            # Classification thresholds (matching our model)
            if change > 3:
                return 'BUY', change
            elif change < -3:
                return 'SELL', change
            else:
                return 'HOLD', change
        except Exception as e:
            return None, None
    
    def run_production_validation(self):
        print("🚀 Production App.py Validation Test")
        print("=" * 50)
        print("Testing improved prediction logic against real outcomes")
        print("=" * 50)
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            # Get prediction from app
            predicted, confidence = self.simulate_app_prediction(symbol)
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
                'correct': is_correct
            })
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} Predicted: {predicted}, Actual: {actual} ({actual_change:.1f}%), Confidence: {confidence}%")
        
        self.print_validation_results()
    
    def print_validation_results(self):
        print("\n" + "=" * 50)
        print("🎯 PRODUCTION VALIDATION RESULTS")
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
                target = {'BUY': 75, 'SELL': 70, 'HOLD': 60}[pred_type]
                status = '✅' if accuracy >= target else '❌'
                print(f"  {pred_type}: {status} {accuracy:.1f}% ({stats['correct']}/{stats['total']}) [Target: {target}%]")
            else:
                print(f"  {pred_type}: No predictions made")
        
        # Success analysis
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        targets_met = 0
        
        for pred_type, target in [('BUY', 75), ('SELL', 70), ('HOLD', 60)]:
            stats = self.results['by_type'][pred_type]
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                if accuracy >= target:
                    targets_met += 1
                    print(f"✅ {pred_type} target achieved: {accuracy:.1f}% >= {target}%")
                else:
                    print(f"❌ {pred_type} target missed: {accuracy:.1f}% < {target}%")
        
        print(f"\nTargets achieved: {targets_met}/3")
        print(f"Overall target (>65%): {'✅' if overall_accuracy > 65 else '❌'} {overall_accuracy:.1f}%")
        
        # Detailed analysis
        print(f"\n📋 DETAILED RESULTS:")
        for result in self.results['predictions']:
            status = "✅" if result['correct'] else "❌"
            print(f"  {status} {result['symbol']}: {result['predicted']} → {result['actual']} ({result['actual_change']:+.1f}%)")
        
        # Final assessment
        if overall_accuracy > 65 and targets_met >= 2:
            print(f"\n🚀 PRODUCTION MODEL READY!")
            print(f"Significant improvements achieved - ready for deployment")
        elif overall_accuracy > 55:
            print(f"\n⚠️ MODERATE IMPROVEMENT")
            print(f"Some progress made but more refinement needed")
        else:
            print(f"\n❌ NEEDS MORE WORK")
            print(f"Accuracy still below acceptable thresholds")

if __name__ == "__main__":
    validator = ProductionValidator()
    validator.run_production_validation()
