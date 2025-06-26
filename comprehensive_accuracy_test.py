#!/usr/bin/env python3
"""
Comprehensive accuracy test using a wide variety of stocks
Tests the enhanced HOLD and BUY systems across different market sectors and cap sizes
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

class ComprehensiveAccuracyTest:
    def __init__(self):
        # Wide variety of stocks across different sectors and market caps
        self.test_stocks = {
            # Large Cap Tech
            'MEGA_TECH': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX'],
            
            # Large Cap Traditional
            'LARGE_CAP': ['JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA', 'HD', 'UNH', 'CVX'],
            
            # Mid Cap Growth
            'MID_CAP': ['PLTR', 'RBLX', 'SNOW', 'CRWD', 'ZM', 'ROKU', 'SQ', 'SHOP', 'TWLO', 'OKTA'],
            
            # Financial Sector
            'FINANCIAL': ['BAC', 'GS', 'MS', 'C', 'WFC', 'AXP', 'BLK', 'SCHW', 'USB', 'PNC'],
            
            # Healthcare & Biotech
            'HEALTHCARE': ['MRNA', 'PFE', 'ABBV', 'TMO', 'DHR', 'ABT', 'BMY', 'GILD', 'AMGN', 'BIIB'],
            
            # Energy & Commodities
            'ENERGY': ['XOM', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY', 'DVN'],
            
            # Consumer & Retail
            'CONSUMER': ['AMZN', 'TGT', 'COST', 'NKE', 'SBUX', 'MCD', 'DIS', 'LOW', 'TJX', 'BKNG'],
            
            # Industrial & Manufacturing
            'INDUSTRIAL': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'EMR'],
            
            # ETFs for Market Representation
            'ETFS': ['SPY', 'QQQ', 'IWM', 'VTI', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY'],
            
            # Volatile/Speculative
            'VOLATILE': ['GME', 'AMC', 'COIN', 'HOOD', 'RIVN', 'LCID', 'SOFI', 'WISH', 'CLOV', 'BB']
        }
        
        # Flatten all stocks into one list
        self.all_stocks = []
        for category, stocks in self.test_stocks.items():
            self.all_stocks.extend(stocks)
        
        # Remove duplicates
        self.all_stocks = list(set(self.all_stocks))
        
        self.results = {
            'total_tests': 0,
            'correct_predictions': 0,
            'predictions': [],
            'by_type': {
                'BUY': {'correct': 0, 'total': 0},
                'SELL': {'correct': 0, 'total': 0},
                'HOLD': {'correct': 0, 'total': 0}
            },
            'by_category': {}
        }
        
        # Initialize category results
        for category in self.test_stocks.keys():
            self.results['by_category'][category] = {
                'total': 0, 'correct': 0,
                'BUY': {'correct': 0, 'total': 0},
                'SELL': {'correct': 0, 'total': 0},
                'HOLD': {'correct': 0, 'total': 0}
            }
    
    def get_stock_data(self, symbol, days_back=35):
        """Get stock data with retry logic"""
        for attempt in range(3):
            try:
                stock = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                data = stock.history(start=start_date, end=end_date)
                
                if len(data) > 20:
                    return data
                else:
                    print(f"    Insufficient data for {symbol} (attempt {attempt + 1})")
                    
            except Exception as e:
                print(f"    Error fetching {symbol} (attempt {attempt + 1}): {e}")
                
            if attempt < 2:
                time.sleep(1)  # Wait before retry
        
        return None
    
    def calculate_enhanced_indicators(self, data):
        """Calculate indicators for enhanced logic"""
        if len(data) < 15:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Moving averages
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(min(20, len(data))).mean().iloc[-1]
            
            # Volume analysis
            avg_volume_5 = data['Volume'].rolling(5).mean().iloc[-1]
            avg_volume_10 = data['Volume'].rolling(10).mean().iloc[-1]
            
            # Momentum calculations
            momentum_2 = (current_price - data['Close'].iloc[-3]) / data['Close'].iloc[-3] * 100 if len(data) >= 3 else 0
            momentum_5 = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100 if len(data) >= 6 else 0
            momentum_10 = (current_price - data['Close'].iloc[-11]) / data['Close'].iloc[-11] * 100 if len(data) >= 11 else 0
            
            # RSI calculation
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
            
            # Volatility
            volatility = data['Close'].pct_change().rolling(5).std().iloc[-1] * 100
            
            # Market regime detection
            if len(data) >= 15:
                price_15_ago = data['Close'].iloc[-16]
                trend_strength = (current_price - price_15_ago) / price_15_ago * 100
                
                if trend_strength > 3:
                    regime = 'BULL'
                elif trend_strength < -3:
                    regime = 'BEAR'
                else:
                    regime = 'SIDEWAYS'
            else:
                regime = 'SIDEWAYS'
            
            return {
                'current_price': current_price,
                'volume': volume,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'avg_volume_5': avg_volume_5,
                'avg_volume_10': avg_volume_10,
                'momentum_2': momentum_2,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'rsi': rsi,
                'volatility': volatility,
                'regime': regime,
                'volume_ratio': volume / avg_volume_10 if avg_volume_10 > 0 else 1,
                'price_momentum': momentum_5,
                'momentum_strength': momentum_5 / 10,  # Normalized
                'price_vs_sma20': (current_price - sma_20) / sma_20 * 100
            }
        except Exception as e:
            print(f"    Error calculating indicators: {e}")
            return None
    
    def enhanced_prediction_logic(self, indicators):
        """Enhanced prediction logic matching app.py implementation"""
        
        # Enhanced BUY patterns
        price_momentum = indicators['price_momentum']
        momentum_strength = indicators['momentum_strength']
        rsi = indicators['rsi']
        current_price = indicators['current_price']
        sma_20 = indicators['sma_20']
        volume_ratio = indicators['volume_ratio']
        volatility = indicators['volatility']
        regime = indicators['regime']
        
        # High confidence BUY patterns
        high_confidence_buy = (
            price_momentum > 1.5 and 
            momentum_strength > 0.3 and
            rsi < 70 and
            current_price > sma_20 and
            volume_ratio > 1.2 and
            volatility < 3.5
        )
        
        strong_momentum_buy = (
            price_momentum > 2.0 and
            rsi < 75 and
            current_price > sma_20 and
            momentum_strength > 0.2 and
            volume_ratio > 1.1
        )
        
        oversold_bounce_buy = (
            rsi < 35 and
            price_momentum > 1.0 and
            volume_ratio > 1.3 and
            current_price > sma_20 and
            momentum_strength > 0.1
        )
        
        breakout_buy = (
            price_momentum > 2.5 and
            volume_ratio > 1.5 and
            rsi < 75 and
            momentum_strength > 0.4
        )
        
        # Enhanced HOLD patterns
        perfect_consolidation = (
            abs(price_momentum) < 0.5 and
            abs(momentum_strength) < 0.1 and
            volatility < 1.5 and
            48 <= rsi <= 52 and
            abs((current_price - sma_20) / sma_20 * 100) < 1 and
            0.9 <= volume_ratio <= 1.1
        )
        
        tight_range_consolidation = (
            volatility < 1.0 and
            abs(price_momentum) < 0.3 and
            45 <= rsi <= 55 and
            abs((current_price - sma_20) / sma_20 * 100) < 0.5 and
            0.95 <= volume_ratio <= 1.05
        )
        
        neutral_momentum_zone = (
            abs(price_momentum) < 1.0 and
            45 <= rsi <= 55 and
            volatility < 2.0 and
            abs(momentum_strength) < 0.15 and
            0.8 <= volume_ratio <= 1.2
        )
        
        sideways_pattern = (
            regime == 'SIDEWAYS' and
            abs(price_momentum) < 1.5 and
            40 <= rsi <= 60 and
            volatility < 2.5 and
            0.85 <= volume_ratio <= 1.15
        )
        
        # Decision logic
        if (high_confidence_buy or strong_momentum_buy or 
            oversold_bounce_buy or breakout_buy):
            return 'BUY'
        elif (perfect_consolidation or tight_range_consolidation or 
              neutral_momentum_zone or sideways_pattern):
            return 'HOLD'
        elif (price_momentum < -2.0 and rsi > 30 and 
              current_price < sma_20 and volume_ratio > 1.1):
            return 'SELL'
        else:
            # Default logic
            if price_momentum > 1.5 and rsi < 70:
                return 'BUY'
            elif price_momentum < -1.5 and rsi > 30:
                return 'SELL'
            else:
                return 'HOLD'
    
    def get_actual_outcome(self, data, index, timeframe=8):
        """Get actual outcome for validation"""
        if index + timeframe >= len(data):
            return None
        
        current_price = data['Close'].iloc[index]
        future_price = data['Close'].iloc[index + timeframe]
        change = (future_price - current_price) / current_price * 100
        
        # Classification thresholds
        if change > 1.2:
            return 'BUY', change
        elif change < -1.2:
            return 'SELL', change
        else:
            return 'HOLD', change
    
    def get_stock_category(self, symbol):
        """Get the category of a stock"""
        for category, stocks in self.test_stocks.items():
            if symbol in stocks:
                return category
        return 'UNKNOWN'
    
    def run_comprehensive_test(self):
        print("🧪 Comprehensive Accuracy Test - Wide Variety of Stocks")
        print("=" * 65)
        print(f"Testing enhanced HOLD and BUY systems across {len(self.all_stocks)} stocks")
        print("Categories: Tech, Traditional, Mid-Cap, Financial, Healthcare, Energy, Consumer, Industrial, ETFs, Volatile")
        print("=" * 65)
        
        # Shuffle stocks for random testing order
        test_stocks = self.all_stocks.copy()
        random.shuffle(test_stocks)
        
        stocks_tested = 0
        stocks_failed = 0
        
        for i, symbol in enumerate(test_stocks, 1):
            print(f"\n[{i}/{len(test_stocks)}] 📊 Testing {symbol}...")
            
            # Get stock category
            category = self.get_stock_category(symbol)
            
            data = self.get_stock_data(symbol)
            if data is None:
                print(f"  ❌ Could not get data for {symbol}")
                stocks_failed += 1
                continue
            
            stocks_tested += 1
            tests_for_symbol = 0
            
            # Test multiple time points for this symbol
            for j in range(15, len(data) - 10, 3):  # Every 3 days
                historical_data = data.iloc[:j+1]
                indicators = self.calculate_enhanced_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                prediction = self.enhanced_prediction_logic(indicators)
                actual_result = self.get_actual_outcome(data, j)
                
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                tests_for_symbol += 1
                
                # Record results
                is_correct = prediction == actual_direction
                
                # Overall results
                self.results['total_tests'] += 1
                self.results['by_type'][prediction]['total'] += 1
                
                if is_correct:
                    self.results['correct_predictions'] += 1
                    self.results['by_type'][prediction]['correct'] += 1
                
                # Category results
                self.results['by_category'][category]['total'] += 1
                self.results['by_category'][category][prediction]['total'] += 1
                
                if is_correct:
                    self.results['by_category'][category]['correct'] += 1
                    self.results['by_category'][category][prediction]['correct'] += 1
                
                self.results['predictions'].append({
                    'symbol': symbol,
                    'category': category,
                    'predicted': prediction,
                    'actual': actual_direction,
                    'actual_change': actual_change,
                    'correct': is_correct
                })
            
            if tests_for_symbol > 0:
                symbol_accuracy = (sum(1 for p in self.results['predictions'] 
                                     if p['symbol'] == symbol and p['correct']) / tests_for_symbol * 100)
                print(f"  ✅ Completed {tests_for_symbol} tests, {symbol_accuracy:.1f}% accuracy")
            else:
                print(f"  ⚠️ No valid tests for {symbol}")
        
        print(f"\n✅ Testing completed!")
        print(f"   Stocks tested: {stocks_tested}")
        print(f"   Stocks failed: {stocks_failed}")
        print(f"   Total tests: {self.results['total_tests']}")
        
        self.print_comprehensive_results()
    
    def print_comprehensive_results(self):
        print("\n" + "=" * 65)
        print("🎯 COMPREHENSIVE ACCURACY TEST RESULTS")
        print("=" * 65)
        
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
        
        # Accuracy by category
        print(f"\n📊 ACCURACY BY STOCK CATEGORY:")
        for category, stats in self.results['by_category'].items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                print(f"\n  {category}:")
                print(f"    Overall: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
                
                for pred_type in ['BUY', 'SELL', 'HOLD']:
                    type_stats = stats[pred_type]
                    if type_stats['total'] > 0:
                        type_accuracy = (type_stats['correct'] / type_stats['total']) * 100
                        print(f"    {pred_type}: {type_accuracy:.1f}% ({type_stats['correct']}/{type_stats['total']})")
        
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
            print(f"Ready for production deployment")
        elif targets_achieved >= 1 and overall_accuracy >= 65:
            print(f"\n🎊 GOOD PERFORMANCE!")
            print(f"Enhanced systems showing strong improvement")
            print(f"Continue optimization for even better results")
        elif overall_accuracy >= 55:
            print(f"\n⚠️ MODERATE PERFORMANCE")
            print(f"Enhanced systems working but need refinement")
            print(f"Focus on improving weaker prediction types")
        else:
            print(f"\n❌ NEEDS SIGNIFICANT IMPROVEMENT")
            print(f"Enhanced systems require major refinement")
        
        # Best and worst performing categories
        category_accuracies = []
        for category, stats in self.results['by_category'].items():
            if stats['total'] > 10:  # Only categories with sufficient data
                accuracy = (stats['correct'] / stats['total']) * 100
                category_accuracies.append((category, accuracy, stats['total']))
        
        if category_accuracies:
            category_accuracies.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🏆 BEST PERFORMING CATEGORIES:")
            for category, accuracy, total in category_accuracies[:3]:
                print(f"  {category}: {accuracy:.1f}% ({total} tests)")
            
            print(f"\n📉 CHALLENGING CATEGORIES:")
            for category, accuracy, total in category_accuracies[-3:]:
                print(f"  {category}: {accuracy:.1f}% ({total} tests)")

if __name__ == "__main__":
    tester = ComprehensiveAccuracyTest()
    tester.run_comprehensive_test()
