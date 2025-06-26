#!/usr/bin/env python3
"""
Ultra-selective prediction model - only makes predictions when extremely confident
Target: 80%+ accuracy by being highly selective
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class UltraSelectivePredictor:
    def __init__(self):
        # Focus on most predictable stocks
        self.test_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA', 'HD',
            'SPY', 'QQQ', 'VTI', 'XLK', 'XLF'
        ]
        
        self.results = {
            'baseline_model': {'correct': 0, 'total': 0, 'predictions': []},
            'ultra_selective_model': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        self.detailed_results = {
            'baseline_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}},
            'ultra_selective_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}}
        }
    
    def get_historical_data(self, symbol, days_back=90):
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 60 else None
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_indicators(self, data):
        if len(data) < 30:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Moving averages
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
            
            ema_12 = data['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['Close'].ewm(span=26).mean().iloc[-1]
            
            # Volume analysis
            avg_volume_5 = data['Volume'].rolling(5).mean().iloc[-1]
            avg_volume_10 = data['Volume'].rolling(10).mean().iloc[-1]
            avg_volume_20 = data['Volume'].rolling(20).mean().iloc[-1]
            
            # Momentum
            momentum_1 = (current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100 if len(data) >= 2 else 0
            momentum_3 = (current_price - data['Close'].iloc[-4]) / data['Close'].iloc[-4] * 100 if len(data) >= 4 else 0
            momentum_5 = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100 if len(data) >= 6 else 0
            momentum_10 = (current_price - data['Close'].iloc[-11]) / data['Close'].iloc[-11] * 100 if len(data) >= 11 else 0
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
            
            # MACD
            macd = ema_12 - ema_26
            macd_signal = data['Close'].ewm(span=9).mean().iloc[-1]
            macd_histogram = macd - macd_signal
            
            # Bollinger Bands
            bb_std = data['Close'].rolling(20).std().iloc[-1]
            bollinger_upper = sma_20 + (bb_std * 2)
            bollinger_lower = sma_20 - (bb_std * 2)
            
            # Volatility
            volatility = data['Close'].pct_change().rolling(10).std().iloc[-1] * 100
            
            # Support/Resistance
            high_10 = data['High'].rolling(10).max().iloc[-1]
            low_10 = data['Low'].rolling(10).min().iloc[-1]
            high_20 = data['High'].rolling(20).max().iloc[-1]
            low_20 = data['Low'].rolling(20).min().iloc[-1]
            
            return {
                'current_price': current_price,
                'volume': volume,
                'sma_5': sma_5, 'sma_10': sma_10, 'sma_20': sma_20, 'sma_50': sma_50,
                'avg_volume_5': avg_volume_5, 'avg_volume_10': avg_volume_10, 'avg_volume_20': avg_volume_20,
                'momentum_1': momentum_1, 'momentum_3': momentum_3, 'momentum_5': momentum_5, 'momentum_10': momentum_10,
                'rsi': rsi, 'macd': macd, 'macd_histogram': macd_histogram,
                'bollinger_upper': bollinger_upper, 'bollinger_lower': bollinger_lower,
                'volatility': volatility, 'high_10': high_10, 'low_10': low_10, 'high_20': high_20, 'low_20': low_20,
                # Derived
                'volume_ratio_5': volume / avg_volume_5 if avg_volume_5 > 0 else 1,
                'volume_ratio_10': volume / avg_volume_10 if avg_volume_10 > 0 else 1,
                'volume_ratio_20': volume / avg_volume_20 if avg_volume_20 > 0 else 1,
                'price_vs_sma20': (current_price - sma_20) / sma_20 * 100,
                'price_vs_sma50': (current_price - sma_50) / sma_50 * 100,
                'bb_position': (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower) if bollinger_upper != bollinger_lower else 0.5,
                'distance_from_high_10': (high_10 - current_price) / high_10 * 100,
                'distance_from_low_10': (current_price - low_10) / low_10 * 100,
                'distance_from_high_20': (high_20 - current_price) / high_20 * 100,
                'distance_from_low_20': (current_price - low_20) / low_20 * 100,
                'ma_alignment_bullish': (sma_5 > sma_10 > sma_20 > sma_50),
                'ma_alignment_bearish': (sma_5 < sma_10 < sma_20 < sma_50),
                'momentum_consistency': 1 if (momentum_1 > 0 and momentum_3 > 0 and momentum_5 > 0) or (momentum_1 < 0 and momentum_3 < 0 and momentum_5 < 0) else 0
            }
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None
    
    def baseline_model(self, indicators):
        """Simple baseline model"""
        if (indicators['momentum_5'] > 2 and indicators['rsi'] < 70):
            return 'BUY'
        elif (indicators['momentum_5'] < -2 and indicators['rsi'] > 30):
            return 'SELL'
        else:
            return 'HOLD'
    
    def ultra_selective_model(self, indicators):
        """Ultra-selective model - only predicts when extremely confident"""
        
        # === ULTRA HIGH CONFIDENCE BUY PATTERNS ===
        
        # Pattern 1: Perfect momentum alignment with volume confirmation
        if (indicators['momentum_1'] > 1.5 and 
            indicators['momentum_3'] > 2.0 and 
            indicators['momentum_5'] > 2.5 and
            indicators['momentum_10'] > 3.0 and
            indicators['momentum_consistency'] == 1 and
            indicators['volume_ratio_5'] > 1.5 and
            indicators['volume_ratio_10'] > 1.3 and
            indicators['ma_alignment_bullish'] and
            40 <= indicators['rsi'] <= 65 and
            indicators['volatility'] < 3 and
            indicators['distance_from_high_10'] > 2):
            return 'BUY'
        
        # Pattern 2: Oversold bounce with strong confirmation
        if (indicators['rsi'] < 30 and
            indicators['momentum_1'] > 1.0 and
            indicators['momentum_3'] > 0.5 and
            indicators['distance_from_low_10'] < 3 and
            indicators['volume_ratio_5'] > 1.4 and
            indicators['current_price'] > indicators['sma_5'] and
            indicators['macd_histogram'] > 0):
            return 'BUY'
        
        # Pattern 3: Breakout with volume
        if (indicators['current_price'] > indicators['high_10'] and
            indicators['volume_ratio_5'] > 2.0 and
            indicators['momentum_5'] > 3.0 and
            indicators['rsi'] < 75 and
            indicators['ma_alignment_bullish'] and
            indicators['volatility'] < 4):
            return 'BUY'
        
        # === ULTRA HIGH CONFIDENCE SELL PATTERNS ===
        
        # Pattern 1: Perfect bearish momentum with volume
        if (indicators['momentum_1'] < -1.5 and
            indicators['momentum_3'] < -2.0 and
            indicators['momentum_5'] < -2.5 and
            indicators['momentum_10'] < -3.0 and
            indicators['momentum_consistency'] == 1 and
            indicators['volume_ratio_5'] > 1.4 and
            indicators['volume_ratio_10'] > 1.2 and
            indicators['ma_alignment_bearish'] and
            35 <= indicators['rsi'] <= 70 and
            indicators['distance_from_low_10'] > 5):
            return 'SELL'
        
        # Pattern 2: Overbought reversal with confirmation
        if (indicators['rsi'] > 75 and
            indicators['momentum_1'] < -1.0 and
            indicators['momentum_3'] < -0.5 and
            indicators['distance_from_high_10'] < 2 and
            indicators['volume_ratio_5'] > 1.3 and
            indicators['current_price'] < indicators['sma_5'] and
            indicators['macd_histogram'] < 0):
            return 'SELL'
        
        # Pattern 3: Breakdown with volume
        if (indicators['current_price'] < indicators['low_10'] and
            indicators['volume_ratio_5'] > 1.8 and
            indicators['momentum_5'] < -3.0 and
            indicators['rsi'] > 25 and
            indicators['ma_alignment_bearish']):
            return 'SELL'
        
        # === ULTRA HIGH CONFIDENCE HOLD PATTERNS ===
        
        # Pattern 1: Perfect consolidation
        if (abs(indicators['momentum_5']) < 0.5 and
            abs(indicators['momentum_10']) < 1.0 and
            indicators['volatility'] < 1.5 and
            48 <= indicators['rsi'] <= 52 and
            abs(indicators['price_vs_sma20']) < 1 and
            0.9 <= indicators['volume_ratio_10'] <= 1.1 and
            0.4 <= indicators['bb_position'] <= 0.6 and
            3 <= indicators['distance_from_high_10'] <= 7 and
            3 <= indicators['distance_from_low_10'] <= 7):
            return 'HOLD'
        
        # Pattern 2: Tight range with low volatility
        if (indicators['volatility'] < 1.0 and
            abs(indicators['momentum_3']) < 0.3 and
            45 <= indicators['rsi'] <= 55 and
            abs(indicators['price_vs_sma20']) < 0.5 and
            0.95 <= indicators['volume_ratio_5'] <= 1.05):
            return 'HOLD'
        
        # === NO PREDICTION IF NOT EXTREMELY CONFIDENT ===
        return None  # No prediction when not confident
    
    def get_actual_outcome(self, data, index, timeframe=15):
        """Get actual outcome"""
        if index + timeframe >= len(data):
            return None
        
        current_price = data['Close'].iloc[index]
        future_price = data['Close'].iloc[index + timeframe]
        change = (future_price - current_price) / current_price * 100
        
        # Conservative thresholds
        if change > 2.5:
            return 'BUY', change
        elif change < -2.5:
            return 'SELL', change
        else:
            return 'HOLD', change
    
    def run_ultra_selective_test(self):
        print("🎯 Ultra-Selective Model Test - Only Extremely Confident Predictions")
        print("=" * 70)
        print("Goal: Achieve 80%+ accuracy by being highly selective")
        print("=" * 70)
        
        symbols_tested = 0
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            data = self.get_historical_data(symbol)
            if data is None:
                continue
            
            symbols_tested += 1
            tests_for_symbol = 0
            predictions_made = 0
            
            # Test every day for maximum coverage
            for i in range(50, len(data) - 20, 1):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                baseline_pred = self.baseline_model(indicators)
                ultra_pred = self.ultra_selective_model(indicators)
                
                actual_result = self.get_actual_outcome(data, i)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                tests_for_symbol += 1
                
                # Record baseline results (always makes prediction)
                self.results['baseline_model']['total'] += 1
                self.detailed_results['baseline_model'][baseline_pred]['total'] += 1
                
                if baseline_pred == actual_direction:
                    self.results['baseline_model']['correct'] += 1
                    self.detailed_results['baseline_model'][baseline_pred]['correct'] += 1
                
                self.results['baseline_model']['predictions'].append({
                    'symbol': symbol,
                    'predicted': baseline_pred,
                    'actual': actual_direction,
                    'actual_change': actual_change,
                    'correct': baseline_pred == actual_direction
                })
                
                # Record ultra-selective results (only when prediction made)
                if ultra_pred is not None:
                    predictions_made += 1
                    self.results['ultra_selective_model']['total'] += 1
                    self.detailed_results['ultra_selective_model'][ultra_pred]['total'] += 1
                    
                    if ultra_pred == actual_direction:
                        self.results['ultra_selective_model']['correct'] += 1
                        self.detailed_results['ultra_selective_model'][ultra_pred]['correct'] += 1
                    
                    self.results['ultra_selective_model']['predictions'].append({
                        'symbol': symbol,
                        'predicted': ultra_pred,
                        'actual': actual_direction,
                        'actual_change': actual_change,
                        'correct': ultra_pred == actual_direction
                    })
            
            print(f"  Completed {tests_for_symbol} tests, Ultra-selective made {predictions_made} predictions")
        
        print(f"\n✅ Tested {symbols_tested} symbols")
        self.print_ultra_selective_results()
    
    def print_ultra_selective_results(self):
        print("\n" + "=" * 70)
        print("🎯 ULTRA-SELECTIVE MODEL RESULTS")
        print("=" * 70)
        
        baseline_accuracy = (self.results['baseline_model']['correct'] / 
                           self.results['baseline_model']['total'] * 100) if self.results['baseline_model']['total'] > 0 else 0
        
        ultra_accuracy = (self.results['ultra_selective_model']['correct'] / 
                        self.results['ultra_selective_model']['total'] * 100) if self.results['ultra_selective_model']['total'] > 0 else 0
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"Baseline Model:      {baseline_accuracy:.1f}% ({self.results['baseline_model']['correct']}/{self.results['baseline_model']['total']})")
        print(f"Ultra-Selective:     {ultra_accuracy:.1f}% ({self.results['ultra_selective_model']['correct']}/{self.results['ultra_selective_model']['total']})")
        
        selectivity = (self.results['ultra_selective_model']['total'] / self.results['baseline_model']['total'] * 100) if self.results['baseline_model']['total'] > 0 else 0
        print(f"Selectivity:         {selectivity:.1f}% (made {selectivity:.1f}% of possible predictions)")
        
        improvement = ultra_accuracy - baseline_accuracy
        print(f"Accuracy Improvement: {improvement:+.1f}%")
        
        # Detailed breakdown
        print(f"\n📈 DETAILED ACCURACY BY PREDICTION TYPE:")
        
        for model_name in ['baseline_model', 'ultra_selective_model']:
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
        
        # Success analysis
        print(f"\n🎯 80% ACCURACY TARGET ANALYSIS:")
        ultra_results = self.detailed_results['ultra_selective_model']
        
        targets_achieved = 0
        for pred_type in ['BUY', 'SELL', 'HOLD']:
            stats = ultra_results[pred_type]
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                if accuracy >= 80:
                    targets_achieved += 1
                    print(f"✅ {pred_type} target achieved: {accuracy:.1f}% >= 80%")
                else:
                    print(f"❌ {pred_type} target missed: {accuracy:.1f}% < 80%")
            else:
                print(f"⚪ {pred_type}: No predictions made")
        
        print(f"\nTargets achieved: {targets_achieved}/3")
        print(f"Overall target (>80%): {'✅' if ultra_accuracy >= 80 else '❌'} {ultra_accuracy:.1f}%")
        
        if ultra_accuracy >= 80 and targets_achieved >= 2:
            print(f"\n🚀 SUCCESS! ULTRA-SELECTIVE MODEL ACHIEVED 80%+ ACCURACY!")
            print(f"Ready for production implementation")
        elif ultra_accuracy >= 75:
            print(f"\n⚠️ CLOSE! Need {80 - ultra_accuracy:.1f}% more accuracy")
            print(f"Consider making selection criteria even more strict")
        else:
            print(f"\n❌ NEED MORE WORK: {80 - ultra_accuracy:.1f}% accuracy gap")
            print(f"Need to refine selection patterns")
        
        # Show some successful predictions
        if self.results['ultra_selective_model']['predictions']:
            print(f"\n📋 SAMPLE ULTRA-SELECTIVE PREDICTIONS:")
            correct_preds = [p for p in self.results['ultra_selective_model']['predictions'] if p['correct']]
            for pred in correct_preds[:10]:  # Show first 10 correct predictions
                print(f"  ✅ {pred['symbol']}: {pred['predicted']} → {pred['actual']} ({pred['actual_change']:+.1f}%)")

if __name__ == "__main__":
    predictor = UltraSelectivePredictor()
    predictor.run_ultra_selective_test()
