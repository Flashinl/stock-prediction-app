#!/usr/bin/env python3
"""
High accuracy model targeting 75%+ BUY, 70%+ SELL, 60%+ HOLD accuracy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class HighAccuracyPredictor:
    def __init__(self):
        # Focus on most predictable stocks
        self.test_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA', 'HD',
            'SPY', 'QQQ', 'IWM', 'VTI', 'XLK', 'XLF',
            'NFLX', 'AMD', 'INTC', 'DIS'
        ]
        
        self.results = {
            'baseline_model': {'correct': 0, 'total': 0, 'predictions': []},
            'high_accuracy_model': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        self.detailed_results = {
            'baseline_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}},
            'high_accuracy_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}}
        }
    
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
            
            # Multiple timeframe moving averages
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            
            # Volume analysis
            avg_volume_5 = data['Volume'].rolling(5).mean().iloc[-1]
            avg_volume_10 = data['Volume'].rolling(10).mean().iloc[-1]
            avg_volume_20 = data['Volume'].rolling(20).mean().iloc[-1]
            
            # Multiple momentum timeframes
            momentum_1 = (current_price - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100 if len(data) >= 1 else 0
            momentum_3 = (current_price - data['Close'].iloc[-3]) / data['Close'].iloc[-3] * 100 if len(data) >= 3 else 0
            momentum_5 = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100 if len(data) >= 5 else 0
            momentum_10 = (current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10] * 100 if len(data) >= 10 else 0
            
            # RSI with multiple periods
            def calc_rsi(prices, period):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
            
            rsi_14 = calc_rsi(data['Close'], 14)
            rsi_21 = calc_rsi(data['Close'], 21)
            
            # MACD
            ema_12 = data['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['Close'].ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            macd_signal = data['Close'].ewm(span=9).mean().iloc[-1]
            macd_histogram = macd - macd_signal
            
            # Bollinger Bands
            bb_std = data['Close'].rolling(20).std().iloc[-1]
            bollinger_upper = sma_20 + (bb_std * 2)
            bollinger_lower = sma_20 - (bb_std * 2)
            
            # Volatility
            volatility_5 = data['Close'].pct_change().rolling(5).std().iloc[-1] * 100
            volatility_10 = data['Close'].pct_change().rolling(10).std().iloc[-1] * 100
            
            # Price position analysis
            recent_high_10 = data['High'].rolling(10).max().iloc[-1]
            recent_low_10 = data['Low'].rolling(10).min().iloc[-1]
            recent_high_20 = data['High'].rolling(20).max().iloc[-1]
            recent_low_20 = data['Low'].rolling(20).min().iloc[-1]
            
            return {
                'current_price': current_price,
                'volume': volume,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'avg_volume_5': avg_volume_5,
                'avg_volume_10': avg_volume_10,
                'avg_volume_20': avg_volume_20,
                'momentum_1': momentum_1,
                'momentum_3': momentum_3,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'rsi_14': rsi_14,
                'rsi_21': rsi_21,
                'macd': macd,
                'macd_histogram': macd_histogram,
                'bollinger_upper': bollinger_upper,
                'bollinger_lower': bollinger_lower,
                'volatility_5': volatility_5,
                'volatility_10': volatility_10,
                'recent_high_10': recent_high_10,
                'recent_low_10': recent_low_10,
                'recent_high_20': recent_high_20,
                'recent_low_20': recent_low_20,
                # Derived indicators
                'volume_ratio_5': volume / avg_volume_5 if avg_volume_5 > 0 else 1,
                'volume_ratio_10': volume / avg_volume_10 if avg_volume_10 > 0 else 1,
                'volume_ratio_20': volume / avg_volume_20 if avg_volume_20 > 0 else 1,
                'price_vs_sma20': (current_price - sma_20) / sma_20 * 100,
                'price_vs_sma50': (current_price - sma_50) / sma_50 * 100,
                'bb_position': (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower) if bollinger_upper != bollinger_lower else 0.5,
                'distance_from_high_10': (recent_high_10 - current_price) / recent_high_10 * 100,
                'distance_from_low_10': (current_price - recent_low_10) / recent_low_10 * 100,
                'ma_alignment_bullish': (sma_5 > sma_10 > sma_20 > sma_50),
                'ma_alignment_bearish': (sma_5 < sma_10 < sma_20 < sma_50)
            }
        except Exception as e:
            return None
    
    def baseline_model(self, indicators):
        """Simple baseline model"""
        score = 50
        
        if indicators['rsi_14'] < 30:
            score += 15
        elif indicators['rsi_14'] > 70:
            score -= 15
        
        if indicators['momentum_5'] > 3:
            score += 10
        elif indicators['momentum_5'] < -3:
            score -= 10
        
        if indicators['volume_ratio_20'] > 1.5:
            score += 8
        
        if indicators['price_vs_sma20'] > 2:
            score += 10
        elif indicators['price_vs_sma20'] < -2:
            score -= 10
        
        if score >= 65:
            return 'BUY'
        elif score <= 35:
            return 'SELL'
        else:
            return 'HOLD'
    
    def high_accuracy_model(self, indicators):
        """High accuracy model with strict criteria"""
        
        # === ULTRA HIGH CONFIDENCE BUY (Target 75%+ accuracy) ===
        buy_score = 0
        
        # Momentum alignment (critical for BUY accuracy)
        if indicators['momentum_3'] > 0.5 and indicators['momentum_5'] > 1.0 and indicators['momentum_10'] > 1.5:
            buy_score += 30  # Strong momentum alignment
        elif indicators['momentum_5'] > 1.5:
            buy_score += 15
        
        # RSI in optimal zone (not overbought)
        if 40 <= indicators['rsi_14'] <= 60:
            buy_score += 25  # Sweet spot
        elif 35 <= indicators['rsi_14'] <= 65:
            buy_score += 15
        elif indicators['rsi_14'] > 70:
            buy_score -= 20  # Overbought penalty
        
        # Moving average alignment
        if indicators['ma_alignment_bullish']:
            buy_score += 25  # Perfect alignment
        elif indicators['current_price'] > indicators['sma_20'] > indicators['sma_50']:
            buy_score += 15
        
        # Volume confirmation
        if indicators['volume_ratio_5'] > 1.3 and indicators['volume_ratio_10'] > 1.1:
            buy_score += 20
        elif indicators['volume_ratio_10'] > 1.2:
            buy_score += 10
        
        # MACD bullish
        if indicators['macd'] > 0 and indicators['macd_histogram'] > 0:
            buy_score += 15
        
        # Not at extreme highs
        if indicators['distance_from_high_10'] > 2:
            buy_score += 10
        
        # Low volatility (more predictable)
        if indicators['volatility_10'] < 2:
            buy_score += 10
        
        # === ULTRA HIGH CONFIDENCE SELL (Target 70%+ accuracy) ===
        sell_score = 0
        
        # Strong negative momentum
        if indicators['momentum_3'] < -0.5 and indicators['momentum_5'] < -1.0 and indicators['momentum_10'] < -1.5:
            sell_score += 30
        elif indicators['momentum_5'] < -2:
            sell_score += 20
        
        # RSI showing clear weakness
        if indicators['rsi_14'] < 35:
            sell_score += 25
        elif indicators['rsi_14'] < 40:
            sell_score += 15
        
        # Moving average breakdown
        if indicators['ma_alignment_bearish']:
            sell_score += 25
        elif indicators['current_price'] < indicators['sma_20'] < indicators['sma_50']:
            sell_score += 15
        
        # Volume on selling
        if indicators['volume_ratio_5'] > 1.2 and indicators['momentum_5'] < -1:
            sell_score += 20
        
        # MACD bearish
        if indicators['macd'] < 0 and indicators['macd_histogram'] < 0:
            sell_score += 15
        
        # Near recent highs (reversal)
        if indicators['distance_from_high_10'] < 1:
            sell_score += 15
        
        # === ULTRA HIGH CONFIDENCE HOLD (Target 60%+ accuracy) ===
        hold_score = 0
        
        # Very low volatility
        if indicators['volatility_5'] < 1:
            hold_score += 25
        elif indicators['volatility_10'] < 1.5:
            hold_score += 15
        
        # Minimal momentum
        if abs(indicators['momentum_5']) < 0.5:
            hold_score += 25
        elif abs(indicators['momentum_5']) < 1:
            hold_score += 15
        
        # RSI in dead neutral
        if 48 <= indicators['rsi_14'] <= 52:
            hold_score += 25
        elif 45 <= indicators['rsi_14'] <= 55:
            hold_score += 15
        
        # Price very close to MA
        if abs(indicators['price_vs_sma20']) < 0.5:
            hold_score += 20
        elif abs(indicators['price_vs_sma20']) < 1:
            hold_score += 10
        
        # Normal volume
        if 0.9 <= indicators['volume_ratio_10'] <= 1.1:
            hold_score += 15
        
        # Bollinger middle
        if 0.4 <= indicators['bb_position'] <= 0.6:
            hold_score += 15
        
        # === ULTRA STRICT DECISION LOGIC ===
        
        # Require very high confidence for any prediction
        if buy_score >= 70:  # Very high threshold for BUY
            return 'BUY'
        elif sell_score >= 60:  # High threshold for SELL
            return 'SELL'
        elif hold_score >= 50:  # Moderate threshold for HOLD
            return 'HOLD'
        else:
            # If no high confidence, use conservative fallback
            if buy_score > 40 and buy_score > sell_score and buy_score > hold_score:
                return 'BUY'
            elif sell_score > 35 and sell_score > buy_score and sell_score > hold_score:
                return 'SELL'
            else:
                return 'HOLD'  # Default to HOLD when uncertain
    
    def get_actual_outcome(self, data, index, timeframe=10):
        """Get actual outcome with shorter timeframe for better accuracy"""
        if index + timeframe >= len(data):
            return None
        
        current_price = data['Close'].iloc[index]
        future_price = data['Close'].iloc[index + timeframe]
        change = (future_price - current_price) / current_price * 100
        
        # More conservative thresholds for better accuracy
        if change > 2:  # 2% threshold for BUY
            return 'BUY', change
        elif change < -2:  # -2% threshold for SELL
            return 'SELL', change
        else:
            return 'HOLD', change
    
    def run_high_accuracy_test(self):
        print("🎯 High Accuracy Model Test - Targeting 75%+ BUY, 70%+ SELL, 60%+ HOLD")
        print("=" * 70)
        
        symbols_tested = 0
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            data = self.get_historical_data(symbol)
            if data is None:
                continue
            
            symbols_tested += 1
            tests_for_symbol = 0
            
            # Test every day for maximum data points
            for i in range(60, len(data) - 12, 1):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                baseline_pred = self.baseline_model(indicators)
                high_acc_pred = self.high_accuracy_model(indicators)
                
                actual_result = self.get_actual_outcome(data, i)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                tests_for_symbol += 1
                
                # Record results
                for model_name, prediction in [('baseline_model', baseline_pred), ('high_accuracy_model', high_acc_pred)]:
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
        self.print_high_accuracy_results()
    
    def print_high_accuracy_results(self):
        print("\n" + "=" * 70)
        print("🎯 HIGH ACCURACY MODEL RESULTS")
        print("=" * 70)
        
        baseline_accuracy = (self.results['baseline_model']['correct'] / 
                           self.results['baseline_model']['total'] * 100) if self.results['baseline_model']['total'] > 0 else 0
        
        high_acc_accuracy = (self.results['high_accuracy_model']['correct'] / 
                           self.results['high_accuracy_model']['total'] * 100) if self.results['high_accuracy_model']['total'] > 0 else 0
        
        print(f"\n📊 OVERALL ACCURACY:")
        print(f"Baseline Model:      {baseline_accuracy:.1f}% ({self.results['baseline_model']['correct']}/{self.results['baseline_model']['total']})")
        print(f"High Accuracy Model: {high_acc_accuracy:.1f}% ({self.results['high_accuracy_model']['correct']}/{self.results['high_accuracy_model']['total']})")
        
        improvement = high_acc_accuracy - baseline_accuracy
        print(f"\nImprovement: {improvement:+.1f}%")
        
        # Detailed accuracy analysis
        print(f"\n📈 DETAILED ACCURACY BY PREDICTION TYPE:")
        
        for model_name in ['baseline_model', 'high_accuracy_model']:
            print(f"\n{model_name.replace('_', ' ').title()}:")
            
            for pred_type in ['BUY', 'SELL', 'HOLD']:
                stats = self.detailed_results[model_name][pred_type]
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    target = {'BUY': 75, 'SELL': 70, 'HOLD': 60}[pred_type]
                    status = '✅' if accuracy >= target else '❌'
                    print(f"  {pred_type}: {status} {accuracy:.1f}% ({stats['correct']}/{stats['total']}) [Target: {target}%]")
                else:
                    print(f"  {pred_type}: No predictions made")
        
        # Success analysis
        print(f"\n🎯 TARGET ACHIEVEMENT ANALYSIS:")
        ha_results = self.detailed_results['high_accuracy_model']
        
        buy_acc = (ha_results['BUY']['correct'] / ha_results['BUY']['total'] * 100) if ha_results['BUY']['total'] > 0 else 0
        sell_acc = (ha_results['SELL']['correct'] / ha_results['SELL']['total'] * 100) if ha_results['SELL']['total'] > 0 else 0
        hold_acc = (ha_results['HOLD']['correct'] / ha_results['HOLD']['total'] * 100) if ha_results['HOLD']['total'] > 0 else 0
        
        targets_met = 0
        if buy_acc >= 75: targets_met += 1
        if sell_acc >= 70: targets_met += 1
        if hold_acc >= 60: targets_met += 1
        
        print(f"Targets achieved: {targets_met}/3")
        print(f"Overall target (>65%): {'✅' if high_acc_accuracy > 65 else '❌'} {high_acc_accuracy:.1f}%")

if __name__ == "__main__":
    predictor = HighAccuracyPredictor()
    predictor.run_high_accuracy_test()
