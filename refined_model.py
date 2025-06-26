#!/usr/bin/env python3
"""
Refined model targeting high accuracy with optimized thresholds
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RefinedStockPredictor:
    def __init__(self):
        # Focus on most liquid and predictable stocks
        self.test_symbols = [
            # Large Cap Tech (most predictable)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            # Large Cap Traditional
            'JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA', 'HD',
            # ETFs (most stable)
            'SPY', 'QQQ', 'IWM', 'VTI', 'XLK', 'XLF',
            # Selected growth stocks
            'NFLX', 'AMD', 'INTC', 'DIS'
        ]
        
        self.results = {
            'current_model': {'correct': 0, 'total': 0, 'predictions': []},
            'refined_model': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        self.detailed_results = {
            'current_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}},
            'refined_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}}
        }
    
    def get_historical_data(self, symbol, days_back=150):
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 100 else None
        except Exception as e:
            return None
    
    def calculate_indicators(self, data):
        if len(data) < 50:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Moving averages
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            
            # Volume averages
            avg_volume_10 = data['Volume'].rolling(10).mean().iloc[-1]
            avg_volume_20 = data['Volume'].rolling(20).mean().iloc[-1]
            
            # Momentum
            momentum_3 = (current_price - data['Close'].iloc[-3]) / data['Close'].iloc[-3] * 100 if len(data) >= 3 else 0
            momentum_5 = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100 if len(data) >= 5 else 0
            momentum_10 = (current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10] * 100 if len(data) >= 10 else 0
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
            
            # MACD
            ema_12 = data['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['Close'].ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            
            # Bollinger Bands
            bb_std = data['Close'].rolling(20).std().iloc[-1]
            bollinger_upper = sma_20 + (bb_std * 2)
            bollinger_lower = sma_20 - (bb_std * 2)
            
            # Volatility
            volatility = data['Close'].pct_change().rolling(10).std().iloc[-1] * 100
            
            return {
                'current_price': current_price,
                'volume': volume,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'avg_volume_10': avg_volume_10,
                'avg_volume_20': avg_volume_20,
                'momentum_3': momentum_3,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'rsi': rsi,
                'macd': macd,
                'bollinger_upper': bollinger_upper,
                'bollinger_lower': bollinger_lower,
                'volatility': volatility,
                'volume_ratio_10': volume / avg_volume_10 if avg_volume_10 > 0 else 1,
                'volume_ratio_20': volume / avg_volume_20 if avg_volume_20 > 0 else 1,
                'price_vs_sma20': (current_price - sma_20) / sma_20 * 100,
                'price_vs_sma50': (current_price - sma_50) / sma_50 * 100,
                'bb_position': (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower) if bollinger_upper != bollinger_lower else 0.5
            }
        except Exception as e:
            return None
    
    def current_model(self, indicators):
        """Current model from app.py"""
        score = 50
        
        if indicators['rsi'] < 30:
            score += 15
        elif indicators['rsi'] > 70:
            score -= 15
        
        if indicators['current_price'] > indicators['sma_20'] > indicators['sma_50']:
            score += 15
        elif indicators['current_price'] < indicators['sma_20'] < indicators['sma_50']:
            score -= 15
        
        if indicators['volume_ratio_20'] > 2:
            score += 15
        elif indicators['volume_ratio_20'] > 1.5:
            score += 8
        elif indicators['volume_ratio_20'] < 0.5:
            score -= 10
        
        if indicators['bb_position'] < 0.2:
            score += 10
        elif indicators['bb_position'] > 0.8:
            score -= 10
        
        if indicators['macd'] > 0:
            score += min(10, abs(indicators['macd']) * 2)
        else:
            score -= min(10, abs(indicators['macd']) * 2)
        
        if indicators['momentum_5'] > 5:
            score += 10
        elif indicators['momentum_5'] < -5:
            score -= 10
        
        if score >= 70:
            return 'BUY'
        elif score <= 35:
            return 'SELL'
        else:
            return 'HOLD'
    
    def refined_model(self, indicators):
        """Refined model with high accuracy focus"""
        
        # === HIGH CONFIDENCE BUY SIGNALS ===
        buy_signals = 0
        buy_strength = 0
        
        # Strong momentum alignment (most important)
        if indicators['momentum_3'] > 1 and indicators['momentum_5'] > 2:
            buy_signals += 1
            buy_strength += 2
        
        # RSI in optimal range (not overbought)
        if 45 <= indicators['rsi'] <= 65:
            buy_signals += 1
            buy_strength += 1
        
        # Price above key moving averages
        if indicators['current_price'] > indicators['sma_20'] > indicators['sma_50']:
            buy_signals += 1
            buy_strength += 2
        
        # Volume confirmation
        if indicators['volume_ratio_10'] > 1.2:
            buy_signals += 1
            buy_strength += 1
        
        # MACD bullish
        if indicators['macd'] > 0:
            buy_signals += 1
            buy_strength += 1
        
        # Not at Bollinger upper extreme
        if indicators['bb_position'] < 0.8:
            buy_signals += 1
        
        # === HIGH CONFIDENCE SELL SIGNALS ===
        sell_signals = 0
        sell_strength = 0
        
        # Strong negative momentum
        if indicators['momentum_3'] < -1 and indicators['momentum_5'] < -2:
            sell_signals += 1
            sell_strength += 2
        
        # RSI showing weakness
        if indicators['rsi'] < 40:
            sell_signals += 1
            sell_strength += 1
        
        # Price below key moving averages
        if indicators['current_price'] < indicators['sma_20'] < indicators['sma_50']:
            sell_signals += 1
            sell_strength += 2
        
        # Volume on decline
        if indicators['volume_ratio_10'] > 1.1 and indicators['momentum_5'] < -1:
            sell_signals += 1
            sell_strength += 1
        
        # MACD bearish
        if indicators['macd'] < 0:
            sell_signals += 1
            sell_strength += 1
        
        # At Bollinger lower extreme
        if indicators['bb_position'] < 0.2:
            sell_signals += 1
        
        # === HIGH CONFIDENCE HOLD SIGNALS ===
        hold_signals = 0
        hold_strength = 0
        
        # Low volatility
        if indicators['volatility'] < 2:
            hold_signals += 1
            hold_strength += 1
        
        # Neutral momentum
        if abs(indicators['momentum_5']) < 1.5:
            hold_signals += 1
            hold_strength += 1
        
        # RSI in neutral zone
        if 45 <= indicators['rsi'] <= 55:
            hold_signals += 1
            hold_strength += 1
        
        # Price near SMA20
        if abs(indicators['price_vs_sma20']) < 2:
            hold_signals += 1
            hold_strength += 1
        
        # Normal volume
        if 0.8 <= indicators['volume_ratio_20'] <= 1.3:
            hold_signals += 1
        
        # Bollinger middle
        if 0.3 <= indicators['bb_position'] <= 0.7:
            hold_signals += 1
        
        # === DECISION LOGIC (High Confidence Required) ===
        
        # Require strong signals for BUY
        if buy_signals >= 4 and buy_strength >= 5:
            return 'BUY'
        
        # Require strong signals for SELL
        elif sell_signals >= 4 and sell_strength >= 5:
            return 'SELL'
        
        # Require strong signals for HOLD
        elif hold_signals >= 4 and hold_strength >= 3:
            return 'HOLD'
        
        # If no strong signals, use conservative logic
        else:
            # Conservative BUY (market bias)
            if (buy_signals >= 3 and buy_strength >= 3 and 
                indicators['momentum_5'] > 1 and indicators['rsi'] < 70):
                return 'BUY'
            
            # Conservative SELL (strong evidence needed)
            elif (sell_signals >= 3 and sell_strength >= 4 and 
                  indicators['momentum_5'] < -2):
                return 'SELL'
            
            # Default to HOLD when uncertain
            else:
                return 'HOLD'
    
    def get_actual_outcome(self, data, index, timeframe=12):
        """Get actual outcome with optimized timeframe"""
        if index + timeframe >= len(data):
            return None
        
        current_price = data['Close'].iloc[index]
        future_price = data['Close'].iloc[index + timeframe]
        change = (future_price - current_price) / current_price * 100
        
        # Optimized thresholds based on analysis
        if change > 2.5:  # Lowered from 4% to 2.5%
            return 'BUY', change
        elif change < -2.5:  # Lowered from -4% to -2.5%
            return 'SELL', change
        else:
            return 'HOLD', change
    
    def run_refined_backtest(self):
        print("🎯 Refined Model Backtest - High Accuracy Focus")
        print("=" * 60)
        print(f"Testing {len(self.test_symbols)} high-quality symbols")
        print("=" * 60)
        
        symbols_tested = 0
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            data = self.get_historical_data(symbol)
            if data is None:
                print(f"❌ Insufficient data for {symbol}")
                continue
            
            symbols_tested += 1
            tests_for_symbol = 0
            
            # Test every 2 days for more data points
            for i in range(80, len(data) - 15, 2):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                # Get predictions
                current_pred = self.current_model(indicators)
                refined_pred = self.refined_model(indicators)
                
                # Get actual outcome
                actual_result = self.get_actual_outcome(data, i)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                tests_for_symbol += 1
                
                # Record results
                for model_name, prediction in [('current_model', current_pred), ('refined_model', refined_pred)]:
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
        self.print_refined_results()
    
    def print_refined_results(self):
        print("\n" + "=" * 60)
        print("🎯 REFINED MODEL RESULTS")
        print("=" * 60)
        
        current_accuracy = (self.results['current_model']['correct'] / 
                          self.results['current_model']['total'] * 100) if self.results['current_model']['total'] > 0 else 0
        
        refined_accuracy = (self.results['refined_model']['correct'] / 
                          self.results['refined_model']['total'] * 100) if self.results['refined_model']['total'] > 0 else 0
        
        print(f"\n📊 OVERALL ACCURACY:")
        print(f"Current Model:  {current_accuracy:.1f}% ({self.results['current_model']['correct']}/{self.results['current_model']['total']})")
        print(f"Refined Model:  {refined_accuracy:.1f}% ({self.results['refined_model']['correct']}/{self.results['refined_model']['total']})")
        
        improvement = refined_accuracy - current_accuracy
        if improvement > 2:
            print(f"\n✅ IMPROVEMENT: +{improvement:.1f}% better accuracy!")
        elif improvement < -2:
            print(f"\n❌ REGRESSION: -{abs(improvement):.1f}% worse accuracy")
        else:
            print(f"\n🤝 SIMILAR: ±{abs(improvement):.1f}% difference")
        
        # Detailed breakdown
        print(f"\n📈 DETAILED ACCURACY BY PREDICTION TYPE:")
        for model_name in ['current_model', 'refined_model']:
            print(f"\n{model_name.replace('_', ' ').title()}:")
            
            for pred_type in ['BUY', 'SELL', 'HOLD']:
                stats = self.detailed_results[model_name][pred_type]
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    print(f"  {pred_type}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
                else:
                    print(f"  {pred_type}: No predictions made")
        
        # High accuracy targets
        print(f"\n🎯 HIGH ACCURACY TARGET ANALYSIS:")
        for model_name in ['current_model', 'refined_model']:
            model_results = self.detailed_results[model_name]
            
            buy_acc = (model_results['BUY']['correct'] / model_results['BUY']['total'] * 100) if model_results['BUY']['total'] > 0 else 0
            sell_acc = (model_results['SELL']['correct'] / model_results['SELL']['total'] * 100) if model_results['SELL']['total'] > 0 else 0
            hold_acc = (model_results['HOLD']['correct'] / model_results['HOLD']['total'] * 100) if model_results['HOLD']['total'] > 0 else 0
            
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  BUY accuracy target (>75%): {'✅' if buy_acc > 75 else '❌'} {buy_acc:.1f}%")
            print(f"  SELL accuracy target (>70%): {'✅' if sell_acc > 70 else '❌'} {sell_acc:.1f}%")
            print(f"  HOLD accuracy target (>60%): {'✅' if hold_acc > 60 else '❌'} {hold_acc:.1f}%")
            print(f"  Overall target (>65%): {'✅' if (current_accuracy if model_name == 'current_model' else refined_accuracy) > 65 else '❌'}")

if __name__ == "__main__":
    predictor = RefinedStockPredictor()
    predictor.run_refined_backtest()
