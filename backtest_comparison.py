#!/usr/bin/env python3
"""
Backtesting script to compare Simple vs Complex prediction models
Tests both approaches against historical data to see which is more accurate
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class BacktestComparison:
    def __init__(self):
        self.test_symbols = [
            # Mix of different categories for comprehensive testing
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Large caps
            'PLTR', 'SOFI', 'CRWD', 'SNOW', 'ZS',     # Small/mid caps
            'TSLA', 'NVDA', 'AMD', 'ROKU', 'COIN',    # High volatility
            'SPY', 'QQQ', 'IWM', 'XLK', 'XLF',       # ETFs for baseline
            'ETSY', 'CHWY', 'RBLX', 'ENPH', 'DKNG'   # More diverse stocks
        ]
        self.results = {
            'simple_model': {'correct': 0, 'total': 0, 'predictions': []},
            'complex_model': {'correct': 0, 'total': 0, 'predictions': []}
        }
    
    def get_historical_data(self, symbol, days_back=90):
        """Get historical data for backtesting"""
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if not data.empty else None
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for both models"""
        if len(data) < 50:
            return None
        
        current_price = data['Close'].iloc[-1]
        
        # Moving averages
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        macd = (exp1 - exp2).iloc[-1]
        
        # Volume analysis
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
        
        # Price momentum (10-day rate of change)
        if len(data) >= 10:
            price_momentum = (current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10] * 100
        else:
            price_momentum = 0
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma = data['Close'].rolling(window=bb_period).mean()
        std = data['Close'].rolling(window=bb_period).std()
        bollinger_upper = (sma + (std * bb_std)).iloc[-1]
        bollinger_lower = (sma - (std * bb_std)).iloc[-1]
        
        return {
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'macd': macd,
            'volume': volume,
            'avg_volume': avg_volume,
            'price_momentum': price_momentum,
            'bollinger_upper': bollinger_upper,
            'bollinger_lower': bollinger_lower
        }
    
    def simple_model_prediction(self, indicators):
        """New simple model: Volume + Momentum"""
        volume = indicators['volume']
        avg_volume = indicators['avg_volume']
        price_momentum = indicators['price_momentum']
        current_price = indicators['current_price']
        sma_20 = indicators['sma_20']
        
        # Volume signal
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 2.0:
            volume_strength = 2
        elif volume_ratio > 1.5:
            volume_strength = 1
        elif volume_ratio < 0.5:
            volume_strength = -1
        else:
            volume_strength = 0
        
        # Momentum signal
        if price_momentum > 5:
            momentum_strength = 1
        elif price_momentum < -5:
            momentum_strength = -1
        else:
            momentum_strength = 0
        
        # Combine signals (Volume 60%, Momentum 40%)
        combined_strength = (volume_strength * 0.6) + (momentum_strength * 0.4)
        
        if combined_strength >= 0.8:
            return 'BUY', combined_strength
        elif combined_strength <= -0.8:
            return 'SELL', abs(combined_strength)
        else:
            return 'HOLD', abs(combined_strength)
    
    def complex_model_prediction(self, indicators):
        """Old complex model: Multi-factor scoring"""
        current_price = indicators['current_price']
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        rsi = indicators['rsi']
        macd = indicators['macd']
        volume = indicators['volume']
        avg_volume = indicators['avg_volume']
        price_momentum = indicators['price_momentum']
        bollinger_upper = indicators['bollinger_upper']
        bollinger_lower = indicators['bollinger_lower']
        
        score = 50  # Start neutral
        
        # Technical score (RSI, MA, Bollinger)
        if rsi < 30:
            score += 15
        elif rsi > 70:
            score -= 15
        
        if current_price > sma_20 > sma_50:
            score += 15
        elif current_price < sma_20 < sma_50:
            score -= 15
        
        bb_position = (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower)
        if bb_position < 0.2:
            score += 10
        elif bb_position > 0.8:
            score -= 10
        
        # Momentum score
        if macd > 0:
            score += min(10, abs(macd) * 2)
        else:
            score -= min(10, abs(macd) * 2)
        
        if price_momentum > 5:
            score += 10
        elif price_momentum < -5:
            score -= 10
        
        # Volume score
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 2:
            score += 15
        elif volume_ratio > 1.5:
            score += 8
        elif volume_ratio < 0.5:
            score -= 10
        
        # Convert score to prediction
        if score >= 70:
            return 'BUY', (score - 50) / 50
        elif score <= 30:
            return 'SELL', (50 - score) / 50
        else:
            return 'HOLD', abs(score - 50) / 50
    
    def test_prediction_accuracy(self, symbol, prediction_date_index, actual_data):
        """Test if prediction was correct based on future price movement"""
        if prediction_date_index + 20 >= len(actual_data):  # Need 20 days future data
            return None
        
        current_price = actual_data['Close'].iloc[prediction_date_index]
        future_price = actual_data['Close'].iloc[prediction_date_index + 20]  # 20 days later
        
        actual_change = (future_price - current_price) / current_price * 100

        if actual_change > 2:  # More realistic threshold
            actual_direction = 'BUY'
        elif actual_change < -2:
            actual_direction = 'SELL'
        else:
            actual_direction = 'HOLD'
        
        return actual_direction, actual_change
    
    def run_backtest(self):
        """Run comprehensive backtest comparing both models"""
        print("🧪 Starting Backtest Comparison: Simple vs Complex Models")
        print("=" * 60)
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            data = self.get_historical_data(symbol, days_back=120)
            if data is None or len(data) < 70:
                print(f"❌ Insufficient data for {symbol}")
                continue
            
            # Test multiple prediction points (every 3 days for last 60 days)
            for i in range(50, len(data) - 20, 3):
                # Get data up to prediction point
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_technical_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                # Get predictions from both models
                simple_pred, simple_confidence = self.simple_model_prediction(indicators)
                complex_pred, complex_confidence = self.complex_model_prediction(indicators)
                
                # Get actual result
                actual_result = self.test_prediction_accuracy(i, i, data)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                
                # Record results
                self.results['simple_model']['total'] += 1
                self.results['complex_model']['total'] += 1
                
                if simple_pred == actual_direction:
                    self.results['simple_model']['correct'] += 1
                
                if complex_pred == actual_direction:
                    self.results['complex_model']['correct'] += 1
                
                # Store detailed results
                self.results['simple_model']['predictions'].append({
                    'symbol': symbol,
                    'predicted': simple_pred,
                    'actual': actual_direction,
                    'actual_change': actual_change,
                    'confidence': simple_confidence,
                    'correct': simple_pred == actual_direction
                })
                
                self.results['complex_model']['predictions'].append({
                    'symbol': symbol,
                    'predicted': complex_pred,
                    'actual': actual_direction,
                    'actual_change': actual_change,
                    'confidence': complex_confidence,
                    'correct': complex_pred == actual_direction
                })
        
        self.print_results()
    
    def print_results(self):
        """Print comprehensive comparison results"""
        print("\n" + "=" * 60)
        print("🎯 BACKTEST RESULTS COMPARISON")
        print("=" * 60)
        
        simple_accuracy = (self.results['simple_model']['correct'] / 
                          self.results['simple_model']['total'] * 100) if self.results['simple_model']['total'] > 0 else 0
        
        complex_accuracy = (self.results['complex_model']['correct'] / 
                           self.results['complex_model']['total'] * 100) if self.results['complex_model']['total'] > 0 else 0
        
        print(f"\n📊 OVERALL ACCURACY:")
        print(f"Simple Model (Volume + Momentum):  {simple_accuracy:.1f}% ({self.results['simple_model']['correct']}/{self.results['simple_model']['total']})")
        print(f"Complex Model (Multi-factor):      {complex_accuracy:.1f}% ({self.results['complex_model']['correct']}/{self.results['complex_model']['total']})")
        
        difference = simple_accuracy - complex_accuracy
        if difference > 2:
            print(f"\n✅ WINNER: Simple Model (+{difference:.1f}% better)")
        elif difference < -2:
            print(f"\n✅ WINNER: Complex Model (+{abs(difference):.1f}% better)")
        else:
            print(f"\n🤝 TIE: Models perform similarly (±{abs(difference):.1f}%)")
        
        # Analyze by prediction type
        print(f"\n📈 PREDICTION TYPE BREAKDOWN:")
        for model_name, model_data in self.results.items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            
            buy_correct = sum(1 for p in model_data['predictions'] if p['predicted'] == 'BUY' and p['correct'])
            buy_total = sum(1 for p in model_data['predictions'] if p['predicted'] == 'BUY')
            
            sell_correct = sum(1 for p in model_data['predictions'] if p['predicted'] == 'SELL' and p['correct'])
            sell_total = sum(1 for p in model_data['predictions'] if p['predicted'] == 'SELL')
            
            hold_correct = sum(1 for p in model_data['predictions'] if p['predicted'] == 'HOLD' and p['correct'])
            hold_total = sum(1 for p in model_data['predictions'] if p['predicted'] == 'HOLD')
            
            if buy_total > 0:
                print(f"  BUY predictions:  {buy_correct/buy_total*100:.1f}% ({buy_correct}/{buy_total})")
            if sell_total > 0:
                print(f"  SELL predictions: {sell_correct/sell_total*100:.1f}% ({sell_correct}/{sell_total})")
            if hold_total > 0:
                print(f"  HOLD predictions: {hold_correct/hold_total*100:.1f}% ({hold_correct}/{hold_total})")

if __name__ == "__main__":
    backtest = BacktestComparison()
    backtest.run_backtest()
