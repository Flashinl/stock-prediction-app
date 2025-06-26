#!/usr/bin/env python3
"""
Test the improved sell and hold logic against the backtest
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ImprovedBacktest:
    def __init__(self):
        self.test_symbols = ['AAPL', 'TSLA', 'PLTR', 'SPY', 'NVDA']
        self.results = {
            'original_model': {'correct': 0, 'total': 0, 'predictions': []},
            'improved_model': {'correct': 0, 'total': 0, 'predictions': []}
        }
    
    def get_historical_data(self, symbol, days_back=120):
        """Get historical data for backtesting"""
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 50 else None
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        if len(data) < 20:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            
            # Price momentum (5-day)
            if len(data) >= 5:
                price_momentum = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100
            else:
                price_momentum = 0
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
            
            # Moving averages
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
            
            # MACD
            ema_12 = data['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['Close'].ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            
            # Bollinger Bands
            bb_period = 20
            bb_std = data['Close'].rolling(bb_period).std().iloc[-1]
            bb_middle = sma_20
            bollinger_upper = bb_middle + (bb_std * 2)
            bollinger_lower = bb_middle - (bb_std * 2)
            
            return {
                'current_price': current_price,
                'volume': volume,
                'avg_volume': avg_volume,
                'price_momentum': price_momentum,
                'rsi': rsi,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'macd': macd,
                'bollinger_upper': bollinger_upper,
                'bollinger_lower': bollinger_lower
            }
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None
    
    def original_model_prediction(self, indicators):
        """Original model prediction logic (from current app.py)"""
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
        
        # RSI component
        if rsi < 30:
            score += 15
        elif rsi > 70:
            score -= 15
        
        # Moving average component
        if current_price > sma_20 > sma_50:
            score += 15
        elif current_price < sma_20 < sma_50:
            score -= 15
        
        # Volume component
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 2:
            score += 15
        elif volume_ratio > 1.5:
            score += 8
        elif volume_ratio < 0.5:
            score -= 10
        
        # Bollinger Bands
        bb_position = (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower)
        if bb_position < 0.2:
            score += 10
        elif bb_position > 0.8:
            score -= 10
        
        # Momentum
        if macd > 0:
            score += min(10, abs(macd) * 2)
        else:
            score -= min(10, abs(macd) * 2)
        
        if price_momentum > 5:
            score += 10
        elif price_momentum < -5:
            score -= 10
        
        # Original thresholds
        if score >= 70:
            return 'BUY', score
        elif score <= 30:
            return 'SELL', score
        else:
            return 'HOLD', score
    
    def calculate_momentum_strength(self, indicators):
        """Calculate overall momentum strength"""
        price_momentum = indicators.get('price_momentum', 0)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        
        momentum_score = 0
        momentum_score += max(-1, min(1, price_momentum / 10))
        
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 1.5:
            momentum_score += 0.3
        elif volume_ratio < 0.7:
            momentum_score -= 0.3
        
        if rsi > 60:
            momentum_score += 0.2
        elif rsi < 40:
            momentum_score -= 0.2
        
        momentum_score += max(-0.2, min(0.2, macd / 5))
        
        return momentum_score
    
    def improved_model_prediction(self, indicators):
        """Improved model with better sell/hold logic"""
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
        
        # Same scoring as original
        if rsi < 30:
            score += 15
        elif rsi > 70:
            score -= 15
        
        if current_price > sma_20 > sma_50:
            score += 15
        elif current_price < sma_20 < sma_50:
            score -= 15
        
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 2:
            score += 15
        elif volume_ratio > 1.5:
            score += 8
        elif volume_ratio < 0.5:
            score -= 10
        
        bb_position = (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower)
        if bb_position < 0.2:
            score += 10
        elif bb_position > 0.8:
            score -= 10
        
        if macd > 0:
            score += min(10, abs(macd) * 2)
        else:
            score -= min(10, abs(macd) * 2)
        
        if price_momentum > 5:
            score += 10
        elif price_momentum < -5:
            score -= 10
        
        # IMPROVED DECISION LOGIC
        momentum_strength = self.calculate_momentum_strength(indicators)
        
        # BUY logic (keep what worked)
        if score >= 70:
            return 'BUY', score
        
        # IMPROVED SELL logic - more decisive
        elif score <= 35:  # Raised threshold
            # Check for strong sell indicators
            strong_sell_indicators = 0
            if current_price < sma_20 < sma_50:
                strong_sell_indicators += 1
            if rsi < 40 and price_momentum < -3:
                strong_sell_indicators += 1
            if volume_ratio > 1.3 and price_momentum < -2:
                strong_sell_indicators += 1
            if momentum_strength < -0.4:
                strong_sell_indicators += 1
            
            if strong_sell_indicators >= 2:
                return 'SELL', score
            else:
                return 'HOLD', score  # Cautious hold instead of weak sell
        
        # IMPROVED HOLD logic - much more restrictive, favor BUY when uncertain
        elif 45 <= score <= 62:
            # Very strict consolidation requirements
            consolidation_score = 0
            if abs(current_price - sma_20) / sma_20 < 0.02:  # Very close to MA
                consolidation_score += 1
            if 48 <= rsi <= 52:  # Very neutral RSI
                consolidation_score += 1
            if abs(price_momentum) < 1.5:  # Very low momentum
                consolidation_score += 1
            if 0.9 <= volume_ratio <= 1.1:  # Very normal volume
                consolidation_score += 1
            if abs(momentum_strength) < 0.15:  # Very low overall momentum
                consolidation_score += 1

            # Only HOLD if we have very strong consolidation signals
            if consolidation_score >= 4:
                return 'HOLD', score
            elif score >= 50:  # Any bullish bias becomes BUY
                return 'BUY', score
            else:
                # Bearish bias - check for sell signals
                if momentum_strength < -0.25 and price_momentum < -2:
                    return 'SELL', score
                else:
                    # When in doubt, lean BUY (market bias)
                    return 'BUY', score

        # Edge cases - favor BUY over HOLD
        else:
            if score > 62:  # 63-69 range
                return 'BUY', score
            elif score >= 40:  # 40-44 range
                # Lean towards BUY unless strong bearish signals
                if momentum_strength < -0.3 and price_momentum < -3:
                    return 'SELL', score
                else:
                    # Default to BUY when uncertain (market bias)
                    return 'BUY', score
            else:  # 36-39 range
                # More likely to be SELL, but still check
                if momentum_strength < -0.2 or price_momentum < -2:
                    return 'SELL', score
                else:
                    # Even here, consider BUY if no strong bearish momentum
                    return 'BUY', score
    
    def test_prediction_accuracy(self, symbol, prediction_date_index, actual_data):
        """Test if prediction was correct based on future price movement"""
        if prediction_date_index + 20 >= len(actual_data):
            return None
        
        current_price = actual_data['Close'].iloc[prediction_date_index]
        future_price = actual_data['Close'].iloc[prediction_date_index + 20]
        
        actual_change = (future_price - current_price) / current_price * 100
        
        if actual_change > 2:
            actual_direction = 'BUY'
        elif actual_change < -2:
            actual_direction = 'SELL'
        else:
            actual_direction = 'HOLD'
        
        return actual_direction, actual_change
    
    def run_backtest(self):
        """Run comprehensive backtest comparing original vs improved models"""
        print("🧪 Testing Improved Sell/Hold Logic")
        print("=" * 60)
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            data = self.get_historical_data(symbol, days_back=120)
            if data is None or len(data) < 70:
                print(f"❌ Insufficient data for {symbol}")
                continue
            
            # Test multiple prediction points
            for i in range(50, len(data) - 20, 3):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_technical_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                # Get predictions from both models
                original_pred, original_score = self.original_model_prediction(indicators)
                improved_pred, improved_score = self.improved_model_prediction(indicators)
                
                # Get actual result
                actual_result = self.test_prediction_accuracy(i, i, data)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                
                # Record results
                self.results['original_model']['total'] += 1
                self.results['improved_model']['total'] += 1
                
                if original_pred == actual_direction:
                    self.results['original_model']['correct'] += 1
                
                if improved_pred == actual_direction:
                    self.results['improved_model']['correct'] += 1
                
                # Store detailed results
                self.results['original_model']['predictions'].append({
                    'symbol': symbol,
                    'predicted': original_pred,
                    'actual': actual_direction,
                    'actual_change': actual_change,
                    'score': original_score,
                    'correct': original_pred == actual_direction
                })
                
                self.results['improved_model']['predictions'].append({
                    'symbol': symbol,
                    'predicted': improved_pred,
                    'actual': actual_direction,
                    'actual_change': actual_change,
                    'score': improved_score,
                    'correct': improved_pred == actual_direction
                })
        
        self.print_results()
    
    def print_results(self):
        """Print comprehensive comparison results"""
        print("\n" + "=" * 60)
        print("🎯 IMPROVED LOGIC BACKTEST RESULTS")
        print("=" * 60)
        
        original_accuracy = (self.results['original_model']['correct'] / 
                           self.results['original_model']['total'] * 100) if self.results['original_model']['total'] > 0 else 0
        
        improved_accuracy = (self.results['improved_model']['correct'] / 
                           self.results['improved_model']['total'] * 100) if self.results['improved_model']['total'] > 0 else 0
        
        print(f"\n📊 OVERALL ACCURACY:")
        print(f"Original Model:  {original_accuracy:.1f}% ({self.results['original_model']['correct']}/{self.results['original_model']['total']})")
        print(f"Improved Model:  {improved_accuracy:.1f}% ({self.results['improved_model']['correct']}/{self.results['improved_model']['total']})")
        
        difference = improved_accuracy - original_accuracy
        if difference > 2:
            print(f"\n✅ IMPROVEMENT: +{difference:.1f}% better accuracy!")
        elif difference < -2:
            print(f"\n❌ REGRESSION: -{abs(difference):.1f}% worse accuracy")
        else:
            print(f"\n🤝 SIMILAR: ±{abs(difference):.1f}% difference")
        
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
                print(f"  BUY:  {buy_correct}/{buy_total} = {buy_correct/buy_total*100:.1f}%")
            if sell_total > 0:
                print(f"  SELL: {sell_correct}/{sell_total} = {sell_correct/sell_total*100:.1f}%")
            if hold_total > 0:
                print(f"  HOLD: {hold_correct}/{hold_total} = {hold_correct/hold_total*100:.1f}%")

if __name__ == "__main__":
    backtest = ImprovedBacktest()
    backtest.run_backtest()
