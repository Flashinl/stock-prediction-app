#!/usr/bin/env python3
"""
Comprehensive backtest with larger dataset to achieve high accuracy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class ComprehensiveBacktest:
    def __init__(self):
        # Expanded dataset with more diverse stocks
        self.test_symbols = [
            # Large Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            # Large Cap Traditional
            'JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA',
            # Mid Cap
            'PLTR', 'RBLX', 'SNOW', 'CRWD', 'ZM', 'ROKU', 'SQ',
            # ETFs for stability
            'SPY', 'QQQ', 'IWM', 'VTI', 'XLK', 'XLF',
            # Volatile/Cyclical
            'AMD', 'INTC', 'NFLX', 'DIS', 'BA', 'CAT', 'GE'
        ]
        
        self.results = {
            'current_model': {'correct': 0, 'total': 0, 'predictions': []},
            'improved_model': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        # Track performance by prediction type
        self.detailed_results = {
            'current_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}},
            'improved_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}}
        }
    
    def get_historical_data(self, symbol, days_back=180):
        """Get more historical data for comprehensive testing"""
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 100 else None
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        if len(data) < 50:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            
            # Price momentum (multiple timeframes)
            price_momentum_5 = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100 if len(data) >= 5 else 0
            price_momentum_10 = (current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10] * 100 if len(data) >= 10 else 0
            price_momentum = (price_momentum_5 + price_momentum_10) / 2
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
            
            # Moving averages
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
            ema_12 = data['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['Close'].ewm(span=26).mean().iloc[-1]
            
            # MACD
            macd = ema_12 - ema_26
            macd_signal = data['Close'].ewm(span=9).mean().iloc[-1]
            macd_histogram = macd - macd_signal
            
            # Bollinger Bands
            bb_period = 20
            bb_std = data['Close'].rolling(bb_period).std().iloc[-1]
            bb_middle = sma_20
            bollinger_upper = bb_middle + (bb_std * 2)
            bollinger_lower = bb_middle - (bb_std * 2)
            
            # Additional indicators
            # Stochastic
            low_14 = data['Low'].rolling(14).min().iloc[-1]
            high_14 = data['High'].rolling(14).max().iloc[-1]
            stoch_k = ((current_price - low_14) / (high_14 - low_14)) * 100 if high_14 != low_14 else 50
            
            # Williams %R
            williams_r = ((high_14 - current_price) / (high_14 - low_14)) * -100 if high_14 != low_14 else -50
            
            return {
                'current_price': current_price,
                'volume': volume,
                'avg_volume': avg_volume,
                'price_momentum': price_momentum,
                'rsi': rsi,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'macd': macd,
                'macd_histogram': macd_histogram,
                'bollinger_upper': bollinger_upper,
                'bollinger_lower': bollinger_lower,
                'stoch_k': stoch_k,
                'williams_r': williams_r
            }
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None
    
    def calculate_momentum_strength(self, indicators):
        """Enhanced momentum strength calculation"""
        price_momentum = indicators.get('price_momentum', 0)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_histogram = indicators.get('macd_histogram', 0)
        stoch_k = indicators.get('stoch_k', 50)
        
        momentum_score = 0
        
        # Price momentum component (-1 to 1)
        momentum_score += max(-1, min(1, price_momentum / 10))
        
        # Volume momentum component (-0.5 to 0.5)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 1.5:
            momentum_score += 0.3
        elif volume_ratio < 0.7:
            momentum_score -= 0.3
        
        # RSI momentum component (-0.3 to 0.3)
        if rsi > 60:
            momentum_score += 0.2
        elif rsi < 40:
            momentum_score -= 0.2
        
        # MACD components (-0.4 to 0.4)
        momentum_score += max(-0.2, min(0.2, macd / 5))
        momentum_score += max(-0.2, min(0.2, macd_histogram / 3))
        
        # Stochastic component (-0.2 to 0.2)
        if stoch_k > 70:
            momentum_score += 0.1
        elif stoch_k < 30:
            momentum_score -= 0.1
        
        return momentum_score
    
    def current_model_prediction(self, indicators):
        """Current model from app.py (baseline)"""
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
        
        # Current scoring logic
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
        
        # Current thresholds (from app.py)
        if score >= 70:
            return 'BUY', score
        elif score <= 35:
            return 'SELL', score
        else:
            return 'HOLD', score
    
    def improved_model_prediction(self, indicators):
        """Improved model with better logic"""
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
        stoch_k = indicators.get('stoch_k', 50)
        williams_r = indicators.get('williams_r', -50)
        
        score = 50  # Start neutral
        
        # Enhanced scoring with additional indicators
        # RSI component (enhanced)
        if rsi < 25:
            score += 20
        elif rsi < 30:
            score += 15
        elif rsi > 75:
            score -= 20
        elif rsi > 70:
            score -= 15
        
        # Moving average component (enhanced)
        if current_price > sma_20 > sma_50:
            ma_strength = (current_price - sma_50) / sma_50 * 100
            if ma_strength > 5:
                score += 20
            else:
                score += 15
        elif current_price < sma_20 < sma_50:
            ma_weakness = (sma_50 - current_price) / sma_50 * 100
            if ma_weakness > 5:
                score -= 20
            else:
                score -= 15
        
        # Volume component (enhanced)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 3:
            score += 20
        elif volume_ratio > 2:
            score += 15
        elif volume_ratio > 1.5:
            score += 8
        elif volume_ratio < 0.3:
            score -= 15
        elif volume_ratio < 0.5:
            score -= 10
        
        # Bollinger Bands (enhanced)
        bb_position = (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower)
        if bb_position < 0.1:
            score += 15
        elif bb_position < 0.2:
            score += 10
        elif bb_position > 0.9:
            score -= 15
        elif bb_position > 0.8:
            score -= 10
        
        # MACD (enhanced)
        if macd > 0:
            score += min(15, abs(macd) * 3)
        else:
            score -= min(15, abs(macd) * 3)
        
        # Price momentum (enhanced)
        if price_momentum > 8:
            score += 15
        elif price_momentum > 5:
            score += 10
        elif price_momentum < -8:
            score -= 15
        elif price_momentum < -5:
            score -= 10
        
        # Additional indicators
        # Stochastic
        if stoch_k < 20:
            score += 8
        elif stoch_k > 80:
            score -= 8
        
        # Williams %R
        if williams_r < -80:
            score += 8
        elif williams_r > -20:
            score -= 8
        
        # Enhanced decision logic
        momentum_strength = self.calculate_momentum_strength(indicators)
        
        # BUY logic (enhanced thresholds)
        if score >= 75:
            return 'BUY', score
        
        # SELL logic (enhanced)
        elif score <= 30:
            # Check for strong sell confirmation
            sell_confirmations = 0
            if current_price < sma_20 < sma_50:
                sell_confirmations += 1
            if rsi < 35 and price_momentum < -3:
                sell_confirmations += 1
            if volume_ratio > 1.5 and price_momentum < -2:
                sell_confirmations += 1
            if momentum_strength < -0.5:
                sell_confirmations += 1
            if stoch_k < 25 and williams_r < -75:
                sell_confirmations += 1
            
            if sell_confirmations >= 3:
                return 'SELL', score
            else:
                return 'HOLD', score
        
        # HOLD logic (very restrictive)
        elif 45 <= score <= 65:
            # Very strict consolidation requirements
            consolidation_signals = 0
            if abs(current_price - sma_20) / sma_20 < 0.015:
                consolidation_signals += 1
            if 47 <= rsi <= 53:
                consolidation_signals += 1
            if abs(price_momentum) < 1:
                consolidation_signals += 1
            if 0.85 <= volume_ratio <= 1.15:
                consolidation_signals += 1
            if abs(momentum_strength) < 0.1:
                consolidation_signals += 1
            if 40 <= stoch_k <= 60:
                consolidation_signals += 1
            
            if consolidation_signals >= 5:
                return 'HOLD', score
            elif score >= 55:
                return 'BUY', score
            else:
                if momentum_strength < -0.2:
                    return 'SELL', score
                else:
                    return 'BUY', score
        
        # Edge cases - be more decisive
        else:
            if score > 65:  # 66-74
                return 'BUY', score
            else:  # 31-44
                if momentum_strength < -0.3 or (price_momentum < -3 and volume_ratio > 1.3):
                    return 'SELL', score
                else:
                    return 'BUY', score
    
    def test_prediction_accuracy(self, prediction_date_index, actual_data, timeframe=15):
        """Test prediction accuracy with configurable timeframe"""
        if prediction_date_index + timeframe >= len(actual_data):
            return None
        
        current_price = actual_data['Close'].iloc[prediction_date_index]
        future_price = actual_data['Close'].iloc[prediction_date_index + timeframe]
        
        actual_change = (future_price - current_price) / current_price * 100
        
        # Adjusted thresholds for better classification
        if actual_change > 3:  # Raised threshold for BUY
            actual_direction = 'BUY'
        elif actual_change < -3:  # Raised threshold for SELL
            actual_direction = 'SELL'
        else:
            actual_direction = 'HOLD'
        
        return actual_direction, actual_change
    
    def run_comprehensive_backtest(self):
        """Run comprehensive backtest with larger dataset"""
        print("🧪 Comprehensive Backtest - Large Dataset")
        print("=" * 60)
        print(f"Testing {len(self.test_symbols)} symbols with 180 days of data each")
        print("=" * 60)
        
        total_symbols_tested = 0
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            data = self.get_historical_data(symbol, days_back=180)
            if data is None or len(data) < 100:
                print(f"❌ Insufficient data for {symbol}")
                continue
            
            total_symbols_tested += 1
            symbol_tests = 0
            
            # Test every 2 days for more comprehensive coverage
            for i in range(80, len(data) - 20, 2):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_technical_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                # Get predictions from both models
                current_pred, current_score = self.current_model_prediction(indicators)
                improved_pred, improved_score = self.improved_model_prediction(indicators)
                
                # Get actual result
                actual_result = self.test_prediction_accuracy(i, data, timeframe=15)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                symbol_tests += 1
                
                # Record results for both models
                for model_name, prediction in [('current_model', current_pred), ('improved_model', improved_pred)]:
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
            
            print(f"  Completed {symbol_tests} tests for {symbol}")
        
        print(f"\n✅ Tested {total_symbols_tested} symbols")
        self.print_comprehensive_results()
    
    def print_comprehensive_results(self):
        """Print detailed results"""
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE BACKTEST RESULTS")
        print("=" * 60)
        
        current_accuracy = (self.results['current_model']['correct'] / 
                          self.results['current_model']['total'] * 100) if self.results['current_model']['total'] > 0 else 0
        
        improved_accuracy = (self.results['improved_model']['correct'] / 
                           self.results['improved_model']['total'] * 100) if self.results['improved_model']['total'] > 0 else 0
        
        print(f"\n📊 OVERALL ACCURACY:")
        print(f"Current Model:   {current_accuracy:.1f}% ({self.results['current_model']['correct']}/{self.results['current_model']['total']})")
        print(f"Improved Model:  {improved_accuracy:.1f}% ({self.results['improved_model']['correct']}/{self.results['improved_model']['total']})")
        
        difference = improved_accuracy - current_accuracy
        if difference > 2:
            print(f"\n✅ IMPROVEMENT: +{difference:.1f}% better accuracy!")
        elif difference < -2:
            print(f"\n❌ REGRESSION: -{abs(difference):.1f}% worse accuracy")
        else:
            print(f"\n🤝 SIMILAR: ±{abs(difference):.1f}% difference")
        
        # Detailed breakdown by prediction type
        print(f"\n📈 DETAILED PREDICTION ACCURACY:")
        for model_name in ['current_model', 'improved_model']:
            print(f"\n{model_name.replace('_', ' ').title()}:")
            
            for pred_type in ['BUY', 'SELL', 'HOLD']:
                stats = self.detailed_results[model_name][pred_type]
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    print(f"  {pred_type}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
                else:
                    print(f"  {pred_type}: No predictions made")

if __name__ == "__main__":
    backtest = ComprehensiveBacktest()
    backtest.run_comprehensive_backtest()
