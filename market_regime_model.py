#!/usr/bin/env python3
"""
Market regime detection model - different strategies for different market conditions
Target: 80%+ accuracy by adapting to market regimes
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MarketRegimePredictor:
    def __init__(self):
        # Focus on liquid stocks and ETFs
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ', 'NVDA', 'TSLA', 'META']
        
        self.results = {
            'simple_model': {'correct': 0, 'total': 0, 'predictions': []},
            'regime_model': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        self.detailed_results = {
            'simple_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}},
            'regime_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}}
        }
        
        self.regime_stats = {'BULL': 0, 'BEAR': 0, 'SIDEWAYS': 0}
    
    def get_data_with_retry(self, symbol, days_back=60):
        """Get data with retry logic"""
        for attempt in range(3):
            try:
                stock = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                data = stock.history(start=start_date, end=end_date)
                
                if len(data) > 40:
                    return data
                else:
                    print(f"  Insufficient data for {symbol} (attempt {attempt + 1})")
                    
            except Exception as e:
                print(f"  Error fetching {symbol} (attempt {attempt + 1}): {e}")
                
            if attempt < 2:  # Wait before retry
                import time
                time.sleep(1)
        
        return None
    
    def calculate_simple_indicators(self, data):
        """Calculate essential indicators only"""
        if len(data) < 20:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Simple moving averages
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            
            # Volume
            avg_volume = data['Volume'].rolling(10).mean().iloc[-1]
            
            # Momentum
            momentum_3 = (current_price - data['Close'].iloc[-4]) / data['Close'].iloc[-4] * 100 if len(data) >= 4 else 0
            momentum_5 = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100 if len(data) >= 6 else 0
            momentum_10 = (current_price - data['Close'].iloc[-11]) / data['Close'].iloc[-11] * 100 if len(data) >= 11 else 0
            
            # RSI (simplified)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
            
            # Volatility
            volatility = data['Close'].pct_change().rolling(10).std().iloc[-1] * 100
            
            return {
                'current_price': current_price,
                'volume': volume,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'avg_volume': avg_volume,
                'momentum_3': momentum_3,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': volume / avg_volume if avg_volume > 0 else 1,
                'price_vs_sma20': (current_price - sma_20) / sma_20 * 100
            }
        except Exception as e:
            print(f"  Error calculating indicators: {e}")
            return None
    
    def detect_market_regime(self, data, indicators):
        """Detect current market regime"""
        if len(data) < 20:
            return 'SIDEWAYS'
        
        try:
            # Calculate trend over different periods
            price_20_days_ago = data['Close'].iloc[-21] if len(data) >= 21 else data['Close'].iloc[0]
            price_10_days_ago = data['Close'].iloc[-11] if len(data) >= 11 else data['Close'].iloc[0]
            current_price = data['Close'].iloc[-1]
            
            # Long-term trend
            long_trend = (current_price - price_20_days_ago) / price_20_days_ago * 100
            
            # Medium-term trend
            medium_trend = (current_price - price_10_days_ago) / price_10_days_ago * 100
            
            # Volatility factor
            volatility = indicators['volatility']
            
            # Regime classification
            if long_trend > 3 and medium_trend > 1.5 and volatility < 4:
                return 'BULL'
            elif long_trend < -3 and medium_trend < -1.5:
                return 'BEAR'
            else:
                return 'SIDEWAYS'
                
        except Exception as e:
            return 'SIDEWAYS'
    
    def simple_model(self, indicators):
        """Simple baseline model"""
        if indicators['momentum_5'] > 2 and indicators['rsi'] < 70:
            return 'BUY'
        elif indicators['momentum_5'] < -2 and indicators['rsi'] > 30:
            return 'SELL'
        else:
            return 'HOLD'
    
    def regime_adaptive_model(self, indicators, regime):
        """Regime-adaptive prediction model"""
        
        if regime == 'BULL':
            # Bull market strategy - favor BUY, avoid SELL
            if (indicators['momentum_3'] > 1 and 
                indicators['rsi'] < 75 and 
                indicators['current_price'] > indicators['sma_10']):
                return 'BUY'
            elif (indicators['rsi'] > 80 and 
                  indicators['momentum_3'] < -1 and
                  indicators['volume_ratio'] > 1.5):
                return 'SELL'  # Only sell on extreme overbought with volume
            else:
                return 'HOLD'
        
        elif regime == 'BEAR':
            # Bear market strategy - favor SELL, be cautious with BUY
            if (indicators['momentum_3'] < -1 and 
                indicators['rsi'] > 25 and 
                indicators['current_price'] < indicators['sma_10']):
                return 'SELL'
            elif (indicators['rsi'] < 25 and 
                  indicators['momentum_3'] > 1 and
                  indicators['volume_ratio'] > 1.3):
                return 'BUY'  # Only buy on extreme oversold with volume
            else:
                return 'HOLD'
        
        else:  # SIDEWAYS
            # Sideways market strategy - range trading
            if (indicators['rsi'] < 35 and 
                indicators['momentum_3'] > 0.5 and
                indicators['price_vs_sma20'] < -2):
                return 'BUY'  # Buy oversold bounces
            elif (indicators['rsi'] > 65 and 
                  indicators['momentum_3'] < -0.5 and
                  indicators['price_vs_sma20'] > 2):
                return 'SELL'  # Sell overbought reversals
            elif (abs(indicators['momentum_5']) < 1 and 
                  45 <= indicators['rsi'] <= 55 and
                  indicators['volatility'] < 2):
                return 'HOLD'  # True consolidation
            else:
                return 'HOLD'  # Default to hold in sideways
    
    def get_actual_outcome(self, data, index, timeframe=10):
        """Get actual outcome with shorter timeframe"""
        if index + timeframe >= len(data):
            return None
        
        current_price = data['Close'].iloc[index]
        future_price = data['Close'].iloc[index + timeframe]
        change = (future_price - current_price) / current_price * 100
        
        # More achievable thresholds
        if change > 2:
            return 'BUY', change
        elif change < -2:
            return 'SELL', change
        else:
            return 'HOLD', change
    
    def run_regime_test(self):
        print("🎯 Market Regime Adaptive Model Test")
        print("=" * 50)
        print("Goal: Achieve 80%+ accuracy by adapting to market conditions")
        print("=" * 50)
        
        symbols_tested = 0
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            data = self.get_data_with_retry(symbol)
            if data is None:
                print(f"  ❌ Could not get data for {symbol}")
                continue
            
            symbols_tested += 1
            tests_for_symbol = 0
            
            # Test every 2 days
            for i in range(25, len(data) - 12, 2):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_simple_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                # Detect market regime
                regime = self.detect_market_regime(historical_data, indicators)
                self.regime_stats[regime] += 1
                
                # Get predictions
                simple_pred = self.simple_model(indicators)
                regime_pred = self.regime_adaptive_model(indicators, regime)
                
                # Get actual outcome
                actual_result = self.get_actual_outcome(data, i)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                tests_for_symbol += 1
                
                # Record results
                for model_name, prediction in [('simple_model', simple_pred), ('regime_model', regime_pred)]:
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
                        'regime': regime,
                        'correct': prediction == actual_direction
                    })
            
            print(f"  Completed {tests_for_symbol} tests for {symbol}")
        
        print(f"\n✅ Tested {symbols_tested} symbols")
        self.print_regime_results()
    
    def print_regime_results(self):
        print("\n" + "=" * 50)
        print("🎯 MARKET REGIME MODEL RESULTS")
        print("=" * 50)
        
        simple_accuracy = (self.results['simple_model']['correct'] / 
                         self.results['simple_model']['total'] * 100) if self.results['simple_model']['total'] > 0 else 0
        
        regime_accuracy = (self.results['regime_model']['correct'] / 
                         self.results['regime_model']['total'] * 100) if self.results['regime_model']['total'] > 0 else 0
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"Simple Model:  {simple_accuracy:.1f}% ({self.results['simple_model']['correct']}/{self.results['simple_model']['total']})")
        print(f"Regime Model:  {regime_accuracy:.1f}% ({self.results['regime_model']['correct']}/{self.results['regime_model']['total']})")
        
        improvement = regime_accuracy - simple_accuracy
        print(f"Improvement:   {improvement:+.1f}%")
        
        # Market regime distribution
        total_regimes = sum(self.regime_stats.values())
        if total_regimes > 0:
            print(f"\n📈 MARKET REGIME DISTRIBUTION:")
            for regime, count in self.regime_stats.items():
                percentage = count / total_regimes * 100
                print(f"  {regime}: {percentage:.1f}% ({count} periods)")
        
        # Detailed accuracy by prediction type
        print(f"\n📈 DETAILED ACCURACY BY PREDICTION TYPE:")
        
        for model_name in ['simple_model', 'regime_model']:
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
        
        # Regime-specific analysis
        if self.results['regime_model']['predictions']:
            print(f"\n🎯 REGIME-SPECIFIC PERFORMANCE:")
            for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
                regime_preds = [p for p in self.results['regime_model']['predictions'] if p['regime'] == regime]
                if regime_preds:
                    correct = sum(1 for p in regime_preds if p['correct'])
                    accuracy = correct / len(regime_preds) * 100
                    print(f"  {regime}: {accuracy:.1f}% ({correct}/{len(regime_preds)})")
        
        # Success analysis
        print(f"\n🎯 80% ACCURACY TARGET ANALYSIS:")
        regime_results = self.detailed_results['regime_model']
        
        targets_achieved = 0
        for pred_type in ['BUY', 'SELL', 'HOLD']:
            stats = regime_results[pred_type]
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
        print(f"Overall target (>80%): {'✅' if regime_accuracy >= 80 else '❌'} {regime_accuracy:.1f}%")
        
        if regime_accuracy >= 80 and targets_achieved >= 2:
            print(f"\n🚀 SUCCESS! REGIME MODEL ACHIEVED 80%+ ACCURACY!")
            print(f"Ready for production implementation")
        elif regime_accuracy >= 70:
            print(f"\n⚠️ CLOSE! Need {80 - regime_accuracy:.1f}% more accuracy")
            print(f"Consider refining regime detection or prediction logic")
        else:
            print(f"\n❌ NEED MORE WORK: {80 - regime_accuracy:.1f}% accuracy gap")

if __name__ == "__main__":
    predictor = MarketRegimePredictor()
    predictor.run_regime_test()
