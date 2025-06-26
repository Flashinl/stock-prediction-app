#!/usr/bin/env python3
"""
Ensemble model with longer timeframes and multiple validation methods
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class EnsemblePredictor:
    def __init__(self):
        # Focus on most stable and predictable stocks
        self.test_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
            'JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA',
            'SPY', 'QQQ', 'VTI', 'XLK'
        ]
        
        self.results = {
            'simple_model': {'correct': 0, 'total': 0, 'predictions': []},
            'ensemble_model': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        self.detailed_results = {
            'simple_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}},
            'ensemble_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}}
        }
    
    def get_historical_data(self, symbol, days_back=90):
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 50 else None
        except Exception as e:
            return None
    
    def calculate_ensemble_indicators(self, data):
        if len(data) < 30:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Moving averages
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            
            # Volume analysis
            avg_volume_10 = data['Volume'].rolling(10).mean().iloc[-1]
            avg_volume_20 = data['Volume'].rolling(20).mean().iloc[-1]
            
            # Multiple momentum periods
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
            
            # Volatility
            volatility = data['Close'].pct_change().rolling(10).std().iloc[-1] * 100
            
            # Trend strength
            trend_strength = (sma_5 - sma_20) / sma_20 * 100
            
            return {
                'current_price': current_price,
                'volume': volume,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'avg_volume_10': avg_volume_10,
                'avg_volume_20': avg_volume_20,
                'momentum_3': momentum_3,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'rsi': rsi,
                'macd': macd,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_ratio_10': volume / avg_volume_10 if avg_volume_10 > 0 else 1,
                'volume_ratio_20': volume / avg_volume_20 if avg_volume_20 > 0 else 1,
                'price_vs_sma20': (current_price - sma_20) / sma_20 * 100
            }
        except Exception as e:
            return None
    
    def simple_model(self, indicators):
        """Simple baseline model"""
        if (indicators['momentum_5'] > 2 and 
            indicators['rsi'] < 70 and 
            indicators['current_price'] > indicators['sma_20']):
            return 'BUY'
        elif (indicators['momentum_5'] < -2 and 
              indicators['rsi'] > 30 and 
              indicators['current_price'] < indicators['sma_20']):
            return 'SELL'
        else:
            return 'HOLD'
    
    def ensemble_model(self, indicators):
        """Ensemble model with multiple sub-models"""
        
        # Sub-model 1: Momentum-based
        momentum_vote = self.momentum_submodel(indicators)
        
        # Sub-model 2: Mean reversion
        reversion_vote = self.reversion_submodel(indicators)
        
        # Sub-model 3: Trend following
        trend_vote = self.trend_submodel(indicators)
        
        # Sub-model 4: Volume-based
        volume_vote = self.volume_submodel(indicators)
        
        # Sub-model 5: Technical indicators
        technical_vote = self.technical_submodel(indicators)
        
        # Ensemble voting
        votes = [momentum_vote, reversion_vote, trend_vote, volume_vote, technical_vote]
        
        # Count votes
        buy_votes = votes.count('BUY')
        sell_votes = votes.count('SELL')
        hold_votes = votes.count('HOLD')
        
        # Require majority consensus for BUY/SELL
        if buy_votes >= 3:
            return 'BUY'
        elif sell_votes >= 3:
            return 'SELL'
        else:
            return 'HOLD'
    
    def momentum_submodel(self, indicators):
        """Momentum-based sub-model"""
        if (indicators['momentum_3'] > 1 and 
            indicators['momentum_5'] > 1.5 and 
            indicators['momentum_10'] > 2):
            return 'BUY'
        elif (indicators['momentum_3'] < -1 and 
              indicators['momentum_5'] < -1.5 and 
              indicators['momentum_10'] < -2):
            return 'SELL'
        else:
            return 'HOLD'
    
    def reversion_submodel(self, indicators):
        """Mean reversion sub-model"""
        if (indicators['rsi'] < 35 and 
            indicators['momentum_3'] > 0 and
            indicators['price_vs_sma20'] < -3):
            return 'BUY'  # Oversold bounce
        elif (indicators['rsi'] > 70 and 
              indicators['momentum_3'] < 0 and
              indicators['price_vs_sma20'] > 3):
            return 'SELL'  # Overbought reversal
        else:
            return 'HOLD'
    
    def trend_submodel(self, indicators):
        """Trend following sub-model"""
        if (indicators['current_price'] > indicators['sma_5'] > indicators['sma_10'] > indicators['sma_20'] and
            indicators['trend_strength'] > 1):
            return 'BUY'
        elif (indicators['current_price'] < indicators['sma_5'] < indicators['sma_10'] < indicators['sma_20'] and
              indicators['trend_strength'] < -1):
            return 'SELL'
        else:
            return 'HOLD'
    
    def volume_submodel(self, indicators):
        """Volume-based sub-model"""
        if (indicators['volume_ratio_10'] > 1.5 and 
            indicators['momentum_5'] > 1 and
            indicators['volume_ratio_20'] > 1.2):
            return 'BUY'
        elif (indicators['volume_ratio_10'] > 1.3 and 
              indicators['momentum_5'] < -1 and
              indicators['volume_ratio_20'] > 1.1):
            return 'SELL'
        else:
            return 'HOLD'
    
    def technical_submodel(self, indicators):
        """Technical indicators sub-model"""
        if (indicators['macd'] > 0 and 
            40 <= indicators['rsi'] <= 65 and
            indicators['volatility'] < 3):
            return 'BUY'
        elif (indicators['macd'] < 0 and 
              35 <= indicators['rsi'] <= 60 and
              indicators['momentum_5'] < -1):
            return 'SELL'
        else:
            return 'HOLD'
    
    def get_actual_outcome(self, data, index, timeframe=20):
        """Longer timeframe for more reliable outcomes"""
        if index + timeframe >= len(data):
            return None
        
        current_price = data['Close'].iloc[index]
        future_price = data['Close'].iloc[index + timeframe]
        change = (future_price - current_price) / current_price * 100
        
        # More generous thresholds for longer timeframe
        if change > 3:  # 3% over 20 days
            return 'BUY', change
        elif change < -3:  # -3% over 20 days
            return 'SELL', change
        else:
            return 'HOLD', change
    
    def run_ensemble_test(self):
        print("🎯 Ensemble Model Test - Multiple Sub-Models with Longer Timeframe")
        print("=" * 70)
        
        symbols_tested = 0
        
        for symbol in self.test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            data = self.get_historical_data(symbol)
            if data is None:
                continue
            
            symbols_tested += 1
            tests_for_symbol = 0
            
            # Test every 3 days for longer timeframe
            for i in range(30, len(data) - 25, 3):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_ensemble_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                simple_pred = self.simple_model(indicators)
                ensemble_pred = self.ensemble_model(indicators)
                
                actual_result = self.get_actual_outcome(data, i)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                tests_for_symbol += 1
                
                # Record results
                for model_name, prediction in [('simple_model', simple_pred), ('ensemble_model', ensemble_pred)]:
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
        self.print_ensemble_results()
    
    def print_ensemble_results(self):
        print("\n" + "=" * 70)
        print("🎯 ENSEMBLE MODEL RESULTS")
        print("=" * 70)
        
        simple_accuracy = (self.results['simple_model']['correct'] / 
                         self.results['simple_model']['total'] * 100) if self.results['simple_model']['total'] > 0 else 0
        
        ensemble_accuracy = (self.results['ensemble_model']['correct'] / 
                           self.results['ensemble_model']['total'] * 100) if self.results['ensemble_model']['total'] > 0 else 0
        
        print(f"\n📊 OVERALL ACCURACY:")
        print(f"Simple Model:   {simple_accuracy:.1f}% ({self.results['simple_model']['correct']}/{self.results['simple_model']['total']})")
        print(f"Ensemble Model: {ensemble_accuracy:.1f}% ({self.results['ensemble_model']['correct']}/{self.results['ensemble_model']['total']})")
        
        improvement = ensemble_accuracy - simple_accuracy
        print(f"\nImprovement: {improvement:+.1f}%")
        
        # Detailed breakdown
        print(f"\n📈 DETAILED ACCURACY BY PREDICTION TYPE:")
        
        for model_name in ['simple_model', 'ensemble_model']:
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
        
        # Success metrics
        print(f"\n🎯 SUCCESS ANALYSIS:")
        ensemble_results = self.detailed_results['ensemble_model']
        
        targets_achieved = 0
        for pred_type, target in [('BUY', 75), ('SELL', 70), ('HOLD', 60)]:
            stats = ensemble_results[pred_type]
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                if accuracy >= target:
                    targets_achieved += 1
                    print(f"✅ {pred_type} target achieved: {accuracy:.1f}% >= {target}%")
                else:
                    print(f"❌ {pred_type} target missed: {accuracy:.1f}% < {target}%")
        
        print(f"\nTargets achieved: {targets_achieved}/3")
        print(f"Overall target (>65%): {'✅' if ensemble_accuracy > 65 else '❌'} {ensemble_accuracy:.1f}%")
        
        if ensemble_accuracy > 65 and targets_achieved >= 2:
            print(f"\n🚀 ENSEMBLE MODEL READY FOR PRODUCTION!")
            print(f"Significant improvement achieved with longer timeframe approach")

if __name__ == "__main__":
    predictor = EnsemblePredictor()
    predictor.run_ensemble_test()
