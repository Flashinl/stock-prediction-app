#!/usr/bin/env python3
"""
Deep analysis of successful vs failed prediction patterns to achieve 80%+ accuracy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class SuccessPatternAnalyzer:
    def __init__(self):
        # Focus on most liquid and predictable stocks
        self.test_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA', 'HD',
            'SPY', 'QQQ', 'VTI', 'XLK', 'XLF'
        ]
        
        self.successful_patterns = {'BUY': [], 'SELL': [], 'HOLD': []}
        self.failed_patterns = {'BUY': [], 'SELL': [], 'HOLD': []}
        
        # Track different timeframes
        self.timeframes = [5, 7, 10, 15, 20, 25, 30]
        self.threshold_pairs = [(1.5, -1.5), (2.0, -2.0), (2.5, -2.5), (3.0, -3.0), (4.0, -4.0)]
    
    def get_historical_data(self, symbol, days_back=120):
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 80 else None
        except Exception as e:
            return None
    
    def calculate_comprehensive_indicators(self, data):
        if len(data) < 50:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Multiple moving averages
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            
            ema_5 = data['Close'].ewm(span=5).mean().iloc[-1]
            ema_10 = data['Close'].ewm(span=10).mean().iloc[-1]
            ema_20 = data['Close'].ewm(span=20).mean().iloc[-1]
            
            # Volume analysis
            avg_volume_5 = data['Volume'].rolling(5).mean().iloc[-1]
            avg_volume_10 = data['Volume'].rolling(10).mean().iloc[-1]
            avg_volume_20 = data['Volume'].rolling(20).mean().iloc[-1]
            avg_volume_50 = data['Volume'].rolling(50).mean().iloc[-1]
            
            # Multiple momentum periods
            momentum_1 = (current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100 if len(data) >= 2 else 0
            momentum_2 = (current_price - data['Close'].iloc[-3]) / data['Close'].iloc[-3] * 100 if len(data) >= 3 else 0
            momentum_3 = (current_price - data['Close'].iloc[-4]) / data['Close'].iloc[-4] * 100 if len(data) >= 4 else 0
            momentum_5 = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100 if len(data) >= 6 else 0
            momentum_10 = (current_price - data['Close'].iloc[-11]) / data['Close'].iloc[-11] * 100 if len(data) >= 11 else 0
            momentum_20 = (current_price - data['Close'].iloc[-21]) / data['Close'].iloc[-21] * 100 if len(data) >= 21 else 0
            
            # RSI multiple periods
            def calc_rsi(prices, period):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
            
            rsi_7 = calc_rsi(data['Close'], 7)
            rsi_14 = calc_rsi(data['Close'], 14)
            rsi_21 = calc_rsi(data['Close'], 21)
            
            # MACD variations
            ema_12 = data['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['Close'].ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            macd_signal = data['Close'].ewm(span=9).mean().iloc[-1]
            macd_histogram = macd - macd_signal
            
            # Bollinger Bands multiple periods
            bb_std_20 = data['Close'].rolling(20).std().iloc[-1]
            bb_upper_20 = sma_20 + (bb_std_20 * 2)
            bb_lower_20 = sma_20 - (bb_std_20 * 2)
            
            bb_std_10 = data['Close'].rolling(10).std().iloc[-1]
            bb_upper_10 = sma_10 + (bb_std_10 * 2)
            bb_lower_10 = sma_10 - (bb_std_10 * 2)
            
            # Volatility measures
            volatility_5 = data['Close'].pct_change().rolling(5).std().iloc[-1] * 100
            volatility_10 = data['Close'].pct_change().rolling(10).std().iloc[-1] * 100
            volatility_20 = data['Close'].pct_change().rolling(20).std().iloc[-1] * 100
            
            # Support/Resistance levels
            high_5 = data['High'].rolling(5).max().iloc[-1]
            low_5 = data['Low'].rolling(5).min().iloc[-1]
            high_10 = data['High'].rolling(10).max().iloc[-1]
            low_10 = data['Low'].rolling(10).min().iloc[-1]
            high_20 = data['High'].rolling(20).max().iloc[-1]
            low_20 = data['Low'].rolling(20).min().iloc[-1]
            
            # Advanced indicators
            # Stochastic
            low_14 = data['Low'].rolling(14).min().iloc[-1]
            high_14 = data['High'].rolling(14).max().iloc[-1]
            stoch_k = ((current_price - low_14) / (high_14 - low_14)) * 100 if high_14 != low_14 else 50
            
            # Williams %R
            williams_r = ((high_14 - current_price) / (high_14 - low_14)) * -100 if high_14 != low_14 else -50
            
            # Commodity Channel Index (CCI)
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            sma_tp = typical_price.rolling(20).mean().iloc[-1]
            mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean()))).iloc[-1]
            cci = (typical_price.iloc[-1] - sma_tp) / (0.015 * mad) if mad != 0 else 0
            
            return {
                'current_price': current_price,
                'volume': volume,
                'sma_5': sma_5, 'sma_10': sma_10, 'sma_20': sma_20, 'sma_50': sma_50,
                'ema_5': ema_5, 'ema_10': ema_10, 'ema_20': ema_20,
                'avg_volume_5': avg_volume_5, 'avg_volume_10': avg_volume_10, 'avg_volume_20': avg_volume_20, 'avg_volume_50': avg_volume_50,
                'momentum_1': momentum_1, 'momentum_2': momentum_2, 'momentum_3': momentum_3,
                'momentum_5': momentum_5, 'momentum_10': momentum_10, 'momentum_20': momentum_20,
                'rsi_7': rsi_7, 'rsi_14': rsi_14, 'rsi_21': rsi_21,
                'macd': macd, 'macd_histogram': macd_histogram,
                'bb_upper_20': bb_upper_20, 'bb_lower_20': bb_lower_20,
                'bb_upper_10': bb_upper_10, 'bb_lower_10': bb_lower_10,
                'volatility_5': volatility_5, 'volatility_10': volatility_10, 'volatility_20': volatility_20,
                'high_5': high_5, 'low_5': low_5, 'high_10': high_10, 'low_10': low_10, 'high_20': high_20, 'low_20': low_20,
                'stoch_k': stoch_k, 'williams_r': williams_r, 'cci': cci,
                # Derived indicators
                'volume_ratio_5': volume / avg_volume_5 if avg_volume_5 > 0 else 1,
                'volume_ratio_10': volume / avg_volume_10 if avg_volume_10 > 0 else 1,
                'volume_ratio_20': volume / avg_volume_20 if avg_volume_20 > 0 else 1,
                'volume_ratio_50': volume / avg_volume_50 if avg_volume_50 > 0 else 1,
                'price_vs_sma20': (current_price - sma_20) / sma_20 * 100,
                'price_vs_sma50': (current_price - sma_50) / sma_50 * 100,
                'bb_position_20': (current_price - bb_lower_20) / (bb_upper_20 - bb_lower_20) if bb_upper_20 != bb_lower_20 else 0.5,
                'bb_position_10': (current_price - bb_lower_10) / (bb_upper_10 - bb_lower_10) if bb_upper_10 != bb_lower_10 else 0.5,
                'distance_from_high_5': (high_5 - current_price) / high_5 * 100,
                'distance_from_low_5': (current_price - low_5) / low_5 * 100,
                'distance_from_high_10': (high_10 - current_price) / high_10 * 100,
                'distance_from_low_10': (current_price - low_10) / low_10 * 100,
                'distance_from_high_20': (high_20 - current_price) / high_20 * 100,
                'distance_from_low_20': (current_price - low_20) / low_20 * 100,
                # Trend indicators
                'ma_trend_short': (sma_5 - sma_10) / sma_10 * 100,
                'ma_trend_medium': (sma_10 - sma_20) / sma_20 * 100,
                'ma_trend_long': (sma_20 - sma_50) / sma_50 * 100,
                'ema_trend': (ema_5 - ema_20) / ema_20 * 100,
                # Volume trends
                'volume_trend_short': (avg_volume_5 - avg_volume_10) / avg_volume_10 * 100 if avg_volume_10 > 0 else 0,
                'volume_trend_medium': (avg_volume_10 - avg_volume_20) / avg_volume_20 * 100 if avg_volume_20 > 0 else 0,
                # Momentum consistency
                'momentum_consistency': 1 if (momentum_1 > 0 and momentum_2 > 0 and momentum_3 > 0) or (momentum_1 < 0 and momentum_2 < 0 and momentum_3 < 0) else 0,
                'momentum_acceleration': momentum_1 - momentum_3,
                # Market regime indicators
                'trend_strength': abs(momentum_20),
                'consolidation_score': 1 if (volatility_10 < 2 and abs(momentum_5) < 1) else 0
            }
        except Exception as e:
            return None
    
    def get_actual_outcome_multiple_timeframes(self, data, index):
        """Get actual outcomes for multiple timeframes"""
        outcomes = {}
        
        for timeframe in self.timeframes:
            if index + timeframe >= len(data):
                continue
            
            current_price = data['Close'].iloc[index]
            future_price = data['Close'].iloc[index + timeframe]
            change = (future_price - current_price) / current_price * 100
            
            outcomes[timeframe] = change
        
        return outcomes
    
    def classify_outcome(self, change, buy_threshold, sell_threshold):
        """Classify outcome based on thresholds"""
        if change > buy_threshold:
            return 'BUY'
        elif change < sell_threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def analyze_success_patterns(self):
        print("🔍 Deep Analysis of Successful vs Failed Prediction Patterns")
        print("=" * 70)
        print("Goal: Identify patterns that lead to 80%+ accuracy")
        print("=" * 70)
        
        all_data_points = []
        
        for symbol in self.test_symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            data = self.get_historical_data(symbol)
            if data is None:
                continue
            
            symbol_points = 0
            
            # Test every 2 days for comprehensive coverage
            for i in range(60, len(data) - 35, 2):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_comprehensive_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                # Get actual outcomes for all timeframes
                outcomes = self.get_actual_outcome_multiple_timeframes(data, i)
                if not outcomes:
                    continue
                
                # Store data point with all information
                data_point = {
                    'symbol': symbol,
                    'indicators': indicators,
                    'outcomes': outcomes
                }
                
                all_data_points.append(data_point)
                symbol_points += 1
            
            print(f"  Collected {symbol_points} data points for {symbol}")
        
        print(f"\n✅ Total data points collected: {len(all_data_points)}")
        
        # Analyze patterns for each timeframe and threshold combination
        self.find_optimal_combinations(all_data_points)
    
    def find_optimal_combinations(self, data_points):
        """Find optimal timeframe and threshold combinations for 80%+ accuracy"""
        print(f"\n🎯 Finding Optimal Combinations for 80%+ Accuracy")
        print("=" * 60)
        
        best_combinations = []
        
        for timeframe in self.timeframes:
            for buy_thresh, sell_thresh in self.threshold_pairs:
                print(f"\nTesting: {timeframe} days, {buy_thresh}%/{sell_thresh}% thresholds")
                
                # Filter data points that have this timeframe
                valid_points = [dp for dp in data_points if timeframe in dp['outcomes']]
                if len(valid_points) < 50:  # Need minimum data
                    continue
                
                # Test different prediction strategies
                strategies = [
                    self.momentum_strategy,
                    self.rsi_strategy,
                    self.volume_strategy,
                    self.trend_strategy,
                    self.volatility_strategy,
                    self.combined_strategy
                ]
                
                for strategy in strategies:
                    accuracy_results = self.test_strategy(strategy, valid_points, timeframe, buy_thresh, sell_thresh)
                    
                    if accuracy_results['overall_accuracy'] >= 80:
                        best_combinations.append({
                            'strategy': strategy.__name__,
                            'timeframe': timeframe,
                            'buy_threshold': buy_thresh,
                            'sell_threshold': sell_thresh,
                            'results': accuracy_results
                        })
                        
                        print(f"  🚀 FOUND 80%+ ACCURACY: {strategy.__name__}")
                        print(f"     Overall: {accuracy_results['overall_accuracy']:.1f}%")
                        print(f"     BUY: {accuracy_results['buy_accuracy']:.1f}%")
                        print(f"     SELL: {accuracy_results['sell_accuracy']:.1f}%")
                        print(f"     HOLD: {accuracy_results['hold_accuracy']:.1f}%")
        
        # Report best combinations
        if best_combinations:
            print(f"\n🎉 FOUND {len(best_combinations)} COMBINATIONS WITH 80%+ ACCURACY!")
            self.report_best_combinations(best_combinations)
        else:
            print(f"\n⚠️ No combinations achieved 80%+ accuracy yet")
            print("Analyzing patterns for further optimization...")
            self.analyze_near_misses(data_points)
    
    def momentum_strategy(self, indicators):
        """Momentum-based prediction strategy"""
        if (indicators['momentum_3'] > 2 and indicators['momentum_5'] > 2.5 and 
            indicators['momentum_consistency'] == 1 and indicators['rsi_14'] < 70):
            return 'BUY'
        elif (indicators['momentum_3'] < -2 and indicators['momentum_5'] < -2.5 and 
              indicators['momentum_consistency'] == 1 and indicators['rsi_14'] > 30):
            return 'SELL'
        else:
            return 'HOLD'
    
    def rsi_strategy(self, indicators):
        """RSI-based prediction strategy"""
        if (indicators['rsi_14'] < 35 and indicators['momentum_2'] > 0.5 and 
            indicators['distance_from_low_10'] < 5):
            return 'BUY'
        elif (indicators['rsi_14'] > 70 and indicators['momentum_2'] < -0.5 and 
              indicators['distance_from_high_10'] < 3):
            return 'SELL'
        else:
            return 'HOLD'
    
    def volume_strategy(self, indicators):
        """Volume-based prediction strategy"""
        if (indicators['volume_ratio_5'] > 1.5 and indicators['momentum_3'] > 1 and 
            indicators['volume_trend_short'] > 10):
            return 'BUY'
        elif (indicators['volume_ratio_5'] > 1.3 and indicators['momentum_3'] < -1 and 
              indicators['volume_trend_short'] > 5):
            return 'SELL'
        else:
            return 'HOLD'
    
    def trend_strategy(self, indicators):
        """Trend-based prediction strategy"""
        if (indicators['ma_trend_short'] > 1 and indicators['ma_trend_medium'] > 0.5 and 
            indicators['current_price'] > indicators['sma_20']):
            return 'BUY'
        elif (indicators['ma_trend_short'] < -1 and indicators['ma_trend_medium'] < -0.5 and 
              indicators['current_price'] < indicators['sma_20']):
            return 'SELL'
        else:
            return 'HOLD'
    
    def volatility_strategy(self, indicators):
        """Volatility-based prediction strategy"""
        if (indicators['volatility_10'] < 2 and indicators['momentum_5'] > 1.5 and 
            indicators['bb_position_20'] < 0.8):
            return 'BUY'
        elif (indicators['volatility_10'] > 3 and indicators['momentum_5'] < -1.5 and 
              indicators['bb_position_20'] > 0.2):
            return 'SELL'
        else:
            return 'HOLD'
    
    def combined_strategy(self, indicators):
        """Combined multi-factor strategy"""
        buy_signals = 0
        sell_signals = 0
        
        # Momentum signals
        if indicators['momentum_3'] > 1.5 and indicators['momentum_5'] > 2:
            buy_signals += 2
        elif indicators['momentum_3'] < -1.5 and indicators['momentum_5'] < -2:
            sell_signals += 2
        
        # RSI signals
        if 40 <= indicators['rsi_14'] <= 60:
            buy_signals += 1
        elif indicators['rsi_14'] < 35:
            sell_signals += 1
        
        # Volume signals
        if indicators['volume_ratio_10'] > 1.3:
            if indicators['momentum_3'] > 0:
                buy_signals += 1
            else:
                sell_signals += 1
        
        # Trend signals
        if indicators['ma_trend_short'] > 0.5:
            buy_signals += 1
        elif indicators['ma_trend_short'] < -0.5:
            sell_signals += 1
        
        # Volatility signals
        if indicators['volatility_10'] < 2.5:
            buy_signals += 1
        
        if buy_signals >= 4:
            return 'BUY'
        elif sell_signals >= 3:
            return 'SELL'
        else:
            return 'HOLD'
    
    def test_strategy(self, strategy, data_points, timeframe, buy_thresh, sell_thresh):
        """Test a strategy and return accuracy results"""
        predictions = []
        
        for dp in data_points:
            prediction = strategy(dp['indicators'])
            actual_change = dp['outcomes'][timeframe]
            actual = self.classify_outcome(actual_change, buy_thresh, sell_thresh)
            
            predictions.append({
                'predicted': prediction,
                'actual': actual,
                'correct': prediction == actual
            })
        
        # Calculate accuracies
        total = len(predictions)
        correct = sum(1 for p in predictions if p['correct'])
        
        buy_preds = [p for p in predictions if p['predicted'] == 'BUY']
        sell_preds = [p for p in predictions if p['predicted'] == 'SELL']
        hold_preds = [p for p in predictions if p['predicted'] == 'HOLD']
        
        buy_accuracy = (sum(1 for p in buy_preds if p['correct']) / len(buy_preds) * 100) if buy_preds else 0
        sell_accuracy = (sum(1 for p in sell_preds if p['correct']) / len(sell_preds) * 100) if sell_preds else 0
        hold_accuracy = (sum(1 for p in hold_preds if p['correct']) / len(hold_preds) * 100) if hold_preds else 0
        
        return {
            'overall_accuracy': correct / total * 100,
            'buy_accuracy': buy_accuracy,
            'sell_accuracy': sell_accuracy,
            'hold_accuracy': hold_accuracy,
            'total_predictions': total,
            'buy_count': len(buy_preds),
            'sell_count': len(sell_preds),
            'hold_count': len(hold_preds)
        }
    
    def report_best_combinations(self, combinations):
        """Report the best performing combinations"""
        print(f"\n🏆 TOP PERFORMING COMBINATIONS (80%+ Accuracy)")
        print("=" * 70)
        
        # Sort by overall accuracy
        combinations.sort(key=lambda x: x['results']['overall_accuracy'], reverse=True)
        
        for i, combo in enumerate(combinations[:5], 1):  # Top 5
            print(f"\n#{i} - {combo['strategy']}")
            print(f"   Timeframe: {combo['timeframe']} days")
            print(f"   Thresholds: {combo['buy_threshold']}% / {combo['sell_threshold']}%")
            print(f"   Overall Accuracy: {combo['results']['overall_accuracy']:.1f}%")
            print(f"   BUY: {combo['results']['buy_accuracy']:.1f}% ({combo['results']['buy_count']} predictions)")
            print(f"   SELL: {combo['results']['sell_accuracy']:.1f}% ({combo['results']['sell_count']} predictions)")
            print(f"   HOLD: {combo['results']['hold_accuracy']:.1f}% ({combo['results']['hold_count']} predictions)")
        
        # Save best combination for implementation
        best = combinations[0]
        print(f"\n🚀 IMPLEMENTING BEST STRATEGY: {best['strategy']}")
        return best
    
    def analyze_near_misses(self, data_points):
        """Analyze combinations that came close to 80% for optimization hints"""
        print(f"\n🔍 Analyzing Near-Miss Patterns (70-79% accuracy)")
        print("=" * 60)
        
        # This would contain logic to find patterns that almost work
        # and suggest modifications to push them over 80%
        pass

if __name__ == "__main__":
    analyzer = SuccessPatternAnalyzer()
    analyzer.analyze_success_patterns()
