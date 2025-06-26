#!/usr/bin/env python3
"""
Advanced prediction model with high accuracy targeting
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AdvancedStockPredictor:
    def __init__(self):
        # Expanded test dataset
        self.test_symbols = [
            # Large Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
            # Large Cap Traditional  
            'JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA', 'HD', 'UNH', 'CVX',
            # Mid Cap Growth
            'PLTR', 'RBLX', 'SNOW', 'CRWD', 'ZM', 'ROKU', 'SQ', 'SHOP',
            # ETFs for stability
            'SPY', 'QQQ', 'IWM', 'VTI', 'XLK', 'XLF', 'XLE', 'XLV',
            # Cyclical/Industrial
            'AMD', 'INTC', 'DIS', 'BA', 'CAT', 'GE', 'F', 'GM'
        ]
        
        self.results = {
            'baseline_model': {'correct': 0, 'total': 0, 'predictions': []},
            'advanced_model': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        self.detailed_results = {
            'baseline_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}},
            'advanced_model': {'BUY': {'correct': 0, 'total': 0}, 'SELL': {'correct': 0, 'total': 0}, 'HOLD': {'correct': 0, 'total': 0}}
        }
    
    def get_historical_data(self, symbol, days_back=200):
        try:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            data = stock.history(start=start_date, end=end_date)
            return data if len(data) > 120 else None
        except Exception as e:
            return None
    
    def calculate_advanced_indicators(self, data):
        if len(data) < 60:
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            # Multiple timeframe analysis
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_10 = data['Close'].rolling(10).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            
            ema_12 = data['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['Close'].ewm(span=26).mean().iloc[-1]
            
            # Volume analysis
            avg_volume_10 = data['Volume'].rolling(10).mean().iloc[-1]
            avg_volume_20 = data['Volume'].rolling(20).mean().iloc[-1]
            avg_volume_50 = data['Volume'].rolling(50).mean().iloc[-1]
            
            # Price momentum (multiple timeframes)
            momentum_3 = (current_price - data['Close'].iloc[-3]) / data['Close'].iloc[-3] * 100 if len(data) >= 3 else 0
            momentum_5 = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100 if len(data) >= 5 else 0
            momentum_10 = (current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10] * 100 if len(data) >= 10 else 0
            
            # RSI (multiple periods)
            def calculate_rsi(prices, period):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs)).iloc[-1] if not pd.isna(rs.iloc[-1]) else 50
            
            rsi_14 = calculate_rsi(data['Close'], 14)
            rsi_21 = calculate_rsi(data['Close'], 21)
            
            # MACD
            macd = ema_12 - ema_26
            macd_signal = data['Close'].ewm(span=9).mean().iloc[-1]
            macd_histogram = macd - macd_signal
            
            # Bollinger Bands
            bb_std = data['Close'].rolling(20).std().iloc[-1]
            bollinger_upper = sma_20 + (bb_std * 2)
            bollinger_lower = sma_20 - (bb_std * 2)
            bb_width = (bollinger_upper - bollinger_lower) / sma_20 * 100
            
            # Volatility measures
            volatility_10 = data['Close'].pct_change().rolling(10).std().iloc[-1] * 100
            volatility_20 = data['Close'].pct_change().rolling(20).std().iloc[-1] * 100
            
            # Support/Resistance levels
            recent_high = data['High'].rolling(20).max().iloc[-1]
            recent_low = data['Low'].rolling(20).min().iloc[-1]
            
            # Trend strength
            trend_strength = (sma_5 - sma_20) / sma_20 * 100
            
            return {
                'current_price': current_price,
                'volume': volume,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'avg_volume_10': avg_volume_10,
                'avg_volume_20': avg_volume_20,
                'avg_volume_50': avg_volume_50,
                'momentum_3': momentum_3,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'rsi_14': rsi_14,
                'rsi_21': rsi_21,
                'macd': macd,
                'macd_histogram': macd_histogram,
                'bollinger_upper': bollinger_upper,
                'bollinger_lower': bollinger_lower,
                'bb_width': bb_width,
                'volatility_10': volatility_10,
                'volatility_20': volatility_20,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'trend_strength': trend_strength,
                # Derived indicators
                'volume_ratio_10': volume / avg_volume_10 if avg_volume_10 > 0 else 1,
                'volume_ratio_20': volume / avg_volume_20 if avg_volume_20 > 0 else 1,
                'price_vs_sma20': (current_price - sma_20) / sma_20 * 100,
                'price_vs_sma50': (current_price - sma_50) / sma_50 * 100,
                'bb_position': (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower) if bollinger_upper != bollinger_lower else 0.5,
                'distance_from_high': (recent_high - current_price) / recent_high * 100,
                'distance_from_low': (current_price - recent_low) / recent_low * 100
            }
        except Exception as e:
            return None
    
    def baseline_model(self, indicators):
        """Simple baseline model"""
        score = 50
        
        # Basic RSI
        if indicators['rsi_14'] < 30:
            score += 15
        elif indicators['rsi_14'] > 70:
            score -= 15
        
        # Basic momentum
        if indicators['momentum_5'] > 3:
            score += 10
        elif indicators['momentum_5'] < -3:
            score -= 10
        
        # Basic volume
        if indicators['volume_ratio_20'] > 1.5:
            score += 8
        elif indicators['volume_ratio_20'] < 0.7:
            score -= 8
        
        # Basic trend
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
    
    def advanced_model(self, indicators):
        """Advanced model with sophisticated logic"""
        # Multi-factor scoring system
        buy_score = 0
        sell_score = 0
        hold_score = 0
        
        # === BUY SIGNALS ===
        
        # Strong momentum alignment
        if (indicators['momentum_3'] > 1 and indicators['momentum_5'] > 2 and 
            indicators['momentum_10'] > 3):
            buy_score += 25
        elif indicators['momentum_5'] > 3:
            buy_score += 15
        
        # RSI in sweet spot (not overbought but showing strength)
        if 50 <= indicators['rsi_14'] <= 65:
            buy_score += 20
        elif 45 <= indicators['rsi_14'] <= 70:
            buy_score += 10
        
        # Volume confirmation
        if (indicators['volume_ratio_10'] > 1.3 and indicators['volume_ratio_20'] > 1.1):
            buy_score += 20
        elif indicators['volume_ratio_20'] > 1.2:
            buy_score += 10
        
        # Trend alignment
        if (indicators['current_price'] > indicators['sma_5'] > indicators['sma_10'] > 
            indicators['sma_20'] > indicators['sma_50']):
            buy_score += 25
        elif indicators['current_price'] > indicators['sma_20'] > indicators['sma_50']:
            buy_score += 15
        
        # MACD bullish
        if indicators['macd'] > 0 and indicators['macd_histogram'] > 0:
            buy_score += 15
        
        # Bollinger position (not at extremes)
        if 0.3 <= indicators['bb_position'] <= 0.7:
            buy_score += 10
        
        # Near recent lows (potential bounce)
        if indicators['distance_from_low'] < 5 and indicators['momentum_3'] > 0:
            buy_score += 15
        
        # === SELL SIGNALS ===
        
        # Strong negative momentum
        if (indicators['momentum_3'] < -1 and indicators['momentum_5'] < -2 and 
            indicators['momentum_10'] < -3):
            sell_score += 25
        elif indicators['momentum_5'] < -3:
            sell_score += 15
        
        # RSI showing weakness
        if indicators['rsi_14'] < 35:
            sell_score += 20
        elif indicators['rsi_14'] < 45:
            sell_score += 10
        
        # Volume on decline
        if (indicators['volume_ratio_10'] > 1.2 and indicators['momentum_5'] < -2):
            sell_score += 20
        
        # Trend breakdown
        if (indicators['current_price'] < indicators['sma_5'] < indicators['sma_10'] < 
            indicators['sma_20']):
            sell_score += 25
        elif indicators['current_price'] < indicators['sma_20'] < indicators['sma_50']:
            sell_score += 15
        
        # MACD bearish
        if indicators['macd'] < 0 and indicators['macd_histogram'] < 0:
            sell_score += 15
        
        # Near recent highs (potential reversal)
        if indicators['distance_from_high'] < 3 and indicators['momentum_3'] < 0:
            sell_score += 15
        
        # High volatility with negative momentum
        if indicators['volatility_10'] > 3 and indicators['momentum_5'] < -2:
            sell_score += 10
        
        # === HOLD SIGNALS ===
        
        # Low volatility
        if indicators['volatility_10'] < 1.5:
            hold_score += 15
        
        # Neutral momentum
        if abs(indicators['momentum_5']) < 1:
            hold_score += 15
        
        # RSI in neutral zone
        if 45 <= indicators['rsi_14'] <= 55:
            hold_score += 15
        
        # Price near moving averages
        if abs(indicators['price_vs_sma20']) < 1:
            hold_score += 15
        
        # Normal volume
        if 0.8 <= indicators['volume_ratio_20'] <= 1.2:
            hold_score += 10
        
        # Bollinger squeeze
        if indicators['bb_width'] < 3:
            hold_score += 10
        
        # === DECISION LOGIC ===
        
        # Require minimum confidence
        min_confidence = 40
        
        if buy_score >= min_confidence and buy_score > sell_score and buy_score > hold_score:
            return 'BUY'
        elif sell_score >= min_confidence and sell_score > buy_score and sell_score > hold_score:
            return 'SELL'
        elif hold_score >= min_confidence and hold_score > buy_score and hold_score > sell_score:
            return 'HOLD'
        else:
            # Default to market bias when uncertain
            if buy_score > sell_score:
                return 'BUY'
            elif sell_score > buy_score:
                return 'SELL'
            else:
                return 'HOLD'
    
    def get_actual_outcome(self, data, index, timeframe=15):
        if index + timeframe >= len(data):
            return None
        
        current_price = data['Close'].iloc[index]
        future_price = data['Close'].iloc[index + timeframe]
        change = (future_price - current_price) / current_price * 100
        
        # Adjusted thresholds for clearer classification
        if change > 4:  # Raised threshold for BUY
            return 'BUY', change
        elif change < -4:  # Raised threshold for SELL
            return 'SELL', change
        else:
            return 'HOLD', change
    
    def run_advanced_backtest(self):
        print("🚀 Advanced Model Backtest - Large Dataset")
        print("=" * 60)
        print(f"Testing {len(self.test_symbols)} symbols with 200 days of data each")
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
            
            # Test every 3 days for comprehensive coverage
            for i in range(100, len(data) - 20, 3):
                historical_data = data.iloc[:i+1]
                indicators = self.calculate_advanced_indicators(historical_data)
                
                if indicators is None:
                    continue
                
                # Get predictions
                baseline_pred = self.baseline_model(indicators)
                advanced_pred = self.advanced_model(indicators)
                
                # Get actual outcome
                actual_result = self.get_actual_outcome(data, i)
                if actual_result is None:
                    continue
                
                actual_direction, actual_change = actual_result
                tests_for_symbol += 1
                
                # Record results
                for model_name, prediction in [('baseline_model', baseline_pred), ('advanced_model', advanced_pred)]:
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
        self.print_advanced_results()
    
    def print_advanced_results(self):
        print("\n" + "=" * 60)
        print("🎯 ADVANCED MODEL RESULTS")
        print("=" * 60)
        
        baseline_accuracy = (self.results['baseline_model']['correct'] / 
                           self.results['baseline_model']['total'] * 100) if self.results['baseline_model']['total'] > 0 else 0
        
        advanced_accuracy = (self.results['advanced_model']['correct'] / 
                           self.results['advanced_model']['total'] * 100) if self.results['advanced_model']['total'] > 0 else 0
        
        print(f"\n📊 OVERALL ACCURACY:")
        print(f"Baseline Model:  {baseline_accuracy:.1f}% ({self.results['baseline_model']['correct']}/{self.results['baseline_model']['total']})")
        print(f"Advanced Model:  {advanced_accuracy:.1f}% ({self.results['advanced_model']['correct']}/{self.results['advanced_model']['total']})")
        
        improvement = advanced_accuracy - baseline_accuracy
        if improvement > 2:
            print(f"\n✅ IMPROVEMENT: +{improvement:.1f}% better accuracy!")
        elif improvement < -2:
            print(f"\n❌ REGRESSION: -{abs(improvement):.1f}% worse accuracy")
        else:
            print(f"\n🤝 SIMILAR: ±{abs(improvement):.1f}% difference")
        
        # Detailed breakdown
        print(f"\n📈 DETAILED ACCURACY BY PREDICTION TYPE:")
        for model_name in ['baseline_model', 'advanced_model']:
            print(f"\n{model_name.replace('_', ' ').title()}:")
            
            for pred_type in ['BUY', 'SELL', 'HOLD']:
                stats = self.detailed_results[model_name][pred_type]
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    print(f"  {pred_type}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
                else:
                    print(f"  {pred_type}: No predictions made")
        
        # Target analysis
        print(f"\n🎯 TARGET ANALYSIS:")
        for model_name in ['baseline_model', 'advanced_model']:
            model_results = self.detailed_results[model_name]
            buy_acc = (model_results['BUY']['correct'] / model_results['BUY']['total'] * 100) if model_results['BUY']['total'] > 0 else 0
            sell_acc = (model_results['SELL']['correct'] / model_results['SELL']['total'] * 100) if model_results['SELL']['total'] > 0 else 0
            hold_acc = (model_results['HOLD']['correct'] / model_results['HOLD']['total'] * 100) if model_results['HOLD']['total'] > 0 else 0
            
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  BUY accuracy target (>70%): {'✅' if buy_acc > 70 else '❌'} {buy_acc:.1f}%")
            print(f"  SELL accuracy target (>60%): {'✅' if sell_acc > 60 else '❌'} {sell_acc:.1f}%")
            print(f"  HOLD accuracy target (>50%): {'✅' if hold_acc > 50 else '❌'} {hold_acc:.1f}%")

if __name__ == "__main__":
    predictor = AdvancedStockPredictor()
    predictor.run_advanced_backtest()
