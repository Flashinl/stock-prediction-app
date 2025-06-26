"""
BACKUP: Original Rule-Based Stock Predictor
This is the original algorithm that achieved 42.86% accuracy
Preserved for reference and potential future use
"""

import yfinance as yf
import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OriginalRuleBasedPredictor:
    """Original rule-based stock prediction algorithm (42.86% accuracy)"""
    
    def __init__(self):
        self._prediction_cache = {}
        self._cache_timeout = 300  # 5 minutes cache
        
    def get_stock_category(self, stock_data):
        """Categorize stock based on market cap and price"""
        market_cap = stock_data.get('market_cap', 0) or 0
        price = stock_data.get('current_price', 0) or 0
        
        if market_cap > 10_000_000_000:  # > $10B
            return 'large_cap'
        elif market_cap > 2_000_000_000:  # $2B - $10B
            return 'mid_cap'
        elif market_cap > 300_000_000:  # $300M - $2B
            return 'small_cap'
        elif price >= 5.0:  # >= $5
            return 'micro_cap'
        elif price >= 1.0:  # $1 - $5
            return 'penny'
        else:  # < $1
            return 'micro_penny'
    
    def calculate_base_confidence(self, category):
        """Calculate base confidence based on stock category"""
        confidence_map = {
            'large_cap': 0.85,
            'mid_cap': 0.75,
            'small_cap': 0.65,
            'micro_cap': 0.50,
            'penny': 0.35,
            'micro_penny': 0.25
        }
        return confidence_map.get(category, 0.50)
    
    def get_sector_multiplier(self, sector):
        """Get sector-based multiplier for predictions"""
        sector_multipliers = {
            'Technology': 1.3,
            'Automotive': 1.1,
            'E-commerce': 1.2,
            'Entertainment': 0.9,
            'ETF': 0.7,
            'Healthcare': 1.1,
            'Financial Services': 1.0,
            'Consumer Discretionary': 1.0,
            'Consumer Staples': 0.9,
            'Energy': 0.8,
            'Materials': 0.9,
            'Industrials': 1.0,
            'Real Estate': 0.8,
            'Utilities': 0.7,
            'Communication Services': 1.1
        }
        return sector_multipliers.get(sector, 1.0)
    
    def calculate_technical_indicators(self, symbol):
        """Calculate technical indicators for stock analysis"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                return None
                
            close = hist['Close']
            volume = hist['Volume']
            high = hist['High']
            low = hist['Low']
            
            current_price = close.iloc[-1]
            
            # Moving averages
            sma_20 = close.rolling(window=20).mean().iloc[-1]
            sma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else sma_20
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            ema_12 = close.ewm(span=12).mean().iloc[-1]
            ema_26 = close.ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            
            # Bollinger Bands
            bb_middle = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            bb_upper = (bb_middle + (bb_std * 2)).iloc[-1]
            bb_lower = (bb_middle - (bb_std * 2)).iloc[-1]
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            
            # Volume analysis
            avg_volume = volume.rolling(window=20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1
            
            # Price momentum
            price_momentum = ((current_price - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) > 21 else 0
            
            return {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'macd': macd,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_position': bb_position,
                'volume_ratio': volume_ratio,
                'price_momentum': price_momentum,
                'avg_volume': avg_volume,
                'volume': volume.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None
    
    def _calculate_comprehensive_score(self, indicators):
        """Calculate comprehensive technical score (original algorithm)"""
        score = 50  # Neutral starting point
        
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        volume_ratio = indicators.get('volume_ratio', 1)
        bb_position = indicators.get('bb_position', 0.5)
        price_momentum = indicators.get('price_momentum', 0)
        
        # RSI analysis (EXACT match from original)
        if rsi < 30:
            score += 15
        elif rsi > 70:
            score -= 15
        
        # Moving average analysis (EXACT match from original)
        if current_price > sma_20 > sma_50:
            score += 15
        elif current_price < sma_20 < sma_50:
            score -= 15
        
        # Volume analysis (EXACT match from original)
        if volume_ratio > 2:
            score += 15
        elif volume_ratio > 1.5:
            score += 8
        elif volume_ratio < 0.5:
            score -= 10
        
        # Bollinger Bands analysis (EXACT match from original)
        if bb_position < 0.2:
            score += 10
        elif bb_position > 0.8:
            score -= 10
        
        # MACD analysis (EXACT match from original)
        if macd > 0:
            score += min(10, abs(macd) * 2)
        else:
            score -= min(10, abs(macd) * 2)
        
        # Price momentum analysis (EXACT match from original)
        if price_momentum > 5:
            score += 10
        elif price_momentum < -5:
            score -= 10
        
        # Enhanced growth patterns (from original algorithm)
        if price_momentum > 5 and volume_ratio > 1.5:
            score += 5  # Strong upward momentum with volume confirmation
        
        if 50 <= rsi <= 60:
            score += 3  # RSI in healthy growth range
        
        if current_price > sma_20 and sma_20 > sma_50:
            ma_separation = (sma_20 - sma_50) / sma_50 * 100 if sma_50 > 0 else 0
            if ma_separation > 3:
                score += 4  # Strong trend with significant separation
        
        if macd > 1.0:
            score += 3  # MACD strong positive momentum
        
        # Ensure score stays within bounds
        return max(0, min(100, score))
    
    def predict_stock_movement(self, symbol, timeframe_override=None):
        """Main prediction function (original algorithm)"""
        try:
            # Get technical indicators
            indicators = self.calculate_technical_indicators(symbol)
            if not indicators:
                return {'error': 'Could not fetch stock data'}
            
            # Get stock info
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Determine category
            stock_data = {
                'market_cap': info.get('marketCap', 0),
                'current_price': indicators['current_price']
            }
            category = self.get_stock_category(stock_data)
            
            # Calculate base confidence
            base_confidence = self.calculate_base_confidence(category)
            
            # Get sector multiplier
            sector = info.get('sector', 'Unknown')
            sector_multiplier = self.get_sector_multiplier(sector)
            
            # Calculate comprehensive score
            score = self._calculate_comprehensive_score(indicators)
            
            # Apply sector multiplier
            adjusted_score = score * sector_multiplier
            adjusted_score = max(0, min(100, adjusted_score))
            
            # Generate prediction based on score thresholds (original logic)
            if adjusted_score >= 85:
                prediction = "STRONG BUY"
                confidence = min(95, base_confidence * 100 + 10)
                expected_change = 8 + (adjusted_score - 85) * 0.5
            elif adjusted_score >= 75:
                prediction = "STRONG BUY"
                confidence = min(90, base_confidence * 100 + 5)
                expected_change = 5 + (adjusted_score - 75) * 0.3
            elif adjusted_score >= 65:
                prediction = "BUY"
                confidence = min(85, base_confidence * 100)
                expected_change = 3 + (adjusted_score - 65) * 0.2
            elif adjusted_score >= 52:
                prediction = "SPECULATIVE BUY"
                confidence = min(75, base_confidence * 100 - 5)
                expected_change = 1 + (adjusted_score - 52) * 0.15
            elif adjusted_score >= 45:
                prediction = "HOLD"
                confidence = min(70, base_confidence * 100 - 10)
                expected_change = 0
            elif adjusted_score >= 35:
                prediction = "SELL"
                confidence = min(75, base_confidence * 100 - 5)
                expected_change = -2 - (45 - adjusted_score) * 0.2
            else:
                prediction = "STRONG SELL"
                confidence = min(80, base_confidence * 100)
                expected_change = -5 - (35 - adjusted_score) * 0.3
            
            # Calculate target price
            current_price = indicators['current_price']
            target_price = current_price * (1 + expected_change / 100)
            
            return {
                'symbol': symbol,
                'prediction': prediction,
                'confidence': round(confidence, 1),
                'expected_change_percent': round(expected_change, 2),
                'target_price': round(target_price, 2),
                'current_price': round(current_price, 2),
                'technical_score': round(adjusted_score, 1),
                'category': category,
                'sector': sector,
                'timeframe': timeframe_override or 'auto',
                'model_type': 'Original Rule-Based Algorithm'
            }
            
        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            return {'error': str(e)}

# Create instance for backward compatibility
original_predictor = OriginalRuleBasedPredictor()
