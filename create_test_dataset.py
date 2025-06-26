"""
Comprehensive Test Dataset Creator
Generates labeled training data using both technical and fundamental factors
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
import json
import os
from app import app, db, Stock, predictor
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDatasetCreator:
    def __init__(self):
        self.dataset = []
        self.failed_symbols = []
        
    def get_comprehensive_stock_list(self):
        """Get a comprehensive list of stocks for testing across all market caps and sectors"""
        
        # Large Cap Stocks (>$10B market cap)
        large_cap = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM',
            'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'INTU', 'AMAT', 'MU', 'LRCX',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'VRTX',
            'REGN', 'BIIB', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'CVS', 'CI', 'HUM', 'ANTM',
            
            # Financial Services
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
            'PNC', 'TFC', 'COF', 'BK', 'STT', 'SPGI', 'ICE', 'CME', 'MCO', 'MSCI',
            
            # Consumer Discretionary
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'DIS',
            
            # Consumer Staples
            'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'HSY',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL',
            
            # Industrials
            'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX'
        ]
        
        # Mid Cap Stocks ($2B - $10B market cap)
        mid_cap = [
            # Technology
            'SNOW', 'PLTR', 'CRWD', 'ZM', 'DOCU', 'TWLO', 'OKTA', 'DDOG', 'NET', 'FSLY',
            'ESTC', 'MDB', 'TEAM', 'ATLASSIAN', 'WDAY', 'VEEV', 'SPLK', 'NOW', 'PANW', 'FTNT',
            
            # Healthcare & Biotech
            'TDOC', 'VEEV', 'DXCM', 'ISRG', 'ALGN', 'IDXX', 'IQV', 'A', 'TECH', 'HOLX',
            
            # Consumer
            'ROKU', 'PTON', 'BYND', 'UBER', 'LYFT', 'DASH', 'ABNB', 'ETSY', 'CHWY', 'PINS',
            
            # Financial
            'SOFI', 'AFRM', 'UPST', 'LC', 'HOOD', 'COIN', 'SQ', 'PYPL', 'ADYEN', 'SHOP',
            
            # Industrial
            'SPCE', 'RKT', 'OPEN', 'Z', 'RDFN', 'COMP', 'JBHT', 'CHRW', 'EXPD', 'LSTR'
        ]
        
        # Small Cap Stocks ($300M - $2B market cap)
        small_cap = [
            # Growth
            'RBLX', 'RIVN', 'LCID', 'NKLA', 'RIDE', 'GOEV', 'HYLN', 'WKHS', 'SOLO', 'AYRO',
            'BLNK', 'CHPT', 'EVGO', 'BLINK', 'SBE', 'ACTC', 'CCIV', 'IPOE', 'CLOV', 'WISH',
            
            # Traditional
            'AAL', 'UAL', 'DAL', 'LUV', 'JBLU', 'ALK', 'SAVE', 'HA', 'MESA', 'SKYW',
            'CCL', 'RCL', 'NCLH', 'CUK', 'ONEW', 'SIX', 'FUN', 'CNK', 'IMAX', 'LYV',
            
            # REITs
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'UDR', 'CPT',
            
            # Biotech
            'MRNA', 'BNTX', 'NVAX', 'INO', 'SRNE', 'VXRT', 'OCGN', 'CODX', 'IBIO', 'BIOL'
        ]
        
        # Penny Stocks and Volatile Stocks (more likely to have SELL performance)
        penny_stocks = [
            'SNDL', 'NAKD', 'GNUS', 'XSPA', 'SHIP', 'TOPS', 'GLBS', 'CTRM', 'CASTOR', 'JAGX',
            'INPX', 'EXPR', 'AMC', 'GME', 'BB', 'NOK', 'SIRI', 'ZYXI', 'GSAT', 'SENS',
            'CLIS', 'OZSC', 'HMBL', 'ALPP', 'ABML', 'AITX', 'TLSS', 'PASO', 'RTON', 'PHIL',
            # Add more volatile stocks for SELL examples
            'SPCE', 'NKLA', 'RIDE', 'WKHS', 'HYLN', 'GOEV', 'SOLO', 'AYRO', 'BLNK', 'CHPT',
            'LCID', 'RIVN', 'PTON', 'BYND', 'WISH', 'CLOV', 'HOOD', 'RBLX', 'COIN', 'UPST'
        ]
        
        # ETFs for market representation
        etfs = [
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG', 'LQD',
            'HYG', 'EMB', 'TLT', 'IEF', 'SHY', 'GLD', 'SLV', 'USO', 'UNG', 'DBA'
        ]
        
        # Combine all categories
        all_stocks = large_cap + mid_cap + small_cap + penny_stocks + etfs
        
        # Remove duplicates and return
        return list(set(all_stocks))
    
    def extract_comprehensive_features(self, symbol, date_range_days=365):
        """Extract all features (technical + fundamental) for a given symbol"""
        try:
            logger.info(f"Extracting features for {symbol}")
            
            # Get stock from database
            stock = Stock.query.filter_by(symbol=symbol).first()
            if not stock:
                logger.warning(f"Stock {symbol} not found in database")
                return None
            
            # Get historical data from yfinance
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=date_range_days)
            
            hist = ticker.history(start=start_date, end=end_date)
            info = ticker.info
            
            if hist.empty or len(hist) < 50:
                logger.warning(f"Insufficient historical data for {symbol}")
                return None
            
            # Calculate technical indicators using the original algorithm logic
            features = self._calculate_all_technical_indicators(hist)
            
            # Add fundamental features
            fundamental_features = self._extract_fundamental_features(stock, info)
            features.update(fundamental_features)
            
            # Add stock metadata
            features.update({
                'symbol': symbol,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'exchange': info.get('exchange', 'Unknown'),
                'extraction_date': datetime.now().isoformat()
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            self.failed_symbols.append(symbol)
            return None
    
    def _calculate_all_technical_indicators(self, hist):
        """Calculate ALL technical indicators from the original algorithm"""
        close = hist['Close']
        volume = hist['Volume']
        high = hist['High']
        low = hist['Low']
        
        # Basic price and volume data
        current_price = close.iloc[-1]
        avg_volume = volume.rolling(window=20).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1
        
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
        
        # Price momentum
        price_momentum = ((current_price - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) > 21 else 0
        
        # Volatility
        volatility = close.pct_change().rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100
        
        # === ORIGINAL ALGORITHM SCORING COMPONENTS ===
        
        # RSI scoring (EXACT from original)
        rsi_score = 15 if rsi < 30 else (-15 if rsi > 70 else 0)
        
        # Moving average scoring (EXACT from original)
        ma_score = 15 if (current_price > sma_20 > sma_50) else (-15 if (current_price < sma_20 < sma_50) else 0)
        
        # Bollinger Bands scoring (EXACT from original)
        bb_score = 10 if bb_position < 0.2 else (-10 if bb_position > 0.8 else 0)
        
        # MACD scoring (EXACT from original)
        macd_score = min(10, abs(macd) * 2) if macd > 0 else -min(10, abs(macd) * 2)
        
        # Price momentum scoring (EXACT from original)
        momentum_score = 10 if price_momentum > 5 else (-10 if price_momentum < -5 else 0)
        
        # Volume scoring (EXACT from original)
        if volume_ratio > 2:
            volume_score = 15
        elif volume_ratio > 1.5:
            volume_score = 8
        elif volume_ratio < 0.5:
            volume_score = -10
        else:
            volume_score = 0
        
        # Enhanced growth patterns (from original)
        momentum_volume_bonus = 5 if (price_momentum > 5 and volume_ratio > 1.5) else 0
        rsi_growth_bonus = 3 if (50 <= rsi <= 60) else 0
        
        # Trend bonus
        trend_bonus = 0
        if current_price > sma_20 and sma_20 > sma_50:
            ma_separation = (sma_20 - sma_50) / sma_50 * 100 if sma_50 > 0 else 0
            trend_bonus = 4 if ma_separation > 3 else 0
        
        # MACD momentum bonus
        macd_momentum_bonus = 3 if macd > 1.0 else 0
        
        # Calculate total technical score (EXACT from original algorithm)
        total_technical_score = (50 + rsi_score + ma_score + bb_score + macd_score + 
                               momentum_score + volume_score + momentum_volume_bonus + 
                               rsi_growth_bonus + trend_bonus + macd_momentum_bonus)
        total_technical_score = max(0, min(100, total_technical_score))
        
        return {
            # Core data
            'current_price': current_price,
            'volume': volume.iloc[-1],
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
            
            # Technical indicators
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'macd': macd,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_position': bb_position,
            'price_momentum': price_momentum,
            'volatility': volatility,
            
            # Original algorithm scoring components
            'rsi_score': rsi_score,
            'ma_score': ma_score,
            'bb_score': bb_score,
            'macd_score': macd_score,
            'momentum_score': momentum_score,
            'volume_score': volume_score,
            'momentum_volume_bonus': momentum_volume_bonus,
            'rsi_growth_bonus': rsi_growth_bonus,
            'trend_bonus': trend_bonus,
            'macd_momentum_bonus': macd_momentum_bonus,
            'total_technical_score': total_technical_score,
            
            # Additional ratios
            'price_to_sma20': current_price / sma_20 if sma_20 > 0 else 1,
            'price_to_sma50': current_price / sma_50 if sma_50 > 0 else 1,
            'sma20_to_sma50': sma_20 / sma_50 if sma_50 > 0 else 1,
            
            # Boolean conditions
            'is_above_sma20': 1 if current_price > sma_20 else 0,
            'is_above_sma50': 1 if current_price > sma_50 else 0,
            'is_sma20_above_sma50': 1 if sma_20 > sma_50 else 0,
            'is_rsi_oversold': 1 if rsi < 30 else 0,
            'is_rsi_overbought': 1 if rsi > 70 else 0,
            'is_macd_positive': 1 if macd > 0 else 0,
            'is_high_volume': 1 if volume_ratio > 1.5 else 0,
            'is_bb_oversold': 1 if bb_position < 0.2 else 0,
            'is_bb_overbought': 1 if bb_position > 0.8 else 0
        }

    def _extract_fundamental_features(self, stock, info):
        """Extract fundamental features from company_data table and yfinance"""
        features = {}

        # === ALL FACTORS FROM COMPANY_DATA TABLE ===
        if hasattr(stock, 'company_data') and stock.company_data:
            cd = stock.company_data
            features.update({
                'revenue': getattr(cd, 'revenue', 0) or 0,
                'revenue_growth_rate_fwd': getattr(cd, 'revenue_growth_rate_fwd', 0) or 0,
                'revenue_growth_rate_trailing': getattr(cd, 'revenue_growth_rate_trailing', 0) or 0,
                'ebitda': getattr(cd, 'ebitda', 0) or 0,
                'ebitda_growth_rate_fwd': getattr(cd, 'ebitda_growth_rate_fwd', 0) or 0,
                'ebitda_growth_rate_trailing': getattr(cd, 'ebitda_growth_rate_trailing', 0) or 0,
                'depreciation_amortization': getattr(cd, 'depreciation_amortization', 0) or 0,
                'ebit': getattr(cd, 'ebit', 0) or 0,
                'capx': getattr(cd, 'capx', 0) or 0,
                'working_capital': getattr(cd, 'working_capital', 0) or 0,
                'net_debt': getattr(cd, 'net_debt', 0) or 0,
                'levered_fcf': getattr(cd, 'levered_fcf', 0) or 0,
                'wacc': getattr(cd, 'wacc', 0) or 0,
                'debt_to_equity_ratio': getattr(cd, 'debt_to_equity_ratio', 0) or 0,
                'current_ratio': getattr(cd, 'current_ratio', 0) or 0,
                'quick_ratio': getattr(cd, 'quick_ratio', 0) or 0,
                'gross_profit_margin': getattr(cd, 'gross_profit_margin', 0) or 0,
                'pe_ratio': getattr(cd, 'pe_ratio', 0) or 0,
                'eps': getattr(cd, 'eps', 0) or 0
            })
        else:
            # Default values if company_data not available
            features.update({
                'revenue': 0, 'revenue_growth_rate_fwd': 0, 'revenue_growth_rate_trailing': 0,
                'ebitda': 0, 'ebitda_growth_rate_fwd': 0, 'ebitda_growth_rate_trailing': 0,
                'depreciation_amortization': 0, 'ebit': 0, 'capx': 0, 'working_capital': 0,
                'net_debt': 0, 'levered_fcf': 0, 'wacc': 0, 'debt_to_equity_ratio': 0,
                'current_ratio': 0, 'quick_ratio': 0, 'gross_profit_margin': 0,
                'pe_ratio': 0, 'eps': 0
            })

        # === YFINANCE FUNDAMENTAL DATA ===
        features.update({
            'market_cap': info.get('marketCap', 0) or 0,
            'enterprise_value': info.get('enterpriseValue', 0) or 0,
            'trailing_pe': info.get('trailingPE', 0) or 0,
            'forward_pe': info.get('forwardPE', 0) or 0,
            'peg_ratio': info.get('pegRatio', 0) or 0,
            'price_to_book': info.get('priceToBook', 0) or 0,
            'price_to_sales': info.get('priceToSalesTrailing12Months', 0) or 0,
            'enterprise_to_revenue': info.get('enterpriseToRevenue', 0) or 0,
            'enterprise_to_ebitda': info.get('enterpriseToEbitda', 0) or 0,
            'profit_margins': info.get('profitMargins', 0) or 0,
            'operating_margins': info.get('operatingMargins', 0) or 0,
            'return_on_assets': info.get('returnOnAssets', 0) or 0,
            'return_on_equity': info.get('returnOnEquity', 0) or 0,
            'revenue_growth': info.get('revenueGrowth', 0) or 0,
            'earnings_growth': info.get('earningsGrowth', 0) or 0,
            'debt_to_equity': info.get('debtToEquity', 0) or 0,
            'total_cash': info.get('totalCash', 0) or 0,
            'total_debt': info.get('totalDebt', 0) or 0,
            'free_cashflow': info.get('freeCashflow', 0) or 0,
            'operating_cashflow': info.get('operatingCashflow', 0) or 0,
            'beta': info.get('beta', 1.0) or 1.0,
            'shares_outstanding': info.get('sharesOutstanding', 0) or 0,
            'book_value': info.get('bookValue', 0) or 0,
            'held_percent_institutions': info.get('heldPercentInstitutions', 0) or 0,
            'held_percent_insiders': info.get('heldPercentInsiders', 0) or 0
        })

        # === STOCK CATEGORIZATION (from original algorithm) ===
        market_cap = features['market_cap']
        current_price = stock.current_price or 0

        if market_cap > 10_000_000_000:
            features['stock_category'] = 'large_cap'
            features['base_confidence'] = 0.85
        elif market_cap > 2_000_000_000:
            features['stock_category'] = 'mid_cap'
            features['base_confidence'] = 0.75
        elif market_cap > 300_000_000:
            features['stock_category'] = 'small_cap'
            features['base_confidence'] = 0.65
        elif current_price >= 5.0:
            features['stock_category'] = 'micro_cap'
            features['base_confidence'] = 0.50
        elif current_price >= 1.0:
            features['stock_category'] = 'penny'
            features['base_confidence'] = 0.35
        else:
            features['stock_category'] = 'micro_penny'
            features['base_confidence'] = 0.25

        # === SECTOR MULTIPLIERS (from original algorithm) ===
        sector = info.get('sector', 'Unknown')
        sector_multipliers = {
            'Technology': 1.3, 'Automotive': 1.1, 'E-commerce': 1.2,
            'Entertainment': 0.9, 'ETF': 0.7, 'Healthcare': 1.1,
            'Financial Services': 1.0, 'Consumer Discretionary': 1.0,
            'Consumer Staples': 0.9, 'Energy': 0.8, 'Materials': 0.9,
            'Industrials': 1.0, 'Real Estate': 0.8, 'Utilities': 0.7,
            'Communication Services': 1.1
        }
        features['sector_multiplier'] = sector_multipliers.get(sector, 1.0)

        # === BINARY CATEGORY FEATURES ===
        categories = ['large_cap', 'mid_cap', 'small_cap', 'micro_cap', 'penny', 'micro_penny']
        for cat in categories:
            features[f'is_{cat}'] = 1 if features['stock_category'] == cat else 0

        return features

    def create_labels_from_future_performance(self, symbol, features, lookforward_days=30):
        """Create labels based on actual future stock performance"""
        try:
            # Get future price data
            ticker = yf.Ticker(symbol)
            end_date = datetime.now() + timedelta(days=lookforward_days + 5)  # Buffer for weekends
            start_date = datetime.now() - timedelta(days=5)  # Small lookback

            hist = ticker.history(start=start_date, end=end_date)

            if len(hist) < 2:
                return None

            current_price = features['current_price']

            # Try to get price from lookforward_days in the future
            if len(hist) >= lookforward_days:
                future_price = hist['Close'].iloc[min(lookforward_days, len(hist)-1)]
            else:
                future_price = hist['Close'].iloc[-1]  # Use latest available

            # Calculate actual performance
            actual_change_percent = ((future_price - current_price) / current_price) * 100

            # Create 3-class classification labels with more realistic thresholds
            if actual_change_percent > 1.5:
                label = 'BUY'  # >1.5% gain
                label_numeric = 2
            elif actual_change_percent < -0.5:
                label = 'SELL'  # <-0.5% loss
                label_numeric = 0
            else:
                label = 'HOLD'  # -0.5% to 1.5%
                label_numeric = 1

            return {
                'actual_change_percent': actual_change_percent,
                'label': label,
                'label_numeric': label_numeric,
                'future_price': future_price,
                'days_forward': lookforward_days
            }

        except Exception as e:
            logger.error(f"Error creating labels for {symbol}: {e}")
            return None

    def get_algorithm_prediction(self, symbol):
        """Get prediction from the existing rule-based algorithm"""
        try:
            with app.app_context():
                prediction = predictor.predict_stock_movement(symbol)

                if prediction and not prediction.get('error'):
                    # Convert prediction to numeric label
                    pred_text = prediction.get('prediction', 'HOLD')
                    if 'STRONG BUY' in pred_text:
                        pred_numeric = 4
                    elif 'BUY' in pred_text:
                        pred_numeric = 3
                    elif 'HOLD' in pred_text:
                        pred_numeric = 2
                    elif 'SELL' in pred_text:
                        pred_numeric = 1
                    else:
                        pred_numeric = 0

                    return {
                        'algorithm_prediction': pred_text,
                        'algorithm_prediction_numeric': pred_numeric,
                        'algorithm_confidence': prediction.get('confidence', 0),
                        'algorithm_expected_change': prediction.get('expected_change_percent', 0),
                        'algorithm_target_price': prediction.get('target_price', 0)
                    }

        except Exception as e:
            logger.error(f"Error getting algorithm prediction for {symbol}: {e}")

        return {
            'algorithm_prediction': 'HOLD',
            'algorithm_prediction_numeric': 2,
            'algorithm_confidence': 50,
            'algorithm_expected_change': 0,
            'algorithm_target_price': 0
        }

    def create_comprehensive_dataset(self, max_stocks=500, save_to_file=True):
        """Create the complete test dataset with features and labels"""
        logger.info("Starting comprehensive dataset creation...")

        # Get stock symbols
        all_symbols = self.get_comprehensive_stock_list()

        # Limit the number of stocks for testing
        symbols_to_process = all_symbols[:max_stocks]
        logger.info(f"Processing {len(symbols_to_process)} stocks")

        dataset = []
        processed_count = 0

        with app.app_context():
            for i, symbol in enumerate(symbols_to_process):
                try:
                    logger.info(f"Processing {symbol} ({i+1}/{len(symbols_to_process)})")

                    # Extract comprehensive features
                    features = self.extract_comprehensive_features(symbol)
                    if features is None:
                        continue

                    # Get algorithm prediction
                    algorithm_pred = self.get_algorithm_prediction(symbol)
                    features.update(algorithm_pred)

                    # Create labels from future performance
                    labels = self.create_labels_from_future_performance(symbol, features)
                    if labels is None:
                        continue

                    features.update(labels)

                    # Add to dataset
                    dataset.append(features)
                    processed_count += 1

                    # Add small delay to avoid rate limiting
                    time.sleep(0.1)

                    # Save intermediate results every 50 stocks
                    if processed_count % 50 == 0:
                        logger.info(f"Processed {processed_count} stocks successfully")
                        if save_to_file:
                            self._save_intermediate_dataset(dataset, f"dataset_intermediate_{processed_count}.json")

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    self.failed_symbols.append(symbol)
                    continue

        logger.info(f"Dataset creation completed. Processed {processed_count} stocks successfully.")
        logger.info(f"Failed symbols: {len(self.failed_symbols)}")

        if save_to_file:
            self._save_final_dataset(dataset)

        self.dataset = dataset
        return dataset

    def _save_intermediate_dataset(self, dataset, filename):
        """Save intermediate dataset results"""
        try:
            os.makedirs('datasets', exist_ok=True)
            filepath = os.path.join('datasets', filename)

            with open(filepath, 'w') as f:
                json.dump(dataset, f, indent=2, default=str)

            logger.info(f"Intermediate dataset saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving intermediate dataset: {e}")

    def _save_final_dataset(self, dataset):
        """Save the final comprehensive dataset"""
        try:
            os.makedirs('datasets', exist_ok=True)

            # Save as JSON
            json_filepath = 'datasets/comprehensive_stock_dataset.json'
            with open(json_filepath, 'w') as f:
                json.dump(dataset, f, indent=2, default=str)

            # Save as CSV for easier analysis
            df = pd.DataFrame(dataset)
            csv_filepath = 'datasets/comprehensive_stock_dataset.csv'
            df.to_csv(csv_filepath, index=False)

            # Save summary statistics
            summary = {
                'total_stocks': len(dataset),
                'failed_symbols': self.failed_symbols,
                'feature_count': len(dataset[0].keys()) if dataset else 0,
                'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {},
                'creation_date': datetime.now().isoformat()
            }

            summary_filepath = 'datasets/dataset_summary.json'
            with open(summary_filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"Final dataset saved:")
            logger.info(f"  JSON: {json_filepath}")
            logger.info(f"  CSV: {csv_filepath}")
            logger.info(f"  Summary: {summary_filepath}")
            logger.info(f"  Total features: {summary['feature_count']}")
            logger.info(f"  Label distribution: {summary['label_distribution']}")

        except Exception as e:
            logger.error(f"Error saving final dataset: {e}")

def main():
    """Main function to create the test dataset"""
    logger.info("Starting comprehensive test dataset creation")

    creator = TestDatasetCreator()

    # Create the dataset
    dataset = creator.create_comprehensive_dataset(
        max_stocks=100,  # Start with 100 stocks for testing
        save_to_file=True
    )

    logger.info("Dataset creation completed!")
    logger.info(f"Created dataset with {len(dataset)} samples")

    if dataset:
        # Print sample features
        sample = dataset[0]
        logger.info(f"Sample features: {list(sample.keys())}")
        logger.info(f"Total features per stock: {len(sample.keys())}")

if __name__ == "__main__":
    main()
