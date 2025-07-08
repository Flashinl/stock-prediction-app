"""
Neural Network Stock Prediction Model (PyTorch)
Integrates technical indicators with fundamental analysis factors
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
from datetime import datetime
import yfinance as yf
from app import db, Stock
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class StockNeuralNetwork(nn.Module):
    """PyTorch Neural Network for Stock Prediction"""

    def __init__(self, input_dim, num_classes=5):
        super(StockNeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            # Input layer with dropout for regularization
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            # Hidden layers with decreasing size
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer for 5 classes
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

class StockNeuralNetworkPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        self.device = device
        
    def extract_features(self, symbol):
        """Extract comprehensive features for a stock symbol"""
        try:
            # Get stock data from database
            stock = Stock.query.filter_by(symbol=symbol).first()
            if not stock:
                logger.warning(f"Stock {symbol} not found in database")
                return None
                
            # Get real-time data from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            info = ticker.info
            
            if hist.empty:
                logger.warning(f"No historical data for {symbol}")
                return None
                
            # Calculate technical indicators
            technical_features = self._calculate_technical_indicators(hist)
            
            # Extract fundamental features from company_data table
            fundamental_features = self._extract_fundamental_features(stock, info)
            
            # Combine all features
            features = {**technical_features, **fundamental_features}
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            return None
    
    def _calculate_technical_indicators(self, hist):
        """Calculate ALL technical indicators used in the original algorithm + additional ones"""
        close = hist['Close']
        volume = hist['Volume']
        high = hist['High']
        low = hist['Low']

        # === EXACT REPLICA OF ORIGINAL ALGORITHM INDICATORS ===

        # Moving averages (EXACT match from original)
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        ema_12_series = close.ewm(span=12).mean()
        ema_26_series = close.ewm(span=26).mean()
        ema_12 = ema_12_series.iloc[-1]
        ema_26 = ema_26_series.iloc[-1]

        # RSI (EXACT match from original)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # MACD (EXACT match from original)
        macd_series = ema_12_series - ema_26_series
        macd = macd_series.iloc[-1]
        macd_signal_series = macd_series.ewm(span=9).mean()
        macd_signal = macd_signal_series.iloc[-1]
        macd_histogram = (macd_series - macd_signal_series).iloc[-1]

        # Bollinger Bands (EXACT match from original)
        bb_middle = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_upper = (bb_middle + (bb_std * 2)).iloc[-1]
        bb_lower = (bb_middle - (bb_std * 2)).iloc[-1]
        bb_position = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

        # Price momentum (EXACT match from original)
        price_momentum = ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) > 21 else 0

        # Volume analysis (EXACT match from original)
        avg_volume = volume.rolling(window=20).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1

        # Trend strength calculation (from original)
        if len(close) > 50:
            trend_strength = abs((sma_20 - sma_50) / sma_50 * 100) if sma_50 > 0 else 0
        else:
            trend_strength = 0

        # Volume trend (from original)
        if len(volume) > 10:
            recent_avg_volume = volume.rolling(window=10).mean().iloc[-1]
            older_avg_volume = volume.rolling(window=10).mean().iloc[-11] if len(volume) > 20 else recent_avg_volume
            volume_trend = ((recent_avg_volume - older_avg_volume) / older_avg_volume * 100) if older_avg_volume > 0 else 0
        else:
            volume_trend = 0

        # === ORIGINAL ALGORITHM SCORING FACTORS ===

        # RSI scoring (EXACT from original algorithm)
        rsi_score = 0
        if rsi < 30:
            rsi_score = 15
        elif rsi > 70:
            rsi_score = -15

        # Moving average scoring (EXACT from original algorithm)
        ma_score = 0
        if close.iloc[-1] > sma_20 > sma_50:
            ma_score = 15
        elif close.iloc[-1] < sma_20 < sma_50:
            ma_score = -15

        # Bollinger Bands scoring (EXACT from original algorithm)
        bb_score = 0
        if bb_position < 0.2:
            bb_score = 10
        elif bb_position > 0.8:
            bb_score = -10

        # MACD scoring (EXACT from original algorithm)
        macd_score = 0
        if macd > 0:
            macd_score = min(10, abs(macd) * 2)
        else:
            macd_score = -min(10, abs(macd) * 2)

        # Price momentum scoring (EXACT from original algorithm)
        momentum_score = 0
        if price_momentum > 5:
            momentum_score = 10
        elif price_momentum < -5:
            momentum_score = -10

        # Volume scoring (EXACT from original algorithm)
        volume_score = 0
        if volume_ratio > 2:
            volume_score = 15
        elif volume_ratio > 1.5:
            volume_score = 8
        elif volume_ratio < 0.5:
            volume_score = -10

        # === ENHANCED GROWTH PATTERNS (from original algorithm) ===

        # Strong upward momentum with volume confirmation
        momentum_volume_bonus = 0
        if price_momentum > 5 and volume_ratio > 1.5:
            momentum_volume_bonus = 5

        # RSI in healthy growth range
        rsi_growth_bonus = 0
        if 50 <= rsi <= 60:
            rsi_growth_bonus = 3

        # Strong trend with significant separation
        trend_bonus = 0
        if close.iloc[-1] > sma_20 and sma_20 > sma_50:
            ma_separation = (sma_20 - sma_50) / sma_50 * 100 if sma_50 > 0 else 0
            if ma_separation > 3:
                trend_bonus = 4

        # MACD strong positive momentum
        macd_momentum_bonus = 0
        if macd > 1.0:
            macd_momentum_bonus = 3

        # Calculate total technical score (EXACT from original algorithm)
        total_technical_score = (50 + rsi_score + ma_score + bb_score + macd_score +
                               momentum_score + volume_score + momentum_volume_bonus +
                               rsi_growth_bonus + trend_bonus + macd_momentum_bonus)
        total_technical_score = max(0, min(100, total_technical_score))

        # === ADDITIONAL TECHNICAL INDICATORS ===

        # Volatility
        volatility = close.pct_change().rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100 if len(close) > 20 else 0

        # Additional price ratios
        price_to_sma20 = close.iloc[-1] / sma_20 if sma_20 > 0 else 1
        price_to_sma50 = close.iloc[-1] / sma_50 if sma_50 > 0 else 1
        sma20_to_sma50 = sma_20 / sma_50 if sma_50 > 0 else 1

        # Price position in recent range
        if len(high) > 20:
            recent_high = high.rolling(window=20).max().iloc[-1]
            recent_low = low.rolling(window=20).min().iloc[-1]
            price_position = ((close.iloc[-1] - recent_low) / (recent_high - recent_low)) if (recent_high - recent_low) > 0 else 0.5
        else:
            price_position = 0.5

        # Stochastic oscillator
        if len(high) > 14:
            lowest_low = low.rolling(window=14).min().iloc[-1]
            highest_high = high.rolling(window=14).max().iloc[-1]
            stochastic_k = ((close.iloc[-1] - lowest_low) / (highest_high - lowest_low) * 100) if (highest_high - lowest_low) > 0 else 50
        else:
            stochastic_k = 50

        return {
            # === CORE PRICE AND VOLUME DATA ===
            'current_price': close.iloc[-1],
            'volume': volume.iloc[-1],
            'avg_volume': avg_volume,

            # === ORIGINAL ALGORITHM TECHNICAL INDICATORS ===
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_position': bb_position,
            'price_momentum': price_momentum,
            'volume_ratio': volume_ratio,
            'trend_strength': trend_strength,
            'volume_trend': volume_trend,

            # === ORIGINAL ALGORITHM SCORING COMPONENTS ===
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

            # === ADDITIONAL TECHNICAL INDICATORS ===
            'volatility': volatility,
            'price_to_sma20': price_to_sma20,
            'price_to_sma50': price_to_sma50,
            'sma20_to_sma50': sma20_to_sma50,
            'price_position': price_position,
            'stochastic_k': stochastic_k,

            # === BOOLEAN TECHNICAL CONDITIONS ===
            'is_above_sma20': 1 if close.iloc[-1] > sma_20 else 0,
            'is_above_sma50': 1 if close.iloc[-1] > sma_50 else 0,
            'is_sma20_above_sma50': 1 if sma_20 > sma_50 else 0,
            'is_rsi_oversold': 1 if rsi < 30 else 0,
            'is_rsi_overbought': 1 if rsi > 70 else 0,
            'is_macd_positive': 1 if macd > 0 else 0,
            'is_high_volume': 1 if volume_ratio > 1.5 else 0,
            'is_bb_oversold': 1 if bb_position < 0.2 else 0,
            'is_bb_overbought': 1 if bb_position > 0.8 else 0
        }
    
    def _extract_fundamental_features(self, stock, info):
        """Extract ALL fundamental analysis features from company_data table and yfinance info"""
        features = {}

        # === ALL FACTORS FROM COMPANY_DATA TABLE ===
        if hasattr(stock, 'company_data') and stock.company_data:
            cd = stock.company_data
            features.update({
                # Revenue metrics
                'revenue': getattr(cd, 'revenue', 0) or 0,
                'revenue_growth_rate_fwd': getattr(cd, 'revenue_growth_rate_fwd', 0) or 0,
                'revenue_growth_rate_trailing': getattr(cd, 'revenue_growth_rate_trailing', 0) or 0,

                # EBITDA metrics
                'ebitda': getattr(cd, 'ebitda', 0) or 0,
                'ebitda_growth_rate_fwd': getattr(cd, 'ebitda_growth_rate_fwd', 0) or 0,
                'ebitda_growth_rate_trailing': getattr(cd, 'ebitda_growth_rate_trailing', 0) or 0,

                # Operating metrics
                'depreciation_amortization': getattr(cd, 'depreciation_amortization', 0) or 0,
                'ebit': getattr(cd, 'ebit', 0) or 0,
                'capx': getattr(cd, 'capx', 0) or 0,
                'working_capital': getattr(cd, 'working_capital', 0) or 0,

                # Debt and cash flow metrics
                'net_debt': getattr(cd, 'net_debt', 0) or 0,
                'levered_fcf': getattr(cd, 'levered_fcf', 0) or 0,
                'wacc': getattr(cd, 'wacc', 0) or 0,

                # Financial ratios
                'debt_to_equity_ratio': getattr(cd, 'debt_to_equity_ratio', 0) or 0,
                'current_ratio': getattr(cd, 'current_ratio', 0) or 0,
                'quick_ratio': getattr(cd, 'quick_ratio', 0) or 0,
                'gross_profit_margin': getattr(cd, 'gross_profit_margin', 0) or 0,

                # Valuation metrics
                'pe_ratio': getattr(cd, 'pe_ratio', 0) or 0,
                'eps': getattr(cd, 'eps', 0) or 0
            })
        else:
            # Set default values if company_data not available
            features.update({
                'revenue': 0, 'revenue_growth_rate_fwd': 0, 'revenue_growth_rate_trailing': 0,
                'ebitda': 0, 'ebitda_growth_rate_fwd': 0, 'ebitda_growth_rate_trailing': 0,
                'depreciation_amortization': 0, 'ebit': 0, 'capx': 0, 'working_capital': 0,
                'net_debt': 0, 'levered_fcf': 0, 'wacc': 0, 'debt_to_equity_ratio': 0,
                'current_ratio': 0, 'quick_ratio': 0, 'gross_profit_margin': 0,
                'pe_ratio': 0, 'eps': 0
            })
        
        # From yfinance info (backup/additional data)
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
            'float_shares': info.get('floatShares', 0) or 0,
            'shares_short': info.get('sharesShort', 0) or 0,
            'short_ratio': info.get('shortRatio', 0) or 0,
            'book_value': info.get('bookValue', 0) or 0,
            'price_to_book': info.get('priceToBook', 0) or 0,
            'held_percent_institutions': info.get('heldPercentInstitutions', 0) or 0,
            'held_percent_insiders': info.get('heldPercentInsiders', 0) or 0
        })
        
        # === ORIGINAL ALGORITHM SECTOR MULTIPLIERS ===
        sector = info.get('sector', 'Unknown')
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
        features['sector_multiplier'] = sector_multipliers.get(sector, 1.0)

        # === ORIGINAL ALGORITHM CONFIDENCE ADJUSTMENTS ===

        # Base confidence by category (from original algorithm)
        if features['market_cap'] > 10_000_000_000:  # Large cap
            features['base_confidence'] = 0.85
            stock_category = 'large_cap'
        elif features['market_cap'] > 2_000_000_000:  # Mid cap
            features['base_confidence'] = 0.75
            stock_category = 'mid_cap'
        elif features['market_cap'] > 300_000_000:  # Small cap
            features['base_confidence'] = 0.65
            stock_category = 'small_cap'
        elif stock.current_price >= 5.0:  # Micro cap
            features['base_confidence'] = 0.50
            stock_category = 'micro_cap'
        elif stock.current_price >= 1.0:  # Penny stock
            features['base_confidence'] = 0.35
            stock_category = 'penny'
        else:  # Micro penny
            features['base_confidence'] = 0.25
            stock_category = 'micro_penny'

        # Beta confidence adjustment (from original algorithm)
        beta = features['beta']
        features['beta_confidence_adj'] = 0
        if beta < 0.8:
            features['beta_confidence_adj'] = 0.05
        elif beta > 1.5:
            features['beta_confidence_adj'] = -0.10
        elif beta > 2.0:
            features['beta_confidence_adj'] = -0.20

        # Calculate additional derived ratios
        if features['market_cap'] > 0 and features['total_debt'] > 0:
            features['debt_to_market_cap'] = features['total_debt'] / features['market_cap']
        else:
            features['debt_to_market_cap'] = 0

        if features['free_cashflow'] != 0 and features['market_cap'] > 0:
            features['fcf_yield'] = features['free_cashflow'] / features['market_cap']
        else:
            features['fcf_yield'] = 0

        # === STOCK CATEGORY BINARY FEATURES ===
        features['is_large_cap'] = 1 if stock_category == 'large_cap' else 0
        features['is_mid_cap'] = 1 if stock_category == 'mid_cap' else 0
        features['is_small_cap'] = 1 if stock_category == 'small_cap' else 0
        features['is_micro_cap'] = 1 if stock_category == 'micro_cap' else 0
        features['is_penny_stock'] = 1 if stock_category == 'penny' else 0
        features['is_micro_penny'] = 1 if stock_category == 'micro_penny' else 0

        # === SECTOR BINARY FEATURES ===
        major_sectors = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Discretionary',
                        'Consumer Staples', 'Energy', 'Materials', 'Industrials', 'Real Estate',
                        'Utilities', 'Communication Services']

        for sector_name in major_sectors:
            features[f'is_sector_{sector_name.lower().replace(" ", "_")}'] = 1 if sector == sector_name else 0

        # === FUNDAMENTAL RATIOS AND HEALTH SCORES ===

        # Profitability score
        profitability_score = 0
        if features['profit_margins'] > 0.15:
            profitability_score += 2
        elif features['profit_margins'] > 0.10:
            profitability_score += 1
        if features['operating_margins'] > 0.20:
            profitability_score += 2
        elif features['operating_margins'] > 0.15:
            profitability_score += 1
        features['profitability_score'] = profitability_score

        # Growth score
        growth_score = 0
        if features['revenue_growth'] > 0.20:
            growth_score += 3
        elif features['revenue_growth'] > 0.10:
            growth_score += 2
        elif features['revenue_growth'] > 0.05:
            growth_score += 1
        if features['earnings_growth'] > 0.15:
            growth_score += 2
        elif features['earnings_growth'] > 0.10:
            growth_score += 1
        features['growth_score'] = growth_score

        # Financial health score
        financial_health = 0
        if features['current_ratio'] > 2.0:
            financial_health += 2
        elif features['current_ratio'] > 1.5:
            financial_health += 1
        if features['debt_to_equity'] < 0.3:
            financial_health += 2
        elif features['debt_to_equity'] < 0.6:
            financial_health += 1
        if features['return_on_equity'] > 0.15:
            financial_health += 2
        elif features['return_on_equity'] > 0.10:
            financial_health += 1
        features['financial_health_score'] = financial_health

        # Valuation score (lower is better for value)
        valuation_score = 0
        if 0 < features['trailing_pe'] < 15:
            valuation_score += 2
        elif 15 <= features['trailing_pe'] < 25:
            valuation_score += 1
        if 0 < features['price_to_book'] < 2:
            valuation_score += 2
        elif 2 <= features['price_to_book'] < 3:
            valuation_score += 1
        features['valuation_score'] = valuation_score

        return features
    
    def prepare_training_data(self, symbols_list, lookback_days=90):
        """Prepare training data by collecting features and labels for multiple stocks"""
        logger.info(f"Preparing training data for {len(symbols_list)} symbols")

        training_data = []
        labels = []

        # Import app here to avoid circular imports
        from app import app

        with app.app_context():
            for symbol in symbols_list:
                try:
                    logger.info(f"Processing {symbol}...")

                    # Get current features
                    features = self.extract_features(symbol)
                    if features is None:
                        continue

                    # Get historical price data to create labels
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="6mo")  # Get 6 months of data

                    if len(hist) < lookback_days + 30:  # Need enough data for lookback + future
                        continue

                    # Create labels based on future price movement (30 days ahead)
                    for i in range(len(hist) - lookback_days - 30):
                        current_price = hist['Close'].iloc[i + lookback_days]
                        future_price = hist['Close'].iloc[i + lookback_days + 30]

                        price_change = (future_price - current_price) / current_price * 100

                        # Create classification labels
                        if price_change > 8:
                            label = 'STRONG_BUY'  # >8% gain
                        elif price_change > 3:
                            label = 'BUY'  # 3-8% gain
                        elif price_change > -3:
                            label = 'HOLD'  # -3% to 3%
                        elif price_change > -8:
                            label = 'SELL'  # -8% to -3% loss
                        else:
                            label = 'STRONG_SELL'  # >8% loss

                        training_data.append(features)
                        labels.append(label)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue

        logger.info(f"Collected {len(training_data)} training samples")
        return training_data, labels

    def build_model(self, input_dim):
        """Build the PyTorch neural network"""
        model = StockNeuralNetwork(input_dim, num_classes=5)
        model = model.to(self.device)
        return model

    def train(self, symbols_list, epochs=100, validation_split=0.2, save_model=True, learning_rate=0.001):
        """Train the PyTorch neural network model"""
        logger.info("Starting PyTorch neural network training...")

        # Prepare training data
        training_data, labels = self.prepare_training_data(symbols_list)

        if len(training_data) == 0:
            raise ValueError("No training data collected")

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(training_data)

        # Handle missing values
        df = df.fillna(0)

        # Check for non-numeric columns and convert them
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.warning(f"Found non-numeric column: {col}, sample values: {df[col].head().tolist()}")
                # Try to convert to numeric, if it fails, drop the column
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
                except:
                    logger.warning(f"Dropping non-numeric column: {col}")
                    df = df.drop(columns=[col])

        # Store feature names
        self.feature_names = df.columns.tolist()
        logger.info(f"Training with {len(self.feature_names)} features")

        # Prepare features and labels
        X = df.values
        y = self.label_encoder.fit_transform(labels)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Build model
        self.model = self.build_model(X_scaled.shape[1])

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        # Training loop
        logger.info("Training PyTorch neural network...")
        best_accuracy = 0.0
        patience_counter = 0
        patience = 15

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_test_tensor).sum().item() / len(y_test_tensor)

            train_accuracy = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Train Acc: {train_accuracy:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Acc: {val_accuracy:.4f}")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                # Save best model
                if save_model:
                    self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy = accuracy_score(y_test, test_predicted.cpu().numpy())

        logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")

        # Generate classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, test_predicted.cpu().numpy(), target_names=class_names)
        logger.info(f"Classification Report:\n{report}")

        # Confusion matrix
        cm = confusion_matrix(y_test, test_predicted.cpu().numpy())
        logger.info(f"Confusion Matrix:\n{cm}")

        self.is_trained = True

        return {
            'best_accuracy': best_accuracy,
            'final_accuracy': test_accuracy,
            'epochs_trained': epoch + 1
        }

    def predict(self, symbol):
        """Make prediction for a single stock"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Import app here to avoid circular imports
        from app import app

        with app.app_context():
            # Extract features
            features = self.extract_features(symbol)
            if features is None:
                return None

            # Convert to DataFrame and ensure same feature order
            df = pd.DataFrame([features])

            # Add missing features with 0 values
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0

            # Reorder columns to match training data
            df = df[self.feature_names]

            # Handle missing values
            df = df.fillna(0)

            # Scale features
            X_scaled = self.scaler.transform(df.values)

            # Convert to PyTorch tensor
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                prediction_probs = torch.softmax(outputs, dim=1)[0]
                predicted_class_idx = torch.argmax(prediction_probs).item()
                predicted_class = self.label_encoder.classes_[predicted_class_idx]
                confidence = prediction_probs[predicted_class_idx].item() * 100

            # Get all class probabilities
            class_probabilities = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_probabilities[class_name] = prediction_probs[i].item() * 100

            return {
                'symbol': symbol,
                'prediction': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'features_used': len(self.feature_names)
            }

    def save_model(self, model_path='models/stock_nn_model.pth',
                   scaler_path='models/stock_scaler.joblib',
                   encoder_path='models/stock_label_encoder.joblib'):
        """Save the trained PyTorch model and preprocessors"""
        import os
        os.makedirs('models', exist_ok=True)

        # Save PyTorch model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': len(self.feature_names),
            'num_classes': len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else 5
        }, model_path)

        # Save scaler and label encoder
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        joblib.dump(self.feature_names, 'models/feature_names.joblib')

        logger.info(f"PyTorch model saved to {model_path}")

    def load_model(self, model_path='models/stock_nn_model.pth',
                   scaler_path='models/stock_scaler.joblib',
                   encoder_path='models/stock_label_encoder.joblib'):
        """Load a pre-trained PyTorch model and preprocessors"""
        try:
            # Load scaler and label encoder first
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            self.feature_names = joblib.load('models/feature_names.joblib')

            # Load PyTorch model
            checkpoint = torch.load(model_path, map_location=self.device)
            input_dim = checkpoint['input_dim']
            num_classes = checkpoint['num_classes']

            self.model = StockNeuralNetwork(input_dim, num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()

            self.is_trained = True
            logger.info(f"PyTorch model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return False
