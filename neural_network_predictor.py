"""
Neural Network Stock Prediction Model
Integrates technical indicators with fundamental analysis factors
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime
import yfinance as yf
from app import db, Stock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockNeuralNetworkPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        
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
        ema_12 = close.ewm(span=12).mean().iloc[-1]
        ema_26 = close.ewm(span=26).mean().iloc[-1]

        # RSI (EXACT match from original)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # MACD (EXACT match from original)
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean().iloc[-1]
        macd_histogram = (macd - macd.ewm(span=9).mean()).iloc[-1]

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
        if macd.iloc[-1] > 0:
            macd_score = min(10, abs(macd.iloc[-1]) * 2)
        else:
            macd_score = -min(10, abs(macd.iloc[-1]) * 2)

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
        if macd.iloc[-1] > 1.0:
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
            'macd': macd.iloc[-1],
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
            'is_macd_positive': 1 if macd.iloc[-1] > 0 else 0,
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
            features['stock_category'] = 'large_cap'
        elif features['market_cap'] > 2_000_000_000:  # Mid cap
            features['base_confidence'] = 0.75
            features['stock_category'] = 'mid_cap'
        elif features['market_cap'] > 300_000_000:  # Small cap
            features['base_confidence'] = 0.65
            features['stock_category'] = 'small_cap'
        elif stock.current_price >= 5.0:  # Micro cap
            features['base_confidence'] = 0.50
            features['stock_category'] = 'micro_cap'
        elif stock.current_price >= 1.0:  # Penny stock
            features['base_confidence'] = 0.35
            features['stock_category'] = 'penny'
        else:  # Micro penny
            features['base_confidence'] = 0.25
            features['stock_category'] = 'micro_penny'

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
        features['is_large_cap'] = 1 if features['stock_category'] == 'large_cap' else 0
        features['is_mid_cap'] = 1 if features['stock_category'] == 'mid_cap' else 0
        features['is_small_cap'] = 1 if features['stock_category'] == 'small_cap' else 0
        features['is_micro_cap'] = 1 if features['stock_category'] == 'micro_cap' else 0
        features['is_penny_stock'] = 1 if features['stock_category'] == 'penny' else 0
        features['is_micro_penny'] = 1 if features['stock_category'] == 'micro_penny' else 0

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
        """Build the neural network architecture"""
        model = keras.Sequential([
            # Input layer with dropout for regularization
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Hidden layers with decreasing size
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),

            # Output layer for 5 classes (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
            layers.Dense(5, activation='softmax')
        ])

        # Compile with appropriate loss function and metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    def train(self, symbols_list, epochs=100, validation_split=0.2, save_model=True):
        """Train the neural network model"""
        logger.info("Starting neural network training...")

        # Prepare training data
        training_data, labels = self.prepare_training_data(symbols_list)

        if len(training_data) == 0:
            raise ValueError("No training data collected")

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(training_data)

        # Handle missing values
        df = df.fillna(0)

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

        # Build model
        self.model = self.build_model(X_scaled.shape[1])

        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001
            )
        ]

        # Train the model
        logger.info("Training neural network...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate on test set
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")

        # Generate classification report
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred_classes, target_names=class_names)
        logger.info(f"Classification Report:\n{report}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        logger.info(f"Confusion Matrix:\n{cm}")

        self.is_trained = True

        # Save model and preprocessors
        if save_model:
            self.save_model()

        return history

    def predict(self, symbol):
        """Make prediction for a single stock"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

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

        # Make prediction
        prediction_probs = self.model.predict(X_scaled, verbose=0)[0]
        predicted_class_idx = np.argmax(prediction_probs)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        confidence = prediction_probs[predicted_class_idx] * 100

        # Get all class probabilities
        class_probabilities = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_probabilities[class_name] = prediction_probs[i] * 100

        return {
            'symbol': symbol,
            'prediction': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'features_used': len(self.feature_names)
        }

    def save_model(self, model_path='models/stock_nn_model.h5',
                   scaler_path='models/stock_scaler.joblib',
                   encoder_path='models/stock_label_encoder.joblib'):
        """Save the trained model and preprocessors"""
        import os
        os.makedirs('models', exist_ok=True)

        # Save model
        self.model.save(model_path)

        # Save scaler and label encoder
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        joblib.dump(self.feature_names, 'models/feature_names.joblib')

        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path='models/stock_nn_model.h5',
                   scaler_path='models/stock_scaler.joblib',
                   encoder_path='models/stock_label_encoder.joblib'):
        """Load a pre-trained model and preprocessors"""
        try:
            # Load model
            self.model = keras.models.load_model(model_path)

            # Load scaler and label encoder
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            self.feature_names = joblib.load('models/feature_names.joblib')

            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
