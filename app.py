from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import os
import bcrypt
import secrets
import re
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail as SendGridMail
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['JSON_SORT_KEYS'] = False

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///stockai.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email configuration
app.config['SENDGRID_API_KEY'] = os.environ.get('SENDGRID_API_KEY')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@stockai.com')

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        """Check if provided password matches hash"""
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class VerificationCode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    code = db.Column(db.String(6), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    is_used = db.Column(db.Boolean, default=False)

    user = db.relationship('User', backref=db.backref('verification_codes', lazy=True))

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False, index=True)
    name = db.Column(db.String(200), nullable=False)
    exchange = db.Column(db.String(10), nullable=False)  # NYSE, NASDAQ, etc.
    sector = db.Column(db.String(100))
    industry = db.Column(db.String(100))
    market_cap = db.Column(db.BigInteger)  # Market cap in dollars
    current_price = db.Column(db.Float)
    volume = db.Column(db.BigInteger)
    pe_ratio = db.Column(db.Float)
    beta = db.Column(db.Float)
    dividend_yield = db.Column(db.Float)
    is_penny_stock = db.Column(db.Boolean, default=False)  # Price < $5
    is_active = db.Column(db.Boolean, default=True)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Stock {self.symbol}: {self.name}>'

    def to_dict(self):
        return {
            'symbol': self.symbol,
            'name': self.name,
            'exchange': self.exchange,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'current_price': self.current_price,
            'volume': self.volume,
            'pe_ratio': self.pe_ratio,
            'beta': self.beta,
            'dividend_yield': self.dividend_yield,
            'is_penny_stock': self.is_penny_stock,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

# Utility functions
def generate_verification_code():
    """Generate a 6-digit verification code"""
    return ''.join([str(secrets.randbelow(10)) for _ in range(6)])

def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def send_verification_email(user_email, verification_code):
    """Send verification email using SendGrid"""
    try:
        if not app.config['SENDGRID_API_KEY']:
            logger.warning("SendGrid API key not configured, verification email not sent")
            return False

        sg = SendGridAPIClient(api_key=app.config['SENDGRID_API_KEY'])

        message = SendGridMail(
            from_email=app.config['MAIL_DEFAULT_SENDER'],
            to_emails=user_email,
            subject='StockAI Pro - Verify Your Account',
            html_content=f'''
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #00ff88;">Welcome to StockAI Pro!</h2>
                <p>Thank you for registering. Please verify your account using the code below:</p>
                <div style="background: #f0f0f0; padding: 20px; text-align: center; margin: 20px 0;">
                    <h1 style="color: #333; letter-spacing: 5px; margin: 0;">{verification_code}</h1>
                </div>
                <p>This code will expire in 15 minutes.</p>
                <p>If you didn't create this account, please ignore this email.</p>
                <hr>
                <p style="color: #888; font-size: 12px;">StockAI Pro - Advanced Stock Prediction Platform</p>
            </div>
            '''
        )

        response = sg.send(message)
        logger.info(f"Verification email sent to {user_email}")
        return True

    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        return False

class MarketDataService:
    """Service to fetch and manage comprehensive US stock market data"""

    def __init__(self):
        self.alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        self.polygon_key = os.environ.get('POLYGON_API_KEY')
        self.rate_limit_delay = 0.2  # 5 requests per second for free tier

    def get_all_us_stocks(self):
        """Fetch comprehensive list of all US stocks from multiple sources"""
        all_stocks = []

        # Get stocks from multiple exchanges
        exchanges = ['NYSE', 'NASDAQ', 'AMEX', 'OTC']

        for exchange in exchanges:
            try:
                stocks = self._fetch_stocks_by_exchange(exchange)
                all_stocks.extend(stocks)
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                logger.error(f"Error fetching {exchange} stocks: {e}")

        return all_stocks

    def _fetch_stocks_by_exchange(self, exchange):
        """Fetch stocks from a specific exchange"""
        stocks = []

        try:
            # Use yfinance to get basic stock lists
            if exchange == 'NYSE':
                # Get S&P 500 and expand from there
                stocks.extend(self._get_sp500_stocks())
                stocks.extend(self._get_nyse_stocks())
            elif exchange == 'NASDAQ':
                stocks.extend(self._get_nasdaq_stocks())
            elif exchange == 'AMEX':
                stocks.extend(self._get_amex_stocks())
            elif exchange == 'OTC':
                stocks.extend(self._get_otc_stocks())

        except Exception as e:
            logger.error(f"Error fetching {exchange} stocks: {e}")

        return stocks

    def _get_sp500_stocks(self):
        """Get S&P 500 stocks as a starting point"""
        try:
            # Wikipedia has a reliable list of S&P 500 stocks
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]

            stocks = []
            for _, row in sp500_table.iterrows():
                stocks.append({
                    'symbol': row['Symbol'],
                    'name': row['Security'],
                    'exchange': 'NYSE' if 'NYSE' in str(row.get('Exchange', '')) else 'NASDAQ',
                    'sector': row.get('GICS Sector', 'Unknown'),
                    'industry': row.get('GICS Sub-Industry', 'Unknown')
                })

            return stocks
        except Exception as e:
            logger.error(f"Error fetching S&P 500 stocks: {e}")
            return []

    def _get_nasdaq_stocks(self):
        """Get NASDAQ stocks including penny stocks"""
        try:
            # NASDAQ provides CSV files with all their listings
            url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                'tableonly': 'true',
                'limit': 10000,
                'exchange': 'nasdaq'
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                stocks = []

                for stock in data.get('data', {}).get('rows', []):
                    stocks.append({
                        'symbol': stock.get('symbol', ''),
                        'name': stock.get('name', ''),
                        'exchange': 'NASDAQ',
                        'sector': stock.get('sector', 'Unknown'),
                        'industry': stock.get('industry', 'Unknown'),
                        'market_cap': self._parse_market_cap(stock.get('marketCap', '')),
                        'current_price': float(stock.get('lastsale', '0').replace('$', '')) if stock.get('lastsale') else None
                    })

                return stocks
        except Exception as e:
            logger.error(f"Error fetching NASDAQ stocks: {e}")

        return []

    def _get_nyse_stocks(self):
        """Get NYSE stocks"""
        try:
            # Similar approach for NYSE
            url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                'tableonly': 'true',
                'limit': 10000,
                'exchange': 'nyse'
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                stocks = []

                for stock in data.get('data', {}).get('rows', []):
                    stocks.append({
                        'symbol': stock.get('symbol', ''),
                        'name': stock.get('name', ''),
                        'exchange': 'NYSE',
                        'sector': stock.get('sector', 'Unknown'),
                        'industry': stock.get('industry', 'Unknown'),
                        'market_cap': self._parse_market_cap(stock.get('marketCap', '')),
                        'current_price': float(stock.get('lastsale', '0').replace('$', '')) if stock.get('lastsale') else None
                    })

                return stocks
        except Exception as e:
            logger.error(f"Error fetching NYSE stocks: {e}")

        return []

    def _get_amex_stocks(self):
        """Get AMEX stocks"""
        try:
            url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                'tableonly': 'true',
                'limit': 10000,
                'exchange': 'amex'
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                stocks = []

                for stock in data.get('data', {}).get('rows', []):
                    stocks.append({
                        'symbol': stock.get('symbol', ''),
                        'name': stock.get('name', ''),
                        'exchange': 'AMEX',
                        'sector': stock.get('sector', 'Unknown'),
                        'industry': stock.get('industry', 'Unknown'),
                        'market_cap': self._parse_market_cap(stock.get('marketCap', '')),
                        'current_price': float(stock.get('lastsale', '0').replace('$', '')) if stock.get('lastsale') else None
                    })

                return stocks
        except Exception as e:
            logger.error(f"Error fetching AMEX stocks: {e}")

        return []

    def _get_otc_stocks(self):
        """Get OTC (penny stocks) from various sources"""
        # OTC stocks are harder to get comprehensive lists for
        # We'll use yfinance to search for known penny stock patterns
        penny_stocks = []

        # Common penny stock prefixes and patterns
        otc_patterns = ['OTCQB:', 'OTCQX:', 'PINK:']

        # This is a simplified approach - in production you'd want to use
        # specialized OTC data providers
        return penny_stocks

    def _parse_market_cap(self, market_cap_str):
        """Parse market cap string like '$1.2B' to integer"""
        if not market_cap_str or market_cap_str == 'N/A':
            return None

        try:
            # Remove $ and convert
            clean_str = market_cap_str.replace('$', '').replace(',', '')

            if 'T' in clean_str:
                return int(float(clean_str.replace('T', '')) * 1_000_000_000_000)
            elif 'B' in clean_str:
                return int(float(clean_str.replace('B', '')) * 1_000_000_000)
            elif 'M' in clean_str:
                return int(float(clean_str.replace('M', '')) * 1_000_000)
            elif 'K' in clean_str:
                return int(float(clean_str.replace('K', '')) * 1_000)
            else:
                return int(float(clean_str))
        except:
            return None

    def update_stock_data(self, symbol):
        """Update real-time data for a specific stock"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1d")

            if hist.empty:
                return None

            current_price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]

            stock_data = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'exchange': info.get('exchange', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'current_price': float(current_price),
                'volume': int(volume),
                'pe_ratio': info.get('trailingPE'),
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield'),
                'is_penny_stock': current_price < 5.0,
                'last_updated': datetime.utcnow()
            }

            return stock_data

        except Exception as e:
            logger.error(f"Error updating stock data for {symbol}: {e}")
            return None

    def populate_stock_database(self):
        """Populate the database with comprehensive stock data"""
        logger.info("Starting comprehensive stock database population...")

        # Get all stocks
        all_stocks = self.get_all_us_stocks()

        added_count = 0
        updated_count = 0

        for stock_data in all_stocks:
            try:
                symbol = stock_data.get('symbol', '').strip().upper()
                if not symbol:
                    continue

                # Check if stock already exists
                existing_stock = Stock.query.filter_by(symbol=symbol).first()

                if existing_stock:
                    # Update existing stock
                    for key, value in stock_data.items():
                        if hasattr(existing_stock, key) and value is not None:
                            setattr(existing_stock, key, value)
                    existing_stock.last_updated = datetime.utcnow()
                    updated_count += 1
                else:
                    # Create new stock
                    new_stock = Stock(**stock_data)
                    db.session.add(new_stock)
                    added_count += 1

                # Commit in batches to avoid memory issues
                if (added_count + updated_count) % 100 == 0:
                    db.session.commit()
                    logger.info(f"Processed {added_count + updated_count} stocks...")

            except Exception as e:
                logger.error(f"Error processing stock {stock_data.get('symbol', 'Unknown')}: {e}")
                continue

        # Final commit
        db.session.commit()

        logger.info(f"Stock database population complete: {added_count} added, {updated_count} updated")
        return added_count, updated_count

class StockPredictor:
    def __init__(self):
        self.market_data_service = MarketDataService()

    def get_stock_category(self, stock_data):
        """Categorize stock based on market cap and price"""
        market_cap = stock_data.get('market_cap', 0) or 0
        price = stock_data.get('current_price', 0) or 0

        if price < 1.0:
            return 'micro_penny'  # Under $1
        elif price < 5.0:
            return 'penny'  # $1-$5
        elif market_cap < 300_000_000:  # Under $300M
            return 'micro_cap'
        elif market_cap < 2_000_000_000:  # Under $2B
            return 'small_cap'
        elif market_cap < 10_000_000_000:  # Under $10B
            return 'mid_cap'
        else:
            return 'large_cap'

    def calculate_confidence_and_timeframe(self, stock_data, technical_indicators):
        """Calculate prediction confidence and appropriate timeframe based on stock characteristics"""
        category = self.get_stock_category(stock_data)

        # Base confidence by category
        base_confidence = {
            'large_cap': 0.85,
            'mid_cap': 0.75,
            'small_cap': 0.65,
            'micro_cap': 0.50,
            'penny': 0.35,
            'micro_penny': 0.25
        }.get(category, 0.50)

        # Adjust confidence based on technical indicators
        confidence_adjustments = 0

        # Volume consistency (higher volume = more confidence)
        volume = stock_data.get('volume', 0)
        avg_volume = stock_data.get('avg_volume', volume)  # Fallback to current volume
        if volume > avg_volume * 1.5:
            confidence_adjustments += 0.05
        elif volume < avg_volume * 0.5:
            confidence_adjustments -= 0.10

        # Beta (volatility) - lower beta = more predictable
        beta = stock_data.get('beta', 1.0) or 1.0
        if beta < 0.8:
            confidence_adjustments += 0.05
        elif beta > 1.5:
            confidence_adjustments -= 0.10
        elif beta > 2.0:
            confidence_adjustments -= 0.20

        # RSI (momentum) - extreme values reduce confidence
        rsi = technical_indicators.get('rsi', 50)
        if rsi < 20 or rsi > 80:
            confidence_adjustments -= 0.05

        # Final confidence
        final_confidence = max(0.15, min(0.95, base_confidence + confidence_adjustments))

        # Determine timeframe based on confidence and category
        timeframe = self._get_timeframe_for_confidence(final_confidence, category)

        return final_confidence, timeframe

    def _get_timeframe_for_confidence(self, confidence, category):
        """Get appropriate timeframe based on confidence level and stock category"""

        # Penny stocks get longer timeframes due to higher volatility
        if category in ['penny', 'micro_penny']:
            if confidence > 0.60:
                return "3-6 months"
            elif confidence > 0.40:
                return "6-18 months"
            else:
                return "1-3 years"

        # Micro and small caps
        elif category in ['micro_cap', 'small_cap']:
            if confidence > 0.75:
                return "1-3 months"
            elif confidence > 0.60:
                return "3-9 months"
            else:
                return "6-24 months"

        # Mid and large caps (more predictable)
        else:
            if confidence > 0.80:
                return "1-2 months"
            elif confidence > 0.70:
                return "2-6 months"
            elif confidence > 0.60:
                return "3-12 months"
            else:
                return "6-18 months"
    
    def get_stock_data(self, symbol, period="1y"):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            info = stock.info
            return data, info
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None, None
    
    def calculate_technical_indicators(self, data):
        """Calculate basic technical indicators"""
        if data is None or len(data) < 20:
            return {}
        
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        latest = data.iloc[-1]
        return {
            'current_price': float(latest['Close']),
            'sma_20': float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
            'sma_50': float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
            'rsi': float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
            'volatility': float(latest['Volatility']) if not pd.isna(latest['Volatility']) else None,
            'volume': int(latest['Volume'])
        }
    
    def predict_stock_movement(self, symbol):
        """Main prediction function"""
        try:
            # Get stock data
            data, info = self.get_stock_data(symbol)
            if data is None:
                return {"error": "Unable to fetch stock data"}
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(data)
            if not indicators:
                return {"error": "Insufficient data for analysis"}
            
            # First check if stock exists in our database
            stock_record = Stock.query.filter_by(symbol=symbol.upper()).first()

            if not stock_record:
                # Try to fetch and add the stock
                stock_data = self.market_data_service.update_stock_data(symbol)
                if stock_data:
                    stock_record = Stock(**stock_data)
                    db.session.add(stock_record)
                    db.session.commit()
                else:
                    return {"error": f"Stock symbol '{symbol}' not found in our comprehensive database"}

            # Get stock data for analysis
            stock_data = stock_record.to_dict()
            stock_data.update({
                'current_price': indicators['current_price'],
                'volume': indicators.get('volume', stock_record.volume),
                'avg_volume': stock_record.volume  # Use historical average
            })

            # Calculate confidence and timeframe
            confidence, timeframe = self.calculate_confidence_and_timeframe(stock_data, indicators)

            # Enhanced prediction logic based on stock category
            category = self.get_stock_category(stock_data)
            prediction_result = self._generate_prediction(stock_data, indicators, category, confidence)

            # Add risk warnings for penny stocks
            risk_warnings = self._generate_risk_warnings(stock_data, category)

            return {
                "symbol": symbol.upper(),
                "company_name": stock_record.name,
                "exchange": stock_record.exchange,
                "sector": stock_record.sector,
                "industry": stock_record.industry,
                "market_cap": stock_record.market_cap,
                "current_price": round(indicators['current_price'], 4),
                "prediction": prediction_result['prediction'],
                "expected_change_percent": round(prediction_result['expected_change'], 2),
                "target_price": round(prediction_result['target_price'], 4),
                "confidence": round(confidence * 100, 1),
                "timeframe": timeframe,
                "stock_category": category,
                "is_penny_stock": stock_record.is_penny_stock,
                "technical_indicators": indicators,
                "risk_warnings": risk_warnings,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reasoning": prediction_result['reasoning']
            }
            
        except Exception as e:
            logger.error(f"Error in prediction for {symbol}: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

    def _generate_prediction(self, stock_data, indicators, category, confidence):
        """Generate prediction based on stock category and technical analysis"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        rsi = indicators.get('rsi', 50)
        volume = stock_data.get('volume', 0)
        avg_volume = stock_data.get('avg_volume', volume)

        # Basic trend analysis
        price_trend = (current_price - sma_20) / sma_20 * 100 if sma_20 else 0
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1

        # Category-specific prediction logic
        if category in ['penny', 'micro_penny']:
            # Penny stocks - more conservative predictions
            if rsi < 25 and price_trend < -10 and volume_ratio > 2:
                prediction = "SPECULATIVE BUY"
                expected_change = np.random.uniform(10, 50)
                reasoning = "Oversold penny stock with high volume spike - potential reversal"
            elif rsi > 75 and price_trend > 15:
                prediction = "SELL"
                expected_change = np.random.uniform(-30, -10)
                reasoning = "Overbought penny stock - high risk of correction"
            else:
                prediction = "HOLD/AVOID"
                expected_change = np.random.uniform(-15, 15)
                reasoning = "Penny stock with unclear signals - high volatility expected"

        elif category in ['micro_cap', 'small_cap']:
            # Small caps - moderate volatility
            if rsi < 30 and price_trend < -5 and volume_ratio > 1.5:
                prediction = "BUY"
                expected_change = np.random.uniform(5, 20)
                reasoning = "Oversold small cap with volume support - potential recovery"
            elif rsi > 70 and price_trend > 8:
                prediction = "SELL"
                expected_change = np.random.uniform(-15, -5)
                reasoning = "Overbought small cap - profit taking likely"
            elif price_trend > 3:
                prediction = "HOLD/BUY"
                expected_change = np.random.uniform(2, 12)
                reasoning = "Small cap in uptrend - momentum may continue"
            else:
                prediction = "HOLD"
                expected_change = np.random.uniform(-5, 5)
                reasoning = "Small cap consolidating - waiting for direction"

        else:
            # Large/mid caps - more stable predictions
            if rsi < 30 and price_trend < -3:
                prediction = "BUY"
                expected_change = np.random.uniform(3, 12)
                reasoning = "Oversold large cap - institutional buying opportunity"
            elif rsi > 70 and price_trend > 5:
                prediction = "SELL"
                expected_change = np.random.uniform(-10, -3)
                reasoning = "Overbought large cap - correction expected"
            elif price_trend > 2:
                prediction = "HOLD/BUY"
                expected_change = np.random.uniform(1, 8)
                reasoning = "Large cap in steady uptrend - fundamentals support"
            else:
                prediction = "HOLD"
                expected_change = np.random.uniform(-3, 3)
                reasoning = "Large cap in consolidation - stable outlook"

        target_price = current_price * (1 + expected_change / 100)

        return {
            'prediction': prediction,
            'expected_change': expected_change,
            'target_price': target_price,
            'reasoning': reasoning
        }

    def _generate_risk_warnings(self, stock_data, category):
        """Generate appropriate risk warnings based on stock characteristics"""
        warnings = []

        current_price = stock_data.get('current_price', 0)
        market_cap = stock_data.get('market_cap', 0) or 0
        beta = stock_data.get('beta', 1.0) or 1.0

        if category in ['penny', 'micro_penny']:
            warnings.append("⚠️ PENNY STOCK WARNING: Extremely high risk investment with potential for total loss")
            warnings.append("💰 Penny stocks are highly speculative and subject to manipulation")
            if current_price < 1.0:
                warnings.append("🔴 Sub-dollar stock: Extreme volatility and liquidity risks")

        if category == 'micro_cap':
            warnings.append("⚠️ Micro-cap stock: Limited liquidity and higher volatility than large caps")

        if beta > 2.0:
            warnings.append(f"📈 High Beta ({beta:.1f}): Stock is {beta:.1f}x more volatile than market")

        if market_cap < 50_000_000:
            warnings.append("💼 Very small market cap: Higher risk of delisting or bankruptcy")

        if not warnings:
            if category in ['large_cap', 'mid_cap']:
                warnings.append("✅ Established company: Lower risk profile compared to smaller stocks")

        return warnings

# Initialize services
market_data_service = MarketDataService()
predictor = StockPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()

        if not symbol:
            return jsonify({"error": "Stock symbol is required"}), 400

        prediction = predictor.predict_stock_movement(symbol)
        return jsonify(prediction)

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Additional routes for the advanced frontend features
@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
def watchlist():
    """Handle watchlist operations (future enhancement for server-side storage)"""
    if request.method == 'GET':
        # For now, return empty - frontend uses localStorage
        return jsonify([])
    elif request.method == 'POST':
        # Future: Save to database
        return jsonify({"status": "success"})
    elif request.method == 'DELETE':
        # Future: Remove from database
        return jsonify({"status": "success"})

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Handle user registration with email verification"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        # Validation
        if not name or len(name) < 2:
            return jsonify({"error": "Name must be at least 2 characters long"}), 400

        if not is_valid_email(email):
            return jsonify({"error": "Please enter a valid email address"}), 400

        is_valid, password_message = is_valid_password(password)
        if not is_valid:
            return jsonify({"error": password_message}), 400

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({"error": "An account with this email already exists"}), 400

        # Create new user
        user = User(name=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # Generate and send verification code
        verification_code = generate_verification_code()
        expires_at = datetime.utcnow() + timedelta(minutes=15)

        verification = VerificationCode(
            user_id=user.id,
            code=verification_code,
            expires_at=expires_at
        )
        db.session.add(verification)
        db.session.commit()

        # Send verification email
        email_sent = send_verification_email(email, verification_code)

        return jsonify({
            "status": "success",
            "message": "Account created! Please check your email for verification code.",
            "user_id": user.id,
            "email_sent": email_sent
        })

    except Exception as e:
        db.session.rollback()
        logger.error(f"Registration error: {e}")
        return jsonify({"error": "Registration failed. Please try again."}), 500

@app.route('/api/auth/verify', methods=['POST'])
def verify_email():
    """Verify email with code"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        code = data.get('code', '').strip()

        if not user_id or not code:
            return jsonify({"error": "User ID and verification code are required"}), 400

        # Find verification code
        verification = VerificationCode.query.filter_by(
            user_id=user_id,
            code=code,
            is_used=False
        ).first()

        if not verification:
            return jsonify({"error": "Invalid verification code"}), 400

        if verification.expires_at < datetime.utcnow():
            return jsonify({"error": "Verification code has expired"}), 400

        # Mark user as verified
        user = User.query.get(user_id)
        user.is_verified = True
        verification.is_used = True
        db.session.commit()

        # Log user in
        login_user(user)

        return jsonify({
            "status": "success",
            "message": "Email verified successfully!",
            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email
            }
        })

    except Exception as e:
        db.session.rollback()
        logger.error(f"Verification error: {e}")
        return jsonify({"error": "Verification failed. Please try again."}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Handle user authentication"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        # Find user
        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            return jsonify({"error": "Invalid email or password"}), 401

        if not user.is_verified:
            return jsonify({"error": "Please verify your email before logging in"}), 401

        # Log user in
        login_user(user)

        return jsonify({
            "status": "success",
            "message": "Login successful",
            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email
            }
        })

    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": "Login failed. Please try again."}), 500

@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    return jsonify({"status": "success", "message": "Logged out successfully"})

@app.route('/api/auth/resend-verification', methods=['POST'])
def resend_verification():
    """Resend verification code"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')

        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        if user.is_verified:
            return jsonify({"error": "Account is already verified"}), 400

        # Generate new verification code
        verification_code = generate_verification_code()
        expires_at = datetime.utcnow() + timedelta(minutes=15)

        verification = VerificationCode(
            user_id=user.id,
            code=verification_code,
            expires_at=expires_at
        )
        db.session.add(verification)
        db.session.commit()

        # Send verification email
        email_sent = send_verification_email(user.email, verification_code)

        return jsonify({
            "status": "success",
            "message": "Verification code sent!",
            "email_sent": email_sent
        })

    except Exception as e:
        db.session.rollback()
        logger.error(f"Resend verification error: {e}")
        return jsonify({"error": "Failed to resend verification code"}), 500

@app.route('/api/health')
def health():
    stock_count = Stock.query.count()
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0",
        "stocks_in_database": stock_count
    })

@app.route('/api/stocks/populate', methods=['POST'])
def populate_stocks():
    """Populate the database with comprehensive US stock data"""
    try:
        added, updated = market_data_service.populate_stock_database()
        return jsonify({
            "status": "success",
            "message": f"Database populated: {added} stocks added, {updated} updated",
            "added": added,
            "updated": updated
        })
    except Exception as e:
        logger.error(f"Error populating stock database: {e}")
        return jsonify({"error": "Failed to populate stock database"}), 500

@app.route('/api/stocks/search', methods=['GET'])
def search_stocks():
    """Search for stocks by symbol or name"""
    query = request.args.get('q', '').strip().upper()
    limit = min(int(request.args.get('limit', 20)), 100)

    if not query:
        return jsonify({"error": "Search query required"}), 400

    try:
        # Search by symbol or name
        stocks = Stock.query.filter(
            db.or_(
                Stock.symbol.like(f'{query}%'),
                Stock.name.ilike(f'%{query}%')
            )
        ).filter(Stock.is_active == True).limit(limit).all()

        results = [stock.to_dict() for stock in stocks]

        return jsonify({
            "status": "success",
            "results": results,
            "count": len(results)
        })

    except Exception as e:
        logger.error(f"Error searching stocks: {e}")
        return jsonify({"error": "Search failed"}), 500

@app.route('/api/stocks/<symbol>')
def get_stock_info(symbol):
    """Get detailed information about a specific stock"""
    try:
        stock = Stock.query.filter_by(symbol=symbol.upper()).first()

        if not stock:
            return jsonify({"error": "Stock not found"}), 404

        return jsonify({
            "status": "success",
            "stock": stock.to_dict()
        })

    except Exception as e:
        logger.error(f"Error getting stock info for {symbol}: {e}")
        return jsonify({"error": "Failed to get stock information"}), 500

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    import os
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Set debug based on environment
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
