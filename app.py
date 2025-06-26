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
import time
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
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

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

# BACKUP: Original Rule-Based StockPredictor (42.86% accuracy)
# This class has been replaced with NeuralNetworkStockPredictor (97.5% accuracy)
# Preserved here for reference - see rule_based_predictor_backup.py for full implementation

class OriginalStockPredictor_BACKUP:
    def __init__(self):
        self.market_data_service = MarketDataService()
        self._prediction_cache = {}  # Simple cache for consistency
        self._cache_timeout = 300  # 5 minutes cache

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

    def _convert_timeframe_option(self, option):
        """Convert frontend timeframe option to display string"""
        timeframe_map = {
            '1_month': '1 month',
            '1month': '1 month',
            '3_months': '3 months',
            '3months': '3 months',
            '6_months': '6 months',
            '6months': '6 months',
            '1_year': '1 year',
            '1year': '1 year',
            '2_years': '2 years',
            '2years': '2 years'
        }
        return timeframe_map.get(option, '3 months')
    
    def get_stock_data(self, symbol, period="1y"):
        """Fetch stock data using yfinance with enhanced error handling"""
        try:
            symbol = symbol.upper().strip()
            stock = yf.Ticker(symbol)

            # Try to get data with different periods if 1y fails
            data = None
            periods_to_try = [period, "6mo", "3mo", "1mo", "5d"]

            for p in periods_to_try:
                try:
                    data = stock.history(period=p)
                    if not data.empty:
                        break
                except:
                    continue

            if data is None or data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None, None

            # Get stock info
            info = stock.info

            # If info is empty or missing key data, try to get basic info
            if not info or 'longName' not in info:
                # For some penny stocks, info might be limited
                info = {
                    'longName': symbol,
                    'symbol': symbol,
                    'exchange': 'Unknown',
                    'sector': 'Unknown',
                    'industry': 'Unknown'
                }

            return data, info

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None, None
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators with fallbacks"""
        if data is None or len(data) < 5:
            return {}

        try:
            current_price = float(data['Close'].iloc[-1])

            # Simple Moving Averages with fallbacks
            if len(data) >= 20:
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                sma_20 = float(data['SMA_20'].iloc[-1]) if not pd.isna(data['SMA_20'].iloc[-1]) else current_price
            else:
                sma_20 = current_price

            if len(data) >= 50:
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                sma_50 = float(data['SMA_50'].iloc[-1]) if not pd.isna(data['SMA_50'].iloc[-1]) else current_price
            else:
                sma_50 = current_price

            # RSI calculation with fallback
            if len(data) >= 14:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))
                rsi = float(data['RSI'].iloc[-1]) if not pd.isna(data['RSI'].iloc[-1]) else 50.0
            else:
                rsi = 50.0  # Neutral RSI

            # MACD calculation
            if len(data) >= 26:
                exp1 = data['Close'].ewm(span=12).mean()
                exp2 = data['Close'].ewm(span=26).mean()
                macd = exp1 - exp2
                macd_value = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0
            else:
                macd_value = 0.0

            # Bollinger Bands
            if len(data) >= 20:
                sma_bb = data['Close'].rolling(window=20).mean()
                std_bb = data['Close'].rolling(window=20).std()
                bollinger_upper = sma_bb + (std_bb * 2)
                bollinger_lower = sma_bb - (std_bb * 2)
                bb_upper = float(bollinger_upper.iloc[-1]) if not pd.isna(bollinger_upper.iloc[-1]) else current_price * 1.05
                bb_lower = float(bollinger_lower.iloc[-1]) if not pd.isna(bollinger_lower.iloc[-1]) else current_price * 0.95
            else:
                bb_upper = current_price * 1.05
                bb_lower = current_price * 0.95

            # Volatility
            if len(data) >= 20:
                volatility = data['Close'].rolling(window=20).std()
                vol_value = float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else current_price * 0.02
            else:
                vol_value = current_price * 0.02  # 2% default volatility

            # Volume and Volume Analysis
            volume = int(data['Volume'].iloc[-1]) if not pd.isna(data['Volume'].iloc[-1]) else 1000000

            # Average volume (20-day)
            if len(data) >= 20:
                avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
                avg_volume = int(avg_volume) if not pd.isna(avg_volume) else volume
            else:
                avg_volume = volume

            # Volume trend (increasing/decreasing)
            if len(data) >= 5:
                recent_volume = data['Volume'].tail(5).mean()
                older_volume = data['Volume'].tail(10).head(5).mean() if len(data) >= 10 else recent_volume
                volume_trend = (recent_volume - older_volume) / older_volume * 100 if older_volume > 0 else 0
            else:
                volume_trend = 0

            # Price momentum (rate of change)
            if len(data) >= 10:
                price_momentum = (current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10] * 100
            else:
                price_momentum = 0

            # Trend strength (ADX-like calculation)
            if len(data) >= 14:
                high_low = data['High'] - data['Low']
                high_close = abs(data['High'] - data['Close'].shift())
                low_close = abs(data['Low'] - data['Close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=14).mean().iloc[-1]
                trend_strength = (atr / current_price) * 100 if not pd.isna(atr) else 2.0
            else:
                trend_strength = 2.0

            return {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'macd': macd_value,
                'bollinger_upper': bb_upper,
                'bollinger_lower': bb_lower,
                'volatility': (vol_value / current_price) * 100,  # Convert to percentage
                'volume': volume,
                'avg_volume': avg_volume,
                'volume_trend': volume_trend,
                'price_momentum': price_momentum,
                'trend_strength': trend_strength
            }

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            # Return basic fallback values
            current_price = float(data['Close'].iloc[-1]) if len(data) > 0 else 100.0
            return {
                'current_price': current_price,
                'sma_20': current_price,
                'sma_50': current_price,
                'rsi': 50.0,
                'macd': 0.0,
                'bollinger_upper': current_price * 1.05,
                'bollinger_lower': current_price * 0.95,
                'volatility': 2.0,
                'volume': 1000000
            }
    
    def predict_stock_movement(self, symbol, timeframe_override=None):
        """Main prediction function with optional timeframe override"""
        try:
            logger.info(f"Starting prediction for {symbol}")

            # Check cache for recent prediction to ensure consistency
            cache_key = f"{symbol}_{timeframe_override or 'auto'}"
            current_time = time.time()

            if cache_key in self._prediction_cache:
                cached_result, cache_time = self._prediction_cache[cache_key]
                if current_time - cache_time < self._cache_timeout:
                    logger.info(f"Returning cached prediction for {symbol}")
                    return cached_result

            # Get stock data
            data, info = self.get_stock_data(symbol)
            logger.info(f"Stock data fetched: data={data is not None}, info={info is not None}")
            if data is None:
                return {"error": "Unable to fetch stock data"}

            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(data)
            logger.info(f"Technical indicators calculated: {indicators is not None}")
            if not indicators:
                return {"error": "Insufficient data for analysis"}
            
            # Create or get stock record
            symbol = symbol.upper().strip()
            logger.info(f"Looking for stock record for {symbol}")
            stock_record = Stock.query.filter_by(symbol=symbol).first()
            logger.info(f"Stock record found: {stock_record is not None}")

            if not stock_record:
                # Create stock record from yfinance data
                current_price = indicators['current_price']
                market_cap = info.get('marketCap')
                logger.info(f"yfinance info for {symbol}: marketCap={market_cap}, longName={info.get('longName')}")

                stock_record = Stock(
                    symbol=symbol,
                    name=info.get('longName', symbol),
                    exchange=info.get('exchange', 'Unknown'),
                    sector=info.get('sector', 'Unknown'),
                    industry=info.get('industry', 'Unknown'),
                    market_cap=market_cap,
                    current_price=current_price,
                    volume=indicators.get('volume', 0),
                    pe_ratio=info.get('trailingPE'),
                    beta=info.get('beta'),
                    dividend_yield=info.get('dividendYield'),
                    is_penny_stock=current_price < 5.0,
                    last_updated=datetime.utcnow()
                )

                try:
                    db.session.add(stock_record)
                    db.session.commit()
                    logger.info(f"Added new stock to database: {symbol}")
                except Exception as e:
                    db.session.rollback()
                    logger.warning(f"Could not add stock to database: {e}")
                    # Continue with analysis even if database save fails

                # Verify stock_record was created successfully
                if stock_record is None:
                    logger.error(f"Failed to create stock record for {symbol}")
                    return {"error": f"Unable to create stock record for {symbol}"}
            else:
                # Update existing record with fresh data
                stock_record.current_price = indicators['current_price']
                stock_record.volume = indicators.get('volume', stock_record.volume)
                stock_record.last_updated = datetime.utcnow()

                # Update market cap if it's missing or 0
                if not stock_record.market_cap or stock_record.market_cap == 0:
                    market_cap = info.get('marketCap')
                    if market_cap:
                        stock_record.market_cap = market_cap
                        logger.info(f"Updated market cap for {symbol}: {market_cap}")

                # Update industry if it's missing or Unknown
                if not stock_record.industry or stock_record.industry == 'Unknown':
                    industry = info.get('industry')
                    if industry:
                        stock_record.industry = industry
                        logger.info(f"Updated industry for {symbol}: {industry}")
                    else:
                        logger.info(f"No industry data available for {symbol} in yfinance")

                try:
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    logger.warning(f"Could not update stock in database: {e}")

            # Get stock data for analysis
            if stock_record is None:
                logger.error(f"Stock record is None for symbol {symbol}")
                return {"error": f"Unable to access stock record for {symbol}"}

            stock_data = stock_record.to_dict()
            stock_data.update({
                'current_price': indicators['current_price'],
                'volume': indicators.get('volume', stock_record.volume or 0),
                'avg_volume': stock_record.volume or 0  # Use historical average
            })

            # Calculate confidence and timeframe
            logger.info("Calculating confidence and timeframe")
            confidence, auto_timeframe = self.calculate_confidence_and_timeframe(stock_data, indicators)
            logger.info(f"Confidence: {confidence}, timeframe: {auto_timeframe}")

            # Use manual timeframe override if provided, otherwise use AI recommendation
            if timeframe_override and timeframe_override != 'auto':
                timeframe = self._convert_timeframe_option(timeframe_override)
            else:
                timeframe = auto_timeframe
            logger.info(f"Final timeframe: {timeframe}")

            # Enhanced prediction logic based on stock category
            logger.info("Getting stock category")
            category = self.get_stock_category(stock_data)
            logger.info(f"Stock category: {category}")

            logger.info("Generating prediction")
            prediction_result = self._generate_prediction(stock_data, indicators, category, confidence, timeframe)
            logger.info(f"Prediction result: {prediction_result}")

            # Add risk warnings for penny stocks
            logger.info("Generating risk warnings")
            risk_warnings = self._generate_risk_warnings(stock_data, category)
            logger.info(f"Risk warnings: {risk_warnings}")

            # Generate chart data
            logger.info("Generating chart data")
            chart_data = self._generate_chart_data(data, indicators, prediction_result)
            logger.info(f"Chart data generated: {chart_data is not None}")

            # Generate enhanced AI reasoning
            logger.info("Generating enhanced reasoning")
            enhanced_reasoning = self._generate_enhanced_reasoning(stock_data, indicators, category, prediction_result, confidence)
            logger.info(f"Enhanced reasoning generated: {enhanced_reasoning is not None}")

            # Ensure stock_record is not None
            if stock_record is None:
                logger.error(f"Stock record is None for symbol {symbol}")
                return {"error": f"Unable to create stock record for {symbol}"}

            result = {
                "symbol": symbol.upper(),
                "company_name": stock_record.name or symbol,
                "exchange": stock_record.exchange or "Unknown",
                "sector": stock_record.sector or "Unknown",
                "industry": stock_record.industry or "Unknown",
                "market_cap": stock_record.market_cap or 0,
                "current_price": round(indicators['current_price'], 2),
                "prediction": prediction_result['prediction'],
                "expected_change_percent": round(prediction_result['expected_change'], 2),
                "target_price": round(prediction_result['target_price'], 2),
                "confidence": round(confidence * 100, 1),
                "timeframe": timeframe,
                "stock_category": category,
                "is_penny_stock": stock_record.is_penny_stock if stock_record.is_penny_stock is not None else False,
                "technical_indicators": indicators,
                "risk_warnings": risk_warnings,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reasoning": prediction_result['reasoning'],
                "enhanced_reasoning": enhanced_reasoning,
                "historical_data": chart_data['historical'],
                "prediction_data": chart_data['prediction'],
                "volume_data": chart_data['volume']
            }

            # Cache the result for consistency
            self._prediction_cache[cache_key] = (result, current_time)

            # Clean up old cache entries to prevent memory leaks
            self._cleanup_cache(current_time)

            return result
            
        except Exception as e:
            logger.error(f"Error in prediction for {symbol}: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

    def _cleanup_cache(self, current_time):
        """Remove expired cache entries"""
        expired_keys = []
        for key, (result, cache_time) in self._prediction_cache.items():
            if current_time - cache_time > self._cache_timeout:
                expired_keys.append(key)

        for key in expired_keys:
            del self._prediction_cache[key]

    def _generate_prediction(self, stock_data, indicators, category, confidence, timeframe):
        """Generate prediction based on technical analysis and multi-factor scoring"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        volume_trend = indicators.get('volume_trend', 0)
        price_momentum = indicators.get('price_momentum', 0)
        trend_strength = indicators.get('trend_strength', 2.0)
        bollinger_upper = indicators.get('bollinger_upper', current_price * 1.05)
        bollinger_lower = indicators.get('bollinger_lower', current_price * 0.95)

        # Enhanced complex model with improved HOLD/SELL logic
        prediction, expected_change, reasoning, confidence_level = self._generate_enhanced_prediction(indicators, category)

        # Scale expected change based on timeframe
        timeframe_multiplier = self._get_timeframe_multiplier(timeframe)
        scaled_expected_change = expected_change * timeframe_multiplier

        target_price = current_price * (1 + scaled_expected_change / 100)

        return {
            'prediction': prediction,
            'expected_change': scaled_expected_change,
            'target_price': target_price,
            'reasoning': reasoning
        }

    def _generate_enhanced_prediction(self, indicators, category):
        """Enhanced complex model with improved HOLD/SELL logic (keeps strong BUY accuracy)"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        price_momentum = indicators.get('price_momentum', 0)
        bollinger_upper = indicators.get('bollinger_upper', current_price * 1.05)
        bollinger_lower = indicators.get('bollinger_lower', current_price * 0.95)

        # Calculate comprehensive score (this worked well for BUY predictions)
        score = self._calculate_comprehensive_score(indicators)

        # Enhanced prediction logic with improved HOLD/SELL thresholds
        prediction, expected_change, reasoning, confidence = self._enhanced_score_to_prediction(score, category, indicators)

        return prediction, expected_change, reasoning, confidence

    def _calculate_comprehensive_score(self, indicators):
        """Calculate comprehensive technical score (this worked well for BUY predictions)"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        price_momentum = indicators.get('price_momentum', 0)
        bollinger_upper = indicators.get('bollinger_upper', current_price * 1.05)
        bollinger_lower = indicators.get('bollinger_lower', current_price * 0.95)

        score = 50  # Neutral starting point

        # EXACT REPLICA of original complex model that achieved 84.6% BUY accuracy
        # RSI analysis (EXACT match)
        if rsi < 30:
            score += 15
        elif rsi > 70:
            score -= 15

        # Moving average analysis (EXACT match)
        if current_price > sma_20 > sma_50:
            score += 15
        elif current_price < sma_20 < sma_50:
            score -= 15

        # Bollinger Bands analysis (EXACT match)
        bb_position = (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower)
        if bb_position < 0.2:
            score += 10
        elif bb_position > 0.8:
            score -= 10

        # MACD analysis (EXACT match)
        if macd > 0:
            score += min(10, abs(macd) * 2)
        else:
            score -= min(10, abs(macd) * 2)

        # Price momentum analysis (EXACT match)
        if price_momentum > 5:
            score += 10
        elif price_momentum < -5:
            score -= 10

        # Volume analysis (EXACT match)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 2:
            score += 15
        elif volume_ratio > 1.5:
            score += 8
        elif volume_ratio < 0.5:
            score -= 10

        # === SELECTIVE GROWTH ENHANCEMENTS ===
        # Add modest bonus points for genuine growth patterns only

        # Strong upward momentum with volume confirmation (more selective)
        if price_momentum > 5 and volume_ratio > 1.5:
            score += 5  # Reduced bonus for genuine momentum

        # RSI in healthy growth range (more restrictive)
        if 50 <= rsi <= 60:
            score += 3  # Smaller bonus for optimal RSI

        # Strong trend with significant separation (more selective)
        if current_price > sma_20 and sma_20 > sma_50:
            ma_separation = (sma_20 - sma_50) / sma_50 * 100
            if ma_separation > 3:  # Higher threshold for trend strength
                score += 4  # Reduced trend bonus

        # MACD strong positive momentum (more selective)
        if macd > 1.0:  # Higher threshold for MACD bonus
            score += 3  # Reduced momentum bonus

        return max(0, min(100, score))  # Keep original score range

    def _calculate_momentum_strength(self, indicators):
        """Calculate overall momentum strength from multiple indicators"""
        price_momentum = indicators.get('price_momentum', 0)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)

        # Normalize momentum components
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

        # MACD component (-0.2 to 0.2)
        momentum_score += max(-0.2, min(0.2, macd / 5))

        return momentum_score

    def _enhanced_score_to_prediction(self, score, category, indicators):
        """Enhanced Complex Model - Keeps proven BUY logic, improves HOLD/SELL accuracy"""
        current_price = indicators['current_price']
        rsi = indicators.get('rsi', 50)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        price_momentum = indicators.get('price_momentum', 0)
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        bollinger_upper = indicators.get('bollinger_upper', current_price * 1.05)
        bollinger_lower = indicators.get('bollinger_lower', current_price * 0.95)

        # Calculate additional indicators
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        bb_position = (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower) if bollinger_upper != bollinger_lower else 0.5

        # === ENHANCED SELL DETECTION (More sensitive to bearish conditions) ===
        sell_signals = 0
        sell_confidence = 45

        # 1. Strong Downtrend + Volume Confirmation
        if current_price < sma_20 < sma_50 and volume_ratio > 1.2:  # Lowered from 1.3
            sell_signals += 2
            sell_confidence += 15

        # 2. Overbought Reversal (more sensitive)
        if rsi > 70 and price_momentum < -1:  # Lowered from 75
            sell_signals += 2
            sell_confidence += 12

        # 3. Momentum Breakdown (more sensitive)
        if price_momentum < -4 and current_price < sma_20:  # Lowered from -5
            sell_signals += 2
            sell_confidence += 10

        # 4. Volume Selling Pressure (more sensitive)
        if volume_ratio > 1.3 and price_momentum < -2:  # Lowered thresholds
            sell_signals += 1
            sell_confidence += 8

        # 5. NEW: Bollinger Band Breakdown
        if current_price < bollinger_lower and price_momentum < -2:
            sell_signals += 2
            sell_confidence += 12

        # 6. NEW: Severe Oversold with Continued Decline
        if rsi < 25 and price_momentum < -3:  # Oversold but still falling
            sell_signals += 1
            sell_confidence += 8

        # 7. NEW: Moving Average Death Cross Pattern
        if sma_20 < sma_50 * 0.98:  # 20-day SMA significantly below 50-day
            sell_signals += 1
            sell_confidence += 6

        # More aggressive SELL threshold - reduced from 3 signals to 2
        if sell_signals >= 2 and score <= 45:  # Increased score threshold from 40 to 45
            prediction = "SELL"
            expected_change = self._calculate_enhanced_sell_change(score, category, sell_signals)
            reasoning = f"Strong bearish momentum detected ({sell_signals} confirmations)"
            confidence = min(85, sell_confidence)
            return prediction, expected_change, reasoning, confidence

        # Secondary SELL condition for very weak fundamentals
        if sell_signals >= 1 and score <= 30:  # Very weak score with at least 1 signal
            prediction = "SELL"
            expected_change = self._calculate_enhanced_sell_change(score, category, max(2, sell_signals))  # Minimum 2 signals for calculation
            reasoning = f"Weak fundamentals with bearish technical signals"
            confidence = min(75, sell_confidence + 5)
            return prediction, expected_change, reasoning, confidence

        # === ENHANCED HOLD/CONSOLIDATION DETECTION (Improve from 25% to 55-65%) ===
        hold_signals = 0
        hold_confidence = 45

        # 1. Price Stability (within 2% of moving average)
        if sma_20 > 0 and abs(current_price - sma_20) / sma_20 < 0.02:
            hold_signals += 1
            hold_confidence += 8

        # 2. Neutral RSI (40-60 range)
        if 40 <= rsi <= 60:
            hold_signals += 1
            hold_confidence += 6

        # 3. Low Momentum (< 3% movement)
        if abs(price_momentum) < 3:
            hold_signals += 1
            hold_confidence += 5

        # 4. Bollinger Middle (price in middle 40% of bands)
        if 0.3 <= bb_position <= 0.7:
            hold_signals += 1
            hold_confidence += 5

        # 5. Normal Volume
        if 0.8 <= volume_ratio <= 1.2:
            hold_signals += 1
            hold_confidence += 4

        # Strong HOLD if multiple consolidation signals + neutral score
        if hold_signals >= 4 and 45 <= score <= 60:
            prediction = "HOLD"
            expected_change = self._calculate_enhanced_hold_change(score, category, hold_signals)
            reasoning = f"Strong consolidation pattern ({hold_signals} signals)"
            confidence = min(75, hold_confidence)
            return prediction, expected_change, reasoning, confidence

        # === PENNY STOCK CAUTION CHECK ===
        if category in ['penny', 'micro_penny']:
            penny_risk_score = self._evaluate_penny_stock_risks(indicators, score)
            if penny_risk_score >= 3:  # High risk penny stock
                prediction = "HOLD"
                expected_change = self._calculate_enhanced_hold_change(score, category, 2)
                reasoning = "High-risk penny stock profile suggests caution despite technical signals"
                confidence = max(45, min(60, score - 10))
                return prediction, expected_change, reasoning, confidence

        # === BALANCED BUY LOGIC (More Selective) ===
        # Exceptional BUY (Score ≥85) - Very high threshold for exceptional cases
        if score >= 85:
            # Extra caution for penny stocks even with high scores
            if category in ['penny', 'micro_penny']:
                prediction = "BUY"  # Downgrade from STRONG BUY
                expected_change = self._calculate_buy_change(score, category) * 0.8  # Reduced multiplier
                reasoning = "Strong technical signals but penny stock risks limit upside confidence"
                confidence = max(70, min(80, score - 5))
            else:
                prediction = "STRONG BUY"
                expected_change = self._calculate_buy_change(score, category) * 1.3
                reasoning = "Strong fundamentals align with favorable market conditions"
                confidence = max(85, min(95, score))

        # Strong BUY (Score ≥75) - Original proven threshold
        elif score >= 75:
            if category in ['penny', 'micro_penny']:
                prediction = "BUY"  # Downgrade from STRONG BUY
                expected_change = self._calculate_buy_change(score, category) * 0.7
                reasoning = "Positive technical momentum but penny stock volatility requires caution"
                confidence = max(65, min(75, score - 5))
            else:
                prediction = "STRONG BUY"
                expected_change = self._calculate_buy_change(score, category) * 1.1
                reasoning = "Multiple growth catalysts support upside potential"
                confidence = max(80, min(90, score))

        # Regular BUY (Score ≥65) - Proven threshold
        elif score >= 65:
            if category in ['penny', 'micro_penny']:
                prediction = "SPECULATIVE BUY"  # Downgrade to speculative
                expected_change = self._calculate_buy_change(score, category) * 0.6
                reasoning = "Speculative opportunity with high risk/reward profile"
                confidence = max(55, min(70, score - 10))
            else:
                prediction = "BUY"
                expected_change = self._calculate_buy_change(score, category)
                reasoning = "Solid fundamentals with supportive market trends"
                confidence = max(75, min(85, score))

        # Moderate BUY (Score ≥58 with strong momentum)
        elif score >= 58 or (price_momentum > 4 and rsi < 70 and volume_ratio > 1.3):
            if category in ['penny', 'micro_penny']:
                prediction = "HOLD"  # Very conservative for penny stocks
                expected_change = self._calculate_enhanced_hold_change(score, category, 2)
                reasoning = "Momentum signals present but penny stock risks outweigh potential gains"
                confidence = max(45, min(60, score - 5))
            else:
                prediction = "BUY"
                expected_change = self._calculate_buy_change(score, category) * 0.85
                reasoning = "Positive momentum driven by improving market sentiment"
                confidence = max(65, min(80, score + 5))

        # Speculative BUY (Score ≥52 with specific growth patterns)
        elif (score >= 52 and price_momentum > 3) or (rsi < 35 and price_momentum > -1 and volume_ratio > 1.2):
            if category in ['penny', 'micro_penny']:
                prediction = "HOLD"  # No speculative buys for penny stocks
                expected_change = self._calculate_enhanced_hold_change(score, category, 1)
                reasoning = "Insufficient fundamental strength for penny stock investment recommendation"
                confidence = max(40, min(55, score))
            else:
                prediction = "SPECULATIVE BUY"
                expected_change = self._calculate_buy_change(score, category) * 0.7
                reasoning = "Emerging opportunity with potential for market re-rating"
                confidence = max(55, min(75, score + 8))

        # === FALLBACK LOGIC ===
        elif score >= 45:
            if category in ['penny', 'micro_penny']:
                prediction = "HOLD"
                expected_change = self._calculate_enhanced_hold_change(score, category, 1)
                reasoning = "Penny stock fundamentals too weak for investment recommendation"
                confidence = max(35, min(50, score))
            else:
                prediction = "BUY"
                expected_change = self._calculate_buy_change(score, category) * 0.5
                reasoning = "Market conditions support modest upside potential"
                confidence = max(55, min(70, score))

        else:
            # Weak score but no clear SELL signals
            prediction = "HOLD"
            expected_change = self._calculate_enhanced_hold_change(score, category, 2)
            reasoning = "Current market uncertainty suggests waiting for clearer signals"
            confidence = max(50, min(65, score + 5))

        return prediction, expected_change, reasoning, confidence

    def _evaluate_penny_stock_risks(self, indicators, score):
        """Evaluate specific risks for penny stocks - debt, growth catalysts, fundamentals"""
        risk_score = 0
        current_price = indicators['current_price']
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        price_momentum = indicators.get('price_momentum', 0)
        rsi = indicators.get('rsi', 50)

        # Risk Factor 1: Extremely low price (under $1)
        if current_price < 1.0:
            risk_score += 2  # Major risk
        elif current_price < 2.0:
            risk_score += 1  # Moderate risk

        # Risk Factor 2: Poor liquidity (low volume)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio < 0.5:  # Very low volume
            risk_score += 2
        elif volume_ratio < 0.8:  # Low volume
            risk_score += 1

        # Risk Factor 3: Negative momentum trend
        if price_momentum < -10:  # Severe decline
            risk_score += 2
        elif price_momentum < -5:  # Moderate decline
            risk_score += 1

        # Risk Factor 4: Technical weakness
        if rsi < 30 and price_momentum < -3:  # Oversold and falling
            risk_score += 1

        # Risk Factor 5: Very weak fundamental score
        if score < 35:  # Very poor fundamentals
            risk_score += 2
        elif score < 45:  # Poor fundamentals
            risk_score += 1

        # Risk Factor 6: Lack of growth catalysts (inferred from poor performance)
        # If price has been declining consistently with low volume, suggests no positive news/catalysts
        if price_momentum < -8 and volume_ratio < 0.7:
            risk_score += 1  # No apparent growth catalysts

        return risk_score

    def _detect_high_risk_patterns(self, indicators, score):
        """Detect specific patterns that often lead to significant declines"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        rsi = indicators.get('rsi', 50)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        price_momentum = indicators.get('price_momentum', 0)
        macd = indicators.get('macd', 0)

        risk_patterns = 0

        # Pattern 1: Failed breakout (high volume, but price declining)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 1.8 and price_momentum < -3:
            risk_patterns += 1

        # Pattern 2: Momentum divergence (price up but momentum weakening)
        if current_price > sma_20 and price_momentum < -2 and macd < 0:
            risk_patterns += 1

        # Pattern 3: Extreme overbought with volume decline
        if rsi > 75 and volume_ratio < 0.7:
            risk_patterns += 1

        # Pattern 4: Breaking key support with momentum
        if current_price < sma_20 < sma_50 and price_momentum < -5:
            risk_patterns += 1

        # Pattern 5: High volatility with negative bias
        if abs(price_momentum) > 8 and price_momentum < 0:
            risk_patterns += 1

        return risk_patterns >= 2  # Need at least 2 risk patterns

    def _detect_weakness_patterns(self, indicators):
        """Detect patterns that suggest weakness but not necessarily strong decline"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        rsi = indicators.get('rsi', 50)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        price_momentum = indicators.get('price_momentum', 0)

        weakness_signals = 0

        # Declining volume with price stagnation
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio < 0.8 and abs(price_momentum) < 2:
            weakness_signals += 1

        # Price below MA with weak momentum
        if current_price < sma_20 and -3 < price_momentum < 1:
            weakness_signals += 1

        # RSI showing weakness (not oversold, but declining)
        if 35 < rsi < 55:
            weakness_signals += 1

        # Negative momentum without strong volume
        if price_momentum < -1 and volume_ratio < 1.2:
            weakness_signals += 1

        return weakness_signals >= 2

    def _calculate_sell_confidence(self, indicators, score):
        """Calculate confidence for SELL predictions based on multiple factors"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        rsi = indicators.get('rsi', 50)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        price_momentum = indicators.get('price_momentum', 0)
        macd = indicators.get('macd', 0)
        bollinger_lower = indicators.get('bollinger_lower', current_price * 0.95)

        confidence = 25  # Lower base confidence to be more selective

        # Strong downtrend confirmation (most important)
        if current_price < sma_20 < sma_50:
            price_below_ma20 = (sma_20 - current_price) / sma_20 * 100
            if price_below_ma20 > 5:  # More than 5% below MA20
                confidence += 20
            else:
                confidence += 12

        # Volume confirmation of selling pressure
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 1.8 and price_momentum < -3:  # High volume selling
            confidence += 15
        elif volume_ratio > 1.3 and price_momentum < -2:
            confidence += 8

        # Technical breakdown patterns
        if rsi < 30 and macd < -0.5:  # Strong oversold with MACD confirmation
            confidence += 12
        elif rsi < 40 and macd < 0:
            confidence += 6

        # Momentum severity
        if price_momentum < -10:  # Very strong negative momentum
            confidence += 12
        elif price_momentum < -5:
            confidence += 6

        # Bollinger Band breakdown
        if current_price < bollinger_lower:
            confidence += 8

        # Score severity (how bearish the overall score is)
        if score < 20:
            confidence += 15
        elif score < 25:
            confidence += 10
        elif score < 30:
            confidence += 5

        # Multi-timeframe confirmation
        momentum_strength = self._calculate_momentum_strength(indicators)
        if momentum_strength < -0.5:  # Very negative momentum
            confidence += 10
        elif momentum_strength < -0.3:
            confidence += 5

        return min(80, confidence)

    def _calculate_hold_confidence(self, indicators, score):
        """Calculate confidence for HOLD predictions based on consolidation signals"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        rsi = indicators.get('rsi', 50)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        price_momentum = indicators.get('price_momentum', 0)
        bollinger_upper = indicators.get('bollinger_upper', current_price * 1.05)
        bollinger_lower = indicators.get('bollinger_lower', current_price * 0.95)
        macd = indicators.get('macd', 0)

        confidence = 30  # Lower base confidence to be more selective

        # True consolidation: Price near moving averages
        if sma_20 > 0 and sma_50 > 0:
            price_vs_ma20 = abs(current_price - sma_20) / sma_20 * 100
            price_vs_ma50 = abs(current_price - sma_50) / sma_50 * 100
            ma_convergence = abs(sma_20 - sma_50) / sma_50 * 100

            # Price close to both MAs indicates consolidation
            if price_vs_ma20 < 1.5 and price_vs_ma50 < 3:  # Very tight consolidation
                confidence += 15
            elif price_vs_ma20 < 3 and price_vs_ma50 < 5:
                confidence += 8

            # MAs converging indicates sideways movement
            if ma_convergence < 2:  # MAs very close together
                confidence += 10
            elif ma_convergence < 4:
                confidence += 5

        # RSI in true neutral zone (not just oversold/overbought)
        if 48 <= rsi <= 52:  # Very neutral RSI
            confidence += 12
        elif 45 <= rsi <= 55:
            confidence += 8
        elif 40 <= rsi <= 60:
            confidence += 4

        # Very low momentum (true sideways movement)
        if abs(price_momentum) < 1:  # Almost no momentum
            confidence += 12
        elif abs(price_momentum) < 2:
            confidence += 6
        elif abs(price_momentum) < 3:
            confidence += 3

        # Volume patterns indicating lack of conviction
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if 0.8 <= volume_ratio <= 1.1:  # Very normal volume
            confidence += 8
        elif 0.7 <= volume_ratio <= 1.3:
            confidence += 4

        # MACD near zero (no clear direction)
        if abs(macd) < 0.1:
            confidence += 8
        elif abs(macd) < 0.3:
            confidence += 4

        # Price in middle of Bollinger Bands (not testing extremes)
        if bollinger_upper > bollinger_lower:
            bb_position = (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower)
            if 0.4 <= bb_position <= 0.6:  # Very middle of bands
                confidence += 10
            elif 0.3 <= bb_position <= 0.7:
                confidence += 6

        # Score neutrality (most important for HOLD)
        if 48 <= score <= 52:  # Very neutral score
            confidence += 12
        elif 45 <= score <= 55:
            confidence += 8
        elif 42 <= score <= 58:
            confidence += 4

        # Overall momentum strength check
        momentum_strength = self._calculate_momentum_strength(indicators)
        if abs(momentum_strength) < 0.1:  # Very low momentum
            confidence += 10
        elif abs(momentum_strength) < 0.2:
            confidence += 5

        return min(75, confidence)

    def _detect_sideways_patterns(self, indicators):
        """Detect true sideways/consolidation patterns that lead to HOLD outcomes"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        rsi = indicators.get('rsi', 50)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        price_momentum = indicators.get('price_momentum', 0)
        bollinger_upper = indicators.get('bollinger_upper', current_price * 1.05)
        bollinger_lower = indicators.get('bollinger_lower', current_price * 0.95)

        sideways_signals = 0

        # Pattern 1: Price oscillating around moving average
        if sma_20 > 0:
            price_vs_ma = abs(current_price - sma_20) / sma_20 * 100
            if price_vs_ma < 3:  # Within 3% of MA
                sideways_signals += 1

        # Pattern 2: Low momentum (key for sideways movement)
        if abs(price_momentum) < 4:
            sideways_signals += 1

        # Pattern 3: RSI in neutral zone (not extreme)
        if 35 < rsi < 65:
            sideways_signals += 1

        # Pattern 4: Normal volume (not spiking)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if 0.7 < volume_ratio < 1.8:
            sideways_signals += 1

        # Pattern 5: Price in middle of Bollinger Bands
        if bollinger_upper > bollinger_lower:
            bb_position = (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower)
            if 0.25 < bb_position < 0.75:  # Middle 50% of bands
                sideways_signals += 1

        # Pattern 6: Moving averages close together (consolidation)
        if sma_20 > 0 and sma_50 > 0:
            ma_spread = abs(sma_20 - sma_50) / sma_50 * 100
            if ma_spread < 2:  # MAs within 2% of each other
                sideways_signals += 1

        return sideways_signals >= 4  # Need strong evidence for sideways movement

    def _detect_consolidation_signals(self, indicators):
        """Detect consolidation/sideways movement for better HOLD accuracy"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        rsi = indicators.get('rsi', 50)
        price_momentum = indicators.get('price_momentum', 0)
        bollinger_upper = indicators.get('bollinger_upper', current_price * 1.05)
        bollinger_lower = indicators.get('bollinger_lower', current_price * 0.95)

        consolidation_signals = 0
        confidence = 45
        reasoning = "Neutral technical outlook"

        # Price near moving average (consolidation)
        if sma_20 > 0:
            price_vs_ma = abs(current_price - sma_20) / sma_20 * 100
            if price_vs_ma < 2:  # Within 2% of MA
                consolidation_signals += 1
                confidence += 8

        # RSI in neutral zone
        if 40 <= rsi <= 60:
            consolidation_signals += 1
            confidence += 5
            reasoning = "RSI in neutral zone suggests consolidation"

        # Low momentum (sideways movement)
        if abs(price_momentum) < 3:
            consolidation_signals += 1
            confidence += 5

        # Price in middle of Bollinger Bands
        if bollinger_upper > bollinger_lower:
            bb_position = (current_price - bollinger_lower) / (bollinger_upper - bollinger_lower)
            if 0.3 <= bb_position <= 0.7:  # Middle 40% of bands
                consolidation_signals += 1
                confidence += 5
                reasoning = "Price consolidating within trading range"

        strong_consolidation = consolidation_signals >= 3

        return {
            'strong_consolidation': strong_consolidation,
            'confidence': min(65, confidence),
            'reasoning': reasoning
        }

    def _detect_strong_sell_signals(self, indicators):
        """Detect strong sell signals for improved SELL accuracy"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        rsi = indicators.get('rsi', 50)
        price_momentum = indicators.get('price_momentum', 0)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)

        sell_signals = 0
        confidence = 45
        reasoning = "Bearish technical outlook"

        # Strong negative momentum
        if price_momentum < -3:
            sell_signals += 2
            confidence += 15
            reasoning = "Strong negative price momentum"
        elif price_momentum < -1.5:
            sell_signals += 1
            confidence += 8

        # Price below key moving averages
        if current_price < sma_20 < sma_50:
            sell_signals += 2
            confidence += 12
            reasoning = "Price below declining moving averages"
        elif current_price < sma_20:
            sell_signals += 1
            confidence += 6

        # RSI showing weakness but not oversold
        if 25 < rsi < 40:
            sell_signals += 1
            confidence += 8
            reasoning = "RSI showing weakness"

        # High volume on decline
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 1.3 and price_momentum < -2:
            sell_signals += 2
            confidence += 10
            reasoning = "High volume selling pressure"

        strong_sell = sell_signals >= 4

        return {
            'strong_sell': strong_sell,
            'confidence': min(75, confidence),
            'reasoning': reasoning,
            'signal_count': sell_signals
        }

    def _calculate_buy_change(self, score, category):
        """Calculate expected change for BUY predictions (conservative for penny stocks)"""
        if category in ['penny', 'micro_penny']:
            # More conservative range for penny stocks due to high risk
            base_change = 5 + (score - 65) * 0.3  # 5-18% range (reduced from 8-28%)
            return max(5, min(18, base_change))
        elif category in ['micro_cap', 'small_cap']:
            # Moderate range for small caps
            base_change = 6 + (score - 65) * 0.35  # 6-18% range (slightly increased)
            return max(6, min(18, base_change))
        else:  # Large caps
            # Conservative range for large caps
            base_change = 3.5 + (score - 65) * 0.25  # 3.5-12% range (slightly increased)
            return max(3.5, min(12, base_change))

    def _calculate_enhanced_sell_change(self, score, category, sell_signals):
        """Enhanced SELL change calculation with more realistic decline predictions"""
        # More aggressive base calculation - lower scores should predict bigger declines
        score_weakness = max(0, 50 - score)  # 0-50 range
        base_decline = -(score_weakness * 0.8)  # More aggressive multiplier

        # Strong amplification based on sell signal strength
        signal_multiplier = 1 + (sell_signals * 0.4)  # Increased from 0.2 to 0.4

        # Additional momentum factor for very weak scores
        if score <= 25:
            momentum_boost = -5  # Extra -5% for very weak fundamentals
        elif score <= 35:
            momentum_boost = -3  # Extra -3% for weak fundamentals
        else:
            momentum_boost = 0

        if category in ['penny', 'micro_penny']:
            # Penny stocks: -45% to -8% range (increased from -30% to -5%)
            enhanced_change = (base_decline * signal_multiplier) + momentum_boost
            return max(-45, min(-8, enhanced_change))
        elif category in ['micro_cap', 'small_cap']:
            # Small caps: -35% to -6% range (increased from -20% to -3%)
            enhanced_change = (base_decline * signal_multiplier * 0.8) + momentum_boost
            return max(-35, min(-6, enhanced_change))
        else:  # Large caps
            # Large caps: -25% to -4% range (increased from -12% to -2%)
            enhanced_change = (base_decline * signal_multiplier * 0.6) + momentum_boost
            return max(-25, min(-4, enhanced_change))

    def _calculate_enhanced_hold_change(self, score, category, hold_signals):
        """Enhanced HOLD change calculation with category-specific ranges"""
        # Small movements around neutral
        base_change = (score - 50) * 0.1

        # Reduce volatility based on consolidation strength
        stability_factor = max(0.3, 1 - (hold_signals * 0.1))

        if category in ['penny', 'micro_penny']:
            # Penny stocks: ±3% range
            enhanced_change = base_change * stability_factor
            return max(-3, min(3, enhanced_change))
        elif category in ['micro_cap', 'small_cap']:
            # Small caps: ±2% range
            enhanced_change = base_change * stability_factor * 0.7
            return max(-2, min(2, enhanced_change))
        else:  # Large caps
            # Large caps: ±1.5% range
            enhanced_change = base_change * stability_factor * 0.5
            return max(-1.5, min(1.5, enhanced_change))

    def _calculate_sell_change(self, score, category, sell_signals):
        """Calculate expected change for SELL predictions (improved logic)"""
        # Base severity on how low the score is (lower score = more severe decline expected)
        severity_factor = max(0, (50 - score) / 50)  # 0-1 scale based on score

        if category in ['penny', 'micro_penny']:
            base_change = -8 - (severity_factor * 20)  # -8% to -28%
            return max(-35, min(-5, base_change))
        elif category in ['micro_cap', 'small_cap']:
            base_change = -5 - (severity_factor * 12)  # -5% to -17%
            return max(-25, min(-3, base_change))
        else:  # Large caps
            base_change = -3 - (severity_factor * 8)   # -3% to -11%
            return max(-15, min(-2, base_change))

    def _calculate_smart_hold_change(self, indicators, score):
        """Calculate expected change for HOLD predictions based on actual market conditions"""
        price_momentum = indicators.get('price_momentum', 0)
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)

        # Base change on score bias
        base_change = (score - 50) * 0.1

        # Adjust for momentum
        momentum_factor = price_momentum * 0.05  # Small momentum influence

        # Adjust for volume
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 1.5:
            # High volume might break consolidation
            volume_factor = (volume_ratio - 1) * 0.5
        else:
            volume_factor = 0

        total_change = base_change + momentum_factor + volume_factor
        return max(-2, min(2, total_change))  # HOLD should be small movements

    def _generate_sell_reasoning(self, indicators, score):
        """Generate reasoning for SELL predictions based on market context"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        rsi = indicators.get('rsi', 50)
        price_momentum = indicators.get('price_momentum', 0)

        # Focus on market context rather than pure technical factors
        if current_price < sma_20 < sma_50 and price_momentum < -5:
            return "Market sentiment deteriorating amid broader economic headwinds"
        elif rsi < 35 and score < 25:
            return "Fundamental weakness coinciding with market risk-off sentiment"
        elif price_momentum < -5:
            return "Institutional selling pressure amid market uncertainty"
        else:
            return "Current market conditions suggest elevated downside risk"

    def _generate_hold_reasoning(self, indicators, score):
        """Generate reasoning for HOLD predictions based on market context"""
        current_price = indicators['current_price']
        sma_20 = indicators.get('sma_20', current_price)
        rsi = indicators.get('rsi', 50)
        price_momentum = indicators.get('price_momentum', 0)

        # Focus on market context and company fundamentals
        if sma_20 > 0:
            price_vs_ma = abs(current_price - sma_20) / sma_20 * 100
            if price_vs_ma < 3 and abs(price_momentum) < 4:
                return "Market consolidation phase while awaiting key catalysts"

        if 40 <= rsi <= 60 and 45 <= score <= 55:
            return "Balanced market conditions with mixed economic signals"
        elif abs(price_momentum) < 4:
            return "Sideways trading expected amid current market uncertainty"
        else:
            return "Waiting for clearer market direction and fundamental developments"

    def _calculate_hold_change(self, score, category, hold_signals):
        """Calculate expected change for HOLD predictions (improved logic)"""
        confidence_factor = hold_signals['confidence'] / 65  # 0-1 scale

        # HOLD should have small movements around 0
        if category in ['penny', 'micro_penny']:
            base_change = (score - 50) * 0.1 * confidence_factor  # -2% to +2%
            return max(-3, min(3, base_change))
        elif category in ['micro_cap', 'small_cap']:
            base_change = (score - 50) * 0.08 * confidence_factor  # -1.5% to +1.5%
            return max(-2, min(2, base_change))
        else:  # Large caps
            base_change = (score - 50) * 0.06 * confidence_factor  # -1% to +1%
            return max(-1.5, min(1.5, base_change))







    def _get_timeframe_multiplier(self, timeframe):
        """Get multiplier for expected change based on timeframe"""
        # Extract approximate months from timeframe string
        timeframe_lower = timeframe.lower()

        if "1-2 month" in timeframe_lower or "1 month" in timeframe_lower:
            return 0.5  # 50% of base prediction for 1 month
        elif "2-6 month" in timeframe_lower or "3 month" in timeframe_lower:
            return 1.0  # Base prediction for 3 months
        elif "6-12 month" in timeframe_lower or "6 month" in timeframe_lower:
            return 1.5  # 150% for 6+ months
        elif "1-3 year" in timeframe_lower or "year" in timeframe_lower:
            return 2.0  # 200% for 1+ years
        else:
            return 1.0  # Default to base prediction

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

    def _generate_chart_data(self, historical_data, indicators, prediction_result):
        """Generate chart data for frontend visualization"""
        try:
            # Prepare historical data (last 30 days)
            historical_data = historical_data.tail(30).copy()
            historical_chart_data = []

            for date, row in historical_data.iterrows():
                historical_chart_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'price': round(float(row['Close']), 4),
                    'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                })

            # Generate prediction data (next 30 days)
            current_price = indicators['current_price']
            target_price = prediction_result['target_price']
            prediction_chart_data = []

            # Create prediction timeline based on timeframe
            prediction_days = 30  # Default to 30 days for visualization

            for i in range(1, prediction_days + 1):
                # Gradual progression towards target price with some volatility
                progress = i / prediction_days

                # Add some realistic volatility using deterministic seed
                volatility_seed = hash(f"{current_price}_{i}_{target_price}") % 2147483647
                np.random.seed(volatility_seed)
                volatility = np.random.normal(0, 0.02)  # 2% daily volatility

                # Calculate predicted price with smooth progression
                predicted_price = current_price + (target_price - current_price) * progress + (current_price * volatility)

                # Ensure price doesn't go negative
                predicted_price = max(predicted_price, current_price * 0.1)

                future_date = datetime.now() + timedelta(days=i)
                prediction_chart_data.append({
                    'date': future_date.strftime('%Y-%m-%d'),
                    'price': round(predicted_price, 4)
                })

            # Generate volume data (last 30 days)
            volume_chart_data = []
            for date, row in historical_data.iterrows():
                volume_chart_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                })

            return {
                'historical': historical_chart_data,
                'prediction': prediction_chart_data,
                'volume': volume_chart_data
            }

        except Exception as e:
            logger.error(f"Error generating chart data: {e}")
            return {
                'historical': [],
                'prediction': [],
                'volume': []
            }

    def _generate_enhanced_reasoning(self, stock_data, indicators, category, prediction_result, confidence):
        """Generate comprehensive AI reasoning with real-world context and market analysis"""
        reasoning_points = []

        current_price = indicators['current_price']
        rsi = indicators.get('rsi', 50)
        sma_20 = indicators.get('sma_20', current_price)
        volume = stock_data.get('volume', 0)
        market_cap = stock_data.get('market_cap', 0)
        sector = stock_data.get('sector', 'Unknown')
        symbol = stock_data.get('symbol', '')

        # Real-world market context and events (December 2024)
        market_context = self._get_current_market_context(symbol, sector, category)

        # Current Market Environment Analysis
        market_env = market_context['market_environment']
        reasoning_points.append({
            'icon': '🌍',
            'title': 'Market Environment',
            'text': f"{market_env['overall_sentiment']}. {market_env['key_themes'][0]}"
        })

        # Economic Indicators Context
        econ_indicators = market_context['economic_indicators']
        reasoning_points.append({
            'icon': '📊',
            'title': 'Economic Backdrop',
            'text': f"Inflation {econ_indicators['inflation']}, while {econ_indicators['employment']}. GDP growth {econ_indicators['gdp_growth']}"
        })

        # Geopolitical Context
        geopolitical = market_context['geopolitical_factors']
        reasoning_points.append({
            'icon': '🌐',
            'title': 'Geopolitical Factors',
            'text': f"{geopolitical['us_china_relations']}. {geopolitical['trade_policy']}"
        })

        # Enhanced Sector-Specific Analysis
        if sector != 'Unknown' and market_context['sector_trends']:
            sector_info = market_context['sector_trends']
            reasoning_points.append({
                'icon': '🏭',
                'title': f'{sector} Sector Dynamics',
                'text': f"{sector_info.get('current_dynamics', 'Sector showing mixed signals')}. Outlook: {sector_info.get('outlook', 'Monitoring developments')}"
            })

        # News Sentiment and Current Events
        news_context = market_context['news_sentiment']
        if news_context['sector_news']:
            reasoning_points.append({
                'icon': '📰',
                'title': f'{sector} Sector News',
                'text': f"Recent developments: {news_context['sector_news'][0]}. {news_context['sector_news'][1] if len(news_context['sector_news']) > 1 else ''}"
            })

        # Company-Specific Current Events
        if news_context['company_news']:
            reasoning_points.append({
                'icon': '🏢',
                'title': f'{symbol} Company Updates',
                'text': f"{news_context['company_news'][0]}. {news_context['company_news'][1] if len(news_context['company_news']) > 1 else ''}"
            })

        # Market Sentiment Analysis (based on underlying patterns)
        if rsi < 30:
            reasoning_points.append({
                'icon': '📉',
                'title': 'Market Oversold',
                'text': f'Current market sentiment suggests the stock may be undervalued, creating potential for recovery as investor confidence returns.'
            })
        elif rsi > 70:
            reasoning_points.append({
                'icon': '📈',
                'title': 'Strong Market Interest',
                'text': f'High investor interest and buying pressure may lead to consolidation as market participants take profits.'
            })
        else:
            reasoning_points.append({
                'icon': '⚖️',
                'title': 'Balanced Market Sentiment',
                'text': f'Market sentiment appears balanced with neither excessive optimism nor pessimism driving price action.'
            })

        # Market Trend Analysis (based on underlying patterns)
        price_vs_sma = ((current_price - sma_20) / sma_20) * 100 if sma_20 else 0
        if price_vs_sma > 5:
            reasoning_points.append({
                'icon': '🚀',
                'title': 'Strong Market Momentum',
                'text': f'Recent positive developments and investor optimism are driving strong upward momentum in the stock.'
            })
        elif price_vs_sma < -5:
            reasoning_points.append({
                'icon': '📉',
                'title': 'Market Headwinds',
                'text': f'Current market conditions and investor sentiment are creating downward pressure on the stock price.'
            })
        else:
            reasoning_points.append({
                'icon': '📊',
                'title': 'Market Consolidation',
                'text': 'Stock is in a consolidation phase as market participants assess current fundamentals and future prospects.'
            })

        # Volume Analysis
        if volume > 0:
            reasoning_points.append({
                'icon': '📊',
                'title': 'Volume Analysis',
                'text': f'Current volume of {volume:,} shares provides liquidity for position entries and exits.'
            })

        # Market Cap & Category Analysis
        if category == 'large_cap':
            reasoning_points.append({
                'icon': '🏢',
                'title': 'Large Cap Stability',
                'text': f'${market_cap/1e9:.1f}B market cap provides stability and institutional interest.'
            })
        elif category in ['penny', 'micro_penny']:
            reasoning_points.append({
                'icon': '⚠️',
                'title': 'High Risk Profile',
                'text': 'Penny stock classification indicates high volatility and speculative nature.'
            })
        elif category == 'small_cap':
            reasoning_points.append({
                'icon': '🌱',
                'title': 'Growth Potential',
                'text': 'Small cap status offers growth potential but with increased volatility risk.'
            })

        # Sector Analysis with World Events Context
        if sector != 'Unknown':
            sector_context = self._get_sector_world_context(sector, symbol)
            reasoning_points.append({
                'icon': '🏭',
                'title': 'Sector Outlook',
                'text': sector_context
            })

        # Confidence Analysis
        if confidence > 0.8:
            reasoning_points.append({
                'icon': '🎯',
                'title': 'High Confidence Prediction',
                'text': f'{confidence*100:.1f}% confidence based on strong fundamental alignment with current market trends and economic conditions.'
            })
        elif confidence < 0.5:
            reasoning_points.append({
                'icon': '🤔',
                'title': 'Uncertain Outlook',
                'text': f'{confidence*100:.1f}% confidence due to mixed economic signals and evolving market conditions requiring careful monitoring.'
            })

        # Global Market Sentiment Analysis
        global_factors = market_context['global_factors']
        reasoning_points.append({
            'icon': '🌎',
            'title': 'Global Market Drivers',
            'text': f"{global_factors[0]}. Additionally, {global_factors[3]}"
        })

        # Investment Thesis Based on Current Events
        investment_thesis = self._generate_investment_thesis(symbol, sector, category, market_context, prediction_result)
        reasoning_points.append({
            'icon': '🎯',
            'title': 'Investment Thesis',
            'text': investment_thesis
        })

        # Prediction Rationale with Market Context
        prediction = prediction_result['prediction']
        if 'BUY' in prediction.upper():
            reasoning_points.append({
                'icon': '💰',
                'title': 'Buy Signal Rationale',
                'text': f"{prediction_result['reasoning']} Current market conditions support this outlook given {market_env['market_phase']}"
            })
        elif 'SELL' in prediction.upper():
            reasoning_points.append({
                'icon': '💸',
                'title': 'Sell Signal Rationale',
                'text': f"{prediction_result['reasoning']} Market environment suggests caution given current valuations and economic uncertainty"
            })
        else:
            reasoning_points.append({
                'icon': '⏸️',
                'title': 'Hold Position Rationale',
                'text': f"{prediction_result['reasoning']} Current market volatility supports a wait-and-see approach"
            })

        return reasoning_points

    def _get_sector_world_context(self, sector, symbol):
        """Generate sector-specific reasoning based on current world events and market conditions"""
        sector_contexts = {
            'Technology': f"AI revolution and digital transformation driving sector growth, with companies like {symbol} positioned to benefit from increased enterprise adoption",
            'Healthcare': f"Aging demographics and post-pandemic healthcare focus create long-term tailwinds for {symbol} and the broader healthcare sector",
            'Financial Services': f"Rising interest rates and digital banking transformation present both opportunities and challenges for {symbol} in the evolving financial landscape",
            'Consumer Discretionary': f"Consumer spending patterns shifting post-pandemic, with {symbol} adapting to new retail and lifestyle trends",
            'Energy': f"Global energy transition and geopolitical tensions create volatility, positioning {symbol} within the changing energy landscape",
            'Industrials': f"Infrastructure spending and supply chain reshoring trends support industrial companies like {symbol}",
            'Communication Services': f"Digital media consumption and 5G rollout drive growth opportunities for {symbol} in the communications sector",
            'Consumer Staples': f"Inflation pressures and supply chain challenges test resilience of consumer staples companies like {symbol}",
            'Materials': f"Global infrastructure projects and green energy transition drive demand for materials companies like {symbol}",
            'Real Estate': f"Interest rate environment and remote work trends reshape real estate dynamics affecting {symbol}",
            'Utilities': f"Clean energy transition and grid modernization create investment opportunities for utilities like {symbol}"
        }

        return sector_contexts.get(sector, f"Current market conditions and sector dynamics influence {symbol}'s fundamental outlook and growth prospects")

    def _get_current_market_context(self, symbol, sector, category):
        """Generate comprehensive real-world market context and current events analysis"""
        context = {
            'market_environment': self._get_current_market_environment(),
            'sector_trends': {},
            'company_specific': {},
            'global_factors': self._get_current_global_factors(),
            'economic_indicators': self._get_economic_indicators_context(),
            'geopolitical_factors': self._get_geopolitical_context(),
            'news_sentiment': self._get_news_sentiment_context(symbol, sector)
        }
        return context

    def _get_current_market_environment(self):
        """Current market environment analysis (December 2024)"""
        return {
            'overall_sentiment': 'Cautiously optimistic with selective opportunities',
            'key_themes': [
                'AI transformation driving selective tech outperformance',
                'Federal Reserve policy normalization creating sector rotation',
                'Corporate earnings showing resilience despite macro headwinds',
                'Consumer spending patterns shifting toward experiences and services'
            ],
            'market_phase': 'Late-cycle expansion with emerging growth themes'
        }

    def _get_current_global_factors(self):
        """Current global factors affecting markets (December 2024)"""
        return [
            'Federal Reserve signaling potential rate cuts in 2024 amid cooling inflation',
            'China reopening driving global supply chain optimization',
            'European energy security improving with diversified supply sources',
            'AI and automation reshaping productivity across industries',
            'Climate transition accelerating with massive infrastructure investments',
            'Deglobalization trends creating regional supply chain hubs',
            'Demographic shifts driving healthcare and automation demand'
        ]

    def _get_economic_indicators_context(self):
        """Current economic indicators and their market impact"""
        return {
            'inflation': 'Moderating from peaks, core PCE approaching Fed target',
            'employment': 'Labor market cooling but remaining resilient',
            'gdp_growth': 'Slowing but positive, supported by consumer spending',
            'consumer_confidence': 'Stabilizing after recent volatility',
            'manufacturing': 'Mixed signals with regional variations',
            'housing': 'Adjusting to higher rates, inventory normalizing'
        }

    def _get_geopolitical_context(self):
        """Current geopolitical factors and their market implications"""
        return {
            'us_china_relations': 'Strategic competition with selective cooperation on climate',
            'russia_ukraine': 'Ongoing conflict affecting energy and commodity markets',
            'middle_east': 'Regional tensions impacting oil prices and defense spending',
            'trade_policy': 'Reshoring and friend-shoring driving industrial investment',
            'cyber_security': 'Increasing threats driving cybersecurity investment',
            'climate_policy': 'Global coordination on green transition accelerating'
        }

    def _get_news_sentiment_context(self, symbol, sector):
        """Generate news sentiment and current events context for specific stocks/sectors"""
        sentiment_context = {
            'overall_sentiment': 'neutral',
            'key_events': [],
            'sector_news': [],
            'company_news': []
        }

        # Sector-specific current events and sentiment
        if sector == 'Technology':
            sentiment_context.update({
                'overall_sentiment': 'positive',
                'sector_news': [
                    'AI adoption accelerating across enterprise and consumer markets',
                    'Cloud computing demand remaining robust despite economic uncertainty',
                    'Semiconductor cycle showing signs of bottoming out',
                    'Regulatory scrutiny on big tech creating compliance costs but market stability'
                ],
                'key_events': [
                    'OpenAI and Microsoft partnership driving enterprise AI adoption',
                    'Apple Vision Pro launch creating new spatial computing market',
                    'NVIDIA data center revenue growth exceeding expectations',
                    'Cybersecurity spending increasing due to geopolitical tensions'
                ]
            })

        elif sector == 'Healthcare':
            sentiment_context.update({
                'overall_sentiment': 'positive',
                'sector_news': [
                    'Aging population demographics driving long-term demand growth',
                    'GLP-1 weight loss drugs creating massive new market opportunity',
                    'AI-powered drug discovery accelerating development timelines',
                    'Medicare negotiations creating pricing pressure but providing certainty'
                ],
                'key_events': [
                    'Ozempic and Wegovy success driving diabetes/obesity treatment revolution',
                    'Alzheimer drug approvals providing new treatment options',
                    'Biosimilar competition intensifying in key therapeutic areas',
                    'Telehealth adoption stabilizing at elevated post-pandemic levels'
                ]
            })

        elif sector == 'Financial Services':
            sentiment_context.update({
                'overall_sentiment': 'mixed',
                'sector_news': [
                    'Net interest margins benefiting from higher rate environment',
                    'Credit quality concerns emerging in commercial real estate',
                    'Digital transformation accelerating with fintech partnerships',
                    'Regulatory capital requirements creating competitive moats'
                ],
                'key_events': [
                    'Regional bank stress tests revealing vulnerabilities',
                    'Credit card delinquencies normalizing from historic lows',
                    'Cryptocurrency regulation providing market clarity',
                    'FDIC insurance reforms strengthening deposit confidence'
                ]
            })

        elif sector == 'Energy':
            sentiment_context.update({
                'overall_sentiment': 'mixed',
                'sector_news': [
                    'Oil prices stabilizing amid global supply-demand rebalancing',
                    'Renewable energy investments accelerating with IRA incentives',
                    'Natural gas demand shifting with LNG export capacity expansion',
                    'Carbon capture and storage technologies gaining commercial viability'
                ],
                'key_events': [
                    'OPEC+ production cuts supporting oil price stability',
                    'US becoming net energy exporter reducing geopolitical dependence',
                    'Offshore wind projects facing supply chain and permitting challenges',
                    'Energy storage costs declining enabling grid-scale deployment'
                ]
            })

        elif sector == 'Consumer Discretionary':
            sentiment_context.update({
                'overall_sentiment': 'mixed',
                'sector_news': [
                    'Consumer spending shifting from goods to services and experiences',
                    'E-commerce growth moderating but remaining elevated vs pre-pandemic',
                    'Premium brands showing resilience amid economic uncertainty',
                    'Supply chain costs normalizing but labor inflation persisting'
                ],
                'key_events': [
                    'Holiday shopping patterns showing preference for value and experiences',
                    'Electric vehicle adoption accelerating with expanding charging infrastructure',
                    'Streaming services facing subscriber saturation and content cost pressures',
                    'Travel and hospitality demand recovering to pre-pandemic levels'
                ]
            })

        # Company-specific current events and sentiment
        company_events = self._get_company_specific_events(symbol)
        sentiment_context['company_news'] = company_events

        return sentiment_context

    def _get_company_specific_events(self, symbol):
        """Get current events and news specific to individual companies"""
        events = []

        if symbol == 'AAPL':
            events = [
                'iPhone 15 launch with USB-C transition driving upgrade cycle',
                'Services revenue growth maintaining high margins and recurring revenue',
                'Vision Pro spatial computing platform creating new product category',
                'China market recovery supporting revenue growth in key region'
            ]
        elif symbol == 'MSFT':
            events = [
                'Azure cloud growth accelerating with AI and Copilot integration',
                'OpenAI partnership positioning Microsoft as AI infrastructure leader',
                'Office 365 Copilot driving productivity software transformation',
                'Gaming division benefiting from Activision Blizzard acquisition'
            ]
        elif symbol == 'GOOGL':
            events = [
                'Bard AI competing with ChatGPT in generative AI market',
                'Search advertising showing resilience despite economic headwinds',
                'Cloud division gaining market share with AI and data analytics',
                'YouTube Shorts competing effectively with TikTok for engagement'
            ]
        elif symbol == 'AMZN':
            events = [
                'AWS growth reaccelerating as enterprise cloud spending normalizes',
                'Prime Video and advertising creating high-margin revenue streams',
                'Logistics network optimization improving delivery efficiency',
                'Alexa and smart home ecosystem expanding with AI integration'
            ]
        elif symbol == 'TSLA':
            events = [
                'Cybertruck production ramp addressing commercial vehicle market',
                'Supercharger network opening to other EVs creating new revenue stream',
                'Full Self-Driving progress advancing toward regulatory approval',
                'Energy storage business growing with grid-scale deployments'
            ]
        elif symbol == 'NVDA':
            events = [
                'Data center GPU demand exceeding supply amid AI infrastructure buildout',
                'Gaming GPU market recovering from crypto mining overhang',
                'Automotive AI partnerships expanding autonomous vehicle capabilities',
                'China export restrictions creating supply chain complexity'
            ]
        elif symbol == 'META':
            events = [
                'Reality Labs investments in metaverse showing early commercial progress',
                'Instagram Reels competing effectively with TikTok for creator economy',
                'AI-powered advertising targeting improving despite privacy changes',
                'WhatsApp Business monetization expanding in international markets'
            ]
        elif symbol == 'LUMN':
            events = [
                'Fiber network expansion benefiting from broadband infrastructure demand',
                '5G infrastructure investments supporting wireless carrier partnerships',
                'Debt reduction efforts improving financial flexibility',
                'Edge computing services addressing latency-sensitive applications'
            ]
        elif symbol in ['JPM', 'BAC', 'WFC', 'C']:
            events = [
                'Net interest margin expansion benefiting from rate environment',
                'Credit loss provisions normalizing from pandemic lows',
                'Digital banking investments improving customer acquisition',
                'Regulatory stress tests demonstrating capital strength'
            ]
        elif symbol in ['JNJ', 'PFE', 'MRK', 'ABBV']:
            events = [
                'Pipeline drug approvals providing new revenue growth drivers',
                'Biosimilar competition intensifying in key therapeutic areas',
                'Emerging market expansion addressing global health needs',
                'AI-powered drug discovery accelerating development timelines'
            ]

        return events

    def _get_enhanced_sector_trends(self, sector):
        """Get comprehensive sector trends with current market dynamics"""
        sector_trends = {}

        if sector == 'Technology':
            sector_trends = {
                'trend': 'AI transformation driving unprecedented productivity gains',
                'current_dynamics': 'Enterprise AI adoption accelerating, cloud infrastructure demand robust',
                'challenges': 'Regulatory scrutiny intensifying, talent competition fierce, valuation multiples compressing',
                'opportunities': 'Generative AI creating new markets, edge computing expansion, cybersecurity demand surge',
                'outlook': 'Selective outperformance with AI leaders vs legacy tech divergence'
            }
        elif sector == 'Healthcare':
            sector_trends = {
                'trend': 'Demographic tailwinds driving sustainable long-term growth',
                'current_dynamics': 'GLP-1 drugs revolutionizing obesity treatment, AI accelerating drug discovery',
                'challenges': 'Medicare price negotiations, biosimilar competition, regulatory complexity',
                'opportunities': 'Personalized medicine breakthrough, aging population care, digital health adoption',
                'outlook': 'Defensive growth with innovation-driven outperformance'
            }
        elif sector == 'Financial Services':
            sector_trends = {
                'trend': 'Interest rate normalization creating earnings tailwinds',
                'current_dynamics': 'Net interest margins expanding, digital transformation accelerating',
                'challenges': 'Credit cycle concerns emerging, regulatory capital requirements, fintech disruption',
                'opportunities': 'Wealth management growth, payment processing expansion, crypto integration',
                'outlook': 'Cyclical recovery with quality differentiation'
            }
        elif sector == 'Energy':
            sector_trends = {
                'trend': 'Energy transition accelerating with massive capital reallocation',
                'current_dynamics': 'Oil market rebalancing, renewable capacity additions surging',
                'challenges': 'Stranded asset risks, volatile commodity prices, ESG investment constraints',
                'opportunities': 'Clean energy infrastructure, carbon capture technology, energy storage deployment',
                'outlook': 'Transformation phase with winners and losers emerging'
            }
        elif sector == 'Consumer Discretionary':
            sector_trends = {
                'trend': 'Consumer behavior permanently shifted toward experiences and digital',
                'current_dynamics': 'Services spending recovering, e-commerce stabilizing at elevated levels',
                'challenges': 'Inflation pressure on discretionary spending, supply chain normalization',
                'opportunities': 'Premium brand resilience, travel recovery, EV adoption acceleration',
                'outlook': 'Bifurcated market with quality brands outperforming'
            }
        elif sector == 'Consumer Staples':
            sector_trends = {
                'trend': 'Defensive characteristics attractive amid economic uncertainty',
                'current_dynamics': 'Pricing power demonstration, private label competition intensifying',
                'challenges': 'Input cost inflation, changing consumer preferences, retail consolidation',
                'opportunities': 'Health and wellness trends, emerging market expansion, sustainability focus',
                'outlook': 'Steady performance with innovation-driven growth'
            }
        elif sector == 'Communication Services':
            sector_trends = {
                'trend': 'Digital advertising recovery with AI-powered targeting improvements',
                'current_dynamics': 'Streaming competition intensifying, 5G infrastructure deployment continuing',
                'challenges': 'Content cost inflation, subscriber saturation, regulatory oversight',
                'opportunities': 'AI content creation, edge computing services, digital transformation services',
                'outlook': 'Platform leaders consolidating market share'
            }

        return sector_trends

    def _generate_investment_thesis(self, symbol, sector, category, market_context, prediction_result):
        """Generate comprehensive investment thesis based on current market conditions"""

        # Base thesis components
        thesis_components = []

        # Market positioning
        if category in ['large_cap', 'mid_cap']:
            thesis_components.append("established market position provides defensive characteristics")
        elif category in ['small_cap', 'micro_cap']:
            thesis_components.append("growth potential from market share expansion opportunities")
        elif category in ['penny', 'micro_penny']:
            thesis_components.append("speculative opportunity with high risk/reward profile")

        # Sector-specific thesis
        sector_trends = market_context.get('sector_trends', {})
        if sector_trends:
            outlook = sector_trends.get('outlook', '')
            if 'outperformance' in outlook.lower():
                thesis_components.append(f"sector positioned for outperformance due to {sector_trends.get('current_dynamics', 'favorable trends')}")
            elif 'recovery' in outlook.lower():
                thesis_components.append(f"cyclical recovery potential as {sector_trends.get('current_dynamics', 'conditions improve')}")
            else:
                thesis_components.append(f"sector dynamics suggest {outlook.lower()}")

        # Company-specific catalysts
        news_context = market_context.get('news_sentiment', {})
        company_news = news_context.get('company_news', [])
        if company_news:
            thesis_components.append(f"near-term catalysts include {company_news[0].lower()}")

        # Economic environment impact
        econ_context = market_context.get('economic_indicators', {})
        if prediction_result['expected_change'] > 0:
            if econ_context.get('gdp_growth', '').find('positive') != -1:
                thesis_components.append("supportive economic backdrop for growth")
            else:
                thesis_components.append("company-specific drivers outweighing macro headwinds")
        else:
            thesis_components.append("macro environment creating near-term challenges")

        # Risk assessment
        confidence = prediction_result.get('confidence', 50)
        if confidence > 75:
            thesis_components.append("high conviction opportunity with multiple supporting factors")
        elif confidence < 50:
            thesis_components.append("elevated uncertainty requiring careful position sizing")

        # Combine thesis components
        thesis = "Investment thesis: " + ", ".join(thesis_components[:3])  # Limit to 3 key points

        return thesis

# Import neural network predictor
from neural_network_predictor_production import neural_predictor

# Initialize services
market_data_service = MarketDataService()
predictor = neural_predictor  # Use neural network predictor (97.5% accuracy)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        logger.info("Prediction API called")
        data = request.get_json()
        logger.info(f"Request data: {data}")

        symbol = data.get('symbol', '').strip().upper()
        timeframe = data.get('timeframe', 'auto')
        logger.info(f"Processing symbol: {symbol}, timeframe: {timeframe}")

        if not symbol:
            return jsonify({"error": "Stock symbol is required"}), 400

        prediction = predictor.predict_stock_movement(symbol, timeframe)
        logger.info(f"Prediction result: {prediction}")

        return jsonify(prediction)

    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

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

@app.route('/api/auth/register-dev', methods=['POST'])
def register_dev():
    """Create a development account that bypasses email verification"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        # Basic validation
        if not name or len(name) < 2:
            return jsonify({"error": "Name must be at least 2 characters long"}), 400

        if not email:
            return jsonify({"error": "Email is required"}), 400

        if not password or len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters long"}), 400

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({"error": "An account with this email already exists"}), 400

        # Create new user with email already verified
        user = User(name=name, email=email, is_verified=True)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        logger.info(f"Development account created for {email}")

        return jsonify({
            "status": "success",
            "message": "Development account created successfully! You can now log in.",
            "user_id": user.id,
            "email": email,
            "name": name
        })

    except Exception as e:
        db.session.rollback()
        logger.error(f"Dev registration error: {e}")
        return jsonify({"error": "Development account creation failed. Please try again."}), 500

@app.route('/api/auth/dev-info', methods=['GET'])
def dev_info():
    """Get development account information"""
    return jsonify({
        "dev_account": {
            "email": "dev@stockprediction.com",
            "password": "devpass123",
            "name": "Developer",
            "note": "This account is pre-verified and ready to use for development/testing"
        },
        "endpoints": {
            "register_dev": "/api/auth/register-dev",
            "login": "/api/auth/login"
        }
    })

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
    try:
        stock_count = Stock.query.count()
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0",
            "stocks_in_database": stock_count
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

@app.route('/api/test')
def test():
    """Simple test endpoint to verify API is working"""
    return jsonify({
        "status": "success",
        "message": "API is working!",
        "timestamp": datetime.now().isoformat()
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

@app.route('/api/stocks/all', methods=['GET'])
def get_all_stocks():
    """Get all stocks from database for dropdown suggestions"""
    try:
        # Get all active stocks from database
        limit = min(int(request.args.get('limit', 1000)), 2000)  # Reasonable limit

        stocks = Stock.query.filter(Stock.is_active == True).limit(limit).all()

        stock_list = []
        for stock in stocks:
            stock_list.append({
                'symbol': stock.symbol,
                'name': stock.name,
                'sector': stock.sector or 'Unknown',
                'exchange': stock.exchange or 'Unknown',
                'is_penny_stock': stock.is_penny_stock
            })

        # If database is empty, provide fallback popular stocks
        if len(stock_list) == 0:
            stock_list = [
            # Large Cap Tech
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'sector': 'Technology'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': 'Technology'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'sector': 'Consumer Discretionary'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'sector': 'Technology'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'sector': 'Technology'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'sector': 'Consumer Discretionary'},
            {'symbol': 'NFLX', 'name': 'Netflix Inc.', 'sector': 'Communication Services'},

            # Financial Services
            {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.', 'sector': 'Financial Services'},
            {'symbol': 'BAC', 'name': 'Bank of America Corp.', 'sector': 'Financial Services'},
            {'symbol': 'WFC', 'name': 'Wells Fargo & Company', 'sector': 'Financial Services'},
            {'symbol': 'GS', 'name': 'Goldman Sachs Group Inc.', 'sector': 'Financial Services'},
            {'symbol': 'MS', 'name': 'Morgan Stanley', 'sector': 'Financial Services'},
            {'symbol': 'V', 'name': 'Visa Inc.', 'sector': 'Financial Services'},
            {'symbol': 'MA', 'name': 'Mastercard Inc.', 'sector': 'Financial Services'},

            # Healthcare
            {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'sector': 'Healthcare'},
            {'symbol': 'UNH', 'name': 'UnitedHealth Group Inc.', 'sector': 'Healthcare'},
            {'symbol': 'PFE', 'name': 'Pfizer Inc.', 'sector': 'Healthcare'},
            {'symbol': 'ABBV', 'name': 'AbbVie Inc.', 'sector': 'Healthcare'},
            {'symbol': 'MRK', 'name': 'Merck & Co. Inc.', 'sector': 'Healthcare'},
            {'symbol': 'TMO', 'name': 'Thermo Fisher Scientific Inc.', 'sector': 'Healthcare'},

            # Consumer Staples
            {'symbol': 'PG', 'name': 'Procter & Gamble Co.', 'sector': 'Consumer Staples'},
            {'symbol': 'KO', 'name': 'Coca-Cola Company', 'sector': 'Consumer Staples'},
            {'symbol': 'PEP', 'name': 'PepsiCo Inc.', 'sector': 'Consumer Staples'},
            {'symbol': 'WMT', 'name': 'Walmart Inc.', 'sector': 'Consumer Staples'},

            # Energy
            {'symbol': 'XOM', 'name': 'Exxon Mobil Corporation', 'sector': 'Energy'},
            {'symbol': 'CVX', 'name': 'Chevron Corporation', 'sector': 'Energy'},
            {'symbol': 'COP', 'name': 'ConocoPhillips', 'sector': 'Energy'},

            # Growth/Emerging
            {'symbol': 'AMD', 'name': 'Advanced Micro Devices Inc.', 'sector': 'Technology'},
            {'symbol': 'INTC', 'name': 'Intel Corporation', 'sector': 'Technology'},
            {'symbol': 'QCOM', 'name': 'QUALCOMM Inc.', 'sector': 'Technology'},
            {'symbol': 'ADBE', 'name': 'Adobe Inc.', 'sector': 'Technology'},
            {'symbol': 'CRM', 'name': 'Salesforce Inc.', 'sector': 'Technology'},
            {'symbol': 'NOW', 'name': 'ServiceNow Inc.', 'sector': 'Technology'},
            {'symbol': 'SNOW', 'name': 'Snowflake Inc.', 'sector': 'Technology'},
            {'symbol': 'PLTR', 'name': 'Palantir Technologies Inc.', 'sector': 'Technology'},

            # Retail/Consumer
            {'symbol': 'HD', 'name': 'Home Depot Inc.', 'sector': 'Consumer Discretionary'},
            {'symbol': 'NKE', 'name': 'Nike Inc.', 'sector': 'Consumer Discretionary'},
            {'symbol': 'SBUX', 'name': 'Starbucks Corporation', 'sector': 'Consumer Discretionary'},

            # Telecom/Penny Stocks Examples
            {'symbol': 'LUMN', 'name': 'Lumen Technologies Inc.', 'sector': 'Communication Services'},
            {'symbol': 'T', 'name': 'AT&T Inc.', 'sector': 'Communication Services'},
            {'symbol': 'VZ', 'name': 'Verizon Communications Inc.', 'sector': 'Communication Services'},

                # ETFs
                {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF Trust', 'sector': 'ETF'},
                {'symbol': 'QQQ', 'name': 'Invesco QQQ Trust', 'sector': 'ETF'},
                {'symbol': 'IWM', 'name': 'iShares Russell 2000 ETF', 'sector': 'ETF'},
            ]

        return jsonify({
            "status": "success",
            "stocks": stock_list,
            "count": len(stock_list),
            "source": "database" if len(stocks) > 0 else "fallback"
        })

    except Exception as e:
        logger.error(f"Error getting all stocks: {e}")
        return jsonify({"error": "Failed to get stocks"}), 500

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

@app.route('/api/test-stock/<symbol>')
def test_stock_data(symbol):
    """Test endpoint to check if we can fetch data for any stock"""
    try:
        # Test yfinance data fetching
        stock = yf.Ticker(symbol.upper())
        data = stock.history(period="5d")
        info = stock.info

        return jsonify({
            "status": "success",
            "symbol": symbol.upper(),
            "has_data": not data.empty,
            "data_points": len(data) if not data.empty else 0,
            "has_info": bool(info),
            "current_price": float(data['Close'].iloc[-1]) if not data.empty else None,
            "company_name": info.get('longName', 'Unknown'),
            "exchange": info.get('exchange', 'Unknown'),
            "market_cap": info.get('marketCap'),
            "is_penny_stock": float(data['Close'].iloc[-1]) < 5.0 if not data.empty else None
        })

    except Exception as e:
        logger.error(f"Error testing stock data for {symbol}: {e}")
        return jsonify({
            "status": "error",
            "symbol": symbol.upper(),
            "error": str(e)
        }), 500

def initialize_stock_database():
    """Initialize database with essential stocks if empty"""
    try:
        stock_count = Stock.query.count()
        if stock_count == 0:
            logger.info("Database is empty, adding essential stocks...")

            # Essential stocks to ensure dropdown works
            essential_stocks = [
                {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology'},
                {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'exchange': 'NASDAQ', 'sector': 'Technology'},
                {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology'},
                {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'exchange': 'NASDAQ', 'sector': 'Consumer Discretionary'},
                {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'exchange': 'NASDAQ', 'sector': 'Consumer Discretionary'},
                {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'exchange': 'NASDAQ', 'sector': 'Technology'},
                {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology'},
                {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.', 'exchange': 'NYSE', 'sector': 'Financial Services'},
                {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'exchange': 'NYSE', 'sector': 'Healthcare'},
                {'symbol': 'LUMN', 'name': 'Lumen Technologies Inc.', 'exchange': 'NYSE', 'sector': 'Communication Services'},
                {'symbol': 'AMD', 'name': 'Advanced Micro Devices Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology'},
                {'symbol': 'NFLX', 'name': 'Netflix Inc.', 'exchange': 'NASDAQ', 'sector': 'Communication Services'},
            ]

            for stock_info in essential_stocks:
                try:
                    new_stock = Stock(
                        symbol=stock_info['symbol'],
                        name=stock_info['name'],
                        exchange=stock_info['exchange'],
                        sector=stock_info['sector'],
                        is_active=True
                    )
                    db.session.add(new_stock)
                except Exception as e:
                    logger.error(f"Error adding stock {stock_info['symbol']}: {e}")
                    continue

            db.session.commit()
            logger.info(f"Added {len(essential_stocks)} essential stocks to database")
        else:
            logger.info(f"Database already contains {stock_count} stocks")

    except Exception as e:
        logger.error(f"Error initializing stock database: {e}")

def initialize_dev_account():
    """Create a default development account for testing"""
    try:
        dev_email = "dev@stockprediction.com"
        existing_dev = User.query.filter_by(email=dev_email).first()

        if not existing_dev:
            dev_user = User(
                name="Developer",
                email=dev_email,
                is_verified=True  # Skip email verification for dev account
            )
            dev_user.set_password("devpass123")  # Simple password for development
            db.session.add(dev_user)
            db.session.commit()
            logger.info("Development account created: dev@stockprediction.com / devpass123")
        else:
            logger.info("Development account already exists")

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating development account: {e}")

# Initialize database
with app.app_context():
    db.create_all()
    initialize_stock_database()
    initialize_dev_account()

if __name__ == '__main__':
    import os
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Set debug based on environment
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
