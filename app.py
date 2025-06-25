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

class StockPredictor:
    def __init__(self):
        self.model_confidence = 0.75  # Simulated model confidence
    
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
            
            # Simple prediction logic (replace with your LLM model)
            current_price = indicators['current_price']
            sma_20 = indicators.get('sma_20', current_price)
            rsi = indicators.get('rsi', 50)
            
            # Basic trend analysis
            price_trend = (current_price - sma_20) / sma_20 * 100 if sma_20 else 0
            
            # Prediction logic
            if rsi < 30 and price_trend < -5:  # Oversold and downtrend
                prediction = "BUY"
                expected_change = np.random.uniform(3, 8)
                confidence = 0.8
            elif rsi > 70 and price_trend > 5:  # Overbought and uptrend
                prediction = "SELL"
                expected_change = np.random.uniform(-8, -3)
                confidence = 0.75
            elif price_trend > 2:
                prediction = "HOLD/BUY"
                expected_change = np.random.uniform(1, 5)
                confidence = 0.65
            else:
                prediction = "HOLD"
                expected_change = np.random.uniform(-2, 2)
                confidence = 0.6
            
            # Calculate target price
            target_price = current_price * (1 + expected_change / 100)
            
            return {
                "symbol": symbol.upper(),
                "company_name": info.get('longName', symbol.upper()) if info else symbol.upper(),
                "current_price": round(current_price, 2),
                "prediction": prediction,
                "expected_change_percent": round(expected_change, 2),
                "target_price": round(target_price, 2),
                "confidence": round(confidence * 100, 1),
                "timeframe": "1-3 months",
                "technical_indicators": indicators,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error in prediction for {symbol}: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

# Initialize predictor
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
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "2.0"})

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
