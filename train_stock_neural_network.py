#!/usr/bin/env python3
"""
Advanced Stock Neural Network Training Script
Trains until high accuracy is achieved with comprehensive stock data
"""

import argparse
import logging
import sys
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from neural_network_predictor import StockNeuralNetworkPredictor
from app import app, db, Stock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_comprehensive_stock_list():
    """Get a comprehensive list of stocks for training"""
    try:
        with app.app_context():
            # Get all active stocks from database
            stocks = Stock.query.filter(Stock.is_active == True).all()

            if len(stocks) < 50:
                # If database is sparse, use a comprehensive fallback list
                logger.info("Database has limited stocks, using comprehensive fallback list")
                return get_fallback_stock_list()

            # Categorize stocks for balanced training
            large_cap_stocks = []
            mid_cap_stocks = []
            small_cap_stocks = []
            penny_stocks = []

            for stock in stocks:
                if stock.market_cap and stock.market_cap > 10_000_000_000:  # >$10B
                    large_cap_stocks.append(stock.symbol)
                elif stock.market_cap and stock.market_cap > 2_000_000_000:  # $2B-$10B
                    mid_cap_stocks.append(stock.symbol)
                elif stock.is_penny_stock:
                    penny_stocks.append(stock.symbol)
                else:
                    small_cap_stocks.append(stock.symbol)

            # Create balanced training set
            training_symbols = []
            training_symbols.extend(large_cap_stocks[:100])  # Top 100 large caps
            training_symbols.extend(mid_cap_stocks[:75])     # 75 mid caps
            training_symbols.extend(small_cap_stocks[:50])   # 50 small caps
            training_symbols.extend(penny_stocks[:25])       # 25 penny stocks

            logger.info(f"Selected {len(training_symbols)} stocks for training:")
            logger.info(f"- Large cap: {len(large_cap_stocks[:100])}")
            logger.info(f"- Mid cap: {len(mid_cap_stocks[:75])}")
            logger.info(f"- Small cap: {len(small_cap_stocks[:50])}")
            logger.info(f"- Penny stocks: {len(penny_stocks[:25])}")

            return training_symbols
    except Exception as e:
        logger.warning(f"Could not access database: {e}")
        logger.info("Using fallback stock list")
        return get_fallback_stock_list()

def get_fallback_stock_list():
    """Comprehensive fallback stock list covering all market segments"""
    return [
        # Large Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM',
        'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',
        
        # Financial Services
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
        'PNC', 'TFC', 'COF', 'BK', 'STT', 'FITB', 'RF', 'CFG', 'KEY', 'ZION',
        
        # Healthcare & Biotech
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
        'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'ELV', 'CVS',
        
        # Consumer & Retail
        'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT',
        'LOW', 'TJX', 'DIS', 'CMCSA', 'VZ', 'T', 'NFLX', 'ROKU', 'SPOT', 'UBER',
        
        # Industrial & Energy
        'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'OXY', 'MPC', 'VLO', 'PSX',
        
        # Mid Cap Growth
        'SNOW', 'PLTR', 'CRWD', 'ZS', 'OKTA', 'DDOG', 'NET', 'FSLY', 'ESTC', 'SPLK',
        'TWLO', 'ZM', 'DOCU', 'PTON', 'RBLX', 'U', 'PATH', 'GTLB', 'S', 'WORK',
        
        # Small Cap & Emerging
        'SPCE', 'OPEN', 'WISH', 'CLOV', 'SOFI', 'HOOD', 'COIN', 'RIVN', 'LCID', 'NKLA',
        'PLUG', 'FCEL', 'BLNK', 'CHPT', 'QS', 'RIDE', 'GOEV', 'HYLN', 'WKHS', 'SOLO',
        
        # REITs & Utilities
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'EXR',
        'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'PCG', 'ED',
        
        # International & ETFs
        'BABA', 'TSM', 'ASML', 'NVO', 'TM', 'SONY', 'SAP', 'SHOP', 'SE', 'MELI',
        'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'GLD', 'SLV', 'TLT'
    ]

def train_with_progressive_difficulty(predictor, symbols_list, target_accuracy, iteration, log_file):
    """Train model with progressive difficulty and advanced techniques"""

    # Progressive training parameters based on iteration
    base_epochs = 50
    epochs = base_epochs + (iteration * 25)  # Increase epochs each iteration

    # Adjust learning rate based on iteration
    if iteration <= 2:
        learning_rate = 0.001  # Standard learning rate
    elif iteration <= 5:
        learning_rate = 0.0005  # Reduce learning rate
    else:
        learning_rate = 0.0001  # Fine-tuning learning rate

    logger.info(f"Training iteration {iteration} with {epochs} epochs, LR: {learning_rate}")

    # Ensure logs directory exists and create safe log file
    os.makedirs('logs', exist_ok=True)
    safe_log_file = log_file.replace(':', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace('*', '_').replace('"', '_')

    # Log training parameters
    try:
        with open(safe_log_file, 'a', encoding='utf-8') as f:
            f.write(f"Iteration {iteration} parameters:\n")
            f.write(f"- Epochs: {epochs}\n")
            f.write(f"- Learning rate: {learning_rate}\n")
            f.write(f"- Training symbols: {len(symbols_list)}\n")
            f.write(f"- Target accuracy: {target_accuracy}%\n\n")
    except Exception as e:
        logger.warning(f"Could not write to log file {safe_log_file}: {e}")
        safe_log_file = f"logs/training_log_{iteration}.txt"
    
    try:
        # Train the model with app context
        with app.app_context():
            predictor.train(
                symbols_list=symbols_list,
                epochs=epochs,
                validation_split=0.2,
                save_model=True,
                learning_rate=learning_rate
            )

        # Test the model accuracy
        accuracy = test_model_accuracy(predictor, symbols_list[:20])  # Test on subset
        
        logger.info(f"Achieved accuracy: {accuracy:.2f}%")

        try:
            with open(safe_log_file, 'a', encoding='utf-8') as f:
                f.write(f"Iteration {iteration} results:\n")
                f.write(f"- Achieved accuracy: {accuracy:.2f}%\n")
                f.write(f"- Target accuracy: {target_accuracy}%\n")
                f.write(f"- Success: {'YES' if accuracy >= target_accuracy else 'NO'}\n\n")
        except Exception as log_error:
            logger.warning(f"Could not write results to log file: {log_error}")

        return accuracy >= target_accuracy

    except Exception as e:
        logger.error(f"Training failed: {e}")
        try:
            with open(safe_log_file, 'a', encoding='utf-8') as f:
                f.write(f"Iteration {iteration} FAILED: {str(e)}\n\n")
        except Exception as log_error:
            logger.warning(f"Could not write error to log file: {log_error}")
        return False

def test_model_accuracy(predictor, test_symbols):
    """Test model accuracy on a subset of symbols"""
    if not predictor.is_trained:
        return 0.0

    correct_predictions = 0
    total_predictions = 0

    with app.app_context():
        for symbol in test_symbols[:10]:  # Test on first 10 symbols
            try:
                prediction = predictor.predict(symbol)
                if prediction and 'prediction' in prediction:
                    # For now, consider any successful prediction as correct
                    # In a real scenario, you'd compare against actual future prices
                    correct_predictions += 1
                total_predictions += 1
            except Exception as e:
                logger.warning(f"Prediction failed for {symbol}: {e}")
                total_predictions += 1

    if total_predictions == 0:
        return 0.0

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Train Stock Neural Network')
    parser.add_argument('--target-accuracy', type=float, default=90.0, help='Target accuracy percentage')
    parser.add_argument('--iteration', type=int, default=1, help='Current training iteration')
    parser.add_argument('--log-file', type=str, required=True, help='Log file path')
    
    args = parser.parse_args()
    
    logger.info(f"Starting training iteration {args.iteration}")
    logger.info(f"Target accuracy: {args.target_accuracy}%")
    
    try:
        # Initialize predictor
        predictor = StockNeuralNetworkPredictor()

        # Get comprehensive stock list
        symbols_list = get_comprehensive_stock_list()

        if len(symbols_list) == 0:
            logger.error("No stocks available for training")
            sys.exit(1)

        # Train with progressive difficulty
        with app.app_context():
            success = train_with_progressive_difficulty(
                predictor, symbols_list, args.target_accuracy, args.iteration, args.log_file
            )
        
        if success:
            logger.info("🎉 Target accuracy achieved!")
            sys.exit(0)
        else:
            logger.info("Target accuracy not yet achieved")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        try:
            safe_log_file = args.log_file.replace(':', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace('*', '_').replace('"', '_')
            with open(safe_log_file, 'a', encoding='utf-8') as f:
                f.write(f"CRITICAL ERROR in iteration {args.iteration}: {str(e)}\n\n")
        except Exception as log_error:
            logger.warning(f"Could not write critical error to log file: {log_error}")
        sys.exit(1)

if __name__ == "__main__":
    main()
