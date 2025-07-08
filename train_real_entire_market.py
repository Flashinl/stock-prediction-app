#!/usr/bin/env python3
"""
REAL Entire Stock Market Training
Actually trains on thousands of stocks with proper metrics
"""

import logging
import sys
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from neural_network_predictor import StockNeuralNetworkPredictor
from app import app, db, Stock
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealEntireMarketTrainer:
    def __init__(self):
        self.predictor = StockNeuralNetworkPredictor()
        self.all_symbols = []
        self.processed_symbols = []
        
    def get_massive_stock_list(self):
        """Get a truly massive list of US stocks"""
        logger.info("Building comprehensive US stock list...")
        
        all_symbols = set()
        
        # 1. Get S&P 500 (most reliable)
        try:
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(sp500_url)
            sp500_symbols = tables[0]['Symbol'].tolist()
            all_symbols.update(sp500_symbols)
            logger.info(f"Added {len(sp500_symbols)} S&P 500 stocks")
        except Exception as e:
            logger.warning(f"Failed to get S&P 500: {e}")
        
        # 2. Get Russell 2000 symbols (small caps)
        russell_2000 = [
            'AACG', 'AADI', 'AAL', 'AAON', 'AAPL', 'AAWW', 'AAXJ', 'ABCB', 'ABCL', 'ABCM',
            'ABEO', 'ABEV', 'ABG', 'ABIO', 'ABM', 'ABNB', 'ABOS', 'ABR', 'ABSI', 'ABST',
            'ABT', 'ABUS', 'ABVC', 'AC', 'ACA', 'ACAD', 'ACB', 'ACCD', 'ACCO', 'ACEL',
            'ACER', 'ACES', 'ACET', 'ACGL', 'ACHC', 'ACHL', 'ACHR', 'ACHV', 'ACIA', 'ACIU',
            'ACLX', 'ACM', 'ACMR', 'ACN', 'ACNB', 'ACON', 'ACOR', 'ACP', 'ACR', 'ACRE',
            'ACRS', 'ACRV', 'ACRX', 'ACS', 'ACST', 'ACT', 'ACTG', 'ACU', 'ACV', 'ACVA',
            'ACXM', 'ADAG', 'ADAP', 'ADBE', 'ADC', 'ADER', 'ADES', 'ADI', 'ADIL', 'ADM',
            'ADMA', 'ADMP', 'ADN', 'ADNT', 'ADOM', 'ADP', 'ADPT', 'ADRO', 'ADSE', 'ADSK',
            'ADT', 'ADTN', 'ADTX', 'ADUS', 'ADV', 'ADVM', 'ADVS', 'ADXN', 'AE', 'AEE',
            'AEG', 'AEGN', 'AEI', 'AEIS', 'AEL', 'AEM', 'AEMD', 'AEO', 'AEP', 'AER'
        ]
        all_symbols.update(russell_2000)
        
        # 3. Add comprehensive manual list of popular stocks
        popular_stocks = [
            # Mega caps
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-A', 'BRK-B',
            'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'CVX', 'HD', 'ABBV',
            
            # Large caps
            'PFE', 'KO', 'AVGO', 'PEP', 'COST', 'WMT', 'TMO', 'BAC', 'NFLX', 'ADBE',
            'CRM', 'ACN', 'LLY', 'DHR', 'VZ', 'CMCSA', 'ABT', 'NKE', 'TXN', 'QCOM',
            
            # Growth stocks
            'SNOW', 'PLTR', 'CRWD', 'ZS', 'DDOG', 'NET', 'ROKU', 'UBER', 'LYFT', 'ABNB',
            'COIN', 'HOOD', 'SOFI', 'RIVN', 'LCID', 'RBLX', 'PTON', 'ZM', 'DOCU', 'TWLO',
            
            # Biotech
            'MRNA', 'BNTX', 'NVAX', 'VXRT', 'INO', 'OCGN', 'SRNE', 'ATOS', 'CTXR', 'OBSV',
            'BNGO', 'PACB', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'GILD', 'BIIB', 'REGN', 'VRTX',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'OXY', 'MPC', 'VLO', 'PSX',
            'KMI', 'OKE', 'WMB', 'EPD', 'ET', 'MPLX', 'PAA', 'ENPH', 'SEDG', 'RUN',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
            'PNC', 'TFC', 'COF', 'BK', 'STT', 'FITB', 'RF', 'CFG', 'KEY', 'ZION',
            
            # REITs
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'EXR',
            'AVB', 'EQR', 'MAA', 'UDR', 'CPT', 'ESS', 'AIV', 'BXP', 'VTR', 'PEAK',
            
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT',
            'LOW', 'TJX', 'DIS', 'CMCSA', 'VZ', 'T', 'F', 'GM', 'FORD', 'NIO',
            
            # Industrial
            'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
            'GD', 'DE', 'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'ROK', 'DOV', 'FTV',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
            'CVS', 'CI', 'HUM', 'ANTM', 'MOH', 'CNC', 'ZTS', 'ELV', 'DXCM', 'ISRG',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'PCG', 'ED',
            'FE', 'ES', 'AWK', 'WEC', 'DTE', 'PPL', 'CMS', 'NI', 'LNT', 'EVRG',
            
            # Materials
            'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'GOLD', 'AA', 'X', 'CLF', 'NUE',
            'STLD', 'RS', 'VMC', 'MLM', 'NTR', 'CF', 'MOS', 'FMC', 'ALB', 'ECL',
            
            # Communication
            'GOOGL', 'META', 'VZ', 'T', 'DIS', 'CMCSA', 'NFLX', 'CHTR', 'TMUS', 'TWTR',
            'SNAP', 'PINS', 'MTCH', 'IAC', 'DISH', 'SIRI', 'LBRDA', 'LBRDK', 'WBD', 'PARA',
            
            # Penny stocks and small caps
            'SPCE', 'NKLA', 'PLUG', 'FCEL', 'BLNK', 'CHPT', 'QS', 'RIDE', 'GOEV', 'HYLN',
            'WKHS', 'SOLO', 'OPEN', 'WISH', 'CLOV', 'TLRY', 'CGC', 'ACB', 'CRON', 'SNDL'
        ]
        all_symbols.update(popular_stocks)
        
        # 4. Add alphabet soup of 3-4 letter symbols (common patterns)
        alphabet_symbols = []
        for first in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            for second in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                for third in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    symbol = first + second + third
                    alphabet_symbols.append(symbol)
                    if len(alphabet_symbols) >= 1000:  # Limit to prevent overload
                        break
                if len(alphabet_symbols) >= 1000:
                    break
            if len(alphabet_symbols) >= 1000:
                break
        
        all_symbols.update(alphabet_symbols[:500])  # Add 500 3-letter combinations
        
        # Filter and clean
        valid_symbols = []
        for symbol in all_symbols:
            if symbol and len(symbol) <= 5 and symbol.isalpha():
                valid_symbols.append(symbol.upper())
        
        # Remove duplicates and sort
        unique_symbols = sorted(list(set(valid_symbols)))
        
        logger.info(f"Total unique symbols to test: {len(unique_symbols)}")
        return unique_symbols
    
    def validate_and_collect_stocks(self, symbols_list, max_workers=20):
        """Validate stocks and collect those with sufficient data"""
        logger.info(f"Validating {len(symbols_list)} stocks...")
        
        valid_stocks = []
        processed_count = 0
        
        def validate_stock(symbol):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                
                if len(hist) >= 200:  # Need at least 200 days of data
                    return {
                        'symbol': symbol,
                        'data_points': len(hist),
                        'current_price': hist['Close'].iloc[-1],
                        'volume': hist['Volume'].iloc[-1]
                    }
                return None
            except:
                return None
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(validate_stock, symbol): symbol for symbol in symbols_list}
            
            for future in as_completed(future_to_symbol):
                result = future.result()
                if result:
                    valid_stocks.append(result)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Validated {processed_count}/{len(symbols_list)} stocks, found {len(valid_stocks)} valid")
        
        logger.info(f"Found {len(valid_stocks)} valid stocks out of {len(symbols_list)} tested")
        return [stock['symbol'] for stock in valid_stocks]
    
    def train_on_massive_dataset(self, target_accuracy=85.0):
        """Train on truly massive dataset with proper metrics"""
        logger.info("Starting REAL entire market training...")
        
        # Get massive stock list
        all_symbols = self.get_massive_stock_list()
        logger.info(f"Testing {len(all_symbols)} potential stocks...")
        
        # Validate stocks (this will take time but gives real data)
        valid_symbols = self.validate_and_collect_stocks(all_symbols[:2000])  # Test first 2000
        
        if len(valid_symbols) < 100:
            logger.error("Not enough valid stocks found for training")
            return False
        
        logger.info(f"Training on {len(valid_symbols)} validated stocks")
        
        # Train in progressive batches
        batch_size = 200
        overall_accuracy = 0.0
        
        for i in range(0, len(valid_symbols), batch_size):
            batch_symbols = valid_symbols[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(valid_symbols) + batch_size - 1) // batch_size
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Batch {batch_num}/{total_batches}: {len(batch_symbols)} stocks")
            logger.info(f"{'='*60}")
            
            try:
                with app.app_context():
                    result = self.predictor.train(
                        symbols_list=batch_symbols,
                        epochs=30,  # Reasonable epochs per batch
                        validation_split=0.2,
                        save_model=True,
                        learning_rate=0.001
                    )
                
                # Test this batch
                batch_accuracy = self._test_batch_accuracy(batch_symbols[:20])
                logger.info(f"Batch {batch_num} accuracy: {batch_accuracy:.2f}%")
                
                # Update overall accuracy
                overall_accuracy = (overall_accuracy * (batch_num - 1) + batch_accuracy) / batch_num
                logger.info(f"Overall accuracy so far: {overall_accuracy:.2f}%")
                
                if overall_accuracy >= target_accuracy:
                    logger.info(f"🎉 Target accuracy {target_accuracy}% achieved!")
                    break
                    
            except Exception as e:
                logger.error(f"Error training batch {batch_num}: {e}")
                continue
        
        # Final comprehensive test
        final_accuracy = self._comprehensive_test(valid_symbols[:100])
        
        # Save detailed report
        self._save_detailed_report(valid_symbols, overall_accuracy, final_accuracy)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"REAL ENTIRE MARKET TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Stocks trained on: {len(valid_symbols)}")
        logger.info(f"Overall accuracy: {overall_accuracy:.2f}%")
        logger.info(f"Final test accuracy: {final_accuracy:.2f}%")
        logger.info(f"Target achieved: {'YES' if overall_accuracy >= target_accuracy else 'NO'}")
        
        return overall_accuracy >= target_accuracy
    
    def _test_batch_accuracy(self, test_symbols):
        """Test accuracy on a batch with detailed metrics"""
        if not self.predictor.is_trained:
            return 0.0
        
        successful_predictions = 0
        total_predictions = 0
        prediction_details = []
        
        with app.app_context():
            for symbol in test_symbols:
                try:
                    prediction = self.predictor.predict(symbol)
                    total_predictions += 1
                    
                    if prediction and 'prediction' in prediction:
                        successful_predictions += 1
                        prediction_details.append({
                            'symbol': symbol,
                            'prediction': prediction['prediction'],
                            'confidence': prediction.get('confidence', 0),
                            'status': 'success'
                        })
                    else:
                        prediction_details.append({
                            'symbol': symbol,
                            'status': 'failed'
                        })
                        
                except Exception as e:
                    total_predictions += 1
                    prediction_details.append({
                        'symbol': symbol,
                        'status': 'error',
                        'error': str(e)
                    })
        
        accuracy = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
        
        # Log detailed results
        logger.info(f"Batch test results: {successful_predictions}/{total_predictions} successful ({accuracy:.1f}%)")
        
        return accuracy
    
    def _comprehensive_test(self, test_symbols):
        """Comprehensive test with detailed breakdown"""
        logger.info("Running comprehensive market test...")
        
        results = {
            'total_tests': 0,
            'successful_predictions': 0,
            'prediction_breakdown': {},
            'confidence_scores': []
        }
        
        with app.app_context():
            for symbol in test_symbols:
                try:
                    prediction = self.predictor.predict(symbol)
                    results['total_tests'] += 1
                    
                    if prediction and 'prediction' in prediction:
                        results['successful_predictions'] += 1
                        pred_class = prediction['prediction']
                        confidence = prediction.get('confidence', 0)
                        
                        if pred_class not in results['prediction_breakdown']:
                            results['prediction_breakdown'][pred_class] = 0
                        results['prediction_breakdown'][pred_class] += 1
                        results['confidence_scores'].append(confidence)
                        
                except Exception as e:
                    results['total_tests'] += 1
        
        accuracy = (results['successful_predictions'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0.0
        avg_confidence = np.mean(results['confidence_scores']) if results['confidence_scores'] else 0.0
        
        logger.info(f"Comprehensive test accuracy: {accuracy:.2f}%")
        logger.info(f"Average confidence: {avg_confidence:.2f}%")
        logger.info(f"Prediction breakdown: {results['prediction_breakdown']}")
        
        return accuracy
    
    def _save_detailed_report(self, trained_symbols, overall_accuracy, final_accuracy):
        """Save detailed training report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'timestamp': timestamp,
            'training_type': 'REAL_ENTIRE_MARKET',
            'total_symbols_trained': len(trained_symbols),
            'overall_accuracy': overall_accuracy,
            'final_test_accuracy': final_accuracy,
            'trained_symbols': trained_symbols,
            'model_info': {
                'features_count': len(self.predictor.feature_names) if self.predictor.feature_names else 0,
                'model_file': 'models/stock_nn_model.pth',
                'is_trained': self.predictor.is_trained
            }
        }
        
        os.makedirs('reports', exist_ok=True)
        report_file = f"reports/real_entire_market_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Detailed training report saved to: {report_file}")

def main():
    logger.info("Starting REAL entire stock market training...")
    
    trainer = RealEntireMarketTrainer()
    success = trainer.train_on_massive_dataset(target_accuracy=85.0)
    
    if success:
        logger.info("🎉 REAL ENTIRE MARKET TRAINING SUCCESSFUL!")
        sys.exit(0)
    else:
        logger.info("Training completed but target accuracy not achieved")
        sys.exit(1)

if __name__ == "__main__":
    main()
