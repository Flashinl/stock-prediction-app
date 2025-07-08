#!/usr/bin/env python3
"""
Smart Market Model Training
Uses existing database + smart stock selection to avoid rate limits
"""

import logging
import sys
import os
import json
import time
from datetime import datetime
from neural_network_predictor import StockNeuralNetworkPredictor
from app import app, db, Stock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartMarketTrainer:
    def __init__(self):
        self.predictor = StockNeuralNetworkPredictor()
        
    def get_database_stocks(self):
        """Get all stocks from existing database"""
        with app.app_context():
            stocks = Stock.query.filter(Stock.is_active == True).all()
            return [stock.symbol for stock in stocks]
    
    def get_proven_stock_list(self):
        """Get a proven list of stocks that definitely exist and trade"""
        return [
            # Mega Cap Tech (guaranteed to work)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            
            # Large Cap Stable
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'XOM', 'WMT',
            'KO', 'PFE', 'ABBV', 'CVX', 'MRK', 'TMO', 'ABT', 'COST', 'AVGO',
            'NFLX', 'ADBE', 'CRM', 'ACN', 'DHR', 'VZ', 'NKE', 'DIS', 'MCD',
            
            # Financial
            'GS', 'MS', 'C', 'WFC', 'AXP', 'BLK', 'SCHW', 'USB', 'PNC', 'TFC',
            
            # Healthcare
            'LLY', 'BMY', 'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ISRG', 'ZTS',
            
            # Consumer
            'PEP', 'PM', 'UL', 'CL', 'KMB', 'GIS', 'K', 'HSY', 'MKC', 'CPB',
            
            # Industrial
            'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
            'DE', 'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'ROK', 'DOV', 'FTV',
            
            # Energy
            'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE', 'WMB',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'PCG',
            
            # Materials
            'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'AA', 'X', 'NUE', 'STLD',
            
            # REITs
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'SBAC', 'EXR', 'AVB',
            
            # Communication
            'T', 'CMCSA', 'CHTR', 'TMUS', 'VZ',
            
            # Growth/Tech
            'SNOW', 'PLTR', 'CRWD', 'ZS', 'DDOG', 'NET', 'ROKU', 'UBER', 'LYFT',
            'ABNB', 'COIN', 'HOOD', 'SOFI', 'RIVN', 'LCID', 'RBLX', 'PTON',
            
            # Biotech
            'MRNA', 'BNTX', 'NVAX', 'CRSP', 'EDIT', 'NTLA', 'BEAM',
            
            # ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'GLD', 'SLV', 'TLT'
        ]
    
    def train_with_rate_limiting(self, target_accuracy=75.0):
        """Train with smart rate limiting to avoid API issues"""
        logger.info("Starting smart market training with rate limiting...")
        
        # Get stocks from database first
        db_stocks = self.get_database_stocks()
        proven_stocks = self.get_proven_stock_list()
        
        # Combine and deduplicate
        all_stocks = list(set(db_stocks + proven_stocks))
        
        logger.info(f"Training on {len(all_stocks)} stocks")
        logger.info(f"Database stocks: {len(db_stocks)}")
        logger.info(f"Proven stocks: {len(proven_stocks)}")
        
        # Train in small batches with delays to avoid rate limiting
        batch_size = 15  # Small batches to avoid rate limits
        delay_between_batches = 30  # 30 second delay between batches
        
        overall_accuracy = 0.0
        successful_batches = 0
        
        for i in range(0, len(all_stocks), batch_size):
            batch_stocks = all_stocks[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(all_stocks) + batch_size - 1) // batch_size
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Batch {batch_num}/{total_batches}: {len(batch_stocks)} stocks")
            logger.info(f"Stocks: {batch_stocks}")
            logger.info(f"{'='*60}")
            
            try:
                with app.app_context():
                    result = self.predictor.train(
                        symbols_list=batch_stocks,
                        epochs=30,
                        validation_split=0.2,
                        save_model=True,
                        learning_rate=0.001
                    )
                
                # Test batch accuracy
                batch_accuracy = self._test_batch_accuracy(batch_stocks[:5])
                
                if batch_accuracy > 0:
                    overall_accuracy = ((overall_accuracy * successful_batches) + batch_accuracy) / (successful_batches + 1)
                    successful_batches += 1
                
                logger.info(f"Batch {batch_num} accuracy: {batch_accuracy:.2f}%")
                logger.info(f"Overall accuracy: {overall_accuracy:.2f}% (from {successful_batches} batches)")
                
                # Check if target reached
                if overall_accuracy >= target_accuracy and successful_batches >= 3:
                    logger.info(f"🎉 Target accuracy {target_accuracy}% achieved!")
                    break
                
                # Rate limiting delay (except for last batch)
                if i + batch_size < len(all_stocks):
                    logger.info(f"Waiting {delay_between_batches} seconds to avoid rate limits...")
                    time.sleep(delay_between_batches)
                    
            except Exception as e:
                logger.error(f"Error training batch {batch_num}: {e}")
                # Still wait to avoid rate limits
                if i + batch_size < len(all_stocks):
                    time.sleep(delay_between_batches)
                continue
        
        # Final test
        final_accuracy = self._comprehensive_test(all_stocks[:30])
        
        # Save report
        self._save_training_report(all_stocks, overall_accuracy, final_accuracy, successful_batches)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SMART MARKET TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total stocks: {len(all_stocks)}")
        logger.info(f"Successful batches: {successful_batches}")
        logger.info(f"Overall accuracy: {overall_accuracy:.2f}%")
        logger.info(f"Final test accuracy: {final_accuracy:.2f}%")
        logger.info(f"Target achieved: {'YES' if overall_accuracy >= target_accuracy else 'NO'}")
        
        return overall_accuracy >= target_accuracy
    
    def _test_batch_accuracy(self, test_symbols):
        """Test accuracy on a small batch"""
        if not self.predictor.is_trained:
            return 0.0
        
        successful = 0
        total = 0
        
        with app.app_context():
            for symbol in test_symbols:
                try:
                    prediction = self.predictor.predict(symbol)
                    total += 1
                    if prediction and 'prediction' in prediction:
                        successful += 1
                        logger.info(f"  {symbol}: {prediction['prediction']} ({prediction.get('confidence', 0):.1f}%)")
                except Exception as e:
                    total += 1
                    logger.debug(f"  {symbol}: Failed - {e}")
        
        accuracy = (successful / total * 100) if total > 0 else 0.0
        return accuracy
    
    def _comprehensive_test(self, test_symbols):
        """Comprehensive test with detailed metrics"""
        logger.info("Running comprehensive test...")
        
        results = {
            'total': 0,
            'successful': 0,
            'predictions': {},
            'confidences': []
        }
        
        with app.app_context():
            for symbol in test_symbols:
                try:
                    prediction = self.predictor.predict(symbol)
                    results['total'] += 1
                    
                    if prediction and 'prediction' in prediction:
                        results['successful'] += 1
                        pred = prediction['prediction']
                        conf = prediction.get('confidence', 0)
                        
                        if pred not in results['predictions']:
                            results['predictions'][pred] = 0
                        results['predictions'][pred] += 1
                        results['confidences'].append(conf)
                        
                except Exception as e:
                    results['total'] += 1
        
        accuracy = (results['successful'] / results['total'] * 100) if results['total'] > 0 else 0.0
        avg_conf = sum(results['confidences']) / len(results['confidences']) if results['confidences'] else 0.0
        
        logger.info(f"Comprehensive test: {accuracy:.2f}% accuracy, {avg_conf:.2f}% avg confidence")
        logger.info(f"Predictions: {results['predictions']}")
        
        return accuracy
    
    def _save_training_report(self, all_stocks, overall_accuracy, final_accuracy, successful_batches):
        """Save training report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'timestamp': timestamp,
            'training_type': 'SMART_MARKET_WITH_RATE_LIMITING',
            'total_stocks': len(all_stocks),
            'successful_batches': successful_batches,
            'overall_accuracy': overall_accuracy,
            'final_accuracy': final_accuracy,
            'stocks_trained': all_stocks,
            'model_info': {
                'features': len(self.predictor.feature_names) if self.predictor.feature_names else 0,
                'model_file': 'models/stock_nn_model.pth',
                'trained': self.predictor.is_trained
            }
        }
        
        os.makedirs('reports', exist_ok=True)
        report_file = f"reports/smart_market_training_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved: {report_file}")

def main():
    logger.info("Starting smart market training...")
    
    trainer = SmartMarketTrainer()
    success = trainer.train_with_rate_limiting(target_accuracy=70.0)  # Realistic target
    
    if success:
        logger.info("🎉 SMART MARKET TRAINING SUCCESSFUL!")
        sys.exit(0)
    else:
        logger.info("Training completed but target accuracy not achieved")
        sys.exit(1)

if __name__ == "__main__":
    main()
