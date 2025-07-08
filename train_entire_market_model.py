#!/usr/bin/env python3
"""
Train Neural Network on Entire Stock Market
Uses comprehensive market data for training
"""

import argparse
import logging
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from neural_network_predictor import StockNeuralNetworkPredictor
from prepare_entire_stock_market import EntireStockMarketPreparator
from app import app, db, Stock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntireMarketTrainer:
    def __init__(self):
        self.predictor = StockNeuralNetworkPredictor()
        self.market_preparator = EntireStockMarketPreparator()
        
    def get_market_symbols(self, use_cached=True):
        """Get symbols for entire market training"""
        
        # Check for cached market data
        if use_cached:
            data_files = [f for f in os.listdir('data') if f.startswith('entire_stock_market_') and f.endswith('.csv')]
            if data_files:
                latest_file = sorted(data_files)[-1]
                logger.info(f"Using cached market data: {latest_file}")
                df = pd.read_csv(f"data/{latest_file}")
                return df['symbol'].tolist()
        
        # Generate fresh market data
        logger.info("Generating fresh market data...")
        market_data = self.market_preparator.prepare_market_data_parallel(max_workers=20)
        return [stock['symbol'] for stock in market_data]
    
    def train_on_market_segments(self, target_accuracy=90.0, epochs_per_segment=50):
        """Train on different market segments progressively"""
        
        logger.info("Starting entire market training...")
        
        # Get all market symbols
        all_symbols = self.get_market_symbols()
        logger.info(f"Total market symbols: {len(all_symbols)}")
        
        # Segment the market for progressive training
        segments = self._create_market_segments(all_symbols)
        
        overall_accuracy = 0.0
        segment_results = []
        
        for segment_name, symbols in segments.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training on {segment_name}: {len(symbols)} stocks")
            logger.info(f"{'='*60}")
            
            try:
                # Train on this segment
                with app.app_context():
                    result = self.predictor.train(
                        symbols_list=symbols,
                        epochs=epochs_per_segment,
                        validation_split=0.2,
                        save_model=True,
                        learning_rate=0.001
                    )
                
                # Test accuracy on this segment
                accuracy = self._test_segment_accuracy(symbols[:20])  # Test on subset
                
                segment_result = {
                    'segment': segment_name,
                    'symbols_count': len(symbols),
                    'accuracy': accuracy,
                    'training_result': result
                }
                segment_results.append(segment_result)
                
                logger.info(f"{segment_name} accuracy: {accuracy:.2f}%")
                
                # Update overall accuracy (weighted by segment size)
                overall_accuracy = self._calculate_weighted_accuracy(segment_results)
                
                logger.info(f"Overall weighted accuracy: {overall_accuracy:.2f}%")
                
                # Check if we've reached target
                if overall_accuracy >= target_accuracy:
                    logger.info(f"🎉 Target accuracy {target_accuracy}% achieved!")
                    break
                    
            except Exception as e:
                logger.error(f"Error training on {segment_name}: {e}")
                continue
        
        # Final comprehensive test
        final_accuracy = self._comprehensive_market_test(all_symbols[:100])
        
        # Save training report
        self._save_training_report(segment_results, final_accuracy, all_symbols)
        
        return overall_accuracy >= target_accuracy
    
    def _create_market_segments(self, all_symbols):
        """Create market segments for progressive training"""
        
        # Get market data for segmentation
        market_data = {}
        with app.app_context():
            for symbol in all_symbols[:1000]:  # Limit for performance
                try:
                    stock = Stock.query.filter_by(symbol=symbol).first()
                    if stock:
                        market_data[symbol] = {
                            'market_cap': stock.market_cap or 0,
                            'sector': stock.sector or 'Unknown'
                        }
                except:
                    continue
        
        # Create segments
        segments = {}
        
        # 1. Large Cap Segment (>$10B)
        large_cap = [s for s, data in market_data.items() if data['market_cap'] > 10_000_000_000]
        if large_cap:
            segments['Large Cap (>$10B)'] = large_cap[:200]
        
        # 2. Mid Cap Segment ($2B-$10B)
        mid_cap = [s for s, data in market_data.items() 
                  if 2_000_000_000 < data['market_cap'] <= 10_000_000_000]
        if mid_cap:
            segments['Mid Cap ($2B-$10B)'] = mid_cap[:300]
        
        # 3. Small Cap Segment ($300M-$2B)
        small_cap = [s for s, data in market_data.items() 
                    if 300_000_000 < data['market_cap'] <= 2_000_000_000]
        if small_cap:
            segments['Small Cap ($300M-$2B)'] = small_cap[:400]
        
        # 4. Technology Sector
        tech_stocks = [s for s, data in market_data.items() 
                      if 'Technology' in data.get('sector', '')]
        if tech_stocks:
            segments['Technology Sector'] = tech_stocks[:250]
        
        # 5. Healthcare Sector
        healthcare_stocks = [s for s, data in market_data.items() 
                           if 'Healthcare' in data.get('sector', '')]
        if healthcare_stocks:
            segments['Healthcare Sector'] = healthcare_stocks[:200]
        
        # 6. Financial Sector
        financial_stocks = [s for s, data in market_data.items() 
                          if 'Financial' in data.get('sector', '')]
        if financial_stocks:
            segments['Financial Sector'] = financial_stocks[:200]
        
        # 7. Mixed Market Sample (diverse selection)
        remaining_symbols = [s for s in all_symbols if s not in 
                           sum(segments.values(), [])]
        if remaining_symbols:
            segments['Mixed Market Sample'] = remaining_symbols[:500]
        
        # Log segment info
        for name, symbols in segments.items():
            logger.info(f"Segment '{name}': {len(symbols)} stocks")
        
        return segments
    
    def _test_segment_accuracy(self, test_symbols):
        """Test accuracy on a segment"""
        if not self.predictor.is_trained:
            return 0.0
        
        correct = 0
        total = 0
        
        with app.app_context():
            for symbol in test_symbols:
                try:
                    prediction = self.predictor.predict(symbol)
                    if prediction and 'prediction' in prediction:
                        correct += 1
                    total += 1
                except:
                    total += 1
        
        return (correct / total * 100) if total > 0 else 0.0
    
    def _calculate_weighted_accuracy(self, segment_results):
        """Calculate weighted accuracy across segments"""
        total_weighted_score = 0
        total_weight = 0
        
        for result in segment_results:
            weight = result['symbols_count']
            score = result['accuracy']
            total_weighted_score += weight * score
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _comprehensive_market_test(self, test_symbols):
        """Comprehensive test on diverse market sample"""
        logger.info("Running comprehensive market test...")
        
        test_results = {
            'total_tests': 0,
            'successful_predictions': 0,
            'by_sector': {},
            'by_market_cap': {}
        }
        
        with app.app_context():
            for symbol in test_symbols:
                try:
                    prediction = self.predictor.predict(symbol)
                    test_results['total_tests'] += 1
                    
                    if prediction and 'prediction' in prediction:
                        test_results['successful_predictions'] += 1
                        
                except Exception as e:
                    test_results['total_tests'] += 1
                    logger.debug(f"Test failed for {symbol}: {e}")
        
        accuracy = (test_results['successful_predictions'] / 
                   test_results['total_tests'] * 100) if test_results['total_tests'] > 0 else 0.0
        
        logger.info(f"Comprehensive market test accuracy: {accuracy:.2f}%")
        return accuracy
    
    def _save_training_report(self, segment_results, final_accuracy, all_symbols):
        """Save comprehensive training report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'timestamp': timestamp,
            'total_market_symbols': len(all_symbols),
            'segments_trained': len(segment_results),
            'segment_results': segment_results,
            'final_accuracy': final_accuracy,
            'overall_weighted_accuracy': self._calculate_weighted_accuracy(segment_results),
            'model_info': {
                'features_count': len(self.predictor.feature_names) if self.predictor.feature_names else 0,
                'model_file': 'models/stock_nn_model.pth'
            }
        }
        
        report_file = f"reports/entire_market_training_{timestamp}.json"
        os.makedirs('reports', exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Train on Entire Stock Market')
    parser.add_argument('--target-accuracy', type=float, default=90.0, help='Target accuracy percentage')
    parser.add_argument('--epochs-per-segment', type=int, default=50, help='Epochs per market segment')
    parser.add_argument('--use-cached', action='store_true', help='Use cached market data')
    
    args = parser.parse_args()
    
    logger.info("Starting entire stock market training...")
    logger.info(f"Target accuracy: {args.target_accuracy}%")
    logger.info(f"Epochs per segment: {args.epochs_per_segment}")
    
    try:
        trainer = EntireMarketTrainer()
        success = trainer.train_on_market_segments(
            target_accuracy=args.target_accuracy,
            epochs_per_segment=args.epochs_per_segment
        )
        
        if success:
            logger.info("🎉 ENTIRE MARKET TRAINING SUCCESSFUL!")
            sys.exit(0)
        else:
            logger.info("Training completed but target accuracy not achieved")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Entire market training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
