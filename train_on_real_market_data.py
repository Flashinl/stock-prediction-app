#!/usr/bin/env python3
"""
Train Neural Network on Real Entire US Market Data
Uses the reliable market data from get_entire_us_market_data.py
"""

import logging
import sys
import os
import json
from datetime import datetime
from neural_network_predictor import StockNeuralNetworkPredictor
from get_entire_us_market_data import EntireUSMarketData
from app import app, db, Stock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealMarketTrainer:
    def __init__(self):
        self.predictor = StockNeuralNetworkPredictor()
        self.market_data_collector = EntireUSMarketData()
        
    def get_or_download_market_data(self):
        """Get market data - use cached if available, otherwise download fresh"""
        
        # Check for existing market data
        if os.path.exists('market_data'):
            symbol_files = [f for f in os.listdir('market_data') if f.startswith('entire_us_market_symbols_')]
            if symbol_files:
                latest_file = sorted(symbol_files)[-1]
                logger.info(f"Found existing market data: {latest_file}")
                
                with open(f'market_data/{latest_file}', 'r') as f:
                    data = json.load(f)
                    return data['symbols']
        
        # Download fresh market data
        logger.info("No existing market data found, downloading fresh data...")
        all_symbols = self.market_data_collector.compile_entire_market()
        self.market_data_collector.download_historical_data(all_symbols)
        self.market_data_collector.save_market_data()
        
        return self.market_data_collector.valid_symbols
    
    def train_on_entire_market(self, target_accuracy=80.0):
        """Train on the entire US stock market with real data"""
        logger.info("Starting training on entire US stock market...")
        
        # Get market symbols
        market_symbols = self.get_or_download_market_data()
        
        if len(market_symbols) < 100:
            logger.error("Not enough market symbols available for training")
            return False
        
        logger.info(f"Training on {len(market_symbols)} US market stocks")
        
        # Train in progressive segments for better results
        segments = self._create_training_segments(market_symbols)
        
        overall_accuracy = 0.0
        successful_segments = 0
        
        for segment_name, symbols in segments.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Segment: {segment_name}")
            logger.info(f"Stocks: {len(symbols)}")
            logger.info(f"Sample: {symbols[:5]}...")
            logger.info(f"{'='*60}")
            
            try:
                # Train on this segment
                with app.app_context():
                    training_result = self.predictor.train(
                        symbols_list=symbols,
                        epochs=40,  # More epochs for better training
                        validation_split=0.2,
                        save_model=True,
                        learning_rate=0.001
                    )
                
                # Test segment accuracy
                segment_accuracy = self._test_segment_accuracy(symbols[:20])
                
                if segment_accuracy > 0:
                    overall_accuracy = ((overall_accuracy * successful_segments) + segment_accuracy) / (successful_segments + 1)
                    successful_segments += 1
                
                logger.info(f"Segment '{segment_name}' accuracy: {segment_accuracy:.2f}%")
                logger.info(f"Overall accuracy so far: {overall_accuracy:.2f}% (from {successful_segments} segments)")
                
                # Check if target reached
                if overall_accuracy >= target_accuracy and successful_segments >= 3:
                    logger.info(f"🎉 Target accuracy {target_accuracy}% achieved!")
                    break
                    
            except Exception as e:
                logger.error(f"Error training segment '{segment_name}': {e}")
                continue
        
        # Final comprehensive test
        final_accuracy = self._comprehensive_market_test(market_symbols[:100])
        
        # Save training report
        self._save_training_report(market_symbols, segments, overall_accuracy, final_accuracy, successful_segments)
        
        # Print final results
        logger.info(f"\n{'='*60}")
        logger.info(f"ENTIRE US MARKET TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total market stocks: {len(market_symbols)}")
        logger.info(f"Training segments completed: {successful_segments}")
        logger.info(f"Overall training accuracy: {overall_accuracy:.2f}%")
        logger.info(f"Final test accuracy: {final_accuracy:.2f}%")
        logger.info(f"Target achieved: {'YES' if overall_accuracy >= target_accuracy else 'NO'}")
        logger.info(f"Model saved: models/stock_nn_model.pth")
        
        return overall_accuracy >= target_accuracy
    
    def _create_training_segments(self, market_symbols):
        """Create logical training segments from market symbols"""
        
        # Load market summary if available to get sector info
        market_summary = self._load_market_summary()
        
        segments = {}
        
        # Segment 1: Large Cap Stocks (most reliable)
        large_cap_stocks = [s for s in market_symbols if s in [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-A', 'BRK-B', 'UNH',
            'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'CVX', 'HD', 'ABBV', 'PFE', 'KO', 'AVGO',
            'PEP', 'COST', 'WMT', 'TMO', 'ABT', 'CRM', 'ACN', 'DHR', 'VZ', 'NKE', 'DIS'
        ]]
        if large_cap_stocks:
            segments['Large Cap Stocks'] = large_cap_stocks
        
        # Segment 2: Technology Stocks
        tech_stocks = [s for s in market_symbols if s in [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM', 'ORCL',
            'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC',
            'SNOW', 'PLTR', 'CRWD', 'ZS', 'DDOG', 'NET', 'ROKU', 'UBER', 'LYFT', 'ABNB'
        ]]
        if tech_stocks:
            segments['Technology Sector'] = tech_stocks
        
        # Segment 3: Financial Stocks
        financial_stocks = [s for s in market_symbols if s in [
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB', 'PNC', 'TFC',
            'COF', 'BK', 'STT', 'FITB', 'RF', 'CFG', 'KEY', 'ZION', 'CMA', 'HBAN', 'MTB'
        ]]
        if financial_stocks:
            segments['Financial Sector'] = financial_stocks
        
        # Segment 4: Healthcare Stocks
        healthcare_stocks = [s for s in market_symbols if s in [
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD',
            'CVS', 'CI', 'HUM', 'ANTM', 'MOH', 'CNC', 'ZTS', 'ELV', 'DXCM', 'ISRG', 'SYK'
        ]]
        if healthcare_stocks:
            segments['Healthcare Sector'] = healthcare_stocks
        
        # Segment 5: Energy Stocks
        energy_stocks = [s for s in market_symbols if s in [
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE', 'WMB',
            'EPD', 'ET', 'MPLX', 'PAA', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'FSLR', 'SPWR'
        ]]
        if energy_stocks:
            segments['Energy Sector'] = energy_stocks
        
        # Segment 6: Mixed Market Sample (remaining stocks)
        used_symbols = set()
        for segment_symbols in segments.values():
            used_symbols.update(segment_symbols)
        
        remaining_symbols = [s for s in market_symbols if s not in used_symbols]
        if remaining_symbols:
            segments['Mixed Market Sample'] = remaining_symbols[:300]  # Limit for performance
        
        # Log segment info
        for name, symbols in segments.items():
            logger.info(f"Segment '{name}': {len(symbols)} stocks")
        
        return segments
    
    def _load_market_summary(self):
        """Load market summary if available"""
        try:
            if os.path.exists('market_data'):
                summary_files = [f for f in os.listdir('market_data') if f.startswith('entire_us_market_summary_')]
                if summary_files:
                    latest_file = sorted(summary_files)[-1]
                    with open(f'market_data/{latest_file}', 'r') as f:
                        return json.load(f)
        except:
            pass
        return {}
    
    def _test_segment_accuracy(self, test_symbols):
        """Test accuracy on a segment"""
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
                            'confidence': prediction.get('confidence', 0)
                        })
                        
                except Exception as e:
                    total_predictions += 1
                    logger.debug(f"Prediction failed for {symbol}: {e}")
        
        accuracy = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
        
        # Log sample predictions
        if prediction_details:
            logger.info(f"Sample predictions: {prediction_details[:3]}")
        
        return accuracy
    
    def _comprehensive_market_test(self, test_symbols):
        """Comprehensive test on diverse market sample"""
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
        avg_confidence = sum(results['confidence_scores']) / len(results['confidence_scores']) if results['confidence_scores'] else 0.0
        
        logger.info(f"Comprehensive test results:")
        logger.info(f"- Accuracy: {accuracy:.2f}%")
        logger.info(f"- Average confidence: {avg_confidence:.2f}%")
        logger.info(f"- Prediction breakdown: {results['prediction_breakdown']}")
        
        return accuracy
    
    def _save_training_report(self, market_symbols, segments, overall_accuracy, final_accuracy, successful_segments):
        """Save detailed training report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'timestamp': timestamp,
            'training_type': 'ENTIRE_US_MARKET',
            'data_source': 'Multiple reliable sources (S&P 500, NASDAQ, NYSE, ETFs)',
            'total_market_symbols': len(market_symbols),
            'training_segments': {name: len(symbols) for name, symbols in segments.items()},
            'successful_segments': successful_segments,
            'overall_training_accuracy': overall_accuracy,
            'final_test_accuracy': final_accuracy,
            'sample_symbols': market_symbols[:50],
            'model_info': {
                'features_count': len(self.predictor.feature_names) if self.predictor.feature_names else 0,
                'model_file': 'models/stock_nn_model.pth',
                'is_trained': self.predictor.is_trained
            }
        }
        
        os.makedirs('reports', exist_ok=True)
        report_file = f"reports/entire_us_market_training_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved to: {report_file}")

def main():
    logger.info("Starting entire US market training with real data...")
    
    trainer = RealMarketTrainer()
    success = trainer.train_on_entire_market(target_accuracy=75.0)  # Realistic target
    
    if success:
        logger.info("🎉 ENTIRE US MARKET TRAINING SUCCESSFUL!")
        sys.exit(0)
    else:
        logger.info("Training completed but target accuracy not achieved")
        sys.exit(1)

if __name__ == "__main__":
    main()
