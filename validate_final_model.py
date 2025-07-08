#!/usr/bin/env python3
"""
Final Model Validation Script
Performs comprehensive validation of the trained model
"""

import logging
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from neural_network_predictor import StockNeuralNetworkPredictor
from app import app, db, Stock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_model_files():
    """Validate that all required model files exist"""
    required_files = [
        'models/stock_nn_model.pth',
        'models/stock_scaler.joblib',
        'models/stock_label_encoder.joblib',
        'models/feature_names.joblib'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing required model files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    logger.info("✅ All required model files found")
    return True

def test_model_loading():
    """Test if the model can be loaded successfully"""
    try:
        predictor = StockNeuralNetworkPredictor()
        success = predictor.load_model()
        
        if success and predictor.is_trained:
            logger.info("✅ Model loaded successfully")
            return predictor
        else:
            logger.error("❌ Model loading failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        return None

def test_prediction_functionality(predictor):
    """Test basic prediction functionality"""
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    logger.info("Testing prediction functionality...")
    
    successful_predictions = 0
    total_tests = len(test_symbols)
    
    for symbol in test_symbols:
        try:
            with app.app_context():
                result = predictor.predict(symbol)
            
            if result and 'prediction' in result:
                prediction = result['prediction']
                confidence = result.get('confidence', 0)
                logger.info(f"✅ {symbol}: {prediction} (confidence: {confidence:.1f}%)")
                successful_predictions += 1
            else:
                logger.warning(f"❌ {symbol}: No prediction returned")
                
        except Exception as e:
            logger.error(f"❌ {symbol}: Prediction error - {e}")
    
    success_rate = (successful_predictions / total_tests) * 100
    logger.info(f"Prediction success rate: {success_rate:.1f}% ({successful_predictions}/{total_tests})")
    
    return success_rate >= 80  # Require at least 80% success rate

def test_prediction_consistency(predictor):
    """Test that predictions are consistent (deterministic)"""
    test_symbol = 'AAPL'
    num_tests = 5
    
    logger.info(f"Testing prediction consistency for {test_symbol}...")
    
    predictions = []
    
    try:
        with app.app_context():
            for i in range(num_tests):
                result = predictor.predict(test_symbol)
                if result and 'prediction' in result:
                    predictions.append(result['prediction'])
                else:
                    logger.warning(f"Test {i+1}: No prediction returned")
                    return False
        
        # Check if all predictions are the same
        if len(set(predictions)) == 1:
            logger.info(f"✅ Predictions are consistent: {predictions[0]}")
            return True
        else:
            logger.warning(f"❌ Predictions are inconsistent: {predictions}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Consistency test error: {e}")
        return False

def test_prediction_diversity(predictor):
    """Test that the model can make diverse predictions"""
    diverse_symbols = [
        'AAPL',  # Large cap tech
        'XOM',   # Energy
        'JPM',   # Financial
        'JNJ',   # Healthcare
        'WMT',   # Consumer staples
        'TSLA',  # Growth stock
        'T',     # Utility/dividend
        'PLTR',  # Speculative growth
    ]
    
    logger.info("Testing prediction diversity...")
    
    predictions = []
    
    try:
        with app.app_context():
            for symbol in diverse_symbols:
                result = predictor.predict(symbol)
                if result and 'prediction' in result:
                    predictions.append(result['prediction'])
        
        unique_predictions = set(predictions)
        diversity_score = len(unique_predictions)
        
        logger.info(f"Unique predictions made: {diversity_score}")
        logger.info(f"Prediction types: {list(unique_predictions)}")
        
        # Require at least 2 different prediction types
        if diversity_score >= 2:
            logger.info("✅ Model shows good prediction diversity")
            return True
        else:
            logger.warning("❌ Model lacks prediction diversity")
            return False
            
    except Exception as e:
        logger.error(f"❌ Diversity test error: {e}")
        return False

def generate_validation_report(predictor):
    """Generate comprehensive validation report"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Test comprehensive stock list
    test_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK',
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR',
        'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD'
    ]
    
    validation_results = {
        'timestamp': timestamp,
        'model_info': {
            'model_file': 'models/stock_nn_model.pth',
            'feature_count': len(predictor.feature_names) if predictor.feature_names else 0,
            'is_trained': predictor.is_trained
        },
        'test_results': [],
        'summary': {
            'total_tests': 0,
            'successful_predictions': 0,
            'prediction_distribution': {},
            'average_confidence': 0.0
        }
    }
    
    logger.info(f"Generating validation report for {len(test_stocks)} stocks...")
    
    confidences = []
    
    try:
        with app.app_context():
            for symbol in test_stocks:
                try:
                    result = predictor.predict(symbol)
                    
                    if result and 'prediction' in result:
                        prediction = result['prediction']
                        confidence = result.get('confidence', 0)
                        
                        validation_results['test_results'].append({
                            'symbol': symbol,
                            'prediction': prediction,
                            'confidence': confidence,
                            'status': 'success'
                        })
                        
                        validation_results['summary']['successful_predictions'] += 1
                        confidences.append(confidence)
                        
                        # Track prediction distribution
                        if prediction in validation_results['summary']['prediction_distribution']:
                            validation_results['summary']['prediction_distribution'][prediction] += 1
                        else:
                            validation_results['summary']['prediction_distribution'][prediction] = 1
                    
                    else:
                        validation_results['test_results'].append({
                            'symbol': symbol,
                            'prediction': None,
                            'confidence': 0,
                            'status': 'failed'
                        })
                
                except Exception as e:
                    validation_results['test_results'].append({
                        'symbol': symbol,
                        'prediction': None,
                        'confidence': 0,
                        'status': 'error',
                        'error': str(e)
                    })
                
                validation_results['summary']['total_tests'] += 1
        
        # Calculate summary statistics
        if confidences:
            validation_results['summary']['average_confidence'] = np.mean(confidences)
        
        validation_results['summary']['success_rate'] = (
            validation_results['summary']['successful_predictions'] / 
            validation_results['summary']['total_tests'] * 100
        )
        
        # Save report
        report_file = f"models/validation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation report saved to: {report_file}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error generating validation report: {e}")
        return None

def main():
    logger.info("=" * 60)
    logger.info("FINAL MODEL VALIDATION")
    logger.info("=" * 60)
    
    # Step 1: Validate model files
    if not validate_model_files():
        logger.error("Model validation failed: Missing files")
        sys.exit(1)
    
    # Step 2: Test model loading
    predictor = test_model_loading()
    if not predictor:
        logger.error("Model validation failed: Cannot load model")
        sys.exit(1)
    
    # Step 3: Test prediction functionality
    if not test_prediction_functionality(predictor):
        logger.error("Model validation failed: Prediction functionality issues")
        sys.exit(1)
    
    # Step 4: Test prediction consistency
    if not test_prediction_consistency(predictor):
        logger.warning("Model validation warning: Predictions may not be consistent")
    
    # Step 5: Test prediction diversity
    if not test_prediction_diversity(predictor):
        logger.warning("Model validation warning: Limited prediction diversity")
    
    # Step 6: Generate comprehensive validation report
    validation_report = generate_validation_report(predictor)
    
    if validation_report:
        summary = validation_report['summary']
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Successful predictions: {summary['successful_predictions']}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Average confidence: {summary['average_confidence']:.1f}%")
        logger.info(f"Prediction distribution: {summary['prediction_distribution']}")
        logger.info("=" * 60)
        
        if summary['success_rate'] >= 80:
            logger.info("🎉 MODEL VALIDATION SUCCESSFUL!")
            logger.info("Your model is ready for production use!")
        else:
            logger.warning("⚠️  Model validation completed with warnings")
            logger.warning("Consider additional training for better performance")
    
    else:
        logger.error("Model validation failed: Could not generate validation report")
        sys.exit(1)

if __name__ == "__main__":
    main()
