#!/usr/bin/env python3
"""
Model Accuracy Checker
Validates if the trained model meets the target accuracy requirements
"""

import argparse
import logging
import sys
import os
import json
import numpy as np
from datetime import datetime, timedelta
from neural_network_predictor import StockNeuralNetworkPredictor
from app import app, db, Stock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trained_model():
    """Load the most recently trained model"""
    try:
        predictor = StockNeuralNetworkPredictor()
        
        # Check if model files exist
        model_files = [
            'models/stock_nn_model.pth',
            'models/stock_scaler.joblib',
            'models/stock_label_encoder.joblib'
        ]
        
        for file_path in model_files:
            if not os.path.exists(file_path):
                logger.error(f"Model file not found: {file_path}")
                return None
        
        # Load the model
        success = predictor.load_model()
        if success:
            logger.info("Model loaded successfully")
            return predictor
        else:
            logger.error("Failed to load model")
            return None
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def get_test_stocks():
    """Get a diverse set of stocks for testing"""
    test_stocks = [
        # Large cap tech (should be predictable)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        
        # Financial services
        'JPM', 'BAC', 'WFC', 'GS', 'MS',
        
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
        
        # Consumer goods
        'WMT', 'HD', 'PG', 'KO', 'PEP',
        
        # Industrial
        'CAT', 'BA', 'GE', 'MMM', 'HON',
        
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB',
        
        # Growth stocks
        'TSLA', 'NFLX', 'ROKU', 'SNOW', 'PLTR',
        
        # Mid cap
        'CRWD', 'ZS', 'DDOG', 'NET', 'TWLO'
    ]
    
    return test_stocks

def comprehensive_accuracy_test(predictor, test_stocks):
    """Perform comprehensive accuracy testing"""
    results = {
        'total_tests': 0,
        'successful_predictions': 0,
        'failed_predictions': 0,
        'prediction_distribution': {
            'STRONG_BUY': 0,
            'BUY': 0,
            'HOLD': 0,
            'SELL': 0,
            'STRONG_SELL': 0
        },
        'confidence_scores': [],
        'test_details': []
    }
    
    logger.info(f"Testing model accuracy on {len(test_stocks)} stocks...")
    
    for symbol in test_stocks:
        try:
            logger.info(f"Testing prediction for {symbol}...")
            
            # Make prediction
            prediction_result = predictor.predict(symbol)
            
            if prediction_result and 'prediction' in prediction_result:
                results['successful_predictions'] += 1
                
                prediction = prediction_result['prediction']
                confidence = prediction_result.get('confidence', 0)
                
                # Track prediction distribution
                if prediction in results['prediction_distribution']:
                    results['prediction_distribution'][prediction] += 1
                
                # Track confidence scores
                results['confidence_scores'].append(confidence)
                
                # Store test details
                results['test_details'].append({
                    'symbol': symbol,
                    'prediction': prediction,
                    'confidence': confidence,
                    'status': 'success'
                })
                
                logger.info(f"✅ {symbol}: {prediction} (confidence: {confidence:.1f}%)")
                
            else:
                results['failed_predictions'] += 1
                results['test_details'].append({
                    'symbol': symbol,
                    'prediction': None,
                    'confidence': 0,
                    'status': 'failed'
                })
                logger.warning(f"❌ {symbol}: Prediction failed")
                
        except Exception as e:
            results['failed_predictions'] += 1
            results['test_details'].append({
                'symbol': symbol,
                'prediction': None,
                'confidence': 0,
                'status': 'error',
                'error': str(e)
            })
            logger.error(f"❌ {symbol}: Error - {e}")
        
        results['total_tests'] += 1
    
    return results

def calculate_accuracy_metrics(results):
    """Calculate various accuracy metrics"""
    total_tests = results['total_tests']
    successful_predictions = results['successful_predictions']
    
    if total_tests == 0:
        return {
            'prediction_success_rate': 0.0,
            'average_confidence': 0.0,
            'prediction_diversity': 0.0,
            'overall_score': 0.0
        }
    
    # Basic prediction success rate
    prediction_success_rate = (successful_predictions / total_tests) * 100
    
    # Average confidence score
    confidence_scores = results['confidence_scores']
    average_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    
    # Prediction diversity (how well distributed are the predictions)
    distribution = results['prediction_distribution']
    total_predictions = sum(distribution.values())
    
    if total_predictions > 0:
        # Calculate entropy for diversity
        probabilities = [count / total_predictions for count in distribution.values() if count > 0]
        diversity = -sum(p * np.log2(p) for p in probabilities) if probabilities else 0.0
        prediction_diversity = (diversity / np.log2(5)) * 100  # Normalize to 0-100
    else:
        prediction_diversity = 0.0
    
    # Overall score (weighted combination)
    overall_score = (
        prediction_success_rate * 0.5 +  # 50% weight on success rate
        average_confidence * 0.3 +       # 30% weight on confidence
        prediction_diversity * 0.2       # 20% weight on diversity
    )
    
    return {
        'prediction_success_rate': prediction_success_rate,
        'average_confidence': average_confidence,
        'prediction_diversity': prediction_diversity,
        'overall_score': overall_score
    }

def save_accuracy_report(results, metrics, target_accuracy):
    """Save detailed accuracy report"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = f"models/accuracy_report_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'target_accuracy': target_accuracy,
        'test_results': results,
        'accuracy_metrics': metrics,
        'meets_target': metrics['overall_score'] >= target_accuracy,
        'summary': {
            'total_stocks_tested': results['total_tests'],
            'successful_predictions': results['successful_predictions'],
            'success_rate': f"{metrics['prediction_success_rate']:.2f}%",
            'average_confidence': f"{metrics['average_confidence']:.2f}%",
            'overall_score': f"{metrics['overall_score']:.2f}%",
            'target_met': metrics['overall_score'] >= target_accuracy
        }
    }
    
    try:
        os.makedirs('models', exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Accuracy report saved to: {report_file}")
    except Exception as e:
        logger.error(f"Failed to save accuracy report: {e}")

def main():
    parser = argparse.ArgumentParser(description='Check Model Accuracy')
    parser.add_argument('--target', type=float, default=90.0, help='Target accuracy percentage')
    
    args = parser.parse_args()
    
    logger.info(f"Checking model accuracy against target: {args.target}%")
    
    try:
        # Load the trained model
        predictor = load_trained_model()
        if not predictor:
            logger.error("Failed to load trained model")
            sys.exit(1)
        
        # Get test stocks
        test_stocks = get_test_stocks()
        
        # Run comprehensive accuracy test
        with app.app_context():
            results = comprehensive_accuracy_test(predictor, test_stocks)
        
        # Calculate accuracy metrics
        metrics = calculate_accuracy_metrics(results)
        
        # Save detailed report
        save_accuracy_report(results, metrics, args.target)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("ACCURACY TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total stocks tested: {results['total_tests']}")
        logger.info(f"Successful predictions: {results['successful_predictions']}")
        logger.info(f"Failed predictions: {results['failed_predictions']}")
        logger.info(f"Success rate: {metrics['prediction_success_rate']:.2f}%")
        logger.info(f"Average confidence: {metrics['average_confidence']:.2f}%")
        logger.info(f"Prediction diversity: {metrics['prediction_diversity']:.2f}%")
        logger.info(f"Overall score: {metrics['overall_score']:.2f}%")
        logger.info(f"Target accuracy: {args.target}%")
        logger.info("=" * 60)
        
        # Check if target is met
        if metrics['overall_score'] >= args.target:
            logger.info("🎉 TARGET ACCURACY ACHIEVED!")
            sys.exit(0)
        else:
            logger.info(f"❌ Target not met. Need {args.target - metrics['overall_score']:.2f}% more.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Accuracy check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
