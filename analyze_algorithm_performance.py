"""
Analyze Rule-Based Algorithm Performance vs Neural Network
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_algorithm_performance():
    """Analyze how the rule-based algorithm performed vs actual outcomes"""
    
    # Load the dataset
    df = pd.read_csv('datasets/comprehensive_stock_dataset.csv')
    
    logger.info("=" * 60)
    logger.info("RULE-BASED ALGORITHM PERFORMANCE ANALYSIS")
    logger.info("=" * 60)
    
    # Get algorithm predictions and actual outcomes
    algorithm_predictions = df['algorithm_prediction'].values
    actual_labels = df['label'].values
    
    logger.info(f"Total predictions analyzed: {len(df)}")
    
    # Show distribution of algorithm predictions
    logger.info("\nAlgorithm Prediction Distribution:")
    algo_counts = pd.Series(algorithm_predictions).value_counts()
    for pred, count in algo_counts.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  {pred}: {count} ({percentage:.1f}%)")
    
    # Show distribution of actual outcomes
    logger.info("\nActual Outcome Distribution:")
    actual_counts = pd.Series(actual_labels).value_counts()
    for label, count in actual_counts.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Convert to binary classification for fair comparison
    # Algorithm: BUY/STRONG BUY = BUY, others = HOLD
    # Actual: BUY = BUY, HOLD = HOLD
    
    algo_binary = []
    for pred in algorithm_predictions:
        if 'BUY' in pred:
            algo_binary.append('BUY')
        else:
            algo_binary.append('HOLD')
    
    # Calculate accuracy
    accuracy = accuracy_score(actual_labels, algo_binary)
    
    logger.info("=" * 60)
    logger.info("RULE-BASED ALGORITHM PERFORMANCE")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    logger.info("\nClassification Report:")
    report = classification_report(actual_labels, algo_binary, zero_division=0)
    logger.info(report)
    
    # Confusion matrix
    logger.info("Confusion Matrix:")
    cm = confusion_matrix(actual_labels, algo_binary)
    logger.info(cm)
    
    # Detailed analysis by prediction type
    logger.info("\n" + "=" * 60)
    logger.info("DETAILED PREDICTION ANALYSIS")
    logger.info("=" * 60)
    
    # Create detailed comparison
    comparison_df = pd.DataFrame({
        'Symbol': df['symbol'],
        'Algorithm_Prediction': algorithm_predictions,
        'Algorithm_Binary': algo_binary,
        'Actual_Outcome': actual_labels,
        'Algorithm_Confidence': df['algorithm_confidence'],
        'Algorithm_Expected_Change': df['algorithm_expected_change'],
        'Actual_Change_Percent': df['actual_change_percent'],
        'Correct': [1 if a == b else 0 for a, b in zip(algo_binary, actual_labels)]
    })
    
    # Show correct vs incorrect predictions
    correct_predictions = comparison_df[comparison_df['Correct'] == 1]
    incorrect_predictions = comparison_df[comparison_df['Correct'] == 0]
    
    logger.info(f"Correct Predictions: {len(correct_predictions)}/{len(comparison_df)} ({len(correct_predictions)/len(comparison_df)*100:.1f}%)")
    logger.info(f"Incorrect Predictions: {len(incorrect_predictions)}/{len(comparison_df)} ({len(incorrect_predictions)/len(comparison_df)*100:.1f}%)")
    
    # Analyze by prediction type
    logger.info("\nPerformance by Prediction Type:")
    
    for pred_type in ['BUY', 'HOLD']:
        pred_subset = comparison_df[comparison_df['Algorithm_Binary'] == pred_type]
        if len(pred_subset) > 0:
            correct_count = len(pred_subset[pred_subset['Correct'] == 1])
            accuracy_type = correct_count / len(pred_subset)
            logger.info(f"  {pred_type}: {correct_count}/{len(pred_subset)} correct ({accuracy_type*100:.1f}%)")
    
    # Show incorrect predictions in detail
    if len(incorrect_predictions) > 0:
        logger.info("\nIncorrect Predictions Details:")
        for _, row in incorrect_predictions.iterrows():
            logger.info(f"  {row['Symbol']}: Predicted {row['Algorithm_Binary']}, Actual {row['Actual_Outcome']} "
                       f"(Expected: {row['Algorithm_Expected_Change']:.1f}%, Actual: {row['Actual_Change_Percent']:.1f}%)")
    
    # Compare with Neural Network performance
    logger.info("\n" + "=" * 60)
    logger.info("ALGORITHM COMPARISON")
    logger.info("=" * 60)
    
    # Load neural network metrics
    try:
        import json
        with open('models/optimized_metrics.json', 'r') as f:
            nn_metrics = json.load(f)
        
        nn_accuracy = nn_metrics['test_accuracy']
        nn_cv_accuracy = nn_metrics['cv_mean_accuracy']
        
        logger.info(f"Rule-Based Algorithm Accuracy: {accuracy*100:.2f}%")
        logger.info(f"Neural Network Test Accuracy: {nn_accuracy*100:.2f}%")
        logger.info(f"Neural Network CV Accuracy: {nn_cv_accuracy*100:.2f}%")
        
        improvement = (nn_accuracy - accuracy) * 100
        logger.info(f"Neural Network Improvement: +{improvement:.2f} percentage points")
        
        if improvement > 0:
            logger.info("🎉 Neural Network significantly outperforms the rule-based algorithm!")
        else:
            logger.info("⚠️ Rule-based algorithm performs comparably to neural network")
            
    except Exception as e:
        logger.warning(f"Could not load neural network metrics: {e}")
    
    # Analyze confidence vs accuracy
    logger.info("\n" + "=" * 60)
    logger.info("CONFIDENCE vs ACCURACY ANALYSIS")
    logger.info("=" * 60)
    
    # Group by confidence levels
    high_conf = comparison_df[comparison_df['Algorithm_Confidence'] >= 80]
    med_conf = comparison_df[(comparison_df['Algorithm_Confidence'] >= 70) & (comparison_df['Algorithm_Confidence'] < 80)]
    low_conf = comparison_df[comparison_df['Algorithm_Confidence'] < 70]
    
    for conf_group, name in [(high_conf, 'High Confidence (≥80%)'), 
                            (med_conf, 'Medium Confidence (70-79%)'), 
                            (low_conf, 'Low Confidence (<70%)')]:
        if len(conf_group) > 0:
            correct_count = len(conf_group[conf_group['Correct'] == 1])
            accuracy_conf = correct_count / len(conf_group)
            logger.info(f"{name}: {correct_count}/{len(conf_group)} correct ({accuracy_conf*100:.1f}%)")
    
    return comparison_df, accuracy

def main():
    """Main analysis function"""
    comparison_df, accuracy = analyze_algorithm_performance()
    
    # Save detailed results
    comparison_df.to_csv('datasets/algorithm_performance_analysis.csv', index=False)
    logger.info(f"\nDetailed analysis saved to: datasets/algorithm_performance_analysis.csv")

if __name__ == "__main__":
    main()
