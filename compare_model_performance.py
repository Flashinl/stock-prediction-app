"""
Compare Performance: Neural Network vs Original Algorithm
Comprehensive evaluation of both prediction systems
"""

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPerformanceComparator:
    def __init__(self):
        self.neural_network = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.comparison_results = {}
        
    def load_neural_network_model(self):
        """Load the trained neural network model and artifacts"""
        try:
            logger.info("Loading neural network model...")
            
            # Load model
            self.neural_network = joblib.load('models/optimized_stock_model.joblib')
            
            # Load preprocessing artifacts
            self.scaler = joblib.load('models/feature_scaler.joblib')
            self.label_encoder = joblib.load('models/label_encoder.joblib')
            self.feature_names = joblib.load('models/feature_names.joblib')
            
            logger.info(f"Neural network model loaded successfully")
            logger.info(f"Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading neural network model: {e}")
            return False
    
    def load_test_dataset(self, dataset_path='datasets/comprehensive_stock_dataset.csv'):
        """Load the test dataset for comparison"""
        try:
            logger.info(f"Loading test dataset from {dataset_path}")
            
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            else:
                df = pd.read_json(dataset_path)
            
            logger.info(f"Test dataset loaded: {len(df)} samples")
            
            # Show label distribution
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                logger.info("Test dataset label distribution:")
                for label, count in label_counts.items():
                    percentage = (count / len(df)) * 100
                    logger.info(f"  {label}: {count} ({percentage:.1f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading test dataset: {e}")
            return None
    
    def prepare_neural_network_features(self, df):
        """Prepare features for neural network prediction"""
        # Separate features from labels and metadata
        exclude_columns = [
            'symbol', 'sector', 'industry', 'exchange', 'extraction_date',
            'label', 'label_numeric', 'actual_change_percent', 'future_price',
            'days_forward', 'algorithm_prediction', 'stock_category'
        ]
        
        # Get feature columns that match our trained model
        available_features = [col for col in df.columns if col not in exclude_columns]
        
        # Ensure we have the same features as training
        feature_df = df[available_features].copy()
        
        # Convert any object columns to numeric
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                try:
                    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                except:
                    feature_df = feature_df.drop(columns=[col])
        
        # Handle missing values
        feature_df = feature_df.fillna(feature_df.median())
        
        # Remove infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        # Scale features
        X_scaled = self.scaler.transform(feature_df.values)
        
        return X_scaled
    
    def get_algorithm_predictions(self, df):
        """Extract original algorithm predictions from dataset"""
        if 'algorithm_prediction' in df.columns:
            return df['algorithm_prediction'].values
        else:
            logger.warning("No algorithm predictions found in dataset")
            return None
    
    def get_actual_outcomes(self, df):
        """Extract actual outcomes from dataset"""
        if 'label' in df.columns:
            return df['label'].values
        else:
            logger.warning("No actual outcomes found in dataset")
            return None
    
    def predict_neural_network(self, X):
        """Make predictions using neural network"""
        try:
            predictions = self.neural_network.predict(X)
            
            # Convert numeric predictions back to labels
            if hasattr(self.label_encoder, 'classes_'):
                label_predictions = self.label_encoder.inverse_transform(predictions)
            else:
                # Map numeric to labels
                label_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                label_predictions = [label_map.get(pred, 'UNKNOWN') for pred in predictions]
            
            return label_predictions
            
        except Exception as e:
            logger.error(f"Error making neural network predictions: {e}")
            return None
    
    def calculate_performance_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive performance metrics"""
        try:
            # Basic accuracy
            accuracy = accuracy_score(y_true, y_pred)
            
            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=['SELL', 'HOLD', 'BUY'])
            
            # Per-class metrics
            class_metrics = {}
            for label in ['SELL', 'HOLD', 'BUY']:
                if label in report:
                    class_metrics[label] = {
                        'precision': report[label]['precision'],
                        'recall': report[label]['recall'],
                        'f1_score': report[label]['f1-score'],
                        'support': report[label]['support']
                    }
            
            metrics = {
                'model_name': model_name,
                'accuracy': accuracy,
                'macro_avg_precision': report['macro avg']['precision'],
                'macro_avg_recall': report['macro avg']['recall'],
                'macro_avg_f1': report['macro avg']['f1-score'],
                'weighted_avg_precision': report['weighted avg']['precision'],
                'weighted_avg_recall': report['weighted avg']['recall'],
                'weighted_avg_f1': report['weighted avg']['f1-score'],
                'class_metrics': class_metrics,
                'confusion_matrix': cm.tolist(),
                'total_predictions': len(y_true)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {e}")
            return None
    
    def compare_models(self, df):
        """Compare neural network vs original algorithm performance"""
        logger.info("=" * 60)
        logger.info("COMPARING MODEL PERFORMANCE")
        logger.info("=" * 60)
        
        # Prepare data
        X = self.prepare_neural_network_features(df)
        actual_outcomes = self.get_actual_outcomes(df)
        algorithm_predictions = self.get_algorithm_predictions(df)
        
        if actual_outcomes is None:
            logger.error("Cannot compare without actual outcomes")
            return None
        
        # Get neural network predictions
        neural_predictions = self.predict_neural_network(X)
        
        if neural_predictions is None:
            logger.error("Failed to get neural network predictions")
            return None
        
        # Calculate metrics for neural network
        logger.info("Calculating Neural Network performance...")
        nn_metrics = self.calculate_performance_metrics(
            actual_outcomes, neural_predictions, "Neural Network"
        )
        
        # Calculate metrics for original algorithm (if available)
        algo_metrics = None
        if algorithm_predictions is not None:
            logger.info("Calculating Original Algorithm performance...")
            algo_metrics = self.calculate_performance_metrics(
                actual_outcomes, algorithm_predictions, "Original Algorithm"
            )
        
        # Store comparison results
        self.comparison_results = {
            'neural_network': nn_metrics,
            'original_algorithm': algo_metrics,
            'comparison_date': datetime.now().isoformat(),
            'test_samples': len(actual_outcomes)
        }
        
        return self.comparison_results
    
    def print_comparison_results(self):
        """Print detailed comparison results"""
        if not self.comparison_results:
            logger.error("No comparison results available")
            return
        
        logger.info("=" * 80)
        logger.info("DETAILED PERFORMANCE COMPARISON")
        logger.info("=" * 80)
        
        nn_metrics = self.comparison_results['neural_network']
        algo_metrics = self.comparison_results['original_algorithm']
        
        # Overall accuracy comparison
        logger.info(f"\n📊 OVERALL ACCURACY:")
        logger.info(f"Neural Network:     {nn_metrics['accuracy']:.4f} ({nn_metrics['accuracy']*100:.2f}%)")
        if algo_metrics:
            logger.info(f"Original Algorithm: {algo_metrics['accuracy']:.4f} ({algo_metrics['accuracy']*100:.2f}%)")
            improvement = nn_metrics['accuracy'] - algo_metrics['accuracy']
            logger.info(f"Improvement:        {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        # Per-class performance
        logger.info(f"\n📈 PER-CLASS PERFORMANCE:")
        
        for class_name in ['SELL', 'HOLD', 'BUY']:
            logger.info(f"\n{class_name} Class:")
            
            if class_name in nn_metrics['class_metrics']:
                nn_class = nn_metrics['class_metrics'][class_name]
                logger.info(f"  Neural Network:")
                logger.info(f"    Precision: {nn_class['precision']:.4f} ({nn_class['precision']*100:.1f}%)")
                logger.info(f"    Recall:    {nn_class['recall']:.4f} ({nn_class['recall']*100:.1f}%)")
                logger.info(f"    F1-Score:  {nn_class['f1_score']:.4f} ({nn_class['f1_score']*100:.1f}%)")
                logger.info(f"    Support:   {nn_class['support']} samples")
            
            if algo_metrics and class_name in algo_metrics['class_metrics']:
                algo_class = algo_metrics['class_metrics'][class_name]
                logger.info(f"  Original Algorithm:")
                logger.info(f"    Precision: {algo_class['precision']:.4f} ({algo_class['precision']*100:.1f}%)")
                logger.info(f"    Recall:    {algo_class['recall']:.4f} ({algo_class['recall']*100:.1f}%)")
                logger.info(f"    F1-Score:  {algo_class['f1_score']:.4f} ({algo_class['f1_score']*100:.1f}%)")
                logger.info(f"    Support:   {algo_class['support']} samples")
                
                # Calculate improvements
                prec_imp = nn_class['precision'] - algo_class['precision']
                rec_imp = nn_class['recall'] - algo_class['recall']
                f1_imp = nn_class['f1_score'] - algo_class['f1_score']
                
                logger.info(f"  Improvements:")
                logger.info(f"    Precision: {prec_imp:+.4f} ({prec_imp*100:+.1f}%)")
                logger.info(f"    Recall:    {rec_imp:+.4f} ({rec_imp*100:+.1f}%)")
                logger.info(f"    F1-Score:  {f1_imp:+.4f} ({f1_imp*100:+.1f}%)")
        
        # Confusion matrices
        logger.info(f"\n🎯 CONFUSION MATRICES:")
        
        logger.info(f"\nNeural Network:")
        nn_cm = np.array(nn_metrics['confusion_matrix'])
        logger.info("Predicted:  SELL  HOLD  BUY")
        logger.info("Actual:")
        for i, actual_class in enumerate(['SELL', 'HOLD', 'BUY']):
            row = "  ".join([f"{val:4d}" for val in nn_cm[i]])
            logger.info(f"{actual_class:4s}      {row}")
        
        if algo_metrics:
            logger.info(f"\nOriginal Algorithm:")
            algo_cm = np.array(algo_metrics['confusion_matrix'])
            logger.info("Predicted:  SELL  HOLD  BUY")
            logger.info("Actual:")
            for i, actual_class in enumerate(['SELL', 'HOLD', 'BUY']):
                row = "  ".join([f"{val:4d}" for val in algo_cm[i]])
                logger.info(f"{actual_class:4s}      {row}")
    
    def save_comparison_results(self, output_path='results/performance_comparison.json'):
        """Save comparison results to file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self.comparison_results, f, indent=2, default=str)
            
            logger.info(f"Comparison results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving comparison results: {e}")

def main():
    """Main comparison function"""
    logger.info("=" * 60)
    logger.info("MODEL PERFORMANCE COMPARISON")
    logger.info("Neural Network vs Original Algorithm")
    logger.info("=" * 60)
    
    # Initialize comparator
    comparator = ModelPerformanceComparator()
    
    # Load neural network model
    if not comparator.load_neural_network_model():
        logger.error("Failed to load neural network model")
        return
    
    # Load test dataset
    test_df = comparator.load_test_dataset()
    if test_df is None:
        logger.error("Failed to load test dataset")
        return
    
    # Compare models
    results = comparator.compare_models(test_df)
    if results is None:
        logger.error("Failed to compare models")
        return
    
    # Print detailed results
    comparator.print_comparison_results()
    
    # Save results
    comparator.save_comparison_results()
    
    logger.info("=" * 60)
    logger.info("PERFORMANCE COMPARISON COMPLETED!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
