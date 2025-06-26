#!/usr/bin/env python3
"""
Script to run neural network training for stock prediction
"""

import sys
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neural_network_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import tensorflow
        import sklearn
        import pandas
        import numpy
        logger.info("✅ All dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Please install required packages:")
        logger.error("pip install tensorflow scikit-learn pandas numpy matplotlib")
        return False

def check_dataset():
    """Check if the dataset exists"""
    dataset_paths = [
        'datasets/comprehensive_stock_dataset.csv',
        'datasets/comprehensive_stock_dataset.json'
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found dataset: {path}")
            return path
    
    logger.error("❌ No dataset found!")
    logger.error("Please run the dataset creation first:")
    logger.error("python run_dataset_creation.py")
    return None

def main():
    """Main function to run neural network training"""
    logger.info("=" * 70)
    logger.info("🧠 NEURAL NETWORK TRAINING FOR STOCK PREDICTION")
    logger.info("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Cannot proceed without required dependencies")
        return 1
    
    # Check dataset
    dataset_path = check_dataset()
    if not dataset_path:
        logger.error("❌ Cannot proceed without dataset")
        return 1
    
    try:
        # Import training module
        from train_neural_network import StockNeuralNetworkTrainer
        
        logger.info("🚀 Starting neural network training...")
        
        # Initialize trainer
        trainer = StockNeuralNetworkTrainer()
        
        # Load dataset
        logger.info(f"📊 Loading dataset from {dataset_path}")
        df = trainer.load_dataset(dataset_path)
        
        if df is None:
            logger.error("❌ Failed to load dataset")
            return 1
        
        logger.info(f"✅ Dataset loaded successfully: {len(df)} samples")
        
        # Prepare features and labels
        logger.info("🔧 Preparing features and labels...")
        X = trainer.prepare_features(df)
        y = trainer.prepare_labels(df)
        
        logger.info(f"✅ Features prepared: {X.shape}")
        logger.info(f"✅ Labels prepared: {y.shape}")
        
        # Train the model
        logger.info("🎯 Starting neural network training...")
        logger.info("This may take several minutes...")
        
        history = trainer.train_model(
            X, y,
            validation_split=0.2,
            epochs=100,  # Reduced for faster training
            batch_size=32
        )
        
        # Save everything
        logger.info("💾 Saving model and artifacts...")
        trainer.save_model_and_artifacts()
        
        # Plot training history (if matplotlib is available)
        try:
            trainer.plot_training_history()
        except Exception as e:
            logger.warning(f"Could not plot training history: {e}")
        
        # Print final results
        logger.info("=" * 70)
        logger.info("🎉 NEURAL NETWORK TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        
        metrics = trainer.model_metrics
        logger.info(f"📈 Final Accuracy: {metrics['final_accuracy']:.4f} ({metrics['final_accuracy']*100:.2f}%)")
        logger.info(f"🔧 Features Used: {metrics['num_features']}")
        logger.info(f"📊 Training Samples: {metrics['training_samples']}")
        logger.info(f"🧪 Test Samples: {metrics['test_samples']}")
        logger.info(f"🏷️  Classes: {metrics['num_classes']}")
        
        logger.info("=" * 70)
        logger.info("📁 FILES CREATED:")
        logger.info("🤖 models/stock_neural_network.h5 - Trained neural network")
        logger.info("⚖️  models/feature_scaler.joblib - Feature scaler")
        logger.info("🏷️  models/label_encoder.joblib - Label encoder")
        logger.info("📋 models/feature_names.joblib - Feature names")
        logger.info("📊 models/training_metrics.json - Training metrics")
        logger.info("📈 models/training_history.json - Training history")
        logger.info("🖼️  models/training_history.png - Training plots")
        logger.info("=" * 70)
        
        # Show classification performance
        if 'classification_report' in metrics:
            logger.info("📊 CLASSIFICATION PERFORMANCE:")
            report = metrics['classification_report']
            for class_name, class_metrics in report.items():
                if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                    precision = class_metrics['precision']
                    recall = class_metrics['recall']
                    f1 = class_metrics['f1-score']
                    logger.info(f"   {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        logger.info("🎯 Neural network is now ready for stock predictions!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.error("Check the logs for more details")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
