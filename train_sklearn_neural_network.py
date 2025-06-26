"""
Scikit-learn Neural Network Training for Stock Prediction
Uses MLPClassifier (Multi-layer Perceptron) with comprehensive features
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockMLPTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.model_metrics = {}
        
    def load_dataset(self, dataset_path='datasets/comprehensive_stock_dataset.csv'):
        """Load the comprehensive dataset"""
        try:
            logger.info(f"Loading dataset from {dataset_path}")
            
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            else:
                df = pd.read_json(dataset_path)
            
            logger.info(f"Dataset loaded: {len(df)} samples, {len(df.columns)} features")
            
            # Show basic dataset info
            logger.info("Label distribution:")
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                for label, count in label_counts.items():
                    percentage = (count / len(df)) * 100
                    logger.info(f"  {label}: {count} ({percentage:.1f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features for training"""
        logger.info("Preparing features for training...")
        
        # Separate features from labels and metadata
        exclude_columns = [
            'symbol', 'sector', 'industry', 'exchange', 'extraction_date',
            'label', 'label_numeric', 'actual_change_percent', 'future_price',
            'days_forward', 'algorithm_prediction', 'stock_category'
        ]
        
        # Get feature columns
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Handle any remaining non-numeric columns
        feature_df = df[feature_columns].copy()
        
        # Convert any object columns to numeric
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                try:
                    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                except:
                    logger.warning(f"Could not convert column {col} to numeric, dropping it")
                    feature_df = feature_df.drop(columns=[col])
        
        # Fill NaN values with 0
        feature_df = feature_df.fillna(0)
        
        # Remove any infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        self.feature_names = feature_df.columns.tolist()
        logger.info(f"Prepared {len(self.feature_names)} features for training")
        
        # Show feature categories
        technical_features = [f for f in self.feature_names if any(x in f.lower() for x in 
                             ['rsi', 'macd', 'sma', 'volume', 'price', 'bb_', 'momentum', 'technical'])]
        fundamental_features = [f for f in self.feature_names if any(x in f.lower() for x in 
                               ['revenue', 'ebitda', 'debt', 'ratio', 'margin', 'growth', 'pe_', 'eps', 'cash'])]
        
        logger.info(f"Technical features: {len(technical_features)}")
        logger.info(f"Fundamental features: {len(fundamental_features)}")
        logger.info(f"Other features: {len(self.feature_names) - len(technical_features) - len(fundamental_features)}")
        
        return feature_df.values
    
    def prepare_labels(self, df):
        """Prepare labels for training"""
        if 'label_numeric' in df.columns:
            labels = df['label_numeric'].values
        elif 'label' in df.columns:
            # Convert text labels to numeric
            labels = self.label_encoder.fit_transform(df['label'].values)
        else:
            raise ValueError("No label column found in dataset")
        
        logger.info(f"Prepared labels: {len(np.unique(labels))} classes")
        return labels
    
    def train_model(self, X, y, use_grid_search=True):
        """Train the MLP neural network"""
        logger.info("Starting neural network training with scikit-learn MLPClassifier...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        if use_grid_search and len(X_train) > 50:  # Only use grid search if we have enough data
            logger.info("Performing hyperparameter tuning with GridSearchCV...")
            
            # Define parameter grid
            param_grid = {
                'hidden_layer_sizes': [
                    (100,), (200,), (100, 50), (200, 100), (300, 150, 75)
                ],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000]
            }
            
            # Create base model
            mlp = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
            
            # Grid search
            grid_search = GridSearchCV(
                mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
        else:
            logger.info("Training with default parameters...")
            
            # Create and train model with good default parameters
            self.model = MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),
                activation='relu',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            )
            
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store metrics
        self.model_metrics = {
            'test_accuracy': float(accuracy),
            'num_features': len(self.feature_names),
            'num_classes': len(np.unique(y)),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'model_type': 'MLPClassifier'
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Generate classification report
        if hasattr(self.label_encoder, 'classes_'):
            class_names = self.label_encoder.classes_
        else:
            # Get unique classes from the actual data
            unique_classes = np.unique(y)
            class_names = [f'Class_{i}' for i in unique_classes]

        report = classification_report(y_test, y_pred,
                                     target_names=class_names,
                                     output_dict=True,
                                     zero_division=0)

        logger.info("Classification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        
        # Store detailed metrics
        self.model_metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.model_metrics['confusion_matrix'] = cm.tolist()
        
        logger.info("Confusion Matrix:")
        logger.info(cm)
        
        return self.model
    
    def save_model_and_artifacts(self):
        """Save the trained model and all artifacts"""
        logger.info("Saving model and artifacts...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        model_path = 'models/stock_mlp_model.joblib'
        joblib.dump(self.model, model_path)
        
        # Save the scaler
        scaler_path = 'models/feature_scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        # Save the label encoder
        encoder_path = 'models/label_encoder.joblib'
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save feature names
        features_path = 'models/feature_names.joblib'
        joblib.dump(self.feature_names, features_path)
        
        # Save training metrics
        metrics_path = 'models/training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.model_metrics, f, indent=2, default=str)
        
        logger.info("Model and artifacts saved:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Scaler: {scaler_path}")
        logger.info(f"  Label Encoder: {encoder_path}")
        logger.info(f"  Features: {features_path}")
        logger.info(f"  Metrics: {metrics_path}")
        
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        if 'confusion_matrix' not in self.model_metrics:
            logger.warning("No confusion matrix available")
            return
        
        try:
            cm = np.array(self.model_metrics['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            
            if hasattr(self.label_encoder, 'classes_'):
                class_names = self.label_encoder.classes_
            else:
                class_names = ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Confusion matrix plot saved to models/confusion_matrix.png")
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")

def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("NEURAL NETWORK TRAINING FOR STOCK PREDICTION")
    logger.info("Using scikit-learn MLPClassifier")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = StockMLPTrainer()
    
    # Load dataset
    df = trainer.load_dataset()
    if df is None:
        logger.error("Failed to load dataset")
        return
    
    # Prepare features and labels
    X = trainer.prepare_features(df)
    y = trainer.prepare_labels(df)
    
    # Train the model
    model = trainer.train_model(X, y, use_grid_search=True)
    
    # Save everything
    trainer.save_model_and_artifacts()
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix()
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Final Accuracy: {trainer.model_metrics['test_accuracy']:.4f}")
    logger.info(f"Features Used: {trainer.model_metrics['num_features']}")
    logger.info(f"Training Samples: {trainer.model_metrics['training_samples']}")

if __name__ == "__main__":
    main()
