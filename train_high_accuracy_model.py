"""
High-Accuracy Stock Prediction Model Training
Uses ensemble methods and advanced techniques for maximum accuracy
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
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

class HighAccuracyStockPredictor:
    def __init__(self):
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.feature_names = []
        self.selected_features = []
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
        """Prepare features for training with advanced preprocessing"""
        logger.info("Preparing features for maximum accuracy...")
        
        # Separate features from labels and metadata
        exclude_columns = [
            'symbol', 'sector', 'industry', 'exchange', 'extraction_date',
            'label', 'label_numeric', 'actual_change_percent', 'future_price',
            'days_forward', 'algorithm_prediction', 'stock_category'
        ]
        
        # Get feature columns
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        feature_df = df[feature_columns].copy()
        
        # Advanced feature preprocessing
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                try:
                    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                except:
                    logger.warning(f"Dropping non-numeric column: {col}")
                    feature_df = feature_df.drop(columns=[col])
        
        # Handle missing values with advanced imputation
        feature_df = feature_df.fillna(feature_df.median())
        
        # Remove infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        # Create additional engineered features for higher accuracy
        if 'current_price' in feature_df.columns and 'sma_20' in feature_df.columns:
            feature_df['price_sma20_ratio'] = feature_df['current_price'] / (feature_df['sma_20'] + 1e-8)
        
        if 'rsi' in feature_df.columns:
            feature_df['rsi_momentum'] = feature_df['rsi'].diff().fillna(0)
            feature_df['rsi_oversold_signal'] = (feature_df['rsi'] < 30).astype(int)
            feature_df['rsi_overbought_signal'] = (feature_df['rsi'] > 70).astype(int)
        
        if 'volume_ratio' in feature_df.columns:
            feature_df['volume_surge'] = (feature_df['volume_ratio'] > 2).astype(int)
        
        # Technical indicator combinations
        if all(col in feature_df.columns for col in ['rsi', 'macd', 'bb_position']):
            feature_df['technical_composite'] = (
                feature_df['rsi'] * 0.4 + 
                feature_df['macd'] * 0.3 + 
                feature_df['bb_position'] * 0.3
            )
        
        self.feature_names = feature_df.columns.tolist()
        logger.info(f"Prepared {len(self.feature_names)} features (including engineered features)")
        
        return feature_df.values
    
    def prepare_labels(self, df):
        """Prepare labels for 3-class classification"""
        if 'label_numeric' in df.columns:
            labels = df['label_numeric'].values
        elif 'label' in df.columns:
            labels = self.label_encoder.fit_transform(df['label'].values)
        else:
            raise ValueError("No label column found in dataset")
        
        unique_labels = np.unique(labels)
        logger.info(f"Prepared labels: {len(unique_labels)} classes - {unique_labels}")
        return labels
    
    def select_best_features(self, X, y, k=50):
        """Select the best features for maximum accuracy"""
        logger.info(f"Selecting top {k} features for maximum accuracy...")
        
        # Use SelectKBest with f_classif for feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        
        logger.info(f"Selected {len(self.selected_features)} best features")
        logger.info(f"Top 10 features: {self.selected_features[:10]}")
        
        return X_selected
    
    def create_ensemble_model(self):
        """Create an ensemble model for maximum accuracy"""
        logger.info("Creating high-accuracy ensemble model...")
        
        # Individual models with optimized parameters
        models = [
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(300, 200, 100),
                activation='relu',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=2000,
                random_state=42,
                early_stopping=True
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )),
            ('svm', SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            ))
        ]
        
        # Create voting ensemble
        self.ensemble_model = VotingClassifier(
            estimators=models,
            voting='soft',  # Use probability voting for better accuracy
            n_jobs=-1
        )
        
        return self.ensemble_model
    
    def train_model(self, X, y, use_feature_selection=True):
        """Train the high-accuracy ensemble model"""
        logger.info("Starting high-accuracy model training...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection for maximum accuracy
        if use_feature_selection and X_train_scaled.shape[1] > 20:
            X_train_selected = self.select_best_features(X_train_scaled, y_train, k=min(50, X_train_scaled.shape[1]))
            X_test_selected = self.feature_selector.transform(X_test_scaled)
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
            self.selected_features = self.feature_names
        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        logger.info(f"Using {X_train_selected.shape[1]} features")
        
        # Create and train ensemble model
        self.ensemble_model = self.create_ensemble_model()
        
        logger.info("Training ensemble model...")
        self.ensemble_model.fit(X_train_selected, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        y_pred = self.ensemble_model.predict(X_test_selected)
        y_pred_proba = self.ensemble_model.predict_proba(X_test_selected)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation for robust accuracy estimate
        cv_scores = cross_val_score(self.ensemble_model, X_train_selected, y_train, cv=5, scoring='accuracy')
        
        # Store metrics
        self.model_metrics = {
            'test_accuracy': float(accuracy),
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std_accuracy': float(cv_scores.std()),
            'num_features': len(self.feature_names),
            'num_selected_features': len(self.selected_features),
            'num_classes': len(np.unique(y)),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'model_type': 'Ensemble (MLP + RF + GB + SVM)'
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Classification report
        if hasattr(self.label_encoder, 'classes_'):
            class_names = self.label_encoder.classes_
        else:
            class_names = ['SELL', 'HOLD', 'BUY']
        
        report = classification_report(y_test, y_pred, 
                                     target_names=class_names, 
                                     output_dict=True,
                                     zero_division=0)
        
        logger.info("Classification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        
        self.model_metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.model_metrics['confusion_matrix'] = cm.tolist()
        
        logger.info("Confusion Matrix:")
        logger.info(cm)
        
        return self.ensemble_model
    
    def save_model_and_artifacts(self):
        """Save the trained model and all artifacts"""
        logger.info("Saving high-accuracy model and artifacts...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save the ensemble model
        model_path = 'models/high_accuracy_ensemble_model.joblib'
        joblib.dump(self.ensemble_model, model_path)
        
        # Save preprocessing artifacts
        scaler_path = 'models/feature_scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        encoder_path = 'models/label_encoder.joblib'
        joblib.dump(self.label_encoder, encoder_path)
        
        features_path = 'models/feature_names.joblib'
        joblib.dump(self.feature_names, features_path)
        
        selected_features_path = 'models/selected_features.joblib'
        joblib.dump(self.selected_features, selected_features_path)
        
        if self.feature_selector:
            selector_path = 'models/feature_selector.joblib'
            joblib.dump(self.feature_selector, selector_path)
        
        # Save metrics
        metrics_path = 'models/high_accuracy_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.model_metrics, f, indent=2, default=str)
        
        logger.info("High-accuracy model saved:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Accuracy: {self.model_metrics['test_accuracy']:.4f}")

def main():
    """Main training function for maximum accuracy"""
    logger.info("=" * 60)
    logger.info("HIGH-ACCURACY STOCK PREDICTION MODEL TRAINING")
    logger.info("Using Ensemble Methods for Maximum Accuracy")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = HighAccuracyStockPredictor()
    
    # Load dataset
    df = trainer.load_dataset()
    if df is None:
        logger.error("Failed to load dataset")
        return
    
    if len(df) < 50:
        logger.warning(f"Dataset too small ({len(df)} samples). Creating larger dataset...")
        # Run dataset creation first
        os.system('python run_dataset_creation.py')
        df = trainer.load_dataset()
        
        if df is None or len(df) < 50:
            logger.error("Still insufficient data for high-accuracy training")
            return
    
    # Prepare features and labels
    X = trainer.prepare_features(df)
    y = trainer.prepare_labels(df)
    
    # Train the high-accuracy model
    model = trainer.train_model(X, y, use_feature_selection=True)
    
    # Save everything
    trainer.save_model_and_artifacts()
    
    logger.info("=" * 60)
    logger.info("HIGH-ACCURACY TRAINING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Final Test Accuracy: {trainer.model_metrics['test_accuracy']:.4f}")
    logger.info(f"Cross-Validation Accuracy: {trainer.model_metrics['cv_mean_accuracy']:.4f}")
    logger.info(f"Selected Features: {trainer.model_metrics['num_selected_features']}")

if __name__ == "__main__":
    main()
