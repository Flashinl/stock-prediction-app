"""
High-Accuracy Stock Prediction with Data Augmentation
Creates synthetic samples to improve training data size while maintaining quality
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AugmentedStockPredictor:
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
    
    def augment_data(self, df, target_samples=200):
        """Create synthetic samples through intelligent data augmentation"""
        logger.info(f"Augmenting data from {len(df)} to {target_samples} samples...")
        
        # Separate by class for balanced augmentation
        classes = df['label'].unique()
        augmented_dfs = []
        
        samples_per_class = target_samples // len(classes)
        
        for class_label in classes:
            class_df = df[df['label'] == class_label].copy()
            current_count = len(class_df)
            
            logger.info(f"Augmenting {class_label}: {current_count} -> {samples_per_class} samples")
            
            if current_count >= samples_per_class:
                # If we have enough, just sample
                augmented_class = class_df.sample(n=samples_per_class, replace=False, random_state=42)
            else:
                # Need to create synthetic samples
                needed_samples = samples_per_class - current_count
                synthetic_samples = self._create_synthetic_samples(class_df, needed_samples)
                augmented_class = pd.concat([class_df, synthetic_samples], ignore_index=True)
            
            augmented_dfs.append(augmented_class)
        
        # Combine all classes
        augmented_df = pd.concat(augmented_dfs, ignore_index=True)
        
        # Shuffle the dataset
        augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Augmentation completed: {len(augmented_df)} total samples")
        
        # Show new distribution
        new_label_counts = augmented_df['label'].value_counts()
        for label, count in new_label_counts.items():
            percentage = (count / len(augmented_df)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        return augmented_df
    
    def _create_synthetic_samples(self, class_df, num_samples):
        """Create synthetic samples for a specific class using intelligent noise"""
        synthetic_samples = []
        
        # Get numeric columns only
        exclude_columns = [
            'symbol', 'sector', 'industry', 'exchange', 'extraction_date',
            'label', 'label_numeric', 'actual_change_percent', 'future_price',
            'days_forward', 'algorithm_prediction', 'stock_category'
        ]
        
        numeric_columns = [col for col in class_df.columns if col not in exclude_columns]
        
        for _ in range(num_samples):
            # Select a random base sample
            base_sample = class_df.sample(n=1).iloc[0].copy()
            
            # Add intelligent noise to numeric features
            for col in numeric_columns:
                if pd.api.types.is_numeric_dtype(class_df[col]):
                    original_value = base_sample[col]
                    
                    # Calculate noise based on feature type
                    if 'ratio' in col.lower() or 'percentage' in col.lower():
                        # For ratios and percentages, use smaller noise
                        noise_factor = 0.05  # 5% noise
                    elif 'price' in col.lower() or 'value' in col.lower():
                        # For prices and values, use moderate noise
                        noise_factor = 0.10  # 10% noise
                    elif 'score' in col.lower():
                        # For scores, use very small noise
                        noise_factor = 0.02  # 2% noise
                    else:
                        # Default noise
                        noise_factor = 0.08  # 8% noise
                    
                    # Add Gaussian noise
                    noise = np.random.normal(0, abs(original_value) * noise_factor)
                    base_sample[col] = original_value + noise
                    
                    # Ensure reasonable bounds
                    if col in ['rsi']:
                        base_sample[col] = np.clip(base_sample[col], 0, 100)
                    elif 'ratio' in col.lower() and original_value >= 0:
                        base_sample[col] = max(0, base_sample[col])
            
            synthetic_samples.append(base_sample)
        
        return pd.DataFrame(synthetic_samples)
    
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
        feature_df = df[feature_columns].copy()
        
        # Convert any object columns to numeric
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                try:
                    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                except:
                    logger.warning(f"Dropping non-numeric column: {col}")
                    feature_df = feature_df.drop(columns=[col])
        
        # Handle missing values
        feature_df = feature_df.fillna(feature_df.median())
        
        # Remove infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        self.feature_names = feature_df.columns.tolist()
        logger.info(f"Prepared {len(self.feature_names)} features for training")
        
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
    
    def create_optimized_model(self):
        """Create an optimized ensemble model for maximum accuracy"""
        logger.info("Creating optimized ensemble model...")
        
        # Individual models optimized for small datasets
        models = [
            ('mlp1', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )),
            ('mlp2', MLPClassifier(
                hidden_layer_sizes=(150, 75, 25),
                activation='tanh',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=43,
                early_stopping=True
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ]
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble
    
    def train_model(self, X, y):
        """Train the optimized model"""
        logger.info("Starting optimized model training...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Create and train model
        self.model = self.create_optimized_model()
        
        logger.info("Training ensemble model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation for robust accuracy estimate
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Store metrics
        self.model_metrics = {
            'test_accuracy': float(accuracy),
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std_accuracy': float(cv_scores.std()),
            'num_features': len(self.feature_names),
            'num_classes': len(np.unique(y)),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'model_type': 'Optimized Ensemble (2x MLP + RF + GB)'
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Classification report
        if hasattr(self.label_encoder, 'classes_'):
            class_names = self.label_encoder.classes_
        else:
            # Get actual unique classes from the data
            unique_classes = np.unique(y)
            class_names = [f'Class_{i}' for i in unique_classes]
        
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
        
        return self.model
    
    def save_model_and_artifacts(self):
        """Save the trained model and all artifacts"""
        logger.info("Saving optimized model and artifacts...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        model_path = 'models/optimized_stock_model.joblib'
        joblib.dump(self.model, model_path)
        
        # Save preprocessing artifacts
        scaler_path = 'models/feature_scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        encoder_path = 'models/label_encoder.joblib'
        joblib.dump(self.label_encoder, encoder_path)
        
        features_path = 'models/feature_names.joblib'
        joblib.dump(self.feature_names, features_path)
        
        # Save metrics
        metrics_path = 'models/optimized_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.model_metrics, f, indent=2, default=str)
        
        logger.info("Optimized model saved:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Accuracy: {self.model_metrics['test_accuracy']:.4f}")

def main():
    """Main training function with data augmentation"""
    logger.info("=" * 60)
    logger.info("OPTIMIZED STOCK PREDICTION WITH DATA AUGMENTATION")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = AugmentedStockPredictor()
    
    # Load dataset
    df = trainer.load_dataset()
    if df is None:
        logger.error("Failed to load dataset")
        return
    
    # Augment data to create more training samples
    augmented_df = trainer.augment_data(df, target_samples=200)
    
    # Prepare features and labels
    X = trainer.prepare_features(augmented_df)
    y = trainer.prepare_labels(augmented_df)
    
    # Train the model
    model = trainer.train_model(X, y)
    
    # Save everything
    trainer.save_model_and_artifacts()
    
    logger.info("=" * 60)
    logger.info("OPTIMIZED TRAINING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Final Test Accuracy: {trainer.model_metrics['test_accuracy']:.4f}")
    logger.info(f"Cross-Validation Accuracy: {trainer.model_metrics['cv_mean_accuracy']:.4f}")
    logger.info(f"Training Samples: {trainer.model_metrics['training_samples']}")

if __name__ == "__main__":
    main()
