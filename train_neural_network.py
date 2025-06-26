"""
Neural Network Training Pipeline for Stock Prediction
Uses comprehensive dataset with technical + fundamental features
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockNeuralNetworkTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.training_history = None
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
            logger.info(f"Label distribution:")
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

    def build_model(self, input_dim, num_classes=5):
        """Build the neural network architecture"""
        logger.info(f"Building neural network: {input_dim} inputs -> {num_classes} classes")

        model = keras.Sequential([
            # Input layer with batch normalization
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            # Hidden layers with decreasing size
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),

            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])

        # Compile with appropriate optimizer and metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print model summary
        model.summary()

        return model

    def train_model(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Train the neural network"""
        logger.info("Starting neural network training...")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build the model
        num_classes = len(np.unique(y))
        self.model = self.build_model(X_train_scaled.shape[1], num_classes)

        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_stock_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Create models directory
        os.makedirs('models', exist_ok=True)

        # Train the model
        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples")

        self.training_history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)

        # Make predictions for detailed metrics
        y_pred = self.model.predict(X_test_scaled, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_classes)

        # Store metrics
        self.model_metrics = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'final_accuracy': float(accuracy),
            'num_features': len(self.feature_names),
            'num_classes': num_classes,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

        logger.info(f"Final Test Accuracy: {accuracy:.4f}")
        logger.info(f"Final Test Loss: {test_loss:.4f}")

        # Generate classification report
        if hasattr(self.label_encoder, 'classes_'):
            class_names = self.label_encoder.classes_
        else:
            class_names = ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']

        report = classification_report(y_test, y_pred_classes,
                                     target_names=class_names,
                                     output_dict=True)

        logger.info("Classification Report:")
        logger.info(classification_report(y_test, y_pred_classes, target_names=class_names))

        # Store detailed metrics
        self.model_metrics['classification_report'] = report

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        self.model_metrics['confusion_matrix'] = cm.tolist()

        logger.info("Confusion Matrix:")
        logger.info(cm)

        return self.training_history

    def save_model_and_artifacts(self):
        """Save the trained model and all artifacts"""
        logger.info("Saving model and artifacts...")

        os.makedirs('models', exist_ok=True)

        # Save the model
        model_path = 'models/stock_neural_network.h5'
        self.model.save(model_path)

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

        # Save training history
        if self.training_history:
            history_path = 'models/training_history.json'
            history_dict = {
                'loss': self.training_history.history['loss'],
                'accuracy': self.training_history.history['accuracy'],
                'val_loss': self.training_history.history['val_loss'],
                'val_accuracy': self.training_history.history['val_accuracy']
            }
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)

        logger.info("Model and artifacts saved:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Scaler: {scaler_path}")
        logger.info(f"  Label Encoder: {encoder_path}")
        logger.info(f"  Features: {features_path}")
        logger.info(f"  Metrics: {metrics_path}")

    def plot_training_history(self):
        """Plot training history"""
        if not self.training_history:
            logger.warning("No training history available")
            return

        try:
            plt.figure(figsize=(12, 4))

            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.training_history.history['accuracy'], label='Training Accuracy')
            plt.plot(self.training_history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(self.training_history.history['loss'], label='Training Loss')
            plt.plot(self.training_history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
            plt.show()

            logger.info("Training history plot saved to models/training_history.png")

        except Exception as e:
            logger.error(f"Error plotting training history: {e}")

def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("NEURAL NETWORK TRAINING FOR STOCK PREDICTION")
    logger.info("=" * 60)

    # Initialize trainer
    trainer = StockNeuralNetworkTrainer()

    # Load dataset
    df = trainer.load_dataset()
    if df is None:
        logger.error("Failed to load dataset")
        return

    # Prepare features and labels
    X = trainer.prepare_features(df)
    y = trainer.prepare_labels(df)

    # Train the model
    history = trainer.train_model(
        X, y,
        validation_split=0.2,
        epochs=150,
        batch_size=32
    )

    # Save everything
    trainer.save_model_and_artifacts()

    # Plot training history
    trainer.plot_training_history()

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Final Accuracy: {trainer.model_metrics['final_accuracy']:.4f}")
    logger.info(f"Features Used: {trainer.model_metrics['num_features']}")
    logger.info(f"Training Samples: {trainer.model_metrics['training_samples']}")

if __name__ == "__main__":
    main()
