"""
Advanced ASL Model Training Framework
State-of-the-art training system with custom loss functions and optimization.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
from datetime import datetime
import wandb

from ..models.advanced_asl_model import AdvancedASLModel
from ..data.asl_dataset import ASLDataset

logger = logging.getLogger(__name__)

class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance in ASL recognition."""
    
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        
        # Calculate cross entropy
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Calculate focal weight
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, self.gamma)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return tf.reduce_mean(focal_loss)

class LabelSmoothingLoss(keras.losses.Loss):
    """Label smoothing for better generalization."""
    
    def __init__(self, smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing
    
    def call(self, y_true, y_pred):
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            num_classes = tf.shape(y_pred)[-1]
            y_true = tf.one_hot(y_true, depth=num_classes)
        
        # Apply label smoothing
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        smooth_labels = y_true * (1 - self.smoothing) + self.smoothing / num_classes
        
        return tf.keras.losses.categorical_crossentropy(smooth_labels, y_pred)

class CosineWarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing with warmup for stable training."""
    
    def __init__(self, warmup_steps, total_steps, max_lr, min_lr=1e-6):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Warmup phase
        warmup_lr = self.max_lr * step / self.warmup_steps
        
        # Cosine annealing phase
        cosine_steps = step - self.warmup_steps
        cosine_total = self.total_steps - self.warmup_steps
        cosine_lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
            1 + tf.cos(tf.constant(np.pi) * cosine_steps / cosine_total)
        )
        
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

class AdvancedMetrics:
    """Custom metrics for ASL recognition evaluation."""
    
    @staticmethod
    def top_k_accuracy(k=5):
        """Top-k accuracy metric."""
        def metric(y_true, y_pred):
            return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=k)
        metric.__name__ = f'top_{k}_accuracy'
        return metric
    
    @staticmethod
    def per_class_accuracy(num_classes):
        """Per-class accuracy tracking."""
        def metric(y_true, y_pred):
            y_true_class = tf.argmax(y_true, axis=-1)
            y_pred_class = tf.argmax(y_pred, axis=-1)
            
            accuracies = []
            for i in range(num_classes):
                class_mask = tf.equal(y_true_class, i)
                if tf.reduce_sum(tf.cast(class_mask, tf.float32)) > 0:
                    class_acc = tf.reduce_mean(
                        tf.cast(tf.equal(y_true_class, y_pred_class), tf.float32)[class_mask]
                    )
                    accuracies.append(class_acc)
            
            return tf.reduce_mean(accuracies) if accuracies else 0.0
        
        metric.__name__ = 'per_class_accuracy'
        return metric

class ASLTrainer:
    """Advanced training framework for ASL recognition models."""
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 data_config: Dict[str, Any]):
        
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Initialize model
        self.model = AdvancedASLModel(**model_config)
        
        # Initialize dataset
        self.dataset = ASLDataset(**data_config)
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.training_history = []
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup advanced logging and monitoring."""
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(f"experiments/asl_training_{timestamp}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        log_file = self.experiment_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Initialize Weights & Biases if available
        try:
            wandb.init(
                project="advanced-asl-recognition",
                config={**self.model_config, **self.training_config, **self.data_config},
                dir=str(self.experiment_dir)
            )
            self.use_wandb = True
            logger.info("Weights & Biases initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False
    
    def create_callbacks(self) -> List[callbacks.Callback]:
        """Create advanced training callbacks."""
        callback_list = []
        
        # Model checkpointing
        checkpoint_path = self.experiment_dir / "checkpoints" / "model_{epoch:03d}_{val_accuracy:.4f}.h5"
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callback_list.append(checkpoint_callback)
        
        # Early stopping with patience
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.training_config.get('early_stopping_patience', 15),
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # CSV logging
        csv_logger = callbacks.CSVLogger(
            str(self.experiment_dir / "training_log.csv"),
            append=True
        )
        callback_list.append(csv_logger)
        
        # Custom callback for advanced metrics
        class AdvancedMetricsCallback(callbacks.Callback):
            def __init__(self, trainer):
                super().__init__()
                self.trainer = trainer
            
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                
                # Log to W&B if available
                if self.trainer.use_wandb:
                    wandb.log(logs, step=epoch)
                
                # Save training state
                self.trainer.current_epoch = epoch
                if logs.get('val_accuracy', 0) > self.trainer.best_val_accuracy:
                    self.trainer.best_val_accuracy = logs['val_accuracy']
                
                # Save training history
                self.trainer.training_history.append({
                    'epoch': epoch,
                    'timestamp': datetime.now().isoformat(),
                    **logs
                })
                
                # Save history to file
                history_file = self.trainer.experiment_dir / "training_history.json"
                with open(history_file, 'w') as f:
                    json.dump(self.trainer.training_history, f, indent=2)
        
        callback_list.append(AdvancedMetricsCallback(self))
        
        return callback_list
    
    def compile_model(self):
        """Compile model with advanced optimization."""
        # Learning rate schedule
        total_steps = len(self.dataset.train_samples) // self.data_config['batch_size'] * self.training_config['epochs']
        warmup_steps = total_steps // 10  # 10% warmup
        
        lr_schedule = CosineWarmupSchedule(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            max_lr=self.training_config['learning_rate'],
            min_lr=self.training_config['learning_rate'] / 100
        )
        
        # Advanced optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.training_config.get('weight_decay', 1e-4),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Loss function
        if self.training_config.get('use_focal_loss', False):
            loss = FocalLoss(alpha=0.25, gamma=2.0)
        elif self.training_config.get('use_label_smoothing', False):
            loss = LabelSmoothingLoss(smoothing=0.1)
        else:
            loss = 'sparse_categorical_crossentropy'
        
        # Metrics
        metrics = [
            'accuracy',
            AdvancedMetrics.top_k_accuracy(k=5),
            AdvancedMetrics.per_class_accuracy(self.model_config['num_classes']),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
        
        # Compile model
        compiled_model = self.model.compile_model(self.training_config['learning_rate'])
        compiled_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return compiled_model
    
    def train(self):
        """Execute complete training pipeline."""
        logger.info("Starting advanced ASL model training...")
        
        # Compile model
        compiled_model = self.compile_model()
        
        # Print model summary
        logger.info("Model Architecture:")
        compiled_model.summary(print_fn=logger.info)
        
        # Create datasets
        train_dataset = self.dataset.create_tf_dataset(training=True)
        val_dataset = self.dataset.create_tf_dataset(training=False)
        
        # Calculate class weights
        class_weights = self.dataset.get_class_weights()
        
        # Create callbacks
        callbacks_list = self.create_callbacks()
        
        # Start training
        start_time = time.time()
        
        history = compiled_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.training_config['epochs'],
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Save final model
        final_model_path = self.experiment_dir / "final_model.h5"
        compiled_model.save(str(final_model_path))
        
        # Save training summary
        summary = {
            'training_time_seconds': training_time,
            'best_val_accuracy': self.best_val_accuracy,
            'total_epochs': self.current_epoch + 1,
            'model_parameters': compiled_model.count_params(),
            'final_model_path': str(final_model_path),
            'experiment_dir': str(self.experiment_dir)
        }
        
        summary_file = self.experiment_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        logger.info(f"Model saved to: {final_model_path}")
        
        if self.use_wandb:
            wandb.finish()
        
        return history, summary
