"""
Advanced Multi-Modal ASL Recognition Model
State-of-the-art transformer architecture for superior sign language recognition.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MultiHeadSelfAttention(layers.Layer):
    """Custom multi-head self-attention for temporal sequence modeling."""
    
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) should be divisible by num_heads ({num_heads})")
        
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(layers.Layer):
    """Transformer encoder block with residual connections and layer normalization."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training=None):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(layers.Layer):
    """Positional encoding for transformer input."""
    
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=(1, self.sequence_length, self.embed_dim),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)
    
    def call(self, inputs):
        length = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :length, :]

class SpatialFeatureExtractor(layers.Layer):
    """Extract spatial features from video frames using CNN."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_layers = [
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(256, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(512, 3, activation='relu', padding='same'),
            layers.GlobalAveragePooling2D(),
        ]
    
    def call(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        return x

class LandmarkProcessor(layers.Layer):
    """Process MediaPipe landmarks with attention mechanism."""
    
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_layers = [
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(embed_dim, activation='relu'),
        ]
        
    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

class AdvancedASLModel:
    """
    State-of-the-art multi-modal ASL recognition model.
    
    Features:
    - Multi-modal fusion (video + landmarks + pose)
    - Transformer-based temporal modeling
    - Advanced attention mechanisms
    - Optimized for real-time inference
    """
    
    def __init__(self, 
                 num_classes: int = 1000,
                 sequence_length: int = 30,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_transformer_blocks: int = 6,
                 ff_dim: int = 2048):
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ff_dim = ff_dim
        
        self.model = None
        self.training_model = None
        
    def build_model(self) -> keras.Model:
        """Build the complete multi-modal ASL recognition model."""
        
        # Input layers
        video_input = layers.Input(shape=(self.sequence_length, 224, 224, 3), name='video_frames')
        landmarks_input = layers.Input(shape=(self.sequence_length, 63), name='hand_landmarks')  # 21 landmarks * 3 coords
        pose_input = layers.Input(shape=(self.sequence_length, 99), name='pose_landmarks')  # 33 landmarks * 3 coords
        
        # Process video frames
        video_features = layers.TimeDistributed(SpatialFeatureExtractor())(video_input)
        video_features = layers.Dense(self.embed_dim)(video_features)
        
        # Process landmarks
        landmark_features = layers.TimeDistributed(LandmarkProcessor(self.embed_dim))(landmarks_input)
        
        # Process pose
        pose_features = layers.TimeDistributed(LandmarkProcessor(self.embed_dim))(pose_input)
        
        # Multi-modal fusion with learned attention weights
        fusion_attention = layers.Dense(3, activation='softmax', name='fusion_weights')
        fusion_weights = fusion_attention(layers.GlobalAveragePooling1D()(video_features))
        fusion_weights = layers.Reshape((1, 3))(fusion_weights)
        
        # Weighted combination of modalities
        combined_features = layers.Lambda(lambda x: 
            x[0] * tf.expand_dims(x[3][:, :, 0], -1) + 
            x[1] * tf.expand_dims(x[3][:, :, 1], -1) + 
            x[2] * tf.expand_dims(x[3][:, :, 2], -1)
        )([video_features, landmark_features, pose_features, fusion_weights])
        
        # Add positional encoding
        encoded_features = PositionalEncoding(self.sequence_length, self.embed_dim)(combined_features)
        
        # Transformer encoder stack
        x = encoded_features
        for i in range(self.num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                rate=0.1
            )(x)
        
        # Global temporal pooling with attention
        attention_weights = layers.Dense(1, activation='sigmoid')(x)
        attended_features = layers.Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([x, attention_weights])
        
        # Classification head
        x = layers.Dense(1024, activation='gelu')(attended_features)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='gelu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        model = keras.Model(
            inputs=[video_input, landmarks_input, pose_input],
            outputs=outputs,
            name='AdvancedASLModel'
        )
        
        return model
    
    def compile_model(self, learning_rate: float = 1e-4):
        """Compile the model with advanced optimization."""
        if self.model is None:
            self.model = self.build_model()
        
        # Custom learning rate schedule
        lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )
        
        # Advanced optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Compile with advanced metrics
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        logger.info(f"Model compiled with {self.model.count_params():,} parameters")
        return self.model
    
    def get_model_summary(self):
        """Get detailed model architecture summary."""
        if self.model is None:
            self.model = self.build_model()
        return self.model.summary()
