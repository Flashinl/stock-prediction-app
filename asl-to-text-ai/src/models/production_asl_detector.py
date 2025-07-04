"""
Production-Ready Advanced ASL Detector
Integrates the advanced transformer model with real-time inference capabilities.
"""

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import json
import time
from collections import deque

from ..utils.config import MODEL_CONFIG, REALTIME_CONFIG

logger = logging.getLogger(__name__)

class ProductionASLDetector:
    """
    Production-ready ASL detector with advanced transformer model.
    
    Features:
    - Real-time inference with optimized performance
    - Multi-modal input processing (video + landmarks + pose)
    - Confidence-based prediction filtering
    - Temporal smoothing for stable predictions
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the production ASL detector.
        
        Args:
            model_path: Path to trained advanced model
        """
        self.model_path = model_path
        self.confidence_threshold = MODEL_CONFIG.get("confidence_threshold", 0.7)
        self.sequence_length = MODEL_CONFIG.get("sequence_length", 30)
        
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        # Optimized for real-time performance
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        # Buffers for sequence processing
        self.frame_buffer = deque(maxlen=self.sequence_length)
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        self.pose_buffer = deque(maxlen=self.sequence_length)
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)
        self.last_prediction = None
        self.prediction_confidence = 0.0
        
        # Load model and vocabulary
        self.model = None
        self.vocabulary = []
        self.demo_mode = True
        
        if model_path and Path(model_path).exists():
            self.load_advanced_model(model_path)
        else:
            self.setup_demo_mode()
        
        logger.info(f"Production ASL Detector initialized ({'Advanced Model' if not self.demo_mode else 'Demo Mode'})")
    
    def setup_demo_mode(self):
        """Setup demo mode with basic gesture recognition."""
        self.demo_mode = True
        self.vocabulary = [
            "hello", "goodbye", "thank_you", "please", "yes", "no",
            "good", "bad", "happy", "sad", "love", "help",
            "water", "food", "home", "work", "family", "friend",
            "morning", "evening", "today", "tomorrow", "yesterday",
            "big", "small", "hot", "cold", "fast", "slow"
        ]
        
        # Simple gesture patterns for demo
        self.demo_patterns = {
            "hello": {"hand_movement": "wave", "confidence": 0.85},
            "goodbye": {"hand_movement": "wave_away", "confidence": 0.80},
            "thank_you": {"hand_movement": "touch_chin", "confidence": 0.75},
            "yes": {"head_movement": "nod", "confidence": 0.70},
            "no": {"head_movement": "shake", "confidence": 0.70}
        }
        
        logger.info("Demo mode initialized with basic gesture patterns")
    
    def load_advanced_model(self, model_path: str) -> bool:
        """
        Load the advanced transformer model.
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # Load the advanced model
            self.model = tf.keras.models.load_model(model_path, compile=False)
            
            # Load vocabulary if available
            vocab_path = Path(model_path).parent / "vocabulary.json"
            if vocab_path.exists():
                with open(vocab_path, 'r') as f:
                    vocab_data = json.load(f)
                    self.vocabulary = vocab_data.get('words', [])
            else:
                # Default vocabulary for 1000 classes
                self.vocabulary = [f"sign_{i:04d}" for i in range(1000)]
            
            self.demo_mode = False
            logger.info(f"Advanced model loaded: {len(self.vocabulary)} classes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load advanced model: {e}")
            self.setup_demo_mode()
            return False
    
    def extract_landmarks(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract hand and pose landmarks from frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary containing landmark data
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract hand landmarks
        hand_results = self.hands.process(rgb_frame)
        hand_landmarks = np.zeros((21, 3))
        
        if hand_results.multi_hand_landmarks:
            # Use primary hand (first detected)
            landmarks = hand_results.multi_hand_landmarks[0]
            for i, landmark in enumerate(landmarks.landmark):
                hand_landmarks[i] = [landmark.x, landmark.y, landmark.z]
        
        # Extract pose landmarks
        pose_results = self.pose.process(rgb_frame)
        pose_landmarks = np.zeros((33, 3))
        
        if pose_results.pose_landmarks:
            for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                pose_landmarks[i] = [landmark.x, landmark.y, landmark.z]
        
        return {
            'hand_landmarks': hand_landmarks.flatten(),  # Shape: (63,)
            'pose_landmarks': pose_landmarks.flatten(),  # Shape: (99,)
            'hand_detected': hand_results.multi_hand_landmarks is not None,
            'pose_detected': pose_results.pose_landmarks is not None
        }
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input.
        
        Args:
            frame: Raw video frame
            
        Returns:
            Preprocessed frame
        """
        # Resize to model input size
        frame = cv2.resize(frame, (224, 224))
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def demo_prediction(self, landmarks_data: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """
        Simple demo prediction based on basic patterns.
        
        Args:
            landmarks_data: Extracted landmark data
            
        Returns:
            Tuple of (predicted_word, confidence)
        """
        if not landmarks_data['hand_detected']:
            return "no_gesture", 0.0
        
        # Simple heuristic-based recognition for demo
        hand_landmarks = landmarks_data['hand_landmarks'].reshape(21, 3)
        
        # Calculate hand center
        hand_center = np.mean(hand_landmarks[:, :2], axis=0)
        
        # Simple gesture classification based on hand position and movement
        if len(self.landmark_buffer) > 5:
            # Calculate movement
            prev_landmarks = list(self.landmark_buffer)[-5].reshape(21, 3)
            prev_center = np.mean(prev_landmarks[:, :2], axis=0)
            movement = np.linalg.norm(hand_center - prev_center)
            
            # Basic pattern matching
            if movement > 0.1:  # Significant movement
                if hand_center[1] < 0.5:  # Upper part of frame
                    return "hello", 0.75
                else:
                    return "goodbye", 0.70
            else:  # Static gesture
                if hand_center[0] < 0.3:  # Left side
                    return "yes", 0.65
                elif hand_center[0] > 0.7:  # Right side
                    return "no", 0.65
                else:  # Center
                    return "thank_you", 0.60
        
        return "processing", 0.50
    
    def advanced_prediction(self, video_sequence: np.ndarray, 
                          landmark_sequence: np.ndarray, 
                          pose_sequence: np.ndarray) -> Tuple[str, float]:
        """
        Advanced prediction using transformer model.
        
        Args:
            video_sequence: Sequence of video frames
            landmark_sequence: Sequence of hand landmarks
            pose_sequence: Sequence of pose landmarks
            
        Returns:
            Tuple of (predicted_word, confidence)
        """
        try:
            # Prepare input for model
            video_input = np.expand_dims(video_sequence, axis=0)
            landmark_input = np.expand_dims(landmark_sequence, axis=0)
            pose_input = np.expand_dims(pose_sequence, axis=0)
            
            # Run inference
            predictions = self.model.predict([video_input, landmark_input, pose_input], verbose=0)
            
            # Get top prediction
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Map to vocabulary
            if predicted_class < len(self.vocabulary):
                predicted_word = self.vocabulary[predicted_class]
            else:
                predicted_word = f"unknown_{predicted_class}"
            
            return predicted_word, confidence
            
        except Exception as e:
            logger.error(f"Advanced prediction failed: {e}")
            return "error", 0.0
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame and update buffers.
        
        Args:
            frame: Input video frame
            
        Returns:
            Processing results
        """
        # Extract landmarks
        landmarks_data = self.extract_landmarks(frame)
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Update buffers
        self.frame_buffer.append(processed_frame)
        self.landmark_buffer.append(landmarks_data['hand_landmarks'])
        self.pose_buffer.append(landmarks_data['pose_landmarks'])
        
        # Check if we have enough frames for prediction
        can_predict = len(self.frame_buffer) >= self.sequence_length
        
        return {
            'landmarks_detected': landmarks_data['hand_detected'] or landmarks_data['pose_detected'],
            'hand_detected': landmarks_data['hand_detected'],
            'pose_detected': landmarks_data['pose_detected'],
            'can_predict': can_predict,
            'buffer_size': len(self.frame_buffer)
        }
    
    def predict(self) -> Dict[str, Any]:
        """
        Make prediction based on current buffer state.
        
        Returns:
            Prediction results
        """
        if len(self.frame_buffer) < self.sequence_length:
            return {
                'prediction': 'buffering',
                'confidence': 0.0,
                'status': 'insufficient_frames'
            }
        
        try:
            if self.demo_mode:
                # Use demo prediction
                latest_landmarks = {
                    'hand_landmarks': list(self.landmark_buffer)[-1],
                    'hand_detected': np.any(list(self.landmark_buffer)[-1] != 0)
                }
                predicted_word, confidence = self.demo_prediction(latest_landmarks)
            else:
                # Use advanced model
                video_sequence = np.array(list(self.frame_buffer))
                landmark_sequence = np.array(list(self.landmark_buffer))
                pose_sequence = np.array(list(self.pose_buffer))
                
                predicted_word, confidence = self.advanced_prediction(
                    video_sequence, landmark_sequence, pose_sequence
                )
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                predicted_word = "low_confidence"
            
            # Update prediction history for smoothing
            self.prediction_history.append((predicted_word, confidence))
            
            # Smooth predictions
            smoothed_prediction = self.smooth_predictions()
            
            return {
                'prediction': smoothed_prediction['word'],
                'confidence': smoothed_prediction['confidence'],
                'raw_prediction': predicted_word,
                'raw_confidence': confidence,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def smooth_predictions(self) -> Dict[str, Any]:
        """
        Smooth predictions over time for stability.
        
        Returns:
            Smoothed prediction result
        """
        if not self.prediction_history:
            return {'word': 'no_prediction', 'confidence': 0.0}
        
        # Count occurrences of each prediction
        word_counts = {}
        total_confidence = 0.0
        
        for word, conf in self.prediction_history:
            if word not in word_counts:
                word_counts[word] = {'count': 0, 'total_conf': 0.0}
            word_counts[word]['count'] += 1
            word_counts[word]['total_conf'] += conf
            total_confidence += conf
        
        # Find most frequent prediction with highest average confidence
        best_word = None
        best_score = 0.0
        
        for word, data in word_counts.items():
            avg_confidence = data['total_conf'] / data['count']
            frequency_weight = data['count'] / len(self.prediction_history)
            score = avg_confidence * frequency_weight
            
            if score > best_score:
                best_score = score
                best_word = word
        
        avg_confidence = total_confidence / len(self.prediction_history)
        
        return {
            'word': best_word or 'uncertain',
            'confidence': min(best_score, avg_confidence)
        }
    
    def reset_buffers(self):
        """Reset all buffers for new session."""
        self.frame_buffer.clear()
        self.landmark_buffer.clear()
        self.pose_buffer.clear()
        self.prediction_history.clear()
        logger.info("Buffers reset for new session")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current detector status."""
        return {
            'mode': 'demo' if self.demo_mode else 'advanced',
            'model_loaded': self.model is not None,
            'vocabulary_size': len(self.vocabulary),
            'buffer_size': len(self.frame_buffer),
            'sequence_length': self.sequence_length,
            'confidence_threshold': self.confidence_threshold
        }
