"""
Core ASL Detection Model for real-time sign language recognition.
"""

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

from ..utils.config import MODEL_CONFIG, REALTIME_CONFIG

logger = logging.getLogger(__name__)

class ASLDetector:
    """
    Advanced ASL detection model with real-time processing capabilities.
    
    This class handles:
    - Hand and pose detection using MediaPipe
    - Feature extraction from video frames
    - Sign classification using deep learning models
    - Real-time processing with minimal latency
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ASL detector.
        
        Args:
            model_path: Path to pre-trained model file
        """
        self.model_path = model_path
        self.confidence_threshold = MODEL_CONFIG["confidence_threshold"]
        self.sequence_length = MODEL_CONFIG["sequence_length"]
        
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        
        # Initialize hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize face mesh for facial expressions
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Frame buffer for sequence processing
        self.frame_buffer = []
        self.feature_buffer = []
        
        # Load the classification model
        self.classification_model = None
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        logger.info("ASL Detector initialized successfully")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a pre-trained ASL classification model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            self.classification_model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def extract_landmarks(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract hand, pose, and facial landmarks from a video frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Dictionary containing extracted landmarks and metadata
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        
        # Process face
        face_results = self.face_mesh.process(rgb_frame)
        
        landmarks = {
            "hands": [],
            "pose": None,
            "face": None,
            "frame_shape": frame.shape,
            "timestamp": None  # Will be set by caller
        }
        
        # Extract hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    hand_points.append([landmark.x, landmark.y, landmark.z])
                landmarks["hands"].append(np.array(hand_points))
        
        # Extract pose landmarks
        if pose_results.pose_landmarks:
            pose_points = []
            for landmark in pose_results.pose_landmarks.landmark:
                pose_points.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            landmarks["pose"] = np.array(pose_points)
        
        # Extract facial landmarks (key points for expressions)
        if face_results.multi_face_landmarks:
            face_points = []
            for face_landmarks in face_results.multi_face_landmarks:
                # Extract only key facial expression points to reduce dimensionality
                key_indices = [10, 151, 9, 8, 168, 6, 297, 334, 296, 336]  # Key facial points
                for idx in key_indices:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        face_points.append([landmark.x, landmark.y, landmark.z])
            if face_points:
                landmarks["face"] = np.array(face_points)
        
        return landmarks
    
    def normalize_landmarks(self, landmarks: Dict[str, Any]) -> np.ndarray:
        """
        Normalize and flatten landmarks into a feature vector.
        
        Args:
            landmarks: Dictionary of extracted landmarks
            
        Returns:
            Normalized feature vector
        """
        features = []
        
        # Process hand landmarks
        for hand in landmarks["hands"]:
            if hand is not None and len(hand) > 0:
                # Normalize relative to wrist (first landmark)
                if len(hand) >= 21:  # Standard MediaPipe hand has 21 landmarks
                    wrist = hand[0]
                    normalized_hand = hand - wrist
                    features.extend(normalized_hand.flatten())
                else:
                    # Pad if insufficient landmarks
                    features.extend([0.0] * (21 * 3))
            else:
                # No hand detected, pad with zeros
                features.extend([0.0] * (21 * 3))
        
        # Ensure we have features for both hands
        while len(landmarks["hands"]) < 2:
            features.extend([0.0] * (21 * 3))
        
        # Process pose landmarks
        if landmarks["pose"] is not None and len(landmarks["pose"]) > 0:
            # Use upper body landmarks (shoulders, elbows, wrists)
            upper_body_indices = [11, 12, 13, 14, 15, 16]  # Key upper body points
            pose_features = []
            for idx in upper_body_indices:
                if idx < len(landmarks["pose"]):
                    pose_features.extend(landmarks["pose"][idx][:3])  # x, y, z only
                else:
                    pose_features.extend([0.0, 0.0, 0.0])
            features.extend(pose_features)
        else:
            # No pose detected, pad with zeros
            features.extend([0.0] * (6 * 3))
        
        # Process facial landmarks
        if landmarks["face"] is not None and len(landmarks["face"]) > 0:
            features.extend(landmarks["face"].flatten())
        else:
            # No face detected, pad with zeros
            features.extend([0.0] * (10 * 3))  # 10 key facial points
        
        return np.array(features, dtype=np.float32)
    
    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> Dict[str, Any]:
        """
        Process a single video frame and extract ASL features.
        
        Args:
            frame: Input video frame
            timestamp: Frame timestamp
            
        Returns:
            Dictionary containing processed features and metadata
        """
        # Extract landmarks
        landmarks = self.extract_landmarks(frame)
        landmarks["timestamp"] = timestamp
        
        # Normalize features
        features = self.normalize_landmarks(landmarks)
        
        # Add to buffer
        self.feature_buffer.append(features)
        
        # Maintain buffer size
        if len(self.feature_buffer) > self.sequence_length:
            self.feature_buffer.pop(0)
        
        return {
            "features": features,
            "landmarks": landmarks,
            "buffer_size": len(self.feature_buffer),
            "ready_for_prediction": len(self.feature_buffer) >= self.sequence_length
        }
    
    def predict_sign(self) -> Optional[Dict[str, Any]]:
        """
        Predict ASL sign from the current feature buffer.
        
        Returns:
            Dictionary containing prediction results or None if not ready
        """
        if not self.classification_model:
            logger.warning("No classification model loaded")
            return None
        
        if len(self.feature_buffer) < self.sequence_length:
            return None
        
        try:
            # Prepare sequence for prediction
            sequence = np.array(self.feature_buffer[-self.sequence_length:])
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            
            # Make prediction
            predictions = self.classification_model.predict(sequence, verbose=0)
            
            # Get top prediction
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get top 3 predictions for context
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {
                    "class_id": int(idx),
                    "confidence": float(predictions[0][idx])
                }
                for idx in top_3_indices
            ]
            
            return {
                "predicted_class": int(predicted_class),
                "confidence": confidence,
                "top_predictions": top_3_predictions,
                "is_confident": confidence >= self.confidence_threshold,
                "timestamp": self.feature_buffer[-1] if self.feature_buffer else None
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def reset_buffer(self):
        """Reset the feature buffer."""
        self.feature_buffer.clear()
        logger.debug("Feature buffer reset")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
