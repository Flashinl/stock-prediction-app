"""
Advanced ASL Dataset Pipeline
High-performance data loading and augmentation for ASL model training.
"""

import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from typing import List, Tuple, Dict, Any, Optional
import logging
import json
import os
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor
import albumentations as A

logger = logging.getLogger(__name__)

class ASLDataAugmentation:
    """Advanced data augmentation specifically designed for ASL videos."""
    
    def __init__(self):
        # Video-specific augmentations
        self.spatial_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        ])
        
        # Geometric augmentations (careful with hand positions)
        self.geometric_transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),
        ])
    
    def augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single frame."""
        # Apply spatial augmentations
        frame = self.spatial_transforms(image=frame)['image']
        
        # Apply geometric augmentations with probability
        if random.random() < 0.3:
            frame = self.geometric_transforms(image=frame)['image']
        
        return frame
    
    def augment_landmarks(self, landmarks: np.ndarray, augment_params: Dict) -> np.ndarray:
        """Apply consistent augmentation to landmarks based on frame augmentation."""
        # Apply same geometric transformations to landmarks
        # This ensures consistency between frames and landmarks
        if 'rotation' in augment_params:
            # Apply rotation to landmarks
            angle = augment_params['rotation']
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Rotate x,y coordinates (keep z unchanged)
            xy_coords = landmarks[:, :2]
            rotated_xy = np.dot(xy_coords, rotation_matrix.T)
            landmarks[:, :2] = rotated_xy
        
        return landmarks

class MediaPipeProcessor:
    """Optimized MediaPipe processing for landmark extraction."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        
        # Initialize with optimized settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all landmarks from a frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract hand landmarks
        hand_results = self.hands.process(rgb_frame)
        hand_landmarks = np.zeros((21, 3))  # Default to zeros if no hands detected
        
        if hand_results.multi_hand_landmarks:
            # Use the first detected hand (can be extended for two hands)
            landmarks = hand_results.multi_hand_landmarks[0]
            for i, landmark in enumerate(landmarks.landmark):
                hand_landmarks[i] = [landmark.x, landmark.y, landmark.z]
        
        # Extract pose landmarks
        pose_results = self.pose.process(rgb_frame)
        pose_landmarks = np.zeros((33, 3))  # Default to zeros if no pose detected
        
        if pose_results.pose_landmarks:
            for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                pose_landmarks[i] = [landmark.x, landmark.y, landmark.z]
        
        return {
            'hand_landmarks': hand_landmarks.flatten(),  # Shape: (63,)
            'pose_landmarks': pose_landmarks.flatten()   # Shape: (99,)
        }

class ASLVideoProcessor:
    """High-performance video processing for ASL datasets."""
    
    def __init__(self, sequence_length: int = 30, target_size: Tuple[int, int] = (224, 224)):
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.mp_processor = MediaPipeProcessor()
        self.augmentation = ASLDataAugmentation()
    
    def load_video(self, video_path: str) -> Optional[np.ndarray]:
        """Load video and extract frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, self.target_size)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            logger.error(f"No frames extracted from video: {video_path}")
            return None
        
        return np.array(frames)
    
    def sample_frames(self, frames: np.ndarray, training: bool = True) -> np.ndarray:
        """Sample frames to match sequence length."""
        num_frames = len(frames)
        
        if num_frames >= self.sequence_length:
            if training:
                # Random sampling during training
                start_idx = random.randint(0, num_frames - self.sequence_length)
                return frames[start_idx:start_idx + self.sequence_length]
            else:
                # Center sampling during inference
                start_idx = (num_frames - self.sequence_length) // 2
                return frames[start_idx:start_idx + self.sequence_length]
        else:
            # Pad or repeat frames if video is too short
            if training:
                # Random padding/repetition
                indices = np.random.choice(num_frames, self.sequence_length, replace=True)
            else:
                # Linear interpolation for consistent inference
                indices = np.linspace(0, num_frames - 1, self.sequence_length, dtype=int)
            
            return frames[indices]
    
    def process_video(self, video_path: str, label: int, training: bool = True) -> Optional[Dict[str, np.ndarray]]:
        """Complete video processing pipeline."""
        # Load video
        frames = self.load_video(video_path)
        if frames is None:
            return None
        
        # Sample frames
        sampled_frames = self.sample_frames(frames, training)
        
        # Process each frame
        processed_frames = []
        hand_landmarks_seq = []
        pose_landmarks_seq = []
        
        for frame in sampled_frames:
            # Extract landmarks first (before augmentation)
            landmarks = self.mp_processor.extract_landmarks(frame)
            
            # Apply augmentation during training
            if training:
                frame = self.augmentation.augment_frame(frame)
            
            # Normalize frame
            frame = frame.astype(np.float32) / 255.0
            
            processed_frames.append(frame)
            hand_landmarks_seq.append(landmarks['hand_landmarks'])
            pose_landmarks_seq.append(landmarks['pose_landmarks'])
        
        return {
            'video_frames': np.array(processed_frames),
            'hand_landmarks': np.array(hand_landmarks_seq),
            'pose_landmarks': np.array(pose_landmarks_seq),
            'label': label
        }

class ASLDataset:
    """High-performance ASL dataset with advanced preprocessing."""
    
    def __init__(self, 
                 data_dir: str,
                 sequence_length: int = 30,
                 batch_size: int = 8,
                 num_classes: int = 1000,
                 validation_split: float = 0.2):
        
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.validation_split = validation_split
        
        self.processor = ASLVideoProcessor(sequence_length)
        
        # Load dataset metadata
        self.train_samples = []
        self.val_samples = []
        self._load_dataset_info()
    
    def _load_dataset_info(self):
        """Load dataset information and create train/val splits."""
        # This would be customized based on your dataset structure
        # Example for WLASL-style dataset
        
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Create train/val split
            all_samples = []
            for item in metadata:
                video_path = self.data_dir / item['video_path']
                if video_path.exists():
                    all_samples.append({
                        'video_path': str(video_path),
                        'label': item['label'],
                        'word': item.get('word', '')
                    })
            
            # Shuffle and split
            random.shuffle(all_samples)
            split_idx = int(len(all_samples) * (1 - self.validation_split))
            self.train_samples = all_samples[:split_idx]
            self.val_samples = all_samples[split_idx:]
            
            logger.info(f"Dataset loaded: {len(self.train_samples)} train, {len(self.val_samples)} val samples")
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
    
    def create_tf_dataset(self, training: bool = True) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset."""
        samples = self.train_samples if training else self.val_samples
        
        def generator():
            for sample in samples:
                processed = self.processor.process_video(
                    sample['video_path'], 
                    sample['label'], 
                    training=training
                )
                if processed is not None:
                    yield (
                        processed['video_frames'],
                        processed['hand_landmarks'],
                        processed['pose_landmarks']
                    ), processed['label']
        
        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.sequence_length, 224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.sequence_length, 63), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.sequence_length, 99), dtype=tf.float32)
                ),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        # Optimize dataset
        if training:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.EXPERIMENTAL_AUTOTUNE)
        
        return dataset
    
    def get_class_weights(self) -> Dict[int, float]:
        """Calculate class weights for imbalanced dataset."""
        label_counts = {}
        for sample in self.train_samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total_samples = len(self.train_samples)
        class_weights = {}
        
        for label, count in label_counts.items():
            class_weights[label] = total_samples / (len(label_counts) * count)
        
        return class_weights
