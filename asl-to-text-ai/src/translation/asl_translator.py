"""
Main ASL Translation Engine for converting sign language to text.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from collections import deque
import json
from pathlib import Path

from ..models.asl_detector import ASLDetector
from ..utils.config import TRANSLATION_CONFIG, MODEL_CONFIG
from .vocabulary import ASLVocabulary
from .grammar_processor import GrammarProcessor

logger = logging.getLogger(__name__)

class ASLTranslator:
    """
    Main translation engine that coordinates ASL detection, vocabulary lookup,
    and grammar processing to produce coherent English text.
    """
    
    def __init__(self, model_path: Optional[str] = None, vocabulary_path: Optional[str] = None):
        """
        Initialize the ASL translator.
        
        Args:
            model_path: Path to the trained ASL detection model
            vocabulary_path: Path to the ASL vocabulary database
        """
        # Initialize components
        self.detector = ASLDetector(model_path)
        self.vocabulary = ASLVocabulary(vocabulary_path)
        self.grammar_processor = GrammarProcessor()
        
        # Translation configuration
        self.max_sentence_length = TRANSLATION_CONFIG["max_sentence_length"]
        self.context_window = TRANSLATION_CONFIG["context_window"]
        self.enable_grammar = TRANSLATION_CONFIG["grammar_rules"]
        
        # Translation state
        self.current_sentence = []
        self.word_buffer = deque(maxlen=self.context_window)
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.5  # Minimum time between predictions (seconds)
        
        # Performance tracking
        self.translation_stats = {
            "total_signs_processed": 0,
            "successful_translations": 0,
            "average_confidence": 0.0,
            "processing_times": deque(maxlen=100)
        }
        
        logger.info("ASL Translator initialized successfully")
    
    def translate_frame(self, frame: np.ndarray, timestamp: float = None) -> Dict[str, Any]:
        """
        Translate a single video frame to text.
        
        Args:
            frame: Input video frame
            timestamp: Frame timestamp
            
        Returns:
            Dictionary containing translation results
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = time.time()
        
        # Process frame through detector
        detection_result = self.detector.process_frame(frame, timestamp)
        
        translation_result = {
            "text": "",
            "confidence": 0.0,
            "word_added": False,
            "sentence_complete": False,
            "processing_time": 0.0,
            "detection_ready": detection_result["ready_for_prediction"],
            "buffer_size": detection_result["buffer_size"]
        }
        
        # Check if we can make a prediction
        if detection_result["ready_for_prediction"]:
            # Apply prediction cooldown to avoid rapid-fire predictions
            current_time = time.time()
            if current_time - self.last_prediction_time >= self.prediction_cooldown:
                
                # Get sign prediction
                prediction = self.detector.predict_sign()
                
                if prediction and prediction["is_confident"]:
                    # Look up word in vocabulary
                    word_info = self.vocabulary.get_word(prediction["predicted_class"])
                    
                    if word_info:
                        # Add word to current translation
                        word_result = self._add_word_to_translation(
                            word_info, 
                            prediction["confidence"],
                            timestamp
                        )
                        
                        translation_result.update(word_result)
                        self.last_prediction_time = current_time
                        
                        # Update statistics
                        self._update_stats(prediction["confidence"], start_time)
        
        translation_result["processing_time"] = time.time() - start_time
        return translation_result
    
    def translate_video_sequence(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Translate a sequence of video frames (for pre-recorded videos).
        
        Args:
            frames: List of video frames
            
        Returns:
            Complete translation results
        """
        logger.info(f"Starting translation of {len(frames)} frames")
        
        # Reset state for new video
        self.reset_translation_state()
        
        all_words = []
        frame_results = []
        
        for i, frame in enumerate(frames):
            timestamp = i / 30.0  # Assume 30 FPS
            result = self.translate_frame(frame, timestamp)
            frame_results.append(result)
            
            if result["word_added"]:
                all_words.append({
                    "word": result["text"],
                    "confidence": result["confidence"],
                    "timestamp": timestamp
                })
        
        # Process complete sentence with grammar rules
        if all_words and self.enable_grammar:
            word_sequence = [w["word"] for w in all_words]
            processed_text = self.grammar_processor.process_sentence(word_sequence)
        else:
            processed_text = " ".join([w["word"] for w in all_words])
        
        return {
            "final_text": processed_text,
            "words": all_words,
            "frame_results": frame_results,
            "total_frames": len(frames),
            "processing_stats": self.get_translation_stats()
        }
    
    def _add_word_to_translation(self, word_info: Dict[str, Any], 
                               confidence: float, timestamp: float) -> Dict[str, Any]:
        """
        Add a detected word to the current translation.
        
        Args:
            word_info: Word information from vocabulary
            confidence: Detection confidence
            timestamp: Word timestamp
            
        Returns:
            Dictionary with word addition results
        """
        word = word_info["word"]
        
        # Check for duplicate consecutive words
        if self.word_buffer and self.word_buffer[-1]["word"] == word:
            # Skip duplicate word within short time window
            time_diff = timestamp - self.word_buffer[-1]["timestamp"]
            if time_diff < 2.0:  # 2 second window
                return {
                    "text": "",
                    "confidence": confidence,
                    "word_added": False,
                    "sentence_complete": False
                }
        
        # Add word to buffers
        word_entry = {
            "word": word,
            "confidence": confidence,
            "timestamp": timestamp,
            "category": word_info.get("category", "unknown")
        }
        
        self.word_buffer.append(word_entry)
        self.current_sentence.append(word_entry)
        
        # Check for sentence completion
        sentence_complete = self._check_sentence_completion(word_info)
        
        result = {
            "text": word,
            "confidence": confidence,
            "word_added": True,
            "sentence_complete": sentence_complete,
            "word_info": word_info
        }
        
        # If sentence is complete, process with grammar rules
        if sentence_complete and self.enable_grammar:
            sentence_words = [w["word"] for w in self.current_sentence]
            processed_sentence = self.grammar_processor.process_sentence(sentence_words)
            result["processed_sentence"] = processed_sentence
            
            # Reset for next sentence
            self.current_sentence = []
        
        return result
    
    def _check_sentence_completion(self, word_info: Dict[str, Any]) -> bool:
        """
        Check if the current sentence is complete.
        
        Args:
            word_info: Information about the last added word
            
        Returns:
            True if sentence is complete
        """
        # Check for sentence-ending indicators
        if word_info.get("category") == "punctuation":
            return True
        
        # Check for sentence length
        if len(self.current_sentence) >= self.max_sentence_length:
            return True
        
        # Check for pause in signing (would be implemented with timing logic)
        # This is a simplified version
        return False
    
    def _update_stats(self, confidence: float, start_time: float):
        """Update translation performance statistics."""
        self.translation_stats["total_signs_processed"] += 1
        self.translation_stats["successful_translations"] += 1
        
        # Update average confidence
        total = self.translation_stats["total_signs_processed"]
        current_avg = self.translation_stats["average_confidence"]
        self.translation_stats["average_confidence"] = (
            (current_avg * (total - 1) + confidence) / total
        )
        
        # Track processing time
        processing_time = time.time() - start_time
        self.translation_stats["processing_times"].append(processing_time)
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get current translation performance statistics."""
        processing_times = list(self.translation_stats["processing_times"])
        
        return {
            "total_signs_processed": self.translation_stats["total_signs_processed"],
            "successful_translations": self.translation_stats["successful_translations"],
            "success_rate": (
                self.translation_stats["successful_translations"] / 
                max(1, self.translation_stats["total_signs_processed"])
            ),
            "average_confidence": self.translation_stats["average_confidence"],
            "average_processing_time": np.mean(processing_times) if processing_times else 0,
            "max_processing_time": np.max(processing_times) if processing_times else 0,
            "min_processing_time": np.min(processing_times) if processing_times else 0
        }
    
    def get_current_sentence(self) -> str:
        """Get the current sentence being built."""
        if not self.current_sentence:
            return ""
        
        words = [w["word"] for w in self.current_sentence]
        if self.enable_grammar:
            return self.grammar_processor.process_sentence(words)
        else:
            return " ".join(words)
    
    def reset_translation_state(self):
        """Reset the translation state for a new session."""
        self.current_sentence = []
        self.word_buffer.clear()
        self.detector.reset_buffer()
        self.last_prediction_time = 0
        
        logger.info("Translation state reset")
    
    def set_vocabulary(self, vocabulary_path: str) -> bool:
        """
        Load a new vocabulary database.
        
        Args:
            vocabulary_path: Path to vocabulary file
            
        Returns:
            True if vocabulary loaded successfully
        """
        try:
            self.vocabulary = ASLVocabulary(vocabulary_path)
            logger.info(f"Vocabulary updated from {vocabulary_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load vocabulary from {vocabulary_path}: {e}")
            return False
    
    def set_model(self, model_path: str) -> bool:
        """
        Load a new ASL detection model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if model loaded successfully
        """
        return self.detector.load_model(model_path)
