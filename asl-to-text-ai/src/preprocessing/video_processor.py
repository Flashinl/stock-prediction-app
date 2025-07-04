"""
Video Processing Pipeline for ASL-to-Text AI system.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Generator
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from ..utils.config import VIDEO_CONFIG

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Handles video processing operations including frame extraction,
    preprocessing, and quality assessment for ASL detection.
    """
    
    def __init__(self):
        """Initialize the video processor."""
        self.supported_formats = VIDEO_CONFIG["supported_formats"]
        self.target_fps = VIDEO_CONFIG["fps"]
        self.max_video_length = VIDEO_CONFIG["max_video_length"]
        self.frame_width = VIDEO_CONFIG["frame_width"]
        self.frame_height = VIDEO_CONFIG["frame_height"]
        self.quality_threshold = VIDEO_CONFIG["quality_threshold"]
        
        # Threading for real-time processing
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        logger.info("Video processor initialized")
    
    def validate_video_file(self, video_path: str) -> Dict[str, Any]:
        """
        Validate video file format, duration, and basic properties.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with validation results
        """
        path = Path(video_path)
        
        validation_result = {
            "valid": False,
            "error": None,
            "duration": 0,
            "fps": 0,
            "frame_count": 0,
            "resolution": (0, 0),
            "file_size": 0
        }
        
        # Check file existence
        if not path.exists():
            validation_result["error"] = "File does not exist"
            return validation_result
        
        # Check file extension
        if path.suffix.lower() not in self.supported_formats:
            validation_result["error"] = f"Unsupported format. Supported: {self.supported_formats}"
            return validation_result
        
        # Get file size
        validation_result["file_size"] = path.stat().st_size
        
        # Check video properties
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            validation_result["error"] = "Could not open video file"
            return validation_result
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = frame_count / fps if fps > 0 else 0
            
            validation_result.update({
                "fps": fps,
                "frame_count": frame_count,
                "resolution": (width, height),
                "duration": duration
            })
            
            # Validate duration
            if duration > self.max_video_length:
                validation_result["error"] = f"Video too long. Max: {self.max_video_length}s"
                return validation_result
            
            # Validate resolution
            if width < 320 or height < 240:
                validation_result["error"] = "Resolution too low. Minimum: 320x240"
                return validation_result
            
            validation_result["valid"] = True
            
        except Exception as e:
            validation_result["error"] = f"Error reading video properties: {str(e)}"
        
        finally:
            cap.release()
        
        return validation_result
    
    def extract_frames(self, video_path: str, 
                      max_frames: Optional[int] = None,
                      skip_frames: int = 1) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            skip_frames: Number of frames to skip between extractions
            
        Returns:
            List of video frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return frames
        
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                
                # Quality check
                if self.assess_frame_quality(processed_frame) >= self.quality_threshold:
                    frames.append(processed_frame)
                    extracted_count += 1
                    
                    # Check max frames limit
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
                
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        
        finally:
            cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def extract_frames_generator(self, video_path: str,
                               skip_frames: int = 1) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames one by one for memory efficiency.
        
        Args:
            video_path: Path to video file
            skip_frames: Number of frames to skip between extractions
            
        Yields:
            Preprocessed video frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                
                # Quality check
                if self.assess_frame_quality(processed_frame) >= self.quality_threshold:
                    yield processed_frame
                
                frame_count += 1
                
        except Exception as e:
            logger.error(f"Error in frame generator: {e}")
        
        finally:
            cap.release()
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a video frame for ASL detection.
        
        Args:
            frame: Raw video frame
            
        Returns:
            Preprocessed frame
        """
        # Resize to target resolution
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Enhance contrast and brightness
        frame = self.enhance_frame(frame)
        
        # Noise reduction
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        
        return frame
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame contrast and brightness for better detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def assess_frame_quality(self, frame: np.ndarray) -> float:
        """
        Assess the quality of a frame for ASL detection.
        
        Args:
            frame: Video frame
            
        Returns:
            Quality score between 0 and 1
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # Calculate brightness
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Prefer mid-range brightness
        
        # Calculate contrast
        contrast = gray.std() / 255.0
        contrast_score = min(contrast * 2, 1.0)  # Normalize
        
        # Combine scores
        quality_score = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
        
        return quality_score
    
    def process_real_time_frame(self, frame: np.ndarray, 
                              timestamp: float = None) -> Dict[str, Any]:
        """
        Process a single frame for real-time translation.
        
        Args:
            frame: Input frame from camera
            timestamp: Frame timestamp
            
        Returns:
            Dictionary with processed frame and metadata
        """
        start_time = time.time()
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Assess quality
        quality_score = self.assess_frame_quality(processed_frame)
        
        processing_time = time.time() - start_time
        
        return {
            "frame": processed_frame,
            "quality_score": quality_score,
            "processing_time": processing_time,
            "timestamp": timestamp or time.time(),
            "original_shape": frame.shape,
            "processed_shape": processed_frame.shape
        }
    
    def batch_process_frames(self, frames: List[np.ndarray], 
                           num_workers: int = 4) -> List[np.ndarray]:
        """
        Process multiple frames in parallel.
        
        Args:
            frames: List of input frames
            num_workers: Number of worker threads
            
        Returns:
            List of processed frames
        """
        processed_frames = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all frames for processing
            futures = [executor.submit(self.preprocess_frame, frame) for frame in frames]
            
            # Collect results
            for future in futures:
                try:
                    processed_frame = future.result(timeout=5.0)
                    processed_frames.append(processed_frame)
                except Exception as e:
                    logger.error(f"Frame processing failed: {e}")
        
        return processed_frames
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        try:
            info = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": 0,
                "codec": None,
                "file_size": Path(video_path).stat().st_size
            }
            
            # Calculate duration
            if info["fps"] > 0:
                info["duration"] = info["frame_count"] / info["fps"]
            
            # Try to get codec information
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            if fourcc:
                codec_bytes = int(fourcc).to_bytes(4, byteorder='little')
                info["codec"] = codec_bytes.decode('ascii', errors='ignore')
            
            return info
            
        except Exception as e:
            return {"error": f"Error reading video info: {str(e)}"}
        
        finally:
            cap.release()
    
    def create_video_thumbnail(self, video_path: str, 
                             timestamp: float = 1.0) -> Optional[np.ndarray]:
        """
        Create a thumbnail image from video at specified timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds
            
        Returns:
            Thumbnail frame or None if failed
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        try:
            # Seek to timestamp
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                # Resize for thumbnail
                thumbnail = cv2.resize(frame, (320, 240))
                return thumbnail
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
        
        finally:
            cap.release()
        
        return None
