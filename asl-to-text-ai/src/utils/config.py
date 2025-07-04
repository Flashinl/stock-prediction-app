"""
Configuration settings for the ASL-to-Text AI system.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
DATASETS_DIR = DATA_DIR / "datasets"
VOCABULARY_DIR = DATA_DIR / "vocabulary"

# Model configuration
MODEL_CONFIG = {
    "input_shape": (224, 224, 3),  # Standard input size for video frames
    "sequence_length": 30,  # Number of frames to analyze for one sign
    "num_classes": 1000,  # Number of ASL signs in vocabulary
    "confidence_threshold": 0.7,  # Minimum confidence for sign recognition
    "batch_size": 32,
    "learning_rate": 0.001,
}

# Video processing configuration
VIDEO_CONFIG = {
    "fps": 30,  # Frames per second for processing
    "max_video_length": 300,  # Maximum video length in seconds
    "supported_formats": [".mp4", ".mov", ".avi", ".webm"],
    "frame_width": 640,
    "frame_height": 480,
    "quality_threshold": 0.5,  # Minimum video quality score
}

# Real-time processing configuration
REALTIME_CONFIG = {
    "max_latency_ms": 200,  # Maximum acceptable latency
    "buffer_size": 10,  # Number of frames to buffer
    "processing_threads": 4,  # Number of parallel processing threads
    "websocket_timeout": 30,  # WebSocket connection timeout
}

# Translation configuration
TRANSLATION_CONFIG = {
    "max_sentence_length": 100,  # Maximum words in a sentence
    "context_window": 5,  # Number of previous signs to consider for context
    "grammar_rules": True,  # Enable ASL grammar to English conversion
    "regional_dialects": ["ASL", "PSE"],  # Supported sign language variants
}

# Web application configuration
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": os.getenv("DEBUG", "False").lower() == "true",
    "secret_key": os.getenv("SECRET_KEY", "asl-ai-secret-key-change-in-production"),
    "max_upload_size": 100 * 1024 * 1024,  # 100MB max file upload
    "allowed_origins": ["*"],  # CORS origins
}

# Database configuration (for vocabulary and user data)
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "sqlite:///asl_ai.db"),
    "pool_size": 10,
    "max_overflow": 20,
    "pool_timeout": 30,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "asl_ai.log",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# Performance monitoring
MONITORING_CONFIG = {
    "enable_metrics": True,
    "metrics_port": 8080,
    "health_check_interval": 30,  # seconds
    "performance_logging": True,
}

# Error handling
ERROR_CONFIG = {
    "max_retries": 3,
    "retry_delay": 1.0,  # seconds
    "fallback_message": "[unintelligible sign]",
    "error_reporting": True,
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary."""
    return {
        "model": MODEL_CONFIG,
        "video": VIDEO_CONFIG,
        "realtime": REALTIME_CONFIG,
        "translation": TRANSLATION_CONFIG,
        "web": WEB_CONFIG,
        "database": DATABASE_CONFIG,
        "logging": LOGGING_CONFIG,
        "monitoring": MONITORING_CONFIG,
        "error": ERROR_CONFIG,
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_dir": str(DATA_DIR),
            "models_dir": str(MODELS_DIR),
            "datasets_dir": str(DATASETS_DIR),
            "vocabulary_dir": str(VOCABULARY_DIR),
        }
    }

def validate_config() -> bool:
    """Validate that all required directories exist and configuration is valid."""
    try:
        # Create directories if they don't exist
        for dir_path in [DATA_DIR, MODELS_DIR, DATASETS_DIR, VOCABULARY_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Validate configuration values
        assert MODEL_CONFIG["confidence_threshold"] > 0 and MODEL_CONFIG["confidence_threshold"] <= 1
        assert VIDEO_CONFIG["fps"] > 0
        assert REALTIME_CONFIG["max_latency_ms"] > 0
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    config = get_config()
    if validate_config():
        print("Configuration is valid!")
        print(f"Project root: {config['paths']['project_root']}")
    else:
        print("Configuration validation failed!")
