#!/usr/bin/env python3
"""
Advanced ASL Model Training Script
Train state-of-the-art ASL recognition model with multi-modal transformer architecture.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from training.advanced_trainer import ASLTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_default_config():
    """Create default configuration for training."""
    return {
        'model_config': {
            'num_classes': 1000,
            'sequence_length': 30,
            'embed_dim': 512,
            'num_heads': 8,
            'num_transformer_blocks': 6,
            'ff_dim': 2048
        },
        'training_config': {
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'use_focal_loss': True,
            'use_label_smoothing': False,
            'early_stopping_patience': 15
        },
        'data_config': {
            'data_dir': 'data/asl_dataset',
            'sequence_length': 30,
            'batch_size': 8,
            'num_classes': 1000,
            'validation_split': 0.2
        }
    }

def setup_dataset_structure():
    """Setup basic dataset structure for training."""
    data_dir = Path("data/asl_dataset")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample metadata file
    metadata_file = data_dir / "metadata.json"
    if not metadata_file.exists():
        sample_metadata = [
            {
                "video_path": "videos/sample_001.mp4",
                "label": 0,
                "word": "hello"
            },
            {
                "video_path": "videos/sample_002.mp4", 
                "label": 1,
                "word": "goodbye"
            }
        ]
        
        with open(metadata_file, 'w') as f:
            json.dump(sample_metadata, f, indent=2)
        
        logger.info(f"Created sample metadata file: {metadata_file}")
        logger.info("Please add your ASL video dataset to the data/asl_dataset/videos/ directory")
        logger.info("Update the metadata.json file with your actual video paths and labels")

def download_sample_dataset():
    """Download a sample ASL dataset for training."""
    logger.info("Setting up sample dataset...")
    
    # This would download a real dataset like WLASL or ASL-LEX
    # For now, we'll create the structure
    setup_dataset_structure()
    
    # Instructions for users
    print("\n" + "="*60)
    print("DATASET SETUP INSTRUCTIONS")
    print("="*60)
    print("1. Download an ASL dataset (recommended: WLASL, MS-ASL, or ASL-LEX)")
    print("2. Extract videos to: data/asl_dataset/videos/")
    print("3. Update metadata.json with your video paths and labels")
    print("4. Ensure videos are in MP4 format")
    print("5. Run this script again to start training")
    print("="*60)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Advanced ASL Recognition Model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data/asl_dataset', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of ASL classes')
    parser.add_argument('--setup-dataset', action='store_true', help='Setup sample dataset structure')
    
    args = parser.parse_args()
    
    # Setup dataset if requested
    if args.setup_dataset:
        download_sample_dataset()
        return
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        config = create_default_config()
        logger.info("Using default configuration")
    
    # Override with command line arguments
    if args.data_dir:
        config['data_config']['data_dir'] = args.data_dir
    if args.epochs:
        config['training_config']['epochs'] = args.epochs
    if args.batch_size:
        config['data_config']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training_config']['learning_rate'] = args.learning_rate
    if args.num_classes:
        config['model_config']['num_classes'] = args.num_classes
        config['data_config']['num_classes'] = args.num_classes
    
    # Validate dataset exists
    data_dir = Path(config['data_config']['data_dir'])
    metadata_file = data_dir / "metadata.json"
    
    if not data_dir.exists() or not metadata_file.exists():
        logger.error(f"Dataset not found at: {data_dir}")
        logger.error("Run with --setup-dataset to create sample structure")
        logger.error("Or provide a valid dataset directory with --data-dir")
        return
    
    # Check if we have actual video files
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    video_count = 0
    for item in metadata:
        video_path = data_dir / item['video_path']
        if video_path.exists():
            video_count += 1
    
    if video_count == 0:
        logger.error("No video files found in dataset!")
        logger.error("Please add ASL videos to your dataset directory")
        return
    
    logger.info(f"Found {video_count} videos in dataset")
    
    # Save current configuration
    config_save_path = Path("experiments") / "current_config.json"
    config_save_path.parent.mkdir(exist_ok=True)
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # Initialize trainer
    logger.info("Initializing Advanced ASL Trainer...")
    trainer = ASLTrainer(
        model_config=config['model_config'],
        training_config=config['training_config'],
        data_config=config['data_config']
    )
    
    # Start training
    try:
        history, summary = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}")
        logger.info(f"Model saved to: {summary['final_model_path']}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
