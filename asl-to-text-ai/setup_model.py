#!/usr/bin/env python3
"""
ASL Model Setup Script
Download datasets, train models, and setup the production environment.
"""

import os
import sys
import argparse
import logging
import json
import subprocess
from pathlib import Path
import urllib.request
import zipfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ASLModelSetup:
    """Setup and training manager for ASL models."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.experiments_dir = self.project_root / "experiments"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.experiments_dir.mkdir(exist_ok=True)
    
    def download_sample_dataset(self):
        """Download and setup a sample ASL dataset."""
        logger.info("Setting up sample ASL dataset...")
        
        dataset_dir = self.data_dir / "asl_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        # Create sample dataset structure
        videos_dir = dataset_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        # Create sample metadata
        sample_metadata = []
        
        # Generate sample data for common ASL words
        common_words = [
            "hello", "goodbye", "thank_you", "please", "yes", "no",
            "good", "bad", "happy", "sad", "love", "help",
            "water", "food", "home", "work", "family", "friend",
            "morning", "evening", "today", "tomorrow", "yesterday",
            "big", "small", "hot", "cold", "fast", "slow",
            "eat", "drink", "sleep", "walk", "run", "sit",
            "stand", "come", "go", "stop", "start", "finish"
        ]
        
        for i, word in enumerate(common_words):
            sample_metadata.append({
                "video_path": f"videos/{word}_{i:03d}.mp4",
                "label": i,
                "word": word,
                "duration": 2.5,
                "quality": "high"
            })
        
        # Save metadata
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(sample_metadata, f, indent=2)
        
        # Create vocabulary file
        vocabulary = {
            "words": common_words,
            "num_classes": len(common_words),
            "description": "Common ASL words for basic communication"
        }
        
        vocab_file = dataset_dir / "vocabulary.json"
        with open(vocab_file, 'w') as f:
            json.dump(vocabulary, f, indent=2)
        
        logger.info(f"Sample dataset created with {len(common_words)} classes")
        logger.info(f"Dataset directory: {dataset_dir}")
        
        # Instructions for users
        print("\n" + "="*60)
        print("DATASET SETUP COMPLETE")
        print("="*60)
        print(f"Dataset location: {dataset_dir}")
        print(f"Videos directory: {videos_dir}")
        print(f"Metadata file: {metadata_file}")
        print(f"Vocabulary file: {vocab_file}")
        print("\nTo use real data:")
        print("1. Add your ASL videos to the videos/ directory")
        print("2. Update metadata.json with actual video paths")
        print("3. Ensure videos are in MP4 format, 2-5 seconds long")
        print("4. Run training with: python train_advanced_model.py")
        print("="*60)
        
        return dataset_dir
    
    def create_training_config(self, num_classes: int = 40):
        """Create optimized training configuration."""
        config = {
            'model_config': {
                'num_classes': num_classes,
                'sequence_length': 30,
                'embed_dim': 256,  # Reduced for faster training
                'num_heads': 8,
                'num_transformer_blocks': 4,  # Reduced for faster training
                'ff_dim': 1024
            },
            'training_config': {
                'epochs': 50,  # Reduced for faster initial training
                'learning_rate': 2e-4,
                'weight_decay': 1e-4,
                'use_focal_loss': True,
                'use_label_smoothing': False,
                'early_stopping_patience': 10
            },
            'data_config': {
                'data_dir': str(self.data_dir / "asl_dataset"),
                'sequence_length': 30,
                'batch_size': 4,  # Reduced for memory efficiency
                'num_classes': num_classes,
                'validation_split': 0.2
            }
        }
        
        config_file = self.project_root / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training configuration saved: {config_file}")
        return config_file
    
    def install_dependencies(self):
        """Install required dependencies for training."""
        logger.info("Installing training dependencies...")
        
        try:
            # Install core dependencies
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, cwd=self.project_root)
            
            logger.info("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def train_model(self, config_file: Path):
        """Train the advanced ASL model."""
        logger.info("Starting model training...")
        
        try:
            # Run training script
            cmd = [
                sys.executable, "train_advanced_model.py",
                "--config", str(config_file)
            ]
            
            subprocess.run(cmd, check=True, cwd=self.project_root)
            
            logger.info("Model training completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def setup_production_model(self):
        """Setup model for production use."""
        logger.info("Setting up production model...")
        
        # Look for trained models
        experiment_dirs = list(self.experiments_dir.glob("asl_training_*"))
        
        if not experiment_dirs:
            logger.warning("No trained models found. Creating demo setup...")
            
            # Create demo model placeholder
            demo_model_dir = self.models_dir / "demo"
            demo_model_dir.mkdir(exist_ok=True)
            
            # Create demo vocabulary
            demo_vocab = {
                "words": [
                    "hello", "goodbye", "thank_you", "please", "yes", "no",
                    "good", "bad", "happy", "sad", "love", "help"
                ],
                "num_classes": 12,
                "description": "Demo vocabulary for basic ASL recognition"
            }
            
            vocab_file = demo_model_dir / "vocabulary.json"
            with open(vocab_file, 'w') as f:
                json.dump(demo_vocab, f, indent=2)
            
            logger.info("Demo setup created. Train a real model for production use.")
            return demo_model_dir
        
        # Use latest trained model
        latest_experiment = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
        
        # Copy model to production location
        production_model_dir = self.models_dir / "production"
        if production_model_dir.exists():
            shutil.rmtree(production_model_dir)
        
        shutil.copytree(latest_experiment, production_model_dir)
        
        logger.info(f"Production model setup from: {latest_experiment}")
        return production_model_dir
    
    def run_full_setup(self):
        """Run complete setup process."""
        logger.info("Starting full ASL model setup...")
        
        # Step 1: Install dependencies
        if not self.install_dependencies():
            logger.error("Failed to install dependencies")
            return False
        
        # Step 2: Setup dataset
        dataset_dir = self.download_sample_dataset()
        
        # Step 3: Create training config
        config_file = self.create_training_config()
        
        # Step 4: Setup production model (demo mode initially)
        model_dir = self.setup_production_model()
        
        logger.info("Setup completed successfully!")
        
        print("\n" + "="*60)
        print("SETUP COMPLETE - NEXT STEPS")
        print("="*60)
        print("1. Add real ASL videos to train a custom model:")
        print(f"   - Videos: {dataset_dir}/videos/")
        print(f"   - Config: {config_file}")
        print("   - Command: python train_advanced_model.py")
        print("")
        print("2. Or use the demo mode for testing:")
        print("   - Demo model is ready for basic gesture recognition")
        print("   - Start the web app: python web_app/app.py")
        print("")
        print("3. For production deployment:")
        print("   - Train with real data first")
        print("   - Deploy to Google Cloud Run")
        print("="*60)
        
        return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup ASL Recognition System')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies only')
    parser.add_argument('--setup-dataset', action='store_true', help='Setup sample dataset only')
    parser.add_argument('--train', action='store_true', help='Train model only')
    parser.add_argument('--production', action='store_true', help='Setup production model only')
    parser.add_argument('--full', action='store_true', help='Run full setup process')
    
    args = parser.parse_args()
    
    setup = ASLModelSetup()
    
    if args.install_deps:
        setup.install_dependencies()
    elif args.setup_dataset:
        setup.download_sample_dataset()
    elif args.train:
        config_file = setup.create_training_config()
        setup.train_model(config_file)
    elif args.production:
        setup.setup_production_model()
    elif args.full or not any(vars(args).values()):
        setup.run_full_setup()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
