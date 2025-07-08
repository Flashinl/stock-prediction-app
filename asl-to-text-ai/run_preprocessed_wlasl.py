#!/usr/bin/env python3
"""
Run Pre-processed WLASL Training Pipeline
Downloads pre-processed dataset and trains model.
"""

import sys
import os
from pathlib import Path
import subprocess

def main():
    print("Pre-processed WLASL Training Pipeline")
    print("=" * 60)
    print("Using Kaggle pre-processed WLASL dataset")
    print("Target: 25 samples per word, ~2000 words")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("\nStep 1: Setting up Kaggle and downloading dataset...")
    
    # Run dataset download
    try:
        result = subprocess.run([sys.executable, 'download_preprocessed_wlasl.py'], 
                              capture_output=False, text=True)
        
        if result.returncode != 0:
            print("Dataset download/setup failed!")
            return 1
            
    except Exception as e:
        print(f"Error in dataset setup: {e}")
        return 1
    
    print("\nStep 2: Training model on pre-processed data...")
    
    # Run training
    try:
        result = subprocess.run([sys.executable, 'train_wlasl_model.py'], 
                              capture_output=False, text=True)
        
        if result.returncode != 0:
            print("Training failed!")
            return 1
            
    except Exception as e:
        print(f"Error in training: {e}")
        return 1
    
    print("\nTraining pipeline complete!")
    print("Check models/ directory for trained model")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        input("Press Enter to exit...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
